"""Exact ACE-Step checkpoint switching using one native dynamic model owner.

The checkpoint tensors stay on the CPU (normally backed by ComfyUI's mmap
loader).  A variant changes the source of a parameter, rather than adding a
full-sized LoRA replacement patch.  ComfyUI still owns all GPU allocations,
eviction, adapter application, and device transfers.
"""

import logging
import os
import uuid
from dataclasses import dataclass, field

import torch
from tqdm.auto import tqdm

import comfy.model_detection
import comfy.lora
import comfy.model_management
import comfy.model_patcher
import comfy.sd
import comfy.utils


_LOG = logging.getLogger(__name__)
_DYNAMIC_PATCHER = getattr(comfy.model_patcher, "ModelPatcherDynamic", None)


class _SharingUnsupported(ValueError):
    """A checkpoint needs the ordinary native loader."""


def _fingerprint(value):
    """Conservative identity/version comparison; never equate different LoRAs."""
    if isinstance(value, torch.Tensor):
        try:
            version = value._version
        except RuntimeError:  # Inference tensors do not have version counters.
            version = None
        return ("tensor", id(value), version, tuple(value.shape), value.dtype)
    if isinstance(value, (str, bytes, int, float, bool, type(None), uuid.UUID)):
        return value
    if isinstance(value, (tuple, list)):
        return (type(value).__name__, tuple(_fingerprint(item) for item in value))
    if isinstance(value, dict):
        return ("dict", tuple(sorted((str(key), _fingerprint(item)) for key, item in value.items())))
    # Native adapters carry their tensors in ``weights``.  Object identity also
    # distinguishes arbitrary callable wrappers whose state we cannot inspect.
    weights = getattr(value, "weights", None)
    return (type(value), id(value), _fingerprint(weights) if weights is not None else None)


def _patch_fingerprints(patcher):
    return {key: _fingerprint(value) for key, value in patcher.patches.items()}


def _wrapper_fingerprints(patcher):
    return {key: _fingerprint(value) for key, value in patcher.weight_wrapper_patches.items()}


def _same_tensor_bytes(first, second):
    """Checkpoint equality includes signed zero and NaN payload bits."""
    if first.shape != second.shape or first.dtype != second.dtype:
        return False
    first_flat = first.reshape(-1)
    second_flat = second.reshape(-1)
    # Most independently trained large matrices already differ at their start.
    # Reject them before touching the remaining mmap pages.  A matching prefix
    # is only a probe: sharing still requires equality of every checkpoint byte.
    prefix = min(first_flat.numel(), 256)
    if not torch.equal(
        first_flat[:prefix].view(torch.uint8), second_flat[:prefix].view(torch.uint8)
    ):
        return False
    return torch.equal(first_flat.view(torch.uint8), second_flat.view(torch.uint8))


@dataclass
class _VariantFamily:
    """CPU source ownership and activation metadata for one shared model graph."""

    sources: tuple
    parameter_keys: frozenset
    buffer_keys: frozenset
    paths: tuple
    names: tuple
    checkpoint_sources: object = None
    runtime_cache_bytes: int = 0
    device: object = None
    active_variant: object = None
    active_patches: dict = field(default_factory=dict)
    active_wrappers: dict = field(default_factory=dict)
    active_uuid: object = None
    active_sources: object = None
    last_transition: dict = field(default_factory=dict)

    def for_new_model(self):
        # CPU sources may be shared by separate native graph owners.  GPU and
        # activation state must never follow a deepclone onto another graph.
        return _VariantFamily(
            self.sources, self.parameter_keys, self.buffer_keys, self.paths, self.names,
            checkpoint_sources=self.checkpoint_sources,
            runtime_cache_bytes=self.runtime_cache_bytes,
        )


def _runtime_cache_plan(model, sources):
    """Only cache weights that native dynamic loading casts before LoRA."""
    targets = {}
    for name, module in model.named_modules():
        if not hasattr(module, "comfy_cast_weights"):
            continue
        params = dict(module.named_parameters(recurse=False))
        if set(params) - {"weight", "bias"} or any(module.children()):
            continue
        if sum(tensor.nbytes for tensor in params.values()) <= 16 * 1024:
            continue
        module_targets = {}
        for suffix in ("weight", "bias"):
            key = f"{name}.{suffix}" if name else suffix
            dtype = getattr(module, suffix + "_comfy_model_dtype", None)
            if key in sources[0] and dtype in (torch.bfloat16, torch.float16):
                module_targets[key] = dtype
        runtime_size = sum(
            tensor.numel() * module_targets.get(f"{name}.{suffix}" if name else suffix, tensor.dtype).itemsize
            for suffix, tensor in params.items()
        )
        runtime_size += sum(tensor.nbytes for tensor in module.buffers(recurse=False))
        # Native DynamicVRAM force-loads small modules and applies their LoRA
        # before casting.  Staging must not move a module across that boundary.
        if runtime_size > 16 * 1024:
            targets.update(module_targets)
    unique = {}
    largest_variant = 0
    for source in sources:
        variant_bytes = sum(tensor.nbytes for tensor in source.values())
        for key, tensor in source.items():
            dtype = targets.get(key, tensor.dtype)
            if key in targets:
                unique[(id(tensor), dtype)] = tensor.numel() * dtype.itemsize
        largest_variant = max(largest_variant, variant_bytes)
    return targets, sum(unique.values()), largest_variant


def _cache_runtime_sources(model, sources):
    targets, required, native_host_bytes = _runtime_cache_plan(model, sources)
    if not targets:
        return sources, 0
    available = comfy.model_management.get_free_memory(torch.device("cpu"))
    headroom = max(2 * 1024 ** 3, int(comfy.model_management.get_total_memory(torch.device("cpu")) * 0.08))
    # Keep capacity for raw native DiT loading, subsequent text encoder/VAE
    # work, and the OS.  This node-owned optional cache cannot be evicted by
    # ComfyUI's host-pin manager, so never consume all currently free RAM.
    reserve = native_host_bytes + 9 * 1024 ** 3 + headroom
    budget = min(4 * 1024 ** 3, max(0, available - reserve))
    if not budget:
        _LOG.warning(
            "[AceStep cascade] Runtime RAM cache disabled: %.1f GiB reserved "
            "for native loading, text encoders/VAE and headroom; %.1f GiB available.",
            reserve / 1024 ** 3, available / 1024 ** 3,
        )
        return sources, 0
    # Prefer large transfers, distributing capacity across the variants rather
    # than filling it with SFT alone.  Identical checkpoint tensors share one
    # cache allocation; every unselected source stays at its original dtype.
    keys = sorted(targets, key=lambda key: max(source[key].numel() * targets[key].itemsize for source in sources), reverse=True)
    selected = {}
    cache_bytes = 0
    for key in keys:
        dtype = targets[key]
        for source in sources:
            tensor = source[key]
            identity = (id(tensor), dtype)
            size = tensor.numel() * dtype.itemsize
            if identity not in selected and cache_bytes + size <= budget:
                selected[identity] = (tensor, dtype, size)
                cache_bytes += size
    if not selected:
        _LOG.info("[AceStep cascade] Runtime RAM cache skipped: no eligible tensor fits the safe RAM budget.")
        return sources, 0
    cache = {}
    progress = comfy.utils.ProgressBar(len(selected))
    for completed, (identity, (tensor, dtype, size)) in enumerate(selected.items(), 1):
        comfy.model_management.throw_exception_if_processing_interrupted()
        if comfy.model_management.get_free_memory(torch.device("cpu")) < reserve + size:
            _LOG.warning("[AceStep cascade] Runtime RAM cache disabled: available memory fell during staging.")
            return sources, 0
        cache[identity] = tensor.to(device="cpu", dtype=dtype, copy=True)
        progress.update_absolute(completed)
    results = tuple(
        {key: cache.get((id(tensor), targets.get(key)), tensor) for key, tensor in source.items()}
        for source in sources
    )
    _LOG.info(
        "[AceStep cascade] %.2f / %.2f GiB eligible runtime weights cached at native inference dtypes "
        "(4 GiB family cap); remaining checkpoint sources unchanged.",
        cache_bytes / 1024 ** 3, required / 1024 ** 3,
    )
    return results, cache_bytes


def _reload_variant(path, name, disable_dynamic=False):
    patcher = comfy.sd.load_diffusion_model(path, disable_dynamic=disable_dynamic)
    patcher.model_options["acestep_checkpoint_name"] = name
    return patcher


class _SharedVariantPatcher(_DYNAMIC_PATCHER or comfy.model_patcher.ModelPatcher):
    """Native dynamic patcher with checkpoint-aware per-module invalidation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._variant_family = None
        self._variant_index = None

    def clone(self, disable_dynamic=False, model_override=None, force_deepcopy=False):
        cloned = super().clone(
            disable_dynamic=disable_dynamic,
            model_override=model_override,
            force_deepcopy=force_deepcopy,
        )
        if isinstance(cloned, _SharedVariantPatcher):
            cloned._variant_index = self._variant_index
            family = self._variant_family
            cloned._variant_family = (
                family if cloned.model is self.model or family is None else family.for_new_model()
            )
        return cloned

    def _unknown_weight_mutations(self):
        # Native hooks and graph injections may modify parameters outside the
        # adapter dictionaries.  Keep exact behavior but invalidate all caches.
        return bool(
            self.hook_patches
            or self.hook_backup
            or self.current_hooks
            or self.forced_hooks
            or self.injections
            or self.get_all_callbacks(comfy.model_patcher.CallbacksMP.ON_LOAD)
        )

    def _variant_sources(self, family, variant, unknown_mutations):
        desired = family.sources[variant]
        if family.checkpoint_sources is None:
            return desired
        original = family.checkpoint_sources[variant]
        if unknown_mutations or self.weight_wrapper_patches or self.weight_inplace_update:
            # The native raw-weight fallback can need twice the host memory.
            # Release the optional cache instead of retaining both copies.
            family.sources = family.checkpoint_sources
            family.checkpoint_sources = None
            family.runtime_cache_bytes = 0
            _LOG.warning("[AceStep cascade] Runtime RAM cache released for a custom weight transform; using original checkpoint sources.")
            return original
        force_modules = set()
        for key, patches in self.patches.items():
            if key in original and comfy.lora.calculate_shape(patches, original[key], key) != original[key].shape:
                force_modules.add(key.rpartition(".")[0])
        if not force_modules:
            return desired
        # Native resizing-LoRA force loading applies patches to the raw source
        # before its final cast, unlike the normal dynamic cast-before-LoRA path.
        return {
            key: original[key] if key.rpartition(".")[0] in force_modules else tensor
            for key, tensor in desired.items()
        }

    def load(self, device_to=None, lowvram_model_memory=0,
             force_patch_weights=False, full_load=False, dirty=False):
        family = self._variant_family
        variant = self._variant_index
        if family is None or variant is None:
            return super().load(
                device_to=device_to,
                lowvram_model_memory=lowvram_model_memory,
                force_patch_weights=force_patch_weights,
                full_load=full_load,
                dirty=dirty,
            )

        if family.device is not None and device_to != family.device:
            raise RuntimeError(
                "A shared ACE-Step cascade cannot be retargeted to another device. "
                "Create a fresh cascade loader or use independent model loaders."
            )
        if family.device is None:
            family.device = device_to

        known_owner = self.model.current_weight_patches_uuid in (None, family.active_uuid)
        unknown_mutations = self._unknown_weight_mutations()
        desired = self._variant_sources(family, variant, unknown_mutations)
        previous = None if family.active_variant is None else family.active_sources
        source_changes = {
            key for key, tensor in desired.items()
            if previous is None or not known_owner or unknown_mutations or previous[key] is not tensor
        }
        invalidate_all = previous is None or not known_owner or unknown_mutations
        del previous
        patches = _patch_fingerprints(self)
        wrappers = _wrapper_fingerprints(self)
        changed_keys = set(source_changes)
        changed_keys.update(
            key for key in patches.keys() | family.active_patches.keys()
            if patches.get(key) != family.active_patches.get(key)
        )
        changed_keys.update(
            key for key in wrappers.keys() | family.active_wrappers.keys()
            if wrappers.get(key) != family.active_wrappers.get(key)
        )
        changing = bool(changed_keys) or invalidate_all

        if changing and getattr(device_to, "type", None) == "cuda":
            # Host-buffer release must wait for previous asynchronous prefetches.
            torch.cuda.synchronize(device_to)

        # Tiny/forced weights and buffers are restored by the native loader too;
        # restore them *before* changing their pristine CPU checkpoint source.
        self.restore_loaded_backups()
        if changing:
            # Native host pins contain gathered SOURCE bytes.  Reusing them after
            # a source switch would silently resurrect the previous checkpoint.
            # Flush through the native owner API; GPU VBAR blocks stay resident.
            self.partially_unload_ram(1e32)

        module_changes = {key.rpartition(".")[0] for key in changed_keys}
        invalidated = 0
        retained = 0
        for name, module in self.model.named_modules():
            if invalidate_all or name in module_changes:
                if hasattr(module, "_v_signature"):
                    module._v_signature = None
                    invalidated += 1
                # Prefetch objects refer to the old source/adapter work buffers.
                if hasattr(module, "_prefetch"):
                    del module._prefetch
            elif getattr(module, "_v_signature", None) is not None:
                retained += 1

        # A failed load cannot leave the family claiming a valid active source.
        # A subsequent retry will reassign every source and invalidate all GPU
        # signatures, even if this load failed halfway through parameter setup.
        family.active_variant = None
        family.active_uuid = None
        family.active_sources = None
        try:
            for key in source_changes:
                if key in family.parameter_keys:
                    comfy.utils.set_attr_param(self.model, key, desired[key])
                elif key in family.buffer_keys:
                    comfy.utils.set_attr_buffer(self.model, key, desired[key])
                else:
                    raise RuntimeError(f"Unregistered cascade source tensor: {key}")

            # The source/adapter comparison above replaces core's global dirty
            # flag.  Core retains ownership of every actual weight transfer.
            result = super().load(
                device_to=device_to,
                lowvram_model_memory=lowvram_model_memory,
                force_patch_weights=force_patch_weights,
                full_load=full_load,
                dirty=False,
            )
        except Exception:
            family.active_patches = {}
            family.active_wrappers = {}
            raise

        family.active_variant = variant
        family.active_uuid = self.patches_uuid
        family.active_sources = desired
        family.active_patches = patches
        family.active_wrappers = wrappers
        family.last_transition = {
            "variant": family.names[variant],
            "source_tensors_changed": len(source_changes),
            "effective_tensors_changed": len(changed_keys),
            "gpu_module_signatures_retained": retained,
            "gpu_module_signatures_invalidated": invalidated,
            "host_pins_flushed": changing,
            "full_invalidation": invalidate_all,
        }
        if changing:
            _LOG.info(
                "[AceStep cascade] %s: %d source tensors switched; %d GPU module "
                "signatures retained, %d invalidated; native host pins refreshed.",
                family.names[variant], len(source_changes), retained, invalidated,
            )
        return result


def _wrap_native_patcher(native, family, variant):
    """Construct a managed owner; never change the class of an existing patcher."""
    if not native.is_dynamic():
        raise _SharingUnsupported("Native dynamic VRAM loading is not enabled")
    if (native.patches or native.object_patches or native.hook_patches
            or native.injections or native.weight_wrapper_patches):
        raise _SharingUnsupported("Cascade construction requires a pristine checkpoint")
    wrapped = _SharedVariantPatcher(
        native.model, native.load_device, native.offload_device, native.model_size(),
        weight_inplace_update=native.weight_inplace_update, fast_disk=native.fast_disk,
    )
    wrapped.model_options = comfy.utils.deepcopy_list_dict(native.model_options)
    wrapped.force_cast_weights = native.force_cast_weights
    wrapped._variant_family = family
    wrapped._variant_index = variant
    if family.device is None:
        family.device = native.load_device
    wrapped.model_options["acestep_checkpoint_name"] = family.names[variant]
    wrapped.cached_patcher_init = (_reload_variant, (family.paths[variant], family.names[variant]))
    return wrapped


def _normalized_target_sources(path, native, reference_keys):
    state, metadata = comfy.utils.load_torch_file(path, return_metadata=True)
    state, metadata = comfy.utils.convert_old_quants(state, "", metadata=metadata)
    prefix = comfy.model_detection.unet_prefix_from_state_dict(state)
    stripped = comfy.utils.state_dict_prefix_replace(state, {prefix: ""}, filter_keys=True)
    if stripped:
        state = stripped
    config = comfy.model_detection.model_config_from_unet(state, "", metadata=metadata)
    base_config = native.model.model_config
    # The loaded config has already had its inference dtype installed; a freshly
    # detected target config has not.  Apply that same native configuration step
    # before comparing architectural options.
    if config is not None:
        config.set_inference_dtype(
            base_config.unet_config.get("dtype"), base_config.manual_cast_dtype,
            device=native.load_device,
        )
    if (config is None or type(config) is not type(base_config)
            or config.unet_config != base_config.unet_config
            or config.quant_config is not None):
        raise _SharingUnsupported("Checkpoint architectures or quantization differ")
    state = base_config.process_unet_state_dict(state)
    result = {"diffusion_model." + key: tensor for key, tensor in state.items()}
    if result.keys() != reference_keys:
        raise _SharingUnsupported("Checkpoint parameter/buffer names differ")
    return result


def load_cascade_models(paths, names, cache_runtime_weights=False):
    """Return exact SFT/Base/Turbo MODELs with one shared dynamic graph.

    ``paths`` are resolved local checkpoint paths, in SFT/Base/Turbo order.
    ``names`` are their ComfyUI selector names.  Unsupported native/static/CPU
    configurations return three ordinary independently managed MODELs.
    ``cache_runtime_weights`` optionally stages eligible native casts in RAM,
    within a conservative memory reserve and a 4 GiB shared-family limit.
    Original checkpoint tensors remain available for native forced-load paths.
    """
    paths = tuple(os.fspath(path) for path in paths)
    names = tuple(str(name) for name in names)
    if len(paths) != 3 or len(names) != 3:
        raise ValueError("The cascade loader requires SFT, Base, and Turbo checkpoints")
    native = _reload_variant(paths[0], names[0])
    try:
        if native.model.model_config.unet_config.get("audio_model") != "ace1.5":
            raise _SharingUnsupported("The shared cascade requires ACE-Step 1.5 checkpoints")
        if not native.is_dynamic() or comfy.model_management.is_device_cpu(native.load_device):
            raise _SharingUnsupported("Dynamic VRAM is unavailable (static or CPU loading)")
        if _DYNAMIC_PATCHER is None or type(native) is not _DYNAMIC_PATCHER:
            raise _SharingUnsupported("An unsupported custom dynamic patcher is active")
        if getattr(native.model.model_config, "quant_config", None) is not None:
            raise _SharingUnsupported("Quantized checkpoints require the ordinary loader")

        params = dict(native.model.named_parameters())
        buffers = dict(native.model.named_buffers())
        original = {key: value.detach() for key, value in {**params, **buffers}.items()}
        if any(type(tensor) is not torch.Tensor or tensor.device.type != "cpu"
               for tensor in original.values()):
            raise _SharingUnsupported("Cascade sources must be plain CPU tensors")
        # Runtime rotary/FSQ caches are registered non-persistent buffers and
        # deliberately absent from a checkpoint.  Compare only saved state;
        # retain those runtime buffers in the owner's original CPU sources.
        diffusion_keys = frozenset(
            key for key in native.model.state_dict() if key.startswith("diffusion_model.")
        )
        if not diffusion_keys.issubset(original):
            raise _SharingUnsupported("Checkpoint aliases require the ordinary native loader")
        targets = [_normalized_target_sources(path, native, diffusion_keys) for path in paths[1:]]
        progress = comfy.utils.ProgressBar(sum(len(target) for target in targets))
        completed = 0
        sources = [original]
        for name, target in zip(names[1:], targets):
            source = original.copy()
            shared = 0
            shared_bytes = 0
            different = 0
            for key, tensor in tqdm(
                target.items(), total=len(target), unit="tensor",
                desc=f"ACE-Step cascade compare: {name}",
            ):
                comfy.model_management.throw_exception_if_processing_interrupted()
                baseline = original[key]
                if (type(tensor) is not torch.Tensor or tensor.device.type != "cpu"
                        or tensor.shape != baseline.shape or tensor.dtype != baseline.dtype):
                    raise _SharingUnsupported(f"Incompatible tensor shape/dtype: {key}")
                # Deduplicate against all prior variants, including Base/Turbo
                # weights that are equal to one another but differ from SFT.
                equal = next((previous[key] for previous in sources
                              if _same_tensor_bytes(previous[key], tensor)), None)
                if equal is not None:
                    source[key] = equal
                    shared += 1
                    shared_bytes += tensor.numel() * tensor.element_size()
                else:
                    source[key] = tensor
                    different += 1
                completed += 1
                progress.update_absolute(completed)
            sources.append(source)
            _LOG.info(
                "[AceStep cascade] %s: %d shared tensors (%.2f GiB CPU source), "
                "%d distinct tensors; original checkpoint dtypes preserved.",
                name, shared, shared_bytes / (1024 ** 3), different,
            )
        sources = tuple(sources)
        runtime_sources, cache_bytes = _cache_runtime_sources(native.model, sources) if cache_runtime_weights else (sources, 0)
        family = _VariantFamily(
            runtime_sources, frozenset(params), frozenset(buffers), paths, names,
            checkpoint_sources=sources if cache_bytes else None,
            runtime_cache_bytes=cache_bytes,
        )
        first = _wrap_native_patcher(native, family, 0)
        outputs = [first]
        for variant in (1, 2):
            clone = first.clone()
            clone._variant_index = variant
            clone.patches_uuid = uuid.uuid4()
            clone.model_options["acestep_checkpoint_name"] = names[variant]
            clone.cached_patcher_init = (_reload_variant, (paths[variant], names[variant]))
            outputs.append(clone)
        return tuple(outputs)
    except _SharingUnsupported as exc:
        _LOG.warning(
            "[AceStep cascade] Shared switching unavailable: %s. Falling back to "
            "three native checkpoint owners; selective GPU retention is disabled.", exc,
        )
        return (native,) + tuple(_reload_variant(path, name) for path, name in zip(paths[1:], names[1:]))
