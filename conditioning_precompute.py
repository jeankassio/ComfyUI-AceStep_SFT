"""Prepare invariant ACE-Step conditioning through normal sampler conditions."""

import functools
from types import MethodType

import torch

import comfy.conds
import comfy.model_management


_ENCODER_COND = "acestep_precomputed_encoder"
_CONTEXT_COND = "acestep_precomputed_context"


def install_conditioning_precompute(model):
    """Install execution-scoped ModelPatcher methods; retain no tensor cache."""
    extra_key = "extra_conds"
    forward_key = "diffusion_model.forward"
    original_extra = model.get_model_object(extra_key)
    if getattr(original_extra, "_acestep_conditioning_precompute", False):
        return True
    transformer = model.model_options.get("transformer_options", {})
    if (
        forward_key in model.object_patches
        or "diffusion_model.prepare_condition" in model.object_patches
        or model.model_options.get("model_function_wrapper") is not None
        or model.hook_patches or model.forced_hooks or model.current_hooks
        or model.injections or model.wrappers or model.callbacks
        or model.weight_wrapper_patches
        or any(transformer.get(name) for name in ("patches", "patches_replace", "wrappers", "callbacks"))
    ):
        return False
    original_forward = model.get_model_object(forward_key)

    @functools.wraps(original_extra)
    def extra_conds(self, **kwargs):
        out = original_extra(**kwargs)
        noise = kwargs["noise"]
        if noise.ndim != 3 or noise.shape[0] != 1:
            return out
        required = ("c_crossattn", "lyric_embed", "refer_audio")
        if any(name not in out or out[name].cond is None for name in required):
            return out
        is_covers = out["is_covers"].cond if "is_covers" in out else None
        if is_covers is not None and type(is_covers) is not bool:
            return out
        if any(out[name].cond.shape[0] != 1 for name in required):
            return out

        device = kwargs["device"]
        dtype = self.get_dtype_inference()
        context, lyrics, reference = (
            comfy.model_management.cast_to_device(out[name].cond, device, dtype)
            for name in required
        )
        codes = out["audio_codes"].cond if "audio_codes" in out else None
        if codes is not None:
            codes = comfy.model_management.cast_to_device(codes, device, None)
        # For the three native Python cover modes, prepare_condition replaces
        # this source with fixed LM hints or reference audio. Only its shape is
        # consumed before replacement; the evolving denoising input is not used.
        source = torch.empty(
            (1, noise.shape[-1], noise.shape[1]), device=device, dtype=dtype
        )
        encoder_hidden, encoder_mask, context_latents = self.diffusion_model.prepare_condition(
            context, None, lyrics, None, reference.movedim(-1, -2),
            torch.zeros(1, device=device, dtype=torch.long), source,
            torch.ones_like(source), is_covers, audio_codes=codes,
        )
        if encoder_mask is not None:
            return out
        replace_null = out["replace_with_null_embeds"].cond if "replace_with_null_embeds" in out else False
        if type(replace_null) is not bool:
            return out
        if replace_null:
            encoder_hidden[:] = self.diffusion_model.null_condition_emb.to(encoder_hidden)
        out[_ENCODER_COND] = comfy.conds.CONDRegular(encoder_hidden)
        out[_CONTEXT_COND] = comfy.conds.CONDRegular(context_latents)
        return out

    @functools.wraps(original_forward)
    def forward(self, x, timestep, context, lyric_embed=None, refer_audio=None,
                audio_codes=None, is_covers=None, replace_with_null_embeds=False,
                acestep_precomputed_encoder=None, acestep_precomputed_context=None,
                **kwargs):
        if (
            acestep_precomputed_encoder is None or acestep_precomputed_context is None
            or x.ndim != 3 or x.shape[0] != 1
            or acestep_precomputed_encoder.shape[0] != x.shape[0]
            or acestep_precomputed_context.shape != (x.shape[0], x.shape[-1], x.shape[1] * 2)
        ):
            return original_forward(
                x, timestep, context, lyric_embed=lyric_embed, refer_audio=refer_audio,
                audio_codes=audio_codes, is_covers=is_covers,
                replace_with_null_embeds=replace_with_null_embeds, **kwargs,
            )
        out = self.decoder(
            hidden_states=x.movedim(-1, -2), timestep=timestep, timestep_r=timestep,
            attention_mask=None, encoder_hidden_states=acestep_precomputed_encoder,
            encoder_attention_mask=None, context_latents=acestep_precomputed_context,
        )
        return out.movedim(-1, -2)

    extra_conds._acestep_conditioning_precompute = True
    extra_conds._acestep_previous_patch = model.object_patches.get(extra_key)
    forward._acestep_conditioning_precompute = True
    forward._acestep_previous_patch = model.object_patches.get(forward_key)
    model.add_object_patch(extra_key, MethodType(extra_conds, model.model))
    model.add_object_patch(forward_key, MethodType(forward, model.model.diffusion_model))
    return True


def clear_conditioning_precompute(model):
    applied = False
    for key in ("extra_conds", "diffusion_model.forward"):
        patched = model.object_patches.get(key)
        if not getattr(patched, "_acestep_conditioning_precompute", False):
            continue
        previous = patched._acestep_previous_patch
        if previous is None:
            model.object_patches.pop(key, None)
        else:
            model.add_object_patch(key, previous)
        applied = applied or key in model.object_patches_backup
    if applied:
        model.unpatch_model(unpatch_weights=False)
        model.patch_model(load_weights=False)
