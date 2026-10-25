"""A private, differentiable ACE-Step instance for a bounded voice fit."""

from contextlib import contextmanager
import functools
import math
import time
import uuid

import torch

import comfy.model_management
from comfy.ldm.modules.attention import ComfyAttention, attention_pytorch


class _AdaptationClock:
    def __init__(self, seconds):
        seconds = float(seconds)
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("Voice adaptation time must be finite and nonnegative.")
        self.seconds = min(seconds, 60.0)
        self.started = None

    def __call__(self):
        now = time.perf_counter()
        if self.started is None:
            self.started = now
        return now

    def check(self):
        if self.started is not None and time.perf_counter() - self.started >= self.seconds:
            raise TimeoutError("Voice adaptation reached its time budget.")


@contextmanager
def adaptation_runtime(model, seconds=60.0):
    """Yield an unloaded static delegate and a lazily started monotonic clock.

    The caller prepares/rebinds its own model patches, then loads this delegate
    through model management. No original weights or methods are modified.
    """
    clock = _AdaptationClock(seconds)
    missing = object()
    previous_patches = {}
    with torch.inference_mode(False), torch.enable_grad():
        static = model.clone(disable_dynamic=True, force_deepcopy=True)
        # A deep clone has separate weights but inherits the parent's family ID.
        # Give this disposable instance its own family so cleanup cannot unload
        # the original model or a user-owned clone.
        static.clone_base_uuid = uuid.uuid4()

        def patch(path, value):
            previous_patches[path] = static.object_patches.get(path, missing)
            static.add_object_patch(path, value)

        def check_gradient(gradient):
            clock.check()
            return gradient

        try:
            for name, module in static.model.named_modules():
                if isinstance(module, ComfyAttention):
                    patch(f"{name}.function", attention_pytorch)

            layers = static.get_model_object("diffusion_model.decoder.layers")
            for index in range(len(layers)):
                path = f"diffusion_model.decoder.layers.{index}.forward"
                original_forward = static.get_model_object(path)

                @functools.wraps(original_forward)
                def timed_forward(*args, _forward=original_forward, **kwargs):
                    clock.check()
                    output = _forward(*args, **kwargs)
                    clock.check()
                    if output.requires_grad:
                        output.register_hook(check_gradient)
                    return output

                patch(path, timed_forward)

            yield static, clock
        finally:
            for path, previous in previous_patches.items():
                if previous is missing:
                    static.object_patches.pop(path, None)
                else:
                    static.add_object_patch(path, previous)
            try:
                static.unpatch_model(unpatch_weights=False)
            finally:
                comfy.model_management.unload_model_and_clones(
                    static, unload_additional_models=False
                )
