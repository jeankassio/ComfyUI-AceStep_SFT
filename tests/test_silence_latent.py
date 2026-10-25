"""Official silence conditioning and patch lifetime, without model weights or CUDA."""
import functools
import math
import os
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from types import MethodType
import unittest
from unittest.mock import patch

import torch

from test_sampling import NODE_PATH, TensorCondition, load_definitions


def node_namespace(node_file=NODE_PATH):
    namespace = dict(torch=torch, math=math, os=os, functools=functools,
                     MethodType=MethodType, contextmanager=contextmanager,
                     __file__=str(node_file))
    load_definitions(NODE_PATH, {
        "_get_acestep_silence_latent", "_ensure_acestep_silence_latent",
        "_clear_acestep_silence_latent_patch", "_acestep_sampling_model",
    }, namespace)
    for name in ("clear_voice_profile", "clear_conditioning_precompute", "_clear_acestep_reference_timbre_repeats",
                 "_clear_acestep_reference_attention",
                 "_clear_acestep_native_task_patches",
                 "_clear_acestep_timbre_cls_patch"):
        namespace[name] = lambda model: None
    return namespace


class ConditioningModel:
    def __init__(self):
        self.calls = []
        self.other = "original"

    def extra_conds(self, **kwargs):
        self.calls.append(kwargs)
        length = kwargs["noise"].shape[-1]
        references = kwargs.get("reference_audio_timbre_latents")
        # Deliberately unlike the official asset, representing the core's fallback.
        reference = torch.full((1, 64, length), -99.0)
        output = {"keep": object(), "is_covers": bool(references)}
        if references:
            voice = references[-1][..., :length]
            reference[..., :voice.shape[-1]] = voice
        elif "audio_codes" in kwargs:
            output["audio_codes"] = kwargs["audio_codes"]
        output["refer_audio"] = TensorCondition(reference)
        return output


class ObjectPatcher:
    def __init__(self, model=None):
        self.model = model or ConditioningModel()
        self.object_patches = {}
        self.object_patches_backup = {}
        self.model_options = {}

    def clone(self):
        clone = ObjectPatcher(self.model)
        clone.object_patches = self.object_patches.copy()
        clone.model_options = self.model_options.copy()
        return clone

    def get_model_object(self, name):
        return self.object_patches.get(name, self.object_patches_backup.get(
            name, getattr(self.model, name)))

    def add_object_patch(self, name, value):
        self.object_patches[name] = value

    def patch_model(self, load_weights=False):
        for name, value in self.object_patches.items():
            self.object_patches_backup.setdefault(name, getattr(self.model, name))
            setattr(self.model, name, value)

    def unpatch_model(self, unpatch_weights=False):
        for name, value in self.object_patches_backup.items():
            setattr(self.model, name, value)
        self.object_patches_backup.clear()


class SilenceLatentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.official = torch.load(NODE_PATH.parent / "silence_latent.pt",
                                  map_location="cpu", weights_only=True)

    def setUp(self):
        self.ns = node_namespace()

    def silence(self, length, dtype=None):
        return self.ns["_get_acestep_silence_latent"](length, "cpu", dtype)

    def test_bundled_official_tensor_is_full_length_finite_and_not_a_constant_tail(self):
        self.assertEqual(self.official.shape, (1, 64, 15000))
        self.assertEqual(self.official.dtype, torch.float32)
        self.assertTrue(torch.isfinite(self.official).all())
        self.assertFalse(torch.equal(self.official[..., 4], self.official[..., 1000]))

    def test_requested_prefixes_and_long_sequences_match_the_official_tensor_exactly(self):
        for length in (0, 1, 2, 3, 4, 750, 4500, 15000, 15007):
            with self.subTest(length=length):
                repeats = max(1, math.ceil(length / self.official.shape[-1]))
                expected = self.official.repeat(1, 1, repeats)[..., :length]
                actual = self.silence(length)
                self.assertEqual(actual.shape, (1, 64, length))
                self.assertEqual(actual.device.type, "cpu")
                self.assertTrue(actual.is_contiguous())
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_dtype_conversion_is_exact_and_returned_storage_is_independent_and_compact(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                actual = self.silence(7, dtype)
                torch.testing.assert_close(actual, self.official[..., :7].to(dtype), rtol=0, atol=0)
                self.assertEqual(actual.untyped_storage().nbytes(), actual.numel() * actual.element_size())
                actual.zero_()
                torch.testing.assert_close(self.silence(7, dtype), self.official[..., :7].to(dtype),
                                           rtol=0, atol=0)

    def test_loading_is_cpu_only_and_uses_weights_only_and_mmap_without_advancing_rng(self):
        before = torch.get_rng_state().clone()
        with patch.object(torch, "load", wraps=torch.load) as loader:
            self.silence(4)
        loader.assert_called_once()
        self.assertEqual(loader.call_args.kwargs,
                         dict(map_location="cpu", weights_only=True, mmap=True))
        self.assertEqual(Path(loader.call_args.args[0]), NODE_PATH.parent / "silence_latent.pt")
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_malformed_assets_fail_instead_of_falling_back_to_approximate_silence(self):
        values = ({"samples": torch.zeros(1, 64, 4)}, torch.zeros(1, 32, 4),
                  torch.zeros(2, 64, 4), torch.zeros(1, 64, 0),
                  torch.zeros(1, 64, 4, dtype=torch.int64))
        with TemporaryDirectory() as directory:
            namespace = node_namespace(Path(directory) / "nodes.py")
            for value in values:
                with self.subTest(type=type(value).__name__, shape=getattr(value, "shape", None)):
                    torch.save(value, Path(directory) / "silence_latent.pt")
                    with self.assertRaisesRegex(ValueError, "silence_latent.pt.*64"):
                        namespace["_get_acestep_silence_latent"](4, "cpu")

    def test_default_and_empty_reference_conditions_keep_lm_codes_and_other_outputs(self):
        patcher = ObjectPatcher()
        original = patcher.model.extra_conds
        self.ns["_ensure_acestep_silence_latent"](patcher)
        installed = patcher.object_patches["extra_conds"]
        self.ns["_ensure_acestep_silence_latent"](patcher)
        self.assertIs(patcher.object_patches["extra_conds"], installed)
        self.assertEqual(patcher.model.extra_conds, original)
        codes = torch.tensor([[11, 22, 33]])
        for references in (None, []):
            kwargs = dict(noise=torch.zeros(2, 64, 4500), device="cpu", audio_codes=codes,
                          reference_audio_timbre_latents=references)
            result = installed(**kwargs)
            torch.testing.assert_close(result["refer_audio"].cond, self.official[..., :4500],
                                       rtol=0, atol=0)
            self.assertIs(result["audio_codes"], codes)
            self.assertFalse(result["is_covers"])
            self.assertIn("keep", result)
            self.assertIs(patcher.model.calls[-1]["reference_audio_timbre_latents"], references)

    def test_explicit_reference_prefix_remains_exact_and_short_tail_uses_official_silence_start(self):
        patcher = ObjectPatcher()
        self.ns["_ensure_acestep_silence_latent"](patcher)
        installed = patcher.object_patches["extra_conds"]
        for voice_length in (3, 40, 57):
            voice = torch.arange(voice_length, dtype=torch.float32).reshape(1, 1, -1).repeat(1, 64, 1)
            before = voice.clone()
            references = [torch.ones(1, 64, 2), voice]
            result = installed(noise=torch.zeros(1, 64, 40), device="cpu",
                               reference_audio_timbre_latents=references,
                               audio_codes=torch.tensor([[5]]))
            prefix = min(voice_length, 40)
            torch.testing.assert_close(result["refer_audio"].cond[..., :prefix], voice[..., :prefix],
                                       rtol=0, atol=0)
            if prefix < 40:
                torch.testing.assert_close(result["refer_audio"].cond[..., prefix:],
                                           self.official[..., :40-prefix], rtol=0, atol=0)
            self.assertTrue(result["is_covers"])
            self.assertNotIn("audio_codes", result)
            torch.testing.assert_close(voice, before, rtol=0, atol=0)
            self.assertIs(patcher.model.calls[-1]["reference_audio_timbre_latents"], references)

    def test_cleanup_restores_previous_applied_patch_and_keeps_unrelated_objects(self):
        patcher = ObjectPatcher()
        delegated = []
        def previous(self, **kwargs):
            delegated.append(kwargs)
            return ConditioningModel.extra_conds(self, **kwargs)
        external = MethodType(previous, patcher.model)
        patcher.add_object_patch("extra_conds", external)
        patcher.add_object_patch("other", "preserved")
        self.ns["_ensure_acestep_silence_latent"](patcher)
        patcher.patch_model()
        result = patcher.model.extra_conds(noise=torch.zeros(1, 64, 4), device="cpu")
        torch.testing.assert_close(result["refer_audio"].cond, self.official[..., :4], rtol=0, atol=0)
        self.assertEqual(len(delegated), 1)
        self.ns["_clear_acestep_silence_latent_patch"](patcher)
        self.assertIs(patcher.model.extra_conds, external)
        self.assertIs(patcher.object_patches["extra_conds"], external)
        self.assertEqual(patcher.model.other, "preserved")
        self.ns["_clear_acestep_silence_latent_patch"](patcher)
        self.assertIs(patcher.model.extra_conds, external)
        patcher.unpatch_model()

    def test_sampling_exception_removes_applied_patch_from_shared_model_and_clone(self):
        patcher = ObjectPatcher()
        original = patcher.model.extra_conds
        patcher.add_object_patch("other", "preserved")
        with self.assertRaisesRegex(RuntimeError, "sampling failed"):
            with self.ns["_acestep_sampling_model"](patcher) as clone:
                self.ns["_ensure_acestep_silence_latent"](clone)
                clone.patch_model()
                raise RuntimeError("sampling failed")
        self.assertEqual(patcher.model.extra_conds, original)
        self.assertNotIn("extra_conds", clone.object_patches)
        self.assertEqual(clone.object_patches, {"other": "preserved"})
        self.assertEqual(patcher.object_patches, {"other": "preserved"})
        self.assertEqual(patcher.model.other, "preserved")
        clone.unpatch_model()


if __name__ == "__main__":
    unittest.main()
