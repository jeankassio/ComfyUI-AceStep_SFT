"""Scoped reference attention uses checkpoint windows without altering weights."""
import ast
from contextlib import contextmanager
import functools
import itertools
import sys
from types import MethodType
import unittest

import torch
from torch import nn
import torch.nn.functional as F

import test_reference_audio_generate
import test_generate_advanced
from test_sampling import COMFY_ROOT, NODE_PATH, load_definitions

sys.path.insert(0, str(COMFY_ROOT))
from comfy.cli_args import args
args.cpu = True
import comfy.model_management
from comfy.model_patcher import ModelPatcher
from comfy.ldm.modules.attention import AttentionTensorContainer, ComfyAttention, wrap_attn


@wrap_attn
def attention(q, k, v, heads, mask=None, **kwargs):
    result = F.scaled_dot_product_attention(
        q, k, v, attn_mask=mask, enable_gqa=kwargs.get("enable_gqa", False)
    )
    return result.transpose(1, 2).reshape(q.shape[0], q.shape[2], -1)


core_ns = dict(torch=torch, nn=nn, F=F, itertools=itertools,
               optimized_attention=attention, ComfyAttention=ComfyAttention,
               AttentionTensorContainer=AttentionTensorContainer)
for name in ("AceStepAudioTokenizer", "AudioTokenDetokenizer"):
    core_ns[name] = lambda *args, **kwargs: nn.Identity()
load_definitions(COMFY_ROOT / "comfy" / "ldm" / "ace" / "ace_step15.py", {
    "get_layer_class", "RotaryEmbedding", "rotate_half", "apply_rotary_pos_emb",
    "MLP", "TimestepEmbedding", "AceStepAttention", "AceStepDiTLayer",
    "AceStepEncoderLayer", "AceStepLyricEncoder", "AceStepTimbreEncoder",
    "AceStepConditionEncoder", "AceStepDiTModel", "AceStepConditionGenerationModel",
}, core_ns)
node_ns = dict(torch=torch, comfy=comfy, functools=functools, MethodType=MethodType,
               contextmanager=contextmanager, clear_conditioning_precompute=lambda model: None,
               clear_voice_profile=lambda model: None)
load_definitions(NODE_PATH, {
    "_get_acestep_diffusion_model", "_ensure_acestep_reference_attention",
    "_clear_acestep_reference_attention", "_clear_acestep_native_task_patches",
    "_clear_acestep_timbre_cls_patch", "_acestep_sampling_model",
    "_clear_acestep_reference_timbre_repeats",
    "_clear_acestep_silence_latent_patch",
}, node_ns)

PATCH_PREFIX = "diffusion_model.encoder.timbre_encoder.layers."
PATCH_STATE = "_acestep_sft_reference_attention_patches"


class ReferenceAttentionTests(unittest.TestCase):
    def model(self, **options):
        torch.manual_seed(41)
        diffusion = core_ns["AceStepConditionGenerationModel"](
            hidden_size=16, text_hidden_dim=8, timbre_hidden_dim=4,
            num_dit_layers=4, num_lyric_layers=4, num_timbre_layers=4,
            num_heads=2, num_kv_heads=1, head_dim=8, intermediate_size=32,
            encoder_hidden_size=16, encoder_intermediate_size=32,
            encoder_num_heads=2, operations=nn, **options,
        )
        for value in diffusion.parameters():
            nn.init.uniform_(value, -.3, .3)
        base = nn.Module()
        base.diffusion_model = diffusion
        base.other = "original"
        return ModelPatcher(base, torch.device("cpu"), torch.device("cpu")), diffusion

    def windows(self, encoder):
        return [layer.self_attn.sliding_window for layer in encoder.layers]

    def test_default_sampling_context_leaves_attention_and_weights_exactly_unchanged(self):
        patcher, diffusion = self.model()
        weights = {name: value.clone() for name, value in diffusion.state_dict().items()}
        original = self.windows(diffusion.encoder.timbre_encoder)
        with node_ns["_acestep_sampling_model"](patcher) as clone:
            self.assertEqual(clone.object_patches, {})
            self.assertNotIn(PATCH_STATE, clone.model_options)
        self.assertEqual(self.windows(diffusion.encoder.timbre_encoder), original)
        self.assertEqual(patcher.object_patches, {})
        for name, value in diffusion.state_dict().items():
            torch.testing.assert_close(value, weights[name], rtol=0, atol=0)

    def test_configured_decoder_windows_patch_both_encoders_only_in_the_clone(self):
        patcher, diffusion = self.model(sliding_window=4)
        timbre = diffusion.encoder.timbre_encoder
        lyric = diffusion.encoder.lyric_encoder
        weights = {name: value.clone() for name, value in diffusion.state_dict().items()}
        original_forward = timbre.forward
        clone = patcher.clone()
        node_ns["_ensure_acestep_reference_attention"](clone)
        self.assertEqual(self.windows(timbre), [None] * 4)
        self.assertEqual(self.windows(lyric), [None] * 4)
        self.assertEqual(patcher.object_patches, {})
        self.assertEqual(set(clone.object_patches), {
            f"diffusion_model.encoder.{name}.layers.{index}.self_attn.{property_name}"
            for name in ("lyric_encoder", "timbre_encoder")
            for index in range(4)
            for property_name in ("sliding_window", "forward")
        })
        clone.patch_model(load_weights=False)
        try:
            self.assertEqual(self.windows(timbre), [4, None, 4, None])
            self.assertEqual(self.windows(lyric), [4, None, 4, None])
            self.assertEqual(self.windows(diffusion.decoder), [4, None, 4, None])
            self.assertEqual(timbre.forward, original_forward)
            for name, value in diffusion.state_dict().items():
                torch.testing.assert_close(value, weights[name], rtol=0, atol=0)
        finally:
            node_ns["_clear_acestep_reference_attention"](clone)
        self.assertEqual(self.windows(timbre), [None] * 4)
        self.assertEqual(self.windows(lyric), [None] * 4)
        self.assertNotIn(PATCH_STATE, clone.model_options)

    def test_real_sdpa_local_attention_ignores_distant_voice_tokens(self):
        patcher, diffusion = self.model(sliding_window=4)
        clone = patcher.clone()
        node_ns["_ensure_acestep_reference_attention"](clone)
        clone.patch_model(load_weights=False)
        try:
            original = torch.randn(2, 20, 16)
            changed = original.clone()
            changed[:, 12:] += torch.randn_like(changed[:, 12:]) * 10
            local, full = [layer.self_attn for layer in diffusion.encoder.timbre_encoder.layers[:2]]
            torch.testing.assert_close(local(original)[:, 0], local(changed)[:, 0], rtol=0, atol=0)
            self.assertFalse(torch.allclose(full(original)[:, 0], full(changed)[:, 0]))
        finally:
            node_ns["_clear_acestep_reference_attention"](clone)

    def test_short_reference_attention_remains_identical_without_a_cls_patch(self):
        patcher, diffusion = self.model(sliding_window=8)
        encoder = diffusion.encoder.timbre_encoder
        x = torch.randn(2, 5, 16)
        layer = encoder.layers[0].self_attn
        before = layer(x)
        original_forward = encoder.forward
        clone = patcher.clone()
        node_ns["_ensure_acestep_reference_attention"](clone)
        clone.patch_model(load_weights=False)
        try:
            torch.testing.assert_close(layer(x), before, rtol=0, atol=0)
            self.assertEqual(encoder.forward, original_forward)
            self.assertNotIn("diffusion_model.encoder.timbre_encoder.forward", clone.object_patches)
        finally:
            node_ns["_clear_acestep_reference_attention"](clone)

    def test_explicit_full_and_sliding_layer_configuration_is_not_hardcoded(self):
        patcher, diffusion = self.model(
            sliding_window=5, layer_types=["full_attention", "sliding_attention"]
        )
        clone = patcher.clone()
        node_ns["_ensure_acestep_reference_attention"](clone)
        clone.patch_model(load_weights=False)
        try:
            self.assertEqual(self.windows(diffusion.encoder.timbre_encoder), [None, 5, None, 5])
        finally:
            node_ns["_clear_acestep_reference_attention"](clone)

    def test_cleanup_restores_prior_property_patches_and_is_idempotent(self):
        patcher, diffusion = self.model(sliding_window=4)
        key = PATCH_PREFIX + "0.self_attn.sliding_window"
        patcher.add_object_patch(key, 7)
        patcher.add_object_patch("other", "patched")
        clone = patcher.clone()
        node_ns["_ensure_acestep_reference_attention"](clone)
        initial_patches = clone.object_patches.copy()
        node_ns["_ensure_acestep_reference_attention"](clone)
        self.assertEqual(clone.object_patches, initial_patches)
        clone.patch_model(load_weights=False)
        node_ns["_clear_acestep_reference_attention"](clone)
        node_ns["_clear_acestep_reference_attention"](clone)
        self.assertEqual(clone.object_patches, {key: 7, "other": "patched"})
        self.assertEqual(patcher.object_patches, {key: 7, "other": "patched"})
        self.assertEqual(self.windows(diffusion.encoder.timbre_encoder), [7, None, None, None])
        self.assertEqual(patcher.model.other, "patched")
        clone.unpatch_model(unpatch_weights=False)

    def test_sampling_exception_restores_shared_attention_properties(self):
        patcher, diffusion = self.model(sliding_window=4)
        patcher.add_object_patch("other", "patched")
        with self.assertRaisesRegex(RuntimeError, "sampling failed"):
            with node_ns["_acestep_sampling_model"](patcher) as clone:
                node_ns["_ensure_acestep_reference_attention"](clone)
                clone.patch_model(load_weights=False)
                self.assertEqual(self.windows(diffusion.encoder.timbre_encoder), [4, None, 4, None])
                raise RuntimeError("sampling failed")
        self.assertEqual(self.windows(diffusion.encoder.timbre_encoder), [None] * 4)
        self.assertEqual(clone.object_patches, {"other": "patched"})
        self.assertNotIn(PATCH_STATE, clone.model_options)
        self.assertEqual(patcher.model.other, "patched")
        clone.unpatch_model(unpatch_weights=False)


class ReferenceAttentionGenerateTests(unittest.TestCase):
    def setUp(self):
        self.reference = test_reference_audio_generate.ReferenceAudioGenerateTests()
        self.reference.setUp()
        self.ns = self.reference.ns
        self.requested = []
        self.ns["_ensure_acestep_reference_attention"] = self.requested.append

    def test_default_uses_reference_attention_and_explicit_opt_out_preserves_sampling(self):
        for advanced in (False, True):
            self.requested.clear()
            original = self.reference.generate(advanced=advanced)[4]["samples"]
            self.assertEqual(len(self.requested), 1)
            self.requested.clear()
            disabled = self.reference.generate(advanced=advanced, reference_xl_attention=False)[4]["samples"]
            torch.testing.assert_close(original, disabled, rtol=0, atol=0)
            self.assertEqual(self.requested, [])
            harness = test_generate_advanced.GenerateAdvancedTests()
            harness.setUp()
            harness.ns["_ensure_acestep_reference_attention"] = self.requested.append
            if advanced:
                harness.generate()
            else:
                harness.ns["AceStepSFTGenerate"]().generate(
                    model=test_generate_advanced.AdvancedFakeModel(),
                    positive=self.reference.positive, negative=self.reference.negative,
                    seed=27, steps=50, cfg=7.0, sampler_name="jkass_quality",
                    scheduler="normal", denoise=1.0, duration=1.6,
                    infer_method="ode", guidance_mode="standard_cfg",
                )
            self.assertEqual(self.requested, [])

    def test_direct_and_native_task_references_enable_attention_independently_of_cls(self):
        for advanced in (False, True):
            for task in (None, self.reference.native.cover(),
                         self.reference.native.cover(reference_audio=None),
                         self.reference.native.repaint(reference_audio=self.reference.reference)):
                with self.subTest(advanced=advanced, task=None if task is None else task["task_type"]):
                    self.requested.clear()
                    direct = self.reference.reference if task is None else None
                    self.reference.generate(advanced=advanced, task=task, reference_audio=direct,
                        reference_xl_attention=True, timbre_cls_token=False)
                    self.assertEqual(len(self.requested), 1)
            self.requested.clear()
            self.reference.generate(advanced=advanced, reference_audio=None,
                task=self.reference.native.cover(reference_audio=None, preserve_singer_identity=False),
                reference_xl_attention=True)
            self.assertEqual(self.requested, [])

    def test_advanced_continuation_reapplies_the_attention_option(self):
        first = self.reference.generate(advanced=True, reference_xl_attention=True,
            end_at_step=25, return_with_leftover_noise="enable")[4]
        self.reference.generate(advanced=True, reference_xl_attention=True,
            latent_or_audio=first, add_noise="disable", start_at_step=25)
        self.assertEqual(len(self.requested), 2)
        self.assertTrue(torch.count_nonzero(self.reference.native.calls[-1]["noise"]) == 0)
        self.assertEqual(self.reference.native.calls[-1]["positive"][0][1]["audio_codes"],
                         self.reference.codes)


if __name__ == "__main__":
    unittest.main()
