"""Exercise the scoped timbre CLS fix with ComfyUI's encoder and patcher."""
import ast
from contextlib import contextmanager
import functools
from pathlib import Path
import sys
from types import MethodType
import unittest

import torch
from torch import nn


SOURCE = Path(__file__).resolve().parents[1] / "nodes.py"
sys.path.insert(0, str(SOURCE.parents[2]))
from comfy.cli_args import args
args.cpu = True
import comfy.model_management
import comfy.ops
from comfy.ldm.ace.ace_step15 import AceStepTimbreEncoder
from comfy.model_patcher import ModelPatcher


names = {
    "_ensure_acestep_timbre_cls", "_clear_acestep_timbre_cls_patch",
    "_clear_acestep_native_task_patches", "_acestep_sampling_model",
    "_clear_acestep_reference_attention",
    "_clear_acestep_reference_timbre_repeats",
    "_clear_acestep_silence_latent_patch",
}
tree = ast.parse(SOURCE.read_text(encoding="utf-8-sig"))
ns = dict(torch=torch, comfy=comfy, functools=functools, MethodType=MethodType,
          contextmanager=contextmanager, clear_conditioning_precompute=lambda model: None,
          clear_voice_profile=lambda model: None)
exec(compile(ast.Module(body=[node for node in tree.body
                             if getattr(node, "name", None) in names],
                        type_ignores=[]), str(SOURCE), "exec"), ns)

PATCH_KEY = "diffusion_model.encoder.timbre_encoder.forward"


class RecordingLayer(nn.Module):
    def forward(self, hidden_states, position_embeddings, attention_mask=None):
        self.hidden_states = hidden_states.clone()
        self.position_embeddings = position_embeddings
        self.attention_mask = attention_mask
        return hidden_states


class TimbreCLSTests(unittest.TestCase):
    def model(self, layers=1, dtype=torch.float32):
        torch.manual_seed(19)
        encoder = AceStepTimbreEncoder(
            8, 8, layers, 2, 1, 4, 16, dtype=dtype,
            operations=comfy.ops.disable_weight_init,
        )
        for parameter in encoder.parameters():
            nn.init.uniform_(parameter, -.3, .3)
        base = nn.Module()
        base.diffusion_model = nn.Module()
        base.diffusion_model.encoder = nn.Module()
        base.diffusion_model.encoder.timbre_encoder = encoder
        base.other = "original"
        patcher = ModelPatcher(base, torch.device("cpu"), torch.device("cpu"))
        return patcher, encoder

    def test_pooling_uses_trained_cls_and_extends_padding_mask_before_rope(self):
        patcher, encoder = self.model(layers=0)
        layer = RecordingLayer()
        encoder.layers.append(layer)
        references = torch.randn(2, 5, 8)
        order = torch.tensor([0, 1])
        for dtype in (torch.bool, torch.int64, torch.float32):
            with self.subTest(mask_dtype=dtype):
                mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]], dtype=dtype)
                clone = patcher.clone()
                original = encoder.forward
                ns["_ensure_acestep_timbre_cls"](clone)
                self.assertEqual(encoder.forward, original)
                clone.patch_model(load_weights=False)
                result, unpacked_mask = encoder(references, order, attention_mask=mask)
                expected_cls = encoder.norm(encoder.special_token).expand(2, 1, -1)
                torch.testing.assert_close(result, expected_cls)
                torch.testing.assert_close(unpacked_mask, torch.ones(2, 1, dtype=torch.long))
                torch.testing.assert_close(layer.hidden_states[:, :1], encoder.special_token.expand(2, 1, -1))
                torch.testing.assert_close(layer.hidden_states[:, 1:], encoder.embed_tokens(references))
                self.assertEqual(layer.position_embeddings[0].shape[0], 6)
                self.assertEqual(layer.attention_mask.dtype, dtype)
                torch.testing.assert_close(layer.attention_mask[:, 0], torch.ones(2, dtype=dtype))
                torch.testing.assert_close(layer.attention_mask[:, 1:], mask)
                ns["_clear_acestep_timbre_cls_patch"](clone)
                self.assertEqual(encoder.forward, original)

    def test_real_attention_batches_match_individual_and_respect_reference_order(self):
        patcher, encoder = self.model()
        references = torch.randn(3, 7, 8)
        clone = patcher.clone()
        ns["_ensure_acestep_timbre_cls"](clone)
        clone.patch_model(load_weights=False)
        try:
            order = torch.tensor([2, 0, 1])
            batched, mask = encoder(references, order)
            individual = torch.cat([encoder(references[i:i + 1], torch.zeros(1, dtype=torch.long))[0]
                                    for i in (1, 2, 0)])
            torch.testing.assert_close(batched, individual)
            torch.testing.assert_close(mask, torch.ones(3, 1, dtype=torch.long))
            before = batched.clone()
            nn.init.constant_(encoder.special_token, .9)
            after, _ = encoder(references, order)
            self.assertFalse(torch.allclose(before, after))
        finally:
            ns["_clear_acestep_timbre_cls_patch"](clone)

    def test_raw_cls_parameter_is_cast_to_compute_dtype_without_changing_storage(self):
        patcher, encoder = self.model(layers=0, dtype=torch.bfloat16)
        encoder.special_token = nn.Parameter(encoder.special_token.float())
        cls_before = encoder.special_token.clone()
        layer = RecordingLayer()
        encoder.layers.append(layer)
        clone = patcher.clone()
        ns["_ensure_acestep_timbre_cls"](clone)
        clone.patch_model(load_weights=False)
        try:
            encoder(torch.randn(1, 3, 8, dtype=torch.bfloat16), torch.zeros(1, dtype=torch.long))
            self.assertEqual(layer.hidden_states.dtype, torch.bfloat16)
            self.assertEqual(encoder.special_token.dtype, torch.float32)
            torch.testing.assert_close(encoder.special_token, cls_before)
        finally:
            ns["_clear_acestep_timbre_cls_patch"](clone)

    def test_idempotent_patch_and_cleanup_preserve_existing_object_patch_and_weights(self):
        patcher, encoder = self.model(layers=0)
        def previous_forward(self, references, order, attention_mask=None):
            return references, order
        previous = MethodType(previous_forward, encoder)
        patcher.add_object_patch(PATCH_KEY, previous)
        patcher.add_object_patch("other", "patched")
        weights = {name: value.clone() for name, value in encoder.state_dict().items()}
        clone = patcher.clone()
        ns["_ensure_acestep_timbre_cls"](clone)
        added = clone.object_patches[PATCH_KEY]
        ns["_ensure_acestep_timbre_cls"](clone)
        self.assertIs(clone.object_patches[PATCH_KEY], added)
        clone.patch_model(load_weights=False)
        ns["_clear_acestep_timbre_cls_patch"](clone)
        self.assertIs(clone.object_patches[PATCH_KEY], previous)
        self.assertIs(encoder.forward, previous)
        self.assertEqual(patcher.model.other, "patched")
        self.assertIs(patcher.object_patches[PATCH_KEY], previous)
        for name, value in encoder.state_dict().items():
            torch.testing.assert_close(value, weights[name])
        clone.unpatch_model(unpatch_weights=False)

    def test_sampling_exception_restores_shared_encoder_and_unrelated_patches(self):
        patcher, encoder = self.model(layers=0)
        patcher.add_object_patch("other", "patched")
        original = encoder.forward
        with self.assertRaisesRegex(RuntimeError, "sampling failed"):
            with ns["_acestep_sampling_model"](patcher) as clone:
                ns["_ensure_acestep_timbre_cls"](clone)
                clone.patch_model(load_weights=False)
                raise RuntimeError("sampling failed")
        self.assertEqual(encoder.forward, original)
        self.assertNotIn(PATCH_KEY, clone.object_patches)
        self.assertNotIn(PATCH_KEY, patcher.object_patches)
        self.assertEqual(patcher.model.other, "patched")
        clone.unpatch_model(unpatch_weights=False)


if __name__ == "__main__":
    unittest.main()
