"""Regression coverage for ACE-Step's configured lyric/timbre attention windows."""
import ast
import copy
import itertools
from pathlib import Path
import unittest

import torch
from torch import nn
import torch.nn.functional as F


SOURCE = Path(__file__).resolve().parents[3] / "comfy/ldm/ace/ace_step15.py"


def attention(q, k, v, heads, mask=None, **kwargs):
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                        enable_gqa=kwargs.get("enable_gqa", False))
    return out.transpose(1, 2).reshape(q.shape[0], q.shape[2], -1)


names = {
    "get_layer_class", "RotaryEmbedding", "rotate_half", "apply_rotary_pos_emb",
    "MLP", "AceStepAttention", "AceStepEncoderLayer", "AceStepLyricEncoder",
    "AceStepTimbreEncoder", "AceStepConditionEncoder", "AceStepConditionGenerationModel",
}
ns = {"torch": torch, "nn": nn, "F": F, "itertools": itertools,
      "optimized_attention": attention}
# Only the conditioning encoder is exercised; the other large components are
# placeholders so the real top-level constructor can be checked on CPU.
for name in ("AceStepDiTModel", "AceStepAudioTokenizer", "AudioTokenDetokenizer"):
    ns[name] = lambda *args, **kwargs: nn.Identity()
tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
exec(compile(ast.Module(body=[n for n in tree.body if getattr(n, "name", None) in names],
                        type_ignores=[]), str(SOURCE), "exec"), ns)


class EncoderAttentionTests(unittest.TestCase):
    def model(self, **options):
        torch.manual_seed(41)
        return ns["AceStepConditionGenerationModel"](
            hidden_size=16, text_hidden_dim=8, timbre_hidden_dim=4,
            num_dit_layers=4, num_lyric_layers=4, num_timbre_layers=4,
            num_heads=2, num_kv_heads=1, head_dim=8, intermediate_size=32,
            encoder_hidden_size=16, encoder_intermediate_size=32,
            encoder_num_heads=2, operations=nn, **options,
        )

    def test_default_model_propagates_official_window_to_both_encoders(self):
        model = self.model()
        for encoder in (model.encoder.lyric_encoder, model.encoder.timbre_encoder):
            self.assertEqual([layer.self_attn.sliding_window for layer in encoder.layers],
                             [128, None, 128, None])

    def test_explicit_layer_types_and_window_are_preserved(self):
        model = self.model(layer_types=["full_attention", "sliding_attention"] * 2,
                           sliding_window=4)
        for encoder in (model.encoder.lyric_encoder, model.encoder.timbre_encoder):
            self.assertEqual([layer.self_attn.sliding_window for layer in encoder.layers],
                             [None, 4, None, 4])

    def test_local_layer_ignores_distant_lyric_tokens_but_global_layer_reads_them(self):
        model = self.model(sliding_window=4)
        x = torch.randn(1, 20, 16)
        changed = x.clone()
        changed[:, 12:] += torch.randn_like(changed[:, 12:]) * 10
        local, global_ = [layer.self_attn for layer in model.encoder.lyric_encoder.layers[:2]]
        torch.testing.assert_close(local(x)[:, 0], local(changed)[:, 0])
        self.assertFalse(torch.allclose(global_(x)[:, 0], global_(changed)[:, 0]))

    def test_short_sequences_match_full_attention(self):
        model = self.model(sliding_window=4)
        local = model.encoder.lyric_encoder.layers[0].self_attn
        global_ = copy.deepcopy(local)
        global_.sliding_window = None
        x = torch.randn(1, 5, 16)
        torch.testing.assert_close(local(x), global_(x))


if __name__ == "__main__":
    unittest.main()
