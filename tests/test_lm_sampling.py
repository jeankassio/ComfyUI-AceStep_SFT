"""Check native ACE-Step audio-token sampling without loading the LM."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch


SOURCE = Path(__file__).resolve().parents[3] / "comfy" / "text_encoders" / "ace15.py"
tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
ns = {"torch": torch}
exec(compile(ast.Module(body=[node for node in tree.body
     if getattr(node, "name", None) == "_sample_audio_token"], type_ignores=[]), str(SOURCE), "exec"), ns)


class LMSamplingTests(unittest.TestCase):
    def sample(self, logits, eos=None, **overrides):
        ns["_audio_logits"] = lambda *args: (logits.clone(), None if eos is None else eos.clone())
        options = dict(cfg_scale=2.0, temperature=.85, top_p=1.0,
                       top_k=0, min_p=0, generator=torch.Generator().manual_seed(13),
                       audio_start_id=2, audio_end_id=2 + logits.shape[1],
                       eos_token_id=1 if eos is not None else None)
        options.update(overrides)
        return ns["_sample_audio_token"](SimpleNamespace(vocab_size=2 + logits.shape[1]), None, **options)

    def capture_probabilities(self, logits, **kwargs):
        captured = []

        def multinomial(probabilities, **unused):
            captured.append(probabilities.clone())
            return probabilities.argmax(dim=-1, keepdim=True)

        with patch("torch.multinomial", side_effect=multinomial):
            self.sample(logits, **kwargs)
        return captured[0]

    def test_half_precision_cfg_and_temperature_do_not_overflow(self):
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                logits = torch.tensor([[60000, -60000, 3, 2],
                                       [-60000, 60000, -1, -2]], dtype=dtype)
                actual = self.capture_probabilities(logits, temperature=.01)
                self.assertEqual(actual.dtype, torch.float32)
                self.assertTrue(torch.isfinite(actual).all())
                self.assertEqual(actual[0, 2].item(), 1.0)

    def test_sampling_matches_float32_cfg_reference(self):
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                logits = torch.tensor([[1.1, .5, -.2, 2.1],
                                       [.3, .8, 1.2, -.4]], dtype=dtype)
                actual = self.capture_probabilities(logits, temperature=.6)
                expected = torch.softmax((logits[1:2].float() +
                    2 * (logits[0:1].float() - logits[1:2].float())) / .6, dim=-1)
                torch.testing.assert_close(actual[:, 2:], expected)
                self.assertTrue(torch.equal(actual[:, :2], torch.zeros(1, 2)))

    def test_top_p_zero_disables_filtering_like_official_backend(self):
        logits = torch.tensor([[0., 1., 2., 3.], [0., 0., 0., 0.]])
        actual = self.capture_probabilities(logits, top_p=0)
        reference = self.capture_probabilities(logits, top_p=1)
        torch.testing.assert_close(actual, reference)
        self.assertEqual(torch.count_nonzero(actual).item(), 4)

    def test_eos_cfg_score_also_uses_float32(self):
        logits = torch.tensor([[3., 2., 1.], [0., 0., 0.]], dtype=torch.float16)
        eos = torch.tensor([[60000.], [-60000.]], dtype=torch.float16)
        actual = self.capture_probabilities(logits, eos=eos)
        self.assertTrue(torch.isfinite(actual).all())
        self.assertEqual(actual[0, 1].item(), 1.0)

    def test_seeded_sampling_and_greedy_mode_remain_valid(self):
        logits = torch.tensor([[0., 1., 2., 3.], [1., 0., -1., 2.]])
        first = self.sample(logits)
        second = self.sample(logits)
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(2 <= first.item() < 6)
        greedy = self.sample(logits, temperature=0)
        expected = 2 + (logits[1] + 2 * (logits[0] - logits[1])).argmax()
        self.assertEqual(greedy.item(), expected.item())


if __name__ == "__main__":
    unittest.main()
