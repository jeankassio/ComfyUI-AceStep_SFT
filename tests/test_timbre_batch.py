"""Ensure native text-to-music batches keep their own timbre references."""
import ast
from pathlib import Path
import unittest

import torch
import torch.nn.functional as F


SOURCE = Path(__file__).resolve().parents[3] / "comfy" / "ldm" / "ace" / "ace_step15.py"
tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
methods = []
for class_name, method_name in (("AceStepConditionGenerationModel", "forward"),
                                ("AceStepTimbreEncoder", "unpack_timbre_embeddings")):
    cls = next(n for n in tree.body if getattr(n, "name", None) == class_name)
    methods.append(next(n for n in cls.body if getattr(n, "name", None) == method_name))
ns = {"torch": torch, "F": F}
exec(compile(ast.Module(body=methods, type_ignores=[]), str(SOURCE), "exec"), ns)


class ReferenceModel:
    forward = ns["forward"]

    def prepare_condition(self, text, text_mask, lyrics, lyric_mask, references,
                          order, source, chunks, covers, **kwargs):
        timbre, mask = ns["unpack_timbre_embeddings"](None, references[:, 0, :], order)
        return timbre, mask, source

    def decoder(self, hidden_states, encoder_hidden_states, **kwargs):
        return encoder_hidden_states.sum(dim=1).unsqueeze(1).expand_as(hidden_states)


class TimbreBatchTests(unittest.TestCase):
    def test_batched_reference_conditioning_matches_individual_samples(self):
        model = ReferenceModel()
        for batch_size in (1, 2, 4):
            with self.subTest(batch_size=batch_size):
                x = torch.zeros(batch_size, 64, 10)
                timbre = torch.arange(1, batch_size + 1).float().reshape(-1, 1, 1).expand_as(x)
                context = torch.ones(batch_size, 3, 4)
                actual = model.forward(x, torch.ones(batch_size), context, refer_audio=timbre)
                expected = torch.cat([
                    model.forward(x[i:i+1], torch.ones(1), context[i:i+1], refer_audio=timbre[i:i+1])
                    for i in range(batch_size)
                ])
                torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
