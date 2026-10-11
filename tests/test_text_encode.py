"""Exercise text/audio planning without importing ComfyUI or loading models."""

import ast
import copy
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
import yaml


SOURCE = Path(__file__).resolve().parents[1] / "nodes.py"


def load_encoding_nodes():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8-sig"))
    functions = {
        "_build_null_negative", "_clone_processed_cond_value",
        "_estimate_duration_from_lyrics", "_is_audio_payload",
        "_is_latent_payload", "_get_source_duration_seconds",
        "_get_source_latent_metadata",
        "_retime_lm_codes_conditioning", "_retime_cover_conditioning",
        "_select_cover_conditioning_branch", "_sigma_to_percent",
    }
    selected = [
        node for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name in functions)
        or (isinstance(node, ast.ClassDef) and node.name == "AceStepSFTTextEncode")
        or (isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "ACESTEP_INSTRUCTIONS"
            for target in node.targets
        ))
    ]
    generate_class = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "AceStepSFTGenerate"
    )
    generate = copy.deepcopy(next(
        node for node in generate_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "generate"
    ))
    # Execute the real preflight and duration selection, stopping before model
    # loading, latent creation, or GPU sampling.
    duration_end = next(
        index for index, node in enumerate(generate.body)
        if isinstance(node, ast.If) and any(
            isinstance(child, ast.Name) and child.id == "latent_length"
            and isinstance(child.ctx, ast.Store)
            for child in ast.walk(node)
        )
    )
    generate.name = "resolve_generation_duration"
    generate.body = generate.body[:duration_end + 1] + [
        ast.parse("return duration, latent_length, batch_size").body[0]
    ]
    selected.append(generate)
    namespace = {
        "math": math, "torch": torch, "yaml": yaml,
        "_validate_acestep_task": lambda task: task,
        "node_helpers": SimpleNamespace(conditioning_set_values=lambda values, extras: [
            [embedding, {**metadata, **extras}] for embedding, metadata in values
        ]),
    }
    exec(compile(ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[])),
                 str(SOURCE), "exec"), namespace)
    return namespace


class FakeClip:
    def __init__(self):
        self.tokenizer = SimpleNamespace(qwen3_06b=SimpleNamespace(
            tokenize_with_weights=lambda text, *args, **kwargs: text
        ))
        self.encodings = []

    def tokenize(self, caption, **kwargs):
        return {"lm_metadata": kwargs}

    def encode_from_tokens_scheduled(self, tokens):
        self.encodings.append(tokens)
        metadata = {"conditioning_lyrics": torch.ones(1, 2, 4)}
        if tokens["lm_metadata"]["generate_audio_codes"]:
            metadata["audio_codes"] = [[1, 2, 3]]
        return [[torch.ones(1, 3, 4), metadata]]


class TextEncodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nodes = load_encoding_nodes()

    def encode(self, **overrides):
        clip = FakeClip()
        kwargs = dict(
            clip=clip, caption="acoustic rock", lyrics="[Instrumental]",
            instrumental=True, seed=123, duration=60.0, bpm=120,
            timesignature="4", language="en", keyscale="C major",
        )
        kwargs.update(overrides)
        positive, negative = self.nodes["AceStepSFTTextEncode"]().encode(**kwargs)
        return clip, positive, negative

    def duration(self, positive, **overrides):
        kwargs = dict(
            self=None, model=SimpleNamespace(model_options={}), positive=positive,
            negative=positive, seed=123, steps=50, cfg=7,
            sampler_name="euler", scheduler="normal", denoise=1.0,
            duration=0, infer_method="ode", guidance_mode="standard_cfg",
        )
        kwargs.update(overrides)
        return self.nodes["resolve_generation_duration"](**kwargs)

    def test_lm_enabled_preserves_official_prompt_and_null_cfg(self):
        clip, positive, negative = self.encode()
        tokens = clip.encodings[0]
        instruction = "Generate audio semantic tokens based on the given conditions:"
        self.assertIn(instruction, tokens["lm_prompt"])
        self.assertTrue(tokens["lm_prompt"].endswith("</think>\n\n"))
        self.assertIn("user\nNO USER INPUT<|im_end|>", tokens["lm_prompt_negative"])
        self.assertIn(instruction, tokens["qwen3_06b"])
        self.assertEqual(torch.count_nonzero(negative[0][0]).item(), 0)
        self.assertEqual(positive[0][1]["audio_codes"], negative[0][1]["audio_codes"])

    def test_lm_disabled_uses_text2music_instruction(self):
        clip, positive, _ = self.encode(generate_audio_codes=False)
        self.assertIn("Fill the audio semantic mask based on the given conditions:",
                      clip.encodings[0]["qwen3_06b"])
        self.assertNotIn("audio_codes", positive[0][1])

    def test_full_lm_strength_preserves_single_conditioning_branch(self):
        clip, positive, negative = self.encode(lm_codes_strength=1.0)
        self.assertEqual(len(clip.encodings), 1)
        for conditioning in (positive, negative):
            self.assertEqual(len(conditioning), 1)
            self.assertIn("audio_codes", conditioning[0][1])
            self.assertNotIn("acestep_sft_lm_codes_strength", conditioning[0][1])
            self.assertNotIn("end_percent", conditioning[0][1])

    def test_zero_lm_strength_skips_code_generation(self):
        clip, positive, negative = self.encode(lm_codes_strength=0.0)
        self.assertEqual(len(clip.encodings), 1)
        self.assertFalse(clip.encodings[0]["lm_metadata"]["generate_audio_codes"])
        self.assertIn("Fill the audio semantic mask", clip.encodings[0]["qwen3_06b"])
        for conditioning in (positive, negative):
            self.assertNotIn("audio_codes", conditioning[0][1])
            self.assertNotIn("acestep_sft_lm_codes_strength", conditioning[0][1])

    def test_partial_lm_strength_encodes_aligned_code_and_native_branches(self):
        clip, positive, negative = self.encode(lm_codes_strength=0.75)
        self.assertEqual(len(clip.encodings), 2)
        self.assertTrue(clip.encodings[0]["lm_metadata"]["generate_audio_codes"])
        self.assertFalse(clip.encodings[1]["lm_metadata"]["generate_audio_codes"])
        self.assertIn("Generate audio semantic tokens", clip.encodings[0]["qwen3_06b"])
        self.assertIn("Fill the audio semantic mask", clip.encodings[1]["qwen3_06b"])
        for conditioning in (positive, negative):
            self.assertEqual(len(conditioning), 2)
            self.assertIn("audio_codes", conditioning[0][1])
            self.assertNotIn("audio_codes", conditioning[1][1])
            self.assertEqual(conditioning[0][1]["end_percent"], 0.75)
            self.assertEqual(conditioning[1][1]["start_percent"], 0.75)
            for _, metadata in conditioning:
                self.assertEqual(metadata["acestep_sft_lm_codes_strength"], 0.75)
                self.assertEqual(metadata["acestep_sft_duration"], 60)
        self.assertEqual(positive[0][1]["audio_codes"], negative[0][1]["audio_codes"])
        for embedding, _ in negative:
            self.assertEqual(torch.count_nonzero(embedding).item(), 0)

    def test_partial_strength_is_ignored_when_codes_disabled(self):
        clip, positive, _ = self.encode(generate_audio_codes=False, lm_codes_strength=0.5)
        self.assertEqual(len(clip.encodings), 1)
        self.assertNotIn("acestep_sft_lm_codes_strength", positive[0][1])

    def test_lm_strength_counts_actual_steps_instead_of_sigma_distance(self):
        _, positive, negative = self.encode(lm_codes_strength=0.5)
        model_sampling = SimpleNamespace(percent_to_sigma=lambda percent: 1.0 - percent)
        sigmas = torch.tensor([1.0, 0.9, 0.6, 0.2, 0.0])
        for conditioning in (positive, negative):
            retimed = self.nodes["_retime_lm_codes_conditioning"](
                conditioning, sigmas, model_sampling
            )
            self.assertAlmostEqual(retimed[0][1]["end_percent"], 0.25)
            self.assertAlmostEqual(retimed[1][1]["start_percent"], 0.25)
            self.assertEqual(conditioning[0][1]["end_percent"], 0.5)

    def test_partial_strength_rounding_to_zero_selects_native_branch(self):
        _, positive, _ = self.encode(lm_codes_strength=0.2)
        retimed = self.nodes["_retime_lm_codes_conditioning"](
            positive, torch.tensor([1.0, 0.8, 0.4, 0.1, 0.0]), None
        )
        self.assertEqual(len(retimed), 1)
        self.assertNotIn("audio_codes", retimed[0][1])
        self.assertNotIn("start_percent", retimed[0][1])
        self.assertNotIn("end_percent", retimed[0][1])

    def test_legacy_conditioning_is_not_retimed(self):
        _, positive, _ = self.encode()
        retimed = self.nodes["_retime_lm_codes_conditioning"](
            positive, torch.tensor([1.0, 0.0]), None
        )
        self.assertIs(retimed, positive)

    def test_invalid_lm_sampling_parameters_fail_before_encoding(self):
        for name, value in (
            ("lm_cfg_scale", -1), ("lm_cfg_scale", float("inf")),
            ("lm_temperature", float("nan")), ("lm_temperature", -0.1),
            ("lm_top_p", 2000), ("lm_top_p", -0.1),
            ("lm_top_k", -1), ("lm_min_p", 1.1),
            ("lm_codes_strength", -0.1), ("lm_codes_strength", 1.1),
            ("lm_codes_strength", float("nan")),
        ):
            with self.subTest(name=name, value=value), self.assertRaisesRegex(ValueError, name):
                self.encode(**{name: value})

    def test_greedy_and_unfiltered_lm_settings_are_preserved(self):
        clip, _, _ = self.encode(lm_temperature=0, lm_top_p=0)
        metadata = clip.encodings[0]["lm_metadata"]
        self.assertEqual(metadata["temperature"], 0)
        self.assertEqual(metadata["top_p"], 0)

    def test_lm_code_budget_uses_exact_duration_before_prompt_rounding(self):
        for duration, count in ((60, 300), (60.1, 300), (60.2, 301), (60.9, 304)):
            with self.subTest(duration=duration):
                clip, positive, _ = self.encode(duration=duration)
                metadata = clip.encodings[0]["lm_metadata"]
                self.assertEqual(metadata["min_tokens"], count)
                self.assertEqual(metadata["max_tokens"], count)
                self.assertEqual(positive[0][1]["acestep_sft_duration"], duration)
                self.assertEqual(metadata["duration"], duration)

    def test_nonfinite_encoding_duration_is_rejected(self):
        for duration in (float("nan"), float("inf")):
            with self.subTest(duration=duration), self.assertRaisesRegex(ValueError, "duration"):
                self.encode(duration=duration)

    def test_style_bpm_keeps_requested_duration(self):
        clip, positive, negative = self.encode(style_bpm=90)
        self.assertEqual(clip.encodings[0]["lm_metadata"]["bpm"], 90)
        self.assertEqual(clip.encodings[0]["lm_metadata"]["duration"], 60)
        for conditioning in (positive, negative):
            self.assertEqual(conditioning[0][1]["acestep_sft_duration"], 60)

    def test_auto_duration_is_shared_with_generate(self):
        clip, positive, negative = self.encode(duration=0)
        self.assertEqual(clip.encodings[0]["lm_metadata"]["duration"], 90)
        self.assertEqual(negative[0][1]["acestep_sft_duration"], 90)
        self.assertEqual(self.duration(positive), (90, 2250, 1))

    def test_explicit_generate_duration_has_precedence(self):
        _, positive, _ = self.encode(duration=90)
        self.assertEqual(self.duration(positive, duration=45), (45, 1125, 1))

    def test_audio_source_has_precedence_in_auto_mode(self):
        _, positive, _ = self.encode(duration=90)
        source = {"waveform": torch.zeros(1, 2, 1700), "sample_rate": 100}
        self.assertEqual(self.duration(positive, latent_or_audio=source), (17, 425, 1))

    def test_latent_source_controls_length_and_batch(self):
        _, positive, _ = self.encode(duration=90)
        source = {"samples": torch.zeros(2, 64, 250)}
        self.assertEqual(self.duration(positive, latent_or_audio=source), (10, 250, 2))

    def test_legacy_conditioning_without_duration_keeps_fallback(self):
        self.assertEqual(self.duration([[torch.ones(1), {}]]), (60, 1500, 1))

    def test_invalid_numeric_duration_keeps_fallback(self):
        for duration in (float("nan"), float("inf"), -1, 0):
            with self.subTest(duration=duration):
                conditioning = [[torch.ones(1), {"acestep_sft_duration": duration}]]
                self.assertEqual(self.duration(conditioning), (60, 1500, 1))

    def test_cover_branches_keep_task_and_duration_metadata(self):
        source = {"waveform": torch.zeros(1, 2, 500), "sample_rate": 100}
        task = {"task_type": "cover", "source": source, "audio_cover_strength": 0.5}
        clip, positive, negative = self.encode(task=task, lm_codes_strength=0.0)
        self.assertEqual(len(clip.encodings), 2)
        self.assertIn("Generate audio semantic tokens", clip.encodings[0]["qwen3_06b"])
        self.assertIn("Fill the audio semantic mask", clip.encodings[1]["qwen3_06b"])
        for conditioning in positive + negative:
            self.assertEqual(conditioning[1]["acestep_sft_duration"], 5)
            self.assertEqual(conditioning[1]["acestep_sft_task_type"], "cover")
            self.assertNotIn("audio_codes", conditioning[1])
            self.assertNotIn("acestep_sft_lm_codes_strength", conditioning[1])
        self.assertEqual(self.duration(positive, task=task, duration=90), (5, 125, 1))

    def test_repaint_keeps_its_instruction(self):
        source = {"waveform": torch.zeros(1, 2, 500), "sample_rate": 100}
        task = {"task_type": "repaint", "source": source}
        clip, positive, _ = self.encode(task=task)
        self.assertIn("Repaint the mask area", clip.encodings[0]["qwen3_06b"])
        self.assertEqual(positive[0][1]["acestep_sft_duration"], 5)


if __name__ == "__main__":
    unittest.main()
