"""Offline math/integration checks; optional comparisons to pinned upstream source."""
import ast
import math
import io
from contextlib import redirect_stderr
from tqdm.auto import trange
import os
from pathlib import Path
from typing import Union, List
import unittest

import torch
import torch.nn.functional as F

import test_sampling
from test_sampling import NODE_PATH, load_definitions


NAMES = {
    "_calculate_acestep_official_sigmas", "_prepare_gradio_noise",
    "sample_jkass_quality", "_gradio_guidance_interval", "_guidance_step_position",
    "_normalize_audio_peak", "MomentumBuffer", "_project", "apg_guidance",
    "_cos_sim", "_perpendicular", "adg_guidance",
}
ns = {"torch": torch, "math": math, "F": F, "model_trange": trange}
load_definitions(NODE_PATH, NAMES, ns)
REFERENCE = Path(os.environ.get("ACESTEP_REFERENCE_DIR", str(
    NODE_PATH.parents[3] / "_ace_gradio_alignment_20260930" / "upstream"
)))


class GradioAlignmentTests(unittest.TestCase):
    def test_shifted_schedule_matches_reference_in_model_dtype(self):
        for dtype in (torch.float32, torch.bfloat16):
            expected = torch.linspace(1, 0, 51, dtype=dtype)
            expected = 3 * expected / (1 + 2 * expected)
            actual = ns["_calculate_acestep_official_sigmas"](50, 3, dtype=dtype)
            self.assertTrue(torch.equal(actual, expected))
            self.assertEqual(len(actual), 51)

    def test_interval_bounds_are_timesteps_not_elapsed_fraction(self):
        sigmas = [1.0, .95, .8, .6, .2, 0.0]
        interval = ns["_gradio_guidance_interval"]
        self.assertEqual(interval(.95, sigmas, 1, .6, .95), (True, 0.0))
        self.assertEqual(interval(.6, sigmas, 1, .6, .95), (True, 1.0))
        self.assertFalse(interval(.2, sigmas, 1, .6, .95)[0])

    def test_noise_uses_btc_layout_and_preserves_rng(self):
        state = torch.get_rng_state().clone()
        generator = torch.Generator().manual_seed(42)
        expected = torch.randn(1, 25, 64, generator=generator, dtype=torch.bfloat16)
        actual = ns["_prepare_gradio_noise"](torch.zeros(1, 64, 25), 42, "cpu", torch.bfloat16)
        self.assertTrue(torch.equal(actual, expected.movedim(-1, 1).float()))
        self.assertTrue(torch.equal(state, torch.get_rng_state()))

    def test_batch_indices_produce_stable_variations(self):
        actual = ns["_prepare_gradio_noise"](torch.zeros(3, 64, 25), 7, "cpu", torch.float32, [0, 2, 0])
        self.assertTrue(torch.equal(actual[0], actual[2]))
        self.assertFalse(torch.equal(actual[0], actual[1]))

    def test_jkass_heun_correction_and_final_clean_prediction(self):
        calls, evaluations = [], []
        def model(x, sigma, **kwargs):
            self.assertEqual(kwargs["seed"], 123)
            evaluations.append(float(sigma[0]))
            return x - sigma[:, None, None] * (x.square() + .2 * sigma[:, None, None])
        result = ns["sample_jkass_quality"](
            model, torch.ones(2, 3, 5), torch.tensor([1., .5, 0.]),
            extra_args={"seed": 123}, callback=calls.append, disable=True,
        )
        # First Euler predictor = .4; corrected state = .635; final x0 = .3833875.
        torch.testing.assert_close(result, torch.full_like(result, .3833875))
        self.assertEqual(evaluations, [1., .5, .5])
        self.assertEqual([c["i"] for c in calls], [0, 1])
        torch.testing.assert_close(calls[1]["x"], torch.full_like(result, .635))

    def test_jkass_empty_schedule_preserves_source(self):
        x = torch.ones(1, 3, 5)
        def model(*args, **kwargs):
            self.fail("Empty schedule must not evaluate the model")
        for sigmas in (torch.tensor([]), torch.tensor([0.])):
            self.assertIs(ns["sample_jkass_quality"](model, x, sigmas), x)

    def test_sampler_choices_are_local_and_unique(self):
        from types import SimpleNamespace
        harness = test_sampling.SamplingTests()
        harness.setUp()
        harness.ns.update(GUIDANCE_MODES=["apg", "adg", "standard_cfg"],
                          ACESTEP_SFT_TASK_TYPE="ACESTEP_SFT_TASK")
        for existing in (["euler", "heun"], ["euler", "jkass_quality", "ace_step_euler"]):
            registry = list(existing)
            harness.ns["comfy"].samplers.KSampler = SimpleNamespace(
                SAMPLERS=registry, SCHEDULERS=["normal"])
            inputs = harness.ns["AceStepSFTGenerate"].INPUT_TYPES()["required"]
            choices, settings = inputs["sampler_name"]
            self.assertEqual(choices.count("jkass_quality"), 1)
            self.assertEqual(settings["default"], "jkass_quality")
            self.assertNotIn("ace_step_euler", choices)
            self.assertIn("ace_step", inputs["scheduler"][0])
            self.assertEqual(registry, existing)

    def test_console_progress_respects_disable_and_reaches_completion(self):
        def model(x, sigma, **kwargs):
            return torch.zeros_like(x)
        for disabled in (False, True):
            output = io.StringIO()
            with redirect_stderr(output):
                ns["sample_jkass_quality"](
                    model, torch.ones(1, 2, 4), torch.tensor([1., .5, 0.]),
                    disable=disabled,
                )
            if disabled:
                self.assertEqual(output.getvalue(), "")
            else:
                self.assertIn("100%", output.getvalue())
                self.assertIn("2/2", output.getvalue())

    @unittest.skipUnless(REFERENCE.exists(), "Set ACESTEP_REFERENCE_DIR for upstream comparisons")
    def test_noise_and_apg_against_official_implementation(self):
        source = REFERENCE / "acestep/models/sft/modeling_acestep_v15_base.py"
        method = next(n for n in ast.walk(ast.parse(source.read_text(encoding="utf-8")))
                      if isinstance(n, ast.FunctionDef) and n.name == "prepare_noise")
        official = {"torch": torch, "Union": Union, "List": List}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), official)
        load_definitions(REFERENCE / "acestep/models/common/apg_guidance.py",
                         {"MomentumBuffer", "project", "apg_forward", "call_cos_tensor",
                          "compute_perpendicular_component", "adg_forward"}, official)
        for device in (["cpu", "cuda:0"] if torch.cuda.is_available() else ["cpu"]):
            context = torch.zeros(1, 25, 128, device=device, dtype=torch.bfloat16)
            expected = official["prepare_noise"](None, context, 42).movedim(-1, 1).float().cpu()
            actual = ns["_prepare_gradio_noise"](torch.zeros(1, 64, 25), 42, device, torch.bfloat16)
            self.assertTrue(torch.equal(actual, expected))
        cond, uncond = torch.randn(2, 25, 64), torch.randn(2, 25, 64)
        expected = official["apg_forward"](cond, uncond, 7, dims=[1])
        actual = ns["apg_guidance"](cond.movedim(-1, 1), uncond.movedim(-1, 1), 7)
        torch.testing.assert_close(actual.movedim(-1, 1), expected)
        latent = torch.randn_like(cond)
        expected = official["adg_forward"](latent, cond, uncond, .7, 7)
        actual = ns["adg_guidance"](latent, cond, uncond, .7, 7)
        torch.testing.assert_close(actual, expected)

    def test_generate_routes_local_sampler_with_official_and_comfy_schedules(self):
        harness = test_sampling.SamplingTests()
        harness.setUp()
        harness.ns.update(ns)
        original_generate = harness.ns["AceStepSFTGenerate"]
        captured = {}
        def sample_custom(model, noise, cfg, sampler, sigmas, positive, negative, latent, **kwargs):
            captured.update(sampler=sampler, sigmas=sigmas, noise=noise, model=model)
            return latent + 1
        harness.ns["comfy"].sample.sample_custom = sample_custom
        harness.ns["comfy"].samplers.KSAMPLER = lambda fn, options: (fn, options)
        from test_sampling import FakeModel
        model = FakeModel()
        model.load_device = "cpu"
        model.model_dtype = lambda: torch.bfloat16
        original_clone = model.clone
        def clone():
            cloned = original_clone()
            cloned.load_device, cloned.model_dtype = model.load_device, model.model_dtype
            return cloned
        model.clone = clone
        for scheduler in ("ace_step", "normal"):
            for infer_method in ("ode", "sde"):
                with self.subTest(scheduler=scheduler, infer_method=infer_method):
                    result = harness.generate(model=model, sampler_name="jkass_quality", scheduler=scheduler,
                                              noise_source="official", infer_method=infer_method,
                                              schedule_precision="model")
                    self.assertIs(result[0], model)
                    self.assertIs(captured["sampler"][0], ns["sample_jkass_quality"])
                    self.assertEqual(captured["sampler"][1], {})
                    expected = (ns["_calculate_acestep_official_sigmas"](50, 3, dtype=torch.bfloat16).float()
                                if scheduler == "ace_step" else torch.linspace(1, 0, 51))
                    torch.testing.assert_close(captured["sigmas"], expected)
                    self.assertIs(harness.ns["AceStepSFTGenerate"], original_generate)



if __name__ == "__main__":
    unittest.main()
