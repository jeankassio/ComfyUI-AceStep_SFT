"""CPU regressions for flow-schedule precision and the Generate sampler route."""
import copy
import math
from types import SimpleNamespace
import unittest

import torch

import test_sampling as sampling


class ScheduleModel(sampling.FakeModel):
    def __init__(self, dtype=torch.bfloat16):
        super().__init__("acestep_v1.5_sft_xl.safetensors")
        self.load_device = "cpu"
        self.dtype = dtype

    def model_dtype(self):
        return self.dtype

    def clone(self):
        cloned = ScheduleModel(self.dtype)
        cloned.model_options = self.model_options.copy()
        return cloned


class SchedulePrecisionTests(unittest.TestCase):
    def setUp(self):
        self.harness = sampling.SamplingTests("runTest")
        self.harness.setUp()
        self.ns = self.harness.ns
        sampling.load_definitions(
            sampling.NODE_PATH,
            {"_calculate_acestep_official_sigmas", "_prepare_gradio_noise",
             "_gradio_guidance_interval", "sample_jkass_quality"},
            self.ns,
        )
        self.ns["model_trange"] = lambda count, **kwargs: range(count)
        self.ns["comfy"].samplers.KSAMPLER = lambda function, options: (function, options)
        self.ns["comfy"].samplers.ksampler = lambda name: name
        self.ns["comfy"].sample.sample_custom = self.capture_sampling
        self.captured = {}

    def capture_sampling(self, model, noise, cfg, sampler, sigmas,
                         positive, negative, latent, **kwargs):
        self.captured = dict(
            model=model, noise=noise.clone(), sampler=sampler,
            sigmas=sigmas.clone(), positive=positive, negative=negative,
        )
        return latent + 1

    def generate(self, **overrides):
        positive = [[torch.ones(1, 2, 3), {
            "conditioning_lyrics": torch.ones(1, 2, 3),
            "audio_codes": [[3, 9]], "acestep_sft_duration": .4,
        }]]
        options = dict(
            model=ScheduleModel(), positive=positive,
            negative=copy.deepcopy(positive), scheduler="ace_step",
            sampler_name="jkass_quality", shift=3.,
            guidance_mode="apg_denoised", guidance_schedule="normal",
            noise_source="official",
        )
        options.update(overrides)
        return self.harness.generate(**options)

    def test_default_bf16_model_schedule_has_no_zero_length_steps(self):
        for steps in (50, 100, 200):
            with self.subTest(steps=steps):
                self.generate(steps=steps)
                sigmas = self.captured["sigmas"]
                self.assertEqual(sigmas.dtype, torch.float32)
                self.assertEqual(sigmas.device.type, "cpu")
                self.assertEqual(sigmas.numel(), steps + 1)
                self.assertEqual(float(sigmas[0]), 1.)
                self.assertEqual(float(sigmas[-1]), 0.)
                self.assertTrue(torch.all(sigmas[:-1] > sigmas[1:]))
                # A float64 reference catches rounding before the schedule is
                # delivered to the sampler, including the shift transform.
                time = torch.arange(steps, -1, -1, dtype=torch.float64) / steps
                reference = 3 * time / (1 + 2 * time)
                self.assertLess(float((sigmas.double() - reference).abs().max()), 1.5e-7)

    def test_model_precision_preserves_legacy_bf16_mesh_exactly(self):
        for steps, repeats in ((50, 3), (100, 13), (200, 35)):
            with self.subTest(steps=steps):
                self.generate(steps=steps, schedule_precision="model")
                time = torch.linspace(1., 0., steps + 1, dtype=torch.bfloat16)
                expected = (3 * time / (1 + 2 * time)).float()
                sigmas = self.captured["sigmas"]
                self.assertTrue(torch.equal(sigmas, expected))
                self.assertEqual(int((sigmas[:-1] == sigmas[1:]).sum()), repeats)

    def test_float32_models_match_in_both_precision_modes(self):
        self.generate(model=ScheduleModel(torch.float32), schedule_precision="float32")
        expected = self.captured["sigmas"]
        self.generate(model=ScheduleModel(torch.float32), schedule_precision="model")
        self.assertTrue(torch.equal(self.captured["sigmas"], expected))

    def test_precision_choice_preserves_seeded_noise_and_full_lm_conditioning(self):
        self.generate(seed=123, schedule_precision="model")
        legacy = self.captured
        self.generate(seed=123, schedule_precision="float32")
        improved = self.captured
        self.assertTrue(torch.equal(improved["noise"], legacy["noise"]))
        self.assertIs(improved["sampler"][0], legacy["sampler"][0])
        for branch in ("positive", "negative"):
            self.assertEqual(improved[branch][0][1]["audio_codes"], [[3, 9]])
            torch.testing.assert_close(
                improved[branch][0][1]["conditioning_lyrics"],
                legacy[branch][0][1]["conditioning_lyrics"], rtol=0, atol=0,
            )

    def test_shift_one_retains_evenly_spaced_float32_steps(self):
        self.generate(steps=100, shift=1.)
        torch.testing.assert_close(
            self.captured["sigmas"], torch.linspace(1., 0., 101), rtol=0, atol=0
        )

    def test_other_scheduler_keeps_its_mesh_in_both_precision_modes(self):
        for precision in ("float32", "model"):
            with self.subTest(precision=precision):
                self.generate(scheduler="normal", schedule_precision=precision)
                torch.testing.assert_close(
                    self.captured["sigmas"], torch.linspace(1., 0., 51), rtol=0, atol=0
                )

    def test_jk_heun_correctors_repeat_evaluations_without_repeated_outer_steps(self):
        self.generate(steps=100)
        sigmas = self.captured["sigmas"]
        evaluations, outer_steps = [], []

        def model(x, sigma):
            evaluations.append(float(sigma[0]))
            velocity = torch.full_like(x, math.sin(float(sigma[0])))
            return x - sigma[:, None, None] * velocity

        result = self.ns["sample_jkass_quality"](
            model, torch.ones(1, 2, 3), sigmas, disable=True,
            callback=lambda data: outer_steps.append(float(data["sigma"])),
        )
        self.assertEqual(len(evaluations), 199)
        self.assertEqual(len(outer_steps), 100)
        self.assertEqual(len(set(outer_steps)), 100)
        self.assertTrue(all(a > b for a, b in zip(outer_steps, outer_steps[1:])))
        for index in range(1, len(evaluations), 2):
            self.assertEqual(evaluations[index], evaluations[index + 1])
        self.assertTrue(torch.isfinite(result).all())

    def test_precision_control_is_appended_and_defaults_to_float32(self):
        self.ns.update(
            GUIDANCE_MODES=["apg", "adg", "standard_cfg", "apg_denoised"],
            ACESTEP_SFT_TASK_TYPE="ACESTEP_SFT_TASK",
        )
        self.ns["comfy"].samplers.KSampler = SimpleNamespace(
            SAMPLERS=["euler", "heun"], SCHEDULERS=["normal"]
        )
        optional = self.ns["AceStepSFTGenerate"].INPUT_TYPES()["optional"]
        choices, settings = optional["schedule_precision"]
        self.assertEqual(list(choices), ["float32", "model"])
        self.assertEqual(settings["default"], "float32")
        self.assertEqual(
            list(optional).index("schedule_precision"),
            list(optional).index("guidance_rescale") + 1,
        )


if __name__ == "__main__":
    unittest.main()
