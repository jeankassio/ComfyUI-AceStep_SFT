"""CPU regressions for Advanced sampling and ComfyUI-compatible continuation.

Definitions are AST-loaded: these tests do not load model weights or start CUDA.
The sampler uses the installed ComfyUI CONST noise/inverse-noise transforms;
ignoring those transforms would hide errors in chained partial sampling.
"""
import ast
import copy
from types import SimpleNamespace
import unittest

import torch

import test_sampling
from test_sampling import (
    COMFY_ROOT, NODE_PATH, FakeModel, load_definitions,
)


class AdvancedFakeModel(FakeModel):
    def clone(self):
        result = AdvancedFakeModel()
        result.model_options = self.model_options.copy()
        return result

    def add_object_patch(self, name, value):
        if name == "model_sampling":
            self.sampling = value

    def model_dtype(self):
        return torch.float32


class GenerateAdvancedTests(unittest.TestCase):
    def setUp(self):
        harness = test_sampling.SamplingTests()
        harness.setUp()
        self.ns = harness.ns
        self.sampler_ns = harness.sampler_ns
        load_definitions(
            NODE_PATH,
            {
                "AceStepSFTGenerateAdvanced", "sample_jkass_quality",
                "_calculate_acestep_official_sigmas", "_gradio_guidance_interval",
                "_prepare_gradio_noise", "_get_source_latent_metadata",
                "_match_latent_length", "_match_noise_mask_length",
            },
            self.ns,
        )
        const_ns = {"torch": torch}
        load_definitions(COMFY_ROOT / "comfy" / "model_sampling.py",
                         {"reshape_sigma", "CONST"}, const_ns)
        self.ns["comfy"].model_sampling.CONST = const_ns["CONST"]
        self.ns["model_trange"] = lambda count, **kwargs: range(count)
        self.ns["comfy"].samplers.KSAMPLER = lambda function, options: (function, options)
        self.ns["comfy"].samplers.ksampler = lambda name: (
            self.sampler_ns["sample_" + name], {}
        )
        self.ns["comfy"].samplers.KSampler = SimpleNamespace(
            SAMPLERS=["euler", "heun"], SCHEDULERS=["normal"],
        )
        self.ns["comfy"].sample.sample_custom = self.sample_custom
        self.ns["comfy"].sample.sample = self.sample
        self.ns["comfy"].sample.prepare_noise = self.prepare_noise
        self.ns["comfy"].sample.prepare_empty_noise = torch.zeros_like
        self.ns["_calculate_sampling_sigmas"] = self.calculate_sigmas
        self.ns["_vae_decode_with_optional_tiling"] = self.decode
        self.ns["_normalize_audio_peak"] = lambda audio: audio
        tree = ast.parse(NODE_PATH.read_text(encoding="utf-8"))
        self.ns["GUIDANCE_MODES"] = ast.literal_eval(next(
            item.value for item in tree.body if isinstance(item, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "GUIDANCE_MODES"
                    for target in item.targets)
        ))
        self.ns["ACESTEP_SFT_TASK_TYPE"] = "ACESTEP_SFT_TASK"
        self.calls = []
        self.evaluations = []
        self.noise_requests = []
        self.decodes = []

    def calculate_sigmas(self, sampling, scheduler, sampler, steps):
        if scheduler == "ace_step":
            return self.ns["_calculate_acestep_official_sigmas"](steps, sampling.shift)
        return torch.linspace(1.0, 0.0, steps + 1)

    def prepare_noise(self, latent, seed, indices):
        self.noise_requests.append((seed, indices))
        generator = torch.Generator().manual_seed(seed)
        return torch.randn(latent.shape, generator=generator, dtype=latent.dtype)

    def decode(self, vae, latent, tiled):
        self.decodes.append(latent.clone())
        return torch.zeros(latent.shape[0], latent.shape[-1] * 1920, 2)

    def sample(self, model, noise, steps, cfg, sampler_name, scheduler,
               positive, negative, latent_image, **kwargs):
        return self.sample_custom(
            model, noise, cfg,
            (self.sampler_ns["sample_" + sampler_name], {}), kwargs.pop("sigmas"),
            positive, negative, latent_image, **kwargs,
        )

    def sample_custom(self, model, noise, cfg, sampler, sigmas,
                      positive, negative, latent_image, **kwargs):
        self.calls.append({
            "noise": noise.clone(), "sigmas": sigmas.clone(),
            "positive": copy.deepcopy(positive), "negative": copy.deepcopy(negative),
            "seed": kwargs.get("seed"), "noise_mask": kwargs.get("noise_mask"),
        })
        options = model.model_options.copy()
        # Real ComfyUI advertises the sliced mesh here. Guidance must still
        # remain anchored to the full requested mesh when chaining windows.
        options["transformer_options"] = {"sample_sigmas": sigmas}

        def evaluate(x, sigma, **extra):
            scalar_sigma = float(sigma[0])
            code_active = any(
                "audio_codes" in metadata
                and scalar_sigma >= model.sampling.percent_to_sigma(
                    metadata.get("end_percent", 1.0)
                )
                for _, metadata in positive
            )
            sigma_r = sigma.reshape(-1, 1, 1)
            # Smooth nonlinear velocity makes an omitted Heun correction
            # or incorrect restart scaling visible in the final latent.
            velocity = 0.12 * x.tanh() + 0.035 * x.sin() + 0.025 * sigma_r
            uncond = x - sigma_r * velocity
            cond = x - sigma_r * (velocity + (0.018 if code_active else 0.009))
            cfg_function = options.get("sampler_cfg_function")
            if cfg_function is None:
                denoised = uncond + (cond - uncond) * cfg
            else:
                denoised = x - cfg_function({
                    "cond_denoised": cond, "uncond_denoised": uncond,
                    "cond_scale": cfg, "input": x, "sigma": sigma,
                    "model_options": options,
                })
            measured_scale = float(((denoised - uncond) / (cond - uncond)).mean())
            self.evaluations.append((scalar_sigma, measured_scale, code_active))
            return denoised

        x = model.sampling.noise_scaling(sigmas[0], noise, latent_image)
        sampled = sampler[0](evaluate, x, sigmas, extra_args={"seed": kwargs.get("seed")},
                             disable=True)
        return model.sampling.inverse_noise_scaling(sigmas[-1], sampled)

    def generate(self, **overrides):
        embedding = torch.ones(1, 2, 3)
        positive = [
            [embedding, {"audio_codes": [[1, 2]], "end_percent": 0.5,
                         "acestep_sft_lm_codes_strength": 0.5}],
            [embedding, {"start_percent": 0.5,
                         "acestep_sft_lm_codes_strength": 0.5}],
        ]
        settings = dict(
            model=AdvancedFakeModel(), positive=positive, negative=copy.deepcopy(positive),
            add_noise="enable", noise_seed=27, steps=6, cfg=7.0,
            sampler_name="jkass_quality", scheduler="normal", duration=0.4,
            infer_method="ode", guidance_mode="standard_cfg", guidance_interval=1.0,
            guidance_schedule="decay", min_guidance_scale=1.0,
            start_at_step=0, end_at_step=10000, return_with_leftover_noise="disable",
        )
        settings.update(overrides)
        return self.ns["AceStepSFTGenerateAdvanced"]().generate_advanced(**settings)

    def test_schema_defaults_and_registration_match_advanced_controls(self):
        advanced = self.ns["AceStepSFTGenerateAdvanced"]
        inputs = advanced.INPUT_TYPES()
        required, optional = inputs["required"], inputs["optional"]
        self.assertNotIn("seed", required)
        self.assertNotIn("denoise", required)
        self.assertNotIn("task", optional)
        for name, choices, default in (
            ("add_noise", ["enable", "disable"], "enable"),
            ("return_with_leftover_noise", ["disable", "enable"], "disable"),
        ):
            self.assertEqual(set(required[name][0]), set(choices))
            self.assertEqual(required[name][1]["default"], default)
        for name, default in (("start_at_step", 0), ("end_at_step", 10000),
                              ("noise_seed", 0)):
            self.assertEqual(required[name][1]["default"], default)
        basic = self.ns["AceStepSFTGenerate"]
        self.assertEqual(advanced.RETURN_TYPES, basic.RETURN_TYPES)
        self.assertEqual(advanced.RETURN_NAMES, basic.RETURN_NAMES)
        self.assertEqual(advanced.FUNCTION, "generate_advanced")
        for name, definition in basic.INPUT_TYPES()["optional"].items():
            if name not in ("task", "latent_or_audio"):
                self.assertEqual(optional[name], definition)
        self.assertIn("latent_or_audio", optional)
        tree = ast.parse(NODE_PATH.read_text(encoding="utf-8"))
        registries = {
            target.id: item.value
            for item in tree.body if isinstance(item, ast.Assign)
            for target in item.targets if isinstance(target, ast.Name)
            and target.id in {"NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"}
        }
        mapping = dict(zip(
            [ast.literal_eval(key) for key in registries["NODE_CLASS_MAPPINGS"].keys],
            registries["NODE_CLASS_MAPPINGS"].values,
        ))
        self.assertEqual(mapping["AceStepSFTGenerateAdvanced"].id, advanced.__name__)
        display = ast.literal_eval(registries["NODE_DISPLAY_NAME_MAPPINGS"])
        self.assertIn("Advanced", display["AceStepSFTGenerateAdvanced"])

    def test_two_windows_equal_one_shot_with_full_guidance_and_lm_handoff(self):
        for scheduler in ("normal", "ace_step"):
            for schedule in ("normal", "decay", "ramp_up"):
                with self.subTest(scheduler=scheduler, guidance_schedule=schedule):
                    self.calls.clear()
                    self.evaluations.clear()
                    whole = self.generate(scheduler=scheduler, guidance_schedule=schedule)[4]
                    first = self.generate(
                        scheduler=scheduler, guidance_schedule=schedule,
                        end_at_step=3, return_with_leftover_noise="enable",
                    )[4]
                    second = self.generate(
                        scheduler=scheduler, guidance_schedule=schedule,
                        latent_or_audio=first, add_noise="disable", start_at_step=3,
                    )[4]
                    torch.testing.assert_close(second["samples"], whole["samples"],
                                               rtol=2e-5, atol=2e-6)
                    full_mesh = self.calls[0]["sigmas"]
                    self.assertEqual(len(self.calls[1]["sigmas"]), 4)
                    self.assertGreater(float(self.calls[1]["sigmas"][-1]), 0)
                    torch.testing.assert_close(self.calls[2]["sigmas"], full_mesh[3:])
                    self.assertTrue(torch.count_nonzero(self.calls[2]["noise"]) == 0)
                    for call in self.calls:
                        for branch in (call["positive"], call["negative"]):
                            cutoff = branch[0][1]["end_percent"]
                            expected_sigma = float((full_mesh[2] + full_mesh[3]) / 2)
                            self.assertAlmostEqual(
                                3 * (1 - cutoff) / (1 + 2 * (1 - cutoff)),
                                expected_sigma, places=6,
                            )
                    self.assertTrue(any(active for _, _, active in self.evaluations))
                    self.assertTrue(any(not active for _, _, active in self.evaluations))
                    for sigma, measured_scale, _ in self.evaluations:
                        position = self.ns["_guidance_step_position"](sigma, full_mesh)
                        progress = position / (len(full_mesh) - 2)
                        expected_scale = {
                            "normal": 7.0,
                            "decay": 7.0 - 6.0 * progress,
                            "ramp_up": 1.0 + 6.0 * progress,
                        }[schedule]
                        self.assertAlmostEqual(measured_scale, expected_scale, delta=1e-4)

    def test_native_euler_and_heun_routes_continue_the_full_mesh(self):
        for sampler in ("euler", "heun"):
            with self.subTest(sampler_name=sampler):
                whole = self.generate(sampler_name=sampler)[4]
                first = self.generate(sampler_name=sampler, end_at_step=2,
                                      return_with_leftover_noise="enable")[4]
                final = self.generate(sampler_name=sampler, start_at_step=2,
                                      latent_or_audio=first, add_noise="disable")[4]
                torch.testing.assert_close(final["samples"], whole["samples"],
                                           rtol=2e-5, atol=2e-6)

    def test_default_full_window_equals_regular_generate(self):
        settings = dict(
            model=AdvancedFakeModel(), positive=[[torch.ones(1, 2, 3), {}]],
            negative=[[torch.ones(1, 2, 3), {}]], seed=27, steps=6, cfg=7.0,
            sampler_name="jkass_quality", scheduler="normal", denoise=1.0,
            duration=0.4, infer_method="ode", guidance_mode="standard_cfg",
            guidance_schedule="normal",
        )
        regular = self.ns["AceStepSFTGenerate"]().generate(**settings)[4]
        advanced_settings = {key: value for key, value in settings.items()
                             if key not in ("seed", "denoise")}
        advanced = self.generate(**advanced_settings)[4]
        torch.testing.assert_close(advanced["samples"], regular["samples"])

    def test_forced_terminal_zero_decodes_but_partial_window_does_not(self):
        vae = SimpleNamespace(audio_sample_rate=48000)
        partial = self.generate(vae=vae, end_at_step=3,
                                return_with_leftover_noise="enable")
        self.assertGreater(float(self.calls[-1]["sigmas"][-1]), 0)
        self.assertIsNone(partial[5])
        self.assertEqual(self.decodes, [])
        forced = self.generate(vae=vae, end_at_step=3,
                               return_with_leftover_noise="disable")
        self.assertEqual(float(self.calls[-1]["sigmas"][-1]), 0)
        self.assertIsNotNone(forced[5])
        self.assertEqual(len(self.decodes), 1)
        self.assertEqual(forced[5]["sample_rate"], 48000)

    def test_empty_or_reversed_windows_preserve_latent_and_skip_decode(self):
        source = {"samples": torch.randn(1, 64, 10), "type": "audio"}
        vae = SimpleNamespace(audio_sample_rate=48000)
        for start, end in ((3, 3), (4, 2), (6, 10000), (10000, 10000)):
            with self.subTest(start=start, end=end):
                result = self.generate(latent_or_audio=source, vae=vae,
                                       start_at_step=start, end_at_step=end)
                torch.testing.assert_close(result[4]["samples"], source["samples"])
                self.assertIsNone(result[5])
        self.assertEqual(self.calls, [])
        self.assertEqual(self.decodes, [])

    def test_disabled_noise_does_not_draw_noise_and_still_passes_seed(self):
        for source in ("comfy", "official"):
            with self.subTest(noise_source=source):
                self.generate(add_noise="disable", noise_source=source,
                              noise_seed=197, start_at_step=2)
                self.assertTrue(torch.count_nonzero(self.calls[-1]["noise"]) == 0)
                self.assertEqual(self.calls[-1]["seed"], 197)
        self.assertEqual(self.noise_requests, [])

    def test_enabled_noise_uses_selected_source_and_noise_seed(self):
        for source in ("comfy", "official"):
            with self.subTest(noise_source=source):
                self.generate(noise_source=source, noise_seed=19)
                actual = self.calls[-1]["noise"]
                if source == "comfy":
                    expected = torch.randn(actual.shape,
                                           generator=torch.Generator().manual_seed(19))
                else:
                    expected = torch.randn((1, actual.shape[-1], actual.shape[1]),
                                           generator=torch.Generator().manual_seed(19)).movedim(-1, 1)
                torch.testing.assert_close(actual, expected)
                self.assertEqual(self.calls[-1]["seed"], 19)

    def test_continuation_metadata_does_not_alias_source_noise_mask_or_indices(self):
        mask = torch.linspace(0, 1, 10).reshape(1, 1, 10)
        source = {"samples": torch.randn(1, 64, 10), "type": "audio",
                  "noise_mask": mask, "batch_index": [3], "custom_metadata": "retained"}
        result = self.generate(latent_or_audio=source, end_at_step=2,
                               return_with_leftover_noise="enable")[4]
        self.assertEqual(result["custom_metadata"], "retained")
        torch.testing.assert_close(result["noise_mask"], mask)
        self.assertNotEqual(result["noise_mask"].data_ptr(), mask.data_ptr())
        self.assertIsNot(result["batch_index"], source["batch_index"])
        with torch.inference_mode():
            result["noise_mask"].zero_()
        result["batch_index"][0] = 99
        torch.testing.assert_close(mask, torch.linspace(0, 1, 10).reshape(1, 1, 10))
        self.assertEqual(source["batch_index"], [3])


if __name__ == "__main__":
    unittest.main()
