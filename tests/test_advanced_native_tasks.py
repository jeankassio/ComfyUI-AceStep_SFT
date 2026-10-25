"""Exercise native task conditioning and split sampling without model weights."""
import copy
import random
from types import SimpleNamespace
import unittest

import torch

import test_generate_advanced
from test_generate_advanced import AdvancedFakeModel
from test_sampling import COMFY_ROOT, NODE_PATH, load_definitions


class NativeFakeModel(AdvancedFakeModel):
    def clone(self):
        result = NativeFakeModel()
        result.model_options = self.model_options.copy()
        return result

    def set_model_unet_function_wrapper(self, wrapper):
        self.model_options["model_function_wrapper"] = wrapper

    def set_model_denoise_mask_function(self, function):
        self.model_options["denoise_mask_function"] = function


class RecordingVAE:
    audio_sample_rate = 48000
    latent_dim = 1
    downscale_ratio = 1920
    device = "cpu"

    def __init__(self):
        self.encodes = []

    def encode(self, audio):
        self.encodes.append(audio.clone())
        frames = audio.shape[1] // self.downscale_ratio
        return audio[:, :frames * self.downscale_ratio].reshape(
            audio.shape[0], frames, self.downscale_ratio, 2
        ).mean(dim=(2, 3)).unsqueeze(1).repeat(1, 64, 1)

    def encode_tiled(self, audio, **kwargs):
        return self.encode(audio)


class NativeSampler:
    def __init__(self, sampler_function, extra_options=None, inpaint_options=None):
        self.sampler_function = sampler_function
        self.extra_options = extra_options or {}
        self.inpaint_options = inpaint_options or {}


class AdvancedNativeTaskTests(unittest.TestCase):
    def setUp(self):
        self.harness = test_generate_advanced.GenerateAdvancedTests()
        self.harness.setUp()
        self.ns = self.harness.ns
        load_definitions(NODE_PATH, {
            "_validate_acestep_task", "_get_source_duration_seconds",
            "_build_source_latent", "_repeat_native_tensor",
            "_audio_to_stereo_48k", "_normalize_audio_to_stereo_48k", "_normalize_audio_peak",
            "_vae_uses_1d_tiling", "_vae_encode_with_optional_tiling",
            "_select_timbre_reference_audio", "_build_timbre_reference_latent",
            "_same_audio_payload", "_configure_acestep_native_task_model",
            "_resolve_repaint_span", "_resolve_repaint_profile",
            "_build_soft_repaint_mask", "_splice_repaint_waveform",
        }, self.ns)
        load_definitions(COMFY_ROOT / "comfy" / "samplers.py",
                         {"KSamplerX0Inpaint"}, self.ns)
        self.ns["random"] = random
        self.ns["_get_acestep_diffusion_model"] = lambda model: object()
        self.ns["_ensure_acestep_native_forward"] = lambda model: None
        self.ns["_get_acestep_silence_latent"] = lambda length, device, dtype: torch.zeros(
            1, 64, length, device=device, dtype=dtype
        )
        self.ns["_extract_cover_semantic_hints"] = lambda model, source: source * 0.6
        self.ns["comfy"].model_management.load_model_gpu = lambda model: None
        self.ns["comfy"].samplers.KSAMPLER = NativeSampler
        self.ns["comfy"].samplers.ksampler = lambda name: NativeSampler(
            self.harness.sampler_ns["sample_" + name]
        )
        self.harness.sample_custom = self.sample_custom
        self.ns["comfy"].sample.sample_custom = self.sample_custom
        self.calls = []
        self.native_calls = []
        self.blends = []
        self.splices = []
        self.raw_samples = []
        self.runtime_conditions = {}
        self.blend = self.ns["_build_soft_repaint_mask"]
        self.splice = self.ns["_splice_repaint_waveform"]
        self.ns["_build_soft_repaint_mask"] = self.record_blend
        self.ns["_splice_repaint_waveform"] = self.record_splice
        self.source = {"samples": torch.linspace(-0.5, 0.8, 40).reshape(1, 1, 40).repeat(1, 64, 1),
                       "type": "audio", "batch_index": [3]}
        self.reference = {"samples": torch.full((1, 64, 30), 0.65)}

    def record_blend(self, *args):
        self.blends.append(args[0].clone())
        return self.blend(*args)

    def record_splice(self, *args):
        self.splices.append(args)
        return self.splice(*args)

    def cover(self, **overrides):
        task = dict(task_type="cover", source=self.source,
                    reference_audio=self.reference, source_conditioning="semantic_fsq",
                    audio_cover_strength=0.4, cover_noise_strength=0.2,
                    preserve_singer_identity=True)
        task.update(overrides)
        return task

    def repaint(self, **overrides):
        task = dict(task_type="repaint", source=self.source,
                    start_seconds=0.48, end_seconds=1.08,
                    repaint_mode="balanced", repaint_strength=0.5)
        task.update(overrides)
        return task

    def generate(self, task, **overrides):
        embedding = torch.ones(1, 2, 3)
        positive = [[embedding, {"acestep_sft_task_type": task["task_type"], "end_percent": 0.4}],
                    [embedding, {"acestep_sft_task_type": task["task_type"], "start_percent": 0.4}]]
        settings = dict(model=NativeFakeModel(), task=task, positive=positive,
                        negative=copy.deepcopy(positive), steps=50,
                        guidance_schedule="normal", duration=8.0)
        settings.update(overrides)
        return self.harness.generate(**settings)

    def sample_custom(self, model, noise, cfg, sampler, sigmas,
                      positive, negative, latent_image, **kwargs):
        self.calls.append({"noise": noise.clone(), "sigmas": sigmas.clone(),
                           "latent_image": latent_image.clone(),
                           "positive": copy.deepcopy(positive),
                           "mask": kwargs.get("noise_mask"), "model": model})

        def core(x, sigma, **c):
            source = c.get("src_latents", torch.zeros_like(x))
            reference = c.get("refer_audio")
            reference_mean = 0.0 if reference is None else float(reference.mean())
            self.native_calls.append({
                "sigma": float(sigma[0]), "source": source.clone(),
                "reference_mean": reference_mean,
                "reference_shape": None if reference is None else reference.shape,
                "chunks": c.get("chunk_masks"), "hints": c.get("precomputed_lm_hints_25Hz"),
            })
            sigma_r = sigma.reshape(-1, 1, 1)
            velocity = (0.12 * x.tanh() + 0.035 * x.sin() + 0.025 * sigma_r
                        + 0.013 * source.tanh() + 0.009 * reference_mean + 0.008 * cfg)
            return x - sigma_r * velocity

        def evaluate(x, sigma, model_options=None, seed=None):
            wrapper = model_options.get("model_function_wrapper")
            return wrapper(core, {"input": x, "timestep": sigma,
                                  "c": self.runtime_conditions.copy()})

        evaluate.inner_model = SimpleNamespace(
            scale_latent_inpaint=lambda x, sigma, noise, latent_image, denoise_mask:
                model.sampling.noise_scaling(sigma, noise, latent_image)
        )
        model_k = self.ns["KSamplerX0Inpaint"](evaluate, sigmas)
        model_k.noise = noise
        model_k.latent_image = latent_image
        x = model.sampling.noise_scaling(sigmas[0], noise, latent_image)
        sampler_function = sampler[0] if isinstance(sampler, tuple) else sampler.sampler_function
        sampled = sampler_function(
            model_k, x, sigmas,
            extra_args={"denoise_mask": kwargs.get("noise_mask"),
                        "model_options": model.model_options, "seed": kwargs.get("seed")},
            disable=True,
        )
        result = model.sampling.inverse_noise_scaling(sigmas[-1], sampled)
        self.raw_samples.append(result.clone())
        return result

    def test_cover_reference_and_source_reach_both_sampling_stages(self):
        task = self.cover()
        source_before = self.source["samples"].clone()
        ref_before = self.reference["samples"].clone()
        first = self.generate(task, end_at_step=25, return_with_leftover_noise="enable")[4]
        first_count = len(self.native_calls)
        final = self.generate(task, latent_or_audio=first, add_noise="disable", start_at_step=25)[4]
        self.assertEqual(first["samples"].shape, (1, 64, 40))
        self.assertEqual(final["samples"].shape, (1, 64, 40))
        for stage in (self.native_calls[:first_count], self.native_calls[first_count:]):
            self.assertTrue(stage)
            self.assertTrue(all(abs(call["reference_mean"] - 0.65) < 1e-6 for call in stage))
            self.assertTrue(all(call["reference_shape"] == (1, 64, 750) for call in stage))
        torch.testing.assert_close(self.native_calls[0]["source"], source_before * 0.6)
        torch.testing.assert_close(self.source["samples"], source_before)
        torch.testing.assert_close(self.reference["samples"], ref_before)
        self.assertTrue(torch.count_nonzero(self.calls[-1]["noise"]) == 0)
        torch.testing.assert_close(self.calls[-1]["latent_image"], first["samples"])
        self.assertEqual(len(self.harness.noise_requests), 1)

    def test_cover_one_shot_matches_split_for_native_samplers_and_noise_strengths(self):
        for sampler in ("euler", "heun", "jkass_quality"):
            for noise_strength in (0.0, 0.2):
                with self.subTest(sampler=sampler, cover_noise_strength=noise_strength):
                    task = self.cover(cover_noise_strength=noise_strength)
                    whole = self.generate(task, sampler_name=sampler)[4]
                    first = self.generate(task, sampler_name=sampler, end_at_step=25,
                                          return_with_leftover_noise="enable")[4]
                    final = self.generate(task, sampler_name=sampler, latent_or_audio=first,
                                          add_noise="disable", start_at_step=25)[4]
                    torch.testing.assert_close(final["samples"], whole["samples"], rtol=2e-5, atol=2e-6)

    def test_cover_noise_truncation_keeps_absolute_full_schedule_indices(self):
        self.generate(self.cover(), end_at_step=25, return_with_leftover_noise="enable")
        expected = self.ns["_calculate_acestep_official_sigmas"](50, 3.0)
        source_start = int(torch.argmin((expected[:-1] - 0.8).abs()))
        torch.testing.assert_close(self.calls[-1]["sigmas"], expected[source_start:26])
        first = self.generate(self.cover(), end_at_step=25, return_with_leftover_noise="enable")[4]
        self.generate(self.cover(), latent_or_audio=first, add_noise="disable", start_at_step=25)
        torch.testing.assert_close(self.calls[-1]["sigmas"], expected[25:])
        self.assertEqual(len(self.calls[-1]["sigmas"]), 26)
        cutoff = self.calls[-1]["positive"][0][1]["end_percent"]
        self.assertEqual(cutoff, self.calls[-2]["positive"][0][1]["end_percent"])

    def test_cover_window_before_source_noise_start_can_continue_without_reinitializing(self):
        mesh = self.ns["_calculate_acestep_official_sigmas"](50, 1.0)
        source_start = int(torch.argmin((mesh[:-1] - 0.8).abs()))
        self.assertEqual(source_start, 10)
        for split in (5, source_start):
            with self.subTest(split=split):
                task = self.cover()
                whole = self.generate(task, shift=1.0)[4]
                call_count = len(self.calls)
                first = self.generate(task, shift=1.0, end_at_step=split,
                                      return_with_leftover_noise="enable")
                self.assertEqual(len(self.calls), call_count)
                self.assertIsNone(first[5])
                sigma = mesh[source_start]
                noise = self.harness.prepare_noise(self.source["samples"], 27, [3])
                initial = sigma * noise + (1 - sigma) * self.source["samples"]
                torch.testing.assert_close(first[4]["samples"], initial / (1 - sigma))
                final = self.generate(task, shift=1.0, latent_or_audio=first[4],
                                      add_noise="disable", start_at_step=split)[4]
                torch.testing.assert_close(final["samples"], whole["samples"], rtol=2e-5, atol=2e-6)

    def test_audio_source_and_distinct_reference_are_encoded_for_each_stage(self):
        vae = RecordingVAE()
        source = {"waveform": torch.full((1, 2, 19200), 0.25), "sample_rate": 48000}
        reference = {"waveform": torch.full((1, 2, 9600), 0.7), "sample_rate": 48000}
        task = self.cover(source=source, reference_audio=reference)
        first = self.generate(task, vae=vae, end_at_step=25, return_with_leftover_noise="enable")[4]
        self.assertEqual(len(vae.encodes), 2)
        final = self.generate(task, vae=vae, latent_or_audio=first,
                              add_noise="disable", start_at_step=25)
        self.assertEqual([audio.shape[1] for audio in vae.encodes], [19200, 1440000] * 2)
        self.assertIsNotNone(final[5])
        self.assertTrue(torch.equal(source["waveform"], torch.full((1, 2, 19200), 0.25)))
        self.assertTrue(torch.equal(reference["waveform"], torch.full((1, 2, 9600), 0.7)))
        self.assertAlmostEqual(self.native_calls[-1]["reference_mean"], 0.7, places=6)

    def test_same_audio_reference_reuses_source_encode_in_each_stage(self):
        vae = RecordingVAE()
        source = {"waveform": torch.full((1, 2, 19200), 0.25), "sample_rate": 48000}
        task = self.cover(source=source, reference_audio=None)
        first = self.generate(task, vae=vae, end_at_step=25, return_with_leftover_noise="enable")[4]
        self.generate(task, vae=vae, latent_or_audio=first, add_noise="disable", start_at_step=25)
        self.assertEqual([audio.shape[1] for audio in vae.encodes], [19200, 19200])
        self.assertAlmostEqual(self.native_calls[-1]["reference_mean"], 0.25, places=6)

    def test_basic_generate_keeps_task_and_refinement_mutually_exclusive(self):
        with self.assertRaisesRegex(ValueError, "requires Generate Advanced"):
            self.ns["AceStepSFTGenerate"]().generate(
                model=NativeFakeModel(), positive=[], negative=[], task=self.cover(),
                latent_or_audio=self.source, seed=0, steps=50, cfg=7.0,
                sampler_name="euler", scheduler="normal", denoise=1.0,
                duration=1.6, infer_method="ode", guidance_mode="standard_cfg")

    def test_task_continuation_rejects_audio_or_new_noise(self):
        audio = {"waveform": torch.zeros(1, 2, 19200), "sample_rate": 48000}
        for payload, add_noise in ((audio, "disable"), (self.source, "enable")):
            with self.subTest(add_noise=add_noise, is_audio="waveform" in payload):
                with self.assertRaisesRegex(ValueError, "LATENT|add_noise|disable"):
                    self.generate(self.cover(), latent_or_audio=payload, add_noise=add_noise,
                                  start_at_step=25)

    def test_native_audio_source_requires_a_vae_before_sampling(self):
        audio = {"waveform": torch.zeros(1, 2, 19200), "sample_rate": 48000}
        with self.assertRaisesRegex(ValueError, "VAE is required.*AUDIO source"):
            self.generate(self.cover(source=audio))
        self.assertEqual(self.calls, [])
        self.assertEqual(self.native_calls, [])

    def test_continuation_preserves_batch_and_caller_metadata(self):
        task = self.cover()
        first = self.generate(task, batch_size=2, end_at_step=25,
                              return_with_leftover_noise="enable")[4]
        first["batch_index"] = [8, 4]
        first["custom_metadata"] = "continue-this-stage"
        first_before = first["samples"].clone()
        final = self.generate(task, latent_or_audio=first, add_noise="disable",
                              start_at_step=25)[4]
        self.assertEqual(final["samples"].shape, (2, 64, 40))
        self.assertEqual(final["batch_index"], [8, 4])
        self.assertEqual(final["custom_metadata"], "continue-this-stage")
        self.assertIsNot(final["batch_index"], first["batch_index"])
        torch.testing.assert_close(first["samples"], first_before)
        final["batch_index"][0] = 99
        self.assertEqual(first["batch_index"], [8, 4])
        self.assertEqual(self.source["batch_index"], [3])

    def test_repaint_injection_cutoff_uses_the_full_schedule_in_both_stages(self):
        task = self.repaint()
        first = self.generate(task, end_at_step=25, return_with_leftover_noise="enable")[4]
        self.generate(task, latent_or_audio=first, add_noise="disable", start_at_step=25)
        mesh = self.harness.calculate_sigmas(SimpleNamespace(shift=3.0), "normal", "euler", 50)
        mask = self.calls[-1]["mask"]
        for call in self.calls[-2:]:
            function = call["model"].model_options["denoise_mask_function"]
            for index in (24, 25, 26, 40):
                actual = function(mesh[index].reshape(1), mask,
                                  {"sigmas": call["sigmas"]})
                expected = mask if index < 25 else torch.ones_like(mask)
                torch.testing.assert_close(actual, expected)

    def test_repaint_only_blends_and_splices_the_finished_stage(self):
        vae = RecordingVAE()
        task = self.repaint()
        first = self.generate(task, vae=vae, end_at_step=25,
                              return_with_leftover_noise="enable")
        self.assertIsNone(first[5])
        self.assertEqual(self.blends, [])
        self.assertEqual(self.splices, [])
        self.assertEqual(self.harness.decodes, [])
        torch.testing.assert_close(first[4]["samples"], self.raw_samples[-1])
        final = self.generate(task, vae=vae, latent_or_audio=first[4], add_noise="disable",
                              start_at_step=25)
        self.assertEqual(len(self.blends), 1)
        self.assertEqual(len(self.splices), 1)
        self.assertIsNotNone(final[5])
        mask = self.blend(self.blends[-1], 12).to(final[4]["samples"])
        torch.testing.assert_close(final[4]["samples"],
                                   mask * self.raw_samples[-1] + (1 - mask) * self.source["samples"])

    def test_repaint_one_shot_matches_split_with_comfy_inpaint_masking(self):
        for sampler in ("euler", "heun", "jkass_quality"):
            for split in (17, 25):
                with self.subTest(sampler=sampler, split=split):
                    task = self.repaint()
                    whole = self.generate(task, sampler_name=sampler)[4]
                    first = self.generate(task, sampler_name=sampler, end_at_step=split,
                                          return_with_leftover_noise="enable")[4]
                    final = self.generate(task, sampler_name=sampler, latent_or_audio=first,
                                          add_noise="disable", start_at_step=split)[4]
                    torch.testing.assert_close(final["samples"], whole["samples"], rtol=2e-5, atol=2e-6)


if __name__ == "__main__":
    unittest.main()
