"""Reference-only singer conditioning preserves text2music and staged sampling."""
import ast
import copy
import functools
import inspect
import math
from types import MethodType, SimpleNamespace
import unittest

import torch

import test_advanced_native_tasks
from test_sampling import COMFY_ROOT, NODE_PATH, load_definitions


class ReferenceAudioGenerateTests(unittest.TestCase):
    def setUp(self):
        self.native = test_advanced_native_tasks.AdvancedNativeTaskTests()
        self.native.setUp()
        self.ns = self.native.ns
        load_definitions(NODE_PATH, {"_configure_acestep_reference_model"}, self.ns)
        self.reference = {"samples": torch.full((1, 64, 30), 0.65)}
        self.embedding = torch.ones(1, 2, 3)
        self.lyrics = torch.full((1, 4, 3), 0.4)
        self.codes = [[12, 34, 56]]
        self.positive = [[self.embedding, {"audio_codes": self.codes,
                         "conditioning_lyrics": self.lyrics,
                         "acestep_sft_lm_codes_strength": 1.0}]]
        self.negative = copy.deepcopy(self.positive)
        self.native.runtime_conditions = {"context": self.embedding,
                                          "lyric_embed": self.lyrics,
                                          "audio_codes": self.codes}

    def generate(self, advanced=False, **overrides):
        options = dict(
            model=test_advanced_native_tasks.NativeFakeModel(),
            positive=self.positive, negative=self.negative,
            reference_audio=self.reference, steps=50, cfg=7.0,
            sampler_name="jkass_quality", scheduler="normal", duration=1.6,
            infer_method="ode", guidance_mode="standard_cfg",
            guidance_schedule="normal", noise_source="comfy",
        )
        options.update(overrides)
        if advanced:
            return self.native.harness.generate(**options)
        options["seed"] = 27
        options["denoise"] = 1.0
        return self.ns["AceStepSFTGenerate"]().generate(**options)

    def test_reference_latent_conditions_basic_and_advanced_without_a_source_or_vae(self):
        reference_before = self.reference["samples"].clone()
        for advanced in (False, True):
            with self.subTest(advanced=advanced):
                result = self.generate(advanced=advanced)
                self.assertEqual(result[4]["samples"].shape, (1, 64, 40))
                self.assertIsNone(result[5])
                self.assertNotIn("model_function_wrapper", result[0].model_options)
                self.assertTrue(torch.count_nonzero(self.native.calls[-1]["latent_image"]) == 0)
                for call in self.native.native_calls:
                    self.assertAlmostEqual(call["reference_mean"], 0.65, places=6)
                    self.assertEqual(call["reference_shape"], (1, 64, 750))
                    self.assertTrue(torch.count_nonzero(call["source"]) == 0)
        torch.testing.assert_close(self.reference["samples"], reference_before)

    def test_lm_codes_lyrics_and_caller_conditioning_remain_intact(self):
        original_positive = copy.deepcopy(self.positive)
        original_negative = copy.deepcopy(self.negative)
        for advanced in (False, True):
            self.generate(advanced=advanced)
            received = self.native.calls[-1]["positive"]
            self.assertEqual(len(received), 1)
            self.assertEqual(received[0][1]["audio_codes"], self.codes)
            torch.testing.assert_close(received[0][1]["conditioning_lyrics"], self.lyrics)
            self.assertNotIn("acestep_sft_task_type", received[0][1])
            self.assertNotIn("start_percent", received[0][1])
            self.assertNotIn("end_percent", received[0][1])
        for actual, expected in ((self.positive, original_positive), (self.negative, original_negative)):
            self.assertEqual(actual[0][1]["audio_codes"], expected[0][1]["audio_codes"])
            torch.testing.assert_close(actual[0][0], expected[0][0])
            torch.testing.assert_close(actual[0][1]["conditioning_lyrics"], expected[0][1]["conditioning_lyrics"])

    def test_reference_wrapper_keeps_lm_context_and_delegates_the_previous_wrapper(self):
        model = test_advanced_native_tasks.NativeFakeModel()
        delegated = []
        def previous(apply_model, args):
            delegated.append(args["c"])
            return apply_model(args["input"], args["timestep"], **args["c"])
        model.set_model_unet_function_wrapper(previous)
        self.ns["_configure_acestep_reference_model"](model, self.reference, None, 48000, True, 27)
        wrapper = model.model_options["model_function_wrapper"]
        codes = torch.tensor([[12, 34, 56]])
        hints = torch.full((1, 64, 40), 0.9)
        c = {"audio_codes": codes, "lyric_embed": self.lyrics,
             "context": self.embedding, "precomputed_lm_hints_25Hz": hints}
        received = []
        def apply_model(x, sigma, **conditions):
            received.append(conditions)
            return x
        x = torch.zeros(2, 64, 40)
        wrapper(apply_model, {"input": x, "timestep": torch.ones(2), "c": c})
        self.assertEqual(len(delegated), 1)
        self.assertIs(received[0]["audio_codes"], codes)
        self.assertIs(received[0]["precomputed_lm_hints_25Hz"], hints)
        self.assertIs(received[0]["lyric_embed"], self.lyrics)
        self.assertIs(received[0]["context"], self.embedding)
        self.assertEqual(received[0]["refer_audio"].shape, (2, 64, 750))
        self.assertNotIn("refer_audio", c)
        self.assertNotIn("src_latents", c)
        self.assertNotIn("chunk_masks", c)

    def test_audio_reference_requires_a_vae_and_keeps_requested_song_duration(self):
        audio = {"waveform": torch.full((1, 2, 9600), 0.7), "sample_rate": 48000}
        original = audio["waveform"].clone()
        for advanced in (False, True):
            with self.subTest(advanced=advanced):
                with self.assertRaisesRegex(ValueError, "VAE is required.*AUDIO"):
                    self.generate(advanced=advanced, reference_audio=audio)
                vae = test_advanced_native_tasks.RecordingVAE()
                result = self.generate(advanced=advanced, reference_audio=audio, vae=vae, duration=2.4)
                self.assertEqual(len(vae.encodes), 1)
                self.assertEqual(vae.encodes[0].shape, (1, 1440000, 2))
                self.assertEqual(result[4]["samples"].shape, (1, 64, 60))
                self.assertIsNotNone(result[5])
                self.assertAlmostEqual(self.native.native_calls[-1]["reference_mean"], 0.7, places=6)
        torch.testing.assert_close(audio["waveform"], original)

    def test_reference_without_lm_codes_keeps_a_silent_source_context(self):
        model = test_advanced_native_tasks.NativeFakeModel()
        self.ns["_configure_acestep_reference_model"](model, self.reference, None, 48000, True, 27)
        wrapper = model.model_options["model_function_wrapper"]
        received = []
        def apply_model(x, sigma, **conditions):
            received.append(conditions)
            return x
        wrapper(apply_model, {"input": torch.full((1, 64, 40), 0.8),
                              "timestep": torch.ones(1),
                              "c": {"context": self.embedding, "lyric_embed": self.lyrics}})
        conditions = received[0]
        self.assertTrue(torch.count_nonzero(conditions["src_latents"]) == 0)
        self.assertTrue(torch.count_nonzero(conditions["precomputed_lm_hints_25Hz"]) == 0)
        torch.testing.assert_close(conditions["chunk_masks"], torch.ones(1, 64, 40))
        self.assertAlmostEqual(float(conditions["refer_audio"].mean()), 0.65, places=6)
        self.assertNotIn("audio_codes", conditions)

    def test_real_native_forward_keeps_180_seconds_separate_from_a_10_second_voice(self):
        source = COMFY_ROOT / "comfy" / "ldm" / "ace" / "ace_step15.py"
        cls = next(item for item in ast.parse(source.read_text(encoding="utf-8")).body
                   if getattr(item, "name", None) == "AceStepConditionGenerationModel")
        methods = [item for item in cls.body if getattr(item, "name", None)
                   in {"forward", "prepare_condition"}]
        core_ns = {"torch": torch, "math": math}
        exec(compile(ast.Module(body=methods, type_ignores=[]), str(source), "exec"), core_ns)
        self.ns.update(functools=functools, inspect=inspect, MethodType=MethodType)
        load_definitions(NODE_PATH, {"_ensure_acestep_native_forward", "_get_acestep_diffusion_model"}, self.ns)
        encoded_references = []
        code_requests = []
        contexts = []
        def encoder(text, text_mask, lyrics, lyric_mask, reference, order):
            encoded_references.append(reference.clone())
            torch.testing.assert_close(lyrics, self.lyrics)
            return text, None
        def get_codes(codes, dtype):
            code_requests.append(codes.clone())
            return torch.full((codes.shape[0], codes.shape[1], 64), 0.3, dtype=dtype)
        def tokenize_reference(reference):
            self.fail("Reference timbre must not become the song's LM composition")
        def decoder(hidden_states, context_latents, **kwargs):
            contexts.append(context_latents.clone())
            return torch.zeros_like(hidden_states)
        diffusion = SimpleNamespace(encoder=encoder, decoder=decoder,
            tokenizer=SimpleNamespace(tokenize=tokenize_reference,
                                      quantizer=SimpleNamespace(get_output_from_indices=get_codes)),
            detokenizer=lambda values: values.repeat_interleave(5, dim=1))
        diffusion.forward = MethodType(core_ns["forward"], diffusion)
        diffusion.prepare_condition = MethodType(core_ns["prepare_condition"], diffusion)
        voice = {"samples": torch.full((1, 64, 250), 0.65)}
        x = torch.zeros(1, 64, 4500)
        for condition_name, lm_condition, source_value in (
            (None, None, 0.0),
            ("audio_codes", torch.tensor([[12, 34, 56]]), 0.3),
            ("precomputed_lm_hints_25Hz", torch.full((1, 64, 4500), 0.9), 0.9),
        ):
            model = test_advanced_native_tasks.NativeFakeModel()
            model.model.diffusion_model = diffusion
            model.object_patches = {}
            model.get_model_object = lambda name: model.object_patches.get(name, diffusion.forward)
            model.add_object_patch = lambda name, value: model.object_patches.__setitem__(name, value)
            self.ns["_configure_acestep_reference_model"](model, voice, None, 48000, True, 27)
            patched_forward = model.object_patches["diffusion_model.forward"]
            c = {"context": self.embedding, "lyric_embed": self.lyrics}
            if condition_name is not None:
                c[condition_name] = lm_condition
            model.model_options["model_function_wrapper"](
                patched_forward, {"input": x, "timestep": torch.ones(1), "c": c})
            self.assertEqual(contexts[-1].shape, (1, 4500, 128))
            expected_source = torch.full((1, 4500, 64), source_value)
            torch.testing.assert_close(contexts[-1][..., :64], expected_source)
            torch.testing.assert_close(contexts[-1][..., 64:], torch.ones(1, 4500, 64))
            self.assertEqual(encoded_references[-1].shape, (1, 750, 64))
            self.assertAlmostEqual(float(encoded_references[-1].mean()), 0.65, places=6)
        self.assertEqual(len(code_requests), 1)
        torch.testing.assert_close(code_requests[0][:, :3], torch.tensor([[12, 34, 56]]))

    def test_partial_lm_handoff_keeps_reference_as_timbre_when_codes_end(self):
        model = test_advanced_native_tasks.NativeFakeModel()
        self.ns["_configure_acestep_reference_model"](model, self.reference, None, 48000, True, 27)
        received = []
        def apply_model(x, sigma, **conditions):
            received.append(conditions)
            return x
        wrapper = model.model_options["model_function_wrapper"]
        x = torch.zeros(1, 64, 4500)
        for codes in (torch.tensor([[12, 34, 56]]), None):
            c = {"context": self.embedding, "lyric_embed": self.lyrics}
            if codes is not None:
                c["audio_codes"] = codes
            wrapper(apply_model, {"input": x, "timestep": torch.ones(1), "c": c})
        self.assertIs(received[0]["is_covers"], True)
        self.assertNotIn("precomputed_lm_hints_25Hz", received[0])
        self.assertNotIn("audio_codes", received[1])
        self.assertTrue(torch.count_nonzero(received[1]["precomputed_lm_hints_25Hz"]) == 0)
        self.assertEqual(received[1]["precomputed_lm_hints_25Hz"].shape, (1, 64, 4500))
        self.assertIsNot(received[1]["is_covers"], False)
        for conditions in received:
            self.assertAlmostEqual(float(conditions["refer_audio"].mean()), 0.65, places=6)

    def test_invalid_reference_payloads_are_rejected_before_sampling(self):
        for advanced in (False, True):
            for reference, error in (({"unsupported": True}, "AUDIO or LATENT"),
                                     ({"samples": torch.zeros(1, 32, 30)}, "64 channels"),
                                     ({"samples": torch.zeros(1, 64, 2, 30)}, "64 channels")):
                with self.subTest(advanced=advanced, error=error):
                    with self.assertRaisesRegex(ValueError, error):
                        self.generate(advanced=advanced, reference_audio=reference)
        self.assertEqual(self.native.calls, [])

    def test_advanced_reapplies_reference_and_preserves_partial_latent_batch(self):
        audio = {"waveform": torch.full((1, 2, 9600), 0.7), "sample_rate": 48000}
        vae = test_advanced_native_tasks.RecordingVAE()
        whole = self.generate(advanced=True, reference_audio=audio, vae=vae, batch_size=2)[4]
        first = self.generate(advanced=True, reference_audio=audio, vae=vae, batch_size=2,
                              end_at_step=25, return_with_leftover_noise="enable")[4]
        first_before = first["samples"].clone()
        first["batch_index"] = [6, 8]
        final = self.generate(advanced=True, reference_audio=audio, vae=vae, latent_or_audio=first,
                              add_noise="disable", start_at_step=25)[4]
        self.assertEqual(len(vae.encodes), 3)
        self.assertEqual(final["samples"].shape, (2, 64, 40))
        self.assertEqual(final["batch_index"], [6, 8])
        self.assertEqual(self.native.native_calls[-1]["reference_shape"], (2, 64, 750))
        self.assertTrue(torch.count_nonzero(self.native.calls[-1]["noise"]) == 0)
        torch.testing.assert_close(self.native.calls[-1]["latent_image"], first_before)
        torch.testing.assert_close(first["samples"], first_before)
        torch.testing.assert_close(final["samples"], whole["samples"], rtol=2e-5, atol=2e-6)
        self.assertEqual(len(self.native.harness.noise_requests), 2)

    def test_direct_reference_overrides_task_reference_without_mutating_the_task(self):
        task = self.native.cover()
        existing_reference = task["reference_audio"]
        direct_reference = {"samples": torch.full((1, 64, 30), 0.2)}
        task_keys = set(task)
        for advanced in (False, True):
            self.generate(advanced=advanced, task=task, reference_audio=direct_reference,
                          positive=[[self.embedding, {"acestep_sft_task_type": "cover"}]],
                          negative=[[self.embedding, {"acestep_sft_task_type": "cover"}]])
            self.assertAlmostEqual(self.native.native_calls[-1]["reference_mean"], 0.2, places=6)
            self.assertIs(task["reference_audio"], existing_reference)
            self.assertEqual(set(task), task_keys)

    def test_reference_socket_is_appended_without_moving_legacy_widgets_or_inputs(self):
        legacy_optional = [
            "vae", "latent_or_audio", "task", "batch_size", "latent_shift", "latent_rescale",
            "fade_in_duration", "fade_out_duration", "use_tiled_vae", "unload_models_after_generate",
            "voice_boost", "apg_eta", "apg_momentum", "apg_norm_threshold", "guidance_interval",
            "guidance_schedule", "min_guidance_scale", "guidance_scale_text", "guidance_scale_lyric",
            "omega_scale", "erg_scale", "cfg_interval_start", "cfg_interval_end", "shift",
            "noise_source", "timbre_cls_token", "guidance_rescale", "schedule_precision", "apg_norm_rms",
        ]
        for name in ("AceStepSFTGenerate", "AceStepSFTGenerateAdvanced"):
            inputs = self.ns[name].INPUT_TYPES()
            optional = list(inputs["optional"])
            expected = legacy_optional
            if name.endswith("Advanced"):
                expected = [key for key in legacy_optional if key != "task"] + ["task"]
            self.assertEqual(optional[:-4], expected)
            self.assertEqual(optional[-4:], ["reference_audio", "reference_xl_attention",
                                            "reference_guidance_scale", "reference_timbre_repeats"])
            self.assertTrue(inputs["optional"]["reference_xl_attention"][1]["default"])
            self.assertEqual(inputs["optional"]["reference_guidance_scale"][1]["default"], 1.0)
            self.assertEqual(inputs["optional"]["reference_timbre_repeats"][1]["default"], 1)
            self.assertEqual(inputs["optional"]["reference_audio"][0], "AUDIO,LATENT")
            self.assertNotIn("reference_audio", inputs["required"])
            for key, value in (("steps", 50), ("cfg", 7.0), ("duration", 60.0)):
                self.assertEqual(inputs["required"][key][1]["default"], value)


if __name__ == "__main__":
    unittest.main()
