"""CPU regressions for latent-only cascade stages with a connected VAE.

The existing sampling harness executes Generate and the real sampler without
loading checkpoints or starting CUDA. Disabling output decode must leave both
sampling and source/reference encoding intact.
"""
import copy
import inspect
from types import SimpleNamespace
import unittest

import torch

import test_generate_advanced
import test_reference_audio_generate
import test_advanced_native_tasks


class GenerateDecodeTests(unittest.TestCase):
    def setUp(self):
        self.harness = test_generate_advanced.GenerateAdvancedTests()
        self.harness.setUp()
        self.ns = self.harness.ns

    def generate(self, advanced=False, **overrides):
        if advanced:
            return self.harness.generate(**overrides)
        embedding = torch.ones(1, 2, 3)
        positive = [[embedding, {}]]
        settings = dict(
            model=test_generate_advanced.AdvancedFakeModel(), positive=positive,
            negative=copy.deepcopy(positive), seed=27, steps=6, cfg=7.0,
            sampler_name="jkass_quality", scheduler="normal", denoise=1.0,
            duration=0.4, infer_method="ode", guidance_mode="standard_cfg",
        )
        settings.update(overrides)
        return self.ns["AceStepSFTGenerate"]().generate(**settings)

    def test_default_stays_enabled_and_widget_is_appended_for_both_nodes(self):
        basic = self.ns["AceStepSFTGenerate"]
        parameters = inspect.signature(basic.generate).parameters
        self.assertIs(parameters["decode_audio"].default, True)
        self.assertEqual(list(parameters)[-3:], ["decode_audio", "precompute_conditioning", "voice_profile"])
        self.assertIs(parameters["precompute_conditioning"].default, True)
        self.assertIsNone(parameters["voice_profile"].default)
        for name in ("AceStepSFTGenerate", "AceStepSFTGenerateAdvanced"):
            with self.subTest(node=name):
                optional = self.ns[name].INPUT_TYPES()["optional"]
                self.assertEqual(optional["decode_audio"][0], "BOOLEAN")
                self.assertIs(optional["decode_audio"][1]["default"], True)
                self.assertEqual(list(optional)[-3:], ["decode_audio", "precompute_conditioning", "voice_profile"])

    def test_disabling_decode_preserves_exact_latent_for_basic_and_advanced(self):
        vae = SimpleNamespace(audio_sample_rate=48000)
        for advanced in (False, True):
            with self.subTest(advanced=advanced):
                self.harness.decodes.clear()
                settings = dict(vae=vae, latent_shift=0.1, latent_rescale=1.2)
                decoded = self.generate(advanced=advanced, **settings)
                self.assertIsNotNone(decoded[5])
                self.assertEqual(len(self.harness.decodes), 1)
                torch.testing.assert_close(
                    self.harness.decodes[0],
                    decoded[4]["samples"] * 1.2 + 0.1,
                )
                latent_only = self.generate(
                    advanced=advanced, decode_audio=False, **settings,
                )
                self.assertEqual(len(self.harness.decodes), 1)
                self.assertIsNone(latent_only[5])
                self.assertIs(latent_only[1], vae)
                self.assertEqual(latent_only[4]["type"], "audio")
                torch.testing.assert_close(
                    latent_only[4]["samples"], decoded[4]["samples"],
                    rtol=0, atol=0,
                )

    def test_audio_reference_still_encodes_without_output_decode(self):
        reference_harness = test_reference_audio_generate.ReferenceAudioGenerateTests()
        reference_harness.setUp()
        def unexpected_decode(*args):
            self.fail("A latent-only stage must not decode VAE output")
        reference_harness.ns["_vae_decode_with_optional_tiling"] = unexpected_decode
        reference = {
            "waveform": torch.full((1, 2, 9600), 0.7),
            "sample_rate": 48000,
        }
        for advanced in (False, True):
            with self.subTest(advanced=advanced):
                vae = test_advanced_native_tasks.RecordingVAE()
                result = reference_harness.generate(
                    advanced=advanced, reference_audio=reference, vae=vae,
                    decode_audio=False, duration=2.4,
                )
                self.assertEqual(len(vae.encodes), 1)
                self.assertEqual(result[4]["samples"].shape, (1, 64, 60))
                self.assertIsNone(result[5])
                self.assertIs(result[1], vae)
                self.assertAlmostEqual(
                    reference_harness.native.native_calls[-1]["reference_mean"],
                    0.7, places=6,
                )


if __name__ == "__main__":
    unittest.main()
