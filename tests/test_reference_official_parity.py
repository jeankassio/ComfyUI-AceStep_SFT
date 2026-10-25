"""Official reference preprocessing and encoder masks, using CPU fixtures only."""
import math
import random
import unittest

import torch
import torchaudio

import test_reference_attention
from test_advanced_native_tasks import RecordingVAE
from test_sampling import NODE_PATH, load_definitions
import comfy.lora


def reference_namespace():
    namespace = dict(torch=torch, torchaudio=torchaudio, math=math, random=random)
    load_definitions(NODE_PATH, {
        "_audio_to_stereo_48k", "_normalize_audio_to_stereo_48k", "_normalize_audio_peak",
        "_select_timbre_reference_audio", "_build_timbre_reference_latent",
        "_is_audio_payload", "_is_latent_payload", "_vae_uses_1d_tiling",
        "_vae_encode_with_optional_tiling",
    }, namespace)
    return namespace


class ReferencePreprocessingParityTests(unittest.TestCase):
    def setUp(self):
        self.ns = reference_namespace()

    def test_three_ten_second_segments_match_official_seeded_offsets(self):
        # Golden offsets from ACE-Step process_reference_audio with Random(37).
        cases = ((91, (715133, 2093279, 3008052)),
                 (47, (48026, 1022017, 1522581)),
                 (12, (89391, 655659, 1164006)),
                 (3, (0, 480000, 960000)))
        for seconds, starts in cases:
            with self.subTest(seconds=seconds):
                left = torch.linspace(-0.8, 0.8, seconds * 48000)
                original = torch.stack((left, -left))
                audio = original.unsqueeze(0)
                if seconds < 30:
                    audio = audio.repeat(1, 1, math.ceil(30 / seconds))
                expected = torch.cat([audio[..., start:start + 480000]
                                      for start in starts], dim=-1)
                result = self.ns["_select_timbre_reference_audio"](
                    original, 48000, rng=random.Random(37))
                self.assertEqual(result.shape, (1, 2, 1440000))
                torch.testing.assert_close(result, expected, rtol=0, atol=0)
                torch.testing.assert_close(original[0], left, rtol=0, atol=0)

    def test_reference_clips_over_range_samples_without_scaling_quiet_voice(self):
        original = torch.tensor([2.0, -2.0, 0.25, 0.5]).repeat(12000)
        before = original.clone()
        result = self.ns["_select_timbre_reference_audio"](
            original, 48000, rng=random.Random(37))
        expected = torch.tensor([1.0, -1.0, 0.25, 0.5]).repeat(360000)
        self.assertEqual(result.shape, (1, 2, 1440000))
        torch.testing.assert_close(result[0, 0], expected, rtol=0, atol=0)
        torch.testing.assert_close(result[0, 1], expected, rtol=0, atol=0)
        torch.testing.assert_close(original, before, rtol=0, atol=0)

    def test_resampling_matches_official_stereo_48k_input(self):
        original = (torch.sin(torch.arange(16000) * 0.07) * 0.2).unsqueeze(0)
        stereo = original.repeat(2, 1)
        expected = torchaudio.transforms.Resample(16000, 48000)(stereo).clamp(-1, 1)
        result = self.ns["_select_timbre_reference_audio"](
            original, 16000, rng=random.Random(37))
        expected = expected.repeat(1, 30).unsqueeze(0)
        torch.testing.assert_close(result, expected, rtol=1e-6, atol=1e-7)

    def test_silent_and_empty_audio_never_reaches_the_vae(self):
        for waveform in (torch.empty(1, 2, 0), torch.zeros(1, 1, 48000),
                         torch.full((1, 2, 48000), 5e-7)):
            with self.subTest(shape=tuple(waveform.shape), empty=not waveform.numel()):
                vae = RecordingVAE()
                with self.assertRaisesRegex(ValueError, "empty or silent"):
                    self.ns["_build_timbre_reference_latent"](
                        vae, {"waveform": waveform, "sample_rate": 48000},
                        48000, False, seed=37)
                self.assertEqual(vae.encodes, [])

    def test_empty_latent_is_rejected_before_repeat_or_sampling(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            self.ns["_build_timbre_reference_latent"](
                None, {"samples": torch.empty(1, 64, 0)}, 48000, False, seed=37)

    def test_small_non_silent_voice_is_not_amplified_and_keeps_global_rng(self):
        vae = RecordingVAE()
        reference = {"waveform": torch.full((1, 1, 48000), 2e-6),
                     "sample_rate": 48000}
        python_state = random.getstate()
        torch_state = torch.random.get_rng_state().clone()
        result = self.ns["_build_timbre_reference_latent"](
            vae, reference, 48000, False, seed=37)
        self.assertEqual(random.getstate(), python_state)
        torch.testing.assert_close(torch.random.get_rng_state(), torch_state, rtol=0, atol=0)
        self.assertEqual(len(vae.encodes), 1)
        self.assertEqual(vae.encodes[0].shape, (1, 1440000, 2))
        torch.testing.assert_close(vae.encodes[0], torch.full_like(vae.encodes[0], 2e-6),
                                   rtol=0, atol=0)
        self.assertEqual(result.shape, (1, 64, 750))


class ReferenceEncoderParityTests(unittest.TestCase):
    def setUp(self):
        self.fixture = test_reference_attention.ReferenceAttentionTests()
        self.ns = test_reference_attention.node_ns

    def test_both_encoders_ignore_padded_tokens_in_sliding_and_full_attention(self):
        patcher, diffusion = self.fixture.model(sliding_window=4)
        with self.ns["_acestep_sampling_model"](patcher) as clone:
            self.ns["_ensure_acestep_reference_attention"](clone)
            clone.patch_model(load_weights=False)
            for name in ("lyric_encoder", "timbre_encoder"):
                encoder = getattr(diffusion.encoder, name)
                for index in (0, 1):
                    for dtype in (torch.bool, torch.float32):
                        with self.subTest(encoder=name, layer=index, mask_dtype=dtype):
                            original = torch.randn(2, 12, 16)
                            changed = original.clone()
                            mask = torch.ones(2, 12, dtype=dtype)
                            mask[0, 7:] = 0
                            mask[1, 5:] = 0
                            changed[0, 7:] += 100
                            changed[1, 5:] -= 100
                            attention = encoder.layers[index].self_attn
                            first = attention(original, attention_mask=mask)
                            second = attention(changed, attention_mask=mask)
                            torch.testing.assert_close(first[0, :7], second[0, :7], rtol=0, atol=0)
                            torch.testing.assert_close(first[1, :5], second[1, :5], rtol=0, atol=0)
                            self.assertTrue(torch.isfinite(first).all())

    def test_sampling_failure_restores_both_encoders_and_keeps_applied_lora(self):
        patcher, diffusion = self.fixture.model(sliding_window=4)
        key = "diffusion_model.decoder.layers.0.self_attn.q_proj.weight"
        original = patcher.model.state_dict()[key].clone()
        up = torch.full((original.shape[0], 2), 0.05)
        down = torch.full((2, original.shape[1]), 0.1)
        adapter = comfy.lora.load_lora({
            "test.lora_up.weight": up, "test.lora_down.weight": down,
            "test.alpha": torch.tensor(2.0),
        }, {"test": key})
        self.assertEqual(patcher.add_patches(adapter, 0.75), [key])
        patcher.patch_weight_to_device(key, device_to=torch.device("cpu"))
        expected = original + (up @ down) * 0.75
        torch.testing.assert_close(patcher.model.state_dict()[key], expected)
        forwards = {name: [layer.self_attn.forward for layer in getattr(diffusion.encoder, name).layers]
                    for name in ("lyric_encoder", "timbre_encoder")}
        windows = {name: self.fixture.windows(getattr(diffusion.encoder, name))
                   for name in forwards}
        with self.assertRaisesRegex(RuntimeError, "sampling failed"):
            with self.ns["_acestep_sampling_model"](patcher) as clone:
                self.ns["_ensure_acestep_reference_attention"](clone)
                clone.patch_model(load_weights=False)
                for name in forwards:
                    self.assertEqual(self.fixture.windows(getattr(diffusion.encoder, name)),
                                     [4, None, 4, None])
                raise RuntimeError("sampling failed")
        for name in forwards:
            encoder = getattr(diffusion.encoder, name)
            self.assertEqual(self.fixture.windows(encoder), windows[name])
            self.assertEqual([layer.self_attn.forward for layer in encoder.layers], forwards[name])
        self.assertEqual(clone.object_patches, {})
        self.assertNotIn(test_reference_attention.PATCH_STATE, clone.model_options)
        self.assertIs(clone.patches[key][0][1], adapter[key])
        torch.testing.assert_close(patcher.model.state_dict()[key], expected)
        self.assertEqual(patcher.patches[key][0][0], 0.75)


if __name__ == "__main__":
    unittest.main()
