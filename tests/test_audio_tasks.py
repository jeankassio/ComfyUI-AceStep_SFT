"""Check source-audio reproducibility and scoped native editing patches."""

import ast
import functools
import inspect
import math
from pathlib import Path
import random
from types import MethodType, SimpleNamespace
import unittest

import torch
import torch.nn.functional as F
import torchaudio


SOURCE = Path(__file__).resolve().parents[1] / "nodes.py"
names = {
    "_audio_to_stereo_48k", "_normalize_audio_to_stereo_48k", "_normalize_audio_peak", "_vae_uses_1d_tiling",
    "_vae_encode_with_optional_tiling", "_is_audio_payload", "_is_latent_payload",
    "_vae_decode_with_optional_tiling",
    "_get_source_duration_seconds", "_match_latent_length", "_build_source_latent",
    "_select_timbre_reference_audio", "_build_timbre_reference_latent",
    "_get_acestep_diffusion_model", "_ensure_acestep_native_forward",
    "_clear_acestep_native_task_patches", "_repeat_native_tensor",
    "_match_batch_indices",
}
tree = ast.parse(SOURCE.read_text(encoding="utf-8-sig"))
selected = [node for node in tree.body if (
    isinstance(node, ast.FunctionDef) and node.name in names
) or (isinstance(node, ast.ClassDef) and node.name in {
    "AceStepSFTAudioDuration", "AceStepSFTLoadReferenceAudio",
})]
ns = dict(torch=torch, F=F, torchaudio=torchaudio, math=math, random=random,
          functools=functools, inspect=inspect, MethodType=MethodType,
          model_trange=lambda *args, **kwargs: range(*args))
exec(compile(ast.Module(body=selected, type_ignores=[]), str(SOURCE), "exec"), ns)


class SamplingVAE:
    latent_dim = 1
    downscale_ratio = 1920

    def __init__(self, device="cpu"):
        self.device = torch.device(device)
        self.tiles = None

    def encode(self, samples):
        return torch.randn(samples.shape[0], 64, samples.shape[1] // 1920,
                           device=self.device)

    def encode_tiled(self, samples, **kwargs):
        self.tiles = kwargs
        return self.encode(samples)


class FakeDiffusion:
    tokenizer = detokenizer = encoder = object()

    def forward(self, x, timestep, context, **kwargs):
        return x + 1

    def prepare_condition(self, context, _, lyrics, __, refer, order,
                          source, chunks, covers, **kwargs):
        self.last_source = source
        return context, None, source

    def decoder(self, hidden_states, **kwargs):
        return hidden_states + kwargs["context_latents"]


class FakePatcher:
    """The object-patch lifecycle, without importing or loading Comfy models."""

    def __init__(self):
        self.model = SimpleNamespace(diffusion_model=FakeDiffusion(), other="base")
        self.model_options = {}
        self.object_patches = {}
        self.object_patches_backup = {}

    def add_object_patch(self, name, value):
        self.object_patches[name] = value

    def get_model_object(self, name):
        return self.object_patches.get(name, self.object_patches_backup.get(
            name, self.model.diffusion_model.forward))

    def patch_model(self, load_weights=False):
        for name, value in self.object_patches.items():
            obj = self.model.diffusion_model if name.startswith("diffusion_model.") else self.model
            attr = name.split(".")[-1]
            self.object_patches_backup.setdefault(name, getattr(obj, attr))
            setattr(obj, attr, value)

    def unpatch_model(self, unpatch_weights=False):
        for name, value in self.object_patches_backup.items():
            obj = self.model.diffusion_model if name.startswith("diffusion_model.") else self.model
            setattr(obj, name.split(".")[-1], value)
        self.object_patches_backup.clear()


class AudioTaskTests(unittest.TestCase):
    def test_source_peak_scaling_preserves_waveform_and_stereo_balance(self):
        waveform = torch.tensor([[[2.0, 1.5, -0.5, -2.0],
                                  [1.0, 0.75, -0.25, -1.0]]])
        original = waveform.clone()
        actual = ns["_normalize_audio_to_stereo_48k"](waveform, 48000)
        torch.testing.assert_close(actual, original / 2)
        self.assertTrue(torch.equal(waveform, original))
        quiet = original / 4
        torch.testing.assert_close(
            ns["_normalize_audio_to_stereo_48k"](quiet, 48000), quiet
        )

    def test_invalid_source_audio_is_rejected_before_vae_encoding(self):
        for value in (float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaisesRegex(RuntimeError, "NaN or Inf"):
                ns["_normalize_audio_to_stereo_48k"](torch.full((1, 2, 20), value), 48000)

    def test_chunked_decode_matches_full_convolution_at_every_seam(self):
        class ConvolutionVAE:
            latent_dim = 1
            upscale_ratio = 2

            def decode(self, samples):
                audio = F.avg_pool1d(samples[:, :2], 7, stride=1, padding=3)
                return audio.repeat_interleave(2, dim=-1).movedim(1, -1)

        vae = ConvolutionVAE()
        generator = torch.Generator().manual_seed(34)
        for length in (255, 256, 257, 384, 512, 513, 529):
            with self.subTest(length=length):
                samples = torch.randn(2, 64, length, generator=generator)
                expected = vae.decode(samples)
                actual = ns["_vae_decode_with_optional_tiling"](vae, samples, True)
                self.assertEqual(actual.shape, (2, length * 2, 2))
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_non_1d_decode_keeps_native_tiler(self):
        class VAE:
            latent_dim = 2

            def decode_tiled(self, samples):
                return samples + 1

            def decode(self, samples):
                return samples + 2

        samples = torch.zeros(1, 2, 300)
        decode = ns["_vae_decode_with_optional_tiling"]
        torch.testing.assert_close(decode(VAE(), samples, True), samples + 1)
        torch.testing.assert_close(decode(VAE(), samples, False), samples + 2)

    def test_batch_indices_follow_source_repeat_without_changing_caller(self):
        indices = [4, 7]
        self.assertEqual(ns["_match_batch_indices"](indices, 3), [4, 7, 4])
        self.assertEqual(ns["_match_batch_indices"](indices, 1), [4])
        self.assertEqual(indices, [4, 7])
        self.assertIsNone(ns["_match_batch_indices"](None, 3))
        self.assertIsNone(ns["_match_batch_indices"]([], 3))

    def test_duration_handles_both_channel_layouts(self):
        for shape in ((48000,), (2, 48000), (48000, 2),
                      (1, 2, 48000), (1, 48000, 2)):
            with self.subTest(shape=shape):
                audio = {"waveform": torch.zeros(shape), "sample_rate": 48000}
                self.assertEqual(ns["_get_source_duration_seconds"](audio, 48000), 1.0)
                self.assertEqual(ns["AceStepSFTAudioDuration"]().get_duration(audio), (1,))

    def test_duration_of_disabled_reference_is_zero_without_loading_a_file(self):
        reference, = ns["AceStepSFTLoadReferenceAudio"]().load_reference_audio(
            audio="missing.wav", enabled=False,
        )
        self.assertIsNone(reference)
        self.assertEqual(ns["AceStepSFTAudioDuration"]().get_duration(reference), (0,))

    def test_duration_still_rejects_malformed_audio(self):
        duration = ns["AceStepSFTAudioDuration"]().get_duration
        for audio in ({}, {"waveform": None, "sample_rate": 48000},
                      {"waveform": [], "sample_rate": 48000},
                      {"waveform": torch.tensor(0.0), "sample_rate": 48000}):
            with self.subTest(audio=audio), self.assertRaisesRegex(ValueError, "valid waveform"):
                duration(audio)
        for sample_rate in (None, 0, -48000, "48000"):
            with self.subTest(sample_rate=sample_rate), self.assertRaisesRegex(ValueError, "valid sample rate"):
                duration({"waveform": torch.zeros(1, 1, 48000), "sample_rate": sample_rate})

    def test_encode_tiles_follow_vae_hop(self):
        for hop in (1920, 2048):
            vae = SamplingVAE()
            vae.downscale_ratio = hop
            ns["_vae_encode_with_optional_tiling"](vae, torch.zeros(1, 48000, 2), True, seed=7)
            self.assertEqual(vae.tiles, dict(tile_x=256 * hop, tile_y=1, overlap=64 * hop))

    def assert_seeded_encode(self, device):
        vae = SamplingVAE(device)
        samples = torch.zeros(1, 48000, 2)
        cpu_rng = torch.get_rng_state().clone()
        gpu_rng = torch.cuda.get_rng_state(device).clone() if device != "cpu" else None
        encode = ns["_vae_encode_with_optional_tiling"]
        first = encode(vae, samples, True, seed=123)
        self.assertTrue(torch.equal(cpu_rng, torch.get_rng_state()))
        if gpu_rng is not None:
            self.assertTrue(torch.equal(gpu_rng, torch.cuda.get_rng_state(device)))
        torch.rand(3)
        second = encode(vae, samples, True, seed=123)
        different = encode(vae, samples, True, seed=124)
        self.assertTrue(torch.equal(first, second))
        self.assertFalse(torch.equal(first, different))

    def test_seeded_encode_preserves_cpu_rng(self):
        self.assert_seeded_encode("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_seeded_encode_preserves_cuda_rng(self):
        self.assert_seeded_encode("cuda:0")

    def test_source_and_reference_forward_seed(self):
        vae = SamplingVAE()
        audio = {"waveform": torch.full((1, 2, 48000), 0.2), "sample_rate": 48000}
        source = ns["_build_source_latent"]
        ref = ns["_build_timbre_reference_latent"]
        self.assertTrue(torch.equal(source(vae, audio, 1, 25, 48000, True, seed=4),
                                    source(vae, audio, 1, 25, 48000, True, seed=4)))
        self.assertTrue(torch.equal(ref(vae, audio, 48000, True, seed=4),
                                    ref(vae, audio, 48000, True, seed=4)))

    def test_native_forward_leaves_class_and_other_instances_unchanged(self):
        patcher = FakePatcher()
        untouched = FakeDiffusion()
        original = FakeDiffusion.forward
        ns["_ensure_acestep_native_forward"](patcher)
        self.assertIs(FakeDiffusion.forward, original)
        self.assertIs(untouched.forward.__func__, original)
        self.assertIs(patcher.model.diffusion_model.forward.__func__, original)
        installed = patcher.object_patches["diffusion_model.forward"]
        ns["_ensure_acestep_native_forward"](patcher)
        self.assertIs(installed, patcher.object_patches["diffusion_model.forward"])
        patcher.patch_model()
        x = torch.zeros(1, 64, 5)
        context = torch.zeros(1, 2, 3)
        self.assertTrue(torch.equal(patcher.model.diffusion_model.forward(x, 1, context), x + 1))
        out = patcher.model.diffusion_model.forward(
            x, 1, context, refer_audio=x, src_latents=torch.full_like(x, 2)
        )
        self.assertTrue(torch.equal(out, x + 2))
        ns["_clear_acestep_native_task_patches"](patcher)
        self.assertIs(patcher.model.diffusion_model.forward.__func__, original)
        self.assertNotIn("diffusion_model.forward", patcher.object_patches)

    def test_cleanup_restores_external_forward_and_unrelated_patches(self):
        patcher = FakePatcher()
        def external(self, x, timestep, context, **kwargs):
            return x + 3
        previous = MethodType(external, patcher.model.diffusion_model)
        patcher.add_object_patch("diffusion_model.forward", previous)
        patcher.add_object_patch("other", "external")
        ns["_ensure_acestep_native_forward"](patcher)
        patcher.patch_model()
        ns["_clear_acestep_native_task_patches"](patcher)
        self.assertIs(patcher.object_patches["diffusion_model.forward"], previous)
        self.assertIs(patcher.model.diffusion_model.forward, previous)
        self.assertEqual(patcher.model.other, "external")

    def test_forward_supporting_native_context_needs_no_patch(self):
        patcher = FakePatcher()
        def native(self, x, timestep, context, src_latents=None, chunk_masks=None,
                   precomputed_lm_hints_25Hz=None):
            return x
        supported = MethodType(native, patcher.model.diffusion_model)
        patcher.add_object_patch("diffusion_model.forward", supported)
        ns["_ensure_acestep_native_forward"](patcher)
        self.assertIs(patcher.object_patches["diffusion_model.forward"], supported)


if __name__ == "__main__":
    unittest.main()
