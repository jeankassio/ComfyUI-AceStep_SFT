"""Adapt Voice preprocessing contracts exercised through the actual node AST."""

import ast
from contextlib import contextmanager
import copy
import importlib.util
import json
import math
from pathlib import Path
import random
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
NODE_PATH = ROOT / "nodes.py"


class SpeechEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lyric_inputs = []

    def forward(self, inputs_embeds, attention_mask=None):
        self.lyric_inputs.append(inputs_embeds.detach().clone())
        return inputs_embeds


class TimbreEncoder(torch.nn.Module):
    def forward(self, reference, order):
        return reference.mean(dim=1, keepdim=True)[..., :8], None


class Diffusion(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Module()
        self.encoder.text_projector = torch.nn.Identity()
        self.encoder.lyric_encoder = SpeechEncoder()
        self.encoder.timbre_encoder = TimbreEncoder()
        self.decoder = torch.nn.Identity()
        self.tokenizer = torch.nn.Identity()
        self.detokenizer = torch.nn.Identity()

    def prepare_condition(self, *args, **kwargs):
        raise AssertionError("The fit should prepare each crop directly.")


class Model:
    def __init__(self, diffusion=None):
        self.model = SimpleNamespace(diffusion_model=diffusion or Diffusion())
        self.model_options = {"caller_option": "preserved"}
        self.object_patches = {}
        self.load_device = torch.device("cpu")
        self.clones = []

    def clone(self):
        result = Model(self.model.diffusion_model)
        result.model_options = self.model_options.copy()
        result.object_patches = self.object_patches.copy()
        self.clones.append(result)
        return result

    def model_dtype(self):
        return torch.float32


class VAE:
    device = "cpu"

    def __init__(self):
        self.inputs = []

    def encode(self, waveform):
        self.inputs.append(waveform.detach().clone())
        frames = waveform.shape[1] // 1920
        mono = waveform[..., :2].mean(dim=-1)
        return mono[:, :frames * 1920].reshape(1, frames, 1920).mean(dim=-1).unsqueeze(1).repeat(1, 64, 1)


class Clock:
    def __init__(self):
        self.started = None
        self.now = 0.0

    def __call__(self):
        if self.started is None:
            self.started = self.now
        return self.now


class AdaptVoiceNodeTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.fit_calls = []
        self.encode_calls = []
        self.transcribed = []
        self.load_calls = []
        self.conditionings = []
        self.progress = []
        self.accepted = True
        self.fit_error = None
        self.model = Model()
        self.vae = VAE()
        self.clip = SimpleNamespace(marker="original clip")
        self.audio = {
            "waveform": (0.2 + torch.sin(torch.arange(3 * 48000) * 0.03) * 0.1).reshape(1, 1, -1),
            "sample_rate": 48000,
        }
        self.original_waveform = self.audio["waveform"].clone()
        self.original_options = copy.deepcopy(self.model.model_options)

        audio_spec = importlib.util.spec_from_file_location(
            "adapt_voice_audio_contract", ROOT / "voice_adaptation_audio.py"
        )
        self.audio_module = importlib.util.module_from_spec(audio_spec)
        audio_spec.loader.exec_module(self.audio_module)

        def transcribe(segments, language):
            self.transcribed.append((len(segments), language))
            for index, segment in enumerate(segments):
                segment["text"] = f"spoken reference phrase {index + 1}"

        self.transcription_patch = patch.object(self.audio_module, "_transcribe_segments", transcribe)
        self.transcription_patch.start()
        self.addCleanup(self.transcription_patch.stop)

        owner = self

        class TextEncode:
            def encode(self, clip, tags, lyrics, *args, **kwargs):
                owner.encode_calls.append((clip, tags, lyrics, args, kwargs))
                text = torch.full((1, 2, 8), 0.25)
                lyric_embedding = torch.full((1, 3, 8), len(owner.encode_calls) / 10)
                # This is the real TextEncode public metadata contract. The
                # diffusion-model argument named lyric_embed is a later mapping.
                positive = [[text, {"conditioning_lyrics": lyric_embedding}]]
                owner.conditionings.append((positive, lyric_embedding.clone()))
                return positive, [[text.clone(), {"conditioning_lyrics": lyric_embedding.clone()}]]

        @contextmanager
        def runtime(model, seconds):
            owner.events.append(("runtime_enter", model, seconds))
            with torch.inference_mode(False), torch.enable_grad():
                temporary = Model()
                owner.runtime_model = temporary
                try:
                    yield temporary, Clock()
                finally:
                    owner.events.append(("runtime_exit", temporary))

        def load_models(models, **kwargs):
            owner.load_calls.append((models, kwargs))

        def fit(decoder, initial, examples, **kwargs):
            owner.fit_calls.append((decoder, initial, examples, kwargs))
            owner.fit_modes = (torch.is_inference_mode_enabled(), torch.is_grad_enabled())
            if owner.fit_error is not None:
                raise owner.fit_error
            kwargs["clock"]()
            kwargs["progress"]()
            return initial + 0.1, {
                "accepted": owner.accepted, "steps": 1, "elapsed_seconds": 0.1,
                "stop_reason": "finished", "timed_out": False,
            }

        class Progress:
            def __init__(self, total):
                self.total = total

            def update_absolute(self, value, total):
                owner.progress.append((value, total))

        def clear(name):
            return lambda model: owner.events.append((name, model))

        namespace = {
            "torch": torch, "F": F, "math": math, "random": random, "json": json,
            "contextmanager": contextmanager,
            "AceStepSFTTextEncode": TextEncode,
            "adaptation_runtime": runtime,
            "fit_timbre": fit,
            "prepare_reference_segments": self.audio_module.prepare_reference_segments,
            "_ensure_acestep_timbre_cls": lambda model: owner.events.append(("install_cls", model)),
            "_ensure_acestep_reference_attention": lambda model: owner.events.append(("install_attention", model)),
            "_get_acestep_silence_latent": lambda length, device, dtype: torch.zeros(1, 64, length, device=device, dtype=dtype),
            "comfy": SimpleNamespace(
                model_management=SimpleNamespace(
                    throw_exception_if_processing_interrupted=lambda: None,
                    load_models_gpu=load_models,
                ),
                utils=SimpleNamespace(ProgressBar=Progress),
            ),
        }
        for name in (
            "clear_voice_profile", "clear_conditioning_precompute",
            "_clear_acestep_silence_latent_patch", "_clear_acestep_reference_timbre_repeats",
            "_clear_acestep_reference_attention", "_clear_acestep_native_task_patches",
            "_clear_acestep_timbre_cls_patch",
        ):
            namespace[name] = clear(name)
        selected = {
            "AceStepSFTAdaptVoice", "_acestep_sampling_model", "_get_acestep_diffusion_model",
            "_audio_to_stereo_48k", "_normalize_audio_to_stereo_48k", "_normalize_audio_peak",
            "_is_audio_payload", "_is_latent_payload", "_match_latent_length", "_build_source_latent",
            "_vae_uses_1d_tiling", "_vae_encode_with_optional_tiling",
            "_select_timbre_reference_audio", "_build_timbre_reference_latent",
        }
        tree = ast.parse(NODE_PATH.read_text(encoding="utf-8"))
        definitions = [item for item in tree.body if getattr(item, "name", None) in selected]
        exec(compile(ast.Module(body=definitions, type_ignores=[]), str(NODE_PATH), "exec"), namespace)
        self.namespace = namespace
        self.node = namespace["AceStepSFTAdaptVoice"]()

    def adapt(self, **kwargs):
        with torch.inference_mode():
            return self.node.adapt(
                self.model, self.clip, self.vae, self.audio,
                max_seconds=kwargs.pop("max_seconds", 1), language="pt", seed=37, **kwargs,
            )

    def assert_inputs_preserved(self):
        torch.testing.assert_close(self.audio["waveform"], self.original_waveform, rtol=0, atol=0)
        self.assertEqual(self.audio["sample_rate"], 48000)
        self.assertEqual(self.model.model_options, self.original_options)
        self.assertEqual(self.model.object_patches, {})
        self.assertEqual(self.model.clones, [])
        for positive, lyrics in self.conditionings:
            self.assertEqual(set(positive[0][1]), {"conditioning_lyrics"})
            torch.testing.assert_close(positive[0][1]["conditioning_lyrics"], lyrics, rtol=0, atol=0)

    def test_real_textencode_metadata_prepares_three_speech_examples(self):
        result = self.adapt()
        profile, rendered = result["result"]
        self.assertEqual(self.transcribed, [(3, "pt")])
        self.assertEqual([call[2] for call in self.encode_calls], [
            "spoken reference phrase 1", "spoken reference phrase 2", "spoken reference phrase 3",
        ])
        for clip, tags, text, args, kwargs in self.encode_calls:
            self.assertIs(clip, self.clip)
            self.assertIn("spoken voice", tags)
            self.assertFalse(kwargs["generate_audio_codes"])
        self.assertEqual(self.fit_modes, (False, True))
        _, initial, examples, settings = self.fit_calls[0]
        self.assertEqual(tuple(initial.shape), (1, 1, 8))
        self.assertEqual(settings["seconds"], 1)
        self.assertEqual(settings["seed"], 37)
        self.assertEqual(len(examples), 3)
        for index, example in enumerate(examples):
            self.assertEqual(tuple(example["target"].shape), (1, 25, 64))
            self.assertEqual(tuple(example["context"].shape), (1, 25, 128))
            self.assertEqual(float(example["context"][..., :64].abs().max()), 0)
            self.assertEqual(float(example["context"][..., 64:].min()), 1)
            torch.testing.assert_close(example["lyrics"], torch.full((1, 3, 8), (index + 1) / 10))
            for value in example.values():
                self.assertFalse(value.is_inference())
                self.assertFalse(value.requires_grad)
        self.assertEqual(len(self.vae.inputs), 4)
        self.assertEqual(profile["version"], 1)
        self.assertEqual(tuple(profile["embedding"].shape), (1, 1, 8))
        self.assertEqual(tuple(profile["reference_audio"]["samples"].shape), (1, 64, 750))
        self.assertEqual(profile["embedding"].device.type, "cpu")
        self.assertEqual(profile["embedding"].dtype, torch.float32)
        self.assertEqual(json.loads(rendered)["segments"][1]["text"], "spoken reference phrase 2")
        self.assertEqual(result["ui"]["text"], [rendered])
        self.assertTrue(self.load_calls[0][1]["force_full_load"])
        self.assertEqual(self.progress[-1], (100, 100))
        self.assert_inputs_preserved()

    def test_rejected_fit_keeps_native_reference_and_omits_learned_embedding(self):
        self.accepted = False
        profile, rendered = self.adapt()["result"]
        self.assertIsNone(profile["embedding"])
        self.assertFalse(json.loads(rendered)["accepted"])
        self.assertEqual(tuple(profile["reference_audio"]["samples"].shape), (1, 64, 750))
        self.assert_inputs_preserved()

    def test_zero_budget_bypasses_preprocessing_and_model_runtime(self):
        profile, rendered = self.adapt(max_seconds=0)
        self.assertIs(profile["reference_audio"], self.audio)
        self.assertIsNone(profile["embedding"])
        self.assertEqual(json.loads(rendered)["stop_reason"], "disabled")
        self.assertEqual(self.events, [])
        self.assertEqual(self.transcribed, [])
        self.assertEqual(self.encode_calls, [])
        self.assertEqual(self.vae.inputs, [])
        self.assertEqual(self.fit_calls, [])
        self.assert_inputs_preserved()

    def test_absent_reference_skips_transcription_and_adaptation(self):
        with torch.inference_mode():
            profile, rendered = self.node.adapt(self.model, self.clip, self.vae, None)
        self.assertIsNone(profile)
        self.assertEqual(json.loads(rendered)["stop_reason"], "no_reference_audio")
        self.assertEqual(self.events, [])
        self.assertEqual(self.transcribed, [])
        self.assertEqual(self.encode_calls, [])
        self.assertEqual(self.vae.inputs, [])
        self.assertEqual(self.fit_calls, [])
        self.assert_inputs_preserved()

    def test_success_and_error_both_clean_scoped_model_patches(self):
        for error in (None, RuntimeError("forced fit failure")):
            with self.subTest(error=error):
                self.events.clear()
                self.fit_error = error
                if error is None:
                    self.adapt()
                else:
                    with self.assertRaisesRegex(RuntimeError, "forced fit failure"):
                        self.adapt()
                self.assertEqual(self.events[0][0], "runtime_enter")
                self.assertEqual(self.events[-1][0], "runtime_exit")
                names = [event[0] for event in self.events]
                self.assertEqual(names.count("clear_voice_profile"), 2)
                self.assertEqual(names.count("_clear_acestep_timbre_cls_patch"), 2)
                self.assertEqual(names.count("install_cls"), 1)
                self.assertEqual(names.count("install_attention"), 1)
                self.assert_inputs_preserved()

    def test_supplied_transcript_is_for_reference_not_the_new_song(self):
        spoken = "These are the words in the uploaded recording."
        self.adapt(transcript=spoken)
        self.assertEqual(self.transcribed, [])
        self.assertEqual([call[2] for call in self.encode_calls], [spoken])
        self.assertEqual(tuple(self.fit_calls[0][2][0]["target"].shape), (1, 75, 64))
        self.assert_inputs_preserved()


if __name__ == "__main__":
    unittest.main()
