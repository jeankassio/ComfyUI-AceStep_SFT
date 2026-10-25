"""Aligned audio examples and offline CPU transcription for voice adaptation."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch


SPEC = importlib.util.spec_from_file_location(
    "voice_adaptation_audio", Path(__file__).resolve().parents[1] / "voice_adaptation_audio.py",
)
AUDIO = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIO)


class ReferenceSegmentTests(unittest.TestCase):
    def audio(self, seconds=30, sample_rate=100, channels=1):
        return {"waveform": torch.arange(seconds * sample_rate, dtype=torch.float32)
                .reshape(1, 1, -1).repeat(1, channels, 1), "sample_rate": sample_rate}

    def fake_transcribe(self, segments, language):
        for index, segment in enumerate(segments):
            segment["text"] = f"spoken crop {index}"

    def test_front_middle_back_crops_are_nonoverlapping_independent_copies(self):
        source = self.audio(channels=2)
        before = source["waveform"].clone()
        with patch.object(AUDIO, "_transcribe_segments", side_effect=self.fake_transcribe) as transcribe:
            segments = AUDIO.prepare_reference_segments(source, language="pt")
        self.assertEqual([(item["start"], item["end"]) for item in segments], [(0, 6), (12, 18), (24, 30)])
        self.assertEqual([item["text"] for item in segments], ["spoken crop 0", "spoken crop 1", "spoken crop 2"])
        transcribe.assert_called_once_with(segments, "pt")
        for segment in segments:
            waveform = segment["audio"]["waveform"]
            self.assertEqual(waveform.shape, (1, 2, 600))
            self.assertNotEqual(waveform.untyped_storage().data_ptr(), source["waveform"].untyped_storage().data_ptr())
            waveform.zero_()
        torch.testing.assert_close(source["waveform"], before)

    def test_short_reference_is_divided_in_three_without_repeating_audio(self):
        with patch.object(AUDIO, "_transcribe_segments", side_effect=self.fake_transcribe):
            segments = AUDIO.prepare_reference_segments(self.audio(seconds=3))
        self.assertEqual([(item["start"], item["end"]) for item in segments], [(0, 1), (1, 2), (2, 3)])
        self.assertEqual([item["audio"]["waveform"][0, 0, 0].item() for item in segments], [0, 100, 200])

    def test_tiny_executable_audio_does_not_repeat_samples(self):
        source = {"waveform": torch.tensor([[[0.1, 0.2]]]), "sample_rate": 16000}
        with patch.object(AUDIO, "_transcribe_segments", side_effect=self.fake_transcribe):
            segments = AUDIO.prepare_reference_segments(source)
        self.assertEqual(len(segments), 2)
        self.assertEqual(sum(item["audio"]["waveform"].shape[-1] for item in segments), 2)

    def test_plain_transcript_stays_with_whole_reference_without_asr(self):
        source = self.audio()
        with patch.object(AUDIO, "_transcribe_segments") as transcribe:
            segments = AUDIO.prepare_reference_segments(source, transcript="  A long sentence.  ")
        transcribe.assert_not_called()
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0]["text"], "A long sentence.")
        self.assertEqual((segments[0]["start"], segments[0]["end"]), (0, 30))
        self.assertNotEqual(segments[0]["audio"]["waveform"].data_ptr(), source["waveform"].data_ptr())

    def test_json_segments_use_supplied_alignment_and_do_not_interpret_paths(self):
        timed = [{"start": 2.5, "end": 4.5, "text": "K:/private/file.txt"},
                 {"start": 10, "end": 12, "text": " second phrase "}]
        with patch.object(AUDIO, "_transcribe_segments") as transcribe:
            segments = AUDIO.prepare_reference_segments(self.audio(), json.dumps(timed))
        transcribe.assert_not_called()
        self.assertEqual([(item["start"], item["end"]) for item in segments], [(2.5, 4.5), (10, 12)])
        self.assertEqual([item["text"] for item in segments], ["K:/private/file.txt", "second phrase"])
        self.assertEqual(segments[0]["audio"]["waveform"][0, 0, 0].item(), 250)

    def test_rejects_malformed_audio(self):
        valid = self.audio()
        invalid = [None, {}, {"waveform": valid["waveform"], "sample_rate": 0},
                   {"waveform": valid["waveform"], "sample_rate": True},
                   {"waveform": torch.zeros(1, 3, 4), "sample_rate": 100},
                   {"waveform": torch.zeros(2, 1, 4), "sample_rate": 100},
                   {"waveform": torch.zeros(1, 1, 0), "sample_rate": 100},
                   {"waveform": torch.tensor([[[float("nan")]]]), "sample_rate": 100},
                   {"waveform": torch.ones(1, 1, 3, dtype=torch.int32), "sample_rate": 100}]
        for value in invalid:
            with self.subTest(value=str(value)[:100]), self.assertRaises(ValueError):
                AUDIO.prepare_reference_segments(value, "spoken text")

    def test_rejects_invalid_alignment_instead_of_guessing_transcript_times(self):
        invalid = ["[]", "{}", "[bad", '[{"start":0,"end":1}]',
                   '[{"start":0,"end":31,"text":"outside"}]',
                   '[{"start":2,"end":1,"text":"reverse"}]',
                   '[{"start":0,"end":NaN,"text":"nan"}]',
                   '[{"start":false,"end":1,"text":"boolean"}]',
                   '[{"start":0,"end":1,"text":null}]',
                   '[{"start":0,"end":0.001,"text":"sub-sample"}]']
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                AUDIO.prepare_reference_segments(self.audio(), value)

    def test_rejects_invalid_crop_duration(self):
        for value in (0, -1, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                AUDIO.prepare_reference_segments(self.audio(), "phrase", crop_seconds=value)


class OfflineTranscriptionTests(unittest.TestCase):
    def backend(self):
        processor = Mock()
        processor.return_value = {"input_features": torch.zeros(1, 128, 3000),
                                  "attention_mask": torch.ones(1, 3000)}
        processor.batch_decode.side_effect = [["first phrase"], ["second phrase"], ["third phrase"]]
        model = Mock()
        model.to.return_value = model
        model.eval.return_value = model
        model.generate.return_value = torch.tensor([[1, 2]])
        processors = SimpleNamespace(from_pretrained=Mock(return_value=processor))
        models = SimpleNamespace(from_pretrained=Mock(return_value=model))
        backend = SimpleNamespace(WhisperProcessor=processors, WhisperForConditionalGeneration=models)
        return backend, processor, model

    def test_auto_asr_loads_once_offline_runs_all_crops_on_cpu(self):
        backend, processor, model = self.backend()
        source = {"waveform": torch.ones(1, 2, 36000), "sample_rate": 12000}
        with patch.dict(sys.modules, {"transformers": backend}):
            segments = AUDIO.prepare_reference_segments(source, language="pt")
        backend.WhisperProcessor.from_pretrained.assert_called_once_with(AUDIO.WHISPER_MODEL, local_files_only=True)
        backend.WhisperForConditionalGeneration.from_pretrained.assert_called_once_with(
            AUDIO.WHISPER_MODEL, local_files_only=True, dtype=torch.float32, attn_implementation="sdpa",
        )
        model.to.assert_called_once_with("cpu")
        model.eval.assert_called_once_with()
        self.assertEqual(model.generate.call_count, 3)
        self.assertEqual([item["text"] for item in segments], ["first phrase", "second phrase", "third phrase"])
        for call in processor.call_args_list:
            self.assertEqual(len(call.args[0]), 16000)
            self.assertEqual(call.kwargs["sampling_rate"], 16000)
        for call in model.generate.call_args_list:
            self.assertEqual(call.kwargs["language"], "pt")
            self.assertEqual(call.kwargs["input_features"].device.type, "cpu")

    def test_auto_language_does_not_force_an_incorrect_language(self):
        backend, _, model = self.backend()
        with patch.dict(sys.modules, {"transformers": backend}):
            AUDIO.prepare_reference_segments({"waveform": torch.ones(1, 1, 3), "sample_rate": 16000})
        self.assertIsNone(model.generate.call_args.kwargs["language"])

    def test_missing_cached_model_fails_without_network_retry(self):
        backend, _, _ = self.backend()
        backend.WhisperProcessor.from_pretrained.side_effect = OSError("not in cache")
        with patch.dict(sys.modules, {"transformers": backend}), self.assertRaisesRegex(RuntimeError, "does not download"):
            AUDIO.prepare_reference_segments({"waveform": torch.ones(1, 1, 30), "sample_rate": 100})
        self.assertEqual(backend.WhisperProcessor.from_pretrained.call_count, 1)
        backend.WhisperForConditionalGeneration.from_pretrained.assert_not_called()

    def test_silent_crop_can_keep_an_empty_transcript(self):
        backend, processor, _ = self.backend()
        processor.batch_decode.side_effect = [["  "], ["some speech"], ["more speech"]]
        with patch.dict(sys.modules, {"transformers": backend}):
            segments = AUDIO.prepare_reference_segments({"waveform": torch.ones(1, 1, 30), "sample_rate": 100})
        self.assertEqual([item["text"] for item in segments], ["", "some speech", "more speech"])


if __name__ == "__main__":
    unittest.main()
