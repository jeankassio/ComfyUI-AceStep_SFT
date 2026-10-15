"""CPU tests for the optional loader using ComfyUI's real path and audio code.

AST loading avoids importing node/model modules or initializing CUDA. WAVs are
decoded by the installed PyAV implementation, not a replacement decoder.
"""
import ast
from contextlib import ExitStack
import hashlib
import mimetypes
import os
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
from typing import List, Literal
import unittest
from unittest.mock import Mock, patch
import wave

import av
import torch

import test_generate_advanced
from test_sampling import COMFY_ROOT, NODE_PATH, load_definitions


class LoadReferenceAudioTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.input = self.root / "input"
        self.output = self.root / "output"
        self.temp = self.root / "temp"
        for directory in (self.input, self.output, self.temp):
            directory.mkdir()
        paths = {
            "os": os, "mimetypes": mimetypes, "List": List, "Literal": Literal,
            "input_directory": str(self.input),
            "output_directory": str(self.output),
            "temp_directory": str(self.temp),
            "extension_mimetypes_cache": {},
        }
        load_definitions(COMFY_ROOT / "folder_paths.py", {
            "get_input_directory", "get_output_directory", "get_temp_directory",
            "filter_files_content_types", "annotated_filepath",
            "is_within_directory", "get_annotated_filepath",
            "exists_annotated_filepath",
        }, paths)
        self.folder_paths = SimpleNamespace(**paths)
        audio = {"av": av, "torch": torch}
        load_definitions(COMFY_ROOT / "comfy_extras" / "nodes_audio.py",
                         {"f32_pcm", "load"}, audio)
        self.ns = {
            "os": os, "hashlib": hashlib, "folder_paths": self.folder_paths,
            "load_audio": audio["load"],
        }
        load_definitions(NODE_PATH, {"AceStepSFTLoadReferenceAudio"}, self.ns)
        self.cls = self.ns["AceStepSFTLoadReferenceAudio"]
        self.node = self.cls()

    def write_wav(self, path, samples=None, sample_rate=16000):
        samples = samples or [[-32768], [-12345], [0], [12345], [32767]]
        channels = len(samples[0])
        flattened = [value for frame in samples for value in frame]
        with wave.open(str(path), "wb") as output:
            output.setnchannels(channels)
            output.setsampwidth(2)
            output.setframerate(sample_rate)
            output.writeframes(struct.pack("<" + "h" * len(flattened), *flattened))
        return torch.tensor(samples, dtype=torch.float32).t().unsqueeze(0) / 32768

    def test_disabled_ignores_missing_invalid_traversal_and_selected_files_without_io(self):
        self.write_wav(self.input / "selected.wav")
        (self.input / "invalid.dat").write_bytes(b"not an audio file")
        selections = (
            None, "", "missing.wav", "invalid.dat", "../outside.wav",
            str(self.root / "outside.wav"), "selected.wav",
        )
        with ExitStack() as stack:
            probes = [stack.enter_context(patch.object(
                self.folder_paths, name,
                side_effect=AssertionError("Disabled reference must not resolve or validate a file"),
            )) for name in ("exists_annotated_filepath", "get_annotated_filepath")]
            probes.append(stack.enter_context(patch("builtins.open")))
            probes.append(stack.enter_context(patch.object(os.path, "isfile")))
            probes.append(stack.enter_context(patch.object(hashlib, "sha256")))
            decoder = Mock(side_effect=AssertionError("Disabled reference must not decode"))
            stack.enter_context(patch.dict(self.ns, load_audio=decoder))
            for selection in selections:
                with self.subTest(selection=selection):
                    self.assertEqual(self.node.load_reference_audio(selection, enabled=False), (None,))
                    self.assertIs(self.cls.VALIDATE_INPUTS(selection, enabled=False), True)
                    self.assertEqual(self.cls.IS_CHANGED(selection, enabled=False), "disabled")
            for probe in probes:
                probe.assert_not_called()
            decoder.assert_not_called()

    def test_omitted_arguments_return_no_reference(self):
        with patch.dict(self.ns, load_audio=Mock(side_effect=AssertionError("Unexpected decode"))):
            self.assertEqual(self.node.load_reference_audio(), (None,))
            self.assertIs(self.cls.VALIDATE_INPUTS(audio=""), True)
            self.assertEqual(self.cls.IS_CHANGED(audio=""), "disabled")

    def test_enabled_mono_and_stereo_wav_have_native_batch_rate_and_pcm_values(self):
        for samples, rate in (
            ([[-32768], [-12345], [0], [12345], [32767]], 16000),
            ([[-32768, 32767], [-16384, 16384], [0, 123], [12345, -12345]], 48000),
        ):
            with self.subTest(channels=len(samples[0]), sample_rate=rate):
                path = self.input / "selected.wav"
                expected = self.write_wav(path, samples, rate)
                self.assertIs(self.cls.VALIDATE_INPUTS(path.name, enabled=True), True)
                result, = self.node.load_reference_audio(path.name, enabled=True)
                self.assertEqual(result["sample_rate"], rate)
                self.assertEqual(result["waveform"].dtype, torch.float32)
                self.assertEqual(result["waveform"].device.type, "cpu")
                self.assertEqual(result["waveform"].shape, (1, len(samples[0]), len(samples)))
                torch.testing.assert_close(result["waveform"], expected, rtol=0, atol=0)

    def test_enabled_requires_an_existing_file_and_reports_directories(self):
        (self.input / "voices").mkdir()
        for selection in ("", "missing.wav", "../outside.wav"):
            with self.subTest(selection=selection):
                self.assertIsInstance(self.cls.VALIDATE_INPUTS(selection, enabled=True), str)
        self.assertIn("must be a file", self.cls.VALIDATE_INPUTS("voices", enabled=True))
        with self.assertRaisesRegex(ValueError, "Select a reference audio file"):
            self.node.load_reference_audio(enabled=True)
        with self.assertRaises(FileNotFoundError):
            self.node.load_reference_audio("missing.wav", enabled=True)

    def test_linked_filename_defers_unresolved_validation_but_runtime_still_requires_audio(self):
        with patch.object(self.folder_paths, "exists_annotated_filepath", side_effect=AssertionError("Unresolved filename")):
            self.assertIs(self.cls.VALIDATE_INPUTS(None, enabled=True), True)
        with self.assertRaisesRegex(ValueError, "Select a reference audio file"):
            self.node.load_reference_audio(None, enabled=True)
        self.write_wav(self.input / "linked.wav")
        audio, = self.node.load_reference_audio("linked.wav", enabled=True)
        self.assertEqual(audio["sample_rate"], 16000)

    def test_existing_corrupt_file_fails_the_real_decoder(self):
        (self.input / "broken.wav").write_bytes(b"not an audio container")
        self.assertIs(self.cls.VALIDATE_INPUTS("broken.wav", enabled=True), True)
        with self.assertRaises(av.error.InvalidDataError):
            self.node.load_reference_audio("broken.wav", enabled=True)

    def test_native_annotations_and_subfolder_inputs_resolve_in_the_correct_directory(self):
        (self.input / "voices").mkdir()
        for selection, path, rate in (
            ("voices/nested.wav", self.input / "voices" / "nested.wav", 8000),
            ("voices/nested.wav [input]", self.input / "voices" / "nested.wav", 12000),
            ("reference.wav [output]", self.output / "reference.wav", 22050),
            ("reference.wav [temp]", self.temp / "reference.wav", 44100),
        ):
            with self.subTest(selection=selection):
                expected = self.write_wav(path, sample_rate=rate)
                self.assertIs(self.cls.VALIDATE_INPUTS(selection, enabled=True), True)
                result, = self.node.load_reference_audio(selection, enabled=True)
                self.assertEqual(result["sample_rate"], rate)
                torch.testing.assert_close(result["waveform"], expected, rtol=0, atol=0)
                self.assertEqual(self.cls.IS_CHANGED(selection, enabled=True),
                                 hashlib.sha256(path.read_bytes()).hexdigest())

    def test_load_and_fingerprint_reject_path_traversal_before_reading_or_decoding(self):
        self.write_wav(self.root / "outside.wav")
        (self.input / "voices").mkdir()
        selections = (
            "../outside.wav", "..\\outside.wav", str(self.root / "outside.wav"),
            "voices/../../outside.wav", "../outside.wav [input]",
            "../outside.wav [output]", "..\\outside.wav [temp]",
        )
        decoder = Mock(side_effect=AssertionError("Traversal must not decode"))
        with patch.dict(self.ns, load_audio=decoder), patch("builtins.open") as file_open:
            for selection in selections:
                with self.subTest(selection=selection):
                    self.assertIsInstance(self.cls.VALIDATE_INPUTS(selection, enabled=True), str)
                    with self.assertRaisesRegex(ValueError, "Invalid file path"):
                        self.node.load_reference_audio(selection, enabled=True)
                    with self.assertRaisesRegex(ValueError, "Invalid file path"):
                        self.cls.IS_CHANGED(selection, enabled=True)
            decoder.assert_not_called()
            file_open.assert_not_called()

    def test_fingerprint_tracks_contents_after_the_first_megabyte_and_is_stable_disabled(self):
        path = self.input / "selected.wav"
        self.write_wav(path)
        path.write_bytes(path.read_bytes() + b"A" * (1024 * 1024 + 8))
        before = self.cls.IS_CHANGED(path.name, enabled=True)
        self.assertEqual(before, hashlib.sha256(path.read_bytes()).hexdigest())
        unchanged = self.cls.IS_CHANGED(path.name, enabled=True)
        self.assertEqual(before, unchanged)
        content = path.read_bytes()
        path.write_bytes(content[:-1] + b"B")
        after = self.cls.IS_CHANGED(path.name, enabled=True)
        self.assertEqual(after, hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertNotEqual(before, after)
        self.assertEqual(self.cls.IS_CHANGED(path.name, enabled=False), "disabled")
        path.unlink()
        self.assertEqual(self.cls.IS_CHANGED(path.name, enabled=False), "disabled")

    def test_schema_has_upload_empty_selection_and_disabled_default(self):
        for name in ("z.wav", "a.wav", "movie.mp4", "notes.txt"):
            (self.input / name).write_bytes(b"")
        (self.input / "voices").mkdir()
        schema = self.cls.INPUT_TYPES()["required"]
        self.assertEqual(schema["enabled"][0], "BOOLEAN")
        self.assertIs(schema["enabled"][1]["default"], False)
        self.assertEqual(schema["audio"][0], ["", "a.wav", "movie.mp4", "z.wav"])
        self.assertEqual(schema["audio"][1]["default"], "")
        self.assertIs(schema["audio"][1]["audio_upload"], True)
        self.assertEqual(self.cls.RETURN_TYPES, ("AUDIO",))
        self.assertEqual(self.cls.RETURN_NAMES, ("reference_audio",))
        self.assertEqual(self.cls.FUNCTION, "load_reference_audio")

    def test_node_class_and_display_name_are_registered(self):
        tree = ast.parse(NODE_PATH.read_text(encoding="utf-8"))
        mappings = {
            target.id: item.value
            for item in tree.body if isinstance(item, ast.Assign)
            for target in item.targets if isinstance(target, ast.Name)
            and target.id in {"NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"}
        }
        classes = dict(zip(
            [ast.literal_eval(key) for key in mappings["NODE_CLASS_MAPPINGS"].keys],
            mappings["NODE_CLASS_MAPPINGS"].values,
        ))
        self.assertEqual(classes[self.cls.__name__].id, self.cls.__name__)
        self.assertEqual(ast.literal_eval(mappings["NODE_DISPLAY_NAME_MAPPINGS"])[self.cls.__name__],
                         "Load Reference Audio")

    def test_disabled_output_matches_absent_reference_in_generate_and_advanced(self):
        reference, = self.node.load_reference_audio("../missing.wav", enabled=False)
        for advanced in (False, True):
            with self.subTest(advanced=advanced):
                harness = test_generate_advanced.GenerateAdvancedTests()
                harness.setUp()
                forbidden = Mock(side_effect=AssertionError("None must not configure a reference"))
                harness.ns.update({
                    "_configure_acestep_reference_model": forbidden,
                    "_build_timbre_reference_latent": forbidden,
                    "_ensure_acestep_reference_attention": forbidden,
                })
                settings = dict(
                    model=test_generate_advanced.AdvancedFakeModel(),
                    positive=[[torch.ones(1, 2, 3), {}]],
                    negative=[[torch.ones(1, 2, 3), {}]], steps=6, cfg=7.0,
                    sampler_name="jkass_quality", scheduler="normal", duration=0.4,
                    infer_method="ode", guidance_mode="standard_cfg", guidance_schedule="normal",
                )
                if advanced:
                    absent = harness.generate(**settings)
                    result = harness.generate(**settings, reference_audio=reference)
                else:
                    settings.update(seed=27, denoise=1.0)
                    generator = harness.ns["AceStepSFTGenerate"]()
                    absent = generator.generate(**settings)
                    result = generator.generate(**settings, reference_audio=reference)
                torch.testing.assert_close(result[4]["samples"], absent[4]["samples"], rtol=0, atol=0)
                self.assertNotIn("model_function_wrapper", result[0].model_options)
                for key in ("positive", "negative"):
                    received = harness.calls[-1][key]
                    self.assertEqual(received[0][1], {})
                    torch.testing.assert_close(received[0][0], settings[key][0][0])
                forbidden.assert_not_called()


if __name__ == "__main__":
    unittest.main()
