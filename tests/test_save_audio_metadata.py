"""CPU audio-encoding regressions; no models or ComfyUI server are loaded."""
from io import BytesIO
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import av
import torch

from test_sampling import NODE_PATH, load_definitions


class SaveAudioMetadataTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.output = Path(temporary.name)
        self.args = SimpleNamespace(disable_metadata=False)
        paths = SimpleNamespace(
            get_output_directory=lambda: str(self.output),
            get_save_image_path=lambda prefix, output: (output, prefix, 1, "", prefix),
        )
        namespace = {
            "av": av, "BytesIO": BytesIO, "json": json, "os": os,
            "args": self.args, "folder_paths": paths,
        }
        load_definitions(NODE_PATH, {"AceStepSFTSaveAudio"}, namespace)
        self.cls = namespace["AceStepSFTSaveAudio"]
        self.node = self.cls()
        times = torch.arange(7200, dtype=torch.float32) / 48000
        waveform = torch.stack((
            0.2 * torch.sin(2 * torch.pi * 220 * times),
            0.1 * torch.sin(2 * torch.pi * 440 * times),
        )).unsqueeze(0)
        self.audio = {"waveform": waveform, "sample_rate": 48000}
        self.prompt = {"12": {"class_type": "AceStepSFTGenerate", "inputs": {"seed": 42}}}
        self.extra = {"workflow": {"nodes": [{"id": 12, "title": "Canção de teste"}]}}

    def read_audio(self, result):
        entry, = result["ui"]["audio"]
        self.assertEqual(entry["type"], "output")
        path = self.output / entry["filename"]
        self.assertTrue(path.is_file())
        with av.open(str(path)) as container:
            stream = container.streams.audio[0]
            tags = {key.lower(): value for key, value in container.metadata.items()}
            tags.update({key.lower(): value for key, value in stream.metadata.items()})
            self.assertEqual(stream.rate, 48000)
            frames = [frame.to_ndarray() for frame in container.decode(stream)]
        return tags, frames

    def assert_workflow_tags(self, tags):
        self.assertEqual(json.loads(tags["prompt"]), self.prompt)
        self.assertEqual(json.loads(tags["workflow"]), self.extra["workflow"])

    def assert_no_workflow_tags(self, tags):
        self.assertNotIn("prompt", tags)
        self.assertNotIn("workflow", tags)

    def test_switch_is_optional_enabled_by_default_and_named(self):
        definition = self.cls.INPUT_TYPES()["optional"]["save_metadata"]
        self.assertEqual(definition[0], "BOOLEAN")
        self.assertIs(definition[1]["default"], True)
        self.assertEqual(definition[1]["display_name"], "Save Metadata")

    def test_real_formats_embed_or_omit_workflow_without_changing_audio(self):
        for format_name in ("flac", "mp3", "opus"):
            with self.subTest(format=format_name):
                enabled = self.node.save_audio(
                    self.audio, format_name + "-enabled", format_name,
                    prompt=self.prompt, extra_pnginfo=self.extra, save_metadata=True,
                )
                disabled = self.node.save_audio(
                    self.audio, format_name + "-disabled", format_name,
                    prompt=self.prompt, extra_pnginfo=self.extra, save_metadata=False,
                )
                enabled_tags, enabled_frames = self.read_audio(enabled)
                disabled_tags, disabled_frames = self.read_audio(disabled)
                self.assert_workflow_tags(enabled_tags)
                self.assert_no_workflow_tags(disabled_tags)
                self.assertEqual(len(enabled_frames), len(disabled_frames))
                samples = 0
                for left, right in zip(enabled_frames, disabled_frames):
                    torch.testing.assert_close(torch.from_numpy(left), torch.from_numpy(right), rtol=0, atol=0)
                    samples += left.size // 2
                self.assertEqual(samples, self.audio["waveform"].shape[-1])

    def test_disabled_does_not_serialize_hidden_inputs(self):
        with patch.object(json, "dumps", side_effect=AssertionError("Metadata must not be serialized")):
            for format_name in ("flac", "mp3", "opus"):
                with self.subTest(format=format_name):
                    result = self.node.save_audio(
                        self.audio, format_name, format_name,
                        prompt=object(), extra_pnginfo={"workflow": object()},
                        save_metadata=False,
                    )
                    tags, _ = self.read_audio(result)
                    self.assert_no_workflow_tags(tags)

    def test_global_disable_overrides_enabled_switch(self):
        self.args.disable_metadata = True
        with patch.object(json, "dumps", side_effect=AssertionError("Global disable must win")):
            result = self.node.save_audio(
                self.audio, "globally-disabled", prompt=object(), extra_pnginfo={"workflow": object()},
                save_metadata=True,
            )
        tags, _ = self.read_audio(result)
        self.assert_no_workflow_tags(tags)

    def test_legacy_positional_arguments_keep_metadata_enabled(self):
        result = self.node.save_audio(
            self.audio, "legacy", "flac", "128k", self.prompt, self.extra,
        )
        tags, _ = self.read_audio(result)
        self.assert_workflow_tags(tags)


if __name__ == "__main__":
    unittest.main()
