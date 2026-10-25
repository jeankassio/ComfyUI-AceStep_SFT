"""Trigger detection uses tiny CPU files and mocked Civitai responses only."""
import ast
import hashlib
import importlib.util
from http.client import IncompleteRead
from io import BytesIO
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError, URLError

from safetensors import safe_open
from safetensors.torch import load_file, save_file
import torch

from test_sampling import NODE_PATH, load_definitions


MODULE_PATH = Path(__file__).resolve().parents[1] / "lora_triggers.py"
SPEC = importlib.util.spec_from_file_location("acestep_lora_triggers_test", MODULE_PATH)
TRIGGERS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRIGGERS)


class LoraTriggerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.lora = self.directory / "my.style.v1.safetensors"
        self.save_weights()

    def save_weights(self, metadata=None):
        save_file({"test.lora_down.weight": torch.ones(1, 2)}, str(self.lora), metadata=metadata)

    def sidecar(self, suffix, value):
        path = self.lora.with_suffix(suffix)
        path.write_text(value if isinstance(value, str) else json.dumps(value), encoding="utf-8")
        return path

    def resolve(self, **kwargs):
        return TRIGGERS.resolve_trigger_word(str(self.lora), lookup_civitai=False, **kwargs)

    def test_override_wins_over_metadata_sidecar_and_remote(self):
        self.save_weights({"modelspec.trigger_phrase": "embedded trigger"})
        self.sidecar(".trigger.txt", "sidecar trigger")
        with patch.object(TRIGGERS, "urlopen") as request:
            result = TRIGGERS.resolve_trigger_word(
                str(self.lora), trigger_word_override="  my chosen phrase  ", lookup_civitai=True,
            )
        self.assertEqual(result, "my chosen phrase")
        request.assert_not_called()

    def test_explicit_embedded_keys_preserve_phrases(self):
        cases = {
            "modelspec.trigger_phrase": "warm guitar tone",
            "trigger_word": "warm guitar tone",
            "trigger_words": json.dumps(["warm guitar tone", "crooner", "warm guitar tone"]),
            "trainedWords": json.dumps(["warm guitar tone", "crooner", "warm guitar tone"]),
            "ss_trained_words": json.dumps(["warm guitar tone", "crooner", "warm guitar tone"]),
            "custom_tag": "warm guitar tone",
        }
        for key, value in cases.items():
            with self.subTest(key=key):
                self.save_weights({key: value})
                expected = "warm guitar tone, crooner" if value.startswith("[") else value
                self.assertEqual(self.resolve(), expected)

    def test_embedded_trigger_wins_over_sidecar(self):
        self.save_weights({"custom_tag": "embedded"})
        self.sidecar(".trigger.txt", "sidecar")
        self.assertEqual(self.resolve(), "embedded")

    def test_model_specific_sidecars(self):
        cases = (
            (".trigger.txt", "  silky baritone  ", "silky baritone"),
            (".json", {"trigger_word": "silky baritone"}, "silky baritone"),
            (".json", {"trainedWords": ["silky baritone", "warm strings", "silky baritone"]},
             "silky baritone, warm strings"),
            (".civitai.info", {"trainedWords": ["silky baritone"]}, "silky baritone"),
            (".info", {"trainedWords": ["silky baritone"]}, "silky baritone"),
        )
        for suffix, value, expected in cases:
            with self.subTest(suffix=suffix, value=value):
                sidecar = self.sidecar(suffix, value)
                try:
                    self.assertEqual(self.resolve(), expected)
                finally:
                    sidecar.unlink()

    def test_empty_or_malformed_sidecar_does_not_prevent_embedded_loading(self):
        self.save_weights({"custom_tag": "embedded"})
        self.sidecar(".json", "{broken json")
        self.assertEqual(self.resolve(), "embedded")

    def test_malformed_sidecar_falls_back_without_error(self):
        self.sidecar(".json", "{broken json")
        self.assertEqual(self.resolve(), "")

    def test_tag_frequency_and_filename_are_never_inferred(self):
        self.save_weights({"ss_tag_frequency": json.dumps({"dataset": {"popular": 99, "other": 2}})})
        self.assertEqual(self.resolve(), "")

    def test_missing_trigger_is_empty_and_offline_does_not_request(self):
        with patch.object(TRIGGERS, "urlopen") as request:
            self.assertEqual(self.resolve(), "")
        request.assert_not_called()

    def test_side_step_preprocessing_tag_requires_dedicated_adapter_directory(self):
        metadata = self.directory / "preprocess_meta.json"
        metadata.write_text(json.dumps({"custom_tag": "my trained voice"}), encoding="utf-8")
        self.assertEqual(self.resolve(), "")
        (self.directory / "adapter_config.json").write_text("{}", encoding="utf-8")
        self.assertEqual(self.resolve(), "my trained voice")

    def test_side_step_explicit_tag_wins_over_civitai_metadata(self):
        self.save_weights({"custom_tag": "trained voice", "trainedWords": json.dumps(["civitai voice"])})
        self.sidecar(".civitai.info", {"trainedWords": ["sidecar voice"]})
        with patch.object(TRIGGERS, "urlopen") as request:
            self.assertEqual(TRIGGERS.resolve_trigger_word(str(self.lora), lookup_civitai=True), "trained voice")
        request.assert_not_called()

    def test_portable_side_step_metadata_requires_its_own_adapter_directory(self):
        portable = self.directory / "sidestep_metadata.json"
        portable.write_text(json.dumps({"custom_tag": "saved training tag"}), encoding="utf-8")
        self.assertEqual(self.resolve(), "")
        (self.directory / "adapter_config.json").write_text("{}", encoding="utf-8")
        self.assertEqual(self.resolve(), "saved training tag")

    def test_portable_side_step_metadata_changes_invalidate_trigger_cache(self):
        (self.directory / "adapter_config.json").write_text("{}", encoding="utf-8")
        before = TRIGGERS.trigger_file_signature(self.lora)
        portable = self.directory / "sidestep_metadata.json"
        portable.write_text(json.dumps({"custom_tag": "saved training tag"}), encoding="utf-8")
        self.assertNotEqual(before, TRIGGERS.trigger_file_signature(self.lora))

    def test_portable_side_step_metadata_rejects_ambiguous_sample_triggers(self):
        (self.directory / "adapter_config.json").write_text("{}", encoding="utf-8")
        (self.directory / "sidestep_metadata.json").write_text(json.dumps({
            "samples": [{"custom_tag": "one voice"}, {"custom_tag": "another voice"}],
        }), encoding="utf-8")
        self.assertEqual(self.resolve(), "")

    def test_dedicated_side_step_tag_wins_over_civitai_metadata(self):
        self.save_weights({"trainedWords": json.dumps(["civitai voice"])})
        (self.directory / "adapter_config.json").write_text("{}", encoding="utf-8")
        (self.directory / "preprocess_meta.json").write_text(
            json.dumps({"custom_tag": "trained voice"}), encoding="utf-8",
        )
        self.assertEqual(self.resolve(), "trained voice")

    def test_model_specific_side_step_tag_does_not_require_shared_folder_config(self):
        self.save_weights({"modelspec.trigger_phrase": "generic phrase", "trainedWords": json.dumps(["civitai voice"])})
        self.sidecar(".json", {"custom_tag": "trained voice"})
        self.assertEqual(self.resolve(), "trained voice")

    def test_side_step_named_sidecar_has_priority(self):
        self.save_weights({"modelspec.trigger_phrase": "generic phrase"})
        self.sidecar(".sidestep.json", {"custom_tag": "trained voice", "tag_position": "prepend"})
        self.sidecar(".civitai.info", {"trainedWords": ["civitai voice"]})
        self.assertEqual(self.resolve(), "trained voice")

    def test_side_step_dataset_effective_single_trigger(self):
        (self.directory / "adapter_config.json").write_text("{}", encoding="utf-8")
        path = self.directory / "dataset.json"
        path.write_text(json.dumps({
            "metadata": {"custom_tag": "my trained voice"},
            "samples": [{"custom_tag": "my trained voice"}, {}, {"custom_tag": ""}],
        }), encoding="utf-8")
        self.assertEqual(self.resolve(), "my trained voice")

    def test_side_step_dataset_per_sample_unique_trigger(self):
        (self.directory / "adapter_config.json").write_text("{}", encoding="utf-8")
        (self.directory / "dataset.json").write_text(json.dumps({
            "samples": [{"custom_tag": "my trained voice"}, {"custom_tag": "my trained voice"}],
        }), encoding="utf-8")
        self.assertEqual(self.resolve(), "my trained voice")

    def test_side_step_dataset_distinct_triggers_are_not_combined(self):
        (self.directory / "adapter_config.json").write_text("{}", encoding="utf-8")
        (self.directory / "dataset.json").write_text(json.dumps({
            "metadata": {"custom_tag": "default voice"},
            "samples": [{"custom_tag": "voice one"}, {"custom_tag": "voice two"}],
        }), encoding="utf-8")
        self.assertEqual(self.resolve(), "")

    def test_training_metadata_paths_are_not_followed(self):
        elsewhere = self.directory / "elsewhere"
        elsewhere.mkdir()
        foreign_dataset = elsewhere / "dataset.json"
        foreign_dataset.write_text(json.dumps({"metadata": {"custom_tag": "wrong model"}}), encoding="utf-8")
        (self.directory / "adapter_config.json").write_text("{}", encoding="utf-8")
        (self.directory / "training_config.json").write_text(json.dumps({
            "dataset_json": str(foreign_dataset), "dataset_dir": str(elsewhere),
        }), encoding="utf-8")
        self.assertEqual(self.resolve(), "")

    def test_metadata_files_include_matching_stem_without_truncating_dots(self):
        sidecar = self.sidecar(".trigger.txt", "my voice")
        unrelated = self.directory / "other.trigger.txt"
        unrelated.write_text("other voice", encoding="utf-8")
        files = {Path(path) for path in TRIGGERS.trigger_metadata_files(str(self.lora))}
        self.assertIn(sidecar, files)
        self.assertNotIn(unrelated, files)
        self.assertTrue(all(path.parent == self.directory for path in files))

    def test_local_trigger_avoids_remote_request_even_when_enabled(self):
        self.save_weights({"trigger_word": "local voice"})
        with patch.object(TRIGGERS, "urlopen") as request:
            self.assertEqual(TRIGGERS.resolve_trigger_word(str(self.lora), lookup_civitai=True), "local voice")
        request.assert_not_called()

    def test_civitai_uses_actual_full_file_hash_and_declared_words(self):
        payload = {"trainedWords": ["deep warm voice", "soul guitar", "deep warm voice"]}
        with patch.object(TRIGGERS, "urlopen", return_value=BytesIO(json.dumps(payload).encode())) as request:
            result = TRIGGERS.resolve_trigger_word(str(self.lora), lookup_civitai=True)
        self.assertEqual(result, "deep warm voice, soul guitar")
        self.assertEqual(request.call_count, 1)
        args, kwargs = request.call_args
        url = getattr(args[0], "full_url", args[0])
        digest = hashlib.sha256(self.lora.read_bytes()).hexdigest()
        self.assertEqual(url, "https://civitai.com/api/v1/model-versions/by-hash/" + digest)
        self.assertEqual(kwargs.get("timeout"), 10)

    def test_civitai_missing_or_invalid_words_do_not_fail_loading(self):
        for payload in ({}, {"trainedWords": []}, {"trainedWords": None}, {"trainedWords": [None, 5]}, [], None):
            with self.subTest(payload=payload):
                with patch.object(TRIGGERS, "urlopen", return_value=BytesIO(json.dumps(payload).encode())):
                    self.assertEqual(TRIGGERS.resolve_trigger_word(str(self.lora), lookup_civitai=True), "")

    def test_civitai_invalid_json_does_not_fail_loading(self):
        with patch.object(TRIGGERS, "urlopen", return_value=BytesIO(b"<html>unavailable</html>")):
            self.assertEqual(TRIGGERS.resolve_trigger_word(str(self.lora), lookup_civitai=True), "")

    def test_civitai_truncated_response_does_not_fail_loading(self):
        response = BytesIO()
        with patch.object(response, "read", side_effect=IncompleteRead(b'{"trainedWords":', 100)):
            with patch.object(TRIGGERS, "urlopen", return_value=response):
                self.assertEqual(TRIGGERS.resolve_trigger_word(str(self.lora), lookup_civitai=True), "")

    def test_civitai_excessively_nested_json_does_not_fail_loading(self):
        raw = b"[" * 25000 + b"]" * 25000
        with patch.object(TRIGGERS, "urlopen", return_value=BytesIO(raw)):
            self.assertEqual(TRIGGERS.resolve_trigger_word(str(self.lora), lookup_civitai=True), "")

    def test_excessively_nested_sidecar_json_does_not_fail_loading(self):
        self.sidecar(".json", "[" * 25000 + "]" * 25000)
        self.assertEqual(self.resolve(), "")

    def test_excessively_nested_embedded_trigger_json_does_not_fail_loading(self):
        self.save_weights({"trigger_words": "[" * 25000 + "]" * 25000})
        self.assertEqual(self.resolve(), "")

    def test_excessively_nested_embedded_metadata_json_does_not_fail_loading(self):
        self.save_weights({"metadata": "[" * 25000 + "]" * 25000})
        self.assertEqual(self.resolve(), "")

    def test_deep_sidecar_metadata_wrappers_are_not_recursed(self):
        data = {"trainedWords": ["too deeply nested"]}
        for _ in range(64):
            data = {"metadata": data}
        self.sidecar(".json", data)
        self.assertEqual(self.resolve(), "")

    def test_civitai_network_failures_do_not_fail_loading(self):
        errors = (
            HTTPError("https://civitai.com", 404, "Not Found", {}, None),
            URLError("offline"),
            TimeoutError("request timed out"),
        )
        for error in errors:
            with self.subTest(error=type(error).__name__):
                with patch.object(TRIGGERS, "urlopen", side_effect=error):
                    self.assertEqual(TRIGGERS.resolve_trigger_word(str(self.lora), lookup_civitai=True), "")


class LoraTriggerNodeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.lora = Path(temporary.name) / "adapter.safetensors"
        save_file({"test.lora_down.weight": torch.ones(1, 2)}, str(self.lora))
        self.paths = SimpleNamespace(
            get_full_path_or_raise=Mock(return_value=str(self.lora)),
            get_filename_list=Mock(return_value=[self.lora.name]),
        )
        namespace = {"folder_paths": self.paths, "torch": torch}
        load_definitions(NODE_PATH, {"AceStepSFTLoraLoader"}, namespace)
        self.base = namespace["AceStepSFTLoraLoader"]
        self.resolve = Mock(return_value="trained voice")
        namespace.update(resolve_trigger_word=self.resolve, trigger_file_signature=TRIGGERS.trigger_file_signature)
        load_definitions(NODE_PATH, {"AceStepSFTLoraLoaderWithTriggerWord"}, namespace)
        self.cls = namespace["AceStepSFTLoraLoaderWithTriggerWord"]

    def test_new_node_keeps_loader_controls_and_adds_string_output(self):
        self.assertEqual(self.cls.RETURN_TYPES, ("MODEL", "CLIP", "STRING"))
        self.assertEqual(self.cls.RETURN_NAMES, ("model", "clip", "Trigger word"))
        self.assertEqual(self.cls.FUNCTION, self.base.FUNCTION)
        self.assertEqual(self.cls.INPUT_TYPES()["required"], self.base.INPUT_TYPES()["required"])
        self.assertEqual(self.base.RETURN_TYPES, ("MODEL", "CLIP"))
        self.assertEqual(self.base.RETURN_NAMES, ("model", "clip"))

    def test_new_node_uses_same_adapter_loader_and_preserves_strengths(self):
        model, clip, loaded_model, loaded_clip = object(), object(), object(), object()
        with patch.object(self.base, "load_lora", return_value=(loaded_model, loaded_clip)) as loader:
            result = self.cls().load_lora(
                model, clip, self.lora.name, 0.7, 0.3,
                trigger_word_override="manual voice", lookup_civitai=False,
            )
        loader.assert_called_once_with(model, clip, self.lora.name, 0.7, 0.3)
        self.resolve.assert_called_once_with(str(self.lora), "manual voice", False)
        self.assertEqual(result, (loaded_model, loaded_clip, "trained voice"))

    def test_new_node_rechecks_trigger_when_sidecar_changes(self):
        signature = self.cls.IS_CHANGED(self.lora.name)
        self.lora.with_suffix(".trigger.txt").write_text("my first voice", encoding="utf-8")
        added = self.cls.IS_CHANGED(self.lora.name)
        self.assertNotEqual(signature, added)
        self.lora.with_suffix(".trigger.txt").write_text("my second different voice", encoding="utf-8")
        changed = self.cls.IS_CHANGED(self.lora.name)
        self.assertNotEqual(added, changed)
        self.lora.with_suffix(".trigger.txt").unlink()
        self.assertNotEqual(changed, self.cls.IS_CHANGED(self.lora.name))

    def test_new_node_is_registered_with_requested_display_name(self):
        tree = ast.parse(NODE_PATH.read_text(encoding="utf-8"))
        mappings = {}
        for statement in tree.body:
            if not isinstance(statement, ast.Assign) or not isinstance(statement.value, ast.Dict):
                continue
            for target in statement.targets:
                if isinstance(target, ast.Name):
                    mappings[target.id] = dict(zip(
                        [key.value for key in statement.value.keys if isinstance(key, ast.Constant)],
                        statement.value.values,
                    ))
        registration = mappings["NODE_CLASS_MAPPINGS"]["AceStepSFTLoraLoaderWithTriggerWord"]
        self.assertIsInstance(registration, ast.Name)
        self.assertEqual(registration.id, "AceStepSFTLoraLoaderWithTriggerWord")
        self.assertEqual(
            mappings["NODE_DISPLAY_NAME_MAPPINGS"]["AceStepSFTLoraLoaderWithTriggerWord"].value,
            "AceStep 1.5 SFT Lora Loader With Trigger Word",
        )


class LoraTriggerConversionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.adapter = self.directory / "voice_adapter"
        self.adapter.mkdir()
        self.output = self.directory / "voice_comfyui.safetensors"
        self.weights = {
            "base_model.model.layers.0.q_proj.lora_A.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
            "base_model.model.layers.0.q_proj.lora_B.weight": torch.arange(8, dtype=torch.float32).reshape(4, 2),
            "base_model.model.layers.1.v_proj.lora_A.weight": torch.ones(2, 3),
            "base_model.model.layers.1.v_proj.lora_B.weight": torch.ones(4, 2),
        }
        (self.adapter / "adapter_config.json").write_text(json.dumps({
            "lora_alpha": 8, "r": 2, "alpha_pattern": {"layers.0.q_proj": 4},
        }), encoding="utf-8")
        self.namespace = {
            "os": os, "json": json, "safe_open": safe_open,
            "local_trigger_word": TRIGGERS.local_trigger_word, "print": Mock(),
        }
        self.init_path = NODE_PATH.with_name("__init__.py")
        load_definitions(self.init_path, {"_convert_peft_to_comfyui", "_prepare_loras_folder"}, self.namespace)

    def convert(self, metadata=None):
        save_file(self.weights, str(self.adapter / "adapter_model.safetensors"), metadata=metadata)
        with patch.object(TRIGGERS, "urlopen") as request:
            self.namespace["_convert_peft_to_comfyui"](str(self.adapter), str(self.output))
        request.assert_not_called()
        with safe_open(str(self.output), framework="pt", device="cpu") as adapter:
            metadata = adapter.metadata() or {}
        converted = load_file(str(self.output))
        expected = {
            key.replace(".lora_A.weight", ".lora_down.weight").replace(".lora_B.weight", ".lora_up.weight"): tensor
            for key, tensor in self.weights.items()
        }
        expected["base_model.model.layers.0.q_proj.alpha"] = torch.tensor(4.0)
        expected["base_model.model.layers.1.v_proj.alpha"] = torch.tensor(8.0)
        self.assertEqual(converted.keys(), expected.keys())
        for key in expected:
            torch.testing.assert_close(converted[key], expected[key], rtol=0, atol=0)
        return metadata

    def test_conversion_preserves_explicit_source_trigger_and_other_metadata(self):
        metadata = self.convert({"custom_tag": "my trained voice", "format": "pt", "trainer": "test trainer"})
        self.assertEqual(metadata["custom_tag"], "my trained voice")
        self.assertEqual(metadata["modelspec.trigger_phrase"], "my trained voice")
        self.assertEqual(metadata["format"], "pt")
        self.assertEqual(metadata["trainer"], "test trainer")

    def test_conversion_carries_dedicated_side_step_preprocessing_trigger(self):
        (self.adapter / "preprocess_meta.json").write_text(json.dumps({
            "custom_tag": "my side step voice", "tag_position": "prepend",
        }), encoding="utf-8")
        metadata = self.convert({"format": "pt"})
        self.assertEqual(metadata["modelspec.trigger_phrase"], "my side step voice")
        self.assertEqual(TRIGGERS.resolve_trigger_word(str(self.output), lookup_civitai=False), "my side step voice")

    def test_conversion_keeps_portable_trigger_after_original_dataset_is_gone(self):
        (self.adapter / "sidestep_metadata.json").write_text(json.dumps({
            "custom_tag": "portable style", "tag_position": "prepend",
        }), encoding="utf-8")
        metadata = self.convert({"format": "pt"})
        self.assertEqual(metadata["modelspec.trigger_phrase"], "portable style")
        self.assertEqual(TRIGGERS.resolve_trigger_word(str(self.output), lookup_civitai=False), "portable style")

    def test_preexisting_conversion_is_not_rewritten_at_startup(self):
        save_file(self.weights, str(self.adapter / "adapter_model.safetensors"))
        previous = self.directory / "voice_adapter_comfyui.safetensors"
        previous.write_bytes(b"existing user file")
        self.namespace["_LORAS_DIR"] = str(self.directory)
        convert = Mock(side_effect=AssertionError("Existing conversion must remain unchanged"))
        self.namespace["_convert_peft_to_comfyui"] = convert
        self.namespace["_prepare_loras_folder"]()
        convert.assert_not_called()
        self.assertEqual(previous.read_bytes(), b"existing user file")


if __name__ == "__main__":
    unittest.main()
