"""CPU prompt-preflight regressions for native If/Else audio routing.

Use the installed ComfyUI validator and real SwitchNode schema. ACE-Step
definitions are AST-loaded so no checkpoints, custom-node startup, servers,
sampling, or CUDA initialization are needed.
"""
import ast
import asyncio
import copy
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch


NODE_PATH = Path(__file__).resolve().parents[1] / "nodes.py"
COMFY_ROOT = NODE_PATH.parents[2]
sys.path.insert(0, str(COMFY_ROOT))
from comfy.cli_args import args
args.cpu = True
import execution
import comfy.samplers
from comfy_extras.nodes_logic import SwitchNode
from comfy_execution.validation import validate_node_input


def load_node_definitions():
    # An optional snapshot allows proving this regression against an earlier
    # file without editing the installed node or relying on its Git state.
    source = Path(os.environ.get("ACESTEP_VALIDATION_TEST_SOURCE", NODE_PATH))
    tree = ast.parse(source.read_text(encoding="utf-8-sig"))
    names = {
        "AceStepSFTGenerate", "AceStepSFTGenerateAdvanced",
        "AceStepSFTCoverRemix", "AceStepSFTInpaint", "_is_audio_payload",
        "_is_latent_payload", "_validate_acestep_task",
    }
    namespace = {"torch": torch, "math": math, "comfy": comfy,
                 "validate_node_input": validate_node_input}
    for name in ("GUIDANCE_MODES", "ACESTEP_SFT_TASK_TYPE"):
        assignment = next(item for item in tree.body
                          if isinstance(item, ast.Assign)
                          and any(isinstance(target, ast.Name) and target.id == name
                                  for target in item.targets))
        namespace[name] = ast.literal_eval(assignment.value)
    selected = [item for item in tree.body if getattr(item, "name", None) in names]
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), "exec"), namespace)
    return namespace


def source_class(return_type):
    class Source:
        RETURN_TYPES = (return_type,)

        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {}}

    return Source


class SwitchValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.namespace = load_node_definitions()
        cls.node_names = ("AceStepSFTGenerate", "AceStepSFTGenerateAdvanced",
                          "AceStepSFTCoverRemix", "AceStepSFTInpaint")
        cls.registered = {name: cls.namespace[name] for name in cls.node_names}
        cls.registered["ComfySwitchNode"] = SwitchNode
        for return_type in ("MODEL", "CONDITIONING", "VAE", "AUDIO", "LATENT",
                            "AUDIO,LATENT", "*", "STRING", "IMAGE"):
            cls.registered["ValidationSource_" + return_type] = source_class(return_type)
        cls.previous = {name: execution.nodes.NODE_CLASS_MAPPINGS.get(name)
                        for name in cls.registered}
        execution.nodes.NODE_CLASS_MAPPINGS.update(cls.registered)

    @classmethod
    def tearDownClass(cls):
        for name, previous in cls.previous.items():
            if previous is None:
                execution.nodes.NODE_CLASS_MAPPINGS.pop(name, None)
            else:
                execution.nodes.NODE_CLASS_MAPPINGS[name] = previous

    def default_inputs(self, name):
        schema = self.namespace[name].INPUT_TYPES()
        inputs = {}
        for key, definition in schema["required"].items():
            options = definition[1] if len(definition) > 1 else {}
            if "default" in options:
                inputs[key] = options["default"]
            elif isinstance(definition[0], (list, tuple)):
                inputs[key] = definition[0][0]
            else:
                inputs[key] = ["source_" + definition[0].split(",")[0], 0]
        return inputs

    def prompt(self, name, switch=False):
        prompt = {
            "source_" + return_type: {"class_type": "ValidationSource_" + return_type,
                                      "inputs": {}}
            for return_type in ("MODEL", "CONDITIONING", "VAE", "AUDIO", "LATENT",
                                "AUDIO,LATENT", "*", "STRING", "IMAGE")
        }
        # Both branches have AUDIO contracts, matching Load Reference Audio
        # (possibly returning None when disabled) and OmniVoice Generate Audio.
        prompt["switch"] = {"class_type": "ComfySwitchNode", "inputs": {
            "switch": switch, "on_false": ["source_AUDIO", 0],
            "on_true": ["source_AUDIO", 0],
        }}
        prompt["target"] = {"class_type": name, "inputs": self.default_inputs(name)}
        return prompt

    def validate(self, prompt):
        return asyncio.run(execution.validate_inputs(
            "acestep-switch-regression", copy.deepcopy(prompt), "target", {},
        ))

    def assert_valid(self, prompt):
        result = self.validate(prompt)
        self.assertTrue(result[0], result[1])

    def test_native_switch_audio_is_valid_for_every_audio_input_and_branch(self):
        for name in self.node_names:
            for switch in (False, True):
                audio_inputs = ("reference_audio", "latent_or_audio") if "Generate" in name else (
                    "source", "reference_audio")
                for input_name in audio_inputs:
                    with self.subTest(node=name, switch=switch, input=input_name):
                        prompt = self.prompt(name, switch)
                        prompt["target"]["inputs"][input_name] = ["switch", 0]
                        self.assert_valid(prompt)

    def test_direct_audio_latent_union_and_wildcard_remain_compatible(self):
        for name in self.node_names:
            for return_type in ("AUDIO", "LATENT", "AUDIO,LATENT", "*"):
                with self.subTest(node=name, source=return_type):
                    prompt = self.prompt(name)
                    prompt["target"]["inputs"]["reference_audio"] = ["source_" + return_type, 0]
                    self.assert_valid(prompt)

    def test_three_generate_stages_accept_one_shared_reference_switch(self):
        prompt = self.prompt("AceStepSFTGenerate")
        for index in range(3):
            inputs = self.default_inputs("AceStepSFTGenerate")
            inputs["reference_audio"] = ["switch", 0]
            inputs["vae"] = ["source_VAE", 0]
            if index:
                previous = "stage_" + str(index - 1)
                inputs.update(positive=[previous, 2], negative=[previous, 3],
                              latent_or_audio=[previous, 4], denoise=0.9)
            node_id = "target" if index == 2 else "stage_" + str(index)
            prompt[node_id] = {"class_type": "AceStepSFTGenerate", "inputs": inputs}
        self.assert_valid(prompt)

    def test_invalid_concrete_audio_links_still_fail(self):
        for name in self.node_names:
            for return_type in ("MODEL", "STRING", "IMAGE"):
                with self.subTest(node=name, source=return_type):
                    prompt = self.prompt(name)
                    prompt["target"]["inputs"]["reference_audio"] = ["source_" + return_type, 0]
                    result = self.validate(prompt)
                    self.assertFalse(result[0])
                    self.assertTrue(any("AUDIO or LATENT" in error["details"]
                                        for error in result[1]), result[1])

    def test_switch_preserves_disabled_none_and_selected_audio(self):
        audio = {"waveform": torch.zeros(1, 1, 1), "sample_rate": 48000}
        # Disabled Load Reference Audio returns None. A false selection must
        # preserve that optional value; selecting OmniVoice returns its audio.
        for switch in (False, True):
            result = SwitchNode.execute(switch, on_false=None, on_true=audio)
            self.assertIs(result.result[0], audio if switch else None)

    def test_invalid_resolved_payload_is_rejected_before_sampling(self):
        for name in self.node_names:
            with self.subTest(node=name):
                inputs = self.default_inputs(name)
                if "Generate" in name:
                    inputs.update(model=SimpleNamespace(model_options={}), positive=[], negative=[])
                else:
                    inputs["source"] = {"waveform": torch.zeros(1, 1, 1), "sample_rate": 48000}
                inputs["reference_audio"] = "this is not audio"
                node = self.namespace[name]()
                with self.assertRaisesRegex(ValueError, "reference_audio must be AUDIO or LATENT"):
                    getattr(node, node.FUNCTION)(**inputs)


if __name__ == "__main__":
    unittest.main()
