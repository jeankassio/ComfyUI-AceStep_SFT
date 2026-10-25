"""CPU state-machine regressions for exact checkpoint switching.

The native dynamic GPU path is independently exercised by the tiny benchmark.
Here AST loading isolates source ownership/invalidation from CUDA and ComfyUI
startup; no checkpoints or model downloads are required.
"""
import ast
import copy
from dataclasses import dataclass, field
import logging
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
import uuid

import torch


SOURCE = Path(__file__).resolve().parents[1] / "model_variants.py"


def set_attr(model, key, tensor, parameter=True):
    path, _, name = key.rpartition(".")
    owner = model.get_submodule(path) if path else model
    setattr(owner, name, torch.nn.Parameter(tensor, requires_grad=False) if parameter else tensor)


class NativeDynamic:
    """Small contract double; actual native implementation is tested on GPU."""
    def __init__(self, model, *args, **kwargs):
        self.model = model
        self.patches = {}
        self.weight_wrapper_patches = {}
        self.hook_patches = {}
        self.hook_backup = {}
        self.current_hooks = None
        self.forced_hooks = None
        self.injections = {}
        self.patches_uuid = uuid.uuid4()
        self.flushes = 0
        self.restores = 0
        self.fail_next_load = False
        self.callbacks = []

    def get_all_callbacks(self, which):
        return self.callbacks

    def restore_loaded_backups(self):
        self.restores += 1

    def partially_unload_ram(self, size):
        self.flushes += 1

    def load(self, **kwargs):
        self.last_native_load = kwargs
        if self.fail_next_load:
            self.fail_next_load = False
            raise RuntimeError("injected native loading failure")
        self.model.current_weight_patches_uuid = self.patches_uuid

    def clone(self, **kwargs):
        cloned = copy.copy(self)
        cloned.patches = {key: list(value) for key, value in self.patches.items()}
        cloned.weight_wrapper_patches = self.weight_wrapper_patches.copy()
        return cloned


def load_module():
    fake_comfy = SimpleNamespace(
        model_patcher=SimpleNamespace(
            ModelPatcherDynamic=NativeDynamic,
            CallbacksMP=SimpleNamespace(ON_LOAD="on_load"),
        ),
        utils=SimpleNamespace(
            set_attr_param=lambda model, key, tensor: set_attr(model, key, tensor),
            set_attr_buffer=lambda model, key, tensor: set_attr(model, key, tensor, parameter=False),
        ),
    )
    namespace = {
        "__name__": __name__, "logging": logging, "os": os, "uuid": uuid,
        "dataclass": dataclass, "field": field, "torch": torch, "comfy": fake_comfy,
        "_LOG": logging.getLogger("test_model_variants"),
        "_DYNAMIC_PATCHER": NativeDynamic,
    }
    names = {"_fingerprint", "_patch_fingerprints", "_wrapper_fingerprints", "_same_tensor_bytes", "_VariantFamily", "_SharedVariantPatcher"}
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    nodes = [node for node in tree.body if getattr(node, "name", None) in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return SimpleNamespace(**namespace)


class VariantOwnershipTests(unittest.TestCase):
    def setUp(self):
        self.api = load_module()
        self.model = torch.nn.Module()
        self.model.common = torch.nn.Linear(2, 2)
        self.model.changed = torch.nn.Linear(2, 2)
        self.model.register_buffer("counter", torch.tensor(1, dtype=torch.int64))
        self.model.current_weight_patches_uuid = None
        original = {key: tensor.detach().clone() for key, tensor in self.model.state_dict().items()}
        self.sources = []
        for index in range(3):
            source = original.copy()
            source["changed.weight"] = torch.full((2, 2), index + 1, dtype=torch.float32)
            source["changed.bias"] = torch.full((2,), index + 3, dtype=torch.float32)
            source["counter"] = torch.tensor(index + 1, dtype=torch.int64)
            self.sources.append(source)
        self.family = self.api._VariantFamily(
            tuple(self.sources), frozenset(dict(self.model.named_parameters())),
            frozenset(dict(self.model.named_buffers())), ("a", "b", "c"), ("SFT", "Base", "Turbo"),
        )
        self.patchers = []
        for index in range(3):
            patcher = self.api._SharedVariantPatcher(self.model)
            patcher._variant_family = self.family
            patcher._variant_index = index
            self.patchers.append(patcher)

    def activate(self, index, patcher=None):
        patcher = self.patchers[index] if patcher is None else patcher
        patcher.load(device_to=torch.device("cpu"), dirty=True)
        for key, tensor in self.model.state_dict().items():
            self.assertTrue(torch.equal(tensor, self.sources[index][key]), key)
        self.assertFalse(patcher.last_native_load["dirty"])
        return patcher

    def signatures(self):
        self.model.common._v_signature = "common-resident"
        self.model.changed._v_signature = "changed-resident"

    def test_exact_sft_base_turbo_return_and_shared_storage(self):
        self.activate(0)
        for index in (1, 2, 0, 2, 1, 0):
            self.signatures()
            pointer = self.model.common.weight.data_ptr()
            self.activate(index)
            self.assertEqual(self.model.common.weight.data_ptr(), pointer)
            self.assertEqual(self.model.common._v_signature, "common-resident")
            self.assertIsNone(self.model.changed._v_signature)
            self.assertEqual(self.family.last_transition["source_tensors_changed"], 3)
            self.assertEqual(self.model.counter.dtype, torch.int64)

    def test_unchanged_activation_does_not_flush_pins_or_invalidate(self):
        patcher = self.activate(0)
        self.signatures()
        flushes = patcher.flushes
        self.activate(0)
        self.assertEqual(patcher.flushes, flushes)
        self.assertEqual(self.model.changed._v_signature, "changed-resident")
        self.assertFalse(self.family.last_transition["host_pins_flushed"])

    def test_independent_adapter_changes_invalidate_common_and_return(self):
        self.activate(0)
        adapter = self.patchers[1].clone()
        adapter.patches = {"common.weight": [(0.75, (torch.ones(2, 2),), 1.0, None, None)]}
        adapter.patches_uuid = uuid.uuid4()
        self.signatures()
        self.activate(1, adapter)
        self.assertIsNone(self.model.common._v_signature)
        self.signatures()
        self.activate(1, adapter)
        self.assertEqual(self.model.common._v_signature, "common-resident")
        self.activate(2)
        self.assertIsNone(self.model.common._v_signature)
        self.assertFalse(self.patchers[1].patches)
        self.signatures()
        self.activate(1, adapter)
        self.assertIsNone(self.model.common._v_signature)

    def test_mutating_adapter_tensor_is_detected(self):
        patcher = self.patchers[0]
        delta = torch.ones(2, 2)
        patcher.patches["common.weight"] = [(1.0, (delta,), 1.0, None, None)]
        self.activate(0)
        self.signatures()
        delta.add_(1)
        self.activate(0)
        self.assertIsNone(self.model.common._v_signature)

    def test_failure_invalidates_owner_and_next_load_recovers_exactly(self):
        self.activate(0)
        broken = self.patchers[1]
        broken.fail_next_load = True
        with self.assertRaisesRegex(RuntimeError, "injected"):
            broken.load(device_to=torch.device("cpu"))
        self.assertIsNone(self.family.active_variant)
        self.signatures()
        self.activate(2)
        self.assertTrue(self.family.last_transition["full_invalidation"])
        self.assertIsNone(self.model.common._v_signature)
        self.activate(0)

    def test_unknown_native_owner_and_hook_force_full_invalidation(self):
        self.activate(0)
        self.model.current_weight_patches_uuid = uuid.uuid4()
        set_attr(self.model, "common.weight", torch.full((2, 2), 999.0))
        self.signatures()
        self.activate(0)
        self.assertTrue(self.family.last_transition["full_invalidation"])
        self.assertIsNone(self.model.common._v_signature)
        self.patchers[0].hook_patches = {"external": {}}
        self.signatures()
        self.activate(0)
        self.assertTrue(self.family.last_transition["full_invalidation"])

    def test_new_model_owner_shares_sources_but_not_activation_state(self):
        self.activate(2)
        copied = self.family.for_new_model()
        self.assertIs(copied.sources, self.family.sources)
        self.assertIsNone(copied.active_variant)
        self.assertEqual(copied.active_patches, {})
        self.assertIsNot(copied.active_patches, self.family.active_patches)

    def test_byte_equality_preserves_signed_zero_and_nan_payloads(self):
        same_bytes = self.api._same_tensor_bytes
        positive_zero = torch.tensor([0], dtype=torch.int32).view(torch.float32)
        negative_zero = torch.tensor([-2147483648], dtype=torch.int32).view(torch.float32)
        self.assertFalse(same_bytes(positive_zero, negative_zero))
        first_nan = torch.tensor([0x7fc00001], dtype=torch.int32).view(torch.float32)
        second_nan = torch.tensor([0x7fc00002], dtype=torch.int32).view(torch.float32)
        self.assertTrue(same_bytes(first_nan, first_nan.clone()))
        self.assertFalse(same_bytes(first_nan, second_nan))
        self.assertFalse(same_bytes(torch.ones(1), torch.ones(1, dtype=torch.float64)))
        self.assertFalse(same_bytes(torch.ones(2, 1), torch.ones(1, 2)))

    def test_matching_probe_still_checks_the_rest_of_the_checkpoint_tensor(self):
        same_bytes = self.api._same_tensor_bytes
        first = torch.arange(1024, dtype=torch.float32)
        second = first.clone()
        self.assertTrue(same_bytes(first, second))
        second[700] += 1
        self.assertFalse(same_bytes(first, second))
        second[0] += 1
        self.assertFalse(same_bytes(first, second))


if __name__ == "__main__":
    unittest.main()
