"""CPU-only ownership, gradient and time-budget checks for voice adaptation."""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import patch
import uuid

import torch


class FakeComfyAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.function = None


class Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.comfy_attention = FakeComfyAttention()
        self.linear = torch.nn.Linear(2, 2, bias=False)
        self.linear.requires_grad_(False)
        self.calls = 0

    def forward(self, value):
        self.calls += 1
        return self.linear(value)


def make_model():
    model = torch.nn.Module()
    model.diffusion_model = torch.nn.Module()
    model.diffusion_model.decoder = torch.nn.Module()
    model.diffusion_model.decoder.layers = torch.nn.ModuleList([Layer(), Layer()])
    return model


def get_attr(model, path):
    for part in path.split("."):
        model = getattr(model, part)
    return model


def set_attr(model, path, value):
    owner_path, _, name = path.rpartition(".")
    setattr(get_attr(model, owner_path), name, value)


class Patcher:
    def __init__(self):
        self.model = make_model()
        self.object_patches = {}
        self.object_patches_backup = {}
        self.clone_base_uuid = uuid.uuid4()
        self.clone_args = None
        self.clone_inference_mode = None
        self.last_clone = None
        self.unpatches = 0

    def clone(self, **kwargs):
        self.clone_args = kwargs
        self.clone_inference_mode = torch.is_inference_mode_enabled()
        result = Patcher()
        result.model.load_state_dict(self.model.state_dict())
        result.clone_base_uuid = self.clone_base_uuid
        result.object_patches = self.object_patches.copy()
        self.last_clone = result
        return result

    def get_model_object(self, path):
        if path in self.object_patches:
            return self.object_patches[path]
        return get_attr(self.model, path)

    def add_object_patch(self, path, value):
        self.object_patches[path] = value

    def patch_model(self, load_weights=False):
        for path, value in self.object_patches.items():
            self.object_patches_backup.setdefault(path, get_attr(self.model, path))
            set_attr(self.model, path, value)

    def unpatch_model(self, unpatch_weights=False):
        self.unpatches += 1
        for path, value in self.object_patches_backup.items():
            set_attr(self.model, path, value)
        self.object_patches_backup.clear()


class ManualTime:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class AdaptationRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.unloaded = []
        comfy = ModuleType("comfy")
        management = ModuleType("comfy.model_management")
        management.unload_model_and_clones = lambda model, **kwargs: cls.unloaded.append((model, kwargs))
        comfy.model_management = management
        attention = ModuleType("comfy.ldm.modules.attention")
        attention.ComfyAttention = FakeComfyAttention
        attention.attention_pytorch = lambda *args, **kwargs: None
        module_path = Path(__file__).resolve().parents[1] / "voice_adaptation_runtime.py"
        spec = importlib.util.spec_from_file_location("voice_adaptation_runtime_cpu", module_path)
        cls.runtime = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {
            "comfy": comfy,
            "comfy.model_management": management,
            "comfy.ldm.modules.attention": attention,
        }):
            spec.loader.exec_module(cls.runtime)

    def setUp(self):
        self.unloaded.clear()
        self.timer = ManualTime()
        self.time_patch = patch.object(self.runtime.time, "perf_counter", self.timer)
        self.time_patch.start()
        self.addCleanup(self.time_patch.stop)

    def test_inference_escape_fresh_weights_and_original_preservation(self):
        with torch.inference_mode():
            original = Patcher()
            before = original.model.diffusion_model.decoder.layers[0].linear.weight.clone()
            self.assertTrue(before.is_inference())
            with self.runtime.adaptation_runtime(original) as (static, clock):
                self.assertFalse(torch.is_inference_mode_enabled())
                self.assertTrue(torch.is_grad_enabled())
                static.patch_model()
                value = torch.ones(1, 2, requires_grad=True)
                result = static.model.diffusion_model.decoder.layers[0](value)
                gradient, = torch.autograd.grad(result.sum(), value)
                self.assertTrue(torch.isfinite(gradient).all())
                self.assertFalse(static.model.diffusion_model.decoder.layers[0].linear.weight.is_inference())
                self.assertNotEqual(static.clone_base_uuid, original.clone_base_uuid)
                self.assertFalse(any(parameter.requires_grad for parameter in static.model.parameters()))
            self.assertTrue(torch.is_inference_mode_enabled())
        self.assertEqual(original.clone_args, {"disable_dynamic": True, "force_deepcopy": True})
        self.assertFalse(original.clone_inference_mode)
        torch.testing.assert_close(original.model.diffusion_model.decoder.layers[0].linear.weight, before)
        self.assertEqual(original.object_patches, {})
        self.assertEqual(self.unloaded, [(static, {"unload_additional_models": False})])

    def test_runtime_patches_restore_inherited_entries(self):
        original = Patcher()
        path = "diffusion_model.decoder.layers.0.comfy_attention.function"
        previous = lambda *args: "user configuration"
        original.object_patches[path] = previous
        with self.runtime.adaptation_runtime(original) as (static, clock):
            self.assertIs(static.object_patches[path], self.runtime.attention_pytorch)
            static.patch_model()
        self.assertEqual(static.object_patches, {path: previous})
        self.assertEqual(original.object_patches, {path: previous})
        self.assertIsNone(static.model.diffusion_model.decoder.layers[0].comfy_attention.function)
        self.assertGreater(static.unpatches, 0)

    def test_preparation_does_not_start_the_budget(self):
        with self.runtime.adaptation_runtime(Patcher(), seconds=1) as (static, clock):
            static.patch_model()
            self.timer.now = 1000
            static.model.diffusion_model.decoder.layers[0](torch.ones(1, 2))
            self.assertIsNone(clock.started)
            self.assertEqual(clock(), 1000)
            self.timer.now = 1000.9
            clock.check()
            self.timer.now = 1001
            with self.assertRaises(TimeoutError):
                clock.check()

    def test_forward_stops_before_an_expired_layer(self):
        with self.runtime.adaptation_runtime(Patcher(), seconds=1) as (static, clock):
            static.patch_model()
            layer = static.model.diffusion_model.decoder.layers[0]
            clock()
            self.timer.now = 1
            with self.assertRaises(TimeoutError):
                layer(torch.ones(1, 2))
            self.assertEqual(layer.calls, 0)

    def test_backward_stops_when_budget_expires(self):
        with self.runtime.adaptation_runtime(Patcher(), seconds=1) as (static, clock):
            static.patch_model()
            clock()
            value = torch.ones(1, 2, requires_grad=True)
            output = static.model.diffusion_model.decoder.layers[0](value)
            self.timer.now = 1
            with self.assertRaises(TimeoutError):
                torch.autograd.grad(output.sum(), value)

    def test_cleanup_runs_on_caller_failure(self):
        original = Patcher()
        with self.assertRaisesRegex(RuntimeError, "fit failed"):
            with self.runtime.adaptation_runtime(original) as (static, clock):
                static.patch_model()
                raise RuntimeError("fit failed")
        self.assertEqual(static.object_patches, {})
        self.assertEqual(len(self.unloaded), 1)
        self.assertIs(self.unloaded[0][0], static)
        self.assertEqual(original.unpatches, 0)

    def test_budget_is_capped_at_sixty_seconds(self):
        with self.runtime.adaptation_runtime(Patcher(), seconds=120) as (static, clock):
            clock()
            self.timer.now = 60
            with self.assertRaises(TimeoutError):
                clock.check()

    def test_invalid_budgets_do_not_clone_models(self):
        for seconds in (-1, float("nan"), float("inf")):
            with self.subTest(seconds=seconds):
                original = Patcher()
                with self.assertRaises(ValueError):
                    with self.runtime.adaptation_runtime(original, seconds):
                        pass
                self.assertIsNone(original.last_clone)


if __name__ == "__main__":
    unittest.main()
