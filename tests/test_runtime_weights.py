"""Native-inference-dtype CPU source caching without CUDA or checkpoint I/O."""
import ast
from types import SimpleNamespace
import unittest

import torch

import test_model_variants


GIB = 1024 ** 3


class RuntimeSourceTests(unittest.TestCase):
    def setUp(self):
        api = test_model_variants.load_module()
        self.ns = api._SharedVariantPatcher.__init__.__globals__
        self.available = 32 * GIB
        self.ns["comfy"].model_management = SimpleNamespace(
            get_free_memory=lambda device: self.available,
            get_total_memory=lambda device: 64 * GIB,
            throw_exception_if_processing_interrupted=lambda: None,
        )
        self.ns["comfy"].utils.ProgressBar = lambda total: SimpleNamespace(update_absolute=lambda value: None)
        self.ns["comfy"].lora = SimpleNamespace(
            calculate_shape=lambda patches, weight, key: (weight.shape[0] + 1, *weight.shape[1:]) if patches == "resize" else weight.shape,
        )
        tree = ast.parse(test_model_variants.SOURCE.read_text(encoding="utf-8"))
        selected = [node for node in tree.body if getattr(node, "name", None) in {"_runtime_cache_plan", "_cache_runtime_sources"}]
        exec(compile(ast.Module(body=selected, type_ignores=[]), str(test_model_variants.SOURCE), "exec"), self.ns)
        self.model = torch.nn.Module()
        self.model.large = torch.nn.Linear(128, 128)
        self.model.small = torch.nn.Linear(4, 4)
        self.model.direct = torch.nn.Linear(128, 128)
        self.model.unspecified = torch.nn.Linear(128, 128)
        for layer in (self.model.large, self.model.small):
            layer.comfy_cast_weights = True
            layer.weight_comfy_model_dtype = torch.bfloat16
            layer.bias_comfy_model_dtype = torch.bfloat16
        self.model.unspecified.comfy_cast_weights = True
        self.model.register_buffer("indices", torch.arange(4, dtype=torch.int64))
        baseline = {key: tensor.detach().clone() for key, tensor in self.model.state_dict().items()}
        second = baseline.copy()
        second["large.weight"] = baseline["large.weight"] + 0.125
        third = baseline.copy()
        third["large.weight"] = second["large.weight"]
        self.sources = (baseline, second, third)
        self.api = api

    def cache(self):
        return self.ns["_cache_runtime_sources"](self.model, self.sources)

    def family(self):
        staged, size = self.cache()
        return self.api._VariantFamily(
            staged, frozenset(dict(self.model.named_parameters())),
            frozenset(dict(self.model.named_buffers())), ("sft", "base", "turbo"),
            ("SFT", "Base", "Turbo"), checkpoint_sources=self.sources,
            runtime_cache_bytes=size,
        )

    def patcher(self, family):
        patcher = self.api._SharedVariantPatcher(self.model)
        patcher._variant_family = family
        patcher._variant_index = 0
        patcher.weight_inplace_update = False
        return patcher

    def test_cache_uses_native_dtype_and_deduplicates_all_variants(self):
        staged, size = self.cache()
        self.assertGreater(size, 0)
        self.assertIs(staged[0]["large.bias"], staged[1]["large.bias"])
        self.assertIs(staged[1]["large.weight"], staged[2]["large.weight"])
        self.assertIsNot(staged[0]["large.weight"], staged[1]["large.weight"])
        for source, result in zip(self.sources, staged):
            self.assertEqual(result["large.weight"].dtype, torch.bfloat16)
            torch.testing.assert_close(result["large.weight"], source["large.weight"].to(torch.bfloat16), rtol=0, atol=0)
            self.assertEqual(source["large.weight"].dtype, torch.float32)

    def test_small_direct_unspecified_and_integer_sources_remain_original(self):
        staged, _ = self.cache()
        for source, result in zip(self.sources, staged):
            for key in ("small.weight", "small.bias", "direct.weight", "direct.bias", "unspecified.weight", "unspecified.bias", "indices"):
                with self.subTest(key=key):
                    self.assertIs(result[key], source[key])
        self.assertEqual(staged[0]["indices"].dtype, torch.int64)

    def test_materialized_source_has_no_file_slice_metadata(self):
        tensor = self.sources[0]["large.weight"]
        marker = object()
        tensor.untyped_storage()._comfy_tensor_file_slice = marker
        staged, _ = self.cache()
        self.assertFalse(hasattr(staged[0]["large.weight"].untyped_storage(), "_comfy_tensor_file_slice"))
        self.assertIs(tensor.untyped_storage()._comfy_tensor_file_slice, marker)

    def test_staging_cannot_cross_native_small_module_threshold(self):
        self.model.boundary = torch.nn.Linear(64, 64)
        self.model.boundary.comfy_cast_weights = True
        self.model.boundary.weight_comfy_model_dtype = torch.bfloat16
        self.model.boundary.bias_comfy_model_dtype = torch.bfloat16
        for source in self.sources:
            source["boundary.weight"] = self.model.boundary.weight.detach().clone()
            source["boundary.bias"] = self.model.boundary.bias.detach().clone()
        staged, size = self.cache()
        self.assertGreater(size, 0)
        for source, result in zip(self.sources, staged):
            self.assertIs(result["boundary.weight"], source["boundary.weight"])
            self.assertIs(result["boundary.bias"], source["boundary.bias"])

    def test_insufficient_ram_disables_cache_without_partial_sources(self):
        self.available = GIB
        staged, size = self.cache()
        self.assertIs(staged, self.sources)
        self.assertEqual(size, 0)

    def test_ram_pressure_during_build_abandons_cache(self):
        calls = []
        def available(device):
            calls.append(device)
            return 32 * GIB if len(calls) == 1 else GIB
        self.ns["comfy"].model_management.get_free_memory = available
        staged, size = self.cache()
        self.assertIs(staged, self.sources)
        self.assertEqual(size, 0)

    def test_partial_budget_distributes_large_weights_across_variants(self):
        self.model.large2 = torch.nn.Linear(128, 128)
        self.model.large2.comfy_cast_weights = True
        self.model.large2.weight_comfy_model_dtype = torch.bfloat16
        self.model.large2.bias_comfy_model_dtype = torch.bfloat16
        for index, source in enumerate(self.sources):
            source["large.weight"] = source["large.weight"].clone() + index
            source["large2.weight"] = self.model.large2.weight.detach().clone() + index
            source["large2.bias"] = self.model.large2.bias.detach().clone()
        raw_host = max(sum(tensor.nbytes for tensor in source.values()) for source in self.sources)
        budget = 3 * 128 * 128 * torch.bfloat16.itemsize
        self.available = raw_host + 9 * GIB + int(64 * GIB * 0.08) + budget
        staged, size = self.cache()
        self.assertEqual(size, budget)
        for source, result in zip(self.sources, staged):
            self.assertEqual(result["large.weight"].dtype, torch.bfloat16)
            for key in ("large.bias", "large2.weight", "large2.bias"):
                self.assertIs(result[key], source[key])
        self.assertEqual(sum(result["large.weight"].nbytes for result in staged), size)

    def test_future_text_encoder_capacity_is_reserved(self):
        raw_host = max(sum(tensor.nbytes for tensor in source.values()) for source in self.sources)
        self.available = raw_host + int(64 * GIB * 0.08) + 8 * GIB
        staged, size = self.cache()
        self.assertIs(staged, self.sources)
        self.assertEqual(size, 0)

    def test_family_cache_is_capped_at_four_gib_without_large_allocations(self):
        class LogicalTensor:
            def __init__(self, numel, dtype=torch.float32):
                self.elements = numel
                self.dtype = dtype
                self.nbytes = numel * dtype.itemsize

            def numel(self):
                return self.elements

            def to(self, *, device, dtype, copy):
                self.assert_copy = copy
                return LogicalTensor(self.elements, dtype)

        for source in self.sources:
            source["large.weight"] = LogicalTensor(9 * GIB // 8)
        self.available = 64 * GIB
        staged, size = self.cache()
        self.assertLessEqual(size, 4 * GIB)
        self.assertEqual(staged[0]["large.weight"].dtype, torch.bfloat16)
        self.assertIs(staged[1]["large.weight"], self.sources[1]["large.weight"])
        self.assertIs(staged[2]["large.weight"], self.sources[2]["large.weight"])

    def test_resizing_adapter_uses_raw_source_for_weight_and_bias(self):
        family = self.family()
        patcher = self.patcher(family)
        patcher.patches["large.weight"] = "resize"
        selected = patcher._variant_sources(family, 0, False)
        self.assertIs(selected["large.weight"], self.sources[0]["large.weight"])
        self.assertIs(selected["large.bias"], self.sources[0]["large.bias"])
        patcher.patches["large.weight"] = "ordinary-lora"
        self.assertIs(patcher._variant_sources(family, 0, False), family.sources[0])

    def test_custom_transform_releases_runtime_copy_before_raw_fallback(self):
        family = self.family()
        patcher = self.patcher(family)
        patcher.weight_wrapper_patches["large.weight"] = [object()]
        selected = patcher._variant_sources(family, 0, False)
        self.assertIs(selected, self.sources[0])
        self.assertIs(family.sources, self.sources)
        self.assertIsNone(family.checkpoint_sources)
        self.assertEqual(family.runtime_cache_bytes, 0)

    def test_deepclone_shares_cpu_ownership_with_fresh_activation_state(self):
        family = self.family()
        family.active_variant = 2
        family.active_sources = family.sources[2]
        copied = family.for_new_model()
        self.assertIs(copied.sources, family.sources)
        self.assertIs(copied.checkpoint_sources, family.checkpoint_sources)
        self.assertEqual(copied.runtime_cache_bytes, family.runtime_cache_bytes)
        self.assertIsNone(copied.active_variant)
        self.assertIsNone(copied.active_sources)


if __name__ == "__main__":
    unittest.main()
