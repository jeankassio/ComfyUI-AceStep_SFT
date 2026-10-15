"""CPU arithmetic and conditioning regressions; these are not voice-quality tests."""
import ast
import copy
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import torch

import test_reference_attention
import test_reference_audio_generate
import test_timbre_cls
from test_sampling import load_definitions


NODE_PATH = Path(__file__).resolve().parents[1] / "nodes.py"


class Model:
    def __init__(self, previous=None):
        self.model_options = {} if previous is None else {"model_function_wrapper": previous}

    def set_model_unet_function_wrapper(self, wrapper):
        self.model_options["model_function_wrapper"] = wrapper


class ReferenceGuidanceTests(unittest.TestCase):
    def setUp(self):
        tree = ast.parse(NODE_PATH.read_text(encoding="utf-8"))
        definition = next(item for item in tree.body if getattr(item, "name", None) == "_configure_acestep_reference_model")
        self.namespace = {"torch": torch,
                          "_is_latent_payload": lambda value: isinstance(value, dict) and "samples" in value,
                          "_build_timbre_reference_latent": lambda vae, reference, sr, tiled, seed: reference["samples"].clone(),
                          "_ensure_acestep_native_forward": lambda model: None,
                          "_get_acestep_silence_latent": lambda frames, device, dtype: torch.zeros(1, 64, frames, device=device, dtype=dtype),
                          "_repeat_native_tensor": lambda value, batch: value.repeat(batch // value.shape[0], 1, 1)}
        exec(compile(ast.Module(body=[definition], type_ignores=[]), str(NODE_PATH), "exec"), self.namespace)
        self.calls = []
        self.reference = {"samples": torch.full((1, 64, 750), .5)}
        self.x = torch.arange(3, dtype=torch.float32).reshape(3, 1, 1).expand(3, 64, 40).clone()
        self.conditions = {"context": torch.tensor([1., 4., 7.]).reshape(3, 1, 1),
                           "lyric_embed": torch.ones(3, 4, 2), "audio_codes": torch.tensor([[1, 2], [3, 4], [5, 6]]),
                           "precomputed_lm_hints_25Hz": torch.full((3, 64, 40), .2)}
        self.args = {"input": self.x, "timestep": torch.full((3,), .3), "c": self.conditions,
                     "cond_or_uncond": [0, 0, 0]}

    def configure(self, scale=None, reference=None, previous=None, repeat_state=None):
        model = Model(previous)
        if repeat_state is not None:
            model.model_options["_acestep_sft_reference_timbre_repeat_state"] = repeat_state
        arguments = (model, reference or self.reference, None, 48000, False, 27)
        if scale is None:
            self.namespace["_configure_acestep_reference_model"](*arguments)
        else:
            self.namespace["_configure_acestep_reference_model"](*arguments, reference_guidance_scale=scale)
        return model.model_options["model_function_wrapper"]

    def predict(self, x, sigma, **conditions):
        self.calls.append(conditions)
        reference = conditions["refer_audio"].mean(dim=(1, 2))[:, None, None]
        # Each row deliberately has a distinct result, making reordering visible.
        return x + conditions["context"] + sigma[:, None, None] + reference * 4

    def expected(self, scale):
        return self.x + self.conditions["context"] + .3 + scale * 2

    def test_default_and_explicit_one_preserve_the_single_reference_prediction(self):
        for scale in (None, 1.0):
            with self.subTest(scale=scale):
                self.calls.clear()
                output = self.configure(scale)(self.predict, self.args)
                torch.testing.assert_close(output, self.expected(1))
                self.assertEqual(len(self.calls), 1)

    def test_zero_uses_reference_free_prediction_and_two_boosts_only_reference_difference(self):
        for scale in (0.0, 2.0):
            with self.subTest(scale=scale):
                self.calls.clear()
                output = self.configure(scale)(self.predict, self.args)
                torch.testing.assert_close(output, self.expected(scale))
                self.assertEqual(len(self.calls), 2)
                self.assertEqual(self.calls[0]["refer_audio"].shape, (3, 64, 750))
                self.assertEqual(self.calls[1]["refer_audio"].shape, (3, 64, 750))
                self.assertEqual(torch.count_nonzero(self.calls[1]["refer_audio"]).item(), 0)

    def test_both_predictions_keep_codes_lyrics_lm_hints_sigma_and_source_context(self):
        before = copy.deepcopy(self.conditions)
        original_keys = set(self.conditions)
        self.configure(2.0)(self.predict, self.args)
        self.assertEqual(len(self.calls), 2)
        for key in ("context", "lyric_embed", "audio_codes", "precomputed_lm_hints_25Hz"):
            self.assertIs(self.calls[0][key], self.conditions[key])
            self.assertIs(self.calls[1][key], self.conditions[key])
            torch.testing.assert_close(self.conditions[key], before[key], rtol=0, atol=0)
        self.assertIs(self.calls[0]["src_latents"], self.calls[1]["src_latents"])
        self.assertIs(self.calls[0]["chunk_masks"], self.calls[1]["chunk_masks"])
        self.assertIs(self.calls[0]["is_covers"], True)
        self.assertIs(self.calls[1]["is_covers"], True)
        self.assertEqual(set(self.conditions), original_keys)
        self.assertNotIn("refer_audio", self.conditions)
        torch.testing.assert_close(self.args["input"], self.x, rtol=0, atol=0)

    def test_reference_free_lm_path_does_not_turn_reference_into_composition(self):
        args = self.args.copy()
        args["c"] = {"context": self.conditions["context"], "lyric_embed": self.conditions["lyric_embed"]}
        self.configure(2.0)(self.predict, args)
        for call in self.calls:
            self.assertNotIn("audio_codes", call)
            self.assertEqual(call["src_latents"].shape, self.x.shape)
            self.assertEqual(torch.count_nonzero(call["src_latents"]).item(), 0)
            self.assertEqual(call["precomputed_lm_hints_25Hz"].shape, self.x.shape)
            self.assertEqual(torch.count_nonzero(call["precomputed_lm_hints_25Hz"]).item(), 0)

    def test_null_text_branch_is_not_amplified_and_requires_one_prediction(self):
        for null in (True, torch.tensor(True)):
            with self.subTest(null=null):
                self.calls.clear()
                args = self.args.copy()
                args["c"] = self.conditions | {"replace_with_null_embeds": null}
                output = self.configure(3.0)(self.predict, args)
                torch.testing.assert_close(output, self.expected(1))
                self.assertEqual(len(self.calls), 1)

    def test_mixed_batch_preserves_row_order_and_null_row_without_stripping_other_rows(self):
        args = self.args.copy()
        args["c"] = self.conditions | {"replace_with_null_embeds": torch.tensor([False, True, False])}
        output = self.configure(2.0)(self.predict, args)
        expected = self.expected(2)
        expected[1] = self.expected(1)[1]
        torch.testing.assert_close(output, expected)
        self.assertEqual(len(self.calls), 2)

    def test_previous_wrapper_is_delegated_for_each_branch_without_mutating_caller(self):
        delegated = []
        def previous(apply_model, args):
            delegated.append(args)
            return apply_model(args["input"], args["timestep"], **args["c"]) + 7
        output = self.configure(2.0, previous=previous)(self.predict, self.args)
        torch.testing.assert_close(output, self.expected(2) + 7)
        self.assertEqual(len(delegated), 2)
        for args in delegated:
            self.assertIs(args["input"], self.x)
            self.assertIs(args["timestep"], self.args["timestep"])
            self.assertIs(args["cond_or_uncond"], self.args["cond_or_uncond"])
        self.assertIs(self.args["c"], self.conditions)
        self.assertNotIn("refer_audio", self.conditions)

    def test_each_fresh_stage_uses_its_supplied_reference_and_keeps_reference_tensor_unchanged(self):
        before = self.reference["samples"].clone()
        first = self.configure(2.0)(self.predict, self.args)
        next_reference = {"samples": torch.full((1, 64, 750), .25)}
        second = self.configure(2.0, reference=next_reference)(self.predict, self.args)
        torch.testing.assert_close(first, self.expected(2))
        torch.testing.assert_close(second, self.expected(1))
        torch.testing.assert_close(self.reference["samples"], before, rtol=0, atol=0)
        torch.testing.assert_close(next_reference["samples"], torch.full((1, 64, 750), .25), rtol=0, atol=0)

    def test_token_repetition_is_active_only_for_reference_prediction_and_restores_state(self):
        state, enabled = {"enabled": False}, []
        def apply(x, sigma, **conditions):
            enabled.append(state["enabled"])
            return self.predict(x, sigma, **conditions)
        self.configure(1.25, repeat_state=state)(apply, self.args)
        self.assertEqual(enabled, [True, False])
        self.assertFalse(state["enabled"])
        enabled.clear()
        args = self.args.copy()
        args["c"] = self.conditions | {"replace_with_null_embeds": True}
        self.configure(1.25, repeat_state=state)(apply, args)
        self.assertEqual(enabled, [False])
        self.assertFalse(state["enabled"])

    def test_prediction_failure_does_not_leave_repetition_active_for_the_next_call(self):
        for fail_on in (1, 2):
            with self.subTest(fail_on=fail_on):
                state = {"enabled": False}
                calls = 0
                def failing(x, sigma, **conditions):
                    nonlocal calls
                    calls += 1
                    if calls == fail_on:
                        raise RuntimeError("prediction failed")
                    return x
                with self.assertRaisesRegex(RuntimeError, "prediction failed"):
                    self.configure(1.25, repeat_state=state)(failing, self.args)
                self.assertFalse(state["enabled"])


class TimbreRepeatTests(unittest.TestCase):
    def setUp(self):
        self.ns = test_reference_attention.node_ns.copy()
        load_definitions(NODE_PATH, {"_ensure_acestep_timbre_cls", "_clear_acestep_timbre_cls_patch",
                                    "_ensure_acestep_reference_timbre_repeats",
                                    "_clear_acestep_reference_timbre_repeats"}, self.ns)

    def test_default_one_installs_no_patch_and_is_bit_identical(self):
        patcher, encoder = test_timbre_cls.TimbreCLSTests().model(layers=0)
        reference, order = torch.randn(2, 6, 8), torch.tensor([0, 1])
        before, before_mask = encoder(reference, order)
        clone = patcher.clone()
        self.ns["_ensure_acestep_reference_timbre_repeats"](clone, 1)
        self.assertEqual(clone.object_patches, {})
        self.assertNotIn("_acestep_sft_reference_timbre_repeat_state", clone.model_options)
        after, after_mask = encoder(reference, order)
        torch.testing.assert_close(after, before, rtol=0, atol=0)
        torch.testing.assert_close(after_mask, before_mask, rtol=0, atol=0)

    def test_repeated_tokens_keep_amplitude_padding_mask_and_reference_batch_order(self):
        patcher, encoder = test_timbre_cls.TimbreCLSTests().model(layers=0)
        reference, order = torch.randn(3, 6, 8), torch.tensor([1, 0, 1])
        before, before_mask = encoder(reference, order)
        original, weights = encoder.forward, {key: value.clone() for key, value in encoder.state_dict().items()}
        clone = patcher.clone()
        self.ns["_ensure_acestep_reference_timbre_repeats"](clone, 3)
        state = clone.model_options["_acestep_sft_reference_timbre_repeat_state"]
        self.assertEqual(encoder.forward, original)
        clone.patch_model(load_weights=False)
        try:
            disabled, disabled_mask = encoder(reference, order)
            torch.testing.assert_close(disabled, before, rtol=0, atol=0)
            torch.testing.assert_close(disabled_mask, before_mask, rtol=0, atol=0)
            state["enabled"] = True
            repeated, mask = encoder(reference, order)
            self.assertEqual(repeated.shape[1], before.shape[1] * 3)
            for index in range(before.shape[1]):
                for repetition in range(3):
                    torch.testing.assert_close(repeated[:, index * 3 + repetition], before[:, index], rtol=0, atol=0)
            self.assertEqual(mask[0].tolist(), [1, 1, 1, 0, 0, 0])
            self.assertEqual(mask[1].tolist(), [1, 1, 1, 1, 1, 1])
            for key, value in encoder.state_dict().items():
                torch.testing.assert_close(value, weights[key], rtol=0, atol=0)
        finally:
            self.ns["_clear_acestep_reference_timbre_repeats"](clone)
        self.assertEqual(encoder.forward, original)

    def test_repetition_coexists_with_cls_and_local_windows_and_cleanup_restores_each(self):
        patcher, diffusion = test_reference_attention.ReferenceAttentionTests().model(sliding_window=4)
        encoder, reference, order = diffusion.encoder.timbre_encoder, torch.randn(2, 10, 4), torch.tensor([0, 1])
        original, weights = encoder.forward, {key: value.clone() for key, value in encoder.state_dict().items()}
        clone = patcher.clone()
        self.ns["_ensure_acestep_timbre_cls"](clone)
        self.ns["_ensure_acestep_reference_attention"](clone)
        self.ns["_ensure_acestep_reference_timbre_repeats"](clone, 2)
        clone.patch_model(load_weights=False)
        try:
            baseline, baseline_mask = encoder(reference, order)
            clone.model_options["_acestep_sft_reference_timbre_repeat_state"]["enabled"] = True
            repeated, mask = encoder(reference, order)
            torch.testing.assert_close(repeated[:, 0], baseline[:, 0], rtol=0, atol=0)
            torch.testing.assert_close(repeated[:, 1], baseline[:, 0], rtol=0, atol=0)
            torch.testing.assert_close(mask[:, 0], baseline_mask[:, 0], rtol=0, atol=0)
            self.assertEqual([layer.self_attn.sliding_window for layer in encoder.layers], [4, None, 4, None])
            self.ns["_clear_acestep_reference_timbre_repeats"](clone)
            cls_output, _ = encoder(reference, order)
            torch.testing.assert_close(cls_output, baseline, rtol=0, atol=0)
            self.assertTrue(getattr(encoder.forward, "_acestep_sft_timbre_cls", False))
        finally:
            self.ns["_clear_acestep_reference_timbre_repeats"](clone)
            self.ns["_clear_acestep_reference_attention"](clone)
            self.ns["_clear_acestep_timbre_cls_patch"](clone)
        self.assertEqual(encoder.forward, original)
        self.assertEqual([layer.self_attn.sliding_window for layer in encoder.layers], [None] * 4)
        for key, value in encoder.state_dict().items():
            torch.testing.assert_close(value, weights[key], rtol=0, atol=0)

    def test_sampling_exception_restores_encoder_without_losing_unrelated_patch(self):
        patcher, diffusion = test_reference_attention.ReferenceAttentionTests().model(sliding_window=4)
        original = diffusion.encoder.timbre_encoder.forward
        patcher.add_object_patch("other", "preserved")
        with self.assertRaisesRegex(RuntimeError, "sampling failed"):
            with self.ns["_acestep_sampling_model"](patcher) as clone:
                self.ns["_ensure_acestep_timbre_cls"](clone)
                self.ns["_ensure_acestep_reference_attention"](clone)
                self.ns["_ensure_acestep_reference_timbre_repeats"](clone, 3)
                clone.patch_model(load_weights=False)
                raise RuntimeError("sampling failed")
        self.assertEqual(diffusion.encoder.timbre_encoder.forward, original)
        self.assertEqual([layer.self_attn.sliding_window for layer in diffusion.encoder.timbre_encoder.layers], [None] * 4)
        self.assertNotIn("diffusion_model.encoder.timbre_encoder.forward", clone.object_patches)
        self.assertNotIn("_acestep_sft_reference_timbre_repeat_state", clone.model_options)
        self.assertEqual(patcher.model.other, "preserved")
        clone.unpatch_model(unpatch_weights=False)


class ReferenceOptionRoutingTests(unittest.TestCase):
    def test_basic_and_advanced_forward_explicit_repetition_only_when_reference_exists(self):
        harness = test_reference_audio_generate.ReferenceAudioGenerateTests()
        harness.setUp()
        ensure = Mock()
        harness.ns["_ensure_acestep_reference_timbre_repeats"] = ensure
        for advanced in (False, True):
            with self.subTest(advanced=advanced):
                ensure.reset_mock()
                harness.generate(advanced=advanced, reference_timbre_repeats=3)
                self.assertEqual(ensure.call_count, 1)
                self.assertEqual(ensure.call_args.args[1], 3)
                ensure.reset_mock()
                with patch.object(harness.ns["comfy"].sample, "sample_custom", return_value=torch.zeros(1, 64, 40)):
                    harness.generate(advanced=advanced, reference_timbre_repeats=3, reference_audio=None)
                ensure.assert_not_called()

    def test_invalid_repeat_controls_fail_before_sampling_in_basic_and_advanced(self):
        harness = test_reference_audio_generate.ReferenceAudioGenerateTests()
        harness.setUp()
        for advanced in (False, True):
            for value in (True, 0, 9, 2.5):
                with self.subTest(advanced=advanced, value=value):
                    with self.assertRaises(ValueError):
                        harness.generate(advanced=advanced, reference_timbre_repeats=value)
        self.assertEqual(harness.native.calls, [])


if __name__ == "__main__":
    unittest.main()
