"""Reference fitting on CPU; no ComfyUI imports, model downloads or CUDA."""
import importlib.util
from pathlib import Path
import sys
from types import MethodType
import unittest

import torch


SPEC = importlib.util.spec_from_file_location(
    "voice_adaptation", Path(__file__).resolve().parents[1] / "voice_adaptation.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
fit_timbre = MODULE.fit_timbre
install_voice_profile = MODULE.install_voice_profile
clear_voice_profile = MODULE.clear_voice_profile


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class TokenDecoder(torch.nn.Module):
    def __init__(self, clock, detached=False, nonfinite=False, forward_seconds=0.01):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.clock = clock
        self.detached = detached
        self.nonfinite = nonfinite
        self.forward_seconds = forward_seconds
        self.inputs = []

    def forward(self, hidden_states, timestep, timestep_r, attention_mask,
                encoder_hidden_states, encoder_attention_mask, context_latents):
        self.clock.now += self.forward_seconds
        self.inputs.append((timestep.detach().clone(), context_latents.detach().clone()))
        torch.testing.assert_close(timestep, timestep_r, rtol=0, atol=0)
        token = encoder_hidden_states[:, 2:3].mean(dim=-1, keepdim=True)
        result = -self.scale * token.expand_as(hidden_states)
        if self.detached:
            result = result.detach()
        if self.nonfinite:
            result = result * float("nan")
        return result


def example(value=2.0):
    return {
        "target": torch.full((1, 128, 64), value),
        "lyrics": torch.full((1, 2, 8), 0.25),
        "text": torch.full((1, 3, 8), 0.5),
        "context": torch.cat((torch.zeros((1, 128, 64)), torch.ones((1, 128, 64))), dim=-1),
    }


class DeadlineDecoder(TokenDecoder):
    def __init__(self, clock, fail_on_call=None, fail_backward=False):
        super().__init__(clock)
        self.fail_on_call = fail_on_call
        self.fail_backward = fail_backward

    def forward(self, **kwargs):
        if len(self.inputs) + 1 == self.fail_on_call:
            self.clock.now += self.forward_seconds
            raise TimeoutError("adaptation deadline")
        output = super().forward(**kwargs)
        if self.fail_backward and output.requires_grad:
            def stop_backward(gradient):
                raise TimeoutError("adaptation backward deadline")
            output.register_hook(stop_backward)
        return output


class ProfileEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.arange(8.0), requires_grad=False)

    def forward(self, reference, order):
        embeddings = self.weight.to(reference).reshape(1, 1, -1).expand(reference.shape[0], 3, -1)
        mask = torch.tensor([[1, 1, 0]], device=reference.device).expand(reference.shape[0], -1)
        return embeddings, mask


class VoiceProfileTests(unittest.TestCase):
    def model(self):
        # The existing fixture implements the same patch/restore protocol as
        # ModelPatcher without importing a GPU runtime.
        from test_conditioning_precompute import Patcher
        base = torch.nn.Module()
        base.diffusion_model = torch.nn.Module()
        base.diffusion_model.encoder = torch.nn.Module()
        encoder = ProfileEncoder()
        base.diffusion_model.encoder.timbre_encoder = encoder
        return Patcher(base), encoder

    def test_profile_is_opt_in_and_keeps_batch_repeats_dtype_and_mask(self):
        model, encoder = self.model()
        reference = torch.zeros((2, 10, 64), dtype=torch.float64)
        order = torch.arange(2)
        before, original_mask = encoder(reference, order)
        learned = torch.full((1, 1, 8), 0.75)
        install_voice_profile(model, learned)
        model.patch_model(load_weights=False)
        current, mask = encoder(reference, order)
        torch.testing.assert_close(current, before, rtol=0, atol=0)
        torch.testing.assert_close(mask, original_mask, rtol=0, atol=0)
        model.model_options["_acestep_sft_voice_profile"]["enabled"] = True
        current, mask = encoder(reference, order)
        self.assertEqual(current.shape, (2, 3, 8))
        self.assertEqual(current.dtype, torch.float64)
        torch.testing.assert_close(current, learned.to(current).expand_as(current), rtol=0, atol=0)
        torch.testing.assert_close(mask, original_mask, rtol=0, atol=0)
        # The caller disables the profile for null/no-reference branches.
        model.model_options["_acestep_sft_voice_profile"]["enabled"] = False
        current, _ = encoder(reference, order)
        torch.testing.assert_close(current, before, rtol=0, atol=0)
        clear_voice_profile(model)
        self.assertNotIn("_acestep_sft_voice_profile", model.model_options)
        self.assertEqual(model.object_patches, {})
        current, _ = encoder(reference, order)
        torch.testing.assert_close(current, before, rtol=0, atol=0)

    def test_cleanup_restores_prior_patch_and_leaves_weights_unchanged(self):
        model, encoder = self.model()
        original = encoder.forward
        weights = {name: value.clone() for name, value in encoder.state_dict().items()}

        def prior(self, reference, order):
            embeddings, mask = original(reference, order)
            return embeddings + 4, mask

        prior_method = MethodType(prior, encoder)
        model.add_object_patch("diffusion_model.encoder.timbre_encoder.forward", prior_method)
        model.patch_model(load_weights=False)
        install_voice_profile(model, torch.zeros((1, 1, 8)))
        model.patch_model(load_weights=False)
        model.model_options["_acestep_sft_voice_profile"]["enabled"] = True
        clear_voice_profile(model)
        self.assertIs(encoder.forward, prior_method)
        self.assertIs(model.object_patches["diffusion_model.encoder.timbre_encoder.forward"], prior_method)
        for name, value in encoder.state_dict().items():
            torch.testing.assert_close(value, weights[name], rtol=0, atol=0)
        clear_voice_profile(model)
        self.assertIs(encoder.forward, prior_method)

    def test_reinstallation_replaces_profile_without_stacking(self):
        model, encoder = self.model()
        install_voice_profile(model, torch.ones((1, 1, 8)))
        model.patch_model(load_weights=False)
        install_voice_profile(model, torch.full((1, 1, 8), 2.0))
        model.patch_model(load_weights=False)
        model.model_options["_acestep_sft_voice_profile"]["enabled"] = True
        result, _ = encoder(torch.zeros((1, 10, 64)), torch.zeros(1, dtype=torch.long))
        torch.testing.assert_close(result, torch.full((1, 3, 8), 2.0), rtol=0, atol=0)
        clear_voice_profile(model)
        self.assertEqual(model.object_patches, {})


class VoiceAdaptationTests(unittest.TestCase):
    def test_installed_native_decoder_propagates_timbre_gradients(self):
        comfy_root = Path(__file__).resolve().parents[3]
        sys.path.insert(0, str(comfy_root))
        from comfy.cli_args import args
        args.cpu = True
        from comfy.ldm.ace.ace_step15 import AceStepDiTModel

        generator_state = torch.get_rng_state().clone()
        torch.manual_seed(321)
        try:
            native = AceStepDiTModel(
                in_channels=192, hidden_size=8, num_layers=1,
                num_heads=2, num_kv_heads=1, head_dim=4, intermediate_size=16,
                patch_size=2, audio_acoustic_hidden_dim=64, condition_dim=8,
                dtype=torch.float32, device="cpu", operations=torch.nn,
            )
            for parameter in native.parameters():
                torch.nn.init.uniform_(parameter, -0.2, 0.2)
                parameter.requires_grad_(False)
            clock = Clock()

            def decoder(**kwargs):
                clock.now += 0.01
                return native(**kwargs)

            samples = [example(0.2), example(0.2)]
            for item in samples:
                item["target"] = item["target"][:, :9].clone()
                item["context"] = item["context"][:, :9].clone()
            result, report = fit_timbre(
                decoder, torch.ones((1, 1, 8)), samples,
                seconds=0.15, clock=clock,
            )
            self.assertGreater(report["steps"], 0)
            self.assertNotIn(report["stop_reason"], (
                "missing_timbre_gradient", "zero_timbre_gradient", "backward_failed",
            ))
            self.assertTrue(torch.isfinite(result).all())
            self.assertTrue(all(parameter.grad is None for parameter in native.parameters()))
        finally:
            torch.set_rng_state(generator_state)

    def test_fit_updates_only_timbre_and_improves_validation(self):
        clock = Clock()
        decoder = TokenDecoder(clock)
        weights_before = {key: value.clone() for key, value in decoder.state_dict().items()}
        initial = torch.ones((1, 1, 8))
        examples = [example(), example()]
        originals = [{key: value.clone() for key, value in item.items()} for item in examples]
        result, report = fit_timbre(decoder, initial, examples, seconds=0.5, clock=clock)
        self.assertTrue(report["accepted"])
        self.assertGreater(report["steps"], 0)
        self.assertLess(report["final_validation_loss"], report["initial_validation_loss"])
        self.assertLessEqual(report["relative_change"], 0.150001)
        self.assertFalse(result.requires_grad)
        self.assertTrue(report["validation_held_out"])
        self.assertLessEqual(report["elapsed_seconds"], 0.5)
        torch.testing.assert_close(initial, torch.ones_like(initial), rtol=0, atol=0)
        for key, value in decoder.state_dict().items():
            torch.testing.assert_close(value, weights_before[key], rtol=0, atol=0)
        self.assertTrue(all(parameter.grad is None for parameter in decoder.parameters()))
        for current, original in zip(examples, originals):
            for name in current:
                torch.testing.assert_close(current[name], original[name], rtol=0, atol=0)
        for timestep, context in decoder.inputs:
            self.assertGreater(float(timestep), 0.0)
            self.assertLess(float(timestep), 1.0)
            torch.testing.assert_close(context, examples[0]["context"], rtol=0, atol=0)

    def test_validation_rejects_training_that_hurts_other_reference(self):
        clock = Clock()
        initial = torch.ones((1, 1, 8))
        result, report = fit_timbre(
            TokenDecoder(clock), initial, [example(2.0), example(0.0)],
            seconds=0.25, clock=clock,
        )
        self.assertFalse(report["accepted"])
        torch.testing.assert_close(result, initial, rtol=0, atol=0)
        self.assertEqual(report["final_validation_loss"], report["initial_validation_loss"])

    def test_zero_budget_is_exact_bypass_even_in_inference_mode(self):
        with torch.inference_mode():
            initial = torch.ones((1, 1, 8))
            result, report = fit_timbre(None, initial, [], seconds=0)
        self.assertIs(result, initial)
        self.assertEqual(report["steps"], 0)
        self.assertEqual(report["elapsed_seconds"], 0.0)
        self.assertEqual(report["stop_reason"], "disabled")

    def test_reserves_validation_budget_before_backward(self):
        clock = Clock()
        result, report = fit_timbre(
            TokenDecoder(clock), torch.ones((1, 1, 8)), [example(), example()],
            seconds=0.03, clock=clock,
        )
        self.assertEqual(report["steps"], 0)
        self.assertEqual(report["elapsed_seconds"], 0.01)
        self.assertFalse(report["accepted"])

    def test_uncancellable_initial_forward_reports_actual_overrun(self):
        clock = Clock()
        result, report = fit_timbre(
            TokenDecoder(clock, forward_seconds=0.2), torch.ones((1, 1, 8)),
            [example()], seconds=0.05, clock=clock,
        )
        self.assertEqual(report["steps"], 0)
        self.assertAlmostEqual(report["elapsed_seconds"], 0.2)
        self.assertAlmostEqual(report["deadline_overrun_seconds"], 0.15)

    def test_global_rng_is_unchanged_and_seed_is_reproducible(self):
        torch.manual_seed(123)
        before = torch.get_rng_state().clone()
        outputs = []
        reports = []
        for _ in range(2):
            clock = Clock()
            result, report = fit_timbre(
                TokenDecoder(clock), torch.ones((1, 1, 8)),
                [example(), example()], seconds=0.2, seed=15, clock=clock,
            )
            outputs.append(result)
            reports.append(report)
        torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)
        torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
        self.assertEqual(reports[0], reports[1])

    def test_disconnected_decoder_returns_original_without_fake_learning(self):
        clock = Clock()
        initial = torch.ones((1, 1, 8))
        result, report = fit_timbre(
            TokenDecoder(clock, detached=True), initial, [example(), example()],
            seconds=0.2, clock=clock,
        )
        self.assertEqual(report["stop_reason"], "missing_timbre_gradient")
        self.assertFalse(report["accepted"])
        torch.testing.assert_close(result, initial, rtol=0, atol=0)

    def test_nonfinite_validation_is_rejected(self):
        clock = Clock()
        initial = torch.ones((1, 1, 8))
        result, report = fit_timbre(
            TokenDecoder(clock, nonfinite=True), initial, [example()],
            seconds=0.2, clock=clock,
        )
        self.assertEqual(report["stop_reason"], "nonfinite_validation")
        self.assertFalse(report["accepted"])
        torch.testing.assert_close(result, initial, rtol=0, atol=0)

    def test_caller_must_disable_inference_mode(self):
        with torch.inference_mode():
            with self.assertRaisesRegex(RuntimeError, "inference_mode"):
                fit_timbre(None, torch.ones((1, 1, 8)), [example()], seconds=0.2)

    def test_inference_inputs_are_cloned_into_ordinary_autograd_tensors(self):
        with torch.inference_mode():
            initial = torch.ones((1, 1, 8))
            examples = [example(), example()]
        clock = Clock()
        result, report = fit_timbre(
            TokenDecoder(clock), initial, examples, seconds=0.2, clock=clock,
        )
        self.assertTrue(report["accepted"])
        self.assertFalse(result.is_inference())

    def test_cooperative_deadlines_never_accept_unvalidated_candidates(self):
        cases = ((1, "initial_validation", 0), (2, "training", 0),
                 (3, "candidate_validation", 1))
        initial = torch.ones((1, 1, 8))
        for call, phase, steps in cases:
            with self.subTest(phase=phase):
                clock = Clock()
                result, report = fit_timbre(
                    DeadlineDecoder(clock, fail_on_call=call), initial,
                    [example(), example()], seconds=0.5, clock=clock,
                )
                self.assertTrue(report["timed_out"])
                self.assertEqual(report["stop_reason"], "deadline_during_" + phase)
                self.assertEqual(report["steps"], steps)
                self.assertFalse(report["accepted"])
                torch.testing.assert_close(result, initial, rtol=0, atol=0)

    def test_backward_deadline_returns_original_and_keeps_weights_clean(self):
        clock = Clock()
        decoder = DeadlineDecoder(clock, fail_backward=True)
        initial = torch.ones((1, 1, 8))
        result, report = fit_timbre(
            decoder, initial, [example(), example()], seconds=0.5, clock=clock,
        )
        self.assertTrue(report["timed_out"])
        self.assertEqual(report["stop_reason"], "deadline_during_backward")
        self.assertFalse(report["accepted"])
        torch.testing.assert_close(result, initial, rtol=0, atol=0)
        self.assertTrue(all(parameter.grad is None for parameter in decoder.parameters()))

    def test_deadline_preserves_already_validated_improvement(self):
        clock = Clock()
        initial = torch.ones((1, 1, 8))
        result, report = fit_timbre(
            DeadlineDecoder(clock, fail_on_call=5), initial,
            [example(), example()], seconds=0.5, clock=clock,
        )
        self.assertTrue(report["timed_out"])
        self.assertTrue(report["accepted"])
        self.assertEqual(report["steps"], 2)
        self.assertLess(report["final_validation_loss"], report["initial_validation_loss"])
        # Only the first update was validated; the second must be discarded.
        self.assertLess(float(result.mean()), 1.03)

    def test_invalid_reference_context_and_settings_fail_clearly(self):
        invalid = example()
        invalid["context"] = invalid["target"]
        with self.assertRaisesRegex(ValueError, "silence"):
            fit_timbre(None, torch.ones((1, 1, 8)), [invalid], seconds=1)
        with self.assertRaisesRegex(ValueError, "finite"):
            fit_timbre(None, torch.ones((1, 1, 8)), [], seconds=float("nan"))


if __name__ == "__main__":
    unittest.main()
