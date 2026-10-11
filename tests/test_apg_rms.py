"""CPU invariants for an APG RMS cap independent of song duration."""
import ast
import math
from types import SimpleNamespace
import unittest

import torch
import torch.nn.functional as F

import test_apg_denoised as apg_tests
import test_sampling as sampling


class APGRMSMathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        namespace = {"torch": torch, "F": F, "math": math}
        sampling.load_definitions(
            sampling.NODE_PATH, {"MomentumBuffer", "_project", "apg_guidance"}, namespace
        )
        cls.apg = staticmethod(namespace["apg_guidance"])
        cls.momentum = namespace["MomentumBuffer"]

    def predictions(self):
        conditional = torch.ones(1, 2, 4)
        delta = torch.tensor([[[.5, -.5, .5, -.5], [1., -1., 1., -1.]]])
        return conditional, conditional - delta

    def test_repeating_duration_preserves_rms_capped_prediction(self):
        conditional, unconditional = self.predictions()
        base = self.apg(conditional, unconditional, 2., norm_rms=.075, dims=(1, 2))
        repeated = self.apg(
            conditional.repeat(1, 1, 9), unconditional.repeat(1, 1, 9),
            2., norm_rms=.075, dims=(1, 2)
        )
        torch.testing.assert_close(repeated, base.repeat(1, 1, 9))
        correction_rms = float((base - conditional).square().mean().sqrt())
        self.assertAlmostEqual(correction_rms, .075, places=6)
        legacy = self.apg(conditional, unconditional, 2., dims=(1, 2))
        legacy_repeated = self.apg(
            conditional.repeat(1, 1, 9), unconditional.repeat(1, 1, 9), 2., dims=(1, 2)
        )
        self.assertLess(float((legacy_repeated - conditional.repeat(1, 1, 9)).square().mean()),
                        float((legacy - conditional).square().mean()))

    def test_joint_cap_is_independent_between_batch_items(self):
        conditional, unconditional = self.predictions()
        conditional = torch.cat([conditional, 3. * conditional])
        unconditional = torch.cat([unconditional, -.5 * unconditional])
        together = self.apg(conditional, unconditional, 4., norm_rms=.075, dims=(1, 2))
        separate = torch.cat([
            self.apg(conditional[i:i + 1], unconditional[i:i + 1], 4.,
                     norm_rms=.075, dims=(1, 2))
            for i in range(2)
        ])
        torch.testing.assert_close(together, separate)
        conditional[1] *= 1000.
        unconditional[1] *= .001
        changed = self.apg(conditional, unconditional, 4., norm_rms=.075, dims=(1, 2))
        torch.testing.assert_close(changed[0], together[0])

    def test_per_channel_and_joint_caps_use_their_projected_element_counts(self):
        conditional, unconditional = self.predictions()
        per_channel = self.apg(conditional, unconditional, 2., norm_rms=.075, dims=[-1])
        per_channel_rms = (per_channel - conditional).square().mean(dim=-1).sqrt()
        torch.testing.assert_close(per_channel_rms, torch.full((1, 2), .075))
        joint = self.apg(conditional, unconditional, 2., norm_rms=.075, dims=(1, 2))
        # Original joint delta RMS=sqrt((.5**2+1**2)/2)=sqrt(.625).
        scale = .075 / math.sqrt(.625)
        expected = conditional + (conditional - unconditional) * scale
        torch.testing.assert_close(joint, expected)
        joint_rms = (joint - conditional).square().mean().sqrt()
        self.assertAlmostEqual(float(joint_rms), .075, places=6)
        channel_rms = (joint - conditional).square().mean(dim=-1).sqrt()
        self.assertGreater(float(channel_rms[0, 1]), .075)
        self.assertLess(float(channel_rms[0, 0]), .075)

    def test_zero_override_preserves_fixed_l2_budget_exactly(self):
        conditional, unconditional = self.predictions()
        implicit = self.apg(conditional, unconditional, 4., norm_threshold=.2, dims=(1, 2))
        explicit = self.apg(conditional, unconditional, 4., norm_threshold=.2,
                            dims=(1, 2), norm_rms=0.)
        self.assertTrue(torch.equal(explicit, implicit))
        self.assertAlmostEqual(float((explicit - conditional).norm()), .6, places=6)

    def test_positive_override_clips_when_fixed_l2_clipping_is_disabled(self):
        conditional, unconditional = self.predictions()
        result = self.apg(conditional, unconditional, 2., norm_threshold=0.,
                          dims=(1, 2), norm_rms=.05)
        self.assertAlmostEqual(float((result - conditional).square().mean().sqrt()),
                               .05, places=6)
        unbounded = self.apg(conditional, unconditional, 2., norm_threshold=0., dims=(1, 2))
        self.assertGreater(float((unbounded - conditional).square().mean().sqrt()), .05)

    def test_momentum_history_stores_unclipped_update(self):
        conditional = torch.zeros(1, 2, 4)
        pattern = torch.tensor([[[1., -1., 1., -1.], [1., -1., 1., -1.]]])
        buffer = self.momentum(momentum=-.5)
        first = self.apg(conditional, -.8 * pattern, 2., momentum_buffer=buffer,
                         dims=(1, 2), timestep=.8, norm_rms=.05)
        self.assertAlmostEqual(float(first.square().mean().sqrt()), .05, places=6)
        torch.testing.assert_close(buffer.running_average, .8 * pattern)
        # .4*pattern - .5*(.8*pattern)=0, even though the first returned
        # update was capped to RMS .05. Clipping must not change the history.
        second = self.apg(conditional, -.4 * pattern, 2., momentum_buffer=buffer,
                          dims=(1, 2), timestep=.4, norm_rms=.05)
        self.assertTrue(torch.equal(second, conditional))

    def test_zero_update_stays_finite_with_rms_override(self):
        zero = torch.zeros(2, 3, 4)
        for dims in ([-1], (1, 2)):
            with self.subTest(dims=dims):
                result = self.apg(zero, zero, 7., dims=dims, norm_rms=.075)
                self.assertTrue(torch.isfinite(result).all())
                self.assertTrue(torch.equal(result, zero))


class APGRMSIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.apg = apg_tests.APGDenoisedTests("runTest")
        self.apg.setUp()

    def test_generate_routes_rms_override_to_clean_and_velocity_apg(self):
        namespace = self.apg.harness.ns
        original = namespace["apg_guidance"]
        calls = []

        def capture(*args, **kwargs):
            calls.append((kwargs["norm_rms"], tuple(kwargs["dims"])))
            return original(*args, **kwargs)

        namespace["apg_guidance"] = capture
        conditional = torch.tensor([[[1., 1.], [2., 0.]]])
        unconditional = torch.tensor([[[0., 1.], [2., -1.]]])
        for mode, expected_dims in (
            ("apg", (-1,)), ("apg_denoised", (-1,)), ("apg_denoised_joint", (1, 2))
        ):
            with self.subTest(mode=mode):
                function = self.apg.configure(guidance_mode=mode, apg_norm_rms=.075)
                result = self.apg.evaluate(function, torch.zeros_like(conditional),
                                           conditional, unconditional, .5)
                self.assertEqual(calls[-1], (.075, expected_dims))
                self.assertEqual(result.dtype, conditional.dtype)
                self.assertTrue(torch.isfinite(result).all())

    def test_zero_override_preserves_existing_generate_output_exactly(self):
        options = dict(guidance_mode="apg_denoised_joint", apg_eta=.25,
                       apg_momentum=-.5, apg_norm_threshold=2.5)
        default = self.apg.configure(**options)
        disabled = self.apg.configure(**options, apg_norm_rms=0.)
        conditional = torch.tensor([[[1., 1.], [2., 0.]]])
        unconditional = torch.tensor([[[0., 1.], [2., -1.]]])
        x = torch.zeros_like(conditional)
        for sigma in (.8, .4, .4, .02):
            expected = self.apg.evaluate(default, x, conditional, unconditional, sigma)
            actual = self.apg.evaluate(disabled, x, conditional, unconditional, sigma)
            self.assertTrue(torch.equal(actual, expected))

    def test_jk_rms_cap_runs_in_every_phase_without_extra_model_calls(self):
        function = self.apg.configure(guidance_mode="apg_denoised_joint", cfg=2.,
                                      apg_norm_rms=.075)
        namespace = self.apg.harness.ns
        sampling.load_definitions(sampling.NODE_PATH, {"sample_jkass_quality"}, namespace)
        namespace["model_trange"] = lambda count, **kwargs: range(count)
        original = namespace["apg_guidance"]
        calls, evaluations = [], []

        def capture(*args, **kwargs):
            calls.append((kwargs["norm_rms"], tuple(kwargs["dims"])))
            return original(*args, **kwargs)

        namespace["apg_guidance"] = capture
        conditional = torch.ones(1, 2, 4)
        delta = torch.tensor([[[.5, -.5, .5, -.5], [1., -1., 1., -1.]]])

        def evaluate(x, sigma):
            value = float(sigma[0])
            evaluations.append(value)
            return self.apg.evaluate(function, x, conditional, conditional - delta,
                                     value, cfg=2.)

        result = namespace["sample_jkass_quality"](
            evaluate, torch.zeros_like(conditional), torch.linspace(1., 0., 31), disable=True
        )
        self.assertEqual(len(evaluations), 59)
        self.assertEqual(calls, [(.075, (1, 2))] * 59)
        for index in range(1, 59, 2):
            self.assertEqual(evaluations[index], evaluations[index + 1])
        expected = conditional + delta * (.075 / math.sqrt(.625))
        torch.testing.assert_close(result, expected)
        at_zero = self.apg.evaluate(function, conditional, conditional, conditional, 0., cfg=2.)
        self.assertTrue(torch.isfinite(at_zero).all())
        torch.testing.assert_close(at_zero, conditional)

    def test_nonfinite_rms_override_is_rejected(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "apg_norm_rms"):
                    self.apg.harness.generate(apg_norm_rms=value)

    def test_rms_widget_is_appended_and_disabled_by_default(self):
        namespace = self.apg.harness.ns
        tree = ast.parse(sampling.NODE_PATH.read_text(encoding="utf-8"))
        for item in tree.body:
            if isinstance(item, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "GUIDANCE_MODES"
                for target in item.targets
            ):
                namespace["GUIDANCE_MODES"] = ast.literal_eval(item.value)
                break
        namespace["ACESTEP_SFT_TASK_TYPE"] = "ACESTEP_SFT_TASK"
        namespace["comfy"].samplers.KSampler = SimpleNamespace(
            SAMPLERS=["euler", "heun"], SCHEDULERS=["normal"]
        )
        optional = namespace["AceStepSFTGenerate"].INPUT_TYPES()["optional"]
        self.assertEqual(list(optional)[-1], "apg_norm_rms")
        kind, settings = optional["apg_norm_rms"]
        self.assertEqual(kind, "FLOAT")
        self.assertEqual(settings["default"], 0.)
        self.assertEqual(settings["min"], 0.)
        self.assertEqual(settings["max"], 1.)
        self.assertEqual(settings["step"], .005)


if __name__ == "__main__":
    unittest.main()
