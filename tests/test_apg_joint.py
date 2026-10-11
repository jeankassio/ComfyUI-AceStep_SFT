"""CPU geometry and sampler checks for joint clean-latent APG."""
import ast
import math
from types import SimpleNamespace
import unittest

import torch

import test_apg_denoised as apg_tests
import test_sampling as sampling


class APGJointTests(unittest.TestCase):
    def setUp(self):
        self.apg = apg_tests.APGDenoisedTests("runTest")
        self.apg.setUp()

    def configure(self, **overrides):
        options = {"guidance_mode": "apg_denoised_joint"}
        options.update(overrides)
        return self.apg.configure(**options)

    def test_two_channel_projection_matches_hand_calculation(self):
        function = self.configure()
        conditional = torch.tensor([[[1., 1.], [2., 0.]]])
        unconditional = torch.tensor([[[0., 1.], [2., -1.]]])
        # delta=[1,0,0,1], dot(delta,cond)=1, dot(cond,cond)=6.
        # At CFG 4: cond + 3*(delta-cond/6).
        expected = torch.tensor([[[3.5, .5], [1., 3.]]])
        actual = self.apg.evaluate(
            function, torch.zeros_like(conditional), conditional, unconditional, .5, cfg=4.
        )
        torch.testing.assert_close(actual, expected)
        channelwise = self.apg.configure()
        previous = self.apg.evaluate(
            channelwise, torch.zeros_like(conditional), conditional, unconditional, .5, cfg=4.
        )
        torch.testing.assert_close(previous, torch.tensor([[[2.5, -.5], [2., 3.]]]))
        self.assertFalse(torch.equal(actual, previous))

    def test_batch_projection_does_not_mix_songs(self):
        function = self.configure()
        conditional = torch.tensor([[[1., 1.], [2., 0.]], [[2., -1.], [.5, 3.]]])
        unconditional = torch.tensor([[[0., 1.], [2., -1.]], [[-.5, 2.], [1., -1.]]])
        x = torch.zeros_like(conditional)
        together = self.apg.evaluate(function, x, conditional, unconditional, .5)
        separate = torch.cat([
            self.apg.evaluate(function, x[i:i + 1], conditional[i:i + 1],
                              unconditional[i:i + 1], .5)
            for i in range(2)
        ])
        torch.testing.assert_close(together, separate)
        changed_conditional = conditional.clone()
        changed_unconditional = unconditional.clone()
        changed_conditional[1] *= 1000.
        changed_unconditional[1] *= .001
        changed = self.apg.evaluate(function, x, changed_conditional,
                                    changed_unconditional, .5)
        torch.testing.assert_close(changed[0], together[0])

    def test_joint_radius_matches_aggregate_channel_budget(self):
        function = self.configure(apg_norm_threshold=.75)
        conditional = torch.tensor([[[1., 0.], [0., 1.]]])
        unconditional = torch.tensor([[[1., -10.], [-10., 1.]]])
        # delta=[0,10,10,0] is orthogonal to cond. The global radius is
        # .75*sqrt(2), so each nonzero clipped component becomes .75.
        actual = self.apg.evaluate(
            function, torch.zeros_like(conditional), conditional, unconditional, .5, cfg=3.
        )
        torch.testing.assert_close(actual, torch.tensor([[[1., 1.5], [1.5, 1.]]]))
        update = (actual - conditional) / 2.
        self.assertAlmostEqual(float(update.norm()), .75 * math.sqrt(2), places=6)

    def test_eta_one_without_clipping_or_momentum_is_cfg(self):
        function = self.configure(apg_eta=1.)
        conditional = torch.tensor([[[1., 1.], [2., 0.]]])
        unconditional = torch.tensor([[[0., 1.], [2., -1.]]])
        expected = unconditional + 7. * (conditional - unconditional)
        for sigma in (1., .5, .01):
            with self.subTest(sigma=sigma):
                actual = self.apg.evaluate(function, torch.zeros_like(conditional),
                                          conditional, unconditional, sigma)
                torch.testing.assert_close(actual, expected)

    def test_zero_predictions_and_zero_sigma_stay_finite(self):
        function = self.configure(apg_norm_threshold=2.5)
        zero = torch.zeros(2, 3, 4)
        for sigma in (1., .5, 0.):
            with self.subTest(sigma=sigma):
                actual = self.apg.evaluate(function, zero, zero, zero, sigma)
                self.assertTrue(torch.isfinite(actual).all())
                self.assertTrue(torch.equal(actual, zero))

    def test_one_channel_matches_previous_clean_apg(self):
        options = dict(apg_eta=.25, apg_norm_threshold=.75, apg_momentum=-.5)
        joint = self.configure(**options)
        previous = self.apg.configure(**options)
        for sigma, conditional, unconditional in (
            (.8, [2., 1.], [-1., 2.]),
            (.4, [1., 3.], [-2., 1.]),
        ):
            conditional = torch.tensor([[conditional]])
            unconditional = torch.tensor([[unconditional]])
            x = torch.zeros_like(conditional)
            expected = self.apg.evaluate(previous, x, conditional, unconditional, sigma)
            actual = self.apg.evaluate(joint, x, conditional, unconditional, sigma)
            torch.testing.assert_close(actual, expected)

    def test_rescale_runs_after_joint_projection_in_velocity_space(self):
        function = self.configure(guidance_rescale=.5)
        x = torch.tensor([[[0., 1.], [1., -1.]]])
        conditional = torch.tensor([[[1., 1.], [2., 0.]]])
        unconditional = torch.tensor([[[0., 1.], [2., -1.]]])
        # Joint APG at CFG 4 is [[3.5,.5],[1,3]]. The residual has
        # std=sqrt(65)/4; the conditional residual has std=sqrt(3)/4.
        guided_output = torch.tensor([[[-3.5, .5], [0., -4.]]])
        scale = .5 + .5 * math.sqrt(3. / 65.)
        expected = x - guided_output * scale
        actual = self.apg.evaluate(function, x, conditional, unconditional, .5, cfg=4.)
        torch.testing.assert_close(actual, expected)

    def test_jk_guides_every_evaluation_without_extra_model_calls(self):
        function = self.configure(apg_eta=1., guidance_rescale=.5)
        namespace = self.apg.harness.ns
        sampling.load_definitions(sampling.NODE_PATH, {"sample_jkass_quality"}, namespace)
        namespace["model_trange"] = lambda count, **kwargs: range(count)
        original_apg = namespace["apg_guidance"]
        original_rescale = namespace["_rescale_guidance_output"]
        apg_calls, rescale_calls, evaluations = [], [], []

        def capture_apg(*args, **kwargs):
            apg_calls.append(tuple(kwargs["dims"]))
            return original_apg(*args, **kwargs)

        def capture_rescale(guided, conditional, strength):
            rescale_calls.append(strength)
            return original_rescale(guided, conditional, strength)

        namespace["apg_guidance"] = capture_apg
        namespace["_rescale_guidance_output"] = capture_rescale
        velocity = torch.tensor([[[1., -1.], [2., -2.]]])

        def evaluate(x, sigma):
            value = float(sigma[0])
            evaluations.append(value)
            return self.apg.evaluate(function, x, x - value * velocity, x, value)

        result = namespace["sample_jkass_quality"](
            evaluate, torch.zeros_like(velocity), torch.linspace(1., 0., 31), disable=True
        )
        self.assertEqual(len(evaluations), 59)
        self.assertEqual(apg_calls, [(1, 2)] * 59)
        self.assertEqual(rescale_calls, [.5] * 59)
        for index in range(1, 59, 2):
            self.assertEqual(evaluations[index], evaluations[index + 1])
        torch.testing.assert_close(result, -4. * velocity)

    def test_previous_mode_remains_identical_to_channelwise_helper(self):
        function = self.apg.configure(apg_eta=.25, apg_norm_threshold=.75, apg_momentum=-.5)
        namespace = self.apg.harness.ns
        momentum = namespace["MomentumBuffer"](momentum=-.5)
        for sigma, conditional, unconditional in (
            (.8, [[2., 1.], [1., -2.]], [[-1., 2.], [.5, 1.]]),
            (.4, [[1., 3.], [-1., 2.]], [[-2., 1.], [1., .5]]),
        ):
            conditional = torch.tensor([conditional])
            unconditional = torch.tensor([unconditional])
            expected = namespace["apg_guidance"](
                conditional, unconditional, 7., momentum_buffer=momentum,
                eta=.25, norm_threshold=.75, dims=[-1], timestep=sigma
            )
            actual = self.apg.evaluate(function, torch.zeros_like(conditional),
                                      conditional, unconditional, sigma)
            self.assertTrue(torch.equal(actual, expected))

    def test_schema_exposes_joint_choice_without_replacing_previous_modes(self):
        tree = ast.parse(sampling.NODE_PATH.read_text(encoding="utf-8"))
        for item in tree.body:
            if isinstance(item, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "GUIDANCE_MODES"
                for target in item.targets
            ):
                modes = ast.literal_eval(item.value)
                break
        else:
            self.fail("GUIDANCE_MODES is missing")
        namespace = self.apg.harness.ns
        namespace["GUIDANCE_MODES"] = modes
        namespace["ACESTEP_SFT_TASK_TYPE"] = "ACESTEP_SFT_TASK"
        namespace["comfy"].samplers.KSampler = SimpleNamespace(
            SAMPLERS=["euler", "heun"], SCHEDULERS=["normal"]
        )
        choices = namespace["AceStepSFTGenerate"].INPUT_TYPES()["required"]["guidance_mode"][0]
        self.assertEqual(choices.count("apg_denoised_joint"), 1)
        for previous in ("apg", "adg", "standard_cfg", "apg_denoised"):
            self.assertIn(previous, choices)


if __name__ == "__main__":
    unittest.main()
