"""Math and real CFG-hook/Heun integration checks for velocity rescale."""
import math
import unittest

import torch

import test_apg_denoised as apg_tests
import test_sampling as sampling


class RescaleMathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        namespace = {"torch": torch}
        sampling.load_definitions(
            sampling.NODE_PATH, {"_rescale_guidance_output"}, namespace
        )
        cls.rescale = staticmethod(namespace["_rescale_guidance_output"])

    def test_disabled_returns_original_without_reading_conditional(self):
        guided = torch.tensor([[[1.0, -2.0]]])
        self.assertIs(self.rescale(guided, None, 0.0), guided)

    def test_full_and_partial_rescale_use_per_sample_channel_time_statistics(self):
        conditional = torch.tensor([[[-1., 1.], [-1., 1.]],
                                    [[-2., 2.], [-2., 2.]]])
        guided = torch.tensor([[[-4., 4.], [-4., 4.]],
                               [[-6., 6.], [-6., 6.]]])
        torch.testing.assert_close(self.rescale(guided, conditional, 1.), conditional)
        expected = torch.tensor([[[-2.5, 2.5], [-2.5, 2.5]],
                                 [[-4., 4.], [-4., 4.]]])
        torch.testing.assert_close(self.rescale(guided, conditional, .5), expected)
        torch.testing.assert_close(guided[0], torch.tensor([[-4., 4.], [-4., 4.]]))

    def test_batch_item_does_not_depend_on_other_song(self):
        guided = torch.tensor([[[2., -6., 4.]], [[-3., 9., 6.]]])
        conditional = torch.tensor([[[1., -1., 0.]], [[-2., 0., 1.]]])
        together = self.rescale(guided, conditional, .5)
        separate = torch.cat([
            self.rescale(guided[i:i + 1], conditional[i:i + 1], .5)
            for i in range(2)
        ])
        torch.testing.assert_close(together, separate)
        guided[1] *= 1000
        conditional[1] *= .001
        changed = self.rescale(guided, conditional, .5)
        torch.testing.assert_close(changed[0], together[0])

    def test_sigma_factor_cancels_without_dividing_by_sigma(self):
        guided_velocity = torch.tensor([[[-4., 4.]]])
        conditional_velocity = torch.tensor([[[-1., 1.]]])
        for sigma in (1., .5, .01, .0001):
            with self.subTest(sigma=sigma):
                actual = self.rescale(
                    sigma * guided_velocity, sigma * conditional_velocity, .25
                )
                torch.testing.assert_close(actual, sigma * .8125 * guided_velocity)

    def test_constant_singleton_and_zero_predictions_stay_finite(self):
        for guided, conditional in (
            (torch.full((1, 2, 3), 7.), torch.arange(6.).reshape(1, 2, 3)),
            (torch.tensor([[[9.]]]), torch.tensor([[[2.]]])),
            (torch.zeros(1, 2, 3), torch.zeros(1, 2, 3)),
            (torch.tensor([[[-1e-10, 1e-10]]]), torch.tensor([[[-1., 1.]]])),
        ):
            with self.subTest(shape=guided.shape, value=float(guided.flatten()[0])):
                result = self.rescale(guided, conditional, 1.)
                self.assertTrue(torch.isfinite(result).all())
                torch.testing.assert_close(result, guided, rtol=0, atol=0)

    def test_fp32_statistics_preserve_output_dtype_and_device(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            with self.subTest(dtype=dtype):
                guided = torch.tensor([[[1., 3., 5., 7.]]], dtype=dtype)
                conditional = torch.tensor([[[-1., 1., -1., 1.]]], dtype=dtype)
                expected = (guided.float() * (.5 + .5 / math.sqrt(5))).to(dtype)
                actual = self.rescale(guided, conditional, .5)
                self.assertEqual(actual.dtype, dtype)
                self.assertEqual(actual.device, guided.device)
                torch.testing.assert_close(actual, expected)


class RescaleIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.apg = apg_tests.APGDenoisedTests("runTest")
        self.apg.setUp()

    def test_zero_preserves_apg_denoised_results_exactly(self):
        default = self.apg.configure()
        disabled = self.apg.configure(guidance_rescale=0.)
        x = torch.tensor([[[0., 1.]]])
        conditional = torch.tensor([[[1., 1.]]])
        unconditional = torch.tensor([[[0., 1.]]])
        expected = self.apg.evaluate(default, x, conditional, unconditional, .5)
        actual = self.apg.evaluate(disabled, x, conditional, unconditional, .5)
        self.assertTrue(torch.equal(actual, expected))

    def test_rescale_follows_apg_projection_in_velocity_space(self):
        function = self.apg.configure(guidance_rescale=.5)
        # APG x0=[4,-2], hence sigma*v=[-4,3]. Its std=3.5;
        # conditional sigma*v=[-1,0] has std=.5. Half rescale gives 4/7.
        actual = self.apg.evaluate(
            function, torch.tensor([[[0., 1.]]]),
            torch.tensor([[[1., 1.]]]), torch.tensor([[[0., 1.]]]), .5
        )
        torch.testing.assert_close(actual, torch.tensor([[[16 / 7, -5 / 7]]]))

    def test_standard_cfg_rescales_velocity_rather_than_clean_latent(self):
        function = self.apg.configure(guidance_mode="standard_cfg", guidance_rescale=1.)
        actual = self.apg.evaluate(
            function, torch.tensor([[[2., 3.]]]),
            torch.tensor([[[1., -1.]]]), torch.zeros(1, 1, 2), .5, cfg=2.
        )
        # Guided output [0,5] has std=2.5; conditional [1,4] has std=1.5.
        torch.testing.assert_close(actual, torch.tensor([[[2., 0.]]]))

    def test_rescale_alone_keeps_cfg_one_native(self):
        self.apg.harness.generate(
            steps=3, cfg=1., guidance_mode="apg_denoised",
            guidance_schedule="normal", guidance_rescale=.5,
        )
        options = self.apg.harness.sampled_model.model_options
        self.assertNotIn("sampler_cfg_function", options)
        self.assertNotIn("disable_cfg1_optimization", options)

    def test_conditional_passes_and_zero_sigma_remain_unchanged(self):
        x = torch.tensor([[[2., 3.]]])
        conditional = torch.tensor([[[1., -1.]]])
        unconditional = torch.full_like(x, float("nan"))
        function = self.apg.configure(
            cfg=1., guidance_rescale=.5, guidance_schedule="decay", min_guidance_scale=1.
        )
        actual = self.apg.evaluate(function, x, conditional, unconditional, .5, cfg=1.)
        torch.testing.assert_close(actual, conditional)
        function = self.apg.configure(
            guidance_rescale=.5, guidance_interval=-1.,
            cfg_interval_start=.2, cfg_interval_end=.8,
        )
        for sigma in (1., .01):
            actual = self.apg.evaluate(function, x, conditional, unconditional, sigma)
            torch.testing.assert_close(actual, conditional)
        actual = self.apg.evaluate(function, x, x, x, 0.)
        self.assertTrue(torch.isfinite(actual).all())
        torch.testing.assert_close(actual, x)

    def test_jk_rescales_every_predictor_corrector_without_extra_evaluations(self):
        function = self.apg.configure(apg_eta=1., guidance_rescale=.5)
        namespace = self.apg.harness.ns
        sampling.load_definitions(sampling.NODE_PATH, {"sample_jkass_quality"}, namespace)
        namespace["model_trange"] = lambda count, **kwargs: range(count)
        original = namespace["_rescale_guidance_output"]
        rescale_calls = []
        evaluated_sigmas = []

        def rescale(guided, conditional, strength):
            rescale_calls.append(strength)
            return original(guided, conditional, strength)

        namespace["_rescale_guidance_output"] = rescale
        velocity = torch.tensor([[[1., -1.]]])

        def evaluate(x, sigma):
            value = float(sigma[0])
            evaluated_sigmas.append(value)
            return self.apg.evaluate(function, x, x - value * velocity, x, value)

        result = namespace["sample_jkass_quality"](
            evaluate, torch.zeros_like(velocity), torch.linspace(1., 0., 31), disable=True
        )
        self.assertEqual(len(evaluated_sigmas), 59)
        self.assertEqual(rescale_calls, [.5] * 59)
        for index in range(1, 59, 2):
            self.assertEqual(evaluated_sigmas[index], evaluated_sigmas[index + 1])
        torch.testing.assert_close(result, -4 * velocity)


if __name__ == "__main__":
    unittest.main()
