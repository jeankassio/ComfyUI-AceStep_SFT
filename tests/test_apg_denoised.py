"""CPU checks for APG in clean-latent space and the bundled Heun sampler."""
import math
import unittest

import torch

import test_sampling as sampling


class APGDenoisedTests(unittest.TestCase):
    def setUp(self):
        self.harness = sampling.SamplingTests("runTest")
        self.harness.setUp()
        self.harness.ns["comfy"].sample.sample = self.capture_model

    def capture_model(self, model, noise, steps, cfg, sampler_name, scheduler,
                      positive, negative, latent_image, **kwargs):
        self.harness.sampled_model = model
        return latent_image

    def configure(self, **overrides):
        options = dict(
            steps=30, guidance_mode="apg_denoised", guidance_schedule="normal",
            apg_eta=0.0, apg_momentum=0.0, apg_norm_threshold=0.0,
        )
        options.update(overrides)
        self.harness.generate(**options)
        return self.harness.sampled_model.model_options["sampler_cfg_function"]

    def evaluate(self, function, x, conditional, unconditional, sigma, cfg=7.0):
        return x - function({
            "cond_denoised": conditional,
            "uncond_denoised": unconditional,
            "cond_scale": cfg,
            "input": x,
            "sigma": torch.full((x.shape[0],), sigma, dtype=x.dtype),
            "model_options": self.harness.sampled_model.model_options,
        })

    def test_clean_prediction_projection_differs_from_velocity_projection(self):
        x = torch.tensor([[[0.0, 1.0]]])
        conditional = torch.tensor([[[1.0, 0.0]]])
        unconditional = torch.zeros_like(conditional)
        clean_function = self.configure()
        velocity_function = self.configure(guidance_mode="apg")
        clean = self.evaluate(clean_function, x, conditional, unconditional, 0.5)
        velocity = self.evaluate(velocity_function, x, conditional, unconditional, 0.5)
        torch.testing.assert_close(clean, conditional)
        torch.testing.assert_close(velocity, torch.tensor([[[4.0, 3.0]]]))

    def test_eta_one_without_clipping_or_momentum_is_exact_cfg(self):
        function = self.configure(apg_eta=1.0)
        x = torch.tensor([[[0.0, 1.0], [2.0, -1.0]]])
        conditional = torch.tensor([[[1.0, 0.0], [0.5, 2.0]]])
        unconditional = torch.tensor([[[0.0, 0.0], [-1.0, 0.5]]])
        expected = unconditional + 7.0 * (conditional - unconditional)
        for sigma in (1.0, 0.5, 0.01):
            with self.subTest(sigma=sigma):
                actual = self.evaluate(function, x, conditional, unconditional, sigma)
                torch.testing.assert_close(actual, expected)

    def test_cfg_one_ignores_nan_unconditional_prediction(self):
        function = self.configure(
            cfg=1.0, guidance_schedule="decay", min_guidance_scale=1.0,
        )
        x = torch.tensor([[[0.0, 1.0]]])
        conditional = torch.tensor([[[1.0, 0.0]]])
        unconditional = torch.full_like(conditional, float("nan"))
        actual = self.evaluate(function, x, conditional, unconditional, 0.5, cfg=1.0)
        torch.testing.assert_close(actual, conditional)

    def test_outside_interval_returns_conditional_prediction(self):
        function = self.configure(
            guidance_interval=-1.0, cfg_interval_start=0.2, cfg_interval_end=0.8,
        )
        x = torch.tensor([[[0.0, 1.0]]])
        conditional = torch.tensor([[[1.0, 1.0]]])
        unconditional = torch.tensor([[[0.0, 1.0]]])
        for sigma in (1.0, 0.01):
            with self.subTest(sigma=sigma):
                actual = self.evaluate(function, x, conditional, unconditional, sigma)
                torch.testing.assert_close(actual, conditional)
        guided = self.evaluate(function, x, conditional, unconditional, 0.5)
        torch.testing.assert_close(guided, torch.tensor([[[4.0, -2.0]]]))

    def test_zero_sigma_with_zero_update_stays_finite(self):
        function = self.configure(apg_norm_threshold=2.5)
        x = torch.tensor([[[0.0, 1.0]]])
        actual = self.evaluate(function, x, x, x, 0.0)
        self.assertTrue(torch.isfinite(actual).all())
        torch.testing.assert_close(actual, x)

    def test_momentum_and_clipping_use_clean_prediction_differences(self):
        function = self.configure(apg_momentum=-0.5, apg_norm_threshold=0.75,
                                  apg_eta=0.25)
        previous = [0.0, 0.0]
        for sigma, conditional, unconditional in (
            (0.8, [2.0, 1.0], [-1.0, 2.0]),
            (0.4, [1.0, 3.0], [-2.0, 1.0]),
        ):
            difference = [c - u - 0.5 * p
                          for c, u, p in zip(conditional, unconditional, previous)]
            previous = difference
            norm = math.sqrt(sum(value * value for value in difference))
            clipped = [value * min(1.0, 0.75 / norm) for value in difference]
            base_norm_sq = sum(value * value for value in conditional)
            coefficient = sum(d * c for d, c in zip(clipped, conditional)) / base_norm_sq
            parallel = [coefficient * value for value in conditional]
            expected = [c + 3.0 * (d - p + 0.25 * p)
                        for c, d, p in zip(conditional, clipped, parallel)]
            actual = self.evaluate(
                function, torch.tensor([[[0.0, 1.0]]]),
                torch.tensor([[conditional]]), torch.tensor([[unconditional]]),
                sigma, cfg=4.0,
            )
            torch.testing.assert_close(actual, torch.tensor([[expected]]))

    def test_jk_uses_clean_apg_for_predictor_corrector_and_final_step(self):
        function = self.configure(apg_eta=1.0)
        sampling.load_definitions(sampling.NODE_PATH, {"sample_jkass_quality"}, self.harness.ns)
        self.harness.ns["model_trange"] = lambda count, **kwargs: range(count)
        original = self.harness.ns["apg_guidance"]
        inputs = []

        def capture(conditional, unconditional, *args, **kwargs):
            inputs.append((conditional.clone(), unconditional.clone()))
            return original(conditional, unconditional, *args, **kwargs)

        self.harness.ns["apg_guidance"] = capture
        conditional = torch.tensor([[[1.0, 0.0]]])
        unconditional = torch.zeros_like(conditional)
        evaluated_sigmas = []

        def evaluate(x, sigma):
            evaluated_sigmas.append(float(sigma[0]))
            return self.evaluate(function, x, conditional, unconditional, float(sigma[0]))

        sigmas = torch.linspace(1.0, 0.0, 31)
        result = self.harness.ns["sample_jkass_quality"](
            evaluate, torch.zeros_like(conditional), sigmas, disable=True,
        )
        self.assertEqual(len(inputs), 59)
        self.assertEqual(evaluated_sigmas[0], 1.0)
        self.assertEqual(evaluated_sigmas[-1], float(sigmas[-2]))
        self.assertNotIn(0.0, evaluated_sigmas)
        for index in range(1, 59, 2):
            self.assertEqual(evaluated_sigmas[index], evaluated_sigmas[index + 1])
        for actual_conditional, actual_unconditional in inputs:
            torch.testing.assert_close(actual_conditional, conditional)
            torch.testing.assert_close(actual_unconditional, unconditional)
        torch.testing.assert_close(result, 7.0 * conditional)


if __name__ == "__main__":
    unittest.main()
