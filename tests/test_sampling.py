"""CPU regressions for guidance; execute node code and installed Comfy samplers.

AST loading avoids initializing CUDA, downloading weights or starting ComfyUI.
Run with the portable Python: python -m unittest discover -s tests -p test_sampling.py
"""
import ast
import copy
import inspect
import math
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
import torch.nn.functional as F


NODE_PATH = Path(__file__).resolve().parents[1] / "nodes.py"
COMFY_ROOT = NODE_PATH.parents[2]


def load_definitions(path, names, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    selected = [item for item in tree.body if getattr(item, "name", None) in names]
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)


class FakeModel:
    def __init__(self, name="sft"):
        self.model_options = {"acestep_checkpoint_name": name}
        self.model = SimpleNamespace(model_config=None)
        self.load_device = "cpu"

    def clone(self):
        result = FakeModel()
        result.model_options = self.model_options.copy()
        return result

    def add_object_patch(self, name, value):
        pass

    def set_model_sampler_cfg_function(self, function, disable_cfg1_optimization=False):
        self.model_options["sampler_cfg_function"] = function
        self.model_options["disable_cfg1_optimization"] = disable_cfg1_optimization

    def set_model_sampler_calc_cond_batch_function(self, function):
        self.model_options["sampler_calc_cond_batch_function"] = function


class FakeSampling:
    def __init__(self, config):
        pass

    def set_parameters(self, **kwargs):
        self.shift = kwargs["shift"]

    def percent_to_sigma(self, percent):
        time = 1.0 - percent
        return self.shift * time / (1.0 + (self.shift - 1.0) * time)


class TensorCondition:
    def __init__(self, cond):
        self.cond = cond

    def _copy_with(self, value):
        return TensorCondition(value)


class SamplingTests(unittest.TestCase):
    def setUp(self):
        self.trace = []
        self.schedule_override = None
        self.unconditional_nan = False
        self.sampled_model = None
        self.cleanup_models = []
        self.ns = {"torch": torch, "F": F, "math": math, "contextmanager": contextmanager}
        names = {
            "MomentumBuffer", "_project", "apg_guidance", "_cos_sim", "_perpendicular",
            "adg_guidance", "_guidance_step_position", "_guidance_interval_context",
            "_clone_conditioning_value", "_clone_conditioning", "_clone_runtime_conditioning",
            "_clone_processed_cond_value", "_zero_conditioning_value", "_build_text_only_conditioning",
            "_build_processed_text_only_conditioning", "_erg_tau_from_scale", "_apply_omega_scale",
            "_rescale_guidance_output",
            "_resolve_sampler_for_infer_method", "_is_audio_payload", "_is_latent_payload",
            "_match_batch_indices",
            "_acestep_sampling_model",
            "_retime_lm_codes_conditioning", "_retime_cover_conditioning",
            "_select_cover_conditioning_branch", "_sigma_to_percent",
            "AceStepSFTGenerate",
        }
        load_definitions(NODE_PATH, names, self.ns)
        self.sampler_ns = {
            "torch": torch, "trange": lambda count, **kw: range(count),
            "to_d": lambda x, sigma, denoised: (x - denoised) / sigma,
        }
        load_definitions(
            COMFY_ROOT / "comfy" / "k_diffusion" / "sampling.py",
            {"sample_euler", "sample_heun", "sample_dpm_2"}, self.sampler_ns,
        )
        comfy = SimpleNamespace(
            model_sampling=SimpleNamespace(ModelSamplingDiscreteFlow=FakeSampling, CONST=type("CONST", (), {})),
            model_management=SimpleNamespace(intermediate_device=lambda: "cpu"),
            sample=SimpleNamespace(
                fix_empty_latent_channels=lambda model, latent, ratio: latent,
                prepare_noise=lambda latent, seed, inds: torch.zeros_like(latent),
                sample=self.sample,
            ),
            samplers=SimpleNamespace(calc_cond_batch=self.calc_cond_batch),
            utils=SimpleNamespace(PROGRESS_BAR_ENABLED=False),
        )
        self.ns.update({
            "comfy": comfy,
            "latent_preview": SimpleNamespace(prepare_callback=lambda *args: None),
            "_validate_acestep_task": lambda task: task,
            "_get_source_latent_metadata": lambda source: None,
            "_match_noise_mask_length": lambda mask, length: mask,
            "_clear_acestep_native_task_patches": self.cleanup_models.append,
            "_ensure_acestep_silence_latent": lambda model: None,
            "_clear_acestep_silence_latent_patch": lambda model: None,
            "clear_conditioning_precompute": lambda model: None,
            "clear_voice_profile": lambda model: None,
            "install_conditioning_precompute": lambda model: None,
            "_ensure_acestep_timbre_cls": lambda model: None,
            "_clear_acestep_timbre_cls_patch": lambda model: None,
            "_clear_acestep_reference_attention": lambda model: None,
            "_ensure_acestep_reference_attention": lambda model: None,
            "_clear_acestep_reference_timbre_repeats": lambda model: None,
            "_ensure_acestep_reference_timbre_repeats": lambda model, repeats: None,
            "_calculate_sampling_sigmas": lambda sampling, scheduler, sampler, steps: torch.linspace(1, 0, steps + 1),
        })

    def calc_cond_batch(self, model, conds, x, sigma, model_options):
        outputs = []
        for cond in conds:
            if cond is None:
                outputs.append(torch.zeros_like(x))
            elif cond == "unconditional":
                outputs.append(torch.full_like(x, float("nan") if self.unconditional_nan else 1.0))
            else:
                lyric = cond[0]["model_conds"]["lyric_embed"].cond
                outputs.append(torch.full_like(x, 4.0 if torch.count_nonzero(lyric) == 0 else 2.0))
        return outputs

    def sample(self, model, noise, steps, cfg, sampler_name, scheduler,
               positive, negative, latent_image, **kwargs):
        self.sampled_model = model
        self.sampled_positive = positive
        self.sampled_negative = negative
        sigmas = self.schedule_override
        if sigmas is None:
            sigmas = kwargs["sigmas"]
        if sigmas is None:
            sigmas = torch.linspace(1, 0, steps + 1)
        model_options = model.model_options.copy()
        model_options["transformer_options"] = {"sample_sigmas": sigmas}
        cond = [{"model_conds": {"lyric_embed": TensorCondition(torch.ones(1, 2, 3))}}]

        def evaluate(x, sigma, **extra):
            cond_args = {
                "model": model, "conds": [cond, "unconditional"],
                "input": x, "sigma": sigma, "model_options": model_options,
            }
            batch_function = model_options.get("sampler_calc_cond_batch_function")
            if batch_function:
                conditional, unconditional = batch_function(cond_args)
            else:
                conditional, unconditional = self.calc_cond_batch(
                    model, [cond, "unconditional"], x, sigma, model_options
                )
            cfg_function = model_options.get("sampler_cfg_function")
            if cfg_function:
                result = x - cfg_function({
                    "cond_denoised": conditional, "uncond_denoised": unconditional,
                    "cond_scale": cfg, "input": x, "sigma": sigma,
                    "model_options": model_options,
                })
            else:
                result = unconditional + (conditional - unconditional) * cfg
            self.trace.append((float(sigma[0]), result.clone()))
            return result

        return self.sampler_ns["sample_" + sampler_name](
            evaluate, noise.clone(), sigmas, disable=True
        )

    def generate(self, **overrides):
        positive = [[torch.ones(1, 2, 3), {"conditioning_lyrics": torch.ones(1, 2, 3)}]]
        options = dict(
            model=FakeModel(), positive=positive, negative=copy.deepcopy(positive),
            seed=0, steps=50, cfg=7.0, sampler_name="euler", scheduler="normal",
            denoise=1.0, duration=0.4, infer_method="ode", guidance_mode="standard_cfg",
            guidance_interval=1.0, guidance_schedule="decay", min_guidance_scale=1.0,
        )
        options.update(overrides)
        return self.ns["AceStepSFTGenerate"]().generate(**options)

    def assert_guidance_scales(self, scales):
        self.assertEqual(len(self.trace), len(scales))
        for (_, denoised), scale in zip(self.trace, scales):
            torch.testing.assert_close(denoised, torch.full_like(denoised, scale + 1.0), atol=2e-5, rtol=1e-5)

    def test_standard_cfg_decays_from_seven_to_one(self):
        self.generate()
        self.assert_guidance_scales([7.0 - 6.0 * index / 49 for index in range(50)])

    def test_guidance_schedule_modes_have_exact_endpoints(self):
        for schedule, expected in (
            ("normal", [7.0, 7.0, 7.0]),
            ("decay", [7.0, 4.0, 1.0]),
            ("ramp_up", [1.0, 4.0, 7.0]),
        ):
            with self.subTest(schedule=schedule):
                self.trace.clear()
                self.generate(steps=3, guidance_schedule=schedule)
                self.assert_guidance_scales(expected)

    def test_ramp_up_uses_shortened_runtime_schedule(self):
        self.schedule_override = torch.tensor([1.0, 0.7, 0.25, 0.0])
        self.generate(steps=50, guidance_schedule="ramp_up")
        self.assert_guidance_scales([1.0, 4.0, 7.0])

    def test_ramp_up_only_changes_scale_inside_guidance_interval(self):
        self.generate(steps=8, guidance_interval=0.5, guidance_schedule="ramp_up")
        self.assert_guidance_scales([1, 1, 1, 3, 5, 7, 1, 1])

    def test_ramp_up_reverses_split_text_and_lyric_scales(self):
        self.generate(steps=3, guidance_schedule="ramp_up",
                      guidance_scale_text=7.0, guidance_scale_lyric=5.0)
        for (_, actual), expected in zip(self.trace, [2.0, 7.0, 12.0]):
            torch.testing.assert_close(actual, torch.full_like(actual, expected))

    def test_jkass_ramp_up_uses_actual_sigmas_for_both_evaluations(self):
        load_definitions(NODE_PATH, {"sample_jkass_quality", "_gradio_guidance_interval",
                                     "_calculate_acestep_official_sigmas"}, self.ns)
        self.ns["model_trange"] = lambda count, **kwargs: range(count)
        self.ns["comfy"].samplers.KSAMPLER = lambda function, options: (function, options)

        def sample_custom(model, noise, cfg, sampler, sigmas, positive, negative, latent, **kwargs):
            self.assertIs(sampler[0], self.ns["sample_jkass_quality"])
            self.sampler_ns["sample_jkass_quality"] = sampler[0]
            return self.sample(model, noise, len(sigmas) - 1, cfg, "jkass_quality", "normal",
                               positive, negative, latent, sigmas=sigmas)

        self.ns["comfy"].sample.sample_custom = sample_custom
        for scheduler in ("normal", "ace_step"):
            with self.subTest(scheduler=scheduler):
                self.trace.clear()
                self.generate(steps=3, sampler_name="jkass_quality", scheduler=scheduler,
                              guidance_schedule="ramp_up")
                self.assert_guidance_scales([1.0, 4.0, 4.0, 7.0, 7.0])

    def test_guidance_schedule_is_a_combo_in_the_same_widget_position(self):
        tree = ast.parse(NODE_PATH.read_text(encoding="utf-8"))
        modes = next(item.value for item in tree.body if isinstance(item, ast.Assign)
                     and any(isinstance(target, ast.Name) and target.id == "GUIDANCE_MODES"
                             for target in item.targets))
        self.ns.update(GUIDANCE_MODES=ast.literal_eval(modes), ACESTEP_SFT_TASK_TYPE="ACESTEP_SFT_TASK")
        self.ns["comfy"].samplers.KSampler = SimpleNamespace(SAMPLERS=["euler"], SCHEDULERS=["normal"])
        inputs = self.ns["AceStepSFTGenerate"].INPUT_TYPES()
        optional = inputs["optional"]
        choices, settings = optional["guidance_schedule"]
        self.assertEqual(choices, ["normal", "decay", "ramp_up"])
        self.assertEqual(settings["default"], "normal")
        position = list(optional).index("guidance_schedule")
        self.assertEqual(list(optional)[position - 1], "guidance_interval")
        self.assertEqual(list(optional)[position + 1], "min_guidance_scale")
        parameter = inspect.signature(self.ns["AceStepSFTGenerate"].generate).parameters["guidance_schedule"]
        self.assertEqual(parameter.default, "normal")
        self.assertEqual(inputs["required"]["guidance_mode"][0],
                         ["apg", "adg", "standard_cfg", "apg_denoised", "apg_denoised_joint"])

    def test_generate_retimes_lm_handoff_for_both_cfg_branches(self):
        embedding = torch.ones(1, 2, 3)
        positive = [
            [embedding, {"audio_codes": [[1, 2]], "end_percent": 0.5,
                         "acestep_sft_lm_codes_strength": 0.5}],
            [embedding, {"start_percent": 0.5,
                         "acestep_sft_lm_codes_strength": 0.5}],
        ]
        self.generate(steps=4, positive=positive, negative=copy.deepcopy(positive))
        # The real sample schedule is [1, .75, .5, .25, 0]. Two code
        # steps end at the midpoint .625, irrespective of shifted percent.
        sampling = FakeSampling(None)
        sampling.set_parameters(shift=3.0)
        for branches in (self.sampled_positive, self.sampled_negative):
            cutoff = branches[0][1]["end_percent"]
            self.assertAlmostEqual(sampling.percent_to_sigma(cutoff), 0.625)
            self.assertEqual(cutoff, branches[1][1]["start_percent"])
            self.assertIn("audio_codes", branches[0][1])
            self.assertNotIn("audio_codes", branches[1][1])
        self.assertEqual(positive[0][1]["end_percent"], 0.5)

    def test_constant_standard_cfg_remains_native(self):
        self.generate(steps=6, guidance_schedule="normal")
        self.assertNotIn("sampler_cfg_function", self.sampled_model.model_options)
        self.assert_guidance_scales([7.0] * 6)

    def test_real_dpm2_intermediate_evaluations_do_not_advance_steps(self):
        self.generate(steps=6, sampler_name="dpm_2")
        main_passes = self.trace[::2]
        self.assertEqual(len(main_passes), 6)
        for index, (_, denoised) in enumerate(main_passes):
            torch.testing.assert_close(denoised, torch.full_like(denoised, 8.0 - 6.0 * index / 5))

    def test_real_heun_repeated_sigmas_have_the_same_scale(self):
        self.generate(steps=6, sampler_name="heun")
        for previous, current in zip(self.trace, self.trace[1:]):
            if previous[0] == current[0]:
                torch.testing.assert_close(previous[1], current[1])
        torch.testing.assert_close(self.trace[-1][1], torch.full_like(self.trace[-1][1], 2.0))

    def test_shortened_runtime_schedule_still_reaches_final_cfg(self):
        self.schedule_override = torch.tensor([1.0, 0.7, 0.25, 0.0])
        self.generate(steps=50)
        self.assert_guidance_scales([7.0, 4.0, 1.0])

    def test_manual_interval_uses_exact_endpoints(self):
        self.generate(guidance_interval=-1.0, cfg_interval_start=0.2, cfg_interval_end=0.8)
        expected = [7.0 - 6.0 * (i - 10) / 29 if 10 <= i <= 39 else 1.0 for i in range(50)]
        self.assert_guidance_scales(expected)

    def test_official_interval_without_decay_applies_only_inside(self):
        self.generate(steps=8, guidance_interval=0.5, guidance_schedule="normal")
        self.assert_guidance_scales([1, 1, 7, 7, 7, 7, 1, 1])

    def test_split_text_and_lyrics_decay_to_conditional(self):
        self.generate(steps=3, guidance_scale_text=7.0, guidance_scale_lyric=5.0)
        for (_, actual), expected in zip(self.trace, [12.0, 7.0, 2.0]):
            torch.testing.assert_close(actual, torch.full_like(actual, expected))

    def test_cfg_one_does_not_mix_a_nan_unconditional_prediction(self):
        self.unconditional_nan = True
        self.generate(steps=3, cfg=1.0)
        self.assert_guidance_scales([1.0] * 3)

    def test_apg_and_adg_follow_the_same_curve_and_skip_cfg_one(self):
        for mode in ("apg", "adg"):
            with self.subTest(mode=mode):
                self.trace.clear()
                scales = []
                function_name = mode + "_guidance"
                original = self.ns[function_name]

                def capture(*args, **kwargs):
                    scales.append(args[4] if mode == "adg" else args[2])
                    return original(*args, **kwargs)

                self.ns[function_name] = capture
                self.generate(steps=3, guidance_mode=mode)
                self.assertEqual(scales, [7.0, 4.0])
                torch.testing.assert_close(self.trace[-1][1], torch.full_like(self.trace[-1][1], 2.0))
                self.ns[function_name] = original

    def test_apg_and_adg_ramp_up_starts_with_conditional_prediction(self):
        for mode in ("apg", "adg", "apg_denoised", "apg_denoised_joint"):
            with self.subTest(mode=mode):
                self.trace.clear()
                scales = []
                function_name = "adg_guidance" if mode == "adg" else "apg_guidance"
                original = self.ns[function_name]

                def capture(*args, **kwargs):
                    scales.append(args[4] if mode == "adg" else args[2])
                    return original(*args, **kwargs)

                self.ns[function_name] = capture
                self.generate(steps=3, guidance_mode=mode, guidance_schedule="ramp_up")
                self.assertEqual(scales, [4.0, 7.0])
                torch.testing.assert_close(self.trace[0][1], torch.full_like(self.trace[0][1], 2.0))
                self.ns[function_name] = original

    def test_single_step_outside_manual_interval_has_no_guidance(self):
        self.generate(steps=1, guidance_interval=-1.0, cfg_interval_start=0.2, cfg_interval_end=0.8)
        self.assert_guidance_scales([1.0])

    def test_turbo_filename_never_overrides_requested_cfg(self):
        self.generate(steps=3, model=FakeModel("turbo_custom.safetensors"))
        self.assert_guidance_scales([7.0, 4.0, 1.0])

    def test_model_output_preserves_input_and_external_hooks(self):
        model = FakeModel()
        external_hook = lambda args: args["input"] - 3.0
        model.model_options["sampler_cfg_function"] = external_hook
        original_options = model.model_options.copy()
        output_model = self.generate(model=model, steps=3)[0]
        self.assertIs(output_model, model)
        self.assertEqual(model.model_options, original_options)
        self.assertIs(output_model.model_options["sampler_cfg_function"], external_hook)

    def test_chained_generation_does_not_keep_previous_guidance(self):
        model = self.generate(steps=3, guidance_mode="apg")[0]
        self.trace.clear()
        self.generate(model=model, steps=3, guidance_schedule="normal")
        self.assertNotIn("sampler_cfg_function", model.model_options)
        self.assert_guidance_scales([7.0] * 3)

    def test_heun_repeated_timestep_does_not_advance_apg_momentum(self):
        history = []
        original = self.ns["apg_guidance"]

        def capture(*args, **kwargs):
            result = original(*args, **kwargs)
            history.append(kwargs["momentum_buffer"].running_average.clone())
            return result

        self.ns["apg_guidance"] = capture
        self.generate(steps=6, sampler_name="heun", guidance_mode="apg",
                      guidance_schedule="normal")
        repeated = 0
        for index in range(1, len(self.trace)):
            if self.trace[index - 1][0] == self.trace[index][0]:
                repeated += 1
                torch.testing.assert_close(history[index - 1], history[index])
        self.assertGreater(repeated, 0)

    def test_euler_apg_momentum_keeps_official_recurrence(self):
        expected = 0
        original = self.ns["apg_guidance"]

        def capture(*args, **kwargs):
            nonlocal expected
            expected = args[0] - args[1] - 0.75 * expected
            result = original(*args, **kwargs)
            torch.testing.assert_close(kwargs["momentum_buffer"].running_average,
                                       expected)
            return result

        self.ns["apg_guidance"] = capture
        self.generate(steps=6, guidance_mode="apg", guidance_schedule="normal")

    def test_apg_repeated_timestep_recomputes_changed_prediction(self):
        buffer = self.ns["MomentumBuffer"](momentum=-0.75)
        buffer.update(torch.tensor([4.0]), timestep=1.0)
        buffer.update(torch.tensor([8.0]), timestep=0.8)
        torch.testing.assert_close(buffer.running_average, torch.tensor([5.0]))
        buffer.update(torch.tensor([9.0]), timestep=0.8)
        torch.testing.assert_close(buffer.running_average, torch.tensor([6.0]))
        buffer.update(torch.tensor([10.0]), timestep=0.6)
        torch.testing.assert_close(buffer.running_average, torch.tensor([5.5]))

    def test_apg_without_timestep_keeps_official_recurrence(self):
        buffer = self.ns["MomentumBuffer"](momentum=-0.75)
        expected = 0
        for value in (4.0, 8.0, 9.0, 10.0):
            expected = value - 0.75 * expected
            buffer.update(torch.tensor([value]))
            torch.testing.assert_close(buffer.running_average, torch.tensor([expected]))

    def test_sampling_error_cleans_the_clone_without_changing_input(self):
        model = FakeModel()
        original_options = model.model_options.copy()

        def fail_sampling(cloned_model, *args, **kwargs):
            self.sampled_model = cloned_model
            raise RuntimeError("sampling failed")

        self.ns["comfy"].sample.sample = fail_sampling
        with self.assertRaisesRegex(RuntimeError, "sampling failed"):
            self.generate(model=model)
        self.assertIsNot(self.sampled_model, model)
        self.assertEqual(self.cleanup_models, [self.sampled_model, self.sampled_model])
        self.assertEqual(model.model_options, original_options)

    def test_configuration_error_also_cleans_the_clone(self):
        model = FakeModel()

        class BrokenSampling(FakeSampling):
            def set_parameters(self, **kwargs):
                raise RuntimeError("configuration failed")

        self.ns["comfy"].model_sampling.ModelSamplingDiscreteFlow = BrokenSampling
        with self.assertRaisesRegex(RuntimeError, "configuration failed"):
            self.generate(model=model)
        self.assertEqual(len(self.cleanup_models), 2)
        self.assertIs(self.cleanup_models[0], self.cleanup_models[1])
        self.assertIsNot(self.cleanup_models[0], model)
        self.assertEqual(model.model_options, {"acestep_checkpoint_name": "sft"})

    def test_invalid_shift_is_rejected_before_sampling(self):
        for shift in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(shift=shift), self.assertRaisesRegex(ValueError, "shift"):
                self.generate(shift=shift)
        self.assertEqual(self.trace, [])

    def test_invalid_guidance_is_rejected_before_sampling(self):
        for name, value in (
            ("apg_momentum", 2.5), ("apg_momentum", float("nan")),
            ("apg_eta", float("inf")), ("apg_norm_threshold", -1.0),
            ("guidance_interval", 2.0),
            ("min_guidance_scale", -1.0), ("cfg_interval_start", -0.1),
            ("cfg_interval_end", 3.5),
        ):
            with self.subTest(name=name, value=value), self.assertRaisesRegex(ValueError, name):
                self.generate(**{name: value})
        self.assertEqual(self.trace, [])

    def test_invalid_guidance_schedule_is_rejected_before_sampling(self):
        for schedule in ("invalid", 0.0, -1.0, None):
            with self.subTest(schedule=schedule), self.assertRaisesRegex(ValueError, "guidance_schedule"):
                self.generate(guidance_schedule=schedule)
        self.assertEqual(self.trace, [])

    def test_adg_noncontiguous_batch_matches_individual_results(self):
        generator = torch.Generator().manual_seed(22)
        tensors = [torch.randn(2, 64, 25, generator=generator).movedim(1, -1) for _ in range(3)]
        actual = self.ns["adg_guidance"](*tensors, 0.5, 7.0)
        expected = torch.cat([
            self.ns["adg_guidance"](*(tensor[i:i + 1] for tensor in tensors), 0.5, 7.0)
            for i in range(2)
        ])
        self.assertTrue(torch.isfinite(actual).all())
        torch.testing.assert_close(actual, expected)

    def test_apg_temporal_axis_matches_btc_reference(self):
        generator = torch.Generator().manual_seed(23)
        cond, uncond = [torch.randn(2, 64, 25, generator=generator) for _ in range(2)]
        result = self.ns["apg_guidance"](cond, uncond, 7.0, dims=[-1])
        btc_result = self.ns["apg_guidance"](cond.movedim(1, -1), uncond.movedim(1, -1), 7.0, dims=[1])
        torch.testing.assert_close(result, btc_result.movedim(-1, 1))

    def test_step_position_handles_repeated_queries_and_terminal_zero(self):
        position = self.ns["_guidance_step_position"]
        schedule = [1.0, 0.8, 0.4, 0.0]
        self.assertEqual([position(s, schedule) for s in [1.0, 0.9, 0.8, 0.8, 0.6, 0.4, 0.0]],
                         [0.0, 0.5, 1.0, 1.0, 1.5, 2.0, 2.0])


if __name__ == "__main__":
    unittest.main()
