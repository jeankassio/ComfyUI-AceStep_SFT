"""CPU regressions using the installed native ACE forward/prepare methods."""

import ast
import functools
import math
from pathlib import Path
from types import MethodType, SimpleNamespace
import unittest

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = ROOT.parents[1]


class Condition:
    def __init__(self, value):
        self.cond = value


def assign(model, path, value):
    parent, _, name = path.rpartition(".")
    owner = model.get_submodule(parent) if parent else model
    setattr(owner, name, value)


def read_attr(model, path):
    value = model
    for name in path.split("."):
        value = getattr(value, name)
    return value


class Patcher:
    def __init__(self, model):
        self.model = model
        self.object_patches = {}
        self.object_patches_backup = {}
        self.model_options = {"transformer_options": {}}
        self.hook_patches = {}
        self.forced_hooks = None
        self.current_hooks = None
        self.injections = {}
        self.wrappers = {}
        self.callbacks = {}
        self.weight_wrapper_patches = {}
        self.unpatches = 0

    def get_model_object(self, path):
        if path in self.object_patches:
            return self.object_patches[path]
        if path in self.object_patches_backup:
            return self.object_patches_backup[path]
        return read_attr(self.model, path)

    def add_object_patch(self, path, value):
        self.object_patches[path] = value

    def patch_model(self, load_weights=False):
        for path, value in self.object_patches.items():
            self.object_patches_backup.setdefault(path, read_attr(self.model, path))
            assign(self.model, path, value)

    def unpatch_model(self, unpatch_weights=False):
        self.unpatches += 1
        for path, value in self.object_patches_backup.items():
            assign(self.model, path, value)
        self.object_patches_backup.clear()


class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, text, text_mask, lyrics, lyric_mask, reference, order):
        self.calls += 1
        timbre = reference.mean(dim=1, keepdim=True)
        return torch.cat((text, lyrics, timbre), dim=1), None


class Tokenizer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0
        self.quantizer = SimpleNamespace(get_output_from_indices=self.decode)

    def decode(self, codes, dtype):
        self.calls += 1
        return F.one_hot(codes % 4, num_classes=4).to(dtype)

    def tokenize(self, reference):
        self.calls += 1
        return reference[:, ::5, :], None


class Detokenizer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, hints):
        self.calls += 1
        return hints.repeat_interleave(5, dim=1)


class Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0
        self.fail = False

    def forward(self, hidden_states, timestep, timestep_r, attention_mask,
                encoder_hidden_states, encoder_attention_mask, context_latents):
        self.calls += 1
        if self.fail:
            raise RuntimeError("decoder failure")
        return (
            hidden_states * .25 + context_latents[..., :4] * .5
            + encoder_hidden_states.mean(dim=(1, 2))[:, None, None]
            + timestep[:, None, None] * .125
        )


def load_api():
    fake_comfy = SimpleNamespace(
        conds=SimpleNamespace(CONDRegular=Condition),
        model_management=SimpleNamespace(cast_to_device=lambda value, device, dtype: value.to(device=device, dtype=dtype)),
    )
    namespace = dict(torch=torch, comfy=fake_comfy, functools=functools, MethodType=MethodType)
    tree = ast.parse((ROOT / "conditioning_precompute.py").read_text(encoding="utf-8"))
    selected = [node for node in tree.body if isinstance(node, (ast.Assign, ast.FunctionDef))]
    exec(compile(ast.Module(body=selected, type_ignores=[]), "conditioning_precompute.py", "exec"), namespace)

    tree = ast.parse((COMFY_ROOT / "comfy/ldm/ace/ace_step15.py").read_text(encoding="utf-8"))
    original = next(node for node in tree.body if getattr(node, "name", None) == "AceStepConditionGenerationModel")
    methods = [node for node in original.body if getattr(node, "name", None) in {"forward", "prepare_condition"}]
    tiny_class = ast.ClassDef(name="Native", bases=[ast.Attribute(value=ast.Name(id="torch", ctx=ast.Load()), attr="nn", ctx=ast.Load())], keywords=[], body=methods, decorator_list=[])
    # Use the native methods with tiny independent conditioning/decoder modules.
    tiny_class.bases = [ast.Attribute(value=tiny_class.bases[0], attr="Module", ctx=ast.Load())]
    native_namespace = dict(torch=torch, math=math)
    native_tree = ast.fix_missing_locations(ast.Module(body=[tiny_class], type_ignores=[]))
    exec(compile(native_tree, "ace_step15.py", "exec"), native_namespace)
    return SimpleNamespace(**namespace), native_namespace["Native"]


class ConditioningPrecomputeTests(unittest.TestCase):
    def setUp(self):
        self.api, native = load_api()
        self.diffusion = native()
        self.diffusion.encoder = Encoder()
        self.diffusion.tokenizer = Tokenizer()
        self.diffusion.detokenizer = Detokenizer()
        self.diffusion.decoder = Decoder()
        self.diffusion.null_condition_emb = torch.nn.Parameter(torch.full((1, 1, 4), -.375))
        self.model = torch.nn.Module()
        self.model.diffusion_model = self.diffusion
        self.model.get_dtype_inference = lambda: torch.float32
        self.model.extra_conds = self.extra
        self.patcher = Patcher(self.model)
        self.args = dict(
            noise=torch.zeros(1, 4, 10), device=torch.device("cpu"),
            cross_attn=torch.arange(12, dtype=torch.float64).reshape(1, 3, 4) / 10,
            conditioning_lyrics=torch.arange(8, dtype=torch.float64).reshape(1, 2, 4) / 20,
            reference=torch.arange(40, dtype=torch.float64).reshape(1, 4, 10) / 40,
            codes=torch.tensor([[1]], dtype=torch.long),
        )

    def extra(self, **kwargs):
        result = {
            "c_crossattn": Condition(kwargs["cross_attn"]),
            "lyric_embed": Condition(kwargs["conditioning_lyrics"]),
            "refer_audio": Condition(kwargs["reference"]),
        }
        if kwargs.get("codes") is not None:
            result["audio_codes"] = Condition(kwargs["codes"])
        if "is_covers" in kwargs:
            result["is_covers"] = Condition(kwargs["is_covers"])
        if kwargs.get("null", False):
            result["replace_with_null_embeds"] = Condition(True)
        return result

    def call(self, conditions, index=0, batch=1):
        value = torch.arange(batch * 40, dtype=torch.float32).reshape(batch, 4, 10) / 50 + index * .1
        kwargs = {}
        for key, condition in conditions.items():
            tensor = condition.cond
            if torch.is_tensor(tensor):
                tensor = tensor.to(dtype=None if tensor.dtype in (torch.int32, torch.int64) else torch.float32)
            kwargs[key] = tensor
        context = kwargs.pop("c_crossattn")
        return self.diffusion(value, torch.full((batch,), .7 - index * .1), context, **kwargs).detach().clone()

    def install(self):
        self.assertTrue(self.api.install_conditioning_precompute(self.patcher))
        self.patcher.patch_model()

    def test_preparation_preserves_full_native_forward_and_runs_only_once(self):
        original = self.extra(**self.args)
        expected = [self.call(original, index) for index in range(4)]
        self.assertEqual(self.diffusion.encoder.calls, 4)
        self.diffusion.encoder.calls = self.diffusion.detokenizer.calls = 0
        self.install()
        prepared = self.model.extra_conds(**self.args)
        self.assertEqual(self.diffusion.encoder.calls, 1)
        for index in range(4):
            torch.testing.assert_close(self.call(prepared, index), expected[index], rtol=0, atol=0)
        self.assertEqual(self.diffusion.encoder.calls, 1)
        self.assertEqual(self.diffusion.detokenizer.calls, 1)

    def test_null_and_positive_conditions_are_independent(self):
        original_positive = self.call(self.extra(**self.args))
        original_null = self.call(self.extra(**dict(self.args, null=True)))
        self.install()
        positive = self.model.extra_conds(**self.args)
        saved_positive = positive["acestep_precomputed_encoder"].cond.clone()
        negative = self.model.extra_conds(**dict(self.args, null=True))
        for _ in range(2):
            torch.testing.assert_close(self.call(negative), original_null, rtol=0, atol=0)
            torch.testing.assert_close(self.call(positive), original_positive, rtol=0, atol=0)
        torch.testing.assert_close(positive["acestep_precomputed_encoder"].cond, saved_positive, rtol=0, atol=0)

    def test_no_codes_and_all_python_cover_modes_match_native(self):
        for codes in (None, self.args["codes"]):
            for cover in (None, False, True):
                with self.subTest(codes=codes is not None, cover=cover):
                    args = dict(self.args, codes=codes, is_covers=cover)
                    expected = self.call(self.extra(**args))
                    self.install()
                    prepared = self.model.extra_conds(**args)
                    torch.testing.assert_close(self.call(prepared), expected, rtol=0, atol=0)
                    self.api.clear_conditioning_precompute(self.patcher)

    def test_batch_two_and_dynamic_cover_mode_fall_back(self):
        args = {key: value.repeat(2, 1, 1) if key in ("noise", "cross_attn", "conditioning_lyrics", "reference") else value for key, value in self.args.items()}
        args["codes"] = args["codes"].repeat(2, 1)
        expected = self.call(self.extra(**args), batch=2)
        self.install()
        prepared = self.model.extra_conds(**args)
        self.assertNotIn("acestep_precomputed_encoder", prepared)
        torch.testing.assert_close(self.call(prepared, batch=2), expected, rtol=0, atol=0)
        unsupported = self.model.extra_conds(**dict(self.args, is_covers=torch.tensor([False])))
        self.assertNotIn("acestep_precomputed_encoder", unsupported)

    def test_mismatched_prepared_context_uses_original_forward(self):
        expected = self.call(self.extra(**self.args))
        self.install()
        prepared = self.model.extra_conds(**self.args)
        prepared["acestep_precomputed_context"] = Condition(torch.zeros(1, 3, 8))
        calls = self.diffusion.encoder.calls
        torch.testing.assert_close(self.call(prepared), expected, rtol=0, atol=0)
        self.assertEqual(self.diffusion.encoder.calls, calls + 1)

    def test_clone_cleanup_and_reuse_do_not_retain_prepared_conditions(self):
        original_forward = self.diffusion.forward
        self.install()
        first = self.model.extra_conds(**self.args)
        clone = Patcher(self.model)
        clone.object_patches = self.patcher.object_patches.copy()
        # Native clones share the model and its object backup container.
        clone.object_patches_backup = self.patcher.object_patches_backup
        self.api.clear_conditioning_precompute(clone)
        self.assertEqual(self.diffusion.forward, original_forward)
        self.assertFalse(clone.object_patches)
        changed_args = dict(self.args, cross_attn=self.args["cross_attn"] + 2)
        expected = self.call(self.extra(**changed_args))
        self.assertTrue(self.api.install_conditioning_precompute(clone))
        clone.patch_model()
        second = self.model.extra_conds(**changed_args)
        self.assertFalse(torch.equal(first["acestep_precomputed_encoder"].cond,
                                     second["acestep_precomputed_encoder"].cond))
        torch.testing.assert_close(self.call(second), expected, rtol=0, atol=0)
        self.api.clear_conditioning_precompute(clone)
        self.assertEqual(self.diffusion.forward, original_forward)

    def test_cleanup_after_decoder_failure_restores_existing_patches(self):
        extra_original = self.model.extra_conds
        forward_original = self.diffusion.forward
        self.patcher.add_object_patch("extra_conds", extra_original)
        self.patcher.add_object_patch("unrelated", 27)
        self.model.unrelated = 0
        self.install()
        prepared = self.model.extra_conds(**self.args)
        self.diffusion.decoder.fail = True
        try:
            with self.assertRaisesRegex(RuntimeError, "decoder failure"):
                self.call(prepared)
        finally:
            self.api.clear_conditioning_precompute(self.patcher)
        self.assertIs(self.model.extra_conds, extra_original)
        self.assertEqual(self.diffusion.forward, forward_original)
        self.assertIs(self.patcher.object_patches["extra_conds"], extra_original)
        self.assertEqual(self.model.unrelated, 27)
        self.assertNotIn("diffusion_model.forward", self.patcher.object_patches)
        self.assertEqual(self.patcher.unpatches, 1)

    def test_foreign_forward_and_hook_paths_are_left_unchanged(self):
        original = self.diffusion.forward
        def foreign(*args, **kwargs):
            return original(*args, **kwargs) + 13
        self.patcher.add_object_patch("diffusion_model.forward", foreign)
        before = self.patcher.object_patches.copy()
        self.assertFalse(self.api.install_conditioning_precompute(self.patcher))
        self.assertEqual(self.patcher.object_patches, before)
        self.patcher.object_patches.clear()
        for owner, key, value in (
            (self.patcher.hook_patches, "external", object()),
            (self.patcher.model_options, "model_function_wrapper", foreign),
            (self.patcher.model_options["transformer_options"], "patches", {"external": [foreign]}),
        ):
            with self.subTest(key=key):
                owner[key] = value
                self.assertFalse(self.api.install_conditioning_precompute(self.patcher))
                self.assertFalse(self.patcher.object_patches)
                del owner[key]


if __name__ == "__main__":
    unittest.main()
