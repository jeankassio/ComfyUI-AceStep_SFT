"""Cascade wiring and path validation, without loading ComfyUI or checkpoints."""
import ast
import ntpath
from pathlib import Path
from types import SimpleNamespace
import unittest


class CascadeLoaderTests(unittest.TestCase):
    def setUp(self):
        self.resolved = []
        self.calls = []
        names = ["acestep_sft_xl.safetensors", "acestep_base_xl.safetensors", "acestep_turbo_xl.safetensors"]
        tree = ast.parse((Path(__file__).resolve().parents[1] / "nodes.py").read_text(encoding="utf-8"))
        selected = [node for node in tree.body if getattr(node, "name", None) in {
            "AceStepSFTModelLoader", "AceStepSFTCascadeModelLoader",
        }]
        def resolve(category, name):
            self.resolved.append((category, name))
            return f"/models/{category}/{name}"
        self.ns = {
            "ntpath": ntpath,
            "folder_paths": SimpleNamespace(get_filename_list=lambda category: names,
                                            get_full_path_or_raise=resolve),
            "load_cascade_models": lambda paths, names, **options: self.calls.append((paths, names, options)) or ("sft", "base", "turbo"),
        }
        exec(compile(ast.Module(body=selected, type_ignores=[]), "nodes.py", "exec"), self.ns)
        self.node = self.ns["AceStepSFTCascadeModelLoader"]()
        self.aux_calls = []
        self.node._load_text_and_vae = lambda *args: self.aux_calls.append(args) or ("clip", "vae")
        self.args = dict(sft_model=names[0], base_model=names[1], turbo_model=names[2],
                         text_encoder_1="qwen_small.safetensors", text_encoder_2="qwen_lm.safetensors", vae_name="audio_vae.safetensors")

    def test_outputs_keep_three_checkpoint_identities_and_one_auxiliary_pair(self):
        self.assertEqual(self.node.load_cascade(**self.args), ("sft", "base", "turbo", "clip", "vae"))
        self.assertEqual(self.calls[0][1], tuple(self.args[key] for key in ("sft_model", "base_model", "turbo_model")))
        self.assertEqual(self.calls[0][2], {"cache_runtime_weights": True})
        self.assertEqual(len(self.resolved), 3)
        self.assertEqual(self.aux_calls, [("qwen_small.safetensors", "qwen_lm.safetensors", "audio_vae.safetensors")])
        self.assertEqual(self.node.RETURN_TYPES, ("MODEL", "MODEL", "MODEL", "CLIP", "VAE"))

    def test_absolute_drive_unc_and_parent_paths_rejected_before_loading(self):
        for field in self.args:
            for value in ("C:\\model.safetensors", "C:model.safetensors", "\\\\host\\share\\model.safetensors", "/tmp/model.safetensors", "folder/../model.safetensors"):
                with self.subTest(field=field, value=value):
                    args = dict(self.args, **{field: value})
                    with self.assertRaises(ValueError):
                        self.node.load_cascade(**args)
        self.assertFalse(self.resolved)
        self.assertFalse(self.calls)
        self.assertFalse(self.aux_calls)

    def test_old_loader_schema_is_unchanged(self):
        self.assertEqual(list(self.ns["AceStepSFTModelLoader"].INPUT_TYPES()["required"]),
                         ["diffusion_model", "text_encoder_1", "text_encoder_2", "vae_name"])
        self.assertEqual(list(self.node.INPUT_TYPES()["required"]), list(self.args))


if __name__ == "__main__":
    unittest.main()
