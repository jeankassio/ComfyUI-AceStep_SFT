"""
ComfyUI-AceStepSFT - AceStep 1.5 SFT All-in-One Generation Node

Provides an all-in-one node for AceStep 1.5 SFT music generation that matches
the quality of the official AceStep Gradio pipeline by using APG guidance.
"""

import json
import os

import folder_paths

# ---------------------------------------------------------------------------
# Register local Loras/ folder and auto-convert PEFT/DoRA LoRAs
# ---------------------------------------------------------------------------

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_LORAS_DIR = os.path.join(_THIS_DIR, "Loras")


def _convert_peft_to_comfyui(peft_dir, output_path):
    """Convert a PEFT/DoRA LoRA directory to a single ComfyUI .safetensors file.

    Remaps:
      - lora_A.weight  → lora_down.weight
      - lora_B.weight  → lora_up.weight
      - lora_magnitude_vector → dora_scale
    Injects per-layer .alpha scalars from adapter_config.json.
    """
    import torch
    from safetensors.torch import load_file, save_file

    config_path = os.path.join(peft_dir, "adapter_config.json")
    model_path = os.path.join(peft_dir, "adapter_model.safetensors")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    global_alpha = config.get("lora_alpha", 64)
    alpha_pattern = config.get("alpha_pattern", {})

    sd = load_file(model_path)
    new_sd = {}

    # Track which layer keys we've seen (for alpha injection)
    layer_keys_seen = set()

    for key, tensor in sd.items():
        new_key = key
        if ".lora_A.weight" in key:
            new_key = key.replace(".lora_A.weight", ".lora_down.weight")
        elif ".lora_B.weight" in key:
            new_key = key.replace(".lora_B.weight", ".lora_up.weight")
        elif ".lora_magnitude_vector" in key:
            new_key = key.replace(".lora_magnitude_vector", ".dora_scale")

        new_sd[new_key] = tensor

        # Extract layer prefix for alpha injection
        for suffix in (".lora_A.weight", ".lora_B.weight", ".lora_magnitude_vector"):
            if key.endswith(suffix):
                layer_prefix = key[: -len(suffix)]
                layer_keys_seen.add(layer_prefix)

    # Inject .alpha for each layer
    for layer_prefix in layer_keys_seen:
        # Strip "base_model.model." prefix to match alpha_pattern keys
        short_key = layer_prefix
        if short_key.startswith("base_model.model."):
            short_key = short_key[len("base_model.model."):]
        alpha_val = alpha_pattern.get(short_key, global_alpha)
        new_sd[f"{layer_prefix}.alpha"] = torch.tensor(float(alpha_val))
    # Unsqueeze 1D dora_scale to [N, 1] so ComfyUI's weight_decompose
    # broadcasts correctly: [N,1]/[N,1]=[N,1] instead of [1,N]/[N,1]=[N,N].
    for key in list(new_sd.keys()):
        if key.endswith(".dora_scale") and new_sd[key].dim() == 1:
            new_sd[key] = new_sd[key].unsqueeze(-1)
    save_file(new_sd, output_path)
    print(f"[AceStep SFT] Converted PEFT/DoRA → ComfyUI: {os.path.basename(output_path)}")


def _prepare_loras_folder():
    """Scan the Loras/ folder and prepare all LoRAs for ComfyUI.

    - PEFT directories (adapter_config.json + adapter_model.safetensors)
      are auto-converted to ComfyUI format .safetensors files.
    - Nested .safetensors files (from zip extraction) are copied to the
      Loras/ root so ComfyUI can find them.
    """
    if not os.path.isdir(_LORAS_DIR):
        os.makedirs(_LORAS_DIR, exist_ok=True)
        return

    for entry in os.listdir(_LORAS_DIR):
        entry_path = os.path.join(_LORAS_DIR, entry)
        if not os.path.isdir(entry_path):
            continue

        # --- PEFT/DoRA directory ---
        adapter_config = os.path.join(entry_path, "adapter_config.json")
        adapter_model = os.path.join(entry_path, "adapter_model.safetensors")
        if os.path.isfile(adapter_config) and os.path.isfile(adapter_model):
            out_name = entry.replace(" ", "_") + "_comfyui.safetensors"
            out_path = os.path.join(_LORAS_DIR, out_name)
            if not os.path.isfile(out_path):
                try:
                    _convert_peft_to_comfyui(entry_path, out_path)
                except Exception as e:
                    print(f"[AceStep SFT] Failed to convert {entry}: {e}")
            continue

        # --- Nested .safetensors (zip extraction artifact) ---
        for sub in os.listdir(entry_path):
            if sub.lower().endswith(".safetensors") and sub != "__MACOSX":
                src = os.path.join(entry_path, sub)
                if not os.path.isfile(src):
                    continue
                dst = os.path.join(_LORAS_DIR, sub)
                # If a directory with the same name exists at the destination,
                # rename the directory out of the way first (zip extraction artifact).
                if os.path.isdir(dst):
                    renamed_dir = dst + "_extracted_dir"
                    try:
                        os.rename(dst, renamed_dir)
                    except OSError:
                        continue
                    # Now copy from the renamed directory
                    src = os.path.join(renamed_dir, sub)
                if not os.path.isfile(dst):
                    import shutil
                    shutil.copy2(src, dst)
                    print(f"[AceStep SFT] Extracted nested LoRA: {sub}")


try:
    _prepare_loras_folder()
except Exception as e:
    print(f"[AceStep SFT] Warning: LoRA preparation failed: {e}")

# Register Loras/ folder so ComfyUI picks up the files
if os.path.isdir(_LORAS_DIR):
    folder_paths.add_model_folder_path("loras", _LORAS_DIR)

# ---------------------------------------------------------------------------

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
