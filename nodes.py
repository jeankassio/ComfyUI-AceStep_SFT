"""
AceStep 1.5 SFT - All-in-one Generation Node for ComfyUI

Provides a single node that handles latent creation, text encoding,
sampling with APG/ADG guidance (matching the AceStep Gradio pipeline),
and VAE decoding to produce audio output.
"""

import math
import random

import torch
import torch.nn.functional as F
import torchaudio
import yaml

import comfy.sample
import comfy.samplers
import comfy.model_management
import comfy.sd
import comfy.utils
import folder_paths
import node_helpers
import latent_preview

# ---------------------------------------------------------------------------
# APG (Adaptive Projected Guidance) - ported from AceStep SFT pipeline
# ---------------------------------------------------------------------------

class MomentumBuffer:
    def __init__(self, momentum: float = -0.75):
        self.momentum = momentum
        self.running_average = 0

    def update(self, update_value: torch.Tensor):
        new_average = self.momentum * self.running_average
        self.running_average = update_value + new_average


def _project(v0, v1, dims=[-1]):
    dtype = v0.dtype
    device_type = v0.device.type
    if device_type == "mps":
        v0, v1 = v0.cpu(), v1.cpu()
    v0, v1 = v0.double(), v1.double()
    v1 = F.normalize(v1, dim=dims)
    v0_parallel = (v0 * v1).sum(dim=dims, keepdim=True) * v1
    v0_orthogonal = v0 - v0_parallel
    return v0_parallel.to(dtype).to(device_type), v0_orthogonal.to(dtype).to(device_type)


def apg_guidance(pred_cond, pred_uncond, guidance_scale, momentum_buffer=None,
                 eta=0.0, norm_threshold=2.5, dims=[-1]):
    """APG guidance as used by AceStep SFT's generate_audio."""
    diff = pred_cond - pred_uncond
    if momentum_buffer is not None:
        momentum_buffer.update(diff)
        diff = momentum_buffer.running_average
    if norm_threshold > 0:
        ones = torch.ones_like(diff)
        diff_norm = diff.norm(p=2, dim=dims, keepdim=True)
        scale_factor = torch.minimum(ones, norm_threshold / diff_norm)
        diff = diff * scale_factor
    diff_parallel, diff_orthogonal = _project(diff, pred_cond, dims)
    normalized_update = diff_orthogonal + eta * diff_parallel
    return pred_cond + (guidance_scale - 1) * normalized_update


# ---------------------------------------------------------------------------
# ADG (Angle-based Dynamic Guidance) - ported from AceStep SFT pipeline
# ---------------------------------------------------------------------------

def _cos_sim(t1, t2):
    t1 = t1 / torch.linalg.norm(t1, dim=1, keepdim=True).clamp(min=1e-8)
    t2 = t2 / torch.linalg.norm(t2, dim=1, keepdim=True).clamp(min=1e-8)
    return torch.sum(t1 * t2, dim=1, keepdim=True).clamp(min=-1.0 + 1e-6, max=1.0 - 1e-6)


def _perpendicular(diff, base):
    n, t, c = diff.shape
    diff = diff.view(n * t, c).float()
    base = base.view(n * t, c).float()
    dot = torch.sum(diff * base, dim=1, keepdim=True)
    norm_sq = torch.sum(base * base, dim=1, keepdim=True)
    proj = (dot / (norm_sq + 1e-8)) * base
    perp = diff - proj
    return proj.view(n, t, c), perp.reshape(n, t, c)


def adg_guidance(latents, v_cond, v_uncond, sigma, guidance_scale,
                angle_clip=3.14159265 / 6, apply_norm=False, apply_clip=True):
    """ADG guidance (Angle-based Dynamic Guidance) for flow matching.

    Operates on velocity predictions in [B, T, C] layout.
    """
    n, t, c = v_cond.shape
    if isinstance(sigma, (int, float)):
        sigma = torch.tensor(sigma, device=latents.device, dtype=latents.dtype)
    sigma = sigma.view(-1, 1, 1).expand(n, 1, 1)

    weight = max(guidance_scale - 1, 0) + 1e-3

    x0_cond = latents - sigma * v_cond
    x0_uncond = latents - sigma * v_uncond
    x0_diff = x0_cond - x0_uncond

    theta = torch.acos(_cos_sim(
        x0_cond.view(-1, c).float(), x0_uncond.reshape(-1, c).contiguous().float()
    ))
    theta_new = torch.clip(weight * theta, -angle_clip, angle_clip) if apply_clip else weight * theta
    proj, perp = _perpendicular(x0_diff, x0_uncond)
    v_part = torch.cos(theta_new) * x0_cond
    mask = (torch.sin(theta) > 1e-3).float()
    p_part = perp * torch.sin(theta_new) / torch.sin(theta) * mask + perp * weight * (1 - mask)
    x0_new = v_part + p_part
    if apply_norm:
        x0_new = x0_new * (torch.linalg.norm(x0_cond, dim=1, keepdim=True)
                           / torch.linalg.norm(x0_new, dim=1, keepdim=True))

    v_out = (latents - x0_new) / sigma
    return v_out.reshape(n, t, c).to(latents.dtype)


def _clone_conditioning_value(value):
    if torch.is_tensor(value):
        return value.clone()
    if isinstance(value, dict):
        return {k: _clone_conditioning_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clone_conditioning_value(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_clone_conditioning_value(v) for v in value)
    return value


def _clone_conditioning(conditioning):
    return [
        [_clone_conditioning_value(value) for value in cond_item]
        for cond_item in conditioning
    ]


def _zero_conditioning_value(value):
    if torch.is_tensor(value):
        return torch.zeros_like(value)
    if isinstance(value, list):
        return [_zero_conditioning_value(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_zero_conditioning_value(v) for v in value)
    if isinstance(value, dict):
        return {k: _zero_conditioning_value(v) for k, v in value.items()}
    return value


def _reweight_conditioning_energy(tensor, erg_scale):
    if not torch.is_tensor(tensor) or abs(erg_scale) < 1e-8:
        return tensor
    mean = tensor.mean(dim=-1, keepdim=True)
    return mean + (tensor - mean) * (1.0 + erg_scale)


def _apply_erg_to_conditioning(conditioning, erg_scale):
    if abs(erg_scale) < 1e-8:
        return conditioning

    conditioned = _clone_conditioning(conditioning)
    for cond_item in conditioned:
        if cond_item and torch.is_tensor(cond_item[0]):
            cond_item[0] = _reweight_conditioning_energy(cond_item[0], erg_scale)
        if len(cond_item) > 1 and isinstance(cond_item[1], dict):
            lyrics_cond = cond_item[1].get("conditioning_lyrics")
            if lyrics_cond is not None:
                cond_item[1]["conditioning_lyrics"] = _reweight_conditioning_energy(
                    lyrics_cond, erg_scale
                )

    return conditioned


def _build_text_only_conditioning(conditioning):
    text_only = _clone_conditioning(conditioning)
    has_lyrics_branch = False

    for cond_item in text_only:
        if len(cond_item) > 1 and isinstance(cond_item[1], dict):
            lyrics_cond = cond_item[1].get("conditioning_lyrics")
            if lyrics_cond is not None:
                cond_item[1]["conditioning_lyrics"] = _zero_conditioning_value(
                    lyrics_cond
                )
                has_lyrics_branch = True

    return text_only if has_lyrics_branch else None


def _apply_omega_scale(model_output, omega_scale):
    if abs(omega_scale) < 1e-8:
        return model_output

    omega = 0.9 + 0.2 / (1.0 + math.exp(-float(omega_scale)))
    reduce_dims = tuple(range(1, model_output.ndim))
    mean = model_output.mean(dim=reduce_dims, keepdim=True)
    return mean + (model_output - mean) * omega


def _clone_processed_cond_value(value):
    if torch.is_tensor(value):
        return value.clone()
    if isinstance(value, dict):
        return {k: _clone_processed_cond_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clone_processed_cond_value(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_clone_processed_cond_value(v) for v in value)
    if hasattr(value, "_copy_with") and hasattr(value, "cond"):
        cond_value = _clone_processed_cond_value(value.cond)
        return value._copy_with(cond_value)
    return value


def _build_processed_text_only_conditioning(processed_conditioning):
    if processed_conditioning is None:
        return None

    text_only = []
    has_lyric_branch = False
    for cond_item in processed_conditioning:
        cloned = cond_item.copy()
        model_conds = cloned.get("model_conds")
        if model_conds is not None:
            cloned_model_conds = model_conds.copy()

            for key in ("lyric_embed", "lyric_token_idx"):
                lyric_cond = cloned_model_conds.get(key)
                if lyric_cond is not None and hasattr(lyric_cond, "_copy_with") and hasattr(lyric_cond, "cond"):
                    cloned_model_conds[key] = lyric_cond._copy_with(
                        torch.zeros_like(lyric_cond.cond)
                    )
                    has_lyric_branch = True

            lyrics_strength = cloned_model_conds.get("lyrics_strength")
            if lyrics_strength is not None and hasattr(lyrics_strength, "_copy_with"):
                cloned_model_conds["lyrics_strength"] = lyrics_strength._copy_with(0.0)

            cloned["model_conds"] = cloned_model_conds

        text_only.append(cloned)

    return text_only if has_lyric_branch else None


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LANGUAGES = [
    "en", "ja", "zh", "es", "de", "fr", "pt", "ru", "it", "nl",
    "pl", "tr", "vi", "cs", "fa", "id", "ko", "uk", "hu", "ar",
    "sv", "ro", "el",
]

KEYSCALES_LIST = [
    f"{root} {quality}"
    for quality in ["major", "minor"]
    for root in [
        "C", "C#", "Db", "D", "D#", "Eb", "E", "F", "F#", "Gb",
        "G", "G#", "Ab", "A", "A#", "Bb", "B",
    ]
]

GUIDANCE_MODES = ["apg", "adg", "standard_cfg"]


# ---------------------------------------------------------------------------
# Duration estimation from lyrics (auto mode for Comfy ACE encoder)
# ---------------------------------------------------------------------------

def _estimate_duration_from_lyrics(lyrics, bpm=120):
    """Estimate duration from lyric density and song structure.

    ComfyUI's ACE tokenizer requires fixed duration upfront (duration*5 tokens),
    so true free-form duration selection by Qwen is not available here.
    """
    if not lyrics or lyrics.strip().lower() in ("", "[instrumental]"):
        return 90.0

    lines = [line.strip() for line in lyrics.split("\n") if line.strip()]
    if not lines:
        return 90.0

    section_bar_map = {
        "intro": 4,
        "outro": 4,
        "verse": 8,
        "chorus": 8,
        "pre-chorus": 4,
        "prechorus": 4,
        "bridge": 4,
        "hook": 4,
        "refrain": 4,
        "instrumental": 8,
        "interlude": 4,
        "solo": 4,
        "break": 2,
    }

    section_bars = 0
    words = 0
    for line in lines:
        if line.startswith("[") and line.endswith("]"):
            tag = line[1:-1].lower().strip()
            tag = tag.split(":", 1)[0]
            section_bars += section_bar_map.get(tag, 2)
            continue

        # Remove direction cues like (rhythmic), (shouting)
        normalized = line
        while True:
            start = normalized.find("(")
            end = normalized.find(")", start + 1) if start >= 0 else -1
            if start >= 0 and end > start:
                normalized = (normalized[:start] + " " + normalized[end + 1:]).strip()
            else:
                break
        words += len([w for w in normalized.split() if w])

    # Conservative delivery for rap/funk-like dense lyrics.
    words_per_second = 2.0
    lyric_seconds = words / words_per_second

    # Convert structural bars to seconds with bpm consideration.
    effective_bpm = max(70, min(180, bpm if bpm > 0 else 120))
    sec_per_bar = 240.0 / effective_bpm
    structure_seconds = section_bars * sec_per_bar

    # Add safety margin for breath, transitions, adlibs.
    total = lyric_seconds + structure_seconds + 8.0

    # Keep within practical range for quality/perf.
    return max(20.0, min(round(total), 360.0))


# ---------------------------------------------------------------------------
# LoRA Loader Node
# ---------------------------------------------------------------------------

class AceStepSFTLoraLoader:
    """Chainable LoRA loader for AceStep 1.5 SFT.

    Accumulates LoRA specifications into a stack that is applied
    when the Generate node loads its models.  Multiple Lora Loader
    nodes can be chained together before connecting to Generate.
    """

    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora_name": (folder_paths.get_filename_list("loras"), {
                    "tooltip": "LoRA file to apply to the AceStep model.",
                }),
                "strength_model": ("FLOAT", {
                    "default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01,
                    "tooltip": "How strongly to modify the diffusion model.",
                }),
                "strength_clip": ("FLOAT", {
                    "default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01,
                    "tooltip": "How strongly to modify the CLIP/text encoder model.",
                }),
            },
            "optional": {
                "lora": ("ACESTEP_LORA", {
                    "tooltip": "Optional upstream LoRA stack to chain with.",
                }),
            },
        }

    RETURN_TYPES = ("ACESTEP_LORA",)
    RETURN_NAMES = ("lora",)
    FUNCTION = "load_lora"
    CATEGORY = "audio/AceStep SFT"
    DESCRIPTION = (
        "Loads a LoRA for AceStep 1.5 SFT. Chain multiple nodes "
        "together and connect the final output to the Generate node."
    )

    def load_lora(self, lora_name, strength_model, strength_clip, lora=None):
        lora_stack = list(lora) if lora is not None else []
        lora_stack.append({
            "lora_name": lora_name,
            "strength_model": strength_model,
            "strength_clip": strength_clip,
        })
        return (lora_stack,)


# ---------------------------------------------------------------------------
# Main Node
# ---------------------------------------------------------------------------

class AceStepSFTGenerate:
    """All-in-one AceStep 1.5 SFT music generation node.

    Generates its own latent from duration, encodes text (caption + lyrics +
    metadata) via CLIP, runs the diffusion sampler, and decodes the result
    with the VAE to produce audio.  Supports reference audio for timbre
    transfer and source audio for img2img-style denoising.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # ---- Model loading ----
                "diffusion_model": (folder_paths.get_filename_list("diffusion_models"), {
                    "tooltip": "AceStep 1.5 diffusion model (DiT). e.g. Audio/acestep_v1.5_sft.safetensors",
                }),
                "text_encoder_1": (folder_paths.get_filename_list("text_encoders"), {
                    "tooltip": "Qwen3-0.6B encoder for captions/lyrics. e.g. Audio/qwen_0.6b_ace15.safetensors",
                }),
                "text_encoder_2": (folder_paths.get_filename_list("text_encoders"), {
                    "tooltip": "Qwen3 LLM for audio codes (1.7B or 4B). e.g. Audio/qwen_1.7b_ace15.safetensors",
                }),
                "vae_name": (folder_paths.get_filename_list("vae"), {
                    "tooltip": "AceStep 1.5 audio VAE. e.g. Audio/ace_1.5_vae.safetensors",
                }),
                # ---- Text inputs ----
                "caption": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Describe the music: genre, mood, instruments, style...",
                    "tooltip": "Text description of the music to generate (tags/caption).",
                }),
                "lyrics": ("STRING", {
                    "multiline": True,
                    "default": "[Instrumental]",
                    "placeholder": "Song lyrics or [Instrumental]",
                    "tooltip": "Lyrics for the music. Use [Instrumental] for instrumental tracks.",
                }),
                "instrumental": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Force instrumental mode (overrides lyrics with [Instrumental]). Enabled by default because the baseline quality profile starts from instrumental generation.",
                }),
                # ---- Sampling ----
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xffffffffffffffff,
                    "control_after_generate": True,
                }),
                "steps": ("INT", {
                    "default": 60, "min": 1, "max": 200, "step": 1,
                    "tooltip": "Diffusion inference steps. The official AceStep 1.5 quality baseline uses 60 steps.",
                }),
                "cfg": ("FLOAT", {
                    "default": 15.0, "min": 1.0, "max": 20.0, "step": 0.1,
                    "tooltip": "Classifier-free guidance scale. The official AceStep 1.5 quality baseline uses 15.0.",
                }),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {
                    "default": "euler",
                    "tooltip": "Official AceStep 1.5 quality baseline uses Euler sampling.",
                }),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {
                    "default": "normal",
                    "tooltip": "Recommended scheduler pairing for the AceStep Euler baseline in ComfyUI.",
                }),
                "denoise": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Denoise strength. 1.0 = full generation from noise. < 1.0 requires source_audio. Auto-set to 1.0 when reference_audio is provided.",
                }),
                # ---- Guidance mode ----
                "guidance_mode": (GUIDANCE_MODES, {
                    "default": "apg",
                    "tooltip": "APG = Adaptive Projected Guidance (AceStep SFT default). ADG = Angle-based Dynamic Guidance. standard_cfg = regular CFG.",
                }),
                # ---- Duration & Metadata ----
                "duration": ("FLOAT", {
                    "default": 60.0, "min": 0.0, "max": 600.0, "step": 0.1,
                    "tooltip": "Duration in seconds. Default 60s matches the strongest quality baseline. Set to 0 for auto duration from lyrics or source_audio.",
                }),
                "bpm": ("INT", {
                    "default": 0, "min": 0, "max": 300,
                    "tooltip": "Beats per minute. 0 = auto (N/A, let model decide). Defaulting to auto usually gives better global musical coherence unless you need a fixed tempo.",
                }),
                "timesignature": (['auto', '4', '3', '2', '6'], {
                    "default": 'auto',
                    "tooltip": "Time signature numerator. 'auto' = let model decide (N/A). Defaulting to auto avoids over-constraining the planner.",
                }),
                "language": (LANGUAGES, {
                    "default": "en",
                    "tooltip": "Language tag for lyrics conditioning. English remains the safest default for broad model support and instrumental prompts.",
                }),
                "keyscale": (["auto"] + KEYSCALES_LIST, {
                    "default": "auto",
                    "tooltip": "Key and scale. 'auto' = let model decide (N/A). Defaulting to auto usually improves natural harmonic choices unless a fixed key is required.",
                }),
            },
            "optional": {
                # ---- Batch size ----
                "batch_size": ("INT", {
                    "default": 1, "min": 1, "max": 16,
                    "tooltip": "Number of audios to generate in parallel.",
                }),
                # ---- Audio inputs ----
                "source_audio": ("AUDIO", {
                    "tooltip": "Source audio to denoise/edit. Use denoise < 1.0 to preserve source characteristics. With duration=0, duration is derived from this audio.",
                }),
                "reference_audio": ("AUDIO", {
                    "tooltip": "Reference audio for style/timbre learning. Model generates new music that resembles this style. Set reference_as_cover=False for pure style transfer (recommended).",
                }),
                "reference_as_cover": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "If False (default): learn style from reference, generate completely new music. If True: use reference as base for remix/cover.",
                }),
                "audio_cover_strength": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Only used when reference_as_cover=True. How much reference content is preserved (0=remix, 1=exact cover).",
                }),
                # ---- LLM / Audio codes ----
                "generate_audio_codes": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable LLM audio code generation. Not used when reference_audio is provided.",
                }),
                "lm_cfg_scale": ("FLOAT", {
                    "default": 2.0, "min": 0.0, "max": 100.0, "step": 0.1,
                    "tooltip": "LLM classifier-free guidance scale.",
                }),
                "lm_temperature": ("FLOAT", {
                    "default": 0.85, "min": 0.0, "max": 2.0, "step": 0.01,
                    "tooltip": "LLM sampling temperature.",
                }),
                "lm_top_p": ("FLOAT", {
                    "default": 0.9, "min": 0.0, "max": 2000.0, "step": 0.01,
                }),
                "lm_top_k": ("INT", {
                    "default": 0, "min": 0, "max": 100,
                }),
                "lm_min_p": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001,
                }),
                "lm_negative_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Negative prompt for LLM audio code generation",
                    "tooltip": "Negative text prompt for LLM CFG.",
                }),
                # ---- Latent post-processing ----
                "latent_shift": ("FLOAT", {
                    "default": 0.0, "min": -0.2, "max": 0.2, "step": 0.01,
                    "tooltip": "Additive shift on DiT latents before VAE decode (anti-clipping).",
                }),
                "latent_rescale": ("FLOAT", {
                    "default": 1.0, "min": 0.5, "max": 1.5, "step": 0.01,
                    "tooltip": "Multiplicative scale on DiT latents before VAE decode.",
                }),
                # ---- Audio normalization ----
                "normalize_peak": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable peak normalization (normalize to max amplitude). Disabled by default to preserve the model's natural dynamics and transient balance.",
                }),
                "voice_boost": ("FLOAT", {
                    "default": 0.0, "min": -12.0, "max": 12.0, "step": 0.5,
                    "tooltip": "Voice boost in dB. Positive = louder voice (use with reference_audio). Default 0 dB.",
                }),
                # ---- APG parameters ----
                "apg_momentum": ("FLOAT", {
                    "default": -0.75, "min": -1.0, "max": 1.0, "step": 0.05,
                    "tooltip": "APG momentum buffer coefficient.",
                }),
                "apg_norm_threshold": ("FLOAT", {
                    "default": 2.5, "min": 0.0, "max": 10.0, "step": 0.1,
                    "tooltip": "APG norm threshold for gradient clipping.",
                }),
                "guidance_interval": ("FLOAT", {
                    "default": 0.5, "min": -1.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Official AceStep guidance interval width. 0.5 applies guidance in the centered middle band. Set to -1 to use legacy cfg_interval_start/end instead.",
                }),
                "guidance_interval_decay": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Linearly decays guidance inside the active interval toward min_guidance_scale, matching AceStep's official control.",
                }),
                "min_guidance_scale": ("FLOAT", {
                    "default": 3.0, "min": 0.0, "max": 30.0, "step": 0.1,
                    "tooltip": "Lowest guidance scale reached when guidance_interval_decay is enabled.",
                }),
                "guidance_scale_text": ("FLOAT", {
                    "default": -1.0, "min": -1.0, "max": 30.0, "step": 0.1,
                    "tooltip": "Independent text guidance scale. -1 inherits cfg. Works by adding a text-only conditioning branch inside the node.",
                }),
                "guidance_scale_lyric": ("FLOAT", {
                    "default": -1.0, "min": -1.0, "max": 30.0, "step": 0.1,
                    "tooltip": "Independent lyric guidance scale. -1 inherits cfg. The full branch remains text+lyrics; this value controls the lyric-only delta against the text-only branch.",
                }),
                "omega_scale": ("FLOAT", {
                    "default": 0.0, "min": -8.0, "max": 8.0, "step": 0.05,
                    "tooltip": "Mean-preserving output reweighting applied inside the node to emulate AceStep's omega_scale scheduler behavior.",
                }),
                "erg_scale": ("FLOAT", {
                    "default": 0.0, "min": -0.9, "max": 2.0, "step": 0.05,
                    "tooltip": "Node-local ERG approximation. Reweights prompt and lyric conditioning energy before sampling to strengthen prompt adherence without changing ComfyUI core.",
                }),
                # ---- CFG interval ----
                "cfg_interval_start": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Start applying CFG/APG guidance at this fraction of the schedule.",
                }),
                "cfg_interval_end": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Stop applying CFG/APG guidance at this fraction of the schedule.",
                }),
                # ---- Shift / Custom timesteps ----
                "shift": ("FLOAT", {
                    "default": 3.0, "min": 1.0, "max": 5.0, "step": 0.1,
                    "tooltip": "Timestep schedule shift. Gradio default = 3.0.",
                }),
                "custom_timesteps": ("STRING", {
                    "default": "",
                    "placeholder": "0.97,0.76,0.615,0.5,0.395,0.28,0.18,0.085,0",
                    "tooltip": "Custom comma-separated timesteps (overrides steps, shift and scheduler).",
                }),
                # ---- LoRA ----
                "lora": ("ACESTEP_LORA", {
                    "tooltip": "LoRA stack from one or more AceStep 1.5 SFT Lora Loader nodes.",
                }),

            },
        }

    RETURN_TYPES = ("AUDIO", "LATENT")
    RETURN_NAMES = ("audio", "latent")
    FUNCTION = "generate"
    CATEGORY = "audio/AceStep SFT"
    DESCRIPTION = (
        "All-in-one AceStep 1.5 SFT music generation with auto-metadata. "
        "Generates latent internally, supports source audio for denoising "
        "and reference audio for timbre/style transfer."
    )

    def generate(
        self,
        diffusion_model,
        text_encoder_1,
        text_encoder_2,
        vae_name,
        caption,
        lyrics,
        instrumental,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        denoise,
        guidance_mode,
        duration,
        bpm,
        timesignature,
        language,
        keyscale,
        # Optional
        batch_size=1,
        source_audio=None,
        reference_audio=None,
        reference_as_cover=False,
        audio_cover_strength=0.0,
        generate_audio_codes=True,
        lm_cfg_scale=2.0,
        lm_temperature=0.85,
        lm_top_p=0.9,
        lm_top_k=0,
        lm_min_p=0.0,
        lm_negative_prompt="",
        latent_shift=0.0,
        latent_rescale=1.0,
        normalize_peak=False,
        voice_boost=0.0,
        apg_momentum=-0.75,
        apg_norm_threshold=2.5,
        guidance_interval=0.5,
        guidance_interval_decay=0.0,
        min_guidance_scale=3.0,
        guidance_scale_text=-1.0,
        guidance_scale_lyric=-1.0,
        omega_scale=0.0,
        erg_scale=0.0,
        cfg_interval_start=0.0,
        cfg_interval_end=1.0,
        shift=3.0,
        custom_timesteps="",
        lora=None,
    ):
        actual_lyrics = "[Instrumental]" if instrumental else lyrics
        cfg_interval_start, cfg_interval_end = sorted(
            (cfg_interval_start, cfg_interval_end)
        )

        # --- Load models internally (matching Gradio pipeline) ---
        unet_path = folder_paths.get_full_path_or_raise(
            "diffusion_models", diffusion_model
        )
        model = comfy.sd.load_diffusion_model(unet_path)
        # Set to eval mode (ComfyUI handles dtype and device management)
        model.model.eval()

        clip_path1 = folder_paths.get_full_path_or_raise(
            "text_encoders", text_encoder_1
        )
        clip_path2 = folder_paths.get_full_path_or_raise(
            "text_encoders", text_encoder_2
        )
        clip = comfy.sd.load_clip(
            ckpt_paths=[clip_path1, clip_path2],
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            clip_type=comfy.sd.CLIPType.ACE,
        )
        # Set to eval mode
        clip.cond_stage_model.eval()

        # --- Apply LoRA stack ---
        if lora is not None:
            for lora_spec in lora:
                lora_path = folder_paths.get_full_path_or_raise(
                    "loras", lora_spec["lora_name"]
                )
                lora_data = comfy.utils.load_torch_file(lora_path, safe_load=True)
                model, clip = comfy.sd.load_lora_for_models(
                    model, clip, lora_data,
                    lora_spec["strength_model"], lora_spec["strength_clip"]
                )

        vae_path = folder_paths.get_full_path_or_raise("vae", vae_name)
        vae_sd = comfy.utils.load_torch_file(vae_path)
        vae = comfy.sd.VAE(sd=vae_sd)
        # Set to eval mode
        vae.first_stage_model.eval()

        vae_sr = getattr(vae, "audio_sample_rate", 48000)

        # --- 1. Determine duration ---
        auto_duration = (duration <= 0)
        if source_audio is not None and auto_duration:
            duration = source_audio["waveform"].shape[-1] / source_audio["sample_rate"]
        elif auto_duration:
            duration = _estimate_duration_from_lyrics(actual_lyrics, bpm)

        latent_length = max(10, round(duration * vae_sr / 1920))
        duration = latent_length * 1920.0 / vae_sr

        # --- 2. Create or encode starting latent ---
        # Auto-force denoise=1.0 when reference_audio is provided (style transfer, not img2img)
        if reference_audio is not None:
            # When using reference_audio for style transfer, we want pure generation,
            # not denoising from source audio. Create zero latent (pure noise).
            denoise = 1.0
            latent_image = torch.zeros(
                [batch_size, 64, latent_length],
                device=comfy.model_management.intermediate_device(),
            )
        elif source_audio is not None:
            src_waveform = source_audio["waveform"]
            src_sr = source_audio["sample_rate"]
            if src_sr != vae_sr:
                src_waveform = torchaudio.functional.resample(
                    src_waveform, src_sr, vae_sr
                )
            if src_waveform.shape[1] == 1:
                src_waveform = src_waveform.repeat(1, 2, 1)
            elif src_waveform.shape[1] > 2:
                src_waveform = src_waveform[:, :2, :]
            target_samples = latent_length * 1920
            if src_waveform.shape[-1] < target_samples:
                src_waveform = F.pad(
                    src_waveform, (0, target_samples - src_waveform.shape[-1])
                )
            elif src_waveform.shape[-1] > target_samples:
                src_waveform = src_waveform[:, :, :target_samples]
            latent_image = vae.encode(src_waveform.movedim(1, -1))
            if latent_image.shape[0] < batch_size:
                latent_image = latent_image.repeat(
                    math.ceil(batch_size / latent_image.shape[0]), 1, 1
                )[:batch_size]
        else:
            latent_image = torch.zeros(
                [batch_size, 64, latent_length],
                device=comfy.model_management.intermediate_device(),
            )

        latent_image = comfy.sample.fix_empty_latent_channels(
            model, latent_image, None,
        )

        # --- 3. Resolve auto metadata ---
        bpm_is_auto = (bpm == 0)
        ts_is_auto = (timesignature == "auto")
        ks_is_auto = (keyscale == "auto")
        tok_bpm = 120 if bpm_is_auto else bpm
        tok_ts = 4 if ts_is_auto else int(timesignature)
        tok_ks = "C major" if ks_is_auto else keyscale

        # --- 4. Encode positive conditioning ---
        if reference_audio is not None:
            generate_audio_codes = False

        tokenize_kwargs = dict(
            lyrics=actual_lyrics,
            bpm=tok_bpm,
            duration=duration,
            timesignature=tok_ts,
            language=language,
            keyscale=tok_ks,
            seed=seed,
            generate_audio_codes=generate_audio_codes,
            cfg_scale=lm_cfg_scale,
            temperature=lm_temperature,
            top_p=lm_top_p,
            top_k=lm_top_k,
            min_p=lm_min_p,
        )
        tokenize_kwargs["caption_negative"] = (
            lm_negative_prompt if lm_negative_prompt else ""
        )
        tokens = clip.tokenize(caption, **tokenize_kwargs)

        # --- Override tokenized prompts to match Gradio pipeline exactly ---
        inner_tok = getattr(clip.tokenizer, "qwen3_06b", None)
        if inner_tok is not None:
            dur_ceil = int(math.ceil(duration))
            # Enriched CoT - exclude auto values (matching Gradio Phase 1)
            cot_items = {}
            if not bpm_is_auto:
                cot_items["bpm"] = bpm
            cot_items["caption"] = caption
            cot_items["duration"] = dur_ceil
            if not ks_is_auto:
                cot_items["keyscale"] = keyscale
            cot_items["language"] = language
            if not ts_is_auto:
                cot_items["timesignature"] = tok_ts
            cot_yaml = yaml.dump(
                cot_items, allow_unicode=True, sort_keys=True
            ).strip()
            enriched_cot = f"<think>\n{cot_yaml}\n</think>"

            lm_tpl = (
                "<|im_start|>system\n# Instruction\n"
                "Generate audio semantic tokens based on the given conditions:\n\n"
                "<|im_end|>\n<|im_start|>user\n# Caption\n{}\n\n# Lyric\n{}\n"
                "<|im_end|>\n<|im_start|>assistant\n{}\n\n<|im_end|>\n"
            )
            tokens["lm_prompt"] = inner_tok.tokenize_with_weights(
                lm_tpl.format(caption, actual_lyrics.strip(), enriched_cot),
                False,
                disable_weights=True,
            )
            neg_caption = lm_negative_prompt if lm_negative_prompt else ""
            tokens["lm_prompt_negative"] = inner_tok.tokenize_with_weights(
                lm_tpl.format(
                    neg_caption, actual_lyrics.strip(), "<think>\n\n</think>"
                ),
                False,
                disable_weights=True,
            )

            # Fix lyrics template: single <|endoftext|> (Gradio uses single)
            tokens["lyrics"] = inner_tok.tokenize_with_weights(
                f"# Languages\n{language}\n\n# Lyric\n{actual_lyrics}<|endoftext|>",
                False,
                disable_weights=True,
            )

            # Fix qwen3_06b template: single <|endoftext|> + N/A for auto
            bpm_str = str(bpm) if not bpm_is_auto else "N/A"
            ts_str = timesignature if not ts_is_auto else "N/A"
            ks_str = keyscale if not ks_is_auto else "N/A"
            dur_str = f"{dur_ceil} seconds"
            meta_cap = (
                f"- bpm: {bpm_str}\n"
                f"- timesignature: {ts_str}\n"
                f"- keyscale: {ks_str}\n"
                f"- duration: {dur_str}"
            )
            tokens["qwen3_06b"] = inner_tok.tokenize_with_weights(
                "# Instruction\n"
                "Generate audio semantic tokens based on the given conditions:\n\n"
                f"# Caption\n{caption}\n\n# Metas\n{meta_cap}\n<|endoftext|>\n",
                True,
                disable_weights=True,
            )

        positive = clip.encode_from_tokens_scheduled(tokens)

        # Read generated audio codes from positive conditioning
        audio_codes_from_pos = None
        for cond_item in positive:
            if len(cond_item) > 1 and "audio_codes" in cond_item[1]:
                audio_codes_from_pos = cond_item[1]["audio_codes"]
                break

        # --- 4.5. Initialize reference audio variables ---
        refer_audio_latents = None
        refer_audio_order_mask = None

        # --- 5. Negative conditioning ---
        neg_tokens = clip.tokenize("", generate_audio_codes=False)
        negative = clip.encode_from_tokens_scheduled(neg_tokens)

        # Share audio_codes from positive → negative
        if audio_codes_from_pos is not None:
            negative = node_helpers.conditioning_set_values(
                negative, {"audio_codes": audio_codes_from_pos}
            )

        # Zero out negative embedding → model uses null_condition_emb
        for n in negative:
            n[0] = torch.zeros_like(n[0])

        # --- 6. Reference audio conditioning (MUST be before inference_mode) ---
        # When reference_as_cover=False (default): model learns style/timbre from reference and generates completely new music
        # When reference_as_cover=True: model uses reference as base for remix/cover
        if reference_audio is not None:
            ref_waveform = reference_audio["waveform"]
            ref_sr = reference_audio["sample_rate"]
            
            # Resample if needed
            if ref_sr != vae_sr:
                ref_waveform = torchaudio.functional.resample(
                    ref_waveform, ref_sr, vae_sr
                )
            
            # Normalize channels to stereo
            if ref_waveform.shape[1] == 1:
                ref_waveform = ref_waveform.repeat(1, 2, 1)
            elif ref_waveform.shape[1] > 2:
                ref_waveform = ref_waveform[:, :2, :]
            
            # Pad/truncate to match latent_length
            target_samples = latent_length * 1920
            if ref_waveform.shape[-1] < target_samples:
                ref_waveform = F.pad(
                    ref_waveform, (0, target_samples - ref_waveform.shape[-1])
                )
            elif ref_waveform.shape[-1] > target_samples:
                ref_waveform = ref_waveform[:, :, :target_samples]
            
            # Encode reference audio to latent space (BEFORE inference_mode)
            ref_latent = vae.encode(ref_waveform.movedim(1, -1))
            
            # Match batch size
            if ref_latent.shape[0] < batch_size:
                ref_latent = ref_latent.repeat(
                    math.ceil(batch_size / ref_latent.shape[0]), 1, 1
                )[:batch_size]
            
            # Prepare for conditioning: create order mask and latents
            refer_audio_latents = ref_latent
            refer_audio_order_mask = torch.arange(batch_size, device=ref_latent.device, dtype=torch.long)
            is_cover = reference_as_cover
            positive = node_helpers.conditioning_set_values(
                positive,
                {
                    "refer_audio_acoustic_hidden_states_packed": refer_audio_latents,
                    "refer_audio_order_mask": refer_audio_order_mask,
                    "is_covers": torch.full(
                        (batch_size,),
                        is_cover,
                        dtype=torch.bool,
                        device=refer_audio_latents.device,
                    ),
                    "audio_cover_strength": (
                        audio_cover_strength if is_cover else 0.0
                    ),
                },
                append=True,
            )

        if abs(erg_scale) > 1e-8:
            positive = _apply_erg_to_conditioning(positive, erg_scale)

        resolved_text_guidance = cfg if guidance_scale_text < 0.0 else guidance_scale_text
        resolved_lyric_guidance = cfg if guidance_scale_lyric < 0.0 else guidance_scale_lyric
        text_only_positive = _build_text_only_conditioning(positive)
        split_guidance_active = (
            text_only_positive is not None and (
                abs(resolved_text_guidance - cfg) > 1e-6
                or abs(resolved_lyric_guidance - cfg) > 1e-6
            )
        )

        # --- 7. Prepare noise ---
        # Wrap all sampling and decoding in torch.inference_mode() for efficiency
        with torch.inference_mode():
            noise = comfy.sample.prepare_noise(latent_image, seed)

            # --- 8. Compute sigmas (Gradio exact schedule) ---
            custom_sigmas = None
            if custom_timesteps and custom_timesteps.strip():
                parts = [x.strip() for x in custom_timesteps.split(",") if x.strip()]
                ts = [float(x) for x in parts]
                if not ts or ts[-1] != 0.0:
                    ts.append(0.0)
                custom_sigmas = torch.FloatTensor(ts)
                steps = len(custom_sigmas) - 1
            else:
                t = torch.linspace(1.0, 0.0, steps + 1)
                if shift != 1.0:
                    custom_sigmas = shift * t / (1 + (shift - 1) * t)
                else:
                    custom_sigmas = t

            use_official_interval = guidance_interval >= 0.0
            official_interval = max(0.0, min(1.0, guidance_interval))
            interval_step_start = int(steps * ((1.0 - official_interval) / 2.0))
            interval_step_end = int(steps * (official_interval / 2.0 + 0.5))

            # --- 9. Apply guidance via model patching ---
            if (
                (guidance_mode in ("apg", "adg") and cfg > 1.0)
                or split_guidance_active
                or abs(omega_scale) > 1e-8
            ):
                momentum_buf = MomentumBuffer(momentum=apg_momentum)
                norm_thresh = apg_norm_threshold
                schedule_state = {
                    "index": 0,
                    "last_sigma": None,
                    "denom": max(steps - 1, 1),
                }
                branch_state = {"text_denoised": None}
                use_adg = (guidance_mode == "adg")

                def get_step_context(sigma, cond_scale):
                    sigma_value = float(sigma.flatten()[0])
                    if schedule_state["last_sigma"] != sigma_value:
                        if schedule_state["last_sigma"] is not None:
                            schedule_state["index"] = min(
                                schedule_state["index"] + 1,
                                schedule_state["denom"],
                            )
                        schedule_state["last_sigma"] = sigma_value

                    step_index = schedule_state["index"]
                    progress = step_index / schedule_state["denom"]
                    if use_official_interval:
                        in_interval = interval_step_start <= step_index < interval_step_end
                    else:
                        in_interval = cfg_interval_start <= progress <= cfg_interval_end

                    current_guidance_scale = cond_scale
                    if guidance_interval_decay > 0.0:
                        if use_official_interval:
                            interval_span = max(interval_step_end - interval_step_start - 1, 1)
                            interval_progress = min(
                                max((step_index - interval_step_start) / interval_span, 0.0),
                                1.0,
                            )
                        else:
                            interval_width = max(cfg_interval_end - cfg_interval_start, 1e-8)
                            interval_progress = min(
                                max((progress - cfg_interval_start) / interval_width, 0.0),
                                1.0,
                            )
                        current_guidance_scale = cond_scale - (
                            (cond_scale - min_guidance_scale)
                            * interval_progress
                            * guidance_interval_decay
                        )

                    return sigma_value, step_index, progress, in_interval, current_guidance_scale

                def calc_cond_batch_function(args):
                    x = args["input"]
                    sigma = args["sigma"]
                    cond, uncond = args["conds"]
                    model_options = args["model_options"]
                    branch_state["text_denoised"] = None

                    if not split_guidance_active:
                        return comfy.samplers.calc_cond_batch(
                            args["model"], [cond, uncond], x, sigma, model_options
                        )

                    _, _, _, in_interval, _ = get_step_context(sigma, cfg)
                    if not in_interval:
                        return comfy.samplers.calc_cond_batch(
                            args["model"], [cond, uncond], x, sigma, model_options
                        )

                    cond_out, uncond_out = comfy.samplers.calc_cond_batch(
                        args["model"], [cond, uncond], x, sigma, model_options
                    )
                    text_only_cond = _build_processed_text_only_conditioning(cond)
                    if text_only_cond is None:
                        return [cond_out, uncond_out]

                    text_out, _ = comfy.samplers.calc_cond_batch(
                        args["model"], [text_only_cond, None], x, sigma, model_options
                    )
                    branch_state["text_denoised"] = text_out
                    return [cond_out, uncond_out]

                def guided_cfg_function(args):
                    cond_denoised = args["cond_denoised"]
                    uncond_denoised = args["uncond_denoised"]
                    cond_scale = args["cond_scale"]
                    x = args["input"]
                    sigma = args["sigma"]

                    sigma_value, _, _, in_interval, current_guidance_scale = get_step_context(
                        sigma, cond_scale
                    )

                    effective_cond_denoised = cond_denoised
                    text_denoised = branch_state.get("text_denoised")
                    if split_guidance_active and text_denoised is not None:
                        base_guidance = max(cond_scale, 1e-8)
                        text_unit = resolved_text_guidance / base_guidance
                        lyric_unit = resolved_lyric_guidance / base_guidance

                        cond_model_output = x - cond_denoised
                        uncond_model_output = x - uncond_denoised
                        text_model_output = x - text_denoised
                        blended_model_output = (
                            uncond_model_output
                            + (text_model_output - uncond_model_output) * text_unit
                            + (cond_model_output - text_model_output) * lyric_unit
                        )
                        effective_cond_denoised = x - blended_model_output

                    if not in_interval:
                        return _apply_omega_scale(x - cond_denoised, omega_scale)

                    if guidance_mode == "standard_cfg" or current_guidance_scale <= 1.0:
                        guided_denoised = uncond_denoised + (
                            effective_cond_denoised - uncond_denoised
                        ) * current_guidance_scale
                        return _apply_omega_scale(x - guided_denoised, omega_scale)

                    sigma_r = sigma.reshape(-1, *([1] * (x.ndim - 1))).clamp(min=1e-8)
                    v_cond = (x - effective_cond_denoised) / sigma_r
                    v_uncond = (x - uncond_denoised) / sigma_r

                    if use_adg:
                        v_guided = adg_guidance(
                            x.movedim(1, -1),
                            v_cond.movedim(1, -1),
                            v_uncond.movedim(1, -1),
                            sigma_value,
                            current_guidance_scale,
                        ).movedim(-1, 1)
                    else:
                        v_guided = apg_guidance(
                            v_cond,
                            v_uncond,
                            current_guidance_scale,
                            momentum_buffer=momentum_buf,
                            norm_threshold=norm_thresh,
                            dims=[-1],
                        )

                    return _apply_omega_scale(v_guided * sigma_r, omega_scale)

                model = model.clone()
                if split_guidance_active:
                    model.set_model_sampler_calc_cond_batch_function(
                        calc_cond_batch_function
                    )
                model.set_model_sampler_cfg_function(
                    guided_cfg_function, disable_cfg1_optimization=True
                )

            # --- 10. Sample ---
            callback = latent_preview.prepare_callback(model, steps)
            disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

            samples = comfy.sample.sample(
                model,
                noise,
                steps,
                cfg,
                sampler_name,
                scheduler,
                positive,
                negative,
                latent_image,
                denoise=denoise,
                sigmas=custom_sigmas,
                callback=callback,
                disable_pbar=disable_pbar,
                seed=seed,
            )

            # --- 11. Post-process latents ---
            if latent_shift != 0.0 or latent_rescale != 1.0:
                samples = samples * latent_rescale + latent_shift

            out_latent = {"samples": samples, "type": "audio"}

            # --- 12. Decode with VAE ---
            audio = vae.decode(samples).movedim(-1, 1)

            if audio.dtype != torch.float32:
                audio = audio.float()

            # Peak normalization (optional)
            if normalize_peak:
                peak = audio.abs().amax(dim=[1, 2], keepdim=True).clamp(min=1e-8)
                audio = audio / peak

            # Apply voice boost if specified (dB to linear: 10^(dB/20))
            if voice_boost != 0.0:
                boost_linear = 10.0 ** (voice_boost / 20.0)
                audio = audio * boost_linear
                # Soft clip to avoid excessive clipping
                audio = torch.tanh(audio * 0.99) / 0.99

            audio_output = {
                "waveform": audio,
                "sample_rate": vae_sr,
            }

        return (audio_output, out_latent)


# ---------------------------------------------------------------------------
# Node Registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "AceStepSFTGenerate": AceStepSFTGenerate,
    "AceStepSFTLoraLoader": AceStepSFTLoraLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AceStepSFTGenerate": "AceStep 1.5 SFT Generate",
    "AceStepSFTLoraLoader": "AceStep 1.5 SFT Lora Loader",
}
