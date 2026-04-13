"""
AceStep 1.5 SFT - All-in-one Generation Node for ComfyUI

Provides a single node that handles latent creation, text encoding,
sampling with APG/ADG guidance (matching the AceStep Gradio pipeline),
and VAE decoding to produce audio output.
"""

import math
import os
import re
import gc
import json
import random
from io import BytesIO

import av
import torch
import torch.nn.functional as F
import torchaudio
import yaml

from comfy.cli_args import args

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


def _clone_runtime_conditioning(conditioning):
    if conditioning is None:
        return None

    cloned = []
    for cond_item in conditioning:
        cloned.append([
            _clone_processed_cond_value(value)
            for value in cond_item
        ])
    return cloned


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


def _erg_tau_from_scale(erg_scale):
    if erg_scale <= 0.0:
        return 1.0
    return max(0.01, math.exp(-2.0 * float(erg_scale)))


def _apply_erg_tau_to_tensor(tensor, erg_scale):
    if not torch.is_tensor(tensor):
        return tensor
    tau = _erg_tau_from_scale(erg_scale)
    if tau >= 0.999999:
        return tensor
    return tensor * tau


def _apply_erg_tau_to_model_output(model_output, erg_scale):
    tau = _erg_tau_from_scale(erg_scale)
    if tau >= 0.999999:
        return model_output
    return model_output * tau


def _build_null_negative(positive):
    """Build null negative conditioning for ACE-Step 1.5 CFG.

    The official pipeline uses a trained ``null_condition_emb`` parameter for
    the unconditional CFG pass.  ComfyUI triggers this via
    ``torch.count_nonzero(cross_attn) == 0`` inside ``ACEStep15.extra_conds``.
    Encoding an empty string with the Qwen-3 text encoder produces a *non-zero*
    tensor (because of the instruction template), so the detection never fires
    and ``null_condition_emb`` is never used — breaking CFG quality.

    This helper creates a zero ``cross_attn`` tensor (triggering the detection)
    and copies every other conditioning value from *positive* so that
    ``context_latents`` (lyrics, audio-codes, reference audio, etc.) match
    exactly, replicating the official behaviour.
    """
    negative = []
    for cond_tensor, cond_dict in positive:
        zero_cross_attn = torch.zeros_like(cond_tensor)
        new_dict = {
            k: _clone_processed_cond_value(v)
            for k, v in cond_dict.items()
        }
        negative.append([zero_cross_attn, new_dict])
    return negative


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


def _build_processed_erg_conditioning(processed_conditioning, erg_scale):
    if processed_conditioning is None:
        return None

    tau = _erg_tau_from_scale(erg_scale)
    if tau >= 0.999999:
        return None

    erg_conditioning = []
    has_erg_branch = False

    for cond_item in processed_conditioning:
        cloned = cond_item.copy()
        model_conds = cloned.get("model_conds")
        if model_conds is not None:
            cloned_model_conds = model_conds.copy()

            cross_attn = cloned_model_conds.get("c_crossattn")
            if cross_attn is not None and hasattr(cross_attn, "_copy_with") and hasattr(cross_attn, "cond"):
                cloned_model_conds["c_crossattn"] = cross_attn._copy_with(
                    _apply_erg_tau_to_tensor(cross_attn.cond, erg_scale)
                )
                has_erg_branch = True

            lyric_embed = cloned_model_conds.get("lyric_embed")
            if lyric_embed is not None and hasattr(lyric_embed, "_copy_with") and hasattr(lyric_embed, "cond"):
                cloned_model_conds["lyric_embed"] = lyric_embed._copy_with(
                    _apply_erg_tau_to_tensor(lyric_embed.cond, erg_scale)
                )
                has_erg_branch = True

            cloned["model_conds"] = cloned_model_conds

        erg_conditioning.append(cloned)

    return erg_conditioning if has_erg_branch else None


def _apply_omega_scale(model_output, omega_scale):
    if abs(omega_scale) < 1e-8:
        return model_output

    omega = 0.9 + 0.2 / (1.0 + math.exp(-float(omega_scale)))
    reduce_dims = tuple(range(1, model_output.ndim))
    mean = model_output.mean(dim=reduce_dims, keepdim=True)
    return mean + (model_output - mean) * omega


def _normalize_audio_to_stereo_48k(waveform, sample_rate, target_sr=48000):
    if waveform.dim() == 2:
        waveform = waveform.unsqueeze(0)
    elif waveform.dim() == 1:
        waveform = waveform.unsqueeze(0).unsqueeze(0)

    # Accept both [B, C, T] and [B, T, C] layouts from external audio nodes.
    if waveform.dim() == 3 and waveform.shape[1] > waveform.shape[2] and waveform.shape[2] <= 8:
        waveform = waveform.movedim(-1, 1)

    if waveform.shape[1] == 1:
        waveform = waveform.repeat(1, 2, 1)
    elif waveform.shape[1] > 2:
        waveform = waveform[:, :2, :]

    if sample_rate != target_sr:
        waveform = torchaudio.functional.resample(waveform, sample_rate, target_sr)

    return torch.clamp(waveform, -1.0, 1.0)


def _apply_fade(audio_data, fade_in_samples=0, fade_out_samples=0):
    if fade_in_samples <= 0 and fade_out_samples <= 0:
        return audio_data

    audio = audio_data.clone()
    total_samples = audio.shape[-1]

    if fade_in_samples > 0:
        actual_in = min(fade_in_samples, total_samples)
        ramp = torch.linspace(0.0, 1.0, actual_in, dtype=audio.dtype, device=audio.device)
        audio[..., :actual_in] = audio[..., :actual_in] * ramp

    if fade_out_samples > 0:
        actual_out = min(fade_out_samples, total_samples)
        ramp = torch.linspace(1.0, 0.0, actual_out, dtype=audio.dtype, device=audio.device)
        audio[..., total_samples - actual_out:] = audio[..., total_samples - actual_out:] * ramp

    return audio


def _resolve_sampler_for_infer_method(sampler_name, infer_method):
    if infer_method != "sde":
        return sampler_name

    sampler_map = {
        "euler": "dpmpp_2m_sde",
        "euler_cfg_pp": "dpmpp_2m_sde",
        "heun": "exp_heun_2_x0_sde",
        "heunpp2": "exp_heun_2_x0_sde",
        "exp_heun_2_x0": "exp_heun_2_x0_sde",
    }
    return sampler_map.get(sampler_name, sampler_name)


def _vae_uses_1d_tiling(vae):
    return getattr(vae, "latent_dim", None) == 1 or getattr(vae, "extra_1d_channel", None) is not None


def _vae_encode_with_optional_tiling(vae, pixel_samples, use_tiled_vae):
    if not use_tiled_vae:
        return vae.encode(pixel_samples)
    if _vae_uses_1d_tiling(vae):
        return vae.encode_tiled(pixel_samples, tile_y=1)
    return vae.encode_tiled(pixel_samples)


def _vae_decode_with_optional_tiling(vae, samples, use_tiled_vae):
    if not use_tiled_vae:
        return vae.decode(samples)
    if _vae_uses_1d_tiling(vae):
        return vae.decode_tiled(samples, tile_y=1)
    return vae.decode_tiled(samples)


def _is_audio_payload(value):
    return isinstance(value, dict) and "waveform" in value and "sample_rate" in value


def _is_latent_payload(value):
    return isinstance(value, dict) and "samples" in value


def _get_source_duration_seconds(source_input, vae_sr):
    if _is_audio_payload(source_input):
        sample_rate = max(int(source_input["sample_rate"]), 1)
        return source_input["waveform"].shape[-1] / sample_rate
    if _is_latent_payload(source_input):
        return source_input["samples"].shape[-1] * 1920.0 / vae_sr
    return None


def _match_latent_length(latent_samples, latent_length):
    if latent_samples.shape[-1] < latent_length:
        latent_samples = F.pad(latent_samples, (0, latent_length - latent_samples.shape[-1]))
    elif latent_samples.shape[-1] > latent_length:
        latent_samples = latent_samples[..., :latent_length]
    return latent_samples


def _match_noise_mask_length(noise_mask, latent_length):
    if noise_mask is None:
        return None
    if noise_mask.shape[-1] < latent_length:
        noise_mask = F.pad(noise_mask, (0, latent_length - noise_mask.shape[-1]))
    elif noise_mask.shape[-1] > latent_length:
        noise_mask = noise_mask[..., :latent_length]
    return noise_mask


def _build_source_latent(vae, source_input, batch_size, latent_length, vae_sr, use_tiled_vae):
    if _is_latent_payload(source_input):
        latent_image = source_input["samples"].clone()
        latent_image = _match_latent_length(latent_image, latent_length)
    elif _is_audio_payload(source_input):
        src_waveform = _normalize_audio_to_stereo_48k(
            source_input["waveform"], source_input["sample_rate"], vae_sr
        )
        target_samples = latent_length * 1920
        if src_waveform.shape[-1] < target_samples:
            src_waveform = F.pad(src_waveform, (0, target_samples - src_waveform.shape[-1]))
        elif src_waveform.shape[-1] > target_samples:
            src_waveform = src_waveform[:, :, :target_samples]
        latent_image = _vae_encode_with_optional_tiling(
            vae, src_waveform.movedim(1, -1), use_tiled_vae
        )
    else:
        raise ValueError("latent_or_audio must be AUDIO or LATENT.")

    if latent_image.shape[0] < batch_size:
        latent_image = latent_image.repeat(
            math.ceil(batch_size / latent_image.shape[0]), 1, 1
        )[:batch_size]

    return latent_image


def _get_source_latent_metadata(source_input):
    if not _is_latent_payload(source_input):
        return None
    return {
        "downscale_ratio_spacial": source_input.get("downscale_ratio_spacial", None),
        "batch_index": source_input.get("batch_index", None),
        "noise_mask": source_input.get("noise_mask", None),
    }


def _release_acestep_generation_models(*loaded_objects):
    for loaded_object in loaded_objects:
        if loaded_object is None:
            continue
        try:
            if hasattr(loaded_object, "model") and hasattr(loaded_object.model, "detach"):
                loaded_object.model.detach(unpatch_all=False)
        except Exception:
            pass

    try:
        device = comfy.model_management.get_torch_device()
        comfy.model_management.free_memory(10**30, device)
    except Exception:
        pass

    try:
        comfy.model_management.free_memory(10**30, torch.device("cpu"))
    except Exception:
        pass

    try:
        comfy.model_management.cleanup_models_gc()
    except Exception:
        pass

    try:
        comfy.model_management.soft_empty_cache()
    except Exception:
        pass

    gc.collect()


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
    lyric_lines = 0
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
        line_words = len([w for w in normalized.split() if w])
        if line_words > 0:
            lyric_lines += 1
        words += line_words

    # Adapt delivery rate to BPM.
    effective_bpm = max(70, min(200, bpm if bpm > 0 else 120))
    if effective_bpm >= 140:
        words_per_second = 2.5  # fast genres: rapid-fire delivery
    elif effective_bpm >= 110:
        words_per_second = 2.2  # moderate tempo
    else:
        words_per_second = 1.8  # slow ballads: drawn-out delivery

    # Lyric time = raw vocal delivery + ~0.3s breath between lines.
    lyric_seconds = words / words_per_second + lyric_lines * 0.3

    # Structure time = musical bars (lyrics are sung *during* these bars).
    sec_per_bar = 240.0 / effective_bpm
    structure_seconds = section_bars * sec_per_bar

    # Lyrics overlap with structure, so take the longer of the two,
    # then add a small safety margin for intro/outro/transitions.
    total = max(lyric_seconds, structure_seconds) + 10.0

    estimated = max(20.0, min(round(total), 360.0))
    print(f"[AceStep SFT] Auto-duration: {estimated}s "
          f"({words} words, {lyric_lines} lines, "
          f"{section_bars} section bars, {effective_bpm} BPM, "
          f"lyric={lyric_seconds:.0f}s struct={structure_seconds:.0f}s)")
    return estimated


# ---------------------------------------------------------------------------
# Music Style Analysis (multi-model + librosa)
# ---------------------------------------------------------------------------

# Supported audio analysis models (HuggingFace repo IDs)
_ANALYSIS_MODELS = {
    "ACE-Step-Transcriber": "ACE-Step/acestep-transcriber",
    "Qwen2-Audio-7B-Instruct": "Qwen/Qwen2-Audio-7B-Instruct",
    "Qwen2.5-Omni-3B": "Qwen/Qwen2.5-Omni-3B",
    "Ke-Omni-R-3B": "KE-Team/Ke-Omni-R-3B",
    "Qwen2.5-Omni-7B": "Qwen/Qwen2.5-Omni-7B",
    "Whisper-large-v3-transcription": "openai/whisper-large-v3",
    "Whisper-large-v3-turbo-transcription": "openai/whisper-large-v3-turbo",
    "Distil-Whisper-large-v3.5-transcription": "distil-whisper/distil-large-v3.5",
    "Distil-Whisper-large-v3-transcription": "distil-whisper/distil-large-v3",
    "AST-AudioSet": "MIT/ast-finetuned-audioset-10-10-0.4593",
    "MERT-v1-330M": "m-a-p/MERT-v1-330M",
    "Whisper-large-v2-audio-captioning": "MU-NLPC/whisper-large-v2-audio-captioning",
    "Whisper-small-audio-captioning": "MU-NLPC/whisper-small-audio-captioning",
    "Whisper-tiny-audio-captioning": "MU-NLPC/whisper-tiny-audio-captioning",
}

_NATIVE_ANALYSIS_MODEL = "ACE-Step-Transcriber"


def _is_whisper_captioning_model(model_key):
    return "audio-captioning" in model_key.lower()


def _is_whisper_asr_model(model_key):
    return model_key.endswith("-transcription")


def _is_acestep_transcriber_model(model_key):
    return model_key == "ACE-Step-Transcriber"

# Singleton cache
_audio_model = None
_audio_processor = None
_audio_model_name = None


def _get_model_dir(model_key):
    """Return the local path for a given model key."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", model_key)


def _ensure_model_downloaded(model_key):
    """Download model to the node folder if not already present."""
    model_dir = _get_model_dir(model_key)
    config_path = os.path.join(model_dir, "config.json")
    if os.path.isfile(config_path):
        return model_dir
    repo_id = _ANALYSIS_MODELS[model_key]
    from huggingface_hub import snapshot_download
    print(f"[AceStep SFT] Downloading {model_key} for music analysis (first time only)...")
    snapshot_download(repo_id, local_dir=model_dir)
    print(f"[AceStep SFT] {model_key} download complete.")
    return model_dir


def _get_analysis_device():
    """Use the same active device selected by the running ComfyUI process."""
    return comfy.model_management.get_torch_device()


def _get_analysis_device_map():
    device = _get_analysis_device()
    return {"": str(device)}


def _load_audio_model(model_key, use_flash_attn=False):
    """Load an audio analysis model + processor (cached singleton).

    If a different model is already loaded, unloads it first.
    """
    global _audio_model, _audio_processor, _audio_model_name
    if _audio_model is not None and _audio_model_name == model_key:
        return _audio_model, _audio_processor
    if _audio_model is not None:
        _unload_audio_model()

    model_dir = _ensure_model_downloaded(model_key)
    target_device = _get_analysis_device()
    load_kwargs = dict(
        torch_dtype=torch.bfloat16,
        device_map=_get_analysis_device_map(),
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    if use_flash_attn:
        load_kwargs["attn_implementation"] = "flash_attention_2"
        print(f"[AceStep SFT] Using flash_attention_2 for {model_key}")
    print(f"[AceStep SFT] Loading {model_key} on {target_device}...")

    if _is_acestep_transcriber_model(model_key):
        import warnings
        from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*Flash Attention 2 without specifying a torch dtype.*")
            warnings.filterwarnings("ignore", message=".*Token2WavModel.*fallback.*")
            _audio_model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                model_dir, **load_kwargs,
            )
        # Native transcriber usage is text-only, so disable speech synthesis modules.
        _audio_model.disable_talker()
        _audio_model.eval()
        _audio_processor = Qwen2_5OmniProcessor.from_pretrained(model_dir, use_fast=False)
    elif model_key.startswith("Qwen2.5-Omni"):
        import warnings
        from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
        # Suppress harmless warnings about Token2Wav flash_attn fallback and dtype
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*Flash Attention 2 without specifying a torch dtype.*")
            warnings.filterwarnings("ignore", message=".*Token2WavModel.*fallback.*")
            _audio_model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                model_dir, **load_kwargs,
            )
        _audio_model.disable_talker()
        _audio_model.eval()
        _audio_processor = Qwen2_5OmniProcessor.from_pretrained(model_dir, use_fast=False)
    elif model_key == "Qwen2-Audio-7B-Instruct":
        from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
        _audio_model = Qwen2AudioForConditionalGeneration.from_pretrained(
            model_dir, **load_kwargs,
        )
        _audio_model.eval()
        _audio_processor = AutoProcessor.from_pretrained(model_dir)
    elif _is_whisper_captioning_model(model_key):
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        _audio_model = WhisperForConditionalGeneration.from_pretrained(
            model_dir,
            torch_dtype=torch.float32,
            device_map=_get_analysis_device_map(),
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        _audio_model.eval()
        _audio_processor = WhisperProcessor.from_pretrained(model_dir)
    elif _is_whisper_asr_model(model_key):
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
        whisper_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        asr_load_kwargs = {
            "torch_dtype": whisper_dtype,
            "device_map": _get_analysis_device_map(),
            "low_cpu_mem_usage": True,
            "use_safetensors": True,
        }
        if use_flash_attn:
            asr_load_kwargs["attn_implementation"] = "flash_attention_2"
        _audio_model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_dir, **asr_load_kwargs,
        )
        _audio_model.eval()
        _audio_processor = AutoProcessor.from_pretrained(model_dir)
    elif model_key == "Ke-Omni-R-3B":
        from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor
        _audio_model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            model_dir, **load_kwargs,
        )
        _audio_model.eval()
        _audio_processor = Qwen2_5OmniProcessor.from_pretrained(model_dir, use_fast=False)
    elif model_key == "MERT-v1-330M":
        from transformers import AutoModel, Wav2Vec2FeatureExtractor
        _audio_model = AutoModel.from_pretrained(
            model_dir, torch_dtype=torch.float32, device_map=_get_analysis_device_map(),
            trust_remote_code=True,
        )
        _audio_model.eval()
        _audio_processor = Wav2Vec2FeatureExtractor.from_pretrained(
            model_dir, trust_remote_code=True,
        )
    elif model_key == "AST-AudioSet":
        from transformers import ASTForAudioClassification, AutoFeatureExtractor
        _audio_model = ASTForAudioClassification.from_pretrained(
            model_dir, torch_dtype=torch.float32, device_map=_get_analysis_device_map(),
        )
        _audio_model.eval()
        _audio_processor = AutoFeatureExtractor.from_pretrained(model_dir)

    _audio_model_name = model_key
    print(f"[AceStep SFT] {model_key} loaded.")
    return _audio_model, _audio_processor


def _unload_audio_model():
    """Unload audio analysis model to free VRAM."""
    global _audio_model, _audio_processor, _audio_model_name
    name = _audio_model_name or "audio model"
    if _audio_model is not None:
        try:
            _audio_model.to("cpu")
        except Exception:
            pass
        del _audio_model
        _audio_model = None
    if _audio_processor is not None:
        del _audio_processor
        _audio_processor = None
    _audio_model_name = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"[AceStep SFT] {name} unloaded from VRAM.")


def _prepare_audio_mono(audio_dict, target_sr, max_seconds):
    """Convert audio dict to mono numpy float32 at target sample rate, limited to max_seconds."""
    import numpy as np

    waveform = audio_dict["waveform"]
    sr = audio_dict["sample_rate"]

    if waveform.dim() == 3:
        y = waveform[0].mean(dim=0)
    elif waveform.dim() == 2:
        y = waveform.mean(dim=0)
    else:
        y = waveform
    y = y.cpu().numpy().astype(np.float32)

    if sr != target_sr:
        import librosa
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)

    max_samples = target_sr * max_seconds
    if len(y) > max_samples:
        start = (len(y) - max_samples) // 2
        y = y[start:start + max_samples]

    return y


def _build_gen_kwargs(temperature, top_p, top_k, repetition_penalty, seed):
    """Build a dict of generation kwargs from user-facing parameters."""
    kwargs = {}
    if temperature > 0:
        kwargs["do_sample"] = True
        kwargs["temperature"] = temperature
    else:
        kwargs["do_sample"] = False
    if top_p < 1.0:
        kwargs["top_p"] = top_p
    if top_k > 0:
        kwargs["top_k"] = top_k
    if repetition_penalty != 1.0:
        kwargs["repetition_penalty"] = repetition_penalty
    return kwargs


def _extract_tags(audio_dict, model_key, max_new_tokens=200, audio_duration=30,
                  use_flash_attn=False, gen_kwargs=None):
    """Extract music tags using the selected model.

    Returns a comma-separated string of descriptive tags.
    """
    if gen_kwargs is None:
        gen_kwargs = {}
    model, processor = _load_audio_model(model_key, use_flash_attn=use_flash_attn)

    if _is_acestep_transcriber_model(model_key):
        return _extract_tags_acestep_transcriber(audio_dict, model, processor, max_new_tokens, audio_duration, gen_kwargs)
    elif model_key.startswith("Qwen2.5-Omni") or model_key == "Ke-Omni-R-3B":
        return _extract_tags_qwen_omni(audio_dict, model, processor, max_new_tokens, audio_duration, gen_kwargs)
    elif model_key == "Qwen2-Audio-7B-Instruct":
        return _extract_tags_qwen2_audio(audio_dict, model, processor, max_new_tokens, audio_duration, gen_kwargs)
    elif model_key == "MERT-v1-330M":
        return _extract_tags_mert(audio_dict, model, processor)
    elif _is_whisper_captioning_model(model_key):
        return _extract_tags_whisper_captioning(audio_dict, model, processor, audio_duration, gen_kwargs)
    elif _is_whisper_asr_model(model_key):
        return _extract_tags_whisper_asr(audio_dict, model, processor, max_new_tokens, audio_duration, gen_kwargs)
    elif model_key == "AST-AudioSet":
        return _extract_tags_ast(audio_dict, model, processor)
    return ""


# Simple tag instruction appended to each model's native prompt
_TAG_TEMPLATE_START = "<<<INICIO_TAGS_TEMPLATE>>>"
_TAG_TEMPLATE_END = "<<<FIM_TAGS_TEMPLATE>>>"

_TAG_INSTRUCTION = (
    "Return the result only inside this exact template, with nothing before or after it:\n"
    "<<<INICIO_TAGS_TEMPLATE>>>\n"
    "tag1, tag2, tag3\n"
    "<<<FIM_TAGS_TEMPLATE>>>\n"
    "Inside the template, write only short lowercase comma-separated tags for this audio. "
    "No labels, no explanation, no sentences, no question, no closing text. "
    "Use only the final tags for rhythm, instrumentation, vocals, production effects, mood, and energy. "
    "Use specific tags such as 'punchy kick drum' instead of generic words. Max 4 words per tag."
)


def _extract_tag_template(result_text):
    """Extract the content between explicit tag template markers.

    Handles exact markers and also fuzzy matching for models that
    hallucinate similar but not identical marker strings (e.g.
    <<<TAGS_END_TAGGING>>> instead of <<<FIM_TAGS_TEMPLATE>>>).
    """
    # Truncate at hallucinated conversation turns (e.g. "Human:", "User:", "Assistant:")
    result_text = re.split(r"\n\s*(?:Human|User|Assistant)\s*:", result_text, maxsplit=1, flags=re.IGNORECASE)[0]
    # Strip markdown code blocks if the model wrapped output in ```
    result_text = re.sub(r"```[a-zA-Z]*\n?", "", result_text)
    # Try exact markers first
    start = result_text.find(_TAG_TEMPLATE_START)
    end = result_text.find(_TAG_TEMPLATE_END)
    if start != -1 and end != -1 and end > start:
        start += len(_TAG_TEMPLATE_START)
        return result_text[start:end].strip()
    # Fuzzy: find any <<...>> or <<<...>>> markers (models hallucinate variants)
    markers = list(re.finditer(r"<{2,3}\s*[^>]+\s*>{2,3}", result_text))
    if len(markers) >= 2:
        return result_text[markers[0].end():markers[1].start()].strip()
    # Only one marker found: check if it's a start or end marker
    if len(markers) == 1:
        marker_text = markers[0].group().lower()
        if "inicio" in marker_text or "start" in marker_text or "begin" in marker_text:
            return result_text[markers[0].end():].strip()
        return result_text[:markers[0].start()].strip()
    return result_text.strip()


def _extract_tags_qwen_omni(audio_dict, model, processor, max_new_tokens, audio_duration, gen_kwargs=None):
    """Qwen2.5-Omni tag extraction — single turn, default system prompt."""
    y = _prepare_audio_mono(audio_dict, 16000, audio_duration)

    DEFAULT_SYS = (
        "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
        "capable of perceiving auditory and visual inputs, as well as generating text and speech."
    )
    conversation = [
        {"role": "system", "content": [{"type": "text", "text": DEFAULT_SYS}]},
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": y, "sampling_rate": 16000},
                {"type": "text", "text": _TAG_INSTRUCTION},
            ],
        },
    ]

    text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=text_prompt, audio=[y], sampling_rate=16000, return_tensors="pt", padding=True)
    inputs = inputs.to(model.device).to(model.dtype)
    input_len = inputs["input_ids"].shape[-1]
    gk = {"max_new_tokens": max_new_tokens}
    # return_audio / use_audio_in_video are only valid for full Qwen2.5-Omni (has talker)
    if hasattr(model, "talker"):
        gk["return_audio"] = False
        gk["use_audio_in_video"] = True
    gk.update(gen_kwargs or {})
    if "repetition_penalty" not in gk:
        gk["repetition_penalty"] = 1.5
    text_ids = model.generate(**inputs, **gk)
    new_tokens = text_ids[:, input_len:]
    raw = processor.batch_decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    result = raw[0].strip() if raw else ""
    print(f"[AceStep SFT] Raw model output: {result[:300]}")
    return _clean_tags(_extract_tag_template(result))


def _extract_tags_mert(audio_dict, model, processor):
    """MERT-v1-330M music embedding — returns top activations as tags.

    MERT is an encoder-only model (no generation). We aggregate the
    last hidden layer and return the dimensions with highest activation
    mapped to a small set of predefined music attribute labels.
    """
    import numpy as np

    _MERT_LABELS = [
        "drums", "bass", "guitar", "piano", "synth", "strings", "brass",
        "woodwind", "vocals", "male vocals", "female vocals", "choir",
        "electronic", "acoustic", "distorted", "clean",
        "fast tempo", "slow tempo", "medium tempo",
        "major key", "minor key",
        "happy", "sad", "aggressive", "calm", "dark", "bright",
        "energetic", "mellow", "groovy", "atmospheric",
        "reverb", "delay", "distortion", "compression",
        "kick drum", "snare", "hi hat", "cymbal", "percussion",
        "sub bass", "pad", "lead synth", "arpeggio",
        "pop", "rock", "jazz", "classical", "hip hop", "electronic music",
        "r&b", "folk", "metal", "funk", "latin", "reggae",
    ]

    y = _prepare_audio_mono(audio_dict, 24000, 30)

    inputs = processor(y, sampling_rate=24000, return_tensors="pt")
    inputs = inputs.to(model.device)
    with torch.inference_mode():
        outputs = model(**inputs, output_hidden_states=True)
    # Use last hidden state, average over time
    hidden = outputs.hidden_states[-1]  # [1, T, 1024]
    features = hidden.mean(dim=1).squeeze().cpu().float().numpy()  # [1024]
    # Map top activations to predefined labels (simple heuristic)
    n_labels = len(_MERT_LABELS)
    # Chunk features into n_labels bins, sum each bin
    chunk_size = len(features) // n_labels
    scores = np.array([
        features[i * chunk_size:(i + 1) * chunk_size].sum()
        for i in range(n_labels)
    ])
    # Return labels with highest scores
    top_indices = scores.argsort()[::-1][:15]
    tags = [_MERT_LABELS[i] for i in top_indices if scores[i] > 0]
    if not tags:
        tags = [_MERT_LABELS[int(top_indices[0])]]
    result = ", ".join(tags)
    print(f"[AceStep SFT] MERT tags (heuristic): {result}")
    return result


def _extract_tags_qwen2_audio(audio_dict, model, processor, max_new_tokens, audio_duration, gen_kwargs=None):
    """Qwen2-Audio-7B-Instruct tag extraction."""
    y = _prepare_audio_mono(audio_dict, 16000, audio_duration)

    conversation = [
        {"role": "user", "content": [
            {"type": "audio", "audio_url": "__PLACEHOLDER__"},
            {"type": "text", "text": _TAG_INSTRUCTION},
        ]},
    ]

    text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=text_prompt, audio=[y], sampling_rate=16000, return_tensors="pt", padding=True)
    inputs = inputs.to(model.device).to(model.dtype)
    input_len = inputs["input_ids"].shape[-1]
    # Qwen2-Audio generate only supports max_new_tokens and repetition_penalty
    gk = {"max_new_tokens": max_new_tokens}
    if gen_kwargs and "repetition_penalty" in gen_kwargs:
        gk["repetition_penalty"] = gen_kwargs["repetition_penalty"]
    try:
        text_ids = model.generate(**inputs, **gk)
    except RuntimeError as e:
        if "cu_seqlens" in str(e):
            # flash_attention_2 incompatible with this flash_attn version — fallback to SDPA
            print("[AceStep SFT] flash_attention_2 incompatible with Qwen2-Audio, reloading with SDPA...")
            _unload_audio_model()
            from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
            global _audio_model, _audio_processor, _audio_model_name
            model_dir = _ensure_model_downloaded("Qwen2-Audio-7B-Instruct")
            _audio_model = Qwen2AudioForConditionalGeneration.from_pretrained(
                model_dir, torch_dtype=torch.bfloat16, device_map="auto",
                attn_implementation="sdpa",
            )
            _audio_model.eval()
            _audio_processor = AutoProcessor.from_pretrained(model_dir)
            _audio_model_name = "Qwen2-Audio-7B-Instruct"
            model, processor = _audio_model, _audio_processor
            print(f"[AceStep SFT] Qwen2-Audio-7B-Instruct reloaded with SDPA.")
            # Re-process inputs with new model/processor
            text_prompt2 = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
            inputs = processor(text=text_prompt2, audio=[y], sampling_rate=16000, return_tensors="pt", padding=True)
            inputs = inputs.to(model.device).to(model.dtype)
            input_len = inputs["input_ids"].shape[-1]
            text_ids = model.generate(**inputs, **gk)
        else:
            raise
    new_tokens = text_ids[:, input_len:]
    raw = processor.batch_decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    result = raw[0].strip() if raw else ""
    print(f"[AceStep SFT] Raw model output: {result[:300]}")
    return _clean_tags(_extract_tag_template(result))


# Dataset-name prefixes that Whisper captioning models prepend to output
_WHISPER_PREFIXES = re.compile(
    r"^\s*(audiosetrain|audioset\s*keywords?|clotho|audiocaps|music\s*role)\s*[,:]?\s*",
    re.IGNORECASE,
)

_TRANSCRIPT_LANGUAGE_PATTERNS = [
    ("japanese", re.compile(r"[\u3040-\u30ff]")),
    ("chinese", re.compile(r"[\u4e00-\u9fff]")),
    ("korean", re.compile(r"[\uac00-\ud7af]")),
    ("russian", re.compile(r"[\u0400-\u04ff]")),
    ("arabic", re.compile(r"[\u0600-\u06ff]")),
]

_TRANSCRIPT_LANGUAGE_HINTS = {
    "english": {"the", "and", "you", "love", "with", "baby", "night", "heart"},
    "portuguese": {"que", "você", "amor", "pra", "não", "meu", "minha", "coração"},
    "spanish": {"que", "amor", "corazón", "noche", "eres", "tengo", "para", "con"},
    "french": {"je", "tu", "amour", "avec", "pas", "dans", "pour", "coeur"},
    "german": {"ich", "du", "und", "nicht", "liebe", "nacht", "mein", "mit"},
    "italian": {"che", "amore", "notte", "con", "sei", "mio", "mia", "cuore"},
}

_LANGUAGE_CODE_TO_NAME = {
    "ar": "arabic",
    "de": "german",
    "el": "greek",
    "en": "english",
    "es": "spanish",
    "fa": "persian",
    "fr": "french",
    "he": "hebrew",
    "hi": "hindi",
    "id": "indonesian",
    "it": "italian",
    "ja": "japanese",
    "ko": "korean",
    "ms": "malay",
    "nl": "dutch",
    "pl": "polish",
    "pt": "portuguese",
    "ru": "russian",
    "th": "thai",
    "tl": "filipino",
    "tr": "turkish",
    "uk": "ukrainian",
    "ur": "urdu",
    "vi": "vietnamese",
    "zh": "chinese",
}


def _infer_transcript_language(text):
    for language_name, pattern in _TRANSCRIPT_LANGUAGE_PATTERNS:
        if pattern.search(text):
            return language_name

    tokens = re.findall(r"[a-zA-ZÀ-ÿ']+", text.lower())
    if not tokens:
        return ""

    token_set = set(tokens)
    best_language = ""
    best_score = 0
    for language_name, hints in _TRANSCRIPT_LANGUAGE_HINTS.items():
        score = len(token_set & hints)
        if score > best_score:
            best_language = language_name
            best_score = score
    return best_language if best_score >= 2 else ""


def _derive_tags_from_transcript(transcript_text, audio_duration):
    transcript_text = transcript_text.strip()
    words = re.findall(r"[a-zA-ZÀ-ÿ0-9']+", transcript_text.lower())
    if len(words) < 3:
        return "instrumental, no clear vocals"

    tags = ["vocals"]
    language_name = _infer_transcript_language(transcript_text)
    if language_name:
        tags.append(f"{language_name} vocals")

    effective_duration = max(float(audio_duration), 1.0)
    word_density = len(words) / effective_duration
    lexical_diversity = len(set(words)) / max(len(words), 1)

    if word_density >= 3.0:
        tags.extend(["fast vocals", "rap-like flow", "lyrical"])
    elif word_density >= 1.6:
        tags.extend(["sung vocals", "lyrical", "clear vocals"])
    else:
        tags.extend(["sparse vocals", "melodic vocals"])

    if lexical_diversity < 0.6:
        tags.append("repeated hook")
    if len(words) >= 40:
        tags.append("storytelling lyrics")
    if any(token in {"yeah", "oh", "la", "na", "hey", "woo"} for token in words):
        tags.append("hook vocals")

    return _clean_tags(", ".join(tags))


def _parse_acestep_transcription(result_text):
    result_text = re.split(r"\n\s*(?:Human|User|Assistant)\s*:", result_text, maxsplit=1, flags=re.IGNORECASE)[0]
    result_text = re.sub(r"```[a-zA-Z]*\n?", "", result_text)
    result_text = result_text.replace("```", "").strip()

    language_match = re.search(r"#\s*Languages\s*\n+(.+?)(?=\n#|\Z)", result_text, flags=re.IGNORECASE | re.DOTALL)
    language_block = language_match.group(1).strip() if language_match else ""
    language_code = next((line.strip().lower() for line in language_block.splitlines() if line.strip()), "")

    lyrics_match = re.search(r"#\s*Lyrics\s*\n+(.+?)(?=\n#\s*[A-Za-z]|\Z)", result_text, flags=re.IGNORECASE | re.DOTALL)
    lyrics_block = lyrics_match.group(1).strip() if lyrics_match else result_text

    section_tags = []
    instrument_tags = []
    lyric_lines = []
    for line in lyrics_block.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("[") and stripped.endswith("]"):
            inside = stripped[1:-1].strip()
            parts = [part.strip() for part in inside.split("-", 1)]
            section_name = re.sub(r"\s+\d+$", "", parts[0].lower())
            section_tags.append(section_name)
            if len(parts) > 1 and parts[1]:
                instrument_tags.append(parts[1].lower())
            continue
        lyric_lines.append(stripped)

    lyrics_text = "\n".join(lyric_lines).strip()
    return {
        "language_code": language_code,
        "lyrics_text": lyrics_text,
        "section_tags": section_tags,
        "instrument_tags": instrument_tags,
    }


def _derive_tags_from_acestep_transcription(result_text, audio_duration):
    parsed = _parse_acestep_transcription(result_text)
    section_tags = parsed["section_tags"]
    instrument_tags = parsed["instrument_tags"]
    lyrics_text = parsed["lyrics_text"]
    language_code = parsed["language_code"]

    tags = []
    if lyrics_text:
        tags.extend([tag.strip() for tag in _derive_tags_from_transcript(lyrics_text, audio_duration).split(",") if tag.strip()])
    else:
        tags.extend(["instrumental", "no clear vocals"])

    language_name = _LANGUAGE_CODE_TO_NAME.get(language_code, language_code)
    if language_name:
        tags.append(f"{language_name} lyrics")

    normalized_sections = []
    for section in section_tags:
        cleaned = re.sub(r"\s+", " ", section).strip()
        normalized_sections.append(cleaned)

    if any("verse" in section for section in normalized_sections):
        tags.append("verse structure")
    if any("chorus" in section for section in normalized_sections):
        tags.append("chorus structure")
    if any("bridge" in section for section in normalized_sections):
        tags.append("bridge section")
    if any("intro" in section for section in normalized_sections):
        tags.append("intro section")
    if any("outro" in section for section in normalized_sections):
        tags.append("outro section")
    if any("spoken" in section for section in normalized_sections):
        tags.append("spoken section")
    if any("instrumental" in section or "interlude" in section for section in normalized_sections):
        tags.append("instrumental section")
    if normalized_sections and len(set(normalized_sections)) >= 2:
        tags.append("structured song form")

    for instrument in instrument_tags[:4]:
        instrument = re.sub(r"\s+", " ", instrument).strip()
        if instrument:
            tags.append(instrument)

    return _clean_tags(", ".join(tags))


def _extract_tags_whisper_captioning(audio_dict, model, processor, audio_duration, gen_kwargs=None):
    """Whisper audio captioning tag extraction (MU-NLPC models)."""
    y = _prepare_audio_mono(audio_dict, 16000, audio_duration)

    inputs = processor(y, sampling_rate=16000, return_tensors="pt")
    inputs = inputs.to(model.device)
    gk = {"max_new_tokens": 200}
    gk.update(gen_kwargs or {})
    with torch.inference_mode():
        gen = model.generate(**inputs, **gk)
    result = processor.batch_decode(gen, skip_special_tokens=True)
    text = result[0] if result else ""
    # Strip dataset-name prefixes the models hallucinate
    text = _WHISPER_PREFIXES.sub("", text)
    return _clean_tags(_extract_tag_template(text))


def _extract_tags_acestep_transcriber(audio_dict, model, processor, max_new_tokens, audio_duration, gen_kwargs=None):
    """ACE-Step transcriber extraction -> lyric, structure and vocal tags."""
    y = _prepare_audio_mono(audio_dict, 16000, audio_duration)

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "*Task* Transcribe this audio in detail"},
                {"type": "audio", "audio": y, "sampling_rate": 16000},
            ],
        },
    ]

    text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=text_prompt, audio=[y], sampling_rate=16000, return_tensors="pt", padding=True)
    inputs = inputs.to(model.device).to(model.dtype)
    input_len = inputs["input_ids"].shape[-1]
    gk = {
        "max_new_tokens": max(256, max_new_tokens),
        "return_audio": False,
    }
    gk.update(gen_kwargs or {})
    if "repetition_penalty" not in gk:
        gk["repetition_penalty"] = 1.1
    text_ids = model.generate(**inputs, **gk)
    new_tokens = text_ids[:, input_len:]
    raw = processor.batch_decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    result = raw[0].strip() if raw else ""
    print(f"[AceStep SFT] ACE-Step transcription: {result[:400]}")
    return _derive_tags_from_acestep_transcription(result, audio_duration)


def _extract_tags_whisper_asr(audio_dict, model, processor, max_new_tokens, audio_duration, gen_kwargs=None):
    """Whisper ASR transcription -> vocal tags heuristic extraction."""
    y = _prepare_audio_mono(audio_dict, 16000, audio_duration)

    inputs = processor(y, sampling_rate=16000, return_tensors="pt")
    input_features = inputs["input_features"].to(model.device)
    if torch.is_floating_point(input_features):
        input_features = input_features.to(dtype=model.dtype)

    gk = {"max_new_tokens": max(64, min(max_new_tokens, 256))}
    if hasattr(getattr(model, "generation_config", None), "task_to_id"):
        gk["task"] = "transcribe"
    gk.update(gen_kwargs or {})

    with torch.inference_mode():
        generated_ids = model.generate(input_features=input_features, **gk)
    result = processor.batch_decode(generated_ids, skip_special_tokens=True)
    transcript_text = result[0].strip() if result else ""
    print(f"[AceStep SFT] Whisper transcript: {transcript_text[:300]}")
    return _derive_tags_from_transcript(transcript_text, audio_duration)


def _extract_tags_ast(audio_dict, model, processor):
    """AST AudioSet classification → top tags."""
    import numpy as np

    y = _prepare_audio_mono(audio_dict, 16000, 30)

    inputs = processor(y, sampling_rate=16000, return_tensors="pt")
    inputs = inputs.to(model.device)
    with torch.inference_mode():
        logits = model(**inputs).logits[0]
    probs = torch.sigmoid(logits)
    top_indices = probs.argsort(descending=True)[:15].cpu().numpy()
    labels = model.config.id2label
    tags = [labels[int(i)] for i in top_indices if probs[i] > 0.1]
    if not tags:
        tags = [labels[int(top_indices[0])]]
    return ", ".join(tags)


def _clean_tags(result_text):
    """Clean model output into short, deduplicated, lowercase tags."""
    result_text = result_text.strip().strip('"').strip("'").strip()
    result_text = result_text.replace("\uff0c", ",").replace("\u3001", ",")
    lines = [ln.strip() for ln in result_text.splitlines() if ln.strip()]
    result_text = ", ".join(lines) if lines else ""

    seen = set()
    unique_tags = []
    for tag in result_text.split(","):
        tag = tag.strip().rstrip(".")
        # Remove numbered prefixes like "1)" or "1."
        tag = re.sub(r"^\d+[).\]]\s*", "", tag).strip()
        # Normalize dashes ("drum - beat" → "drum beat")
        tag = re.sub(r"\s*-\s*", " ", tag).strip()
        if not tag:
            continue
        # Skip BPM numbers (detected separately by librosa)
        if re.match(r"^\d+\s*bpm$", tag, re.IGNORECASE):
            continue
        # Skip filler words like "etc", "and more", "..."
        if tag in ("etc", "and more", "more", "and so on", "..."):
            continue
        # Skip verbose entries (>6 words) — real tags are short
        if len(tag.split()) > 6:
            continue
        tag = tag.lower()
        if tag not in seen:
            seen.add(tag)
            unique_tags.append(tag)
    # Cap at 20 tags to avoid bloated output
    return ", ".join(unique_tags[:20])


_TURBO_SFT_DIRECT_MAPS = [
    (r"\bbrazilian funk mandel[aã]o\b", ["brazilian funk mandelao", "tamborzao beat", "baile funk"]),
    (r"\britualistic phonk atmosphere\b", ["dark phonk", "phonk cowbell", "occult ambience"]),
    (r"\bheavy distorted 808 bass\b", ["distorted 808", "heavy sub bass", "saturated bass"]),
    (r"\britual drums?\b", ["tribal drums", "ritual percussion"]),
    (r"\boccult dark vibes?\b", ["occult mood", "dark ambience"]),
    (r"\bhigh energy dance rhythm\b", ["dance groove", "high energy", "club rhythm"]),
    (r"\bfast bpm\b", ["fast tempo"]),
    (r"\bmale vocal with reverb\b", ["male vocals", "reverb vocals"]),
    (r"\bfemale vocal with reverb\b", ["female vocals", "reverb vocals"]),
    (r"\bexplicit lyrics\b", ["explicit lyrics"]),
    (r"\bgritty synth stabs\b", ["gritty synth", "synth stabs"]),
    (r"\bdark club aesthetic\b", ["dark club", "underground club"]),
]

_TURBO_SFT_WORD_REPLACEMENTS = {
    "atmosphere": "ambience",
    "atmospheric": "atmospheric",
    "vibes": "mood",
    "vibe": "mood",
    "aesthetic": "style",
    "bpm": "tempo",
    "vocal": "vocals",
    "vocalss": "vocals",
    "vocoder": "vocoder",
    "guitars": "guitar",
    "bassline": "bass",
    "pads": "pad",
    "stabs": "stabs",
    "motifs": "motif",
    "ritualistic": "ritual",
    "melodic": "melodic",
    "cinematic": "cinematic",
    "orchestral": "orchestral",
    "distorted": "distorted",
}

_TURBO_SFT_DROP_WORDS = {
    "a",
    "an",
    "and",
    "the",
    "with",
    "very",
    "extremely",
    "super",
}

_TURBO_SFT_GENRE_TERMS = {
    "funk", "mandelao", "phonk", "trap", "drill", "house", "techno", "trance", "edm",
    "dubstep", "dnb", "drum", "bass", "ambient", "cinematic", "orchestral", "rock", "metal",
    "punk", "pop", "jazz", "blues", "soul", "rnb", "hip hop", "hiphop", "rap", "reggaeton",
    "reggae", "dancehall", "afrobeats", "afrobeat", "latin", "salsa", "baile", "funky", "lofi",
    "lo-fi", "gospel", "choir", "folk", "country", "classical", "industrial", "electro", "synthwave",
    "hardstyle", "hardcore", "garage", "ukg", "grime", "boom bap", "bossa", "samba", "pagode",
}

_TURBO_SFT_HEAD_TERMS = {
    "bass", "808", "sub", "drums", "percussion", "kick", "snare", "clap", "hihat", "hat",
    "cowbell", "synth", "stabs", "pad", "arp", "arpeggio", "lead", "guitar", "piano", "keys",
    "strings", "brass", "flute", "organ", "choir", "vocals", "lyrics", "hook", "groove", "rhythm",
    "tempo", "ambience", "mood", "energy", "club", "style", "beat", "drop", "reverb", "delay",
    "distortion", "compression", "sidechain", "texture",
}

_TURBO_SFT_MOOD_TERMS = {
    "dark", "bright", "aggressive", "melancholic", "sad", "happy", "euphoric", "dreamy", "tense",
    "ritual", "occult", "romantic", "epic", "emotional", "mellow", "gritty", "warm", "cold",
    "atmospheric", "cinematic", "uplifting", "moody", "hypnotic", "haunting", "mystic",
}

_TURBO_SFT_PRODUCTION_TERMS = {
    "distorted", "saturated", "clean", "punchy", "wide", "dry", "reverb", "reverbed", "wet",
    "compressed", "sidechained", "lofi", "lo-fi", "glitchy", "crunchy", "noisy", "layered", "gritty",
}

_TURBO_SFT_TEMPO_TERMS = {
    "fast", "slow", "midtempo", "uptempo", "halftime", "doubletime", "driving", "rolling", "syncopated",
    "dance", "groove", "swing", "club", "bouncy",
}

_TURBO_SFT_VOCAL_TERMS = {
    "male", "female", "vocoder", "spoken", "rap", "rapped", "chant", "choir", "hook", "vocal",
    "vocals", "whispered", "shouted", "lyrics", "instrumental", "explicit",
}


def _split_freeform_tags(tag_text):
    parts = re.split(r"[,;\n]+", tag_text or "")
    return [part.strip() for part in parts if part.strip()]


def _dedupe_preserve_order(items):
    seen = set()
    output = []
    for item in items:
        if item not in seen:
            seen.add(item)
            output.append(item)
    return output


def _remove_redundant_subset_tags(tags):
    output = []
    lowered = [tag.lower() for tag in tags]
    for index, tag in enumerate(tags):
        parts = lowered[index].split()
        is_subset = False
        for other_index, other_tag in enumerate(lowered):
            if index == other_index:
                continue
            other_parts = other_tag.split()
            if len(other_parts) > len(parts) and all(part in other_parts for part in parts):
                is_subset = True
                break
        if not is_subset:
            output.append(tag)
    return output


def _turbo_sft_tag_priority(tag):
    high_priority_terms = [
        "funk", "mandelao", "phonk", "cowbell", "808", "bass", "sub bass",
        "drums", "percussion", "beat", "groove", "tempo", "vocals", "lyrics",
        "synth", "stabs",
    ]
    medium_priority_terms = [
        "club", "energy", "dark", "occult", "ambience", "mood", "style",
    ]

    if any(term in tag for term in high_priority_terms):
        return 0
    if any(term in tag for term in medium_priority_terms):
        return 1
    return 2


def _normalize_turbo_tag_words(tag):
    normalized = tag.lower().strip()
    normalized = normalized.replace("–", " ").replace("—", " ")
    normalized = re.sub(r"\bmale vocal\b", "male vocals", normalized)
    normalized = re.sub(r"\bfemale vocal\b", "female vocals", normalized)
    normalized = re.sub(r"\bhigh bpm\b", "fast tempo", normalized)
    normalized = re.sub(r"\blow bpm\b", "slow tempo", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    words = re.sub(r"[^a-z0-9#+\s]", " ", normalized)
    words = [word for word in words.split() if word not in _TURBO_SFT_DROP_WORDS]
    return [_TURBO_SFT_WORD_REPLACEMENTS.get(word, word) for word in words]


def _extract_generic_sft_tags_from_words(words):
    if not words:
        return []

    tags = []
    word_set = set(words)

    genre_words = [word for word in words if word in _TURBO_SFT_GENRE_TERMS]
    if genre_words:
        genre_tag = " ".join(_dedupe_preserve_order(genre_words[:3]))
        tags.append(genre_tag)

    if "808" in word_set and "distorted" in word_set:
        tags.append("distorted 808")
    elif "808" in word_set and "bass" in word_set:
        tags.append("808 bass")

    if "sub" in word_set and "bass" in word_set:
        tags.append("sub bass")
    elif "bass" in word_set:
        bass_adjectives = [word for word in words if word in _TURBO_SFT_PRODUCTION_TERMS or word in _TURBO_SFT_MOOD_TERMS]
        if bass_adjectives:
            tags.append(f"{bass_adjectives[0]} bass")
        else:
            tags.append("bass")

    if any(word in word_set for word in ["drums", "percussion", "kick", "snare", "clap", "hihat", "hat"]):
        drum_focus = [word for word in words if word in ["drums", "percussion", "kick", "snare", "clap", "hihat", "hat", "cowbell"]]
        drum_tag = " ".join(_dedupe_preserve_order(drum_focus[:2]))
        if drum_tag:
            tags.append(drum_tag)

    if any(word in word_set for word in ["synth", "stabs", "pad", "arp", "lead", "keys", "piano", "guitar", "strings", "brass", "flute", "organ"]):
        instrument_focus = [
            word for word in words
            if word in _TURBO_SFT_PRODUCTION_TERMS or word in ["synth", "stabs", "pad", "arp", "lead", "keys", "piano", "guitar", "strings", "brass", "flute", "organ"]
        ]
        instrument_tag = " ".join(_dedupe_preserve_order(instrument_focus[:2]))
        if instrument_tag:
            tags.append(instrument_tag)

    if "male" in word_set and "vocals" in word_set:
        tags.append("male vocals")
    elif "female" in word_set and "vocals" in word_set:
        tags.append("female vocals")
    elif "vocals" in word_set or "vocal" in word_set:
        tags.append("vocals")

    if "reverb" in word_set and any(word in word_set for word in ["vocals", "vocal", "male", "female"]):
        tags.append("reverb vocals")
    elif "reverb" in word_set:
        tags.append("reverb")

    if "explicit" in word_set and "lyrics" in word_set:
        tags.append("explicit lyrics")
    elif "instrumental" in word_set:
        tags.append("instrumental")
    elif "lyrics" in word_set:
        tags.append("lyrics")

    if "fast" in word_set and "tempo" in word_set:
        tags.append("fast tempo")
    elif "slow" in word_set and "tempo" in word_set:
        tags.append("slow tempo")
    elif any(word in word_set for word in ["dance", "groove", "driving", "rolling", "swing", "bouncy"]):
        groove_focus = [word for word in words if word in ["dance", "groove", "driving", "rolling", "swing", "bouncy", "club"]]
        groove_tag = " ".join(_dedupe_preserve_order(groove_focus[:2]))
        if groove_tag:
            tags.append(groove_tag)

    mood_focus = [word for word in words if word in _TURBO_SFT_MOOD_TERMS]
    if mood_focus:
        mood_tag = " ".join(_dedupe_preserve_order(mood_focus[:2]))
        tags.append(mood_tag)

    production_focus = [word for word in words if word in _TURBO_SFT_PRODUCTION_TERMS]
    if production_focus:
        prod_tag = " ".join(_dedupe_preserve_order(production_focus[:2]))
        tags.append(prod_tag)

    return _dedupe_preserve_order([tag for tag in tags if tag])


def _generic_compact_turbo_phrase(tag):
    words = _normalize_turbo_tag_words(tag)
    generic_tags = _extract_generic_sft_tags_from_words(words)
    if generic_tags:
        return generic_tags, "generic"

    head_positions = [idx for idx, word in enumerate(words) if word in _TURBO_SFT_HEAD_TERMS or word in _TURBO_SFT_GENRE_TERMS]
    if head_positions:
        head_idx = head_positions[-1]
        start_idx = max(0, head_idx - 2)
        compact = " ".join(words[start_idx:head_idx + 1]).strip()
        if compact:
            return [compact], "compacted"

    compact = " ".join(words[:4]).strip()
    if compact:
        return [compact], "kept"
    return [], "dropped"


def _simplify_turbo_tag_for_sft(tag):
    normalized = tag.lower().strip()
    normalized = normalized.replace("–", " ").replace("—", " ")
    normalized = re.sub(r"\s+", " ", normalized)

    for pattern, replacements in _TURBO_SFT_DIRECT_MAPS:
        if re.search(pattern, normalized, flags=re.IGNORECASE):
            return replacements, "mapped"

    return _generic_compact_turbo_phrase(tag)


def _adapt_turbo_tags_for_sft(tag_text, adaptation_strength="balanced", keep_unknown_tags=True, add_sft_bias_tags=True):
    source_tags = _split_freeform_tags(tag_text)
    adapted = []
    mapped_count = 0
    simplified_count = 0
    max_replacements_per_tag = {
        "conservative": 1,
        "balanced": 2,
        "aggressive": 3,
    }.get(adaptation_strength, 2)

    for source_tag in source_tags:
        replacements, mode = _simplify_turbo_tag_for_sft(source_tag)
        if replacements:
            adapted.extend(replacements[:max_replacements_per_tag])
            if mode == "mapped":
                mapped_count += 1
            elif mode in ("simplified", "expanded"):
                simplified_count += 1
        elif keep_unknown_tags:
            cleaned = _clean_tags(source_tag)
            if cleaned:
                adapted.extend(_split_freeform_tags(cleaned))

    lowered_source = (tag_text or "").lower()
    if add_sft_bias_tags:
        bias_tags = []
        if any(keyword in lowered_source for keyword in ["funk", "mandelao", "baile"]):
            bias_tags.extend(["favela funk", "mc vocals"])
        if "phonk" in lowered_source:
            bias_tags.extend(["dark cowbell", "drift phonk"])
        if any(keyword in lowered_source for keyword in ["club", "dance", "high energy"]):
            bias_tags.extend(["club energy", "driving groove"])
        if any(keyword in lowered_source for keyword in ["dark", "occult", "ritual"]):
            bias_tags.extend(["dark mood", "tense ambience"])
        if any(keyword in lowered_source for keyword in ["orchestral", "cinematic", "strings", "brass"]):
            bias_tags.extend(["wide arrangement", "cinematic tension"])
        if any(keyword in lowered_source for keyword in ["ambient", "dreamy", "ethereal"]):
            bias_tags.extend(["spacious ambience", "slow evolution"])
        if any(keyword in lowered_source for keyword in ["rock", "metal", "guitar", "live drums"]):
            bias_tags.extend(["live band feel", "driving drums"])

        if adaptation_strength == "conservative":
            bias_tags = bias_tags[:2]
        elif adaptation_strength == "balanced":
            bias_tags = bias_tags[:4]

        adapted.extend(bias_tags)

    adapted = _dedupe_preserve_order(_split_freeform_tags(_clean_tags(", ".join(adapted))))
    adapted = _remove_redundant_subset_tags(adapted)
    adapted = [
        tag for _, _, tag in sorted(
            [(_turbo_sft_tag_priority(tag), idx, tag) for idx, tag in enumerate(adapted)]
        )
    ]

    if adaptation_strength == "conservative":
        adapted = adapted[:10]
        suggested_cfg = 6.5
        suggested_steps = 50
    elif adaptation_strength == "aggressive":
        adapted = adapted[:18]
        suggested_cfg = 7.5
        suggested_steps = 56
    else:
        adapted = adapted[:14]
        suggested_cfg = 7.0
        suggested_steps = 50

    notes = (
        f"Converted {len(source_tags)} Turbo tags into {len(adapted)} SFT-oriented tags; "
        f"direct maps: {mapped_count}, simplified/expanded: {simplified_count}. "
        "Turbo and SFT do not share the same text-conditioning space, so equal seed/config does not guarantee the same song. "
        "This adapter shifts abstract Turbo phrasing toward shorter SFT-friendly genre, rhythm, timbre, vocal, mood, and production tags."
    )
    return ", ".join(adapted), notes, suggested_cfg, suggested_steps


def _detect_bpm_keyscale(audio_dict):
    """Detect BPM and key/scale using librosa signal processing."""
    try:
        import librosa
        import numpy as np
    except ImportError:
        return {"bpm": 0, "keyscale": ""}

    waveform = audio_dict["waveform"]
    sr = audio_dict["sample_rate"]

    if waveform.dim() == 3:
        y = waveform[0].mean(dim=0)
    elif waveform.dim() == 2:
        y = waveform.mean(dim=0)
    else:
        y = waveform
    y = y.cpu().numpy().astype(np.float32)

    target_sr = 22050
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    if isinstance(tempo, np.ndarray):
        tempo = float(tempo[0])
    detected_bpm = int(round(tempo))

    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                              2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                              2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    pitch_names = ["C", "C#", "D", "D#", "E", "F",
                   "F#", "G", "G#", "A", "A#", "B"]

    chromagram = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_vals = chromagram.mean(axis=1)

    best_corr = -2.0
    best_key = "C"
    best_scale = "major"
    for i in range(12):
        maj_corr = float(np.corrcoef(chroma_vals, np.roll(major_profile, -i))[0, 1])
        min_corr = float(np.corrcoef(chroma_vals, np.roll(minor_profile, -i))[0, 1])
        if maj_corr > best_corr:
            best_corr = maj_corr
            best_key = pitch_names[i]
            best_scale = "major"
        if min_corr > best_corr:
            best_corr = min_corr
            best_key = pitch_names[i]
            best_scale = "minor"

    return {"bpm": detected_bpm, "keyscale": f"{best_key} {best_scale}"}


# ---------------------------------------------------------------------------
# Music Analyzer Node
# ---------------------------------------------------------------------------

class AceStepSFTMusicAnalyzer:
    """Analyzes audio to extract descriptive tags, BPM and key/scale.

    Tags are extracted using the native ACE-Step transcriber.
    BPM and key/scale are detected via librosa signal processing.
    Outputs can be wired to the Generate node or to text display nodes.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {
                    "tooltip": "Audio to analyze for style, BPM and key/scale.",
                }),
                "get_tags": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Extract descriptive tags from the audio using the native ACE-Step transcriber.",
                }),
                "get_bpm": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Detect BPM from audio using librosa.",
                }),
                "get_keyscale": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Detect key and scale from audio using librosa.",
                }),
            },
            "optional": {
                "max_new_tokens": ("INT", {
                    "default": 256, "min": 64, "max": 2000, "step": 16,
                    "tooltip": "Maximum tokens for the native ACE-Step transcription output. Higher values preserve more lyric/structure detail.",
                }),
                "audio_duration": ("INT", {
                    "default": 60, "min": 10, "max": 300, "step": 5,
                    "tooltip": "Max seconds of audio to analyze (center crop). ACE-Step Transcriber benefits from more context for sections and lyrics.",
                }),
                "unload_model": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Unload the ACE-Step transcriber after use to free VRAM for generation.",
                }),
                "use_flash_attn": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Use FlashAttention-2 for the ACE-Step transcriber. Requires flash-attn package installed. Faster and uses less VRAM.",
                }),
                "temperature": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Sampling temperature for transcription generation. 0 = deterministic and recommended for stable structure extraction.",
                }),
                "top_p": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Nucleus sampling for transcription generation. Keep at 1.0 for native deterministic behavior unless you are experimenting.",
                }),
                "top_k": ("INT", {
                    "default": 0, "min": 0, "max": 200, "step": 1,
                    "tooltip": "Top-K sampling for transcription generation. 0 preserves the native default behavior.",
                }),
                "repetition_penalty": ("FLOAT", {
                    "default": 1.1, "min": 1.0, "max": 3.0, "step": 0.05,
                    "tooltip": "Penalty against repeated tokens in the transcription output. 1.1 is a mild safeguard against loops.",
                }),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xffffffffffffffff,
                    "control_after_generate": True,
                    "tooltip": "Random seed for reproducible transcription generation when sampling is enabled.",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "INT", "STRING", "STRING")
    RETURN_NAMES = ("tags", "bpm", "keyscale", "music_infos")
    FUNCTION = "analyze"
    CATEGORY = "audio/AceStep SFT"
    DESCRIPTION = (
        "Analyzes audio with the native ACE-Step Transcriber to extract lyric, vocal and song-structure tags, plus BPM and key/scale via librosa. "
        "Wire outputs to Generate node or to text display nodes to inspect results."
    )

    def analyze(self, audio, get_tags, get_bpm, get_keyscale,
                max_new_tokens=256, audio_duration=60, unload_model=True, use_flash_attn=False,
                temperature=0.0, top_p=1.0, top_k=0, repetition_penalty=1.1, seed=0):
        tags = ""
        detected_bpm = 0
        keyscale = ""
        model = _NATIVE_ANALYSIS_MODEL

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        gen_kwargs = _build_gen_kwargs(temperature, top_p, top_k, repetition_penalty, seed)

        if get_tags:
            try:
                tags = _extract_tags(audio, model, max_new_tokens, audio_duration,
                                     use_flash_attn=use_flash_attn, gen_kwargs=gen_kwargs)
                print(f"[AceStep SFT] Extracted tags: {tags}")
            except Exception as e:
                print(f"[AceStep SFT] Tag extraction failed: {e}")

        if get_bpm or get_keyscale:
            try:
                dsp = _detect_bpm_keyscale(audio)
                if get_bpm:
                    detected_bpm = dsp["bpm"]
                if get_keyscale:
                    keyscale = dsp["keyscale"]
                print(f"[AceStep SFT] Detected BPM: {dsp['bpm']} | Key: {dsp['keyscale']}")
            except Exception as e:
                print(f"[AceStep SFT] librosa detection failed: {e}")

        if unload_model and get_tags:
            _unload_audio_model()

        import json
        music_infos = json.dumps({
            "tags": tags,
            "bpm": f"{detected_bpm}bpm",
            "keyscale": keyscale,
        }, ensure_ascii=False, indent=4)

        return (tags, detected_bpm, keyscale, music_infos)


# ---------------------------------------------------------------------------
# Model Loader Node
# ---------------------------------------------------------------------------

class AceStepSFTModelLoader:
    """Loads the AceStep 1.5 SFT diffusion model, CLIP encoders, and VAE.

    Outputs MODEL, CLIP, and VAE that can be connected to the
    Lora Loader, TextEncode, and Generate nodes.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
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
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("model", "clip", "vae")
    FUNCTION = "load_model"
    CATEGORY = "audio/AceStep SFT"
    DESCRIPTION = (
        "Loads the AceStep 1.5 SFT diffusion model, dual CLIP text encoders, "
        "and audio VAE. Connect MODEL to Generate (or Lora Loader first), "
        "CLIP to TextEncode, and VAE to Generate."
    )

    def load_model(self, diffusion_model, text_encoder_1, text_encoder_2, vae_name):
        # --- Load diffusion model ---
        unet_path = folder_paths.get_full_path_or_raise(
            "diffusion_models", diffusion_model
        )
        loaded_model = comfy.sd.load_diffusion_model(unet_path)
        loaded_model.model.eval()

        # --- Load CLIP encoders ---
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
        clip.cond_stage_model.eval()

        # --- Load VAE ---
        vae_path = folder_paths.get_full_path_or_raise("vae", vae_name)
        vae_sd = comfy.utils.load_torch_file(vae_path)
        loaded_vae = comfy.sd.VAE(sd=vae_sd)
        loaded_vae.first_stage_model.eval()

        return (loaded_model, clip, loaded_vae)


# ---------------------------------------------------------------------------
# LoRA Loader Node
# ---------------------------------------------------------------------------

class AceStepSFTLoraLoader:
    """Applies a LoRA to the AceStep 1.5 SFT model and CLIP.

    Takes MODEL and CLIP from the Model Loader (or a previous Lora Loader),
    applies the LoRA, and outputs the modified MODEL and CLIP.
    Multiple Lora Loader nodes can be chained.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {
                    "tooltip": "MODEL from Model Loader or previous Lora Loader.",
                }),
                "clip": ("CLIP", {
                    "tooltip": "CLIP from Model Loader or previous Lora Loader.",
                }),
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
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    RETURN_NAMES = ("model", "clip")
    FUNCTION = "load_lora"
    CATEGORY = "audio/AceStep SFT"
    DESCRIPTION = (
        "Applies a LoRA to the AceStep 1.5 SFT model and CLIP. "
        "Chain multiple Lora Loader nodes before connecting to Generate/TextEncode."
    )

    def load_lora(self, model, clip, lora_name, strength_model, strength_clip):
        lora_path = folder_paths.get_full_path_or_raise(
            "loras", lora_name
        )
        lora_data = comfy.utils.load_torch_file(lora_path, safe_load=True)
        # Fix DoRA scale dimensions for AceStep compatibility
        for k in list(lora_data.keys()):
            if k.endswith(".dora_scale") and lora_data[k].dim() == 1:
                lora_data[k] = lora_data[k].unsqueeze(-1)
        model, clip = comfy.sd.load_lora_for_models(
            model, clip, lora_data, strength_model, strength_clip
        )
        return (model, clip)


# ---------------------------------------------------------------------------
# Main Node
# ---------------------------------------------------------------------------

class AceStepSFTTextEncode:
    """Text encoder node for AceStep 1.5 SFT.

    Encodes caption + lyrics + metadata into positive and negative
    conditioning for the Generate node.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP", {
                    "tooltip": "CLIP model (loaded via DualCLIPLoaderAudio or similar).",
                }),
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
                    "tooltip": "Force instrumental mode (overrides lyrics with [Instrumental]).",
                }),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xffffffffffffffff,
                    "control_after_generate": True,
                }),
                "duration": ("FLOAT", {
                    "default": 60.0, "min": 0.0, "max": 600.0, "step": 0.1,
                    "tooltip": "Duration in seconds. Set to 0 for auto duration from lyrics.",
                }),
                "bpm": ("INT", {
                    "default": 0, "min": 0, "max": 300,
                    "tooltip": "Beats per minute. 0 = auto (N/A, let model decide).",
                }),
                "timesignature": (['auto', '4', '3', '2', '6'], {
                    "default": 'auto',
                    "tooltip": "Time signature numerator. 'auto' = let model decide.",
                }),
                "language": (LANGUAGES, {
                    "default": "en",
                    "tooltip": "Language tag for lyrics conditioning.",
                }),
                "keyscale": (["auto"] + KEYSCALES_LIST, {
                    "default": "auto",
                    "tooltip": "Key and scale. 'auto' = let model decide.",
                }),
            },
            "optional": {
                "generate_audio_codes": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable LLM audio code generation for semantic structure.",
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
                # ---- Style overrides (from Music Analyzer node) ----
                "style_tags": ("STRING", {
                    "default": "",
                    "forceInput": True,
                    "tooltip": "Tags from the Music Analyzer node. Appended to caption when connected.",
                }),
                "style_bpm": ("INT", {
                    "default": 0, "min": 0, "max": 300,
                    "forceInput": True,
                    "tooltip": "BPM from the Music Analyzer node. Overrides bpm when > 0.",
                }),
                "style_keyscale": ("STRING", {
                    "default": "",
                    "forceInput": True,
                    "tooltip": "Key/scale from the Music Analyzer node. Overrides keyscale when not empty.",
                }),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "encode"
    CATEGORY = "audio/AceStep SFT"
    DESCRIPTION = (
        "Encodes caption, lyrics, and metadata into positive/negative conditioning "
        "for the AceStep 1.5 SFT Generate node."
    )

    def encode(
        self,
        clip,
        caption,
        lyrics,
        instrumental,
        seed,
        duration,
        bpm,
        timesignature,
        language,
        keyscale,
        generate_audio_codes=True,
        lm_cfg_scale=2.0,
        lm_temperature=0.85,
        lm_top_p=0.9,
        lm_top_k=0,
        lm_min_p=0.0,
        lm_negative_prompt="",
        style_tags="",
        style_bpm=0,
        style_keyscale="",
    ):
        actual_lyrics = "[Instrumental]" if instrumental else lyrics

        # --- Style overrides from Music Analyzer node ---
        if style_tags and style_tags.strip():
            caption = f"{caption}, {style_tags}" if caption.strip() else style_tags
        if style_bpm > 0:
            if duration > 0:
                original_bpm = bpm if bpm > 0 else 120
                if original_bpm != style_bpm:
                    new_duration = round(duration * original_bpm / style_bpm, 1)
                    print(f"[AceStep SFT] Duration adjusted: {duration}s @ {original_bpm} BPM → {new_duration}s @ {style_bpm} BPM (same bar count)")
                    duration = new_duration
            bpm = style_bpm
        if style_keyscale and style_keyscale.strip():
            keyscale = style_keyscale

        # --- Determine duration ---
        auto_duration = (duration <= 0)
        if auto_duration:
            duration = _estimate_duration_from_lyrics(actual_lyrics, bpm)

        # --- Resolve auto metadata ---
        bpm_is_auto = (bpm == 0)
        ts_is_auto = (timesignature == "auto")
        ks_is_auto = (keyscale == "auto")
        tok_bpm = 120 if bpm_is_auto else bpm
        tok_ts = 4 if ts_is_auto else int(timesignature)
        tok_ks = "C major" if ks_is_auto else keyscale

        # --- Encode conditioning ---
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

            tokens["lyrics"] = inner_tok.tokenize_with_weights(
                f"# Languages\n{language}\n\n# Lyric\n{actual_lyrics}<|endoftext|>",
                False,
                disable_weights=True,
            )

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
        negative = _build_null_negative(positive)

        return (positive, negative)


class AceStepSFTGenerate:
    """AceStep 1.5 SFT music generation node (sampler + decoder).

    Receives a MODEL and conditioning, runs the diffusion sampler, and
    optionally decodes with VAE to produce audio output.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {
                    "tooltip": "AceStep 1.5 diffusion model (from Load Diffusion Model or with LoRA applied).",
                }),
                "positive": ("CONDITIONING", {
                    "tooltip": "Positive conditioning from AceStep 1.5 SFT TextEncode.",
                }),
                "negative": ("CONDITIONING", {
                    "tooltip": "Negative conditioning from AceStep 1.5 SFT TextEncode.",
                }),
                # ---- Sampling ----
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xffffffffffffffff,
                    "control_after_generate": True,
                }),
                "steps": ("INT", {
                    "default": 50, "min": 1, "max": 200, "step": 1,
                    "tooltip": "Diffusion inference steps.",
                }),
                "cfg": ("FLOAT", {
                    "default": 7.0, "min": 1.0, "max": 20.0, "step": 0.1,
                    "tooltip": "Classifier-free guidance scale.",
                }),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {
                    "default": "euler",
                }),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {
                    "default": "normal",
                }),
                "denoise": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Denoise strength. 1.0 = full generation. < 1.0 requires latent_or_audio.",
                }),
                "duration": ("FLOAT", {
                    "default": 60.0, "min": 0.0, "max": 600.0, "step": 0.1,
                    "tooltip": "Duration in seconds. Set to 0 for auto from latent_or_audio.",
                }),
                "infer_method": (("ode", "sde"), {
                    "default": "ode",
                    "tooltip": "ode = deterministic diffusion. sde = stochastic (remaps sampler).",
                }),
                "guidance_mode": (GUIDANCE_MODES, {
                    "default": "apg",
                    "tooltip": "APG = Adaptive Projected Guidance. ADG = Angle-based Dynamic Guidance. standard_cfg = normal CFG.",
                }),
            },
            "optional": {
                "vae": ("VAE", {
                    "tooltip": "VAE for decoding latents to audio. Audio output requires this.",
                }),
                "latent_or_audio": ("AUDIO,LATENT", {
                    "tooltip": "Base input for refinement (img2img). Use denoise < 1.0.",
                }),
                "batch_size": ("INT", {
                    "default": 1, "min": 1, "max": 16,
                    "tooltip": "Number of audios to generate in parallel.",
                }),
                # ---- Latent post-processing ----
                "latent_shift": ("FLOAT", {
                    "default": 0.0, "min": -0.2, "max": 0.2, "step": 0.01,
                    "tooltip": "Additive shift on latents before VAE decode.",
                }),
                "latent_rescale": ("FLOAT", {
                    "default": 1.0, "min": 0.5, "max": 1.5, "step": 0.01,
                    "tooltip": "Multiplicative scale on latents before VAE decode.",
                }),
                "fade_in_duration": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 10.0, "step": 0.1,
                }),
                "fade_out_duration": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 10.0, "step": 0.1,
                }),
                "use_tiled_vae": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Use tiled VAE for long audio / low VRAM.",
                }),
                "unload_models_after_generate": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Unload models from memory after generation.",
                }),
                "voice_boost": ("FLOAT", {
                    "default": 0.0, "min": -12.0, "max": 12.0, "step": 0.5,
                    "tooltip": "Voice boost in dB.",
                }),
                # ---- APG parameters ----
                "apg_eta": ("FLOAT", {
                    "default": 0.0, "min": -10.0, "max": 10.0, "step": 0.05,
                    "tooltip": "APG eta: parallel component retention.",
                }),
                "apg_momentum": ("FLOAT", {
                    "default": -0.75, "min": -1.0, "max": 1.0, "step": 0.05,
                    "tooltip": "APG momentum buffer coefficient.",
                }),
                "apg_norm_threshold": ("FLOAT", {
                    "default": 2.5, "min": 0.0, "max": 15.0, "step": 0.1,
                    "tooltip": "APG norm threshold for gradient clipping.",
                }),
                "guidance_interval": ("FLOAT", {
                    "default": 0.5, "min": -1.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Guidance interval width. -1 = use legacy cfg_interval_start/end.",
                }),
                "guidance_interval_decay": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                }),
                "min_guidance_scale": ("FLOAT", {
                    "default": 3.0, "min": 0.0, "max": 30.0, "step": 0.1,
                }),
                "guidance_scale_text": ("FLOAT", {
                    "default": -1.0, "min": -1.0, "max": 30.0, "step": 0.1,
                    "tooltip": "Split text guidance. Active when both text and lyric > 1.0.",
                }),
                "guidance_scale_lyric": ("FLOAT", {
                    "default": -1.0, "min": -1.0, "max": 30.0, "step": 0.1,
                    "tooltip": "Split lyric guidance. Active when both text and lyric > 1.0.",
                }),
                "omega_scale": ("FLOAT", {
                    "default": 0.0, "min": -8.0, "max": 8.0, "step": 0.05,
                }),
                "erg_scale": ("FLOAT", {
                    "default": 0.0, "min": -0.9, "max": 2.0, "step": 0.05,
                }),
                "cfg_interval_start": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                }),
                "cfg_interval_end": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                }),
                "shift": ("FLOAT", {
                    "default": 3.0, "min": 0.0, "max": 5.0, "step": 0.1,
                    "tooltip": "Timestep schedule shift. ACEStep15 default is 3.0.",
                }),
            },
        }

    RETURN_TYPES = ("MODEL", "VAE", "CONDITIONING", "CONDITIONING", "LATENT", "AUDIO")
    RETURN_NAMES = ("model", "vae", "positive", "negative", "latent", "audio")
    FUNCTION = "generate"
    CATEGORY = "audio/AceStep SFT"
    DESCRIPTION = (
        "AceStep 1.5 SFT sampler + decoder. Requires MODEL and conditioning inputs. "
        "Audio output requires VAE connected."
    )

    @classmethod
    def VALIDATE_INPUTS(cls, input_types=None, **kwargs):
        if input_types is not None:
            src = input_types.get("latent_or_audio")
            if src is not None and src not in ("AUDIO", "LATENT"):
                return "latent_or_audio must be AUDIO or LATENT"
        return True

    def generate(
        self,
        model,
        positive,
        negative,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        denoise,
        duration,
        infer_method,
        guidance_mode,
        # Optional
        vae=None,
        latent_or_audio=None,
        batch_size=1,
        latent_shift=0.0,
        latent_rescale=1.0,
        fade_in_duration=0.0,
        fade_out_duration=0.0,
        use_tiled_vae=True,
        unload_models_after_generate=False,
        voice_boost=0.0,
        apg_momentum=-0.75,
        apg_eta=0.0,
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
    ):
        if denoise < 1.0 and latent_or_audio is None:
            raise ValueError(
                "denoise < 1.0 requires latent_or_audio connected with AUDIO or LATENT."
            )

        if latent_or_audio is not None and not (_is_audio_payload(latent_or_audio) or _is_latent_payload(latent_or_audio)):
            raise ValueError("latent_or_audio must receive AUDIO or LATENT.")

        source_latent_meta = _get_source_latent_metadata(latent_or_audio)

        # Use 48000 as default; if VAE is connected we can query it
        vae_sr = getattr(vae, "audio_sample_rate", 48000) if vae is not None else 48000

        cfg_interval_start, cfg_interval_end = sorted(
            (cfg_interval_start, cfg_interval_end)
        )

        # --- 1. Determine duration ---
        auto_duration = (duration <= 0)
        if latent_or_audio is not None and auto_duration:
            duration = _get_source_duration_seconds(latent_or_audio, vae_sr)
        elif auto_duration:
            duration = 60.0  # fallback when no source and no duration

        latent_length = max(10, round(duration * vae_sr / 1920))
        duration = latent_length * 1920.0 / vae_sr

        if source_latent_meta is not None:
            batch_size = latent_or_audio["samples"].shape[0]
            latent_length = latent_or_audio["samples"].shape[-1]
            duration = latent_length * 1920.0 / vae_sr

        # --- 2. Create starting latent ---
        if latent_or_audio is not None:
            if source_latent_meta is not None:
                latent_image = latent_or_audio["samples"].clone()
                latent_image = _match_latent_length(latent_image, latent_length)
            else:
                if vae is None:
                    raise ValueError("VAE is required to encode AUDIO input into latent.")
                latent_image = _build_source_latent(
                    vae,
                    latent_or_audio,
                    batch_size,
                    latent_length,
                    vae_sr,
                    use_tiled_vae,
                )
        else:
            latent_image = torch.zeros(
                [batch_size, 64, latent_length],
                device=comfy.model_management.intermediate_device(),
            )

        latent_image = comfy.sample.fix_empty_latent_channels(
            model,
            latent_image,
            source_latent_meta["downscale_ratio_spacial"] if source_latent_meta is not None else None,
        )

        # --- 3. Text-only conditioning branch detection ---
        text_only_positive = _build_text_only_conditioning(positive)
        _has_lyric_branch = text_only_positive is not None
        if not _has_lyric_branch and positive is not None:
            for _ci in positive:
                if len(_ci) > 1 and isinstance(_ci[1], dict):
                    _mc = _ci[1].get("model_conds")
                    if isinstance(_mc, dict) and "lyric_embed" in _mc:
                        _has_lyric_branch = True
                        break
        split_guidance_active = (
            _has_lyric_branch
            and guidance_scale_text > 1.0
            and guidance_scale_lyric > 1.0
        )

        # --- 7. Prepare noise ---
        # Wrap all sampling and decoding in torch.inference_mode() for efficiency
        with torch.inference_mode():
            batch_inds = (
                source_latent_meta["batch_index"]
                if source_latent_meta is not None else None
            )
            noise_mask = (
                source_latent_meta["noise_mask"]
                if source_latent_meta is not None else None
            )
            noise_mask = _match_noise_mask_length(noise_mask, latent_length)
            noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)

            model = model.clone()

            # --- Shift patch: force the user's shift value ---
            sampling_base = comfy.model_sampling.ModelSamplingDiscreteFlow
            sampling_type = comfy.model_sampling.CONST

            class ModelSamplingShifted(sampling_base, sampling_type):
                pass

            model_sampling_obj = ModelSamplingShifted(model.model.model_config)
            model_sampling_obj.set_parameters(shift=shift, multiplier=1.0)
            model.add_object_patch("model_sampling", model_sampling_obj)

            custom_sigmas = None

            # --- INÍCIO BLOCO NÃO NATIVO: intervalo de guidance e APG/ADG/split ---
            use_official_interval = guidance_interval >= 0.0
            official_interval = max(0.0, min(1.0, guidance_interval))
            interval_step_start = int(steps * ((1.0 - official_interval) / 2.0))
            interval_step_end = int(steps * (official_interval / 2.0 + 0.5))
            # --- 9. Apply guidance via model patching ---
            if (
                (guidance_mode in ("apg", "adg") and cfg > 1.0)
                or split_guidance_active
                or _erg_tau_from_scale(erg_scale) < 0.999999
                or abs(omega_scale) > 1e-8
            ):
                momentum_buf = MomentumBuffer(momentum=apg_momentum)
                norm_thresh = apg_norm_threshold
                eta_val = apg_eta
                schedule_state = {
                    "index": 0,
                    "last_sigma": None,
                    "denom": max(steps - 1, 1),
                }
                branch_state = {
                    "text_denoised": None,
                    "erg_denoised": None,
                }
                use_adg = (guidance_mode == "adg")
                erg_active = _erg_tau_from_scale(erg_scale) < 0.999999

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
                    _, step_index, _, _, _ = get_step_context(sigma, cfg)
                    branch_state["text_denoised"] = None
                    branch_state["erg_denoised"] = None

                    if not split_guidance_active and not erg_active:
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

                    if erg_active:
                        erg_cond = _build_processed_erg_conditioning(cond, erg_scale)
                        if erg_cond is not None:
                            erg_out, _ = comfy.samplers.calc_cond_batch(
                                args["model"], [erg_cond, None], x, sigma, model_options
                            )
                            erg_model_output = _apply_erg_tau_to_model_output(
                                x - erg_out, erg_scale
                            )
                            branch_state["erg_denoised"] = x - erg_model_output

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
                    erg_denoised = branch_state.get("erg_denoised")
                    effective_uncond_denoised = (
                        erg_denoised if erg_denoised is not None else uncond_denoised
                    )
                    if split_guidance_active and text_denoised is not None:
                        cond_model_output = x - cond_denoised
                        uncond_model_output = x - effective_uncond_denoised
                        text_model_output = x - text_denoised
                        blended_model_output = (
                            (1.0 - guidance_scale_text) * uncond_model_output
                            + (guidance_scale_text - guidance_scale_lyric) * text_model_output
                            + guidance_scale_lyric * cond_model_output
                        )
                        effective_cond_denoised = x - blended_model_output

                    if not in_interval:
                        return _apply_omega_scale(x - cond_denoised, omega_scale)

                    if split_guidance_active and text_denoised is not None:
                        return _apply_omega_scale(x - effective_cond_denoised, omega_scale)

                    if guidance_mode == "standard_cfg" or current_guidance_scale <= 1.0:
                        guided_denoised = effective_uncond_denoised + (
                            effective_cond_denoised - effective_uncond_denoised
                        ) * current_guidance_scale
                        return _apply_omega_scale(x - guided_denoised, omega_scale)

                    sigma_r = sigma.reshape(-1, *([1] * (x.ndim - 1))).clamp(min=1e-8)
                    v_cond = (x - effective_cond_denoised) / sigma_r
                    v_uncond = (x - effective_uncond_denoised) / sigma_r

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
                            eta=eta_val,
                            norm_threshold=norm_thresh,
                            dims=[-1],
                        )

                    return _apply_omega_scale(v_guided * sigma_r, omega_scale)

                if split_guidance_active or erg_active:
                    model.set_model_sampler_calc_cond_batch_function(
                        calc_cond_batch_function
                    )
                if (
                    guidance_mode in ("apg", "adg") and cfg > 1.0
                ) or split_guidance_active or _erg_tau_from_scale(erg_scale) < 0.999999 or abs(omega_scale) > 1e-8:
                    model.set_model_sampler_cfg_function(
                        guided_cfg_function, disable_cfg1_optimization=True
                    )
            # --- FIM BLOCO NÃO NATIVO ---

            # --- 10. Sample ---
            callback = latent_preview.prepare_callback(model, steps)
            disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
            resolved_sampler_name = _resolve_sampler_for_infer_method(
                sampler_name, infer_method
            )
            if resolved_sampler_name != sampler_name:
                print(
                    f"[AceStep SFT] infer_method={infer_method} remapped sampler "
                    f"{sampler_name} -> {resolved_sampler_name}."
                )


            samples_raw = comfy.sample.sample(
                model,
                noise,
                steps,
                cfg,
                resolved_sampler_name,
                scheduler,
                positive,
                negative,
                latent_image,
                denoise=denoise,
                disable_noise=False,
                start_step=None,
                last_step=None,
                force_full_denoise=False,
                noise_mask=noise_mask,
                sigmas=custom_sigmas,
                callback=callback,
                disable_pbar=disable_pbar,
                seed=seed,
            )

            # --- Keep LATENT output in raw diffusion space (no shift/rescale) ---
            out_latent = latent_or_audio.copy() if _is_latent_payload(latent_or_audio) else {}
            out_latent.pop("downscale_ratio_spacial", None)
            out_latent.pop("noise_mask", None)
            out_latent["type"] = "audio"
            out_latent["samples"] = samples_raw

            # --- 12. Decode with VAE (only if VAE is connected) ---
            audio_output = None
            if vae is not None:
                samples_for_audio = samples_raw
                if latent_shift != 0.0 or latent_rescale != 1.0:
                    samples_for_audio = samples_for_audio * latent_rescale + latent_shift

                audio = _vae_decode_with_optional_tiling(
                    vae, samples_for_audio, use_tiled_vae
                ).movedim(-1, 1)

                if audio.dtype != torch.float32:
                    audio = audio.float()

                if voice_boost != 0.0:
                    boost_linear = 10.0 ** (voice_boost / 20.0)
                    audio = audio * boost_linear

                if fade_in_duration > 0.0 or fade_out_duration > 0.0:
                    audio = _apply_fade(
                        audio,
                        fade_in_samples=round(fade_in_duration * vae_sr),
                        fade_out_samples=round(fade_out_duration * vae_sr),
                    )

                audio = torch.clamp(audio, -1.0, 1.0)
                audio_output = {
                    "waveform": audio,
                    "sample_rate": vae_sr,
                }
            else:
                print("[AceStep SFT] No VAE connected — skipping audio decode. Connect a VAE to get audio output.")

            if unload_models_after_generate:
                _release_acestep_generation_models(model, None, vae)
            return (
                model,
                vae,
                _clone_runtime_conditioning(positive),
                _clone_runtime_conditioning(negative),
                out_latent,
                audio_output,
            )


class AceStepSFTSaveAudio:
    """Saves audio to disk with waveform spectrum visualization.

    Supports FLAC, MP3, and Opus formats with configurable filename prefix.
    """

    _OPUS_RATES = [8000, 12000, 16000, 24000, 48000]

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {
                    "tooltip": "Audio to save.",
                }),
                "filename_prefix": ("STRING", {
                    "default": "audio/AceStep",
                    "tooltip": "Filename prefix. May include subfolder path.",
                }),
                "format": (("flac", "mp3", "opus"), {
                    "default": "flac",
                    "tooltip": "Audio format to save.",
                }),
            },
            "optional": {
                "quality": (("V0", "64k", "96k", "128k", "192k", "320k"), {
                    "default": "128k",
                    "tooltip": "Quality/bitrate for MP3 and Opus formats. Ignored for FLAC.",
                }),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_audio"
    OUTPUT_NODE = True
    CATEGORY = "audio/AceStep SFT"
    DESCRIPTION = (
        "Saves audio with waveform visualization. "
        "Supports FLAC, MP3, and Opus formats."
    )

    def save_audio(self, audio, filename_prefix="audio/AceStep", format="flac",
                   quality="128k", prompt=None, extra_pnginfo=None):
        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, self.output_dir
        )

        metadata = {}
        if not args.disable_metadata:
            if prompt is not None:
                metadata["prompt"] = json.dumps(prompt)
            if extra_pnginfo is not None:
                for x in extra_pnginfo:
                    metadata[x] = json.dumps(extra_pnginfo[x])

        results = []
        for batch_number, waveform in enumerate(audio["waveform"].cpu()):
            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            file = f"{filename_with_batch_num}_{counter:05}_.{format}"
            output_path = os.path.join(full_output_folder, file)

            sample_rate = audio["sample_rate"]

            if format == "opus":
                if sample_rate > 48000:
                    sample_rate = 48000
                elif sample_rate not in self._OPUS_RATES:
                    for rate in sorted(self._OPUS_RATES):
                        if rate > sample_rate:
                            sample_rate = rate
                            break
                    if sample_rate not in self._OPUS_RATES:
                        sample_rate = 48000
                if sample_rate != audio["sample_rate"]:
                    waveform = torchaudio.functional.resample(waveform, audio["sample_rate"], sample_rate)

            output_buffer = BytesIO()
            output_container = av.open(output_buffer, mode="w", format=format)

            for key, value in metadata.items():
                output_container.metadata[key] = value

            layout = "mono" if waveform.shape[0] == 1 else "stereo"

            if format == "opus":
                out_stream = output_container.add_stream("libopus", rate=sample_rate, layout=layout)
                bitrates = {"64k": 64000, "96k": 96000, "128k": 128000, "192k": 192000, "320k": 320000}
                out_stream.bit_rate = bitrates.get(quality, 128000)
            elif format == "mp3":
                out_stream = output_container.add_stream("libmp3lame", rate=sample_rate, layout=layout)
                if quality == "V0":
                    out_stream.codec_context.qscale = 1
                elif quality == "128k":
                    out_stream.bit_rate = 128000
                elif quality == "320k":
                    out_stream.bit_rate = 320000
                else:
                    bitrates = {"64k": 64000, "96k": 96000, "192k": 192000}
                    out_stream.bit_rate = bitrates.get(quality, 128000)
            else:
                out_stream = output_container.add_stream("flac", rate=sample_rate, layout=layout)

            frame = av.AudioFrame.from_ndarray(
                waveform.movedim(0, 1).reshape(1, -1).float().numpy(),
                format="flt",
                layout=layout,
            )
            frame.sample_rate = sample_rate
            frame.pts = 0
            output_container.mux(out_stream.encode(frame))
            output_container.mux(out_stream.encode(None))
            output_container.close()

            output_buffer.seek(0)
            with open(output_path, "wb") as f:
                f.write(output_buffer.getbuffer())

            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type,
            })
            counter += 1

        return {"ui": {"audio": results}}


class AceStepSFTPreviewAudio:
    """Previews audio with waveform spectrum visualization.

    Saves a temporary FLAC for playback and displays a waveform visualizer.
    """

    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_temp_" + "".join(
            random.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(5)
        )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {
                    "tooltip": "Audio to preview.",
                }),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "preview_audio"
    OUTPUT_NODE = True
    CATEGORY = "audio/AceStep SFT"
    DESCRIPTION = "Previews audio with waveform visualization."

    def preview_audio(self, audio, prompt=None, extra_pnginfo=None):
        filename_prefix = "AceStep" + self.prefix_append
        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, self.output_dir
        )

        results = []
        for batch_number, waveform in enumerate(audio["waveform"].cpu()):
            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            file = f"{filename_with_batch_num}_{counter:05}_.flac"
            output_path = os.path.join(full_output_folder, file)

            sample_rate = audio["sample_rate"]
            layout = "mono" if waveform.shape[0] == 1 else "stereo"

            output_buffer = BytesIO()
            output_container = av.open(output_buffer, mode="w", format="flac")
            out_stream = output_container.add_stream("flac", rate=sample_rate, layout=layout)

            frame = av.AudioFrame.from_ndarray(
                waveform.movedim(0, 1).reshape(1, -1).float().numpy(),
                format="flt",
                layout=layout,
            )
            frame.sample_rate = sample_rate
            frame.pts = 0
            output_container.mux(out_stream.encode(frame))
            output_container.mux(out_stream.encode(None))
            output_container.close()

            output_buffer.seek(0)
            with open(output_path, "wb") as f:
                f.write(output_buffer.getbuffer())

            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type,
            })
            counter += 1

        return {"ui": {"audio": results}}


class AceStepSFTTurboTagAdapter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "turbo_tags": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Aggressive Brazilian Funk Mandelao, Ritualistic Phonk atmosphere, Heavy distorted 808 bass",
                    "tooltip": "Turbo-style tags or caption to adapt for AceStep 1.5 SFT. Best results when tags are comma-separated.",
                }),
                "adaptation_strength": (("conservative", "balanced", "aggressive"), {
                    "default": "balanced",
                    "tooltip": "How strongly to rewrite Turbo phrasing into SFT-style tags.",
                }),
                "keep_unknown_tags": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Keep tags that were not explicitly mapped, after simplification.",
                }),
                "add_sft_bias_tags": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Add a few extra SFT-oriented anchor tags for genre, groove, and mood.",
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "FLOAT", "INT")
    RETURN_NAMES = ("sft_tags", "notes", "suggested_cfg", "suggested_steps")
    FUNCTION = "adapt"
    CATEGORY = "audio/AceStep SFT"
    DESCRIPTION = (
        "Adapts Turbo-oriented music tags into shorter SFT-friendly tags. "
        "Useful when Turbo captions sound semantically right but produce very different results on AceStep 1.5 SFT."
    )

    def adapt(self, turbo_tags, adaptation_strength, keep_unknown_tags, add_sft_bias_tags):
        return _adapt_turbo_tags_for_sft(
            turbo_tags,
            adaptation_strength=adaptation_strength,
            keep_unknown_tags=keep_unknown_tags,
            add_sft_bias_tags=add_sft_bias_tags,
        )


# ---------------------------------------------------------------------------
# Node Registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "AceStepSFTGenerate": AceStepSFTGenerate,
    "AceStepSFTTextEncode": AceStepSFTTextEncode,
    "AceStepSFTModelLoader": AceStepSFTModelLoader,
    "AceStepSFTLoraLoader": AceStepSFTLoraLoader,
    "AceStepSFTMusicAnalyzer": AceStepSFTMusicAnalyzer,
    "AceStepSFTTurboTagAdapter": AceStepSFTTurboTagAdapter,
    "AceStepSFTSaveAudio": AceStepSFTSaveAudio,
    "AceStepSFTPreviewAudio": AceStepSFTPreviewAudio,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AceStepSFTGenerate": "AceStep 1.5 SFT Generate",
    "AceStepSFTTextEncode": "AceStep 1.5 SFT TextEncode",
    "AceStepSFTModelLoader": "AceStep 1.5 SFT Model Loader",
    "AceStepSFTLoraLoader": "AceStep 1.5 SFT Lora Loader",
    "AceStepSFTMusicAnalyzer": "AceStep 1.5 SFT Get Music Infos",
    "AceStepSFTTurboTagAdapter": "AceStep 1.5 SFT Turbo Tag Adapter",
    "AceStepSFTSaveAudio": "AceStep 1.5 SFT Save Audio",
    "AceStepSFTPreviewAudio": "AceStep 1.5 SFT Preview Audio",
}
