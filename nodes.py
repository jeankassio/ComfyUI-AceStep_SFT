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
    t1 = t1 / torch.linalg.norm(t1, dim=1, keepdim=True)
    t2 = t2 / torch.linalg.norm(t2, dim=1, keepdim=True)
    return torch.sum(t1 * t2, dim=1, keepdim=True)


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
# Duration estimation from lyrics (matching Gradio auto-duration behavior)
# ---------------------------------------------------------------------------

def _estimate_duration_from_lyrics(lyrics, bpm=120):
    """Estimate song duration from lyrics structure and content.

    When Gradio's AceStep pipeline uses auto-duration, the LLM reasons about
    the lyrics to choose an appropriate length.  This heuristic provides a
    similar estimate based on the number of lines, section tags, and BPM.
    """
    if not lyrics or lyrics.strip().lower() in ("[instrumental]", ""):
        return 60.0

    effective_bpm = max(60, min(bpm if bpm > 0 else 120, 240))
    beat_duration = 60.0 / effective_bpm
    lines = lyrics.strip().split("\n")
    total_beats = 0
    instrumental_tags = frozenset(
        {"instrumental", "interlude", "solo", "break"}
    )

    for line in lines:
        stripped = line.strip()
        if not stripped:
            total_beats += 2  # Brief pause for empty lines
            continue

        if stripped.startswith("[") and stripped.endswith("]"):
            tag = stripped[1:-1].lower().split(":")[0].split()[0]
            if tag in instrumental_tags:
                total_beats += 16  # ~2 bars of 4/4
            elif tag in ("intro", "outro"):
                total_beats += 8  # ~1 bar
            else:
                total_beats += 4  # Section transition
            continue

        # Lyrics line: ~2 words per beat (typical singing rate)
        words = len(stripped.split())
        total_beats += max(4, math.ceil(words / 2.0))

    duration = total_beats * beat_duration
    return max(10.0, min(round(duration, 1), 240.0))


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
                    "default": False,
                    "tooltip": "Force instrumental mode (overrides lyrics with [Instrumental]).",
                }),
                # ---- Sampling ----
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xffffffffffffffff,
                    "control_after_generate": True,
                }),
                "steps": ("INT", {
                    "default": 32, "min": 1, "max": 200, "step": 1,
                    "tooltip": "Diffusion inference steps.",
                }),
                "cfg": ("FLOAT", {
                    "default": 7.0, "min": 1.0, "max": 20.0, "step": 0.1,
                    "tooltip": "Classifier-free guidance scale.",
                }),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
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
                    "default": 0.0, "min": 0.0, "max": 600.0, "step": 0.1,
                    "tooltip": "Duration in seconds. 0 = auto (estimates from lyrics length, or uses source_audio duration).",
                }),
                "bpm": ("INT", {
                    "default": 120, "min": 0, "max": 300,
                    "tooltip": "Beats per minute. 0 = auto (N/A, let model decide).",
                }),
                "timesignature": (['auto', '4', '3', '2', '6'], {
                    "tooltip": "Time signature numerator. 'auto' = let model decide (N/A).",
                }),
                "language": (LANGUAGES,),
                "keyscale": (["auto"] + KEYSCALES_LIST, {
                    "tooltip": "Key and scale. 'auto' = let model decide (N/A).",
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
                    "default": True,
                    "tooltip": "Enable peak normalization (normalize to max amplitude). Disable for manual loudness control.",
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
        normalize_peak=True,
        voice_boost=0.0,
        apg_momentum=-0.75,
        apg_norm_threshold=2.5,
        cfg_interval_start=0.0,
        cfg_interval_end=1.0,
        shift=3.0,
        custom_timesteps="",
    ):
        actual_lyrics = "[Instrumental]" if instrumental else lyrics

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

        # --- 4.5. Initialize reference audio variables ---
        refer_audio_latents = None
        refer_audio_order_mask = None

        # --- 5. Negative conditioning ---
        neg_tokens = clip.tokenize("", generate_audio_codes=False)
        negative = clip.encode_from_tokens_scheduled(neg_tokens)

        # Share audio_codes from positive → negative
        audio_codes_from_pos = None
        for cond_item in positive:
            if len(cond_item) > 1 and "audio_codes" in cond_item[1]:
                audio_codes_from_pos = cond_item[1]["audio_codes"]
                break
        if audio_codes_from_pos is not None:
            negative = node_helpers.conditioning_set_values(
                negative, {"audio_codes": audio_codes_from_pos}
            )

        # Zero out negative embedding → model uses null_condition_emb
        for n in negative:
            n[0] = torch.zeros_like(n[0])

        # Add reference audio conditioning if provided
        if refer_audio_latents is not None:
            # Determine if this is a cover or pure style transfer
            is_cover = reference_as_cover
            
            positive = node_helpers.conditioning_set_values(
                positive,
                {
                    "refer_audio_acoustic_hidden_states_packed": refer_audio_latents,
                    "refer_audio_order_mask": refer_audio_order_mask,
                    "is_covers": torch.full((batch_size,), is_cover, dtype=torch.bool, device=refer_audio_latents.device),
                    "audio_cover_strength": audio_cover_strength if is_cover else 0.0,
                },
                append=True,
            )

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

            # --- 9. Apply guidance via model patching ---
            if guidance_mode in ("apg", "adg") and cfg > 1.0:
                momentum_buf = MomentumBuffer(momentum=apg_momentum)
                norm_thresh = apg_norm_threshold
                interval_start = cfg_interval_start
                interval_end = cfg_interval_end
                use_adg = (guidance_mode == "adg")

                def guided_cfg_function(args):
                    cond_denoised = args["cond_denoised"]
                    uncond_denoised = args["uncond_denoised"]
                    cond_scale = args["cond_scale"]
                    x = args["input"]
                    sigma = args["sigma"]

                    t_curr = float(sigma.flatten()[0])
                    if t_curr < interval_start or t_curr > interval_end:
                        return x - cond_denoised

                    sigma_r = sigma.reshape(-1, *([1] * (x.ndim - 1))).clamp(min=1e-8)
                    v_cond = (x - cond_denoised) / sigma_r
                    v_uncond = (x - uncond_denoised) / sigma_r

                    if use_adg:
                        v_guided = adg_guidance(
                            x.movedim(1, -1),
                            v_cond.movedim(1, -1),
                            v_uncond.movedim(1, -1),
                            t_curr,
                            cond_scale,
                        ).movedim(-1, 1)
                    else:
                        v_guided = apg_guidance(
                            v_cond,
                            v_uncond,
                            cond_scale,
                            momentum_buffer=momentum_buf,
                            norm_threshold=norm_thresh,
                            dims=[-1],
                        )

                    return v_guided * sigma_r

                model = model.clone()
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
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AceStepSFTGenerate": "AceStep 1.5 SFT Generate",
}
