# ComfyUI-AceStep SFT

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)

An all-in-one node for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) that implements **AceStep 1.5 SFT** (Supervised Fine-Tuning), a state-of-the-art music generation model. It starts from the official AceStep workflow and extends it with stronger conditioning control and practical ComfyUI-oriented quality options.

> **SFT = Supervised Fine-Tuning**: A specialized version of AceStep optimized for generating superior quality audio through supervised training.

## 📋 Overview

This package currently provides four nodes under `audio/AceStep SFT`:

- **AceStep 1.5 SFT Generate**: all-in-one generation, editing, and decoding
- **AceStep 1.5 SFT Music Analyzer**: AI-powered audio analysis (tags, BPM, key/scale)
- **AceStep 1.5 SFT Lora Loader**: chainable LoRA stack builder for AceStep 1.5 SFT
- **AceStep 1.5 SFT Turbo Tag Adapter**: rewrites Turbo-oriented tags into shorter SFT-friendly prompt tags

The **AceStepSFTGenerate** node encapsulates the entire music generation workflow:

1. **Latent Creation** - Generates initial latents or loads from `latent_or_audio` input
2. **Text Encoding** - Processes captions, lyrics, and metadata via multiple CLIP encoders
3. **Diffusion Sampling** - Runs the diffusion model with advanced guidance control
4. **Audio Decoding** - Converts latents to high-quality audio via VAE

### Example Configuration

![AceStep SFT Node Configuration](example.png)

## 🎯 Key Features

### ✨ Advanced Guidance

The node supports three classifier-free guidance modes, each with unique characteristics:

- **APG (Adaptive Projected Guidance)** ⭐ *Recommended*
  - Dynamic adaptation via momentum buffering
  - Gradient clipping with adaptive thresholds
  - Orthogonal projection to eliminate unwanted noise
  - **AceStep SFT Default** - best quality and stability balance

- **ADG (Angle-based Dynamic Guidance)**
  - Angle-based guidance between conditions
  - Operates in velocity space (flow matching)
  - Ideal for aggressive style distortion
  - Adaptive clipping based on angle between x0_cond and x0_uncond

- **Standard CFG**
  - Traditional Classifier-Free Guidance
  - Simple and predictable implementation
  - Useful as a comparison baseline

### 🎵 Intelligent Metadata Processing

- **Auto-Duration**: Automatically estimates music duration by analyzing lyric structure
- **LLM Encoding**: Use Qwen LLM (0.6B or 1.7B/4B) to generate semantic audio codes
- **Auto Values**: BPM, Time Signature, and Key/Scale automatic (model decides)
- **Multilingual Support**: Over 23 languages supported

### 🎧 AI Music Analyzer

- **Audio Tag Extraction**: Uses the native ACE-Step Transcriber to extract lyric, vocal, and song-structure tags from audio
- **BPM Detection**: Automatic tempo detection via librosa
- **Key/Scale Detection**: Detects musical key and scale (e.g. "G minor")
- **JSON Output**: Structured `music_infos` output with all analysis results
- **Generation Parameters**: Control temperature, top_p, top_k, repetition_penalty, and seed
- **Auto Model Download**: Models are downloaded on first use (~1-7 GB each)

#### Native Analysis Model:

| Model | Size | Type | Best For |
|-------|------|------|----------|
| ACE-Step-Transcriber | 22.4 GB download | Audio-to-Text | Native ACE-Step 1.5 transcription for lyrics, singing voice, structure tags, and instrument hints |

This node is now dedicated to the native ACE-Step-Transcriber workflow. It uses the model's native prompt format, structured transcription output, and derives tags from language, lyrics, section markers such as verse/chorus/bridge, and optional instrument annotations.

### 🔄 Audio Refinement (img2img)

- **Latent-based Refinement**: Use `denoise < 1.0` with `latent_or_audio` connected to refine existing audio
- **Accepts AUDIO or LATENT**: Connect any audio or latent output for img2img-style editing
- **Batch Generation**: Generate multiple variations in parallel

### 🧠 Extended Conditioning Control

- **Split Text/Lyric Guidance**: Independent `guidance_scale_text` and `guidance_scale_lyric`
- **Omega Scale**: Mean-preserving output reweighting to approximate AceStep scheduler behavior
- **ERG Approximation**: Node-local prompt energy reweighting via `erg_scale`
- **Guidance Interval Decay**: Smoothly decay guidance inside the active interval

### 🎚️ AceStep LoRA Workflow

- **Chainable LoRA Loader**: Stack one or more AceStep LoRAs before generation
- **Separate strengths**: Independent `strength_model` and `strength_clip`
- **Single Generate input**: Final LoRA stack plugs into the `lora` input on Generate
- **Local `Loras/` folder**: Drop LoRA files directly into the node's `Loras/` folder — they are automatically registered at startup
- **Auto PEFT/DoRA conversion**: PEFT-format LoRAs (`adapter_config.json` + `adapter_model.safetensors`) placed in `Loras/` are automatically converted to ComfyUI format on first startup
- **DoRA support**: Full DoRA (Weight-Decomposed Low-Rank Adaptation) support with automatic `dora_scale` dimension fix for ComfyUI compatibility

### 🛠️ Latent Post-processing

- **Latent Shift**: Additive anti-clipping correction
- **Latent Rescale**: Multiplicative scaling for dynamic control

## 📦 Installation

### Prerequisites

- ComfyUI installed and functional
- CUDA/GPU or equivalent (modern processors)
- Recommended for better output quality (based on practical testing): use the merged SFT+Turbo model.
- Required model files:
  - Diffusion model (DiT): `acestep_v1.5_sft.safetensors`
  - Text Encoders: `qwen_0.6b_ace15.safetensors`, `qwen_1.7b_ace15.safetensors` (or 4B)
  - VAE: `ace_1.5_vae.safetensors`

### Download Model Files

Download the required models from HuggingFace:

1. **Diffusion Model (Recommended: merged SFT+Turbo)**:
  - [AceStep 1.5 Merged SFT+Turbo Model](https://huggingface.co/Aryanne/acestep-v15-test-merges/blob/main/acestep_v1.5_merge_sft_turbo_ta_0.5.safetensors)

2. **Alternative Diffusion Model (official SFT)**:
   - [AceStep 1.5 SFT Model](https://huggingface.co/ACE-Step/acestep-v15-sft/blob/main/model.safetensors)

3. **Text Encoders** (choose any versions):
   - [Text Encoders Collection](https://huggingface.co/Comfy-Org/ace_step_1.5_ComfyUI_files/tree/main/split_files/text_encoders)
     - `qwen_0.6b_ace15.safetensors` (caption processing)
     - `qwen_1.7b_ace15.safetensors` or `qwen_4b_ace15.safetensors` (audio code generation)

4. **VAE** (Audio codec):
   - [AceStep 1.5 VAE](https://huggingface.co/Comfy-Org/ace_step_1.5_ComfyUI_files/blob/main/split_files/vae/ace_1.5_vae.safetensors)

### Installation Steps

1. Clone the repository to your custom nodes folder:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/jeankassio/ComfyUI-AceStep_SFT.git
```

2. Place model files in the appropriate directories:
```
ComfyUI/models/diffusion_models/     # AceStep 1.5 SFT model
ComfyUI/models/text_encoders/        # Qwen encoders
ComfyUI/models/vae/                  # VAE
ComfyUI/models/loras/                # Optional AceStep 1.5 LoRAs
```

3. **(Optional) Place LoRAs in the local folder:**
```
ComfyUI/custom_nodes/ComfyUI-AceStep_SFT/Loras/   # Local LoRA folder
```
   You can place LoRAs here in **any** of these formats:
   - **ComfyUI format**: Single `.safetensors` file (ready to use)
   - **PEFT/DoRA format**: A folder containing `adapter_config.json` + `adapter_model.safetensors` (auto-converted on startup)
   - **Nested zip artifacts**: If your zip extracted a folder-inside-folder, the node detects this and fixes it automatically

4. Restart ComfyUI - the node will appear under `audio/AceStep SFT`

## 🧩 Available Nodes

### AceStep 1.5 SFT Generate

Main all-in-one node for text-to-music generation, latent-based audio refinement, and VAE decoding.

### AceStep 1.5 SFT Music Analyzer

AI-powered audio analysis node that extracts descriptive tags, BPM, and key/scale from audio input.

Inputs:
- `audio`: Audio input to analyze
- `model`: AI model selection (9 models, auto-downloaded)
- `get_tags` / `get_bpm` / `get_keyscale`: Enable/disable each analysis
- `max_new_tokens`: Maximum tokens for generative models
- `audio_duration`: Max seconds of audio to analyze
- `temperature`, `top_p`, `top_k`, `repetition_penalty`, `seed`: Generation parameters
- `unload_model`: Free VRAM after analysis
- `use_flash_attn`: Enable Flash Attention 2 (if compatible)

Outputs:
- `tags`: Comma-separated descriptive tags (STRING)
- `bpm`: Detected BPM as string e.g. "129bpm" (STRING)
- `keyscale`: Key and scale e.g. "G minor" (STRING)
- `music_infos`: JSON with all results (STRING)

### AceStep 1.5 SFT Lora Loader

Chainable utility node that builds a LoRA stack for AceStep 1.5 SFT.

Inputs:
- `lora_name`: LoRA file from `ComfyUI/models/loras` or the local `Loras/` folder
- `strength_model`: strength applied to the diffusion model
- `strength_clip`: strength applied to the text encoder stack
- `lora` (optional): upstream AceStep LoRA stack

Output:
- `lora`: connect to another Lora Loader or directly into Generate

#### Supported LoRA Formats

| Format | What to place in `Loras/` | Action |
|--------|--------------------------|--------|
| ComfyUI `.safetensors` | Single file | Used directly |
| PEFT/DoRA directory | Folder with `adapter_config.json` + `adapter_model.safetensors` | Auto-converted to `*_comfyui.safetensors` on startup |
| Nested zip artifact | Folder containing a `.safetensors` inside | Auto-extracted to root on startup |

The auto-conversion handles:
- Key remapping: `lora_A`/`lora_B` → `lora_down`/`lora_up`
- DoRA support: `lora_magnitude_vector` → `dora_scale` (with correct 2D shape)
- Per-layer alpha injection from `adapter_config.json` (supports `alpha_pattern` and `rank_pattern`)

## 🎛️ Node Parameters

### Required Parameters

| Parameter | Range | Description |
|-----------|-------|-------------|
| **diffusion_model** | - | Path to DiT model (AceStep 1.5 SFT) |
| **text_encoder_1** | - | Qwen3 0.6B Encoder (caption processing) |
| **text_encoder_2** | - | Qwen3 1.7B/4B Encoder (audio code generation) |
| **vae_name** | - | AceStep 1.5 VAE |
| **caption** | - | Text description of music (genre, mood, instruments) |
| **lyrics** | - | Song lyrics or `[Instrumental]` |
| **instrumental** | boolean | Force instrumental mode (overrides lyrics) |
| **seed** | 0 - 2^64 | Seed for reproducibility |
| **steps** | 1 - 200 | Diffusion inference steps (default: 50 for ACE-Step 1.5 SFT) |
| **cfg** | 1.0 - 20.0 | Classifier-free guidance scale (default: 7.0; typical 7.0-9.0 for ACE-Step 1.5) |
| **sampler_name** | - | Sampler (euler, dpmpp, etc.) |
| **scheduler** | - | Scheduler (normal, karras, exponential, etc.; default: normal) |
| **denoise** | 0.0 - 1.0 | Denoising strength (1.0 = fresh generation, < 1.0 = editing) |
| **infer_method** | ode/sde | ODE keeps the selected sampler behavior; SDE remaps default Euler/Heun choices to a stochastic sampler |
| **guidance_mode** | apg/adg/standard_cfg | Guidance type (default: apg) |
| **duration** | 0.0 - 600.0 | Duration in seconds (default: 60.0, 0 = auto) |
| **bpm** | 0 - 300 | Beats per minute (0 = auto, model decides) |
| **timesignature** | auto/2/3/4/6 | Time signature numerator |
| **language** | - | Lyric language (en, ja, zh, es, pt, etc.) |
| **keyscale** | auto/... | Key and scale (e.g., "C major" or "D minor") |

### Optional Parameters

#### Batch Generation
- **batch_size** (1-16): Number of audios to generate in parallel

#### Audio Input
- **latent_or_audio**: Base input for refinement (img2img). Accepts AUDIO or LATENT. Use `denoise < 1.0` to refine this input. With `duration=0`, duration is derived from the connected input.
- **lora**: AceStep LoRA stack from one or more `AceStep 1.5 SFT Lora Loader` nodes

#### LLM Configuration (Audio Code Generation)
- **generate_audio_codes** (default: True): Enable/disable LLM audio code generation for semantic structure
- **lm_cfg_scale** (0.0-100.0, default: 2.0): LLM classifier-free guidance scale
- **lm_temperature** (0.0-2.0, default: 0.85): LLM sampling temperature
- **lm_top_p** (0.0-2000.0, default: 0.9): Nucleus sampling parameter
- **lm_top_k** (0-100, default: 0): Top-k sampling
- **lm_min_p** (0.0-1.0, default: 0.0): Minimum probability threshold
- **lm_negative_prompt**: Negative prompt for LLM CFG

#### Latent Post-processing
- **latent_shift** (-0.2-0.2, default: 0.0): Additive shift (anti-clipping)
- **latent_rescale** (0.5-1.5, default: 1.0): Multiplicative scaling
- **normalize_peak** (default: False): Legacy hard normalization to 0 dBFS after VAE decode
- **enable_normalization** (default: True): Peak-normalize output to a target dBFS level
- **normalization_db** (-10.0-0.0, default: -1.0): Target peak level when normalization is enabled
- **fade_in_duration / fade_out_duration** (0.0-10.0, default: 0.0): Optional linear fades after normalization
- **use_tiled_vae** (default: True): Uses tiled VAE encode/decode for better long-audio and low-VRAM robustness
- **voice_boost** (-12.0-12.0, default: 0.0): Simple output gain in dB before normalization

#### APG Configuration
- **apg_momentum** (-1.0-1.0, default: -0.75): Momentum buffer coefficient
- **apg_norm_threshold** (0.0-10.0, default: 2.5): Norm threshold for gradient clipping

#### Extended Guidance Controls
- **guidance_interval** (-1.0-1.0, default: 0.5): Official centered guidance interval control
- **guidance_interval_decay** (0.0-1.0, default: 0.0): Linear decay inside the active guidance interval
- **min_guidance_scale** (0.0-30.0, default: 3.0): Lower bound when interval decay is enabled
- **guidance_scale_text** (-1.0-30.0, default: -1.0): Text-only guidance scale, `-1` inherits `cfg`
- **guidance_scale_lyric** (-1.0-30.0, default: -1.0): Lyric-only delta guidance scale, `-1` inherits `cfg`
- **omega_scale** (-8.0-8.0, default: 0.0): Mean-preserving output reweighting
- **erg_scale** (-0.9-2.0, default: 0.0): Prompt/lyric conditioning energy reweighting

#### Guidance Interval
- **cfg_interval_start** (0.0-1.0, default: 0.0): Start applying guidance at this schedule fraction
- **cfg_interval_end** (0.0-1.0, default: 1.0): Stop applying guidance at this schedule fraction

#### Custom Timesteps
- **shift** (1.0-5.0, default: 3.0): Schedule shift (3.0 = Gradio default)
- **custom_timesteps**: Custom comma-separated timesteps (overrides steps, shift, scheduler)

## 🔍 How It Works - Technical Foundation

### 1. Latent Pipeline

The node automatically manages latent creation or reuse:

```
├─ If latent_or_audio provided:
│  ├─ AUDIO: Resamples to VAE SR (48kHz), normalizes channels, encodes via VAE
│  ├─ LATENT: Uses directly as latent_image
│  └─ Duration derived from input when duration=0
│
└─ If no latent_or_audio:
   └─ Creates zero latent (pure noise) [batch_size, 64, latent_length]
```

**Automatic Sizing**: Duration in seconds is converted to latent length via:
```
latent_length = max(10, round(duration * vae_sample_rate / 1920))
```

### 2. Auto-Duration Estimation

When `duration <= 0`, the node analyzes lyric structure:

```
[Intro/Outro] = 8 beats (~1 bar 4/4)
[Instrumental/Solo] = 16 beats (~2 bars 4/4)  
Verse/Chorus → ~2 beats per 2 words (typical singing rate)
Section transitions = 4 beats
Empty lines = 2 beats (pause)
```

Result: `duration = beats * (60 / bpm)`

### 3. Metadata Processing

Metadata (bpm, duration, key/scale, time sig) are encoded in multiple representations:

1. **Structured YAML** (Chain-of-Thought):
```yaml
bpm: 120
caption: "upbeat electronic dance"
duration: 120
keyscale: "G major"
language: "en"
timesignature: 4
```

2. **LLM Template** (for audio code generation via Qwen):
```
<|im_start|>system
# Instruction
Generate audio semantic tokens...
<|im_end|>
<|im_start|>user
# Caption
upbeat electronic dance

# Lyric
[Verse 1]...
<|im_end|>
<|im_start|>assistant
<think>
{YAML above}
</think>

<|im_end|>
```

3. **Qwen3-0.6B Template** (direct metadata):
```
# Instruction
# Caption
upbeat electronic dance

# Metas
- bpm: 120
- timesignature: 4
- keyscale: G major
- duration: 120 seconds
<|endoftext|>
```

### 4. Guidance Strategy

#### APG (Adaptive Projected Guidance) - **Recommended**

```python
# Phase 1: Compute conditional difference
diff = pred_cond - pred_uncond

# Phase 2: Apply smooth momentum
if momentum_buffer:
    diff = momentum * running_avg + diff

# Phase 3: Norm clipping
norm = ||diff||₂
scale = min(1, norm_threshold / norm)
diff = diff * scale

# Phase 4: Orthogonal decomposition
diff_parallel = projection of diff onto pred_cond
diff_orthogonal = diff - diff_parallel

# Phase 5: Final guidance
guidance = pred_cond + (cfg_scale - 1) * (diff_orthogonal + eta * diff_parallel)
```

**Why It Works**: 
- **Orthogonal projection** removes collinear components that amplify noise
- **Momentum** smooths large jumps between timesteps
- **Adaptive clipping** prevents gradient explosion
- Result: **cleaner and more stable audio**

#### ADG (Angle-based Dynamic Guidance)

```
# Based on cosine angles between x0_cond and x0_uncond
# Dynamically adjusts guidance based on alignment
# Uses trigonometry for aggressive style deformation
```

### 5. Latent Refinement (img2img)

When `latent_or_audio` is connected with `denoise < 1.0`, the node operates in img2img mode:

- The input audio is encoded via VAE (or the latent is used directly)
- A fraction of noise is added based on `denoise` strength
- The diffusion model refines the noisy latent while preserving the original structure

## 🎚️ Quality Tips

- Use `guidance_mode=apg` with `steps=50` to `64` for best quality
- For img2img refinement, start with `denoise=0.5` to `0.7` to preserve the original character
- Mild vocal hiss is usually a generation artifact; APG and slightly higher step counts generally help more than raw `cfg`
- Simplify overly dense or contradictory tags for cleaner results

## 📊 Guidance Modes Comparison

| Aspect | APG | ADG | Standard CFG |
|--------|-----|-----|----------|
| **Quality** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Stability** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ |
| **Dynamics** | Natural | Aggressive | Predictable |
| **Computation** | Normal | Normal | Minimal |
| **Recommended** | ✅ Yes | For extreme styles | Baseline |

## 🎨 Workflow Examples

### Example 1: Quality Baseline (Recommended)

```
AceStepSFTGenerate:
  caption: "upbeat electronic dance music with synthesizers"
  lyrics: [Instrumental]
  instrumental: True
  duration: 60.0
  cfg: 7.0
  steps: 50
  sampler_name: "euler"
  scheduler: "normal"
  guidance_mode: "apg"
  → Generates a strong 60s ACE-Step 1.5 SFT baseline render
```

### Example 2: Audio Refinement (img2img)

```
AceStepSFTGenerate:
  latent_or_audio: (mixer output)
  caption: "make it more orchestral"
  denoise: 0.7 (preserves 30% of source)
  duration: 0 (uses input duration)
  → Refines audio while preserving original characteristics
```

### Example 3: Batch Generation with Varied Seeds

```
AceStepSFTGenerate:
  batch_size: 4
  seed: 42 (varies automatically)
  → Creates 4 variations with similar characteristics
```

### Example 4: Chained LoRAs

```
AceStep 1.5 SFT Lora Loader:
  lora_name: "Ace-Step1.5/ace-step15-style1.safetensors"
  strength_model: 0.7
  strength_clip: 0.0
  ↓
AceStep 1.5 SFT Lora Loader:
  lora_name: "Ace-Step1.5/Ace-Step1.5-TechnoRain.safetensors"
  strength_model: 0.35
  strength_clip: 0.0
  ↓
AceStep 1.5 SFT Generate:
  lora: (stack output)
```

Note: AceStep LoRAs are now supported directly by this package. If a specific LoRA produces unstable audio, start by lowering `strength_model` and compare `apg` against `standard_cfg`.

### Example 5: Music Analysis → Generation Pipeline

```
AceStepSFTMusicAnalyzer:
  audio: (input audio file)
  model: "Qwen2-Audio-7B-Instruct"
  → tags: "dancehall beat, powerful bassline, vocal samples, melancholic"
  → bpm: "129bpm"
  → keyscale: "G minor"
  ↓
AceStepSFTGenerate:
  caption: (tags from analyzer)
  bpm: 129
  keyscale: "G minor"
  → Generates new music matching the analyzed style
```

## 🐛 Troubleshooting

### Audio Distortion/Clipping

**Solution**: Use negative `latent_shift` (e.g., -0.1) to reduce amplitude before VAE decoding

### High Variance Results

**Solution**: Increase `apg_norm_threshold` (e.g., 3.0-4.0) for more gradient clipping

### Lower Than Expected Quality

**Solution**: 
1. Use `guidance_mode: "apg"` (recommended)
2. Start from `steps: 50`, `cfg: 7.0`, `sampler_name: "euler"`, `scheduler: "normal"`, `infer_method: "ode"`
3. Keep `enable_normalization: True` with `normalization_db: -1.0` for cleaner final level management

### LoRA Sounds Deformed or Overcooked

**Solution**:
1. Lower `strength_model` first, e.g. `0.2` to `0.6`
2. Set `strength_clip` to `0.0` unless the LoRA explicitly targets the text encoders
3. Compare `guidance_mode: "standard_cfg"` vs `"apg"` for that LoRA
4. Avoid stacking multiple strong LoRAs at full strength

### LoRA Dimension Mismatch Error (`The size of tensor a must match...`)

**Cause**: DoRA LoRAs store `dora_scale` as a 1D tensor `[N]`. ComfyUI's `weight_decompose` divides it by `weight_norm [N,1]`, which causes PyTorch to broadcast `[1,N]/[N,1]` → `[N,N]` instead of the expected `[N,1]`.

**Solution**: This is automatically fixed by the node — all `dora_scale` tensors are unsqueezed to 2D `[N,1]` at load time. If you still see this error, ensure you are using the latest version of this node.

### PEFT/DoRA LoRA Not Showing in Dropdown

**Solution**:
1. Place the PEFT folder (containing `adapter_config.json` + `adapter_model.safetensors`) inside `ComfyUI-AceStep_SFT/Loras/`
2. Restart ComfyUI — the conversion runs automatically on startup
3. Check the console for `[AceStep SFT] Converted PEFT/DoRA → ComfyUI: ...` message
4. The converted file appears as `*_comfyui.safetensors` in the dropdown

### Slow Generation

**Solution**: Reduce `batch_size`, lower `steps` to ~20, or use "karras" scheduler

## 📚 Technical References

- **AceStep 1.5**: ICML 2024 (Learning Universal Features for Efficient Audio Generation)
- **Flow Matching**: Liphardt et al. 2024 (Generative Modeling by Estimating Gradients of the Data Distribution)
- **APG/ADG**: Techniques aligned with official AceStep paper
- **ComfyUI**: Modular node graph architecture for batch generation

## 📝 License

MIT License - Feel free to use in personal or commercial projects

## 🤝 Contributing

Issues and PRs are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## ⚠️ Important Notes

- **Recommended maximum duration**: 240 seconds (GPU memory)
- **Maximum batch size**: Depends on your GPU (start with 1-2)
- **SFT models**: These models are specific to Supervised Fine-Tuning - not tested with non-SFT models
- **Rights and attribution**: Respect model and dataset usage rights

---

**Built on the AceStep SFT workflow and extended with advanced guidance and quality controls for ComfyUI.**

For bugs, questions, or suggestions: open an issue on the repository! 🎵
