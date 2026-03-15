# ComfyUI-AceStep SFT

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)

An all-in-one node for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) that implements **AceStep 1.5 SFT** (Supervised Fine-Tuning), a state-of-the-art music generation model. This node replicates the complete functionality of the official Gradio pipeline, offering fine-grained control over audio synthesis parameters.

> **SFT = Supervised Fine-Tuning**: A specialized version of AceStep optimized for generating superior quality audio through supervised training.

## 📋 Overview

The **AceStepSFTGenerate** is a unified node that encapsulates the entire music generation workflow:

1. **Latent Creation** - Generates initial latents or loads from source audio
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

### 🔄 Audio Editing & Style Transfer

- **Source Audio Denoising**: Use `denoise < 1.0` with source audio for editing
- **Timbre Transfer**: Reference audio for style transfer
- **Batch Generation**: Generate multiple variations in parallel

### 🛠️ Latent Post-processing

- **Latent Shift**: Additive anti-clipping correction
- **Latent Rescale**: Multiplicative scaling for dynamic control

## 📦 Installation

### Prerequisites

- ComfyUI installed and functional
- CUDA/GPU or equivalent (modern processors)
- Required model files:
  - Diffusion model (DiT): `acestep_v1.5_sft.safetensors`
  - Text Encoders: `qwen_0.6b_ace15.safetensors`, `qwen_1.7b_ace15.safetensors` (or 4B)
  - VAE: `ace_1.5_vae.safetensors`

### Download Model Files

Download the required models from HuggingFace:

1. **Diffusion Model (SFT)**:
   - [AceStep 1.5 SFT Model](https://huggingface.co/ACE-Step/acestep-v15-sft/blob/main/model.safetensors)

2. **Text Encoders** (choose any versions):
   - [Text Encoders Collection](https://huggingface.co/Comfy-Org/ace_step_1.5_ComfyUI_files/tree/main/split_files/text_encoders)
     - `qwen_0.6b_ace15.safetensors` (caption processing)
     - `qwen_1.7b_ace15.safetensors` or `qwen_4b_ace15.safetensors` (audio code generation)

3. **VAE** (Audio codec):
   - [AceStep 1.5 VAE](https://huggingface.co/Comfy-Org/ace_step_1.5_ComfyUI_files/blob/main/split_files/vae/ace_1.5_vae.safetensors)

### Installation Steps

1. Clone the repository to your custom nodes folder:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/your-username/ComfyUI-AceStep_SFT.git
```

2. Place model files in the appropriate directories:
```
ComfyUI/models/diffusion_models/     # AceStep 1.5 SFT model
ComfyUI/models/text_encoders/        # Qwen encoders
ComfyUI/models/vae/                  # VAE
```

3. Restart ComfyUI - the node will appear under `audio/AceStep SFT`

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
| **steps** | 1 - 200 | Diffusion inference steps (default: 32) |
| **cfg** | 1.0 - 20.0 | Classifier-free guidance scale (default: 7.0) |
| **sampler_name** | - | Sampler (euler, dpmpp, etc.) |
| **scheduler** | - | Scheduler (karras, exponential, etc.) |
| **denoise** | 0.0 - 1.0 | Denoising strength (1.0 = fresh generation, < 1.0 = editing) |
| **guidance_mode** | apg/adg/standard_cfg | Guidance type (default: apg) |
| **duration** | 0.0 - 600.0 | Duration in seconds (0 = auto) |
| **bpm** | 0 - 300 | Beats per minute (0 = auto, model decides) |
| **timesignature** | auto/2/3/4/6 | Time signature numerator |
| **language** | - | Lyric language (en, ja, zh, es, pt, etc.) |
| **keyscale** | auto/... | Key and scale (e.g., "C major" or "D minor") |

### Optional Parameters

#### Batch Generation
- **batch_size** (1-16): Number of audios to generate in parallel

#### Audio Inputs
- **source_audio**: Source audio for denoising/editing with `denoise < 1.0`
- **reference_audio**: Reference audio for timbre/style transfer

#### LLM Configuration (Audio Code Generation)
- **generate_audio_codes** (default: True): Enable/disable LLM audio code generation
- **lm_cfg_scale** (0.0-100.0, default: 2.0): LLM classifier-free guidance scale
- **lm_temperature** (0.0-2.0, default: 0.85): LLM sampling temperature
- **lm_top_p** (0.0-2000.0, default: 0.9): Nucleus sampling parameter
- **lm_top_k** (0-100, default: 0): Top-k sampling
- **lm_min_p** (0.0-1.0, default: 0.0): Minimum probability threshold
- **lm_negative_prompt**: Negative prompt for LLM CFG

#### Latent Post-processing
- **latent_shift** (-0.2-0.2, default: 0.0): Additive shift (anti-clipping)
- **latent_rescale** (0.5-1.5, default: 1.0): Multiplicative scaling

#### APG Configuration
- **apg_momentum** (-1.0-1.0, default: -0.75): Momentum buffer coefficient
- **apg_norm_threshold** (0.0-10.0, default: 2.5): Norm threshold for gradient clipping

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
├─ If source_audio provided:
│  ├─ Resamples to VAE SR (48kHz default)
│  ├─ Normalizes channels (mono→stereo, truncates >2ch)
│  └─ Encodes via VAE to latent_image
│
└─ If no source_audio:
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

### 5. Exact Gradio Pipeline Matching

The node is engineered to **replicate byte-for-byte** the official pipeline:

✅ Identical LLM prompts (same YAML format, same CoT structure)  
✅ Audio encoding via same Qwen encoders  
✅ Same VAE and timestep scheduler  
✅ Identical peak normalization (prevent clipping)  
✅ Audio codes support in negative conditioning  

## 📊 Guidance Modes Comparison

| Aspect | APG | ADG | Standard CFG |
|--------|-----|-----|----------|
| **Quality** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Stability** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ |
| **Dynamics** | Natural | Aggressive | Predictable |
| **Computation** | Normal | Normal | Minimal |
| **Recommended** | ✅ Yes | For extreme styles | Baseline |

## 🎨 Workflow Examples

### Example 1: Basic Generation (Recommended)

```
AceStepSFTGenerate:
  caption: "upbeat electronic dance music with synthesizers"
  lyrics: [Instrumental]
  duration: 0 (auto)
  cfg: 7.0
  steps: 32
  guidance_mode: "apg"
  → Generates 30s of high-quality electronic music
```

### Example 2: Source Audio Editing

```
AceStepSFTGenerate:
  source_audio: (mixer output)
  caption: "make it more orchestral"
  denoise: 0.7 (preserves 30% of source)
  duration: 0 (uses source duration)
  → Transforms audio while preserving original characteristics
```

### Example 3: Timbre Transfer

```
AceStepSFTGenerate:
  reference_audio: (piano sample)
  caption: "upbeat pop song"
  lyrics: "Verse 1..."
  generate_audio_codes: False
  → Synthesizes using timbral characteristics from piano
```

### Example 4: Batch Generation with Varied Seeds

```
AceStepSFTGenerate:
  batch_size: 4
  seed: 42 (varies automatically)
  → Creates 4 variations with similar characteristics
```

## 🐛 Troubleshooting

### Audio Distortion/Clipping

**Solution**: Use negative `latent_shift` (e.g., -0.1) to reduce amplitude before VAE decoding

### High Variance Results

**Solution**: Increase `apg_norm_threshold` (e.g., 3.0-4.0) for more gradient clipping

### Lower Than Expected Quality

**Solution**: 
1. Use `guidance_mode: "apg"` (recommended)
2. Increase `steps` (e.g., 50-100 for maximum quality)
3. Adjust `cfg` to 6.0-8.0

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

**Engineered to precisely replicate the official AceStep SFT pipeline in ComfyUI.**

For bugs, questions, or suggestions: open an issue on the repository! 🎵
