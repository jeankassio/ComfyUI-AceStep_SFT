# ComfyUI-AceStep SFT

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)

Um node all-in-one para [ComfyUI](https://github.com/comfyanonymous/ComfyUI) que implementa **AceStep 1.5 SFT** (Supervised Fine-Tuning), um modelo de geração de música de alta qualidade. Ele parte do workflow oficial do AceStep e o estende com controles extras de conditioning e opções práticas de qualidade para uso no ComfyUI.

> **SFT = Supervised Fine-Tuning**: Uma versão especializada do AceStep otimizada para gerar áudio com qualidade superior através de treinamento supervisionado.

## 📋 Visão Geral

Este pacote atualmente fornece quatro nodes em `audio/AceStep SFT`:

- **AceStep 1.5 SFT Generate**: geração, edição e decodificação all-in-one
- **AceStep 1.5 SFT Music Analyzer**: análise de áudio com IA (tags, BPM, tom/escala)
- **AceStep 1.5 SFT Lora Loader**: construtor encadeável de stack de LoRA para AceStep 1.5 SFT
- **AceStep 1.5 SFT Turbo Tag Adapter**: reescreve tags do Turbo em tags mais curtas e amigáveis ao SFT

O **AceStepSFTGenerate** é um node unificado que encapsula todo o fluxo de trabalho de geração de música:

1. **Criação de Latentes** - Gera latentes iniciais ou carrega a partir de `latent_or_audio`
2. **Codificação de Texto** - Processa caption, letras e metadados via múltiplos encoders CLIP
3. **Amostragem de Difusão** - Executa o diffusion model com controle avançado de guidance
4. **Decodificação de Áudio** - Converte latentes em áudio de alta qualidade via VAE

### Exemplo de Uso

![AceStep SFT Node Configuration](example.png)

## 🎯 Recursos Principais

### ✨ Guidance Avançada

O node suporta três modos de guidance classif-livres, cada um com características únicas:

- **APG (Adaptive Projected Guidance)** ⭐ *Recomendado*
  - Adaptação dinâmica baseada em momentum
  - Clipping de gradientes com threshold adaptativo
  - Projeção ortogonal para evitar ruído indesejado
  - **Padrão do AceStep SFT** - oferece o melhor equilíbrio entre qualidade e estabilidade

- **ADG (Angle-based Dynamic Guidance)**
  - Guidance baseada em ângulos cosine entre condições
  - Operação em espaço de velocidade (flow matching)
  - Ideal para distorção de estilo mais agressiva
  - Clipping adaptativo baseado no ângulo entre x0_cond e x0_uncond

- **Standard CFG**
  - Classifier-Free Guidance tradicional
  - Implementação simples e previsível
  - Útil como baseline de comparação

### 🎵 Processamento Inteligente de Metadados

- **Auto-duração**: Estima automaticamente a duração da música analisando a estrutura das letras
- **Codificação LLM**: Utilize Qwen LLM (0.6B ou 1.7B/4B) para gerar códigos semânticos de áudio
- **Valores Auto**: BPM, Time Signature e Key/Scale automáticos (modelo decide)
- **Suporte multilíngue**: Mais de 23 idiomas suportados

### 🎧 Analisador Musical com IA

- **Extração de Tags**: Usa o ACE-Step-Transcriber nativo para extrair tags de letra, voz e estrutura musical do áudio
- **Detecção de BPM**: Detecção automática de tempo via librosa
- **Detecção de Tom/Escala**: Detecta tonalidade e escala (ex: "G minor")
- **Saída JSON**: Output estruturado `music_infos` com todos os resultados
- **Parâmetros de Geração**: Controle de temperature, top_p, top_k, repetition_penalty e seed
- **Download Automático**: Modelos são baixados no primeiro uso (~1-7 GB cada)

#### Modelo de Análise Nativo:

| Modelo | Tamanho | Tipo | Ideal Para |
|--------|---------|------|------------|
| ACE-Step-Transcriber | Download de 22.4 GB | Audio-to-Text | Workflow nativo do ACE-Step 1.5 para letras, voz cantada, estrutura musical e instrumentos opcionais |

Este node agora é dedicado ao workflow nativo do ACE-Step-Transcriber. Ele usa o prompt nativo do modelo, a saída estruturada de transcrição e deriva tags a partir do idioma, da letra, de marcações como verse/chorus/bridge e de instrumentos opcionais quando disponíveis.

### 🔄 Refinamento de Áudio (img2img)

- **Refinamento baseado em latentes**: Use `denoise < 1.0` com `latent_or_audio` conectado para refinar áudio existente
- **Aceita AUDIO ou LATENT**: Conecte qualquer saída de áudio ou latent para edição no estilo img2img
- **Tamanho de Lote**: Gere múltiplas variações em paralelo

### 🧠 Controle Estendido de Conditioning

- **Guidance separada para texto/letra**: `guidance_scale_text` e `guidance_scale_lyric`
- **Omega Scale**: reweighting com preservação da média para aproximar o scheduler do AceStep
- **Aproximação ERG**: reweighting local da energia do prompt via `erg_scale`
- **Decaimento do intervalo de guidance**: reduz a força da guidance dentro da faixa ativa

### 🎚️ Workflow com LoRA do AceStep

- **Lora Loader encadeável**: empilhe um ou mais LoRAs antes da geração
- **Forças separadas**: `strength_model` e `strength_clip` independentes
- **Input único no Generate**: a stack final entra no input `lora`
- **Pasta local `Loras/`**: solte arquivos LoRA diretamente na pasta `Loras/` do node — são registrados automaticamente na inicialização
- **Conversão automática PEFT/DoRA**: LoRAs em formato PEFT (`adapter_config.json` + `adapter_model.safetensors`) colocados em `Loras/` são convertidos automaticamente para formato ComfyUI na primeira inicialização
- **Suporte DoRA**: Suporte completo a DoRA (Weight-Decomposed Low-Rank Adaptation) com correção automática de dimensão do `dora_scale` para compatibilidade com ComfyUI

### 🛠️ Pós-processamento Latente

- **Latent Shift**: Correção anti-clipping aditiva
- **Latent Rescale**: Scaling multiplicativo para controle dinâmico

## 📦 Instalação

### Pré-requisitos

- ComfyUI instalado e funcional
- CUDA/GPU ou equivalente (processadores modernos)
- Recomendado para melhor qualidade de saída (com base em testes práticos): usar o modelo merged SFT+Turbo.
- Models necessários:
  - Diffusion model: `acestep_v1.5_sft.safetensors`
  - Text Encoders: `qwen_0.6b_ace15.safetensors` e `qwen_1.7b_ace15.safetensors` (ou 4B)
  - VAE: `ace_1.5_vae.safetensors`

### Download dos Modelos

Baixe os modelos necessários no HuggingFace:

1. **Diffusion Model (Recomendado: merged SFT+Turbo)**:
  - [AceStep 1.5 Merged SFT+Turbo Model](https://huggingface.co/Aryanne/acestep-v15-test-merges/blob/main/acestep_v1.5_merge_sft_turbo_ta_0.5.safetensors)

2. **Modelo alternativo (SFT oficial)**:
  - [AceStep 1.5 SFT Model](https://huggingface.co/ACE-Step/acestep-v15-sft/blob/main/model.safetensors)

3. **Text Encoders** (escolha qualquer versão):
  - [Coleção de Text Encoders](https://huggingface.co/Comfy-Org/ace_step_1.5_ComfyUI_files/tree/main/split_files/text_encoders)
    - `qwen_0.6b_ace15.safetensors` (processamento de caption)
    - `qwen_1.7b_ace15.safetensors` ou `qwen_4b_ace15.safetensors` (geração de códigos de áudio)

4. **VAE** (codec de áudio):
  - [AceStep 1.5 VAE](https://huggingface.co/Comfy-Org/ace_step_1.5_ComfyUI_files/blob/main/split_files/vae/ace_1.5_vae.safetensors)

### Passos

1. Clone o repositório na pasta de custom nodes:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/jeankassio/ComfyUI-AceStep_SFT.git
```

2. Os arquivos de modelo devem ser colocados em:
```
ComfyUI/models/diffusion_models/     # arquivo .safetensors do model DiT
ComfyUI/models/text_encoders/        # arquivos dos encoders CLIP
ComfyUI/models/vae/                  # arquivo da VAE
ComfyUI/models/loras/                # LoRAs opcionais do AceStep 1.5
```

3. **(Opcional) Coloque LoRAs na pasta local:**
```
ComfyUI/custom_nodes/ComfyUI-AceStep_SFT/Loras/   # Pasta local de LoRAs
```
   Você pode colocar LoRAs aqui em **qualquer** um destes formatos:
   - **Formato ComfyUI**: Arquivo `.safetensors` único (pronto para uso)
   - **Formato PEFT/DoRA**: Pasta contendo `adapter_config.json` + `adapter_model.safetensors` (convertido automaticamente na inicialização)
   - **Artefatos de zip aninhados**: Se o zip extraiu uma pasta-dentro-de-pasta, o node detecta e corrige automaticamente

4. Reinicie o ComfyUI - o node aparecerá em `audio/AceStep SFT`

## 🧩 Nodes Disponíveis

### AceStep 1.5 SFT Generate

Node principal all-in-one para text-to-music, refinamento de áudio baseado em latentes e decodificação via VAE.

### AceStep 1.5 SFT Music Analyzer

Node de análise de áudio com IA que extrai tags descritivas, BPM e tom/escala a partir de áudio.

Entradas:
- `audio`: Áudio de entrada para análise
- `model`: Seleção do modelo de IA (9 modelos, download automático)
- `get_tags` / `get_bpm` / `get_keyscale`: Ativar/desativar cada análise
- `max_new_tokens`: Máximo de tokens para modelos generativos
- `audio_duration`: Máximo de segundos de áudio para análise
- `temperature`, `top_p`, `top_k`, `repetition_penalty`, `seed`: Parâmetros de geração
- `unload_model`: Liberar VRAM após análise
- `use_flash_attn`: Ativar Flash Attention 2 (se compatível)

Saídas:
- `tags`: Tags descritivas separadas por vírgula (STRING)
- `bpm`: BPM detectado, ex: "129bpm" (STRING)
- `keyscale`: Tom e escala, ex: "G minor" (STRING)
- `music_infos`: JSON com todos os resultados (STRING)

### AceStep 1.5 SFT Lora Loader

Node utilitário encadeável que monta uma stack de LoRA para o AceStep 1.5 SFT.

Entradas:
- `lora_name`: arquivo LoRA em `ComfyUI/models/loras` ou na pasta local `Loras/`
- `strength_model`: força aplicada ao diffusion model
- `strength_clip`: força aplicada à pilha de text encoders
- `lora` (opcional): stack AceStep LoRA vinda de outro loader

Saída:
- `lora`: conecte em outro Lora Loader ou diretamente no Generate

#### Formatos de LoRA Suportados

| Formato | O que colocar em `Loras/` | Ação |
|---------|--------------------------|------|
| ComfyUI `.safetensors` | Arquivo único | Usado diretamente |
| Diretório PEFT/DoRA | Pasta com `adapter_config.json` + `adapter_model.safetensors` | Auto-convertido para `*_comfyui.safetensors` na inicialização |
| Artefato de zip aninhado | Pasta contendo um `.safetensors` dentro | Auto-extraído para a raiz na inicialização |

A auto-conversão trata:
- Remapeamento de chaves: `lora_A`/`lora_B` → `lora_down`/`lora_up`
- Suporte DoRA: `lora_magnitude_vector` → `dora_scale` (com shape 2D correto)
- Injeção de alpha por camada a partir do `adapter_config.json` (suporta `alpha_pattern` e `rank_pattern`)

## 🎛️ Parâmetros do Node

### Parametros Obrigatórios

| Parâmetro | Intervalo | Descrição |
|-----------|-----------|-----------|
| **diffusion_model** | - | Caminho do modelo DiT (AceStep 1.5 SFT) |
| **text_encoder_1** | - | Encoder Qwen3 0.6B (processamento de caption) |
| **text_encoder_2** | - | Encoder Qwen3 1.7B/4B (LLM para códigos de áudio) |
| **vae_name** | - | VAE do AceStep 1.5 |
| **caption** | - | Descrição textual da música (gênero, mood, instrumentos) |
| **lyrics** | - | Letras ou `[Instrumental]` |
| **instrumental** | boolean | Força modo instrumental (sobrescreve letras) |
| **seed** | 0 - 2^64 | Seed para reprodutibilidade |
| **steps** | 1 - 200 | Passos de difusão (padrão: 50 para ACE-Step 1.5 SFT) |
| **cfg** | 1.0 - 20.0 | Classifier-free guidance scale (padrão: 7.0; faixa típica 7.0-9.0 no ACE-Step 1.5) |
| **sampler_name** | - | Sampler (euler, dpmpp, etc.) |
| **scheduler** | - | Scheduler (normal, karras, exponential, etc.; padrão: normal) |
| **denoise** | 0.0 - 1.0 | Força de denoising (1.0 = novo, < 1.0 = edição) |
| **infer_method** | ode/sde | ODE mantém o comportamento do sampler escolhido; SDE remapeia Euler/Heun padrão para um sampler estocástico |
| **guidance_mode** | apg/adg/standard_cfg | Tipo de guidance (padrão: apg) |
| **duration** | 0.0 - 600.0 | Duração em segundos (padrão: 60.0, 0 = auto) |
| **bpm** | 0 - 300 | BPM (0 = auto, modelo decide) |
| **timesignature** | auto/2/3/4/6 | Numerador da fórmula de compasso |
| **language** | - | Idioma da letra (en, ja, zh, es, pt, etc.) |
| **keyscale** | auto/... | Tom e escala ex: "C major" ou "D minor" |

### Parâmetros Opcionais

#### Geração em Lote
- **batch_size** (1-16): Número de áudios para gerar em paralelo

#### Entrada de Áudio
- **latent_or_audio**: Entrada base para refinamento (img2img). Aceita AUDIO ou LATENT. Use `denoise < 1.0` para refinar esta entrada. Com `duration=0`, a duração é derivada da entrada conectada.
- **lora**: stack AceStep LoRA vinda de um ou mais nodes `AceStep 1.5 SFT Lora Loader`

#### Configuração LLM (Geração de Códigos de Áudio)
- **generate_audio_codes** (default: True): Ativar/desativar geração de códigos de áudio via LLM para estrutura semântica
- **lm_cfg_scale** (0.0-100.0, default: 2.0): CFG scale para LLM
- **lm_temperature** (0.0-2.0, default: 0.85): Temperatura de sampling
- **lm_top_p** (0.0-2000.0, default: 0.9): Nucleus sampling
- **lm_top_k** (0-100, default: 0): Top-k sampling
- **lm_min_p** (0.0-1.0, default: 0.0): Minimum probability
- **lm_negative_prompt**: Prompt negativo para CFG do LLM

#### Pós-processamento de Latentes
- **latent_shift** (-0.2-0.2, default: 0.0): Shift aditivo (anti-clipping)
- **latent_rescale** (0.5-1.5, default: 1.0): Scaling multiplicativo
- **normalize_peak** (default: False): Normalização legada para 0 dBFS após o decode da VAE
- **enable_normalization** (default: True): Normaliza o pico de saída para um alvo em dBFS
- **normalization_db** (-10.0-0.0, default: -1.0): Alvo de pico quando a normalização está ativa
- **fade_in_duration / fade_out_duration** (0.0-10.0, default: 0.0): Fades lineares opcionais após a normalização
- **use_tiled_vae** (default: True): Usa encode/decode tiled da VAE para áudio longo e menor VRAM
- **voice_boost** (-12.0-12.0, default: 0.0): Ganho simples em dB antes da normalização

#### Configuração APG
- **apg_momentum** (-1.0-1.0, default: -0.75): Coeficiente do buffer de momentum
- **apg_norm_threshold** (0.0-10.0, default: 2.5): Threshold de norma para clipping

#### Controles Estendidos de Guidance
- **guidance_interval** (-1.0-1.0, default: 0.5): Controle centralizado do intervalo oficial de guidance
- **guidance_interval_decay** (0.0-1.0, default: 0.0): Decaimento linear dentro do intervalo ativo
- **min_guidance_scale** (0.0-30.0, default: 3.0): Limite inferior quando o decaimento está habilitado
- **guidance_scale_text** (-1.0-30.0, default: -1.0): Guidance do branch de texto, `-1` herda `cfg`
- **guidance_scale_lyric** (-1.0-30.0, default: -1.0): Guidance do delta de letra, `-1` herda `cfg`
- **omega_scale** (-8.0-8.0, default: 0.0): Reweighting da saída com preservação da média
- **erg_scale** (-0.9-2.0, default: 0.0): Reweighting da energia do conditioning de prompt/letra

#### Intervalo de Guidance
- **cfg_interval_start** (0.0-1.0, default: 0.0): Iniciar guidance nesta fração do schedule
- **cfg_interval_end** (0.0-1.0, default: 1.0): Parar guidance nesta fração do schedule

#### Timesteps Personalizados
- **shift** (1.0-5.0, default: 3.0): Shift do schedule (3.0 = padrão Gradio)
- **custom_timesteps**: Timesteps customizados em formato CSV (ex: `0.97,0.76,0.615,...`)

## 🔍 Como Funciona - Fundamentação Técnica

### 1. Pipeline de Latentes

O node gerencia automaticamente a criação ou reutilização de latentes:

```
├─ Se latent_or_audio fornecido:
│  ├─ AUDIO: Reamostrado para VAE SR (48kHz), normaliza canais, codifica via VAE
│  ├─ LATENT: Usado diretamente como latent_image
│  └─ Duração derivada da entrada quando duration=0
│
└─ Se nenhum latent_or_audio:
   └─ Cria latente zero (ruído puro) [batch_size, 64, latent_length]
```

**Dimensionamento automático**: A duração em segundos é convertida em comprimento de latente via:
```
latent_length = max(10, round(duration * vae_sample_rate / 1920))
```

### 2. Estimativa Auto de Duração

Quando `duration <= 0`, o node analisa a estrutura das letras:

```
[Intro/Outro] = 8 beats (~1 bar 4/4)
[Instrumental/Solo] = 16 beats (~2 bars 4/4)  
Verso/Ch → ~2 beats por 2 palavras (ritmo típico de canto)
Secções de transição = 4 beats
Linhas vazias = 2 beats (pausa)
```

Resultado: `duration = beats * (60 / bpm)`

### 3. Processamento de Metadados

Os metadados (bpm, duration, key/scale, time sig) são incorporados em múltiplas representações:

1. **YAML estruturado** (Chain-of-Thought):
```yaml
bpm: 120
caption: "upbeat electronic dance"
duration: 120
keyscale: "G major"
language: "en"
timesignature: 4
```

2. **Template LLM** (para geração de códigos de áudio via Qwen):
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
{YAML acima}
</think>

<|im_end|>
```

3. **Template Qwen3-0.6B** (metadata direto):
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

### 4. Estratégia de Guidance

#### APG (Adaptive Projected Guidance) - **Recomendado**

```python
# Fase 1: Computar diferença condicional
diff = pred_cond - pred_uncond

# Fase 2: Aplicar momentum suave
if momentum_buffer:
    diff = momentum * running_avg + diff

# Fase 3: Clipping de norma
norm = ||diff||₂
scale = min(1, norm_threshold / norm)
diff = diff * scale

# Fase 4: Decomposição ortogonal
diff_parallel = projeção de diff em pred_cond
diff_orthogonal = diff - diff_parallel

# Fase 5: Guidance final
guidance = pred_cond + (cfg_scale - 1) * (diff_orthogonal + eta * diff_parallel)
```

**Por que funciona**: 
- A **projeção ortogonal** remove componentes colineares que amplificam ruído
- O **momentum** suaviza grandes saltos entre timesteps
- O **clipping adaptativo** previne explosão de gradientes
- Isso resulta em **audio mais limpo e estável visualmente**

#### ADG (Angle-based Dynamic Guidance)

```
# Baseado em ângulos cosine entre x0_cond e x0_uncond
# Ajusta a guidance dinamicamente baseado no alinhamento
# Usa trigonometria para deformação de estilo mais agressiva
```

### 5. Refinamento com Latentes (img2img)

Quando `latent_or_audio` está conectado com `denoise < 1.0`, o node opera em modo img2img:

- O áudio de entrada é codificado via VAE (ou o latent é usado diretamente)
- Uma fração de ruído é adicionada conforme a força de `denoise`
- O diffusion model refina o latent ruidoso preservando a estrutura original

## 🎚️ Dicas de Qualidade

- Use `guidance_mode=apg` com `steps=50` a `64` para melhor qualidade
- Para refinamento img2img, comece com `denoise=0.5` a `0.7` para preservar o caráter original
- Um leve chiado vocal geralmente é artefato de geração; APG e um número um pouco maior de steps costumam ajudar mais do que apenas subir `cfg`
- Simplifique tags muito densas ou contraditórias para resultados mais limpos

## 📊 Comparação de Modos de Guidance

| Aspecto | APG | ADG | Standard CFG |
|---------|-----|-----|----------|
| **Qualidade** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Estabilidade** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ |
| **Dinâmica** | Natural | Agressiva | Previsível |
| **Computação** | Normal | Normal | Mínima |
| **Recomendado** | ✅ Sim | Para estilos extremos | Baseline |

## 🎨 Exemplos de Workflow

### Exemplo 1: Baseline de Qualidade (Recomendado)

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
  → Gera um render de 60s com a baseline do ACE-Step 1.5 SFT
```

### Exemplo 2: Refinamento de Áudio (img2img)

```
AceStepSFTGenerate:
  latent_or_audio: (saída do mixer)
  caption: "make it more orchestral"
  denoise: 0.7 (preserva 30% da fonte)
  duration: 0 (usa a duração da entrada)
  → Refina o áudio preservando características originais
```

### Exemplo 3: Geração em Lote com Seed Variado

```
AceStepSFTGenerate:
  batch_size: 4
  seed: 42 (varia automaticamente)
  → Cria 4 variações com características similares
```

### Exemplo 4: LoRAs Encadeados

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
  lora: (saída da stack)
```

Observação: os LoRAs do AceStep agora são suportados diretamente por este pacote. Se um LoRA específico gerar áudio instável, comece reduzindo `strength_model` e compare `apg` com `standard_cfg`.

### Exemplo 5: Pipeline Análise Musical → Geração

```
AceStepSFTMusicAnalyzer:
  audio: (arquivo de áudio de entrada)
  model: "Qwen2-Audio-7B-Instruct"
  → tags: "dancehall beat, powerful bassline, vocal samples, melancholic"
  → bpm: "129bpm"
  → keyscale: "G minor"
  ↓
AceStepSFTGenerate:
  caption: (tags do analyzer)
  bpm: 129
  keyscale: "G minor"
  → Gera nova música seguindo o estilo analisado
```

## 🐛 Troubleshooting

### Áudio com Distorção/Clipping

**Solução**: Use `latent_shift` negativo (ex: -0.1) para reduzir amplitude antes da decodificação VAE

### Resultado com Alta Variância

**Solução**: Aumente `apg_norm_threshold` (ex: 3.0-4.0) para mais clipping de gradientes

### Qualidade Inferior ao Esperado

**Solução**: 
1. Use `guidance_mode: "apg"` (recomendado)
2. Comece com `steps: 50`, `cfg: 7.0`, `sampler_name: "euler"`, `scheduler: "normal"`, `infer_method: "ode"`
3. Mantenha `enable_normalization: True` com `normalization_db: -1.0` para um nível final mais limpo

### LoRA Soando Deformado ou Forte Demais

**Solução**:
1. Reduza `strength_model` primeiro, por exemplo `0.2` a `0.6`
2. Deixe `strength_clip` em `0.0` a menos que o LoRA também tenha sido treinado para os text encoders
3. Compare `guidance_mode: "standard_cfg"` com `"apg"` para esse LoRA
4. Evite empilhar múltiplos LoRAs fortes todos em `1.0`

### Erro de Dimensão no LoRA (`The size of tensor a must match...`)

**Causa**: LoRAs DoRA armazenam `dora_scale` como tensor 1D `[N]`. A função `weight_decompose` do ComfyUI divide por `weight_norm [N,1]`, o que faz o PyTorch transmitir como `[1,N]/[N,1]` → `[N,N]` em vez do esperado `[N,1]`.

**Solução**: Isso é corrigido automaticamente pelo node — todos os tensores `dora_scale` são expandidos para 2D `[N,1]` no momento do carregamento. Se você ainda vir esse erro, certifique-se de estar usando a versão mais recente deste node.

### LoRA PEFT/DoRA Não Aparece no Dropdown

**Solução**:
1. Coloque a pasta PEFT (contendo `adapter_config.json` + `adapter_model.safetensors`) dentro de `ComfyUI-AceStep_SFT/Loras/`
2. Reinicie o ComfyUI — a conversão roda automaticamente na inicialização
3. Verifique no console a mensagem `[AceStep SFT] Converted PEFT/DoRA → ComfyUI: ...`
4. O arquivo convertido aparece como `*_comfyui.safetensors` no dropdown

### Geração Lenta

**Solução**: Reduza `batch_size`, reduza `steps` para ~20, ou use scheduler "karras"

## 📚 Referências Técnicas

- **AceStep 1.5**: ICML 2024 (Learning Universal Features for Efficient Audio Generation)
- **Flow Matching**: Liphardt et al. 2024 (Generative Modeling by Estimating Gradients of the Data Distribution)
- **APG/ADG**: Técnicas alinhadas com o paper oficial do AceStep
- **ComfyUI**: Arquitetura modular de node graph para geração em lote

## 📝 Licença

MIT License - Sinta-se livre para usar em projetos pessoais ou comerciais

## 🤝 Contribuições

Issues e PRs são bem-vindos! Por favor:

1. Fork o repositório
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## ⚠️ Notas Importantes

- **Duração máxima recomendada**: 240 segundos (memória GPU)
- **Batch size máximo**: Depende de sua GPU (comece com 1-2)
- **Models SFT**: Estes models são específicos para Supervised Fine-Tuning - não copie com models não-SFT
- **Direitos autorais**: Respeite direitos de uso de modelos e dados de treinamento

---

**Construído sobre o workflow do AceStep SFT e estendido com controles avançados de guidance e qualidade para ComfyUI.**

Para bugs, dúvidas ou sugestões: abra uma issue no repositório! 🎵
