# ComfyUI-AceStep SFT

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)

Uma suíte modular de nós para [ComfyUI](https://github.com/comfyanonymous/ComfyUI) que implementa o **AceStep 1.5 SFT** (Supervised Fine-Tuning), um modelo de geração musical de última geração. Parte do workflow oficial do AceStep e o estende com controle de condicionamento mais forte e opções práticas de qualidade orientadas ao ComfyUI.

> **SFT = Supervised Fine-Tuning**: Uma versão especializada do AceStep otimizada para gerar áudio de qualidade superior por meio de treinamento supervisionado.

## 📋 Visão Geral

Este pacote fornece **oito nós** em `audio/AceStep SFT`:

| Nó | Finalidade |
|----|-----------|
| **AceStep 1.5 SFT Model Loader** | Carrega o modelo de difusão, encoders CLIP e VAE |
| **AceStep 1.5 SFT Lora Loader** | Aplica um LoRA no MODEL + CLIP (encadeável) |
| **AceStep 1.5 SFT TextEncode** | Codifica caption, letras e metadados em condicionamento |
| **AceStep 1.5 SFT Generate** | Sampler de difusão + decodificação opcional com VAE |
| **AceStep 1.5 SFT Preview Audio** | Reprodução de áudio com visualizador de espectro |
| **AceStep 1.5 SFT Save Audio** | Salva áudio (FLAC/MP3/Opus) com visualizador de espectro |
| **AceStep 1.5 SFT Get Music Infos** | Análise de áudio com IA (tags, BPM, tom/escala) |
| **AceStep 1.5 SFT Turbo Tag Adapter** | Reescreve tags orientadas a Turbo em tags amigáveis ao SFT (Beta) |

### Arquitetura Modular

O workflow é dividido em nós dedicados para máxima flexibilidade:

```
Model Loader → (model, clip, vae)
       │            │        │
       │   Lora Loader (opcional, encadeável)
       │     │    │          │
       │     │  TextEncode   │
       │     │   │    │      │
       ▼     ▼   ▼    ▼      ▼
      Generate (model, positive, negative, vae)
         │         │
    Preview Audio  Save Audio
```

### Exemplo de Configuração

![Configuração do Nó AceStep SFT](example.png)

## 🎯 Recursos Principais

### ✨ Guidance Avançado

O nó suporta três modos de classifier-free guidance, cada um com características únicas:

- **APG (Adaptive Projected Guidance)** ⭐ *Recomendado*
  - Adaptação dinâmica via buffer de momentum
  - Clipping de gradiente com limiares adaptativos
  - Projeção ortogonal para eliminar ruído indesejado
  - **Padrão do AceStep SFT** - melhor equilíbrio entre qualidade e estabilidade

- **ADG (Angle-based Dynamic Guidance)**
  - Guidance baseado em ângulo entre condições
  - Opera no espaço de velocidade (flow matching)
  - Ideal para distorção agressiva de estilo

- **CFG Padrão**
  - Classifier-Free Guidance tradicional
  - Implementação simples e previsível
  - Útil como referência de comparação

### 🎵 Processamento Inteligente de Metadados

- **Auto-Duration**: Estima automaticamente a duração da música analisando a estrutura da letra
- **Codificação LLM**: Usa Qwen LLM (0.6B ou 1.7B/4B) para gerar códigos semânticos de áudio
- **Valores Automáticos**: BPM, fórmula de compasso e tom/escala automáticos (o modelo decide)
- **Suporte Multilíngue**: Mais de 23 idiomas suportados

### 🎧 Analisador Musical com IA

- **Extração de Tags de Áudio**: Usa o ACE-Step Transcriber nativo para extrair tags de letra, voz e estrutura da música
- **Detecção de BPM**: Detecção automática de andamento via librosa
- **Detecção de Tom/Escala**: Detecta tom e escala musical (ex: "G minor")
- **Saída JSON**: Saída estruturada `music_infos` com todos os resultados da análise

### 🔊 Preview e Save de Áudio com Visualizador de Forma de Onda

Ambos os nós Preview Audio e Save Audio possuem:
- **Espectro interativo de forma de onda** exibido diretamente no nó (fundo escuro com barras de amplitude)
- **Botão Play/Pause** com clique para buscar na forma de onda
- **Exibição de tempo** mostrando posição atual e duração total

O Save Audio adicionalmente suporta:
- **Múltiplos formatos**: FLAC (lossless), MP3 e Opus
- **Opções de qualidade**: V0, 64k, 96k, 128k, 192k, 320k
- **Nomes de arquivo auto-incrementais** com prefixo configurável

### 🔄 Refinamento de Áudio (img2img)

- **Refinamento Baseado em Latent**: Use `denoise < 1.0` com `latent_or_audio` conectado para refinar áudio existente
- **Aceita AUDIO ou LATENT**: Conecte qualquer saída de áudio ou latent para edição estilo img2img
- **Geração em Lote**: Gere múltiplas variações em paralelo

### 🧠 Controle Estendido de Condicionamento

- **Split Text/Lyric Guidance**: `guidance_scale_text` e `guidance_scale_lyric` independentes
- **Omega Scale**: Rebalanceamento de saída preservando a média para aproximar o comportamento do scheduler AceStep
- **Aproximação ERG**: Rebalanceamento local de energia do prompt via `erg_scale`
- **Decaimento do Intervalo de Guidance**: Decaimento suave do guidance dentro do intervalo ativo

### 🎚️ Workflow de LoRA para AceStep

- **Aplicação Direta de LoRA**: O Lora Loader recebe MODEL + CLIP, aplica o LoRA via `comfy.sd.load_lora_for_models()` e retorna MODEL + CLIP modificados
- **Encadeável**: Empilhe múltiplos Lora Loaders em sequência
- **Forças separadas**: `strength_model` e `strength_clip` independentes
- **Suporte DoRA**: Suporte completo a DoRA (Weight-Decomposed Low-Rank Adaptation) com correção automática de dimensão do `dora_scale`
- **Pasta local `Loras/`**: Coloque LoRAs diretamente na pasta `Loras/` do nó — são registrados automaticamente na inicialização
- **Conversão automática PEFT/DoRA**: LoRAs em formato PEFT (`adapter_config.json` + `adapter_model.safetensors`) colocados em `Loras/` são convertidos automaticamente para o formato ComfyUI na primeira inicialização

### 🛠️ Pós-processamento de Latent

- **Latent Shift**: Correção aditiva anti-clipping
- **Latent Rescale**: Escalonamento multiplicativo para controle dinâmico

## 📦 Instalação

### Pré-requisitos

- ComfyUI instalado e funcional
- CUDA/GPU ou equivalente (processadores modernos)
- Recomendado para melhor qualidade de saída (baseado em testes práticos): use o modelo mesclado (merged) SFT+Turbo.
- Arquivos de modelo necessários:
  - Modelo de difusão (DiT): `acestep_v1.5_sft.safetensors`
  - Encoders de Texto: `qwen_0.6b_ace15.safetensors`, `qwen_1.7b_ace15.safetensors` (ou 4B)
  - VAE: `ace_1.5_vae.safetensors`

### Download dos Modelos

Baixe os modelos necessários do HuggingFace:

1. **Modelo de Difusão (Recomendado: merged SFT+Turbo)**:
  - [AceStep 1.5 Merged SFT+Turbo Model](https://huggingface.co/Aryanne/acestep-v15-test-merges/blob/main/acestep_v1.5_merge_sft_turbo_ta_0.5.safetensors)

2. **Modelo de Difusão Alternativo (SFT oficial)**:
   - [AceStep 1.5 SFT Model](https://huggingface.co/ACE-Step/acestep-v15-sft/blob/main/model.safetensors)

3. **Encoders de Texto** (escolha qualquer versão):
   - [Coleção de Encoders de Texto](https://huggingface.co/Comfy-Org/ace_step_1.5_ComfyUI_files/tree/main/split_files/text_encoders)
     - `qwen_0.6b_ace15.safetensors` (processamento de caption)
     - `qwen_1.7b_ace15.safetensors` ou `qwen_4b_ace15.safetensors` (geração de códigos de áudio)

4. **VAE** (Codec de áudio):
   - [AceStep 1.5 VAE](https://huggingface.co/Comfy-Org/ace_step_1.5_ComfyUI_files/blob/main/split_files/vae/ace_1.5_vae.safetensors)

### Passos de Instalação

1. Clone o repositório na pasta de custom nodes:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/jeankassio/ComfyUI-AceStep_SFT.git
```

2. Coloque os arquivos de modelo nos diretórios apropriados:
```
ComfyUI/models/diffusion_models/     # Modelo AceStep 1.5 SFT
ComfyUI/models/text_encoders/        # Encoders Qwen
ComfyUI/models/vae/                  # VAE
ComfyUI/models/loras/                # LoRAs opcionais do AceStep 1.5
```

3. **(Opcional) Coloque LoRAs na pasta local:**
```
ComfyUI/custom_nodes/ComfyUI-AceStep_SFT/Loras/   # Pasta local de LoRA
```
   Você pode colocar LoRAs aqui em **qualquer** um destes formatos:
   - **Formato ComfyUI**: Arquivo `.safetensors` único (pronto para uso)
   - **Formato PEFT/DoRA**: Uma pasta contendo `adapter_config.json` + `adapter_model.safetensors` (convertido automaticamente na inicialização)
   - **Artefato zip aninhado**: Se seu zip extraiu uma pasta-dentro-de-pasta, o nó detecta e corrige automaticamente

4. Reinicie o ComfyUI - os nós aparecerão em `audio/AceStep SFT`

## 🧩 Nós Disponíveis

### AceStep 1.5 SFT Model Loader

Carrega o modelo de difusão AceStep 1.5, os dois encoders CLIP de texto e o VAE de áudio.

Entradas:
- `diffusion_model`: Modelo de difusão AceStep 1.5 (.safetensors)
- `text_encoder_1`: Encoder Qwen3-0.6B (processamento de caption)
- `text_encoder_2`: Qwen3 LLM (1.7B ou 4B, geração de códigos de áudio)
- `vae_name`: VAE de áudio AceStep 1.5

Saídas:
- `model`: MODEL — conecte ao Lora Loader ou Generate
- `clip`: CLIP — conecte ao Lora Loader ou TextEncode
- `vae`: VAE — conecte ao Generate

### AceStep 1.5 SFT Lora Loader

Aplica um LoRA diretamente no MODEL e CLIP. Múltiplos Lora Loaders podem ser encadeados.

Entradas:
- `model`: MODEL do Model Loader ou Lora Loader anterior
- `clip`: CLIP do Model Loader ou Lora Loader anterior
- `lora_name`: Arquivo LoRA de `ComfyUI/models/loras` ou da pasta local `Loras/`
- `strength_model`: intensidade aplicada ao modelo de difusão
- `strength_clip`: intensidade aplicada à pilha de encoders de texto

Saídas:
- `model`: MODEL — conecte ao próximo Lora Loader ou Generate
- `clip`: CLIP — conecte ao próximo Lora Loader ou TextEncode

#### Formatos de LoRA Suportados

| Formato | O que colocar em `Loras/` | Ação |
|---------|--------------------------|------|
| ComfyUI `.safetensors` | Arquivo único | Usado diretamente |
| Diretório PEFT/DoRA | Pasta com `adapter_config.json` + `adapter_model.safetensors` | Convertido automaticamente para `*_comfyui.safetensors` na inicialização |
| Artefato zip aninhado | Pasta contendo um `.safetensors` dentro | Extraído automaticamente para a raiz na inicialização |

### AceStep 1.5 SFT TextEncode

Codifica caption, letras e metadados em condicionamento positivo e negativo para o nó Generate.

Entradas:
- `clip`: CLIP do Model Loader ou Lora Loader
- `caption`: Descrição textual da música (gênero, humor, instrumentos)
- `lyrics`: Letra da música ou `[Instrumental]`
- `instrumental`: Forçar modo instrumental
- `seed`, `duration`, `bpm`, `timesignature`, `language`, `keyscale`
- Opcional: `generate_audio_codes`, `lm_cfg_scale`, `lm_temperature`, `lm_top_p`, `lm_top_k`, `lm_min_p`, `lm_negative_prompt`
- Sobreposições de estilo opcionais: `style_tags`, `style_bpm`, `style_keyscale` (do Music Analyzer)

Saídas:
- `positive`: CONDITIONING — conecte ao Generate
- `negative`: CONDITIONING — conecte ao Generate

### AceStep 1.5 SFT Generate

Sampler de difusão + decodificador VAE opcional. Requer MODEL e entradas de condicionamento.

Entradas:
- `model`: MODEL do Model Loader ou Lora Loader
- `positive`: CONDITIONING do TextEncode
- `negative`: CONDITIONING do TextEncode
- Amostragem: `seed`, `steps`, `cfg`, `sampler_name`, `scheduler`, `denoise`, `duration`, `infer_method`, `guidance_mode`
- Opcional: `vae` (para saída de áudio), `latent_or_audio` (para img2img), `batch_size`
- Pós-processamento opcional: `latent_shift`, `latent_rescale`, `fade_in_duration`, `fade_out_duration`, `voice_boost`, `use_tiled_vae`
- Guidance opcional: `apg_eta`, `apg_momentum`, `apg_norm_threshold`, `guidance_interval`, `guidance_interval_decay`, `min_guidance_scale`, `guidance_scale_text`, `guidance_scale_lyric`, `omega_scale`, `erg_scale`, `cfg_interval_start`, `cfg_interval_end`, `shift`

Saídas:
- `model`: MODEL (passthrough para encadeamento)
- `vae`: VAE (passthrough para encadeamento)
- `positive`: CONDITIONING (passthrough)
- `negative`: CONDITIONING (passthrough)
- `latent`: LATENT (saída bruta da difusão)
- `audio`: AUDIO (áudio decodificado, apenas quando VAE está conectado)

### AceStep 1.5 SFT Preview Audio

Pré-visualiza áudio com um visualizador interativo de espectro de forma de onda diretamente no nó.

Entradas:
- `audio`: AUDIO para pré-visualizar

Recursos:
- Visualização interativa de forma de onda com botão play/pause
- Clique para buscar (seek) na forma de onda
- Exibição de tempo atual / duração total

### AceStep 1.5 SFT Save Audio

Salva áudio em disco com um visualizador interativo de espectro de forma de onda.

Entradas:
- `audio`: AUDIO para salvar
- `filename_prefix`: Prefixo do nome do arquivo (suporta caminhos de subpasta, ex: `audio/AceStep`)
- `format`: FLAC, MP3 ou Opus
- `quality` (opcional): V0, 64k, 96k, 128k, 192k, 320k (para MP3/Opus)

Recursos:
- Nomes de arquivo auto-incrementais (ex: `AceStep_00001_.flac`, `AceStep_00002_.flac`)
- Visualizador de forma de onda com play/pause e seek
- Incorporação de metadados (prompt, workflow)

### AceStep 1.5 SFT Get Music Infos

Nó de análise de áudio com IA que extrai tags descritivas, BPM e tom/escala a partir de uma entrada de áudio.

Entradas:
- `audio`: Entrada de áudio para análise
- `get_tags` / `get_bpm` / `get_keyscale`: Ativar/desativar cada análise
- `max_new_tokens`: Máximo de tokens para saída de transcrição
- `audio_duration`: Máximo de segundos de áudio para análise
- `temperature`, `top_p`, `top_k`, `repetition_penalty`, `seed`: Parâmetros de geração
- `unload_model`: Liberar VRAM após análise
- `use_flash_attn`: Ativar Flash Attention 2 (se compatível)

Saídas:
- `tags`: Tags descritivas separadas por vírgula (STRING)
- `bpm`: BPM detectado (INT)
- `keyscale`: Tom e escala ex: "G minor" (STRING)
- `music_infos`: JSON com todos os resultados (STRING)

### AceStep 1.5 SFT Turbo Tag Adapter

Reescreve tags de música orientadas a Turbo em tags de prompt mais curtas e amigáveis ao SFT.

Entradas:
- `turbo_tags`: Tags ou caption no estilo Turbo
- `adaptation_strength`: conservative / balanced / aggressive
- `keep_unknown_tags`: Manter tags que não foram mapeadas explicitamente
- `add_sft_bias_tags`: Adicionar tags âncora extras orientadas ao SFT

Saídas:
- `sft_tags`: Tags adaptadas separadas por vírgula (STRING)
- `notes`: Notas de conversão (STRING)
- `suggested_cfg`: Valor sugerido de CFG (FLOAT)
- `suggested_steps`: Valor sugerido de steps (INT)

## 🎛️ Parâmetros dos Nós

### Generate - Parâmetros Obrigatórios

| Parâmetro | Faixa | Descrição |
|-----------|-------|-----------|
| **model** | MODEL | Modelo de difusão AceStep 1.5 do Model Loader ou Lora Loader |
| **positive** | CONDITIONING | Condicionamento positivo do TextEncode |
| **negative** | CONDITIONING | Condicionamento negativo do TextEncode |
| **seed** | 0 - 2^64 | Seed para reprodutibilidade |
| **steps** | 1 - 200 | Passos de inferência da difusão (padrão: 50) |
| **cfg** | 1.0 - 20.0 | Escala de classifier-free guidance (padrão: 7.0) |
| **sampler_name** | - | Sampler (euler, dpmpp, etc.) |
| **scheduler** | - | Scheduler (normal, karras, etc.) |
| **denoise** | 0.0 - 1.0 | Intensidade de denoising (1.0 = novo, < 1.0 = edição) |
| **duration** | 0.0 - 600.0 | Duração em segundos (0 = automático) |
| **infer_method** | ode/sde | ODE = determinístico, SDE = estocástico |
| **guidance_mode** | apg/adg/standard_cfg | Tipo de guidance (padrão: apg) |

### Generate - Parâmetros Opcionais

#### Geração em Lote
- **batch_size** (1-16): Número de áudios para gerar em paralelo

#### Entrada de Áudio
- **vae**: VAE do Model Loader (necessário para saída de áudio)
- **latent_or_audio**: Entrada base para refinamento (img2img). Aceita AUDIO ou LATENT

#### Pós-processamento de Latent
- **latent_shift** (-0.2-0.2, padrão: 0.0): Deslocamento aditivo (anti-clipping)
- **latent_rescale** (0.5-1.5, padrão: 1.0): Escalonamento multiplicativo
- **fade_in_duration / fade_out_duration** (0.0-10.0, padrão: 0.0): Fades lineares opcionais
- **use_tiled_vae** (padrão: True): Usa VAE em blocos para áudio longo / pouca VRAM
- **voice_boost** (-12.0-12.0, padrão: 0.0): Ganho de saída em dB

#### Configuração APG
- **apg_eta** (-10.0-10.0, padrão: 0.0): Retenção do componente paralelo
- **apg_momentum** (-1.0-1.0, padrão: -0.75): Coeficiente do buffer de momentum
- **apg_norm_threshold** (0.0-15.0, padrão: 2.5): Limiar de norma para clipping de gradiente

#### Controles de Guidance Estendidos
- **guidance_interval** (-1.0-1.0, padrão: 0.5): Largura do intervalo de guidance centrado
- **guidance_interval_decay** (0.0-1.0, padrão: 0.0): Decaimento linear dentro do intervalo
- **min_guidance_scale** (0.0-30.0, padrão: 3.0): Limite inferior com decaimento
- **guidance_scale_text** (-1.0-30.0, padrão: -1.0): Guidance apenas de texto (split)
- **guidance_scale_lyric** (-1.0-30.0, padrão: -1.0): Guidance apenas de letra (split)
- **omega_scale** (-8.0-8.0, padrão: 0.0): Rebalanceamento preservando a média
- **erg_scale** (-0.9-2.0, padrão: 0.0): Rebalanceamento de energia do prompt
- **cfg_interval_start / cfg_interval_end** (0.0-1.0): Faixa de fração do schedule
- **shift** (0.0-5.0, padrão: 3.0): Deslocamento do schedule de timestep

### TextEncode - Parâmetros

| Parâmetro | Faixa | Descrição |
|-----------|-------|-----------|
| **clip** | CLIP | CLIP do Model Loader ou Lora Loader |
| **caption** | texto | Descrição da música (gênero, humor, instrumentos) |
| **lyrics** | texto | Letra da música ou `[Instrumental]` |
| **instrumental** | booleano | Forçar modo instrumental |
| **seed** | 0 - 2^64 | Seed |
| **duration** | 0.0 - 600.0 | Duração em segundos (0 = automático a partir das letras) |
| **bpm** | 0 - 300 | Batidas por minuto (0 = automático) |
| **timesignature** | auto/2/3/4/6 | Numerador da fórmula de compasso |
| **language** | - | Idioma da letra (en, ja, zh, es, pt, etc.) |
| **keyscale** | auto/... | Tom e escala (ex: "C major") |

#### TextEncode - Configuração LLM Opcional
- **generate_audio_codes** (padrão: True): Ativar geração de códigos de áudio por LLM
- **lm_cfg_scale** (0.0-100.0, padrão: 2.0): Escala CFG do LLM
- **lm_temperature** (0.0-2.0, padrão: 0.85): Temperatura de amostragem do LLM
- **lm_top_p** (0.0-2000.0, padrão: 0.9): Amostragem nucleus
- **lm_top_k** (0-100, padrão: 0): Amostragem top-k
- **lm_min_p** (0.0-1.0, padrão: 0.0): Probabilidade mínima
- **lm_negative_prompt**: Prompt negativo para CFG do LLM

#### TextEncode - Sobreposições de Estilo (do Music Analyzer)
- **style_tags**: Adicionados ao caption quando conectados
- **style_bpm**: Sobrescreve bpm quando > 0
- **style_keyscale**: Sobrescreve keyscale quando não vazio

## 🎨 Exemplos de Workflow

### Exemplo 1: Geração Básica

```
Model Loader:
  diffusion_model: "acestep_v1.5_sft.safetensors"
  text_encoder_1: "qwen_0.6b_ace15.safetensors"
  text_encoder_2: "qwen_1.7b_ace15.safetensors"
  vae_name: "ace_1.5_vae.safetensors"
  → model, clip, vae

TextEncode:
  clip: (do Model Loader)
  caption: "upbeat electronic dance music with synthesizers"
  lyrics: [Instrumental]
  instrumental: True
  duration: 60.0
  → positive, negative

Generate:
  model: (do Model Loader)
  positive: (do TextEncode)
  negative: (do TextEncode)
  vae: (do Model Loader)
  cfg: 7.0, steps: 50, guidance_mode: "apg"
  → audio

Preview Audio:
  audio: (do Generate)
```

### Exemplo 2: Com LoRA

```
Model Loader → model, clip, vae
  ↓ model, clip
Lora Loader:
  lora_name: "ace-step15-style1.safetensors"
  strength_model: 0.7
  strength_clip: 0.0
  → model, clip
  ↓ model, clip
Lora Loader:
  lora_name: "Ace-Step1.5-TechnoRain.safetensors"
  strength_model: 0.35
  strength_clip: 0.0
  → model, clip

TextEncode (clip do último Lora Loader) → positive, negative
Generate (model do último Lora Loader, vae do Model Loader) → audio
Save Audio (format: mp3, quality: 320k)
```

### Exemplo 3: Refinamento de Áudio (img2img)

```
Generate:
  latent_or_audio: (áudio existente)
  denoise: 0.7 (preserva 30% da fonte)
  duration: 0 (usa a duração da entrada)
  → Refina o áudio preservando as características originais
```

### Exemplo 4: Análise Musical → Geração

```
Music Analyzer:
  audio: (arquivo de áudio de entrada)
  → tags, bpm, keyscale

TextEncode:
  style_tags: (do Music Analyzer)
  style_bpm: (do Music Analyzer)
  style_keyscale: (do Music Analyzer)
  → positive, negative

Generate → Save Audio (format: flac)
```

## 🐛 Solução de Problemas

### Distorção/Clipping no Áudio

**Solução**: Use `latent_shift` negativo (ex: -0.1) para reduzir a amplitude antes da decodificação VAE

### Resultados com Alta Variância

**Solução**: Aumente `apg_norm_threshold` (ex: 3.0-4.0) para mais clipping de gradiente

### Qualidade Inferior ao Esperado

**Solução**: 
1. Use `guidance_mode: "apg"` (recomendado)
2. Comece com `steps: 50`, `cfg: 7.0`, `sampler_name: "euler"`, `scheduler: "normal"`, `infer_method: "ode"`

### LoRA com Som Deformado ou Excessivo

**Solução**:
1. Diminua `strength_model` primeiro, ex: `0.2` a `0.6`
2. Defina `strength_clip` como `0.0` a menos que o LoRA especificamente mire nos encoders de texto
3. Compare `guidance_mode: "standard_cfg"` vs `"apg"` para aquele LoRA
4. Evite empilhar múltiplos LoRAs fortes em intensidade máxima

### Erro de Incompatibilidade de Dimensão do LoRA (`The size of tensor a must match...`)

**Causa**: LoRAs DoRA armazenam `dora_scale` como tensor 1D `[N]`. O `weight_decompose` do ComfyUI espera `[N,1]`.

**Solução**: Isso é corrigido automaticamente pelo Lora Loader — todos os tensores `dora_scale` recebem unsqueeze para 2D `[N,1]` no momento do carregamento.

### LoRA PEFT/DoRA Não Aparece no Dropdown

**Solução**:
1. Coloque a pasta PEFT (contendo `adapter_config.json` + `adapter_model.safetensors`) dentro de `ComfyUI-AceStep_SFT/Loras/`
2. Reinicie o ComfyUI — a conversão roda automaticamente na inicialização
3. Verifique o console pela mensagem `[AceStep SFT] Converted PEFT/DoRA → ComfyUI: ...`
4. O arquivo convertido aparece como `*_comfyui.safetensors` no dropdown

### Geração Lenta

**Solução**: Reduza `batch_size`, diminua `steps` para ~20, ou use scheduler "karras"

## 📊 Comparação de Modos de Guidance

| Aspecto | APG | ADG | CFG Padrão |
|---------|-----|-----|------------|
| **Qualidade** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Estabilidade** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ |
| **Dinâmica** | Natural | Agressiva | Previsível |
| **Computação** | Normal | Normal | Mínima |
| **Recomendado** | ✅ Sim | Para estilos extremos | Linha de base |

## 🎚️ Dicas de Qualidade

- Use `guidance_mode=apg` com `steps=50` a `64` para melhor qualidade
- Para refinamento img2img, comece com `denoise=0.5` a `0.7` para preservar o caráter original
- Chiado leve de voz geralmente é um artefato de geração; APG e contagens de steps um pouco maiores geralmente ajudam mais do que `cfg` puro
- Simplifique tags excessivamente densas ou contraditórias para resultados mais limpos

## 📚 Referências Técnicas

- **AceStep 1.5**: ICML 2024 (Learning Universal Features for Efficient Audio Generation)
- **Flow Matching**: Liphardt et al. 2024 (Generative Modeling by Estimating Gradients of the Data Distribution)
- **APG/ADG**: Técnicas alinhadas com o paper oficial do AceStep
- **ComfyUI**: Arquitetura de grafo de nós modular para geração em lote

## 📝 Licença

Licença MIT - Livre para uso em projetos pessoais ou comerciais

## 🤝 Contribuindo

Issues e PRs são bem-vindos! Por favor:

1. Faça um fork do repositório
2. Crie uma branch para a feature (`git checkout -b feature/RecursoIncrivel`)
3. Commit suas mudanças (`git commit -m 'Adicionar RecursoIncrivel'`)
4. Push para a branch (`git push origin feature/RecursoIncrivel`)
5. Abra um Pull Request

## ⚠️ Notas Importantes

- **Duração máxima recomendada**: 240 segundos (memória da GPU)
- **Batch size máximo**: Depende da sua GPU (comece com 1-2)
- **Modelos SFT**: Estes modelos são específicos para Supervised Fine-Tuning - não testados com modelos não-SFT
- **Direitos e atribuição**: Respeite os direitos de uso dos modelos e datasets

---

**Construído sobre o workflow AceStep SFT e estendido com nós modulares, guidance avançado, visualização de forma de onda e controles de qualidade para ComfyUI.**

Para bugs, dúvidas ou sugestões: abra uma issue no repositório! 🎵
