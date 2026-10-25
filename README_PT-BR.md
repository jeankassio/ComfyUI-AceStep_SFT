# ComfyUI-AceStep SFT

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Uma suíte modular de nós para [ComfyUI](https://github.com/comfyanonymous/ComfyUI) que implementa o **AceStep 1.5 SFT** (Supervised Fine-Tuning), um modelo de geração musical de última geração. Parte do workflow oficial do AceStep e o estende com controle de condicionamento mais forte e opções práticas de qualidade orientadas ao ComfyUI.

> **SFT = Supervised Fine-Tuning**: Uma versão especializada do AceStep otimizada para gerar áudio de qualidade superior por meio de treinamento supervisionado.

## 📋 Visão Geral

### Carregamento experimental compartilhado SFT → Base → Turbo

O **AceStep 1.5 SFT Cascade Model Loader** carrega os três checkpoints originais em uma arquitetura nativa compartilhada, com um único par CLIP/VAE. Cada saída MODEL conserva a identidade do checkpoint e pode receber seu próprio LoRA. A troca seleciona os tensores exatos do modelo de destino, preservando os dtypes armazenados. A precisão de inferência e o gerenciamento de memória continuam sob controle do ComfyUI.

Na implementação nativa de VRAM dinâmica compatível, o ModelPatcher do node substitui as origens CPU dos pesos diferentes e invalida somente os módulos GPU cujos pesos ou adapters efetivos mudaram. Os módulos iguais que continuam residentes podem ser reutilizados. Os buffers nativos de memória host são limpos antes da troca para evitar dados antigos. Retornar ao SFT depois do Turbo também funciona quando o ComfyUI reutiliza partes do workflow em cache. Nenhum arquivo do core ou classe global do ComfyUI é alterado.

1. Substitua os três Model Loader por um **Cascade Model Loader** e selecione os checkpoints originais SFT, Base e Turbo, com a mesma arquitetura, formas e dtypes dos tensores.
2. Conecte `sft_model`, `base_model` e `turbo_model` aos respectivos LoRA/Generate. Aplique o LoRA de cada etapa **depois** desse carregador.
3. Use o `clip` compartilhado no TextEncode/LoRA e o `vae` nos Generate.
4. Encadeie as saídas **LATENT**: SFT → Base → Turbo, mantendo os parâmetros de geração que você já utiliza.
5. Nos Generate SFT e Base, defina `decode_audio=false`; no Turbo final, `true`. O AUDIO intermediário será `None`: previews e salvamento devem receber a etapa final. O VAE continua disponível para codificar áudio-fonte e referência.
6. Mantenha `unload_models_after_generate=false` e a mesma GPU nas três etapas. Reinicie o ComfyUI e atualize o navegador para carregar os controles novos.

O primeiro carregamento compara os tensores e pode demorar mais; depois, o ComfyUI mantém a saída do carregador em cache. Em CPU, carregamento estático, patchers personalizados não suportados ou checkpoints incompatíveis/quantizados, o node usa carregadores nativos independentes e informa no console que a retenção seletiva está desativada. A pressão de memória também pode remover blocos GPU iguais. A otimização reduz inicializações, transferências e decodificações intermediárias, **não** o número de avaliações de denoising. Os SFT/Base XL locais compartilham cerca de 16% dos bytes de parâmetros; é necessário medir o ganho no workflow real.

Validação local em RTX 3090: sete forwards nativos curtos com os checkpoints XL reais deram resultados idênticos, bit a bit, na troca seletiva e na invalidação completa. A transferência caiu 14,5–14,6%; as trocas medidas uma única vez foram apenas 0,5–3,5% mais rápidas. O teste exclui CLIP/VAE e a geração de uma música inteira; não comprova um ganho confiável no tempo total. Treze forwards nativos pequenos também verificaram LoRAs independentes, retorno com cache e recuperação após offload.

### Amostragem mais rápida mantendo o sampler

Generate e Generate Advanced agora oferecem `precompute_conditioning=true`. Na geração comum de uma música, o encoder nativo de letra/timbre e o detokenizador semântico executam uma vez por ramo de condicionamento, em vez de repetir a cada avaliação de denoising. Os tensores passam pelas entradas normais de condicionamento do ComfyUI e são liberados ao terminar a amostragem. Decoder, aplicação de LoRA, precisão de inferência, ruído, guidance e quantidade de passos continuam iguais. Referência de voz, tarefas nativas de edição, batches, guidance separado, ERG e wrappers personalizados mantêm o caminho normal. Desative a opção para comparar.

O Cascade loader também oferece `cache_runtime_weights`: uma área de RAM limitada a 4 GiB no conjunto dos três modelos, para preparar pesos grandes elegíveis na precisão de inferência que o ComfyUI já escolheu. Os tensores originais dos checkpoints permanecem intactos, incluindo parâmetros FP32. As verificações de memória reservam espaço para o carregamento nativo e os encoders; RAM insuficiente reduz ou desativa essa preparação. Transformações personalizadas e adapters que redimensionam pesos usam as origens originais. Esse cache não quantiza modelos, move arquivos nem reserva uma segunda GPU.

A validação local na RTX 3090 usou o SFT XL real e um LoRA com 352 patches: 50 avaliações nativas, 1.500 frames latentes, 64 tokens de descrição e 256 de letra. Os resultados foram idênticos bit a bit; encoder/detokenizador passaram de 50 para 2 execuções. O teste mede a amostragem com pesos carregados, excluindo carregamento do checkpoint, geração textual e decodificação VAE. O ganho na música completa depende desses custos e da RAM disponível. Continue usando a inicialização que preserva sua qualidade; `--fast` não é necessário.

### Latente de silêncio oficial

O node inclui o `silence_latent.pt` original do [ACE-Step XL Base](https://huggingface.co/ACE-Step/acestep-v15-xl-base), sem conversão. Generate e Generate Advanced usam esse arquivo no condicionamento silencioso padrão e no preenchimento de referências; Cover/Remix, Repaint e o padding semântico usam o mesmo tensor. Não é necessário adicionar entradas nem alterar o workflow.

O arquivo contém 15.000 frames (600 segundos a 25 Hz). Pedidos menores usam o início do tensor; pedidos maiores repetem a sequência completa e recortam o comprimento necessário, como no carregador oficial. Isso substitui a aproximação do ComfyUI de quatro frames seguidos por um valor constante, usando o ModelPatcher apenas durante a execução, sem editar o núcleo do ComfyUI. A mudança alinha os dados de silêncio, não todo o pipeline de condicionamento, e não garante uma melhoria audível. Reinicie o ComfyUI após terminarem as gerações em andamento.

### Carregador de referência opcional

O **Load Reference Audio**, na categoria `audio/AceStep SFT`, permite selecionar ou enviar uma gravação e oferece o switch `enabled`. Durante a execução, com `true`, carrega o arquivo pelo decoder de áudio do ComfyUI; com `false`, retorna `None`, sem ler ou validar o arquivo selecionado. Ele começa desligado e funciona sem nenhum arquivo enviado. Conecte a saída ao `reference_audio` do Generate, Generate Advanced, Cover/Remix ou Repaint. Nenhuma onda vazia é enviada ao encoder. O switch controla a saída para geração; selecionar um arquivo continua oferecendo uma prévia local no navegador.

Em Cover/Remix, conecte o carregador ao node que cria a tarefa. A opção separada `preserve_singer_identity=true` continua usando a música de origem como referência quando não há referência explícita; desligue essa opção também se quiser remover esse condicionamento. Se a tarefa já contém uma referência, desligar apenas a entrada direta do Generate não limpa a referência da tarefa. Reinicie o ComfyUI e atualize a página para encontrar o novo node.

### Adaptação experimental da voz enviada

O **AceStep 1.5 SFT Adapt Voice (Experimental)** ajusta um vetor temporário de timbre à gravação antes de gerar voz e instrumentos juntos. Ele modifica somente 2.048 valores de condicionamento, com deslocamento limitado em relação ao timbre original. O ajuste não treina LoRA nem altera pesos do checkpoint; também não separa instrumentos ou converte o vocal depois da geração. O perfil conserva a referência codificada da gravação; nenhum arquivo de modelo treinado é salvo.

1. Conecte `model`, `clip` e `vae` do carregador SFT, além do **Load Reference Audio** com `enabled=true`, ao **Adapt Voice**.
2. Comece com `max_seconds=60`, `learning_rate=0.02` e `seed` fixa. Em `language`, informe o idioma da gravação, como `pt`, ou use `auto`.
3. Conecte a saída `voice_profile` ao Generate ou Generate Advanced. Ela substitui o `reference_audio` desse Generate. Mantenha a descrição e a letra da música desejada no TextEncode normal.
4. Para SFT → Base → Turbo, passe LATENT entre as etapas e conecte o mesmo perfil aos três Generate. Transferir um vetor ajustado no SFT para outro checkpoint continua experimental. As tarefas nativas de edição ainda não aceitam `voice_profile`.

Deixe `transcript` vazio para transcrever os recortes em CPU usando o **Whisper large-v3-turbo já disponível no cache local do Hugging Face**. Esse caminho não baixa modelos; informe uma transcrição se o cache estiver ausente. Texto simples corresponde à gravação inteira. Para recortes alinhados, forneça uma lista JSON de registros `{"start":0,"end":5,"text":"Palavras exatas faladas neste intervalo"}`, com tempos em segundos. Não informe aqui a letra da nova música. Com vários recortes, o último fica reservado para validação.

O orçamento de 0–60 segundos cobre o ajuste, **excluindo transcrição, codificação e carregamento dos modelos**. Há verificações cooperativas entre camadas e operações de backward; uma operação GPU já iniciada pode causar um pequeno excedente, registrado junto ao tempo real. `max_seconds=0` desativa a adaptação. O vetor só é aceito se melhorar a reconstrução na validação; caso contrário, a geração conserva o condicionamento normal pela referência. A saída JSON `adaptation_report` informa aceite, passos, perdas e tempos.

Um teste local no SFT concluiu seis atualizações em 49,13 segundos, dentro do orçamento de 60 segundos, reduzindo a perda de validação de 0,99217 para 0,97656. Isso verifica a otimização da reconstrução, **não uma voz cantada idêntica nem maior qualidade musical**. Compare os áudios ouvindo, com seeds e parâmetros de geração iguais. Reinicie o ComfyUI e atualize o navegador para carregar o novo node e a entrada.

### Testes de qualidade do SFT puro

Foram acrescentados três ajustes na geração, sem novos tratamentos da onda:

- Generate `timbre_cls_token=true` corrige a ausência do token aprendido de resumo no encoder de timbre deste core. Ele passa a ser inserido antes do áudio e das posições RoPE, como no [XL oficial](https://github.com/ace-step/ACE-Step-1.5/blob/main/acestep/models/xl_sft/modeling_acestep_v15_xl_base.py). A correção usa ModelPatcher somente durante esta execução; `false` permite comparar o comportamento anterior.
- Generate `guidance_mode=apg_denoised` aplica momentum, limite de norma e projeção à previsão limpa `x0`, conforme a análise de prediction type do [artigo APG](https://arxiv.org/html/2410.02416v2). `apg` mantém a implementação em velocidade. Ambos conservam a projeção temporal do ACE-Step; o novo modo não reproduz integralmente a agregação de dimensões do artigo. `apg_norm_threshold` passa a medir unidades de latent limpo nesse modo. Funciona também no preditor e corretor de `jkass_quality`.
- TextEncode `lm_codes_strength=0.75` utiliza os códigos do LM nos primeiros 75% dos passos principais e troca para uma codificação text2music sem códigos no final. `1` preserva o uso integral; `0` evita executar o LM. A troca também acompanha o sigma das avaliações corretoras. Valores intermediários exigem uma codificação de texto adicional, sem uma segunda geração do LM. Cover/Repaint conservam seus próprios controles.

Reinicie o ComfyUI e atualize a página para carregar os controles. Para um teste comparável, mantenha caption, letra, ambas as seeds, duração, sampler, scheduler, steps, CFG e shift iguais:

| Teste | `timbre_cls_token` | `guidance_mode` | `lm_codes_strength` |
|-------|--------------------|-----------------|---------------------|
| Comportamento anterior | false | apg | 1 |
| Apenas correção de timbre | true | apg | 1 |
| APG na previsão limpa | true | apg_denoised | 1 |
| Liberação final dos códigos | true | apg_denoised | 0.75 |

Comece com os demais valores atuais, incluindo `apg_norm_threshold=2.5`. Omitir os novos parâmetros conserva APG/códigos anteriores e ativa a correção de timbre. A omissão do token foi verificada no código; melhora musical dos dois experimentos depende de comparação auditiva. Os testes automatizados verificam matemática, condicionamento, compatibilidade e restauração dos patches, não qualidade perceptual.

Generate também oferece `guidance_rescale`, desligado por padrão (`0`). Após CFG/APG/ADG, ele aproxima o desvio padrão da velocidade guiada ao da previsão condicional, por amostra do lote, seguindo a fórmula de [rescale do EzAudio](https://arxiv.org/html/2409.10819). A adaptação atua durante a amostragem, inclusive no preditor e corretor do `jkass_quality`, sem avaliações extras do modelo. O APG continua projetando em `x0` quando `apg_denoised` está selecionado; o rescale posterior atua na velocidade, não diretamente no latent limpo. Estatísticas são calculadas em FP32; previsões constantes usam escala neutra.

Para comparar, mantenha `apg_denoised`, `timbre_cls_token=true` e `lm_codes_strength=1`, com as mesmas seeds e demais parâmetros. Teste somente `guidance_rescale`: `0` → `0.25` → `0.50`. `1` aplica a correção completa. O ganho musical no ACE-Step depende do teste auditivo. Reinicie o ComfyUI e atualize a página para carregar o widget, acrescentado ao final do Generate para preservar a ordem dos controles existentes.

Generate agora calcula o schedule `ace_step` em FP32 por padrão (`schedule_precision=float32`), sem mudar a precisão dos pesos ou do ruído. Calcular primeiro em BF16 e converter depois já perdeu precisão: nos testes CPU com `shift=3`, 50/100/200 passos apresentavam 3/13/35 intervalos de sigma repetido, nos quais o integrador não avançava. `schedule_precision=model` reproduz a precisão anterior para comparação. Os outros schedulers e o schedule nativo de Cover com scheduler diferente de `ace_step` conservam sua precisão. O controle foi acrescentado ao final dos widgets; reinicie o ComfyUI após terminar as gerações em andamento.

O modo experimental `guidance_mode=apg_denoised_joint` projeta a previsão limpa conjuntamente sobre canais e tempo, separadamente para cada áudio do lote. Essa geometria segue o [algoritmo APG](https://arxiv.org/html/2410.02416v2) e também aparece na [implementação de áudio do Stable Audio 3](https://github.com/Stability-AI/stable-audio-3/blob/main/optimized/mlx/scripts/sa3_mlx.py). O `apg_denoised` anterior continua projetando por canal. A projeção conjunta remove uma direção paralela em vez de uma por canal; preservar mais variação pode ajudar, mas não há validação perceptual para ACE-Step. Para evitar um clipping global oito vezes mais restritivo, o raio conjunto é `apg_norm_threshold * sqrt(canais)`: 2.5 corresponde a 20 para 64 canais. Essa escala é uma adaptação ao orçamento anterior, não uma recomendação do artigo. Momentum e rescale permanecem disponíveis, sem avaliações extras do modelo.

Compare primeiro `schedule_precision=model` e `float32`, mantendo `apg_denoised`. Depois, com `float32` fixo, compare `apg_denoised` e `apg_denoised_joint`. Mantenha códigos completos do LM, correção de timbre ligada, seeds e demais valores iguais; comece com `guidance_rescale=0` para isolar a projeção. Avalie tanto instrumentos e ritmo quanto palavras cantadas.

### Novos testes de clareza com SFT puro

Generate oferece `apg_norm_rms`, acrescentado ao final dos widgets. `0` conserva o limite L2 de `apg_norm_threshold`. Um valor positivo substitui esse limite por um teto RMS da atualização APG, antes da projeção e do CFG, sem depender da duração. Não é o RMS ou volume do áudio. Nas músicas de 180 segundos com 64 canais e 4.500 frames, `apg_denoised_joint` com norma 2.5 permite RMS de até 0.0373; o mesmo limite em 30 segundos permite 0.0913. Essa dependência da duração é matemática; seu efeito auditivo ainda precisa ser confirmado. O [APG](https://arxiv.org/html/2410.02416v2) alerta para clipping excessivo, mas a normalização RMS para áudio é nossa adaptação.

Primeiro mantenha a configuração que ficou boa, incluindo `apg_denoised_joint`, e compare somente `apg_norm_rms=0`, `0.05` e `0.075`. Aos 180 segundos, os dois valores positivos dão raios 26.83 e 40.25, contra 20 do limite anterior. Esse controle também se aplica aos outros modos APG na previsão que cada um usa; ADG/CFG não o utilizam. Reinicie o ComfyUI depois de terminar as gerações para carregar o novo widget.


### Clareza do áudio: teste opcional de VAE

O [ScragVAE](https://huggingface.co/scragnog/Ace-Step-1.5-ScragVAE) é um decoder retreinado para recuperar detalhes de agudos e transientes. Os ganhos publicados são do autor e ainda exigem comparação auditiva no seu áudio. Para este node, use a [conversão para ComfyUI](https://huggingface.co/megagrump/Ace-Step-1.5-ScragVAE-ComfyUI), `ace_1.5_scrag_vae.safetensors`, na pasta `ComfyUI/models/vae`; selecione-a no Model Loader em `vae_name`. O checkpoint original em formato Diffusers não pode ser usado apenas renomeando o arquivo.

Compare primeiro mudando somente o VAE, com a mesma seed e demais parâmetros. Para isolar completamente a decodificação, conecte a saída `latent` do Generate a dois nós `VAE Decode Audio`, cada um com seu VAE carregado por `Load VAE`, e compare os previews. Use `latent_shift=0`, `latent_rescale=1` e nenhum fade ou voice boost nesse teste. A troca modifica a reconstrução dentro da geração, sem equalização da onda. O sampler, incluindo `jkass_quality`, continua funcionando da mesma forma; não é preciso alterar o guidance.

Nos dois checkpoints locais comparados, os pesos do encoder não são idênticos. Para text2music sem referência, ele não participa da geração; em Cover/Remix com áudio de referência, a troca de VAE também pode alterar a codificação. Comparar o mesmo `latent` isola o decoder.

### Padrões atuais de geração e alinhamento com o Gradio

Referência: [ACE-Step 1.5, commit ca1e85f](https://github.com/ace-step/ACE-Step-1.5/tree/ca1e85fe9430179831e6bc6be790c332190a3866). Foram conferidos os controles do Gradio e o `generate_audio` do SFT; os defaults da API nem sempre são iguais aos da interface.

No Generate, use `sampler_name=jkass_quality`, `scheduler=ace_step`, `noise_source=official`, `steps=50`, `cfg=7`, `shift=3`, `infer_method=ode`, `guidance_mode=apg`. São os padrões dos novos widgets/nodes. Workflows salvos mantêm seus valores: selecione essas opções explicitamente para migrar. A duração de TextEncode e Generate deve coincidir; `duration=0` no Generate utiliza a duração do TextEncode quando não há áudio/latent de entrada.

- `ace_step`: timesteps lineares deslocados, calculados em FP32 por padrão; `schedule_precision=model` conserva o dtype anterior. `cfg_interval_start/end` são limites do **timestep**, que desce de 1 a 0. Nos outros schedulers, continuam sendo frações do progresso. `guidance_interval` pode restringir adicionalmente a faixa central.
- `jkass_quality`: incorporado do JK-AceStep-Nodes, preservando as atualizações Heun de segunda ordem e os callbacks originais. Não é necessário instalar o pacote JK. Em um schedule terminado em zero, N passos usam 2N−1 avaliações do modelo (99 para 50 passos). O último passo retorna a previsão limpa.
- `noise_source=official`: ruído gerado no dispositivo/dtype do modelo em layout BTC; `comfy` mantém o ruído CPU do ComfyUI. Para lotes, usamos sementes explícitas `seed + índice`, mantendo variações reproduzíveis; a interface Gradio pode sortear sementes adicionais.
- O áudio mantém a proteção anterior contra clipping, sem normalização para um volume-alvo. O sampler mostra o progresso no console pelo ComfyUI.
- APG/ADG e condicionamento nulo seguem as fórmulas do SFT. O prompt interno do LM e seus defaults continuam: CFG 2, temperatura 0.85, top-p 0.9 e top-k 0. O node não reescreve automaticamente sua letra/caption com o CoT do Gradio.
- No APG, reavaliações consecutivas do mesmo timestep (como no Heun/`jkass_quality`) reutilizam o histórico anterior de momentum e recalculam a previsão atual. Isso evita acumular momentum duas vezes no mesmo timestep; Euler mantém a recorrência original.
- Nesta instalação, o core não propaga as janelas de atenção aos encoders de letra e timbre. A opção `reference_xl_attention` do Generate aplica ao timbre as janelas configuradas no modelo, seguindo a alternância local/global do XL. Ela usa ModelPatcher durante a geração com referência, sem editar o core ou o encoder de letra, e fica ligada por padrão. `false` permite comparar o comportamento anterior.

Correções locais de estabilidade: a amostragem nativa do LM calcula CFG e probabilidades em FP32, seguindo o backend PyTorch oficial, e `lm_top_p=0` desativa o filtro. O orçamento de códigos usa `int(duration * 5)`, sem arredondar primeiro a duração para segundos inteiros. Generate avisa quando a duração diverge mais de um segundo da usada pelo LM. Áudios de entrada acima do limite são reduzidos proporcionalmente, preservando a forma de onda e o balanço estéreo. O forward nativo também conserva uma referência de timbre independente por amostra do lote. As duas correções no core do ComfyUI estão em `comfy/text_encoders/ace15.py` e `comfy/ldm/ace/ace_step15.py`; uma atualização do ComfyUI pode sobrescrevê-las.

O Gradio SFT usa CFG constante: `guidance_schedule=normal`. Para CFG 7 → 1, use `decay` e `min_guidance_scale=1`; para 1 → 7, use `ramp_up`. A rampa segue os passos reais e também funciona em CFG normal e split. O nome do checkpoint não força CFG 1, pois arquivos merge não podem ser classificados com segurança pelo nome.

**Limites de equivalência:** ComfyUI continua responsável pelo LM, precisão de integração, atenção e VAE. `jkass_quality` sempre usa integração ODE. `infer_method=sde` apenas remapeia as opções Euler/Heun compatíveis do ComfyUI para alternativas SDE; não transforma o JKASS em SDE. O sampler incorporado não é o solver Euler do Gradio. DCW/retake/flow-edit e o CoT automático do Gradio não foram portados; DCW é desligado por padrão no Gradio SFT. Os mesmos parâmetros não garantem áudio idêntico entre as duas implementações.

Validação local: `python -m unittest discover -s tests -v`. Para comparações diretas de ruído/APG/ADG com o código oficial, defina `ACESTEP_REFERENCE_DIR` apontando para o checkout dessa referência.

Este pacote fornece **quinze nós** em `audio/AceStep SFT`:

| Nó | Finalidade |
|----|-----------|
| **AceStep 1.5 SFT Model Loader** | Carrega o modelo de difusão, encoders CLIP e VAE |
| **AceStep 1.5 SFT Cascade Model Loader** | Carrega SFT/Base/Turbo com arquitetura e encoders/VAE compartilhados; retenção seletiva de GPU experimental |
| **AceStep 1.5 SFT Lora Loader** | Aplica um LoRA no MODEL + CLIP (encadeável) |
| **AceStep 1.5 SFT Lora Loader With Trigger Word** | Mesmo carregamento de adapter, com saída STRING da trigger; prioriza Side-Step |
| **AceStep 1.5 SFT TextEncode** | Codifica caption, letras e metadados em condicionamento |
| **AceStep 1.5 SFT Generate** | Sampler de difusão + decodificação opcional com VAE |
| **AceStep 1.5 SFT Generate Advanced** | Controle explícito de ruído e intervalo de passos para geração em etapas |
| **AceStep 1.5 SFT Cover / Remix** | Condicionamento nativo de cover/remix a partir de um áudio-fonte |
| **AceStep 1.5 SFT Inpaint / Repaint** | Regeneração nativa de um intervalo, preservando a fonte |
| **AceStep 1.5 SFT Preview Audio** | Reprodução de áudio com visualizador de espectro |
| **AceStep 1.5 SFT Save Audio** | Salva áudio (FLAC/MP3/Opus) com visualizador de espectro |
| **AceStep 1.5 SFT Audio Duration** | Retorna a duração do áudio como um número inteiro de segundos |
| **Load Reference Audio** | Envio/seleção de áudio opcional com switch de ativação |
| **AceStep 1.5 SFT Get Music Infos** | Transcrição nativa com letra formatada para ACE-Step, tags, BPM e tom/escala |
| **AceStep 1.5 SFT Turbo Tag Adapter** | Reescreve tags orientadas a Turbo em tags amigáveis ao SFT (Beta) |


### Migração do sampler

O sampler `ace_step_euler` foi removido. Em workflows salvos que o utilizavam, selecione `jkass_quality` no Generate. O scheduler `ace_step` foi mantido. Reinicie o ComfyUI e atualize a página para recarregar a lista. A cópia local não altera o registro global de samplers nem o pacote JK; mesmo que o JK esteja instalado, o Generate usa a implementação incorporada e mostra apenas uma entrada `jkass_quality`.

O decaimento CFG 7 → 1 continua disponível com `guidance_schedule=decay` e `min_guidance_scale=1`; `ramp_up` inverte a direção. A rampa segue os timesteps reais, sem contar a avaliação corretora do Heun como um passo adicional. Os controles `enable_normalization`, `normalization_db`, `velocity_norm_threshold` e `velocity_ema_factor` foram removidos; permanece a proteção de pico da saída.

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

O nó suporta quatro modos de classifier-free guidance:

- **APG (Adaptive Projected Guidance)** ⭐ *Recomendado*
  - Adaptação dinâmica via buffer de momentum
  - Clipping de gradiente com limiares adaptativos
  - Projeção ortogonal para eliminar ruído indesejado
  - **Padrão do AceStep SFT** - melhor equilíbrio entre qualidade e estabilidade

- **APG na previsão limpa (`apg_denoised`)**
  - Experimento de APG sobre `x0`, com os mesmos controles de momentum e norma
  - Consulte os testes de qualidade do SFT puro acima

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

- **Extração de Tags de Áudio**: Deriva tags de letra, voz e estrutura a partir do modelo de transcrição selecionado
- **Saída de Letra Formatada**: Retorna `lyrics` com marcações de seção do ACE-Step; o Whisper usa blocos limitados `[Verse N]`, enquanto o modelo nativo preserva as seções detectadas por ele
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

### Música nova com referência de voz

Conecte uma gravação vocal do Load Audio diretamente à entrada opcional `reference_audio` do Generate, sem task de Cover/Remix nem música-fonte. Preencha caption, lyrics, idioma e duração da música nova no TextEncode normalmente. Conecte o VAE ao Generate quando a referência for AUDIO. A referência orienta características acústicas globais, incluindo o timbre vocal; ela não fornece a letra nova nem garante um cantor idêntico.

O Generate Advanced aceita a mesma entrada. Em uma geração dividida, conecte a referência a cada etapa e continue o LATENT da primeira com `add_noise=disable`. A referência é codificada em cada etapa; os códigos de áudio LM e a duração solicitada mantêm o comportamento normal. Sem códigos LM, o contexto da composição usa silêncio codificado, sem tratar a gravação falada como música-fonte.

Para investigar um resultado abafado, salve primeiro o AUDIO direto do Generate e compare com qualquer processamento de Music Fix, estéreo ou volume da mesma geração. Use uma referência clara, com volume de gravação adequado e poucas pausas; incluir canto é uma comparação útil com um trecho curto de fala. Não se aplica EQ, ganho ou mudança de precisão automaticamente para compensar a referência.

`reference_xl_attention=true` é o padrão para aproximar a extração de timbre do [encoder XL oficial](https://github.com/ace-step/ACE-Step-1.5/blob/main/acestep/models/xl_sft/modeling_acestep_v15_xl_base.py). A opção faz o encoder de timbre usar as mesmas janelas de atenção configuradas no decoder do modelo; não altera os códigos LM nem transforma a fala em fonte de composição. Para comparar, use `false` no teste A e `true` no B, mantendo referência, seed e demais campos iguais. A opção funciona no Generate e Advanced, independentemente de `timbre_cls_token`, e restaura os patches ao terminar ou ocorrer erro. Sem referência, não há alteração de atenção. Os testes verificam as janelas e sua restauração; melhora de clareza e semelhança vocal depende da comparação auditiva.

Dois controles adicionais permitem experimentar um condicionamento mais forte da referência direta no Generate e Generate Advanced:

- `reference_guidance_scale` (padrão `1`): em `1`, mantém a previsão original única. Outros valores comparam previsões com a referência e com silêncio codificado, mantendo texto, letra e contexto de composição iguais, e reforçam essa diferença. Isso acrescenta uma avaliação do modelo a cada avaliação condicional. O cálculo acontece **antes** do CFG/APG: com CFG 7, comece comparando `1.2–1.25`, em vez de usar diretamente `2–4`.
- `reference_timbre_repeats` (padrão `1`): repete o token de timbre codificado antes da atenção cruzada, aumentando seu peso relativo sem amplificar sua amplitude. Compare `1` com `2` ou `3` separadamente do guidance. Não repete o áudio nem a música. O reforço atua nas características acústicas globais da referência, não em uma representação treinada exclusivamente para identidade vocal.

São métodos experimentais somente de inferência, não recursos oficiais de clonagem do ACE-Step. Aplicam-se a `reference_audio` direto, sem `task`, e seus patches são restaurados após a geração, inclusive em erros. Use ambos em `1` para manter o comportamento anterior. Valores fortes podem mudar composição, dicção ou qualidade sonora. Mantenha seed, referência, caption, letra e configurações LM iguais nas comparações; pontuações de embeddings entre fala e canto não são porcentagens de identidade nem substituem a escuta. Não são acrescentados arquivos de log; a geração usa o progresso nativo existente do ComfyUI.

### 🎛️ Cover, Remix e Inpaint Nativos

Essas operações usam os caminhos de condicionamento do ACE-Step 1.5 nos quais o modelo foi treinado. Elas são separadas de `latent_or_audio`; o fluxo audio-to-audio atual, no estilo img2img, continua inalterado.

Conecte a saída do nó de tarefa às entradas `task` de **TextEncode e Generate**:

```text
Load Audio/Latent → Cover / Remix ─┬→ TextEncode.task
                                   └→ Generate.task

Load Audio/Latent → Inpaint / Repaint ─┬→ TextEncode.task
                                       └→ Generate.task
```

- **Cover / Remix**: Recebe a música-fonte completa e regenera uma saída completa — tanto a interpretação vocal quanto os instrumentos — no estilo solicitado. No ACE-Step upstream, Remix é o nome exibido na interface para a mesma tarefa `cover`.
- **Semantic FSQ**: Modo recomendado; extrai códigos semânticos em 5 Hz e reconstrói hints estruturais em 25 Hz, permitindo trocar mais o estilo e o timbre enquanto segue a composição da fonte.
- **Raw no-FSQ**: Modo avançado que condiciona diretamente com o latent acústico e, por isso, retém mais do som da fonte.
- **Remix strength**: `remix_strength = 0.4` é o padrão e o ponto inicial recomendado neste node. O guia oficial sugere aproximadamente `0.3–0.5` para uma mudança dramática de gênero; esse valor é a fração dos passos de amostragem que usa o condicionamento Cover.
- **Identidade do cantor (aproximada)**: Com `preserve_singer_identity` ativado, a música-fonte também é usada como `reference_audio` por padrão. Isso orienta o timbre vocal, mas não é clonagem exata de voz, pois Cover regenera o waveform completo. Um `reference_audio` explícito substitui a referência automática da fonte.
- **Letra explícita**: Para manter as palavras originais, cole a letra original no TextEncode. O Cover nativo não infere nem preserva a letra quando o campo está vazio, e `generate_audio_codes` é desativado nessa tarefa.
- **Presets de modelo**: Use CFG `1`, `8` steps, ODE e shift `3` no Turbo. Use CFG `7` no Base/SFT.
- **Inpaint / Repaint**: Regenera somente `start_seconds:end_seconds` com a instrução nativa de repaint, latent da fonte com a região silenciada e máscara ACE de 64 canais.
- **Preservação da fonte**: Reinjeta a fonte com o nível correto de ruído durante a amostragem, mistura os limites no latent e restaura o waveform original fora do intervalo com um crossfade curto.

Nas tarefas nativas, a duração e o batch vêm da fonte e a geração de audio codes é desativada automaticamente. O caminho legado de img2img por `latent_or_audio` continua inalterado e deve ficar desconectado neste fluxo.

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

- **Latent Shift**: Deslocamento aditivo opcional do latent (padrão 0)
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

4. Instale `requirements.txt` no mesmo Python usado pelo ComfyUI. No Windows portable, execute a partir da raiz:
```powershell
.\python_embeded\python.exe -m pip install -r .\ComfyUI\custom_nodes\ComfyUI-AceStep_SFT\requirements.txt
```

5. Reinicie o ComfyUI - os nós aparecerão em `audio/AceStep SFT`. Use uma versão com suporte nativo ao ACE-Step 1.5. As correções locais no core descritas acima ficam fora deste repositório de custom node.

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

### AceStep 1.5 SFT Lora Loader With Trigger Word

Reutiliza a aplicação de adapters e os controles de intensidade do loader existente, acrescentando a saída `Trigger word` (`STRING`). Junte essa string às tags musicais antes de conectar o caption ao TextEncode; o node não reescreve o caption nem a letra.

- O `custom_tag` declarado pelo Side-Step tem prioridade sobre os dados do Civitai. É lido do cabeçalho safetensors ou de um JSON local correspondente, incluindo `metadata.custom_tag` no JSON de dataset. Tags diferentes por música não são combinadas em uma trigger inventada.
- Também aceita outros metadados explícitos, como `modelspec.trigger_phrase`, e arquivos locais `.civitai.info` / `.info`. Vários `trainedWords` declarados no Civitai saem separados por vírgula.
- `trigger_word_override`: valor manual opcional, com prioridade máxima.
- `lookup_civitai`: ativado por padrão. Somente se não encontrar a trigger localmente, envia o SHA-256 do arquivo à [consulta pública do Civitai](https://github.com/civitai/civitai-developer-docs/blob/main/site/reference/model-versions.md). Desative para uso offline. O modelo não é enviado.

**Exportações do Side-Step:** definir a trigger no treinamento não garante que o [safetensors exportado a preserve](https://github.com/koda-dernet/Side-Step/blob/main/sidestep_engine/core/comfyui_export.py). Se a exportação não tiver esse metadado, copie o `preprocess_meta.json` correspondente para junto de `meu_estilo.safetensors`, renomeando a cópia para `meu_estilo.sidestep.json`. O loader lê o `custom_tag` automaticamente. Outra opção é criar esse arquivo correspondente com:

```json
{"custom_tag": "trigger_do_meu_estilo"}
```

Para diretórios de adapters PEFT, aceita `sidestep_metadata.json`, `preprocess_meta.json` ou `dataset.json` junto dos pesos quando existe `adapter_config.json`. O `sidestep_metadata.json` mantém a tag declarada no treinamento disponível mesmo sem o dataset original. Novas conversões automáticas deste pacote preservam os metadados originais e incorporam a trigger local detectada em `modelspec.trigger_phrase`. Arquivos já convertidos não são sobrescritos.

Os dados são lidos apenas como metadados. O loader não segue caminhos de dataset dos arquivos de configuração nem adivinha triggers pelo nome do arquivo ou `ss_tag_frequency`. Sem trigger declarada ou em caso de falha na consulta opcional, retorna string vazia e continua carregando o adapter. Alterações no adapter ou nos arquivos correspondentes de metadados invalidam o cache de execução do novo node.

### AceStep 1.5 SFT Cover / Remix

Cria uma tarefa nativa de cover/remix da música inteira. Conecte a música completa em `source` e ramifique a saída `task` para as entradas `task` do TextEncode e do Generate. O Generate devolve uma mix completa recém-sintetizada: o cantor interpreta no estilo-alvo e os instrumentos são regenerados junto com ele.

Entradas:
- `source`: AUDIO ou LATENT que fornece melodia, ritmo e estrutura
- `source_conditioning`: `semantic_fsq` (recomendado) ou `raw_no_fsq`
- `remix_strength`: fração da amostragem que usa o condicionamento Cover; padrão `0.4`. O guia oficial recomenda aproximadamente `0.3–0.5` para uma mudança dramática de gênero
- `cover_noise_strength`: padrão `0.2`; `0` começa em ruído puro, enquanto valores maiores inicializam mais perto da fonte
- `preserve_singer_identity`: ativado por padrão; também usa a música-fonte como `reference_audio` para orientar o timbre do cantor regenerado. Desative para gerar uma voz livremente
- `reference_audio` opcional: referência explícita de timbre/voz em AUDIO ou LATENT; quando conectada, substitui a referência automática da fonte

A identidade do cantor é aproximada, não clonagem exata de voz: Cover/Remix regenera o waveform completo, inclusive a interpretação vocal. Mantenha `source_conditioning = semantic_fsq` no fluxo normal de remix. Cole a letra original explicitamente no TextEncode se o cover precisar cantar as mesmas palavras; a entrada de letra vazia não transcreve a fonte. As tarefas nativas desativam `generate_audio_codes`, pois o áudio-fonte real fornece o condicionamento nativo.

Para **Turbo**, use CFG `1`, `8` steps, ODE e shift `3`. O Cover nativo usa os timesteps shifted-linear do ACE-Step e normaliza o pico após o decode em vez de recortar a onda. Para **Base/SFT**, use CFG `7`. A entrada legada de img2img `latent_or_audio` continua inalterada e desconectada desse workflow.


### AceStep 1.5 SFT Inpaint / Repaint

Cria uma tarefa nativa de repaint por intervalo. Ramifique a saída `task` para as entradas `task` do TextEncode e do Generate.

Entradas:
- `source`: AUDIO ou LATENT a editar
- `start_seconds`, `end_seconds`: intervalo regenerado (`-1` significa o fim da fonte)
- `repaint_mode`: `conservative`, `balanced` ou `aggressive`
- `repaint_strength`: equilíbrio entre preservação e liberdade usado pelo modo `balanced`
- `reference_audio` opcional: referência independente de timbre/estilo para a região regenerada

### AceStep 1.5 SFT TextEncode

Codifica caption, letras e metadados em condicionamento positivo e negativo para o nó Generate.

Entradas:
- `clip`: CLIP do Model Loader ou Lora Loader
- `caption`: Descrição textual da música (gênero, humor, instrumentos)
- `lyrics`: Letra da música ou `[Instrumental]`
- `instrumental`: Forçar modo instrumental
- `seed`, `duration`, `bpm`, `timesignature`, `language`, `keyscale`
- Opcional: `task`, `generate_audio_codes`, `lm_cfg_scale`, `lm_temperature`, `lm_top_p`, `lm_top_k`, `lm_min_p`, `lm_negative_prompt`
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
- Opcional: `vae` (para saída de áudio), `latent_or_audio` (para o img2img atual), `task` (para edição nativa), `batch_size`
- Pós-processamento opcional: `latent_shift`, `latent_rescale`, `fade_in_duration`, `fade_out_duration`, `voice_boost`, `use_tiled_vae`
- Guidance opcional: `apg_eta`, `apg_momentum`, `apg_norm_threshold`, `guidance_interval`, `guidance_schedule`, `min_guidance_scale`, `guidance_scale_text`, `guidance_scale_lyric`, `omega_scale`, `erg_scale`, `cfg_interval_start`, `cfg_interval_end`, `shift`

Saídas:
- `model`: MODEL (passthrough para encadeamento)
- `vae`: VAE (passthrough para encadeamento)
- `positive`: CONDITIONING (passthrough)
- `negative`: CONDITIONING (passthrough)
- `latent`: LATENT (saída bruta da difusão)
- `audio`: AUDIO (áudio decodificado, apenas quando VAE está conectado)

### AceStep 1.5 SFT Generate Advanced

Acrescenta controles como os do KSampler Advanced, mantendo os samplers do Generate (incluindo `jkass_quality`), controles de guidance e saídas. O Generate comum permanece inalterado. O Advanced usa `noise_seed` no lugar de `seed`, não possui `denoise` e acrescenta:

| Controle | Padrão | Comportamento |
|----------|--------|---------------|
| `add_noise` | `enable` | `disable` continua um latent com ruído sem acrescentar ruído novo |
| `start_at_step` | `0` | Primeiro passo executado no schedule completo de `steps` |
| `end_at_step` | `10000` | Limite de parada, limitado ao último passo do schedule completo |
| `return_with_leftover_noise` | `disable` | `enable` preserva o nível de ruído no limite intermediário; `disable` termina com ruído zero |

Para duas etapas com `steps=50`, use o mesmo sampler, scheduler, shift e `noise_seed` nos dois nós:

| Etapa | `add_noise` | `start_at_step` | `end_at_step` | `return_with_leftover_noise` |
|-------|-------------|-----------------|---------------|------------------------------|
| Primeira | `enable` | `0` | `25` | `enable` |
| Final | `disable` | `25` | `50` | `disable` |

Conecte a saída `latent` do primeiro nó à entrada `latent_or_audio` do segundo. Intervalos incompletos ou que não executam nenhum passo não decodificam AUDIO, mesmo com VAE conectado.

O Advanced também aceita Cover/Remix e Repaint nativos pela entrada opcional `task`. Conecte o mesmo task ao TextEncode e a cada etapa Advanced, com o VAE em cada etapa que recebe task para codificar a fonte e a referência. A etapa final pode receber `task` junto com o LATENT da etapa anterior quando `add_noise=disable`: ela continua esse latent e reaplica o condicionamento de fonte e referência no modelo selecionado. Task junto com uma entrada AUDIO ou com ruído inicial novo é ambíguo e não é aceito.

No Cover, `start_at_step` e `end_at_step` usam o schedule completo original, mesmo quando `cover_noise_strength` pula os passos iniciais. Com 50 passos, `cover_noise_strength=0.2` e `shift=1`, o início automático ocorre no passo 10: o primeiro intervalo 0–25 executa 10–25, seguido de 25–50. O blend com a fonte e a emenda da waveform no Repaint ocorrem somente quando a etapa final termina com ruído zero.

O `guidance_schedule` e a troca de condicionamento dos códigos LM seguem o schedule completo, sem recomeçar em cada etapa. Dividir a geração dá mais controle, sem garantir melhora de qualidade. Momentum APG, histórico de samplers de múltiplos passos e estado do ruído SDE podem reiniciar entre nós; nem toda configuração reproduz uma execução contínua. O `standard_cfg` determinístico com o `jkass_quality`, baseado em Heun, permite continuação. Não são necessárias alterações no core do ComfyUI nem dependências adicionais.

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
- **Save Metadata** (`save_metadata`, opcional, ativado por padrão): Incorpora o prompt e o workflow do ComfyUI no áudio. Desative para salvar sem esses metadados em FLAC, MP3 ou Opus. A flag global `--disable-metadata` sempre impede essa incorporação. Workflows existentes mantêm o comportamento anterior quando esse campo não é informado.

Recursos:
- Nomes de arquivo auto-incrementais (ex: `AceStep_00001_.flac`, `AceStep_00002_.flac`)
- Visualizador de forma de onda com play/pause e seek
- Incorporação opcional de metadados (prompt, workflow)

### AceStep 1.5 SFT Audio Duration

Recebe um valor `AUDIO` do ComfyUI e retorna `duration_seconds` como `INT`, arredondado para o segundo inteiro mais próximo. A saída desativada do Load Reference Audio (`None`) retorna `0`; áudio malformado continua gerando erro. O node lê diretamente a forma de onda em memória e não carrega nenhum modelo.

### AceStep 1.5 SFT Get Music Infos

Nó de análise de áudio com IA que extrai tags descritivas, letra formatada para ACE-Step, BPM e tom/escala. O Whisper large-v3 é o padrão de maior precisão para a letra; músicas longas são transcritas em trechos limitados e reunidas antes do retorno de tags e letra.

Entradas:
- `audio`: Entrada de áudio para análise
- `get_tags` / `get_lyrics` / `get_bpm` / `get_keyscale`: Ativar/desativar cada análise
- `max_new_tokens`: Teto máximo de tokens para cada trecho de transcrição
- `audio_duration`: Máximo de segundos de áudio para análise
- `transcription_chunk_seconds`: Duração dos trechos de músicas longas; o padrão `30` impede que um loop do decoder em uma passagem consuma toda a saída
- `lyrics_model`: `Whisper-large-v3-transcription` (padrão preciso), Whisper Turbo (mais rápido) ou ACE-Step Transcriber nativo
- `lyrics_language`: Dica opcional de idioma; padrão `auto`, ou use `pt` para português brasileiro quando a voz tiver muitos efeitos
- `temperature`, `top_p`, `top_k`, `repetition_penalty`, `seed`: Parâmetros de geração
- `unload_model`: Liberar VRAM após análise
- `use_flash_attn`: Ativar Flash Attention 2 (se compatível)
- `free_vram_before`: Descarregar os modelos de geração do ComfyUI antes de carregar o modelo de transcrição selecionado

Saídas:
- `tags`: Tags descritivas separadas por vírgula (STRING)
- `bpm`: BPM detectado (INT)
- `keyscale`: Tom e escala ex: "G minor" (STRING)
- `music_infos`: JSON com todos os resultados (STRING)
- `lyrics`: Letra pronta para o TextEncode, preservando as seções do ACE-Step (STRING)

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
| **sampler_name** | jkass_quality / ComfyUI | Heun JKASS local (padrão) |
| **scheduler** | ace_step | ACE-Step shifted-linear / schedulers ComfyUI |
| **denoise** | 0.0 - 1.0 | Intensidade de denoising (1.0 = novo, < 1.0 = edição) |
| **duration** | 0.0 - 600.0 | Duração em segundos (0 = automático) |
| **infer_method** | ode/sde | SDE remapeia samplers ComfyUI compatíveis; JKASS permanece ODE |
| **guidance_mode** | apg/adg/standard_cfg/apg_denoised | Tipo de guidance (padrão: apg) |

### Generate - Parâmetros Opcionais

#### Geração em Lote
- **batch_size** (1-16): Número de áudios para gerar em paralelo

#### Controle de execução
- **noise_source** (`official`/`comfy`, padrão: `official`): Ruído no dispositivo/dtype do modelo ou ruído CPU do ComfyUI
- **unload_models_after_generate** (padrão: False): Descarrega modelos após gerar

#### Entrada de Áudio
- **vae**: VAE do Model Loader (necessário para saída de áudio)
- **latent_or_audio**: Entrada base para refinamento (img2img). Aceita AUDIO ou LATENT. `denoise=0` preserva o latent de origem; `denoise<1` começa na fração de ruído solicitada
- **task**: Tarefa nativa Cover/Remix ou Inpaint/Repaint, conectada também ao TextEncode. No Generate comum, é mutuamente exclusivo com `latent_or_audio`; no Advanced, pode acompanhar um LATENT de continuação com `add_noise=disable`.
- **reference_audio**: Referência opcional de timbre em AUDIO/LATENT para uma música nova, independente de `task` e `latent_or_audio`. Com task conectado, essa entrada substitui a referência dele sem alterar sua música-fonte.
- **reference_xl_attention**: Ligada por padrão para usar no encoder de timbre as janelas de atenção configuradas no modelo. Atua somente quando há referência vocal; `false` restaura a atenção anterior.

#### Pós-processamento de Latent
- **latent_shift** (-0.2-0.2, padrão: 0.0): Deslocamento do latent; mantenha 0 salvo alteração intencional do decode
- **latent_rescale** (0.5-1.5, padrão: 1.0): Escalonamento multiplicativo
- **fade_in_duration / fade_out_duration** (0.0-10.0, padrão: 0.0): Fades lineares opcionais
- **use_tiled_vae** (padrão: True): Decodifica áudio 1D em blocos com contexto e descarta previsões das bordas; reduz o pico de VRAM
- **voice_boost** (-12.0-12.0, padrão: 0.0): Ganho em dB da mix inteira antes da proteção de pico; não isola o vocal

#### Configuração APG
- **apg_eta** (-10.0-10.0, padrão: 0.0): Retenção do componente paralelo
- **apg_momentum** (-1.0-1.0, padrão: -0.75): Coeficiente do buffer de momentum
- **apg_norm_threshold** (0.0-15.0, padrão: 2.5): Limiar de norma para clipping de gradiente

#### Controles de Guidance Estendidos
- **guidance_interval** (-1.0-1.0, padrão: 1.0): Largura do intervalo de guidance centrado; `1.0` cobre todo o schedule oficial
- **guidance_schedule** (`normal`, `decay`, `ramp_up`; padrão: `normal`): `normal` mantém CFG constante, `decay` vai de CFG → mínimo e `ramp_up` vai de mínimo → CFG. As rampas são lineares dentro do intervalo ativo de guidance e também funcionam com `jkass_quality`. Ao atualizar um workflow antigo, selecione o modo desejado no Generate.
- **min_guidance_scale** (0.0-30.0, padrão: 3.0): CFG final em `decay` ou inicial em `ramp_up`; não é usado em `normal`.
- **guidance_scale_text** (-1.0-30.0, padrão: -1.0): Guidance apenas de texto (split)
- **guidance_scale_lyric** (-1.0-30.0, padrão: -1.0): Guidance apenas de letra (split)
- **omega_scale** (-8.0-8.0, padrão: 0.0): Rebalanceamento preservando a média
- **erg_scale** (-0.9-2.0, padrão: 0.0): Rebalanceamento de energia do prompt
- **cfg_interval_start / cfg_interval_end** (0.0-1.0): Limites de timestep no `ace_step`; frações do progresso nos demais schedulers
- **shift** (0.1-5.0, padrão: 3.0): Deslocamento do schedule de timestep

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
- **lm_top_p** (0.0-1.0, padrão: 0.9): Amostragem nucleus; 0 ou 1 desativa o filtro
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
  sampler_name: "jkass_quality", scheduler: "ace_step"
  noise_source: "official", infer_method: "ode", shift: 3.0
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
  denoise: 0.7 (fração inicial de ruído; não garante preservar 30% da música)
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

O Generate reduz proporcionalmente picos acima de 1, sem cortar a forma de onda. Mantenha `latent_shift=0` e `latent_rescale=1`: alterar os latentes não equivale a ajustar o volume. Com `use_tiled_vae=true`, a VAE 1D decodifica blocos com 64 frames de contexto de cada lado e descarta as bordas, seguindo a política overlap-discard do ACE-Step. Isso evita misturar previsões nas emendas; repetições de sílabas já presentes nos latentes ainda dependem da geração.

### Resultados com Alta Variância

Comece com `apg_eta=0`, `apg_momentum=-0.75` e `apg_norm_threshold=2.5`. Um limiar menor limita mais a atualização APG; aumentar esse valor permite atualizações maiores. Se a voz estiver instável, compare com o sampler `jkass_quality` e scheduler `ace_step`, mantendo seed e os demais parâmetros. O Generate rejeita parâmetros fora das faixas dos widgets; em workflows antigos, confira se os valores ficaram nos campos corretos.

### Qualidade Inferior ao Esperado

**Solução**: 
1. Use `guidance_mode: "apg"` (recomendado)
2. Comece com `steps: 50`, `cfg: 7.0`, `sampler_name: "jkass_quality"`, `scheduler: "ace_step"`, `infer_method: "ode"`

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

## 📚 Referências de implementação

- [ACE-Step 1.5 reference](https://github.com/ace-step/ACE-Step-1.5/tree/ca1e85fe9430179831e6bc6be790c332190a3866)
- [ComfyUI](https://github.com/Comfy-Org/ComfyUI)
- JK-AceStep-Nodes: `py/jkass_sampler.py`, `sample_jkass_quality`; [MIT attribution](THIRD_PARTY_NOTICES.md).

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
