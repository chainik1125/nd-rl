---
written_on: 2026-11-08
written_by: agent:claude
---

# Frontier and flagship industry open-weight releases: which stage checkpoints around RL are public

How it was checked (2026-11-08). I listed every repo for 22 orgs through the Hugging Face (HF) API (`/api/models?author=…`, with `expand[]=safetensors,cardData,config,gated`). I pulled 196 candidate model cards (`/raw/main/README.md`) and the `/refs` of 190 public repos plus 6 extra pairs. For the key pairs I compared `tokenizer_config.json` and `chat_template.jinja` by md5. The datasets were listed for 4 orgs.

Tags used below:
- **[HF]** means I verified it on HF myself: the repo exists, its parameter count from safetensors, its licence, its creation date, its branches, its template, or text on its card.
- **[report]** means it is only claimed in a paper or blog that I fetched.
- **[secondary]** means it comes from press or aggregator pages.

The 6 gated repos (Gemma 3 pt and it, and the 4 Llama 4 repos) returned "Access … restricted" to unauthenticated `/refs`. Their branches are **not** verified.

## Repos and sizes: which stages each lab put on HF

### Takeaway
Most industry labs publish at most **base + final**. A few have shipped a **released checkpoint that is the exact init of a released RL model**:
- Xiaomi MiMo-7B;
- Microsoft Phi-4-reasoning → Phi-4-reasoning-plus;
- Prime Intellect INTELLECT-3 → INTELLECT-3.1;
- StepFun RLVR-8B → PaCoRe-8B;
- Qwen3-4B → Qwen3-4B-SafeRL;
- NVIDIA's Nemotron-3 Ultra Math SFT/RL pair, which is probable but unconfirmed;
- DeepSeek-Prover-V1.5 SFT/RL and MiMo-VL SFT/RL.

Every one of these is small (≤15B), except INTELLECT-3.x (106B-A12B) and the 560B NVIDIA pair. Every 2025–26 flagship (≥200B) is base + final at best.

### Cited findings

The parameter counts in this section are safetensors totals from the HF listing. Active parameters come from the model cards.

**DeepSeek**
- These repos exist [HF] ([org listing](https://huggingface.co/api/models?author=deepseek-ai&limit=200)):
  - `DeepSeek-V3-Base`, `DeepSeek-V3`, `DeepSeek-R1-Zero`, `DeepSeek-R1`, `DeepSeek-R1-0528`: all 684.5B FP8, 2024-12-25 to 2025-05-28;
  - `DeepSeek-V3.1-Base` / `V3.1` / `V3.1-Terminus`;
  - `DeepSeek-V3.2-Exp-Base` / `V3.2-Exp` / `V3.2` / `V3.2-Speciale`: 685.4B, 2025-09-29 to 2025-12-01;
  - `DeepSeek-V4-Pro-Base` (1600.8B) and `DeepSeek-V4-Flash-Base` (292.0B), both 2026-04-22, plus their finals and the updates `V4-Flash-0731`, `V4-Pro-0813`, `V4.1-Flash` (763.2B incl. Engram) and `V4.1-Flash-1017`.
- No SFT or "cold-start" checkpoint of any DeepSeek flagship is on HF [HF] ([org listing](https://huggingface.co/api/models?author=deepseek-ai&limit=200)).
- The R1 card says 671B total / 37B activated. It also says "DeepSeek-R1-Zero & DeepSeek-R1 are trained based on DeepSeek-V3-Base" [HF] ([card](https://huggingface.co/deepseek-ai/DeepSeek-R1)).
- The V3.2 card's YAML gives `base_model: deepseek-ai/DeepSeek-V3.2-Exp-Base` [HF] ([card](https://huggingface.co/deepseek-ai/DeepSeek-V3.2)).
- The V4-Pro is 1.6T / 49B activated and the V4-Flash is 284B / 13B activated [HF] ([card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro)).
- Older small pairs exist [HF] ([Prover-V1.5-RL](https://huggingface.co/deepseek-ai/DeepSeek-Prover-V1.5-RL), [math-7b-rl](https://huggingface.co/deepseek-ai/deepseek-math-7b-rl)):
  - `DeepSeek-Prover-V1.5-SFT` → `-RL`, 6.9B, 2024-08. Its card says "including base, SFT and RL models", and the RL card's YAML gives `base_model: …Prover-V1.5-SFT`.
  - `deepseek-math-7b-instruct` / `-rl`, 2024-02.

**Moonshot**
- These repos exist [HF] ([org listing](https://huggingface.co/api/models?author=moonshotai)):
  - `Kimi-K2-Base`, `Kimi-K2-Instruct`, `-Instruct-0905`: 1026.4B FP8;
  - `Kimi-K2-Thinking` (INT4 compressed-tensors), `Kimi-K2.5`, `Kimi-K2.6`, `Kimi-K2.7-Code`;
  - `Kimi-K3` (2779.9B, 2026-06-13) and `Kimi-K3.1-Pro` (2780.1B, 2026-11-02).
- There is no K2.5/K2.6/K3 base and no K2-Thinking pre-RL checkpoint [HF] ([org listing](https://huggingface.co/api/models?author=moonshotai)).
- The K2 cards say 1T total / 32B activated. The K3 and K3.1-Pro cards say 2.8T total / 104B activated, with MXFP4 weights and MXFP8 activations from QAT [HF] ([K3 card](https://huggingface.co/moonshotai/Kimi-K3), [K3.1-Pro card](https://huggingface.co/moonshotai/Kimi-K3.1-Pro)).
- The small pairs `Kimi-Linear-48B-A3B-Base` / `-Instruct` (49.1B) and `Moonlight-16B-A3B` / `-Instruct` exist [HF] ([org listing](https://huggingface.co/api/models?author=moonshotai)).

**Zhipu (zai-org)**
- These repos exist [HF] ([org listing](https://huggingface.co/api/models?author=zai-org)):
  - `GLM-4.5-Base` / `GLM-4.5`: 358.3B, cards say 355B/A32B;
  - `GLM-4.5-Air-Base` / `GLM-4.5-Air`: 110.5B, cards say 106B/A12B;
  - all four created 2025-07-20.
- `GLM-4.6`, `GLM-4.7`, `GLM-4.7-Flash` (31.2B), `GLM-5`, `5.1`, `5.2`, `5.3` (≈754B) and `GLM-5.3-Flash` (321.3B, card says 320B/A18B) have **no base repos** [HF] ([org listing](https://huggingface.co/api/models?author=zai-org)).
- The GLM-5.3 card says it "uses the same base model as GLM-5.2 — every gain comes from post-training" [HF] ([card](https://huggingface.co/zai-org/GLM-5.3)).

**Qwen**
- Base models are released for the Qwen3 dense ≤14B and `Qwen3-30B-A3B-Base`. There is **no** base for `Qwen3-235B-A22B` [HF] ([org listing](https://huggingface.co/api/models?author=Qwen)).
- There is no base for Qwen3-2507 Thinking or Instruct, or for `Qwen3-Next-80B-A3B-Thinking` / `-Instruct` (81.3B) [HF] ([org listing](https://huggingface.co/api/models?author=Qwen)).
- Qwen3.5 (2026-02) ships these bases: `Qwen3.5-35B-A3B-Base`, `9B-Base`, `4B-Base`, `2B-Base`, `0.8B-Base`. It ships **no** base for 27B, 122B-A10B or 397B-A17B [HF] ([org listing](https://huggingface.co/api/models?author=Qwen)).
- Qwen3.6-27B / 35B-A3B and Qwen3.8-27B / 2.4T-A95B / Flash-Next have no base [HF] ([org listing](https://huggingface.co/api/models?author=Qwen)).
- No thinking-only RL-stage checkpoint (for example, after Reasoning RL) is released [HF] ([org listing](https://huggingface.co/api/models?author=Qwen)).

**inclusionAI (Ling and Ring)**
- These pairs exist [HF] ([org listing](https://huggingface.co/api/models?author=inclusionAI)):
  - `Ling-mini-base-2.0` plus pretraining checkpoints at 5T, 10T, 15T and 20T tokens, `Ling-mini-2.0`, `Ring-mini-2.0` (16.3B, card says A1.4B);
  - `Ling-flash-base-2.0` (106.2B), `Ling-flash-2.0`, `Ring-flash-2.0` (102.9B, card says 100B/A6.1B);
  - `Ling-1T` and `Ring-1T` (999.7B, card says ≈50B active);
  - `Ling-2.5-1T` and `Ring-2.5-1T`;
  - `Ling-2.6-1T-base` (created 2026-06-02, *after* the finals), `Ling-2.6-1T`, `Ring-2.6-1T` (1025.4B).
- The repos below are absent from the org listing [HF] ([org listing](https://huggingface.co/api/models?author=inclusionAI)):
  - no `Ling-1T-base`, which is Ring-1T's stated base;
  - no `Ling-2.5-1T-base`;
  - no SFT or "long-CoT SFT" checkpoint for any Ring.
- The Ling-3.0 releases are `-base-30T` (pretrained), `-base-midtrain` and `-base` (WSM-merged), "not undergone post-training" [HF] ([card](https://huggingface.co/inclusionAI/Ling-3.0-flash-base)).
- `Ling-3.1-flash-base` is 572.1B (card says 560B/A25B) [HF] ([card](https://huggingface.co/inclusionAI/Ling-3.1-flash-base)).

**Meituan LongCat**
- These repos exist [HF] ([org listing](https://huggingface.co/api/models?author=meituan-longcat)):
  - `LongCat-Flash-Chat`, `LongCat-Flash-Thinking` and `-Thinking-2601`: 561.9B, card says 560B with 18.6–31.3B dynamic active;
  - `-Thinking-ZigZag`, `LongCat-Flash-Lite` (69.1B), `LongCat-Flash-Prover`, `LongCat-2.0` (1775.6B, card says 1.6T/≈48B).
- **No LongCat base** checkpoint exists at all [HF] ([org listing](https://huggingface.co/api/models?author=meituan-longcat)).

**MiniMax**
- These repos exist [HF] ([org listing](https://huggingface.co/api/models?author=MiniMaxAI)):
  - `MiniMax-Text-01` (a post-trained chat model, 456B/A45.9B);
  - `MiniMax-M1-40k` / `-80k` (456.1B);
  - `MiniMax-M2`, `M2.1`, `M2.5`, `M2.7` (228.7B FP8, card says 230B/A10B);
  - `MiniMax-M3` (427B, card says ≈428B/≈23B) and `M3.1-Flash`.
- There is **no base model** for any of them [HF] ([org listing](https://huggingface.co/api/models?author=MiniMaxAI)).

**ByteDance Seed**
- `Seed-OSS-36B-Base`, `Seed-OSS-36B-Base-woSyn` and `Seed-OSS-36B-Instruct` exist, 36.2B dense, created 2025-08-20 [HF] ([card](https://huggingface.co/ByteDance-Seed/Seed-OSS-36B-Instruct)).
- The card says both base variants were released because "synthetic instruction data in pre-training may affect the post-training research" [HF] ([card](https://huggingface.co/ByteDance-Seed/Seed-OSS-36B-Instruct)).

**Tencent**
- `Hunyuan-A13B-Pretrain` / `-Instruct` (80.4B, card says 80B/A13B) and the dense `Hunyuan-{0.5,1.8,4,7}B-Pretrain` / `-Instruct` exist [HF] ([org listing](https://huggingface.co/api/models?author=tencent)).
- No newer large Hunyuan text LLM appears in the 2025-05 to 2026-11 listing. The newer releases are the Hy-MT / SimulMT / Memory 30B-A3B task models [HF] ([org listing](https://huggingface.co/api/models?author=tencent)).

**Baidu**
- `ERNIE-4.5-300B-A47B-Base-PT` / `-PT` and `ERNIE-4.5-21B-A3B-Base-PT` / `-PT` exist, plus `ERNIE-4.5-21B-A3B-Thinking` (2025-09-08) [HF] ([org listing](https://huggingface.co/api/models?author=baidu)).
- `ERNIE-5.5-Flash-Base` (286.2B, card says 280B-A14B) was released **alone**, with no post-trained ERNIE 5.5 on HF [HF] ([card](https://huggingface.co/baidu/ERNIE-5.5-Flash-Base)).

**StepFun**
- `Step-3.5-Flash-Base`, `Step-3.5-Flash-Base-Midtrain` (197.8B) and `Step-3.5-Flash` (199.4B, card says 11B active) exist [HF] ([org listing](https://huggingface.co/api/models?author=stepfun-ai)).
- `step3` (321B/A38B), `Step-3.7-Flash` (201.4B) and `Step-5-Preview` (604.3B, card says 600B/A27B) have no base [HF] ([org listing](https://huggingface.co/api/models?author=stepfun-ai)).
- The small research pair `RLVR-8B-0926` → `PaCoRe-8B` (8.2B) exists [HF] ([card](https://huggingface.co/stepfun-ai/RLVR-8B-0926)).

**Xiaomi MiMo**
- `MiMo-7B-Base`, `MiMo-7B-SFT`, `MiMo-7B-RL`, `MiMo-7B-RL-Zero` (all 7.8B, 2025-04-29) and `MiMo-7B-RL-0530` exist [HF] ([card](https://huggingface.co/XiaomiMiMo/MiMo-7B-RL)).
- `MiMo-VL-7B-SFT` / `-RL` and `MiMo-VL-7B-SFT-2508` / `-RL-2508` exist [HF] ([org listing](https://huggingface.co/api/models?author=XiaomiMiMo)).
- The larger pairs are `MiMo-V2-Flash-Base` / `MiMo-V2-Flash` (309.8B, card says 309B/A15B), `MiMo-V2.5-Base` / `MiMo-V2.5` (310.8B) and `MiMo-V2.5-Pro-Base` / `MiMo-V2.5-Pro` (1023.2B, card says 1.02T/A42B) [HF] ([org listing](https://huggingface.co/api/models?author=XiaomiMiMo)).
- `MiMo-V2.6-Flash-RL` (177.8 GB on disk) and `MiMo-V2.6-Pro-RL` (573.5 GB) were created 2026-09-21. `MiMo-V2.6-Flash-MOPD` and `-Pro-MOPD` followed on 2026-09-27 [HF] ([org listing](https://huggingface.co/api/models?author=XiaomiMiMo)).
- There is **no V2.6 base or V2.6 SFT** checkpoint [HF] ([org listing](https://huggingface.co/api/models?author=XiaomiMiMo)).

**OpenAI**
- Only `gpt-oss-120b` (116.8B, card says 117B/5.1B active) and `gpt-oss-20b` (21B/3.6B) exist, both MXFP4. There is **no base** [HF] ([card](https://huggingface.co/openai/gpt-oss-120b)).

**Mistral**
- These repos exist [HF] ([org listing](https://huggingface.co/api/models?author=mistralai)):
  - `Magistral-Small-2506` / `-2509` (24B);
  - `Ministral-3-{3,8,14}B-Base` / `-Instruct` / `-Reasoning-2512`;
  - `Mistral-Large-3-675B-Base-2512` / `-Instruct-2512` (card says 675B/41B).
- `Mistral-Large-4-1T-A52B`, `Mistral-Medium-3.5-128B` and `Mistral-Small-4-119B-2603` have no base [HF] ([org listing](https://huggingface.co/api/models?author=mistralai)).
- No Magistral Medium weights are released [HF] ([org listing](https://huggingface.co/api/models?author=mistralai)).

**Google**
- `gemma-4-{E2B,E4B,26B-A4B,31B}` exist in pre-trained and `-it` forms, plus `gemma-4-12B` / `-it` (2026-03 to 2026-05). They are Apache-2.0 and ungated [HF] ([card](https://huggingface.co/google/gemma-4-31B-it)).
- Gemma 3 `-pt` / `-it` are gated (`manual`) under the Gemma licence [HF] ([gemma-3-27b-it](https://huggingface.co/google/gemma-3-27b-it)).

**Meta**
- `Llama-4-Scout-17B-16E` / `-Instruct` (108.6B) and `Llama-4-Maverick-17B-128E` / `-Instruct` (401.6B) are gated. No Llama 5 was found [HF] ([org listing](https://huggingface.co/api/models?author=meta-llama)).

**Microsoft**
- `phi-4` (14.7B), `Phi-4-reasoning`, `Phi-4-reasoning-plus` (all 14.7B), `Phi-4-mini-reasoning` (3.8B) and `Phi-4-mini-flash-reasoning` exist [HF] ([org listing](https://huggingface.co/api/models?author=microsoft)).

**NVIDIA** (brief)
- `NVIDIA-Nemotron-3-Ultra-550B-A55B-Base-BF16` / `-BF16` (560.5B), `Nemotron-3-Labs-Ultra-Math-SFT` and `Nemotron-3-Labs-Ultra-Math-RL` (560.5B, 2026-09-02) exist [HF] ([org listing](https://huggingface.co/api/models?author=nvidia)).
- Both Math repos have YAML `base_model: nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16` [HF] ([RL card](https://huggingface.co/nvidia/Nemotron-3-Labs-Ultra-Math-RL)).
- `Llama-3_1-Nemotron-Ultra-253B-v1` exists [HF] ([org listing](https://huggingface.co/api/models?author=nvidia)).

**Other industry labs**
- Prime Intellect [HF] ([org listing](https://huggingface.co/api/models?author=PrimeIntellect)):
  - `INTELLECT-3` (106.9B, 2025-11-26) and `INTELLECT-3.1` (106.9B, 2026-01-20), both described as 106B/A12B;
  - `INTELLECT-3-FP8`;
  - `INTELLECT-3-Base`, which the card calls "a clone of GLM-4.5-Air-Base … with a custom chat template".
- Arcee [HF] ([card](https://huggingface.co/arcee-ai/Trinity-Large-Thinking)):
  - `Trinity-Large-TrueBase`, `Trinity-Large-Base`, `Trinity-Large-Preview` ("lightly post-trained" instruct) and `Trinity-Large-Thinking`;
  - all 398.6B, card says ≈13B active.
- Poolside: `Laguna-M.1-base` / `Laguna-M.1` (225.8B, card says 23B active) exist [HF] ([card](https://huggingface.co/poolside/Laguna-M.1-base)).
- Upstage: `Solar-Open2-250B` (no base) exists [HF] ([org listing](https://huggingface.co/api/models?author=upstage)).

### Inferences
- The "base + final, nothing in between" pattern holds for **every 2025–26 Chinese and US/EU flagship I checked**.
- The labs that release more (Xiaomi, Microsoft, StepFun, Prime Intellect, NVIDIA) do so for **research-sized or specialist** models, not their flagships.
- MiMo-V2.6 is the one exception. Its RL checkpoint is a flagship, but its *start* is not released.

### Gaps
- Gemma 3 and Llama 4 refs were not checked (gated without a token).
- I did not enumerate Hugging Face *collections*, only org repos. A checkpoint hosted under a personal or other-org account would be missed.

## The RL start: is the exact init of the final RL stage public?

### Takeaway
The **exact RL start is public only for the pairs below**. Everywhere else the base is the only pre-RL weight. SFT, cold-start and (often) mid-training sit between the base and the RL model, and none of those intermediates is released.

| Pair | Model size | Status |
|---|---|---|
| MiMo-7B-SFT → MiMo-7B-RL, and MiMo-7B-Base → MiMo-7B-RL-Zero | 7B | B |
| Phi-4-reasoning → Phi-4-reasoning-plus | 14B | B |
| INTELLECT-3 → INTELLECT-3.1 | 106B-A12B | B |
| RLVR-8B-0926 → PaCoRe-8B | 8B | B |
| Qwen3-4B → Qwen3-4B-SafeRL | 4B | B, safety RL |
| DeepSeek-V3-Base → DeepSeek-R1-Zero | 671B | RL-Zero; graded C by the brief's rule |
| DeepSeek-Prover-V1.5-SFT → -RL | 7B | 2024 |
| MiMo-VL-7B-SFT(-2508) → -RL(-2508) | 7B | VLM |
| Nemotron-3-Labs-Ultra-Math-SFT → -RL | 560B | probable |

### Cited findings
- **MiMo-7B.** The card says: "We open-source MiMo-7B series, including checkpoints of the base model, SFT model, RL model trained from base model, and RL model trained from the SFT model." `MiMo-7B-RL-0530` came from a scaled-up SFT set (≈500K → 6M instances) that is *not* released, so it is not paired [HF] ([card](https://huggingface.co/XiaomiMiMo/MiMo-7B-RL)).
- **Phi-4-reasoning-plus** [report] ([arXiv 2504.21318](https://arxiv.org/html/2504.21318)):
  - The report says: "Starting from a strong SFT model, i.e., Phi-4-reasoning, additional GRPO training for only 90 steps".
  - The selected checkpoint is "the model trained for 90 steps".
- The plus card says it "has been trained additionally with Reinforcement Learning". Its YAML gives `base_model: microsoft/phi-4` (the grandparent) [HF] ([card](https://huggingface.co/microsoft/Phi-4-reasoning-plus)).
- **INTELLECT-3.1** is "a continued training of INTELLECT-3 with additional reinforcement learning on math, coding, software engineering, and agentic tasks" [HF] ([card](https://huggingface.co/PrimeIntellect/INTELLECT-3.1)).
- **INTELLECT-3** itself is "post-trained from GLM-4.5-Air-Base using SFT followed by large-scale RL". Its SFT checkpoint is not on HF [HF] ([card](https://huggingface.co/PrimeIntellect/INTELLECT-3); [org listing](https://huggingface.co/api/models?author=PrimeIntellect)).
- **PaCoRe** [HF] ([card](https://huggingface.co/stepfun-ai/RLVR-8B-0926)):
  - The card calls RLVR-8B-0926 "The initial checkpoint of our study, conducted strong reasoning-oriented post-trained on Qwen3-8B-Base".
  - PaCoRe-8B is "Trained via large-scale, outcome-based reinforcement learning".
- **Qwen3-4B-SafeRL** is "a safety-aligned version of the Qwen3-4B model … trained using RL with a reward signal from Qwen3Guard-Gen". Its YAML gives `base_model: Qwen/Qwen3-4B` [HF] ([card](https://huggingface.co/Qwen/Qwen3-4B-SafeRL)).
- **R1-Zero** is "trained via large-scale RL without SFT as a preliminary step". Both R1-Zero and R1 are "trained based on DeepSeek-V3-Base" [HF] ([card](https://huggingface.co/deepseek-ai/DeepSeek-R1)).
- **R1** "incorporates cold-start data before RL", and its pipeline has "two RL stages … two SFT stages". The cold-start checkpoint is not released [HF] ([card](https://huggingface.co/deepseek-ai/DeepSeek-R1); [org listing](https://huggingface.co/api/models?author=deepseek-ai)).
- **Nemotron Math** [report] ([arXiv 2609.10712](https://arxiv.org/abs/2609.10712)):
  - The abstract says: "Starting from Nemotron 3 Ultra, we train two specialist checkpoints using supervised fine-tuning and reinforcement learning".
  - It also says: "We release the two post-trained checkpoints as well as the training data, the training and inference code".
  - The abstract does **not** state that RL starts from the released SFT checkpoint.
- **Kimi K3.1 Pro** is "Post-trained from Kimi K3 with large-scale joint reinforcement learning across hundreds of downstream domains". The same card says that compared with K3 "we replace the full-attention Gated MLA layers with Gated DeepSeek Sparse Attention (Gated DSA)". So the RL init is **not** the released K3 weights [HF] ([card](https://huggingface.co/moonshotai/Kimi-K3.1-Pro)).
- **Kimi K2-Thinking**: "Starting with Kimi K2, we built it as a thinking agent". Whether it starts from Base or Instruct, and through which SFT, is not stated [HF] ([card](https://huggingface.co/moonshotai/Kimi-K2-Thinking)).
- **Kimi K2.7 Code** is "built upon Kimi K2.6". The method is not stated [HF] ([card](https://huggingface.co/moonshotai/Kimi-K2.7-Code)).
- **MiMo-V2.6** [report] ([arXiv 2610.11959](https://arxiv.org/html/2610.11959)):
  - RL follows "mid-training on a broad multimodal corpus" and a "short supervised fine-tuning (SFT) stage".
  - The report says: "we initialize RL by carrying over the FP32 master weights … from the SFT checkpoint".
  - That checkpoint is not on HF [HF] ([org listing](https://huggingface.co/api/models?author=XiaomiMiMo)).
  - The `-MOPD` repos are "the MOPD upgrade of the MiMo-V2.6-Flash-RL checkpoint", so they are post-RL [HF] ([card](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-MOPD)).
- **GLM-4.5** [report] ([arXiv 2508.06471](https://arxiv.org/html/2508.06471)):
  - The pipeline is expert training (cold-start SFT + expert RL), then unified self-distillation, then general RL.
  - The report says: "As the final stage of post-training, general RL needs to rectify potential issues".
  - The report releases only the GLM-4.5 and GLM-4.5-Air weights.
- **Qwen3 flagship** [report] ([arXiv 2505.09388](https://arxiv.org/html/2505.09388)):
  - It has four stages: Long-CoT Cold Start, Reasoning RL, Thinking Mode Fusion, General RL (last).
  - The small models (0.6B–14B dense and 30B-A3B) are instead made by strong-to-weak distillation, with on-policy distillation last.
- **Ring-flash-2.0**: "lightweight Long-CoT SFT equips the Ling-flash-2.0-base model with diverse thinking patterns", followed by RLVR and then RLHF [HF] ([card](https://huggingface.co/inclusionAI/Ring-flash-2.0)).
- **Ring-mini-2.0**'s YAML gives `base_model: inclusionAI/Ling-mini-base-2.0-20T` [HF] ([card](https://huggingface.co/inclusionAI/Ring-mini-2.0)).
- **Ring-1T** "is trained on the Ling-1T-base foundation model". That base is not on HF [HF] ([card](https://huggingface.co/inclusionAI/Ring-1T); [org listing](https://huggingface.co/api/models?author=inclusionAI)).
- The Ring-1T paper describes three stages: Long-CoT SFT, RLVR, then RLHF [report/secondary] ([arXiv 2510.18855](https://arxiv.org/abs/2510.18855)).
- **Ling-2.6-1T-base** "is the base checkpoint behind the Ling-2.6-1T and Ring-2.6-1T" [HF] ([card](https://huggingface.co/inclusionAI/Ling-2.6-1T-base)).
- **LongCat-Flash-Thinking** goes through "a curriculum learning strategy during mid-training … followed by a SFT stage", then large-scale RL [HF] ([card](https://huggingface.co/meituan-longcat/LongCat-Flash-Thinking)).
- **Magistral Small** is "Building upon Mistral Small 3.1 (2503) … undergoing SFT from Magistral Medium traces and RL on top". Its YAML `base_model` is the *Instruct* model, so the start is an unreleased SFT of a released instruct model [HF] ([card](https://huggingface.co/mistralai/Magistral-Small-2506)).
- **Llama 4** [report] ([Meta blog](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)):
  - The pipeline is "lightweight SFT > online RL > lightweight DPO".
  - The final stage is DPO, so Llama 4 is **not RL-final**.
- **DeepSeek-V4** [HF] ([V4 card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro); [V4.1 card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)):
  - Post-training is "independent cultivation of domain-specific experts (through SFT and RL with GRPO), followed by unified model consolidation via on-policy distillation".
  - V4.1-Flash follows "SFT → RL → on-policy distillation (OPD)".
  - So the final stage is OPD, not reward-RL.
- **MiMo-V2-Flash / V2.5 / V2.5-Pro** post-training "incorporates SFT, large-scale agentic RL, and Multi-Teacher On-Policy Distillation (MOPD)" [HF] ([card](https://huggingface.co/XiaomiMiMo/MiMo-V2.5-Pro)).
- **Kimi-VL-A3B-Thinking** has YAML `base_model: moonshotai/Kimi-VL-A3B-Instruct` and is built through "long CoT SFT and RL". The SFT checkpoint is not released [HF] ([card](https://huggingface.co/moonshotai/Kimi-VL-A3B-Thinking)).

### Inferences
- **INTELLECT-3 → INTELLECT-3.1 is the only RL-on-RL pair at ≥100B with a public exact start.**
  - It fits 2 H200s in BF16.
  - It is also the most open recipe found.
  - Caveat: INTELLECT-3 is itself an RL model. So 3.1-vs-3 measures *continued* RL, not SFT → RL.
- **Phi-4-reasoning → plus** is the cleanest SFT → RL industry pair. It is the closest analogue to the Olmo DPO-start → RL design, but short (90 GRPO steps) and math-only.
- **MiMo-7B** gives both SFT → RL and Base → RL-Zero on the same base.
- For DeepSeek-V4, MiMo-V2/V2.5, Qwen3 ≤30B and Llama 4, the final stage is distillation or DPO. These fail the "RL is final" constraint and are excluded.

### Gaps
- I did not confirm that `Nemotron-3-Labs-Ultra-Math-RL` was initialised from the released `-SFT` weights; the abstract is ambiguous.
- I did not confirm the MiMo-VL RL init in this session (the MiMo-VL report was not fetched).
- I did not verify that `deepseek-math-7b-rl` started from the released `-instruct`; the card does not say.
- The final-stage order for Kimi K2-Instruct, Seed-OSS, Hunyuan-A13B, ERNIE-4.5, Ministral-3-Reasoning, Mistral Large 3, Gemma 3/4 and Qwen3.5 small models is **not stated on their HF cards**. I did not fetch their reports. "RL is final" is unverified for them.
- The MiMo-V2-Flash and V2.5 cards do not unambiguously order agentic RL vs MOPD.

## Intermediate RL checkpoints

### Takeaway
**No industry release in scope publishes intermediate RL checkpoints**:
- none as HF branches or tags;
- none as separate step repos;
- none as a documented step series.

So **no release qualifies for grade A.** The nearest substitutes are a public metrics dashboard (MiMo-V2.6), a single selected early RL step (Phi-4-reasoning-plus at step 90), and successive RL generations (INTELLECT-3 → 3.1).

### Cited findings
- `/refs` for 190 public candidate repos plus 6 extra pairs showed only `main`, `pr/*` and three incidental branches: `MiniMax-M2: prepare-transformers`, `MiniMax-H3: switch-library-name-h3` and `Step-3.5-Flash-Base-Midtrain: update-config`. No tags exist [HF] ([example: MiMo-7B-RL refs](https://huggingface.co/api/models/XiaomiMiMo/MiMo-7B-RL/refs); [Phi-4-reasoning-plus refs](https://huggingface.co/api/models/microsoft/Phi-4-reasoning-plus/refs); [INTELLECT-3.1 refs](https://huggingface.co/api/models/PrimeIntellect/INTELLECT-3.1/refs); [MiMo-V2.6-Pro-RL refs](https://huggingface.co/api/models/XiaomiMiMo/MiMo-V2.6-Pro-RL/refs)).
- MiMo-V2.6 RL metrics were streamed live: reward curves, rollout counts, step timings, compute cost and mid-training coding scores. The runs started 2026-09-15 and the weights went up 2026-09-21 [secondary] ([RITS](https://rits.shanghai.nyu.edu/ai/xiaomi-open-sources-mimo-v2-6-pro-and-flash-plus-its-rl-stack/)).
- The report points to a running log at mimo.xiaomi.com/rl/mimo-v26 and says "We open-source the training dynamics". It does not release step checkpoints [report] ([arXiv 2610.11959](https://arxiv.org/html/2610.11959)).
- The Phi-4-reasoning-plus checkpoint is a chosen step (90) of a longer GRPO run (its figure covers 125 updates). Other steps are not released [report] ([arXiv 2504.21318](https://arxiv.org/html/2504.21318)).
- Several labs release *pretraining* or mid-training intermediates instead:
  - Ling-mini-base-2.0 at 5T/10T/15T/20T tokens [HF] ([card](https://huggingface.co/inclusionAI/Ling-mini-2.0));
  - Ling-3.0 `-base-30T` / `-base-midtrain` / `-base` [HF] ([card](https://huggingface.co/inclusionAI/Ling-3.0-flash-base));
  - Step-3.5-Flash-Base-Midtrain [HF] ([card](https://huggingface.co/stepfun-ai/Step-3.5-Flash-Base-Midtrain));
  - Trinity-Large-TrueBase / Base [HF] ([card](https://huggingface.co/arcee-ai/Trinity-Large-Base));
  - ERNIE-5.5-Flash-Base, "the checkpoint at the end of pre-training, before any mid- or post-training" [HF] ([card](https://huggingface.co/baidu/ERNIE-5.5-Flash-Base)).

### Inferences
- For intermediate-RL analyses (reach growth across RL steps), industry releases give at best **two or three points on one lineage**: Base → RL-Zero and SFT → RL (MiMo-7B), or INTELLECT-3 → 3.1. Run-internal steps remain an AllenAI-only resource (Olmo), plus NVIDIA's separately covered releases.

### Gaps
- I could not see refs for gated Gemma 3 and Llama 4.
- I did not check whether any lab hosts RL step checkpoints outside HF (ModelScope, or object storage linked from reports). Only MiMo-V2.6 mentions a dashboard, and it carries metrics, not weights.

## RL setup (algorithm and domains, as reported)

### Takeaway
GRPO-family methods dominate. Newer flagships do agentic, multi-environment RL, often followed by on-policy distillation.

| Model | Algorithm | Domains |
|---|---|---|
| Phi-4-reasoning-plus | GRPO | math only |
| MiMo-7B | rule-rewarded RL | math + code |
| INTELLECT-3.x | prime-rl | math, code, SWE, agentic |
| MiMo-V2.6 | asynchronous GRPO | code, general, visual, cyber, in one mixed run |
| Ring | Long-CoT SFT → RLVR → RLHF, with IcePop | reasoning, then alignment |
| MiniMax-M1 | CISPO | not recorded here |

### Cited findings
- **Phi-4-reasoning-plus** [report] ([arXiv 2504.21318](https://arxiv.org/html/2504.21318)):
  - The algorithm is GRPO.
  - The seed set is 72,401 math problems, of which ≈6K were used, with 8 responses per prompt.
  - The report says: "The RL training focused exclusively on mathematical reasoning".
- **MiMo-7B**: "130K mathematics and code problems as RL training data, which can be verified by rule-based verifiers … only rule-based accuracy rewards". It also uses a "test difficulty driven code reward" and easy-problem re-sampling [HF] ([card](https://huggingface.co/XiaomiMiMo/MiMo-7B-RL-Zero)).
- **INTELLECT-3.1** was trained with prime-rl using verifiers environments, on math, coding, SWE and agentic tasks [HF] ([card](https://huggingface.co/PrimeIntellect/INTELLECT-3.1)).
- **MiMo-V2.6** [HF] ([card](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-RL)):
  - It uses "One mixed RL run across coding, general agents, visual, and cybersecurity".
  - It uses "Fully asynchronous Group Relative Policy Optimization (GRPO) … 1,568 prompts × 16 rollouts per step".
  - Rewards use groupwise agentic grading (Groupwise Reward Synthesis).
  - The training uses "GRPO with asynchronous partial rollouts at a staleness of 4" [report] ([arXiv 2610.11959](https://arxiv.org/html/2610.11959)).
- **Ring-flash-2.0 and Ring-1T**: the pipeline is Long-CoT SFT → RLVR → RLHF, with IcePop correcting the training/inference mismatch in MoE RL [HF] ([Ring-flash-2.0 card](https://huggingface.co/inclusionAI/Ring-flash-2.0); [Ring-1T card](https://huggingface.co/inclusionAI/Ring-1T)).
- **GLM-4.5**: "Our overall RL algorithm builds upon the GRPO framework, excluding the KL loss term". Agentic RL covers web search and SWE [report] ([arXiv 2508.06471](https://arxiv.org/html/2508.06471)).
- **MiniMax-M1** uses CISPO, "a novel algorithm that clips importance sampling weights instead of token updates" [HF] ([card](https://huggingface.co/MiniMaxAI/MiniMax-M1-80k)).
- **LongCat-Flash-Thinking** uses an extended GRPO on the DORA asynchronous system, "domain-parallel training … across STEM, coding, and agentic tasks" and then merges the domain models [HF] ([card](https://huggingface.co/meituan-longcat/LongCat-Flash-Thinking)).
- **Qwen3-Next-80B-A3B-Thinking** uses GSPO for RL [HF] ([card](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Thinking)).
- **Kimi K3.1 Pro** uses "large-scale joint reinforcement learning across hundreds of downstream domains" [HF] ([card](https://huggingface.co/moonshotai/Kimi-K3.1-Pro)).
- **Nemotron-3-Ultra Math** is specialised for olympiad maths and proof checking. The card's YAML lists the datasets `Nemotron-Math-Proofs-v3-SFT` and `-RL` [HF] ([card](https://huggingface.co/nvidia/Nemotron-3-Labs-Ultra-Math-RL)).

### Inferences
- The pairs that suit capability-vs-elicitation analysis on **verifiable math and code** are Phi-4-reasoning-plus (math), MiMo-7B (math + code), INTELLECT-3.1 (math, code, SWE, agentic) and PaCoRe-8B (math and code, but with a parallel-reasoning inference protocol).
- Qwen3-4B-SafeRL is safety RL, so it is off-topic for reasoning reach.

### Gaps
- I recorded no algorithm detail for Seed-OSS, Hunyuan-A13B, ERNIE-4.5, Gemma 4, Mistral Large 3 or Ministral 3; their cards do not state it.

## Openness of RL data and reward code

### Takeaway
RL data and reward code are **not public for most flagships**. The exceptions are:
- **Prime Intellect** (SFT + RL datasets, environments and trainer);
- **Xiaomi MiMo-V2.6** (RL environments, Docker images, a verl fork);
- **inclusionAI Ring-lite** (partial RL data);
- **StepFun PaCoRe** (training data);
- **NVIDIA Nemotron Math** (data and code, per its abstract).

### Cited findings
- **Prime Intellect** has `INTELLECT-3-RL` and `INTELLECT-3-SFT` on HF [HF] ([INTELLECT-3-RL](https://huggingface.co/datasets/PrimeIntellect/INTELLECT-3-RL); [INTELLECT-3-SFT](https://huggingface.co/datasets/PrimeIntellect/INTELLECT-3-SFT)).
  - The RL configs are code (8,579), logic (11,647), math (21,161) and science (29,307 examples).
  - The SFT set includes, for example, `am_chat` with 952,046 examples.
- The INTELLECT-3 card points to "Environments released on Environments Hub", and training uses prime-rl and verifiers [HF] ([card](https://huggingface.co/PrimeIntellect/INTELLECT-3)).
- **Xiaomi MiMo-V2.6** published the dataset `MiMo-V2.6-RL-oss` (Apache-2.0), titled "Agentic RL Environments" [HF] ([dataset](https://huggingface.co/datasets/XiaomiMiMo/MiMo-V2.6-RL-oss)).
  - It covers code (executable tests), cyber (rule checks), general (rubric judging), webdev (visual grading) and music.
  - It links Docker images (`xiaomimimo/mimo-v2.6-rl-oss`) and training code (`github.com/XiaomiMiMo/verl`).
- The MiMo-V2.6 report lists ≈3k code, ≈1k cyber, ≈1k general and ≈2k visual environments [report] ([arXiv 2610.11959](https://arxiv.org/html/2610.11959)).
- The MiMo-7B RL data is not in the XiaomiMiMo HF dataset list [HF] ([datasets listing](https://huggingface.co/api/datasets?author=XiaomiMiMo)).
- **inclusionAI** published `Ring-lite-rl-data` ("Over 39,000" math and "Approximately 8,400" code problems). The card notes "Only a partial subset … is publicly released". It also published `Ring-lite-sft-data` [HF] ([dataset](https://huggingface.co/datasets/inclusionAI/Ring-lite-rl-data)).
- **StepFun** [HF] ([PaCoRe-Train-8k](https://huggingface.co/datasets/stepfun-ai/PaCoRe-Train-8k); [Step-3.5-Flash-SFT](https://huggingface.co/datasets/stepfun-ai/Step-3.5-Flash-SFT)):
  - `PaCoRe-Train-8k` (MIT) comes with "We open-source model checkpoints, training data, and the full inference pipeline".
  - `Step-3.5-Flash-SFT` is SFT data, not RL data, with a StepTronOSS reproduction script.
- **NVIDIA Nemotron Math** releases "the training data, the training and inference code" [report] ([arXiv 2609.10712](https://arxiv.org/abs/2609.10712)).
- **DeepSeek, Moonshot, Zhipu, Qwen, Meituan, MiniMax, ByteDance, Tencent, Baidu, Mistral, Google, Meta, OpenAI** link no RL dataset or reward code from the model cards I read [HF] (cards above).
  - GLM-5 links its RL *infrastructure*, `slime`, but no data [HF] ([card](https://huggingface.co/zai-org/GLM-5)).
  - MiniMax-M2.5 describes the Forge RL framework without releasing it [HF] ([card](https://huggingface.co/MiniMaxAI/MiniMax-M2.5)).

### Inferences
- A **reconstructable** RL start is possible where SFT data and a base are both public: INTELLECT-3 (GLM-4.5-Air-Base + INTELLECT-3-SFT) and Step-3.5-Flash (Base-Midtrain + Step-3.5-Flash-SFT).
- A reconstruction is not the exact released start. It would be a new SFT run whose outputs differ from the lab's.

### Gaps
- HF dataset listings were checked only for XiaomiMiMo, PrimeIntellect, stepfun-ai and inclusionAI. For the other orgs, "not public" rests on the model cards not linking any data.

## Tokenizer and template parity between released stages

### Takeaway
Within the B-grade pairs, **templates are identical** (or differ trivially), so per-prompt pass@k can use one prompt format. Watch the **EOS differences between base-line and SFT-line checkpoints** (MiMo-7B), and the template change in the Nemotron Math SFT.

### Cited findings
All comparisons are md5 hashes of the HF `tokenizer_config.json` `chat_template` or of `chat_template.jinja` [HF].
- **MiMo-7B** ([SFT config](https://huggingface.co/XiaomiMiMo/MiMo-7B-SFT/blob/main/tokenizer_config.json)):
  - Base, SFT, RL and RL-Zero share one template (`435dad37`).
  - EOS is `<|endoftext|>` for Base and RL-Zero and `<|im_end|>` for SFT and RL.
- **Phi-4-reasoning** and **-plus** have the same template (`2b0d2091`) and EOS `<|im_end|>` [HF] ([plus config](https://huggingface.co/microsoft/Phi-4-reasoning-plus/blob/main/tokenizer_config.json)).
- **INTELLECT-3** vs **3.1**: the `chat_template.jinja` files are the same size (7,053 B) and differ in one line, the default system prompt ("You are INTELLECT-3" vs "You are INTELLECT-4"). `INTELLECT-3-Base` has a different template [HF] ([3.1 template](https://huggingface.co/PrimeIntellect/INTELLECT-3.1/blob/main/chat_template.jinja)).
- **RLVR-8B-0926** and **PaCoRe-8B** have the same template (`44cd0f1b`) [HF] ([PaCoRe-8B](https://huggingface.co/stepfun-ai/PaCoRe-8B)).
- **Qwen3-4B** and **Qwen3-4B-SafeRL** have the same template (`ae518d20`) [HF] ([SafeRL](https://huggingface.co/Qwen/Qwen3-4B-SafeRL)).
- **DeepSeek**: R1-Zero and R1 share a template (`1f688cce`), while V3-Base differs (`c1136b69`) [HF] ([R1-Zero](https://huggingface.co/deepseek-ai/DeepSeek-R1-Zero)).
- **Prover-V1.5** SFT and RL share a template (`68394676`). **MiMo-VL-7B-SFT-2508** and **RL-2508** share one (`5dfbb06f`). **MiMo-V2.6-Flash-RL** and **-MOPD** share one (`cf0424e5`) [HF] ([MiMo-V2.6-Flash-RL](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL)).
- **Ling-flash-base-2.0** and **Ring-flash-2.0** share a 293-character template (`3599557b`) [HF] ([Ring-flash-2.0](https://huggingface.co/inclusionAI/Ring-flash-2.0)).
- **Nemotron-3-Labs-Ultra-Math-SFT** and **-RL** templates differ: the SFT template has a `low_effort` flag where the RL template has `medium_effort`. The RL template equals the general Nemotron-3-Ultra template (`fa104c9a`) [HF] ([RL](https://huggingface.co/nvidia/Nemotron-3-Labs-Ultra-Math-RL)).
- **Kimi K2-Instruct** (`67cbd838`) and **K2-Thinking** (`d57721ac`) templates differ. K3 and K3.1-Pro ship no `chat_template.jinja` [HF] ([K2-Thinking](https://huggingface.co/moonshotai/Kimi-K2-Thinking)).

### Inferences
- For Base → RL-Zero comparisons (MiMo-7B, DeepSeek), the base and RL-Zero share the base tokenizer and EOS. RL-Zero models were trained with a plain-text prompt rather than the chat template, so evaluate both with the same raw prompt.

### Gaps
- I did not compare `tokenizer.json` vocab files. The parity claims cover templates and EOS only, and same-family checkpoints share the vocab by construction.

## Practicalities: licence, dates, serving footprint, and fit

### Takeaway
The B-grade pairs that fit 1–2 H200s (141 GB each) are:
- MiMo-7B (1 GPU);
- Phi-4-reasoning → plus (1);
- RLVR-8B → PaCoRe-8B (1);
- Qwen3-4B → SafeRL (1);
- MiMo-VL-7B (1);
- INTELLECT-3 → 3.1 (2 × H200 in BF16).

The flagship C-grade pairs that fit 1–2 GPUs are Ring-mini-2.0 (1), Ring-flash-2.0 (2), GLM-4.5-Air (2), Seed-OSS-36B (1), Hunyuan-A13B (2), Magistral Small (1, start = instruct), Ministral-3-14B-Reasoning (1), Qwen3.5-35B-A3B (1) and Gemma 4 31B / 26B-A4B (1). For those Gemma, Qwen3.5, Seed, Hunyuan and Ministral entries, "RL final" is unverified.

Everything at 300B+ needs 4–16 H200s, which is about $15–170/hour at the quoted rates.

### Cited findings
- **On-disk sizes** [HF] ([MiMo-V2.6-Flash-RL](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL)), from HF `usedStorage`:

  | Repo | Size |
  |---|---|
  | `MiMo-V2.6-Flash-RL` | 177.8 GB |
  | `MiMo-V2.6-Pro-RL` | 573.5 GB |
  | `INTELLECT-3.1` | 213.7 GB |
  | `Nemotron-3-Labs-Ultra-Math-RL` | 1,121.1 GB |

- **Quantisation configs** from the HF config `quantization_config` [HF] ([org listings](https://huggingface.co/api/models?author=XiaomiMiMo)):
  - FP8: DeepSeek V3/R1, Kimi K2 Base/Instruct, MiniMax-M2.x, MiMo-V2.x, GLM-5.3 and Step-5-Preview.
  - Kimi-K2-Thinking: compressed-tensors (INT4).
  - gpt-oss: MXFP4.
  - INTELLECT-3: an FP8 repo exists (`INTELLECT-3-FP8`), but not for 3.1.
  - Phi-4-reasoning-plus: NVIDIA FP8 and NVFP4 conversions exist (`nvidia/Phi-4-reasoning-plus-FP8`).
- The Mistral Large 3 card says: "The Mistral Large 3 Instruct FP8 format can be used on one 8xH200 node" [HF] ([card](https://huggingface.co/mistralai/Mistral-Large-3-675B-Base-2512)).
- **Licences** [HF], from the org listings' `cardData`:
  - **MIT**: MiMo (all), Phi-4 family, INTELLECT-3/3.1, RLVR-8B/PaCoRe, DeepSeek R1/V3.1+/V4, GLM-4.5 through 5.2, inclusionAI 2.0+ (but `Ling-1T` card has no YAML licence), LongCat.
  - **Apache-2.0**: Qwen3/3.5/3.6/3.8-27B, Seed-OSS, ERNIE-4.5, Step-3.5-Flash, Magistral/Ministral/Large 3, Gemma 4, gpt-oss, Laguna, MiniMax-M1.
  - **"other"**: Kimi (K2 and K3 licences), Hunyuan, Nemotron (NVIDIA Open Model License), Llama 4, Trinity, MiniMax-M2+, GLM-5.3, Qwen3.8-2.4T and Flash-Next, Mistral Large 4 and Medium 3.5, ERNIE-5.5.
  - **Gemma licence**, gated: Gemma 3.
  - DeepSeek-V3-Base and V3 have no YAML licence. The card says "DeepSeek-V3 series (including Base and Chat) supports commercial use" under the Model License [HF] ([card](https://huggingface.co/deepseek-ai/DeepSeek-V3-Base)).
- Release dates in the table are the HF `createdAt` dates from the org listings [HF].

### Inferences
- The footprints in the table are my estimates, not measurements: weights plus KV headroom for 256–1,024-sample pass@k, on H200 = 141 GB.
- **Best new candidates for the Olmo-style test, in order:**
  1. **Phi-4-reasoning → Phi-4-reasoning-plus**: exact SFT start, 14B, 1 GPU, identical template. The caveat is that the RL is short and math-only.
  2. **MiMo-7B**: SFT → RL plus Base → RL-Zero; 7B; math and code.
  3. **INTELLECT-3 → 3.1**: continued RL at 106B-A12B on 2 H200, with fully open data and environments.
  4. **RLVR-8B → PaCoRe-8B**, keeping in mind that PaCoRe's gains are framed around a parallel-reasoning inference protocol.
- **MiMo-V2.6-Flash-RL** (2 × H200) is the best *flagship* RL-final model. It comes with public RL environments and code, but has no released start. Base-only reach comparisons would have to use MiMo-V2.5-Base as a non-exact proxy.

### Gaps
- Footprints are estimates. KV-cache needs depend on context length and on the hybrid or linear attention used by many 2026 models, and I did not benchmark them.
- I did not verify FP8 availability for every base checkpoint. Most bases are BF16-only even where the final has an FP8 repo (for example GLM-4.5-Air-Base, Hunyuan-A13B-Pretrain and Step-3.5-Flash-Base).

### Summary table

The **template parity** column covers chat-template parity between the released stages (md5). The **fit** grades are A, B, C or X:
- **A**: exact RL start + RL model + intermediate RL checkpoints.
- **B**: exact RL start + RL model.
- **C**: only base + RL model; the RL-Zero pair counts here.
- **X**: excluded, with the reason.

| model | total / active params | base released | exact RL start released | RL final | intermediate RL ckpts | RL-Zero variant | template parity | licence | serving footprint (H200) | fit |
|---|---|---|---|---|---|---|---|---|---|---|
| Xiaomi MiMo-7B (Base/SFT/RL/RL-Zero, 2025-04) | 7.8B dense | yes [HF] | **yes: MiMo-7B-SFT → RL** [HF card] | MiMo-7B-RL | none [HF refs] | **yes: MiMo-7B-RL-Zero (from Base)** | same template; EOS differs base-line vs SFT-line [HF] | MIT | 1 GPU | **B** |
| Microsoft Phi-4-reasoning → Phi-4-reasoning-plus (2025-04) | 14.7B dense | phi-4 [HF] | **yes: Phi-4-reasoning** [report] | plus (GRPO, 90 steps, math) | none [HF refs] | no | identical [HF] | MIT | 1 GPU | **B** |
| Prime Intellect INTELLECT-3 → INTELLECT-3.1 (2025-11 / 2026-01) | 106.9B / 12B | GLM-4.5-Air-Base (+ clone `INTELLECT-3-Base`) [HF] | **yes for 3.1: INTELLECT-3** [HF card]; INTELLECT-3's own SFT start not released | 3.1 (continued RL) | none [HF refs] | no | identical except system-prompt name [HF] | MIT | 2 GPU BF16 (213.7 GB) | **B** (RL-on-RL); open data and environments |
| StepFun RLVR-8B-0926 → PaCoRe-8B (2025-12) | 8.2B dense | Qwen3-8B-Base (Qwen) | **yes: RLVR-8B-0926** [HF card] | PaCoRe-8B | none [HF refs] | no | identical [HF] | MIT | 1 GPU (PaCoRe inference is very long) | **B** |
| Qwen3-4B → Qwen3-4B-SafeRL (2025-09) | 4.0B dense | Qwen3-4B-Base | **yes: Qwen3-4B** [HF YAML] | SafeRL (safety reward) | none [HF refs] | no | identical [HF] | Apache-2.0 | 1 GPU | **B** (safety-only RL) |
| Xiaomi MiMo-VL-7B SFT → RL (2025-05 / -2508) | 8.3B dense | no VL base on HF (LLM is MiMo-7B; config class Qwen2_5_VL) | SFT released; RL init from it claimed, not verified here | RL / RL-2508 | none [HF refs] | no | identical [HF] | MIT | 1 GPU | **B** (VLM; init unverified) |
| DeepSeek-Prover-V1.5 SFT → RL (2024-08) | 6.9B dense | Prover-V1.5-Base | **yes** (RL YAML `base_model` = SFT) [HF] | RL | none | no | identical [HF] | DeepSeek licence | 1 GPU | **B** (old, Lean-only) |
| NVIDIA Nemotron-3-Labs-Ultra-Math SFT → RL (2026-09) | 560.5B / 55B | Ultra-Base [HF] | SFT released; RL-from-SFT not confirmed [report] | Math-RL | none [HF refs] | no | differ (effort flags) [HF] | NVIDIA Open Model License | ≥16 GPU BF16 (1.12 TB) | **B?** (covered elsewhere) |
| DeepSeek V3-Base → R1-Zero / R1 (2025-01) | 671B / 37B | yes [HF] | R1: no (cold-start SFT unreleased); **R1-Zero: start = V3-Base** | R1 (RL) | none [HF refs] | **yes: R1-Zero** | R1-Zero = R1 ≠ V3-Base [HF] | V3-Base DeepSeek Model License; R1 MIT | 8 GPU FP8 (≈685 GB) | **C** (RL-Zero pair) |
| DeepSeek V3.1-Base → V3.1 / Terminus | 671B / 37B | yes [HF] | no | not stated on card | none | no | not compared | MIT | 8 GPU FP8 | **C?** (final stage unverified) |
| DeepSeek V3.2-Exp-Base → V3.2 / Speciale (2025-12) | 685B / 37B | yes [HF YAML] | no | yes (RL framework per card) | none | no | not compared | MIT | 8 GPU FP8 | **C** |
| DeepSeek V4-Pro / Flash (+ Base), V4.1-Flash (2026) | 1.6T / 49B; 284B / 13B | yes (V4) [HF] | no | **no: final = on-policy distillation** [HF card] | none | no | n/a | MIT | V4-Flash 2–4 GPU; V4-Pro 8+ | **X** (OPD final) |
| Kimi K2-Base → K2-Instruct / K2-Thinking (2025) | 1T / 32B | yes [HF] | no | Instruct (post-trained); Thinking (start not stated) | none [HF refs] | no | Instruct ≠ Thinking [HF] | other (Kimi licence) | 8–16 GPU FP8 (≈1.03 TB); Thinking INT4 | **C** |
| Kimi K3 → K3.1 Pro (2026-06 / 2026-11) | 2.8T / 104B | no base [HF] | **no**: post-trained from K3 but attention layers swapped [HF card] | K3.1 Pro (joint RL) | none | no | no jinja | Kimi K3 licence ("other") | multi-node (≥16) | **X** (not exact; too large) |
| GLM-4.5 / GLM-4.5-Air (+ Base) (2025-07) | 355B / 32B; 106B / 12B | yes [HF] | no | yes: general RL last [report] | none [HF refs] | no | not compared | MIT | Air: 2 GPU BF16; 4.5: 4 GPU FP8 / 8 BF16 | **C** |
| GLM-4.6 through 5.3 | 355B, then ≈754B | no [HF] | no | not stated | none | no | n/a | MIT / other | 8+ GPU | **X** (no base) |
| Qwen3-235B-A22B (+ 2507 Thinking) | 235B / 22B | no [HF] | no | yes (general RL last) [report] | none | no | n/a | Apache-2.0 | 4 GPU BF16 | **X** (no base) |
| Qwen3-30B-A3B / 8B / ≤14B (+ Base) | 30.5B / 3.3B, … | yes [HF] | no | **no: strong-to-weak distillation** [report] | none | no | n/a | Apache-2.0 | 1 GPU | **X** (distillation final) |
| Qwen3-Next-80B-A3B-Thinking | 81.3B / 3B | no [HF] | no | RL (GSPO) [HF card] | none | no | n/a | Apache-2.0 | 2 GPU | **X** (no base) |
| Qwen3.5-35B-A3B / 9B (+ Base) (2026-02) | 36B / 3B; 9.7B | yes [HF] | no | RL claimed in card highlights; final stage unverified | none [HF refs] | no | not compared | Apache-2.0 | 1 GPU | **C?** |
| Qwen3.5-397B, 3.6, 3.8 (incl. 2.4T-A95B) | 403B / 17B … 2.4T / 95B | no [HF] | no | — | none | no | n/a | Apache-2.0 / other | 8–40 GPU | **X** (no base) |
| inclusionAI Ling-mini-base-2.0(-20T) → Ring-mini-2.0 (2025-09) | 16.3B / 1.4B | yes (+5T/10T/15T/20T pretrain ckpts) [HF] | no (long-CoT SFT unreleased) | Ring (RLVR → RLHF) | none [HF refs] | no | not compared | MIT | 1 GPU | **C** |
| inclusionAI Ling-flash-base-2.0 → Ring-flash-2.0 (2025-09) | 102.9B / 6.1B | yes [HF] | no | Ring (RLVR → RLHF) | none [HF refs] | no | identical [HF] | MIT | 2 GPU BF16 | **C** |
| inclusionAI Ring-1T (2025-10) | 1T / 50B | **no** (Ling-1T-base not on HF) | no | yes | none | no | n/a | MIT | ≥16 GPU BF16 | **X** (no base) |
| inclusionAI Ling-2.6-1T-base → Ring-2.6-1T (2026) | 1.03T / ≈63B (63B is from the Ling-2.5-1T card) | yes (released after the finals) [HF] | no | yes (async RL, IcePop) | none [HF refs] | no | not compared | MIT | 8–16 GPU FP8 | **C** |
| inclusionAI Ring-lite (2025-06) | 16.8B / 2.75B | Ling-lite-base-1.5 / Ling-lite-1.5 [HF] | no | yes | none [HF refs] | no | not compared | MIT | 1 GPU | **C** (+ partial RL data public) |
| Meituan LongCat-Flash-Chat / Thinking / 2601, LongCat-2.0 | 560B / ≈27B; 1.6T / 48B | **no** [HF] | no | Thinking: RL | none [HF refs] | no | n/a | MIT | 8 GPU FP8 | **X** (no base) |
| MiniMax-M1 (2025-06) | 456B / 45.9B | Text-01 only (a post-trained model) | no | RL (CISPO) | none [HF refs] | no | not compared | Apache-2.0 | 8 GPU BF16 (tight) | **X/C-** (no true base) |
| MiniMax-M2 / M2.x / M3 | 229B / 10B; 427B / 23B | no [HF] | no | RL claimed | none | no | n/a | other | 2–8 GPU | **X** (no base) |
| ByteDance Seed-OSS-36B (Base, Base-woSyn, Instruct) (2025-08) | 36.2B dense | yes, two variants [HF] | no | not stated on card | none [HF refs] | no | not compared | Apache-2.0 | 1 GPU | **C?** |
| Tencent Hunyuan-A13B (Pretrain / Instruct) (2025-06) | 80.4B / 13B | yes [HF] | no | not stated on card | none [HF refs] | no | not compared | Tencent licence ("other") | 2 GPU BF16 (Instruct-FP8: 1) | **C?** |
| Baidu ERNIE-4.5-21B-A3B / 300B-A47B (+ Base), 21B-A3B-Thinking | 21.8B / 3B; 300B / 47B | yes [HF] | no | not stated on card | none [HF refs] | no | not compared | Apache-2.0 | 1 GPU; 8 GPU | **C?** |
| Baidu ERNIE-5.5-Flash-Base (2026-10) | 286B / 14B | yes | — | no post-trained model released | — | — | — | other | 3–4 GPU FP8 (≈286 GB) | **X** (no RL model) |
| StepFun Step-3.5-Flash (+ Base, Base-Midtrain, SFT data) (2026-02/03) | 199B / 11B | yes, two [HF] | no (but SFT data + trainer open) | RL claimed ("scalable RL framework") | none [HF refs] | no | not compared | Apache-2.0 | 4 GPU BF16 (final FP8: 2) | **C** |
| StepFun step3, Step-3.7, Step-5-Preview | 321B / 38B; 201B / 11B; 604B / 27B | no [HF] | no | — | none | no | n/a | Apache-2.0 / other | 4–8 GPU | **X** (no base) |
| Xiaomi MiMo-V2-Flash / V2.5 / V2.5-Pro (+ Base) | 310B / 15B; 1.02T / 42B | yes [HF] | no | MOPD is part of post-training; order not explicit on cards | none [HF refs] | no | not compared | MIT | 3–4 GPU FP8; 8 GPU | **X?** (final likely OPD) |
| Xiaomi MiMo-V2.6-Flash-RL / Pro-RL (+ MOPD) (2026-09) | 310B / 15B; 1.02T / 42B | **no V2.6 base** (V2.5 bases only) | no (SFT start unreleased) [report] | **RL** (MOPD is a separate later repo) | none (metrics dashboard only) [HF refs] | no | RL = MOPD [HF] | MIT | Flash 2 GPU (177.8 GB); Pro 8 GPU (573.5 GB) | **C-** (proxy base; open RL envs + code) |
| OpenAI gpt-oss-120b / 20b (2025-08) | 117B / 5.1B; 21B / 3.6B | **no** | no | — | none | no | n/a | Apache-2.0 | 1 GPU MXFP4 | **X** (no base) |
| Mistral Magistral Small 2506 / 2509 | 24B dense | Mistral-Small-3.1/3.2 *Instruct* (base also public) | no (SFT on Medium traces unreleased) | RL on top [HF card] | none [HF refs] | no | not compared | Apache-2.0 | 1 GPU | **C** (start ≈ instruct) |
| Mistral Ministral-3-14B Base / Instruct / Reasoning (2025-10) | 13.9B | yes [HF] | no | not stated on card | none [HF refs] | no | not compared | Apache-2.0 | 1 GPU | **C?** |
| Mistral Large 3 675B Base / Instruct (2025-11) | 675B / 41B | yes [HF] | no | not stated | none [HF refs] | no | not compared | Apache-2.0 | 8 GPU FP8 [HF card] | **C?** |
| Mistral Large 4, Medium 3.5, Small 4 (2026) | 1T / 52B; 128B; 119B / 6.5B | no [HF] | no | — | none | no | n/a | other / Apache-2.0 | 1–16 GPU | **X** (no base) |
| Google Gemma 4 (31B, 26B-A4B, 12B, E2B/E4B; pt + it) (2026) | 31.3B; 25.8B / 4B | yes [HF] | no | post-training not described on card | none [HF refs] | no | not compared | Apache-2.0 | 1 GPU | **C?** |
| Google Gemma 3 (pt + it) | 27B … | yes (gated) | no | not verified | refs not checkable (gated) | no | n/a | Gemma licence | 1 GPU | **C?** |
| Meta Llama 4 Scout / Maverick (base + instruct) | 109B / 17B; 402B / 17B | yes (gated) | no | **no: lightweight DPO last** [report] | refs not checkable | no | n/a | Llama 4 licence | 2 GPU; 8 GPU | **X** (DPO final) |
| Microsoft Phi-4-mini-reasoning | 3.8B | Phi-4-mini-instruct | no | not verified here | none | no | not compared | MIT | 1 GPU | **C?** |
| Arcee Trinity-Large (TrueBase / Base / Preview / Thinking) (2026) | 398.6B / 13B | yes, two [HF] | no (Preview is a lightly post-trained sibling, not the stated start) | Thinking ("agentic RL") | none [HF refs] | no | not compared | other | 8 GPU BF16 | **C** |
| Poolside Laguna-M.1 (+ base) (2026-06) | 225.8B / 23B | yes [HF] | no | "pre-training, post-training and RL stages" [HF card] | none [HF refs] | no | not compared | Apache-2.0 | 4 GPU BF16 (final FP8: 2) | **C** |
| Upstage Solar-Open2-250B | 250B / 15B | no | no | — | none | no | n/a | other | 4 GPU | **X** (no base) |
