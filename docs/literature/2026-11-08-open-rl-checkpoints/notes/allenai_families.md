---
written_on: 2026-11-08
written_by: agent:claude
---

# AllenAI (Ai2) model releases ≥ 7B with a public RL start, RL final model and intermediate RL checkpoints

Scope: every AllenAI / Ai2 release of at least 7B parameters whose final stage is online RL (RLVR, GRPO-family, PPO). This excludes the two families already in use: Olmo 3.1 Think 32B (and its DPO start) and Olmo 3 7B RL-Zero IF/Math plus Olmo 3.1 RL-Zero Math. Current as of 2026-11-08.

Method: I listed every `allenai` model repo through the Hugging Face API, which returned 983 repos, sorted by creation date, and also checked the `rl-research` org (DR Tulu). For each candidate I pulled `/api/models/{id}` (parameter count, `gated`, licence, `base_model`, `createdAt`), `/api/models/{id}/refs` (branches) and `/api/models/{id}/tree/{rev}` (LFS SHA-256 of weight shards, `tokenizer.json`, `chat_template.jinja`), plus the raw `README.md` and `tokenizer_config.json`. Papers were read from the arXiv HTML versions, and training scripts from `allenai/open-instruct@main`.

Evidence tags:
- **[HF]**: verified on Hugging Face in this session (refs, file hashes, metadata).
- **[card]**: a claim made only in model-card prose.
- **[paper]**: a claim from the technical report.
- **[script]**: from an open-instruct training script.

Hash comparisons use the first 8–12 hex characters of the LFS SHA-256 (weights, `tokenizer.json`) or of the SHA-1 of the chat-template string.

## Q1. Repos and sizes: which AllenAI releases ≥ 7B end in online RL?

### Takeaway
Outside the two families already in use, Ai2 has released these text LLMs ≥ 7B with RL as the final stage:
- **Olmo 3:** 7B Think, 7B Instruct, 3.1 32B Instruct, 3.0 32B Think, and RL-Zero Code/General/Mix plus 3.1 RL-Zero Code.
- **OLMo 2 Instruct:** 7B, 13B (three chained RLVR stages, each its own repo) and 32B.
- **Tülu 3:** 8B, 3.1 8B, 70B and 405B.
- **2026 Tmax terminal agents:** 9B, 27B and 8B, plus six 9B dataset ablations.
- **DR Tulu 8B:** in the `rl-research` org.
- **BAR 2x7B experts:** Math and Code.
- **Older Tülu 2.5 PPO models:** 13B and Llama-3 8B.

There is no Olmo 4, Olmo 3.2 model or Tülu 4 on HF. Olmo 3.2 exists only as tokenizer repos. The 2026 Olmo Hybrid 7B family stops at DPO/SFT.

### Cited Findings
- **Enumeration [HF]:** the API returned 983 `allenai` repos. The newest LLM-relevant 2026 repos are:
  - `tmax-*` (2026-06-17/21);
  - `qwen35-9b-*` (2026-06-18/19);
  - `BAR-*` (2026-04-19);
  - `Olmo-Hybrid-*` (2026-01-28 to 2026-02-28);
  - `SERA-*` (2026-01-27 / 02-03);
  - `AstaBrief_8B` (2026-02-09);
  - `EMO`/`StdMoE_*` pretraining research models (2026-04/05);
  - `Bwen-8B`, `Llama-3-Blama-8B`, `Bolmo-7B-Stage1` (2026-08-26).

  Olmo 3.2 appears only as `allenai/olmo-3.2-tokenizer-think-release` (2025-11-03) and `allenai/olmo-3.2-tokenizer-think-dev` (2026-02-03), both with 0 downloads. No repo named Olmo 4, Olmo-3.2-* model or Tülu 4 exists — [HF listing](https://huggingface.co/api/models?author=allenai&limit=1000&sort=createdAt&direction=-1)
- **Olmo 3 7B Think [HF]:**
  - base `allenai/Olmo-3-1025-7B` (7.30B);
  - SFT `allenai/Olmo-3-7B-Think-SFT`;
  - DPO `allenai/Olmo-3-7B-Think-DPO`;
  - RL `allenai/Olmo-3-7B-Think` (7.30B; `base_model` = Think-DPO).

  — [model API](https://huggingface.co/api/models/allenai/Olmo-3-7B-Think), [card](https://huggingface.co/allenai/Olmo-3-7B-Think)
- **Olmo 3 7B Instruct [HF]:** `allenai/Olmo-3-7B-Instruct-SFT` → `allenai/Olmo-3-7B-Instruct-DPO` → `allenai/Olmo-3-7B-Instruct` (7.30B; `base_model` = Instruct-DPO). Base: Olmo-3-1025-7B — [card](https://huggingface.co/allenai/Olmo-3-7B-Instruct)
- **Olmo 3.1 32B Instruct [HF]:** `allenai/Olmo-3.1-32B-Instruct-SFT` → `allenai/Olmo-3.1-32B-Instruct-DPO` → `allenai/Olmo-3.1-32B-Instruct` (32.23B; `base_model` = 3.1-Instruct-DPO). Base: `allenai/Olmo-3-1125-32B` — [card](https://huggingface.co/allenai/Olmo-3.1-32B-Instruct)
- **Olmo 3 32B Think, the 3.0 version [HF]:** `allenai/Olmo-3-32B-Think` (`base_model` = `allenai/Olmo-3-32B-Think-DPO`). The card sets `new_version: allenai/Olmo-3.1-32B-Think` — [card](https://huggingface.co/allenai/Olmo-3-32B-Think)
- **Olmo 3 RL-Zero variants not yet used [HF]:** all 7.30B, with `base_model` = `allenai/Olmo-3-1025-7B`:
  - `allenai/Olmo-3-7B-RL-Zero-Code`;
  - `allenai/Olmo-3-7B-RL-Zero-General`;
  - `allenai/Olmo-3-7B-RL-Zero-Mix` (created 2025-12-01);
  - `allenai/Olmo-3.1-7B-RL-Zero-Code` (2025-12-10).

  — [Mix card](https://huggingface.co/allenai/Olmo-3-7B-RL-Zero-Mix)
- **OLMo 2 [HF]:**
  - 7B: `allenai/OLMo-2-1124-7B` → `-7B-SFT` → `-7B-DPO` → `-7B-Instruct` (7.30B).
  - 13B: `allenai/OLMo-2-1124-13B` → `-13B-SFT` → `-13B-DPO` → `-13B-Instruct-RLVR1` → `-13B-Instruct-RLVR2` → `-13B-Instruct` (13.72B).
  - 32B: `allenai/OLMo-2-0325-32B` → `-32B-SFT` → `-32B-DPO` → `-32B-Instruct` (32.23B).
  - Reward models `OLMo-2-1124-7B-RM` and `-13B-RM` are also public.

  — [7B card](https://huggingface.co/allenai/OLMo-2-1124-7B-Instruct), [13B card](https://huggingface.co/allenai/OLMo-2-1124-13B-Instruct), [32B card](https://huggingface.co/allenai/OLMo-2-0325-32B-Instruct)
- **Tülu 3 [HF]:**
  - 8B (8.03B): `allenai/Llama-3.1-Tulu-3-8B-SFT` → `-8B-DPO` → `allenai/Llama-3.1-Tulu-3-8B`. A PPO value-init RM, `Llama-3.1-Tulu-3-8B-RM`, is also public.
  - 3.1 8B: `allenai/Llama-3.1-Tulu-3.1-8B` (`base_model` = Tulu-3-8B-DPO).
  - 70B (70.55B): `-70B-SFT` → `-70B-DPO` → `allenai/Llama-3.1-Tulu-3-70B`.
  - 405B (405.85B): `-405B-SFT` → `-405B-DPO` → `allenai/Llama-3.1-Tulu-3-405B`.

  — [8B card](https://huggingface.co/allenai/Llama-3.1-Tulu-3-8B), [3.1 card](https://huggingface.co/allenai/Llama-3.1-Tulu-3.1-8B), [405B card](https://huggingface.co/allenai/Llama-3.1-Tulu-3-405B)
- **Tmax, 2026 terminal agents [HF]:**
  - `allenai/tmax-9b` (8.95B; `base_model` Qwen/Qwen3.5-9B);
  - `allenai/tmax-27b` (26.9B BF16 parameters; the API's `safetensors.total` field reads 2.6M, a metadata glitch; `base_model` Qwen/Qwen3.6-27B);
  - `allenai/tmax-8b` (8.19B; `base_model` `allenai/tmax-sft-8b`, which is itself SFT from Qwen/Qwen3-8B);
  - six RL ablations `allenai/qwen35-9b-{openthoughts,terminaltraj,cli-gym,swesmith,termigen,endless}` (8.95B each);
  - `tmax-4b` (4.21B) and `tmax-2b` (1.88B), which are below the size cut-off.

  — [tmax-9b card](https://huggingface.co/allenai/tmax-9b), [tmax-27b card](https://huggingface.co/allenai/tmax-27b), [Tmax paper](https://arxiv.org/abs/2606.23321)
- **DR Tulu [HF]:**
  - `rl-research/DR-Tulu-SFT-8B` (8.19B, `base_model` Qwen/Qwen3-8B) → `rl-research/DR-Tulu-8B` (`base_model` DR-Tulu-SFT-8B);
  - ablation `rl-research/DR-Tulu-No-RLER-8B`;
  - snapshot `rl-research/DR-Tulu-8B-Step-1900`;
  - `rl-research/dr-tulu-shortform-rl-400step`, which has an empty card.

  The cards say "Ai2's Responsible Use Guidelines" — [DR-Tulu-8B card](https://huggingface.co/rl-research/DR-Tulu-8B), [paper](https://arxiv.org/abs/2511.19399)
- **BAR [HF]:**
  - `allenai/BAR-7B` (7.30B dense, the starting point);
  - `BAR-2x7B-Math-SFT` → `BAR-2x7B-Math` and `BAR-2x7B-Code-SFT` → `BAR-2x7B-Code` (11.63B two-expert MoEs);
  - `BAR-2x7B-Tool-Use` and `BAR-2x7B-Safety`, which are SFT only;
  - `BAR-5x7B` (24.61B, the final merge with a trained router).

  The card says "All BAR models are built on top of Olmo 2 7B" — [BAR card](https://huggingface.co/allenai/BAR-2x7B-Math), [paper](https://arxiv.org/abs/2604.18473)
- **Tülu 2.5 PPO (2024) [HF]:**
  - nine `allenai/tulu-v2.5-ppo-13b-*` variants (13.02B, `base_model` `allenai/tulu-2-13b`);
  - three `allenai/llama-3-tulu-v2.5-8b-*` PPO variants (8.03B, `base_model` `allenai/llama-3-tulu-2-8b`).

  — [card](https://huggingface.co/allenai/tulu-v2.5-ppo-13b-uf-mean-70b-uf-rm), [paper](https://arxiv.org/abs/2406.09279)
- **Multimodal RL releases [HF]:**
  - `allenai/SAGE-MM-{Qwen2.5-VL-7B,Qwen3-VL-8B,Molmo2-8B}-SFT` → `-SFT_RL` (8.29B, 8.77B and 8.66B);
  - `allenai/olmOCR-2-7B-1025` (8.29B), which the card says was "additionally fine tuned using GRPO RL training".

  — [SAGE card](https://huggingface.co/allenai/SAGE-MM-Qwen3-VL-8B-SFT_RL), [olmOCR-2 card](https://huggingface.co/allenai/olmOCR-2-7B-1025)
- **Below 7B [HF]:**
  - `allenai/OLMoE-1B-7B-0125-Instruct` (6.92B total, about 1B active), RLVR from `OLMoE-1B-7B-0125-DPO`, with 44 RL revisions `step_40`…`step_1760` (every 40);
  - `allenai/OLMo-2-0425-1B-RLVR1` / `-Instruct` (1.48B);
  - `tmax-4b` and `tmax-2b`.

  — [OLMoE refs](https://huggingface.co/api/models/allenai/OLMoE-1B-7B-0125-Instruct/refs)

### Inferences
- The richest new candidates for per-prompt reach tests on verifiable text tasks are:
  - Olmo 3 7B Think (same recipe as the 32B Think already in use, a fifth of the size);
  - Olmo 3 7B Instruct;
  - OLMo 2 13B and 32B;
  - Tülu 3.1 8B.
- Tmax and DR Tulu add agentic, non-math domains but need tool environments to evaluate.

### Gaps
- Private or unlisted repos cannot be seen through the API.
- I did not open the cards for the byte-level `Bwen-8B` / `Llama-3-Blama-8B` / `Bolmo-7B` repos beyond their names and `base_model`. They appear to be byteification conversions with no RL, but this was not verified card by card.

## Q2. The RL start: is the exact initialisation checkpoint released?

### Takeaway
- **Exact starts:** the RL start is released and named as the exact initialisation, either in a public training command or in HF `base_model` metadata, for:
  - Olmo 3 7B Think;
  - OLMo 2 32B;
  - Tülu 3.1 8B;
  - Tmax 8B;
  - Tmax 9B/27B, where the start weights are byte-identical to the Qwen release;
  - the RL-Zero runs;
  - DR Tulu.
- **Starts that are metadata-only or ambiguous:**
  - Olmo 3 7B Instruct and 3.1 32B Instruct (their public scripts point to internal checkpoints, and the paper says RL ran on two DPO candidates);
  - Tülu 3 8B (two DPO starts were tried);
  - OLMo 2 13B (a chain of three RL runs);
  - BAR (the paper implies, but does not state, that RL starts from the released SFT).

### Cited Findings
- **Olmo 3 7B Think [script]:** `scripts/train/olmo3/7b_think_rl.sh` has `--model_name_or_path allenai/Olmo-3-7B-Think-DPO` and `--dataset_mixer_list allenai/Dolci-Think-RL-7B 1.0`. The HF `base_model` is the same — [script](https://github.com/allenai/open-instruct/blob/main/scripts/train/olmo3/7b_think_rl.sh), [HF API](https://huggingface.co/api/models/allenai/Olmo-3-7B-Think)
- **Two 7B Think runs [paper]:** the 7B Think RL run first used older infrastructure (about 15 days) and was "later replicated … achieving similar performance in just 6 days". The open-instruct repo has both `7b_think_rl.sh` (PipelineRL) and `7b_think_rl_no_pipeline.sh` — [Olmo 3 report](https://arxiv.org/abs/2512.13961), [script dir](https://github.com/allenai/open-instruct/tree/main/scripts/train/olmo3)
- **Olmo 3 Instruct starts [paper]:** "we apply RL on top of two DPO candidates", and the final RL checkpoint was chosen by average score, length analysis and a "vibe test" — [Olmo 3 report §5.4](https://arxiv.org/abs/2512.13961)
- **Olmo 3 7B Instruct script [script]:** `7b_instruct_rl.sh` sets `model_name_or_path` to an internal path, `/weka/oe-adapt-default/scottg/olmo/merging/ckpts/olmo3-instruct-dpo-1116-vibes/...`, with the comment "nov 16 tentative final checkpoint row 33 # was replace by row 27". The HF `base_model` is `allenai/Olmo-3-7B-Instruct-DPO` — [script](https://github.com/allenai/open-instruct/blob/main/scripts/train/olmo3/7b_instruct_rl.sh)
- **Olmo 3.1 32B Instruct script [script]:** `32b_instruct_rl.sh` sets `model_path=/weka/.../olmo3-32b-DPO-1116-match-7b-1e-6__42__1763403219` (header `Commit: 8d8232cc`). The HF `base_model` is `allenai/Olmo-3.1-32B-Instruct-DPO` — [script](https://github.com/allenai/open-instruct/blob/main/scripts/train/olmo3/32b_instruct_rl.sh)
- **Olmo 3 32B Think script [script]:** the public `32b_think_rl.sh` points to an internal SFT merge path (`olmo3-merge-32b-1e-4-5e-5`) and is named `test_olmo3_32b_rl_run`. It is therefore not the exact production config, even though the HF `base_model` is Think-DPO — [script](https://github.com/allenai/open-instruct/blob/main/scripts/train/olmo3/32b_think_rl.sh)
- **RL-Zero starts [script]:** `7b_rlzero_mix.sh` uses `MODEL_NAME_OR_PATH="allenai/Olmo-3-1025-7B"`. `7b_rlzero_general.sh` uses an internal long-context checkpoint `.../olmo25_7b_lc_64k_6T_M100B_round5-.../step11921-hf` — [mix script](https://github.com/allenai/open-instruct/blob/main/scripts/train/olmo3/7b_rlzero_mix.sh), [general script](https://github.com/allenai/open-instruct/blob/main/scripts/train/olmo3/7b_rlzero_general.sh)
- **OLMo 2 32B [card]:** the published GRPO command has `--model_name_or_path allenai/OLMo-2-0325-32B-DPO` — [card](https://huggingface.co/allenai/OLMo-2-0325-32B-Instruct)
- **OLMo 2 7B/13B [paper]:** RLVR was applied "to the highest-performing 7B and 13B DPO checkpoints".
  - The 13B then had two more RLVR passes: RLVR2 on GSM8K, and the final model on MATH.
  - "the value function was re-initialized from the reward model in each RLVR run".

  — [OLMo 2 paper](https://arxiv.org/abs/2501.00656)
- **OLMo 2 13B chain [HF]:**
  - RLVR2 `base_model` = `OLMo-2-1124-13B-Instruct-RLVR1`;
  - Instruct `base_model` = `-RLVR2`;
  - but RLVR1 `base_model` wrongly reads `allenai/OLMo-2-1124-7B-DPO`, a 7B repo, which is a card error.

  — [RLVR1 API](https://huggingface.co/api/models/allenai/OLMo-2-1124-13B-Instruct-RLVR1)
- **Tülu 3 8B [paper]:** "at 8B scale, we tested the best overall DPO model and the best model with IFEval persona data mixed in". The value model was initialised from an RM trained from Tülu 3 SFT. The HF `base_model` for `Llama-3.1-Tulu-3-8B` is `-8B-DPO` — [Tülu 3 paper](https://arxiv.org/abs/2411.15124)
- **Tülu 3.1 8B [card]:** the GRPO command uses `--model_name_or_path allenai/Llama-3.1-Tulu-3-8B-DPO` — [card](https://huggingface.co/allenai/Llama-3.1-Tulu-3.1-8B)
- **Tülu 3 405B [paper]:** RLVR started from the 405B DPO model and used an 8B value model — [Tülu 3 paper](https://arxiv.org/abs/2411.15124)
- **Tmax 9B and 27B starts [HF]:** the hyperparameter block in `tmax-9b` names `hamishivi/Qwen3.5-9B`. Its weight shards have identical LFS SHAs to `Qwen/Qwen3.5-9B`. The same holds for `hamishivi/Qwen3.6-27B` versus `Qwen/Qwen3.6-27B`.
  - Only the chat template differs between these copies.
  - The paper says: "When training Qwen 3.5 and 3.6 models, we do not perform any initial SFT".

  — [tmax-9b card](https://huggingface.co/allenai/tmax-9b), [Tmax paper](https://arxiv.org/abs/2606.23321)
- **Tmax 8B start [card]:** "base model: allenai/tmax-sft-8b" — [card](https://huggingface.co/allenai/tmax-8b)
- **DR Tulu start [card]/[paper]:** "trained on top of rl-research/DR-Tulu-SFT-8B". The paper says only that RL follows the SFT cold start — [card](https://huggingface.co/rl-research/DR-Tulu-8B), [paper](https://arxiv.org/html/2511.19399)
- **BAR start [paper]:** each expert goes mid-training → SFT → RLVR.
  - "During RLVR, all shared parameters are unfrozen", while the anchor FFN stays frozen.
  - The paper does not state that RL starts from the released `*-SFT` repo.
  - None of the BAR repos set `base_model`.

  — [BAR paper](https://arxiv.org/html/2604.18473), [BAR HF API](https://huggingface.co/api/models/allenai/BAR-2x7B-Math)

### Inferences
- For Olmo 3 7B Instruct and 3.1 32B Instruct, the released DPO repo is the probable RL start (HF lineage says so), but it cannot be confirmed by hashing because no `step_0` revision exists. Treat it as "claimed exact".
- Tülu 3 8B has the same status. Tülu 3.1 8B is cleaner, because the exact command names the public DPO repo.
- For OLMo 2 13B, the immediate start of the final model is RLVR2. Reach can be measured against the DPO repo (the start of the whole RL chain) or against each stage's start.

### Gaps
- It is unknown which of the two 7B Think runs (old or new infrastructure) produced the released `Olmo-3-7B-Think`.
- The open-instruct commits for the Tülu 3 PPO runs and the OLMo 2 7B/13B runs were not pinned. Their scripts (`ppo_vllm_thread_ray_gtrl.py`, `grpo_vllm_thread_ray_gtrl.py`) return 404 on `main`.

## Q3. Intermediate RL checkpoints: how many, at which steps, and where?

### Takeaway
Intermediate RL checkpoints are exposed as HF branches named `step_*` on:
- Olmo 3 7B Think (55), 7B Instruct (8), 32B Think 3.0 (15), and every RL-Zero repo;
- OLMo 2 7B (6), 13B (13 per stage across 3 stages) and 32B (17);
- Tülu 3 8B (10), Tülu 3.1 8B (61), 70B (1) and 405B (2, plus the final step);
- Tmax 9B/27B/8B (5 each);
- DR Tulu 8B (80).

Olmo 3.1 32B Instruct, BAR, the Tmax ablations, Tülu 2.5 PPO and SAGE-MM expose `main` only.

### Cited Findings
- **Olmo 3 7B Think [HF]:** 55 revisions, `step_0025` … `step_1375` (every 25), plus `main`. `main` differs from `step_1375` (the first weight-shard SHA differs). The paper's Table 49 gives 1,400 training steps — [refs](https://huggingface.co/api/models/allenai/Olmo-3-7B-Think/refs), [Olmo 3 report](https://arxiv.org/abs/2512.13961)
- **Olmo 3 7B Instruct [HF]:** 8 revisions, `step_050` … `step_400` (every 50), plus `main`. `main` ≠ `step_400`. Table 49 gives 450 steps — [refs](https://huggingface.co/api/models/allenai/Olmo-3-7B-Instruct/refs)
- **Olmo 3.1 32B Instruct [HF]:** `main` only. The card says "due to a checkpointing issue, we only are releasing the final few checkpoints", but the refs show none — [refs](https://huggingface.co/api/models/allenai/Olmo-3.1-32B-Instruct/refs), [card](https://huggingface.co/allenai/Olmo-3.1-32B-Instruct)
- **Olmo 3 32B Think, 3.0 version [HF]:** 15 revisions, `step_050` … `step_750` (every 50). The paper says it "was trained for 750 steps", and 3.1 continued the run to 2,300 — [refs](https://huggingface.co/api/models/allenai/Olmo-3-32B-Think/refs), [Olmo 3 report fn. 30](https://arxiv.org/abs/2512.13961)
- **RL-Zero repos not yet used [HF]:**
  - Code: 29 revisions, `step_100` … `step_2900` (every 100).
  - General: 8, `step_100` … `step_800`.
  - Mix: 19, `step_50` … `step_950` (every 50). `main` ≠ `step_950`.
  - 3.1 Code: 33 irregular revisions, `step_050` … `step_1500` every 50 except 750 and 950, then 1600, 1700, 1750, 1900 and 1950.

  — [Code refs](https://huggingface.co/api/models/allenai/Olmo-3-7B-RL-Zero-Code/refs), [Mix refs](https://huggingface.co/api/models/allenai/Olmo-3-7B-RL-Zero-Mix/refs), [3.1 Code refs](https://huggingface.co/api/models/allenai/Olmo-3.1-7B-RL-Zero-Code/refs)
- **OLMo 2 7B Instruct [HF]:** 6 revisions, `step_60` … `step_360` (every 60). `main` ≠ `step_360` — [refs](https://huggingface.co/api/models/allenai/OLMo-2-1124-7B-Instruct/refs)
- **OLMo 2 13B [HF]:** RLVR1, RLVR2 and Instruct each have 13 revisions, `step_60` … `step_780` (every 60). The same branch names hold different weights in each repo; first-shard SHAs at `step_60`, `step_780` and `main` are all distinct. That gives about 39 intermediate checkpoints across the 3-stage chain — [RLVR1 refs](https://huggingface.co/api/models/allenai/OLMo-2-1124-13B-Instruct-RLVR1/refs)
- **OLMo 2 32B [HF]:** 17 revisions, `step_20` … `step_340` (every 20). `main` has the same weights as `step_320`, which matches the card's "we took step 320 as the final checkpoint, corresponding to episode 573,440" — [refs](https://huggingface.co/api/models/allenai/OLMo-2-0325-32B-Instruct/refs), [card](https://huggingface.co/allenai/OLMo-2-0325-32B-Instruct)
- **Tülu 3 8B [HF]:** 10 revisions, `step_100` … `step_1000` (every 100). `main` ≠ `step_1000`. The paper says checkpoints were evaluated "every 100 training steps (40 for 70B)" — [refs](https://huggingface.co/api/models/allenai/Llama-3.1-Tulu-3-8B/refs)
- **Tülu 3.1 8B [HF]:** 61 revisions, `step_40` … `step_2440` (every 40). `main` has the same weights as `step_1920`, which matches the card ("step 1920 … episode 1,474,560") — [refs](https://huggingface.co/api/models/allenai/Llama-3.1-Tulu-3.1-8B/refs)
- **Tülu 3 70B [HF]:** a single intermediate revision, `step_120` — [refs](https://huggingface.co/api/models/allenai/Llama-3.1-Tulu-3-70B/refs)
- **Tülu 3 405B [HF]:** `step_25`, `step_50` and `step_75`. `main` has the same weights as `step_75`. The paper says "we only trained for 75 steps" — [refs](https://huggingface.co/api/models/allenai/Llama-3.1-Tulu-3-405B/refs)
- **Tmax [HF]:**
  - `tmax-9b`: `step_100` … `step_500` (every 100); `main` = `step_200`.
  - `tmax-27b`: `step_100`, `160`, `200`, `240`, `300`; `main` = `step_160`.
  - `tmax-8b`: `step_100` … `step_500`; `main` = `step_300`.
  - `tmax-9b` also ships a `rollouts/` folder; the card says it holds "all model rollouts and logprobs over the course of training".
  - The six `qwen35-9b-*` ablations have `main` only, although their cards say "we provide model checkpoints as branches".

  — [tmax-9b refs](https://huggingface.co/api/models/allenai/tmax-9b/refs), [tmax-9b tree](https://huggingface.co/api/models/allenai/tmax-9b/tree/main), [qwen35-9b-openthoughts refs](https://huggingface.co/api/models/allenai/qwen35-9b-openthoughts/refs)
- **DR Tulu [HF]:**
  - `DR-Tulu-8B` has 80 revisions, `step_50` … `step_4000` (every 50, no gaps). The card says the "Final release is step_4000 (main and branch)" and that the earlier releases were `step_1000` and `step_1900`.
  - `DR-Tulu-8B-Step-1900` has the same weights as the `step_1900` branch.
  - `DR-Tulu-No-RLER-8B` has 23 revisions, `step_50` … `step_1150`.

  — [refs](https://huggingface.co/api/models/rl-research/DR-Tulu-8B/refs), [card](https://huggingface.co/rl-research/DR-Tulu-8B)
- **`main` only [HF]:** all BAR repos, `tulu-v2.5-ppo-13b-*`, `llama-3-tulu-v2.5-8b-*`, the SAGE-MM SFT_RL repos and `olmOCR-2-7B-1025` — [BAR refs](https://huggingface.co/api/models/allenai/BAR-2x7B-Math/refs)

### Inferences
- For dense RL trajectories at ≥ 7B on verifiable tasks, the best new options are:
  - Olmo 3 7B Think (every 25 steps over 1,400);
  - Tülu 3.1 8B (every 40 steps over 2,440);
  - DR Tulu (every 50 over 4,000, but with LLM-judge rewards).
- OLMo 2 13B is unusual: it has three back-to-back RL stages, each fully checkpointed.

### Gaps
- The step value of `main` is not stated on HF for Olmo 3 7B Think, 7B Instruct, Tülu 3 8B or OLMo 2 7B/13B. The Olmo 3 values (1,400 and 450) are inferred from the paper's Table 49.

## Q4. RL setup: algorithm, rewards, domains, steps and compute

### Takeaway
- **Olmo 3 (OlmoRL):** GRPO with no KL loss, clip-higher, truncated importance sampling, token-level loss and active sampling. Rewards are verifiable math/code/IF signals plus a Qwen3-32B judge for chat.
- **OLMo 2 7B/13B and Tülu 3:** PPO with a value model initialised from an RM.
- **OLMo 2 32B and Tülu 3.1:** GRPO with KL β = 0.01.
- **Tmax:** DPPO on Docker terminal tasks with unit-test verifiers.
- **DR Tulu:** GRPO with evolving rubrics judged by GPT-4.1-mini.
- **BAR:** GRPO RLVR on the Olmo 3 math/code data.
- **Tülu 2.5:** PPO against learned RMs.

### Cited Findings
- **OlmoRL algorithm [paper]:** "builds on Group Relative Policy Optimization (GRPO)". It adds:
  - zero-gradient filtering;
  - active sampling;
  - token-level loss;
  - "No KL loss";
  - clip-higher;
  - truncated importance sampling;
  - no standard-deviation normalisation of advantages.

  — [Olmo 3 report §4.4.1](https://arxiv.org/abs/2512.13961)
- **Olmo 3 verifiers [paper]:**
  - Math: SymPy-based comparison with a reference answer, binary reward.
  - Code: test cases run on AWS Lambda; the reward is either the pass fraction or all-pass binary.
  - Instruction following: constraint-checking functions; reward 1 only if all constraints are satisfied.
  - Chat: LM judge (Qwen3 32B, thinking off), with or without a reference answer.

  — [Olmo 3 report §4.4.1](https://arxiv.org/abs/2512.13961)
- **Olmo 3 Table 49 hyperparameters [paper]:**

  | run | prompts | LR | steps | unique prompts/batch × group size | max response | learner + actor GPUs |
  |---|---|---|---|---|---|---|
  | 7B Think | 104,869 | 1e-6 | 1,400 | 64 × 8 | 32,768 | 16 + 56 |
  | 32B Think | 104,869 | 2e-6 | 750 | 128 × 8 | 32,768 | 64 + 160 |
  | 7B Instruct | 171,950 | 1e-6 | 450 | 64 × 8 | 8,192 | 8 + 56 |
  | 7B RL-Zero (math) | 13,314 | 1e-6 | 2,000 | 32 × 8 | 16,384 | 8 + 64 |

  Clip bounds are 0.2 / 0.272 for all runs. The 7B Instruct run uses 4 mini-batches; the others use 1.

  — [Olmo 3 report Table 49](https://arxiv.org/abs/2512.13961)
- **Olmo 3 RL data filtering [paper]:**
  - Before RL, the authors "generate eight rollouts for each prompt from the initial checkpoint (e.g., … the DPO checkpoint)" and remove prompts with pass rate > 62.5%. This offline filtering was used for 7B Think.
  - The 32B run "rel[ies] on active sampling … re-using the 7B DPO-filtered data".
  - Instruct RL skips offline difficulty filtering and uses "less challenging datasets in the math and code domains".

  — [Olmo 3 report §4.4.2, §5.4](https://arxiv.org/abs/2512.13961)
- **Olmo 3 Instruct response lengths [paper]:** "maximum response length of 8K tokens for 7B and 16K for 32B" — [Olmo 3 report §5.4.1](https://arxiv.org/abs/2512.13961)
- **Olmo 3 scripts [script]:** the 7B Think script uses `--beta 0.0`, `--num_samples_per_prompt_rollout 8`, `--response_length 32768`, `--llm_judge_model hosted_vllm/Qwen/Qwen3-32B` and `--chat_template_name olmo_thinker`. The 3.1 32B Instruct script uses `--learning_rate 2e-6`, `--response_length 16384`, 12 nodes × 8 GPUs and `open_instruct/grpo_fast.py`.
  - The Instruct data mix is a set of `hamishivi/*` filtered math/code sets (AceCoder, OMEGA, ORZ, Polaris, MathSub, DAPO), `allenai/IF_multi_constraints_upto5_*` and an `allenai/rlvr…` general set.

  — [7B Think script](https://github.com/allenai/open-instruct/blob/main/scripts/train/olmo3/7b_think_rl.sh), [32B Instruct script](https://github.com/allenai/open-instruct/blob/main/scripts/train/olmo3/32b_instruct_rl.sh)
- **Olmo 3 compute [paper]:** "The final RL runs for the initial Olmo 3 Think 32B spanned approximately 5 days … we continued our best RL run for another 21 days on 224 GPUs to produce Olmo 3.1 Think 32B" — [Olmo 3 report](https://arxiv.org/abs/2512.13961)
- **OLMo 2 7B/13B [paper]:** PPO with "PPO's value function [initialised] from the corresponding RMs". The data was GSM8K, MATH and constraint prompts, and the 13B got extra GSM8K-only and MATH-only passes.
  - The cards list the datasets as: 7B `allenai/RLVR-GSM`; 13B RLVR1 `allenai/RLVR-GSM-MATH-IF-Mixed-Constraints`; 13B RLVR2 `allenai/RLVR-GSM`; 13B final `allenai/RLVR-MATH` (YAML).
  - The 13B final card's prose says RLVR-GSM, which conflicts with its YAML. The paper's description of the 7B run (combined mix) also conflicts with the 7B card (RLVR-GSM).

  — [OLMo 2 paper](https://arxiv.org/abs/2501.00656), [13B card](https://huggingface.co/allenai/OLMo-2-1124-13B-Instruct)
- **OLMo 2 32B [card]:** `grpo_vllm_thread_ray_gtrl.py` with:
  - `--beta 0.01`, `--learning_rate 5e-7`, `--number_samples_per_prompt 16`;
  - `--response_length 2048`, `--chat_template_name tulu`;
  - data `RLVR-GSM-MATH-IF-Mixed-Constraints`;
  - "trained using 5 8xH100 nodes".

  — [card](https://huggingface.co/allenai/OLMo-2-0325-32B-Instruct)
- **Tülu 3 PPO [card]:**

  | model | LR | β | effective batch | episodes (card) | data |
  |---|---|---|---|---|---|
  | 8B | 3e-7 | 0.05 | 224 | 100,000 | RLVR-GSM-MATH-IF-Mixed-Constraints |
  | 70B | 1e-7 | 0.07 | 640 | 100,000 | RLVR-GSM-MATH-IF-Mixed-Constraints |
  | 405B | 1e-7 | 0.05 | 1,856 | 300,000 | RLVR-MATH |

  All three use K = 4 PPO updates, response length 2,048 and a −10 penalty when no EOS is produced.

  — [8B card](https://huggingface.co/allenai/Llama-3.1-Tulu-3-8B), [70B card](https://huggingface.co/allenai/Llama-3.1-Tulu-3-70B), [405B card](https://huggingface.co/allenai/Llama-3.1-Tulu-3-405B)
- **Tülu 3 70B card vs paper [paper]:** the paper's 70B text says "400,000 episodes … β = 0.7", contradicting the card's 100,000 and 0.07 — [Tülu 3 paper](https://arxiv.org/abs/2411.15124)
- **Tülu 3 405B [paper]:** "for Tülu 3 405B RLVR we only used the MATH train set … we only trained for 75 steps". Training used 240 GPUs plus a 16-way tensor-parallel vLLM engine, with "an 8B value model" — [Tülu 3 paper](https://arxiv.org/abs/2411.15124)
- **Tülu 3.1 8B [card]:** "We switched from PPO to GRPO (no reward model)", with LR 5e-7, β 0.01, effective batch 768 (48 × 16) and response length 2,048 — [card](https://huggingface.co/allenai/Llama-3.1-Tulu-3.1-8B)
- **Tmax algorithm [paper]:** "DPPO … a variant of GRPO that masks tokens when inference and training logprobs deviate", with a token-level loss and fully asynchronous training on open-instruct. Training runs 500 steps, "group size of 32 and 8 prompts per batch, … 65536 tokens and 64 maximum tool calls".
- **Tmax tasks [paper]:** each task is a Dockerfile plus a unit-test verifier. Verifiers are graded (metric-threshold, fuzz-equivalence, etc.), and an LM judge was used to repair weak verifiers in the data. The cards give KL β 0.0 and LR 1e-6 — [Tmax paper](https://arxiv.org/html/2606.23321), [tmax-8b card](https://huggingface.co/allenai/tmax-8b)
- **DR Tulu [paper]:**
  - GRPO "customized variant" on open-instruct;
  - rubric reward with "GPT-4.1-mini … as the LM judge, and GPT-4.1 as the rubric generator", plus format, search and citation rewards;
  - prompts: about 5K from SearchArena/OpenScholar and about 4K from RaR;
  - "final run using 27,000 GPU hours" on 2 H100 nodes.

  — [DR Tulu paper](https://arxiv.org/html/2511.19399)
- **BAR [paper]:**
  - GRPO with verifiable rewards;
  - 8 samples/prompt, 64 prompts/rollout, LR 6e-7 (constant), response length 4,096, 2 × 8 H100;
  - math data from the Olmo 3 math RLVR pool (ORZ, DAPO-Math, AceReason-Math, OMEGA);
  - code v2 from "OLMo 3 code RLVR data … with execution-based verification";
  - RL step counts are not given.

  — [BAR paper](https://arxiv.org/html/2604.18473)
- **Tülu 2.5 [card]:** PPO from Tulu 2 13B SFT using a learned 13B or 70B UltraFeedback RM. Only the prompts from `allenai/tulu-2.5-preference-data` were used. The RMs and the PPO value models (`*-value` repos) are released — [card](https://huggingface.co/allenai/tulu-v2.5-ppo-13b-uf-mean-70b-uf-rm)

### Inferences
- Olmo 3's offline filtering removes prompts the DPO start already solves at > 62.5% from 8 samples. The RL training prompts are therefore pre-selected to be hard for the start. That matters if training prompts are reused as reach probes.
- Chat-domain rewards in Olmo 3, DR Tulu's rubric judge, and Tülu 2.5's RM rewards are not verifiable. Reach on those domains needs a judge, not a checker.

### Gaps
- No per-run GPU-hour totals were found for Olmo 3 7B Instruct or 3.1 32B Instruct. No RL step counts were found for BAR.

## Q5. Openness: is the RL prompt data public, and is the reward/verifier code public?

### Takeaway
RL prompt datasets are public on HF for every candidate except BAR, which has no dedicated dataset repo. Verifier code is public in `allenai/open-instruct` (`ground_truth_utils.py`, `IFEvalG/`, `if_functions.py`) for Olmo/OLMo/Tülu, and in `hamishivi/tmax` and `rlresearch/dr-tulu` for Tmax and DR Tulu. DR Tulu's reward depends on closed OpenAI judges.

### Cited Findings
- **Datasets that resolve (HTTP 200) [HF]:**
  - `allenai/Dolci-Think-RL-7B` (2025-11-18), `allenai/Dolci-Think-RL-32B` (2025-11-20), `allenai/Dolci-Instruct-RL` (2025-11-19);
  - `allenai/Dolci-RL-Zero-{Code,Math}-7B` (odc-by) and `allenai/Dolci-RL-Zero-{Mix,IF,General}-7B` (resolve; licence not checked);
  - `allenai/RLVR-GSM` (MIT), `allenai/RLVR-MATH`, `allenai/RLVR-GSM-MATH-IF-Mixed-Constraints` (odc-by);
  - `allenai/tmax-15k-open-instruct`, `allenai/TMax-15K`, `allenai/tmax-sft` (odc-by, 2026-06);
  - `rl-research/dr-tulu-rl-data` and `rl-research/dr-tulu-sft-data` (odc-by);
  - `allenai/tulu-2.5-preference-data`, `allenai/SAGE-MM-RL-7k`, `allenai/olmOCR-synthmix-1025`.

  None are gated — [Dolci-Think-RL-7B](https://huggingface.co/datasets/allenai/Dolci-Think-RL-7B), [RLVR-GSM-MATH-IF](https://huggingface.co/datasets/allenai/RLVR-GSM-MATH-IF-Mixed-Constraints), [tmax-15k](https://huggingface.co/datasets/allenai/tmax-15k-open-instruct), [dr-tulu-rl-data](https://huggingface.co/datasets/rl-research/dr-tulu-rl-data)
- **Dataset IDs in cards that do not resolve [HF]:** these return HTTP 307 rather than 200: `allenai/Dolci-Instruct-RL-7B` (7B Instruct card), `allenai/Dolci-Think-RL` (32B Think card), and `allenai/Dolci-RLZero-{Mix,General,Code}-7B` (RL-Zero cards, without the hyphen). The hyphenated `Dolci-RL-Zero-*` IDs resolve — [Dolci-RL-Zero-Mix-7B](https://huggingface.co/datasets/allenai/Dolci-RL-Zero-Mix-7B)
- **Verifier code [HF/GitHub]:** `open_instruct/ground_truth_utils.py`, `open_instruct/IFEvalG/instructions.py`, `open_instruct/if_functions.py` and `open_instruct/grpo_fast.py` exist on `main` (HTTP 200). The older `open_instruct/ppo_vllm_thread_ray_gtrl.py` and `grpo_vllm_thread_ray_gtrl.py` (used for Tülu 3, Tülu 3.1 and OLMo 2) return 404 on `main` — [ground_truth_utils.py](https://github.com/allenai/open-instruct/blob/main/open_instruct/ground_truth_utils.py)
- **Olmo 3 RL scripts [GitHub]:** `scripts/train/olmo3/` contains `7b_think_rl.sh`, `7b_think_rl_no_pipeline.sh`, `7b_instruct_rl.sh`, `32b_instruct_rl.sh`, `32b_think_rl.sh`, `7b_rlzero_{code,general,instruction_following,math,mix}.sh` and `7b_instruct_hybrid_rl.sh` — [directory](https://github.com/allenai/open-instruct/tree/main/scripts/train/olmo3)
- **Tmax code [paper]:** "We release our data, models, and code … at https://github.com/hamishivi/tmax", which extends open-instruct — [Tmax paper](https://arxiv.org/abs/2606.23321)
- **DR Tulu code [card]:** the training script is at `rlresearch/dr-tulu/blob/rl/rl/open-instruct/train_dr_tulu.sh`. The judge is GPT-4.1-mini (paper) — [card](https://huggingface.co/rl-research/DR-Tulu-8B)
- **BAR [paper]:** the paper "does not state that checkpoints, intermediate RL checkpoints, or code will be released". The RL data is described as the Olmo 3 RLVR pools — [BAR paper](https://arxiv.org/html/2604.18473)

### Inferences
- Exact RL-prompt replication is possible for Olmo 3 7B Think (`Dolci-Think-RL-7B` is named in the script), OLMo 2, Tülu 3/3.1, Tmax and DR Tulu.
- For Olmo 3 Instruct, the released `Dolci-Instruct-RL` dataset and the script's `hamishivi/*` mixture may not be identical.

### Gaps
- I did not diff `Dolci-Instruct-RL` against the script's data mix.
- I did not pin the open-instruct commit that holds the Tülu 3 / OLMo 2 PPO code.

## Q6. Parity: do the RL start and the RL model share the tokenizer and chat template?

### Takeaway
- **Byte-identical tokenizer and template between start and RL model:**
  - Olmo 3 7B Think;
  - OLMo 2 7B and 13B (all stages);
  - Tülu 3 8B, 70B and 405B;
  - Tmax 8B;
  - DR Tulu;
  - BAR;
  - SAGE-MM Qwen3-VL.
- **Small differences:**
  - Olmo 3 7B Instruct (the `tools=[]` branch);
  - Olmo 3.1 32B Instruct (the default system prompt);
  - OLMo 2 32B (DPO template prepends BOS, the RL template does not);
  - Tülu 3.1 8B (`tokenizer.json` file differs);
  - Tmax 9B/27B (template edited to keep thinking in multi-turn history; vision tower removed);
  - RL-Zero (the base has no chat template).

### Cited Findings
- **Olmo 3 7B Think [HF]:** DPO and RL share chat template `8d1c91ad` and `tokenizer.json` `5fe17212`. The SFT differs (template `0c4c4e49`, tokenizer `8fb6a1b5`) — [DPO tree](https://huggingface.co/api/models/allenai/Olmo-3-7B-Think-DPO/tree/main), [RL tree](https://huggingface.co/api/models/allenai/Olmo-3-7B-Think/tree/main)
- **Olmo 3 7B Instruct [HF]:** both share `tokenizer.json` `54b3f7c4`. The templates differ by one condition: DPO has `tools is none or (tools | length) == 0`, RL has `tools is none` — [RL chat_template.jinja](https://huggingface.co/allenai/Olmo-3-7B-Instruct/blob/main/chat_template.jinja)
- **Olmo 3.1 32B Instruct [HF]:** both share tokenizer `54b3f7c4`. The default system prompt, used when no system message is given, differs:
  - DPO: "You are a helpful function-calling AI assistant."
  - RL: "You are Olmo, a helpful AI assistant built by Ai2. Your date cutoff is December 2024, …"

  — [RL chat_template.jinja](https://huggingface.co/allenai/Olmo-3.1-32B-Instruct/blob/main/chat_template.jinja), [DPO chat_template.jinja](https://huggingface.co/allenai/Olmo-3.1-32B-Instruct-DPO/blob/main/chat_template.jinja)
- **Olmo 3 32B Think [HF]:** DPO and the 3.0 RL model share template `e353520f` and tokenizer `5fe17212`. The 3.1 Think template is `0c4c4e49` (different) — [3.0 tree](https://huggingface.co/api/models/allenai/Olmo-3-32B-Think/tree/main)
- **RL-Zero [HF]/[paper]:** the base `Olmo-3-1025-7B` has no chat template. The RL-Zero Code/General/Mix repos carry the Think template `8d1c91ad`; 3.1 Code has a 417-character template `b3ce22e0`. Their `tokenizer.json` (`060ff36b`) differs from the base's file (`5fe17212`). The paper says RL-Zero trains with a simple custom prompt, and the mix script uses `--chat_template_name olmo_thinker_rlzero` — [Olmo 3 report §6.1](https://arxiv.org/abs/2512.13961), [mix script](https://github.com/allenai/open-instruct/blob/main/scripts/train/olmo3/7b_rlzero_mix.sh)
- **OLMo 2 7B/13B [HF]:** DPO and every RL stage share template `951f1842` and tokenizer `5fe17212` — [13B tree](https://huggingface.co/api/models/allenai/OLMo-2-1124-13B-Instruct/tree/main)
- **OLMo 2 32B [HF]:** tokenizer `5fe17212` is shared. The DPO template is `{{ bos_token }}` + the `<|user|>`/`<|assistant|>` format; the Instruct template is the identical Tülu template without `{{ bos_token }}`. That matches `--chat_template_name tulu` in the RL command — [Instruct tokenizer_config](https://huggingface.co/allenai/OLMo-2-0325-32B-Instruct/blob/main/tokenizer_config.json)
- **Tülu 3 [HF]:** DPO and RL share template `ae263161` at 8B, 70B and 405B. Tokenizers match at 8B and 70B (`2ac5eaaa`) and at 405B (`9400df98`). Tülu 3.1 8B has the same template but `tokenizer.json` `9400df98`, while its DPO start has `2ac5eaaa` — [3.1 tree](https://huggingface.co/api/models/allenai/Llama-3.1-Tulu-3.1-8B/tree/main)
- **Tmax 9B [HF]:** `Qwen/Qwen3.5-9B` template `f54c154d`; `hamishivi/Qwen3.5-9B` and `tmax-9b` template `a21b791d`. The diff is line 100, `loop.index0 > ns.last_query_index` → `reasoning_content`, which keeps reasoning in earlier turns. The paper says "We edit chat templates such that thinking is preserved on intermediate turns".
  - `tmax-9b` `tokenizer.json` is `87a7830d` vs Qwen's `5f9e4d49`.
  - The RL model drops the vision tower: 8.95B vs 9.65B, and the card says "we removed the vision head".

  — [tmax-9b tree](https://huggingface.co/api/models/allenai/tmax-9b/tree/main), [Tmax paper](https://arxiv.org/html/2606.23321)
- **Tmax 27B [HF]:** the `tmax-27b` template (`2eb19127`) adds a `preserve_thinking` flag relative to `hamishivi/Qwen3.6-27B` (`2185fc38`). That in turn differs from `Qwen/Qwen3.6-27B` (`25abdbb3`) — [tmax-27b tree](https://huggingface.co/api/models/allenai/tmax-27b/tree/main)
- **Exact matches elsewhere [HF]:**
  - Tmax 8B: `tmax-sft-8b` = `tmax-8b` (template `97d57024`, tokenizer `be756060`).
  - DR Tulu: SFT = RL = Step-1900 = No-RLER (template `b066ba71`, tokenizer `aeb13307`).
  - BAR: Math-SFT = Math (`0b688bfe`, `5fe17212`); Code-SFT = Code (`0b688bfe`, `54b3f7c4`).
  - SAGE-MM Qwen3-VL: SFT = SFT_RL (`38dd3df2`, `f54b55fa`).
  - Tülu 2.5: the template `e7fdddc9` is shared with the start; for Llama-3 Tülu 2.5 the `tokenizer.json` differs (`85317b18` vs `02bdaaa9`).

  — [DR-Tulu tree](https://huggingface.co/api/models/rl-research/DR-Tulu-8B/tree/main)

### Inferences
- Where only the template or the `tokenizer.json` file hash differs, the token-ID mapping is probably unchanged (the RL code reuses the start's tokenizer). Two differences still need care:
  - The OLMo 2 32B BOS difference and the Olmo 3.1 32B Instruct default system prompt can change sampling. Evaluate both models with an explicit, identical prompt format.
  - For Tmax 9B/27B, the start must be served as text-only Qwen3.5 / Qwen3.6, with the RL repo's template applied to both models.

### Gaps
- I compared file hashes, not vocabularies. Differing `tokenizer.json` files (Tülu 3.1, Tmax 9B/27B, RL-Zero, Llama-3 Tülu 2.5) were not decoded to confirm identical vocab and merges.
- No Jinja renderer was available in the sandbox, so template differences were assessed by textual diff only.

## Q7. Practicalities: licence, gating, release date, size on disk

### Takeaway
Nothing is gated. Licences:
- Apache-2.0: Olmo/OLMo, Tmax, DR Tulu and BAR.
- Llama 3.1 Community License: Tülu 3.
- Tülu 2.5: the cards say Apache-2.0, but the models are built on Llama 2 / Llama 3.

Several DPO starts and older repos ship only PyTorch `.bin` weights. Tülu 3 405B is about 812 GB (DPO) and about 1.6 TB for the RL repo, which holds both formats.

### Cited Findings
- **Gating [HF]:** `gated: false` for every repo checked: all Olmo/OLMo/Tülu/Tmax/BAR/SAGE repos, the `rl-research` DR Tulu repos and Qwen3.5-9B / Qwen3.6-27B — [example API](https://huggingface.co/api/models/allenai/Olmo-3-7B-Think-DPO)
- **Licence tags [HF]:** `apache-2.0` for Olmo 3/3.1, OLMo 2, OLMoE, Tmax, BAR, DR Tulu, SAGE-MM and the Tülu 2.5 PPO repos; `llama3.1` for every Tülu 3 repo. OLMo 2 cards add that outputs from third-party models are "subject to additional terms: Gemma Terms of Use". `tulu-2-13b` has no licence tag in metadata — [Tülu 3 8B card](https://huggingface.co/allenai/Llama-3.1-Tulu-3-8B), [OLMo 2 7B card](https://huggingface.co/allenai/OLMo-2-1124-7B-Instruct)
- **Creation dates (`createdAt`) [HF]:**
  - Olmo 3 7B Think DPO/RL: 2025-11-18.
  - Olmo 3 7B Instruct DPO/RL: 2025-11-19.
  - Olmo 3.1 32B Instruct: 2025-12-10 (its DPO 2025-11-18).
  - OLMo 2 7B/13B: 2024-12-18. OLMo 2 32B: 2025-03-12.
  - Tülu 3 8B/70B: 2024-11-20. Tülu 3.1 8B: 2025-02-07. Tülu 3 405B: 2025-01-09 (its DPO 2025-01-28).
  - Tmax: 2026-06-17/21.
  - DR Tulu: 2025-11-13, card updated "23/02/2026".
  - BAR: 2026-04-19.
  - Tülu 2.5 PPO: 2024-06-11 (13B) and 2024-10-14 (Llama-3 8B).

  — [HF listing](https://huggingface.co/api/models?author=allenai&limit=1000&sort=createdAt&direction=-1)
- **Weight formats and sizes [HF]:** `OLMo-2-1124-7B-DPO` (14.6 GB), `OLMo-2-0325-32B-DPO` (64.5 GB), `Llama-3.1-Tulu-3-405B-DPO` (811.7 GB), `tulu-2-13b` and the Tülu 2.5 PPO repos ship `pytorch_model-*.bin` only. The `Llama-3.1-Tulu-3-405B` `main` and `step_75` carry 382 weight shards, about 1,623 GB (both `.bin` and `.safetensors`) — [405B-DPO tree](https://huggingface.co/api/models/allenai/Llama-3.1-Tulu-3-405B-DPO/tree/main)
- **Serving notes [card]:** Tmax needs `--language_model_only` in vLLM and the `qwen3_xml` tool parser. Olmo 3 needs transformers ≥ 4.57.0. Olmo Hybrid needs transformers ≥ 5.3.0 — [tmax-9b card](https://huggingface.co/allenai/tmax-9b), [Olmo-3-7B-RL-Zero-Mix card](https://huggingface.co/allenai/Olmo-3-7B-RL-Zero-Mix)

### Inferences
- Tülu 3 405B is fully checkpointed but impractical for sampling 256–1,024 times per prompt. Tülu 3 70B is feasible but has only one intermediate checkpoint.

### Gaps
- The exact licence text of `tulu-2-13b` was not read; its Llama 2 base implies Llama 2 terms.

## Q8. Caveats and exclusions

### Takeaway
The main caveats are:
- the RL start is "claimed" rather than hash-provable for Olmo 3 Instruct and Tülu 3 8B;
- card/paper inconsistencies in OLMo 2, Tülu 3 70B and Tmax 27B;
- non-verifiable rewards (Olmo 3 chat, DR Tulu, Tülu 2.5);
- agentic environments (Tmax, DR Tulu).

Exclusions:
- **Stops at DPO or SFT:** Olmo Hybrid 7B, AstaBrief 8B, SERA, OLMo 1 Instruct and Tülu 2.
- **Final stage is not RL:** BAR-5x7B (router training).
- **Below 7B:** OLMoE.
- **No RL stated:** FlexOlmo and Bolmo.

### Cited Findings
- **Olmo Hybrid 7B [HF]:** the released repos are base `allenai/Olmo-Hybrid-7B` (7.43B, 1,497 branches), `Olmo-Hybrid-Instruct-SFT-7B`, `Olmo-Hybrid-Instruct-DPO-7B` and `Olmo-Hybrid-Think-SFT-7B`. Their cards show "--" in the "Final Models (RLVR)" row.
  - open-instruct nevertheless has `7b_instruct_hybrid_rl.sh` with `model_name_or_path="allenai/Olmo-Hybrid-Instruct-DPO-7B"`. An RL run was configured, but no RL model was public as of 2026-11-08.

  — [Hybrid DPO card](https://huggingface.co/allenai/Olmo-Hybrid-Instruct-DPO-7B), [hybrid RL script](https://github.com/allenai/open-instruct/blob/main/scripts/train/olmo3/7b_instruct_hybrid_rl.sh)
- **AstaBrief 8B [card]:** trained "using only supervised fine-tuning and offline direct preference optimization", so it stops at DPO — [card](https://huggingface.co/allenai/AstaBrief_8B)
- **SERA [card]:** trained on 25,000 synthetic trajectories using "Soft Verified Generation", which the card calls "26x cheaper than reinforcement learning". This is SFT, not RL — [SERA-32B card](https://huggingface.co/allenai/SERA-32B)
- **BAR-5x7B [card]:** the "final 5-expert MoE combining all experts with a trained router". The last stage is router training, and the Tool-Use and Safety experts are SFT only — [card](https://huggingface.co/allenai/BAR-5x7B)
- **BAR expert initialisation [paper]:** "Each domain expert's FFN weights are initialized from the pre-trained (not post-trained) base", and the anchor FFN is frozen. The BAR-2x7B RL models therefore contain a frozen copy of BAR-7B's FFNs — [BAR paper](https://arxiv.org/html/2604.18473)
- **OLMo 2 relabelling [card]:** OLMo 2 post-trained models were retrained after "post-trained models did not share the pre-tokenization logic". The old versions carry a `-preview` suffix, for example `OLMo-2-1124-7B-Instruct-preview`. The 13B RLVR2 card's own text describes the 7B model, which is a copy error — [RLVR2 card](https://huggingface.co/allenai/OLMo-2-1124-13B-Instruct-RLVR2)
- **Tmax 27B card errors [card]/[HF]:** the card states that the main checkpoint is step 160, which matches the SHA check. Its hyperparameter line says "this checkpoint is from 200 steps", and "Finetuned from model: Qwen 3.5 9B" is a copy error — [card](https://huggingface.co/allenai/tmax-27b)
- **SAGE-MM Molmo2 [HF]:** the `base_model` of `SAGE-MM-Molmo2-8B-SFT_RL` is `allenai/SAGE-MM-Molmo2-D-8B-SFT`, a name that does not match the released `allenai/SAGE-MM-Molmo2-8B-SFT` — [API](https://huggingface.co/api/models/allenai/SAGE-MM-Molmo2-8B-SFT_RL)
- **olmOCR 2 [HF]:** `allenai/olmOCR-2-7B-1025` declares `base_model` Qwen2.5-VL-7B-Instruct. No SFT-only olmOCR-2 checkpoint appears in the `allenai` listing; `olmOCR-7B-0825` is an earlier generation — [card](https://huggingface.co/allenai/olmOCR-2-7B-1025)
- **Tmax ablations [card]:** the `qwen35-9b-*` cards say "The main model checkpoint is step 300" and that branches are provided, but the refs list `main` only — [card](https://huggingface.co/allenai/qwen35-9b-openthoughts)

### Inferences
- Treat any Olmo 3 Instruct or Tülu 3 8B reach result as "relative to the released DPO, which is the HF-declared start". Treat Olmo 3 7B Think, OLMo 2 32B, Tülu 3.1 8B and Tmax as having verified starts.

### Gaps
- I could not confirm whether the 7B Instruct `main` (or the 32B Instruct `main`) is a direct RL output or a selected or merged checkpoint. The paper describes selection only, with no merging for RL.

## Q9. Summary inventory and fit grades

### Takeaway
- **A, exact or claimed-exact start + RL model + intermediate checkpoints:**
  - Olmo 3 7B Think, 7B Instruct and 32B Think 3.0;
  - RL-Zero Code, General, Mix and 3.1 Code;
  - OLMo 2 7B, 13B and 32B;
  - Tülu 3 8B, Tülu 3.1 8B and Tülu 3 405B;
  - Tmax 9B, 27B and 8B;
  - DR Tulu 8B.
- **Best practical next targets:** Olmo 3 7B Think, Tülu 3.1 8B and OLMo 2 32B/13B. Olmo 3 7B Instruct is a good fourth if the claimed start is acceptable.

### Cited Findings
All entries below are summarised from the findings and sources above (HF refs, trees, metadata, cards, papers).

| model | params | base | RL start (repo, exact?) | RL final | intermediate RL ckpts | RL domains | RL data public | verifier public | template parity | licence | fit |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Olmo 3 7B Think | 7.30B | allenai/Olmo-3-1025-7B | allenai/Olmo-3-7B-Think-DPO — exact (script + HF lineage); unclear which of two replicate runs was released | allenai/Olmo-3-7B-Think (main ≈ step 1,400) | 55 branches: step_0025…step_1375, every 25 | math, code, IF (verifiers) + chat (Qwen3-32B judge) | yes: Dolci-Think-RL-7B | yes: open-instruct `ground_truth_utils.py`, `IFEvalG`, `7b_think_rl.sh` | identical template + tokenizer | Apache-2.0 | **A** |
| Olmo 3 7B Instruct | 7.30B | allenai/Olmo-3-1025-7B | allenai/Olmo-3-7B-Instruct-DPO — claimed (HF lineage); script uses an internal ckpt; paper says RL on 2 DPO candidates | allenai/Olmo-3-7B-Instruct (main ≈ step 450) | 8: step_050…step_400, every 50 | math, code, IF, chat (judge) | yes: Dolci-Instruct-RL (card's `-7B` id is a 307) | yes | same tokenizer; template differs only for `tools=[]` | Apache-2.0 | **A** (start claimed) |
| Olmo 3.1 32B Instruct | 32.23B | allenai/Olmo-3-1125-32B | allenai/Olmo-3.1-32B-Instruct-DPO — claimed (HF lineage); script uses an internal ckpt | allenai/Olmo-3.1-32B-Instruct | none (main only, despite card) | math, code, IF, chat (judge) | yes: Dolci-Instruct-RL | yes: `32b_instruct_rl.sh` @8d8232cc | same tokenizer; default system prompt differs | Apache-2.0 | **B** |
| Olmo 3 32B Think (3.0) | 32.23B | allenai/Olmo-3-1125-32B | allenai/Olmo-3-32B-Think-DPO — claimed (HF lineage); same run as 3.1, cut at 750 | allenai/Olmo-3-32B-Think | 15: step_050…step_750, every 50 | as 3.1 Think | yes: Dolci-Think-RL-32B | yes | identical | Apache-2.0 | **A** (redundant with 3.1) |
| Olmo 3 7B RL-Zero Code | 7.30B | allenai/Olmo-3-1025-7B | = base (HF lineage) | allenai/Olmo-3-7B-RL-Zero-Code | 29: step_100…step_2900, every 100 | code (tests) | yes: Dolci-RL-Zero-Code-7B | yes: `7b_rlzero_code.sh` | base has no template; RL-Zero prompt; tokenizer file differs | Apache-2.0 | **A** |
| Olmo 3 7B RL-Zero General | 7.30B | allenai/Olmo-3-1025-7B | = base (HF lineage; script uses an internal long-context ckpt) | allenai/Olmo-3-7B-RL-Zero-General | 8: step_100…step_800 | general chat (LLM judge) | yes: Dolci-RL-Zero-General-7B | yes | as above | Apache-2.0 | **A** (judge reward) |
| Olmo 3 7B RL-Zero Mix | 7.30B | allenai/Olmo-3-1025-7B | allenai/Olmo-3-1025-7B — exact (script) | allenai/Olmo-3-7B-RL-Zero-Mix | 19: step_50…step_950, every 50 | math, code, IF, chat | yes: Dolci-RL-Zero-Mix-7B | yes: `7b_rlzero_mix.sh` | as above | Apache-2.0 | **A** |
| Olmo 3.1 7B RL-Zero Code | 7.30B | allenai/Olmo-3-1025-7B | = base (HF lineage) | allenai/Olmo-3.1-7B-RL-Zero-Code | 33, irregular: step_050…step_1950 (missing 750, 950 and some after 1500) | code | yes: Dolci-RL-Zero-Code-7B | yes | 417-char template; tokenizer file differs | Apache-2.0 | **A** |
| OLMo 2 7B Instruct | 7.30B | allenai/OLMo-2-1124-7B | allenai/OLMo-2-1124-7B-DPO — claimed (HF lineage; paper: "highest-performing DPO") | allenai/OLMo-2-1124-7B-Instruct | 6: step_60…step_360, every 60 | math (card: GSM8K; paper: GSM8K+MATH+IF), PPO + RM value | yes: RLVR-GSM | yes, at an older open-instruct commit (PPO script not on main) | identical | Apache-2.0 | **A** |
| OLMo 2 13B Instruct (3 RL stages) | 13.72B | allenai/OLMo-2-1124-13B | OLMo-2-1124-13B-DPO → RLVR1 → RLVR2 → Instruct (chain on HF; RLVR1 metadata error) | allenai/OLMo-2-1124-13B-Instruct | 13 per stage × 3: step_60…step_780, every 60 (distinct weights) | math, IF (PPO) | yes: RLVR-GSM-MATH-IF-Mixed-Constraints, RLVR-GSM, RLVR-MATH | yes (older commit) | identical across all stages | Apache-2.0 | **A** |
| OLMo 2 32B Instruct | 32.23B | allenai/OLMo-2-0325-32B | allenai/OLMo-2-0325-32B-DPO — exact (card command); `.bin` weights | allenai/OLMo-2-0325-32B-Instruct (= step_320) | 17: step_20…step_340, every 20 | math, IF (GRPO, β 0.01) | yes: RLVR-GSM-MATH-IF-Mixed-Constraints | yes (`grpo_vllm_thread_ray_gtrl.py`, older commit) | same tokenizer; DPO template adds BOS, RL (Tülu) template does not | Apache-2.0 | **A** |
| Tülu 3 8B | 8.03B | meta-llama/Llama-3.1-8B | allenai/Llama-3.1-Tulu-3-8B-DPO — claimed (HF lineage; paper tried 2 DPO starts) | allenai/Llama-3.1-Tulu-3-8B | 10: step_100…step_1000, every 100 | math, IF (PPO, RM-initialised value) | yes: RLVR-GSM-MATH-IF-Mixed-Constraints | yes (older commit) | identical | Llama 3.1 | **A** (start claimed) |
| Tülu 3.1 8B | 8.03B | meta-llama/Llama-3.1-8B | allenai/Llama-3.1-Tulu-3-8B-DPO — exact (card command) | allenai/Llama-3.1-Tulu-3.1-8B (= step_1920) | 61: step_40…step_2440, every 40 | math, IF (GRPO) | yes: same | yes (older commit) | same template; tokenizer.json hash differs | Llama 3.1 | **A** |
| Tülu 3 70B | 70.55B | meta-llama/Llama-3.1-70B | allenai/Llama-3.1-Tulu-3-70B-DPO — claimed (HF lineage; paper: "best overall DPO") | allenai/Llama-3.1-Tulu-3-70B | 1: step_120 | math, IF (PPO) | yes | yes (older commit) | identical | Llama 3.1 | **B** (one intermediate) |
| Tülu 3 405B | 405.85B | meta-llama/Llama-3.1-405B | allenai/Llama-3.1-Tulu-3-405B-DPO — claimed (HF lineage + paper); `.bin`, 812 GB | allenai/Llama-3.1-Tulu-3-405B (= step_75) | 2: step_25, step_50 | math only (MATH; PPO, 8B value model) | yes: RLVR-MATH | yes (older commit) | identical | Llama 3.1 | **A** (impractical) |
| Tmax 9B | 8.95B | Qwen/Qwen3.5-9B | hamishivi/Qwen3.5-9B = Qwen/Qwen3.5-9B weights byte-identical (HF SHA); no SFT | allenai/tmax-9b (= step_200) | 5: step_100…step_500 + `rollouts/` logprobs | multi-turn terminal tasks (Docker unit-test verifiers; DPPO) | yes: tmax-15k-open-instruct / TMax-15K | yes: github.com/hamishivi/tmax | template edited (keeps reasoning in history); vision tower dropped; tokenizer file differs | Apache-2.0 | **A** (agentic) |
| Tmax 27B | 26.9B | Qwen/Qwen3.6-27B | hamishivi/Qwen3.6-27B = Qwen/Qwen3.6-27B weights byte-identical | allenai/tmax-27b (= step_160) | 5: step_100, 160, 200, 240, 300 | terminal | yes | yes | template differs (`preserve_thinking`) | Apache-2.0 | **A** (agentic) |
| Tmax 8B | 8.19B | Qwen/Qwen3-8B | allenai/tmax-sft-8b — exact (card) | allenai/tmax-8b (= step_300) | 5: step_100…step_500 | terminal | yes (+ tmax-sft) | yes | identical | Apache-2.0 | **A** (agentic) |
| Tmax Qwen3.5-9B ablations ×6 | 8.95B | Qwen/Qwen3.5-9B | Qwen/Qwen3.5-9B (card) | allenai/qwen35-9b-{openthoughts, terminaltraj, cli-gym, swesmith, termigen, endless} | none (main only, despite card) | terminal (other datasets) | yes (e.g. open-instruct-endless-terminals) | yes | not checked | Apache-2.0 | **B** |
| DR Tulu 8B | 8.19B | Qwen/Qwen3-8B | rl-research/DR-Tulu-SFT-8B — claimed (card + HF lineage) | rl-research/DR-Tulu-8B (main = step_4000) | 80: step_50…step_4000, every 50; plus DR-Tulu-8B-Step-1900 | long-form deep research with search tools; rubric rewards judged by GPT-4.1-mini (GRPO) | yes: dr-tulu-rl-data | code yes (rlresearch/dr-tulu); judge is a closed API | identical | Apache-2.0 | **A** (judge reward, tool env) |
| DR Tulu No-RLER 8B (ablation) | 8.19B | Qwen/Qwen3-8B | rl-research/DR-Tulu-SFT-8B | rl-research/DR-Tulu-No-RLER-8B | 23: step_50…step_1150 | same, without evolving rubrics | yes | yes | identical | Apache-2.0 | **A** (ablation) |
| BAR 2x7B Math | 11.63B (MoE) | Olmo-2-based BAR-7B + Olmo 2 pretrained expert | allenai/BAR-2x7B-Math-SFT — implied by the paper's pipeline, not stated; no `base_model` | allenai/BAR-2x7B-Math | none | math (GRPO RLVR) | partly: "Olmo 3 math RLVR data", no dedicated repo | not stated | identical | Apache-2.0 | **B** |
| BAR 2x7B Code | 11.63B (MoE) | as above | allenai/BAR-2x7B-Code-SFT — implied | allenai/BAR-2x7B-Code | none | code (execution) | partly: Olmo 3 code RLVR | not stated | identical | Apache-2.0 | **B** |
| Tülu 2.5 PPO 13B ×9 | 13.02B | meta-llama/Llama-2-13b | allenai/tulu-2-13b (SFT; HF lineage) | allenai/tulu-v2.5-ppo-13b-* | none | general chat; learned-RM reward (PPO) | prompts yes: tulu-2.5-preference-data; RMs + value models released | RM-based (no verifier) | identical template | apache-2.0 tag (Llama 2 base) | **B** (RM reward) |
| Llama-3 Tülu 2.5 PPO 8B ×3 | 8.03B | meta-llama/Meta-Llama-3-8B | allenai/llama-3-tulu-2-8b | allenai/llama-3-tulu-v2.5-8b-* | none | chat; RM reward | prompts yes | RM-based | same template; tokenizer file differs | apache-2.0 tag (Llama 3 base) | **B** (RM reward) |
| SAGE-MM (VLM) ×3 | 8.29–8.77B | Qwen2.5-VL-7B / Qwen3-VL-8B / Molmo2-8B | allenai/SAGE-MM-*-SFT (HF lineage; Molmo2 name mismatch) | allenai/SAGE-MM-*-SFT_RL | none | video-understanding tool agent | yes: SAGE-MM-RL-7k | github.com/allenai/SAGE | identical (Qwen3-VL pair) | Apache-2.0 | **B** (multimodal) |
| olmOCR 2 7B | 8.29B | Qwen/Qwen2.5-VL-7B-Instruct | SFT-only olmOCR-2 start not released | allenai/olmOCR-2-7B-1025 | none | OCR (GRPO, unit-test rewards) | yes: olmOCR-synthmix-1025 | github.com/allenai/olmocr | n/a | Apache-2.0 | **C** |
| Olmo Hybrid 7B | 7.43B | allenai/Olmo-Hybrid-7B | — | none released (RL script exists) | — | — | — | — | — | Apache-2.0 | **X**: stops at DPO/SFT |
| BAR 5x7B | 24.61B | — | — | — | — | — | — | — | — | Apache-2.0 | **X**: final stage is router training |
| OLMoE 1B-7B 0125 Instruct | 6.92B | allenai/OLMoE-1B-7B-0125 | OLMoE-1B-7B-0125-DPO | OLMoE-1B-7B-0125-Instruct | 44: step_40…step_1760 | math (RLVR-GSM) | yes | yes | identical | Apache-2.0 | **X**: < 7B total (would be A) |
| AstaBrief 8B / SERA 8–32B / FlexOlmo / Bolmo / OLMo 1 Instruct / Tülu 2 DPO | 7–33B | — | — | — | — | — | — | — | — | various | **X**: DPO-only, SFT-only, or no RL |
| Olmo 4 / Olmo 3.2 / Tülu 4 | — | — | — | not released (Olmo 3.2 exists only as tokenizer repos) | — | — | — | — | — | — | **X**: not released by 2026-11-08 |

### Inferences
- **For verifiable-reward reach tests in new families:**
  - Olmo 3 7B Think is the closest analogue to the setup already in use: same recipe as 3.1 Think 32B, 55 checkpoints, exact start.
  - Tülu 3.1 8B adds a different base (Llama 3.1) with GRPO, an exact start and 61 checkpoints.
  - OLMo 2 32B and 13B add the OLMo 2 base with exact or chained starts.
- **Agentic and judge-reward families:** Tmax and DR Tulu are A-grade on checkpoints but need agent harnesses (Docker terminals; search plus MCP) and, for DR Tulu, a GPT-4.1-mini judge.
- **Smaller-scale replication of the 32B result:** Tmax 9B/27B have byte-verified starts equal to the public Qwen release, and Olmo 3 7B Think matches the 32B recipe. Either would test whether the 32B result replicates at a smaller scale or in another family.

### Gaps
- The step value of `main` is inferred, not HF-labelled, for Olmo 3 7B Think/Instruct, Tülu 3 8B and OLMo 2 7B/13B.
- It is not established whether the released Olmo 3 Instruct and Tülu 3 8B DPO repos are bit-identical to their RL inits.
- Tokenizer vocab equivalence was checked by file hash only.
