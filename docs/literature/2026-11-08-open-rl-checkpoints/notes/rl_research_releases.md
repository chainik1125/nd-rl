---
written_on: 2026-11-08
written_by: agent:claude
---

# RL research releases (≥ 7B) that publish the RL start and the RL'd model: a verified checkpoint inventory (as of 2026-11-08)

Method. Every repo, branch, tag, file set, parameter count, licence and creation date below was read from the Hugging Face API on 2026-11-08: `/api/models?author=…` org listings, `/api/models/{id}` (safetensors metadata, `cardData`, file list), `/api/models/{id}/refs`, `/api/models/{id}/commits/main`, `/api/models/{id}/tree/{rev}`, plus each repo's raw `README.md`, `tokenizer_config.json`, `chat_template.jinja` and `config.json`. **HF✓** marks a fact checked directly on HF. **claim** marks something stated in a model card, paper or GitHub README that HF cannot confirm, for example what a model was trained from or how. Parameter counts are the HF safetensors totals. "Template parity" compares the chat templates (hashed, then diffed), `vocab_size` and `eos_token` of the RL start and the RL model. Where both repos store `tokenizer.json` the same way, it also compares `tokenizer.json` hashes.

## 1. Repos and sizes: which ≥ 7B RL research releases exist, with exact repo ids?

### Takeaway
At ≥ 7B there are about 45 relevant RL-final research releases on HF. Most come from Qwen2.5/Qwen3 bases or from DeepSeek-R1-Distill-Qwen, at 7–32B. A few are 100B+ (INTELLECT-3/3.1, Nemotron-3-Ultra-Math, MiMo-V2.x, DeepSeek-R1-Zero). Several 2026 releases postdate common knowledge: Nemotron-Cascade (Dec 2025), INTELLECT-3.1, Nemotron-3-Labs-Ultra-Math, MiMo-V2.6, DR.Kernel, GoLongRL and K2-Horizon.

### Cited Findings
RL straight from a pretrained base ("RL-Zero" style):
- Open-Reasoner-Zero: `Open-Reasoner-Zero/Open-Reasoner-Zero-7B` (7.62B) and `-32B` (32.76B), both created 2025-02-18. The org also hosts value models `Open-Reasoner-Zero-Critic-{0.5B,1.5B,7B,32B}` and `ORZ-R1-Distill-Qwen-14B` (14.77B; the repo has no README) (HF✓) — [org listing](https://huggingface.co/api/models?author=Open-Reasoner-Zero&limit=1000), [ORZ-32B](https://huggingface.co/Open-Reasoner-Zero/Open-Reasoner-Zero-32B)
- DAPO: `BytedTsinghua-SIA/DAPO-Qwen-32B` (32.76B, 2025-04-09) (HF✓) — [card](https://huggingface.co/BytedTsinghua-SIA/DAPO-Qwen-32B)
- SimpleRL-Zoo (all created 2025-03-24) (HF✓) — [org listing](https://huggingface.co/api/models?author=hkust-nlp&limit=1000):
  - `hkust-nlp/Qwen-2.5-7B-SimpleRL-Zoo` (7.62B), `Qwen-2.5-14B-SimpleRL-Zoo` (14.77B), `Qwen-2.5-32B-SimpleRL-Zoo` (32.76B), `Qwen-2.5-Math-7B-SimpleRL-Zoo` (7.62B)
  - `Llama-3.1-8B-SimpleRL-Zoo` (8.03B), `Mistral-7B-v0.1-SimpleRL-Zoo` (7.24B), `Mistral-Small-24B-SimpleRL-Zoo` (23.57B), `DeepSeek-Math-7B-SimpleRL-Zoo` (6.91B)
  - The earlier `Qwen-2.5-Math-7B-SimpleRL-Zero` (7.62B, 2025-02-18)
- Xiaomi MiMo-7B: `XiaomiMiMo/MiMo-7B-Base`, `-SFT`, `-RL-Zero`, `-RL` (all 7.83B, 2025-04-29), plus `MiMo-7B-RL-0530` (2025-05-30) (HF✓) — [org listing](https://huggingface.co/api/models?author=XiaomiMiMo&limit=1000)
- General-Reasoner: `TIGER-Lab/General-Reasoner-Qwen2.5-7B` (7.62B), `-Qwen2.5-14B` (14.77B), `-Qwen3-14B` (14.77B), plus the verifier `TIGER-Lab/general-verifier` (HF✓) — [org listing](https://huggingface.co/api/models?author=TIGER-Lab&limit=1000)
- Other zero-RL 7–14B releases (HF✓):
  - `sail/Qwen2.5-Math-7B-Oat-Zero` (7.62B)
  - `zwhe99/DeepMath-Zero-7B` (7.62B) and `DeepMath-Zero-Math-7B`
  - `andrewzh/Absolute_Zero_Reasoner-Coder-7b` (7.62B) and `-14b` (14.77B); neither has a README
  - `GAIR/ToRL-7B` (7.62B; no README)
  - `PRIME-RL/Eurus-2-7B-PRIME-Zero` (7.62B)
  - Search-R1 variants such as `PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo-v0.3` (7.62B; 11 variants for base or instruct start × PPO or GRPO × v0.2/v0.3)
  - Sources: [sail](https://huggingface.co/sail/Qwen2.5-Math-7B-Oat-Zero), [zwhe99](https://huggingface.co/api/models?author=zwhe99&limit=1000), [andrewzh](https://huggingface.co/api/models?author=andrewzh&limit=1000), [GAIR](https://huggingface.co/api/models?author=GAIR&limit=1000), [PRIME-RL](https://huggingface.co/api/models?author=PRIME-RL&limit=1000), [HF search](https://huggingface.co/api/models?search=SearchR1-nq_hotpotqa_train-qwen2.5-7b)
- `deepseek-ai/DeepSeek-R1-Zero` (684.49B including MTP weights, 2025-01-20) (HF✓) — [card](https://huggingface.co/deepseek-ai/DeepSeek-R1-Zero)

RL from an R1-distilled or other SFT model:
- Skywork-OR1: `Skywork/Skywork-OR1-7B` (7.62B), `-32B` (32.76B), both 2025-05-13; `Skywork-OR1-Math-7B` (7.62B, 2025-04-12) (HF✓) — [card](https://huggingface.co/Skywork/Skywork-OR1-32B)
- NVIDIA: `nvidia/AceReason-Nemotron-7B` (7.62B), `-14B` (14.77B), `AceReason-Nemotron-1.1-7B` (7.62B) and `AceMath-RL-Nemotron-7B` (7.62B). An HF search for "AceReason" returns only those three AceReason repos (HF✓) — [search](https://huggingface.co/api/models?search=AceReason&author=nvidia)
- Nemotron-Cascade (HF✓) — [search](https://huggingface.co/api/models?search=Cascade&author=nvidia):
  - Models: `nvidia/Nemotron-Cascade-8B` (8.19B, 2025-12-16), `-8B-Thinking` (8.19B), `-14B-Thinking` (14.77B) and `Nemotron-Cascade-2-30B-A3B` (31.58B, 2026-03-18)
  - Stage checkpoints: `nvidia/Nemotron-Cascade-8B-Intermediate-ckpts` (2025-12-19) holds five 16.4 GB stage checkpoints as subfolders
- `POLARIS-Project/Polaris-7B-Preview` (7.62B). The 2026 V2 releases are `Polaris-V2-Qwen3-4B-SFT` and `Polaris-V2-Qwen3-4B` (4.02B, below threshold) and `Polaris-V2-Qwen3-235B-A22B`, whose card is an empty template (HF✓) — [org listing](https://huggingface.co/api/models?author=POLARIS-Project&limit=1000)
- `qihoo360/Light-R1-14B-DS` (14.77B, RL). `Light-R1-32B`, `-32B-DS` and `-7B-DS` are SFT/DPO-final (HF✓) — [org listing](https://huggingface.co/api/models?author=qihoo360&limit=1000)
- `microsoft/Phi-4-reasoning` and `microsoft/Phi-4-reasoning-plus` (both 14.66B) (HF✓) — [card](https://huggingface.co/microsoft/Phi-4-reasoning-plus)
- `mistralai/Magistral-Small-2506` (23.57B), `-2507` (23.57B) and `-2509` (24.01B). No Magistral SFT-only repo exists in the mistralai listing (HF✓) — [org listing](https://huggingface.co/api/models?author=mistralai&limit=1000)
- PrimeIntellect INTELLECT-MATH: `PrimeIntellect/INTELLECT-MATH-SFT` (PyTorch .bin, 15.2 GB), `INTELLECT-MATH-step-47` (7.62B) and `INTELLECT-MATH` (7.62B), all 2025-01-17 (HF✓) — [org listing](https://huggingface.co/api/models?author=PrimeIntellect&limit=1000)
- `PRIME-RL/Eurus-2-7B-SFT` and `PRIME-RL/Eurus-2-7B-PRIME` (7.62B) (HF✓) — [org listing](https://huggingface.co/api/models?author=PRIME-RL&limit=1000)
- Kwai-Klear (HF✓) — [org listing](https://huggingface.co/api/models?author=Kwai-Klear&limit=1000):
  - `Kwai-Klear/Klear-Reasoner-8B-SFT` and `Klear-Reasoner-8B` (8.19B)
  - `Klear-AgentForge-8B-SFT` (gated "auto", 16.4 GB) and `Klear-AgentForge-8B` (8.19B; no README)
  - `GoLongRL-30B-A3B` (30.53B, 2026-05-19)
- `agentica-org/DeepCoder-14B-Preview` (14.77B) and `agentica-org/DeepSWE-Preview` (32.76B) (HF✓) — [org listing](https://huggingface.co/api/models?author=agentica-org&limit=1000)
- `rstar2-reproduce/rStar2-Agent-14B` (14.77B) (HF✓) — [card](https://huggingface.co/rstar2-reproduce/rStar2-Agent-14B)
- inclusionAI AReaL (HF✓, weights as PyTorch .bin) — [org listing](https://huggingface.co/api/models?author=inclusionAI&limit=1000):
  - RL models: `inclusionAI/AReaL-boba-RL-7B`; `AReaL-boba-2-8B`, `-14B` and `-32B`; and the open-data variants `-8B-Open` and `-14B-Open`
  - SFT-final: `AReaL-boba-SFT-32B`
- `a-m-team/AM-Thinking-v1` (32.76B) (HF✓) — [card](https://huggingface.co/a-m-team/AM-Thinking-v1)
- DR.Kernel: `hkust-nlp/drkernel-8b` (8.19B), `drkernel-8b-coldstart`, `drkernel-14b` (14.77B) and `drkernel-14b-coldstart` (59.1 GB of shards, i.e. fp32), created 2026-02-05 (HF✓) — [card](https://huggingface.co/hkust-nlp/drkernel-14b)
- `PRIME-RL/P1-30B-A3B` (30.53B) and `PRIME-RL/P1-235B-A22B` (235.09B), 2025-10-22 (HF✓) — [card](https://huggingface.co/PRIME-RL/P1-30B-A3B)

Large RL runs:
- `PrimeIntellect/INTELLECT-2` (32.76B, 2025-05-11) (HF✓)
- `PrimeIntellect/INTELLECT-3` (106.85B, 2025-11-26) and `-3-FP8` (HF✓)
- `PrimeIntellect/INTELLECT-3-Base` (110.47B, 2025-09-09). Per its card it is "a clone of GLM-4.5-Air-Base … with a custom chat template" (HF✓) — [card](https://huggingface.co/PrimeIntellect/INTELLECT-3-Base)
- `PrimeIntellect/INTELLECT-3.1` (106.85B, created 2026-01-20; one commit dated 2026-02-18) (HF✓) — [card](https://huggingface.co/PrimeIntellect/INTELLECT-3.1)
- In the INTELLECT-3 collection, `INTELLECT-3-SFT` and `INTELLECT-3-RL` are datasets, not models — [collection](https://huggingface.co/collections/PrimeIntellect/intellect-3)
- `nvidia/Nemotron-3-Labs-Ultra-Math-SFT` and `nvidia/Nemotron-3-Labs-Ultra-Math-RL` (560.52B each, 2026-09-02) (HF✓) — [card](https://huggingface.co/nvidia/Nemotron-3-Labs-Ultra-Math-RL)
- Xiaomi (HF✓) — [card](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL):
  - MiMo-V2: `XiaomiMiMo/MiMo-V2-Flash` and `-Base` (309.77B, 2025-12-16)
  - MiMo-V2.6: `MiMo-V2.6-Flash-RL` (310.76B) and `MiMo-V2.6-Pro-RL` (1,024.22B), 2026-09-21; then `MiMo-V2.6-{Flash,Pro}-MOPD` (2026-09-27)
- `Qwen/QwQ-32B` (32.76B, 2025-03-05) (HF✓)

AllenAI RLVR releases. These are DPO→RLVR rather than "research RL" papers, but they fit the test design directly; they may also be covered elsewhere (HF✓) — [Tulu 3.1 card](https://huggingface.co/allenai/Llama-3.1-Tulu-3.1-8B), [OLMo 2 32B Instruct card](https://huggingface.co/allenai/OLMo-2-0325-32B-Instruct):
- Tulu 3: `allenai/Llama-3.1-Tulu-3-8B` (8.03B), `Llama-3.1-Tulu-3.1-8B` (8.03B), `Llama-3.1-Tulu-3-70B` and `Llama-3.1-Tulu-3-405B`
- OLMo 2: `allenai/OLMo-2-0325-32B-Instruct` (32.23B), `OLMo-2-1124-7B-Instruct` (7.30B) and `OLMo-2-1124-13B-Instruct` (13.72B)

LLM360/IFM (brief; covered elsewhere): `IFM/K2-Horizon-7B` (9.00B per HF, 2026-09-01) has 70 tags, including six `rl_*` tags (HF✓) — [refs](https://huggingface.co/api/models/IFM/K2-Horizon-7B/refs). K2-Think was not re-verified here.

Smaller models, one line each (HF✓):
- `agentica-org/DeepScaleR-1.5B-Preview`: `main` only.
- `nvidia/Nemotron-Research-Reasoning-Qwen-1.5B` (ProRL): tag `v1`, branches `v2` and `brorl` (BroRL).
- `hbx/JustRL-DeepSeek-1.5B`: 21 branches, `step_0200` … `step_4200` every 200, init `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`. It is an A-grade release, but at 1.5B.
- Archer: `Fate-Zero/Archer-Code-1.5B` and the `Archer2.0-*-1.5B-Preview` models.
- `BytedTsinghua-SIA/Open-MOPD-SmolLM3-3B-{MixSFT,RL-Math,RL-Code,RL-IF,Final}`: stage checkpoints at 3B.
- Sources: [JustRL refs](https://huggingface.co/api/models/hbx/JustRL-DeepSeek-1.5B/refs), [ProRL refs](https://huggingface.co/api/models/nvidia/Nemotron-Research-Reasoning-Qwen-1.5B/refs), [BytedTsinghua-SIA listing](https://huggingface.co/api/models?author=BytedTsinghua-SIA&limit=1000)

### Inferences
- The "public start by construction" pattern dominates: about two-thirds of the ≥ 7B releases start from a Qwen2.5/Qwen3 base or post-trained model, or from DeepSeek-R1-Distill-Qwen-7B/14B/32B.
- The rarer, more valuable cases are projects that also publish their own SFT start: MiMo, Klear, INTELLECT-MATH, Eurus-2, DR.Kernel, Nemotron-Cascade, AllenAI and Phi-4.

### Gaps
- `Fate-Zero/Archer-Math-30B-Preview`, `Fate-Zero/ASPO-Math-30B-Preview` (30.53B, 2026-05) and `GAIR/daVinci-kernel-{8B,14B}-RL` (2026-06) have no README, so their RL start is undocumented.
- `GAIR/daVinci-kernel-14B-SFT` contains only `.gitattributes`.
- The original `miromind-ai/MiroMind-M1-RL-{7B,32B}` and `MiroMind-M1-SFT-7B` are no longer in the miromind-ai org. Only third-party GGUF mirrors were found ([HF search](https://huggingface.co/api/models?search=MiroMind-M1)), so they were not graded.
- The `LLM360` author query returned 0 repos (possibly a case or API quirk), so K2-Think was not verified.

## 2. The RL start: is the init public, and exactly which revision?

### Takeaway
For most releases the RL start is a named public repo, but **no release pins an init commit hash**; "exact" therefore means the init repo's `main`. Releases whose own pre-RL checkpoint is also public are: MiMo (Base, SFT), Klear-Reasoner (SFT), INTELLECT-MATH (SFT), Eurus-2 (SFT), DR.Kernel (cold-start), Nemotron-Cascade (SFT and every stage), Phi-4-reasoning, INTELLECT-3 (as the start for 3.1) and AllenAI (DPO). Not public:
- the SFT start of AceReason-1.1;
- Light-R1-14B-DS-SFT;
- the Magistral SFT checkpoint;
- the INTELLECT-3 SFT checkpoint;
- the AM-Thinking SFT checkpoint;
- the rStar2 SFT checkpoint;
- the MiMo-7B-RL-0530 SFT start;
- the MiMo-V2.6 cold start.

### Cited Findings
Zero-RL starts:
- ORZ is "based on Qwen2.5 Series of base models", meaning Qwen/Qwen2.5-0.5B/1.5B/7B/32B (claim) — [ORZ-7B card](https://huggingface.co/Open-Reasoner-Zero/Open-Reasoner-Zero-7B). ORZ-R1-Distill-Qwen-14B "applies the ORZ recipe" to DeepSeek-R1-Distill-Qwen-14B (claim) — [GitHub](https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero)
- DAPO-Qwen-32B "is trained based on Qwen2.5-32B", with `base_model: Qwen/Qwen2.5-32B` (claim) — [card](https://huggingface.co/BytedTsinghua-SIA/DAPO-Qwen-32B)
- SimpleRL-Zoo starts from the base models Mistral-v0.1-7B, Llama-3.1-8B, DeepSeek-Math-7B, Mistral-Small-24B, Qwen-2.5-0.5B–32B and Qwen-2.5-Math-7B (claim) — [GitHub](https://github.com/hkust-nlp/simpleRL-reason)
- MiMo-7B-RL-Zero is the "RL model trained from base model" (MiMo-7B-Base) and MiMo-7B-RL is the "RL model trained from SFT model" (MiMo-7B-SFT) (claim) — [card](https://huggingface.co/XiaomiMiMo/MiMo-7B-RL-Zero). MiMo-7B-RL-0530 follows an SFT scaled "from approximately 500K to 6M instances", and that SFT checkpoint is not among the 30 XiaomiMiMo repos (HF✓ absence) — [card](https://huggingface.co/XiaomiMiMo/MiMo-7B-RL)
- General-Reasoner starts from Qwen2.5-7B, Qwen2.5-14B or Qwen3-14B-Base (`base_model` tags). Note that the Qwen3 card's link text points to Qwen3-4B-Base by mistake (claim) — [card](https://huggingface.co/TIGER-Lab/General-Reasoner-Qwen3-14B)
- Oat-Zero starts from Qwen/Qwen2.5-Math-7B (claim) — [card](https://huggingface.co/sail/Qwen2.5-Math-7B-Oat-Zero)
- DeepMath-Zero-7B starts from `zwhe99/Qwen2.5-7B-orz`, a public Qwen2.5-7B copy carrying the ORZ template (claim plus HF✓ repo) — [card](https://huggingface.co/zwhe99/DeepMath-Zero-7B)
- DeepSeek-R1-Zero: "We directly apply reinforcement learning (RL) to the base model without … SFT"; the model is "trained based on DeepSeek-V3-Base" (claim) — [card](https://huggingface.co/deepseek-ai/DeepSeek-R1-Zero)

Distilled or SFT starts:
- Skywork-OR1-7B/32B/Math-7B: `base_model` is deepseek-ai/DeepSeek-R1-Distill-Qwen-7B or -32B (claim) — [HF API](https://huggingface.co/api/models/Skywork/Skywork-OR1-32B)
- AceReason-Nemotron-14B is "trained entirely through RL, starting from the DeepSeek-R1-Distilled-Qwen-14B" (claim) — [card](https://huggingface.co/nvidia/AceReason-Nemotron-14B)
- AceReason-1.1-7B: its RL start is "Our SFT-7B (starting point of RL)", built on Qwen2.5-Math-7B with the nvidia/AceReason-1.1-SFT data. No such model repo exists (HF✓ absence) — [card](https://huggingface.co/nvidia/AceReason-Nemotron-1.1-7B)
- AceMath-RL-Nemotron-7B starts from Deepseek-R1-Distilled-Qwen-7B (claim) — [card](https://huggingface.co/nvidia/AceMath-RL-Nemotron-7B)
- Nemotron-Cascade-8B is "post-trained from the Qwen3-8B-Base"; the pipeline is multi-stage SFT, then RLHF → IF-RL → Math RL → Code RL → SWE RL (claim) — [card](https://huggingface.co/nvidia/Nemotron-Cascade-8B). The SFT checkpoint and every intermediate stage are public as subfolders (HF✓) — [intermediate-ckpts card](https://huggingface.co/nvidia/Nemotron-Cascade-8B-Intermediate-ckpts). For Cascade-2-30B-A3B ("post-trained from Nemotron-3-Nano-30B-A3B-Base"), no SFT or stage checkpoint repo was found (HF✓ absence) — [card](https://huggingface.co/nvidia/Nemotron-Cascade-2-30B-A3B)
- Polaris-7B-Preview: `base_model` deepseek-ai/DeepSeek-R1-Distill-Qwen-7B (claim) — [card](https://huggingface.co/POLARIS-Project/Polaris-7B-Preview)
- Light-R1-14B-DS: the card's "Trained From" column says DeepSeek-R1-Distill-Qwen-14B. The paper, however, says it is "the SFT-only version … which subsequently undergoes an additional stage of GRPO", so the true start is an SFT checkpoint (Light-R1-14B-DS-SFT) that is not on HF (paper claim, HF✓ absence) — [paper](https://arxiv.org/html/2503.10460), [card](https://huggingface.co/qihoo360/Light-R1-14B-DS)
- Phi-4-reasoning-plus: "Starting from a strong SFT model, i.e., Phi-4-reasoning" (paper §4.2). The HF `base_model` tag lists microsoft/phi-4 instead (paper claim; tag HF✓) — [paper](https://arxiv.org/html/2504.21318v1), [HF API](https://huggingface.co/api/models/microsoft/Phi-4-reasoning-plus)
- Magistral Small: SFT on Magistral Medium traces from Mistral Small 3 Instruct (the card says "Mistral Small 3.1 (2503)"), then RL from "the best checkpoint on AIME'24". That SFT checkpoint is not on HF (paper claim, HF✓ absence) — [paper](https://arxiv.org/html/2506.10910v1), [card](https://huggingface.co/mistralai/Magistral-Small-2506)
- INTELLECT-MATH: "an SFT stage … fine-tuned on verified QwQ outputs, and an RL stage" using PRIME-RL. The SFT start is `PrimeIntellect/INTELLECT-MATH-SFT` (claim plus HF✓ repo) — [card](https://huggingface.co/PrimeIntellect/INTELLECT-MATH)
- Eurus-2-7B-PRIME: "the policy model and PRM are both initialized with the SFT model", i.e. `PRIME-RL/Eurus-2-7B-SFT` (claim) — [card](https://huggingface.co/PRIME-RL/Eurus-2-7B-PRIME)
- Klear-Reasoner-8B: `base_model` Kwai-Klear/Klear-Reasoner-8B-SFT, whose own base is Qwen/Qwen3-8B-Base (claim plus HF✓ repo) — [HF API](https://huggingface.co/api/models/Kwai-Klear/Klear-Reasoner-8B)
- DeepCoder-14B is "fine-tuned from DeepSeek-R1-Distilled-Qwen-14B" (claim) — [card](https://huggingface.co/agentica-org/DeepCoder-14B-Preview). DeepSWE-Preview is "trained on top of Qwen3-32B with thinking mode enabled", "RL, no SFT" (claim) — [card](https://huggingface.co/agentica-org/DeepSWE-Preview)
- rStar2-Agent-14B starts from Qwen3-14B-Base, which "requires instruction-following SFT before RL"; that SFT checkpoint is not published (claim) — [GitHub](https://github.com/microsoft/rStar)
- AReaL-boba-RL-7B: "We use R1-Distill-Qwen-7B as our foundation model" — [card](https://huggingface.co/inclusionAI/AReaL-boba-RL-7B). AReaL-boba²: "We use Qwen3 as our base model", with Qwen3-8B/14B as baselines (claim) — [card](https://huggingface.co/inclusionAI/AReaL-boba-2-8B-Open)
- AM-Thinking-v1: cold-start SFT on Qwen2.5-32B-Base, then two-stage GRPO; the SFT checkpoint is not on HF (claim, HF✓ absence) — [card](https://huggingface.co/a-m-team/AM-Thinking-v1)
- DR.Kernel: "Cold-start SFT" (`drkernel-14b-coldstart`), then "Multi-turn RL" (claim plus HF✓ repos) — [card](https://huggingface.co/hkust-nlp/drkernel-14b)
- P1-30B-A3B is "Built on Qwen3-30B-A3B-Thinking-2507" (claim) — [card](https://huggingface.co/PRIME-RL/P1-30B-A3B)
- GoLongRL lists only "Qwen3-30B-A3B (MoE)" as the trained family, without naming the exact variant (claim) — [GitHub](https://github.com/xiaoxuanNLP/GoLongRL)
- INTELLECT-2: "Base Model: QwQ-32B" (claim) — [card](https://huggingface.co/PrimeIntellect/INTELLECT-2)
- INTELLECT-3 is "post-trained from GLM-4.5-Air-Base using SFT followed by large-scale RL"; no SFT model checkpoint was found, only the SFT dataset (claim, HF✓ absence) — [card](https://huggingface.co/PrimeIntellect/INTELLECT-3)
- INTELLECT-3.1 is "a continued training of INTELLECT-3 with additional reinforcement learning" (claim) — [card](https://huggingface.co/PrimeIntellect/INTELLECT-3.1)
- Nemotron-3-Labs-Ultra-Math-RL: "Starting from the Nemotron-3-Ultra-GA checkpoint". The SFT and RL specialists are parallel, both derived from GA, so the released SFT model is *not* the RL start. The HF `base_model` is nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16 (paper claim; tag HF✓) — [paper](https://arxiv.org/html/2609.10712v1), [HF API](https://huggingface.co/api/models/nvidia/Nemotron-3-Labs-Ultra-Math-RL)
- MiMo-V2.6: "Cold start from self-correction" precedes RL, and the cold-start checkpoint is not released (claim, HF✓ absence) — [card](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL)

AllenAI:
- `Llama-3.1-Tulu-3-8B` and `Tulu-3.1-8B` both have `base_model: allenai/Llama-3.1-Tulu-3-8B-DPO`; `OLMo-2-0325-32B-Instruct` has `allenai/OLMo-2-0325-32B-DPO`; `OLMo-2-1124-7B-Instruct` has `allenai/OLMo-2-1124-7B-DPO` (HF✓ tags) — [HF API](https://huggingface.co/api/models/allenai/Llama-3.1-Tulu-3.1-8B)
- `OLMo-2-1124-13B-Instruct` has `base_model: allenai/OLMo-2-1124-13B-Instruct-RLVR2`: a three-round RLVR chain, DPO → RLVR1 → RLVR2 → Instruct (HF✓ tags) — [HF API](https://huggingface.co/api/models/allenai/OLMo-2-1124-13B-Instruct)

### Inferences
- Because no release pins the init's commit, re-check that each init repo's weights have not been re-uploaded since the RL run. Most Qwen and DeepSeek base repos have stable weights, but RL repos do change: see the DAPO and ORZ commit histories in §3.
- When an HF `base_model` tag and the paper disagree (Phi-4-reasoning-plus, Light-R1-14B-DS), trust the paper's description of the immediate RL start.

### Gaps
- No exact init revision is documented for any release.
- Two inits rest only on naming: Eurus-2-7B-PRIME-Zero from Qwen2.5-Math-7B, and Klear-AgentForge-8B from Klear-AgentForge-8B-SFT (the RL card is empty).
- The AceReason math-only then code-only staging was not verified from a primary source here.

## 3. Intermediate RL checkpoints: are there any, how many, at which steps, and how are they exposed?

### Takeaway
Step-level RL checkpoints at ≥ 7B are essentially an **AllenAI and Nemotron-Cascade phenomenon**. AllenAI exposes `step_N` revisions on Tulu 3, Tulu 3.1 and OLMo 2. Nemotron-Cascade-8B has five stage-level subfolders. INTELLECT-MATH has one extra step repo, and K2-Horizon has expert-level tags. None of the RL-Zero papers released step checkpoints, although SimpleRL-Zoo promised them. ORZ and DAPO hold earlier weight versions only in their git history.

### Cited Findings
Step-level revisions (AllenAI, DPO start → RLVR):
- `allenai/Llama-3.1-Tulu-3-8B`: branches `step_100` … `step_1000` every 100 (10) (HF✓) — [refs](https://huggingface.co/api/models/allenai/Llama-3.1-Tulu-3-8B/refs)
- `allenai/Llama-3.1-Tulu-3.1-8B`: 61 branches, `step_40` … `step_2440` every 40 (HF✓) — [refs](https://huggingface.co/api/models/allenai/Llama-3.1-Tulu-3.1-8B/refs)
- `allenai/Llama-3.1-Tulu-3-70B`: `step_120` only. `allenai/Llama-3.1-Tulu-3-405B`: `step_25`, `step_50`, `step_75` (HF✓) — [70B refs](https://huggingface.co/api/models/allenai/Llama-3.1-Tulu-3-70B/refs), [405B refs](https://huggingface.co/api/models/allenai/Llama-3.1-Tulu-3-405B/refs)
- `allenai/OLMo-2-0325-32B-Instruct`: 17 branches, `step_20` … `step_340` every 20. The card says "weights are saved every 20 training steps … in the revisions" (HF✓) — [refs](https://huggingface.co/api/models/allenai/OLMo-2-0325-32B-Instruct/refs), [card](https://huggingface.co/allenai/OLMo-2-0325-32B-Instruct)
- `allenai/OLMo-2-1124-7B-Instruct`: `step_60` … `step_360` every 60 (6) (HF✓) — [refs](https://huggingface.co/api/models/allenai/OLMo-2-1124-7B-Instruct/refs)
- `allenai/OLMo-2-1124-13B-Instruct`, `-Instruct-RLVR1` and `-Instruct-RLVR2` each expose the same 13 branch names, `step_60` … `step_780` (HF✓) — [RLVR1 refs](https://huggingface.co/api/models/allenai/OLMo-2-1124-13B-Instruct-RLVR1/refs)

Stage-level checkpoints:
- `nvidia/Nemotron-Cascade-8B-Intermediate-ckpts` (`main` only) has subfolders `Nemotron-Cascade-8B-SFT`, `-RLHF`, `-IFRL`, `-MathRL` and `-CodeRL`, each a full 16.4 GB safetensors checkpoint with its own tokenizer files (HF✓). The card adds that "the final model … is obtained after the concluding SWE RL stage" — [card](https://huggingface.co/nvidia/Nemotron-Cascade-8B-Intermediate-ckpts), [HF API](https://huggingface.co/api/models/nvidia/Nemotron-Cascade-8B-Intermediate-ckpts)
- `nvidia/Nemotron-Cascade-8B-Thinking`, `-14B-Thinking` and `Nemotron-Cascade-2-30B-A3B` have only `main` (HF✓) — [HF search](https://huggingface.co/api/models?search=Cascade&author=nvidia)
- INTELLECT-MATH: `PrimeIntellect/INTELLECT-MATH-step-47` is a separate repo; the card's table compares "Intellect-Math (Step 255)" (= `INTELLECT-MATH`) with Step 47 and the SFT model (HF✓ repos, claim for step labels) — [card](https://huggingface.co/PrimeIntellect/INTELLECT-MATH-step-47)
- `IFM/K2-Horizon-7B` tags (LLM360, brief):
  - Experts: `rl_math` (2,399 steps from `mid_4_10000`), `rl_code1` (601 steps from the math expert), `rl_code2` (1,499 steps), `rl_search`, `rl_tool-use` and `rl_merged`
  - There are no within-RL step tags, and the released `main` continues with SFT and FTPO after the RL merge
  - Sources: HF✓ tags, claim for steps — [refs](https://huggingface.co/api/models/IFM/K2-Horizon-7B/refs), [card](https://huggingface.co/IFM/K2-Horizon-7B)

Weight versions in git history (not labelled as intermediates):
- `Open-Reasoner-Zero-32B` `main` holds **two full 65.5 GB weight sets**:
  - a 14-shard set uploaded 2025-02-18 (commit `5dad505a9f`);
  - a 5-shard set uploaded 2025-03-25 (`b62406c4fa`), which `model.safetensors.index.json` points to.
  - (HF✓) — [tree](https://huggingface.co/api/models/Open-Reasoner-Zero/Open-Reasoner-Zero-32B/tree/main), [commits](https://huggingface.co/api/models/Open-Reasoner-Zero/Open-Reasoner-Zero-32B/commits/main)
- `Open-Reasoner-Zero-7B`: same 4-shard layout, but the shard hashes differ between revision `1eae6d1291` (2025-02-18) and `main` (2025-03-25 upload) (HF✓) — [commits](https://huggingface.co/api/models/Open-Reasoner-Zero/Open-Reasoner-Zero-7B/commits/main). The README lists the models as "ORZ-7B and 32B (updated)" (claim) — [GitHub](https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero)
- `DAPO-Qwen-32B`: an "update model weights" commit on 2025-05-10 replaced the 23-shard upload of 2025-04-09 (revision `5c6ffca125`) with 14 shards of different hashes (HF✓) — [commits](https://huggingface.co/api/models/BytedTsinghua-SIA/DAPO-Qwen-32B/commits/main)

Explicit absences:
- SimpleRL-Zoo's README says the authors "will also release some of the intermediate checkpoints". All ten Zoo repos have one branch and two commits, and no step repos exist under hkust-nlp (HF✓) — [GitHub](https://github.com/hkust-nlp/simpleRL-reason), [refs](https://huggingface.co/api/models/hkust-nlp/Qwen-2.5-32B-SimpleRL-Zoo/refs)
- `main` only, no step repos (HF✓): ORZ, DAPO, MiMo-7B (all four), Skywork-OR1 (all three), AceReason (all), Polaris, Phi-4-reasoning(-plus), Magistral, INTELLECT-2, INTELLECT-3, INTELLECT-3.1, Klear, DeepCoder, DeepSWE, rStar2, AReaL, AM-Thinking, General-Reasoner, ToRL, AZR, Oat-Zero, DeepMath, P1, GoLongRL, DR.Kernel, Eurus-2 and Nemotron-3-Ultra-Math — [refs outputs collected per repo](https://huggingface.co/api/models/PrimeIntellect/INTELLECT-2/refs)
- The DAPO GitHub mentions no intermediate checkpoints; it points to W&B logs only (claim) — [GitHub](https://github.com/BytedTsinghua-SIA/DAPO)

### Inferences
- For a reach-versus-step curve at ≥ 7B, the best candidates are:
  - Tulu-3.1-8B: 61 GRPO steps from a public DPO start, same Tulu chat template.
  - OLMo-2-0325-32B-Instruct: 17 GRPO checkpoints.
  - Nemotron-Cascade-8B: domain-wise stage checkpoints, useful for "does each RL domain add reach".
  - INTELLECT-MATH-7B: SFT → step 47 → step 255 under PRIME.
- The pre-update ORZ weights are a plausible second RL point for ORZ-7B and ORZ-32B, but this is an inference. The "updated" release is consistent with ORZ-32B's annealing stage on the hard 13k set; whether the February weights are an earlier point of the same run is not documented. The DAPO weight swap may be only a re-shard, which shard hashes cannot rule out.

### Gaps
- Whether ORZ's February weights and DAPO's April weights are distinct training points was not established; that would need a tensor-level diff.
- Whether OLMo-2-13B's RLVR1, RLVR2 and Instruct repos really share identical step grids, or reuse branch names, was not checked.
- Which OLMo-2-32B branch equals `main` (a third-party mirror says step 320) was not verified.

## 4. RL setup: algorithm, rewards, domains, steps and compute

### Takeaway
Math-only RLVR with rule-based answer checking dominates the 2025 RL-Zero releases. NVIDIA, Prime Intellect and Xiaomi moved to multi-domain RL (math, code, IF, SWE, agentic) in late 2025–2026. Step counts are short (90–600) where reported, except AllenAI's 1,000–2,440 and K2-Horizon's 2,399.

### Cited Findings
RL-Zero releases:
- ORZ: PPO training scripts (`orz_*_ppo.py`) and released critic models; compute was 16 nodes for 32B and 4 for 7B; "a tenth of the training steps" of DeepSeek-R1-Zero (claim) — [ORZ-7B card](https://huggingface.co/Open-Reasoner-Zero/Open-Reasoner-Zero-7B), [GitHub](https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero)
- DAPO: decoupled clipping, dynamic sampling and token-level policy-gradient loss; reaches 50 on AIME24 "with 50% training steps" of DeepSeek-R1-Zero-Qwen-32B; math only (claim) — [GitHub](https://github.com/BytedTsinghua-SIA/DAPO)
- SimpleRL-Zoo: GRPO with "only rule-based reward and GSM8K/Math datasets", about 8K examples; about 100 steps for 7B/14B (about 15 h on 2×8 H100) and about 1.5 days on 8×8 H100 for 32B (claim) — [GitHub](https://github.com/hkust-nlp/simpleRL-reason)
- MiMo-7B: 130K math and code problems, "only rule-based accuracy rewards", a "test difficulty driven code reward" and easy-problem re-sampling (claim) — [card](https://huggingface.co/XiaomiMiMo/MiMo-7B-RL)
- General-Reasoner: diverse-domain verifiable questions (WebInstruct-verified) with a 1.5B generative model-based verifier (claim) — [card](https://huggingface.co/TIGER-Lab/General-Reasoner-Qwen3-14B)
- Oat-Zero: MATH level 3–5 data, Qwen-Math template (claim) — [card](https://huggingface.co/sail/Qwen2.5-Math-7B-Oat-Zero)
- DeepSeek-R1-Zero: large-scale RL directly on the base (claim) — [card](https://huggingface.co/deepseek-ai/DeepSeek-R1-Zero)

From distilled or SFT starts:
- Skywork-OR1: customised GRPO, "multi-stage training pipeline coupled with adaptive entropy control"; math and code (claim) — [card](https://huggingface.co/Skywork/Skywork-OR1-32B)
- AceReason-Nemotron: math and code RL from R1-Distill (claim) — [card](https://huggingface.co/nvidia/AceReason-Nemotron-14B)
- Nemotron-Cascade-8B: sequential domain-wise RL (RLHF → IF-RL → Math RL → Code RL → SWE RL). The card adds that RLHF "boosts the model's complex reasoning ability" and that later RLVR stages "rarely degrade" earlier domains (claim) — [card](https://huggingface.co/nvidia/Nemotron-Cascade-8B)
- Phi-4-reasoning-plus (paper) — [paper](https://arxiv.org/html/2504.21318v1):
  - Algorithm and scale: GRPO in verl, 90 steps, 72,401-problem seed set with 64 problems per iteration (about 6K used), 8 responses each
  - Reward: length-aware accuracy plus a 5-gram repetition penalty
  - Compute: 32 H100s, 32K maximum length
- Magistral Small (paper) — [paper](https://arxiv.org/html/2506.10910v1):
  - Algorithm: modified GRPO (no KL, ε_high 0.3, zero-advantage groups filtered)
  - Reward: a 0.1 format gate, +0.9 for correctness, a soft length penalty, and a +0.1 language-consistency bonus
  - Domains and settings: math and code; batch 2048; step count not stated
- INTELLECT-2: "GRPO over verifiable rewards" with prime-rl, math and coding (claim) — [card](https://huggingface.co/PrimeIntellect/INTELLECT-2). INTELLECT-3: SFT, then large-scale RL with prime-rl and verifiers environments (claim) — [card](https://huggingface.co/PrimeIntellect/INTELLECT-3). INTELLECT-3.1 adds RL "on math, coding, software engineering, and agentic tasks" (claim) — [card](https://huggingface.co/PrimeIntellect/INTELLECT-3.1)
- INTELLECT-MATH: PRIME-RL recipe (implicit PRM), steps 47 and 255 — [card](https://huggingface.co/PrimeIntellect/INTELLECT-MATH). Eurus-2-7B-PRIME: implicit PRM plus outcome rewards with RLOO and 0.2–0.8 accuracy prompt filtering (claim) — [card](https://huggingface.co/PRIME-RL/Eurus-2-7B-PRIME); 592 steps per the INTELLECT-MATH comparison table (claim) — [card](https://huggingface.co/PrimeIntellect/INTELLECT-MATH)
- Klear-Reasoner: long-CoT SFT "distilled from DeepSeek-R1-0528", then GPPO (gradient-preserving clipping); math and code (claim) — [card](https://huggingface.co/Kwai-Klear/Klear-Reasoner-8B)
- DeepCoder-14B: GRPO+ (no KL, clip-high, offline difficulty filtering) with iterative context lengthening; about 24K code problems (claim) — [card](https://huggingface.co/agentica-org/DeepCoder-14B-Preview)
- DeepSWE: 200 RL steps on 4.5K R2E-Gym problems; sparse outcome reward of 1 only if all tests pass (claim) — [card](https://huggingface.co/agentica-org/DeepSWE-Preview)
- rStar2-Agent: GRPO-RoC, 510 RL steps in about a week on 64 MI300X, DAPO-17k English subset, Python tool calls via Code Judge (claim) — [GitHub](https://github.com/microsoft/rStar)
- AM-Thinking-v1: pass-rate filtering (0 < p < 1), then two-stage GRPO on math and code (claim) — [card](https://huggingface.co/a-m-team/AM-Thinking-v1)
- DR.Kernel: multi-turn RL (TRLOO + MRS + PR + PRS) in KernelGYM, rewarded on compilation, correctness and performance; Triton kernels (claim) — [card](https://huggingface.co/hkust-nlp/drkernel-14b)
- P1: multi-stage RL on physics-olympiad data using slime and verl (claim) — [card](https://huggingface.co/PRIME-RL/P1-30B-A3B)
- GoLongRL: GRPO/TMN-GRPO in verl on 16×8 GPUs; long-context tasks scored by EM, F1, math_verify, IoU, NDCG or ROUGE-L (claim) — [GitHub](https://github.com/xiaoxuanNLP/GoLongRL)

Large runs:
- Nemotron-3-Labs-Ultra-Math-RL: math proofs only; step count not stated (paper) — [paper](https://arxiv.org/html/2609.10712v1):
  - Algorithm: asynchronous NeMo-RL (PipelineRL-style) with DAPO-style dynamic sampling and truncated importance sampling
  - Reward: follows DeepSeekMath-V2 with α=1, β=0, scored by "judge nodes"
  - Batch: 128 prompts × 16 rollouts
  - Compute: 128 trainer, 128 inference and 16 judge nodes of 4 GB200 each
  - Data: 9,597 proof problems
- MiMo-V2.6: one mixed asynchronous GRPO run across coding, general agents, visual and cybersecurity, at "1,568 prompts × 16 rollouts per step", with group-wise agentic grading. MOPD2 distillation follows in the `-MOPD` models (claim) — [card](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL)

AllenAI and K2:
- Tulu 3.1 "switched from PPO to GRPO (no reward model)", so Tulu 3 itself used PPO (claim) — [card](https://huggingface.co/allenai/Llama-3.1-Tulu-3.1-8B). The RLVR data mixes math (GSM8K, MATH) and IF constraints — [HF API](https://huggingface.co/api/models/allenai/Llama-3.1-Tulu-3-8B). OLMo 2 32B was trained with GRPO (claim) — [Ai2 blog](https://allenai.org/blog/olmo2-32b)
- K2-Horizon RL experts: math 2,399 steps (29.1B tokens, 64K context); code 601 + 1,499 steps; search 59; tool use 39 (claim) — [card](https://huggingface.co/IFM/K2-Horizon-7B)

### Inferences
- For "create versus elicit" tests outside math, the multi-domain releases are the main options:
  - IF: Nemotron-Cascade IF-RL, Tulu/OLMo IF constraints
  - SWE: DeepSWE, Cascade SWE RL
  - Kernels: DR.Kernel
  - Long context: GoLongRL
  - Physics: P1
- Most other releases are math, or math plus code.

### Gaps
- Exact step counts are missing for ORZ, DAPO, Skywork-OR1, AceReason, MiMo, Klear, INTELLECT-2/3/3.1, P1, GoLongRL and Magistral.
- ORZ's reward design and Oat-Zero's algorithm were not confirmed from primary text in this pass.

## 5. Openness: is the RL prompt data public (dataset id), and is the reward or verifier code public (repo)?

### Takeaway
RL prompt data is public for most academic releases (DAPO, Skywork, Klear, DeepCoder, DeepSWE, Eurus, INTELLECT-2/3, Cascade, DR.Kernel, GoLongRL, General-Reasoner, AllenAI). It is not public for MiMo, Phi-4-reasoning-plus or Magistral. Reward code usually lives inside the training repo (verl, rLLM, prime-rl or verifiers, NeMo-RL, open-instruct) rather than as a standalone verifier.

### Cited Findings
Data published as HF datasets:
- DAPO: `BytedTsinghua-SIA/DAPO-Math-17k` (`datasets` tag); recipe in verl (claim) — [HF API](https://huggingface.co/api/models/BytedTsinghua-SIA/DAPO-Qwen-32B), [GitHub](https://github.com/BytedTsinghua-SIA/DAPO)
- Skywork-OR1: `Skywork/Skywork-OR1-RL-Data`; code at SkyworkAI/Skywork-OR1 — [card](https://huggingface.co/Skywork/Skywork-OR1-32B)
- AceReason: math RL data `nvidia/AceReason-Math`; SFT data `nvidia/AceReason-1.1-SFT` — [card](https://huggingface.co/nvidia/AceReason-Nemotron-1.1-7B)
- Nemotron-Cascade collection (HF✓): datasets `Nemotron-Cascade-SFT-Stage-1`, `-SFT-Stage-2`, `-SFT-SWE`, `-RL-Math`, `-RL-SWE`, `-RL-RLHF` and `-RM-Training`, plus the reward model `nvidia/Qwen2.5-CascadeRL-RM-72B` — [collection API](https://huggingface.co/api/collections/nvidia/nemotron-cascade)
- INTELLECT-2: `PrimeIntellect/Intellect-2-RL-Dataset`; code prime-rl — [card](https://huggingface.co/PrimeIntellect/INTELLECT-2). INTELLECT-3: datasets `INTELLECT-3-RL` and `INTELLECT-3-SFT`, with environments on the Environments Hub — [collection](https://huggingface.co/collections/PrimeIntellect/intellect-3), [card](https://huggingface.co/PrimeIntellect/INTELLECT-3)
- Eurus-2: `PRIME-RL/Eurus-2-RL-Data` — [card](https://huggingface.co/PRIME-RL/Eurus-2-7B-PRIME)
- Klear-Reasoner: `Kwai-Klear/KlearReasoner-MathSub-30K` and `KlearReasoner-CodeSub-15K` — [HF API](https://huggingface.co/api/models/Kwai-Klear/Klear-Reasoner-8B)
- DeepCoder: `PrimeIntellect/verifiable-coding-problems`, `likaixin/TACO-verified` and `livecodebench/code_generation_lite`; code agentica-project/rllm — [card](https://huggingface.co/agentica-org/DeepCoder-14B-Preview)
- DeepSWE: `R2E-Gym/R2E-Gym-Subset`; code rLLM — [card](https://huggingface.co/agentica-org/DeepSWE-Preview)
- AReaL-boba-RL-7B: `inclusionAI/AReaL-RL-Data` (boba_106k) — [card](https://huggingface.co/inclusionAI/AReaL-boba-RL-7B)
- General-Reasoner: `TIGER-Lab/WebInstruct-verified`, verifier model `TIGER-Lab/general-verifier` — [card](https://huggingface.co/TIGER-Lab/General-Reasoner-Qwen3-14B)
- DeepMath-Zero: `zwhe99/DeepMath-103K`; GitHub zwhe99/DeepMath — [card](https://huggingface.co/zwhe99/DeepMath-Zero-7B)
- DR.Kernel: `hkust-nlp/drkernel-coldstart-8k`, `drkernel-rl-data` and `drkernel-validation-data`, with training scripts in the repo — [card](https://huggingface.co/hkust-nlp/drkernel-14b)
- GoLongRL: `Kwai-Klear/GoLongRL` (23K samples), with training code — [GitHub](https://github.com/xiaoxuanNLP/GoLongRL)
- Nemotron-3-Labs-Ultra-Math-RL: `nvidia/Nemotron-Math-Proofs-v3-RL`; recipe on NVIDIA-NeMo/RL branch `imo-26-ultra-v3` — [card](https://huggingface.co/nvidia/Nemotron-3-Labs-Ultra-Math-RL)
- AllenAI: `allenai/RLVR-GSM-MATH-IF-Mixed-Constraints` (Tulu 3/3.1 and OLMo 2 32B), `allenai/RLVR-GSM` (OLMo 2 7B) and `allenai/RLVR-MATH` (OLMo 2 13B) — [HF API](https://huggingface.co/api/models/allenai/OLMo-2-1124-13B-Instruct)

Data published as GitHub files:
- ORZ: `orz_math_57k_collected.json`, `orz_math_72k_collection_extended.json` and `orz_math_13k_collection_hard.json`, with training scripts — [GitHub](https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero)
- SimpleRL-Zoo: SimpleRL-Zoo-Data (easy, medium and hard MATH/GSM8K tiers) — [GitHub](https://github.com/hkust-nlp/simpleRL-reason)
- Polaris: code and data in ChenxinAn-fdu/POLARIS; reward function from DeepScaleR — [card](https://huggingface.co/POLARIS-Project/Polaris-7B-Preview)
- rStar2-Agent: DAPO-17k plus preprocessing scripts in microsoft/rStar — [GitHub](https://github.com/microsoft/rStar)

Not public or not stated:
- MiMo: the card describes 130K problems but gives no dataset id — [card](https://huggingface.co/XiaomiMiMo/MiMo-7B-RL)
- Phi-4-reasoning-plus: release of the RL seed set is not stated — [paper](https://arxiv.org/html/2504.21318v1)
- Magistral: release of RL data or reward code is not stated — [paper](https://arxiv.org/html/2506.10910v1)
- BytedTsinghua-SIA QuestA-R1-7B and JustRL-R1-7B (2026): "does not currently document the full training dataset" — [card](https://huggingface.co/BytedTsinghua-SIA/JustRL-R1-7B)

### Inferences
- Where the verifier is public, reach can be re-scored with the same checker the RL used: DAPO, Skywork, DeepCoder, DeepSWE, the PRIME family, the AllenAI IF-constraint checker, General-Reasoner's model verifier, and Nemotron-Cascade's RM and RL data.
- That matters because the Olmo work found the IF checkers were heavily gamed.

### Gaps
- Not verified for MiMo: whether training or reward code exists.
- Not verified for Phi-4-plus: whether training or reward code exists.
- Not verified for Magistral: whether training or reward code exists.
- Not verified for QwQ: whether training or reward code exists.
- Not verified for AceReason: whether the code-RL data exists.
- Not verified for Cascade: whether IF-RL and Code-RL datasets exist; they are not in the collection.
- Not verified for INTELLECT-MATH: the RL prompt-set id.

## 6. Parity: do the RL start and the RL model share the tokenizer and chat template?

### Takeaway
Most pairs share vocab and template. The important exceptions:
- **ORZ** ships a custom base-model prompt template inside `chat_template`.
- **AZR** ships a plain-concatenation template.
- **Skywork-OR1-7B, AReaL-boba-RL-7B and Light-R1-14B-DS** drop R1-Distill's forced `<think>\n` generation prefix.
- **Nemotron-Cascade's SFT checkpoint** lacks the default system prompt and the automatic ` /think` tag that the RL stages add.
- **OLMo-2-32B DPO versus Instruct** differ in whether `bos_token` is prepended.
- **SimpleRL-Zoo-7B** and **Qwen2.5 → QwQ** change the template and eos.

### Cited Findings
Identical chat template, same vocab (`tokenizer.json` identical where comparable):
- DAPO-32B, SimpleRL-Zoo-14B/32B and SimpleRL-Zoo-Math-7B versus their Qwen bases. DAPO uses the Qwen2.5 template, with the training prompt "Solve the following math problem step by step … Answer: $Answer" — [DAPO card](https://huggingface.co/BytedTsinghua-SIA/DAPO-Qwen-32B)
- MiMo-7B-Base → RL-Zero and SFT → RL (`tokenizer.json` identical)
- Skywork-OR1-32B, AceReason-7B/14B, DeepCoder-14B and Polaris-7B versus R1-Distill
- Phi-4-reasoning → plus (identical)
- QwQ-32B → INTELLECT-2
- INTELLECT-MATH-SFT → INTELLECT-MATH, which uses a Tulu-style `<|system|>` template
- Klear-Reasoner SFT → RL: identical template and `tokenizer.json`, but `eos_token` changes from `<|endoftext|>` to `<|im_end|>` in `tokenizer_config`
- Qwen3-32B → DeepSWE (identical)
- Qwen2.5-14B → General-Reasoner-14B; Qwen3-14B-Base → General-Reasoner-Qwen3-14B (eos `<|endoftext|>` → `<|im_end|>`)
- Qwen3-30B-A3B-Thinking-2507 → P1-30B-A3B (identical) and → GoLongRL-30B-A3B
- drkernel coldstart → RL (identical)
- Eurus-2-SFT → PRIME (identical); Qwen2.5-Math-7B → Eurus-2-PRIME-Zero, Oat-Zero and ToRL-7B
- Qwen2.5-7B-orz → DeepMath-Zero-7B (identical; ORZ-style template)
- Nemotron-3-Ultra-550B → Ultra-Math-RL (identical)
- Tulu-3-8B-DPO → Tulu-3-8B (identical); Tulu-3-70B-DPO → 70B (identical); OLMo-2-1124-7B and 13B DPO → Instruct (identical)
- Mistral-Small-3.1-Instruct → Magistral-Small-2506: neither stores a `chat_template` in `tokenizer_config`, since they use mistral-common/tekken; vocab 131072 both
- Sources: parity run via [HF tokenizer_config files](https://huggingface.co/BytedTsinghua-SIA/DAPO-Qwen-32B/blob/main/tokenizer_config.json), [Klear SFT](https://huggingface.co/Kwai-Klear/Klear-Reasoner-8B-SFT/blob/main/tokenizer_config.json)

Template differs (HF✓ diff):
- Open-Reasoner-Zero-7B/32B: the template replaces Qwen ChatML with "A conversation between User and Assistant … User: {q}\nAssistant: <think>", in both the base-model prompt and the generation prefix. To test the base on equal terms, render that same template with the Qwen2.5 base (same vocab, 152064) — [ORZ tokenizer_config](https://huggingface.co/Open-Reasoner-Zero/Open-Reasoner-Zero-32B/blob/main/tokenizer_config.json)
- Absolute_Zero_Reasoner-Coder-7b: plain concatenation of message contents instead of Qwen2.5-Coder ChatML — [tokenizer_config](https://huggingface.co/andrewzh/Absolute_Zero_Reasoner-Coder-7b/blob/main/tokenizer_config.json)
- R1-Distill-derived models:
  - Skywork-OR1-7B: the only change from R1-Distill-Qwen-7B is removal of the trailing `<think>\n` after the assistant generation prompt. Skywork-OR1-32B keeps it.
  - AReaL-boba-RL-7B and Light-R1-14B-DS: their templates hash identically to Skywork-OR1-7B's, i.e. the same removal.
  - Source: [Skywork-OR1-7B tokenizer_config](https://huggingface.co/Skywork/Skywork-OR1-7B/blob/main/tokenizer_config.json)
- Nemotron-Cascade-8B — [SFT tokenizer_config](https://huggingface.co/nvidia/Nemotron-Cascade-8B-Intermediate-ckpts/blob/main/Nemotron-Cascade-8B-SFT/tokenizer_config.json):
  - SFT → RLHF: the RL-stage template adds the default system prompt "You are a helpful and harmless assistant." and appends ` /think` (or ` /no_think` if `enable_thinking=False`) to the last user turn.
  - CodeRL → final: the final also appends ` /no_think` to earlier user turns.
  - `tokenizer.json` is identical across all stages.
- INTELLECT-3 → 3.1: only the default system prompt changes, from "You are INTELLECT-3 …" to "You are INTELLECT-4 …"; `tokenizer.json` identical — [3.1 chat template](https://huggingface.co/PrimeIntellect/INTELLECT-3.1/blob/main/chat_template.jinja)
- OLMo-2-0325-32B-DPO → Instruct: the DPO template starts with `{{ bos_token }}` and the Instruct template does not; `tokenizer.json` identical — [Instruct tokenizer_config](https://huggingface.co/allenai/OLMo-2-0325-32B-Instruct/blob/main/tokenizer_config.json)
- Smaller template changes:
  - Qwen2.5-7B → SimpleRL-Zoo-7B: template changed and eos `<|endoftext|>` → `<|im_end|>`.
  - Qwen3-8B → AReaL-boba-2-8B: the RL model ships the Qwen3-*Base*-style template.
  - Qwen2.5-32B → QwQ-32B: template and eos changed.
  - DeepSeek-V3-Base → R1-Zero: template changed.
  - Source: [SimpleRL-Zoo-7B tokenizer_config](https://huggingface.co/hkust-nlp/Qwen-2.5-7B-SimpleRL-Zoo/blob/main/tokenizer_config.json)
- SimpleRL-Zoo trains with per-model prompt formats, "simpler prompts (abel) and complex prompts (qwen)", which are not necessarily the HF chat template (claim) — [GitHub](https://github.com/hkust-nlp/simpleRL-reason)

### Inferences
- For RL-Zero models, use the training prompt from the paper or code, not the HF `chat_template`. ORZ and DeepMath bake theirs in; DAPO, SimpleRL and Oat do not.
- For Nemotron-Cascade, render all stages with one fixed template (for example the final one with ` /think`), so stage comparisons are not confounded by the SFT checkpoint's missing default system prompt.

### Gaps
- `tokenizer.json` identity could not be established where one repo stores it in LFS and the other in git; vocab_size matched in every pair.
- Template parity for the Llama-3.1-8B and Mistral-7B SimpleRL-Zoo pairs (gated Meta base) and for the Klear-AgentForge pair (gated SFT) was not checked.

## 7. Practicalities: licence, release date and caveats

### Takeaway
Licences are mostly permissive (MIT or Apache-2.0); exceptions:
- NVIDIA: the NVIDIA Open Model License, or OpenMDW-1.1 for Ultra-Math.
- Meta: the Llama 3.1 licence for Tulu.
- No licence field: Skywork-OR1, ToRL, AZR, DR.Kernel and several cards.

Main caveats:
- PyTorch `.bin` weights (INTELLECT-MATH-SFT, AReaL).
- fp32 cold-start weights (DR.Kernel).
- Subfolder layout (Nemotron-Cascade).
- Gated SFT repo (Klear-AgentForge).
- Very large models (≥ 100B) that make pass@1,024 impractical.

### Cited Findings
- Licences from `cardData` (HF✓):
  - MIT: ORZ, MiMo, Phi-4, INTELLECT-3/3.1/MATH, DeepCoder, DeepSWE, rStar2, DeepMath, MiMo-V2.x, DeepSeek-R1-Zero
  - Apache-2.0: DAPO, most SimpleRL-Zoo, Polaris, Light-R1, Magistral, INTELLECT-2, Klear-Reasoner, Eurus-2, AReaL, AM-Thinking, General-Reasoner, Oat-Zero, P1, K2-Horizon, OLMo 2
  - "other" (NVIDIA Open Model License): AceReason, Cascade
  - OpenMDW-1.1: Ultra-Math
  - llama3.1: Tulu 3
  - None: Skywork-OR1, SimpleRL-Zoo-14B, ToRL, AZR, DR.Kernel, Polaris-V2
  - Sources: [HF API example](https://huggingface.co/api/models/nvidia/Nemotron-Cascade-8B), [Ultra-Math card](https://huggingface.co/nvidia/Nemotron-3-Labs-Ultra-Math-RL)
- Weight formats (HF✓):
  - `PrimeIntellect/INTELLECT-MATH-SFT` and the AReaL repos ship PyTorch `.bin` files — [HF API](https://huggingface.co/api/models/PrimeIntellect/INTELLECT-MATH-SFT)
  - `hkust-nlp/drkernel-14b-coldstart` is 59.1 GB for 14.77B parameters (fp32) — [HF API](https://huggingface.co/api/models/hkust-nlp/drkernel-14b-coldstart)
- QuestA-R1-7B and JustRL-R1-7B (BytedTsinghua-SIA, 2026-07) are described as "Direct-OPD" (on-policy distillation) research checkpoints mirrored from ModelScope, with no benchmark or dataset documentation (HF✓ card) — [card](https://huggingface.co/BytedTsinghua-SIA/QuestA-R1-7B)
- K2-Horizon's `main` is post-RL SFT plus FTPO anti-doom-loop; RL is not its final stage (claim) — [card](https://huggingface.co/IFM/K2-Horizon-7B)

### Inferences
- Practical shortlist for per-prompt pass@256–1,024 at 7–32B, with exact public start and matched template:
  - Tulu-3.1-8B and OLMo-2-32B (A)
  - Nemotron-Cascade-8B (A, stage-level)
  - INTELLECT-MATH-7B (A, one mid-point)
  - MiMo-7B RL-Zero and RL (B, with both a base and an SFT start from the same pretrain)
  - DeepSWE-32B and Phi-4-reasoning-plus (B)
  - Klear-Reasoner-8B and DR.Kernel (B, own SFT start)
  - DAPO-32B and ORZ-32B (B; ORZ with a second weight version in git history)
  - Skywork-OR1-32B (B)

### Gaps
- Licence terms of the gated Klear-AgentForge-8B-SFT were not read.
- Whether MiMo-7B or Phi-4 repos replaced weights after release (commit histories) was only spot-checked; the six most recent of MiMo-7B-RL's 11 commits are README and config edits.

### Summary table

| model | params | RL init (repo, exact?) | RL final | intermediate RL ckpts | RL domains | RL data public | verifier public | template parity | licence | fit |
|---|---|---|---|---|---|---|---|---|---|---|
| Llama-3.1-Tulu-3.1-8B | 8.03B | allenai/Llama-3.1-Tulu-3-8B-DPO (tag, exact repo) | allenai/Llama-3.1-Tulu-3.1-8B | 61 branches step_40…step_2440 (/40) HF✓ | math, IF (GRPO) | yes, RLVR-GSM-MATH-IF-Mixed-Constraints | open-instruct (not checked) | same HF✓ | llama3.1 | **A** |
| Llama-3.1-Tulu-3-8B | 8.03B | Tulu-3-8B-DPO (tag) | allenai/Llama-3.1-Tulu-3-8B | 10 branches step_100…1000 HF✓ | math, IF (PPO) | yes | not checked | same HF✓ | llama3.1 | **A** |
| Tulu-3-70B / 405B | 70B / 405B | -70B-DPO / -405B-DPO | allenai/Llama-3.1-Tulu-3-70B / -405B | step_120 / step_25,50,75 HF✓ | math, IF | yes | not checked | same (70B) HF✓ | llama3.1 | **A** (sparse) |
| OLMo-2-0325-32B-Instruct | 32.23B | allenai/OLMo-2-0325-32B-DPO (tag) | allenai/OLMo-2-0325-32B-Instruct | 17 branches step_20…340 (/20) HF✓ | math, IF (GRPO) | yes | not checked | bos_token differs | apache-2.0 | **A** |
| OLMo-2-1124-7B / 13B-Instruct | 7.30B / 13.72B | OLMo-2-1124-7B-DPO / 13B: DPO→RLVR1→RLVR2 chain | allenai/OLMo-2-1124-{7B,13B}-Instruct | 7B: step_60…360 (6); 13B: step_60…780 (13) in each of RLVR1, RLVR2, final HF✓ | math (GSM / MATH) | yes | not checked | same HF✓ | apache-2.0 | **A** |
| Nemotron-Cascade-8B | 8.19B | …Intermediate-ckpts/Nemotron-Cascade-8B-SFT subfolder (exact) | nvidia/Nemotron-Cascade-8B | 4 stage subfolders: RLHF, IFRL, MathRL, CodeRL HF✓ | RLHF, IF, math, code, SWE | partly (RL-Math, RL-SWE, RL-RLHF; IF/Code not found) | RM Qwen2.5-CascadeRL-RM-72B public | SFT ckpt lacks default sys prompt and /think; tokenizer same | NVIDIA OML | **A** (stage-level) |
| INTELLECT-MATH | 7.62B | PrimeIntellect/INTELLECT-MATH-SFT (exact, .bin) | PrimeIntellect/INTELLECT-MATH (step 255) | INTELLECT-MATH-step-47 (separate repo) HF✓ | math (PRIME) | not found | PRIME recipe (claim) | same HF✓ | MIT | **A** (1 mid-point) |
| Open-Reasoner-Zero-7B / 32B | 7.62B / 32.76B | Qwen/Qwen2.5-7B / -32B base (claim) | Open-Reasoner-Zero/Open-Reasoner-Zero-7B / -32B | none labelled; Feb-2025 weights in git history (7B rev 1eae6d1291; 32B 14-shard set) HF✓ | math (PPO, critics released) | yes (GitHub json) | training code GitHub | custom ORZ template baked in | MIT | **B** (A-candidate via git history) |
| DAPO-Qwen-32B | 32.76B | Qwen/Qwen2.5-32B (tag) | BytedTsinghua-SIA/DAPO-Qwen-32B | none; weights replaced 2025-05-10 (old rev 5c6ffca125) | math | yes, DAPO-Math-17k | verl recipe | same template; custom "Answer:" prompt | apache-2.0 | **B** |
| SimpleRL-Zoo (7B/14B/32B Qwen2.5, Math-7B, Llama-3.1-8B, Mistral-7B, Mistral-24B, DeepSeek-Math-7B) | 6.9–32.8B | respective base models (claim) | hkust-nlp/*-SimpleRL-Zoo | none (promised, not released) HF✓ | math (GRPO, ~100 steps) | yes (SimpleRL-Zoo-Data, GitHub) | GitHub | 14B/32B/Math same; 7B template and eos changed; abel/qwen prompts | apache-2.0 (14B none) | **B** |
| MiMo-7B-RL-Zero / MiMo-7B-RL | 7.83B | XiaomiMiMo/MiMo-7B-Base / MiMo-7B-SFT (claim) | XiaomiMiMo/MiMo-7B-RL-Zero / -RL | none | math, code | no id found | not found | same, tokenizer same HF✓ | MIT | **B** |
| MiMo-7B-RL-0530 | 7.83B | unreleased 6M-sample SFT | XiaomiMiMo/MiMo-7B-RL-0530 | none | math, code | no | no | – | MIT | **C** |
| Skywork-OR1-7B / 32B / Math-7B | 7.62B / 32.76B | R1-Distill-Qwen-7B / -32B (tag) | Skywork/Skywork-OR1-* | none | math, code | yes, Skywork-OR1-RL-Data | GitHub SkyworkAI/Skywork-OR1 | 32B same; 7B drops forced `<think>\n` | none listed | **B** |
| AceReason-Nemotron-7B / 14B; AceMath-RL-7B | 7.62B / 14.77B | R1-Distill-Qwen-7B / -14B (claim) | nvidia/AceReason-Nemotron-7B / -14B | none | math, code | math yes (AceReason-Math) | not checked | same HF✓ | NVIDIA OML | **B** |
| AceReason-Nemotron-1.1-7B | 7.62B | unreleased "Our SFT-7B" | nvidia/AceReason-Nemotron-1.1-7B | none | math, code | yes | not checked | – | NVIDIA OML | **C** |
| Phi-4-reasoning-plus | 14.66B | microsoft/Phi-4-reasoning (paper; HF tag says phi-4) | microsoft/Phi-4-reasoning-plus | none | math (90 GRPO steps) | no | no | same, tokenizer same HF✓ | MIT | **B** |
| Polaris-7B-Preview | 7.62B | R1-Distill-Qwen-7B (tag) | POLARIS-Project/Polaris-7B-Preview | none | math | yes (GitHub) | DeepScaleR reward | same HF✓ | apache-2.0 | **B** |
| INTELLECT-2 | 32.76B | Qwen/QwQ-32B (claim) | PrimeIntellect/INTELLECT-2 | none | math, code | yes, Intellect-2-RL-Dataset | prime-rl | same HF✓ | apache-2.0 | **B** |
| INTELLECT-3.1 (vs INTELLECT-3) | 106.85B | PrimeIntellect/INTELLECT-3 (claim) | PrimeIntellect/INTELLECT-3.1 | none | math, code, SWE, agentic | yes (INTELLECT-3-RL; Env Hub) | verifiers / prime-rl | only sys-prompt name differs | MIT | **B** |
| INTELLECT-3 | 106.85B | unreleased SFT of GLM-4.5-Air-Base | PrimeIntellect/INTELLECT-3 | none | multi-domain | yes | yes | – | MIT | **C** |
| Klear-Reasoner-8B | 8.19B | Kwai-Klear/Klear-Reasoner-8B-SFT (tag) | Kwai-Klear/Klear-Reasoner-8B | none | math, code (GPPO) | yes (MathSub-30K, CodeSub-15K) | GitHub | same; eos differs | apache-2.0 | **B** |
| Eurus-2-7B-PRIME / -PRIME-Zero | 7.62B | PRIME-RL/Eurus-2-7B-SFT / Qwen2.5-Math-7B (claim) | PRIME-RL/Eurus-2-7B-PRIME / -Zero | none | math, code (PRIME) | yes, Eurus-2-RL-Data | PRIME GitHub | same HF✓ | apache-2.0 | **B** |
| DeepCoder-14B-Preview | 14.77B | R1-Distill-Qwen-14B (tag) | agentica-org/DeepCoder-14B-Preview | none | code | yes | rLLM | same HF✓ | MIT | **B** |
| DeepSWE-Preview | 32.76B | Qwen/Qwen3-32B (tag) | agentica-org/DeepSWE-Preview | none | SWE agent (200 steps) | yes, R2E-Gym-Subset | rLLM / R2E-Gym | same, tokenizer same HF✓ | MIT | **B** |
| DR.Kernel 8B / 14B | 8.19B / 14.77B | hkust-nlp/drkernel-{8b,14b}-coldstart (exact; fp32) | hkust-nlp/drkernel-8b / -14b | none | Triton kernels (multi-turn) | yes | KernelGYM (claim) | same HF✓ | none listed | **B** |
| General-Reasoner (Qwen2.5-7B/14B, Qwen3-14B) | 7.6–14.8B | Qwen2.5-7B/14B, Qwen3-14B-Base (tags) | TIGER-Lab/General-Reasoner-* | none | multi-domain QA (model verifier) | yes, WebInstruct-verified | yes, TIGER-Lab/general-verifier | same HF✓ | apache-2.0 | **B** |
| P1-30B-A3B | 30.53B | Qwen3-30B-A3B-Thinking-2507 (claim) | PRIME-RL/P1-30B-A3B | none | physics | not checked | not checked | same HF✓ | apache-2.0 | **B** |
| GoLongRL-30B-A3B | 30.53B | Qwen3-30B-A3B (variant not named; template = Thinking-2507) | Kwai-Klear/GoLongRL-30B-A3B | none | long-context QA | yes, Kwai-Klear/GoLongRL | GitHub | same as Thinking-2507 | MIT | **B** (init inferred) |
| AReaL-boba-RL-7B; AReaL-boba-2-8B/14B(-Open) | 7.6–14.8B | R1-Distill-Qwen-7B; Qwen3-8B/14B (claim) | inclusionAI/AReaL-boba-* | none | math; code | yes, AReaL-RL-Data | AReaL GitHub | 7B drops `<think>`; boba-2 ships Qwen3-Base-style template | apache-2.0 | **B** |
| Oat-Zero-7B; DeepMath-Zero-7B; ToRL-7B; AZR-Coder-7b/14b | 7.6–14.8B | Qwen2.5-Math-7B; zwhe99/Qwen2.5-7B-orz; Qwen2.5-Math-7B; Qwen2.5-Coder-7B/14B (claims) | sail/…, zwhe99/…, GAIR/ToRL-7B, andrewzh/… | none | math; math; math with tool; self-play code | Oat: MATH L3-5; DeepMath-103K; AZR: none needed | GitHub repos (claim) | same, except AZR plain-concat | apache-2.0 / MIT / none | **B** |
| Nemotron-3-Labs-Ultra-Math-RL | 560.52B | Nemotron-3-Ultra-GA = NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16 (paper + tag) | nvidia/Nemotron-3-Labs-Ultra-Math-RL | none | math proofs (LLM judge) | yes, Math-Proofs-v3-RL | NeMo-RL branch | same HF✓ | OpenMDW-1.1 | **B** (impractically large) |
| DeepSeek-R1-Zero | 684.49B | DeepSeek-V3-Base (claim) | deepseek-ai/DeepSeek-R1-Zero | none | math, code | no | no | template changed | MIT | **B** (impractically large) |
| K2-Horizon-7B rl_math expert (LLM360) | 9.00B | tag mid_4_10000 (exact) | tag rl_math (then rl_code1 → rl_code2) | expert-level tags only | math, code, search, tool | yes (LLM360 data) | xllm (not checked) | not checked | apache-2.0 | **B** (brief; main is SFT-final) |
| Light-R1-14B-DS | 14.77B | unreleased Light-R1-14B-DS-SFT | qihoo360/Light-R1-14B-DS | none | math (220 GRPO steps) | partly | not checked | drops `<think>` | apache-2.0 | **C** |
| Magistral-Small-2506/2507/2509 | 23.6–24.0B | unreleased SFT on Mistral Small 3.x | mistralai/Magistral-Small-* | none | math, code | no | no | n/a (tekken) | apache-2.0 | **C** |
| AM-Thinking-v1 | 32.76B | unreleased SFT of Qwen2.5-32B-Base | a-m-team/AM-Thinking-v1 | none | math, code | queries claimed public | no | – | apache-2.0 | **C** |
| rStar2-Agent-14B | 14.77B | unreleased SFT of Qwen3-14B-Base | rstar2-reproduce/rStar2-Agent-14B | none | math with Python tool (510 steps) | DAPO-17k | rStar GitHub | – | MIT | **C** |
| QwQ-32B | 32.76B | Qwen2.5-32B (tag; multi-stage, cold-start not public) | Qwen/QwQ-32B | none | math, code, general | no | no | changed | apache-2.0 | **C** |
| Nemotron-Cascade-2-30B-A3B; Seed-Coder-8B-Reasoning; Kimi-Dev-72B | 31.6B; 8.25B; 72.7B | unreleased SFT / mid-train (bases: Nemotron-3-Nano-Base; Seed-Coder-8B-Base; Qwen2.5-72B) | nvidia/…, ByteDance-Seed/…, moonshotai/… | none | multi; code; SWE | partly (Cascade-2 SFT data) | – | – | NVIDIA / MIT / MIT | **C** |
| MiMo-V2.6-Flash-RL / Pro-RL | 310.8B / 1,024B | unreleased cold-start | XiaomiMiMo/MiMo-V2.6-*-RL | none | coding, agents, visual, cyber | no | no | – | MIT | **C** |
| Excluded | – | – | – | – | – | – | – | – | – | **X** |

The excluded releases (**X**), with reasons:
- SFT, DPO or distillation is the final stage: AReaL-boba-SFT-32B; Light-R1-32B, -32B-DS and -7B-DS; OpenReasoning-Nemotron; MiMo-V2.6-*-MOPD; QuestA-R1-7B and JustRL-R1-7B ("Direct-OPD"); the K2-Horizon `main`.
- No usable release: MiroMind-M1 (original repos absent); GAIR/daVinci-kernel-14B-SFT (empty repo); daVinci-kernel-*-RL, Archer-Math-30B, ASPO-Math-30B and Polaris-V2-235B (no card or init documented); Nemotron-Research-Tool-N1 (not found on HF).
- Below 7B: DeepScaleR-1.5B (main only); ProRL/BroRL 1.5B (tag v1, branches v2 and brorl); JustRL-DeepSeek-1.5B (21 step branches); Polaris-V2-Qwen3-4B (SFT and RL pair); Open-MOPD-SmolLM3-3B (stage checkpoints).
