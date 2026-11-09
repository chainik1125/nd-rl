---
written_on: 2026-11-08
written_by: agent:claude
---

> Survey written 2026-11-08 at Dan's request: open LLMs ≥ 7B whose RL start, RL-final model and ideally intermediate RL checkpoints are public, for replicating the Olmo create-vs-elicit tests on other families. Synthesised by agent:claude from four research notes in [`notes/`](notes/), each checked against the Hugging Face API on 2026-11-08. Nothing here has been run; it is an inventory, not evidence.

# Tülu 3.1 and Nemotron-Cascade carry the cross-family replication

Two releases outside Olmo 3 let us rerun most of the Olmo create-vs-elicit design on a different pretrained base, on one H200 each. **Tülu 3.1 8B** is built on Llama 3.1. It has an exact, command-named DPO start, **61 GRPO step checkpoints** and RL on math plus instruction following (IF), with public prompts and verifier code. **Nemotron-Cascade-8B** is built on Qwen3-8B-Base. It has an exact SFT start plus **stage checkpoints after its IF-RL, math-RL and code-RL stages**, and it is a thinking model, so the transplant test can run between its RL stages. **OLMo 2 32B** adds step-level dose at our 32B scale. **MiMo-7B** gives the cleanest RL-from-base vs RL-from-SFT contrast on one pretrain, but has no intermediate checkpoints. Across all four notes, about twenty ≥ 7B lineages meet the full bar (exact or claimed-exact start, RL-final model, intermediate RL checkpoints). Only two of them come from outside AllenAI: Nemotron-Cascade-8B and the thin INTELLECT-MATH 7B. The large industry labs never meet it. Refs for 190 industry repos show only `main` and PR branches. Exact RL starts appear only for ≤ 15B specialist models, plus INTELLECT-3 → 3.1 (106B-A12B) and one 560B NVIDIA math model. Every 2025–26 flagship ships base + final at best, and many end in distillation or DPO rather than RL. The replication risks that matter most are template and EOS drift between start and RL model, RL prompt sets filtered to be hard for the start, `main` being an eval-selected step, and stage checkpoints that confound dose with domain.

Verification tags follow the notes and are never upgraded: **HF** means verified on Hugging Face (refs, file hashes, metadata); **card** means model-card prose only; **paper** means a technical report or blog; **script** means a public training script; **secondary** means press or aggregator pages. The RL-research note's "claim" tag is rendered here as card or paper, according to where the claim was made.

## Eight targets fit, and two cover most of the Olmo design

The ranking weighs, in order: an exact RL start, intermediate RL checkpoints, RL domains that match ours (math, IF, code), public RL data and verifier, template parity, and cost. All eight fit 1–2 H200s, roughly $3.6–21 per hour at our rates.

| # | Start → RL model (HF ids) | Base, size | Exact start | Intermediate RL ckpts | RL domains | RL data / verifier | Template parity | H200s |
|---|---|---|---|---|---|---|---|---|
| 1 | `allenai/Llama-3.1-Tulu-3-8B-DPO` → `allenai/Llama-3.1-Tulu-3.1-8B` | Llama 3.1, 8.03B | yes: the card's GRPO command names the DPO repo [card] | 61 branches `step_40`…`step_2440`; `main` = `step_1920` [HF] | GSM8K/MATH + IF constraints, GRPO, 2,048-token responses [card] | `allenai/RLVR-GSM-MATH-IF-Mixed-Constraints` [HF]; open-instruct verifiers [HF], training script at an unpinned older commit | template identical; `tokenizer.json` hash differs [HF] | 1 |
| 2 | `nvidia/Nemotron-Cascade-8B-Intermediate-ckpts` (subfolder `Nemotron-Cascade-8B-SFT/`) → `-RLHF/` → `-IFRL/` → `-MathRL/` → `-CodeRL/` → `nvidia/Nemotron-Cascade-8B` | Qwen3-8B-Base, 8.19B | yes: "RLHF initializes from the SFT checkpoint" [paper] | 4 stage subfolders, no within-stage steps [HF] | RLHF (RM), IF, math, code, SWE; GRPO without KL [paper] | Math, SWE and RLHF prompts plus the 72B RM [HF]; IF and code RL sets not located | tokenizer identical; SFT template lacks the RL stages' default system prompt and ` /think` suffix [HF] | 1 |
| 3 | `allenai/OLMo-2-0325-32B-DPO` → `allenai/OLMo-2-0325-32B-Instruct` | OLMo 2, 32.23B | yes: card command [card] | 17 branches `step_20`…`step_340`; `main` = `step_320` [HF] | GSM8K/MATH + IF, GRPO β 0.01, 2,048 tokens [card] | same RLVR mix [HF]; older open-instruct script | tokenizer identical; DPO template prepends BOS, RL template does not [HF] | 1 |
| 4 | `XiaomiMiMo/MiMo-7B-SFT` → `XiaomiMiMo/MiMo-7B-RL`; `XiaomiMiMo/MiMo-7B-Base` → `XiaomiMiMo/MiMo-7B-RL-Zero` | MiMo, 7.83B | named on the card for both lineages [card] | none [HF] | math + code, rule-based rewards [card] | no dataset id [HF], no reward code | identical within each pair; EOS `<|endoftext|>` on the Base line vs `<|im_end|>` on the SFT line [HF] | 1 |
| 5 | `IFM/K2-Horizon-7B@mid_4_10000` → `@rl_math` → `@rl_code1` → `@rl_code2`; sibling experts `@rl_search`, `@rl_tool-use` | K2 (IFM), 9.00B | yes per the training table [card]; tags exist [HF] | a sequential chain of expert endpoints [HF]; no in-run steps | math (2,399 steps), code (601 + 1,499), search, tool use [card] | no RL set identified; algorithm and rewards undocumented | tokenizer and template identical [HF] | 1 |
| 6 | `Qwen/Qwen2.5-32B` → `BytedTsinghua-SIA/DAPO-Qwen-32B`, `Open-Reasoner-Zero/Open-Reasoner-Zero-32B`, `hkust-nlp/Qwen-2.5-32B-SimpleRL-Zoo`; `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B` → `Skywork/Skywork-OR1-32B` | Qwen2.5, 32.76B | base by construction; Skywork via `base_model` tag [card] | none; ORZ-32B's git history holds a second, undocumented weight set [HF] | math (Skywork: math + code) | DAPO-Math-17k, ORZ JSON, SimpleRL data, `Skywork/Skywork-OR1-RL-Data`; verl and training repos | DAPO, SimpleRL-32B and Skywork-32B identical to their starts; ORZ bakes in a custom template [HF] | 1 |
| 7 | `microsoft/Phi-4-reasoning` → `microsoft/Phi-4-reasoning-plus` | Phi-4, 14.7B | per the report; the HF tag says `phi-4` [paper] | none; plus = step 90 of about 125 [paper] | math only, GRPO [paper] | not released | identical [HF] | 1 |
| 8 | `PrimeIntellect/INTELLECT-3` → `PrimeIntellect/INTELLECT-3.1` | GLM-4.5-Air, 106.9B / 12B active | "continued training of INTELLECT-3" [card] | none [HF] | math, code, SWE, agentic [card] | `PrimeIntellect/INTELLECT-3-RL` [HF], Environments Hub, verifiers [card] | identical except the default system-prompt name [HF] | 2 (213.7 GB BF16, tight) |

**Tülu 3.1 8B is the best first replication.** It is the only non-Olmo-3 base with an exact start, dense step checkpoints, public prompts and public verifiers, all on one GPU ([card](https://huggingface.co/allenai/Llama-3.1-Tulu-3.1-8B), [refs](https://huggingface.co/api/models/allenai/Llama-3.1-Tulu-3.1-8B/refs)). It also comes with two companions:
- **Tülu 3 8B** is a PPO run with 10 checkpoints from the same DPO repo. It gives an algorithm contrast at no extra cost, but its start is only claimed, because the paper tried two DPO starts ([Tülu 3 paper](https://arxiv.org/abs/2411.15124)).
- **`hkust-nlp/Llama-3.1-8B-SimpleRL-Zoo`** supplies a Llama-3.1-8B RL-from-base cell. It is math only, about 100 steps, from a different lab ([SimpleRL](https://github.com/hkust-nlp/simpleRL-reason)).

There are two limits. First, the post-training recipe is still AllenAI's: the open-instruct constraint checkers are probably the same lineage as Olmo 3's, which is an inference because the commit is unpinned. That makes our hand-gating directly portable, but the run is not independent of AllenAI's choices. Second, a 2,048-token non-thinking model trained on GSM8K and MATH will be near zero on AIME for both start and RL model, so math probes should be MATH-level and the easier OMEGA families.

**Nemotron-Cascade-8B is the best second target, for the domain question and the transplant.** Its stages are adjacent RL runs on single domains, so RLHF→IFRL, IFRL→MathRL and MathRL→CodeRL are three create-vs-elicit pairs with known domains ([intermediate ckpts](https://huggingface.co/nvidia/Nemotron-Cascade-8B-Intermediate-ckpts), [paper](https://arxiv.org/html/2512.13607v2)). The card reports IFEval strict-prompt scores of **70.8 (SFT) → 50.1 (RLHF) → 90.4 (IFRL)** [card]. That jump is exactly where our Olmo finding of gamed IF-checker passes should be retested. The RL stages share a template (verified for RLHF and CodeRL, not for IFRL or MathRL), so thinking transplants between stages are feasible. Comparing against the SFT start needs one hand-fixed rendering. The licence is the NVIDIA Open Model License.

| Olmo test | Tülu 3.1 8B | Cascade-8B | OLMo 2 32B | MiMo-7B | K2-Horizon-7B | Qwen2.5-32B set | Phi-4-plus | INTELLECT-3.1 |
|---|---|---|---|---|---|---|---|---|
| Math reach | yes (MATH-level) | yes (MathRL vs IFRL) | yes (MATH-level) | yes | yes | yes | yes | yes |
| IF reach | yes | yes (IFRL vs RLHF) | yes | no | no (the 375B has `rl_if`) | no | no | no |
| Code reach | no | yes (CodeRL vs MathRL) | no | yes | yes (code1, code2) | Skywork only | no | yes |
| Dose | 61 steps | 4 stages, domain-confounded | 17 steps | no | 3-link chain | no | no | no |
| RL-from-base cell | via SimpleRL-Zoo Llama (other lab) | no | no | yes, same pretrain | no (start is mid-training) | base starts vs distill start | no | no |
| Thinking transplant | no (non-thinking) | between RL stages | no (non-thinking) | yes, SFT → RL | likely (identical files; untested) | Skywork-OR1-32B vs R1-Distill-32B | yes | yes, with explicit system prompt |
| Re-score with the RL verifier | yes | partly (RM only) | yes | no | no | yes | no | yes |

**Recommendation.** For reach, hand-gating and dose in math and IF, run Tülu 3.1 8B first. Add SimpleRL-Zoo-Llama-3.1-8B as its base cell if budget allows. Run Nemotron-Cascade-8B second, for "which RL domain creates reach" and for the transplant. If the 32B scale itself matters, use OLMo 2 32B for dose, accepting that it is still a Tülu-style recipe. Use Skywork-OR1-32B for a non-AllenAI 32B thinking transplant, and DAPO-Qwen-32B for a 32B RL-from-base cell; neither has dose. For the SFT-vs-base question on one pretrain, MiMo-7B is cleaner than anything we have run.

Three in-family moves are also cheap:
- **Olmo 3 7B Think** (`allenai/Olmo-3-7B-Think-DPO` → `allenai/Olmo-3-7B-Think`, 55 checkpoints every 25 steps, exact start per script [script]) is a size-matched SFT-start partner for our 7B RL-Zero cell. It removes the 7B-vs-32B confound in our current base-vs-SFT contrast ([script](https://github.com/allenai/open-instruct/blob/main/scripts/train/olmo3/7b_think_rl.sh), [refs](https://huggingface.co/api/models/allenai/Olmo-3-7B-Think/refs)).
- **`allenai/Olmo-3-7B-RL-Zero-Code`** (29 checkpoints) covers the planned code domain inside the existing harness.
- **`allenai/tmax-27b`** has a start byte-identical to `Qwen/Qwen3.6-27B` and 5 checkpoints, but its RL was on terminal agents. It could serve as an off-domain negative control for math and IF reach; this is untested and is an idea, not a finding ([Tmax paper](https://arxiv.org/abs/2606.23321)).

## The four notes agree on repos but disagree on a dozen details

**LLM360 is now IFM.** The RL-research note's query `author=LLM360` returned zero repos, so it left K2-Think unverified. The open-initiatives note found that `huggingface.co/LLM360/*` 307-redirects to `IFM`, which lists 121 repos [HF]. The HF-verified fact stands. K2-Think v1 is `IFM/K2-Think`, with `base_model` `Qwen/Qwen2.5-32B` and no released SFT checkpoint, which makes it grade C ([K2-Think](https://huggingface.co/IFM/K2-Think)). K2-Horizon appears in both notes:
- **Template parity.** The RL-research note left it "not checked". The open-initiatives note verified identical `tokenizer.json`, `chat_template.jinja` and `generation_config.json` between `mid_4_10000`, `rl_math` and `rl_merged` [HF].
- **RL data.** The RL-research note lists RL data as "yes (LLM360 data)". The open-initiatives note found only pretraining and mid-training data, with no RL set in `ifm-ai/DATA_NAVIGATOR` ([DATA_NAVIGATOR](https://github.com/ifm-ai/DATA_NAVIGATOR)). This report follows the more specific check: no public RL set.
- **`main`.** Both agree that `main` is SFT + FTPO, not RL ([merge_info.json](https://huggingface.co/IFM/K2-Horizon-7B/blob/main/merge_info.json)).

**AllenAI facts.** Several RL-research gaps are settled by the AllenAI note's SHA comparisons:
- **OLMo 2 32B.** `main` has the same weights as `step_320` [HF]. The RL-research note had only a third-party mirror for this ([refs](https://huggingface.co/api/models/allenai/OLMo-2-0325-32B-Instruct/refs)).
- **OLMo 2 13B.** The three repos (RLVR1, RLVR2, final) reuse the branch names `step_60`…`step_780` for distinct weights [HF]. RLVR1's `base_model` wrongly names a 7B repo ([RLVR1 API](https://huggingface.co/api/models/allenai/OLMo-2-1124-13B-Instruct-RLVR1)).
- **Tülu 3 8B start.** The RL-research note grades it exact by tag; the AllenAI note downgrades it to claimed because the paper tested two DPO starts. This report keeps "claimed".
- **Tülu 3.1 parity.** The RL-research note says the tokenizer is the same, comparing template, `vocab_size` and EOS. The AllenAI note found the `tokenizer.json` hashes differ (`9400df98` vs `2ac5eaaa`) [HF]. Both are true. Nobody decoded the vocab, so this stays open.
- **Tülu 3 70B grade.** The RL-research note grades it A (sparse) and the AllenAI note B. With one intermediate (`step_120`), it is useless for dose, so this report lists it under B.
- **Verifier code.** The RL-research note left it "not checked". The AllenAI note found `ground_truth_utils.py` and `IFEvalG/` on open-instruct `main`, while the PPO/GRPO scripts used for Tülu 3, Tülu 3.1 and OLMo 2 return 404 there ([ground_truth_utils.py](https://github.com/allenai/open-instruct/blob/main/open_instruct/ground_truth_utils.py)).
- **OLMo 2 7B/13B data.** The cards' YAML and prose disagree, and the paper disagrees with the 7B card. The 13B RLVR1 stage includes IF constraints, which the RL-research table omits.

**Other cross-note conflicts.**
- **Nemotron-3-Labs-Ultra-Math.** The frontier note read only the abstract and called SFT → RL "probable". The RL-research note read the paper body: the SFT and RL specialists are *parallel*, both from the general Ultra checkpoint. So the RL start is `nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16`, not the released Math-SFT [paper] ([paper](https://arxiv.org/html/2609.10712v1)). The fuller reading wins. At 1.12 TB the model is impractical either way.
- **AceReason-7B tokenizer.** The RL-research note calls it "same". The open-initiatives note parsed the files and found identical vocab and template, but different `bos_token`, `eos_token` and `pad_token` settings and differently stored merges [HF]. Pin EOS.
- **DeepSeek-R1-Zero.** The frontier note grades it C because its rubric put RL-Zero pairs there. The RL-research note grades it B. For our purposes the base *is* the exact start, so B (impractical).
- **MiMo-7B parity.** The notes only look inconsistent: within each pair the files are identical, and the EOS differs between the Base line and the SFT line ([SFT config](https://huggingface.co/XiaomiMiMo/MiMo-7B-SFT/blob/main/tokenizer_config.json)).
- **Nemotron-Cascade template.** Both notes independently contradict the card's statement that "All intermediate checkpoints use the same chat template" for the SFT subfolder [HF].
- **Phi-4-reasoning-plus start.** All three notes agree to trust the paper over the `base_model` tag: the start is Phi-4-reasoning, not `phi-4` ([paper](https://arxiv.org/html/2504.21318)).

## About twenty grade-A lineages, all but two from AllenAI

Grades follow the notes:
- **A:** exact or claimed-exact start + RL-final model + intermediate RL checkpoints.
- **B:** start + RL model, no usable intermediates.
- **C:** base + RL model only.
- **X:** excluded.

The families already used (Olmo 3.1 Think 32B and its DPO start; Olmo 3 7B RL-Zero IF/Math; Olmo 3.1 RL-Zero Math) are omitted. Branch counts and parity results are [HF] unless marked otherwise.

### Grade A

| Lineage | Start → RL model | Intermediate RL ckpts | Domains | Start status | Parity | Size, H200s |
|---|---|---|---|---|---|---|
| Olmo 3 7B Think | `allenai/Olmo-3-7B-Think-DPO` → `allenai/Olmo-3-7B-Think` | 55, `step_0025`…`step_1375`; `main` ≠ `step_1375` | math, code, IF, chat (judge) | exact [script, HF]; which of two replicate runs was released is unknown [paper] | identical | 7.3B, 1 |
| Olmo 3 7B Instruct | `allenai/Olmo-3-7B-Instruct-DPO` → `allenai/Olmo-3-7B-Instruct` | 8, `step_050`…`step_400` | math, code, IF, chat | claimed; script points to an internal ckpt; RL on two DPO candidates [script, paper] | `tools=[]` branch differs | 7.3B, 1 |
| Olmo 3 32B Think (3.0) | `allenai/Olmo-3-32B-Think-DPO` → `allenai/Olmo-3-32B-Think` | 15, `step_050`…`step_750` | as 3.1 Think | claimed; first 750 steps of the 3.1 run [paper] | identical | 32B, 1 |
| Olmo 3 RL-Zero Code / General / Mix; 3.1 Code | `allenai/Olmo-3-1025-7B` → `allenai/Olmo-3-7B-RL-Zero-{Code,General,Mix}`, `allenai/Olmo-3.1-7B-RL-Zero-Code` | 29 / 8 / 19 / 33 | code; chat (judge); mixed; code | Mix exact [script]; others by lineage; General's script uses an internal long-context ckpt | base has no template; `tokenizer.json` differs | 7.3B, 1 |
| OLMo 2 7B | `allenai/OLMo-2-1124-7B-DPO` → `allenai/OLMo-2-1124-7B-Instruct` | 6, `step_60`…`step_360` | math (card: GSM8K; paper: + MATH, IF), PPO | claimed [HF, paper] | identical | 7.3B, 1 |
| OLMo 2 13B (3 RL stages) | `allenai/OLMo-2-1124-13B-DPO` → `-13B-Instruct-RLVR1` → `-RLVR2` → `allenai/OLMo-2-1124-13B-Instruct` | 13 per stage, distinct weights | math + IF → GSM8K → MATH, PPO | chain [HF]; RLVR1 metadata error | identical | 13.7B, 1 |
| OLMo 2 32B | see shortlist | 17 | math, IF, GRPO | exact [card] | BOS differs | 32.2B, 1 |
| Tülu 3 8B | `allenai/Llama-3.1-Tulu-3-8B-DPO` → `allenai/Llama-3.1-Tulu-3-8B` | 10, `step_100`…`step_1000` | math, IF, PPO | claimed [paper] | identical | 8B, 1 |
| Tülu 3.1 8B | see shortlist | 61 | math, IF, GRPO | exact [card] | `tokenizer.json` differs | 8B, 1 |
| Tülu 3 405B | `allenai/Llama-3.1-Tulu-3-405B-DPO` → `allenai/Llama-3.1-Tulu-3-405B` | `step_25`, `step_50`; `main` = `step_75` | MATH only, PPO | claimed | identical | 406B, impractical |
| Tmax 9B / 27B | `Qwen/Qwen3.5-9B` → `allenai/tmax-9b`; `Qwen/Qwen3.6-27B` → `allenai/tmax-27b` | 5 each; `main` = `step_200` / `step_160`; 9B ships rollouts and logprobs | terminal agents, Docker unit tests, DPPO | exact: start weights byte-identical [HF] | template edited to keep thinking; vision tower dropped | 9B / 27B, 1 |
| Tmax 8B | `allenai/tmax-sft-8b` → `allenai/tmax-8b` | 5; `main` = `step_300` | terminal | exact [card] | identical | 8.2B, 1 |
| DR Tulu 8B (+ No-RLER ablation) | `rl-research/DR-Tulu-SFT-8B` → `rl-research/DR-Tulu-8B`, `rl-research/DR-Tulu-No-RLER-8B` | 80 (`main` = `step_4000`); 23 | deep research with search; GPT-4.1-mini rubric judge | claimed [card] | identical | 8.2B, 1 |
| Nemotron-Cascade-8B | see shortlist | 4 stage subfolders | RLHF, IF, math, code, SWE | exact [paper] | SFT template differs | 8.2B, 1 |
| INTELLECT-MATH | `PrimeIntellect/INTELLECT-MATH-SFT` → `PrimeIntellect/INTELLECT-MATH-step-47` → `PrimeIntellect/INTELLECT-MATH` (step 255) | 1 | math (PRIME) | implied [card] | `tokenizer_config` identical; SFT ships `.bin` and no `tokenizer.json` | 7.6B, 1 |

Sources: [AllenAI HF listing](https://huggingface.co/api/models?author=allenai&limit=1000&sort=createdAt&direction=-1), [Olmo 3 report](https://arxiv.org/abs/2512.13961), [OLMo 2 paper](https://arxiv.org/abs/2501.00656), [tmax-9b refs](https://huggingface.co/api/models/allenai/tmax-9b/refs), [DR Tulu refs](https://huggingface.co/api/models/rl-research/DR-Tulu-8B/refs), [INTELLECT-MATH](https://huggingface.co/PrimeIntellect/INTELLECT-MATH).

### Grade B

| Lineage | Start → RL model | Domains | Start status | Main caveat | Size |
|---|---|---|---|---|---|
| Olmo 3.1 32B Instruct | `allenai/Olmo-3.1-32B-Instruct-DPO` → `allenai/Olmo-3.1-32B-Instruct` | math, code, IF, chat | claimed; internal script path [script] | `main` only despite the card; default system prompt differs [HF] | 32B |
| Tülu 3 70B | `allenai/Llama-3.1-Tulu-3-70B-DPO` → `allenai/Llama-3.1-Tulu-3-70B` | math, IF, PPO | claimed | only `step_120`; card and paper disagree on episodes and β | 70.6B, 2 GPUs |
| Tmax ablations ×6 | `Qwen/Qwen3.5-9B` → `allenai/qwen35-9b-{openthoughts,terminaltraj,cli-gym,swesmith,termigen,endless}` | terminal | [card] | `main` only, though the cards promise branches [HF] | 9B |
| BAR 2x7B | `allenai/BAR-2x7B-{Math,Code}-SFT` → `allenai/BAR-2x7B-{Math,Code}` | math; code | implied [paper] | frozen anchor FFN; no dedicated RL dataset | 11.6B MoE |
| Tülu 2.5 PPO | `allenai/tulu-2-13b` → `allenai/tulu-v2.5-ppo-13b-*` (9); `allenai/llama-3-tulu-2-8b` → `allenai/llama-3-tulu-v2.5-8b-*` (3) | chat, learned RM | lineage [HF] | not a verifiable reward; `.bin` weights | 13B / 8B |
| SAGE-MM ×3 | `allenai/SAGE-MM-{Qwen2.5-VL-7B,Qwen3-VL-8B,Molmo2-8B}-SFT` → `…-SFT_RL` | video tool agent | lineage; Molmo2 name mismatch | multimodal | 8–9B |
| K2-Horizon-7B experts | see shortlist; also `@rl_merged` | math, code, search, tool use | [card] table, tags [HF] | RL undocumented; `main` not RL-final | 9B |
| K2-Horizon-375B-A23B experts | `IFM/K2-Horizon-375B-A23B@mid_4_6000` → `@rl_reasoning`, `@rl_if`, `@rl_knowledge_work`, `@rl_search`, `@rl_tool-use` | reasoning, IF, knowledge work, search, tools | [card] | step counts not given; very large | 379B / 23B |
| K2-Think-V2 | `IFM/K2-V2-Instruct` → `IFM/K2-Think-V2` | math, code, science | `base_model` [HF]; revision unstated; recipe [secondary] | default `reasoning_effort` changes from medium to high | 72.6B, 2 GPUs |
| MiMo-7B | see shortlist | math, code | [card] | no RL data | 7.8B |
| Phi-4-reasoning-plus | see shortlist | math | [paper] | 90 steps | 14.7B |
| INTELLECT-3 → 3.1 | see shortlist | math, code, SWE, agentic | [card] | RL-on-RL | 106.9B / 12B |
| INTELLECT-2 | `Qwen/QwQ-32B` → `PrimeIntellect/INTELLECT-2` | math, code | [card] | merges stored differently; QwQ is itself RL'd | 32.8B |
| PaCoRe | `stepfun-ai/RLVR-8B-0926` → `stepfun-ai/PaCoRe-8B` | math, code | [card] | parallel-reasoning inference protocol | 8.2B |
| MiMo-VL | `XiaomiMiMo/MiMo-VL-7B-SFT(-2508)` → `XiaomiMiMo/MiMo-VL-7B-RL(-2508)` | vision-language | init not verified | VLM | 8.3B |
| AceReason / AceMath | `deepseek-ai/DeepSeek-R1-Distill-Qwen-{7B,14B}` → `nvidia/AceReason-Nemotron-{7B,14B}`; `nvidia/AceMath-RL-Nemotron-7B` | math, then code | [card] | EOS/BOS/pad config differs (7B) | 7.6 / 14.8B |
| Instella-MoE (B−) | `amd/Instella-MoE-16B-A3B-DPO` → `amd/Instella-MoE-16B-A3B-Think` | IF-RL, then on-policy distillation | exact [paper] | final stage is distillation; IF-RL expert unreleased; ResearchRAIL licence | 15.9B / 2.8B |
| Nemotron-3-Labs-Ultra-Math | `nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16` → `nvidia/Nemotron-3-Labs-Ultra-Math-RL` | math proofs, judge reward | [paper] | 1.12 TB | 560B / 55B |
| DeepSeek-R1-Zero | `deepseek-ai/DeepSeek-V3-Base` → `deepseek-ai/DeepSeek-R1-Zero` | math, code | [card] | template changed; ≥ 8 GPUs FP8 | 671B / 37B |
| Swallow | `tokyotech-llm/Qwen3-Swallow-{8B,30B-A3B,32B}-SFT-v0.2` → `…-RL-v0.2`; `tokyotech-llm/GPT-OSS-Swallow-{20B,120B}-SFT-v0.1` → `…-RL-v0.1` | math (Dolci-Think-RL-7B math subset, slime) | `base_model` [HF]; exactness not stated | parity checked only at 8B and 20B | 8–120B |
| Poro 2 Long Math | `LumiOpen/Llama-Poro-2-8B-Long-Math-Reasoning-SFT-Preview` → `…-RL-Preview` | math, English and Finnish | [card] | step 900 of 1,350 | 8B |
| OpenThoughts-Agent | `open-thoughts/OpenThinkerAgent-8B-ColdStartSFTForRL` → `open-thoughts/OpenThinkerAgent-8B-RL`; `open-thoughts/OpenThinker-Agent-v1-SFT` → `open-thoughts/OpenThinker-Agent-v1` | SWE / terminal agent | [card], `base_model` [HF] | RL step 45; needs a sandbox | 8.2B |
| Apertus v1.5 | `swiss-ai/Apertus-v1.5-{8B,70B}-SFT` → `swiss-ai/Apertus-v1.5-{8B,70B}-RL` | math, code, IF, multilingual, vision | stage mapping [paper] | gated (`auto`); refs and parity unchecked; long-context SFT placement unknown | 8.9 / 72B |
| Qwen2.5 / Llama / Mistral RL-from-base | `Qwen/Qwen2.5-32B` → `BytedTsinghua-SIA/DAPO-Qwen-32B`; `Qwen/Qwen2.5-{7B,32B}` → `Open-Reasoner-Zero/Open-Reasoner-Zero-{7B,32B}`; bases → `hkust-nlp/{Qwen-2.5-7B,Qwen-2.5-14B,Qwen-2.5-32B,Qwen-2.5-Math-7B,Llama-3.1-8B,Mistral-7B-v0.1,Mistral-Small-24B,DeepSeek-Math-7B}-SimpleRL-Zoo` | math | `base_model` / project [card] | custom training prompts; ORZ and DAPO weights replaced in git history | 7–32B |
| Other zero-RL | `sail/Qwen2.5-Math-7B-Oat-Zero`; `zwhe99/DeepMath-Zero-7B` (from `zwhe99/Qwen2.5-7B-orz`); `GAIR/ToRL-7B`; `andrewzh/Absolute_Zero_Reasoner-Coder-{7b,14b}`; `PRIME-RL/Eurus-2-7B-PRIME-Zero`; `TIGER-Lab/General-Reasoner-{Qwen2.5-7B,Qwen2.5-14B,Qwen3-14B}` | math; tool math; self-play code; multi-domain QA with a model verifier | [card] | ToRL and AZR have no README; AZR uses plain concatenation | 7.6–14.8B |
| Skywork-OR1 | `deepseek-ai/DeepSeek-R1-Distill-Qwen-{7B,32B}` → `Skywork/Skywork-OR1-{7B,32B}`, `Skywork/Skywork-OR1-Math-7B` | math, code | `base_model` [card] | 7B drops the forced `<think>\n`; no licence field | 7.6 / 32.8B |
| Own-SFT research pairs | `Kwai-Klear/Klear-Reasoner-8B-SFT` → `Kwai-Klear/Klear-Reasoner-8B`; `PRIME-RL/Eurus-2-7B-SFT` → `PRIME-RL/Eurus-2-7B-PRIME`; `hkust-nlp/drkernel-{8b,14b}-coldstart` → `hkust-nlp/drkernel-{8b,14b}` | math + code; math + code; Triton kernels | [card] | Klear EOS changes; DR.Kernel cold start is fp32 | 7.6–14.8B |
| Public-start research RL | `POLARIS-Project/Polaris-7B-Preview`; `agentica-org/DeepCoder-14B-Preview`; `inclusionAI/AReaL-boba-RL-7B`, `inclusionAI/AReaL-boba-2-{8B,14B}(-Open)`; `Qwen/Qwen3-32B` → `agentica-org/DeepSWE-Preview`; `PRIME-RL/P1-30B-A3B`; `Kwai-Klear/GoLongRL-30B-A3B` | math; code; math/code; SWE; physics; long context | [card]; GoLongRL init inferred | AReaL ships `.bin` and drops `<think>`; boba-2 ships a Qwen3-Base template | 7.6–32.8B |

Sources: [XiaomiMiMo listing](https://huggingface.co/api/models?author=XiaomiMiMo), [Swallow card](https://huggingface.co/tokyotech-llm/Qwen3-Swallow-8B-RL-v0.2), [Apertus v1.5 notes](https://huggingface.co/blog/swiss-ai/apertus-v15-release-notes), [Instella report](https://arxiv.org/html/2609.00791v1), [K2-Think-V2](https://huggingface.co/IFM/K2-Think-V2), [DAPO commits](https://huggingface.co/api/models/BytedTsinghua-SIA/DAPO-Qwen-32B/commits/main).

### Grade C and X, grouped

**Grade C: base and RL-final model, but the RL start is unreleased.** These groups cannot separate SFT effects from RL effects:
- **Flagships with a base:**
  - GLM-4.5 and 4.5-Air: general RL comes last ([GLM-4.5 report](https://arxiv.org/html/2508.06471));
  - DeepSeek V3.2;
  - Kimi K2 Instruct and Thinking;
  - inclusionAI Ring-mini-2.0, Ring-flash-2.0, Ring-lite and Ring-2.6-1T;
  - Step-3.5-Flash;
  - Arcee Trinity-Large-Thinking;
  - Poolside Laguna-M.1;
  - Magistral Small, whose start is an unreleased SFT of a released Instruct model ([Magistral paper](https://arxiv.org/html/2506.10910v1));
  - MiMo-V2.6-Flash-RL, where only the V2.5 base is a proxy.
- **"C?": RL-final not verified on the card.** Seed-OSS-36B, Hunyuan-A13B, ERNIE-4.5, Ministral-3-14B-Reasoning, Mistral Large 3, Gemma 3 and 4, Qwen3.5-35B-A3B and 9B, and DeepSeek V3.1.
- **Open-data labs that never released the SFT start:**
  - NVIDIA: Nemotron 3 Nano and Super, Nemotron 3.5 Lightning, Nemotron-Cascade-8B-Thinking and -14B-Thinking, Llama-Nemotron-Ultra-253B, AceReason-1.1-7B;
  - Granite 4.1 and 4.2, Falcon-H1R-7B, ZAYA1-8B, Moxin-7B-Reasoning, Tri-21B-Think;
  - INTELLECT-3 itself, MiMo-7B-RL-0530, K2-Think v1.
- **Research releases with an unreleased SFT start:** Light-R1-14B-DS (the paper contradicts the card's "trained from" claim, [paper](https://arxiv.org/html/2503.10460)), AM-Thinking-v1, rStar2-Agent-14B, QwQ-32B, Seed-Coder-8B-Reasoning, Kimi-Dev-72B and olmOCR-2.

**Grade X: excluded.** The reasons fall into five groups:
- **Final stage is not RL:**
  - on-policy distillation: DeepSeek V4/V4.1 ([V4 card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro)), probably MiMo-V2/V2.5, MiMo-V2.6-MOPD, Nemotron 3 Ultra and Cascade-2 (two notes grade these last two C, but both end in on-policy distillation), and QuestA/JustRL-R1-7B;
  - strong-to-weak distillation: Qwen3 ≤ 30B ([Qwen3 report](https://arxiv.org/html/2505.09388));
  - DPO last: Llama 4 ([Meta blog](https://ai.meta.com/blog/llama-4-multimodal-intelligence/));
  - checkpoint merges: Llama-Nemotron Nano and Super;
  - other: K2-Horizon `main` (SFT + FTPO), BAR-5x7B (router training).
- **No RL at all:** Olmo Hybrid 7B, AstaBrief, SERA, Marin, K2-Horizon-32B and MoVA, K2-V2-Instruct, Apertus 1.0 (QRPO is offline), llm-jp-4, Poro 2 Instruct, ALIA-40b, EuroLLM, OpenReasoning-Nemotron, Nemotron-Terminal, OpenR1/OlympicCoder, OpenThinker, ZAYA1-74B-preview and ERNIE-5.5-Flash-Base.
- **No base or start at all:**
  - GLM-4.6–5.3, Qwen3-235B, Qwen3-Next, Qwen3.5-397B and Qwen3.6/3.8;
  - Ring-1T, LongCat, MiniMax M1–M3, gpt-oss, step3, Step-3.7 and Step-5;
  - Mistral Large 4, Medium 3.5 and Small 4, and Solar-Open2;
  - Kimi K3 → K3.1 Pro, whose attention layers were swapped, so the released K3 is not the init ([card](https://huggingface.co/moonshotai/Kimi-K3.1-Pro)).
- **Below 7B:** OLMoE-1B-7B-0125 (6.92B with 44 RL checkpoints; it would otherwise be A), JustRL-DeepSeek-1.5B (21 step branches), ProRL/BroRL 1.5B, DeepScaleR, Qwen3-4B-SafeRL, DeepSeek-Prover-V1.5 (6.9B), Instella-3B-Math, SmolLM3, Polaris-V2-4B, Open-MOPD-SmolLM3, and tmax-4b and tmax-2b.
- **Undocumented or missing:** MiroMind-M1 (originals removed), daVinci-kernel, Archer-Math-30B, ASPO-Math-30B, Polaris-V2-235B, Kanana-2, HyperCLOVAX-SEED-Think, Motif-2, and the unreleased Olmo 4 / Olmo 3.2 / Tülu 4.

## Six caveats decide whether a reach gap is real

**Template and EOS drift makes the start look worse than it is.** Many pairs differ only in rendering, and a start that is prompted off-distribution will fail prompts it can solve:
- Nemotron-Cascade's SFT template lacks the RL stages' default system prompt and the automatic ` /think` suffix ([SFT tokenizer_config](https://huggingface.co/nvidia/Nemotron-Cascade-8B-Intermediate-ckpts/blob/main/Nemotron-Cascade-8B-SFT/tokenizer_config.json));
- OLMo 2 32B's DPO template prepends BOS and its RL template does not ([tokenizer_config](https://huggingface.co/allenai/OLMo-2-0325-32B-Instruct/blob/main/tokenizer_config.json));
- the Olmo 3.1 32B Instruct and INTELLECT-3.1 templates change the default system prompt;
- K2-Think-V2 changes the default reasoning effort;
- the EOS token changes between MiMo's Base and SFT lines, and in Klear SFT → RL;
- AceReason-7B's EOS, BOS and pad settings differ from R1-Distill;
- Skywork-OR1-7B, AReaL and Light-R1 drop R1-Distill's forced `<think>\n`;
- ORZ bakes a custom "User:/Assistant: <think>" template into the RL repo only ([ORZ tokenizer_config](https://huggingface.co/Open-Reasoner-Zero/Open-Reasoner-Zero-32B/blob/main/tokenizer_config.json));
- Tmax keeps thinking in multi-turn history and drops the vision tower.

The controls:
1. Render each prompt once, from an explicit template with the system prompt and thinking flags spelled out. Tokenize it with both tokenizers and assert that the token ids are equal. Decode the vocab and merges rather than trusting file hashes, which differ for Tülu 3.1, Tmax and RL-Zero Olmo.
2. Pin the stop tokens identically in vLLM, and report finish-reason and truncation rates per model.
3. For RL-Zero and from-base models, use the training prompt from the paper or code, not the HF `chat_template`.
4. Run one template-swap arm (start under the RL template vs under its own) to measure how much "reach" is rendering.

**RL prompt sets are pre-filtered to be hard for the start, which biases reuse as probes.** Olmo 3 dropped every prompt the DPO start passed at more than 62.5 % over 8 rollouts. The 32B run reused that 7B-filtered set, with active sampling on top ([Olmo 3 report](https://arxiv.org/abs/2512.13961)). Others filter the same way:
- Eurus-2 keeps prompts with 0.2–0.8 accuracy ([card](https://huggingface.co/PRIME-RL/Eurus-2-7B-PRIME));
- AM-Thinking keeps 0 < p < 1 ([card](https://huggingface.co/a-m-team/AM-Thinking-v1));
- DeepCoder filters offline by difficulty ([card](https://huggingface.co/agentica-org/DeepCoder-14B-Preview));
- K2-Think-V2's Guru set was "specially filtered according to the capabilities of K2-V2" ([dataset](https://huggingface.co/datasets/IFM/guru-RL-v1.5)).

Swallow and Instella train on subsets of Dolci-Think-RL-7B, the set behind Olmo 3 7B Think's filtered RL. If the released set is the post-filter one (an inference), it was filtered against *Olmo's* start, not theirs, so the filter and the start come apart there. A training prompt is therefore pre-selected on the start's 8-sample failures, which inflates the RL-only count relative to an unselected benchmark. Probe with held-out benchmarks. If training prompts must be used, select on one sample set and measure on an independent one. Record which model the filter was run against.

**`main` is often an eval-selected step, not the end of training.**
- **`main` equals an earlier branch:** OLMo 2 32B (`step_320` of 340), Tülu 3.1 (`step_1920` of 2,440), Tülu 405B (`step_75`), tmax-9b, tmax-27b and tmax-8b (`step_200`, `step_160`, `step_300`), DR Tulu (`step_4000`) [HF].
- **`main` matches no branch:** Olmo 3 7B Think and Instruct, Tülu 3 8B, OLMo 2 7B and RL-Zero Mix. Their final step is inferred from the papers.
- **Release is a chosen step:** Phi-4-reasoning-plus (step 90), Poro 2 (step 900 of 1,350, the "best checkpoint") and OpenThinkerAgent-8B-RL (step 45) [card, paper]. Magistral started RL from "the best checkpoint on AIME'24" [paper].

So, for a dose curve:
- deduplicate checkpoints by weight hash so `main` is not counted twice;
- treat post-`main` branches as an over-training tail;
- avoid the benchmark used for checkpoint selection, or report that selection.

**A claimed start is not always the init.**
- **Starts that may differ from the init:**
  - Olmo 3 Instruct's scripts point to internal checkpoints, and its RL ran on two DPO candidates ([7B Instruct script](https://github.com/allenai/open-instruct/blob/main/scripts/train/olmo3/7b_instruct_rl.sh));
  - the public Olmo 3 32B Think and RL-Zero General scripts also use internal paths;
  - Tülu 3 8B tested two DPO starts;
  - Phi-4-plus's `base_model` tag names the wrong model;
  - Light-R1's card names the wrong start;
  - Nemotron Ultra-Math's released SFT is a sibling, not the parent;
  - Apertus may or may not include long-context SFT in `-SFT`.
- **No release pins an init commit,** and ORZ and DAPO replaced their RL weights after release [HF].

The controls:
1. Hash-compare weights where a byte-identical start is claimed (Tmax does this).
2. Otherwise, check that the earliest RL step (for example `step_20` or `step_40`) has a small, monotonically growing logit divergence from the claimed start and is closer to it than to any alternative.
3. Report the result as "relative to the HF-declared start" when neither check is possible.

The set whose start is named in a training command or verified by hash is short: Olmo 3 7B Think, OLMo 2 32B, Tülu 3.1, Tmax and RL-Zero Mix.

**Stage checkpoints are not step checkpoints.** Several releases expose stage boundaries rather than steps inside one run:
- Nemotron-Cascade's subfolders;
- K2-Horizon's expert tags;
- OLMo 2 13B's three RL repos;
- MiMo's two lineages;
- INTELLECT-3 → 3.1;
- Apertus's stage repos.

At each such boundary the data, reward, domain and sometimes algorithm change together. A reach jump from IFRL to MathRL is a domain effect, not a dose. Later stages also move earlier domains: Cascade's IFEval goes 90.4 → 92.1 → 90.7 → 90.2 across MathRL, CodeRL and SWE-RL [card]. Use step series (Tülu 3.1, OLMo 2 32B, Olmo 3 7B Think, RL-Zero Code) for dose-response, and stage pairs only for "which domain's RL creates reach". K2-Horizon's sibling experts from one `mid_4_10000` start allow a new comparison, math-RL vs search-RL from an identical start. Its `rl_merged` tag adds a merge confound.

**Large industry labs essentially never release the exact RL start or intermediate RL checkpoints.**
- **No intermediates anywhere.** Refs for 190 public industry repos and 6 extra pairs show only `main`, `pr/*` and three incidental branches, with no tags [HF] ([MiMo-7B-RL refs](https://huggingface.co/api/models/XiaomiMiMo/MiMo-7B-RL/refs)).
- **Exact starts only for small or specialist models.** These are MiMo-7B, Phi-4-reasoning-plus, PaCoRe-8B, Qwen3-4B-SafeRL and Prover-V1.5, plus INTELLECT-3 → 3.1 and the 560B NVIDIA math model.
- **Flagships ship base + final at best.** SFT, cold-start and mid-training checkpoints are never released, and many flagships have no base at all. MiMo-V2.6 published live RL metrics but no step weights [secondary; paper] ([report](https://arxiv.org/html/2610.11959)).

So create-vs-elicit cannot be tested on any frontier-scale open model with its true start. A base-vs-final comparison there would attribute all of SFT and mid-training to RL.

## Conclusion

Cross-family replication is feasible, but its strongest form is narrower than it looks. Step-level dose with an exact start outside Olmo 3 exists essentially only for AllenAI post-training on other bases (Tülu 3/3.1 on Llama 3.1, OLMo 2, Tmax on Qwen). A positive replication on Tülu 3.1 would show the effect survives a change of pretrain, but not a change of RL recipe or checker. The only truly independent recipes with intermediates are Nemotron-Cascade's stage checkpoints and INTELLECT-MATH's single mid-point. That makes Cascade's IF-RL stage the most informative single experiment. If hand-gating finds the same checker gaming behind its 50.1 → 90.4 IFEval jump, the Olmo IF result generalises across both family and recipe. If it does not, the gaming finding is partly specific to AllenAI's checkers.

Two design improvements fall out of the inventory. Olmo 3 7B Think removes the size confound in our own base-vs-SFT contrast at almost no cost. MiMo-7B replicates that contrast on a second pretrain with the same RL data (inferred from the card's shared 130K-problem description). Pre-registration should name, for every cell, the template rendering, the model the RL prompts were filtered against, and whether the dose points are steps or stages. These three choices can produce or erase "reach" independently of anything RL did.
