---
written_on: 2026-10-19
written_by: "agent:claude"
papers:
  - wuInvisibleLeashWhy2025
---

# The Invisible Leash: net support shrinkage in long-RL checkpoints at k up to 16384

Paper: [@wuInvisibleLeashWhy2025] (proposed key)
Source: https://arxiv.org/abs/2507.14843v5 (v5, the latest). Ids are in `ledger/C_threads.tsv`.

## Learnings

Models. The paper runs no long RL itself and re-evaluates public checkpoints. "We adopt ProRL-1.5B-v1 (Liu et al., 2025a) as our main RLVR method due to its robust long-horizon training framework". It also evaluates:
- ProRL-1.5B-v2;
- AceReason-Nemotron-7B/14B, Skywork-OR1-7B and Phi4-Reason-Plus-14B;
- OLMo-2-0425-1B and one vision-language model (C-LEASH-other_models, C-LEASH-olmo, C-LEASH-vlm).

Its own runs (SFT vs DAPO on Qwen2.5-Math-7B; DAPO on R1-Distill-1.5B to step 300) are small, with batch and group size not given.

Budgets: k ∈ {4096, 8192} on math, up to 16384 on Reasoning Gym, 1024/2048 on other non-math tasks, and 256 for the small runs (C-LEASH-kmax).

Verdict: "Across the evaluated model-level benchmark aggregates, the shrinkage of observed support outweighs its expansion".

Numbers:
- Questions gained / lost: ProRL v1 36 / 163; v2 48 / 175 (C-LEASH-prorlv1_row, C-LEASH-prorlv2_row).
- Retention SRR is 0.93–0.99 and new discovery NDR ≤ 0.02 (C-LEASH-srr_ndr).
- AIME24 pass@8192: base 93.3% vs v1 83.3% (C-LEASH-aime8192).
- leg_counting reverses: ProRL leads at k = 1, but the base leads at k = 1024.
- boxnet: base 0.345 vs ProRL 0.492 at k = 16384 (C-LEASH-boxnet16384).

Base-likelihood check. The authors measure the perplexity of DeepSeek-R1 and Sonnet 4 traces. "In every reported shrinkage comparison, ProRL assigns lower likelihood to these traces"; on expansion questions the perplexity falls (C-LEASH-ppl_exp).

Steps (DAPO, pass@256 every 30 steps):
- Olympiad: gains fall from 13 to 5, losses rise from 22 to 61, and pass@256 goes from 88.00% to 79.70% (C-LEASH-ckpt_olympiad).
- "On the harder AIME25 and Minerva, intermediate checkpoints achieve net expansion, but both finish with net shrinkage at step 300."

## Evidence and limitations

- RL length is varied only up to 300 steps. For long RL, the only contrast within one lineage is v1→v2 (2,000 → 3,000 steps): 12 more gains, 12 more losses, NSCR still −0.05. That reading of Tab. 1 is mine, not the paper's.
- The evaluation protocol differs from ProRL's: "These changes distinguish our evaluation protocol from the one used to report ProRL’s original Reasoning Gym scores." This removes format-only expansion. The authors suggest the expansion that remains may be format alignment or subskill composition (C-LEASH-expansion_hyp).

## Connections and questions

- This is the strongest evidence here that long open RL still narrows at very large k. Long, though, means 2–3k steps on a distilled 1.5B model (about 16k H100-hours for v1; [liu2025-prorl-compute.md](liu2025-prorl-compute.md)). The paper does not document the RL scale of the 7–14B checkpoints.
- At a fixed base and dataset, SFT never loses net support and DAPO shrinks on four of six benchmarks (C-LEASH-sft_vs_dapo), in line with [matsutani2025-rl-squeezes.md](matsutani2025-rl-squeezes.md).
- BroRL (co-author Fang Wu) argues that a large N removes "knowledge shrinkage", but only in simulation ([hu2025-brorl.md](hu2025-brorl.md)). The claim is untested at large k.
