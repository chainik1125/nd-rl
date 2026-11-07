---
written_on: 2026-11-07
written_by: "agent:claude"
papers:
  - dragoiBeyondPasskBreadthDepth2025
---

# Cover@τ: a reliability threshold in place of large-k pass@k (it does not check soundness)

Paper: [@dragoiBeyondPasskBreadthDepth2025] (proposed key; Dragoi, Pintilie, Gogianu, Brad, Bitdefender)
Source: https://arxiv.org/abs/2510.08325v2 (v2, 21 Oct 2025), read in full. This paper defines Cover@τ;
"breadth-depth metrics" is its name for the same proposal. Ledger: `ledger/GG.tsv`.

## Setup card

| item | value |
|---|---|
| model size and start | Qwen2.5-7B-Instruct (instruct) for every RLVR arm (GRPO, GSPO, PPO, KL-Cov, Unlikeliness) [GG-9] |
| RL share of pretraining | not computed (not in rl-compute-share's table) |
| RL steps | OMEGA: 30 epochs at batch 500 [GG-97]; Reasoning Gym: 5 epochs (UNVERIFIED) |
| domain | maths: five OMEGA tasks (train IID, test OOD), Reasoning Gym algebra |
| largest k | about 2^13 per problem (Fig. 1 label; figure only, UNVERIFIED) |
| grading | final answer only [GG-10] |
| controls | symmetric budgets: yes (UNVERIFIED beyond figures); format tolerance: no; guessing: partly, a reliability threshold, no guess baseline [GG-2]; contamination: partly, procedural datasets [GG-98] |

## Learnings

**Observed.**
- Cover@τ is the fraction of problems on which at least a share τ of completions is correct [GG-1].
- Pass@k equals a Beta(1, k)-weighted average of the Cover@τ curve [GG-4]; at large k it records only whether a
  problem has nonzero success probability [GG-5]. Majority voting is Cover@0.5 [GG-7].
- On an OMEGA task whose test answers take only 30 values, the base reaches Pass@k = 1 [GG-6].
- On the OOD split the base leads at large k, but its coverage falls sharply once τ reaches 0.025 [GG-8].

**Paper interprets.** On discrete-answer maths, large-k Pass@k mostly reflects the chance of success over many
trials, not reasoning [GG-3]; guess-driven models "degrade rapidly as τ increases" [GG-2].

**Our interpretation.**
- Cover@τ handles guessing only by frequency: a problem counts if the model is right often enough. It neither models
  guessing nor separates a rare sound solution from a lucky answer, and it checks no reasoning [GG-10]. For gaming
  (C4) it is the wrong axis: a policy that games a checker reliably passes every τ.
- On C1: the base resolves the 17 RL-only AIME problems within 18–383 samples (typically ≈ 70), a per-sample rate of
  very roughly 0.003–0.06. Cover@τ would drop them at τ ≈ 0.02. Our hand check answers what Cover@τ cannot: 10 of
  the 17 are sound.
- Our computation, not the paper's: a uniform guesser over AIME's 1,000 answers hits a given problem within k tries
  with probability 1 − 0.999^k: 0.068 at k = 70, 0.32 at k = 383, 0.40 at k = 512. Over 17 problems at k ≈ 70 that
  is ≈ 1.2 expected hits, the order of the one guess our hand check found. Real guessing follows answer priors (see
  Yona et al. in `screen/GG.tsv`), so this is a floor on the threat, not its size.

## Evidence and limitations

- Every arm starts from an instruct model [GG-9]; there is no pretrained-base RL-zero arm.
- No hand check, no guess baseline, no per-problem reach table. "Guess-driven models degrade with τ" follows from
  the definition [GG-2]; it is not measured against known guesses.
- τ is a free parameter (the paper reports 0.2 and 0.8 and an AUC summary); nothing ties it to a guessing rate.
- Exact k and the Reasoning Gym steps are from figures or UNVERIFIED.

## Connections and questions

- **C1, extends.** Same concern (large-k base successes may be luck) [GG-3]; C1 answers it by reading the 17
  resolutions, not by a threshold that would also discard rare sound successes.
- **C7, extends.** "Guessing at large k on integer answers" is a published concern with a published metric
  [GG-1][GG-4]. C7's hand gate and final-answer grade are controls Cover@τ lacks [GG-10].
- **C4, unrelated** (final-answer metric, no IF) [GG-10]. **C2, C3, C5, C6, C8: unrelated.**
- Question: report Cover@τ at τ = 1/64 and 1/16 beside C1's reach counts.
- See [wen2025-cot-pass-at-k.md](wen2025-cot-pass-at-k.md) for the soundness-based alternative and its audit.
