---
written_on: 2026-11-07
written_by: "agent:claude"
papers:
  - zhangReplayFailuresSuccesses2025
---

# HiR: IFBench pass@k up to k = 1024 for an instruct start and two RL models (aggregate curves only)

Paper: [@zhangReplayFailuresSuccesses2025] (proposed key; Zhang et al., "Replay Failures as Successes: Sample-Efficient Reinforcement Learning for Instruction Following")
Source: https://arxiv.org/abs/2512.23457v1 (v1, 29 Dec 2025, the only version). Read in depth: §1–4 and Appendices A and C; proofs skimmed. Ledger ids are in `ledger/IF.tsv`.

## Setup card

| item | value |
|---|---|
| model size and start | 3–7B instruct models; the pass@k figure uses Qwen3-4B-Instruct-2507 [IF-97] |
| RL share of pretraining | not computed (not in rl-compute-share's table) |
| RL steps | 5 epochs over HIR-16K, about 16k multi-constraint queries; 8 rollouts at temperature 1.0 [IF-107][IF-99][IF-98] |
| domain | multi-constraint IF; 7 IF benchmarks, including IFBench with 294 prompts [IF-38] |
| largest k | 1024 [IF-36] |
| grading | instruction-level accuracy: all constraints must pass [IF-100]. IFBench uses its verifiers. A DeepSeek-V3.1 judge handles soft constraints in training and evaluation [IF-42]. Strict or loose: not stated (UNVERIFIED) |
| controls | symmetric budgets: yes, the same k grid for all three models [IF-36], but the sample count and estimator are not stated (UNVERIFIED); format tolerance: not stated; guessing: n/a; contamination: partly (training includes IFTrain [IF-40], which was built disjoint from IFBench [IF-8]) |

## Learnings

**Observed.**
- IFBench pass@1 (mean of five runs [IF-114] at temperature 0.6 with a 4,096-token cap [IF-41]): initial model 29.9, RL-CR 36.9, HiR 40.5 [IF-94]. RL-CR is plain RL with a constraint-level reward [IF-39].
- Figure 3a plots IFBench pass@k from k = 1 to 1024 for the initial model, RL-CR and HiR [IF-35][IF-36]. The curves are aggregate and drawn after training [IF-37].
- The text says only that HiR "consistently outperforms the initial model and RL-CR as k grows" [IF-34].
- My reading of the figure (figure only, UNVERIFIED):
  - initial model: ≈ 29.5 at k = 1, ≈ 61 at k = 1024;
  - RL-CR: ≈ 35.5 → ≈ 65.5;
  - HiR: ≈ 39.5 → ≈ 67.5.

  The curves never cross. The RL-CR minus initial gap shrinks from about 6 points to about 4.5.

**Paper interprets.** The pass@k curves show "an expanded capability ceiling", that is, a larger reasoning boundary [IF-33][IF-34]. The paper motivates HiR by initial models that fail constraints "even after many attempts" [IF-43].

**Our interpretation.**
- This is the closest published counter-example to "no large-k start-vs-RL boundary for IF": an IF-specific RL paper, on IFBench itself, comparing a start with standard IF-RL at k = 1024. In the figure, RL stays above the start.
- It is weaker than C3 on every axis that C3 adds:
  - Aggregate only, with no RL-only (reach) or start-only (reverse) counts. A gap of about 4.5 points is only the RL-only share minus the start-only share.
  - The sample count and estimator are not stated.
  - No fresh-draw confirmation, hand check or gaming audit. By C4, checker-only pass@k can overstate reach.
  - The start is an instruct model, and its RL mix includes IFTrain.
- The direction agrees with C3: no crossover by k = 1024. The setups are not comparable in size: aggregate vs per-prompt, 4B vs 32B, a 4,096-token cap, and a different prompt set.

## Evidence and limitations

- Every pass@k number above k = 1 is a figure reading. The ledger holds only the axis, the caption and the qualitative claim [IF-34][IF-36][IF-37].
- Only one start model is shown at large k. The other two appear only at pass@1.
- IFBench here has 294 prompts [IF-38]; C3 has 218.

## Connections and questions

- **C3: replicates** (direction only, aggregate, figure only). RL stays above the start up to k = 1024 on IFBench.
- **C4: unrelated** (no audit). Its pass@k is exactly the checker-only quantity that C4 says overstates reach.
- **C7: extends** (contrast). Aggregate curves, where C7 uses per-prompt symmetric confirmation and a hand gate.
- **C1, C2, C5, C6, C8: unrelated.**
- **Q3.** It refutes "no large-k start-vs-RL comparison for IF" in the aggregate sense, not as a per-prompt boundary. Open question: if the checkpoints are public (UNVERIFIED), they could be re-scored per prompt.
