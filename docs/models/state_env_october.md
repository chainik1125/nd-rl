---
written_on: 2026-10-04
written_by: agent:claude
---

# Model card: `state_env_october`

Named by Dan, 2026-10-04: Robbie's network size and architecture, plus the proof-state environment and the RL changes
that go with it. It is the project's strongest model as of 2026-10-04. Internal run names for it: **best-cap12**,
**best12**, "Robbie's recipe + proof state, cap 12".

## What it is

| part | setting | from |
|---|---|---|
| network | ALiBiGPT, 6 layers × 384, Peri-LN, 4 ALiBi + 4 NoPE heads, **9,560,832 parameters** (+1.88M MTP head, not saved) | Robbie's autoresearch incumbent 186 (`pretrain_best.py`, `robbie-experiments`) |
| pretraining | Muon + AdamW, multi-token prediction (weight 0.3), token-budget batches, **1,200 s on one A40** (≈ 24,000 steps) | Robbie's recipe; the budget was chosen by `best-state`'s pilot (1,200 s beat 300 s on held-out greedy, 0.963 vs 0.942) |
| interface | **proof-state environment**: the model sees the goal and the hypotheses in scope and writes one step at a time; the environment assigns names (`lean_staten`, SN-v2) | `state-env` (proposal 13) |
| data | **K12**: 155,000 synthetic ND proofs, flat lengths 2–12 lines (`cap-horizon/data/kh/train_k12.jsonl.gz`) | `cap-horizon`, `state-cap12` |
| RL | T1 expert iteration: 8 rounds, k 32, T 0.8, 4,495 `rl_targets`, K12 replay, 600 fine-tune steps per round | `state-cap12` / `best-state` |
| judge | Lean 4 core alone | `lean-judge` (2026-09-27) |
| evaluation | **plain** (independent attempts) and **guided** (`step_check.py` plus redraw without replacement) | `guided-tts`; the policy default since 2026-10-04 |

## Checkpoints (`hf://buckets/dan-pandori/nd-rl/`)

- `best-state/ckpts/bs/ladder/la_T1_best12_s{0,1,2}_r8.pt`: the original three seeds.
- `trajectory/ckpts/tj/ladder/la_T1_best12_s{0,1,2}_r8.pt`: three fresh seeds of the same recipe, plus every
  pretraining checkpoint (`trajectory/ckpts/tj/stage1_best12_s*_b1200_step*.pt`) and every RL round.
- Pretraining only (no RL): `best-state/ckpts/bs/stage1_best12_s{0,1,2}_b1200.pt`.

## Headline numbers (per seed, Lean alone)

- **From `best-state` (plain, r8):**
  - textbook72 at pass@256: **52 / 51 / 52** of 72;
  - Robbie's dev metric: 1,058 / 1,050 / 1,055 of 1,108;
  - holdout250: 239 / 237 / 237;
  - long pool L_true 13–16: 377 / 375 / 376 of 380;
  - lengths ≥ 17: 21 / 21 / 20 of 21.
- **From `guided-tts` (trajectory seeds, 256 attempts, not compute-matched):**

  | set | plain | guided |
  |---|---|---|
  | textbook72 (of 72) | 48 / 48 / 53 | **54 / 52 / 57** |
  | Charles's `candidate_v0` (of 87) | 67 / 65 / 71 | **75 / 71 / 75** |
  | Charles's `candidate_v1` (of 72) | 52 / 51 / 55 | 59 / 57 / 56 |
  | Charles's `batch3` (of 14) | 11 / 8 / 9 | 12 / 10 / 11 |
  | Charles's `hand_proved_14` (of 14) | 0 | 0 |

  At matched sampled tokens (64 plain-equivalent attempts), guided beats plain by +5.4 pp on textbook72 and +9.1 pp
  on long problems.
- **For comparison:** Robbie's combined model (whole proof, cap 6) scores 30–32 of 72 on textbook72 (his protocol)
  and 904 on the dev metric.

## Known limits

- **RL is close to saturating its targets:** 97.1–98.0 % solved by r8 (`rl-continue` is checking rounds 9–16).
- **The long pools are nearly saturated,** so its length frontier cannot be located.
- **Some theorems are not reached by anything:** about 30 of the 322 (`trajectory`'s group C), and every one of
  Charles's long hand-proved problems.
- **Compute:** its RL used 1.6–1.9× the GPU time of our 3.2M recipe. `compute-match` shows extra compute alone does
  not close the gap.
- **Training overlap:** one textbook72 problem (disjunctive syllogism) is in K12 with its premises in the other
  order.

## Experiments that used it

`best-state` (introduced it), `trajectory`, `rl-from-ckpt`, `mcts-a` (search gate failed), `mcts-b` (not run),
`grpo-best`, `compute-match` (control), `organism-analysis`, `claim-audit`, `guided-tts`, and `rl-continue`
(running).
