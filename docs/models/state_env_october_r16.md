---
written_on: 2026-10-04
written_by: agent:claude
---

# Model card: `state_env_october_r16`

Named by Dan, 2026-10-04: [`state_env_october`](state_env_october.md) after **16** rounds of expert iteration instead
of 8. Nothing else changes. `state_env_october` keeps its definition (round 8); this is a separate model. Internal run
name for the extra rounds: `rl-continue`.

## What it is

| part | setting | from |
|---|---|---|
| everything up to round 8 | identical to `state_env_october` (network, pretraining, proof-state interface, K12 data, T1 settings) | `trajectory` |
| rounds 9–16 | the same T1 settings, continued from round 8: k 32, T 0.8, 4,495 `rl_targets`, K12 replay, 600 fine-tune steps per round | `rl-continue` |
| judge | Lean 4 core alone | |

Only `trajectory`'s three seeds were continued, so `best-state`'s original seeds have no round-16 version. Compare
against `trajectory`'s round 8, not `best-state`'s.

## Checkpoints (`hf://buckets/dan-pandori/nd-rl/`)

- `rl-continue/ckpts/rc/ladder/la_T1_best12_s{0,1,2}_r16.pt`.
- Round 12 of the same ladders: `rl-continue/ckpts/rc/ladder/la_T1_best12_s{0,1,2}_r12.pt`.
- Its round-8 start: `trajectory/ckpts/tj/ladder/la_T1_best12_s{0,1,2}_r8.pt`.

## Headline numbers (per seed s0 / s1 / s2; plain sampling, T 0.8; Lean alone)

| set | pretraining only | round 8 | **round 16** |
|---|---|---|---|
| textbook72, pass@256 (of 72) | 34 / 33 / 36 | 48 / 48 / 53 | **51 / 57 / 56** |
| holdout250, pass@256 | 196 / 205 / 200 | 240 / 236 / 237 | 240 / 246 / 239 |
| `rl_targets` solved (of 4,495) | 3,676 / 3,705 / 3,741 | 4,383 / 4,365 / 4,407 | 4,403 / 4,433 / 4,429 |
| long pool L_true 13–16, pass@64 (of 400) | | 377 / 362 / 367 | 382 / 370 / 376 |
| lengths ≥ 17, pass@256 (of 21) | | 21 / 20 / 20 | 21 / 21 / 20 |
| group C, pass@256 | 0 | 0.083 / 0.029 / 0.036 | 0.167 / 0.486 / 0.143 |

Notes on the rows:
- **`rl_targets`:** the "pretraining only" column is the pretrained model's own pass@32 (round 1's sample). The later
  columns count every target found by that round.
- **Group C:** per-seed theorems from textbook72 and holdout250 that neither the pretrained model nor round 8 solved
  (36 / 35 / 28).

Source: `rl-continue`'s `numbers.md` § rl-continue (reviewed, stands).

## Known limits

- **Seeds differ.** s1 learned the excluded-middle pattern (A ∨ ¬A) in rounds 13–14. That gave 34 of its 68 new
  targets, and 13 of its 16 new group C solves are classical. s0 solves none of those 34 targets and s2 solves 8.
  s0 and s2 gained only 3 group C theorems each.
- **Plain read-outs only.** It has no guided read-out yet; the policy default since 2026-10-04 is plain and guided.
- **Length cut-offs:** reads hit the step or action cap on 0.1–1 % of samples per stratum (single textbook72 rows up
  to 6.6 %). This is above the policy's 0.1 %, so group C counts are lower bounds.
- **The training targets are nearly used up:** 62–92 unsolved of 4,495.
- **Compute:** twice `state_env_october`'s RL rounds (attempts and fine-tune steps). Rounds 9–16 used 31,471–33,261
  A40 GPU-seconds per seed.
- Every limit of `state_env_october` also applies (one textbook72 problem overlaps K12; the long pools are nearly
  saturated).

## Experiments that used it

`rl-continue` (introduced it). Its round-16 checkpoints are also the control arm reserved for proposal 23.
