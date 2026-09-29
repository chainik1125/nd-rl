---
written_on: 2026-09-29
written_by: agent:claude
---

# state-cap12: the proof state and cap-12 training proofs, together, move the long-proof frontier far past either alone

Status: **completed 2026-09-29 02:30 – 09:51 UTC (executor), independently reviewed 10:00–10:17 UTC. Verdict: stands,
with five rewordings.** No hard-constraint violation, nothing quarantined. Where the reviewer's wording differs from the
executor's, this summary uses the reviewer's.

Question: `state-env`'s SN-v2 (a policy that sees the Lean proof state and writes one step at a time, with
environment-assigned names) was trained on the cap-6 set. `cap-horizon`'s K12 (whole-proof `lean_seq`) was trained on a
flat cap-12 set. Each moved something. If SN-v2 is trained on K12's 155,000 cap-12 proofs, does the long-proof frontier
move further than with either lever alone, read on `long-pool`'s pool?

The run was on the fork (`dan-pandori/nd-takehome`), branch `dan_state-cap12` (from the fork's `dan` at `3cb0a0f8`). It is
not merged into the fork's `dan`.

- Review: [`review_state-cap12.md`](https://github.com/dan-pandori/nd-takehome/blob/c6b84c82/review_state-cap12.md)
  ([`fe3320f5`](https://github.com/dan-pandori/nd-takehome/commit/fe3320f5) §Recount, written before the reviewer read
  the executor's write-up; [`c6b84c82`](https://github.com/dan-pandori/nd-takehome/commit/c6b84c82) §Comparison /
  §Verdict). Reviewer scripts: `review_sc12/rv_*.py`.
- Executor write-up: [`run_state_cap12.md`](https://github.com/dan-pandori/nd-takehome/blob/c6b84c82/run_state_cap12.md),
  `numbers.md` § state-cap12, `log.md` § state-cap12. Results commit `a6e33e74`.
- Pre-registration: `preregistration/state-cap12.md`, commit `c34abbfb` at 02:34:31 UTC. The first pod (`sc-s0`) was
  created at 02:36:16 UTC, so the pre-registration came first.
- Brief: [`BRIEF_state-cap12.md`](../../docs/proposals/state-env/BRIEF_state-cap12.md).

## Configuration

- **Models.** All are 4 layers, d 256, trained from scratch. Checker: **Lean alone** (`lean_judge`).

  | label | checkpoint | params | format | training |
  |---|---|---:|---|---|
  | SN-cap12 frozen s0–s3 | `ckpts/sc12/stage1_SN12_s{0-3}.pt` | 3,216,384 | `lean_staten` | Stage-1 on K12 (`data/kh/train_k12.jsonl`, cap 12, flat 2–12, 155,000), `state_train.py --mode lean_staten --cap 12 --steps 6000 --recs 128` |
  | SN-cap12 T1 s0–s3 | `ckpts/sc12/ladder/la_T1_SN12_s{0-3}_r8.pt` | 3,216,384 | `lean_staten` | the above + 8 ladder EI rounds (`state_ladder_ei.py`, 8 × k 32, T 0.8, `rl_targets.jsonl`, replay K12) |
  | K12 whole-proof T1 s0–s1 (comparator, re-run here) | `ckpts/ladder/la_T1_K12_s{0,1}_r8.pt` | 3,214,336 | `lean_seq` | `cap-horizon`'s `stage1_k12_s{0,1}.pt` + 8 ladder EI rounds (`ladder_ei.py`, replay K12, `max_new` 512) |
  | SN-v2 cap-6 T1 / frozen s0–s1 (inherited) | `state-env/…/la_T1_SN_s{0,1}_r8.pt`, `stage1_SN_s{0,1}.pt` | 3,216,384 | `lean_staten` | cap-6 control set `train_depth3_f0_a1` (+ ladder for T1); read by `long-pool` |
  | K12 frozen s0 (inherited) | `cap-horizon/ckpts/kh/stage1_k12_s0.pt` | 3,214,336 | `lean_seq` | Stage-1 only, cap 12; read by `long-pool` |
  | C0 T1 s0–s1 (inherited; the reviewer supplied this label) | `ds-generator/ckpts/ladder/la_T1_c0_s{0,1}_r8.pt` | 3,214,336 | `lean_seq` whole-proof | cap-6 set + ladder, the ladder run under Lean ∧ `nd_verify` on 2026-09-24, re-read by `long-pool` under Lean alone |

- **Frozen arm.** A k = 256 re-read of each Stage-1 checkpoint replaces a `--no_train` ladder. This is a recorded
  deviation, made for cost; the transfer read-out uses the same estimator.
- **Gates on K12** (records; the reviewer did not re-run them). Gate 1 (round trip) and gate 3 (environment replay):
  0 / 155,000 failures. Gate 1b: 0 / 3,000 rejected. Gate 2 (Lean's state vs the renderer's): 1,500 / 1,500 cuts agree.
  The reviewer notes that gate 1b checks the `nd2lean` translation of the ND string, not the reassembled literal
  `lean_seq` text. That design is inherited from `state-env`, and every in-loop accepted sample is checked from its
  literal text anyway.
- **Long-pool read-out**, identical to `long-pool` pass 2: `lpool_reread.py`, k 256, T 0.8, seed 0, on
  `transfer_long_rr600.jsonl` (100 theorems per `L_true` bin 11–16) and the 70-theorem `transfer_long_ge17.jsonl`. State
  models: batch 2,048, `max_action` 512, `max_steps` 48. Whole-proof models: batch 1,024, `max_new` 1,536. Peak memory
  12–18.7 GB. `LEAN_GATE_DUMP` kept the literal text of every sample.
- **Primary quantity (pre-registered).** Q = the number of generator theorems of rr600 at `L_true` 13–16 (380 = 90 / 90 /
  100 / 100) with ≥ 1 Lean-accepted sample at k 256. MDD ≈ 119 at 4 vs 2 seeds (sd 37, pooled from the S and SN-v2 T1
  pairs; t-based, 80 % power, α 0.05). The brief's `L*` falsifier fires by construction, because SN-v2 cap-6 already reads
  `L*` ≥ 17 at the pool's ceiling. The pre-registration says so and replaces it with Q.
- **Pods.** `sc-s0`, `sc-s1` RTX 3090 $0.50/h (6.49 h, 6.74 h); `sc-s2`, `sc-s3` A40 $0.49/h (7.17 h, 7.18 h); `sc-k0`,
  `sc-k1` A40 $0.49/h (1.31 h, 1.30 h). Total **30.17 pod-hours, $14.91** of a 40 h / $20 budget, registered at
  02:30 UTC (`~/podhours.log`; the reviewer confirmed it with `podbudget`). All pods deleted.
- **Bucket** `hf://buckets/dan-pandori/nd-rl/state-cap12/`: `ckpts/sc12/` (4 Stage-1 + 32 ladder checkpoints),
  `ckpts/ladder/` (K12 T1 rounds), `artifacts/sc12/` (rounds, found files, re-reads, re-checks, gzipped Lean dumps with the
  literal texts), `data/kh/train_k12_ge7.jsonl.gz`, `figures/state_cap12.png`. K12 itself: `cap-horizon/data/kh/`.

## Observations (reviewed)

**Primary: Q (rr600 generator theorems at `L_true` 13–16, /380), k 256, final checkpoints.** The reviewer reproduced
every cell independently from the literal-text dumps.

| arm | Q per seed | mean | IQM [stratified-bootstrap 95 % CI] | ≥ 17 file /70 | `L*` rr600 |
|---|---|---:|---|---|---|
| SN-cap12 T1 | 233 / 295 / 320 / 317 | 291.3 | 306.0 [233, 320] | 25 / 42 / 42 / 46 | ≥ 17 on 4 / 4 |
| SN-cap12 frozen (Stage-1) | 134 / 212 / 228 / 216 | 197.5 | 214.0 [134, 228] | 10 / 15 / 19 / 18 | ≥ 17 on 4 / 4 |
| K12 whole-proof T1 | 79 / 72 | 75.5 | — (n = 2) | 3 / 2 | 16 / 16 |
| SN-v2 cap-6 T1 (inherited) | 102 / 28 | 65.0 | — | 5 / 0 | ≥ 17 / 15 |
| SN-v2 cap-6 frozen (inherited) | 3 / 0 | 1.5 | — | — | — |
| K12 frozen s0 (inherited) | 38 | — | — | — | — |

- **Compounding is supported by the pre-registered rule.** SN-cap12 T1 minus the better comparator (K12 T1) is
  **+215.8** (bootstrap 95 % [168, 243]), against an MDD of ≈ 119. Every SN-cap12 T1 seed (min 233) is above every
  comparator seed (max 102). An exact permutation test at 4 vs 2 cannot go below p = 2/15, so the significance rests on
  the parametric MDD.
- **The cap lever transfers to the state policy (secondary, not refuted).** SN-cap12 frozen scores Q 134–228 on 4 / 4 seeds,
  against 3 / 0 for SN-v2 cap-6 frozen. Even before RL, the frozen model beats every comparator's T1.
- **RL adds a paired +99 / +83 / +92 / +101** per seed (T1 − Stage-1, same seed).
- **Interaction: an observation, not a finding (reviewer's rewording).** The executor wrote "super-additive": over C0 T1's
  Q of 2, the state alone adds ≈ +63, the cap alone ≈ +74, and both together +289. The arithmetic is correct. But the
  2 × 2 was not pre-registered, and its cells are not matched: C0's ladder ran under Lean ∧ `nd_verify`, and the replay
  sets differ (cap 6 vs K12). Three of the cells also have n = 2 (SN-v2 cap-6 spans 28–102). And a pass@256 count is
  not an additive scale.

**Per bin (rr600) and the ≥ 17 file.**

| model | rr600 /600 | gen 13 / 14 / 15 / 16 % | textbook solved (of **73**) | step-cap hits rr600 / ≥ 17 |
|---|---|---|---|---|
| SN-cap12 T1 s0 / s1 / s2 / s3 | 375 / 454 / 476 / 472 | 66–86 / 71–89 / 63–91 / 47–78 | 29 / 30 / 27 / 28 | 0.13–0.31 % / 0.81–1.46 % |
| SN-cap12 frozen s0–s3 | 248 / 338 / 356 / 344 | 49–67 / 46–68 / 31–63 / 18–45 | 10 / 17 / 14 / 14 | ≤ 0.003 % / ≤ 0.045 % |
| K12 T1 s0 / s1 | 160 / 151 | 36 / 28 / 15 / 7 and 24 / 24 / 21 / 7 | 0 / 0 | (whole-proof `max_new` hit 0.03–0.04 %) |

- **Textbook theorems are no longer near-unsolvable for the state model.** SN-cap12 T1 solves 27–30 of the 73 textbook
  theorems in rr600, where every model `long-pool` read solved ≤ 6. The executor wrote "of 82"; the reviewer counted 73
  (`sc12_analysis.py:188` hard-codes 82). Q excludes textbook theorems, as pre-registered.
- **Step cap.** SN-cap12 T1 hits `max_steps` 48 on 0.13–0.31 % of rr600 samples and 0.8–1.5 % on the ≥ 17 file, above
  the policy's ≈ 0.1 % and against a pre-registered 0.0 %. A `max_steps` 96 re-read of **seeds 0–1 only** moved Q
  233 → 231 and 295 → 296, and did not change the ≥ 17 counts, which is re-draw noise. So on seeds 0–1 the cap does not
  hide proofs. Seeds 2–3 have about twice the cap rate and were not re-read at 96.
- **Term size** (the reviewer's measure: proof-term atoms of the literal text, type ascriptions removed) of the shortest
  accepted proof rises with `L_true` in every arm. Medians at bins 11–16 are 29 / 37 / 34 / 38 / 41 / 42 for SN-cap12 T1 s0
  (other seeds within ±3) and 28 / 29 / 31 / 31 / 35 / 36 for K12 T1. The bins therefore stay ordered in difficulty under
  Lean. This measure is not the pool's `label_term_size`.

**Original transfer pool (`transfer.jsonl`, 2,285).** SN-cap12 T1 in-loop cumulative solves 1,960 / 2,027 / 2,024 / 1,985,
with `L*` 13 on all four seeds. SN-cap12 frozen (k 256) solves 1,661 / 1,769 / 1,796 / 1,727, with `L*` 12 / 12 / 13 / 12. K12 T1
solves 1,703 / 1,634, with `L*` 13 / 13. The pool holds only 23 theorems at `L_true` 13–14, so `L*` 13 there rests on 3–7
solved theorems and reflects the pool's ceiling, not the model's frontier. Held-out greedy (p2, 5,000) for SN-cap12 Stage-1
is 0.969 / 0.977 / 0.978 / 0.974. These are the gate's Lean verdicts; no literal dump exists for this pass.

**Checks.**
- Accounting: 32 / 32 re-read dumps are set-equal to the `solved` rows. In the 6 ladders, 0 counted proofs have no
  accepted literal text.
- The reviewer's own Lean check accepted **3,315 / 3,315** counted literal texts (≥ 112 per arm and seed, covering
  re-reads and in-loop ladder proofs for both SN-cap12 and K12). It rejected **300 / 300** negative controls. The
  executor's own check covered the SN-cap12 re-reads only (3,280 / 3,280; 2,400 / 2,400 controls).
- Disjointness by renaming class, with a key that ignores both atom renaming and premise order: **0 classes shared between
  K12 / `rl_targets` and the long pool**. The executor's "0 shared classes" holds only for the long pool. K12 shares 22
  premise-permutation classes with the p2 held-out set (0.44 %) and 2 with the original pool. These overlaps are
  inherited from cap-horizon and the ladder pools, and the repo's order-sensitive `key` misses them.

**Expected vs outcome** (reviewer's completed list). Every miss ran above its pre-registered range. The
pre-registration gave 0.15 to a win beyond the MDD.
- SN-cap12 T1:
  - Q 60–200 → 233–320.
  - rr600 total 150–350 → 375–476.
  - Generator rates at 13–16 ≤ 50 % → 47–91 %.
  - ≥ 17 file 2–20 → 25–46.
  - Original pool 1,500–1,900 → 1,960–2,027.
- SN-cap12 frozen:
  - Q 15–80 → 134–228.
  - rr600 total 50–200 → 248–356.
  - Per-bin 5–25 / 5–25 / 3–20 / 1–12 % → 46–67 / 46–68 / 31–63 / 18–45 %.
  - `L*` 14–16 → ≥ 17.
  - ≥ 17 file 0–5 → 10–19.
  - Original pool 900–1,500 → 1,661–1,796.
- K12 T1:
  - Bin-13 rate 10–25 % → 24–36 %.
  - Original-pool `L*` 12 → 13.
- Step-cap hits 0.0 % → 0.13–0.31 %.
- Hits: K12 T1 Q (79 / 72, within 30–90), SN-cap12 T1 `L*` ≥ 17 on 4 / 4, held-out, and the gates.

## Limitations and next step

- The comparison is 4 vs 2 seeds. The headline clears the MDD by ≈ 97, but the only non-parametric bound is p ≥ 2/15.
- The replay set is part of the manipulation: SN-cap12 T1 and K12 T1 replay K12, while SN-v2 cap-6 T1 replays cap 6.
- The long pool is now near its ceiling for SN-cap12 T1: 47–91 % at 13–16 and 36–66 % on the ≥ 17 file. The ≥ 17 file
  has no per-theorem `L_true`, so this policy's frontier cannot be located on the current pool.
- The step cap question is open for seeds 2–3.
- Held-out greedy rests on gate verdicts, not literal re-checks. Gates 1–3 were read from records, not re-run.
- Outside this run: since repo-hygiene's `e844a8ea`, `artifacts/TEST_RUN_DONE` is untracked on the fork's `dan` and absent
  from new run worktrees. The fork's `test_run_once.sh` refuse-twice guard therefore would not fire there. (nd-rl's `gate1` was
  already adapted in `e0e2eee`.) This run did not touch the test set.

Next measurements (reviewer):
1. Re-read seeds 2–3 at `max_steps` 96 on rr600 and the ≥ 17 file, to close the step-cap question where the rate is
   highest.
2. To claim the interaction, run a matched 2 × 2: whole-proof vs state policy × cap 6 vs cap 12. All four cells should use
   the same ladder protocol under Lean alone, with ≥ 4 seeds each and the interaction pre-registered.
3. Build a pool labelled by length above 17, to find where this policy's frontier falls off.
