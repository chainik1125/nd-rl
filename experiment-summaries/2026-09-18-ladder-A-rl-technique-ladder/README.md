---
written_on: 2026-09-18
written_by: agent:claude
---

# Ladder A: six expert-iteration variants against a true-length frontier (Phase A, T1–T6)

Status: completed 2026-09-18 18:27 UTC (autonomous executor session, five RTX 3090 pods plus
one for a resume, ≈ $39 of which ≈ $32 was idle pods during a session outage) and independently
reviewed 2026-09-18 19:15 UTC. Review verdict: no hard-constraint violation, no quarantine;
every count, rate, interval, `L*`, reachability figure and budget figure reproduces from the raw
files with independent code (21 claim rows; all 12,515 transfer + 46,471 target proof records
re-verified, 0 rejections). Three of the executor's readings go past the data and are reworded
below: **"the wall is at 11"**, **"T6 beats T1, paired p < 0.001 twice"**, and the
interpretation sentence that reports base p < 1e-5 without the rule-family fact beside it.
Numbers and wording below are the reviewer's. Run id `ladder-A`.

Review: [`review_ladder-A.md`](https://github.com/dan-pandori/nd-takehome/blob/ab629c7/review_ladder-A.md)
(commits `39d374c` phase 1 blind recount, `ab629c7` phase 2 claims table and verdict; reviewer
code `rv_recount.py`, `rv_labelcheck.py`, `rv_reach.py`, `rv_disjoint.py`; outputs
`review_out/`). Executor write-up:
[`run_ladder_A.md`](https://github.com/dan-pandori/nd-takehome/blob/ab629c7/run_ladder_A.md),
tables [`ladder.md`](https://github.com/dan-pandori/nd-takehome/blob/ab629c7/ladder.md), numbers
`numbers.md` §ladder-A, on branch `dan_ladder_a` of `dan-pandori/nd-takehome` (read the review
first: the executor's files were **not** edited after the review, so they still carry the
wording the verdict asks to change). Proposal:
`docs/proposals/2026-09-17-rl-technique-ladder.md` (PR #4, branch `dan_proposals`).

Question: which expert-iteration-family technique raises the **true** proof-length frontier `L*`
of the take-home's cap-6 `abs` Stage-1 model the most, on a transfer pool it never trains on, at
an equal sample budget, against a frozen control at equal attempts?

## Configuration

- **Code and data revision.** `dan-pandori/nd-takehome`, branch `dan_ladder_a`; pre-registration
  [`preregistration/ladder-A.md`](https://github.com/dan-pandori/nd-takehome/blob/ab629c7/preregistration/ladder-A.md)
  committed 01:14 UTC (`b3845eb`), pools 01:33 (`fb7036c`), amendment 01:40 (`4cfbf6e`; its text
  says "01:50"), earliest counted arm `args.json` 01:42:53 — gate 0 holds (reviewer). Final
  executor commit `d8c8a83`. `nd_verify` identical to `origin/main` (tree `9437bb72`);
  `artifacts/TEST_RUN_DONE` untouched (blob `1d5cf064`); no evaluation file read by training code.
- **Pools** (`data/ladder/POOLS.md`): strict long generator (`make_coverage_sets.py gen --long
  --min 7 --max 16`, unchanged generator knobs, seed 7000) + textbook-schema instances
  (`textbook_pool.py`), labelled `L_true` 7–14 by `minlen.py` at bound 14. **Transfer pool 2,285
  theorems** (1,525 generator + 760 textbook; bins 7: 300, 8: 300, 9: 1,010, 10: 451, 11: 99,
  12–14: 125), never sampled in training. **RL-target pool v2: 4,495** (amended from 2,295 after a
  killed round 1 found a foothold of 16 / 2,295; 1,500 generator theorems at `L_true` 7 and 700
  at 8 moved in from the reserve; transfer pool and predictions unchanged; disclosed).
- **Common protocol.** Init `ckpts/stage1_abs.pt`; 8 rounds; per round 600 fine-tune steps, lr
  3e-4, mix = accumulated target proofs (≤ 4 per theorem, ×4) + 20,000 retained Stage-1 records;
  sampling T = 0.8, `max_new` 512; **1,150,720 target samples per arm (4,495 × 256)**; transfer
  8 × 32 attempts per theorem, cumulative. Seeds 0 and 1 per rung. `L*` = largest L with ≥ 5
  distinct theorems solved at `L_true` ≥ L. Base reachability: log p under `stage1_abs.pt` at
  T = 0.8, marginalised over start index; lines at 1e-5 and 1/256.
- **Rungs** (driver `ladder_ei.py`; launch `bash pod/la/arm2.sh <name> --seed <s> …`, queues
  `pod/la/q1c.sh`…`q5.sh`): frozen (`--no_train`); **T1** plain EI, k = 32; **T2** `--alloc
  difficulty` (32–579 samples per target); **T3** `--relabel` (hindsight by-products ≥ 7 lines);
  **T4** `--alloc window` (155–767 per target; window `L_true` 11–13 from round 5); **T5**
  `--inject_pool` (2,505 / 2,509 cap-6 generator records, round 4 only, ×2); **T6** two siblings,
  `--k 16 --siblings 2 --share_dir …`, found sets pooled each round, transfer 2 × 16 per round
  with the union counted. Per-arm `args.json` are in the bucket. (Librarian's reading of the
  `args.json` files, not checked by the review: T1, T2 and frozen s1 ran at `--batch 512`, the
  other arms at 384 — consistent with the reviewer's note that same-seed arms do not share
  round-1 samples.)
- **Pods.** `la-1`…`la-5` RTX 3090 $0.50/h, created 01:15–02:14 UTC, last arm finished 04:42,
  deleted by Dan at 17:20; `la-6` 17:53–18:22 for the T3 s0 resume. Pod time per arm 55–112 min
  (two to three arms per GPU). Cost ≈ $39.3 (78.6 pod-hours), estimate ≈ $12–20, cap $50.
- **Artifacts.** `hf://buckets/dan-pandori/nd-rl/ladder-A/{data/ladder,artifacts/ladder,ckpts/ladder}`
  (64 / 837 / 112 files per `STATUS.md`; all per-arm `found_<r>`, `mix_<r>`, `novelty_*`, 112
  round checkpoints). The reviewer did not pull a bucket listing (row 21: "not derived").
  Figures (executor's, not checked by the review, linked not copied):
  [`ladder_frontier_counts.png`](https://github.com/dan-pandori/nd-takehome/blob/ab629c7/figures/ladder_frontier_counts.png),
  [`ladder_solve_transfer.png`](https://github.com/dan-pandori/nd-takehome/blob/ab629c7/figures/ladder_solve_transfer.png),
  `ladder_lstar.png`, `ladder_reachability.png`, `ladder_solve_targets.png`.

## Observations

Observed results first; the reviewer's readings and what is not supported follow.

**Frontier (round 8; T6 = union of its two siblings).**

| arm | transfer solved / 2,285 (s0, s1) | transfer `L*` | solved at `L_true` ≥ 9 | ≥ 10 | ≥ 11 | targets solved / 4,495 | targets `L*` |
|---|---|---|---|---|---|---|---|
| frozen | 22, 24 | 7, 7 | 0, 0 | 0, 0 | 0, 0 | 426, 408 | 8, 8 |
| T1 plain EI | 612, 623 | 10, 10 | 335, 346 | 35, 34 | 1, 0 | 2,400, 2,455 | 10, 10 |
| T2 difficulty weighting | 643, 635 | 10, 10 | 362, 354 | 39, 37 | 1, 1 | 2,486, 2,506 | 10, 10 |
| T3 hindsight relabelling | 669, 601 | 10, 10 | 390, 328 | 47, 26 | 1, 0 | 2,507, 2,407 | 10, 10 |
| T4 moving window | 632, 605 | 10, 10 | 357, 327 | 43, 31 | 2, 1 | 2,383, 2,353 | 10, 10 |
| T5 precursor injection | 627, 648 | 10, 10 | 351, 368 | 37, 35 | 2, 0 | 2,448, 2,453 | 10, 10 |
| T6 sibling pair | 694, 680 | 10, 10 | 409, 395 | 46, 42 | 1, 1 | 2,537, 2,535 | 10, 10 |

- **Transfer `L*` = 10 for all twelve trained arms, frozen 7 on both seeds, at equal attempts:
  `L* − L*_frozen` = +3 for every rung on both seeds; no rung differs from T1 in `L*`.** Same
  with the reviewer's corrected labels.
- Transfer solve rate by bin (T1 s0; Wilson 95 %): `L_true` 7 46.7 % [41.1, 52.3], 8 45.7 %
  [40.1, 51.3], 9 29.7 % [27.0, 32.6], 10 7.5 % [5.4, 10.4], 11 1 / 99 [0.2, 5.5], 12–14 0 / 125.
  Frozen: 6.7–7.0 % at 7, 0.7–1.0 % at 8, **0 of 1,685 theorems at `L_true` ≥ 9 in 2 × 256
  attempts**. The full 14-arm table is in the review.
- **No arm reaches 11 within 8 rounds** (≤ 2 / 99 at `L_true` 11, 0 / 125 at 12–14; T4 spent 706
  samples on each of 221 targets at `L_true` 11–13 for 3 and 0 solves), **and counts at 10 were
  still rising when the runs stopped** (T1 s0 at ≥ 10 over rounds 4–8: 2, 12, 23, 27, 35; every
  arm). `L*` by round on transfer: 7, 8, 9, 9, 10, 10, 10, 10 for most arms.
- **Base reachability.** Every counted transfer proof at `L_true` ≥ 9 has base p < 1e-5 (0 of
  327–409 solved theorems per arm has one above); at `L_true` 7, 36.5–42.4 % of solved theorems
  (38.9–42.9 % of proofs) are ≥ 1e-5. The reviewer re-scored 1,370 proofs with an independent
  scorer: max |Δ log p| 1.28 nats, one flip at the line.
- **What the solved long theorems are (reviewer's predicates; not in the executor's write-up).**
  Every solved transfer theorem at `L_true` ≥ 9 is a generator theorem; of 760 textbook-schema
  theorems every arm solves 80–94, 79–80 of them `contraposition` + `export` (both `L_true` 7),
  and 17 of 19 schemata are at ≈ 0 in every arm, trained or not. Counted transfer proofs at
  `L_true` ≥ 10 that use any of ORE / NEGI / NEGE / BOTE: **0 in ten arms, 1 in two** (of 44–83
  per arm); they are AS / IMPI towers (depth 3–4) around ANDI / ORI / IMPE steps. Generator
  theorems at `L_true` 10 whose generating proof uses NEGI: 0 / 48 solved (T1 s0, T6 s0).
  **"`L*` = 10" is one already-known proof family being written longer, not the pool's
  `L_true`-10 bin in general.**
- **T6.** Both T6 seeds exceed both T1 seeds in total and in every bin ≥ 8; each sibling *alone*
  at 128 transfer attempts solves 636–657 (357–376 at ≥ 9), more than T1 at 256 (612 / 623;
  335 / 346); T1 s0 ∪ s1 (512 attempts, two models) solves 667. T6 trains two models — twice the
  gradient steps per round, pod time 111 vs 65–93 min: equal sample budget, not equal compute.
- **T3 is almost T1** (55 relabelled records per seed against ≈ 3,400 target proofs in the mix),
  and its two seeds span 328–390 at ≥ 9 — a range containing T1, T2, T4 and T5 on both seeds.
  T6's margin over the best T1-like run is 5–19 theorems at ≥ 9. Treating T1 and T3 as four
  T1-like runs (mean 350, sd 28) against T6's two (mean 402): pooled t = 2.45 on 4 df
  (p ≈ 0.07), Welch p ≈ 0.03, with "T3 counts as T1" the reviewer's post-hoc assumption.
- Held-out greedy 94.76–95.94 % in every round of every arm (read from `round_*.json`; raw
  generations are not saved, so not re-derived). T5 textbook-schema solve rate 11.2 / 10.8 % vs
  T1 10.9 / 10.9 %.
- **Label errors (reviewer).** `minlen.py` never tries `ORE` with one box cited twice on
  `( A v A )`; merging duplicate ORE boxes and re-verifying gives shorter proofs for 18 of 818
  transfer theorems ever solved and 58 of 2,787 targets (**76**, all off by one, none labelled
  ≥ 10; the executor reports 42 in one file and 43 in another, by written proofs only). `L*` is
  unchanged in every arm; this is a lower bound, since only solved theorems can be checked.
- **Splits.** 0 overlaps by the pre-registered renaming class between every training file and
  every evaluation pool. Under a stronger key (premises also sorted) 2 target classes coincide
  with transfer classes (neither trained on in any arm's mix × transfer intersection);
  immaterial here (2 / 2,285).

**Pre-registered expectations** (committed before the first pod). Wrong: T1 transfer `L*` 9
(10); the headline "no rung reaches +3" (all do); T3's and T6's transfer `L*` 9 (both met their
falsifier "≥ 10 on both seeds"; **not listed by the executor**); T2 / T4 target `L*` above T1's
(all 10); T5 schemata + 5 points (85, 82 vs 83, 83 of 760; falsifier met on s1 only); frozen
solve rate at 7 of 25–35 % (6.7–7.0 %, because 236 of the bin's 300 are textbook instances the
base cannot write — the executor lists this under "held" because `L*` = 7 held; the reviewer
labels the rate a miss); elicitation share at `L_true` 7 of 40–60 % (36.5–42.4 %: low edge);
cost ≈ $12–20 (≈ $39). Held: frozen `L*` 7; ≥ 90 % of proofs at ≥ 9 below 1e-5 (100 %);
held-out greedy ≥ 93 %; T3 solve at 8 within ± 5 points of T1.

**Rewordings the verdict requires** (the executor's files still have the left-hand side):

1. "The wall is at 11" / "do not move the wall" / "beneath it" → "no arm reaches 11 within 8
   rounds (≤ 2 / 99); counts at 10 were still rising when the runs stopped". Each earlier length
   went from 0–2 solved to ≥ 5 within one or two rounds of first appearing; the Wilson upper
   bound at 11 is 5.5–7.1 %.
2. "Paired p < 0.001 twice" as support for T6 → dropped. The numbers reproduce (88 / 14 and
   72 / 23), but the same test gives T3 s0 vs T1 s0 73 / 18, p < 0.001 and T3 s1 vs T1 s1
   27 / 45, p = 0.044, and T3 is T1 plus 55 records: the test's unit is the theorem, the noise
   is the training run. What n supports: **"T6 is ahead on both seeds; likely but not
   established"**; the single-sibling result is the stronger evidence.
3. Frozen "held" → `L*` held, rate prediction missed; add T3's and T6's falsified predictions;
   give the 7-bin elicitation share a verdict.
4. Label errors 42 / 43 → say which; 76 by the merge check.
5. `log.md` 18:00 on the T3 s0 resume ("only the shuffle RNG differs") is understated: the
   restarted RNG also re-draws the Stage-1 retention records, so rounds 6–8 repeat 97 / 94 / 91 %
   of rounds 1–3's 20,000; T3 s0 saw 77.8k distinct Stage-1 records against ≈ 103.6k elsewhere.
   Effect unknown, probably small; T3 s0's lead predates the resume (240 vs 204 at ≥ 9 at round 5).
6. `numbers.md`: T3 s0 by-products 49 → 55.

**Not supported by the run** (reviewer): a length wall in the sense of a limit more rounds would
not move; T6 > T1 as an established technique effect (it is the run's best lead, not its
finding); any reading of "+3 lines of `L*`" as new *kinds* of proof — the gain is confined to
the implication-tower family on generator-shaped theorems. T2, T3, T4, T5 are not
distinguishable from T1 with two seeds (the executor says so).

## Limitations and next step

- Two seeds per rung, and "same seed" is not paired (seed-0 arms split into two groups with
  different round-1 samples from the same checkpoint). Run-to-run spread (T3 ≈ T1: 328 vs 390)
  is as large as any rung difference except T6's.
- Eight rounds; nothing had plateaued. `L*` with a ≥ 5-theorem threshold saturates and hides
  differences below it.
- T6 is compute-confounded (two models). T3 s0 is not protocol-identical to T3 s1 (resume).
- `L_true` labels overestimate by one line for at least 76 solved theorems; pools are disjoint
  with premise order fixed, not up to premise permutation. Held-out greedy and bucket contents
  were read, not re-derived.
- The session died ≈ 08:00 UTC on an OAuth expiry with five pods up; they idled 04:42–17:20
  (≈ 63 pod-hours ≈ $32 of the ≈ $39).

Next measurements (reviewer, in order): (1) *rate limit or wall* — T1 continued to 16–24 rounds
from the round-8 checkpoints (2 seeds exist, add 2), solved counts at 10, 11, 12 per round,
with the prediction written first (if 11 behaves as 10 did, ≥ 5 at 11 within ≈ 2–4 more
rounds; ≈ 8 min per round per arm on a 3090); (2) *is T6 real and why* — 4 seeds each of T1 and
T6, the executor's control (one policy trained on the union of two independent samplers'
finds) and T1 at 1,200 steps per round; (3) *length vs kind* — every frontier number split by
rule family and `L*` on the ORE / NEGI / NEGE / BOTE family alone, a re-analysis of existing
files with no pods (reviewer's count: 9 in every arm, on 5–21 theorems, against frozen 7);
(4) fix `minlen.py` (ORE with one box cited twice), relabel, and make the class key
premise-order-invariant before building Phase B pools. The executor's Phase B request
(`QUESTIONS.md` 18:35, ≈ $15–20, default: not started) overlaps (1) and (2) and adds a sliding
`L*` threshold; it awaits Dan's answer.

## Addendum (2026-09-19): per-round curves and saturation

`figures/ladder_saturation.png` (data in `curves_by_round.json`, computed from the arms' round
files and training logs): cumulative transfer solves, solves at `L_true` ≥ 9 / ≥ 10, new
accepted target proofs per round, and the fine-tune loss at step 600 of each round, for T1 and
the two T6 siblings, with the frozen control. Reading: new proofs per round decay roughly
geometrically after round 2 (T1 s0: 377, 813, 680, 417, 281, 213, 161, 172) and cumulative
transfer solves grow by 112 → 74 → 43 → 43 per round over rounds 5–8, so the *count* is
approaching a plateau — extrapolating the ~0.7 ratio gives ≈ 700–720 of 2,285 for T1 at this
budget. Solves at `L_true` ≥ 10 were still rising (2, 12, 23, 27, 35) with no clear
deceleration, and `L*` reached 10 at round 5 and did not move; ≥ 11 stayed at 0–1. Held-out
accuracy is flat at 0.95, and the per-round fine-tune loss drifts down (0.053 → 0.038) as the
mix changes, so neither signals over-fitting. A longer-horizon continuation (24–32 rounds) is
the direct test of where the count plateaus and whether 11 is ever reached from this lineage.
