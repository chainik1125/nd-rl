---
written_on: 2026-09-30
written_by: agent:claude
---

# long-pool-2: a pool past `L_true` 17 (21 new theorems), and no detectable fall-off for SN-cap12 T1 at 17 vs ≥ 18

Status: **completed 2026-09-29 16:00 – 2026-09-30 01:25 UTC (executor), independently reviewed 01:30–01:42 UTC.
Verdict: stands, with seven rewordings and two policy omissions (no term sizes, no compute rows).** No hard-constraint
violation, nothing quarantined. Where the reviewer's wording differs from the executor's, this summary uses the
reviewer's.

Question: SN-cap12 T1 has `L*` ≥ 17 on `long-pool`'s pool, the pool's ceiling (see
[state-cap12](../2026-09-29-state-cap12-state-and-cap12-proofs-compound-on-long-pool/README.md)). Where does its success
fall off with proof length? This needs theorems labelled above 17 (proposal 16 item 0).

The run was on the fork (`dan-pandori/nd-takehome`), branch `dan_long-pool-2` (from the fork's `dan` at `6b7b2f5e`).
**Only the pool is merged into the fork's `dan`** (`d3abe474`: `transfer_long2` (21) + calibration (70), the
`POOLS.md` section, the labelling scripts). The re-read, analysis, write-up and review stay on `dan_long-pool-2`.

- Review: [`review_long-pool-2.md`](https://github.com/dan-pandori/nd-takehome/blob/fbb8dcf6/review_long-pool-2.md)
  ([`9c429c49`](https://github.com/dan-pandori/nd-takehome/commit/9c429c49) §Recount, written before the reviewer read
  the executor's write-up; [`fbb8dcf6`](https://github.com/dan-pandori/nd-takehome/commit/fbb8dcf6) §Comparison /
  §Verdict). Reviewer scripts: `review_lp2/rv_lp2_{pool,rates,termsize,redraw}.py`.
- Executor write-up: [`run_long_pool_2.md`](https://github.com/dan-pandori/nd-takehome/blob/fbb8dcf6/run_long_pool_2.md),
  `numbers.md` § long-pool-2 (LP2-1 … LP2-8), `log.md` 2026-09-29/30. Results commit `07e6a3f7`, DONE `065975e3`.
- Pre-registration: `preregistration/long-pool-2.md`, commit `ea3cfdff` at 16:32:46 UTC. The first pod (`lp2-a`) was
  created at 16:33:47 UTC, so the pre-registration came first.
- Brief: [`BRIEF_long-pool-2.md`](../../docs/proposals/literature/BRIEF_long-pool-2.md).
- Figure (not checked by the reviewer, so linked, not copied):
  [`figures/lpool2_rate.png`](https://github.com/dan-pandori/nd-takehome/blob/fbb8dcf6/figures/lpool2_rate.png).

## Configuration

- **Checker: Lean alone** (`lean_judge` / `lean_check.py`, Lean 4.34). `nd_verify` judges nothing. One caveat from the
  reviewer: the shared labeller `minlen.py` (unchanged) checks each proof it finds with `nd_verify`, so the stage-E/F
  "found a 17/18-line proof" labels are `nd_verify`-checked by construction. The reviewer re-checked all 91 `ub_proof`s
  in Lean (91 / 91 accepted), so every upper bound stands under Lean. Lower bounds are exhaustive-search bounds over
  `minlen`'s ND search space, not proof judgements.
- **Design change before any pod** (pilot, `artifacts/lpool2/pilot/`): the brief's bins by construction upper bound
  (17–18 / 19–20 / 21–22 / 23–24) cannot be filled. Generator proofs run 15–30 lines above minimal, and pruning removes
  nothing. The executor labelled with `minlen` bounds instead: stages A–D as `long-pool` (10 / 12 / 14 / 16), then
  **stage E** (bound 17, 1,800 s): exact 17 or lower bound 18. **Stage F** (bound 18, 3,600 CPU-s) was **added, not
  pre-registered**, and only partly ran (F: 10 theorems; F2: 3 of 30 finished, 2 of them OOM).
- **Generation:** `make_coverage_sets.py gen --long`, generated length 32–90, 25 chunks (b1–b7, c1–c9, d1–d6, e1–e3),
  chunk seeds 35001…, 36001…, 37001…, 38001… (consecutive; see the seed finding below).
- **Pool:** `data/ladder/transfer_long2.jsonl` (21 new) and `transfer_long2_calib.jsonl` (the 70 theorems of
  `transfer_long_ge17.jsonl`, relabelled by stage E).
- **Re-read** (`lpool_reread.py`, k 256, T 0.8, seed 0): state models batch 2,048, `max_action` 512, **`max_steps` 96**;
  whole-proof batch 1,024, `max_new` 1,536. All 91 theorems, 12 checkpoints (all 4 layers, d 256, from scratch):
  - **SN-cap12 T1** s0–s3: `state-cap12/ckpts/sc12/ladder/la_T1_SN12_s{0-3}_r8.pt`, 3,216,384 params, `lean_staten`,
    Stage-1 on K12 (cap 12) + 8 ladder EI rounds.
  - **SN-cap12 frozen** s0–s3: `state-cap12/ckpts/sc12/stage1_SN12_s{0-3}.pt`, same models' Stage-1, no EI.
  - **K12 whole-proof T1** s0/s1: `state-cap12/ckpts/ladder/la_T1_K12_s{0,1}_r8.pt`, 3,214,336 params, `lean_seq`,
    K12 Stage-1 + 8 ladder rounds.
  - **SN-v2 cap-6 T1** s0/s1: `state-env/ckpts/se/ladder/la_T1_SN_s{0,1}_r8.pt`, 3,216,384 params, `lean_staten`,
    cap-6 control set + 8 ladder rounds.
  - Step-cap check: SN-cap12 T1 s2/s3 at `max_steps` 96 on `transfer_long_rr600` and the ≥ 17 file.
- **Pods.** `lp2-a` RTX A5000 (16:33–21:58), `lp2-b` RTX 4000 Ada (16:37–≈01:00; 125 GB memory limit), `lp2-c` RTX 4090
  $0.74/h (21:50–≈01:32). CPU pods and 4090/5090 had no stock at the start (`QUESTIONS.md` 16:37, unanswered, default
  followed). Labelling ran on the pods' CPUs; `lp2-a` was CPU-starved (7.65-CPU quota) and was deleted early.
- **Spend:** RunPod billing (reviewer) **$5.82** ($1.46 + $2.27 + $2.09); `podbudget` 17.08 h / $6.29 (it over-states
  `lp2-c` by ≈ $0.43). Budget $10 / 20 h.
- **Compute record** (policy; no `record.py` rows were written, derived by the reviewer from the re-read summaries):
  generation 38–187 GPU-wall-seconds per pool read and 1,429 / 1,589 s for the two rr600 reads; 5,376 / 17,920 / 153,600
  samples per file; labelling ≈ 210 k `minlen` core-seconds over chunk records (plus calibration and stage F); no
  training. Every model read the same prompts at the same k, so no compute-matching question arises.
- **Bucket:** `hf://buckets/dan-pandori/nd-rl/long-pool-2/{data/lp2,data/ladder,artifacts/lpool2,figures}` (641 objects,
  1.38 GB).

## Observations

All counts below were re-derived by the reviewer from the raw stage files and re-read rows.

**Pool.**
- 231,809 generated rows → 55 lower-bound-17 records → **21 distinct renaming classes**. Strata: new 21 = 13 exact 17 /
  4 exact 18 / 4 ≥ 18 (F open); calibration 70 = 48 / 1 / 21. All 91 = **61 exact 17 / 30 ≥ 18** (5 exact 18 + 25 F-open).
- **Seed overlap (the executor's error, the main cause of the yield miss, reviewer's framing).** Chunk seeds were
  consecutive and each worker is seeded `seed + i`, so chunks re-ran each other's generator streams. **≈ 69 % of
  generated rows** repeat an earlier class (72,765 distinct of 231,809), not "about 60 %". Well-spaced seeds would have
  given ≈ 3× the pool (≈ 55–65 new theorems) at the same spend. The duplicates are otherwise harmless: assembly
  de-duplicates, and stage E gave the same outcome on every copy of all 21 classes. Small-CPU pods and 5–8 GB per
  search also cost yield, but less.
- Stage E: new 62 % exact 17, calibration 69 %, **0 timeouts**. Stage-D timeout rate 0.5 %.
- Upper bounds: construction length never shrinks under pruning (91 / 91); new-pool construction 32–48 (median 38).
  Brief's bins by best upper bound: **66 / 0 / 0 / 1 / 24** (17–18 / 19–20 / 21–22 / 23–24 / 25+) — `numbers.md` says
  65 / … / 25, a transcription error. By construction length alone 0 / 0 / 0 / 2 / 89.
- Every `ub_proof` and `gen_proof` Lean-accepted (91 / 91 each).
- **Disjoint:** the 21 new theorems hit no renaming class in 239 files / ≈ 11.2 M records (the reviewer's scan, a
  superset of the executor's 118 files / 5.74 M). The 70 calibration theorems hit only the ≥ 17 file they came from.

**Re-read (solved = ≥ 1 Lean-accepted proof in 256 samples).**

| model | all 91, per seed | exact 17 (n 61) | ≥ 18 (n 30) | new 21 only |
|---|---|---|---|---|
| SN-cap12 T1 | 34 / 59 / 58 / 61 | 25 / 43 / 38 / 42 | 9 / 16 / 20 / 19 | 8 / 15 / 17 / 14 |
| SN-cap12 frozen | 11 / 16 / 21 / 23 | 9 / 11 / 17 / 19 | 2 / 5 / 4 / 4 | 3 / 1 / 3 / 5 |
| K12 whole-proof T1 | 3 / 2 | 2 / 1 | 1 / 1 | 0 / 0 |
| SN-v2 cap-6 T1 | 4 / 0 | 4 / 0 | 0 / 0 | 0 / 0 |

- SN-cap12 T1 IQM (4 seeds, stratified bootstrap 95 %): all 91 **64.3 % [37.4, 67.0]**; exact 17 65.6 % [41.0, 70.5];
  **≥ 18 (n 30) 58.3 % [30.0, 66.7]**. Frozen ≥ 18: 13.3 % [6.7, 16.7]. (The executor's "L ≥ 18 56.0 %" and frozen
  "10.0 %" are the n 25 F-open sub-stratum, printed beside an n 30 column: mislabelled.)
- **Exact 17 − ≥ 18, SN-cap12 T1: +7.3 pp** (4-seed mean; per seed +11.0 / +17.2 / −4.4 / +5.5; theorem bootstrap 95 %
  [−7.7, +22.6]). The minimum detectable difference between strata of 61 and 30 is ≈ 30 pp. This is **a null at this
  power**: no fall is detected, and a fall of up to ≈ 20–30 pp is not excluded. New and calibration populations agree in
  direction.
- **By construction length** (≤ 28 / 29–32 / 33–36 / 37+, n 8 / 19 / 24 / 40): SN-cap12 T1's rate **does not fall**
  (top bin ≥ bottom bin on 4 / 4 seeds). It is not monotone on any seed, and the bottom bin has n 8 — the executor's
  "rises with construction length" is overstated.
- The brief's falsifier (≥ 10 % in every filled bin) **fires on all 4 seeds**, as the executor predicted. The 23–24 bin
  holds one theorem.
- **Term size** (reviewer only; the write-up reports none, a policy omission). Minimal proofs (`ub_proof`) have term size
  median 12 (10–16) on exact-17 theorems and 25 (12–42) on ≥ 18 — but the ≥ 18 ones are mostly constructions, so upper
  bounds only. SN-cap12 T1's smallest accepted proof per theorem: median 14 (min 10) on exact 17, median 16 (min 9) on
  ≥ 18. The ≥ 18 stratum's easiest theorem in term size is smaller than any exact-17 minimal proof, and on 24 of 78 solved
  theorems the model's proof is smaller in term size than the pool's `ub_proof`. **Under Lean term size, ND lines 17 vs
  18 are not a clean difficulty axis**; "no fall-off up to length 18" is a statement in ND lines only.

**Step cap (state-cap12's open question), closed.** SN-cap12 T1 s2 / s3 at `max_steps` 96: rr600 **471 / 468** (476 /
472 at 48), ≥ 17 file **42 / 41** (42 / 46). The four `max_steps`-96 step-check reads use at most 89 / 86 steps; one pool
sample (`T1_SN12_s3__cal`) hit 96 (1 / 17,920). Every difference is inside the same-settings redraw spread below, so
the step-cap change has no detectable effect.

**Same-settings redraw (reviewer).** Two reads of SN-cap12 T1 s3 on the same 70 theorems with identical settings, on
different pods ≈ 7 h apart, solve **41 vs 47** (8 flips); s2 42 vs 41 (11 flips). Per-theorem solved/unsolved on 70
theorems at k 256 moves by ≈ 6–11 flips between identical reads. The s3 "miss" on the ≥ 17 file (41 vs 46 ± 4) is this
sampling redraw, not the cap.

**Checks.** The reviewer Lean-checked all **3,012** counted pool proofs across the 12 checkpoints (all accepted) and
160 each of the rr600 / ≥ 17 `max_steps`-96 samples (all accepted). No accepted proof is shorter in lines than its row's
lower bound. The executor's literal-text re-check: 292 / 292 accepted, 72 / 72 negative controls rejected. Action-cap and
`max_new` truncation ≤ 0.056 % per file (max `T1_SN12_s2__new`, 3 / 5,376), under the policy's ≈ 0.1 %; peak memory
10.6–17.1 GB.

**Expected vs outcome** (reviewer's completed list; the executor's "model rates were met" is not supported).
- Met: the falsifier fires (predicted p ≈ 0.85); construction ≥ 23 and median 33–38 (100 % ≥ 32, median 38); stage E
  45–70 % exact 17 (62 / 69 %); no fall by construction bin; exact 17 above ≥ 18 by 0–20 pp (+7.3); frozen 10–35 % on all
  91 (12.1–25.3 %); SN-v2 0–10 %; rr600 within ± 12 (471 / 468); step-cap hits ≤ 0.02 % (0.006 %).
- Missed:
  - New lower-bound-17 theorems 120–300 (point 180) → **21** (seed overlap first).
  - Brief's 17–22 bins < 10 each → the 17–18 bin holds 66, because stage E sets `L_ub` = 17.
  - Stage-D timeout 5–10 % → 0.5 %.
  - Stage-E timeout 15–45 % → 0 %.
  - SN-cap12 T1 on the new pool 30–70 % per seed, IQM 40–65 % → 38.1 / **71.4 / 81.0** / 66.7 %, IQM **69.0 %** (above).
  - K12 T1 1–10 % → 0 / 0 on the new pool (calibration 4.3 / 2.9 %).
  - Frozen on the new pool 10–35 % → s1 **4.8 %**.
  - ≥ 17 file at `max_steps` 96 within ± 4 of 42 / 46 → s3 41 (a sampling redraw; see above).

## Limitations and next step

- The pool is small: 21 new theorems, 91 with calibration. Strata of 61 and 30 give an MDD of ≈ 30 pp, so the 17 vs
  ≥ 18 comparison can only detect a large cliff.
- Nothing above 18 is labelled: 25 of 30 ≥ 18 theorems are F-open. Bound-18 searches need > 20 GB each.
- Line strata are ND-derived; under Lean term size the ≥ 18 stratum is not harder (see above).
- Per-theorem results flip by 6–11 of 70 between identical reads, so single-theorem comparisons need care.
- The write-up has no term sizes and no compute rows (policy). Neither changes a count.
- The fork run branch is not merged except the pool (`d3abe474`).

Next measurements (reviewer):
1. Rerun the same pipeline with **chunk seeds spaced ≥ 1,000** (≈ 3× the distinct yield at this spend), and finish stage F
   on the 25 F-open theorems on a pod with a ≥ 125 GB memory limit (like `lp2-b`). That gives ≈ 60–80 new theorems and
   splits ≥ 18 into exact 18 / ≥ 19.
2. For lengths beyond 19, use theorems of **known length by construction** (e.g. chained lemmas), stratified by Lean term
   size as well as ND lines, read by SN-cap12 T1 s0–s3 and frozen s0–s3 at k 256. Pre-register a paired or larger design:
   ≈ 60 theorems per stratum gives an MDD of ≈ 20 pp.
