---
written_on: 2026-09-29
written_by: agent:claude
---

# long-pool: a transfer pool that can measure the length frontier, and the "wall at 13" was mostly the old pool

Status: **completed 2026-09-28 18:19 – 09-29 01:57 UTC (executor) and independently reviewed 02:00–02:19 UTC.
Verdict: stands, with five rewordings.** No hard-constraint violation, nothing quarantined. Where the reviewer's
wording differs from the executor's, this summary uses the reviewer's.

Question: every length claim in the project is read off `data/ladder/transfer.jsonl`, which has only 23 theorems at
`L_true` ≥ 13 and none above 14. With `L*` = max L with ≥ 5 solved at `L_true` ≥ L, `L*` = 13 needs 5 of those 23
and `L*` > 14 cannot be measured at all. Can we build a pool with ≥ 100 theorems in every `L_true` bin 11–16 from the
ladder's generator family, and what do existing checkpoints read on it (no training)?

The run was on the fork (`dan-pandori/nd-takehome`), branch `dan_long-pool`. **The pool itself (not the re-read) is
merged into the fork's `dan`** as `470d39e0`: the `transfer_long*` files, the `POOLS.md` section, `lpool_{assemble,
finalize,shape,stages}.py`, `pod/lpool/{label,sync}.sh`, and `artifacts/lpool/{excl_manifest.txt,stages.md,shape.md,
lean_labels.jsonl,assemble.log}`. The re-read, numbers, log and write-up stay on `dan_long-pool`.

- Review: [`review_long-pool.md`](https://github.com/dan-pandori/nd-takehome/blob/0c19a780/review_long-pool.md)
  ([`372c6ad2`](https://github.com/dan-pandori/nd-takehome/commit/372c6ad2) §Recount, blind;
  [`0c19a780`](https://github.com/dan-pandori/nd-takehome/commit/0c19a780) §Phase 2/§Verdict, with a term-size
  erratum to its own recount; reviewer scripts `review/long-pool-recount/`, including its own ND → Lean translator).
- Executor write-up: [`run_long_pool.md`](https://github.com/dan-pandori/nd-takehome/blob/0c19a780/run_long_pool.md),
  `numbers.md` § long-pool (LP1–LP14), `log.md` § long-pool, `data/ladder/POOLS.md` § long-pool. Results commit
  `5fc3e701`.
- Pre-registration: `preregistration/long-pool.md`, commit `dfce3630` at 18:32:38 UTC, before the first pod (`lp-l1`,
  created 18:34:14 UTC). Its text says "written ≈ 19:00"; the commit time is the one that counts.
- Brief: [`BRIEF_long-pool.md`](../../docs/proposals/state-env/BRIEF_long-pool.md).

## Configuration

- **Generation.** The ladder's strict long generator (`make_coverage_sets.py gen --long`, take-home knobs, no extra
  flags), with only the generated-length output filter raised, in four chunks: g1 24,400 theorems (generated length
  12–40, seed 31000), g2 255,368 (20–60, 32000), g3 224,596 (28–80, 33000), g4 103,852 (32–90, 34000). The pilot showed
  ≥ 13 yield rising with generated length, so g2–g4 keep ≥ 20 lines, an output filter the pre-registration allowed.
  Textbook: `textbook_pool.py --n 3900 --seed 41`, the 6 schemata that reach ≥ 11 (906 instances).
- **Labels.** Four-stage `minlen.py`: A bound 10 / 5 s, B 12 / 30 s, C 14 / 120 s, D 16 / 600 s. Label = first stage
  that finds a proof. A timeout at any reached stage = label unknown, excluded. No proof at bound 16 without a timeout
  = ≥ 17 (separate file). `L_true` is an ND-derived label and so an upper bound under Lean.
- **Disjointness.** Renaming class against every training / evaluation file found (executor 117 files, 5,736,567
  records; reviewer 118 files, 5,746,204 records, a superset, under two keys, one also premise-order invariant).
- **Re-read subset.** `transfer_long_rr600.jsonl`, seed 0, 100 theorems per bin 11–16, plus the 70-theorem ≥ 17 file.
  The whole pool was not re-read: at ≈ 7 h for 14 models on one 4090 it did not fit the budget (a recorded deviation).
- **Re-read settings.** k = 256, T 0.8, seed 0, Lean alone decides (`lean_judge`). Pass 2 is reported: whole-proof batch
  1,024 / `max_new` 1,536; state batch 2,048 / `max_action` 512 / `max_steps` 48. Pass 1 (whole-proof batch 2,048 /
  `max_new` 768; state `max_action` 256) is kept as a re-draw. Peak memory, pass 2: whole-proof 10.7–11.0 GB,
  state 11.6–14.7 GB.
- **Models** (all from scratch, 4 layers, d 256; read from each checkpoint's `extra` by the reviewer; sha256 prefixes in
  `artifacts/lpool/ckpt_sha.txt`):

  | label | checkpoint (bucket) | params | format | training |
  |---|---|---:|---|---|
  | S T1 s0 / s1 | `state-env/ckpts/se/ladder/la_T1_S_s{0,1}_r8.pt` | 3,216,384 | `lean_state` | Stage-1 on the control set `train_depth3_f0_a1` (cap 6) + 8 ladder EI rounds |
  | SN-v2 T1 s0 / s1 | `state-env/…/la_T1_SN_s{0,1}_r8.pt` | 3,216,384 | `lean_staten` | same |
  | C0 T1 s0 / s1 | `ds-generator/ckpts/ladder/la_T1_c0_s{0,1}_r8.pt` | 3,214,336 | `lean_seq` whole-proof | same |
  | S / SN frozen s0 / s1 | `state-env/ckpts/se/stage1_{S,SN}_s{0,1}.pt` | 3,216,384 | `lean_state` / `lean_staten` | Stage-1 only, cap 6 |
  | C0 frozen s0 / s1 | `lean-format/ckpts/lf/stage1_a1_seq_s{0,1}.pt` | 3,214,336 | `lean_seq` | Stage-1 only, cap 6 |
  | K12 / K14 frozen s0 | `cap-horizon/ckpts/kh/stage1_k{12,14}_s0.pt` | 3,214,336 | `lean_seq` | Stage-1 only, **cap 12 / cap 14**, one seed (their T1 checkpoints are not in the bucket) |

- **Pods.** `lp-l1` NVIDIA GeForce RTX 4090 $0.74/h, 1.54 h, $1.14; `lp-l2` RTX 4090 $0.74/h, 6.91 h, $5.11. Total
  **8.44 pod-hours, $6.25** of an $8 / 16 pod-hour budget (`~/podhours.log`, confirmed by the reviewer). CPU pods, A40 and
  3090 were out of stock, so labelling ran on the 4090 pods' CPUs (a recorded deviation). Both pods deleted.
- **Bucket.** `hf://buckets/dan-pandori/nd-rl/long-pool/{data/ladder,data/lp,artifacts/lpool}`.

## Observations (reviewed)

**The pool.** Every label and label proof re-derives exactly from the raw stage files (0 of 1,913 mismatches), every
label proof has exactly `L_true` lines, and the reviewer's own translator + Lean 4.34.1 accepts all 1,913.

| `L_true` | 11 | 12 | 13 | 14 | 15 | 16 | total | ≥ 17 file |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| generator | 300 | 300 | 300 | 300 | 300 | 131 | 1,631 | 70 |
| textbook | 73 | 128 | 41 | 40 | 0 | 0 | 282 | 0 |
| **total** | **373** | **428** | **341** | **340** | **300** | **131** | **1,913** | 70 |
| re-read subset `rr600` (textbook) | 100 (19) | 100 (34) | 100 (10) | 100 (10) | 100 (0) | 100 (0) | 600 | — |

- Bin 16 holds every generator theorem labelled 16 (131 of 131; no cap binding).
- Timeouts (excluded): A 1,003 / 609,122 (0.16 %), B 143 / 21,992 (0.65 %), C 136 / 4,518 (3.0 %), **D 53 / 697
  (7.6 %)**. So membership of bin 16 and of the ≥ 17 file is conditional on the bound-16 search finishing.
- Label-proof term size 4–16, median 9. By bin, median [min–max], after the reviewer's erratum: 11: 7 [4–13] ·
  12: 9 [4–12] · 13: 9 [5–12] · 14: 10 [5–13] · 15: 10 [6–15] · 16: 11 [8–16].
- Disjointness: **0 shared renaming classes** with 118 files under two keys. Raw collisions before exclusion: 161 of
  21,652 labelled ≥ 11 classes = 0.74 %.
- 67 pool theorems have **jointly unsatisfiable premises** (11: 28, 12: 16, 13: 13, 14: 6, 15: 3, 16: 1). The generator
  drops only syntactic `A`, `¬A` pairs. The ladder's `transfer.jsonl` has 125 such theorems, so this is not a deviation,
  but these theorems have an ex-falso route.

**Re-read, solved theorems per bin (pass 2; n = 100 per bin, 70 at ≥ 17).** Identical cell for cell in the recount.

| model | 11 | 12 | 13 | 14 | 15 | 16 | ≥ 17 | total /600 | `L*` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| S T1 s0 / s1 | 23 / 23 | 21 / 11 | 11 / 12 | 15 / 8 | 2 / 6 | 1 / 2 | 0 / 0 | 73 / 62 | 14 / 15 |
| SN-v2 T1 s0 / s1 | 42 / 25 | 35 / 15 | 31 / 14 | 30 / 9 | 22 / 3 | 19 / 2 | 5 / 0 | 179 / 68 | **≥ 17** / 15 |
| C0 T1 s0 / s1 | 11 / 6 | 6 / 3 | 1 / 1 | 1 / 1 | 0 / 0 | 0 / 0 | 0 / 0 | 19 / 11 | 12 / 12 |
| S frozen s0 / s1 | 0 / 1 | 0 / 0 | 0 / 1 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 2 | < 11 / < 11 |
| SN frozen s0 / s1 | 7 / 3 | 1 / 1 | 2 / 0 | 1 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 11 / 4 | 11 / < 11 |
| C0 frozen s0 / s1 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | < 11 / < 11 |
| K12 frozen s0 | 33 | 17 | 12 | 12 | 12 | 2 | 0 | 88 | 15 |
| K14 frozen s0 | 42 | 27 | 16 | 15 | 13 | 5 | 0 | 118 | 16 |

Figure (executor's, not copied): [`figures/lpool_rate_by_bin_rr2.png`](https://github.com/dan-pandori/nd-takehome/blob/0c19a780/figures/lpool_rate_by_bin_rr2.png).
The reviewer found its curves match the table, but bins 11–14 include 10–34 % textbook theorems that almost no model
solves and bins 15–16 include none. The drop from 12 to 13 and the flattening at 15–16 therefore partly reflect
composition. Generator-only rates for S T1 s0: 28 · 30 · 12 · 17 · 2 · 1 %.

- **Textbook theorems are almost never solved**: 10 solves in 73 × 14 reads (SN-v2 T1 s0 6, s1 2, S T1 1 / 1; every
  frozen model and both C0 T1 models 0).
- **Seed spread is large**: SN-v2 T1 s0 solves 179 / 600, s1 68. SN-v2's `L*` of ≥ 17 is **s0 only**.
- **The cap-12 / cap-14 Stage-1 models, with no RL, out-solve every T1 model except SN-v2 s0** at every bin ≥ 13.
  C0 frozen (cap 6) solves 0 of 670.
- Pass 1 vs pass 2: the 8 state models have identical per-bin counts (same batch and keyed noise, so pass 2 is not a
  re-draw for them). Whole-proof models are a genuine re-draw: totals within ±3, no `L*` change. **Up to 3 theorems in
  a bin of 100 is the only sampling-noise evidence on this pool**; the reviewer says to treat per-bin differences of
  ≲ 3–5 between models as noise.
- Lean re-check: all **3,047** distinct stored accepted proofs of both passes re-accepted by the reviewer's own
  translator. 2 of them need the Lean-valid divergences the policy names (`Not.elim`, an omitted premise line). The stored
  proofs are the ND-form decode of each sample; the literal model text is not kept, so only the decoded form can be
  re-checked.
- Truncation (pass 2): whole-proof 0.02–0.44 % of samples on rr600 and up to **0.93 %** on the ≥ 17 file (C0 frozen
  s1). That is above the policy's ≈ 0.1 % for 5 of 6 whole-proof models after one raise. The executor's argument for
  not raising again is that cut rows never emit `<eos>` and are ≥ 4× the longest accepted proof (362 ND tokens), so they
  are loops. State: 0–0.105 % overall, and S frozen s1 0.238 % / SN-v2 T1 s0 0.164 % in one bin each.

**The "wall at 13" was mostly the old pool** (stands as an explanation, with limits). 17 of the old pool's 23 theorems at
≥ 13 are textbook instances, and textbook instances are nearly unsolvable for every model here. Meanwhile T1 state models
solve 1–34 % of *generator* theorems per bin at 13–16. On the old pool's 23 (smoke run, k = 32), SN-v2 T1 s0 solves 5 / 6
generator and 0 / 17 textbook, and C0 T1 s0 0 / 23. The limits: the old-pool check is one model and one seed at k = 32.
Also, state-env's "wall" was a cumulative 8-round number while this re-read is the final checkpoint, so the two are not
directly comparable.

**Expected vs outcome** (reviewer's scoring of `dfce3630`):
- Pool counts came out above the ranges at 13–16 (341 / 340 / 300 / 131 against 150–300 / 100–250 / 40–150 / 10–80).
  The pre-registered "expect to miss ≥ 100 at 16" was wrong.
- The stage-D timeout (7.6 %), the generator share at 13–14 (88 %) and disjointness were all met.
- `L*`:
  - S T1 predicted 13 (12–14): 14 / 15, so s0 is at the top of the range and s1 above.
  - SN-v2 T1 predicted 13: ≥ 17 / 15, above.
  - C0 T1 predicted 12: 12 / 12, met.
  - S / SN frozen predicted 12: < 11 / < 11 and 11 / < 11, below. **SN frozen s1 also came out lower**, which the
    executor's prose omitted.
  - C0 frozen: met.
  - K12 / K14 predicted 12: 15 / 16, above.
- Rates at 13 were predicted 1–8 % and came out 11–31 %.
- "No model ≥ 5 solved at ≥ 16" was **falsified** (SN-v2 T1 s0 19 at 16 + 5 at ≥ 17; K14 5).
- Overall, the pre-registration read the old pool's wall at 13 as a property of the models. On generator theorems it is
  not one.

## Rewordings the review requires

1. "No accepted proof is shorter than its label" → "no accepted proof has fewer derivation steps than its label". One
   Lean proof (K14 pass 2, `lp_transfer_843`) omits the premise line and is 10 lines against `L_true` 11.
2. "Expected vs outcome": add that SN frozen s1 came out lower and that S T1 s0 (14) fell inside its range.
3. Figure / LP6: state the textbook share of bins 11–14, or add generator-only curves.
4. LP9: add the ≥ 17 file's whole-proof truncation (up to 0.93 %), and say that 0.02–0.44 % is still above the policy's
   0.1 % after one raise, giving the loop argument as the reason for not raising it again.
5. `POOLS.md`: "contradictory premises dropped" → only syntactic pairs are dropped; 67 theorems have jointly
   unsatisfiable premises. Also, a textbook schema has one *maximum* length, not one length.

## Limitations and next step

- **Not supported:** any fine ordering among S T1, SN-v2 T1, K12 and K14 frozen. The SN-v2 seed spread (179 vs 68) is
  larger than their differences, K12 / K14 have one seed, and no noise floor exists for this pool.
- The re-read covers a 600-theorem subset, not the pool, so `L*` at 15–16 rests on 100 theorems per bin.
- `L_true` is an ND upper bound under Lean. Bin 16 / ≥ 17 membership is conditional on the bound-16 search finishing
  (7.6 % timeouts).
- Whole-proof truncation is above the policy threshold, and its per-bin distribution is not recorded.
- The literal model text is not stored, only its ND decode.

Next measurements (reviewer):
- (a) A sampling-noise floor on this pool: re-read S T1 and SN-v2 T1 (both seeds) at a second sampling seed on rr600.
- (b) Read bins 15–16 in full (431 theorems) for the T1 state models and K14.
- (c) A second K14 seed and a third SN-v2 T1 seed, to decide whether SN-v2 s0's 19 / 100 at 16 is the model or the seed.

From now on, length claims should be read on `transfer_long.jsonl`, preferably on its generator-only strata, not on
`transfer.jsonl`'s 23-theorem tail.
