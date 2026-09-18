---
written_on: 2026-09-18
written_by: agent:claude
---

# Round 3, run 3: base generalisation by pattern class (72 Stage-1 draws, no RL)

Status: completed 2026-09-18 12:50 UTC (autonomous executor session, four RTX 3090 pods,
≈ $14) and independently reviewed 2026-09-18 13:23 UTC. Review verdict: no hard-constraint
violation, every number in the write-up reproduces (17 claim rows; first-half *rates* over
unverified samples are "consistent", not re-derivable, because failed samples are not stored),
the pre-registered test is decisive (E2 fails, so the base-level class claim is dropped), and
the executor's ✓ / ✗ marks are honest. Two conclusions of the executor's write-up go past the
data and are reworded below: the derived-ORE zero read as a property of the pattern, and the
missing statement of the run's dominant regularity — **every pre-RL pattern proof in either
class, in all 72 draws, is cap + 1 = 7 lines long**. Numbers and wording below are the
reviewer's. Run id `round3-run3`.

Review: [`review_round3-run3.md`](https://github.com/dan-pandori/nd-takehome/blob/2106258/review_round3-run3.md)
(commits `0d46113` phase 1 blind recount, `9f1da69` phase 2 claims table and verdict, `2106258`
one count corrected in the compare table; reviewer code `review_r3_3_recount.py`,
`review_r3_3_tables.py`, `review_r3_3_extra.py`; outputs `artifacts/review_r3_3/*.json`,
`tables.txt`). Executor write-up:
[`run3-3.md`](https://github.com/dan-pandori/nd-takehome/blob/2106258/run3-3.md) on branch
`dan_round3-run3` (read the review first; its "Consequence" sentence and the fourth result
bullet are the parts the review rewords). The brief's file name `run3.md` was already round
2's write-up, hence `run3-3.md`; unlike run 1's `round3_run1.md` it still matches the
reviewer workspace's `run*.md` withholding glob.

Question: with what probability does a cap-6 Stage-1 draw trained with none of a pattern
(f = 0) emit that pattern before any RL, at what per-sample rate, and do the structural
(depth-3) and rule-sequence (reductio, strict derived-ORE) classes differ at the base level
with 95 % intervals that separate? What property of a draw predicts whether it generalises?
Earlier numbers rested on n = 16 (depth-3) and n = 11 (reductio) draws of the ignition study;
[round-3 run 1](../2026-09-18-round3-run1-pool-composition/README.md) asked the RL-side half of
the same class question on 16 other draws.

## Configuration

- Code, data and results: `dan-pandori/nd-takehome`, branch `dan_round3-run3` (worktree
  `~/work/round3-run3` on the agent VPS). Pre-registration `preregistration/round3-run3.md`
  committed in `16378b0` at 05:16:17 UTC; first pod (`r33a`) created 05:23:43 UTC
  (`~/pods.log`); one amendment at 05:28 (the `derived_disj` first-half predicate narrowed,
  before any sample); deep-pass addendum committed in `0c62ffb` at 09:45:47, first deep job
  10:24:36. Results and write-up `2867c6b`, final `f8ce601`. Brief
  `BRIEF_base-generalisation.md`; policy `AGENT_POLICY.md`. Run diff: `train.py` per-rule
  held-out breakdown, `coverage.py` fields `n_parsed` / `fh_by_pred` / `fh_ok_by_pred`,
  `patterns2.py` first-half predicates with tests, `f0_check.py`, `run3_3_analysis.py`,
  `pod/r3_3/*` (queue scripts with every command line).
- **No RL and no training beyond Stage 1.** 72 draws: `python3 train.py --mode abs --steps 6000
  --bs 128 --cap 6 --heldout data/p2/heldout.jsonl`, seeds 30–53 on each of
  `train_depth3_f0_a1.jsonl`, `train_reductio_f0.jsonl`, `train_derived_ore_f0.jsonl`
  (155,000 proofs each, 3.2M-parameter model). Reviewer, all 155,000 proofs of each file,
  pruned and unpruned: maximum 6 lines; 0 depth-3, 0 strict reductio, 0 strict derived-ORE —
  and 0 *first halves* too (no line at depth ≥ 3; no `NEGI` on `~goal`; no `ORE` on a derived
  disjunction), so attempting a first half is itself out-of-distribution.
- Pools (`data/r3_3/`; reviewer: the four sampled pools share 0 renaming classes with any of
  the three training files, `heldout`, `validation_36` or the test prompts):
  - depth-3: the whole campaign pool `targets_depth3.jsonl`, 1,000 targets, in two parts
    (part 1 = the first 300, the ignition study's sample; part 2 = the other 700). By
    *minimum* length part 1 is 198 × 7 + 102 × 8 lines (the pre-registration's "166 + 134" is
    the generator's length). By what the shortest proof needs (unrestricted minimum /
    depth-≤ 2 minimum at bound 8): 402 targets 7 / 7 (pattern optional at 7 lines), **167
    targets 7 / 8 and 6 targets 7 / none (173 whose 7-line proof needs depth 3)**, 104
    targets 8 / 8, 136 targets 8 / none, 185 unknown. The registered required@8 stratum is
    142 targets (97 in part 1): 136 with minimum 8 and 6 with minimum 7.
  - reductio: run 5's required pool, 300 targets, 16 schemata; the 7-line ones are `nand_neg`
    26 + `negimp_to_pos` 26, the other 248 are 8–10 lines.
  - strict derived-ORE, cap 6: 363 strict candidates, oracle-labelled (`necessity.py`, bound
    10): 363 reachable, 0 timeouts, 0 inconsistencies, **25 required** (3 × 7, 16 × 8, 3 × 9,
    3 × 10 lines); pool = 25 required + **275 optional, all 7-line, each with a pattern-free
    proof of the same length**.
  - deep pools (addendum): `deep52` = the 52 seven-line reductio targets (chosen by length);
    `deep45` = the 45 depth-3 targets hit by any of the ignition study's 16 draws (chosen by
    sibling draws' outcomes; only 3 of 45 are required@8; 30 of the 76 targets hit in this
    run's main pass lie in it).
- Sampling: main pass `coverage.py --k 2000 --temperature 0.8 --batch 2000 --procs 4 --seed 0`
  per draw and pool (600k samples per 300 targets; depth-3 2.0M per draw); deep pass
  `--k 20000 --seed 1` on `deep52` (1.04M per draw) and `deep45` (0.9M). Reviewer: 144
  coverage files complete, `n_tried` as stated on every record, Σ `count` = `n_ok`, and all
  10,479 distinct stored proofs re-verified with 0 failures and 0 pattern-label differences.
- Counting rule (pre-registered): a draw *generalises* iff ≥ 1 stored proof verifies and
  carries the pattern on the dependency-pruned proof; hits = Σ `count`; rate = hits /
  Σ `n_tried`; Clopper–Pearson 95 % intervals; frozen@256 = first pattern hit index ≤ 256.
- Pods: r33a–r33d, RTX 3090 24 GB at $0.50/h, created 05:23:43 / 05:24:00 / 05:24:12 /
  06:21:26 UTC (`~/pods.log`), deleted 12:45 / 12:25 / 12:38 / 12:48 (`log.md`); ≈ 28.0
  pod-hours ≈ **$14.0** (pre-registered estimate 36–42 pod-hours ≈ $20 + ≈ $5 for the deep
  pass, stop at $35, ceiling $50). See `docs/infra/spend.md`.
- Bucket: `hf://buckets/dan-pandori/nd-rl/round3-run3/{ckpts,artifacts,data}` (reviewer: 72
  checkpoints, 144 coverage files, 7 data files listed).
- Hard constraints (reviewer): `nd_verify` tree hash equal on the branch head, `origin/main`
  and `upstream/main`; `artifacts/TEST_RUN_DONE` is the same blob as on `dan_novelty`, last
  touched 2026-09-15, no test-file run (the orchestrator's gate 1 flagged its mtime, a
  worktree checkout artefact); cap 6 on all supervised data and all 72 checkpoints; gate 0
  met (the pod gate 0 flagged, `r32a` at 05:15:15, is round-3 run 2's).

## Observations

All counts were re-derived by the reviewer from raw files with independent code (only
`nd_verify` is shared with the executor; the eight first-half predicates were re-implemented
from the pre-registration text). No traceback, OOM or `FAILED` in any log; one coverage job
(`depth3 s42 p1`) first ended after 2 records (the 07:07 OOM incident, logged `rc=0`) and was
resumed to completion at 10:58.

**Headline counts (main pass registered; deep pass from the addendum).**

| class (pool, samples per draw) | draws with ≥ 1 hit | 95 % CP | pre-registered |
|---|---|---|---|
| depth-3, parts 1 + 2 (1,000 targets, 2.0M) | **15 / 24** | 0.41–0.81 | 12–18 ✓ |
| depth-3, part 1 alone (300, 600k) | 15 / 24 | 0.41–0.81 | – |
| reductio (300, 600k) | **12 / 24** | 0.29–0.71 | 5–11 ✗ (one above; falsifier ≥ 14 not reached) |
| strict derived-ORE, cap 6 (300, 600k) | **0 / 24** | 0.00–0.14 | 18–24 ✗ |
| reductio deep (52, 1.04M) / either pass | 17 / 24 / **17 / 24** | 0.49–0.87 | – |
| depth-3 deep (45, 0.9M) / either pass | 18 / 24 / **19 / 24** | 0.58–0.93 | – |

- **E2 fails.** The registered intervals overlap almost entirely; Fisher exact p = 0.56 (main
  pass), 0.74 (either pass). By the pre-registration's own rule, "structural patterns are
  generalised more readily by the base" is dropped. Same-seed draws of the two sets do not
  co-generalise (Fisher p = 1.0; only the seed is shared).
- Hits per draw, s30–s53 (executor's vectors, identical to the recount in all 48 entries):
  depth-3 1, 1, 0, 56, 353, 156, 2490, 2, 128, 0, 5501, 0, 0, 0, 0, 16, 0, 2, 7, 0, 0, 2, 2604,
  1; reductio 0, 1, 0, 0, 2, 1, 0, 3, 1, 7, 6, 0, 1, 0, 2, 0, 0, 421, 0, 453, 0, 1, 0, 0. Zero
  in both passes: depth-3 s32, 39, 42, 44, 50 (5); reductio s30, 33, 36, 43, 46, 50, 52 (7).
- **A one-line length horizon, in both classes (reviewer; not stated in the write-up).**
  Every one of the 11,320 depth-3 hits and 899 reductio hits of the main pass is a 7-line
  proof, one line beyond the supervised cap. No draw of either class produces a single
  verified 8-line pattern proof in 600k–2M samples: reductio 416 hits on `nand_neg`, 483 on
  `negimp_to_pos`, 0 on the 248 longer targets (the deep pass the same two schemata); depth-3
  0 hits on its 240 targets with minimum 8. On the reductio pool nothing but strict reductio
  is ever verified.
- **Depth-3 hits by stratum:** 10,111 on the 167 "7 / 8" targets (the depth-≤ 2 alternative
  is 8 lines and is never sampled once on those targets), 77 on the 6 "7 / none" targets,
  1,132 on the 402 "7 / 7" targets (beside 2.50M pattern-free verified samples there), 0
  elsewhere. Part 1 carries 11,206 hits, part 2 114; no draw is rescued by part 2; 76 targets
  are ever hit. The registered required@8 stratum gets **77 of 11,320 hits, in 4 / 24 draws**
  (s36, 38, 40, 52; 0.05–0.37), all on its six 7-line members; its 136 eight-line members get
  none. On that stratum depth-3 generalises in *fewer* draws than reductio on its
  all-required pool (Fisher p = 0.03), entirely because 136 of 142 targets are 8-line.
- **Like-for-like at the horizon** (targets whose 7-line proof needs the pattern): depth-3
  173 targets, **12 / 24** draws with a hit; reductio 52 targets, **12 / 24** — identical
  counts, with depth-3 having 3.3× the sample budget on such targets. Pooled per-sample rate
  on them: depth-3 1.2·10⁻³, reductio 3.6·10⁻⁴; median among generalisers 3.7·10⁻⁴ vs
  1.9·10⁻⁵. If a class difference exists at the base level it is in the rate among
  generalisers (draws at ≥ 10⁻⁴: 6 vs 2 of 24 main pass, 7 vs 2 deep; Fisher p ≈ 0.25), not
  in the probability of emitting the pattern at all.
- **Derived-ORE's 0 / 24 is not a like-for-like zero.** The pool holds 3 seven-line required
  targets (6,000 samples per draw; floor 1.7·10⁻⁴) and 22 required ones at 8–10 lines, beyond
  the horizon where the other two classes also score 0. Required targets: 0 of 600 draw ×
  target pairs solved, although the draws attempt it (`ore_on_derived` on 11,240 samples of
  required targets). The 275 optional targets are solved readily by pattern-free proofs
  (108–170 of 300 targets per draw, 1.61M verified samples in total, 0 strict and 4 loose
  derived-ORE, the latter in 2 draws). What the number supports: strict derived-ORE is never
  chosen when a same-length alternative exists (depth-3 is chosen 1,132 times in 7 draws in
  the analogous stratum) and is below ≈ 10⁻⁴ where required at 7 lines. It does not support a
  class ranking.
- **Concentration.** Rates among generalisers span 2.7 (reductio) and 3.7 (depth-3) decades.
  Two reductio draws (s47, s49) carry 874 of 899 main hits; s49 has 387 of 453 on one target.
  Depth-3: s40, s52, s36 carry 93 %. 8 of 12 reductio and 6 of 15 depth-3 generalisers have
  1–3 hits. Frozen@256: depth-3 9 draws > 0 (max 27 targets), reductio s39 1, s40 1, s47 13,
  s49 5. Post-hoc thresholds: rate ≥ 10⁻⁵ / ≥ 10⁻⁴ in 7 / 4 of 24 depth-3 draws on the 2M
  denominator, 9 / 6 on part 1 alone (the same 600k budget as reductio's 4 / 2): the 1.4M
  part-2 samples that hit almost nothing dilute the depth-3 rate 3.3×.
- **Deep pass.** Of the main-pass zeros, 5 of 12 reductio draws (s32, 41, 45, 48, 53) and 4
  of 9 depth-3 draws (s41, 43, 46, 49) show a hit in ≈ 1M further samples on the easy targets
  (rates 10⁻⁶–10⁻⁵); draws with ≥ 2 main hits are confirmed 7 / 7 and 12 / 12, with exactly 1
  hit 5 / 5 and 2 / 3 (depth-3 s30: 1 main, 0 deep); deep / main rate ratios on the same
  targets 0.83–1.12 (depth-3, 7 draws) and 1.06, 1.02 (reductio). The main-pass zeros are
  Poisson-consistent with the deep rates (expected main hits 0.1–1.3). "Generalises" is
  therefore a statement about a budget: the fraction moves 0.50 → 0.71 (reductio) and
  0.63 → 0.79 (depth-3) with ≈ 10× more samples on the easy targets. `deep45` is
  outcome-selected by sibling draws, so 19 / 24 is a little more favourable to depth-3 than
  17 / 24 is to reductio by construction; it does not change E2.
- **Predictors: none registered works.**

  | | reductio | depth-3 |
  |---|---|---|
  | final held-out loss vs indicator (registered \|ρ\| < 0.3) | 0.03 ✓ — but the logged value has 4 decimals and takes 7 distinct values in 0.0822–0.0829 | −0.10 ✓ — 8 values in 0.0822–0.0831 |
  | full-precision held-out mean vs indicator / either pass | −0.34 (p 0.11) / −0.36 (0.09) | −0.24 (0.26) / −0.42 (0.044) |
  | primary first-half rate vs indicator (registered ρ > 0.5) | **0.14 (p 0.52)** ✗ | **0.04 (p 0.87)** ✗ |
  | same, outcome from the independent deep pass | −0.10 (0.67) | 0.40 (0.057) |
  | first-half rate vs log rate among generalisers | 0.49 (0.11) | 0.53 (0.043) |
  | first-half ≥ 10× verified rate in every generaliser | min 16× ✓ | min 20× ✓ |
  | zero-hit depth-3 draws have `d3_written` < 10⁻⁴ | – | **1 of 9** ✗ (the others 4·10⁻³ – 7·10⁻²) |

  Zero-hit and generalising draws attempt the first half at the same rates (reductio
  2·10⁻³–2·10⁻² in both groups; zero-hit depth-3 draws open a third box in up to 7 % of
  samples and never close one validly). The loss "passes" on a variable with almost no
  variance; on the full-precision loss the sign is consistently negative at |ρ| 0.24–0.42,
  which n = 24 cannot separate from 0 or from the 0.3 threshold. First-half rates rest on the
  executor's predicate code (read by the reviewer: it matches the pre-registration) plus the
  reviewer's check on the verified subset; failed samples are not stored.
- **Post hoc only** (38 per-rule / per-class losses × 2 classes scanned, no correction):
  depth-3 draws with lower held-out loss on `NEGI` lines generalise more, ρ = −0.74
  (permutation p ≈ 10⁻⁴; ≈ 0.008 after Bonferroni over 76), −0.42 against the independent
  deep-pass indicator, −0.62 either pass. The losses are 10⁻⁴–10⁻³ nats, there is no mechanism
  on the table, and it is one survivor of a scan: a candidate to pre-register on fresh draws,
  not a finding. The reductio analogue (`AS`-line loss, −0.45) does not survive correction.
- **External stratum** (ignition study, recounted): reductio 4 / 11 (0.11–0.69) as quoted in
  the brief. Depth-3 "10 / 16" pools four training sets: on `f0_a1`, the recipe used here, it
  is **5 / 10** (0.19–0.81); the other three sets give 5 / 6. Not disclosed in the write-up.
  Neither stratum contradicts the new counts.

**What stands (reviewer's verdict).** At cap 6 and f = 0, about half of draws emit each of the
two patterns at 600k samples and 70–80 % with ≈ 1M more samples on easy targets; the intervals
do not separate and on like-for-like targets the two classes are tied at 12 / 24. What
separates outcomes is not class but length. Clause (2)'s "or nothing" is a statement about a
sampling budget for about a third of apparent zero-rate draws. No registered draw-level
property predicts generalisation.

**Reworded (reviewer's wording replaces the executor's).**

| executor (`run3-3.md`) | reviewer |
|---|---|
| "The full depth-3 pool changed nothing: 99 % of hits … lie in the 300 shortest" and "every reductio hit is on a 7-line target", as two facts | one fact, the main structure in the data: all 12,219 main-pass pattern hits in both classes are 7-line proofs (cap + 1), and 90 % of the depth-3 hits fall on the 173 targets whose 7-line proof needs depth 3. The write-up never mentions written length for depth-3 |
| "The pattern, not its class, sets the base rate"; "'or nothing' fits … every strict derived-ORE draw" | the derived-ORE pool has three required targets inside the 7-line horizon; the other 22 are 8–10 lines, where depth-3 (240 targets) and reductio (248) also score exactly 0. Depth-3 and reductio are tied like-for-like; derived-ORE has not been measured like-for-like. What stands: strict derived-ORE is never chosen when a same-length alternative exists (0 of 1.61M verified samples) |
| required@8 stratum "77 hits in s36, 38, 40, 52", as a count | 4 / 24 (0.05–0.37), fewer draws than reductio on its all-required pool (p = 0.03), entirely because 136 of 142 targets are 8-line: any future "base rate on the required pool" must be quoted per minimum length |
| either pass 19 / 24 vs 17 / 24 | `deep45` is outcome-selected by sibling draws, `deep52` by length alone |
| rate thresholds 7 / 24 and 4 / 24 (depth-3) | on the 2M denominator; 9 / 24 and 6 / 24 on part 1 alone, the budget comparable to reductio's |
| ignition stratum "depth-3 10 / 16" | 5 / 10 on this run's recipe |
| run 5's and the ignition study's "7-line entry" for reductio (earlier results) | not a property of `nand_neg`: it is where the base model's length reach ends |

**Pre-registered expectations vs outcome** (`16378b0`, 05:16:17 UTC; addendum `0c62ffb`,
09:45:47; all marks identical in the executor's write-up and the recount).

| id | expected | observed (reviewer) | verdict |
|---|---|---|---|
| E1 | depth-3 12–18 / 24; reductio 5–11; strict derived-ORE 18–24 | 15; 12; 0 | ✓; ✗ by one; ✗ (the executor's number was the degenerate `( X v X )` form's rate) |
| E2 | depth-3 and reductio intervals do not overlap | 0.41–0.81 vs 0.29–0.71 | **✗ — base-level class claim dropped** |
| E3 | rates span ≥ 2 decades; reductio hits only on 7-line `nand_neg` / `negimp_to_pos` | 3.7 and 2.7 decades; 0 hits elsewhere | ✓ (and not specific to reductio) |
| E4 | held-out loss \|ρ\| < 0.3; primary first-half ρ > 0.5 (depth-3 / reductio) | −0.10 / 0.03; 0.04 / 0.14 | ✓ on a near-constant variable; ✗ |
| E5 | ≤ 2 draws rescued by part 2; ≥ 90 % of depth-3 hits in part 1 | 0; 99.0 % | ✓ |
| E6 | < 50 of 363 strict derived-ORE candidates required; 0 inconsistencies | 25; 0 | ✓ |
| E7 | first-half ≥ 10× verified rate; zero-hit depth-3 draws `d3_written` < 10⁻⁴ | min 20× (depth-3) / 16× (reductio); 1 of 9 | ✓; ✗ |
| DA1–DA4 | 3–7 of 12; all / 3–5; 2–5 of 9; ratio in [0.5, 2] | 5; 7 of 7 / 5 of 5; 4; 0.83–1.12 | ✓ |
| DA5 | fewer than ⅓ of reductio draws zero in both passes → "or nothing" is a budget statement | 7 / 24 | triggered |
| falsifiers | E2 failing; reductio ≥ 14 / 24; any reductio hit on an 8-line-or-longer schema | fired; 12 (17 either pass, not the registered quantity); none | first fired |
| cost | ≈ $20 + ≈ $5 deep, stop $35 | ≈ $14.0 (28.0 pod-hours) | under |

Figures: the executor's `figures/r3_3_generalisation.png`, `r3_3_rate_hist.png`,
`r3_3_deep.png` and `r3_3_predictors.png` are
[on the run branch](https://github.com/dan-pandori/nd-takehome/tree/2106258/figures). The
review did not open or check them, and none of them shows the length horizon, so they are not
copied here.

## Limitations and next step

- One training set per pattern (depth-3 `f0_a1` only; the ignition study's other three sets
  gave 5 / 6 against `f0_a1`'s 5 / 10), 24 seeds each, one model size, cap 6 only.
- The class comparison is like-for-like only for depth-3 vs reductio (12 / 24 each, on 173
  vs 52 targets, unequal budgets); strict derived-ORE has 3 required 7-line targets and is
  unmeasured like-for-like. The generator yields too few: a fair pool needs schema instances.
- The length horizon is a description of 72 draws at one cap; whether it moves with the cap
  is untested here ([round-2 run 2](../2026-09-18-round2-run2-six-new-patterns/README.md)'s
  cap-8 depth-4 result and
  [round-3 run 2](../2026-09-18-round3-run2-eight-line-stratum/README.md)'s six-line reductio
  entry are the neighbouring evidence, neither designed for it).
- First-half rates cannot be re-derived from pulled files (failures are not stored); a
  1-in-100 sample of failed strings per target would fix it at negligible size.
- The `NEGI`-line loss correlation is a post-hoc scan survivor that weakens on independent
  samples (−0.74 → −0.42).
- `deep45` is outcome-selected; the 185 "unknown" depth-3 targets have no depth-≤ 2 label.
- Disclosed deviations (`log.md`): the whole 1,000-target depth-3 pool in two parts instead of
  run 1's required@8 pool (not committed in time); a fourth pod; the deep-pass addendum; the
  narrowed first-half predicate; the shared kill-switch cron left at 2026-09-19 18:00 UTC; the
  07:07 self-inflicted OOM (queue logged `rc=0`; fixed, file resumed) and a 09:47 mis-launch
  from a stale `queue.sh` caught within 4 minutes. The addendum's "depth-3 part 1 shows 5 of
  15 with 1–2 hits" was written on 23 complete draws (s42 part 1 finished later with 0 hits).
- Not re-derived by the reviewer: spend beyond matching pod-log start times, first-half rates
  over unverified samples, a full diff of the bucket. The review's compare table dates its pod
  check "13:50 UTC" although it was committed at 13:23; the librarian's own
  `runpodctl pod list` at 13:30 UTC shows no `r33*` pod.

Next measurement (reviewer; one experiment, reuses the 72 bucket checkpoints, no training,
≈ $5–8): **length-matched required pools, deep sampling.** For each class a pool of ≥ 50
targets whose minimum proof is 7 lines and needs the pattern at bound 10, and a second of
≥ 50 at 8 lines (depth-3: the existing 173 + 136; reductio: the existing 52 + the 8-line
schemata; strict derived-ORE: new schema instances), all 24 draws per class at 20,000 samples
per target. Pre-register (a) the 7-line generalisation fractions of the three classes with
intervals, the first like-for-like class comparison; (b) the 8-line fraction, expected
≈ 0 / 24 in every class if the horizon reading is right and > 0 somewhere if "7-line entry" is
schema-specific; (c) `NEGI`-line loss vs the depth-3 7-line indicator, registered as ρ < −0.4.
If (b) is ≈ 0 everywhere, the follow-up is a cap-7 Stage-1 on the same seeds: does the horizon
move to 8 lines (length reach is what RL has to elicit from) or stay at the schemata?
