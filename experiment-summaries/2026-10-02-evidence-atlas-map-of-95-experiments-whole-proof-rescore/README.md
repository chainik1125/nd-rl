---
written_on: 2026-10-02
written_by: agent:claude
---

# evidence-atlas: one map of 95 experiments, and a whole-proof re-score showing proof state beats whole proof on dev and holdout250

Status: completed and reviewed 2026-10-02. No hard-constraint violation and nothing quarantined. **The re-score's 24
per-seed counts reproduce exactly**, and every counted theorem is Lean-confirmed. Five rewordings, one table fix and two
minor findings are applied in the nd-rl copy of the atlas, [`ATLAS.md`](ATLAS.md) (changes marked *[review]*). The
deliverable is a **map for Dan**, not a test of a hypothesis. Its one new measurement is the pre-registered whole-proof
re-score (Addendum 1).

Question: what has the project measured, under which protocol, and which cross-experiment comparisons are apples to
apples? The run produces (a) one row per experiment by question; (b) textbook72, dev1108 and holdout250 counts and the
length frontier across every model family that was scored; (c) the create-vs-elicit evidence with strength ratings.
Brief: [`proposals/claude-heavy/BRIEF_evidence-atlas.md`](../../docs/proposals/claude-heavy/BRIEF_evidence-atlas.md)
(proposal 21).

## Configuration

- Fork `dan-pandori/nd-takehome`, branch `dan_evidence-atlas`. Pre-registration `preregistration/evidence-atlas.md`
  (`453cebab`, 16:53Z). Addendum 1, the re-score with its expected ranges, is `b4f08b79` (17:23:40Z). Both were committed
  before the first pod (17:24:44Z). Atlas `d97ec12a`, DONE `2c93f589`. Review
  [`review_evidence-atlas.md`](https://github.com/dan-pandori/nd-takehome/blob/1a04b3e6/review_evidence-atlas.md):
  [`02b6f6b4`](https://github.com/dan-pandori/nd-takehome/commit/02b6f6b4) §Recount (blind),
  [`1a04b3e6`](https://github.com/dan-pandori/nd-takehome/commit/1a04b3e6) §Comparison / §Verdict. Reviewer scripts:
  `review_ea/` on the fork.
- Executor Fri 2026-10-02 16:53–18:36 UTC; reviewer 18:36–18:48.
- **Map.** Seven read-only extraction agents read nd-rl `dan` summaries, fork run branches and other members' nd-rl
  branches into a fixed schema (`atlas/raw/`). They did not push to or comment on anyone's branch. Shared-pool numbers
  were then recomputed from per-theorem files: 243 fork read-outs from the bucket `.jsonl`, Robbie's factorial
  `passk.csv`, and the long-pool per-stratum files (878 rows). There were 0 disagreements with the summaries. Every other
  number is marked `copied`, with its source.
- **Re-score (Addendum 1).** Eight existing checkpoints, all **3,214,336 params (4 × 256), `lean_seq` whole proof,
  from scratch, Lean alone** (`eval_set.py` → `lean_judge`; `nd_verify` unmodified and unused):
  - **C0 cap 6** (cap-6 control set, depth3 f0 a1, 155k): frozen `lean-format/ckpts/lf/stage1_a1_seq_s{0,1}.pt`; T1
    `ds-generator/ckpts/ladder/la_T1_c0_s{0,1}_r8.pt`.
  - **K12 cap 12** (`data/kh/train_k12.jsonl`, 155k): frozen `cap-horizon/ckpts/kh/stage1_k12_s{0,1}.pt`; T1
    `state-cap12/ckpts/ladder/la_T1_K12_s{0,1}_r8.pt`.
  - Pools: textbook72 at k 256, dev1108 at k 64, holdout250 at k 256. T 0.8, sample seed 0, **batch 2,048** (the
    pre-registration said 4,096, which ran out of memory with two jobs per card; the batch is the same in every arm),
    `max_new` 512, peak 8.0–8.4 GB per job. Two jobs per RTX 3090.
  - Truncation check: the 11 reads with > 0.1 % of samples at the cap were re-read at `max_new` 1,024, with the same
    batch and seed.
- **Pods** (`~/pods.log`, `~/podhours.log`): `ea-rescore` A40 (Lean missing on the pod, 0.15 h); `ea-rescore2` RTX 3090
  (out of memory at batch 4,096, 0.19 h); `ea-rescore3` RTX 3090 (the 24 reads, 0.37 h); `ea-trunc` / `ea-trunc2` RTX 3090
  (the truncation re-reads, 0.30 / 0.07 h). **1.08 pod-hours, $0.54** of a $5 / 10 h budget. The $0.54 is `podbudget`'s
  $0.50/h fallback rate, not a verified billed rate.
- **Compute per arm** (`atlas/data/compute_rescore.csv`, parsed from job logs): 491–574 GPU-s, 306,688 attempts,
  45.2 / 50.3 / 56.8 / 62.3 M generated tokens, 0 training steps. Two jobs shared each 3090, so GPU-seconds are
  shared-card wall time. Registry rows are at `hf://buckets/dan-pandori/nd-rl/registry/evidence-atlas/`.
- **Bucket:** `hf://buckets/dan-pandori/nd-rl/evidence-atlas/{artifacts,atlas}`.
- **Files here:** [`ATLAS.md`](ATLAS.md) (the reviewed atlas), [`MAP.md`](MAP.md) (one row per experiment), `figures/`
  (five figures), and `data/` (the small tables; `protocol_differences.csv` re-quoted). The harmonized CSVs, `raw/` and
  `scripts/` are at [`atlas/`](https://github.com/dan-pandori/nd-takehome/tree/1a04b3e6/atlas) on the fork.

## Observations

Observed results below use the reviewer's wording where it differs from the executor's.

**Re-score: solved counts per seed (s0, s1), 3.2M `lean_seq`, from scratch, Lean alone.**

| model | textbook72 /72 | dev1108 /1108 | holdout250 /250 |
|---|---|---|---|
| C0 cap 6 frozen | 12, 11 | 54, 36 | 15, 8 |
| C0 cap 6 T1 | 15, 13 | 417, 465 | 104, 102 |
| K12 cap 12 frozen | 21, 21 | 415, 384 | 115, 104 |
| K12 cap 12 T1 | 25, 22 | 797, 771 | 181, 177 |

- The reviewer recounted all 24 reads from the per-theorem files: identical. They Lean re-checked 4,356 proofs with
  their own theorem headers and harness: 0 rejected, and every solved count was re-derived. Negative controls (goal set
  to `False`; a different theorem): 0 / 1,200 accepted.
- **Proof state vs whole proof** at the same 3.2M recipe, cap and stage (state = SN-v2 cap 6 / SN-cap12, `lean_staten`).
  Whole proof is below state **in all 8 dev1108 / holdout250 cells, by 41–322 theorems** (holdout250 +81.5, +57, +79,
  +41). **On textbook72, only the cap-12 T1 cell (+13.75) is outside the ≈ 7 floor** the addendum set. Cap-12 frozen
  (+7.25) is at the edge. The two cap-6 cells (+3.5, +5) point the same way but are not differences. Whole proof has 2
  seeds, state 2–4, and the state reads used batch 4,096 and `max_steps` 96 / `max_action` 512. Statement (i) holds as
  a direction.
- **C0 T1 vs Robbie's `lean-naive-ei`** (statement (ii)): 15, 13 vs 8, 6, 10 on textbook72, and 104, 102 vs 45, 42, 37
  on holdout250. It **holds**. Robbie's cells are Lean gate ∧ `nd_verify`, and on textbook72 they were RL-trained on
  14 of the 72 theorems. Both differences favour his cells, so the direction stands a fortiori. His model size is not
  recorded.
- **Expected ranges:** 6 of 12 cells are in range in both seeds. The frozen bases fell far below on long theorems:
  C0 frozen holdout250 15 / 8 against an expected 30–90, and dev 54 / 36 against 100–400; K12 frozen holdout250
  115 / 104 against 120–190, and dev 415 / 384 against 450–750. K12 T1 textbook72 s1 (22) fell below its range of 25–38.
- **Truncation, statement (iii): missed.** 11 of 24 reads had more than 0.1 % of samples at `max_new` 512 (max
  0.258 %). At 1,024 all 11 gave the **identical solved set**. Four reads stay above 0.1 % at 1,024 (max 0.233 %), and
  accepted proofs are at most 18 lines, so these are non-terminating samples. The cap does not bias the counts.
- **Re-draw spread:** reading the same checkpoint at batch 4,096 instead of 2,048 moved counts by −5 to +7 on dev, −2
  to +3 on holdout250 and −4 to 0 on textbook72 (11 pairs).
- **Term size** (reviewer's sample): median 8–11; max 11 / 23 / 35 / 44 for C0 frozen / C0 T1 / K12 frozen / K12 T1.

**The map** (reviewed for counts and traceability, not for its judgements):
- 95 experiments: 81 by Dan's agents and the earlier sprints, 14 by other members. Expectation 1 (≥ 55): hit.
- **Top textbook72 family: best-cap12** (9.56M `lean_staten`, cap 12, after the T1 ladder). Its round-8 reads are
  51–53 in best-state and 48–54 in `trajectory`'s re-run. **The highest number on file is 57** (`trajectory`
  `la_T1_best12_s2_r6`, seed 2, round 6), not ≈ 52. **Expectation 2 is a partial miss**: the family is right, the
  number is not. No Robbie factorial cell exceeds 32. 7 of 201 textbook72 families have a k 256, Lean-alone row.
- **Create vs elicit:** 11 lines of evidence, none rated above "moderate" (expectation 4's first part: hit). 14 of the
  23 create-vs-elicit rows were judged by `nd_verify` alone, and the fork's `nd_verify` still accepts forward box
  citations (Leon's finding). The ratings follow a stated rubric and are judgements, not measurements.
- **Recipe after RL:** best (9.56M) minus ours (3.2M) on textbook72 is +4.0 / +0.5 frozen and +18 / +14 after T1. This
  is the recipe **and size**, not the pretraining recipe alone.
- Spot checks: harmonized rows match their per-read files (24 / 24 best-state, 12 SN seeds, the Robbie rows checked).
- **Split hygiene (reviewer's finding).** `train_k12` contains one textbook72 theorem, disjunctive syllogism
  `(P v Q), ~P ⊢ Q` (`textbook_3ed45280…`), with its premises swapped. Every cap-12 family's textbook72 count (K12
  whole proof, SN-cap12, best-cap12) is therefore inflated by ≤ 1. Comparisons between cap-12 families are unaffected.
- Expectation 3 (1–5 headline comparisons in `docs/STATE.md` or the digests mix checkers or k without saying so) was
  not reported against a count by the executor, and the reviewer did not recount it.
- Expectation 5 ("0 pods", p ≈ 0.7): missed, 5 pods. Addendum 1 had already planned one.

## Limitations and next step

- The whole-proof cells have 2 seeds, so textbook72 differences under ≈ 7 cannot be called. The state comparators were
  read at another batch and decode limit (a sampling re-draw, not a correctness change; see `NOISE_FLOOR.md`).
- Fork ↔ Robbie textbook72 comparisons are only **partly** apples to apples: his cells trained on 14 / 72, use a
  different checker, have an unrecorded model size, and ran a 2,400 s pipeline against our T1 ladder.
- The inherited headline numbers (rr600 2 % → 87 %, support survivors 28 / 29, depth-3 13–28 %, Robbie's main effects,
  "6–11 of 70 flips") were not re-derived here. They carry their source runs' review status.
- The billed rate was not verified ($0.54 is the fallback rate; the cost is under $1.5 either way).

Next (reviewer):
- (a) A paired textbook72 interface read, whole proof vs state at matched cap, with 4+ seeds each, to call the cap-6
  cells.
- (b) Fork vs Robbie on the 58 `textbook_dev` theorems only, for a held-out comparison.
- (c) Frozen best-cap12 pass@4,096 on textbook72. This is the base-reachability number that every "RL solved X" on that
  pool lacks.

The executor's largest open question: best-cap6's 13–16-line solve rate goes 2 % → 87 % under RL, but frozen reach was
read only at k 256. A support test at k ≥ 10⁴ would place this, the largest RL effect in the atlas, on the
create/elicit axis.
