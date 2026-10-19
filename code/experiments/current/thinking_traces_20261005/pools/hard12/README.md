---
written_on: 2026-10-06
written_by: agent:claude
---

# hard12: hard RL targets with reference proofs of at most 12 lines, and a matched sealed test set

The earlier hard pools (`rl_hard*.jsonl`) were too easy. The base model reached about 88% pass@32 on them, and about
97% pass@1 after a little RL, so RL learned to drop its thinking. hard12 keeps the reference proofs inside the
pretraining range (7–12 lines, the K12 rule set, Dan's generator with no new rules). It selects theorems that the
no-think Stage-1 model T0_s0 solves on only **2–16 of 32 attempts**: a per-attempt pass rate of 6–50%.

- **RL pool:** `/scratch/cowifornia/nd-rl/think/runs/pools/rl_hard12.jsonl` (5,000 targets). It has the ladder
  `rl_targets.jsonl` fields, plus `t0_n_ok` / `t0_k`, the generator config `cfg`, `band_score` and `f_*` features.
  Meta file: `rl_hard12.meta.json`, next to it.
- **Sealed test set:** `/scratch/cowifornia/nd-rl/think/runs/pools/hard12_test/hard12_test.jsonl` (2,500
  targets). It uses the OOD-set row format that `code.thinking.eval --sets` reads (`name`, `prompt`,
  `reference_lines`, `reference_proof`, `family`, `knob`, ...).
  - Meta file: `hard12_test.meta.json`.
  - Renaming-class sidecar: `hard12_test.keys.jsonl` (`name`, `key`, `pkey`).
- **Copies of the meta files:** [`meta/`](meta/). The folder also holds the task-1 feature analyses, the band-score
  model and `build.json`.
- **Intermediates:** `/scratch/cowifornia/nd-rl/think/runs/pools/hard12_work/`. It holds the candidate streams, the
  pass-rate rows of every measured candidate (`passrate/*/rows.jsonl`) and the logs.
- **Code:** this folder, branch `leon/thinking`, uncommitted, on top of 8644bf4. [`regenerate.sh`](regenerate.sh)
  runs the whole recipe.

**Result in brief.**

- **Yield:** 24,050 score-selected candidates were measured, and 8,939 (37%) landed in the band. 6 were excluded by renaming
  class and 0 references were rejected by Lean, leaving 8,933 usable targets. Of these, 5,000 form the pool,
  2,500 the test set, and 1,433 are spare (`hard12_work/spare_band.jsonl`, disjoint from both).
- **Pass rates on the band:**
  - T0_s0 per-attempt pass@1 is 28% at selection. A fresh T0 re-measurement of 1,000 + 1,000 targets gives 29.4% /
    28.4%, pass@32 97.3% / 97.2%, with 82–83% still in the band.
  - T1v1 (B 256) reaches 44% pass@1 on the same targets but no higher pass@32 (95.6% / 95.8%, 1.4–1.7 pp below
    fresh T0).
- **Features:** the hard ones are non-implication goals (negation, atom), proofs that start forward from the
  premises (ANDE / IMPE / NEGE), jointly inconsistent premises (ex falso), and several disjunctive or implicational
  premises. The generator's own case splits, reductio and classical-only shapes make targets *easier*.

## 1. What makes a short target hard (existing measurements)

**Data.** The previous pool agent measured T0_s0 at k = 16 (T 0.8, plain, Lean-judged) on 9,900 uniformly drawn
7–12-line candidates from the strict long generator (`pools/passrate/short*`). The analysis
([`analyze.py`](analyze.py), [`meta/feature_analysis_k16*.json`](meta/)) computes features of each theorem and its
reference proof ([`features.py`](features.py)). It compares them at fixed reference length in two ways:

- **Binned rates.** Each bin's difference from the length-matched mean, in pass@1 and in the band share (1–8 of 16
  successes, the k = 16 analogue of the final 2–16-of-32 filter).
- **A binomial GLM.** Successes of k, logit link, with length fixed effects and all features jointly.

**Formula depth comes first, and it is a format effect.** Length-adjusted pass@1 by formula depth:

| depth | ≤4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|
| Δ pass@1 | +4 to +8 pp | −1.6 | −13.8 | −27.9 | −54.4 |
| n | 6,037 | 2,247 | 1,155 | 385 | 72 |

At depth 8, 54% of targets are never solved. OOD_CHECK.md already found that the model breaks on format at depth ≥ 7.
K12 has 97% of its theorems at depth ≤ 5 and 2.8% at depth 6. **All hard12 targets therefore have depth ≤ 5.** The
rest of this section uses only depth ≤ 5 (n = 8,284; mean pass@1 75%, band share 15.9%).

**Effects at depth ≤ 5.** Each cell is the length-adjusted difference against targets without the feature: Δ pass@1
(pp) / Δ band share (pp), with n. The GLM column is the joint log-odds of one attempt succeeding.

| feature | Δ pass@1 / Δ band | n | GLM (log-odds) |
|---|---|---|---|
| goal is a negation (proof ends in NEGI) | −15.9 / +15.4 | 190 | −0.86 (vs `>` goals) |
| goal is an atom | −8.5 / +8.4 | 207 | −0.47 |
| goal is a disjunction | +3.5 / −3.5 | 842 | −0.37 |
| reference uses ANDE | −11.5 / +9.8 | 1,554 | −0.85 |
| reference uses IMPE | −7.8 / +7.0 | 1,161 | −0.43 |
| reference uses NEGE | −4.5 / +3.0 | 2,783 | −0.56 |
| reference uses BOTE (ex falso) | −6.3 / +5.3 | 1,655 | +0.02 |
| premises jointly inconsistent | −11.1 / +8.3 | 979 | – |
| first written step is forward from the premises (not an assumption) | −5.7 / +5.2 | 1,697 | −0.31 |
| first step assumes the goal's antecedent (goal-directed) | +1.5 / −1.2 | 5,190 | (reference) |
| first step is a case split on a premise disjunction | +5.7 / −5.3 | 915 | −0.02 |
| 3 premises | −7.9 / +8.6 | 566 | – |
| ≥ 2 disjunctive premises | −17.6 / +30.5 | 54 | – |
| ≥ 2 implication premises | −17.3 / +27.5 | 41 | – |
| a single box in the proof | −25.4 / +27.3 | 92 | – |
| reference uses ORE | +3.0 / −2.4 | 5,838 | +0.67 |
| reductio (DN of a NEGI) | +5.1 / −7.3 | 558 | – |
| classical-only (not intuitionistically valid; fork `intuit.py`) | +0.8 / −0.9 | 423 | +0.09 |
| reference uses DN | −2.4 / +1.4 | 1,464 | −0.43 |
| reference uses R (reiteration) | +7.9 / −6.5 | 3,284 | – |

The band share by depth-≤5 feature mix is the model in [`score.py`](score.py): logistic, length dummies plus 34
features. Its held-out AUC is 0.73. Held-out band share by score quintile is 4.2 / 7.9 / 12.1 / 20.2 / 34.3%.

**Reading (interpretation, not tested causally):**

- **Hard:** short targets whose first move is not dictated by the goal's connective. Examples:
  - a negated or atomic goal;
  - a proof that must first work forward from the premises (ANDE / IMPE / NEGE chains);
  - premises that are jointly contradictory, so the proof has to find the contradiction and use ex falso;
  - several disjunctive or implicational premises in scope.
- **Easy, despite the task's hypotheses:** the generator's own case splits, reductio and classical-only shapes. In
  the strict long generator these are formulaic. ORE often splits `P v P` or copies a branch, and the reductio
  wrapper is always "assume ¬G, derive F, DN". So they are recognisable patterns, not choices.
- **Measurement noise:** the GLM standard errors ignore overdispersion; use them for ranking only.
- **Distractor premises:** the generator's `Gen.finish` prunes every premise the reference does not cite, so
  generator output has no strictly unused premises (`n_unused_prem` is always 0). The nearest observable is
  `n_redundant_prem`: premises the reference cites, but without which the goal still follows classically. It has
  little effect, except at 3+ (−7.0 / +8.3 pp, n = 174). hard12 adds no distractor premises: that would have been a
  theorem transformation outside the generator.

## 2. Steering the generator (existing knobs + rejection sampling)

**Knobs.** [`gen_steer.py`](gen_steer.py) uses the fork's `gen.Gen` unchanged, importing it read-only. Every config
sets only parameters that `Gen` and `sample_one` already expose:

| config | Gen(max_prem, max_depth) | mode | strict | other knobs |
|---|---|---|---|---|
| `strict_long` | 3, 3 | `sample_one(long=True)`: the ladder / previous-pool generator | yes | ore_steps 3, budgets 3–5, gdepth 2–3 |
| `takehome` | 3, 3 | `sample_one(long=False)`: K12's own generator | no | ore_steps 1, bot_p 0.02 |
| `fwd_takehome_d2` | 3, **2** | `Gen.forward`, 4–16 steps, 1–3 initial premises (weights 1,2,2) | no | ore_steps 3 |
| `fwd_strict_d2` | 3, **2** | `Gen.forward`, 6–22 steps, 1–3 initial premises | yes | ore_steps 3 |
| `fwd_strict_d1` | 3, **1** | `Gen.forward`, 6–22 steps, 1–3 initial premises | yes | ore_steps 3 |
| `goal_takehome` | 3, 3 | `Gen.goal`, 0–2 premises, gdepth 2–3, budgets 3–5 | no | ore_steps 3 |
| `goal_strict` | 3, 3 | `Gen.goal`, same | yes | dropped: 100% `>` goals, mean score 0.05 |

**Gates.** Every sample must pass the following:

- reference 7–12 lines;
- no premise whose negation is also a premise (the ladder's `contra_prem` rule; semantically inconsistent premise
  sets are kept);
- `nd_verify` accepts it;
- formula depth ≤ 5;
- prompt ≤ 156 tokens and reference states ≤ 386 tokens (K12's maxima);
- a renaming class that is new to the worker;
- **band score ≥ threshold**: rejection sampling on the task-1 model, ≥ 0.2 for the stream and 0.3–0.4 per batch at
  selection.

The strict_long candidates are the previous agent's `cands_clean.jsonl`, scored, with the top 9,000 unmeasured ones
taken.

**Calibration (pilot, k = 32).** Share of targets in the band, by config, all at band score ≥ 0.3:

| config | band share, pilot (n) | band share, all measured (n) | mean T0 pass@1, all measured | unsolved |
|---|---|---|---|---|
| `fwd_takehome_d2` | 40% (200) | **43.0%** (5,200) | 42.4% | 10.0% |
| `fwd_strict_d2` | 38% (150) | 38.9% (3,150) | 46.0% | 9.4% |
| `fwd_strict_d1` | 36% (200) | 38.4% (2,200) | 47.1% | 8.0% |
| `takehome` | 38% (200) | 36.2% (3,300) | 53.2% | 5.6% |
| `strict_long` (cands_clean, top 9,000 by score) | – | 33.8% (9,000) | 51.8% | 7.5% |
| `goal_takehome` | 30% (200) | 33.3% (1,200) | 55.8% | 5.2% |
| all | – | 37.2% (24,050) | 49.0% | 8.0% |

**Score thresholds and counts by batch** (`*.pick.json`):

- **Pilot:** score ≥ 0.3, 950 candidates, plus 100 unfiltered `takehome`.
- **b1:** the top 9,000 `strict_long` (score ≥ 0.26).
- **b2a:** score ≥ 0.4: `fwd_takehome_d2` 3,500, `fwd_strict_d2` 2,500, `goal_takehome` 1,000.
- **b2b:** score ≥ 0.35: `takehome` 3,000, `fwd_strict_d1` 2,000.
- **b3:** score ≥ 0.4: `fwd_takehome_d2` 1,500, `fwd_strict_d2` 500.

The b2 and b3 picks allow at most 30% per reference length.

The band score doubles the yield over unselected K12-generator output: `takehome` with no score filter, n = 100,
was 18% in the band at a mean pass@1 of 76.5%.

**Feature mix against K12.** The full tables are under `rule_mix_vs_k12` in the meta file:

| | K12 train, 7–12 lines (n = 10,854 of a 20k sample) | hard12 pool | hard12 test |
|---|---|---|---|
| goal `>` / `&` / `v` / atom / `~` / F | 44 / 23 / 19 / 7.0 / 5.1 / 1.9% | 55 / 19 / 9.3 / 6.4 / **10.0** / 0.4% | 56 / 19 / 8.2 / 5.6 / **10.5** / 0.4% |
| first step: forward from premises / assume goal antecedent / case split / other assumption | 37 / 28 / 27 / 8% | **60** / 25 / **5.5** / 10% | **60** / 26 / **5.6** / 9% |
| references using ANDE / IMPE / NEGE / BOTE | 11 / 49 / 31 / 14% | **41** / 55 / 39 / **23%** | **42** / 56 / 38 / **22%** |
| references using ORE / R / DN / NEGI | 80 / 38 / 18 / 20% | **47** / **9.7** / 17 / 17% | **47** / **9.8** / 17 / 17% |
| reductio / classical-only / inconsistent premises | 14 / 4.4 / 29% | 3.8 / 3.6 / 33% | 4.1 / 3.6 / 33% |
| 3 premises / ≥ 2 disjunctive premises | 31 / 1.0% | **42** / **4.3%** | **42** / **5.0%** |
| formula depth 4–5 / ≥ 6 | 40 / 3.4% | **68** / 0% (capped) | **69** / 0% |
| box depth 3 | 0% | 11.8% | 11.7% |
| mean reference lines / prompt tokens | 9.5 / 39 | 9.7 / 47 | 9.7 / 46 |

## 3. Adversarial filter (T0 pass@32)

- **Measurement:** [`run_passrate.sh`](run_passrate.sh) and [`pull2.sh`](pull2.sh) (a 4-GPU work queue) run the
  previous agent's `pools/passrate.py`:
  - T0_s0, k = 32, plain arm, no think;
  - T 0.8, line forcing, `max_steps` 96;
  - Lean on every finished attempt.
- **Kept:** 2 ≤ n_ok ≤ 16.

**Yield, candidates to kept:**

| stage | count |
|---|---|
| steered generator draws (pilot + main stream) | 29.9M tries; 1.29M pass the shape gates and are scored |
| kept by score (≥ 0.2 stream) and new class | 357,010 (main) + 180,926 (pilot, no score floor) |
| after exclusion | 347,088 + 168,242 |
| strict_long stream (`cands_clean`, 7–12 lines, depth ≤ 5) | 88,926 scored, top 9,000 taken |
| **measured (T0, k = 32)** | **24,050** (9,000 strict_long + 15,050 steered) |
| **in the band (2–16 of 32)** | **8,939 (37.2%)**, against about 18% for unfiltered K12-generator output |
| excluded / duplicate within the band / Lean-rejected | 6 / 0 / 0 |
| kept | 8,933 → pool 5,000, test 2,500, spare 1,433 |

**T0 pass@32 histogram at selection** (successes of 32, % of the set):

| set | 2–4 | 5–8 | 9–12 | 13–16 | mean pass@1 |
|---|---|---|---|---|---|
| pool (n = 5,000) | 22.4 | 24.6 | 24.9 | 28.1 | 28.0% |
| test (n = 2,500) | 22.0 | 24.1 | 28.2 | 25.8 | 27.8% |

Per value, 2/32 to 16/32: pool 401 393 328 315 323 279 313 300 308 332 305 331 341 344 387; test 201 198 150 157 152
142 151 163 179 198 165 165 167 161 151 (also in the meta files).

**T1v1** (thinking Stage-1, B 256, plain, T 0.8, k 32) was measured on 1,000 random pool targets and 1,000 test
targets. It is for reference only and was not used to filter.

A fresh T0 re-measurement of the same targets (new seeds 120–127) is the fair baseline: the selection
measurement is 2–16 by construction and would regress toward the mean.

| % of 1,000 targets | 0 | 1 | 2–4 | 5–8 | 9–12 | 13–16 | 17–24 | 25–32 | mean pass@1 | pass@32 | in band |
|---|---|---|---|---|---|---|---|---|---|---|---|
| T1v1 B 256, pool sample | 4.4 | 3.1 | 7.8 | 14.5 | 12.3 | 15.9 | 30.4 | 11.6 | **44.2%** | 95.6% | 50.5% |
| T1v1 B 256, test sample | 4.2 | 2.8 | 10.2 | 12.1 | 12.4 | 16.1 | 30.4 | 11.8 | **44.4%** | 95.8% | 50.8% |
| T0 fresh, pool sample | 2.7 | 3.9 | 14.5 | 25.5 | 24.1 | 17.7 | 11.1 | 0.5 | 29.4% | 97.3% | 81.8% |
| T0 fresh, test sample | 2.8 | 3.8 | 16.9 | 25.5 | 23.6 | 17.3 | 10.0 | 0.1 | 28.4% | 97.2% | 83.3% |
| T0 at selection, pool sample | – | – | 21.0 | 25.0 | 25.7 | 28.3 | – | – | 28.4% | 100% | 100% |

**Observed:**

- Regression to the mean is small: +1.0 pp at pass@1 in the pool sample.
- T1v1 thinks: about 607 tokens per attempt, against about 170 for T0. It has +15 pp pass@1 over fresh T0 and a
  wider spread, but 1.4–1.7 pp *lower* pass@32.

**Interpretation:** T1v1 still has headroom on these targets (half of them are in its own 2–16 band), so the pool
is not saturated for the thinking model either.

**Length histogram** (reference lines):

| lines | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|
| pool | 607 | 956 | 754 | 691 | 986 | 1,006 |
| test | 314 | 499 | 361 | 358 | 495 | 473 |

Measured, all candidates: the band share is 38.0 / 35.2 / 34.0 / 36.7 / 39.0 / 40.1% for 7–12 lines.

## 4. Exclusion and dedupe

- **Keys:** `code.thinking.data.class_keys`, the same keys as `EvalIndex`: the fork's atom-renaming `canon_key` and
  ndbench's premise-order-invariant `pkey`. A target is dropped when either key appears in any of these files
  ([`exclude.py`](exclude.py); 98 files and 301,565 classes at the final build):
  - K12 train, the K12 held-out set and the cap-6 Stage-1 set;
  - every jsonl under the fork's `data/`: rl_targets, transfer, transfer_long*, the long pools, the p2 / r1–r3 / bs /
    gt / lj sets, and textbook72 dev and train;
  - the five OOD sets;
  - `rl_hard.jsonl` and `rl_hard_filtered.jsonl`;
  - `test_21_30/test_21_30.jsonl`: it existed at the final build (2,775 rows, indexed), 0 hits.
- **Within the band:** duplicate classes are dropped, keeping the first measured.
- **Pool vs test:** the two come from one random split of that deduplicated set, so they share no class key and no
  pkey (checked in `build.json`).

**Hits.**

- **Before measurement:** the steered streams were filtered against the same index (then 97 files). It removed 18,660 candidates,
  mostly K12 train 11,767, the cap-6 Stage-1 set 7,949, `cands_ori_ore` 1,072 and K12 held-out 986. It also removed
  `rl_targets` / `_v1` 858, `rl_hard` / `_filtered` 278, textbook72 copies 57 and one OOD-set class.
- **At the final build:** the index removed 6 band targets, all from the strict_long stream, all hitting the cap-6
  Stage-1 set. The previous agent's `prep.py` had not indexed that file.

**Final checks (re-run independently):**

- pool and test have 0 index hits and 0 duplicate keys or pkeys;
- pool ∩ test = ∅ by key and by pkey (`build.json`, and `EvalIndex(test).hit` on every pool prompt);
- every reference has 7–12 lines, ≤ 3 premises, ≤ 4 atoms, formula depth ≤ 5, prompt ≤ 127 tokens and states
  ≤ 316 tokens;
- all 8,933 kept references pass Lean (16 workers);
- the test set loads with the fork's `guided_eval.load_sets`, which `code.thinking.eval --sets` uses.

## Caveats

- **Hardness is measured, not structural.** References are generator proofs, not minimal proofs (as in K12). Some
  targets have proofs much shorter than the reference, such as the pool row `h12_h12b2a_5580`, whose ORE is
  unnecessary. Each target is kept because T0_s0 solves it 2–16 times in 32, not because its shortest proof is long.
- **Selection is on one checkpoint and one draw.** Targets near the band edges move on re-measurement under fresh T0
  sampling:
  - 82–83% stay in 2–16;
  - about 7% fall below 2 (2.7% unsolved);
  - about 11% exceed 16.

  Other T0 seeds or a post-RL model will differ.
- **Failure modes.** In the pool's failed T0 attempts, the state interface reports:
  - an unbound name in 37%;
  - a binder mismatch in 25%;
  - an exact-goal mismatch in 15%;
  - a malformed formula in 8%.

  So most failures are invalid or wrong-direction steps, not syntax, but no failure was audited by hand. Formula
  depth is capped at 5 because deeper formulas fail on format.
- **Out of K12's range on one axis.** 11.8% of pool targets have a reference nested 3 boxes deep (547 strict_long,
  41 takehome; both run `Gen` with max_depth 3); K12's maximum is 2. At fixed length this had no measurable effect
  on pass rate in the task-1 data (−0.1 pp).
- **Mix.** The pool is:
  - 33% strict_long (the previous pool's generator);
  - 43% the take-home generator family (`fwd_takehome_d2` 25%, `takehome` 14%, `goal_takehome` 5%);
  - 23% strict forward streams (`fwd_strict_d2` 14%, `fwd_strict_d1` 9%).

  The test set has the same mix within 1–2 pp. The strict_long stream is the previous agent's `cands_clean.jsonl`,
  minus every `rl_hard*` class.
- **Not enriched:**
  - case splits, reductio and classical-only shapes, because they made targets easier;
  - distractor premises, because the generator prunes unused premises and adding them would have changed the
    theorems outside the generator.
- **GPU incident.** The first queue worker (`pull.sh`, removed) checked only memory. A pass-rate process drops to
  about 400 MiB while Lean judges a chunk, so at 00:05 UTC the worker started a second shard (b2_q05) on GPU 3 next
  to b1_s2.
  - b2_q05 ran out of memory. The worker then started b2_q06 there at 00:09, which was killed after 2 min.
  - Both were re-queued, and `pull2.sh` now also requires no compute process on the GPU.
  - Both processes were mine; no other job was affected, and no data was lost, because rows are written per
    finished chunk and runs resume.
  - Separately, b1_s2 was paused at a chunk boundary so the pilot could run on GPU 3 first.
  - GPU 0 was never used.
- **Not committed.** The previous pool scripts are used unchanged; the only new files are this folder.

## Regenerating

[`regenerate.sh`](regenerate.sh) runs every step in order:

- the task-1 analysis and band model;
- the exclusion index;
- the streams and selections;
- the T0 measurements (4-GPU queue);
- the build (band filter, exclusion, Lean, split, keys);
- T1v1, and a fresh T0 re-measurement of the same targets;
- the meta files.

Sampling on a GPU at T 0.8 is not bit-reproducible, so a rerun gives a statistically equivalent set, not the same
one. The CPU steps are seeded: generator workers use seed + 1000·i; selection seeds 1, 2, 21, 22 and 23; the split
uses seed 12012 and the T1v1 sample 12013.
