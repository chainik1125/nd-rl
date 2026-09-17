# Run brief: round3-run3 — base generalisation by pattern class, many seeds, no RL

Run id `round3-run3`. Role: executor. Fork `~/nd-takehome`, branch `dan_round3-run3` from
`origin/dan_novelty` (or its merged successor). `AGENT_POLICY.md` governs; designs are
suggestions. Read first: `ignition.md` and `review_ignition.md` (refinement 3), `review_followup.md`
(block B caution), and §3 of `~/nd-rl/docs/proposals/2026-09-18-proposals-round-3.md`.

## Question

With what probability does a cap-6 Stage-1 draw at f = 0 emit each pattern at all before any
RL, at what per-sample rate, over the *full* required pool — and does the structural vs
rule-sequence difference exist at the base level with intervals that separate? The current
numbers are 10 / 16 depth-3 draws (95 % 0.35–0.85) and 4 / 11 reductio draws (0.11–0.69),
measured on 300-target samples that missed some draws' reachable targets. What property of a
draw predicts whether it generalises?

## Why it matters

Clause (3) makes the pre-RL base rate the predictor of ignition, and clause (2)'s "or nothing"
applies exactly to zero-rate draws; how common they are per pattern class is the quantity
that says how often RL has nothing to elicit. Both reviews asked for this before the rate is
quoted. If the intervals still overlap at n = 24 each, the base-level class claim is dropped.
No training step is taken; the run is the cheapest way to fix n.

## Gate 0: `preregistration/round3-run3.md`

Commit before the first pod with the expectations below.

### Pre-registered expectations (numbers)

- **E1 generalisation counts (≥ 1 pattern hit in 600k samples over the full pool):** depth-3
  **12–18 of 24**; reductio **5–11 of 24**; strict derived-ORE (cap 6, in-rule) **18–24 of 24**.
- **E2 separation:** the depth-3 and reductio Clopper–Pearson 95 % intervals do not overlap
  (expected roughly 0.41–0.81 vs 0.13–0.53 at the point estimates; overlap is possible at
  these n — that is the test).
- **E3 rates:** among generalising draws, per-sample rates span ≥ 2 orders of magnitude
  (10⁻⁶ to 10⁻³) for both patterns; every reductio hit is on a 7-line `nand_neg` or
  `negimp_to_pos` target (run 5 and the ignition study saw only `nand_neg`), never on an
  8-line schema.
- **E4 predictors:** Stage-1 held-out loss does not predict generalisation (|Spearman ρ| < 0.3
  across the 24 draws of a pattern); the rate of the pattern's first half in the draw's own
  samples (depth-3: fraction of samples with two nested boxes and an `AS` at depth 2 followed
  by another `AS`; reductio: `NEGI` whose hypothesis is the negated goal, yielding `~~G`
  without a following `DN`) predicts it (ρ > 0.5).

### What would falsify the standing rule

- E2 failing (intervals overlap at n = 24 each) kills "structural patterns are generalised
  more readily by the base"; the class difference would then rest only on RL-side behaviour.
- Reductio ≥ 14 / 24: most draws generalise the sequence; clause (2)'s "or nothing" becomes
  the minority case and should be stated as such.
- Any reductio hit on an 8-line schema before RL contradicts the "7-line entry" picture from
  run 5 and the ignition study.

## Design (suggestion)

**Draws.** `train.py --mode abs --steps 6000 --bs 128 --cap 6`, seeds 30–53, on each of
`data/p2/train_depth3_f0_a1.jsonl`, `train_reductio_f0.jsonl`, `train_derived_ore_f0.jsonl`
(cap 6, the campaign's strict-pattern-free set — verify f = 0 for `derived_ore_strict` by
reclassifying the written file; if it is not 0, assemble one with `make_coverage_sets.py
assemble --patterns derived_ore_strict --freqs 0`). 72 Stage-1 models, ≈ 10–12 min each on a
3090. Keep the per-rule held-out loss from each training log (add a per-rule breakdown to
`train.py`'s eval if absent; small change, tested).

**Pools.** Depth-3: `data/r3_1/targets_depth3_req8.jsonl` from run 1 if committed, else the
300 shortest of `targets_depth3.jsonl` (the ignition study's sample; disclose). Reductio:
`targets_reductio_req.jsonl` (300). Derived-ORE, cap 6: the 300 shortest strict targets of
`targets_derived_ore.jsonl` labelled with `necessity.py --pattern ORE_DERIVED --bound 10`
(keep required ones first; 0 inconsistencies; hand-check ten).

**Sampling.** `coverage.py --k 2000 --temperature 0.8 --batch 2000 --procs 4` per draw over
all 300 targets (600k samples; per-proof hit counts and pattern labels stored), one coverage
job at a time per 24 GB card. Also score the sampled proofs for the "first half" predicates
(new predicates in `patterns2.py`, verifier-checked tests). Frozen@256 count per draw for the
side-by-side with the ignition study.

**Analysis.** Per pattern: generalising draws / 24 with Clopper–Pearson 95 %; rate
histogram; per-target hit concentration; predictor table (held-out loss, per-rule loss,
first-half rate) with Spearman ρ and a permutation p; the ignition study's 27 draws appended
as a separate stratum (300-target samples), never pooled with the new ones.

## Necessity

Pools inherit their certificates from runs 5 and 1 (see those reviews); the cap-6 derived-ORE
pool is the one new pool and needs the oracle run, 0 inconsistencies and ten hand-checked
targets before sampling.

## Controls and seeds

The 72 draws are the sample; the 27 ignition-study draws are the external comparison. No RL,
so no frozen arm is needed beyond the frozen@256 count.

## Budget and stop rule

Two RTX 3090 pods ($0.50/h): 72 Stage-1 × 12 min ≈ 14 h; 72 coverage runs × ≈ 8 min (600k
samples of a 3.2M model) ≈ 10 h; total ≈ 24 pod-hours ≈ **$12**; ceiling $50. Stop at 24
draws per pattern or at $35; if short, cut derived-ORE to 12 draws first. Kill switch
re-armed; pods deleted when pulled.

## Deliverables

`preregistration/round3-run3.md`; `run3.md` (≤ 400 words + figures: per-class
generalisation fraction with intervals; rate histograms; predictor scatter); `numbers.md`
§Round 3 run 3 with source files; `artifacts/r3_3/summary.json`, all coverage files;
`STATUS.md` `RUN3-3 DONE <UTC>`; bucket `hf://buckets/dan-pandori/nd-rl/round3-run3/{artifacts,ckpts,data}`
(all 72 checkpoints — later runs choose draws from them); `touch ~/runs/round3-run3/executor.done`.
