# Run brief: round3-run2 — the 8-line reductio stratum and the derived-ORE base-rate dependence

Run id `round3-run2`. Role: executor. Fork `~/nd-takehome`, branch `dan_round3-run2` from
`origin/dan_novelty` (or its merged successor). `AGENT_POLICY.md` governs; designs are
suggestions. Read first: `run5.md`, `review_round2-run5.md` (§Verdict, "Next measurement"),
`followup.md` §Block B, and §2 of `~/nd-rl/docs/proposals/2026-09-18-proposals-round-3.md`.
This is the week-38 digest's open question 1 and 2, given a brief.

## Question

On the required reductio pool, every igniting arm in run 5 solved the 7-line stratum
(51–53 of 52) and ≤ 1 of the 248 targets at 8–10 lines in 8 rounds; the same checkpoints had
acquired 25–31 eight-line targets on the 606 pool. What is the per-run probability that the
8-line stratum ignites, and is it a matter of rounds, of samples per round, or of f? Does a
6-line stratum (inside the pretraining length) change anything for zero-rate draws? And for
strict derived-ORE (cap 8, non-submission): is EI's acquisition at f = 0 proportional to the
draw's base reachability, and is the f-dependence monotone over three points?

## Why it matters

Clause (2) reads "amplifies what the base already generalised to". The base emits reductio
only on `nand_neg` targets (every pre-RL hit in 14 draws); RL carried it to `negimp_to_pos`
(7 lines) in run 5 and to `neg_both` / `chain_neg` (8 lines) on the 606 pool. If the 8-line
stratum ignites with a measurable probability, the sequence generalises across schemata and
lengths after entry — RL producing proofs of schemata the base never emits, which the word
"amplification" hides. If it never ignites in 6 arms × 16 rounds, the elicitation reading
tightens to "the base's easiest schema and its 7-line sibling".

## Gate 0: `preregistration/round3-run2.md`

Commit before the first pod: question, design as run, the expectations below, budget, stop
rule.

### Pre-registered expectations (numbers)

- **E1 (16 rounds, k = 32).** The 8-line stratum (133 targets: `neg_both`, `chain_neg`,
  `contraposition_conv`, … per `review_round2-run5.md`) ignites (≥ 10 targets with a strict
  reductio proof) in **3–5 of the 6** arms (3 models × 2 sampling seeds); first 8-line proof
  between rounds 6 and 12; the 9- and 10-line strata (82 + 33) stay ≤ 3 targets in every arm.
  f = 0.1 arms ignite the 8-line stratum no more often than the f = 0 s0 arm (f does not
  matter after entry).
- **E2 (k = 64, 8 rounds; same attempts, half the rounds).** The 8-line stratum ignites in
  ≤ 1 of 3 arms: rounds (training steps) matter more than attempts.
- **E3 (6-line stratum).** Zero-rate draws s1 and s2 have **0 hits** on the 45 six-line targets
  at pass@10⁴ and stay at 0 / 345 after 8 rounds; the f = 0 s0 draw solves ≥ 30 of the 45
  six-liners by round 2.
- **E4 (derived-ORE, cap 8).** f = 10⁻³ acquisition in **0.08–0.16** for both seeds (between
  the f = 0 0.080–0.087 and the f = 10⁻² 0.160–0.210). The fresh f = 0 draw with base strict
  reachability r ≠ 27 / 300: EI acquisition within **0.5–1.5× r**, EI ≈ 2× frozen, EI-only
  targets (not base-reachable at 10⁴) 5–10.

### What would falsify the standing rule

- E1 at **0 of 6** with E2 also 0: a real length wall for elicited rule sequences — clause
  (2) tightens and "spreads across lengths" is dropped.
- E3 with any six-line hit in a zero-rate draw followed by ignition: the wall was
  length-gated, not pattern-gated; clause (2) must say "base rate at the shortest instance".
- E4 outside 0.5–2× of base reachability: "amplification in proportion to the base rate"
  is dropped; EI-only ≥ 20 targets would mean RL adds substantially beyond the base's reach
  even for this pattern.

## Design (suggestion)

**Reductio pool.** `data/p2/targets_reductio_req.jsonl` (300; strata 7 / 8 / 9 / 10: 52 /
133 / 82 / 33) plus a **6-line stratum**: the 45 candidates in `data/p2/run5_reductio_nec.jsonl`
with `min_lines_ub` 6 (all 945 fail `--forbid DN`; G4ip-check them with
`review_run5_logic.py`; class-disjoint from the training sets; hand-check five). Pool file
`data/r3_2/targets_reductio_req6.jsonl` (345) with a `stratum` field; transfer
`transfer_reductio_req.jsonl` unchanged.

**Reductio arms** (T = 0.8, retain 20k, min-round rule), Stage-1 checkpoints that exist:
`stage1_reductio_f0_s0.pt`, `stage1_reductio_f0.1_s0.pt`, `stage1_reductio_f0.1_s1.pt`:
- A: 3 models × sampling seeds {0, 1} × **16 rounds at k = 32** (6 arms).
- B: 3 models × sampling seed 0 × **8 rounds at k = 64** (3 arms).
- C: zero-rate draws `stage1_reductio_f0_s1.pt`, `s2.pt` × 8 rounds at k = 32 on the 345
  pool (2 arms) plus `coverage.py --k 10000` on the 45 six-line targets for each.
- Frozen controls: for A and B, the round-1 sample at equal attempts is the run-5 frozen arm
  (same init, same seeds) — re-use `artifacts/r5/frozen_*`; for sampling seed 1 run one
  frozen arm per model at 16 × 32 attempts (`--no_train`). Report acquisition per stratum and
  per schema per round; first-proof round per stratum; transfer per stratum.

**Derived-ORE (cap 8; label every number "cap 8, not a submission result").** Pool
`data/p2/targets_derived_ore_req.jsonl` (300) and transfer (65). Arms: f = 10⁻³ s0
(`stage1_derived_ore_strict_f0.001_c8_s0.pt` exists) and s1 (train on
`train_derived_ore_strict_f0.001_c8.jsonl`, seed 1); two fresh f = 0 cap-8 Stage-1 draws
(seeds 2, 3 on `train_derived_ore_strict_f0_c8.jsonl`) with `coverage.py --k 10000` on the
300 targets; keep both if their strict reachability differs from 27 / 300 by ≥ 5 targets,
else train seeds 4, 5 (disclose). EI k = 32 × 8 rounds + frozen for each. Report acquisition,
frozen, base reachability, EI-only count.

## Necessity

The reductio pool is run 5's, reviewer-certified (classically valid, G4ip-unprovable, `DN`
required unboundedly, 0 pattern-free solves in 18 arms). The 6-line stratum must pass the
same two checks before the arms. The derived-ORE pool is run 5's (`ORE_DERIVED` restriction
fails at bound 10, 0 pattern-free solves).

## Controls and seeds

Frozen at equal attempts for every arm; two sampling seeds in A; two seeds per f in
derived-ORE. Differences quoted only with ≥ 2 seeds.

## Budget and stop rule

Two RTX 3090 pods ($0.50/h). Reductio: 9 long arms × ≈ 0.5 h + 2 short + 2 coverage runs
≈ 7 h; derived-ORE: 3 Stage-1 × 20 min (cap 8) + 2 coverage × 1.5 h + 8 arms × 0.4 h ≈ 7 h;
total ≈ 14 pod-hours ≈ **$7–8**; ceiling $50. Stop at 16 rounds for A, or at $30. Coverage
runs never co-scheduled on a 24 GB card. Kill switch re-armed; pods deleted when pulled.

## Deliverables

`preregistration/round3-run2.md`; `run2.md` (≤ 400 words + figures: per-stratum cumulative
curves for the 6 sixteen-round arms; the 6 × 16 ignition table; derived-ORE acquisition vs
base reachability with f as the marker); `numbers.md` §Round 3 run 2 with source files;
`artifacts/r3_2/summary.json`; `STATUS.md` `RUN3-2 DONE <UTC>`; bucket
`hf://buckets/dan-pandori/nd-rl/round3-run2/{artifacts,ckpts,data}`;
`touch ~/runs/round3-run2/executor.done`.
