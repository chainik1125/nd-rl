# Sprint 2 — numbers ledger (provenance for every figure/claim)

## Infrastructure
- Bounded sound ND prover (`code/novelty/prover.py`): proves 36/36 textbook
  targets, every proof verified by the independent checker. Goal-directed +
  elimination closure + classical reductio + backward modus-ponens + case
  analysis. min_proof via iterative deepening.
- Measures (`code/novelty/measures.py`): genus = first Betti number of the
  citation+discharge flow graph; mdl_bits = gzip length of proof tokens;
  lemma_value = #targets unlocked + 0.5·#shortened at shallow budget.
- nd_pack.pdf (Carr 2018, 123pp): 68 validated propositional sequents w/
  solutions extracted (`data/targets_ndpack.jsonl`); 76 first-order skipped.

## W2/W4 — novelty axes (n=135 W4 set: 36 textbook + 49 ndpack + 50 synthetic)
- On 3000 RL targets: Spearman(lemma_value, min_lines) = **0.008**;
  Spearman(genus, min_lines) = **0.798**; Spearman(lemma_value, genus) = -0.119.
  => lemma_value is interest INDEPENDENT of difficulty; genus is difficulty.
- ICL ladder (6 frontier models, 40-shot, 2 samples each, verifier-judged):
  empirical solve rate grok-3-mini .79 > grok-3 .74 > claude-sonnet .70 >
  claude-haiku .56 >> gpt-4o .19 > gpt-4o-mini .07. (GPT follows the strict
  token format far worse — a format-compliance confound on the ladder.)
- Spearman vs ICL ladder height (n=132): mdl_bits **+0.52**, min_lines **+0.50**,
  genus +0.39, lemma_value **-0.04**, gen_lines +0.64 (n=50).
- mean by ladder height: min_lines 3.7/5.1/7.6/9.8/11.0 (heights 0-4) — rises;
  lemma_value 18.8/15.6/13.5/11.4/17.0/19.3/9.1 — no trend (orthogonal).
- FIG: sprint2/figures/fig_w4_axes.png.
- RL-UPLIFT ANCHOR (weak 3.3M model on the SAME 135 W4 targets): SFT base
  greedy 0.363 / pass@8 0.444; +1 HER round pass@8 0.459. Places the tiny
  specialized model ABOVE gpt-4o (0.19) and gpt-4o-mini (0.07), near
  claude-haiku (0.56) on the ladder. Caveat: frontier got 40-shot/2-sample
  (~pass@2) and GPT is format-penalized; not apples-to-apples, but the
  order-of-magnitude "specialization ~ mid-frontier" point holds.

## W5 — novelty-seeded RL (HER-EI, difficulty-matched high vs low lemma-value
## targets, 983 each, min_lines 8.22 both, lemma 15.78 vs 0.00; base L8off)
| round | HIGH disc / harvest | LOW disc / harvest | HIGH tr p1/p8 | LOW tr p1/p8 |
|---|---|---|---|---|
| 0 | — | — | 0.198 / 0.302 | (same base) |
| 1 | 410 / 1441 | 420 / 897 | 0.260 / 0.384 | 0.230 / 0.374 |
| 2 | 522 / 1880 | 503 / 1122 | 0.226 / 0.374 | 0.274 / 0.410 |
| 3 | 560 / 2082 | 560 / 1304 | 0.294 / 0.432 | 0.288 / 0.394 |
| 4 | 623 / 2295 | 595 / 1392 | 0.272 / 0.436 | 0.226 / 0.382 |
- W5 transfer eval n=500 (single-rate SE~2.2pp, difference SE~3pp).
- VERDICT (r4 complete): transfer pass@8 high .384/.374/.432/.436, low
  .374/.410/.394/.382. NON-MONOTONE: low LEADS at r2 (.410 vs .374); high ends
  +5.4pp (~1.8sigma, single seed), trending up while low trends down. So: a
  weak, late, single-seed edge for high-novelty — suggestive, not established.
  The robust difference is HARVEST DENSITY: high yields ~60% more harvestable
  prefix-proofs (2295 vs 1392) at near-equal discoveries (623 vs 595) — richer
  structure — but extra signal gives at most a marginal transfer gain.
  Consistent with Sprint-1 knowledge>>capability-transfer. Needs more seeds.

## W1 — style isolation (same 248 theorems, model-style 9.1 vs generator-style
## 10.3 mean lines; add each to base L8off, retrain depth-4)
| arm | iid | transfer pass@8 |
|---|---|---|
| base (no seed) | 0.859 | 0.286 |
| + model-style proofs | 0.825 (-3.4pp) | 0.284 |
| + generator-style proofs (same thms) | 0.836 (-2.3pp) | 0.364 (+7.8pp) |
- HEADLINE (refutes the style hypothesis): at matched length/theorems,
  generator-authored proofs do NOT collapse iid more than the model's own —
  they collapse LESS and HELP transfer. So Sprint-1's oracle iid-collapse
  (.832->.696) was driven by the extreme LENGTH (17-30 lines) of oracle
  proofs, NOT by their authorship/style. Corrects the "on-manifold style"
  explanation. (Single seed; n=500 transfer SE~2pp so +7.8pp ~3-4 SE.)
  Caveat: cannot test style at 17-30 line length (no model proofs that long).

## Capacity falsifier — nesting vs length (local-only, nesting-0 proofs;
## does depth-4 run past the 12-line ceiling on flat proofs?)
- depth-4 local-only base: iid 0.853, transfer 0.082/0.18 (textbook 3/36 —
  can't do discharging-rule problems, expected).
- max discovered length by round: r1 11, r2 11, r3 12, r4 12 (final buffer 997).
  VERDICT: flat (nesting-0) proofs cap at 12 at depth-4 — IDENTICAL to nested
  depth-4 (sprint-1: 12). The nesting-ceiling hypothesis (theory P3) is
  REFUTED: the ~12-line ceiling is a general length/capacity limit, not
  subproof-stack tracking. Refines sprint-1's "depth=nesting-tracking" reading.
  Caveats: local-only trained on 200k (vs 450k), 4 rounds, reducibility
  (gen_lines != min_lines). But flat==nested ceiling is a clean dissociation
  failure. (Depth-8 reached 13 on nested in sprint-1 → depth lifts a GENERAL
  length ceiling, modestly, not a nesting one.)

## Strengthening runs (post red-team, single-seed -> firmer)
- W5 SEED-1 (shared base, the clean replication): high pass@8
  r0..r4 .322/.430/.390/.406/.400; low .286/.332/.328/.362/.354. High is ABOVE
  low at EVERY round (gap +4-10pp), ends +4.6pp; harvest 2310 vs 1347 (+71%).
  Seed-0 ended high +5.4pp. => TWO-SEED replication: high-lemma-value targeting
  gives a consistent ~+5pp transfer edge + ~60-70% harvest advantage. Not null.
- NESTING data-matched (local-only 450k): max length 11/12/12/12 (r1-4), 956
  discoveries, zero >12 — IDENTICAL to undertrained local-only AND to nested
  depth-4. Finding 5 FIRM: depth ceiling is not nesting-specific, not a
  data-budget artifact.

## Compute
- ~9 RunPod A40 pods across the sprint (base, w5 high/low ×2 seeds, aux/nesting
  ×2, w1, anchor) — all terminated. + frontier-API calls for the ICL ladder.
  Shared-account balance delta this sprint ~$24 (includes other users); my
  pod-hours ≈ $15-20. Well within the $100 budget.
