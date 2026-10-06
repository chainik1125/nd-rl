# Independent review of the campaign claims (2026-09-16)

Reviewer: a second Claude session, at the human's request, after the run had finished. Method:
re-derive the load-bearing numbers from the raw files with independent code (my own depth
counter on the *written* proof, not `patterns.py`; `nd_verify` re-run on every counted proof),
and check the target/training disjointness by renaming class.

## Claims that hold

| claim | independent check | result |
|---|---|---|
| depth-3 f = 0 pretraining set contains zero depth-3 proofs | max box depth of every **written** proof in `data/p2/train_depth3_f0.jsonl` (155,000) | **0** with a third box; written and pruned depth agree on a 2,000-proof sample |
| same for reductio and derived-ORE f = 0 sets (written form) | independent predicates on written proofs | 0 and 0 |
| EI at depth-3 f = 0 produces depth-3 proofs: 341 / 271 theorems, 400 / 327 proofs (s0 / s1) | recount on `found_8.jsonl`, start-index normalised, written-form depth ≥ 3, every proof re-verified against its prompt | **341 / 271 theorems, 400 / 327 proofs**, 0 verifier failures |
| frozen f = 0 control produces none | same recount on `frozen_depth3_f0_s0/found_8.jsonl` | 0 depth-3 proofs (58 distinct proofs, 55 theorems solved) |
| base f = 0 model, 10⁴ samples × 300 depth-3 targets: no depth-3 proof | `cov_depth3_f0_s0_targets.s0.jsonl` | 3 targets solved, all with 7-line depth ≤ 2 proofs; see caveat 3 |
| depth-3 targets are disjoint from the f = 0 training set and have no ≤ 6-line proof | renaming-class intersection; `min_lines_ub` field | 0 overlap; 0 targets with `min_lines_ub < 7` |
| Phase 1: 348 of 1,411 RL-solved transfer theorems unsolved by the base at 10⁴ | recomputed from the three coverage shards and `found_transfer_16.jsonl` | 1,638 theorems covered, 1,080 solved at 10⁴, **348** RL-only |
| Phase 3 counts 161 / 213 / 222 of 623 | `round_8.json` of the three arms | match |

## Corrections and caveats

1. **"Bistable" is the wrong word.** The reductio f = 0 arms solved 609 vs 170 targets from identical
   Stage-1 loss; that is two draws, and two draws cannot distinguish two equilibria from one
   high-variance process. Read it as *instability of expert iteration on hard targets*, i.e. a
   reason to run more seeds before stating any f = 0 number to better than a factor of ~3. The
   depth-3 f = 0 result is less exposed (341 vs 271 theorems, both far from 0 and far from the
   frozen control's 0), but the same caution applies to every per-arm number in `phase2.md`.
2. **The reductio dial measured nothing about reductio's necessity — and the reason is sharper
   than "94% are intuitionistic".** The 56 targets that are *not* intuitionistically provable
   were solved by the f = 0 arms (40 / 56 and 27 / 56) with zero reductio-shaped proofs: every
   one of their 1,693 proofs applies `DN` directly to a `( ~ ( ~ X ) )` premise or hypothesis
   (1,597 cite a `PR` line, 164 an `AS` line). Classical-only here means "a double negation is
   given", not "a contradiction must be derived". So P2 is uninformative at every f, not because
   RL failed but because the targets never required the pattern. A reductio dial needs targets
   whose double negation must be *derived*; none of the campaign's pools do that.
3. **The base pass@10⁴ files store only distinct proofs.** "0 depth-3 proofs in 3×10⁶ samples"
   rests on the sampler's `n_distinct_ok` bookkeeping being complete (e.g. one target had 1,110
   hits and one stored proof). Plausible — the written-length histogram is per distinct proof —
   but it was not re-derivable from the file, so it is one step weaker than the other checks.
4. **Small inconsistency in the log.** `log.md` gives the first depth-3 proof at round 2 (s0) /
   round 4 (s1) in one entry and round 3 / 5 in another; the `found_8.jsonl` round fields say
   **3 / 5**. `phase2.md` should use 3 / 5. The substantive point (the first depth-3 proofs came
   from models fine-tuned only on depth ≤ 2 successes) holds either way.
5. **Two seeds only, everywhere.** Every f = 0 conclusion has n = 2 training seeds; the
   intermediate-f rows have one. The shapes of the three acquisition-vs-f curves (flat-high,
   flat-zero, rising) are clear at this n; the exact levels are not.

## Verdict

The headline — expert iteration with **zero** depth-3 proofs in pretraining produces depth-3
proofs at the same rate as with 10% coverage, while the frozen control and the base model
produce none — is supported by the raw files under independent recomputation. The reductio
"wall" should be restated as "not tested" (caveat 2). The derived-ORE result (amplification of
a degenerate template, base p ≈ 10⁻³) stands as written.
