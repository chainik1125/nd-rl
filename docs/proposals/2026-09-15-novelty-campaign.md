---
written_on: 2026-09-15
written_by: agent:claude
---

# Does RL create capabilities or elicit rare ones? A controlled test in natural deduction

## Summary

Two independent runs have now found the same thing: RL against the verifier extends proof
length *inside* the generator's family of theorem shapes, and transfers almost nothing to
textbook theorems. The [R1-CoT sprint](../../experiment-summaries/2026-09-08-r1-cot/README.md)
(8M/85M inherited parents, GRPO-style updates, textbook targets) found no window growth
beyond six reference lines (0/43 development theorems), and diagnosed missing sampling
support: 44 of 2,048 policy groups had any reward variance. The
[take-home run](https://github.com/dan-pandori/nd-takehome) (3.2M model from scratch,
expert iteration, generator-made targets; 2026-09-15) moved the robust frontier from 8 to 9
written lines on 1,638 transfer theorems against a frozen-resampling control with equal
attempts, two seeds agreeing to 0.5 pp — but validation-36's `>6` bin went 0/24 → 1–2/24 and
the test-long file 9.8% → 14.5%. Its diagnosis: the generator's distribution, not the
length cap, binds textbook transfer — the unsolved theorems need an `ORE` over a *derived*
disjunction or an excluded-middle detour, shapes the sampler never produces.

Neither run can say whether RL merely amplified behaviour the pretrained model already
produced rarely, or composed something absent. This proposal is a campaign to answer that
directly, now running on branch `dan_novelty` of the take-home fork (started 2026-09-15,
budget $145 of RunPod credit, hard stop 2026-09-18 16:00 UTC).

## How the two runs compare

| | R1-CoT sprint | take-home run |
|---|---|---|
| question | learned scratchpad vs direct-proof RL | any RL vs resampling, past cap 6 |
| model | 8M/85M, inherited, prior exposure to long banks | 3.2M from scratch, cap-6 data only |
| line references | 40-token relative-unary codec | absolute indices with random start offset |
| RL | GRPO/LOO, groups of 8, 256 updates | expert iteration, k = 32, 16 rounds |
| targets | textbook (58 dev / 73 sealed) | generator-made 7–16-line pools; validation-36, test files |
| controls | matched tokens, blind hashed splits, sealed final set, independent audit | frozen model at equal attempts, two seeds, pruned-length reporting |
| result | no window growth; CoT ⊂ direct at 2× tokens | frontier 9 vs 8 (synthetic); ~0 textbook transfer |

The sprint's protocol is the stronger one and should be the template. The take-home used its
transfer set for two selection decisions, has 45% renaming overlap between the test-short
file and its training set, and shipped a proof-counting bug (start-index variants counted
as distinct proofs, ~13× inflation) that a review caught afterwards. Its solve-rate results
and frontier survive the correction.

One new data point for the codec question: with plain absolute indices trained only on
`N1..N6` the take-home model wrote zero seventh lines in 26k samples; absolute indices with
a random training offset beat `B<k>` relative references (108 vs 79 seven-line proofs at
pass@16, 44.7% vs 40.5%). The argument — copying an in-context index is length-agnostic,
counting a distance is not — may or may not apply to a unary relative codec. Cheap to test.

## The campaign

**Phase 1 — measure novelty on the existing run (no training).** For every proof the
RL model found, compute the *base* model's teacher-forced log-probability, marginalised over
start indices. A proof at log p ≈ −8 was reachable by resampling; one at −30 was not at any
budget. Validate against base-model pass@10⁴ on the transfer set and pass@10⁵ on
validation-36 (the sampler does ~1,200 proofs/s, so this is hours). Locate where the
improbability sits in each proof (per-token surprisal; rule at the maximum). Output: the
distribution of RL-found proofs against sampling-reachability lines, and the list of theorems
RL solved that the base model never did in 10⁴–10⁵ samples, with their log-probs. This is
Yue et al.'s question with an exact verifier and a model small enough to sample exhaustively.

**Phase 2 — controlled-coverage pretraining (the core).** Three proof patterns the
textbook set needs and the generator rarely makes: `ORE` over a derived disjunction,
classical reductio (`NEGI` then `DN`), box depth ≥ 3. Build pretraining sets of fixed size
with each pattern's frequency set exactly to f ∈ {0, 10⁻⁴, 10⁻³, 10⁻², 10⁻¹} by subsampling a
labelled raw pool (f = 0 means zero occurrences, verified by re-classifying the file). Train
identical Stage-1 models, run identical expert iteration against targets that need the
pattern, with frozen controls; two seeds at the extremes. Metric: *acquisition* — the fraction
of targets solved with a proof that contains the pattern (classifying the model's proof, not
the generating proof) — plotted against f. If acquisition is ≈ 0 at f = 0 and rises with f,
RL amplifies rare-but-present behaviour and f = 0 is a wall at this budget. If it is > 0 at
f = 0 while the base model's pass@10⁴ is ≈ 0, RL composed an absent pattern, and the
campaign reports exactly how many proofs, on how many theorems, at what base log-prob.

**Phase 3 — textbook-shaped curriculum (if time).** A target pool of classical schemata with
fresh atoms, disjoint from validation-36 by renaming class; expert iteration with and without
a generator-made ≤ 6-line "precursor" supervised set (the sprint's
[support strategy](../../experiment-summaries/2026-09-08-r1-cot/support_strategy.md)).
Validation-36 by bin per round, with base reachability for every newly solved theorem.

Also: a bounded minimal-proof-length prover for honest difficulty labels (evaluation only), and
a rule that every distinct-proof count is computed after renumbering from `N1`.

## Evidence standard

Every "RL solved X" claim carries a base-model reachability number. Two seeds before any
comparison is called a difference. Negative results are written up as such. No test-file
runs in this campaign. A review pass on the numbers before they are repeated anywhere.

## Next steps

Results land on `dan_novelty` of the take-home fork as `phase1.md`, `phase2.md`,
`campaign.md`; a summary will be added to `experiment-summaries/` here when the campaign
ends. The Phase 1 reachability table for the take-home's frontier gain is expected first.
