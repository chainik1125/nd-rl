---
written_on: 2026-09-08
written_by: agent:codex
---

# Experiment summaries

- [Synthetic cap-six proof-length frontiers](synthetic-cap6-proof-lengths.md): cross-experiment
  ledger with exact references for the eight-, eleven-, and 129-line synthetic certificates.
- [Verifier-driven proof bootstrapping and length generalization](2026-06-09-nd-bootstrapping-sprint/README.md): cap, token-wall, expert-iteration, protocol, and model-depth studies from the earlier repository.
- [Proof novelty, difficulty, and self-training selection](2026-06-10-proof-novelty-sprint/README.md): formal proof measures, an ICL capability ladder, novelty-selected training, and mechanism controls.
- [First-order natural deduction and eigenvariable conditions](2026-06-11-fol-sprint/README.md): FOL verifier, constant-token wall, and eigenvariable-freshness failure analysis.
- [R1-style scratchpad reasoning sprint](2026-09-08-r1-cot/README.md): active September 8–9, 2026.
- [OLMo-32B longer-proof pilot](2026-09-10-olmo-32b-pilot/README.md): five-method pilot found no longer-proof gain beyond ICL on the fixed 24-target panel.
- [Proof-cap comparison](2026-09-10-proof-cap-comparison/README.md): six completed configurations did not extend verified textbook reach beyond the supervision cap.
- [Proof-fragment selection](2026-09-10-proof-fragment-selection/README.md): supervised transfer scores reproduced, but the tested RL curricula reduced complete-proof coverage.
- [FLoP adaptation](2026-09-10-flop-adaptation/README.md): narrow symbolic depth extrapolation without a held-out coverage gain for the token LM.
- [Capability benchmarks for the base and RL provers](2026-09-30-benchmarking/README.md): 12 audited benchmarks (pass@k, families, glue, skills, laws, padding, prefix, EDL, likelihood, steering, interpolation, probes); RL gains concentrate on things base cannot do on unseen theorems, case splits stay at 0% for every model.
- [Literature for learning beyond initial training data](../docs/literature/astra_suggestions/learning-beyond-training-data.md): completed reading recommendations and experimental-design critique; no new experiment results.
- [GEAR versus review-gated orchestration A/B](2026-09-23-rl-hillclimb-ab/README.md): two agent frameworks hill-climbed the RL stage for three hours; GEAR reached dev 401 versus 339/324, length-10 extrapolation about two percent, nothing at length 11 or on textbook problems.
- [Run 2: seven GEAR workers and an auditor, lengths 12-14](2026-09-24-rl-hillclimb-run2/README.md): twelve hours of hill-climbing on a length-6-11 pool; the two-hour champion proves 707 of 1,931 sealed length-12-14 theorems at pass@256 (pretrained model 0), first length-14 and first textbook solves above six lines.
- [Length generalisation](2026-09-25-length-generalisation/README.md): RL on longer proofs moves the edge of competence by the trained length and no further (near zero two lines past the cap, 0% at 14-18); the model does not compose two solvable halves, and composition targets help only at short lengths.
- [Representation and goal notes for length](2026-09-25-representation-and-goals-length/README.md): relative citations, no positional encoding, line-structured positions and self-written goal notes all leave the wall one to three lines past the 11-line training maximum (0% from length 14).
- [Mid-training on long glued proofs](2026-09-26-midtraining-long-glued/README.md): supervised fine-tuning on glued proofs of 16-24 lines teaches the glue construction, not length; after RL on 6-11 the wall is unchanged (0% from length 14; natural 12-14 at 2.9% against 5.3% without mid-training, one seed).
- [rl-generalization-testbed: a mini RL run (1,000 + 1,000 pool rows from the Stage-1 base) validated against known full-scale effects before any knob is read](2026-10-16-rl-generalization-testbed/README.md): validation finished 2026-10-18, **NO-GO**, and the mini is retired as the knob testbed. At 10 + 10 iterations g64 stayed partial. At 20 + 20 (remedy A), G0 and G1 pass (the ceiling check only through a recorded B0 deviation) and kl1 reproduces (−3.0 points on 13–20 pass@1), but g64 is still partial at four seeds (−1.3 points, against −5.5 for the full-scale seed-0 pilot). The full-scale g64 replicate does not replicate (seed 2: −1.2 points), so the seed-0 E may be inflated; this is post hoc and does not change the gate. No knob has run, and Leon chooses between remedy C, re-validating the mini against a better-measured E, and an exploratory-only knob wave.
