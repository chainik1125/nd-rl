# Sprint 2 — Proof Novelty, ICL Capability Ladder, and Novelty-Seeded RL

**Window:** 2026-06-10 22:52 PDT → 2026-06-11 08:52 PDT (10h wall-clock).
**Repo:** branch `sprint2`. Builds on sprint-1 infra (`code/nd`, `code/train`,
`code/exp`). Deliverable: `sprint2/summary.md` (+ figures, ledger, log).

## The thesis we are testing

A proof's *value as RL training data* should track its *novelty/interest*, and
novelty/interest can be measured two independent ways that ought to correlate:
1. **ICL capability ladder** — give increasingly capable open models
   (Qwen2.5 7B→14B→32B(→72B), Llama-3.1-8B, a DeepSeek distill) a 100-shot
   ND-proof-completion prompt; a theorem's *ladder height* = the capability
   needed to prove it in-context. Treat our weak 3.3M transformer + RL as an
   **uplift** measured against this ladder.
2. **Formal-logic measures** — computable proxies for "interest":
   - **lemma-value**: how many *other* targets become provable (or get shorter
     proofs) if this theorem is available as a premise/lemma;
   - **proof multiplicity**: # distinct proofs (search/enumeration);
   - **genus / cyclomatic complexity** of the proof's logical-flow graph
     (Carbone; flagged in the original proposal appendix A1) — "cleverness";
   - **description length / bits** (minimal proof length, MDL of the proof).

Then the payoff experiment: **seed RL with high- vs low-novelty proofs and
measure the change in RL dynamics** (R_k, transfer uplift, frontier).

## Workstreams

- **W0 Theory (Fable agent):** first-principles account of how verifier-RL helps
  prop logic (RL as search amortization; criticality/bounds; capacity↔depth;
  predicate-logic outlook); cleanest synthetic settings; ND-context proposals
  → feed brainstorm.
- **W1 Style-matched oracle** (mechanism confirm for HER>oracle): re-prove the
  oracle theorems in the model's own style, retrain, check if iid retention
  recovers. Fast — reuses `exp/expert_iter.py`.
- **W2 Novelty infra:** `code/novelty/` — lemma-value, multiplicity, genus,
  bits on existing ND objects. Cheap (CPU).
- **W3 ICL ladder:** `code/icl/` — vLLM on RunPod; 100-shot ND completion for
  Qwen2.5-{7,14,32}B, Llama-3.1-8B, DeepSeek distill; per-theorem ladder height.
- **W4 Correlation:** ladder height vs formal measures vs our weak-model+RL
  solvability. The validation that "novelty" is a coherent axis.
- **W5 Novelty-seeded RL:** partition seed proofs by novelty; run HER-EI from
  each; compare R_k / transfer / frontier. The payoff.

## Agent team

Theory(Fable) → Brainstorm → Rank → Red-team → Eval, plus me as lead running
the GPU experiments and infra. Monitors babysit RunPod.

## Timeline (provisional)

| h | focus |
|---|---|
| 0-1 | setup, launch theory+lit agents, start W1 + W2 infra |
| 1-3 | W2 novelty measures done; W3 ICL ladder stood up; W1 result |
| 3-5 | brainstorm/rank/red-team workflow on design; W4 correlation |
| 5-7 | W5 novelty-seeded RL chains |
| 7-9 | analysis, figures |
| 9-10 | writing, red-team, polish |

## Log
- **22:52** Sprint 2 start. Branch `sprint2`. Launching theory (Fable) + lit agents.
- **22:58** Setup answered: budget ~$100; ICL ladder self-host OK but USER ALSO
  gave API keys (OPENAI/ANTHROPIC/GROK _MATS) — all three auth-verified. ICL
  ladder will be primarily API-based (GPT/Claude/Grok mini→large), far cheaper
  than self-hosting; maybe one self-hosted Qwen rung for open-weights. Base pod
  nd-s2-base provisioning (CA-MTL-1). nd_pack.pdf (123pp) extraction agent running.
- **23:12** Theory(Fable) done (sprint2/theory.md): amortization model
  (β≈0.07-0.2/line predicts token wall + format>RL); criticality R_k needs full
  consolidation (HER raised value not R); nesting d*≈layers-1, L_min≈3d+1
  (depth-4↔13 lines); methodological fix ADOPTED: R_k on fixed pool decays by
  depletion. Predicts: lemma-value = best seeding signal; bits↔ladder height;
  genus inverted-U; W5 novelty×depth ranking inverts at capacity boundary.
- **23:20** W2 prover: 36/36 textbook, sound (verifier-checked). Measures:
  genus(Betti), MDL(gzip), lemma-value. KEY: on 3000 RL targets,
  lemma_value ⊥ min_lines (Spearman 0.008) but genus~min_lines (0.798) — so
  lemma-value is genuine novelty independent of difficulty; genus is difficulty
  in disguise. Built difficulty-MATCHED high/low lemma-value partitions
  (983 each, min_lines 8.22 both, lemma 15.78 vs 0.00).
- **23:28** W5 launched: HER-EI on high-novelty vs low-novelty targets (2 pods).
  Early: high harvests 1441 prefixes vs low 897 (richer structure) at similar
  discoveries (388 vs 410). W3 ICL ladder (API, 6 models x 135 x 2) finishing.
- **22:58** Lit agent done (sprint2/literature.md): top-3 novelty tools =
  lemma-reuse centrality on proof DAG, logical-flow-graph genus/Betti,
  cut-elim/gzip MDL. KEY WARNING (adopt): control pass-rate when varying
  novelty or novelty⊥difficulty confound. Our verifier-filtered>external result
  echoes Yi et al. 2025 but is stronger (sound verifier sidesteps their
  verifier-quality ceiling). Two gaps we fill: no clean 100-shot ND-completion
  curve exists; no paper isolates structural-novelty-selected RL seed.
