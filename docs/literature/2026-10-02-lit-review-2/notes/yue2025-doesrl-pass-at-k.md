---
written_on: 2026-10-02
written_by: agent:claude
papers:
  - yueDoesReinforcementLearning2025
---

# Yue et al.: large-k pass@k and base-model perplexity as the "elicitation" standard

Paper: [@yueDoesReinforcementLearning2025]
Source: https://arxiv.org/abs/2504.13837v5 (v5 reviewed; html text)

## Learnings
Paper claims:
- At small k RLVR models beat their bases, but "base models consistently surpass RLVR models across all benchmarks and LLM families as $k$ increases" (Sec. 1). k goes to 128, 256 or 1024 (App., "typically 128, 256, or 1024").
- Training dynamics: "pass@1 improves, but the coverage of solvable problems (i.e., pass@256) decreases" (Fig. 1 right; Table 4).
- Per-problem four-way table (Table 2, AIME24 k=1024 / MATH500 k=128): base-unsolved but RL-solved is 0.0% / 1.0%; base-solved, RL-unsolved is 13.3% / 3.6%. This is the same partition as our A/B/C.
- Support test: $\text{PPL}_{\text{Base}}(\mathbf{Y}_{\text{RL}}|x)$ "closely matches the lower portion" of the base's own-sample perplexity (Fig. 6, Sec. 4.1). This rests on two AIME24 problems with 16 responses each.
- Distillation from a stronger teacher does expand the boundary (Sec. 4.2, Fig. 7).
- Answer-guessing caveat: for math they hand-check CoTs on a subset (Sec. 3.1).

Our interpretation: the standard is (i) equal-k pass@k for base and RL at large k, done per problem, plus (ii) the likelihood of RL outputs under the base. Both are tests of support at a finite budget. Neither can tell "already in support at low mass" apart from "outside support" unless k is large compared with 1/p.

## Evidence and limitations
- Read Sec. 1, 3.1, 4.1 (Table 2, Fig. 6), 4.2, 4.4, 7. Did not check the per-benchmark appendix tables.
- The perplexity evidence is anecdotal: two problems, and sequence-level PPL. It is not a per-problem, per-step test.
- All base models were pretrained at web scale, so the pretraining content is unknown. Contamination (2507.10532) and spurious rewards (2506.10947) show this matters.
- The limitations paragraph puts the failure down to exploration ("lack of effective exploration strategies", Sec. 7), not to a theorem. The claim is about current RLVR recipes.

## Connections and questions
- Our `trajectory` A/B/C split is Yue's Table 2, but with asymmetric budgets: the base is sampled at k 256, while the RL model's "eventual" solve draws on 8 rounds × 32 samples plus its own evaluation. Yue compares both models at the same k. Cheapest fix: re-sample the end-of-pretraining base on the B set only at k = 2,048–4,096 (B is small), and report pass@k curves for base and RL on B at equal k.
- Our measurement (a), the teacher-forced log p of the RL proof under every checkpoint, is a per-problem, per-checkpoint version of Fig. 6 and is stronger than Yue's. Report it at sequence level as well as the worst step (see chen2025-coverage-principle.md).
- Lean is a step-level verifier, so our pass@k is already a "CoT-pass@k" (2506.14245). The lucky-guess criticism (2510.08325) does not apply to us. This is worth stating.
- Earlier notes: zhao2025-echo-chamber.md and zhang2025-interplay-pre-mid-rl.md (controlled pretraining) are the stronger designs.
