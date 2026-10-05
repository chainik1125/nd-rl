---
written_on: 2026-10-05
written_by: agent:claude
papers:
  - zhou2025general
---

# ADeLe: general demand scales, subject characteristic curves and ability as the 50 % demand level

Paper: [@zhou2025general] (L. Zhou, L. Pacchiardi, F. Martínez-Plumed, …, X. Xie, J. Hernández-Orallo), "General
Scales Unlock AI Evaluation with Explanatory and Predictive Power"
Source: arXiv 2503.06378v2 (HTML rendering; read: abstract, Sec. 1-2, 3.1-3.3, 5.6-5.8, App. 8.2, 8.4; Sec. 3.4
assessor results and the rubric appendix skimmed)

## Learnings

- **Scales (Sec. 1-2).** The method "introduces 18 open scales in the range (0,∞)", with demand levels set by rubrics
  (DeLeAn) applied by an LLM annotator; applied "to 16,108 instances from 63 tasks from 20 benchmarks" and "15 large
  language models" (Sec. 1, 2.2; Abstract). Human inter-rater r_WG "for the 18 demands ranges between 0.70 and 0.91";
  GPT-4o vs Delphi consensus "Spearman Correlation between 0.75 and 0.94 (averaging to 0.86)" (Sec. 3.1).
- **Ability (Sec. 2.2, Fig. 3).** For each dimension, plot the subject's success rate against demand level and fit a
  logistic; ability is "the level of demand where the probability of the subject to succeed is 0.5, assuming all
  other demand levels are lower". Performance is a property of subject × task distribution, "and ability, which is an
  inferred property of a subject that is invariant to the task distribution" (Sec. 2.2).
- **IRT lineage, but non-populational (Sec. 2.2).** "strongly inspired by item response theory (IRT), and the linear
  logistic test model (LLTM) in particular", but abilities are estimated from one LLM's data only: "'non-populational'
  … does not depend on the rest of the population, only on the individual". Populational methods (IRT, PCA, FA)
  "lead to different results for AI system 'populations', whenever a new set of LLMs are added to the inferential
  pool" (Sec. 2.2).
- **Dominant slices (Sec. 3.3; 5.7).** Because instances load on several demands, the curve for dimension d at level
  l uses only instances "for which the demands in all remaining dimensions do not exceed l". A full multidimensional
  surface "would require 6^19 instances" (Sec. 5.7).
- **Curve fitting and floor (Sec. 5.7).** Two-parameter logistic per subject and dimension, with an anchor point "of 0
  at imaginary level 20" carrying 50 % of the total weight (fn. 17; Fig. 7 caption); the curve's value at level 0
  "could be considered as some kind of base reliability of the system, independent of the demands"; and "we simply
  define ability as the area under the curve", which "avoids having negative abilities, which is nonsensical in a
  ratio scale" (Sec. 5.7).
- **What post-training-style changes look like (Sec. 3.3, Figs. 7-8).** Chain-of-thought / reasoning models "maintain
  non-negligible success even at high levels of demand (in some cases above 5), while non-reasoning models generally
  plateau around level 4" on MCr and MS; reasoning models (o1, R1-Distill) "have clear improvements on the two kinds
  of QL (Quantitative and Logical Reasoning)" and on MCr, MS; knowledge dimensions track size. Ability scaling curves
  show diminishing returns between the two largest models of a family (App. 8.2).
- Caveat: predictive power may suffer "particularly if the range of demands is significantly misaligned with the
  abilities of the LLM" (Sec. 2.2).

## Evidence and limitations

- Demand levels come from LLM-applied rubrics; agreement with humans is good but there is no ground truth (Sec. 3.1).
- Per-dimension curves ignore how demands combine (no compensatory model); dominant slicing discards many instances.
- 15 subject models, single responses per instance; the paper does not study fine-tuning or RL directly — the
  reasoning-vs-non-reasoning comparison mixes training recipe, distillation and inference style.
- Not checked: assessor tables (Sec. 3.4), rubric texts (App. 9).

## Connections and questions

- **Definition offered:** ability on demand dimension d = the demand level at which the subject's per-instance success
  probability is 0.5 (equivalently the area under its subject characteristic curve), on an absolute, rubric-defined
  ratio scale; estimated per subject without reference to other models.
- **New vs better access:** not addressed. *Our interpretation:* the profile representation gives three distinguishable
  effects of RL on a checkpoint: a rise of the level-0 intercept ("base reliability … independent of the demands" —
  format, sharpening), a steeper slope, or a rightward shift of the curve on *specific* dimensions. A
  dimension-specific shift beyond the pretraining trajectory's maximum on that dimension (as reasoning models extend
  beyond level 4 on MCr/MS) is the candidate "new capability"; a uniform rise in intercepts across dimensions is
  better access.
- **Null / floor:** the anchor at level 20 forces monotone, bounded curves and AUC abilities are ≥ 0; a random
  subject has a near-zero curve everywhere and ability ≈ 0. Because success is a per-instance rate, k never enters.
- **Transfer to our setting:** our "rubrics" can be exact, mechanical features of the reference proof — no LLM
  annotator: reference length L_true, maximum box nesting depth, numbers of →I / ∨E / ¬I / classical (excluded middle,
  Peirce, byContradiction) steps, number of atoms, inference-node term size — binned to levels 0…5+. For each
  checkpoint (p0…pend, r1…r16 per seed), dominant-slice curves of per-attempt success (k = 256 counts, binomial
  weights) give a ≈ 6-dimensional ability profile; compare pend → r8 → r16 against the pretraining trajectory
  p1600 → pend. Example prediction to test: s1's late-round classical solves (13 of 16 new group-C solves classical,
  `docs/models/state_env_october_r16.md`) should appear as a shift on the classical-steps dimension only. Cost:
  CPU seconds on existing reads. Failure modes: features are strongly correlated (length with depth), so dominant
  slices are sparse at high levels; demands are properties of the *reference* proof, while the policy may prove a
  theorem by another route (the project's route-dependence trap); 322 theorems give few instances per (dimension,
  level) cell.
- Related notes: `burden2023triangulation.md` (same group: demands, 50 % convention, half-normal priors),
  `hernandezorallo2017evaluation.md` (agent characteristic curves, area = capability), `kwa2025timehorizon.md` (50 %
  horizon on one external scale), `burnell2023revealing.md` (the populational alternative they criticise).
