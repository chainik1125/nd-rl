---
written_on: 2026-10-05
written_by: agent:claude
papers:
  - hilton2023singleagent
---

# Scaling laws for single-agent RL — intrinsic performance

Paper: [@hilton2023singleagent]
Source: arXiv 2301.13442v2 (HTML rendering read: abstract, Sec. 1-2, 4.5, 5, App. A)

## Learnings

- **Problem: RL returns are not smooth.** "the main performance objective of interest, mean episode return,
  need not vary smoothly" (Abstract); e.g. StarPilot: "it takes some ability with aiming and dodging to reach
  a mean episode return of 5 or 10, but not much additional skill to reach a mean episode return of 15 or 20"
  (Sec. 2.1).
- **Definition: intrinsic performance.** "a monotonic function of the return defined as the minimum compute
  required to achieve the given return across a family of models of different sizes" (Abstract). Formally,
  given a scalable model family, "the intrinsic performance of an arbitrary policy is the minimum compute
  required to train a model of any size in the family to reach the same return (averaged over random seeds)"
  (Sec. 2.1, Definition). Consequently "the efficient frontier is mapped onto the line y=x by definition"
  (Sec. 2.1, Fig. 1b).
- **Scaling law in those units.** I^(−β) = (N_c/N)^(α_N) + (E_c/E)^(α_E) (Sec. 2.2, Eq. 1), with β
  determined by α_N and α_E (Lemma 1).
- **Natural performance metrics.** Metrics that themselves follow such laws are called "natural performance
  metrics" (Sec. 1). For CoinRun (binary success), "We found the fail-to-success ratio F := (10 − R)/R" to be
  one; "the logarithm of the fail-to-success ratio can also be thought of as the logit function (inverse
  sigmoid) of the failure rate" (Sec. 4.5); F ∝ I^(−0.40) (easy) and I^(−0.48) (hard) (Sec. 4.5). For Dota 2,
  exponentiated scaled TrueSkill works, with a conjectured "analog of an irreducible loss" (Sec. 4.5).
- **Limit of the construction.** "without a natural performance metric, we cannot extrapolate to unseen
  performance levels" (Sec. 5.1).
- **Method.** Jointly fit the power-law constants and a monotonic function f(R) = I by isotonic regression
  inside a black-box optimiser ("jointly fitting the power law constants and a monotonic function", App. A);
  exclude early training ("Typically at least the first 1/64 of training should be excluded", App. A).
- **Pretraining caveat (on sample-efficiency comparisons with humans).** "we should expect this factor to be
  smaller for models that have been pre-trained to learn useful representations" (Sec. 5.1).
- **Their own confidence.** "we do not think conclusions that depend on the precise fitted values of our
  scaling constants can be drawn with confidence" (Sec. 5.3).

## Evidence and limitations

- Evidence: Fig. 1-2, 9-11; environments Procgen, Dota 2, MNIST toy. All models trained from scratch; no
  pretraining-then-RL setting.
- Intrinsic performance needs a family of models of several sizes trained over a range of compute; it is
  defined only up to the best performance the family reached (Sec. 5.1).
- Not checked: Sec. 3-4.4 details, App. B-G.

## Connections and questions

- **Definition offered:** capability of a policy = the minimum training compute that the reference model
  family needs to reach the same performance ("intrinsic performance", in FLOPs); plus "natural performance
  metrics" (e.g. log fail-to-success ratio = logit of failure rate for binary success) that are smooth in
  compute.
- **New vs better access:** not discussed. Intrinsic performance turns any performance gain into a compute
  number relative to a reference family, which is what a rule needs: an RL checkpoint whose performance has an
  intrinsic performance (under the pretraining family) far beyond the compute actually spent (pretraining +
  RL), or beyond the family's reachable range (undefined, as in Sec. 5.1), is doing something the base
  family's own scaling does not supply. Davidson et al.'s CEG is the ratio of two intrinsic performances
  (davidson2023retraining.md).
- **Null / floor:** none needed: compute = 0 corresponds to the untrained (random-init) model, which anchors
  the scale.
- **Transfer to our setting:** define the reference family as our pretraining runs (checkpoints along
  pretraining and, if available, several sizes); compute intrinsic performance of pend, r8, r16 from pass@1
  (or better, the per-theorem logit / log fail-to-success ratio, which the paper found natural for binary
  success). Report r16's intrinsic performance relative to pend's in FLOPs and compare with RL's FLOPs.
  Also test whether logit(pass@1) is a natural metric on our pretraining curve (a straight line in log
  compute). Cost: evaluation on existing checkpoints; a size ladder would add training cost. Failure modes:
  needs several model sizes or a credible compute-optimal frontier; our RL models start from a pretrained
  model, so "minimum compute in the family" mixes pretraining and RL compute; extrapolation beyond the
  family's best performance is impossible without a natural metric (Sec. 5.1).
- Related: davidson2023retraining.md, jones2021boardgames.md, schaeffer2023mirage.md (smooth metrics).
