---
written_on: 2026-10-05
written_by: agent:claude
papers:
  - davidson2023retraining
---

# AI capabilities can be significantly improved without expensive retraining — compute-equivalent gain

Paper: [@davidson2023retraining]
Source: arXiv 2312.07413v1 (HTML rendering read: abstract, Sec. 1-2, Sec. 4, parts of Sec. 3 and 5,
App. A)

## Learnings

- **Definition: compute-equivalent gain (CEG).** "the compute-equivalent gain: how much additional
  training compute would be needed to improve performance by the same amount as the enhancement"
  (Abstract). Procedure (Sec. 2): measure performance p of a model trained with compute C ("The model should
  be trained compute-optimally, so that C is the minimal compute required to attain performance p"); measure
  p* with the enhancement; estimate the compute C′ a non-enhanced model needs to reach p*; "The CEG is given
  by C′/C" (Sec. 2, steps 1-4).
- **Family-relative.** "the exact relation between training compute and capabilities depends on the model
  family, so our definition … is also relative to a model family" (Sec. 2, footnote 9). Rationale: "training
  compute is a good proxy for capability" (Sec. 2).
- **Link to intrinsic performance.** Hilton et al. "define the 'intrinsic performance' of a model as the
  minimum amount of training compute that would be required to reach the same performance obtained by the
  model"; "The CEG can be seen as the ratio of the intrinsic performances of two models" (Sec. 2, footnote
  10).
- **Bounds instead of point estimates.** "we often calculate a lower bound, which can be done if the
  evaluation data contains a model with the post-training enhancement that outperforms some bigger model
  without the enhancement" (Sec. 2); App. A: "If all the above requirements are met, the CEG is lower bounded
  by C(ML) / C(ME)"; alternatively fit a scaling law to the non-enhanced models (App. A.1).
- **Findings.** "most surveyed enhancements improve benchmark performance by more than a 5x increase in
  training compute, some by more than 20x"; "fine-tuning costs are typically <1% of the original training
  cost" (Abstract). Solution selection (best-of-n, majority vote) is one of the five enhancement types
  (Abstract); Minerva (data cleaning + majority voting) reaches "a CEG of 30 in STEM benchmarks and 2400 in
  math benchmarks" (Sec. 5).
- **Interpretive limits stated by the authors.** "a high CEG might not indicate that the post-training
  enhancement significantly improves performance, but instead indicate that additional training compute
  doesn't improve performance" (Sec. 4); "If scaling the baseline does not improve performance, the CEG stops
  being meaningful" (Fig. 22 caption). "some post-training enhancements allow models to perform tasks that
  would be impossible for any model without it. In these cases, even if the enhancement greatly improves
  performance, the CEG is not meaningful" (Sec. 4). Scale dependence creates two non-equivalent definitions:
  extra compute to match, or "the reduction in compute that can be achieved with an enhanced model without
  reducing performance" (Sec. 4). A benchmark-specific CEG of 5 "doesn't necessarily imply that the
  post-training enhancement is as useful overall" as 5× compute (Sec. 1).

## Evidence and limitations

- Non-experimental: CEGs are read off other papers' results, often as bounds, often from non-compute-optimal
  families (Sec. 4, "Suboptimally scaled models"; "Models from different families"). Individual estimates
  are noisy by the authors' own account (Sec. 4, "Takeaway").
- Not checked: the per-enhancement estimates in Sec. 3 beyond a few examples; App. B.

## Connections and questions

- **Definition offered:** the capability gain of a post-training intervention = C′/C, the factor of extra
  (compute-optimal) training compute the original model family would need to reach the same benchmark
  performance; equivalently the ratio of intrinsic performances.
- **New vs better access:** not the paper's framing, but its two failure cases are exactly the two poles we
  need. (a) Finite, modest CEG: the gain is something more pretraining of the same family would also have
  delivered — "more of the same" (access/efficiency). (b) CEG undefined because no amount of baseline scaling
  reaches the performance ("impossible for any model without it", Sec. 4) or because the baseline curve is
  flat: the intervention gives the family something its own scaling does not — the closest the paper comes
  to "a new capability". Turning this into a rule needs a fitted baseline curve and a declared extrapolation
  range.
- **Null / floor:** none explicitly. But the CEG supplies the principled budget that pass@k lacks: compare
  RL's gain against what the same compute would buy in pretraining (and, via solution selection, in
  sampling); random weights correspond to C = 0, whose performance anchors the curve.
- **Transfer to our setting:** (1) Fit performance vs pretraining compute for the base family (intermediate
  pretraining checkpoints of pend, and/or several model sizes; the random-init checkpoint is the C = 0
  anchor) using a smooth metric (per-theorem log p̂_i or teacher-forced log p of reference proofs) and pass@1.
  (2) RL CEG = C′(perf(r16))/C(pend), with RL's own compute (≈ 1.8 M sampled attempts plus updates) as the
  cost; a CEG below the RL cost ratio means RL is a worse use of compute than more pretraining. (3) The
  inference analogue: k* such that pend pass@k* = r16 pass@1 ("sample-equivalent gain"), compared with RL's
  ≈ 400 attempts per target. Cost: cheap if pretraining checkpoints exist (evaluation only); otherwise
  requires training a small scaling ladder. Failure modes: needs a compute-optimal, same-family baseline
  curve (Sec. 4); the CEG depends on the metric (pass@1 vs log p give different C′), and r16 may sit beyond
  the fitted range, where the CEG becomes an extrapolation.
- Related: hilton2023singleagent.md (intrinsic performance), jones2021boardgames.md (train-test trade-off),
  villalobos2023tradingoff (screened), brown2024monkeys.md.
