---
written_on: 2026-09-17
written_by: agent:claude
---

# Proposal 6 (meta): a ladder of RL techniques to maximise the true proof-length frontier

Requested by Dan. This returns to the project's original objective — Dmitry's L − P, the longest
proofs an RL-trained model writes beyond its pretraining cap — with two changes the campaign
made necessary: the length that counts is the **true** length (the shortest proof that exists,
not the one the model wrote), and the choice of RL technique is treated as an experiment with a
leaderboard, equal budgets and pre-registration per rung, rather than a single design decision.

## The metric, fixed before any technique runs

- **True length** `L_true(θ)`: the smallest bound at which the bounded prover (`minlen.py`,
  iterative deepening over the 14 rules with Fitch semantics, formulas restricted to the
  subformula closure plus negations) finds a proof, when the search at every smaller bound
  terminates without one. Report the bound it was run to. Caveat, stated on every table: the
  restricted formula space means "no ≤ L−1 proof found" is a proof of non-existence *within
  that space*, not in general; a theorem's true length can be lower if the only shorter proof
  uses a formula outside the closure. Theorems where the prover times out are labelled "≥ bound,
  unresolved" and never counted toward a frontier.
- **Robust true frontier** `L*`: the largest L such that the model has verifier-accepted proofs
  of **≥ 5 distinct theorems** with `L_true ≥ L`, from a fixed evaluation pool, at a fixed
  attempt budget, counts start-index-normalised. Written and pruned lengths are reported beside
  it but do not define it. The headline for a technique is `L* − L*_frozen` for the same
  Stage-1 model resampled the same number of times, plus the solve rate on the pool by
  `L_true` bin with Wilson intervals.
- **Base reachability** of every counted proof (`novelty.py`), so each frontier gain is also
  classed as elicitation or creation.
- Pools: a **transfer pool** of ≥ 1,500 theorems with `L_true` resolved in 7–14 (from the
  strict long generator plus the textbook-shaped schemata pool), never sampled during
  training; an **RL-target pool** disjoint from it by renaming class; validation-36 reported
  by bin; the test files untouched. The pools are built once, committed, and shared by every
  rung so the ladder compares techniques and nothing else.

## The ladder

Every rung starts from the same Stage-1 model (the take-home's cap-6 `abs` model, and, as a
second lineage reported separately, one 85M-scale or cap-8 model if a later phase funds it),
uses the same RL-target pool and the same **total sample budget** (256 samples per target,
the equivalent of expert iteration's 8 rounds × k = 32), two training seeds, and a frozen
control. Each rung has a pre-registered `L*` prediction and a one-line mechanism before it runs.

| rung | technique | what it changes | why it might raise L* | pre-registered risk |
|---|---|---|---|---|
| T1 | expert iteration (baseline, as run) | — | reference | — |
| T2 | EI + difficulty-weighted sampling | allocate the sample budget to targets with low but non-zero per-sample success; drop saturated ones each round | the ignition study says success at the frontier is what moves the prior; spend samples there | budget concentrates on a few theorems; L* rises on RL targets, not transfer |
| T3 | EI + hindsight relabelling | keep verifier-valid by-products (proofs of other conclusions from the same premises) as training data, class-disjoint from every evaluation pool | more long verified proofs per sample | teaches theorems nobody asked for; must be reported as outside-target data |
| T4 | EI + moving target pool | regenerate targets each round at the current frontier (`L_true` = current L* + 1…3) | keeps signal at the edge as the edge moves | pool drift; requires the necessity oracle every round (CPU) |
| T5 | EI + precursor injection | one training step on generator-made ≤ 6-line instances of the shapes the unsolved targets need (from `patterns.py` on their generating proofs) | the sprint's precursor idea; ignition study: one injection ignites | shapes must be identified without reading evaluation pools; cap-6 data only |
| T6 | EI + sibling ensemble | N Stage-1 seeds trained in parallel; found proofs shared across siblings each round | 11/11 dead arms ignited on sibling transfer | N× training cost; effectively a larger k |
| T7 | GRPO at matched samples (from run 4) | on-policy policy gradient, groups of 32, no KL | the sprint's method, compared fairly | no signal on all-fail groups |
| T8 | preference training on accepted vs rejected pairs (DPO-style) | off-policy update from the same samples | uses failures, which EI discards | reward hacking toward padding; pruned length guards |
| T9 | EI + length-selected training | train on each theorem's longest *pruned* proof | direct pressure on length | negative once at cap 6; retest under `L_true` |
| T10 | verifier-guided search at evaluation (best-of-N with prefix reuse) on the T1 model | no training change; a model-plus-search capability row | separates "the model can" from "the model plus search can" | reported in its own row, never pooled with training rows |

Phase A runs T1–T6 (the cheap, EI-family rungs) — about 12 arms, ≈ $50 on three pods over a
day. Phase B runs T7–T10 and re-runs the best Phase-A rung with a second lineage (cap 8, or a
larger model) — ≈ $50–100, asked for in `QUESTIONS.md` with the Phase-A leaderboard attached.
Phase C is the meta part: with the leaderboard in hand, the **proposer role** writes the next
rung as a mutation of the best one (its one changed parameter and its predicted L*), and the
loop continues while each new rung is cheap and its prediction is falsifiable. The stop rule:
three consecutive rungs that fail to raise `L*` on the transfer pool by ≥ 1 with the frozen
control accounted for, or the true-length oracle failing to resolve the pool at the next L.

## Deliverables

`ladder.md`: the leaderboard (rung, `L*` transfer / RL-target, `L*_frozen`, solve rate by
`L_true` bin, base-reachability split, cost), updated after every rung; `figures/ladder_*.png`
(solve rate vs `L_true` per rung with the frozen control; `L*` over rungs); per-rung
`preregistration/ladder-T<k>.md`; the shared pools committed with their `L_true` labels and the
prover bound; a `review_ladder-*.md` per phase from the reviewer role before any rung is
called the leader.

## What would count as an answer

`L* − L*_frozen ≥ 2` on the transfer pool for some rung, replicated on both seeds and
classed by base reachability, is the first honest positive answer to the project's original
question. Every rung at `L* − L*_frozen ≤ 1` is the honest negative one, and says which
mechanisms were tried.
