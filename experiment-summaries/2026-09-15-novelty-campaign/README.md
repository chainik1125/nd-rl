---
written_on: 2026-09-16
written_by: agent:claude
---

# Does RL create capabilities or elicit rare ones? Novelty campaign

Status: completed 2026-09-16 00:26 UTC (autonomous run, 5.5 h, three A40 pods, ≈ $8), then
independently reviewed (`review_campaign.md` on the branch). Numbers below are the reviewed ones.

Question: when expert iteration against the verifier proves theorems the pretrained model did
not, is it amplifying behaviour the pretrained model already produced rarely, or composing
patterns that were absent from pretraining? See the
[proposal](../../docs/proposals/2026-09-15-novelty-campaign.md).

## Configuration

- Code and results: `dan-pandori/nd-takehome`, branch `dan_novelty` (`phase1.md`, `phase2.md`,
  `phase3.md`, `campaign.md`, `review_campaign.md`, `numbers.md`, `log.md`; figures in
  `figures/phase*.png`). Base setup is the take-home run (3.2M-parameter decoder, 4 layers,
  d = 256, absolute line indices with random start offset, cap-6 generator data, expert
  iteration k = 32, T = 0.8).
- Phase 1: `novelty.py` (teacher-forced log-probabilities, marginalised over start indices, at
  the sampling temperature), `coverage.py` (base model, 10⁴ samples per transfer theorem,
  10⁵ per validation-36 theorem), `minlen.py` (bounded minimal-length prover, bound 8,
  evaluation only).
- Phase 2: `patterns.py` (predicates on dependency-pruned proofs), `make_coverage_sets.py`
  (155k-proof pretraining sets assembled by subsampling an 820k-class raw pool so that one
  pattern's frequency is exactly f; f = 0 verified by reclassifying the written file).
  Patterns: derived-ORE (`ORE` over a non-premise line), reductio (`NEGI` of `( ~ G )` then
  `DN`), depth-3 (a third nested box). f ∈ {0, 10⁻⁴, 10⁻³, 10⁻², 10⁻¹} (derived-ORE max 0.02:
  the generator cannot supply more distinct classes). Targets: 1,000 per pattern from a
  strict 7–12-line pool whose generating proof uses the pattern and for which `minlen.py`
  finds no ≤ 6-line proof; disjoint by renaming class from every training set and from
  validation-36. 30 Stage-1 models, 24 EI arms (8 rounds), 6 frozen controls. Two training
  seeds at f ∈ {0, f_max}, one elsewhere.
- Phase 3: 623 textbook-shaped targets (classical schemata with fresh atoms, disjoint from
  validation-36 by class); EI from the take-home Stage-1 model with and without a
  generator-made ≤ 6-line precursor set; frozen control.
- Every distinct-proof count is computed after renumbering from `N1` (a take-home review had
  found start-index variants inflating counts ~13×).

## Observations

**Phase 1 (take-home run re-measured).** The log-probability estimate of reachability is
calibrated: predicted vs measured frequency of the RL-found proofs in 10⁴ base samples has
median log-ratio 0.00 on 941 theorems, all within 3×. The base model solves 934/1,638 transfer
theorems at 512 samples (the take-home control gave 939) and 1,080 at 10⁴; **348 of the 1,411
theorems RL solved are unsolved by the base at 10⁴**. All 29 nine-line RL proofs have base
probability < 10⁻⁵ (median log p ≈ −70). The improbability is concentrated: a median 66% of a
proof's surprisal sits in one line, and the surprising token is most often the third `|`
(opening a third box) or `F`. On validation-36 at 10⁵ base samples: `contraposition` p ≈ 1.8×10⁻⁴,
found at sample 5,681 (elicited); `export` p ≈ 6×10⁻⁸, 0 hits (RL's round-14 solution was not
reachable by sampling); `explosion` found 57 times by the base but never by EI in 32 samples
per round (a sampling-support miss, not a capability gap).

**Phase 2 (the dial).** Acquisition = fraction of targets solved with a proof that contains
the pattern (classified on the model's proof).

| pattern | f = 0, EI (s0 / s1) | f = 0, frozen | f = 0, base 10⁴ × 300 targets | f = max, EI |
|---|---|---|---|---|
| depth-3 | **0.341 / 0.271** (341 / 271 theorems; 400 / 327 proofs; first at round 3 / 5) | 0 | 3 solved, 0 depth-3 | 0.355 / 0.316 (f = 0.1) |
| derived-ORE | 9 / 2 theorems, all the degenerate `( X v X )` one-line-box form | 1 | 2 of 284 (degenerate) | 0.242 (f = 0.02); 0.222 already at f = 10⁻⁴ |
| reductio | 0 / 0 | 0 | — | 13 theorems vs 11 for the frozen control (f = 0.1) |

Independent recount of the depth-3 row from the raw files: zero written-form depth-3 proofs in
the 155,000-proof f = 0 set; the counts above reproduced exactly; every counted proof
re-verifies; targets and training set share no renaming class. The f = 0 depth-3 proofs have
base log p median −46 (best −22, ≈ 3×10⁹ samples); their surprisal sits on the third `|` in
608/610 and 475/491 proofs. Held-out greedy of the f = 0 models is 0.886 vs 0.948 for the
take-home (the held-out set contains the natural 17–18% reductio/depth-3 proofs).

Interpretations, with the review's corrections:
- *Depth-3 is composition.* With no third box anywhere in pretraining, RL produces third
  boxes at the f = 0.1 rate; the control and the base never do. The pattern is a structural
  repetition of a learned move (a box inside a box).
- *Derived-ORE is amplification.* The base already assigns ≈ 10⁻³ to the easiest instance of
  the never-seen composition; 16 examples in 155k give 22% acquisition. Within cap 6, 97% of
  expressible derived-OREs are the degenerate form, so the dial measures that template; the
  non-degenerate derived ORE that De Morgan-style theorems need appears ≤ 4 times in any arm.
- *Reductio is untested, not a wall.* 94% of the reductio targets are intuitionistically
  provable and the remaining 56 were solved (40 / 27 of 56) by `DN` on a *given* double
  negation. No target required the reductio shape. A reductio dial needs targets whose double
  negation must be derived.
- *Seed spread.* Reductio f = 0 solved 609 vs 170 targets from identical Stage-1 loss. With
  n = 2 this is instability of expert iteration on hard targets, not evidence of two
  equilibria; per-arm levels should be read as ±3× until more seeds run.

**Phase 3 (textbook curriculum).** Targets solved at round 8: frozen 161, EI 213, EI +
precursors 222 of 623. Gains are confined to contraposition (3 → 24/24), consequentia
mirabilis (9 → 24/24), hypothetical syllogism (22 → 24) and export (0 → 12; 19 with
precursors); De Morgan, distribution, dilemmas, Peirce and the negated conditional stay at
0–2/24 in every arm. Validation-36 `> 6` bin: 1/24 (contraposition) without precursors, 2/24
with (`export`, base p ≈ 6×10⁻⁸ — the one theorem the injection bought that sampling could
not reach). This reproduces the R1-CoT sprint's textbook wall and locates it: the schemata
that move are those the base model already assigns ≥ 10⁻⁴ to; the rest need a derived
disjunction or a derived contradiction, which no arm produces.

## Follow-up run (2026-09-16, reviewed)

Design from the review's limitations (`FOLLOWUP_BRIEF.md`, results in `followup.md`,
`phase2.md` §Follow-up, review in `review_followup.md` on the branch). 4.3 h, three A40s, ≈ $6.

- **Depth-3 at f = 0 replicates.** Three new f = 0 pretraining sets × two training seeds
  (8 arms total): acquisition 0.271–0.364, mean 0.339 ± 0.029; six f = 0.1 arms 0.122–0.355
  (0.307 ± 0.092). Distributions overlap; between-set SD 0.010 vs between-training-seed SD
  0.031. Correction to the first campaign: on some Stage-1 draws the base model does open a
  third box at p ≈ 10⁻²–10⁻³ on a few `A > (B > (C > D))` targets (frozen controls 0–10
  theorems), so "the base never does" is draw-dependent; ≥ 94% of each arm's depth-3 proofs
  stay below 10⁻⁴ under the base. Expert-iteration ignition is seed-driven and not specific to
  f = 0: one f = 0.1 arm reached 0.122 after igniting at round 7.
- **Reductio on 606 targets whose double negation must be derived** (21 classical-only
  schemata without `~~`, hand-checked): f = 0 seeds 1 and 2 solve 0/606; seed 0 solves 58, all
  with the strict `NEGI(~G)…DN` shape, confined to two 7-line schemata — and that seed's base
  model already produces the shape at 2×10⁻⁴–4×10⁻³ (pass@10⁴: 5/300, all reductio-shaped).
  Elicitation of a rare base-model generalisation, not RL composition. Peirce, excluded
  middle, De Morgan, converse contraposition: 0 in every arm including f = 0.1.
- **Cap-8 strict derived-ORE dial is flat**: 0.008 / 0.008 / 0.022 / 0.022 at f = 0, 0, 10⁻³,
  10⁻², against a pre-registered 10–35%; the arms solve ~40% of targets by other routes. The
  pattern is base-reachable but never required.
- Reviewed: every count re-derived independently and matched. Carried-forward issue: no pool
  yet *requires* a rule-sequence pattern, so "RL does not invent rule sequences" rests on
  absence-when-optional plus the base-reachability of the one positive case.

## Ignition study (2026-09-16, reviewed)

Eight new Stage-1 seeds each on the depth-3 f = 0 set `a1` and the reductio f = 0 set, a
pre-RL sample (300 targets × 2,000) per draw, identical expert iteration, and interventions
from the round-4 state on the non-igniting arms (`ignition.md`, `review_ignition.md`).

- **Reductio (a rule sequence): ignition is the pre-RL base rate.** 4 of 11 draws produce the
  strict `NEGI(~G)…DN` shape before RL (8×10⁻⁶ – 2.3×10⁻⁴ per sample); exactly those four
  ignite, in rate order; the seven zero-rate draws end at 0/606 after 155k attempts. The
  first-proof round matches 1/(r·k·N) within ~2.5×.
- **Depth-3 (a structural pattern): the base rate predicts early ignition and nothing else.**
  Draws with rate ≥ 2.5×10⁻⁴ ignite by round 3; of five draws with zero depth-3 samples in
  600k, three ignite (rounds 3–5) and two never — decided during training on depth-≤2
  proofs, not by sampling luck. Round-8 acquisition depends only on ignition (0.30–0.36 if
  ignited by round 5, ≤ 0.004 otherwise).
- **Interventions from round 4:** one training step on a sibling arm's found proofs ignites
  11/11 non-igniters in the next round and reaches the plateau (depth-3 0.33–0.35; reductio
  0.11–0.18, above every self-ignited reductio arm). One round at k = 128 or T = 1.0 changes
  almost nothing in either direction (review refinement of the study's wording).
- Base generalisation: 10/16 depth-3 f = 0 draws and 4/11 reductio draws emit the pattern at
  all in 600k samples.
- Next measurements (review): the pattern's base rate on intermediate checkpoints of
  zero-hit draws that later ignite (does the prior drift before the first proof?); a pre-RL
  sample over all targets.

## Limitations and next step

- Two training seeds at the extremes, one elsewhere; cap 6 makes the derived-ORE dial measure
  a degenerate template; the reductio dial is uninformative (targets did not require it); the
  base pass@10⁴ files store only distinct proofs, so "0 depth-3 in 3×10⁶ samples" rests on the
  sampler's bookkeeping; one model size; one generator.
- Next (after the follow-up): the ignition dynamics of expert iteration (why one seed reaches 86 depth-3 theorems by round 2 and another needs seven rounds); the base rate at which cap-6 Stage-1 draws generalise NEGI-then-DN (many seeds, sampling only); a target pool that *requires* a rule-sequence pattern. Earlier list: (1) more seeds at f = 0 for depth-3 and a reductio pool whose double negation must be
  derived (e.g. targets of the form `( P > Q ) , ( ~ Q ) |- ( ~ P )`-style contradictions
  with the negated goal only reachable through a box); (2) a cap-8 arm so a non-degenerate
  derived ORE is expressible in pretraining and its dial can be run; (3) the same dial on the
  85M model and the relative-unary codec used by the sprint, to see whether "structural
  repetition crosses zero coverage, new rule sequences do not" survives scale and tokenisation.
