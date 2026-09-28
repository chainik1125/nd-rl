---
written_on: 2026-09-08
written_by: agent:codex
---

# Project state

The three completed June natural-deduction sprints have been recovered from the earlier local
repository, with their code, summaries, ledgers, and sub-1MB figures now indexed here. The
[bootstrapping study](../experiment-summaries/2026-06-09-nd-bootstrapping-sprint/README.md)
contains the strongest verified written-length result: cap-6 training produced seven distinct
dependency-pruned eight-line certificates, while deeper cap-8 models reached 13 lines. The
[cross-experiment cap-six ledger](../experiment-summaries/synthetic-cap6-proof-lengths.md) links
the exact eight-line records and the specialized FLoP eleven- and 129-line witnesses. The
[proof-novelty study](../experiment-summaries/2026-06-10-proof-novelty-sprint/README.md) found a
modest two-seed benefit from high-lemma-value target selection, and the
[first-order study](../experiment-summaries/2026-06-11-fol-sprint/README.md) reproduced a token
inventory wall while localizing eigenvariable failures to freshness bookkeeping.

Four completed September 10 experiment records have now been imported from `simplex3` alongside
archived code snapshots. Large raw runs and checkpoints remain remote.

- The [OLMo-32B pilot](../experiment-summaries/2026-09-10-olmo-32b-pilot/README.md) found no
  longer-proof gain beyond ICL on its fixed 24-target panel.
- The [proof-cap comparison](../experiment-summaries/2026-09-10-proof-cap-comparison/README.md)
  completed all six configurations but did not extend verified textbook reach beyond the supplied
  supervision cap.
- The [proof-fragment experiment](../experiment-summaries/2026-09-10-proof-fragment-selection/README.md)
  reproduced its supervised transfer ranking, while all four tested long-RL procedures reduced
  complete-proof coverage from the adapted start.
- The [FLoP adaptation](../experiment-summaries/2026-09-10-flop-adaptation/README.md) showed narrow
  symbolic depth extrapolation without a held-out coverage gain for the token LM.

The earlier [R1-style reasoning sprint](../experiment-summaries/2026-09-08-r1-cot/README.md)
completed its first independent pair audit. No endpoint expanded the observed proof-length window
beyond six reference lines; the 73 final-only problems remained unopened in the local record.

A [literature review for learning beyond initial training data](literature/astra_suggestions/learning-beyond-training-data.md) now recommends five core readings and five additional readings, with experiments and a critique of natural deduction as the setting. Its proposed next step is to specify a fresh structural-transfer benchmark and information/compute accounting before choosing the next RL recipe; this does not alter the current sprint's evaluation restrictions.

A [three-hour A/B of two agent frameworks](../experiment-summaries/2026-09-23-rl-hillclimb-ab/README.md) hill-climbing the RL stage from the shared 4-layer pretrain checkpoint (2026-09-23) found GEAR-Evolve reached a sealed length-7-9 dev score of 401 of 1,610 against 339 (324 reviewed) for review-gated orchestration, from 238 for the seed pipeline; the chosen checkpoints solved 15-20 and 5-11 of 907 sealed length-10 theorems, none at length 11-14 beyond one, and no textbook problem above six lines. A companion [RL monitoring survey and dashboard proposal](https://github.com/chainik1125/nd-rl/blob/leon/rl-dashboards/docs/proposals/2026-09-23-rl-run-dashboards.md) is on branch `leon/rl-dashboards`.

A [second, twelve-hour run](../experiment-summaries/2026-09-24-rl-hillclimb-run2/README.md) (2026-09-24) with seven GEAR-Evolve workers and an auditor, on a freshly generated length-6-11 pool, found four levers (line-verified sampling, a CUDA-graph sampler, a capped line-index shift, long-heavy target subsets); the resulting recipe run for two hours proves 707 of 1,931 sealed length-12-14 theorems at pass@256 (37 percent; 10 at length 14) where the pretrained model proves none, and solves 3 of the 43 textbook problems above six lines, the first such solves. Length 15-16 remains untestable (nine labelled theorems). A same-card replicate of the top two recipes afterwards showed seed-to-seed spread of at most 27 score points on one card, against up to 234 across cards, so the card speed under the power cap made most of the search noise.

A [length-generalisation experiment](../experiment-summaries/2026-09-25-length-generalisation/README.md) (2026-09-24/25) trained the run-3 seed recipe with RL targets capped at 7, 9 or 11 lines and tested every length from 6 to 18: RL raises accuracy up to the trained length, but every arm falls to a few percent one line past its cap and near zero two lines past, with nothing solved at 14-18; attempts stay 6-10 lines long whatever the theorem needs. On constructed conjunctions of two small theorems the model solves the halves but not the whole; adding constructed targets to RL teaches composition only at the short lengths where it finds proofs, and leaves the standard length curve unchanged.

A [follow-up on representation](../experiment-summaries/2026-09-25-representation-and-goals-length/README.md) (2026-09-25) re-pretrained the 039 recipe with four proof representations (absolute line indices, relative citations, no positional encoding, line-structured positions) and a variant that writes its own goal notes, then ran the same RL on lengths 6-11: none moves the wall. Every variant is near zero by length 13 and at 0% from 14; removing positional encoding changes nothing, relative citations cannot cite more than 5 lines back, and goal notes add about 1.6 points at length 13 only. A randomized-position variant awaits a decision on how to evaluate it.

A [mid-training test](../experiment-summaries/2026-09-26-midtraining-long-glued/README.md) (2026-09-26, one seed) fine-tuned the 039 pretrain on 2,000 proofs of 16-24 lines, glued together from shorter ones, before the same RL on lengths 6-11. Before RL it solves about 1% of constructed conjunctions at 15-18, but nothing on natural or chain-joined long theorems. After RL it is back at the same wall as every other arm, and slightly worse on natural theorems at 12-14 (2.9% against 5.3% pass@1).
