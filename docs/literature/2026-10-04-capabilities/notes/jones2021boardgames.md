---
written_on: 2026-10-05
written_by: agent:claude
papers:
  - jones2021boardgames
---

# Scaling Scaling Laws with Board Games — compute frontiers, random-play anchor, train/test trade-off

Paper: [@jones2021boardgames]
Source: arXiv 2104.03113v2 (HTML rendering read in full except appendices; the text uses Roman-numeral
sections, cited below as Sec. II-D, IV-A … from the reading)

## Learnings

- **Train-time vs test-time compute are exchangeable at a fixed rate.** "the test-time and train-time compute
  available to an agent can be traded off while maintaining performance" (Abstract). Quantitatively, "the
  trade-off is linear in log-compute: for each additional 10× of train-time compute, about 15× of test-time
  compute can be eliminated, down to a floor of a single-node tree search" (Sec. IV-C, Fig. 9; 9×9 Hex,
  AlphaZero, test-time compute = MCTS tree size).
- **Per-snapshot performance is sigmoid in test-time compute.** "the performance of a specific snapshot is
  sigmoid in the test-time compute budget" (Sec. IV-C, Fig. 8; tree sizes 1-512 nodes).
- **An anchored capability scale.** Elo is fixed so that perfect play (MoHex) is zero: "we fix its play to
  zero for all Elo ratings reported herein" (Sec. II-D). Each training run follows a sigmoid "starting at
  random play and progressing up to some plateau" (Sec. IV-A, Fig. 5). Derived quantities: slope "500 Elo per
  order of magnitude increase in compute" (Sec. IV-A1); takeoff — "The minimum training compute needed to see
  any improvement over random play increases by 4× for each increment of board size" (Sec. IV-A3); "the
  distance between random play and perfect play increases by 500 Elo for each increment of board size"
  (Sec. IV-A4).
- **A stated criterion for a "key insight", and its absence.** "A spike with regards to compute might
  indicate the model had achieved some key insight"; instead "models' performance changes smoothly and
  predictably with both increased compute and increased complexity", though "this could plausibly be a
  property unique to Hex" (Sec. V).
- **Interpretation of the trade-off.** Test-time optimisation "needs only optimise over one sample, while
  train-time compute meanwhile must optimise over the entire distribution of samples" (Sec. V). Toy model:
  play may "reduce to each agent having a 'pool' of strategies proportional to its compute" (Sec. V).
- **Caveat on the rating scale.** "The central limitation of the Elo system is that it assumes
  transitivity" (Sec. II-D).

## Evidence and limitations

- Evidence: Fig. 5-9, Table III (fitted frontier parameters). Single game (Hex, board sizes 3-9), one
  algorithm (AlphaZero), test-time compute = tree search, not independent resampling.
- The trade-off rate (10× train ≈ 15× test) is one empirical number for one domain; it is not a law.
- Not checked: Appendix (implementation details).

## Connections and questions

- **Definition offered:** capability = position on a compute frontier: the best performance (Elo against a
  fixed perfect player, with random play as the other anchor) attainable for a given training compute; and
  an iso-performance curve in (train compute, test compute) space.
- **New vs better access:** implicit. The paper says what a "new insight" would look like (a spike in
  performance vs compute, Sec. V) and finds smooth curves instead. The iso-performance trade-off gives a
  usable rule for us: if RL's gain is matched by an amount of test-time compute (samples or search) on the
  base model that is in line with the train-time compute RL consumed — i.e. RL sits on the base model's
  train/test iso-curve — RL moved the model along a known exchange rate (access/amortisation). A gain that
  no reasonable amount of base test-time compute reproduces (off the iso-curve) is evidence of something
  new.
- **Null / floor:** explicitly anchored at both ends: random play and perfect play, plus a "takeoff" compute
  below which nothing beats random. For us the analogue is the random-init checkpoint (floor) and a takeoff
  point in pretraining compute where pass@k first leaves the random floor.
- **Transfer to our setting:** build the iso-performance map for our family: x = training compute
  (pretraining checkpoints up to pend, plus RL compute for r8, r16), y = test-time compute (samples k, or
  search nodes if best-first search is used), z = pass rate on the eval theorems. Read off the exchange rate
  "10× train compute ≈ M× samples" from pretraining checkpoints alone, then check whether r8/r16 fall on
  the pretraining-derived iso-curves (RL = amortised sampling) or above them. Cost: evaluation of existing
  checkpoints at several k (counts at k ≤ 512 already exist for pend, r8, r16; intermediate pretraining
  checkpoints would need sampling runs). Failure modes: one exchange rate may not hold across the range;
  RL compute (sampling + updates) and pretraining compute are not obviously commensurable; and independent
  resampling is a much weaker test-time method than MCTS, so the k needed may exceed the sampled range,
  sending us back to extrapolation (kazdan2025passk.md).
- Related: davidson2023retraining.md (CEG), hilton2023singleagent.md (intrinsic performance),
  villalobos2023tradingoff (screened), brown2024monkeys.md.
