# NDRL Group meeting - September 27

Sunday, September 27, 2026. Transcribed from the SPAR discussion Google Doc.

## Updates

### Dan

- [Dataset generation styles didn't change much, other than the option for extending the length of the proofs that were pretrained upon.](https://github.com/chainik1125/nd-rl/blob/dan/experiment-summaries/2026-09-24-ds-composition-training-set-composition/README.md#headline-tables-reviewers-values)
- Lean was better.

### Robbie

- 15% to 60% accuracy headline.
- Lean was much better.
- Length 12 proofs reached.
- Todo: pass@k curves, steal NanoGPT ideas, Olmo paper, possibly other papers, base models know how to reason, thinking models know when.

### Charles

- EI seemed to do well.
  - Improved average accuracy.
  - Dmitry had similar experiments.
- GRPO performed less well.
  - Still had some improvements.
- Considering doing a larger run for EI.
- Experiments on EI vs GRPO.

### Leon

- Similar results to Charles.
- Agents found some improvements on sampling efficiency (e.g. upsample hard proofs).
- Was not able to improve length much.
- pass@256, some successes at length + 6.

## Notes

- Goal: review results of previous meeting, standup style.
- Goal: identify blockers.
- Would be good if folks present slides or artifacts to share.
- Dmitry: notes that RL'ing proofs is a mildly difficult project in itself.

## Proposed next steps

- For all: Try to prepare 2-3 slides to summarize what we did/what was successful/what we are planning to do.
- Dmitry: would be good to add explicit elicitation tests.
  - Pass@k curves
  - Activation steering vectors from RL'ed model, applied to pre-trained model
- Could be good to try PPO.
- May want to add explicit notions of compositionality in dataset, see if that improves.
- Could be good to examine the literature to find best way to pre-train + do RL.
  - OLMO feels like a bare necessity.
  - Base models know how to reason, RL models know when
- May need to test using a longer proof length. 9 to start, perhaps?
- Datasets
  - Explicitly increase the size of the heldout textbook sets.
  - Maybe have test vs validation sets.
  - Curating a larger set of interesting textbook problems.
  - Adding a categorization system within the proofs.
- Include more state in RL environment.

## Four new tasks

1. Measures of elicitation (Leon)
   1. Pass@k, excess description length, etc.
2. Continue working on RL (Dan)
   1. Adding state
3. Dataset (Charles)
   1. Go find a much stronger textbook set
4. Test time scaling (Robbie)
   1. MCTS / Beam search / inspiration from games
   2. Also polish off the pretrain, pipe all previous methods together

## Deliverables

1. Update mid-week
2. Update end-of-week
