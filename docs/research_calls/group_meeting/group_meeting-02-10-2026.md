# NDRL Group meeting - October 2

Friday, October 2, 2026. Transcribed from the SPAR discussion Google Doc.

## Standup

### Charles

- New dataset quite hard. Believed harder than the textbook problems. Lowest accuracy so far on models.
- Some theorems quite hard to solve, needed to use Claude to solve some of them.
- Dmitry: could be interesting to attempt to cluster problems. Try to classify problems as saying that proving a particular theorem implies a given capability (e.g. proving theorem backwards).

### Dan

- <https://github.com/chainik1125/nd-rl/blob/dan/docs/project_strategy/2026-10-01-week-in-review.md>
- Models pretty good now. RL env + model changes + longer proof pretraining all additive.
- TODO: read [Claude's output on RL after various amounts of pretraining](https://github.com/chainik1125/nd-rl/blob/dan/experiment-summaries/2026-10-02-rl-from-ckpt-early-start-ladders-reach-end-arm-from-step-5000/README.md).

### Leon

- <https://claude.ai/artifact/GPA4RNULj2YP53tNNUAuks>

### Robbie

- <https://robbiewmthompson.com/blog/combined-model/>
- Notes was not very successful at test-time-scaling. Tried:
  - beam-search, selecting the best. Did not uplift vs EI alone.
  - verify after each step helped (not really TTS).
  - attempt to train on slight variants of the goal proof (test-time RL). Early results did not show improvements vs EI alone.
  - Dmitry: analogy to AlphaProof/AlphaZero etc. MCTS.
  - Dmitry: analogy to LLMs. Try RLVR. Perhaps try curriculum learning.
  - Dmitry: analogy to formal logic. Finding longer proofs paper. Might be not-crazy to RL one of these models?
- Notes that adding test-time-scaling was on an already strong model.

## ToDos

- Dan: Test time scaling in the games tradition.
- Leon: Test time scaling in the LLMs tradition (thinking blocks).
- Robbie: Test time scaling in the looped transformers tradition (new pretraining arch).
- Charles: Even more data! Get some interesting and better pretraining data.

## Action items

- Test-time scaling
  - Dmitry: existence of test-time scaling is the degree of exploration.
  - Adding MCTS to the model [Dan]
  - Add LLM thinking or the broader LLM analogy [Leon]
- Better understand learning dynamics (especially elicitation vs capabilities).
- Charles: more pre-train dataset.
- Extend problem domain, e.g. include predicate logic.
- Try best models on Charles' new dataset.
- Dan: give example of where the reward is assigned. End of proof, vs individual state.

## Notes on where the reward is assigned

1. **Rollouts.** For each target theorem the model makes 32 attempts, one step at a time in the proof-state environment. The environment checks each step's *structure* (syntax, names in scope). If a step fails that check, the attempt **ends immediately** as a failure. There's no resampling and no partial credit. Whether the finished proof is logically correct is decided by Lean on the whole proof at the end.
2. **Selection: outcome only.** An attempt counts only if Lean accepts the complete proof. Failed attempts contribute **nothing**: no negative signal, and no credit for a correct beginning.
3. **Training: every step of each accepted proof, weighted equally.** Each accepted proof is split into (state, action) pairs, one per step, with cross-entropy on the action tokens. In each round:
   - up to 4 distinct proofs per theorem;
   - upweighted ×4 against retained pretraining examples that are mixed in;
   - accumulated across rounds;
   - 600 fine-tuning steps at learning rate 3e-4.
