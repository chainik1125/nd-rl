---
written_on: 2026-09-30
written_by: agent:claude
---

# Bigger model and longer pretraining: no gain, and a seed coin flip

Status: done, and deliberately not pursued further. Read this before spending more compute on
model size, pretraining length or sampling temperature for the autoresearch recipe.

Question: does run 198's recipe (the autoresearch incumbent, 9.6M params, 300 s pretraining, dev
>= 7 = 681) improve with a 4x bigger model or 4x longer pretraining, on the loop's dev metric or on
the 72 textbook problems?

## Configuration

- **Recipe:** 198's `pretrain.py` (branch `autoresearch-pretrain`, commit `c891da8`).
- **Harness:** the frozen harness with three constants changed in an uncommitted local copy, so
  this is off-protocol:
  - `PRETRAIN_SECS` 1200 (was 300);
  - `PARAM_CAP` 50M;
  - `TOTAL_SECS` 7200.
- **RL and eval:** unchanged (4 expert-iteration rounds, k = 32 at T = 0.8, dev eval k = 64).
- **Arms, 3 seeds each:**
  - `198` (baseline): 9.6M params, 300 s;
  - `scale-d384-p1200`: 9.6M params, 1200 s;
  - `scale-d768-p1200`: 38.0M params (d 768, 12 heads, MLP 2560), 1200 s.
- **Checkpoints:** HF `robbiethompson2018/nd-rl-checkpoints`, `runs/<arm>_s<k>`.
- **Textbook scoring:** `score_holdout.py --pool textbook` (72 problems, pass@256).
- **Temperature check:** each run's `pretrain.pt`, meaning the model RL starts from. A fixed 300 RL
  targets, k = 32, T in {0.6, 0.8, 1.0, 1.2}, judged like an RL round. The script isn't committed
  (it lived in the session scratchpad).

## Observations

| arm | dev >= 7 per seed | mean | textbook /72 per seed | mean |
|---|---|---|---|---|
| 198: 9.6M, 300 s | 680 / 670 / 694 | 681 | 18 / 17 / 20 | 18.3 |
| 9.6M, 1200 s | 702 / 294 / 265 | 420 | 22 / 17 / 17 | 18.7 |
| 38M, 1200 s | 710 / 300 / 311 | 440 | 19 / 19 / 16 | 18.0 |

- **Size doesn't help** at equal wall-clock: 38M vs 9.6M is 440 vs 420 on dev and 18.0 vs 18.7 on
  textbook. The 38M model got ~9k steps in 1200 s, against ~23k for the 9.6M.
- **Longer pretraining makes dev bimodal.** In each 1200 s arm, one seed beats 198 (702, 710). The
  other two never take off in RL: about 550–600 RL targets solved against ~1,180. For example, 38M
  seed 2 reaches a round-4 acceptance rate of 0.37, against 0.75 for seed 0. 198's three seeds all
  took off.
- **Textbook doesn't move:** 16–22 of 72 for all nine runs, whatever the size, budget or RL outcome.
  This matches the earlier pretrain-ablations finding that textbook tracks dev only loosely.
- **Nothing measured at the start of RL predicts which seeds take off:**
  - Pass@32 at T = 0.8 on the RL targets: good seeds span 0.13–0.47, failed seeds 0.12–0.21.
  - Distinct samples out of 32: 14–24% everywhere, with no split between good and failed seeds.
  - Raising the temperature from 0.6 to 1.2 adds 1–4 points of pass@32 for every model, failed or
    not.
  - The clearest case is 198's seeds 0 and 1: they start *below* the failed seeds on both measures
    (pass@32 0.13–0.14, 17% distinct) and still reach 680 / 670.
  - In the 9.6M/1200 s arm, held-out greedy accuracy after pretraining is actually higher on the
    failed seeds (0.970 and 0.970, against 0.956 for the good one).

## Interpretation

- **Scaling this recipe up doesn't pay off.** Neither more parameters nor more pretraining time,
  at this data and RL setup, improves dev or textbook. That's the practical conclusion.
- **Whether RL takes off looks like a coin flip.** It isn't visible in the pretraining checkpoint
  by pass@k, sample diversity or temperature. 198 going 3/3 may be partly luck: if each seed takes
  off with probability ~0.5, 3/3 happens about 1 time in 8.
- **This is untested:** RL was never rerun from the same checkpoint with different RL seeds, so
  "the checkpoint" and "the RL draw" haven't been separated.
- **Dan's noise-floor run found the same kind of seed noise** (frozen ladder 62–265 of 2,285 from the
  seed alone).

## Limitations and next step

- **3 seeds per arm with a bimodal outcome**, so every mean above is very noisy.
- **The comparison is at equal wall-clock, not equal tokens.** A compute-matched size sweep
  wasn't run.
- **Next step, if this is ever reopened:** rerun RL from one failed and one good `pretrain.pt` with
  2–3 RL seeds each. That separates "bad checkpoint" from "unlucky RL draw" for about $2. Until
  then, don't tune pretraining on single-seed dev differences smaller than a seed flip.
