# results: textbook72_distract2 — the sealed 72 with two unused premises each

Run 2026-10-09 evening (UTC-7), paused at Charles's request after the first four models; Dan's r8 seeds 1 and 2 not run.
Aggregates only (the 72 are sealed; per-problem outputs are in the gitignored `docs/textbook-problem/textbook72/`,
raw samples were not read for the same reason). Protocol: k = 32, temperature 1.0, seed 0, pinned verifier;
Robbie's pretrains through `eval_ckpt.py` (lean_seq, max 400 tokens), Dan's checkpoints through his environment
sampler (48 steps, 256 tokens per action). Distracted set: 71 of 72 (one problem could not take two satisfiable,
shortcut-free distractors), 25 problems cross to four or more premises, 46 stay at two or three.

## Headline

| model | original: solved ≥ 1 of 32 (avg rate) | distracted: solved (avg rate) | problems crossing to ≥ 4 premises (25): original → distracted | problems staying ≤ 3 (46): original → distracted |
| --- | --- | --- | --- | --- |
| Robbie d384 pretrain (9.56M) | 12 of 72 (0.104) | 5 of 71 (0.027) | 5 → **0** | 7 → 5 |
| Robbie d768 pretrain (38M) | 11 of 72 (0.129) | 6 of 71 (0.030) | 3 → **0** | 8 → 6 |
| Dan Stage-1 base (cap 12, state format) | 25 of 71 (0.124) | 12 of 71 (0.040) | 8 → **0** | 17 → 12 |
| Dan RL round 8, seed 0 | 49 of 71 (0.378) | 28 of 71 (0.140) | 17 → **0** | 32 → 28 |

Sources: `run/robbie_d384/metrics.jsonl`, `run/robbie_d768/metrics.jsonl` (Robbie's bins recomputed from the gitignored
sample dumps), `run/metrics.jsonl` (Dan, with the bins). n = 71 problems, one seed per model.

By original premise count (problems solved at least once, original → distracted; n per bin 16 / 30 / 22 / 3):

| model | 0 premises | 1 premise | 2 premises | 3 premises |
| --- | --- | --- | --- | --- |
| Robbie d384 | 1 → 0 | 6 → 5 | 5 → 0 | 0 → 0 |
| Robbie d768 | 1 → 0 | 7 → 6 | 3 → 0 | 0 → 0 |
| Dan Stage-1 | 5 → 4 | 12 → 8 | 7 → 0 | 1 → 0 |
| Dan r8 s0 | 9 → 7 | 23 → 21 | 14 → 0 | 3 → 0 |

## Conclusion

The pre-registered decisive cell was the number of crossing problems still solved: predicted 0–3 for every model if
the premise-count wall holds, 8 or more for the RL models if it does not. **It is 0 for all four models**, 33 solved
problems lost in total, while the problems that stay at three or fewer premises keep 71–88 % of their solved count
(Dan r8: 32 → 28). The wall found on generator problems is a property of the models: none of them, pretrain or RL,
writes a proof under a prompt with four or more premises, needed or not. The RL model loses 21 of its 49 textbook
solves to two unused premises, all 17 of them on the problems that crossed four.

What this changes: the four-premise textbook cell (0 of 7 in the 2026-10-02 evaluation) was not a premise-selection
failure but a premise-count failure, and every textbook problem with two or more premises is one distractor away
from it. The premise pool (`k12_prem4_10`) and its fractions are the first Stage-1 to run; hypothesis 2 (exposure,
not volume) is the first test. A model that clears this wall should recover most of the 21 lost solves without any
other change, which is a sharp prediction for that run: distracted-72 solved by the ladder on a premise-pool
Stage-1 ≥ 40 of 71, against 28 here.

Original-set numbers reproduce the earlier figures under the same evaluators (d384 12 of 72 on 2026-10-02; r8 seed 0
49 of 72 on 2026-10-02 with Lean gating), so the protocol is not what moved.

## Caveats

- One seed for Dan's RL model (seeds 1 and 2 pending); Robbie's pretrains one seed each; no comparison claim beyond the
  within-model paired change, which is 33 → 0 on the crossing problems and is not a seed effect.
- 27 of the 71 distracted problems have "unknown" shortcut status (the 10 s minimal-proof search timed out); a
  shortcut could only make a distracted problem easier, which works against the result.
- Parse-failure fraction rose for Robbie's models (0.32 → 0.43, 0.38 → 0.48): longer prompts also break the lean_seq
  output format more often; part of the mechanism may be formatting rather than reasoning. Dan's sampler reports no
  parse-failure rate per set.
- One problem of the 72 is absent from the distracted set.
