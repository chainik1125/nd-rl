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

## Release

Shareable folder: `data/pretrain_data/release_2026-10-12_textbook72_distract2/` (README with the construction, four
worked examples on non-sealed *forall x* exercises, the models with hashes, the protocol and every metric;
`distractors.jsonl` + `apply_distractors.py` rebuild the modified set from the sealed files and check its canonical
sha256 `5f91d124…`; `metrics/summary.json` recomputes the tables above, with 95 % intervals over problems for Dan's
checkpoints: original 0.378 ± 0.088 / distracted 0.140 ± 0.061 for r8 s0, 0.124 ± 0.061 / 0.040 ± 0.027 for Stage-1).

## Mechanism (2026-10-12): the wall is the premise block, not the reasoning

Charles: "make some very elementary questions with 4 premises; it seems weird that the RL model collapses". Ladder of
21 trivial problems at 2 / 3 / 4 / 5 premises (`code/pretrain_data/elementary_prem.py`: pick one premise, conjoin two,
one modus ponens with unused premises, chains that use every premise, ∨-introduction), k 32, T 1.0, seed 0, pinned
verifier. Source: `data/pretrain_data/screens/elementary_prem_{la_T1_best12_s0_r8,scale-d384-p1200_s0}.json`.

| samples verified / samples | 2 premises | 3 premises | 4 premises | 5 premises |
| --- | --- | --- | --- | --- |
| Dan r8 s0 | 154 / 160 | 111 / 128 | **0 / 320** | 0 / 64 |
| Robbie d384 base | 115 / 160 | 90 / 128 | **0 / 320** | 0 / 64 |

Even `P , Q , R , S ⊢ P` and `P , Q , R , S ⊢ ( P & Q )` are 0 of 32 on both. The raw samples show why:

- Robbie's base on `P , Q , R , S ⊢ ( P & Q )` writes `N1 P : PR ; N2 Q : PR ; N3 R : PR ; N4 ( P & Q ) : ANDI N1 N2 ;
  QED` in every sample: the proof is logically right, but it lists three premise lines and the pinned verifier
  requires the PR block to equal the declared premises exactly and in order (`nd_verify/verify.py:158`:
  "premise block does not match declared premises"; 28 to 32 of 32 samples per four-premise problem fail with this
  reason). Adding `N4 S : PR` makes the same proof verify (checked). On the chain `P, P→Q, Q→R, R→S ⊢ S` it lists the
  first three premises, then invents the fourth step.
- Dan's r8 through his environment (`code/pretrain_data/elementary_trace.py`, 32 traces per prompt): on three
  premises the first actions are `h1 h2 h3` in 30 of 32; on four premises the first premise line cites `h2` in 25 to
  32 of 32 (the rest `h1` then `h3`), and the environment rejects it at once ("PR position", premise lines must be
  h1, h2, … in order). The model's premise-listing routine starts one premise late when there are four, so it never
  lists more than three.

So no model has learned to write a fourth premise line, which K12 never shows (≤ 3 premises), and both judges
require every premise to be listed before anything else counts. The 0-of-25 result on the distracted 72 and the
0-of-180 on generator problems are this one habit, not a failure to reason with or ignore premises. It is still a
data gap and the premise pool is still the fix, but the prediction becomes sharper and cheaper to test: a Stage-1
that merely shows four- and five-premise PR blocks should remove the wall outright, and the "distractor" results
should then be re-read as premise selection proper. Side note: Robbie's base also fails the one-line problems
`P , Q ⊢ P` (0 of 32, "LEANPARSE name"), another shape K12 never shows.

## Could RL teach the premise block? (2026-10-12, Charles's question)

Three measurements on `la_T1_best12_s0_r8`, all CPU, one seed.

**1. What RL has seen.** The round-8 training mix (`best-state/artifacts/bs/la_T1_best12_s0/mix_8.jsonl` in Dan's
bucket, 89,724 rows) has 14,279 / 37,807 / 28,454 / 9,184 rows at 0 / 1 / 2 / 3 premises and none at four or more.
The ladder never poses a four-premise prompt, so as run it cannot learn one.

**2. What RL would see if it did** (`code/pretrain_data/elementary_forced.py`; source
`data/pretrain_data/screens/elementary_forced_la_T1_best12_s0_r8.json`). Teacher-forced through Dan's environment with
canonical names, the probability the model gives each candidate premise line at each step of the block, on
`P , Q , R , S ⊢ P`: step 1, `h1` 0.0011 vs `h2` 0.0152 (share of the correct line 0.07); step 2 after a correct
`h1`, `h3` 0.999 and `h2` below 1e-6; step 3 after `h1 h2`, `h3` 0.31; step 4 after `h1 h2 h3`, `h4` below 1e-6.
Product for the whole block about 4e-24. The rule the model learned is positional: with k hypothesis lines in the
state it emits the premise line K12 would have next (k = 3 lines → `h1` was the exception, since 3 premises is the
most K12 shows; 4 lines → `h2`; 5 → `h3`; 7 → never a premise line). A plain search confirms there is nothing to
reinforce: 0 of 1,024 samples at T 1.0 and 0 of 1,024 at T 1.5 (environment rejection "PR position" 1,022 and 991
times). Expert iteration with the current environment would receive zero reward on every four-premise prompt.

**3. Take the block out of the model's hands** (`code/pretrain_data/elementary_envpr.py`; source
`data/pretrain_data/screens/elementary_envpr_la_T1_best12_s0_r8.json`, `run/metrics_envpr.jsonl`). The premise lines
are determined by the prompt, so the environment can apply `h1..hn` itself and let the model act from that state;
the verifier is unchanged (the finished proof still contains the lines). Same checkpoint, no training, k 32, T 1.0:

| samples verified / samples | 2 premises | 3 premises | 4 premises | 5 premises |
| --- | --- | --- | --- | --- |
| model writes the block (as before) | 154 / 160 | 111 / 128 | 0 / 320 | 0 / 64 |
| environment writes the block | 153 / 160 | 106 / 128 | **252 / 320** | **57 / 64** |

Per problem at four premises: pick 32 / 32, conjoin 31 and 29 / 32, all-four conjunction 26 / 32, modus ponens 31
and 29 / 32, ∨-introduction 32 / 32; the four-step chain that needs every premise 0 / 32 (the samples skip a link:
`S : IMPE (R→S) Q`), its reversed order 10 / 32. Generator distractor rows (the 2026-10-10 stratified sample,
300 rows): problems solved at least once 9 / 9, 33 / 33, 78 / 78 at 1 / 2 / 3 premises, **118 / 120 at four**
(pass@1 0.83) and **55 / 60 at five** (0.74), against 0 / 180 with the model writing the block. The distracted 72
(aggregates; names only in the gitignored hit file): original 48 of 71 (0.374), distracted **38 of 71** (0.210,
was 28); the 25 crossing problems 16 → **10** (was 17 → 0); the 46 staying problems 32 → 28 (unchanged); by
original premise count 9 → 7, 23 → 21, 13 → 9, 3 → 1.

**Answer.** As the ladder stands, no: it never sees a four-premise prompt, and if it did, the sparse reward would never
fire (success probability about 1e-24, 0 of 2,048 samples), because the failure is a deterministic line-counting
habit, not noise that search could get lucky against. The habit is also not worth RL: the block is boilerplate. Two
cheap fixes, either of which removes the wall without a single RL step: (a) the environment writes the premise block,
which already lifts the same checkpoint to 118 of 120 four-premise generator problems and 38 of 71 distracted
textbook problems; (b) Stage-1 exposure to four- and five-premise premise blocks (the premise pool), a supervised habit
change. After either, RL has a dense signal (pass@1 about 0.8 at four premises) and a real residual to work on: the
all-premises chain (0 of 32) and the 6 of 16 crossing textbook solves still lost to two unused premises, which is
the premise-selection effect proper, now visible for the first time (37 % of crossing solves lost versus 12 % of
staying ones).

Caveats: one seed, one checkpoint; (a) changes the model's job in the environment and the evaluation protocol, so it
is a plan decision (questions.md 10), used here as a diagnostic only; the stated probabilities are under canonical
names with base 0 and the shares across candidate lines are the robust numbers.
