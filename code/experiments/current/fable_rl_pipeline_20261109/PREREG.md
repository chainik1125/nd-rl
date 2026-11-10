---
written_on: 2026-11-09
written_by: agent:claude
---

# fable_rl pipeline: pre-registration

Written 2026-11-09 about 9:15 pm PT, before the first RL run on a fully pretrained base. The task
and its rules are in `/scratch/cowifornia/nd-rl/fable_rl/BRIEF.md` (revised version: x is Dan's
full corpus; long test sets at 13 to 20 and 21 to 30 lines were added). Amendments, if any, are
dated at the end of this file and made before the run they apply to.

## Data

- **x:** `train_k12.jsonl.gz` (md5 c5a8cd37…, 155,000 proofs). 1,500 proofs (shuffle seed 0) are
  held out for a validation loss; 153,500 are trained on.
- **y:** `dan_rl_targets.jsonl`, 4,495 statements. Only `name`, `thm`, `key`, `n_lines`, `source`,
  `schema`, `n_prem` are kept; the `gen_proof` and `rules` fields are dropped when the split is
  written and no code reads them. `n_lines` and `source` are used only to stratify the split and
  to break results down; `schema` is not used.
- **Split of y** (seed 20261109, stratified by source and reference length): 4,022 train, 473 dev
  (10% of each 7 to 10 line stratum, 20% of each 11 to 14 line stratum).
- No other file is read. No test set is read or copied to a pod.

## Pipeline

1. **Format and environment:** Dan's step environment and `lean_staten` tokens, vendored
   unchanged. The policy sees the tactic state and writes one action.
2. **Model:** from-scratch GPT, 8 layers, d 384, 8 heads, 14.3M parameters; a policy head, a
   verifier scalar read at the last token of an action, a value scalar read at `<act>`.
3. **Stage 1, pretraining on x:** 6,000 steps of about 120,000 padded tokens (about 5.9 epochs),
   AdamW, lr 8e-4 with 300 warm-up steps and a cosine to 8e-5, cross-entropy on action tokens,
   random name offset per pair.
4. **Stage 2, RL on y train:** rounds of (attempts, Lean, hindsight, update) for 9 hours per seed
   (`fr/rl.py`, defaults as committed at the launch commit):
   - 16 attempts per theorem with no proof yet, 8 per theorem with one, temperature 1.0.
   - Search at training time: each sampled step is type-checked on its own (Dan's reject-only
     pre-filter applied to that one step); a refused step is resampled up to 5 times at a rising
     temperature; a stuck attempt may abandon its innermost open box up to twice; then it ends. Refused steps become verifier negatives and get an
     unlikelihood loss in the policy.
   - Reward: Lean 4 on every finished proof that is kept (as produced and dependency-pruned).
   - The shortest pruned proof per theorem is kept across rounds; up to 3 fresh proofs per theorem
     per round are kept for two rounds.
   - Hindsight: a failed attempt on a still-unsolved theorem is relabelled as a proof of the
     top-level line with the largest dependency closure (at least 4 lines); Lean must accept it.
   - Update: 600 steps per round at lr 2e-4 on a mixture, by example: x 35%, best proofs 30%,
     fresh proofs 10%, hindsight lemmas 10%, verifier and value examples 15%. Each proof enters
     pruned and as produced (full states, only the used steps as policy targets).
5. **Solver:** `solve.py`. Default mode "search": the model's verifier head screens each atomic
   step (resample below a threshold), an action the environment refuses is resampled, a stuck try
   restarts, all inside one attempt, without Lean or any type check. "plain" mode is one rollout
   with no resampling, for a like-for-like comparison with the baseline's sampling. The search
   settings (temperature, threshold, resamples, tries) are chosen on the dev split with seed 0 and
   then frozen for all seeds. Tokens generated per attempt are reported for both modes.

## Seeds and checkpoints

- Seeds 0, 1, 2: each is its own pretraining and its own RL run.
- Dev is evaluated after round 1 and then every 3 rounds, 5 draws per theorem, both modes.
- **Checkpoint rule:** for each seed the final checkpoint is the evaluated checkpoint with the
  highest dev pass@1 in search mode; ties go to the later round.
- The orchestrator tests each final checkpoint once.

## What I expect (guesses, written before any RL result)

- The pretrained base in plain mode solves few dev theorems, under 10% pass@1: at step 2,000 its
  finished attempts on y were all ill-typed.
- After RL: plain pass@1 on dev 50 to 70%, search pass@1 80 to 90%, search pass@5 above 90%.
- Success falls with reference length; at 13 and 14 lines I expect 40 to 60% pass@1 in search mode
  (only 6 dev theorems there, so this is loose).
- I expect the verifier-screened search to be the largest single contributor to pass@1, and
  checker-in-the-loop sampling to be what makes RL on y move at all.
- On the long test sets I have no measurement to go on. Guess: 30 to 60% at 13 to 20 lines and 5 to
  30% at 21 to 30 lines.

## What would change the plan

A pilot result that shows a bug or a recipe that does not learn. Any such change is written below
with its time before the affected runs start.

## Amendment 1 (2026-11-09, about 10:00 pm PT; before any RL run that is kept)

Made after the first base was scored on dev and after the orchestrator's interface note
(`/scratch/cowifornia/nd-rl/fable_rl/solver_interface/INTERFACE.md`) and its ruling on premise
lines. No RL result existed yet: the seed-0 RL run started at 05:45Z had not finished its first
round and is discarded, as are the seed-1 start of 05:55Z and an earlier single-process start.

- **My guess about the base was wrong.** The seed-0 base in plain mode solves 57.4% of dev at
  pass@1 and 76.5% at pass@5 (seed 1: 57.9 / 74.8), not under 10%. It is weak where I should have
  looked: textbook 10.1% pass@1 against generated 68.3%, 12 lines 4.8%, 13 and 14 lines 0 of 6.
  Expectations after RL are unchanged.
- **Final mode.** The headline mode is `SOLVER_MODE=full` (the "search" mode above). `plain` is
  one pure model rollout with no resampling and no scaffold, reported beside it.
- **Premise scaffold in full mode (a proof-format choice).** Dan's format starts every proof by
  restating premise k as line k. In full mode the solver writes those lines itself: they are a
  function of the statement alone, with no choice, nothing learned or searched. They cost 0
  generated tokens and the solver reports how many it wrote (`scaffold_lines`). Reason: x and y
  have at most 3 premises, so a model trained on them never emits `h4`, and a test theorem with 4
  premises would fail on the format, not on proving. Plain mode has no scaffold.
- **Subgoal variants (new training statements from the model's own proofs).** When the kept proof
  of a y theorem ends by closing an implication or negation box, the same lines are a proof of the
  subgoal 'premises , assumption |- body goal' (falsum for a negation box). Lean checks each
  variant; they are trained on as a sixth pool. This yields 4-premise statements from 3-premise y
  theorems, so the model can learn `h4` itself and plain mode is covered too.
- **Mixture** is now, by example: x 32%, best proofs 28%, fresh proofs 10%, hindsight 8%, subgoal
  variants 7%, verifier and value examples 15%.
- **Collection** runs in 4 worker processes on the one GPU; retry temperature rises 15% per refused
  sample, capped at 1.6 times; an attempt samples at most 400 actions in training. Solver defaults
  before tuning: threshold 0.5, 4 resamples, 2 abandoned boxes, 4 tries, at most 600 sampled
  actions per attempt, at most 128 steps per try and 512 tokens per action.
- **Scoring on dev** uses the route the orchestrator's judge takes (`nd2lean.translate` with
  `require_all_pr=False`, then Lean 4), on the ND proof the solver outputs.
- **Ablation** (one run, seed-0 base, same budget): `--gate none`, that is, no step check while
  sampling and therefore no verifier; everything else equal; scored in plain mode.

## Amendment 2 (2026-11-09, about 11:10 pm PT; before any RL run that is kept)

Made from one-round A/B runs on the seed-0 base (dev, plain mode, pass@1 / pass@5, share of
attempts that end in a complete proof). The base itself: 57.4 / 76.5, 87.6% complete. All RL runs
so far were pilots and are discarded.

| one RL round with | pass@1 / pass@5 | complete |
|---|---|---|
| unlikelihood loss on refused steps, lr 2e-4 | 42.0 / 68.3 | 51.9% |
| no unlikelihood, lr 2e-4 | 49.3 / 73.8 | 72.8% |
| no unlikelihood, no as-produced examples, lr 2e-4 | 50.2 / 73.8 | 77.1% |
| no unlikelihood, lr 1e-4 | 55.5 / 77.8 | 76.0% |
| no unlikelihood, verifier and value losses off, lr 2e-4 | 60.2 / 80.8 | 88.6% |

Changes to the plan:

- **No unlikelihood loss.** It taught the policy to drop the closing `;` (328 of 670 failed plain
  rollouts after one round ended that way).
- **No verifier or value loss on the policy network.** They cost the policy 11 points of pass@1 in
  one round (49.3 against 60.2). The policy network is trained on policy targets only; its two
  scalar outputs are unused.
- **The verifier is a separate small network** (`fr/verifier.py`), trained from each run's own
  saved data: the steps the checker refused while sampling (negatives) and steps it accepted
  (positives). Its input is the step and the hypothesis lines the step cites, copied from the
  proof state (which lines a step cites is read off the step's text; the judgement is the
  network's). It does not see Lean or the type checker at test time. Full mode uses it exactly as
  the shared head was to be used.
- **Attempts per round:** 16 per theorem in round 0; afterwards 6 per theorem that has a proof and
  48 per theorem that has none; at most 300 sampled actions per attempt. Mixture by example:
  x 37%, best proofs 33%, fresh 12%, hindsight 9%, subgoal variants 9%. Budget 11 hours per seed.
- **Checkpoint rule (replaces the earlier one):** dev is scored in plain mode after round 1 and
  then every 3 rounds; for each seed the final policy checkpoint is the scored one with the highest
  dev pass@1 in plain mode, ties to the later round. The seed's verifier is trained on the data of
  all its rounds. Solver settings (threshold, resamples, tries, temperature) are tuned on dev with
  seed 0 and frozen for all seeds. Headline mode stays `SOLVER_MODE=full`.
- **Ablation** as in amendment 1 (`--gate none` from the seed-0 base, same budget).
- Expectations after RL are unchanged, except that plain mode should now rise, not fall.

## Amendment 3 (2026-11-10, 12:00 am PT; applies to the kept runs from their round 4 or 5 on)

Observed in the kept runs after 3 to 4 rounds: the training theorems still without a proof sit in
a few textbook families (in seed 1 after round 4: `excluded_middle` 1 of 36 solved,
`demorgan_nand_to_or` at 14 lines 0 of 10, `peirce_sequent` 7 of 17, `peirce` 7 of 12), about 80 to
140 theorems per seed. The schema names are read here only to describe what is unsolved; training
still uses nothing but "has a kept proof or not".

Change: theorems without a proof get 128 attempts per round instead of 48 (`--k_hard 128`), in all
three seed runs and in the ablation, applied by restarting each run with `--resume` (the round in
flight at the restart is redone). Nothing else changes.

## Amendment 4 (2026-11-10, 1:40 am PT; applies to the kept runs from about round 14 on)

Observed after 12 to 13 rounds: every seed has a kept proof for all but 14 to 53 of the 4,022
training theorems, and in all three the 14-line family is where they stop (1 of 11 solved).
Dev in plain mode is still rising (seed 0: 79.1 / 91.1, seed 1: 81.1 / 92.6, seed 2: 76.2 / 89.6).

Change, by restarting each run with `--resume` on the current code (commit of this amendment):
- theorems without a proof get 512 attempts per round (`--k_hard 512`; they are few, so this is cheap);
- in the three seed runs a refused step may become a subgoal for the same policy, up to 4 times per
  attempt (`--subgoals 4`, the procedure described in `fr/sampler.py`); the ablation has no step
  check and so no subgoals, but gets the same 512 attempts.
The current code also caps a sampled action's length by its state, which only shortens runaway samples.

## Amendment 5 (2026-11-10, 5:00 am PT; before any final checkpoint is packed or scored)

This amendment fixes the final verifier and how the solver settings will be frozen. It was made
from mid-run checkpoints of seed 1 (policy after 6 and after 18 rounds; verifiers trained on
seed 1's rounds 0 to 7 and read on its rounds 10 to 12, which they did not see). The kept RL runs
are not touched by it.

**Observed.** The one-logit verifier of amendment 2 lets near misses through: in full mode on dev,
112 of 2,207 attempts it called verified were wrong (seed 1, 6 rounds, 89.6 / 95.3 pass@1 / pass@5).
Tried, on the same policy:

| verifier | refused steps of unseen rounds it would pass | accepted steps kept | valid dev steps kept | dev full mode |
|---|---|---|---|---|
| A. one logit per step, 4 layers d 256 (amendment 2), threshold 0.2 | 3.9% | 99.9% | 99.2% | 89.6 / 95.3 (at 0.5) |
| B. A at 6 layers d 384 with three rounds of mining its own mistakes | 2.2 to 2.8% | 97.0 to 99.7% | not read | not run |
| C. B's size, plus one-symbol edits of accepted steps labelled by the training-time checker, threshold 0.2 | 2.1% | 99.9% | 99.4% | 92.5 / 96.0 (at 0.15) |
| D. slot model: writes each formula of a step from the rest of the step | 0.03% (at 0.01) | 99.9% | 97.9% | 92.6 / 95.1 |
| E. masked slot model: the same, seeing the formula with part of it hidden | 3.2% (at 0.01) | 99.9% | 98.9% | not run alone |
| **E at 0.001 and A at 0.05, both must pass** | **0.01%** | **99.9%** | **99.1%** | see below |

With the 18-round policy: C 95.3 / 97.3 (20 of 2,239 verified attempts wrong), D 94.9 / 96.8 (0 of
2,166 wrong), E and A together at thresholds 0.003 and 0.05: 95.4 / 96.8 (0 of 2,217 wrong); plain
mode for that policy is 81.8 / 92.6. D's failure is over-refusal: where a rule leaves part of a
formula free it has habits about that part, and on unseen theorems it refuses valid steps (it
blocked whole theorem families on dev). E's failure is the opposite: with part of the formula in
view it explains away gross mismatches. A catches exactly those and misses near misses, which E
catches, so the two together pass 8 of 79,298 refused steps.

**Final verifier (all seeds):** two networks, and a step passes only if both pass.
- E, the masked slot model (`fr/verifier.py`, `--kind mslot`): 6 layers, d 384; 8,000 steps;
  trained only on steps the training-time checker accepted in that seed's own RL rounds (all
  rounds) and on the steps of x proofs. Threshold 0.001 on the lowest probability it gives to any
  hidden symbol.
- A, the one-logit judge (`--kind disc`): 4 layers, d 256; 6,000 steps; that seed's refused and
  accepted steps (at most 150,000 per round) and 400,000 steps of x proofs as positives.
  Threshold 0.05.
- Not used: checker-labelled edits (C), mining (B), the unmasked slot model (D), several name
  views. No refused step, edited step or checker output enters E. At test time nothing but the two
  networks screens a step: no Lean, no type check. The slot construction itself is mine: which
  lines a step cites is read off its text, and one formula at a time is hidden; what the hidden
  symbols must be is the network's judgement.

**Solver settings.** Fixed now: threshold as above, 4 resamples per step, retry temperature rising
by 0.15 per retry, 2 box abandonments per try, premise lines written by the solver in full mode
(orchestrator's ruling: zero-choice, a function of the statement alone, counted separately, zero
generated tokens), one unscreened last try whose proof is handed in with verdict "no".
Still to be chosen, on dev with seed 0's policy and seed 0's own two verifier networks (the first
scored checkpoint available when the tuning runs, about round 30), then frozen for all seeds and
written into `/scratch/cowifornia/nd-rl/fable_rl/final/solver.json` before any final checkpoint
is packed:
- tries per attempt, 4 or 8 (on seed 1: 95.4 / 96.8 at 425 tokens against 95.9 / 97.0 at 558);
- the subgoal procedure in the solver, off or on (4 per attempt);
- plain-mode temperature, 1.0 or 0.8;
- worker processes sharing the GPU, 1 or 3 (speed only).
Rule: the setting with the higher dev pass@1; a difference under 0.5 points goes to the cheaper
setting (fewer generated tokens); a setting that cannot do 832 theorems at 32 attempts in about
15 minutes on one GPU is not taken. Full mode stays the headline; plain mode is reported next to it.

**Unchanged:** RL budget (11 h per run), the checkpoint rule (highest dev plain pass@1 among the
scored rounds, ties to the later), one final test per seed by the orchestrator.

**What I now expect** (replacing the original guesses where they differ): dev full mode 94 to 97%
pass@1 and 96 to 98% pass@5 for every seed; tokens per attempt 400 to 600 in full mode and about
190 in plain mode. On the sealed textbook set, full mode 75 to 90% pass@1. On the long sets I still
have no measurement: dev at 13 and 14 lines (3 theorems each) is 47% and 33% pass@1 in full mode
with seed 1's 18-round policy, and the dev theorems that stay unsolved are, with one exception I
found, ones the policy cannot prove, not ones the verifier blocks. Guess:
25 to 55% pass@1 at 13 to 20 lines and 2 to 20% at 21 to 30 lines, with seed 0 ahead of the
others there (it is the only seed whose training set's 14-line family opened).

### Amendment 5b (2026-11-10, 5:15 am PT; before any tuning run on seed 0 has finished)

Two things read after amendment 5 was committed, both on seed 1's 18-round policy:
- With E at threshold 0.001 and A at 0.05 (the final verifier), dev full mode is 96.1 / 97.3 at 419
  tokens per attempt, no verified attempt wrong.
- With 8 tries, the tries that produced the accepted proof were: first 2,123 of 2,365 attempts,
  second 60, third 24, fourth 13, fifth to eighth 14 together, and the unscreened last try 35 (of
  113 that reached it). Later tries add little, and they set the solver's running time: a profile
  shows the time goes to decoding rounds, whose number is set by the longest-running attempts.
  Measured on one A40 under a 22 GB memory cap, 473 theorems at 32 attempts: full mode with 4 tries
  469 s, plain mode 87 s; three worker processes did not help (459 s), so one process is used.

Added to the settings to be chosen on seed 0 under the same rule: 1, 2 or 3 tries per attempt
(next to 4 and 8). Worker processes: 1 (fixed now).

### Amendment 5c (2026-11-10, 8:35 am PT): solver settings frozen

Chosen as amendments 5 and 5b say, on dev with seed 0's round-30 policy and seed 0's own two
verifier networks (trained on its rounds 0 to 35), 5 attempts per theorem; times are for the 473
dev theorems at 32 attempts on one A40 under a 22 GB memory cap.

| setting | pass@1 | pass@5 | tokens per attempt | time at 32 attempts |
|---|---|---|---|---|
| full, 1 try | 91.0 | 95.3 | 285 | |
| full, 2 tries | 93.4 | 96.6 | 343 | 287 s |
| full, 3 tries | 93.4 | 96.2 | 391 | |
| **full, 4 tries** | **93.9** | **96.8** | **444** | **409 s** |
| full, 4 tries + subgoals | 94.5 | 97.5 | 590 | 552 s |
| full, 8 tries | 94.8 | 97.0 | 564 | 573 s |
| full, 8 tries + subgoals | 94.7 | 97.5 | 701 | |
| plain, temperature 1.0 | 81.8 | 93.7 | 186 | |
| **plain, temperature 0.8** | **83.3** | **92.4** | **186** | **62 s** |

- 8 tries and 4 tries with subgoals have the higher pass@1 but need 552 to 573 s on 473 theorems,
  which is about 16 to 17 minutes on 832 if time grows in proportion; the rule excludes what
  cannot do 832 theorems in about 15 minutes. Of the rest, 4 tries is 0.5 points above 2 and 3
  tries, so 4 tries it is (about 12 minutes on 832 by the same proportion).
- Plain mode: temperature 0.8 (1.5 points more pass@1, 1.3 fewer pass@5).
- Frozen for all seeds (`solver.json` in this folder, packed into every final checkpoint):
  full mode = both verifier networks must pass (masked slot model 0.001, one-logit judge 0.05),
  4 resamples per step, 4 tries, temperature 1.0 rising by 0.15 per resample, 2 box abandonments
  per try, at most 300 sampled actions per attempt, premise lines written by the solver, one
  unscreened last try, no subgoal procedure; plain mode = temperature 0.8, nothing else.
- Added to the solver after amendment 5 and not part of any choice above: a wall-clock budget per
  call (default 0.06 s per attempt, at least 900 s), periodic writing of finished proofs to the
  output file, failed attempts for unreadable statements, and a retry on smaller parts after an
  error. Tried and left off: resampling from the state's cached encoding (same dev result, slower).
