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
