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
