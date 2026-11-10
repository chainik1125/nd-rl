---
written_on: 2026-11-09
written_by: agent:claude
---

# fable_rl pipeline: a one-day RL pipeline for natural-deduction proving

**Task** (`/scratch/cowifornia/nd-rl/fable_rl/BRIEF.md`, 2026-11-09): pretrain a prover on x
(Dan's full corpus, 155,000 proofs of 2 to 12 lines), improve it with RL on the 4,495 y statements,
hand it back for tests on held-out theorems, a sealed textbook set and two sets of long generated
theorems (13 to 20 and 21 to 30 lines). Design free; proofs only from x or from the model's own
checker-accepted output; statements only from y and what the model's attempts on y produce.

**Where things are.** Results: [`RESULTS.md`](RESULTS.md). Plan and its amendments, each written
before the runs it applies to: [`PREREG.md`](PREREG.md). Summary with figures:
`experiment-summaries/2026-11-10-fable-rl-pipeline/`. Branch `leon/fable-rl-pipeline` (from
`leon/knockout-grpo`). Large files: `/scratch/cowifornia/nd-rl/fable_rl/` on serrano (`data/`,
`runs/`, `final/`, `pods.tsv`, `STATUS.md`).

## The pipeline

1. **Format and environment.** Dan's step environment and `lean_staten` tokens, vendored unchanged
   (`vendor_dan/`). The policy sees the Lean tactic state and writes one proof step.
2. **Policy.** A from-scratch GPT, 8 layers, d 384, 14.3M parameters. Pretrained on x for 6,000
   steps (`fr/train.py`).
3. **RL on y** (`fr/rl.py`): expert iteration with **the checker inside the sampling loop**. While
   the policy writes an attempt, each step is type-checked on its own (Dan's reject-only
   pre-filter applied to that one step). A refused step is not applied; it is sampled again, up to
   5 times at a rising temperature; a stuck attempt may abandon its innermost box; a refused step's
   formula may become a subgoal for the same policy. Lean 4 judges every proof that is kept. The
   shortest proof per theorem is kept across rounds. Failed attempts are reused by **hindsight**
   (the attempt is read as a proof of a lemma it did derive) and finished proofs that end by
   closing a box are also read as proofs of the box's **subgoal** (premises plus the assumption
   entail the box's goal). Each round trains 600 steps on a mixture: x 37%, best proofs 33%, fresh
   proofs 12%, hindsight lemmas 9%, subgoal variants 9%. The policy network is trained on policy
   targets only.
4. **Verifier** (`fr/verifier.py`): two small networks, separate from the policy, trained on the
   steps the checker refused or accepted during RL and on the steps of x proofs. One is a
   one-logit judge of a step. The other is a masked "slot" model trained on accepted steps only:
   it is shown the step with part of one formula hidden and writes that formula out; a step is
   doubted when the model gives a low probability to a hidden symbol. A step passes only if both
   networks pass it.
5. **Solver** (`solve.py`). `SOLVER_MODE=full` (default, headline): inside one attempt the verifier
   takes the checker's place. A step the verifier doubts is resampled; a stuck box is abandoned; a
   try that fails is restarted; if every screened try fails, one unscreened try hands in its proof
   with verdict "no". The premise lines (line k restates premise k) are written by the solver, not
   sampled. No call to Lean or to any type checker. `SOLVER_MODE=plain`: one rollout per attempt,
   no verifier, no resampling, nothing written by the solver.

## Files

| file | what |
|---|---|
| `vendor_dan/` | Dan's modules at 13e41bbd, unchanged (`PROVENANCE.md`, `SHA256SUMS`) |
| `fr/common.py` | tokenizer, per-step type check (training only), Lean runner, trace analysis (dependency marks, hindsight lemmas, subgoal variants) |
| `fr/model.py` | the network; loads its own checkpoints |
| `fr/sampler.py` | batched KV-cache decoding of one action per state; the attempt loop with gating, resampling, backtracking, subgoals, restarts |
| `fr/data.py` | example stores, the x loader (1,500 proofs held out), batches, name-offset augmentation |
| `fr/train.py` | losses and stage 1 (pretraining on x) |
| `fr/collect.py` | one round's attempts in worker processes; candidates from finished and failed attempts |
| `fr/rl.py` | stage 2 (RL on y train) |
| `fr/verifier.py` | the verifier networks: inputs, training, use; also the variants that were tried and dropped |
| `fr/solvelib.py`, `solve.py` | the solver and its scoring |
| `pack.py` | policy + verifier networks + frozen solver settings into one checkpoint file |
| `report.py`, `final_tables.py`, `figures.py` | tables from the runs' `rounds.jsonl`; the final dev tables; the figures |
| `pod/` | pod setup, code and data push (x and the stripped y splits only), job launcher, pull, supervisor (`watch.sh`), per-seed finalisation (`finalize.sh`) |
| `tests/` | CPU checks against x and Lean |

## How to run

```bash
PY=/scratch/cowifornia/nd-rl/think/nd-rl/code/.venv/bin/python
# stage 1 and stage 2 (one A40 each; paths as on the pods)
$PY -m fr.train --x /workspace/data/train_k12.jsonl.gz --out /workspace/runs/pre_s0 --seed 0 --steps 6000
$PY -m fr.rl --init /workspace/runs/pre_s0/pre.pt --out /workspace/runs/rl_s0 --seed 0 --hours 11 \
    --aux_attempts 12000 --k_hard 512 --subgoals 4
# the two verifier networks, from the run's saved step data
$PY -m fr.verifier --kind disc  --run RUN --out RUN/verifier_disc.pt  --x X --steps 6000 --max_per_round 150000 --x_pos 400000
$PY -m fr.verifier --kind mslot --run RUN --out RUN/verifier_mslot.pt --x X --steps 8000 --n_layer 6 --d 384 --x_pos 1200000
# one file for the solver
$PY pack.py --policy RUN/r<N>.pt --verifier RUN/verifier_mslot.pt:0.001,RUN/verifier_disc.pt:0.05 \
    --out final.pt --solver "$(cat solver.json)"
# solve (one 24 GB GPU; the lab environment is enough; everything it imports is in this folder)
$PY solve.py --ckpt final.pt --in theorems.jsonl --out rows.jsonl --k 32 --seed 0                  # full mode
SOLVER_MODE=plain $PY solve.py --ckpt final.pt --in theorems.jsonl --out rows.jsonl --k 32 --seed 0
```

Output rows: `name`, `draw`, `proof` (ND text in Dan's format, the body after `PRF`), `verdict`
(the solver's own yes/no), `tokens_generated` (every token the policy generated in the attempt,
refused samples and restarts included), and extras: `tries`, `steps`, `scaffold_lines` (premise
lines written by the solver, zero generated tokens), `fallback`, `verifier_steps` and
`verifier_step_tokens` (steps the verifier networks read; they generate nothing), `confidence`.
