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

**Status:** in progress. Results are in [`RESULTS.md`](RESULTS.md) once the runs end;
[`PREREG.md`](PREREG.md) was written before the first RL run.

Branch `leon/fable-rl-pipeline` (from `leon/knockout-grpo`). Large files:
`/scratch/cowifornia/nd-rl/fable_rl/` on serrano (`data/`, `runs/`, `pods.tsv`, `STATUS.md`).

## The pipeline in one paragraph

Dan's step environment and `lean_staten` format are kept: the policy sees the Lean tactic state
and writes one proof step. The network (8 layers, d 384, 14.3M parameters) has, besides the
policy head, a **verifier** scalar that says whether a step it has just written is well typed and
a value scalar. RL on y is expert iteration with **the checker inside the sampling loop at
training time**: a step with a certain type error is refused on the spot and resampled, so
attempts cannot end in the ill-typed shortcuts the pretrained policy writes on hard theorems, and
each refusal is a labelled negative for the verifier. Lean 4 judges every proof that is kept.
Failed attempts are reused by **hindsight**: the attempt is read as a proof of the lemma it did
derive. At test time **the model's own verifier takes the checker's place**: inside one attempt
the solver resamples steps the verifier doubts, abandons a box it is stuck in, and restarts a try
that fails, with no call to Lean or to any type checker.

## Files

| file | what |
|---|---|
| `vendor_dan/` | Dan's modules at 13e41bbd, unchanged (`PROVENANCE.md`, `SHA256SUMS`) |
| `fr/common.py` | tokenizer, per-step type check (training only), Lean runner, trace analysis (dependency marks, hindsight lemmas) |
| `fr/model.py` | the network; loads its own checkpoints |
| `fr/sampler.py` | batched KV-cache decoding of one action per state; the attempt loop with gating, resampling, backtracking, restarts |
| `fr/data.py` | example stores, the x loader (1,500 proofs held out), batches, name-offset augmentation |
| `fr/train.py` | losses and stage 1 (pretraining on x) |
| `fr/rl.py` | stage 2 (RL on y train) |
| `fr/solvelib.py`, `solve.py` | the solver and its scoring |
| `pod/` | pod setup, code and data push (x and the stripped y splits only), job launcher, pull |
| `tests/test_common.py` | CPU checks against x and Lean |

## How to run

```bash
PY=/scratch/cowifornia/nd-rl/think/nd-rl/code/.venv/bin/python
# stage 1 and stage 2 (one A40 each; paths as on the pods)
$PY -m fr.train --x /workspace/data/train_k12.jsonl.gz --out /workspace/runs/pre_s0 --seed 0 --steps 6000
$PY -m fr.rl --init /workspace/runs/pre_s0/pre.pt --out /workspace/runs/rl_s0 --seed 0 --hours 9
# solve (one 24 GB GPU; the lab environment is enough)
$PY solve.py --ckpt C.pt --in theorems.jsonl --out rows.jsonl --k 5 --seed 0            # default mode
$PY solve.py --ckpt C.pt --in theorems.jsonl --out rows.jsonl --k 5 --seed 0 --mode plain
```
