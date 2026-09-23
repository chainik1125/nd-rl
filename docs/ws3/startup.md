# Workstream 3: startup sequence and first experiments

Written 2026-09-22. Companion to `plan.md` (task sequence, gates, monitoring, automation loop).
This file answers two questions: what has to happen, in what order, before the automated loop
can run its first experiment, and which small experiments that loop should run first.

Steps 1 to 6 need Charles. Steps 7 to 11 are agent work that can happen on the laptop before any
GPU or checkpoint exists. Steps 12 onward need the pod.

## 1. Steps to get the loop running

### Human setup

1. **Accounts and keys.** Fund RunPod. Create the Hugging Face bucket and a fine-grained write
   token. Write `.env` at the repo root (already gitignored)
   and commit a `.env.example` with names only:

   | variable | needed for |
   | --- | --- |
   | `RUNPOD_API_KEY` | pods, watchdog, kill switch |
   | `HF_TOKEN` | checkpoint and eval-dump upload at `DONE`; passed to pods over stdin |
   | `TINKER_API_KEY` | T6 only |
   | `NTFY_TOPIC` | optional push notifications |
   | `DIGITALOCEAN_ACCESS_TOKEN` | only if a droplet is created |

2. **Get the checkpoint and its code.** Done on 2026-09-22: `code/ws3/fork_pin/` holds the
   pinned `model.py`, `tokenizer.py`, `nd_verify/`, the checkpoint's recipe and summary, and a
   `MANIFEST.json` with every sha256. Run `bash code/ws3/fork_pin/fetch.sh` once to pull the
   data files and `pretrain.pt` (gitignored) and verify them. Nothing is needed from Dmitry.
3. **Targets.** `fetch.sh` also brings `validation_36*.jsonl`, the two test prompt files,
   `heldout.jsonl`, `rl_targets.jsonl`, and `transfer.jsonl` into `fork_pin/data/`. The 72
   textbook problems come from the `origin/dmitry/cap-comparison-textbook-eval-data` branch
   (step 5). Test files are scored once, at the end, aggregate only.
4. **Python environment.** A `uv` venv on Python 3.11 or newer. Add TRL, transformers,
   accelerate, datasets, and huggingface_hub to `code/pyproject.toml`, then lock.
5. **Branches.** Create `charles/main` from the current `charles` branch. Commit `CLAUDE.md`,
   `docs/ws3/plan.md`, this file, and an empty `docs/ws3/log.md` and `docs/ws3/questions.md`.
6. **Decide the driver host.** Laptop kept awake with `caffeinate`, or a user on one of Dan's
   two droplets. Log in to Claude there once; credential files are never copied between machines.

### Agent work, no GPU needed

7. **Evaluator module** (plan T3). Done 2026-09-22: `code/ws3/reward.py` (binary reward through
   `strict_verify.py`, TRL-callable, proof dump) and `code/ws3/evaluator.py` (pass@k by length
   with Wilson CIs, textbook slices, T1 next-token accuracy). Tests:
   `code/tests/test_ws3_reward.py` covers the six required cases plus the forward-reference hole
   and TRL's `completion_ids` convention.
8. **HF wrapper** (plan T2). Done 2026-09-22: `code/ws3/hf_wrapper.py`, a custom
   `PreTrainedModel` around the fork's `GPT` plus a fast HF tokenizer over the 99-token `abs`
   vocab. `code/tests/test_ws3_hf_wrapper.py` checks logits to 1e-4 on 100 real heldout sequences
   (with and without left padding), exact tokenizer round-trips, and save/load through the Auto
   registry (what TRL does for the reference model). 14 tests pass.
9. **Training scripts.** Done 2026-09-22: `code/ws3/train_grpo.py` (TRL `GRPOTrainer`) and
   `code/ws3/train_rloo.py` (TRL `RLOOTrainer`, the PPO-family alternative; TRL 1.13 ships no
   value-head PPO). One YAML per run in `code/ws3/configs/`; every run writes `metrics.jsonl`
   (plan section 4 set plus throughput), `completions.jsonl`, `prompts.jsonl`, eval dumps, and a
   `run_manifest.json`. `configs/smoke_cpu.yaml` runs end to end on the laptop in about 15 s.
   Still to write: a `report` script that reads `baseline.json` and every `eval/final.json` and
   draws the three plan graphs. See `code/ws3/README.md`.
10. **Orchestration.** Copy `runrole`, `gate0` to `gate2`, `review_workspace`, `pod_watchdog`,
    and `notify` from `origin/dan_orchestration:code/tools/orchestration/` into
    `code/tools/orchestration/`. Repoint paths at `experiments/ws3/<name>/`. Rename the gate
    checks to the A, B, C definitions in plan section 6. Write two role briefs in
    `docs/ws3/roles/executor.md` and `docs/ws3/roles/reviewer.md`.
11. **CPU smoke test.** Done 2026-09-22 for the training half: `configs/smoke_cpu.yaml` through
    both `train_grpo.py` and `train_rloo.py` produces every output file, the reference model
    reloads from `hf_base/`, and the first evaluator numbers on the real checkpoint are in:
    next-token accuracy 97.8 percent on 1,000 heldout records (T1 gate passes), textbook 5 of 72
    at k=2 with 0 of 43 at reference length 7 or more. Exp-00 proper (the full five parts and a
    reviewer session) is the next thing to run.

### GPU

12. **Pod.** One RunPod GPU (A40 or L4 class is enough for a 3M-parameter model). Install the
    env, rerun the smoke test on CUDA, record throughput.
13. **T1 checkpoint.** Load `pretrain.pt` with the pinned fork and compute the numbers the
    export did not ship: next-token accuracy and loss on the fork's `heldout.jsonl`, greedy solve
    rate by length. Gate: about 95 percent next-token accuracy. The export trained for only
    300 seconds (last loss 0.070), so treat a failure as a real stop-and-ask, not a bug.
14. **Baseline** (`baseline.json`): greedy pass rate by length 2 to 10, pass at 32 by length at
    temperature 1.0, `validation_36` solves. This is the frozen control for every later claim and
    already gives graph 1 with no RL.
15. **First proposal, first loop.** Write `proposal.md` for exp-01 below, start the executor,
    let the orchestrator hand off to the reviewer, write `results.md` from `review.md`. Exp-00
    has already run on the laptop by this point (it replaces the CPU smoke test in step 11 once
    the checkpoint and evaluator exist) and its `results.md` is the reference the pod must match.

## 2. Proof-of-concept experiments, in order

Every row follows the `proposal.md` shape: hypothesis, config, numeric expectation, what we
conclude either way. Exp-00 runs on the laptop before any pod exists. Exp-01 and exp-02 must run.
If exp-02 shows nothing at length 7 on any seed, exp-03 is next, and exp-04 onward wait until
there is signal to sweep around.

| id | where | question | design | expected | conclude either way |
| --- | --- | --- | --- | --- | --- |
| **exp-00 local check** | laptop CPU, under one hour | Is the whole chain wired before a pod is paid for: loader, tokenizer, HF wrapper, reward, TRL GRPO, evaluator, review files? | Five parts, all on CPU. (a) T1 gate: teacher-forced next-token accuracy and loss on the 5,000 heldout records. (b) Greedy solve rate by length on a 1,000-record stride of heldout through the strict verifier. (c) Textbook baseline: the 72 problems at 64 samples, temperature 1.0. (d) Reduced long-pool baseline: 200 `rl_targets` prompts at pass at 32. (e) Reduced GRPO: 50 steps, 32 prompts x group 8 at lengths 3 to 5, lr 1e-5, KL 0.01, temperature 1.0, writing `metrics.jsonl` and a proof dump. Then a reviewer session recounts (b) to (e) from the dumps. | (a) about 95 percent or higher. (b) near 100 percent at lengths 2 to 4, lower at 5 to 6. (c) and (d) small nonzero counts at lengths 7 to 8. (e) mean reward rises within 50 steps, KL under 1. | All pass: the pod can be launched and every later run reuses this code unchanged. (a) fails: stop and ask; the checkpoint may be undertrained. (e) flat: a wiring bug, fix before spending money. Budget: about 40 minutes of CPU, none of it GPU. |
| **exp-01 plumbing on GPU** | pod | Does TRL GRPO move reward at full size, and what is the throughput? | Prompts at lengths 3 to 5 only, in distribution. 100 steps, 64 prompts x group 8, lr 1e-5, KL 0.01, temperature 1.0, one seed. Record completions per second. | Mean reward rises from the greedy baseline toward 1 within 100 steps. KL stays under about 1. Throughput 50 or more completions per second. | Rises: the GPU pipeline matches the laptop one. Under 30 completions per second: consider the plan's sampler fallback before the sweep. |
| **exp-02 frontier** | pod | Does RL lift pass rate past the cap? | Prompt mix at lengths 5 to 8, weighted so each length has nonzero pass at 32 in the baseline. Same config as exp-01, 500 steps, one seed. | Lift at 5 and 6, some lift at 7, little at 8. | This is plan T4's gate and the first real version of graph 1. |
| **exp-03 prompt mix** | pod | Fixed mix vs pass-rate-adaptive mix | As exp-02, but sample lengths in proportion to within-group reward variance, re-measured every 50 steps. | Adaptive mix keeps group variance nonzero longer and reaches a higher pass rate at 7. | Decides how prompts are drawn for the sweep. |
| **exp-04 sweep** | pod, overnight | Which knob matters? | One variable at a time around the best of exp-02 or exp-03: lr 3e-6 to 3e-5, KL 0 to 0.05, group 8 or 16, temperature 0.8 to 1.2. About 12 runs at 200 steps. | Learning rate dominates. KL 0 collapses entropy. | Picks the two configs that get three seeds each (plan T5). |
| **exp-05 GRPO vs expert iteration** | pod | Is on-policy RL better than rejection-sampling fine-tuning? | Same prompt mix and attempt budget. EI uses TRL's SFT trainer on verified proofs from pass at 32, three rounds. | Similar lift at 5 to 6; a difference at 7 to 8 in one direction. Robbie's EI did not move his test numbers. | Answers the "which algorithm" half of the library recommendation. |
| **exp-06 loss variants** | pod | Do TRL's cheap switches help? | Best config with the DAPO-style and Dr. GRPO loss options TRL exposes, and no KL term. | Small or no difference. | Whether the recommendation needs a loss-variant caveat. |
| **exp-07 Tinker** | Tinker, one day | Reference number | Same reward as a custom environment, small open-weight model. | Higher absolute numbers that say nothing about emergence. | Stated with the caveat in every write-up. Optional; if skipped, the summary says "not attempted". |

Common to all rows: binary reward through the strict verifier, generation cap well above the
longest expected proof, the frozen-checkpoint control at equal attempts, and the reviewer recount
from plan section 6 before any number is reported.

## 3. Open questions for Charles

Also to be mirrored in `docs/ws3/questions.md` once it exists.

- Which host runs the driver: laptop, or a user on one of Dan's droplets?
- Decided 2026-09-22: the fixed checkpoint is Robbie's `share_pretrain_4l_seed0/pretrain.pt`
  with Dan's fork at the pinned commit as its code (plan section 1a). `CLAUDE.md` still says
  "take-home defaults" for the fixed components; that wording should be updated to name this
  lineage so agents do not reach for the vendored Dmitry code.
- Decided 2026-09-22: the textbook test set is the 72 problems on
  `origin/dmitry/cap-comparison-textbook-eval-data` (plan section 1a). Merge that branch into
  `charles/main` in step 5. `validation_36` is reported alongside, with the overlap stated.
