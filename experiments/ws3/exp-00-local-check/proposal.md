# exp-00 local check: proposal

Pre-registration (plan section 6, gate A). Written and committed before the run starts.
Spec: `docs/ws3/startup.md` section 2, row **exp-00 local check**. Executor: one Claude Code
session on branch `charles/exp-00-local-check`. Laptop CPU only (Apple M1 Pro, 8 cores, 16 GB).
No pod, no GPU, no Tinker, no spend.

## Hypothesis

The whole chain is wired before a pod is paid for: checkpoint loader, tokenizer, HF wrapper,
binary reward through the strict verifier, TRL `GRPOTrainer`, evaluator, and the review files.
Concretely, the frozen checkpoint reproduces the expected pre-training profile, and 50 GRPO steps
on in-distribution prompts move mean reward up without the KL blowing up.

## Config

`config.yaml` in this folder; launch command in `launch.sh`
(`uv run python ws3/run.py --config experiments/ws3/exp-00-local-check/config.yaml`).
Fixed components unchanged: checkpoint `fork_pin/ckpt/pretrain.pt` (sha256 `f4409e4f…`), fork
code at `ab629c7d`, tokenizer `abs`, strict verifier. Seed 0 everywhere.

| part | what | size |
| --- | --- | --- |
| (a) T1 gate | teacher-forced next-token accuracy and loss on `heldout.jsonl` | all 5,000 records |
| (b) greedy by length | greedy solve rate by length through the strict verifier | 1,000-record stride of heldout (200 per length 2 to 6) |
| (c) textbook baseline | the 72 problems, temperature 1.0 | 64 samples each (4,608 completions) |
| (d) long-pool baseline | `rl_targets.jsonl` stride, pass@32, temperature 1.0 | 200 prompts (6,400 completions) |
| (e) reduced GRPO | TRL GRPOTrainer, lengths 3 to 5 from `train.jsonl`, lr 1e-5, KL 0.01, clip 0.2, temperature 1.0, cap 400 tokens | 50 steps x 32 prompts x group 8 = 12,800 completions; periodic greedy eval every 10 steps |
| evaluate | (b), (c), (d) again on the trained model at equal attempts (same k, temperature, seed) | as above |

Prompts for (e) come only from `train.jsonl`; `data.assert_no_leak` checks the prompt set
against heldout, transfer, validation_36 and the 72 textbook problems by renaming class before
training starts. Nothing trains or tunes on heldout, transfer, validation_36, the test files, or
the 72 problems.

## Expected result (numeric, from the startup.md row and the 2026-09-22 smoke numbers)

- (a) next-token accuracy about 95 percent or higher (smoke on 1,000 records gave 97.8 percent).
  Gate threshold in the config: 0.95. Stop-and-ask threshold from the brief: below 90 percent.
- (b) near 100 percent at lengths 2 to 4, lower at 5 to 6 (a 200-record bench gave
  1.00 / 0.97 / 0.93 / 0.88 / 0.60 for lengths 2 to 6).
- (c) a small nonzero count of the 72 at k=64; 0 to 1 of the 43 at reference length 7 or more
  (the previous project best). 0 to 1 solves is noise, not a result.
- (d) small nonzero pass@32 counts at true lengths 7 to 8, near zero above.
- (e) mean reward rises within 50 steps (the frozen policy at lengths 3 to 5 should start
  around 0.85 to 0.95 at temperature 1.0, so the rise is toward 1.0); KL to the reference under
  1; within-group reward std nonzero for most steps.

## What we conclude either way

- All parts as expected: the pod can be launched (exp-01) and every later run reuses this code
  unchanged; this `results.md` is the reference the pod must match.
- (a) fails (under 90 percent): stop and ask Charles; the checkpoint may be undertrained.
- (e) flat or reward falls with a KL spike: a wiring bug or a bad default; fix before spending
  money. Not a result about RL.
- (c) or (d) surprisingly good (more than a handful of solves at length 7 or more): check for a
  reward bug and a leak before believing it (CLAUDE.md).

## Budget and stop rule

One hour of laptop CPU in total, no GPU. Measured throughput with the KV cache: about 50
greedy or 100 sampled completions per second, so the plan is roughly (a) 1 min, (b) 1 min,
(c) 2 min, (d) 3 min, (e) 15 to 25 min, evaluate 6 min, report under 1 min.
Stop rule: if any part is on track to push the total past one hour, shrink its n (or the step
count) and record the change in `numbers.md`; do not wait. The run is quarantined if any
fixed component or evaluation file is touched by training code.
