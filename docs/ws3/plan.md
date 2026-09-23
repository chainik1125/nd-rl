# Workstream 3 plan: off-the-shelf RL pipeline

Audience: Claude Code agents working for Charles. Standing rules (what is fixed, git, experiment folders, when to stop and ask) live in `experiments/ws3/CLAUDE.md` and are not repeated here. This file covers only what Workstream 3 has to do.

Week: Sep 20 to Sep 26, 2026. Results are presented at the Saturday group meeting.

## 1. Scope

Question to answer by Saturday:

> With dataset and pre-training held fixed at the take-home defaults, what is the best existing RL library we can plug in, how much lift does it give over pre-training alone, and how much work was it to set up?

In scope: choosing and running an existing RL library on the project's pre-trained checkpoint, the reward function and evaluator, RL hyperparameters, RL monitoring, and a one-day Tinker comparison.

Out of scope: dataset design, tokenizer changes, pre-training recipe or architecture changes, and writing an RL algorithm from scratch. If one of these seems to block progress, stop and ask Charles.

The deliverable is two or three graphs and a clear yes/no recommendation, not clean infrastructure.

## 1a. Where the fixed components live

**Decision 2026-09-22 (Charles): the fixed checkpoint is Robbie's pre-training-only export,
not Dmitry's cap-6.** Everything "fixed" in `CLAUDE.md` therefore means the lineage of that
checkpoint, which is Dan's fork of the take-home, not the vendored Dmitry code described further
down. Dmitry's vendored code stays in the repo as the source of the strict verifier and the
metrics helpers only.

### The checkpoint and its lineage

Everything in this table is vendored and hash-pinned in **`code/ws3/fork_pin/`** (code in git;
data and checkpoint fetched by its `fetch.sh` and verified against `MANIFEST.json`). Put that
directory on `PYTHONPATH`; do not put the live fork on it. Only the files listed there are taken
from the fork.

| Component | Where | Notes |
| --- | --- | --- |
| Checkpoint | `fork_pin/ckpt/pretrain.pt`, fetched from `hf://robbiethompson2018/nd-rl-checkpoints/runs/share_pretrain_4l_seed0/pretrain.pt` (12.9 MB, sha256 `f4409e4f…`) | Raw `torch.save` dict: `state`, `tok_mode='abs'`, `stage='pretrain'`, `seed=0`, source hashes. Weights before any RL. **No held-out evaluation was run for this export**; `fork_pin/checkpoint_source/summary.json` records 2999 steps, 4.95 epochs, last training loss 0.070 on an A40 in 300 s. Load with `fork_pin/load_pretrain.py`. |
| Code pin | `github.com/dan-pandori/nd-takehome` at `ab629c7d2cdf3dc50d471e302d14643ba0782749` (branch `dan_ladder_a`, 2026-09-18) | Also pinned in the run's `source/remote.sh`. Only `model.py`, `tokenizer.py`, `nd_verify/`, and the data files below are vendored. |
| Tokenizer | `fork_pin/tokenizer.py`, `Tokenizer('abs')` | 99 tokens: `<pad>`, 19 symbols, 15 rule names, `N1` to `N64` absolute line indices. PAD=0, EOS is `QED`. Encodes spec text by whitespace split; the 72 textbook prompts all tokenize (longest 69 tokens). |
| Model | `fork_pin/model.py`, `GPT(vocab=99, n_layer=4, d=256, n_head=8, max_len=1024)` | 3,210,240 params. LayerNorm with biases, biased qkv/proj/fc, GELU, RoPE with **interleaved** pair rotation (`x[..., ::2]`, `x[..., 1::2]`), KV cache for sampling. Not the 4-head vendored model. |
| Pre-training recipe | `fork_pin/checkpoint_source/pretrain.py` (the agent-editable stage of Robbie's pretrain autoresearch harness) | AdamW lr 1e-3, batch 256, wd 0.1, betas (0.9, 0.95), warmup 200, cosine over a 300 s wall clock, grad clip 1.0, bf16, prompt-masked loss, random shift of all `N<k>` indices per record. |
| Pre-training data | `fork_pin/data/train.jsonl.gz` (154,990 records, lengths 2 to 6, ~31k per length) and `fork_pin/data/heldout.jsonl` (5,000, 1,000 per length) | Built by the fork's `make_splits.py` from its backward generator; disjoint by atom-renaming class (`gen.canon_key`); validation-36 classes removed. |
| RL prompt pools | `fork_pin/data/rl_targets.jsonl` (4,495) and `fork_pin/data/transfer.jsonl` (2,285) | Lengths are true minimal lengths 7 to 14 (`L_true`), mixed generator and textbook-schema instances; see the fork's `data/ladder/POOLS.md`. RL samples only from `rl_targets`; `transfer` is never sampled. |
| Upstream targets | `fork_pin/data/validation_36*.jsonl`, `fork_pin/data/test_*_prompts.jsonl` | Identical to `chainik1125/nd-takehome`. Test files: score once, aggregate only, never inspect. |
| Proof checker | `fork_pin/nd_verify/verify.py` (identical to upstream and to the vendored copy) plus Dmitry's `strict_verify.py` from `code/experiments/current/r1_cot/vendor/` | Always score through the strict wrapper. |
| Textbook test set | the 72 problems, below | Renaming-class overlap with the fork's train, heldout, `rl_targets`, and `transfer`: **0, 0, 0, 0** (checked 2026-09-22 with `gen.canon_key`). `validation_36` overlap with those four pools is also 0. |
| Evaluator metrics | vendored `nd_metrics.py` (Wilson CIs) and fork `eval_set.py` | Either is fine; report Wilson intervals. |

Consequences for the task sequence: T1 becomes "download and verify", T2 converts the fork's
`GPT` rather than the vendored `ProofModel`, and the pre-training baseline numbers in T3 are
computed by us because none were shipped with the export.

### Dmitry's vendored take-home (reference only)

The vendored code at `code/experiments/current/r1_cot/vendor/` (copied from
`simplex1:/data/users/dmitry/nd-rl/take-home` on 2026-09-09; hashes in `source_hashes.json`) is
**not** the lineage of the checkpoint above. It differs in tokenizer (40-token relative unary refs),
model (4 heads), recipe (batch 128, lr 3e-4), and generator (forward sampler). Use it for:
`strict_verify.py`, `nd_metrics.py`, `canonical_theorem` in `nd_data.py` (leakage audits), and
`nd_grpo.py` as a hyperparameter anchor (group 8, lr 1e-5, kl 0.01, clip 0.2, temp 0.8, 768 max
new tokens). Older June sprint code is at `code/experiments/archived/nd_sprints_20260609_11/`;
archive conventions are in `code/experiments/README.md` and `experiment-summaries/README.md`.

Other locations:

- **Held-out textbook problems: the 72-problem set.** Decided 2026-09-22. The textbook test set
  for ws3 is the 72 prompts on branch `origin/dmitry/cap-comparison-textbook-eval-data` under
  `code/experiments/archived/cap_comparison_20260910/evaluation/`:
  `textbook_dev.jsonl` (58 prompts with `reference_lines`; 43 at reference length 7 or more) and
  `textbook_train.jsonl` (14 prompts). Merge that branch into `charles/main` before T3. Rules:
  never train or tune on any of the 72 or on a renaming of one; report them with the frozen
  checkpoint as control; report the 58 and the 14 separately as well as combined, because the
  14 were RL training prompts in the cap-comparison sprint and the 58 were its development
  benchmark. State that history in every write-up. The 73 `old_final` problems from the same
  split are lost (see the branch's `IMPORT.md`).
  Supporting sets: the upstream assignment repo `github.com/chainik1125/nd-takehome` has
  `targets/validation_36.jsonl` (36 classic sequents; 18 of the 58 above are exact-prompt
  matches) and the two test prompt files (267 + 532; score once, aggregate only, never inspect);
  the vendored `nd_grpo.py` and `nd_data.py` expect those under the take-home root. The
  112-problem catalog and Smith set remain remote on simplex1 and are not used this week.
- **Checkpoint and pre-training data.** Not in this repo; downloaded from the HF repo and the
  pinned fork listed above. Dmitry's cap-6 checkpoint is no longer needed.
- **Python env.** No `.venv` yet. `code/pyproject.toml` pins torch but not TRL or transformers.

## 2. Software decisions

**Primary: Hugging Face TRL, `GRPOTrainer`.** It accepts any HF causal LM and a plain Python reward function, which fits a from-scratch checkpoint with a custom vocab. Use plain HF generation for rollouts first. The model is small, so one GPU is enough.

**Fallback if TRL rollouts are too slow or unstable:** verl or AI2 open-instruct (the Olmo RL stack). Only switch after recording in the notes what failed in TRL and the measured throughput.

**Side comparison, timeboxed to one day: Tinker** (`tinker-cookbook`, start from `recipes/rl_loop.py`). Tinker trains LoRA adapters on a fixed list of open-weight base models and cannot load our checkpoint. So this run uses a small open-weight model (about 1B) with the natural deduction task as a custom environment. It is a reference number only. The base model has seen logic in pre-training, so it says little about emergence. State this caveat in every write-up that mentions the Tinker number.

Supporting tools:
- `uv` for the Python environment. Pin versions in `pyproject.toml` or a lock file.
- File-based metrics, following Dmitry's pattern: the train script appends one JSON line per step to `metrics.jsonl` in the run folder, and `report` plots from it. No external logging service.
- RunPod for GPUs. One machine for this week.
- Hugging Face Hub bucket for our checkpoints. Never git. The fixed pre-trained model is read from Robbie's `robbiethompson2018/nd-rl-checkpoints`.
- `tmux` for long runs. Name sessions after the experiment branch.

## 3. Task sequence

Do these in order. Each task has a gate. Do not start the next task until the gate passes. If a gate fails twice, stop and ask Charles.

### T0. Orient
- Read the repo README, the docs folder, and the take-home code. Find: the dataset generator, the tokenizer, the pre-training script, the proof checker, the held-out textbook problems, and the existing conventions for archived and dead experiments.
- Write what you found, with paths, to `docs/ws3/notes.md`.
- Gate: every item above has a path, or is listed as missing with a question for Charles.

### T1. Baseline checkpoint
- Run `code/ws3/fork_pin/fetch.sh`, which downloads `pretrain.pt` and the pinned data and verifies every sha256 against `MANIFEST.json`. Load with `fork_pin/load_pretrain.py`. Do not retrain.
- Compute what the export did not ship: prompt-masked next-token accuracy and teacher-forced loss on the fork's `data/heldout.jsonl`, and greedy solve rate by length 2 to 6 through the strict verifier.
- Gate: next-token accuracy about 95% or higher. The export's last training loss was 0.070 after only 300 s, so this gate is a real check, not a formality. If it fails, stop and ask Charles; the fallback is a longer run of the same `pretrain.py`, which changes the fixed recipe and needs a decision.

### T2. HF conversion
- Wrap the fork's `GPT` as a custom `PreTrainedModel` (config plus a thin `forward` returning `CausalLMOutputWithPast`) rather than mapping onto `GPT2LMHeadModel`: the fork uses RoPE with interleaved pair rotation, LayerNorm with biases, and biased projections, which no stock HF class matches without weight permutation. Add a `PreTrainedTokenizerFast` over the 99-token `abs` vocab with whitespace pre-tokenization, PAD=0, EOS=`QED`.
- Gate: on 100 heldout sequences, logits from the HF wrapper match the fork model to within 1e-4, and tokenizer round-trips of prompt and proof text are exact. TRL's `GRPOTrainer` must accept the wrapper for generation and for the reference model.

### T3. Reward function and evaluator
- `reward(prompt, completion) -> float`: parse the completion, run the proof checker, return 1.0 for a valid proof of the stated goal and 0.0 otherwise. Keep it binary at first. Any shaping is a separate, labelled experiment.
- Write an evaluator that reports: pass rate by proof length, max length solved, and number of the 72 textbook problems solved (section 1a), broken down by `reference_lines` and by the 58/14 slices.
- Unit tests must cover: valid proofs, a valid proof of the wrong goal, truncated proofs, a correct proof followed by junk tokens, empty output, and unparseable output.
- Put them somewhere importable, not inside an experiment folder, so other tracks can score the same way.
- Gate: all tests pass, and the evaluator run on the pre-RL checkpoint gives the pre-training baseline numbers. Save these. Everything later is compared to them.

### T4. First end-to-end GRPO run
- Prompts: from the fork's `data/train.jsonl` prompts at lengths 5 to 6 (already trained on, so in distribution) and `data/ladder/rl_targets.jsonl` at `L_true` 7 to 8. Never from `heldout`, `transfer`, the 72 textbook problems, or `validation_36`.
- Start conservative: small learning rate, non-zero KL coefficient, group size 8 to 16, temperature 1.0, generation length capped well above the longest expected proof.
- Gate: a run completes, `metrics.jsonl` carries all metrics in section 4, and mean reward is not flat at zero. If reward is zero from the start, the prompts are too hard. Add easier lengths so there is signal.

### T5. Small sweep
- Sweep learning rate, KL coefficient, group size, and temperature. Keep it to roughly 10 to 20 runs. One variable at a time around the T4 config is fine.
- Pick the best two configs by validation pass rate on generated proofs, not by textbook problems.
- Rerun those two with 3 seeds each.

### T6. Tinker side run (one day maximum)
- Natural deduction as a custom environment using the same reward function. Small open-weight base model. Record setup time, cost, and the same evaluator metrics.
- If it is not working within the day, write up what blocked it and move on.

### T7. Final evaluation and write-up
- Run the evaluator on: pre-RL baseline, best TRL config (3 seeds), Tinker run.
- Dump every generated proof longer than length 7 that the checker accepted into a file for human review. These are often trivial instances. Do not claim them as results until reviewed.
- Produce the graphs in section 5 and write `docs/ws3/summary.md`.

## 4. Required monitoring

Log every one of these to `metrics.jsonl` for every RL run, one JSON line per step with a `utc` timestamp and the step number so gates A and B can be checked from the file. RL failures are hard to diagnose after the fact, so a run without them does not count.

- mean reward, and reward standard deviation within groups
- pass rate by proof length (periodic eval)
- KL to the reference model
- policy entropy
- fraction of completions that are unparseable or invalid
- completion length (mean and max), and fraction hitting the length cap
- gradient norm, learning rate
- rollout throughput (samples per second)

Common signatures:
- Entropy collapses early and reward plateaus: lower the learning rate or raise the KL coefficient or temperature.
- KL spikes, then reward crashes: learning rate too high.
- Reward standard deviation within groups near zero: no learning signal. Prompts are all too easy or all too hard.
- Many completions hit the length cap: raise the cap or check for a missing end-of-proof token.
- Reward rises but textbook solves do not: check for reward hacking. Read samples.

Read at least 20 raw samples from every run you report on. Put a few in the results file.

## 5. Deliverables for Saturday

1. Pass rate vs. proof length, before and after RL, with the pre-training length cutoff marked. Seeds as error bars.
2. Reward, KL, and entropy curves for the best config.
3. Textbook problems solved out of the 72 (section 1a), per seed, baseline vs. RL, with the 43 at reference length 7 or more shown separately. Treat 0 to 1 solves as noise. The previous best on this project is 0 to 1 of the 43 longer problems for cap-6 arms (`experiment-summaries/2026-09-10-proof-cap-comparison/reports/summary.md`); the "about 3 of 50" figure quoted earlier has no source in the repo.
4. `docs/ws3/summary.md`, one page: which library to standardize on, the lift over pre-training, setup effort for each option, the Tinker limitation and caveat, what failed, and the next three experiments you would run.
## 6. Automation loop

Borrowed from Dan's orchestration (`origin/dan_orchestration`, tools in `code/tools/orchestration/`,
role briefs in `docs/proposals/orchestration/`, design note
`docs/proposals/2026-09-16-automated-research-orchestration.md`) and cut down for one week, one
GPU, and one person reading the output. Dan's evidence from five runs: the executor's counts were
almost always right, and every run still needed an independent pass to fix wording, an inflated
count, or a selection-on-eval-set mistake. So the split here is two agent sessions per experiment
plus Charles, not four roles.

```
proposal.md → gate A → executor run → gate B → reviewer recount → gate C → results.md → log.md
```

### Roles

| role | who | reads | writes | must not |
| --- | --- | --- | --- | --- |
| executor | one Claude Code session per experiment, branch `charles/exp-<name>`, tmux named after the branch | `CLAUDE.md`, this plan, `proposal.md` | `config.yaml`, the launch command, `status.md`, `numbers.md`, raw eval dumps, `metrics.jsonl` | edit fixed components; touch the test files; write `results.md` conclusions |
| reviewer | a separate session, never the executor's | the raw artefacts and `proposal.md` only, with `numbers.md` and any write-up withheld until its recount is committed | `review.md` | edit the executor's files |
| librarian | Charles | `review.md` first, then the run | `results.md` conclusion, `docs/ws3/log.md`, `summary.md` | copy a number that is not in `review.md` |

Executor and reviewer sessions are started by a `runrole`-style driver copied from Dan's branch:
`claude -p` with the policy and role brief as the first prompt, `--continue` on interruption,
30 minute sleep on a real usage limit (detected from the result event, not by grepping text),
stop on an authentication failure or three consecutive failed invocations, a hard stop time, and
a per-role git author so `git log` separates who wrote what.

### Per-experiment files, `experiments/ws3/<exp-name>/`

- `proposal.md` (required by `CLAUDE.md`): hypothesis, config, **numeric expected result**, what we
  conclude either way, budget in GPU-hours, stop rule. This is the pre-registration.
- `config.yaml` and the exact launch command, committed before the run.
- `status.md`: dated lines; ends with `DONE <UTC>`, or `QUARANTINE <UTC> <reason>`.
- `numbers.md`: every number the executor reports, each with the file it came from.
- `review.md`: sections Recount, Claims (claim / independent value / verdict), Verdict.
- `results.md` (required by `CLAUDE.md`): graphs, 20 raw samples, conclusion written
  from `review.md`.

### Gates

- **Gate A, pre-registration.** `proposal.md` is committed with an expected number before the run
  starts, and its commit timestamp precedes the first line of `metrics.jsonl`. A run that fails this is
  labelled "not pre-registered" in `results.md` and cannot feed the Saturday graphs.
- **Gate B, execution clean.** `status.md` ends in `DONE`; the tmux session has exited; `metrics.jsonl`
  carries every section 4 metric; every file cited in `numbers.md` exists;
  `code/ws3/fork_pin/fetch.sh --verify` passes (checkpoint, data, and pinned code all match
  `MANIFEST.json`); `strict_verify.py` is unchanged; no evaluation file is read by training code (grep the run's
  code and transcript for `validation_36`, `test_short`, `test_long`, `heldout`, `transfer`,
  `textbook_dev`, `textbook_train`).
- **Gate C, review.** `review.md` is committed with all three sections and no `QUARANTINE` line.
  Only numbers that appear in `review.md` with a passing check enter `results.md` or `summary.md`.

### What the reviewer recomputes, with its own code

1. Pass rate by proof length, by re-running every dumped proof through `strict_verify.py`, not by
   reading the executor's JSON.
2. Textbook solves on the 72 (section 1a), per seed and per slice, plus the frozen-checkpoint
   control at equal attempts (same k, same temperature). "RL solved N" without that control is
   rejected.
3. Leakage: renaming-class intersection between the RL prompt set and the fork's heldout and
   transfer pools, the 72 textbook problems, and `validation_36`, using `gen.canon_key` from the
   pinned fork (and `canonical_theorem` from the vendored `nd_data.py` as a second opinion).
4. Reward sanity: within-group reward standard deviation is not zero for most of the run; the
   fraction of accepted proofs that are trivial (conclusion is a premise, `F` among premises).
5. Twenty random raw samples read, and every accepted proof above length 7 inspected.
6. Wording against n: a difference needs 3 seeds; 0 to 1 textbook solves is noise (section 5).

### Hard constraints and quarantine

Any of these puts `QUARANTINE <UTC> <reason>` in `status.md`; the branch is kept, nothing from
the run enters `results.md` or `summary.md`, and one line in `docs/ws3/log.md` says why. No
automatic retry; a quarantined experiment needs a new `proposal.md`.

- Training or tuning on the 72 textbook problems, `validation_36`, the test files, or a renaming class of any of them.
- Scoring the test files more than once, or inspecting per-theorem test verdicts.
- Any change to a fixed component (section 1 of `CLAUDE.md`), including the verifier.
- A checkpoint or key committed.

### Questions without blocking

Agents append to `docs/ws3/questions.md`: date, the question, and the default they will follow if
unanswered, then continue. The only things that stop a session are the "stop and ask" cases in
`CLAUDE.md`. Dan's lesson, kept here: briefs that state designs and budgets with the force of rules
make the agent optimise the brief instead of the question. In `proposal.md`, the hypothesis and
the fixed components are rules; the design is a suggestion.

### Money and idle compute

- One GPU for the week. A pod watchdog runs on a cron: if no agent session has been alive for 30
  minutes while a pod exists, pull `experiments/ws3/<name>/` artefacts and stop the pod. Dan lost
  about a fifth of his spend to idle pods before adding this.
- Every run has a hard stop time in the driver; the kill switch at that time pulls artefacts and
  stops the pod.
- Checkpoints and eval dumps go to the Hugging Face bucket at `DONE`; never anything that could
  contain a token.
- **Spend ledger.** RunPod invoices only credit purchases, not pods, so per-run cost is measured
  as a balance delta. The executor reads the account balance through the RunPod API at run start
  and at `DONE`, and writes the delta, pod type, and pod-hours into `numbers.md`. The librarian
  copies one row per run into `docs/ws3/spend.md` (run, GPU, pod-hours, dollars, estimate from
  `proposal.md`, notes) and keeps a running total against the one-GPU-for-the-week ceiling in
  `CLAUDE.md`. Top-up receipts from the RunPod Billing page are the documents for reimbursement.

### Not borrowed

The proposer role (Charles writes proposals this week), the librarian as an agent, weekly digests,
per-role API keys, Lean as checker of record (outside the fixed components), and Dan's fork's
tooling. Revisit after Saturday if the loop is worth keeping.
