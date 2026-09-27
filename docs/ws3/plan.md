# Workstream 3 plan: off-the-shelf RL pipeline

For Claude Code agents working for Charles. Standing rules (what is fixed, git, experiment
folders, when to stop and ask) are in the root `CLAUDE.md`. Week of Sep 20 to 26, 2026; results
go to the Saturday group meeting.

## 1. Scope

> With dataset and pre-training held fixed, what is the best existing RL library we can plug in,
> how much lift does it give over pre-training alone, and how much work was it to set up?

In scope: an existing RL library on the fixed checkpoint, the reward and evaluator, RL
hyperparameters, RL monitoring, and a one-day Tinker comparison. Out of scope: dataset,
tokenizer, pre-training recipe or architecture, and writing an RL algorithm from scratch; if one
blocks progress, stop and ask. Deliverable: two or three graphs and a clear yes/no
recommendation, not clean infrastructure.

## 1a. Fixed components

Everything below is vendored and hash-pinned in `code/ws3/fork_pin/`: code in git, data and
checkpoint fetched by `fetch.sh` and verified against `MANIFEST.json`. Put `fork_pin/` on
`PYTHONPATH`, never the live fork.

| Component | Where | Notes |
| --- | --- | --- |
| Checkpoint | `fork_pin/ckpt/pretrain.pt` from `hf://robbiethompson2018/nd-rl-checkpoints/runs/039_muon-adamw-finish-5pct_s1/pretrain.pt` (12,858,650 bytes, sha256 `418af3f9df382cd12c75b9399eaeddd4dbf70978156bc0799231d0e8c4e81cf2`) | Weights before any RL; dict with `state`, `tok_mode='abs'`, `seed=1`, `pretrain_sha='b9800e68f453'`. Run summary: 2931 steps, 4.87 epochs, 300 s on an A40, last loss 0.059, held-out greedy 0.9885. Ignore the folder's `rl_round_0N.pt` and `final.pt`. Load with `fork_pin/load_pretrain.py`. Why this one: log.md, 2026-09-23. |
| Pre-training recipe | `source/pretrain.py` in the same HF folder | Muon (lr 0.01, momentum 0.95) on block matrices and AdamW (lr 1e-3, wd 0.1) on the rest, AdamW for everything in the last 5%. Batch 256, warmup 200, cosine over 300 s wall clock, bf16, prompt-masked loss, random `N<k>` shift. |
| Code pin | `dan-pandori/nd-takehome` @ `ab629c7d2cdf3dc50d471e302d14643ba0782749` | Only `model.py`, `tokenizer.py`, `nd_verify/`, and the data below are used. |
| Tokenizer | `fork_pin/tokenizer.py`, `Tokenizer('abs')` | 99 tokens: `<pad>`, 19 symbols, 15 rules, `N1` to `N64`. PAD=0, EOS=`QED`, whitespace split. |
| Model | `fork_pin/model.py`, `GPT(99, 4, 256, 8)` | 3,210,240 params; biased LayerNorm and projections, GELU, interleaved RoPE, KV cache. |
| Pre-training data | `fork_pin/data/train.jsonl.gz` (154,990, lengths 2 to 6), `heldout.jsonl` (5,000) | Disjoint by renaming class (`gen.canon_key`). |
| RL prompt pools | `fork_pin/data/rl_targets.jsonl` (4,495), `transfer.jsonl` (2,285) | True minimal lengths 7 to 14. RL samples only `rl_targets`; `transfer` is evaluation only. |
| Upstream targets | `fork_pin/data/validation_36*.jsonl`, `test_*_prompts.jsonl` | Test files: score once, aggregate only, never inspect. |
| Proof checker | `fork_pin/nd_verify/verify.py` via `strict_verify.py` in `code/experiments/current/r1_cot/vendor/` | Always score through the strict wrapper. |
| Textbook test set | `code/ws3/textbook72/`: `textbook_dev.jsonl` (58; 43 at reference length 7+) and `textbook_train.jsonl` (14) | From `origin/dmitry/cap-comparison-textbook-eval-data`. Zero renaming-class overlap with train, heldout, `rl_targets`, `transfer`. Report the 58 and 14 separately and combined, and say in every write-up that the 14 were RL prompts and the 58 the dev benchmark in the cap-comparison sprint. |

Dmitry's vendored take-home (`code/experiments/current/r1_cot/vendor/`) is a different lineage.
Use it only for `strict_verify.py`, `nd_metrics.py`, `canonical_theorem` (leak audits), and
`nd_grpo.py` as a hyperparameter anchor (group 8, lr 1e-5, KL 0.01, clip 0.2, temp 0.8, 768 new
tokens).

## 2. Software

- **Primary: TRL `GRPOTrainer`**, plain HF generation for rollouts, one GPU.
- **Fallback** if TRL is too slow or unstable: verl or open-instruct, only after recording what
  failed and the measured throughput.
- **Tinker, one day max**, reference only: it cannot load our checkpoint, so it runs a ~1B
  open-weight model on the same task. That model has seen logic in pre-training, so the number
  says little about emergence; state this caveat wherever it appears.
- `uv` with a lock file; `metrics.jsonl` per run, no external logging; RunPod, one pod; our
  checkpoints to the HF bucket, never git; `tmux` sessions named after the branch.

## 3. Task sequence

In order. Don't start a task until the previous gate passes; a gate failing twice means stop
and ask.

- **T0 Orient.** Find the generator, tokenizer, pre-training script, checker, textbook problems,
  and archive conventions; write paths to `docs/ws3/notes.md`. Gate: each has a path or a
  question.
- **T1 Baseline checkpoint.** `fetch.sh`, load, then compute prompt-masked next-token accuracy
  and loss on `heldout.jsonl`, and greedy solve rate at lengths 2 to 6. Gate: about 95% accuracy
  or more; if not, stop and ask. Do not retrain.
- **T2 HF conversion.** `code/ws3/hf_wrapper.py`. Gate: logits match the fork within 1e-4 on
  100 heldout sequences, exact tokenizer round-trips, TRL accepts it as policy and reference.
- **T3 Reward and evaluator.** `code/ws3/eval/`. Binary reward through the strict checker;
  shaping is a separate, labelled experiment. The evaluator reports the section 5 metrics by
  length, and textbook results by `reference_lines` and slice. Tests cover a valid proof, a
  valid proof of the wrong goal, a truncated proof, a proof followed by junk, empty output, and
  unparseable output. Gate: tests pass, and the pre-RL baseline numbers are saved; everything is
  compared to them.
- **T4 First GRPO run.** Prompts from `train` lengths 5 to 6 and `rl_targets` 7 to 8, never from
  evaluation sets. Small lr, nonzero KL, group 8 to 16, temperature 1.0, generous length cap.
  Gate: completes, full section 4 metrics, reward not flat at zero (if it is, add easier lengths).
- **T5 Sweep.** lr, KL, group size, temperature, one at a time, 10 to 20 runs. Pick the best two
  by avg solve rate on the `rl_targets` stride, not textbook results; rerun each with 3 seeds.
- **T6 Tinker**, one day. Same reward, record setup time, cost, and evaluator metrics; if blocked,
  write up why and move on.
- **T7 Final eval.** Baseline, best TRL config (3 seeds), Tinker. Dump every accepted proof above
  length 7 for human review before claiming any. Produce the section 5 graphs.

## 4. Required monitoring

Every RL run writes one `metrics.jsonl` line per step with `utc` and step. A run missing any of
these does not count:

- mean reward, and within-group reward std
- greedy solve rate by length (periodic eval)
- KL to reference, policy entropy
- fraction unparseable or invalid
- completion length (mean, max), fraction hitting the cap
- grad norm, learning rate, rollout samples per second

Expert iteration (`run.algo: ei`) has no reference model or policy-gradient step, so it logs one
line per round with its own set instead: verified fraction of samples (`reward`), greedy solve
rate by length (periodic eval), unparseable and invalid fractions, prompts with a verified
proof, verified proofs kept, SFT records and SFT loss, cumulative completions (for comparisons
at the same number of samples), and samples per second. Both sets live in
`code/ws3/metrics_callback.py`, and the report checks the one matching the run.

| Signature | Likely fix |
| --- | --- |
| entropy collapses early, reward plateaus | lower lr, or raise KL or temperature |
| KL spikes, then reward crashes | lr too high |
| within-group reward std near zero | prompts all too easy or too hard |
| many completions at the length cap | raise the cap, check EOS |
| reward rises, textbook solves don't | reward hacking; read samples |

Read at least 20 raw samples per reported run and put a few in `results.md`.

## 5. Metrics and deliverables

A capability claim cannot rest on pass@k at large k: one lucky proof in 1,024 samples says the
proof is in the model's support, not that the model can prove the theorem. So the headline is
the **avg solve rate**: for each problem, the fraction of its n samples (temperature 1.0) that
verify, averaged over problems. This is the unbiased pass@1 estimate. Any-of-k counts (pass@k at
the evaluated k) stay in the tables as a secondary number. All metrics below are computed by
`code/ws3/eval/` from the same samples, with the frozen checkpoint scored identically as control.

| Metric | What it answers |
| --- | --- |
| avg solve rate by length, 95% CI over problems | how often does one attempt succeed |
| reliable fraction (per-problem rate >= 0.5, and >= 0.9) | how many problems are solved most of the time |
| frontier: longest length with avg >= 0.5 at it and every shorter length | one number per checkpoint for the length axis |
| per problem, base vs RL: **new capability** (base 0 of n, RL >= 0.5), sharpened, lost | is RL unlocking problems or making known proofs likelier |
| valid prefix of failed attempts (valid leading lines / reference length) | progress on problems not yet solved; smooth where solve rates jump |
| log-probability of the best known verified proof (union of both models' finds) | sampling-free, continuous companion to the avg |
| self-knowledge: AUROC of sequence log-prob for right vs wrong, likelihood top-1 | does the model know which of its attempts is right |
| consistency over renamings and premise orders of the textbook problems | capability rather than memorised wording |
| k50: samples needed for pass@k >= 0.5 | the cost of a solve, stated instead of hidden |

An emergence claim needs a rise in avg solve rate at lengths 7+ backed by nonzero new
capability counts, on 3 seeds. Gains only in "sharpened" mean RL made existing proofs likelier.

Deliverables:
1. Avg solve rate vs. proof length, before and after RL, pre-training cutoff and frontier
   marked, seeds as error bars.
2. The per-problem base vs RL scatter on the `rl_targets` stride, with new-capability counts.
3. Reward, KL, and entropy curves for the best config.
4. Textbook 72 per seed, baseline vs. RL: avg solve rate, expected solves (sum of per-problem
   rates), reliable count, and consistency over variants, with the 43 at length 7+ separate.
   The project's previous best is 0 to 1 of the 43 solved at all
   (`experiment-summaries/2026-09-10-proof-cap-comparison/reports/summary.md`).
5. `docs/ws3/summary.md`, one page: library recommendation, lift over pre-training, setup effort
   per option, Tinker caveat, what failed, next three experiments.

## 6. Automation loop

Cut down from Dan's orchestration (`origin/dan_orchestration`). His lesson: executor counts are
usually right, but every run needed an independent pass to catch wording, inflated counts, or
selecting on the eval set.

```
proposal.md → gate A → executor → gate B → reviewer recount → gate C → results.md → log.md
```

| Role | Who | Writes | Must not |
| --- | --- | --- | --- |
| executor | one session per experiment on `charles/exp-<name>` | `config.yaml`, launch command, `status.md`, `numbers.md`, dumps, `metrics.jsonl` | edit fixed components or tests; write conclusions |
| reviewer | a separate session; sees raw artefacts and `proposal.md`, not `numbers.md`, until its recount is committed | `review.md` | edit the executor's files |
| librarian | Charles | `results.md` conclusion, `log.md`, `summary.md` | use a number not in `review.md` |

Sessions run under a `runrole`-style driver: `claude -p` with the role brief, `--continue` on
interruption, 30-minute sleep on a real usage limit, stop on auth failure or three failed
invocations, a hard stop time, and a per-role git author.

**Files in `experiments/ws3/<exp-name>/`:** `proposal.md` (hypothesis, config, numeric expected
result, conclusion either way, GPU-hour budget, stop rule), `config.yaml` and launch command
(committed before the run), `status.md` (ends `DONE <UTC>` or `QUARANTINE <UTC> <reason>`),
`numbers.md` (each number with its source file), `review.md` (Recount, Claims, Verdict),
`results.md` (graphs, 20 samples, conclusion from `review.md`).

**Gates.**
- **A, pre-registration:** `proposal.md` with an expected number is committed before the first
  `metrics.jsonl` line. Otherwise the run is labelled "not pre-registered" and stays off the
  Saturday graphs.
- **B, clean execution:** `status.md` is `DONE`, tmux has exited, all section 4 metrics are
  present, every cited file exists, `fetch.sh --verify` passes, `strict_verify.py` is unchanged,
  and training code never reads `validation_36`, `test_short`, `test_long`, `heldout`,
  `transfer`, `textbook_dev`, or `textbook_train`.
- **C, review:** `review.md` has all three sections and no quarantine. Only numbers that pass
  review go into `results.md` or `summary.md`.

**The reviewer recomputes with its own code:**
1. avg solve rate by length, re-verifying every dumped proof, and the per-problem new-capability
   counts from the base and RL dumps
2. textbook avg solve rate and reliable count per seed and slice, against the frozen checkpoint
   at equal k and temperature
3. renaming-class leakage between RL prompts and every evaluation set (`gen.canon_key`, with
   `canonical_theorem` as a second opinion)
4. reward sanity: nonzero group std for most of the run, and the fraction of trivial accepted
   proofs
5. 20 random samples, plus every accepted proof above length 7
6. wording against n: a difference needs 3 seeds, and "emergence" needs new-capability counts,
   not only a higher avg

**Quarantine** (branch kept, nothing reported, one line in `log.md`, a new proposal to retry):
training or tuning on any evaluation set or a renaming of one; scoring the test files twice or
inspecting per-theorem test verdicts; changing a fixed component; committing a checkpoint or key.

**Questions** go in `docs/ws3/questions.md` with the default you'll follow, then keep working.
Only the `CLAUDE.md` stop-and-ask cases halt a session. In `proposal.md` the hypothesis and fixed
components are rules; the design is a suggestion.

**Money.** One GPU for the week. A cron watchdog stops any pod that has had no live agent for
30 minutes, after pulling its artefacts. Every run has a hard stop time. Checkpoints and dumps go
to the HF bucket at `DONE`. Cost per run is the RunPod balance delta from start to `DONE`,
recorded in `numbers.md` and copied into `docs/ws3/spend.md` with a running total.
