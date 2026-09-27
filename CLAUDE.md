You are working for Charles in the NDRL repo. The project studies capability emergence in RL using natural deduction proofs. This current workstream asks: with dataset and pre-training fixed, what is the best existing RL library we can plug in, and how much lift does it give over pre-training alone?

**Before starting any task, read `docs/ws3/plan.md`.** It has the task sequence, gates, required metrics, and deliverables. Then read the latest entries in `docs/ws3/log.md` to see where the last session stopped. This is a scrappy signs-of-life phase. Prefer the quickest path to a trustworthy graph over clean infrastructure.

## Fixed, do not change
- Pre-trained model: Robbie's pre-training-only Muon checkpoint
  `hf://robbiethompson2018/nd-rl-checkpoints/runs/039_muon-adamw-finish-5pct_s1/pretrain.pt`
  (sha256 `418af3f9…`, see
  `docs/ws3/plan.md` section 1a). Its
  tokenizer (`abs` mode), model (`GPT`, 4 layers, d 256, 8 heads), splits, and RL prompt pools
  come from Dan's fork `dan-pandori/nd-takehome` at commit `ab629c7d`, vendored and hash-pinned
  in `code/ws3/fork_pin/` (run its `fetch.sh` once). Nothing else from that fork is used. Do not
  retrain, do not edit anything under `fork_pin/`, do not use Dmitry's vendored model or tokenizer.
- The held-out textbook problem set: Dmitry's 72 problems (58 `textbook_dev` + 14
  `textbook_train`) from branch `origin/dmitry/cap-comparison-textbook-eval-data`. Never train on
  them, never tune on them, never train on a renaming of them.
- Proof checker: the pinned fork's `nd_verify` scored through Dmitry's `strict_verify.py`.

Only the RL library, algorithm, and RL hyperparameters vary. Primary library is TRL `GRPOTrainer`.

## Python
- Run Python only as `uv run python` from `code/`, or via `code/.venv/bin/python`. Never bare `python3`. `PYTHONPATH` for the fixed model is `code/ws3/fork_pin:code/experiments/current/r1_cot/vendor`. On a fresh machine or pod, `uv sync --locked --extra dev` in `code/` first.

## Git
- Work only on `charles/main` and `charles/exp-<short-name>` branches.
- Never push to, merge into, or rebase the repo's `main`. Never force-push. Never touch other people's branches.
- Parallel agents each use their own `git worktree` on this machine.
- Commit message format: `[ws3][exp-name] what changed`.
- Commit configs, seeds, and the exact launch command for every run.
- Never commit checkpoints, large data, `.env`, or any key. Keys live in a gitignored `.env`. Checkpoints go to the Hugging Face bucket.

## Experiments
- Each experiment gets its own folder under `experiments/ws3/<exp-name>/`.
- Write `proposal.md` before any run: hypothesis, config, expected result, and what we conclude either way. No run starts without it.
- Write `results.md` after: graphs, raw samples, conclusion.
- Every RL run writes the full metric set in section 4 of the plan to `metrics.jsonl` in its run folder, one JSON line per step. Read at least 20 raw samples before reporting on a run.

## Stop and ask when
- A gate in the plan fails twice.
- Anything would spend money beyond one GPU for the week, or Tinker usage beyond its one-day timebox.
- You want to change a fixed component, or switch away from TRL.
- A result looks surprisingly good. Check for a reward bug and a train/test leak first.
- An error looks like running out of quota. Confirm it is not a normal process exit first.

## End of every session
Append to `docs/ws3/log.md`: date, what ran, what was learned, what is next, open questions. Keep all notes and decisions in `docs/ws3/` as Markdown.