# Worker A brief: collector, pools, baselines, learned verifier

You are Worker A in a 10-hour research sprint (deadline 2026-10-06 11:42 UTC; your results are due by 09:30 UTC,
hard stop for new runs at 10:15 UTC). Read `start.md` and `schema.md` in this folder first; they are binding.

## Environment

- Python: `~/venvs/proof-state/bin/python`. Before any run: `export PATH=$HOME/.elan/bin:$PATH OMP_NUM_THREADS=8 ND_SAMPLE_PATH=fast LEAN_GATE_WORKERS=16`.
- Dan's code: `~/runs/proof-selection/nd-takehome` (read it; do not modify it; import via `sys.path.insert(0, ...)`).
  Key files: `state_sample.py` (`env_generate`), `state_env.py` (`Env`, `decompose`), `lean_judge.py`
  (`judge_many([(prompt, nd), ...])` → `(ok, reason, n_lines)`), `model.py` (`load_ckpt`), `best_model.py`,
  `state_train.py` (how (state, action) pairs are batched and loss-masked), `lean_tok.py` (tokenizer; `decode_action`,
  `encode_toks`), `STATE_ENV.md` (the interface).
- Generator checkpoint: `~/runs/proof-selection/ckpts/la_T1_best12_s0_r16.pt` (frozen for all sampling).
- GPUs: **only 5 and 6** (`CUDA_VISIBLE_DEVICES`). Check `nvidia-smi` before launching; if someone else's process appears
  on 5 or 6, do not touch it, note it in your log, and share the GPU.
- Your code goes in this folder's `code/` subfolder; heavy outputs in `~/runs/proof-selection/` (pools_v2, scores, verifier).
- Keep a log at `logs/worker_A.md` here (timestamped; decisions; dead ends; commands). No git commits, no branch
  switches, no pushes, no paid compute. Never read, edit, or stop anything under `/data/users/dmitry/ndrl/code/experiments/current/learned_compute_allocation_20261006/`
  (a parallel sprint).

## Tasks, in order

1. **Collector** (`code/collect_pools.py`): a fork of `env_generate`'s loop that records, per attempt, the fields in
   schema §1 (raw action token ids per step, `base`, `done`, `steps`, assembled `nd` when done, `gen_tokens`), judges the
   finished ones with `lean_judge.judge_many` once, and writes labels to the separate `.labels.jsonl` (schema §2).
   Never put reason strings, `LEANREJ`/`LEANPARSE` markers, or verdicts in the pool rows. Keep Dan's sampling settings
   (T 0.8, `max_action 512`, `max_steps 96`, batch 2048, `early='eos'`, SN canonical naming with `assign=True` and a
   recorded per-attempt base). Unit-check on 3 theorems × 8 attempts that replaying `actions` through
   `Env(prompt, canon=True, base=base, assign=True)` reproduces `nd` byte for byte for done attempts.
2. **Pools.** Generate with the frozen s0_r16 generator: `rl_targets` k=16, `dev1108` k=16, `rr600` k=32
   (`data/ladder/transfer_long_rr600.jsonl`), `holdout250` k=64, `textbook72` k=64. Split the GPUs. Write a
   `pools_v2/MANIFEST.json` (file, sha256, n theorems, k, wall time, validity rate, finished rate). Before generating,
   write `splits.json`: selector-train / selector-dev theorem names (theorem level; rl_targets + ~70% of dev1108 and
   rr600 for train, the rest dev), with holdout250 and textbook72 as **frozen test** (never used to pick anything).
   Report exact-prompt overlap between every pair of sets and read `data/ladder/POOLS.md` for the documented
   renaming-class exclusions; write what you can and cannot guarantee into `logs/worker_A.md`.
3. **Baselines** (`code/baseline_scores.py` → schema §3): `random`, `first`, `completion`, `shortest`. Then
   `code/evaluate_selection.py` (schema §4) producing `results/selection_<pool>.json` and a table. Worker B delivers the
   likelihood/entropy scorers to `~/runs/proof-selection/scores/` using the same format; evaluate them when they appear.
4. **Learned verifier** (`code/train_verifier.py`): copy of r16 (`load_ckpt`) with a scalar head on the final token.
   Input = one attempt rendered as the sequence of (state, action) pairs it actually saw, obtained by *replaying*
   `actions` through the environment with the stored base (states are the model's native format; do not invent a new
   prompt format). Concatenate `[state_1 action_1 state_2 action_2 ...]` (clip to `max_len` 1024 by dropping the earliest
   pairs and record the clip rate). Train with BCE on train pools (balance by weighting, not by dropping), AdamW, lr ~1e-4
   on the body, early-stop on dev AUROC, 3 seeds. Output scores `verifier_s{0,1,2}` and `verifier_ens` for every pool,
   including test, written once. Also evaluate the head *only on finished candidates* (does it discriminate beyond
   completion?). Report AUROC by set and by per-theorem difficulty bucket.
5. **Report** `results/worker_A_report.md`: tables, what was run, limits, the commands. Figures as PNG via matplotlib
   in `figures/` (clear axis labels; one message per figure).

Interpret nothing on test until train/dev choices are frozen; write the freeze time in your log. If something blocks
you for more than 20 minutes, write the blocker to your log and move to the next task.
