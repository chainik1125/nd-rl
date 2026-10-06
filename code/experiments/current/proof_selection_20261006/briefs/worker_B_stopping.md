# Worker B brief: intrinsic scorers, stopping simulator, accuracy-vs-attempts figures

You are Worker B in a 10-hour research sprint (deadline 2026-10-06 11:42 UTC; results due by 09:30 UTC, hard stop
for new runs at 10:15 UTC). Read `start.md` and `schema.md` in this folder first; they are binding.

## Environment

- Python: `~/venvs/proof-state/bin/python`. Before any run: `export PATH=$HOME/.elan/bin:$PATH OMP_NUM_THREADS=8 ND_SAMPLE_PATH=fast`.
- Dan's code: `~/runs/proof-selection/nd-takehome` (read, do not modify; `sys.path.insert(0, ...)`). Key files:
  `state_env.py` (`Env`, `decompose`, `canonicalise`), `state_sample.py` (`prompt_ids`, how states are encoded),
  `lean_tok.py` (tokenizer modes `lean_staten`; `encode_toks`, `decode_action`; which ids are name tokens), `model.py`
  (`load_ckpt`, `GPT.forward`), `best_model.py` (ALiBiGPT), `STATE_ENV.md` (the interface; read the "Names, and arm SN"
  section carefully).
- Frozen model: `~/runs/proof-selection/ckpts/la_T1_best12_s0_r16.pt`. GPU: **only 7** (`CUDA_VISIBLE_DEVICES=7`).
- Your code goes in this folder's `code/`; outputs in `~/runs/proof-selection/scores/` and `~/runs/proof-selection/stopping/`.
  Log at `logs/worker_B.md` (timestamped; decisions; dead ends; commands). No git commits/branch switches/pushes, no paid
  compute. Never read, edit, or stop anything under `/data/users/dmitry/ndrl/code/experiments/current/learned_compute_allocation_20261006/`
  (a parallel sprint).

## Tasks, in order

1. **Teacher-forced scorer** (`code/score_loglik.py`): for each attempt in a pool (schema §1), replay its `actions`
   through `Env(prompt, canon=True, base=base, assign=True)` to recover the exact state the model saw at each step, build
   the model input exactly as `state_sample.prompt_ids` does, and compute the per-token log-probabilities of the emitted
   action tokens (including the `<eos>` that ends an action) under the frozen r16 at temperature 1. **Mask the
   environment-assigned name positions** (SN arm: the model's name token after `have` is overridden by the environment,
   so its log-prob is not the model's choice; see STATE_ENV.md). Output scorers `loglik_mean`, `loglik_sum`,
   `loglik_min_step` (min over steps of the step's mean token log-prob), `entropy_mean` (negated, so higher = more
   confident), and `completion_loglik` (done ? loglik_mean : -inf), each to schema §3. Validate on the smoke pool
   `~/runs/proof-selection/smoke/` or the first pool Worker A writes to `~/runs/proof-selection/pools_v2/`: a done attempt
   replays without error; the greedy re-decode of a high-likelihood attempt should mostly reproduce its actions. Batch
   the forward passes (pad right, causal mask) so that 100k attempts score in minutes on one H200.
   Until Worker A's pools exist, develop against the existing valid proofs in `~/runs/proof-selection/pools/*.jsonl`
   (`proofs` lists valid ND texts; use `state_env.decompose(prompt, nd, canon=True)` to get actions; note those have
   no stored base, so use base 0 there).
2. **Stopping simulator** (`code/stopping.py`, schema §5): given a pool, its labels, and any set of score files, sweep
   thresholds τ and N_max ∈ {8, 16, 32, 64}; a policy consumes attempts in index order, stops when the best score so far
   ≥ τ or at N_max, returns the best-scoring attempt so far. Record per theorem: attempts used, generated tokens used,
   selected attempt valid?. Also implement: fixed-k with the same selector (k ∈ 1..64), "stop at first finished"
   (completion score with τ = 1), oracle (stop at first valid), and *random mixtures* of fixed-k that match a given mean
   attempt count. Because attempts within a pool are i.i.d. and order-free, average over R=50 random orderings per
   theorem. Output `stopping/<pool>__<scorer>.json` with the curves and a paired theorem-level bootstrap (B=1000) of
   accuracy differences at matched mean attempts.
3. **Figures** (`figures/`, matplotlib, PNG, self-explanatory axes and legend; title says the pool and generator):
   (a) accuracy vs mean attempts: fixed-k random pick, fixed-k completion pick, adaptive completion-stop, adaptive
   loglik-stop, and, when Worker A publishes `verifier_*` scores, fixed-k verifier pick and adaptive verifier-stop, plus
   the oracle pass@k; (b) the same with x = mean generated tokens; (c) per-theorem attempts used by the adaptive
   policy against the theorem's true pass rate (does it allocate more to hard theorems?). Dev pools for choosing τ;
   test pools (holdout250, textbook72) read once, after you log a freeze time.
4. **Diagnostics**: AUROC of each scorer for "valid" over all attempts and over finished attempts only, by set and by
   per-theorem difficulty bucket (pass rate 0–0.05, 0.05–0.25, 0.25–0.5, 0.5–0.9, 0.9–1). Does likelihood carry
   recognition signal beyond completion?
5. **Report** `results/worker_B_report.md`: what was run, tables, the commands, limits.

If something blocks you for more than 20 minutes, log it and move on. Coordinate file formats strictly through
`schema.md`; if you must change it, write the change in your log and in the schema file with a timestamp.
