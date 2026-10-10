# Notes (finetuning): paths and facts found while orienting

Written 2026-10-10. Everything below was checked by listing or reading on that date; re-verify
hashes at T0 before relying on a file.

## Checkpoints and dumps (Dan's bucket, `hf://buckets/dan-pandori/nd-rl`)

Listing API: `https://huggingface.co/api/buckets/dan-pandori/nd-rl/tree/<prefix>`; download:
`https://huggingface.co/buckets/dan-pandori/nd-rl/resolve/<path>`. Every `.pt` is ≈ 38.3 MB
(ALiBiGPT 9,560,832 params). Local cache: `~/.cache/nd-rl/dan_ckpts/` (already holds
`stage1_best12_s0_b1200.pt` and `la_T1_best12_s{0,1,2}_r8.pt`); Dan's worktree
`~/.cache/nd-rl/fork_dan` (branch `dan`, 0c492902). `~/.cache` survives scratchpad wipes.

| prefix | files | notes |
| --- | --- | --- |
| `best-state/ckpts/bs/ladder/` | `la_T1_best12_s{0,1,2}_r{1..8}.pt` (+ `.args.json`) | the T1 ladder of `run_best_state.md`; textbook72 (Lean) 51–52 at r8 |
| `best-state/ckpts/bs/` | `stage1_best12_s{0,1,2}_b1200.pt` | the bases, K12, 1,200 s |
| `trajectory/ckpts/tj/ladder/` | `la_T1_best12_s*_r{1..8}.pt` (66 files) | the EI arm grpo-best compares against |
| `trajectory-cap6/ckpts/tj6/ladder/` | `la_T1_best6_s*_r{1..8}.pt` (97 files) | cap-6 line, not used unless a cap comparison is wanted |
| `grpo-best/ckpts/gb/` | `gb_{default,unlikely,passk,distinct}_s{0,1,2}_r{1..8}.pt` (distinct only to r4/r5) | `grpo_state.py` arms; `analysis_stdout.txt` and `compute_stdout.txt` under `grpo-best/artifacts/gb/`; per-arm eval dumps under `.../gb/eval/` |
| `rl-from-ckpt/ckpts/rfc/` | 216 files: ladders from Stage-1 starts p1600 / p5000 / p12000 / p16000 / pend, r1..r8; `control/rc_best12_s*_p*_r*.pt` replay-only | `rl-from-ckpt/artifacts/rfc/analysis_stdout.txt`; eval dumps `.../rfc/eval/<arm>__{tb72,h250}_x{0,1}.{json,jsonl}` |

Eval dump row keys (checked on `c0_p12000_r8__tb72_x0.jsonl`): `name, thm, prompt,
reference_lines, solved, n_ok, n_tried (256), proofs (repr of a list), written_lens,
pruned_lens, reasons, fail_example`. Judge: Lean; re-judge with
`code/textbook_problem/dan_metrics.py` before quoting next to our numbers. The 72's dumps
contain per-problem rows: keep them in the gitignored `docs/textbook-problem/textbook72/`.

Groups A / B / C of the 322 `rl_targets` theorems: defined in grpo-best on sample seed 0 at k
256 (A = base solves, B = base rarely, C = neither base nor EI r8); sizes per seed 36 / 35 / 28
for C. The group membership file is inside Dan's analysis inputs; locate at T0 (likely
`grpo-best/artifacts/gb/` or `trajectory/artifacts/`), otherwise recompute from the `pend r0`
and `ei r8` dumps with his rule and say so.

## Scripts in Dan's worktree (`~/.cache/nd-rl/fork_dan`, never edited)

- `state_train.py`: fine-tune / Stage-1; args `--data --init --steps --lr --min_lr --warmup
  --cap 0 --out --seed --recs --mode lean_state --recipe best`. Loads records
  `{prompt, proof, n_lines}` and turns each into (state, action) pairs with
  `state_env.decompose(prompt, proof, canon=tok.canon)`; loss on action tokens.
- `state_ladder_ei.py`: round r fine-tune call (line ≈ 376):
  `python3 state_train.py --data mix_r.jsonl --init <ckpt> --steps 600 --lr 3e-4 --min_lr 3e-5
  --warmup 50 --cap 0 --out <new> --seed <s> --log_every 200 --recs 128`; the mix is the found
  proofs × `rl_weight 4` (≤ 4 per theorem) plus 20,000 K12 replay records. Round checkpoints
  `<ckptdir>/<name>_r<r>.pt`; artifacts `found_<r>.jsonl`, `found_transfer_<r>.jsonl`,
  `round_<r>.json`, `alloc_<r>.json`.
- `grpo_state.py`: `--adv {default,unlikely,passk,distinct}`, `--group 8 --prompts 256
  --lr 1e-4 --kl 0.0 --temperature 0.8`, round-equivalent budgets matching the ladder.
- `state_env.py`: `canonicalise(prompt, nd_body)`, `decompose(prompt, nd_body, canon)`,
  `Env(canon=True, base=b, assign=True)`; `STATE_ENV.md` documents the format.
- `state_sample.py` (`env_generate`), `state_eval.py` (what we ran on 2026-10-02 via
  `experiments/textbook-problem/eval_dan_best_state/launch.sh`), `best_model.py` (ALiBiGPT).
- Lean 4.34.1 is needed only if Lean is the judge; our protocol re-judges with the pinned
  verifier, and M4 / M5 need no judge at all.

## Our code that this workstream reuses

- `code/ws3/eval/evaluator.py`: `pass_at_k`, `pass_at_k_curve(counts, ks)`, `k_at`.
- `code/textbook_problem/dan_metrics.py`: re-judges Dan's dumps with the pinned verifier and
  computes the plan metrics; `eval_analysis.py`, `padding_analysis.py` for per-problem tags.
- `code/textbook_problem/release_1002.py`: `BR.keys` curry-aware renaming classes (leak audits).
- `code/pretrain_data/`: `subproblems.py` (seed-split rendering, permutations), `scan_pools.py`
  (`k12_wrong`, `k12_long16` builders), `mix.py`.
- `code/ws3/rl/ei.py`, `code/ws3/hf_wrapper.py`, `code/ws3/fork_pin/`: the 3.2M replica (F5).

## Data

- Textbook sets: `data/textbook_problem/release_2026-10-02/{candidate_v0,candidate_v1,
  hand_proved_14,batch3,derived}.jsonl` (412; reference proofs verifier-checked; Lean fields).
  Seed split = candidate_v0 + the Lean-sourced 47 (`lean_v2` inputs under
  `experiments/textbook-problem/eval_dan_best_state/inputs/`); test split = the rest.
- `rl_targets` (4,495; Dan's `data/ladder/rl_targets.jsonl`), the 322-theorem evaluation subset
  and the 250 held-out: in the worktree's `data/` after `fetch_artifacts.py`, or in the bucket.
- K12: `~/.cache/nd-rl/...` per `pretrain-data` notes (md5 800b5486…); its first 1,024 rows are
  the val head.

## Facts to keep straight

- EDL (Donoway et al. 2026, Def. 3.2): `EDL = Σ_i ℓ(θ_{i−1}, x_i, y_i) − n · L_test(θ*)`;
  non-negative in expectation under population monotonicity; depends on the algorithm, so A is
  held fixed; score answer (action) tokens only.
- Dan's compute: an EI round's 600-step fine-tune is minutes on an A40; GRPO 561 updates of
  2,048 rollouts = 1,401 s; a k = 256 evaluation of 322 + 72 + 250 theorems ≈ 1,000 GPU-s.
- Pinned-verifier vs Lean: 1–2 disagreeing proofs per set (Dan's loose ⊥E rendering class);
  never resolve by picking one, report both.
