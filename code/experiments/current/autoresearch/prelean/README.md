# Pre-Lean holdout scoring (runs 000-097)

Scores pre-Lean `final.pt` checkpoints on the transfer holdout. The Lean switch (`a546db9`)
replaced the frozen harness, so the old eval lives here:

- `harness.py`: the pre-Lean harness verbatim from `ec802d5` (sha256 `5f0740b15fdb`, the
  `harness_sha` in every 011-097 checkpoint). Its eval functions match 000-010's harness
  (`d602830d7e51`) except `EVAL_K`, so `score_holdout.py` uses k = 32 for 000-010.
- `score_holdout.py` / `.sh`: `../score_holdout.*` with the pre-Lean fork (`ab629c7`, `abs`
  tokenizer, nd_verify only) and no `TOK_MODE`.
- `src/`: `pretrain.py` for 000-010, which have no `source/` on HF. Rebuilt from git and
  sha-checked against `params.json`, except 003, 007 and 008, discarded before patches were kept:
  those use 001's file (008 with 3 layers), and the checkpoints load strictly into it.
- `fanout.py`: rents pods, sets up a worktree of the pre-Lean fork next to the Lean one, and drains
  `jobs.txt`.

Reproduction checks: `081_s0` dev re-scored 433, equal to the recorded value, on an A40. 000-010
dev re-scores land within 6 of the recorded values (008 and 010 exact). Non-A40 GPUs change
bf16 numerics, so seeded sampling diverges.

Results (2026-09-27, 264 checkpoints): `artifacts/autoresearch/holdout/`, read by
`experiment-summaries/2026-09-21-pretrain-ablations/charts/extract.py`.
