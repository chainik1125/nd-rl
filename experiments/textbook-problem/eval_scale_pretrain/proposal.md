# eval_scale_pretrain: Robbie's newest pretrain checkpoints on the textbook-problem datasets

Pre-registered 2026-10-02 (UTC), before the run. Charles asked: "test the most recent pretrain
from Robbie on these datasets".

## Checkpoint

HF bucket `robbiethompson2018/nd-rl-checkpoints`, runs dated 2026-09-30 (the newest): the
`scale-*-p1200` family, six pretrains (two sizes × three seeds), tokenizer mode `lean_seq`, 1,200 s
of pretraining each (off-protocol "scale-textbook" runs per the harness source).

| run | params | arch | dev metric (pretrain, k=64, temp 0.8) |
|---|---|---|---|
| scale-d384-p1200_s0 | 9,557,760 | ALiBiGPT 6L d384 8h, PeriLN, d_ff 1280 | 702 / 1500 |
| scale-d768-p1200_s0 | 37,989,888 | same family, d768 | 710 / 1500 |

Primary: `scale-d384-p1200_s0/pretrain.pt` (sha in `run_*/log.txt` meta). Secondary, if time:
`scale-d768-p1200_s0/pretrain.pt`. Each is one seed: signs of life only, no comparisons claimed.

## Code

`code/textbook_problem/eval_ckpt.py`. Loads the run's own `source/pretrain.py` (ALiBiGPT +
PeriLNBlock) on top of the fork at commit 51604b3 (local clone, read-only), samples with the fork's
`sample.generate_ids`, decodes lean_seq → ND with the fork's `lean_tok.inverse`, judges every proof
with the pinned nd_verify through strict_verify (`prove.safe_verify`). Lean is not consulted.
Prefix tasks seed the decoder with the lean_seq prefix of the reference proof that reveals nothing
beyond the given ND lines except open-box headers (documented in the module docstring).

Sanity checks run before this proposal (k=8, 32 random records each, seed 0):
fork heldout (lengths 2–6) avg solve 0.977, any-of-8 32/32; rl_targets (7–14) at temp 0.8 avg
0.223, any-of-8 11/32. Consistent with the run's own dev metric. Throughput on this laptop
(M1 Pro, CPU) 2.6–4.0k tokens/s at batch 256.

## Protocol

k = 32, temperature 1.0, max_new 400 lean_seq tokens, seed 0, batch 256 (plan T5 settings).
Sets, in run order: v0 (104), v1 (80), hand14 (14), batch3 (21), lean_v2 (47 + 3 unsolved, the
other session's bundle), derived (18 feedable nuances, 202 rows; repair and the clozes cannot be
encoded), textbook72 (72, aggregate only; its sample dump goes to the gitignored
`docs/textbook-problem/textbook72/`).

Headline per set: avg solve rate (mean over problems of the verified fraction of 32 samples) with a
95% interval over problems; any-of-32 count; never-solved count; parse-fail fraction. Derived set
also by nuance. Every judged sample is written to `run_*/<set>.jsonl`.

## Expected result (numbers written before the run)

From the smoke test (0 of 16 samples and 0 of 8 greedy on the first four v0 problems) and the
heldout/rl_targets sanity numbers:

| set | expected avg solve rate | expected any-of-32 |
|---|---|---|
| v0 | 0.05–0.15 | 30–50 % |
| v1 | 0.02–0.08 | 15–35 % |
| hand14 | ≈ 0 | 0–2 of 14 |
| batch3 | 0.02–0.10 | 20–40 % |
| lean_v2 | 0.05–0.15 | 30–50 % |
| derived, prefix nuances | above their bases by ≥ 0.10 | |
| derived, forward_control | highest of the derived families | |
| textbook72 | 0.05–0.15 | 30–50 % |

We conclude either way: within the bands, textbook problems are a different distribution from the
fork's pools for this pretrain and the derived prefix tasks are measurably easier; above the bands,
check for a scoring bug or leak first (CLAUDE.md); below, the model is effectively at floor on
textbook forms and the set is a hard eval for RL to move.

## Budget and stop rule

No money: laptop CPU only. Expected wall time 30–60 min for d384, 2–3 h for d768. Stop rule: if a
set exceeds 2 h or the parse-fail fraction exceeds 0.5 on any set, stop and investigate before
continuing.

## Launch command

```
cd code && nohup uv run python textbook_problem/eval_ckpt.py --run scale-d384-p1200_s0 --stage pretrain \
  --sets v0,v1,hand14,batch3,lean_v2,derived,textbook72 --k 32 --temp 1.0 --max-new 400 --batch 256 --seed 0 \
  --out ../experiments/textbook-problem/eval_scale_pretrain/run_d384_s0 > ../experiments/textbook-problem/eval_scale_pretrain/run_d384_s0.out 2>&1 &
```
then the same with `--run scale-d768-p1200_s0 --out …/run_d768_s0`.
