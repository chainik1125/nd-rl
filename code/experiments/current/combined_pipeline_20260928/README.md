---
written_on: 2026-09-28
written_by: agent:claude
---

# Combined pipeline: everyone's best piece, one run

Robbie (2026-09-28): "combine everyone's best piece of the pipeline and see how it does."
Branch `combined-pipeline` in the `nd-rl-6` checkout, off `autoresearch-pretrain` @ `6382e86`.

## Which piece from whom

| Stage | Piece | Owner | Status here |
| --- | --- | --- | --- |
| Data | cap-6 `data/train.jsonl`, ladder `rl_targets` / `transfer` pools | Dan | already the harness's; Dan's ds-* runs found no cap-6 set that beats it |
| Tokenizer | `lean_seq` | Dan | already the harness's |
| Reward | Lean 4 ∧ nd_verify (fork `51604b3`) | Dan | kept; see skipped |
| Pretraining | autoresearch incumbent 186-claude (6×384 Peri-LN, ALiBi/NoPE, Muon, MTP, 9.56M params; 3-seed metric 668/699/674 = 680) | Robbie's loop | its saved `pretrain.pt` per seed |
| RL | GEAR run-2 champion expert iteration, tokenizer-independent parts (see `leon_rl.py`) | Leon | **the new piece under test** |
| Eval | harness `stage_eval`: dev half of transfer (1,108), 64 samples at T 0.8, metric = solved at L_true ≥ 7 | frozen harness | unchanged |

Charles (ws3, TRL GRPO/EI) found EI ≥ GRPO in half the time, which agrees with Leon; there is no
separate piece of his to add. Dmitry's R1-CoT and cap work is negative; his textbook eval sets
would need a Lean rendering and are left as a follow-up.

Skipped, with reasons:
- Dan's Lean-only judge (`lean_judge.py`, 2026-09-27): +909 accepts over 281,817 (0.3%), and it
  lives on fork heads that render BOTE as `nA .elim`, not the `False.elim nA` the 186 model
  was trained on.
- Dan's fast sampler (efficiency run, 2.16x end to end): speed only (same accepted set), and its
  cache/rope changes are on those same newer fork heads. Worth porting later: it would buy Leon's
  recipe more rounds per wall-clock.
- Leon's line-level verifier-guided sampler (the biggest run-2 gain) parses `abs` proof lines;
  a `lean_seq` port is a separate project.
- Cap > 6 data (Dan's cap-horizon: far better transfer) breaks the take-home cap-6 rule.

## Design

Paired, from the same weights: both arms load 186-claude's `pretrain.pt` for seeds 0, 1, 2 (the
harness's `--seed` also fixes the 1,500 RL targets, so they match 186's), then run RL and the
frozen eval on one A40 each.

- **ei**: the frozen `harness.py` byte for byte; `pretrain.py` is `ckpt_shim.py`. Replays 186's
  RL stage with fresh sampling noise: a plumbing check and a same-day control.
- **leon**: `harness_leon.py` = frozen harness with `stage_rl` swapped for `leon_rl.stage_rl`,
  sample-matched to the harness (192,000 RL samples = 15 rounds of 400 targets × mean k 32).

`ND_TOTAL_SECS=3600` instead of 2400 for both, so a slower recipe is measured, not killed.
Wall-clock per arm is reported.

## Prediction (written before any pod ran)

- ei: 3-seed mean within ±40 of 680 (the replay only redraws RL/eval sampling noise).
- leon: 3-seed mean 690-800; P(leon mean > ei mean) ≈ 0.6. Reasons for doubt: Leon's gains were
  measured on `abs` 3.2M models with a 15k-target pool and 2 h champion budgets, and a third of
  his run-2 gain came from the guided sampler not ported here.

## Run

```bash
source .envrc.local
for arm in ei leon; do for s in 0 1 2; do
  code/.venv/bin/python code/experiments/current/combined_pipeline_20260928/launch.py $arm $s <ckpt-dir> &
done; done
```

Local CPU smoke (`harness.py --smoke`, both arms) passed; a direct check ran GEM fine-tuning and
k allocation on the loaded 186 model. Results and write-up: TODO after the runs.
