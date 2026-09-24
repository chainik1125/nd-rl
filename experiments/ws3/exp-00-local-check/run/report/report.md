# Report for exp-00-local-check

Generated 2026-09-23T15:14:03+00:00.

## Gate B: metrics.jsonl coverage (plan section 4)

- ok: mean reward (`reward`)
- ok: reward std within groups (`reward_std`)
- ok: pass rate by proof length (periodic eval) (`eval/pass_rate_len*`)
- ok: KL to reference (`kl`)
- ok: policy entropy (`entropy`)
- ok: unparseable fraction (`reward/unparseable_frac`)
- ok: invalid fraction (`reward/invalid_frac`)
- ok: completion length mean (`completions/mean_length`)
- ok: completion length max (`completions/max_length`)
- ok: fraction at length cap (`completions/clipped_ratio`)
- ok: gradient norm (`grad_norm`)
- ok: learning rate (`learning_rate`)
- ok: rollout throughput (`rollout/completions_per_sec`)
- ok: utc timestamp (`utc`)
- ok: step (`step`)

50 rollout log lines, 6 periodic eval lines.

![training curves](curves.png)

![periodic eval](periodic_eval.png)

## greedy pass rate by length, heldout stride

![greedy_heldout](greedy_heldout.png)

| length | control solved/n | control rate [95% CI] | RL solved/n | RL rate [95% CI] |
| --- | --- | --- | --- | --- |
| 2 | 200/200 | 1.000 [0.981, 1.000] | 200/200 | 1.000 [0.981, 1.000] |
| 3 | 198/200 | 0.990 [0.964, 0.997] | 199/200 | 0.995 [0.972, 0.999] |
| 4 | 190/200 | 0.950 [0.910, 0.973] | 187/200 | 0.935 [0.892, 0.962] |
| 5 | 174/200 | 0.870 [0.816, 0.910] | 179/200 | 0.895 [0.845, 0.930] |
| 6 | 145/200 | 0.725 [0.659, 0.782] | 146/200 | 0.730 [0.665, 0.787] |
| all | 907/1000 | 0.907 | 911/1000 | 0.911 |

## pass@k by true length, rl_targets stride

![long_pool](long_pool.png)

| length | control solved/n | control rate [95% CI] | RL solved/n | RL rate [95% CI] |
| --- | --- | --- | --- | --- |
| 7 | 14/82 | 0.171 [0.105, 0.266] | 11/82 | 0.134 [0.077, 0.224] |
| 8 | 0/44 | 0.000 [0.000, 0.080] | 0/44 | 0.000 [0.000, 0.080] |
| 9 | 0/40 | 0.000 [0.000, 0.088] | 0/40 | 0.000 [0.000, 0.088] |
| 10 | 0/20 | 0.000 [0.000, 0.161] | 0/20 | 0.000 [0.000, 0.161] |
| 11 | 0/5 | 0.000 [0.000, 0.434] | 0/5 | 0.000 [0.000, 0.434] |
| 12 | 0/8 | 0.000 [0.000, 0.324] | 0/8 | 0.000 [0.000, 0.324] |
| 14 | 0/1 | 0.000 [0.000, 0.793] | 0/1 | 0.000 [0.000, 0.793] |
| all | 14/200 | 0.070 | 11/200 | 0.055 |

## textbook 72 at k=64, temperature 1.0 (control vs RL)

| slice | frozen checkpoint | after RL |
| --- | --- | --- |
| dev58 | 8 / 58 | 8 / 58 |
| train14 | 2 / 14 | 2 / 14 |
| all | 10 / 72 | 10 / 72 |
| reference length >= 7 | 0 / 43 | 0 / 43 |
