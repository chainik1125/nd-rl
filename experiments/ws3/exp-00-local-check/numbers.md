# exp-00 local check: numbers

Generated 2026-09-23T15:15:24+00:00 by `make_numbers.py`. Paths are relative to `experiments/ws3/exp-00-local-check/run/`; the `.jsonl` dumps are committed gzipped (`.jsonl.gz`). Executor numbers only; the reviewer recounts from the dumps (`review.md`).

## Stage wall-clock (`stages.jsonl`)

| stage | seconds | finished (UTC) |
| --- | --- | --- |
| baseline | 213 | 2026-09-23T15:03:10+00:00 |
| train | 426 | 2026-09-23T15:10:16+00:00 |
| evaluate | 189 | 2026-09-23T15:13:24+00:00 |
| report | 1 | 2026-09-23T15:14:04+00:00 |
| total | 829 | |

## Environment (`baseline/environment.json`)

- git commit `a0b2ec600277fa722c769c51eb1961d68a7680cc`; checkpoint sha256 `f4409e4f82d8922f853a5c53eeb2820d05acea17bbd5cf92d24c073d97709e18`
- versions: {'python': '3.11.16', 'torch': '2.11.0', 'transformers': '5.17.0', 'trl': '1.13.0'}; platform macOS-13.6.2-arm64-arm-64bit; torch threads 6

## (a) T1 gate (`baseline/nta.json`)

- next-token accuracy **0.9835** on 5000 heldout records (363090 target tokens); mean loss 0.0669; gate 0.95: **PASS**

## (b) greedy solve rate by length, heldout stride

k=1, temperature 0.0, 1000 records, 1000 samples per model. Length = the record's reference `n_lines`.

| length | frozen solved/n | frozen rate [Wilson 95%] | RL solved/n | RL rate [Wilson 95%] |
| --- | --- | --- | --- | --- |
| 2 | 200/200 | 1.000 [0.981, 1.000] | 200/200 | 1.000 [0.981, 1.000] |
| 3 | 198/200 | 0.990 [0.964, 0.997] | 199/200 | 0.995 [0.972, 0.999] |
| 4 | 190/200 | 0.950 [0.910, 0.973] | 187/200 | 0.935 [0.892, 0.962] |
| 5 | 174/200 | 0.870 [0.816, 0.910] | 179/200 | 0.895 [0.845, 0.930] |
| 6 | 145/200 | 0.725 [0.659, 0.782] | 146/200 | 0.730 [0.665, 0.787] |
| all | 907/1000 | 0.907 [0.887, 0.923] | 911/1000 | 0.911 [0.892, 0.927] |

- max length solved: frozen 6, RL 6
- unparseable fraction of samples: frozen 0.0000, RL 0.0010
- completion tokens mean/max: frozen 73/247, capped fraction 0.0000; RL 72/247, capped 0.0000
- sampler throughput: frozen 58.0 samples/s, RL 45.6
- sources: `baseline/greedy_heldout.json`, `baseline/dumps/greedy_heldout.jsonl`, `evaluate/greedy_heldout.json`, `evaluate/dumps/greedy_heldout.jsonl`

## (c) textbook 72 by reference length

k=64, temperature 1.0, 72 records, 4608 samples per model. Length -1 = the 14 `textbook_train` problems, which carry no reference length.

| length | frozen solved/n | frozen rate [Wilson 95%] | RL solved/n | RL rate [Wilson 95%] |
| --- | --- | --- | --- | --- |
| -1 | 2/14 | 0.143 [0.040, 0.399] | 2/14 | 0.143 [0.040, 0.399] |
| 1 | 1/1 | 1.000 [0.207, 1.000] | 1/1 | 1.000 [0.207, 1.000] |
| 2 | 2/2 | 1.000 [0.342, 1.000] | 2/2 | 1.000 [0.342, 1.000] |
| 3 | 1/1 | 1.000 [0.207, 1.000] | 1/1 | 1.000 [0.207, 1.000] |
| 5 | 1/1 | 1.000 [0.207, 1.000] | 1/1 | 1.000 [0.207, 1.000] |
| 6 | 3/10 | 0.300 [0.108, 0.603] | 3/10 | 0.300 [0.108, 0.603] |
| 7 | 0/4 | 0.000 [0.000, 0.490] | 0/4 | 0.000 [0.000, 0.490] |
| 8 | 0/2 | 0.000 [0.000, 0.658] | 0/2 | 0.000 [0.000, 0.658] |
| 9 | 0/3 | 0.000 [0.000, 0.562] | 0/3 | 0.000 [0.000, 0.562] |
| 10 | 0/5 | 0.000 [0.000, 0.434] | 0/5 | 0.000 [0.000, 0.434] |
| 11 | 0/4 | 0.000 [0.000, 0.490] | 0/4 | 0.000 [0.000, 0.490] |
| 12 | 0/2 | 0.000 [0.000, 0.658] | 0/2 | 0.000 [0.000, 0.658] |
| 13 | 0/3 | 0.000 [0.000, 0.562] | 0/3 | 0.000 [0.000, 0.562] |
| 14 | 0/3 | 0.000 [0.000, 0.562] | 0/3 | 0.000 [0.000, 0.562] |
| 15 | 0/4 | 0.000 [0.000, 0.490] | 0/4 | 0.000 [0.000, 0.490] |
| 16 | 0/2 | 0.000 [0.000, 0.658] | 0/2 | 0.000 [0.000, 0.658] |
| 18 | 0/4 | 0.000 [0.000, 0.490] | 0/4 | 0.000 [0.000, 0.490] |
| 20 | 0/3 | 0.000 [0.000, 0.562] | 0/3 | 0.000 [0.000, 0.562] |
| 22 | 0/1 | 0.000 [0.000, 0.793] | 0/1 | 0.000 [0.000, 0.793] |
| 25 | 0/1 | 0.000 [0.000, 0.793] | 0/1 | 0.000 [0.000, 0.793] |
| 29 | 0/1 | 0.000 [0.000, 0.793] | 0/1 | 0.000 [0.000, 0.793] |
| 33 | 0/1 | 0.000 [0.000, 0.793] | 0/1 | 0.000 [0.000, 0.793] |
| all | 10/72 | 0.139 [0.077, 0.237] | 10/72 | 0.139 [0.077, 0.237] |

- max length solved: frozen 6, RL 6
- unparseable fraction of samples: frozen 0.0577, RL 0.0460
- completion tokens mean/max: frozen 69/198, capped fraction 0.0000; RL 66/191, capped 0.0000
- sampler throughput: frozen 103.4 samples/s, RL 101.6
- sources: `baseline/textbook.json`, `baseline/dumps/textbook72.jsonl`, `evaluate/textbook.json`, `evaluate/dumps/textbook72.jsonl`

### (c) textbook 72 by slice

| slice | frozen solved/n [Wilson 95%] | RL solved/n [Wilson 95%] |
| --- | --- | --- |
| dev58 | 8/58 [0.072, 0.249] | 8/58 [0.072, 0.249] |
| train14 | 2/14 [0.040, 0.399] | 2/14 [0.040, 0.399] |
| all | 10/72 [0.077, 0.237] | 10/72 [0.077, 0.237] |
| reference length >= 7 | 0/43 [0.000, 0.082] | 0/43 [0.000, 0.082] |

## (d) long pool (`rl_targets` stride) pass@32 by true length

k=32, temperature 1.0, 200 records, 6400 samples per model. Length = `L_true`.

| length | frozen solved/n | frozen rate [Wilson 95%] | RL solved/n | RL rate [Wilson 95%] |
| --- | --- | --- | --- | --- |
| 7 | 14/82 | 0.171 [0.105, 0.266] | 11/82 | 0.134 [0.077, 0.224] |
| 8 | 0/44 | 0.000 [0.000, 0.080] | 0/44 | 0.000 [0.000, 0.080] |
| 9 | 0/40 | 0.000 [0.000, 0.088] | 0/40 | 0.000 [0.000, 0.088] |
| 10 | 0/20 | 0.000 [0.000, 0.161] | 0/20 | 0.000 [0.000, 0.161] |
| 11 | 0/5 | 0.000 [0.000, 0.434] | 0/5 | 0.000 [0.000, 0.434] |
| 12 | 0/8 | 0.000 [0.000, 0.324] | 0/8 | 0.000 [0.000, 0.324] |
| 14 | 0/1 | 0.000 [0.000, 0.793] | 0/1 | 0.000 [0.000, 0.793] |
| all | 14/200 | 0.070 [0.042, 0.114] | 11/200 | 0.055 [0.031, 0.096] |

- max length solved: frozen 7, RL 7
- unparseable fraction of samples: frozen 0.0717, RL 0.0778
- completion tokens mean/max: frozen 129/392, capped fraction 0.0000; RL 127/302, capped 0.0000
- sampler throughput: frozen 50.8 samples/s, RL 52.9
- sources: `baseline/long_pool.json`, `baseline/dumps/long_pool.jsonl`, `evaluate/long_pool.json`, `evaluate/dumps/long_pool.jsonl`

## (e) GRPO (`train_summary.json`, `run_manifest.json`, `metrics.jsonl`)

- 50 optimizer steps, 256 completions per step, 424 s wall-clock (includes periodic evals)
- prompt set: 1600 prompts, lengths [3, 4, 5], all from `train.jsonl` (`prompts.jsonl`); leak check passed at start
- trainer: lr 1e-05, beta 0.01, epsilon 0.2, temperature 1.0, group 8, cap 400 tokens, loss_type grpo
- mean reward, first 5 steps: **0.876**; last 5 steps: **0.923**
- max KL over the run: **0.00172**
- within-group reward std: min 0.174, mean 0.315; steps with std == 0: 0
- frac_reward_zero_std (groups with no signal), mean over steps: 0.677
- entropy: first 0.0618, last 0.0616
- completion length mean: first 78, last 74; max over run 226; fraction at cap, max over steps 0.0000
- unparseable fraction, mean over steps 0.0045; invalid fraction mean 0.1166
- grad norm: mean 0.162, max 0.293; learning rate constant 1e-05
- rollout throughput: mean 31.6 completions/s (generation + optimizer step), 8.5 s/step
- completions dumped: 12800 rows in `completions.jsonl.gz`

### (e) mean reward per step (`metrics.jsonl`)

| step | reward | reward_std | kl | entropy | mean len | grad_norm |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.855 | 0.352 | 0.00000 | 0.0618 | 78 | 0.147 |
| 2 | 0.902 | 0.297 | 0.00002 | 0.0677 | 70 | 0.185 |
| 3 | 0.902 | 0.297 | 0.00002 | 0.0716 | 66 | 0.178 |
| 4 | 0.867 | 0.340 | 0.00002 | 0.0636 | 77 | 0.208 |
| 5 | 0.852 | 0.356 | 0.00009 | 0.0643 | 74 | 0.170 |
| 6 | 0.875 | 0.331 | 0.00007 | 0.0673 | 71 | 0.162 |
| 7 | 0.883 | 0.322 | 0.00010 | 0.0706 | 69 | 0.183 |
| 8 | 0.824 | 0.381 | 0.00018 | 0.0664 | 71 | 0.229 |
| 9 | 0.883 | 0.322 | 0.00012 | 0.0687 | 71 | 0.164 |
| 10 | 0.863 | 0.344 | 0.00009 | 0.0675 | 72 | 0.159 |
| 11 | 0.879 | 0.327 | 0.00023 | 0.0690 | 68 | 0.293 |
| 12 | 0.848 | 0.360 | 0.00008 | 0.0628 | 76 | 0.187 |
| 13 | 0.828 | 0.378 | 0.00020 | 0.0742 | 64 | 0.192 |
| 14 | 0.859 | 0.348 | 0.00019 | 0.0702 | 66 | 0.256 |
| 15 | 0.859 | 0.348 | 0.00030 | 0.0678 | 70 | 0.230 |
| 16 | 0.906 | 0.292 | 0.00018 | 0.0587 | 77 | 0.139 |
| 17 | 0.832 | 0.375 | 0.00041 | 0.0704 | 69 | 0.152 |
| 18 | 0.789 | 0.409 | 0.00036 | 0.0697 | 71 | 0.162 |
| 19 | 0.754 | 0.432 | 0.00069 | 0.0759 | 68 | 0.156 |
| 20 | 0.910 | 0.287 | 0.00025 | 0.0697 | 66 | 0.178 |
| 21 | 0.906 | 0.292 | 0.00030 | 0.0682 | 69 | 0.148 |
| 22 | 0.859 | 0.348 | 0.00031 | 0.0636 | 75 | 0.123 |
| 23 | 0.805 | 0.397 | 0.00054 | 0.0687 | 72 | 0.160 |
| 24 | 0.934 | 0.249 | 0.00037 | 0.0681 | 67 | 0.114 |
| 25 | 0.859 | 0.348 | 0.00053 | 0.0781 | 60 | 0.188 |
| 26 | 0.910 | 0.287 | 0.00051 | 0.0545 | 83 | 0.099 |
| 27 | 0.891 | 0.313 | 0.00026 | 0.0651 | 71 | 0.138 |
| 28 | 0.844 | 0.364 | 0.00108 | 0.0638 | 75 | 0.166 |
| 29 | 0.922 | 0.269 | 0.00034 | 0.0651 | 71 | 0.159 |
| 30 | 0.879 | 0.327 | 0.00076 | 0.0618 | 74 | 0.185 |
| 31 | 0.926 | 0.263 | 0.00052 | 0.0594 | 77 | 0.129 |
| 32 | 0.906 | 0.292 | 0.00037 | 0.0627 | 74 | 0.267 |
| 33 | 0.895 | 0.308 | 0.00058 | 0.0630 | 72 | 0.144 |
| 34 | 0.926 | 0.263 | 0.00099 | 0.0713 | 64 | 0.167 |
| 35 | 0.941 | 0.235 | 0.00037 | 0.0659 | 68 | 0.180 |
| 36 | 0.879 | 0.327 | 0.00153 | 0.0644 | 73 | 0.234 |
| 37 | 0.918 | 0.275 | 0.00061 | 0.0619 | 73 | 0.093 |
| 38 | 0.879 | 0.327 | 0.00038 | 0.0600 | 75 | 0.109 |
| 39 | 0.914 | 0.281 | 0.00025 | 0.0673 | 67 | 0.081 |
| 40 | 0.945 | 0.228 | 0.00051 | 0.0598 | 75 | 0.162 |
| 41 | 0.895 | 0.308 | 0.00079 | 0.0608 | 75 | 0.194 |
| 42 | 0.863 | 0.344 | 0.00055 | 0.0618 | 74 | 0.116 |
| 43 | 0.910 | 0.287 | 0.00087 | 0.0695 | 65 | 0.230 |
| 44 | 0.848 | 0.360 | 0.00172 | 0.0637 | 73 | 0.110 |
| 45 | 0.934 | 0.249 | 0.00020 | 0.0650 | 69 | 0.074 |
| 46 | 0.969 | 0.174 | 0.00020 | 0.0708 | 63 | 0.146 |
| 47 | 0.938 | 0.243 | 0.00028 | 0.0624 | 72 | 0.134 |
| 48 | 0.898 | 0.303 | 0.00082 | 0.0625 | 72 | 0.123 |
| 49 | 0.914 | 0.281 | 0.00050 | 0.0612 | 76 | 0.144 |
| 50 | 0.895 | 0.308 | 0.00050 | 0.0616 | 74 | 0.058 |

### (e) periodic greedy eval (`metrics.jsonl`, `eval_periodic.jsonl`; 50 heldout + 20 rl_targets records)

| step | len 2 | len 3 | len 4 | len 5 | len 6 | len 7 | len 8 | len 9 | len 11 | len 12 | all |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 10/10 | 10/10 | 10/10 | 9/10 | 6/10 | 0/10 | 0/3 | 0/5 | 0/1 | 0/1 | 0.643 |
| 10 | 10/10 | 10/10 | 10/10 | 9/10 | 6/10 | 0/10 | 0/3 | 0/5 | 0/1 | 0/1 | 0.643 |
| 20 | 10/10 | 10/10 | 10/10 | 9/10 | 6/10 | 0/10 | 0/3 | 0/5 | 0/1 | 0/1 | 0.643 |
| 30 | 10/10 | 10/10 | 10/10 | 9/10 | 6/10 | 0/10 | 0/3 | 0/5 | 0/1 | 0/1 | 0.643 |
| 40 | 10/10 | 10/10 | 9/10 | 9/10 | 6/10 | 0/10 | 0/3 | 0/5 | 0/1 | 0/1 | 0.629 |
| 50 | 10/10 | 10/10 | 9/10 | 9/10 | 6/10 | 0/10 | 0/3 | 0/5 | 0/1 | 0/1 | 0.629 |

### Gate B metric coverage (`report/metric_coverage.json`)

- all present: **True**; missing: none
