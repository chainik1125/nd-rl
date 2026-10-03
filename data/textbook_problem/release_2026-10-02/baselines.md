# Baselines (aggregate), 2026-10-02

Scored on the sets as they stood before the curry-aware re-cut (candidate_v0 104, candidate_v1 80); k = 32 samples per problem, temperature 1.0; every proof judged by the pinned nd_verify through strict_verify. avg = mean over problems of the fraction of samples accepted; solved = problems with at least one accepted sample. One seed per pretrain, two seeds for the RL model. Details, per-problem rows and the leak audit: `experiments/textbook-problem/eval_scale_pretrain/` and `eval_dan_best_state/`.

| set | n | pretrain d384 avg / solved | pretrain d768 avg / solved | RL best-state r8 s0 avg / solved | RL s1 avg / solved |
|---|---|---|---|---|---|
| candidate_v0 | 104 | 0.111 / 22 | 0.086 / 17 | 0.341 / 76 | 0.344 / 73 |
| candidate_v1 | 80 | 0.050 / 11 | 0.058 / 11 | 0.356 / 49 | 0.362 / 50 |
| hand_proved_14 | 14 | 0.000 / 0 | 0.000 / 0 | 0.000 / 0 | 0.000 / 0 |
| batch3 (21 incl. the 4 hand-proved) | 21 | 0.000 / 0 | 0.002 / 1 | 0.298 / 10 | 0.327 / 12 |
| Lean-sourced (not in this release) | 50 | 0.088 / 11 | 0.091 / 9 | 0.401 / 33 | 0.383 / 31 |
| derived (18 feedable nuances, pretrain) | 202 | 0.038 / 20 | 0.036 / 17 | — | — |
| derived (13 plain nuances, RL) | 132 | — | — | 0.264 / 69 | 0.287 / 66 |
| textbook 72 (sealed, aggregate) | 72 | 0.104 / 12 | 0.129 / 11 | 0.371 / 49 | 0.378 / 47 |

Checkpoints (sha256):

- `scale-d384-p1200_s0/pretrain.pt`: 447fab6c269017669427bfc8cd3101d398994ec7f6ad82d5e23b4c75e4a0060c
- `scale-d768-p1200_s0/pretrain.pt`: 5297ea76bc87c479090e2cb4ab5850baa3d73126090e6862b96a8c60972f2f79
- `best-state/ckpts/bs/ladder/la_T1_best12_s0_r8.pt`: a992485e436a85bbf51d7a1ffdf5b0b8fe789433c11f6faa4568413d2ca75971
- `best-state/ckpts/bs/ladder/la_T1_best12_s1_r8.pt`: fa9d2528e0659891958a344a579d08c501aacfee636f93462ae015f7267e23df
- `best-state/ckpts/bs/ladder/la_T1_best12_s2_r8.pt`: 5cdf8cf78b29ee8f020c81634e617dbea9b4f373349937323012f7cc306126e4

Pretrains: Robbie's `scale-*-p1200` runs of 2026-09-30 (HF `robbiethompson2018/nd-rl-checkpoints`), lean_seq tokens, 400-token cap. RL: Dan's best-state cap-12 T1 ladder round 8 (HF bucket `dan-pandori/nd-rl`, done 2026-10-01), proof-state environment with Lean in the loop, 48 steps; the derived prefix nuances and the three marker-word nuances could not be fed to it. Overlap between every set and the RL model's training pools is at most 3 problems per set (`eval_dan_best_state/run_best12_r8/leak_audit.md`).
