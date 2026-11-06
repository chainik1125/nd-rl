---
written_on: 2026-10-28
written_by: agent:claude
---

# Temperature-matched KL: is RL's drift from pretraining just sharpening?

For an RL checkpoint θ_RL with pretrained start θ_pre, at every proof-body position s of a rollout sampled from π_RL,
KL(π_RL(·|s) ‖ softmax(z_pre(s)/T)) is computed exactly over the vocabulary, and averaged over positions. Reported
per checkpoint: **raw** = the mean at T = 1; **T\*** = the single temperature that minimises the mean (grid of 601
points on log T in [0.05, 20], 1% steps, parabolic refine); **residual** = the mean at T\*. If RL only sharpened the
pretrained model, T\* < 1 and residual ≈ 0. Also recorded: mean per-token entropy of π_RL and of π_pre at T = 1 on the
same positions, both models' −log p of the sampled tokens, and the same fit with π_RL taken at the sampling
temperature 0.8 (`all_t` in `summary.json`).

## What ran (2026-10-28, 19:46–20:15 PDT)

- Checkpoints from HF `runs/long-rl-20261014-*` (the `passk/backfill.py` listing): the whole `v4_untied-ei_s0` curve
  (pretrain, rounds 5, 10, ..., 35, final = after round 36) and `v3_untied-grpo_s0` (pretrain, final = after round 168).
  Each run's own `pretrain.pt` is θ_pre.
- Rollouts: 64 dev theorems per body-line band (7–9 has 61, 35–49 has 44; 425 theorems in all), fixed across
  checkpoints by a key hash, × 8 samples from the checkpoint being scored with the harness's own sampler
  (`harness.sample`: T = 0.8, `n_loops` = 8, irrelevant for untied, caps 64 + 30 × body tokens up to 2,304, seed
  7777), judged by the Lean-only gate (Lean 4.34.0 installed on the pod, as `passk/remote.sh` does). 3,400 rollouts
  per checkpoint.
- Scoring: both models teacher-forced on each rollout, eval mode, `augment=False`, float32 logits from a bf16
  autocast forward; positions = the harness's proof mask (every sampled token, `<eos>` included), never the prompt.
  `tempkl.py` stores per rollout the sums that make KL(T) exact for any T (entropy of π_RL, E_π_RL[z_pre], and
  logsumexp(z_pre/T) on the grid), so any grouping is a sum of rows; `analyze.py` does the fits locally.
- One A40 (`launch.py` → `remote.sh`), 28.3 min of pod time, about $0.28; terminated by the launcher's `finally`.
  Per checkpoint: sampling 43–86 s, scoring 9–22 s.
- Checks: `tempkl.py --smoke` on CPU with the RL weights as θ_RL and the same weights with the output head scaled by
  0.5 as θ_pre gives T\* = 0.5000 and residual 3.8e-8, the exact answer. A second CPU smoke with the real round-5 and
  pretrain checkpoints ran end to end. On the pod, each checkpoint's direct KL at T = 1 matches the reconstruction
  from the stored sums (asserted to 0.1%), and pretrain-vs-itself gives KL 0 and T\* = 1.000.

| file | what |
| --- | --- |
| `tempkl.py` | on the pod: sample, score, store per-rollout sums (`results/<tag>.json`, 39 MB each, not committed) |
| `analyze.py` | the fit (`fit`, `curve`), the groupings, the table; writes `summary.json` |
| `plot.py` | `tempkl.png` (needs `uv run --no-project --with matplotlib,numpy`) |
| `launch.py`, `remote.sh`, `jobs.txt`, `pod.log`, `lean_gate.jsonl` | the pod run and its logs |

## Results

Nats per proof token, over all 3,400 rollouts. `resid%` = residual / raw. H = mean per-token entropy at T = 1.
`acc` = rollouts Lean accepted.

| checkpoint | raw KL | T\* | residual | resid% | H_RL | H_pre | raw nats/proof | tokens/proof | acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v4_untied-ei_s0_pretrain | 0 | 1.000 | 0 | – | 0.163 | 0.163 | 0 | 228 | 0/3400 |
| v4_untied-ei_s0_round005 | 0.262 | 1.437 | 0.218 | 83% | 0.096 | 0.138 | 58 | 222 | 16/3400 |
| v4_untied-ei_s0_round010 | 0.485 | 1.545 | 0.394 | 81% | 0.114 | 0.216 | 134 | 276 | 194/3400 |
| v4_untied-ei_s0_round015 | 0.596 | 1.544 | 0.484 | 81% | 0.105 | 0.230 | 232 | 390 | 517/3400 |
| v4_untied-ei_s0_round020 | 0.794 | 1.637 | 0.627 | 79% | 0.116 | 0.277 | 355 | 447 | 661/3400 |
| v4_untied-ei_s0_round025 | 0.791 | 1.610 | 0.632 | 80% | 0.093 | 0.266 | 363 | 459 | 787/3400 |
| v4_untied-ei_s0_round030 | 0.872 | 1.646 | 0.689 | 79% | 0.101 | 0.287 | 459 | 526 | 822/3400 |
| v4_untied-ei_s0_round035 | 0.957 | 1.706 | 0.737 | 77% | 0.106 | 0.291 | 537 | 561 | 859/3400 |
| v4_untied-ei_s0_final (r36) | 0.943 | 1.689 | 0.736 | 78% | 0.098 | 0.300 | 445 | 472 | 873/3400 |
| v3_untied-grpo_s0_pretrain | 0 | 1.000 | 0 | – | 0.091 | 0.091 | 0 | 169 | 0/3400 |
| v3_untied-grpo_s0_final (r168) | 0.281 | 1.361 | 0.249 | 89% | 0.089 | 0.178 | 70 | 251 | 169/3400 |

With π_RL at the sampling temperature 0.8 instead of 1 (`all_t`): final EI raw 0.957, T\* 1.680, residual 0.756
(79%); GRPO final raw 0.290, T\* 1.339, residual 0.263 (91%). Same picture.

Cross-check: the v4 run's own per-round probes (`summary.json` → `rl.stats.probe.pre_roll`, KL(π‖pre) at T = 1 on
that round's RL-target rollouts) read 0.126, 0.232, 0.408, 0.587, 0.610, 0.679, 0.711 at rounds 5–35 and 0.751 at 36.
Here, on dev rollouts spread evenly over the bands, the same quantity is 1.3–2× that; same trajectory.

### Per band (final EI checkpoint; raw / T\* / resid% / accepted)

| band | 7–9 | 10–14 | 15–19 | 20–24 | 25–34 | 35–49 | 50+ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| v4 final | 0.92 / 1.65 / 81% / 451/488 | 1.10 / 1.75 / 78% / 288/512 | 1.16 / 1.79 / 76% / 87/512 | 1.31 / 1.80 / 77% / 42/512 | 1.33 / 1.78 / 78% / 5/512 | 0.83 / 1.69 / 76% / 0/352 | 0.58 / 1.56 / 80% / 0/512 |
| v4 round005 | 0.57 / 1.58 / 81% / 16/488 | 0.40 / 1.56 / 79% / 0 | 0.27 / 1.49 / 82% / 0 | 0.28 / 1.50 / 81% / 0 | 0.27 / 1.48 / 82% / 0 | 0.18 / 1.40 / 84% / 0 | 0.16 / 1.32 / 86% / 0 |
| v3 GRPO final | 0.50 / 1.43 / 88% / 159/488 | 0.40 / 1.40 / 89% / 10/512 | 0.28 / 1.39 / 88% / 0 | 0.30 / 1.36 / 90% / 0 | 0.28 / 1.35 / 90% / 0 | 0.20 / 1.31 / 89% / 0 | 0.22 / 1.34 / 87% / 0 |

Every band of every checkpoint is in `summary.json` (`band`, `band_ok`); `tempkl.png` panel (d) shows T\* by band
over the EI rounds.

### Accepted vs rejected rollouts

| checkpoint | split | n | tokens/proof | raw | T\* | residual | resid% | H_RL | H_pre | −log p_pre |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v4 round010 | accepted | 194 | 146 | 0.795 | 1.64 | 0.643 | 81% | 0.057 | 0.277 | 0.85 |
| v4 round010 | rejected | 3,206 | 283 | 0.475 | 1.54 | 0.386 | 81% | 0.116 | 0.214 | 0.57 |
| v4 round020 | accepted | 661 | 184 | 1.055 | 1.72 | 0.837 | 79% | 0.048 | 0.341 | 1.10 |
| v4 round020 | rejected | 2,739 | 510 | 0.772 | 1.63 | 0.609 | 79% | 0.122 | 0.271 | 0.87 |
| v4 final | accepted | 873 | 208 | 1.156 | 1.75 | 0.913 | 79% | 0.047 | 0.366 | 1.20 |
| v4 final | rejected | 2,527 | 563 | 0.916 | 1.68 | 0.713 | 78% | 0.104 | 0.291 | 1.00 |
| v3 GRPO final | accepted | 169 | 144 | 0.487 | 1.42 | 0.431 | 89% | 0.055 | 0.244 | 0.54 |
| v3 GRPO final | rejected | 3,231 | 256 | 0.275 | 1.36 | 0.244 | 89% | 0.090 | 0.176 | 0.35 |

Within a band the split is the same way round (final EI, 15–19: accepted 1.48 raw / T\* 1.85, rejected 1.09 / 1.78).

### Shape of KL(T), final EI, all rollouts

T = 0.5: 1.78, 0.8: 1.14, 1.0: 0.943, 1.5: 0.747, **1.69: 0.736**, 2.0: 0.764, 3.0: 1.16, 5.0: 2.31, 20: 4.47. One
convex minimum, well inside the grid, for every group (no fit hit the grid edge). Per-rollout raw KL at the final
checkpoint: 10/50/90/99th percentiles 0.36 / 1.02 / 1.52 / 1.81 nats per token; the top 10% of rollouts carry 32% of
the summed KL, so it is not a few outliers.

## Reading (observations, then interpretation)

Observed:

1. **T\* > 1 everywhere**, 1.44 → 1.71 over the EI rounds and 1.36 for GRPO, never below 1. The best-matching
   pretrained distribution is a *flattened* one, not a sharpened one.
2. **Residual is 77–83% of raw** for EI at every round and 89% for GRPO, and it grows with the rounds in step with
   raw. Temperature explains at most a quarter of the drift.
3. Yet **π_RL is sharper than π_pre on these positions**: H_RL 0.10 vs H_pre 0.30 at the final EI checkpoint, and the
   pretrained model's entropy on RL's rollouts keeps rising with the rounds (0.14 → 0.30) while π_RL's holds at
   ~0.10. On the accepted rollouts π_RL's entropy is 0.047.
4. Raw KL per token rises with target length up to 20–34 lines (1.3 nats/token at the final checkpoint) and falls
   at 35+ (0.58–0.83), where nothing is solved; rollouts there run to 800–1,000 tokens and 9–16% never emit `<eos>`.
5. Accepted rollouts have *higher* per-token KL than rejected ones (1.16 vs 0.92 at the final checkpoint), a higher
   T\*, and the pretrained model's −log p of their tokens is 1.2 nats/token against 1.0 on rejected ones.
6. GRPO after 168 rounds has drifted about as far as EI after 5 (raw 0.28 vs 0.26) and is the closest to pure
   sharpening of anything here, and still only 11% of its KL is temperature.

Interpretation (not checked further): sharpening would mean RL kept the pretrained ranking of next tokens and only
concentrated mass on the pretrained favourite; T\* < 1 and a small residual. What the numbers say instead is that at
a large share of positions π_RL is confident on a token the pretrained model gives little mass to (observation 5:
its −log p of RL's tokens is 1–1.2 nats per token on average, far above its own entropy there). Flattening the
pretrained distribution (T > 1) is the only way a single temperature can reduce that penalty, which is why T\* lands
near 1.7, and the 78% residual is the part no temperature can fix: a change of ranking, i.e. of behaviour. This fits
the long-RL finding that EI reaches 25-line proofs from a ≤ 6-line pretraining set: the proof shapes it learned are
ones the pretrained model does not rank first. The 35+ bands have lower per-token KL simply because neither model
knows what to do there and the long rollouts share the same low-information rambling. Note too that per-token
entropies are small for both models (0.1–0.3 nats); the tokenizer makes most proof tokens near-forced given the
preceding ones, so the KL is concentrated on the few decision tokens per line.

## Not done / caveats

- Only seed 0 of each run; v4 seed 1 (186 dev) and the v2/v5 runs are untouched. The looped arm was not scored.
- Dev theorems, not holdout, and 64 per band; the 7–9 and 35–49 bands have only 61 and 44 dev theorems.
- The fit is per group over proof-token positions, with a single scalar T; nothing position-dependent (e.g. a
  temperature per line position or per rule token) was tried.
- The "KL of a temperature" direction is KL(π_RL ‖ π_pre^T) only, as specified; the reverse direction was not fitted.
- `results/` is 431 MB (the 601-point logsumexp grid per rollout) and is not committed; `summary.json` (113 KB) holds
  every fitted group.
