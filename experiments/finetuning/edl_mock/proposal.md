# edl_mock — prequential EDL of never-solved proofs under Dan's base and ladder checkpoints

Written 2026-10-10, before any run. Laptop CPU only, $0. Workstream `finetuning`, plan measurement M2
(revised: with starting-loss control) and a first reading for M1 (restricted-parameter arm).

## Hypothesis

H-EDL. Under a fixed fine-tuning algorithm A, the excess description length per example of a stream of
verified proofs of theorems the Stage-1 base never solved falls monotonically with the amount of EI:
EDL/example(base) > r2 > r4 > r8. On a control stream the base already solves, EDL/example ≈ 0 for every
checkpoint. The rival reading (Leon, benchmarking): EDL is a rescaling of the starting loss, i.e. the
ratio EDL / (first-block loss − L_test) is the same for every checkpoint and tells us nothing new.

H-dim (signs of life). 10,000 random weights (0.1 % of 9.56M) at lr 3e-3 give at least 70 % of the
full-parameter reduction in prequential cost on the base.

## Data (built by `code/finetuning/edl_stream.py`, manifest in `data/manifest.json`)

- Stream **new**: 256 `rl_targets` theorems not solved by the seed-0 base in ladder round 1 (32 samples,
  T 0.8) and solved by the seed-1 ladder by round 8; one proof each, the first in seed 1's `found_8`, so no
  seed-0 checkpoint trained on the exact string. Pool 800; 0 rejected by the pinned verifier; 0 renaming-class
  matches with the sealed 72, the test prompt files or validation_36. 128 train (the stream), 128 held-out.
- Stream **ctrl**: 256 theorems the seed-0 base solved in round 1, proofs from seed 1's round 1. Pool 3,307;
  0 rejected, 0 leaks. 128 / 128.
- The stream files are Dan's proofs and stay gitignored; md5s are in the manifest.

## Method (`code/finetuning/edl_mock.py`)

Prefixes n ∈ {0, 8, 16, 32, 64, 128}. For each n: reload the starting weights, train on the first n examples
with algorithm A, then code the next block (sum of next-token NLL over action tokens, Dan's `loss_on`,
fixed renaming seed per block). θ* = A on all 128; L_test = its per-token loss on the 128 held-out.
EDL = Σ block nats − (stream action tokens) × L_test, reported in bits per example. Also logged per
block: bits/example, held-out loss (the starting-loss control is the n = 0 block).

| arm | A | lr | epochs per fit | checkpoints | streams |
| --- | --- | --- | --- | --- | --- |
| full | AdamW all weights, batch 8 proofs, clip 1, constant lr | 3e-4 | 2 | base, r2, r4, r8 (seed 0) | new; ctrl for base and r8 |
| m10000 | same, gradient masked to 10,000 random weights (mask seed 0) | 3e-3 | 8 | base, r8 | new |

Recipe choice comes from today's step sweep (log 2026-10-10 afternoon): full held-out loss peaks at ≈ 2
passes; 10k weights still improving at 8 passes. One stream ordering, one mask seed: signs of life only.

## Expected numbers (from the step sweep: base 0.081 nats/token at n = 0, 0.059 after 2 passes on 128; ≈ 253 scored tokens per proof)

- base / new: first block ≈ 29 bits/example, L_test ≈ 21 bits/example, EDL ≈ 3–6 bits/example.
- r8 / new: first block ≈ 15–22 bits/example, EDL ≈ 1–3 bits/example. r2, r4 in between, in order.
- ctrl: first block ≈ 14 bits/example, EDL ≈ 0–1 bits/example for base and r8.
- m10000 / base: prequential cost reduction ≥ 70 % of the full arm's.

## What we conclude

- Monotone EDL with ctrl ≈ 0: the EDL measurement works at this scale; promote M2 to the three-seed
  version over all eight rounds and both GRPO and EI (plan F2), with the starting-loss ratio reported.
- EDL ∝ starting loss with the same ratio across checkpoints: EDL adds nothing over the n = 0 loss here;
  M2 stays at the ratio form only and the budget goes to M1/M5.
- Non-monotone or ctrl ≫ 0: inspect the 20 worst-coded proofs; suspect the stopping rule (A) first.

## Budget and stop rule

≈ 8 runs, 6–15 min each at 2 threads, 3–4 in parallel: under 1 hour of laptop CPU, $0. Stop and quarantine
a run that exceeds 30 min or logs NaN. Kill only the PIDs in `runs/pids.txt`.
