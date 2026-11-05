---
written_on: 2026-11-04
written_by: agent:claude
---

# GRPO on Dan's knockout bases: can GRPO learn proof by contradiction where EI did not?

**Status.** RUNNING. The primary lines end around 10:30–12:20Z on 2026-11-05. Nothing is pushed.

**Question (Dan).** "It would be cool to know if GRPO can actually learn proof-by-contradiction
off the 0% proof-by-contradiction pretrain (where EI failed to learn it)." Leon: "run this too".

**Answer.** PENDING.

**Working folder:**
[`code/experiments/current/knockout_danbase_20261104/`](../../code/experiments/current/knockout_danbase_20261104/README.md)
on branch `leon/knockout-grpo`. It holds the pre-registration (frozen before any line ran), the
run log, the scripts and the numbers.

![RL needed to reach Dan's bar against σ's share of the pretraining proofs, his EI and GRPO on his own bases](rl_needed_danbase.png)

## Design

- **The same bases as his EI.** These are his create-or-elicit Stage-1 checkpoints
  `stage1_ko_s0–s2`, on K12⁻ with no σ proofs, plus his full base `stage1_full_s3` as a
  reference. They came read-only from his bucket, and their md5s match his registry. His own
  round-1 draw, re-run with his code, gives 0 / 0 / 0 of the 106 on the knockout bases (his
  0 / 0 / 0) and 17 on full_s3 (his 13).
- **The same problems, environment and measure.**
  - **Problems:** his 4,495 rl_targets.
  - **Environment:** his state environment with its caps of 256 tokens per action and 48 steps.
  - **Held-out set:** his 106 strict σ theorems, read with his own code (`generate`, `judge`)
    at T 0.8.
  - **Measure:** his own, a cumulative count of k 32 draws. We take 16 draws, one per round of
    his same-seed ladder, each at the latest checkpoint whose compute (on his rule) is within
    his x for that round.
  - **Bar:** his, 14.
- **Only the algorithm differs.** We use Leon's GRPO (`code.thinking.grpo` at f35d3daf, no
  thinking): 256 prompts × 16 per iteration, T 1.0, PPO clip, unlikeliness reward, no KL and no
  replay. His EI fine-tunes on its accumulated verified proofs plus 20,000 replayed pretraining
  records per round.
- **Arms.**
  - **Primary:** the knockout base with his targets, 3 seeds.
  - **Secondary:** the knockout base with our σ-rich pool (L2_pool, 39.5 % of targets
    strict-σ), 3 seeds. It asks whether GRPO learns σ at all when the problems reward it.
  - **Reference:** the full base with his targets, 1 seed.
- **Budget per line.** A line ends at the first of three events:
  - one ladder of his compute (his round-16 x, 5.80–5.93);
  - 15 GPU-h;
  - reaching the bar and holding it.
- **Machines.** The lines ran on serrano, temescal and ynez under a supervisor. It respawned
  dead lines from their checkpoints, moved lines off GPUs that other users took, and gave the
  primary lines priority (`log.md`).

## Results (observed)

<!-- AUTO:lines -->
| line | arm | base (first 32 of the 106) | his measure, by matched round | reached his bar (14)? | x_dan at the end (his round-16 x) | ended | GPU-h | σ in accepted training proofs: all / last 50 iterations |
|---|---|---|---|---|---|---|---|---|
| dko_s0 | **primary**: knockout base, his 4,495 targets | 0 | 0, 0, 0, 0 | no: 0 of 106 by x_dan 1.21 | 1.21 (5.84) | running | 3.9 | 0.07 % / 0.10 % |
| dko_s1 | **primary**: knockout base, his 4,495 targets | 1 | 1, 1, 1, 1, 1 | no: 1 of 106 by x_dan 1.64 | 1.64 (5.93) | running | 3.8 | 0.05 % / 0.05 % |
| dko_s2 | **primary**: knockout base, his 4,495 targets | 0 | 0, 0, 0, 0 | no: 0 of 106 by x_dan 1.32 | 1.32 (5.90) | running | 3.4 | 0.03 % / 0.01 % |
| dfull_s3 | reference: full K12 base, his targets | 15 | 15 | at the bar before RL (15 at round 1) | 0.26 (5.80) | reached (Dan's measure at x_dan 0.000) and held | 0.5 | 4.45 % / 4.45 % |
| dkoour_s0 | secondary: knockout base, our σ-rich pool | 0 | 0, 1, 1, 1, 1 | no: 1 of 106 by x_dan 1.48 | 1.48 (5.84) | waiting | 3.2 | 0.00 % / 0.00 % |
| dkoour_s1 | secondary: knockout base, our σ-rich pool | 0 | 0, 0, 0, 0 | no: 0 of 106 by x_dan 1.24 | 1.24 (5.93) | waiting | 2.5 | 0.00 % / 0.00 % |
| dkoour_s2 | secondary: knockout base, our σ-rich pool | 0 | 0, 0, 0, 0, 0 | no: 0 of 106 by x_dan 1.83 | 1.83 (5.90) | running | 3.1 | 0.00 % / 0.00 % |
<!-- /AUTO:lines -->

His EI on the same bases, from his registry rows:

<!-- AUTO:dan -->
| his ladder | held-out count by round (rounds 1–16) | x at round 16 | crossing |
|---|---|---|---|
| la_ko_s0 | 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 | 5.84 | not reached by round 16 |
| la_ko_s1 | 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 3 | 5.93 | not reached by round 16 |
| la_ko_s2 | 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 | 5.90 | not reached by round 16 |
| la_full_s3 | 13, 40, 51, 55, 58, 58, 60, 62, 63, 64, 64, 64, 65, 65, 65, 65 | 5.80 | round 2, x 0.39 |
<!-- /AUTO:dan -->

![His measure against compute on his rule: GRPO on his bases beside his EI ladders](cum_vs_x.png)

![σ's share of the accepted training proofs, and the snapshot first-32 count, against compute](sigma_danbase.png)

## Pre-registered predictions

PENDING.

## Caveats

- **The algorithms differ in more than the update rule.** Training temperature is 1.0 for
  GRPO and 0.8 for his EI. GRPO has no pretraining replay. A GRPO iteration samples 4,096
  attempts; an EI round samples 143,840.
- **Our draws are fresh draws, not his.** RTX 3090 sampling does not reproduce his A40 stream;
  his full_s3 round-1 count is 13, ours 17 and 15. Our reads sample only the 106 (64 per
  theorem; the first 32 form the draw), while his sample all 832 transfer theorems.
- **Matched draws.** Each of our 16 draws comes from the latest checkpoint at or below his
  round's x. Our model at that point has trained on all the compute spent so far, while his
  round-r draw comes before round r's fine-tune.
- **Machine events.** Lines were respawned and moved: one OOM, other users' processes, ynez's
  reboot, and a GPU reserved for another experiment. Lost iterations are listed below.

## Compute

<!-- AUTO:compute -->
- Lines: 20.2 GPU-h in all (segment wall time, reads included), plus about 0.4 GPU-h for the base check and 0.2 for the pilot.
- Iterations lost to deaths and moves (superseded rows): {'dko_s0': [[11, 11]], 'dko_s2': [[1, 8]]}.
<!-- /AUTO:compute -->

## Configuration

- **Code:** GRPO is `code.thinking.grpo` at f35d3daf, with the `snap_iters` and
  `danbase_log` patches (logging only). His code is state_train / state_ladder_ei / eval_set at
  13e41bbd, and coe_sigma / coe_rounds at 624982dc.
- **Records:** PREREG.md sha256 is `dac8e2df…`. The run log, every respawn and every move are
  in the working folder's `log.md`.
- **Artifacts:** `/scratch/cowifornia/nd-rl/knockout_danbase/` on serrano, temescal and ynez.
  The supervisor's state and events are in `sup/`.
