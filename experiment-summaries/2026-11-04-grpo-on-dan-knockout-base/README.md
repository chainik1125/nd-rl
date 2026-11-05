---
written_on: 2026-11-04
written_by: agent:claude
---

# GRPO on Dan's knockout bases: can GRPO learn proof by contradiction where EI did not?

**Status.** The primary and reference arms are DONE; the last primary line ended at 12:01Z on
2026-11-05. Two secondary lines are still running to their cap on Runpod A40 pods: dkoour_s0
(about 15:10Z) and dkoour_s1 (about 16:10Z). **Their final numbers are pending.** Nothing is
pushed.

**Question (Dan).** "It would be cool to know if GRPO can actually learn proof-by-contradiction
off the 0% proof-by-contradiction pretrain (where EI failed to learn it)." Leon: "run this too".

**Answer (primary arm).** No. GRPO did not learn proof by contradiction from his 0 %-σ
knockout bases within one ladder of his compute. We read it on his own measure: his cumulative
count of k 32 draws on his 106 strict σ held-out theorems, against his bar of 14.
- **Counts.** His three bases with his 4,495 targets reached 0, 1 and 0 of the 106. That
  matches his EI on the same bases (0, 3 and 0 by his round 16).
- **The one solve.** dko_s1's single theorem was already solved by its base before any RL. No
  GRPO checkpoint solved any of the 106 in its first 32 draws at its last read.
- **σ in training.** σ stayed at 0.05–0.09 % of the accepted training proofs throughout.
- **Compute.** dko_s1 and dko_s2 ran to his round-16 x. dko_s0 stopped at the 15 GPU-h cap at
  x_dan 5.07 of 5.84 (87 % of his ladder).

**Secondary arm: in progress, an observation only.** Here the knockout base trains on our
σ-rich pool.
- **dkoour_s0.** σ in its accepted training proofs rose from about 1 % to 5–7 % over
  iterations 384–395 (x_dan about 4.5). Its it-390 snapshot then solved 3 new theorems of the
  106 in its first 32 draws (E32 2.5). On his matched-draw measure that read is the round-13 draw
  (his x 4.55). It is counted once the line has run past 4.55 and the it-400 read lands; the line
  then stands at 4 (1 + 3).
- **dkoour_s1 and dkoour_s2.** Both are at 0. dkoour_s2 is finished.
- Whether dkoour_s0 reaches 14 by his round-16 x (5.84) is pending.

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
| dko_s0 | **primary**: knockout base, his 4,495 targets | 0 | 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 | no: 0 of 106 by x_dan 5.07 | 5.07 (5.84) | GPU-h cap: 15.0 ≥ 15 | 15.1 | 0.09 % / 0.14 % |
| dko_s1 | **primary**: knockout base, his 4,495 targets | 1 | 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 | no: 1 of 106 by x_dan 5.93 | 5.93 (5.93) | compute cap: x_dan 5.93 ≥ his round-16 x 5.93 | 14.3 | 0.05 % / 0.04 % |
| dko_s2 | **primary**: knockout base, his 4,495 targets | 0 | 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 | no: 0 of 106 by x_dan 5.94 | 5.94 (5.90) | compute cap: x_dan 5.92 ≥ his round-16 x 5.90 | 11.8 | 0.07 % / 0.07 % |
| dfull_s3 | reference: full K12 base, his targets | 15 | 15 | at the bar before RL (15 at round 1) | 0.26 (5.80) | reached (Dan's measure at x_dan 0.000) and held | 0.5 | 4.45 % / 4.45 % |
| dkoour_s0 | secondary: knockout base, our σ-rich pool | 0 | 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 | no: 1 of 106 by x_dan 4.60 | 4.60 (5.84) | running | 9.9 | 0.21 % / 1.30 % |
| dkoour_s1 | secondary: knockout base, our σ-rich pool | 0 | 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 | no: 0 of 106 by x_dan 4.21 | 4.21 (5.93) | running | 9.3 | 0.00 % / 0.00 % |
| dkoour_s2 | secondary: knockout base, our σ-rich pool | 0 | 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 | no: 0 of 106 by x_dan 5.94 | 5.94 (5.90) | compute cap: x_dan 5.94 ≥ his round-16 x 5.90 | 10.0 | 0.00 % / 0.00 % |
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

Scored against PREREG.md (frozen before any knockout line was read).

| | prediction | probability | outcome |
|---|---|---|---|
| P1 | primary: no dko seed reaches his bar (14, cumulative) within one ladder | 70 % | **held**: 0, 1, 0. dko_s0 ended at the GPU-h cap at 87 % of its ladder. |
| P2 | primary: σ stays under 1 % of the accepted training proofs on all three dko seeds | 75 % | **held**: 0.09 %, 0.05 % and 0.07 % overall; at most about 0.2 % in any 50-iteration window |
| P3 | secondary: at least 1 of 3 dkoour seeds reaches his bar within one ladder | 35 % | **pending**: dkoour_s2 did not (0); dkoour_s0 is at 1, and 4 once its round-13 draw counts; dkoour_s1 is at 0 |
| P4 | secondary: σ exceeds 5 % of the accepted training proofs on at least 1 dkoour seed | 50 % | **met per iteration** on dkoour_s0 (5–7 % over iterations 391–394); the run-level share is still 0.21 %. Final pending. |
| P5 | reference: dfull_s3 reaches his bar at its first matched draw after the base (round 2) | 80 % | **not met as registered**: it was already at the bar at round 1 (base draw 15, his 13), so the line ended "reached and held" before his round-2 x and has no round-2 draw |
| P6 | pass@1 reward on Dan's pool passes 0.8 within 50 iterations on every dko seed | 70 % | **held**: it passed at iterations 12, 16 and 13 |

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
- **dko_s0 ran out of GPU-h before his compute.** It stopped at the pre-registered 15 GPU-h
  cap at x_dan 5.07 of his round-16 x 5.84. His EI on the same base had 0 through round 16.
- **dko_s1's 1 is the base's.** Its round-1 draw (the base) solved that one theorem. No
  GRPO checkpoint solved any of the 106 in its first 32 draws at its last read; the cumulative
  count keeps the base's solve.
- **Three secondary lines ran partly on Runpod A40 pods** (rp-s0, rp-s1 and rp-s2), with the
  same code and environment; the files were copied by the coordinator on Leon's approval. An
  A40 iteration took 65–135 s, against 60–90 s on the lab's GPUs.
- **Machine events.** Lines were respawned and moved: one OOM, other users' processes, ynez's
  reboot, and a GPU reserved for another experiment. Lost iterations are listed below.

## Compute

<!-- AUTO:compute -->
- Lines: 71.0 GPU-h in all (segment wall time, reads included), plus about 0.4 GPU-h for the base check and 0.2 for the pilot.
- Iterations lost to deaths and moves (superseded rows): {'dko_s0': [[11, 11], [501, 501]], 'dko_s2': [[1, 8]]}.
<!-- /AUTO:compute -->

## Configuration

- **Code:** GRPO is `code.thinking.grpo` at f35d3daf, with the `snap_iters` and
  `danbase_log` patches (logging only). His code is state_train / state_ladder_ei / eval_set at
  13e41bbd, and coe_sigma / coe_rounds at 624982dc.
- **Records:** PREREG.md sha256 is `dac8e2df…`. The run log, every respawn and every move are
  in the working folder's `log.md`.
- **Artifacts:** `/scratch/cowifornia/nd-rl/knockout_danbase/` on serrano, temescal and ynez.
  The supervisor's state and events are in `sup/`.
