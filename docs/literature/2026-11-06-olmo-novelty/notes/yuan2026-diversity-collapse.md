---
written_on: 2026-11-07
written_by: "agent:claude"
papers:
  - yuanUnderstandingDiversityCollapse2026
---

# Yuan et al. 2026: diversity collapse as overtraining, with a noise test for boundary entries

Paper: [@yuanUnderstandingDiversityCollapse2026] (proposed key)
Source: arXiv 2606.15455v1 (13 Jun 2026); shared text `~/onv_sources/2606.15455v1.txt` (HTML rendering). Screened
earlier by rl-compute-share ("sections"). Ledger `DT-58`–`DT-69`, `DT-117`–`DT-119`, `DT-129`, `DT-133`.

## Setup card

| item | value |
|---|---|
| model, start | Qwen2.5-Math-7B (pretrained, maths-specialised) [DT-58] |
| RL share of pretraining | not computed (not in rl-compute-share's table) |
| RL steps | 140 steps, ≈ 20 epochs over the MATH training set [DT-58][DT-118]; n = 8 rollouts [DT-117] |
| domain | maths (MATH train; held-out AIME 2025, MinervaMath, OlympiadBench) |
| largest k | 256 (N = 256 samples, temperature 0.6) [DT-68] |
| grading | any boxed expression matching after normalization [DT-69] |
| controls | symmetric budgets: yes, N = 256 for the base and every checkpoint [DT-68]; format tolerance: yes, any-boxed [DT-69]; guessing: no (not discussed); contamination: no (not discussed); resampling-noise test: yes [DT-62] |

## Learnings

**Observed.**
- Epochs 5 → 20: pass@1 moves 37.3 → 37.7 while pass@256 falls 75.2 → 70.6 [DT-59].
- At epoch 20, 38 of 265 base-unsolved held-out problems are inside the k = 256 boundary [DT-63]. Only 3 of these have ≥ 5 successes [DT-64].
- Losses: 152, for a net of −114 [DT-65].
- Training only on zero-success problems lifts pass@256 above the base [DT-60].

**Paper interprets.**
- Aggregate decline "does not by itself mean" that no new reasoning occurred [DT-119].
- RLVR does expand the boundary, but overtraining on solved problems masks the gain [DT-61].

**Our interpretation.**
- *Entries are mostly weak.* The paper's own noise test undercuts its headline. At the last checkpoint 35 of 38 entries have 1–4 successes in 256 [DT-64], and the paper shows a single success is p = 0.5 under no change [DT-62]. The robust result is the loss side (152 losses against 38 entries).
- *Contrast with our runs.* That loss side contrasts with C1 and C2, where nothing is lost. The settings differ in base model, data and schedule (20 epochs over the MATH training set [DT-58]), and which of these matters is open.
- *Shared design.* Testing entries and losses against the same noise model mirrors our symmetric confirmation. They compute an exact test; we re-draw.

## Evidence and limitations

- A single training run per method [DT-66]; one training set and 7B models only [DT-67].
- Six evaluated checkpoints from the authors' own run [DT-133]. No public intermediate checkpoints are used.
- The any-boxed rule [DT-69] is more lenient than a last-answer grade. The paper reports that it raises the base's scores on every benchmark and helps the trained models too [DT-129]. This is the format effect that C7's final-answer-only grade beside RL's grader controls.
- No hand checks of entries, and no consideration of guessing at k = 256 on AIME's integer answers.

## Connections and questions

- **C1: contradicts.** Standard RLVR here loses 152 base-solvable problems at k = 256 [DT-65]; C1 has reverse 0 at k ≤ 512. The reach side agrees: noise-tested entries are rare [DT-64], and C1's base reaches every RL-only solve.
- **C2: contradicts, for the same loss reason.** C2's one reverse problem is solved within 56 tries.
- **C3: extends.** Noise-tested maths reach is small: 3 of 265 base-unsolved problems with ≥ 5 successes [DT-63][DT-64]. C3's IFBench reach is much larger, but the denominators differ (base-unsolved problems here, all prompts in C3).
- **C7: replicates.** The same symmetric logic for entries and losses, with an exact resampling test [DT-62].
- **C8: extends.** The exact test shows why one-success entries need more draws; C8 Test 3 (1,024 draws) runs them.
- **C4, C5, C6: unrelated.**
- Question: does an Olmo checkpoint series show the same entry-then-exit churn [DT-63][DT-65] between s0100, s0750 and s2300?
