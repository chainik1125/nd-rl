---
written_on: 2026-11-07
written_by: "agent:claude"
papers:
  - zhaiDoesRLExpand2026
---

# Zhai et al. 2026: one tool-use RL run, three task categories, k = 64

Paper: [@zhaiDoesRLExpand2026] (proposed key)
Source: arXiv 2604.14877v1 (16 Apr 2026); text `~/onv_sources/DT_2604.14877v1.txt`. Ledger `DT-20`–`DT-34`, `DT-123`–`DT-125`, `DT-134`.

## Setup card

| item | value |
|---|---|
| model, start | Qwen2.5-7B-Instruct (instruct) with a search-tool prompt [DT-22]; RL is LoRA rank 16 on one 48 GB GPU [DT-24] |
| RL share of pretraining | not computed (not in rl-compute-share's table) |
| RL steps | 1,000 GRPO steps (10 epochs over 200 HotPotQA problems) [DT-134]; G = 8, binary exact-match reward [DT-23] |
| domains | A: MATH-500 without tools (not trained); B: HotPotQA comparison; C: HotPotQA bridge (B and C trained) |
| largest k | 64 (n = 64 per problem and depth) [DT-25] |
| grading | exact match after light normalization (answer match) |
| controls | symmetric budgets: yes, n = 64 for every model [DT-25]; format tolerance: partly (light normalization only); guessing: no; contamination: no (not discussed) |

## Learnings

**Observed.** At PASS@(64,5), against the base:
- Category A: the boundary is unchanged [DT-26].
- Category B: +5 / −1 [DT-27].
- Category C: +5 / −1 [DT-28]; scores 0.81 vs 0.77 (base) and 0.73 (SFT) [DT-124].
- Matched-data SFT on C: −7 / +3 [DT-29].
- At k = 1 the RL agent is slightly below the base on C [DT-125].
- Between steps 850 and 1000, B and C rise and A stays flat (0.85 / 0.84) [DT-32].

**Paper interprets.** Tool-use RL "genuinely enlarges" the boundary, specifically on compositional sequential retrieval [DT-20][DT-21]. Category A replicates Yue et al.'s null [DT-30]. RL "meaningfully teaches new capabilities" when compositional solutions exist [DT-123].

**Our interpretation.**
- *Which split holds.* RL's boundary gain over the base is the same on B and C (+5 / −1 each, [DT-27][DT-28]). What separates B from C is SFT, not RL. So "specific to compositional tasks" rests on the RL-vs-SFT contrast. Within this one RL run, the domain split that holds is trained retrieval (gains) against untrained maths (none).
- *How big.* The reach is 4–5 problems in 100 at k = 64.
- *What the bootstrap can show.* The bootstrap redraws each problem at its observed rate [DT-31]. A problem the base solved 0 / 64 times therefore stays unsolved in every replicate. The confidence interval assumes the 0 / 64 labels are right, and that is exactly what fresh draws or larger base budgets would test.

## Evidence and limitations

- One base model, a 10-paragraph BM25 corpus and one tool [DT-33]; no temperature sweep [DT-34].
- No repeated RL runs are reported (our grep for "seed" finds only a data seed and a sub-sampling seed).
- The step-850 / step-1000 pair [DT-32] is the paper's only trajectory evidence. On C the RL-over-base gap is about 1 point at step 850 and 4 at step 1000 [DT-32][DT-124], so the headline depends on the last 150 steps.
- In Table 1, the SFT column for C reads −4 under a set-size header (|B_SFT \ B_base|); it is the net. This is our reading of the pdftotext layout.
- Curves beyond the quoted values are figure only.

## Connections and questions

- **C2: extends.** One RL run leaves maths unchanged. Here the maths is untrained [DT-26]; in our run, AIME-style maths is in Dolci-Think-RL and still shows almost no reach, which is the stronger null.
- **C3: replicates the direction.** RL-only problems appear on trained task types at k = 64 [DT-27][DT-28]. The magnitude is 5 % (C3: ≈ 20 %), with no hand check and no fresh draws.
- **C7: extends.** k ≤ 64, no confirmation on fresh draws, and a bootstrap that cannot resolve 0 / 64 base problems [DT-25][DT-31]. Our symmetric confirmation addresses exactly this gap.
- **C1, C4, C5, C6, C8: unrelated.** No RL-Zero, no gaming audit, no budget or hint tests.
- Question: would the 5 RL-only bridge problems survive 1,024 base draws (cf. C8 Test 3)?
