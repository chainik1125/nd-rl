---
written_on: 2026-11-07
written_by: "agent:claude"
papers:
  - wuInvisibleLeashWhy2025
---

# The Invisible Leash (v5): empirical support, format tolerance, and what it does to ProRL

Paper: [@wuInvisibleLeashWhy2025] (proposed key)
Source: https://arxiv.org/abs/2507.14843v5 (v5, 2026-10-14, still the latest). Read in full. Ids are in `ledger/IL.tsv`.

## Setup card

| item | value |
|---|---|
| models, start | ProRL-1.5B v1 (main) and v2, from distilled R1-Distill-Qwen-1.5B [IL-8, IL-9]; 7B–14B AceReason, Skywork, Phi4 [IL-10]; OLMo-2-1B from **DPO** [IL-52] |
| RL share | rl-compute-share: ProRL v1 2.6 × 10⁻³ – 1.2 × 10⁻², v2 4.4 × 10⁻³ – 2.1 × 10⁻² (`compute_table.csv` lines 18–19); others not computed |
| RL steps | re-tests; v2 = 3,000 incl. v1's 2,000 [IL-192]; own DAPO run, 300 steps [IL-198] |
| domains | maths, SimpleQA, LiveBench, SciBench, LiveCodeBench, Reasoning Gym, VLM [IL-12] |
| largest k | 16,384 (boxnet only) [IL-25]; maths 8,192; non-maths 2,048 [IL-12, IL-22]; OLMo, SFT and checkpoint runs 256 [IL-13] |
| grading | benchmark checkers; GPT-4.1 for SimpleQA [IL-65]; format-tolerant extraction on Reasoning Gym [IL-36] |
| controls | symmetric budgets **yes** [IL-14]; format tolerance **partly** (Reasoning Gym only; one base-only step) [IL-38, IL-39]; guessing **no** [IL-61]; contamination **no** (not discussed) |

## Learnings

**Observed.**
- Empirical support: at least one verified success at a stated budget; pairs give preservation, expansion, shrinkage
  and out [IL-1, IL-2, IL-5]. v5 concedes that exact support is never empty [IL-3, IL-4].
- ProRL v1 gains 36 questions and loses 163; v2 gains 48 (39 non-maths) and loses 175 [IL-16–IL-19]. NDR ≤ 0.02
  [IL-20]. OLMo-2-1B (DPO → RLVR, k 256): 83 gains, 104 losses [IL-53, IL-54].
- **boxnet, verified:** base R1-Distill-1.5B 0.345 against ProRL-**v1** 0.492 at k 16,384 (0.024 vs 0.097 at k 1)
  [IL-24–IL-26].
- Other expansion tasks: arc_1d 0.570 vs 0.680 at k 2,048; graph_color_vertex20 0.901 vs 1.000 at k 4,096 [IL-27,
  IL-28].
- Nine tasks reach full base coverage, among them dice (0 at k 1 → 1.000 at k 8,192) [IL-31–IL-33].
- On leg_counting, family_relationships and power_function the base overtakes ProRL by k 1,024 [IL-29, IL-30].
- Format: base answers missed the verifier format, while ProRL (trained on Reasoning Gym) followed it. The authors
  clarified the prompts and widened extraction, departing from ProRL's protocol [IL-34–IL-37, IL-40, IL-41].
- Answer entropy falls; ProRL's token entropy rises [IL-46–IL-48]. In their DAPO run, expansion is temporary [IL-49,
  IL-50].
- No IF task: LiveBench's language part is typos only [IL-66]. A text search finds no IFEval or IFBench.

**Paper interprets.** RLVR makes existing successes more reliable and loses others: the "invisible leash" [IL-6, IL-7,
IL-45]. Surviving expansion may be format conformance or subskill composition [IL-42, IL-44]. Its idealised theory
predicts no shrinkage [IL-57]. Limitations: no reasoning check, false acceptances, draw variance [IL-60–IL-62].

**Our interpretation.**
- *Our arithmetic:* Table 6's non-maths aggregate (1,045 / 31 / 107 / 674) is exactly the sum of its seven SimpleQA,
  LiveBench, SciBench and LiveCodeBench rows [IL-21, IL-22], as in the 7B and 14B tables. The headline gain and loss
  counts **exclude Reasoning Gym**, where ProRL's expansion claims sit; Reasoning Gym appears only as curves.
- RL-only share (our arithmetic): v2 48 / 3,404 = 1.4 %; OLMo-2 83 / 1,547 = 5.4 %. Ours on IFBench is 19.7–22.5 %.
- The boxnet configuration (easy or hard set) and the per-task question counts are UNVERIFIED.

## Evidence and limitations

No fresh-draw confirmation [IL-62], hand reading, contamination check or IF; v2 only in aggregate [IL-9].

## Connections and questions

- **C1, contradicts** (the reverse part): losses exceed gains everywhere, e.g. AIME24 at k 8,192 [IL-7, IL-23]. C1 has
  0 reverse at k ≤ 512 from a pretrained base. Reach agrees (NDR ≈ 0) [IL-20].
- **C2, replicates:** small reach, and small losses for large distilled models (AceReason-14B: S 23 against P 2,418)
  [IL-200].
- **C3, contradicts:** it generalises net shrinkage to non-maths [IL-7] but tests no IF. C3 (~20 %, 2 reverse) runs
  against that.
- **C4, extends** (C4 goes further): false acceptance is only named [IL-61].
- **C7, extends:** it shares symmetric budgets and format tolerance [IL-14, IL-36]. It lacks fresh-draw confirmation,
  a hand gate and a guessing control.
- **C8, extends:** its larger-k base test [IL-12, IL-59] is the template for Test 3, never applied to IF.
