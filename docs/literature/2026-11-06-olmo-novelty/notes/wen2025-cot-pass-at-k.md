---
written_on: 2026-11-07
written_by: "agent:claude"
papers:
  - wenReinforcementLearningVerifiable2025
  - tasaltiDoesCoTPasskReally2026
---

# CoT-Pass@K: an LLM judge removes unsound large-k successes, and an audit finds the judge mostly checks the answer

Papers: [@wenReinforcementLearningVerifiable2025] (proposed key; Wen et al., MSRA) and
[@tasaltiDoesCoTPasskReally2026] (proposed key; Taşaltı, Hüdaverdi, Semedo, MRL 2026).
Sources: https://arxiv.org/abs/2506.14245v2 (v2; §1–6, App. A.3–A.8 read) and https://aclanthology.org/2026.mrl-1.34
(= arXiv 2609.32622v1; abstract, §1–2, §4.1, §5.1–5.2, §6 read). Ledger: `ledger/GG.tsv`.

## Setup card (Wen et al.)

| item | value |
|---|---|
| model size and start | maths: pretrained Qwen2.5-32B vs DAPO-Qwen-32B, RL-zero [GG-15]; code: distilled R1-Distill-Qwen-7B vs AceReason-Nemotron-7B [GG-23] |
| RL share of pretraining | rl-compute-share, DAPO-Qwen-32B: 2.4 × 10⁻⁴ – 3.0 × 10⁻² |
| RL steps | released DAPO run; their reproduction to about 400 steps (Fig. 4; figure only) |
| domain | maths (AIME 2024/2025, MATH-500, AMC23, Minerva), code (LiveCodeBench) |
| largest k | 1,024 [GG-17] |
| grading | answer match; CoT-Pass@K adds an LLM judge [GG-12][GG-13], three judgments per chain [GG-14]; hand inspection of disagreements [GG-18] |
| controls | symmetric budgets: yes (UNVERIFIED beyond figures); format tolerance: no; guessing: yes, the subject [GG-11]; contamination: partly (AIME 2025 postdates the base) [GG-99] |

## Learnings

**Observed (Wen et al.).**
- Under answer-only Pass@K the base catches up with and passes DAPO as K grows [GG-16]. Under CoT-Pass@K the RL
  model leads at every K up to 1,024 on AIME 2024 and 2025 [GG-17]; on MATH-500 and AMC23 the base solves the
  problems with enough attempts [GG-19].
- The judge finds flawed chains among RL's own successes: on optimised DAPO training questions the median accepted
  share is about 0.7 [GG-21].
- Code Pass@K is treated as guess-proof [GG-20]; the distilled 7B start gains Pass@K [GG-100].

**Paper interprets.** Base Pass@K on maths is inflated by wrong chains landing on guessable answers [GG-11]; RLVR
does extend the boundary.

**Observed (audit).** Under the metric's own protocol three judges accept corrupted chains almost as often as clean
ones [GG-25]. The metric's own judge accepts 71 % of solutions with a changed final answer [GG-26]. The Pass@k −
CoT-Pass@k gap is 19.7 points for Qwen2.5 solvers but 4.1 for current ones [GG-27]; a larger token budget moves
Pass@64 by over 50 points while the gap stays at zero [GG-28].

**Audit interprets.** The judges mostly check chain-answer agreement; judged metrics should publish their response
to injected errors [GG-29].

**Our interpretation.**
- How CoT-Pass@K treats guessing: an LLM judge re-grades every success, removing guesses and unsound successes on
  both sides if the judge works. The audit shows it is the weak link; Wen et al. call it their key limitation
  [GG-22] and hand-checked only disagreements [GG-18].
- C1's hand check is CoT-Pass@K by hand, mainly on the base side (its first success on each of the 17 problems).
  7 of 17 base resolutions are unsound (6 flawed steps, 1 guess), so answer-only grading overstates the base's
  reach, as in [GG-17]. But 10 of 17 are sound within ≤ 383 samples (review's bound 10–16): closer to Yue than to
  "a gap at every K". The few RL successes read were no better (on aime2024_20 all five are guesses).
- [GG-21] is a maths analogue of C4, but on training questions, not reach-deciding test successes.

## Evidence and limitations

- CoT-Pass@K curves are figures only; no per-problem table to match C1 against.
- The audit tests other solvers at k ≤ 64 [GG-101], so it weakens the instrument rather than refuting Wen et al.'s
  curves. It cites work where frontier judges pass up to half of right-answer, wrong-reasoning solutions [GG-30].

## Connections and questions

- **C1, extends.** Same pair type (RL-zero from a pretrained base, AIME, large k) [GG-15][GG-17]. C1 adds a hand check
  of the reach-deciding base successes and newer AIME years; Wen et al. add a symmetric soundness grade, by LLM.
- **C4, extends.** RL's own successes are partly unsound in maths [GG-21]; no IF, no gaming taxonomy.
- **C7, extends.** C7's blinded hand gate is the control LLM-judged metrics lack [GG-25][GG-29]; it too should be
  shown to catch injected errors.
- **C8, extends** (budget dominates Pass@64) [GG-28]. **C2, unrelated** (code, 7B) [GG-23]. **C3, C5, C6: unrelated.**
- Question: hand-grade C1's RL-side successes on the 17 problems too.
