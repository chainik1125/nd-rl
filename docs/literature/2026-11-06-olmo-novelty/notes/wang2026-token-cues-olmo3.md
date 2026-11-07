---
written_on: 2026-11-07
written_by: "agent:claude"
papers:
  - wangBaseModelsCan2026
---

# Token cues: two fixed opening tokens bring the Olmo 3 7B base close to RL-Zero (C1's model pair, small k)

Paper: [@wangBaseModelsCan2026] (proposed key; Sophie L. Wang et al., "Base Models Can Reason By Taking a Cue From
Training Data")
Source: https://arxiv.org/abs/2610.06851v1 (v1, 5 Oct 2026). Read in depth: §1–2, §5–6, App. B.1, Tables 5–6;
§3–4 skimmed. Ledger: `ledger/GG.tsv`.

The only screened create-vs-elicit test on C1's model pair: Olmo 3 7B base vs the released Olmo-3-7B-RL-Zero-Math
[GG-57][GG-58]. C1 uses the Olmo 3.1 RL-Zero-Math run at steps 500 and 2,800.

## Setup card

| item | value |
|---|---|
| model size and start | Olmo-3-7B pretrained base vs Olmo-3-7B-RL-Zero-Math (RL directly from the base) [GG-57][GG-58]; also Qwen3-14B vs an own GRPO run |
| RL share of pretraining | rl-compute-share, Olmo 3 7B RL-Zero (math): 4.6 × 10⁻⁴ – 2.3 × 10⁻³ |
| RL steps | released model; own 300-step GRPO runs on MATH (§2.3; UNVERIFIED detail) |
| domain | maths (MATH-500, GSM8K, AMC 23, AIME 2024/2025, OlympiadBench), HumanEval |
| largest k | 16 (32 samples, 31,744-token cap) [GG-59] |
| grading | final-answer equivalence by Math-Verify; parse failures wrong [GG-63] |
| controls | symmetric budgets: yes [GG-59]; format tolerance: partly (Math-Verify) [GG-63]; guessing: no; contamination: no |

## Learnings

**Observed.**
- The cue ".\n\n Okay" lifts the base's MATH-500 pass@1 from 42 % to 78 %, against 75 % for RL-Zero [GG-56].
- AIME 2024: the cue lifts the base from 12.0 to 23.3 (pass@1) and 51.9 to 56.6 (pass@16) [GG-60]; RL-Zero-Math
  scores 35.2 and 68.6 [GG-61].
- In the authors' own 300-step GRPO runs (§2.3; our reading of which RL model), the cue's probability rises from 0.14
  to 0.65 [GG-88]; policy changes sit in the first output positions [GG-62].
- The cue is picked by a label-free answer-agreement criterion [GG-102], within 1.1 points of the best of 500
  openings on MATH-500 [GG-103].

**Paper interprets.** Much of RL-Zero's gain is the base's own reasoning, unlocked by openings that RL makes likely.
The claim is limited to RL applied directly to a base [GG-86]; multi-stage pipelines may add more [GG-87].

**Our interpretation.**
- On AIME the cue does not match RL. By our arithmetic from [GG-60][GG-61] it recovers about half of RL's pass@1
  gain (11.3 of 23.2 points), and RL keeps a 12-point pass@16 lead.
- At k ≤ 16 the cue leaves an AIME gap that C1 closes by sampling: C1's uncued base is credited on all 17 RL-only
  AIME problems within 383 samples. Both say RL-Zero's AIME gain lies largely inside the base's distribution.
- The early-token mechanism [GG-62] fits C1's likelihood result (every scored RL solution inside the base's per-token
  range despite a large whole-solution gap): a few early tokens can carry much of a sequence-level gap.

## Evidence and limitations

- Olmo 3 RL-Zero-Math, not the 3.1 run C1 used; k ≤ 16 [GG-59].
- Answer-only grading [GG-63], no hand check: cued AIME successes could include guesses (UNVERIFIED either way).
- AIME means over 30 problems carry SDs of 5–8 points [GG-60][GG-61]. "Match or outperform RL in most cases"
  summarises all models and benchmarks; Olmo's AIME rows are below RL.
- Which problems the cue-selection criterion was computed on is UNVERIFIED.

## Connections and questions

- **C1, extends.** Same pair, same direction [GG-56][GG-60]. C1 adds per-problem reach to k = 512, the reverse
  check, AIME 2025–26 and a hand check; the paper adds a training-free intervention and a mechanism [GG-62][GG-88].
- **C8, extends.** C8's brevity hint elicits `dpo` on 16 of 48 IF reach prompts; here a two-token opening elicits
  most of RL-Zero's maths gain [GG-56].
- **C7, extends.** An opening or prompt baseline belongs beside large-k sampling in elicitation tests.
- **C2, unrelated** (multi-stage pipelines excluded) [GG-87]. **C3, C4, C5, C6: unrelated.**
- Question: run the cue on C1's 17 RL-only problems; if the cued base solves them at k ≤ 32, "within 383 samples"
  becomes "within 32 with a two-token prefix".
