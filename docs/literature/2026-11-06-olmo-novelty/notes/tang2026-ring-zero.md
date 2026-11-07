---
written_on: 2026-11-07
written_by: "agent:claude"
papers:
  - tangRingZeroScalingZero2026
---

# Tang et al. 2026 (Ring-Zero): step-resolved pass@1024 in 1T zero RL

Paper: [@tangRingZeroScalingZero2026] (proposed key)
Source: arXiv 2607.12395v2 (16 Jul 2026); shared text `~/onv_sources/2607.12395v2.txt` (HTML rendering). Screened
earlier by rl-compute-share ("sections"). Ledger `DT-79`–`DT-86`, `DT-110`–`DT-114`, `DT-132`, `DT-135`.

## Setup card

| item | value |
|---|---|
| model, start | Ling-2.5-1T-Base (1T) and Ling-2.5-flash-Base (104B), zero RL from the pretrained base [DT-110] |
| RL share of pretraining | rl-compute-share: not computable (base pretraining tokens not in the text; GPU-hours not reported) |
| RL steps | 3,600 for 1T [DT-82], 5,200 for flash [DT-132]; KL penalty in stage 1 [DT-111]; 16 rollouts at temperature 1.0 [DT-114] |
| domain | maths (seven maths benchmarks [DT-135]) |
| largest k | 1,024 (pass@1024 tracked "during early training") [DT-79]; the benchmark is not named in the text (figure only) |
| grading | training reward is rule-based early and an LLM judge (Qwen3-Next-80B) later [DT-113]; the evaluation grader is UNVERIFIED |
| controls | symmetric budgets: not applicable (the policy's own pass@1024 over steps; the base is the curve's starting point, figure only); format tolerance: no; guessing: no (k = 1,024 on maths answers, not discussed); contamination: no (not discussed) |

## Learnings

**Observed.**
- pass@1024 rises early in training and then plateaus [DT-80][DT-85], while pass@1 keeps climbing [DT-81].
- After 3,600 steps the 1T model scores 89.06 % on AIME 2024 [DT-82].
- The response window grows from 4k to 64k tokens, doubling every 800 steps [DT-84].

**Paper interprets.**
- Early RL "actively expands" the boundary by finding patterns dormant in pretraining. A sharpening phase follows, so discovery and sharpening are sequential stages [DT-112].
- Zero RL is bounded by pretraining knowledge [DT-83].

**Our interpretation.**
- *A budget confound.* The "discovery" phase overlaps the length curriculum: the training window doubles every 800 steps [DT-84]. Whether evaluation used a fixed generation length at every checkpoint is UNVERIFIED. If it did not, part of the early pass@1024 rise could be a token-budget effect, the confound our C5 and C8 Test 1 isolate (C8: the token budget explains 1 of 3 geometry reach prompts).
- *Guessing.* pass@1024 on integer-answer maths is exposed to guessing, which C7 flags.
- *Not reproducible from outside.* This is the largest step-resolved large-k curve in this screen, but the text does not mention a checkpoint release and does not name the panel's benchmark. It cannot be re-run or re-graded.

## Evidence and limitations

- The curve is figure only: no base value, no per-problem counts, no confirmation draws.
- In late stages the training reward uses an LLM judge [DT-113]; the text does not say whether evaluation does.
- Main-table scores are pass@1 averaged over 64 runs [DT-86]; the k = 1,024 sample count per step is not given in the text (UNVERIFIED).
- The prior rl-compute-share review flagged that the pass@1024 panel does not name its model. We confirm the benchmark is likewise unnamed in the text.

## Connections and questions

- **C1: contradicts in direction.** Zero RL here raises pass@1024 early, which the paper reads as discovery [DT-80][DT-112]. C1's base is credited on every RL-only AIME solve within 383 samples. Magnitudes are figure only, and model, scale and grader differ.
- **C5: extends (our reading).** The early rise coincides with response-window growth [DT-84], the budget confound C5 and C8 Test 1 test.
- **C7: extends.** A step-resolved pass@1024 exists at 1T scale, but on non-public checkpoints with an unnamed benchmark [DT-79][DT-85]. A public trajectory (Olmo 3.1) is the reproducible version.
- **C2, C3, C4, C6, C8: unrelated.**
- Question: on Olmo 3.1 7B RL-Zero-Math (28 public step branches), does pass@k at a fixed 32k budget show the same early rise and plateau?
