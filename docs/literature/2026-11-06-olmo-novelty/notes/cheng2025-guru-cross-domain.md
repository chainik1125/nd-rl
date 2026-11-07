---
written_on: 2026-11-07
written_by: "agent:claude"
papers:
  - chengRevisitingReinforcementLearning2025
---

# Cheng et al. 2025 (Guru): one RL run per size; maths elicited, Zebra Puzzle expanded

Paper: [@chengRevisitingReinforcementLearning2025] (proposed key)
Source: arXiv 2506.14965v1 (17 Jun 2025); text `~/onv_sources/DT_2506.14965v1.txt`. Figure axes are not
recoverable from the text, so every pass@k magnitude below is figure only. Ledger `DT-35`–`DT-51`, `DT-115`, `DT-116`, `DT-130`, `DT-131`.

## Setup card

| item | value |
|---|---|
| model, start | Qwen2.5-7B and Qwen2.5-32B base (pretrained) [DT-49] |
| RL share of pretraining | not computed (not in rl-compute-share's table) |
| RL steps | 3 epochs (7B), 2 epochs (32B), about 3 days each [DT-44]; first epoch ≈ 160 steps for 7B [DT-115]; no KL or entropy loss [DT-45] |
| domains | six-domain mix (Math, Code, Science, Logic, Simulation, Tabular); pass@k shown for AIME24 and Zebra Puzzle; over steps for AIME24, ARC-AGI and CodeI/O |
| largest k | figure only (UNVERIFIED); the text cites a crossing at k = 64 and a plateau near k = 32 [DT-38][DT-42] |
| grading | domain-specific rewards; Zebra Puzzle gets partial credit in Table 3 [DT-47]; how Zebra pass@k is scored: UNVERIFIED |
| controls | symmetric budgets: UNVERIFIED (figure only); format tolerance: partly, decoding sensitivity shown [DT-43]; guessing: no; contamination: partly, a new Zebra test set is generated [DT-116], AIME24 exposure of Qwen2.5 not discussed |

## Learnings

**Observed.**
- AIME, 7B: the Guru-7B and base curves cross at k = 64 [DT-37][DT-38].
- AIME, 32B: the curves never intersect [DT-41].
- Zebra Puzzle: both sizes expand the boundary [DT-40]. The 7B curve plateaus near k = 32 while the 32B curve keeps rising [DT-42].
- Pattern across tasks: pass@k patterns differ by task [DT-36].
- Transfer: logic, simulation and tabular need in-domain RL data [DT-35].
- Over steps (Guru-7B): in the first ~160 steps pass@1 rises while pass at the largest k falls sharply; afterwards both rise slowly [DT-115][DT-50][DT-51].

**Paper interprets.**
- Maths gains "largely exploit the base model" [DT-39].
- Less-pretrained domains show real expansion, so RL "is likely to facilitate genuine skill acquisition" there [DT-35].
- pass@k is "informative but not exhaustive" [DT-43].

**Our interpretation.**
- *Closest analogue to C2 vs C3.* This is the nearest published match to our C2-vs-C3 split: the same RL run (one per size) elicits on maths and expands on a domain the paper argues is rare in pretraining.
- *Two caveats on Zebra.* Zebra Puzzle is a synthesized RL training task scored on a newly generated test set from the same generator, so it is in-distribution [DT-48][DT-116]. Its Table 3 metric is also partial credit [DT-47]. If pass@k used partial credit (UNVERIFIED), "solved at least once" would be ill-defined.
- *Start model.* At 32B the AIME curves never cross [DT-41]. In our 32B run from DPO, pass@8 is equal for start and RL (C2). Whether maths gains persist at large k may depend on the start (pretrained base vs DPO).
- *Trajectory.* The first-epoch dip and recovery is a step-resolved large-k curve, but on the authors' own run and figure only.

## Evidence and limitations

- Evaluation uses temperature 1.0 and top-p 0.7 [DT-46]. Training samples at temperature 1.0 [DT-130] (training top-p not stated in the text), so evaluation may not use RL's own sampler, and the paper shows pass@k moves with these settings [DT-43].
- No per-problem reach counts, no confirmation draws, no hand checks.
- The cross-domain transfer result (Figure 1) is accuracy, not pass@k [DT-131].

## Connections and questions

- **C2: extends.** From a pretrained 32B base the AIME advantage persists at every sampled k [DT-41]; ours from DPO does not. This points to the start as a variable.
- **C3: replicates.** Domain-dependent expansion within one RL run [DT-36][DT-40], with the in-distribution and partial-credit caveats above.
- **C1: contradicts.** At 7B the base overtakes RL on AIME at k = 64 [DT-37][DT-38]; C1 sees no narrowing for Olmo 3 7B RL-Zero. Different base model, data and recipe.
- **C7: extends.** The decoding sensitivity [DT-43] supports C7's choice to grade under RL's own sampler.
- **C4, C5, C6, C8: unrelated.**
- Question: how many Zebra test puzzles does Guru solve that the base never solves at the largest k, under binary scoring?
