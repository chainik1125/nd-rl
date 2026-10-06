# Run 2: six new patterns, classes pre-registered

Run 2026-09-17/18; numbers in `numbers.md` §Round 2 — Run 2; classes and expectations pre-registered in `log.md` 19:47 UTC. **Question:** does "RL crosses zero coverage for a structural repetition of a learned move, not for a new rule sequence" survive six patterns chosen in advance?

**Setup.** Cap-6 sets of 155,000 generator proofs: a uniform draw (`struct`; no ≤ 6-line proof can hold a fourth box, an IMPE chain ≥ 4 or a nested ORE, so it is f = 0 for all three) and three draws with one rule-sequence pattern removed (f = 0 asserted on the written files). Pools of 300–500 theorems whose shortest found proof uses the pattern (`necessity.py`, required subset flagged), two Stage-1 seeds, expert iteration (k = 32, 8 rounds), frozen controls, pre-RL pass@2,000. The generator cannot make three of the shapes: depth-4 candidates came from it with its box-depth cap raised to 5, chains and nested OREs from schemata (`QUESTIONS.md`).

| pattern (class) | f = 0 acquisition, 2 seeds | pre-RL rate | f > 0 control |
|---|---|---|---|
| depth-4 (structural) | **0 / 0** of 500 | 0, 0 in 600k | cap-8 models with 0 depth-4 proofs: 0.97 / 0.96 and 0.96 / 0.96 (base rate 0.11–0.20) |
| IMPE chain ≥ 4 (structural) | 0 / 0 of 500 | 0, 0 | — |
| nested ORE (structural, 11–13 lines) | 0 / 0 of 300 | 0, 0 | — (length-limited, as pre-registered) |
| IMPI containing ORE (sequence) | 0 / 0 of 500 | 0, 0 | natural-rate models (130 / 155k): 0.19 / 0.22 |
| NEGI with ANDE on its hypothesis (sequence) | 0 / **0.147** of 400 | 0, 5·10⁻⁶ | natural-rate models (11 / 155k): 0 / 0.03 |
| ORI consumed by ORE (decoration control) | 0 / 0 of 400 (217 / 219 solved without it) | 0, 0 | — |

**What happened.** Nothing at f = 0 crossed zero except negi_ande_hyp seed 1, whose base already emitted the shape at 5·10⁻⁶ — elicitation, as the ignition rule predicts; its round-4 checkpoint, trained once on the first proof, emits the shape on 28 / 300 targets at 8.6·10⁻³ (the drift measurement: nothing drifts before the first proof because nothing is trained before it). The structural negatives are length, not structure: a fourth box needs ≥ 8 lines, two beyond cap 6, and the cap-6 base never writes one; at cap 8 the base models — four draws, **zero** depth-4 proofs in their data — already write a fourth box in 11–20 % of samples and RL takes them to 96–97 %. The decoration control solves 217 / 219 targets and never writes the pointless ORI-then-ORE. "Structural → acquired" held at cap 8 and failed at cap 6; "rule sequence → ≈ 0 unless the base generalised" held in all six arms.

**Restated rule.** Zero-coverage acquisition needs a base rate above zero; a structural repetition has one only when the proof length it needs is inside the pretraining cap, and a rule sequence only by draw-level generalisation. The class labels predicted outcomes only through the base rate — the ignition study's variable, not the pattern's syntax.

![acquisition](figures/run2_acq.png)
![curves](figures/run2_curves.png)
