---
written_on: 2026-11-07
written_by: "agent:claude"
papers:
  - zhouWhenRLVRShrinks2026
---

# Zhou 2026: pass@k inversion is concentrated on rare-but-reachable prompts

Paper: [@zhouWhenRLVRShrinks2026] (proposed key)
Source: arXiv 2607.20543v1 (12 Jul 2026), the only version online (abs page checked 2026-11-07). Shared text
`~/onv_sources/2607.20543.txt` (HTML of v1). Ledger `DT-70`–`DT-78`, `DT-120`–`DT-122`, `DT-126`–`DT-128`.

## Setup card

| item | value |
|---|---|
| model, start | Qwen2.5-7B (pretrained base) [DT-71] |
| RL share of pretraining | not computed (not in rl-compute-share's table) |
| RL steps | 1,000 (final GRPO step) [DT-73]; an early-stop control at step 300 [DT-74] |
| domain | maths (Omni-MATH-Train; Omni-MATH-Test, MATH500) |
| largest k | 256 (256 samples per policy, temperature 0.6) [DT-71] |
| grading | maths verifier (answer match); details UNVERIFIED |
| controls | symmetric budgets: yes, 256 samples for the base and every policy [DT-71]; format tolerance: no (not discussed); guessing: no; contamination: no (not discussed); seeds: 3 for the main comparison [DT-77] |

## Learnings

**Observed.**
- Omni-MATH-Test: base 10.2 → GRPO 25.1 at p@1, but 69.1 → 68.3 at p@256 [DT-126][DT-127]. GRPO ends "slightly below the base" [DT-72].
- In the 3,000-prompt regime study GRPO loses 654 ± 35 boundary prompts; PBA loses 91 ± 18 [DT-122].
- Neither trained method gains much on out-of-reach prompts (0.1 ± 0.0) [DT-75].
- The step-300 early stop keeps more coverage than step 1000: boundary net −16 against −39, p@256 70.2 against 68.4 [DT-74][DT-73].

**Paper interprets.**
- Inversion means fewer distinct problems solved at large k [DT-70]. It is a finite-sample failure to commit to the correct mode on prompts whose correct mode is rare but recoverable [DT-120][DT-121].
- The proposed fix (PBA) preserves base modes and creates none [DT-78].

**Our interpretation.**
- *Losses and gains both sit in the base's support.* Losses concentrate in the boundary regime, where the base's per-sample rate is below 0.10 yet its pass@256 is above 0.4 [DT-128]. Almost nothing is gained outside the base's reach (0.1 ± 0.0; the unit of this cell is not stated, UNVERIFIED) [DT-75]. The trained policy's whole coverage change happens inside base support. For our IFBench reach this reads as follows: a reach prompt shows something the frozen start never does in 256 tries, which this paper finds almost never happens in maths.
- *Template for our candidates.* The four-regime stratification [DT-120][DT-128] could sort our reach and reverse candidates by the start's per-sample rate. That would make clear whether C1's "base credited within 383 samples" cells are the boundary regime.
- *The only time-resolved evidence is one early stop.* Less training means fewer losses [DT-74], which fits losses accumulating over steps. The paper disclaims checkpoint-level validation [DT-76]. A public checkpoint trajectory could measure this directly.

## Evidence and limitations

- One model family, two maths benchmarks, three seeds [DT-77]. All controls, early stop included, are single runs.
- The regime thresholds are operational, not fundamental [DT-120]. Regimes come from frozen-base rollouts, so a base sample count that is too small leaks rare-but-reachable prompts into "out of reach".
- No grader audit, no guessing analysis and no hand checks.

## Connections and questions

- **C1: contradicts on losses.** GRPO falls below the base at pass@256 and loses boundary prompts [DT-72][DT-121]; C1 loses nothing at k ≤ 512. It agrees on reach: out-of-reach gains are about 0 [DT-75], as C1's base reaches every RL-only solve.
- **C2: contradicts on losses** [DT-121]. C2 has one reverse problem, solved within 56 tries.
- **C7: extends.** Its symmetric gained/lost accounting matches ours, and it lacks the checkpoint axis C7 adds [DT-76][DT-120].
- **C3, C4, C5, C6, C8: unrelated.** Maths only, and no gaming, budget or hint tests.
- Question: in Olmo 3.1 Think 32B, do the few reverse candidates sit in the start's boundary regime (per-sample rate < 0.10, pass@256 > 0.4) [DT-128]?
