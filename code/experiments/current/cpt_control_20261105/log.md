---
written_on: 2026-11-04
written_by: agent:claude
---

# Log: CPT control for the paper's base-vs-RL comparison

Branch `leon/circuit-diffing`, worktree `/scratch/cowifornia/nd-rl/worktrees/circuit-diffing`.
Artifacts go to `ynez:/scratch/cowifornia/nd-rl/cpt_control/`. Times are PT, with Z in brackets.

## 2026-11-04

- **5:17 pm (01:17Z).** The coordinator asked for this control, under Leon's standing rule on
  fairness controls. It runs after Stage 3 Part B on the same reserved GPU, ynez GPU 2.
- **5:27 pm (01:27Z).** PREREG.md written before any job: sha256 `f76a3244c447effab240a104c6d34b66b650844e47be99e9d266c39ac621a2a8`.
  - The record order of RL seeds and CPT seeds is described in the PREREG.
  - The base checkpoints `S1g3_s{1,2}` and `S1g3.pkl` on ynez have the same sha256 as on
    serrano: `6535c483…`, `c36c9bc4…` and `3255633f…`.
