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
- **6:30 pm (02:30Z).** PREREG Amendment 1, before any job: the J7 precedent, and a note that the G
  selection draws are independent of Stage 2's base read. sha256 `0693fdff20acc25d429d103ef877bf466602206e9c36455142c1dd361d336a91`.
- **7:15 pm.** PREREG Amendment 2 (the optional t1320s250 k512 read of CPT_s0, output outside `eval/v2`), before it ran: sha256 `fc2e1ada52209b123cb5278191e13687f4e9b7b53f0b67369e4e8be3471ab74a`. `reads.py` gained `--set`; the Charles149 defaults are unchanged.
- **6:58 pm (02:58Z).** The runner started: CPT_s0 read on Charles149, 16 chunks of 190–240 s each,
  with its own guard and watchdog.
- **7:53 pm (03:53Z).** CPT_s0 merged: pass@1 54.2%, pass@1024 87.9%. Copied to serrano's
  `eval/v2/charles149_cpt_s1g3_s0_plain_k1024/`. Seed-0 table (`table.md`):

  | k | base | CPT | RL | CPT − base | RL − CPT | share of RL's gain |
  |---|---|---|---|---|---|---|
  | 1 | 39.2 | 54.2 | 70.3 | +15.0 [+11.2, +18.9] | +16.1 [+10.6, +21.8] | 48% |
  | 1024 | 78.5 | 87.9 | 89.3 | +9.4 [+4.7, +14.8] | +1.3 [−2.7, +5.4] | 87% |

  - Only-one-arm problems at k1024: RL not CPT 6, CPT not RL 4. Messaged the coordinator.
  - CPT_s1 training started.
