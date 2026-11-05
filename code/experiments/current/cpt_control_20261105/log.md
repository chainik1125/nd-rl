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
- **8:09 pm (04:09Z).** CPT_s1 trained: 312 steps in about 16 min, median relative ΔW 5.74%
  (CPT_s0 5.65%), last loss 0.0278, sha256 `8b6d8f76…`. Its Charles149 read started.
- **8:36 pm and 8:47 pm.** Claude restarted twice. The detached runners on ynez were unaffected.
  Checked: our only GPU process is on ynez GPU 2 (Leon's new cap: 4 GPUs in total, 3 of them
  the knockout supervisor's). `run_cptc2.sh` waits for `run_cptc.sh` to exit, and `reads.py`
  refuses to start a chunk on a busy GPU, so they cannot overlap.
- **9:03 pm (05:03Z).** CPT_s1 merged: pass@1 54.4%, pass@1024 88.6%. Copied to serrano's
  `eval/v2`. Two-seed table:

  | seeds | k | base | CPT | RL | RL − CPT | share of RL's gain |
  |---|---|---|---|---|---|---|
  | 1 | 1 | 33.0 | 54.4 | 70.9 | +16.5 [+10.9, +22.2] | 56% |
  | 1 | 1024 | 74.5 | 88.6 | 85.2 | −3.4 [−7.4, +0.7] | 131% |
  | 0+1 | 1 | 36.1 | 54.3 | 70.6 | +16.3 [+10.9, +21.9] | 53% |
  | 0+1 | 1024 | 76.5 | 88.3 | 87.2 | −1.0 [−6.0, +4.0] | 109% |

  CPT_s2 training started.
- **9:19 pm (05:19Z).** CPT_s2 trained (sha256 `a9adde3b…`). Its Charles149 read started; it should finish at about 10:15 pm.
- **9:24–10:37 pm.** The CPT_s2 read ran at 280–320 s per chunk, against about 200 s earlier.
  GPU 2 held only our process, so the likely cause is CPU contention on ynez during Lean judging.
- **10:37 pm (06:37Z).** CPT_s2 merged: pass@1 54.0%, pass@1024 89.3%. `run_cptc.sh` is done, and
  `run_cptc2.sh` started the optional t1320s250 read on GPU 2.
- **10:40 pm.** **Pooled table** (`table.md`, seeds 0–2 paired):

  | k | base | CPT | RL | RL − CPT | share of RL's gain |
  |---|---|---|---|---|---|
  | 1 | 32.6 | 54.2 | 70.5 | +16.4 [+11.2, +21.6] | 57% |
  | 5 | 49.7 | 69.1 | 79.0 | +9.9 [+4.5, +15.4] | 66% |
  | 64 | 65.1 | 81.8 | 83.6 | +1.8 [−2.8, +6.7] | 90% |
  | 1024 | 74.5 | 88.6 | 86.6 | −2.0 [−6.5, +2.2] | 117% |

  - **One-arm-only problems at k1024**, pooled over 3 × 1,024: RL not CPT 0, CPT not RL 4, RL
    not base 9, CPT not base 13.
  - **Pre-registered readings:**
    - "RL's gain survives" fails at pass@1024; RL is ahead at pass@1.
    - "Large part matched by CPT" holds at every k. The PREREG named it an anneal effect; the
      mechanism is untested.
    - "Explains little" fails.
  - Messaged the coordinator.
- **11:00–11:40 pm.** CPU check requested by the coordinator (Leon's hypothesis that Stage 1 is
  underfit): `valloss.py`, 4 torch threads on serrano, no GPU.
  - The first version scored each metric in its own pass and would have exceeded its 30-min
    timeout. I stopped it and rewrote it to score w1 and w0 from one sorted forward per batch.
  - **The scorer reproduces base s0's logged Stage-1 val exactly:** 0.0692 and 0.0772, against
    0.06924 and 0.07722.
  - **Val loss (Stage 1's split, w1):**

    | seed | base | CPT | RL |
    |---|---|---|---|
    | 0 | 0.0692 | 0.0648 | 0.0847 |
    | 1 | 0.0738 | 0.0646 | 0.0935 |
    | 2 | 0.0750 | 0.0650 | 0.0945 |

  - **Charles149 reference-proof loss:** base 0.090–0.109, CPT 0.066–0.072, RL 0.117–0.133.
  - **On L3's own accepted actions** (Stage 3, seed 0, G): base 0.126, CPT 0.145, RL 0.049.
  - **Reading:** CPT converges the base on its own distribution. Every CPT seed lands at about
    0.065, whatever its base's loss. RL moves away from that distribution.
  - The run ended with a NameError (a leftover `del os`) after writing its outputs; fixed in the
    source.
