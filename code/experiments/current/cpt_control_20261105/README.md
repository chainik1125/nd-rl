---
written_on: 2026-11-04
written_by: agent:claude
---

# CPT control: continued pretraining against RL on Charles149

The question is how much of the paper's base-to-RL gain on Charles149 is matched by continued
pretraining of the base on its own data. That control ("CPT") uses no RL, and matches RL's AdamW
steps and trained tokens.

Pre-registration: [`PREREG.md`](PREREG.md). Running record: [`log.md`](log.md). Numbers:
[`table.md`](table.md) and `table.json`.

## Result (Charles149, seeds 0–2 paired, k1024)

pass@k is in % of the 149 problems. Gaps are in points, with 95% paired-seed problem-bootstrap
intervals (`stats.paired_diff`).

| k | base | CPT (no RL) | RL (L3 end) | RL − CPT | CPT − base | share of RL's gain reached by CPT |
|---|---|---|---|---|---|---|
| 1 | 32.6 | 54.2 | 70.5 | **+16.4 [+11.2, +21.6]** | +21.5 [+14.7, +29.3] | 57% |
| 5 | 49.7 | 69.1 | 79.0 | **+9.9 [+4.5, +15.4]** | +19.4 [+10.8, +28.7] | 66% |
| 64 | 65.1 | 81.8 | 83.6 | +1.8 [−2.8, +6.7] | +16.7 [+9.7, +24.5] | 90% |
| 1024 | 74.5 | 88.6 | 86.6 | **−2.0 [−6.5, +2.2]** | +14.1 [+8.1, +21.0] | 117% |

- **Observed: small k.** CPT reads 54.0–54.4% at pass@1 and 87.9–89.3% at pass@1024 across the
  three seeds. Every seed shows RL ahead at pass@1, by about +16 points.
- **Observed: large k.** At pass@1024 CPT is level with or above RL in every seed: RL − CPT is
  +1.3, −3.4 and −4.0.
- **Observed: one-arm-only problems at k1024**, pooled over 3 × 1,024 attempts per arm:
  - all 9 problems that RL solves and the base never does are also solved by CPT;
  - RL solves 0 that CPT does not;
  - CPT solves 4 that RL does not, and 13 that the base does not.
- **Pre-registered readings:**
  - "RL's gain survives the control" at pass@1024 fails.
  - "A large part of the gap is matched by CPT" holds at every k.
  - "The control explains little" fails.
- **Interpretation.**
  - On Charles149, the paper's large-k base-vs-RL gap is reached by 312 more AdamW steps on the
    base's own Stage-1 data, with no RL.
  - RL's distinctive advantage over that matched control is sampling efficiency (pass@1, pass@5),
    not coverage at large k.
  - Why CPT gains is untested. One hypothesis is that Stage 1 ended un-annealed; capability-defs
    J7 (PREREG amendment 1) argues against it.
**13–20-line sample** (t1320s250, 250 problems, seed 0 only, k512; PREREG amendment 2,
descriptive):

| k | base | CPT | RL | RL − CPT | share of RL's gain reached by CPT |
|---|---|---|---|---|---|
| 1 | 11.2 | 34.2 | 38.7 | +4.5 [+0.9, +8.0] | 84% |
| 5 | 28.4 | 60.9 | 66.4 | +5.5 [+1.3, +9.8] | 86% |
| 64 | 59.3 | 84.5 | 87.4 | +3.0 [−0.7, +6.6] | 90% |
| 512 | 77.2 | 93.6 | 94.0 | +0.4 [−2.8, +3.6] | 98% |

On the longer proofs, CPT reaches most of RL's gain even at pass@1.

**Is the base underfit?** This is a CPU check (`valloss.py`, numbers in `valloss.md`): teacher-forced
loss on Stage 1's own held-out val split, which reproduces the logged Stage-1 val exactly.

| seed | val loss: base / CPT / RL | Charles149 reference proofs: base / CPT / RL |
|---|---|---|
| 0 | 0.0692 / 0.0648 / 0.0847 | 0.0898 / 0.0655 / 0.1168 |
| 1 | 0.0738 / 0.0646 / 0.0935 | 0.1036 / 0.0691 / 0.1220 |
| 2 | 0.0750 / 0.0650 / 0.0945 | 0.1087 / 0.0718 / 0.1325 |

- CPT lowers held-out val loss in every seed, by 6–13%.
- The three CPTs land at the same value (0.0646–0.0650), although their bases differ
  (0.069–0.075).
- CPT lowers the loss on Charles149's reference proofs by 27–34%.
- RL raises both.

So the Stage-1 bases were not converged on their own distribution. More training on it, with no
RL, raises Charles149 coverage to RL's level. RL instead moves away from that distribution towards
its own proofs (NLL on its own accepted actions: 0.049 against the base's 0.126). This supports
"base underfit" over "anneal only" (interpretation).

- **Caveats.**
  - CPT matches RL's trained tokens and AdamW steps, but has 0.64× RL's full FLOP ratio, which
    includes sampling.
  - The CPT seeds vary only the base. Their record order is fixed.
  - Three seeds on Charles149; seed 0 only on the 13–20 sample.

| file | what |
|---|---|
| `reads.py` | Charles149 k1024 read of one checkpoint: 16 k64 chunks with the paper's read command (think_hirl's code), merged as `hirl_queue.merge_group` does (ynez) |
| `table.py` | base vs CPT vs the L3 end, seeds paired, at pass@1 / 5 / 64 / 1024, with one-arm-only problem counts → `table.json`, `table.md` |
| `valloss.py` | CPU: teacher-forced loss on Stage 1's val split and on Charles149's reference proofs → `valloss.json`, `valloss.md` |

CPT training is `../circuit_diffing_20261101/cpt.py --base_seed S`.
