---
written_on: 2026-11-04
written_by: agent:claude
---

# CPT control: continued pretraining reaches RL at large k on Charles149, not at small k

Status (2026-11-04, written by agent:claude for Leon; not reviewed): Charles149, three seeds, done.
The optional 13–20 read is reported below if it finished.

Working folder:
[`code/experiments/current/cpt_control_20261105/`](../../code/experiments/current/cpt_control_20261105/README.md)
(branch `leon/circuit-diffing`). It has a pre-registration with two amendments, each hashed in its
log before it ran.

## Question

Stage 3 of the [circuit-diffing study](../2026-11-01-circuit-diffing-rl-new-features/README.md)
found that "CPT" is far from base-like on the RL-gained problems. CPT is the base S1g3 trained 312
more AdamW steps on its own Stage-1 pairs, with no RL. How much of the paper's base-to-RL gain on
Charles149 does it reach?

## Setup

**Arms**, all from base seed s ∈ {0, 1, 2}:

| arm | what |
|---|---|
| base | `S1g3_s` |
| RL | the recipe's L3 end, `grpo_think_vs_mb_kkl0_s` it39 |
| CPT | `circuit_diffing_20261101/cpt.py`: AdamW, lr 1.5e-4, a 10-step warmup and a cosine to 30%, 312 steps of 400 records (82.1M tokens), weight decay 0 |

- **CPT matches RL** in trained tokens (0.98×) and AdamW steps. It has 0.64× RL's full FLOP
  ratio: 0.186 against 0.289, since RL's includes sampling.
- **Seed variation.** The CPT seeds share one record order (`train.finetune` seed 0), so they
  vary only the base. The RL seeds also vary RL's own sampling.
- **Reads.** The paper's exact Charles149 read: k1024 in 16 chunks of k64, plain arm, B64,
  temperature 0.8, Lean-judged, merged as `think_hirl` does. They are in
  `/scratch/cowifornia/nd-rl/think/runs/eval/v2/charles149_cpt_s1g3_s{0,1,2}_plain_k1024/` on
  serrano.
- **Statistics.** `stats.paired_diff`, with seeds resampled in pairs and 4,000 draws, as
  `high_rl_passk_20261030/seeds3.py` does.

## Results (Charles149, 149 problems, seeds 0–2 pooled)

| k | base | CPT | RL | RL − CPT | CPT − base | RL − base | share of RL's gain reached by CPT |
|---|---|---|---|---|---|---|---|
| 1 | 32.6% | 54.2% | 70.5% | **+16.4 [+11.2, +21.6]** | +21.5 [+14.7, +29.3] | +37.9 [+30.0, +46.2] | 57% |
| 5 | 49.7% | 69.1% | 79.0% | **+9.9 [+4.5, +15.4]** | +19.4 [+10.8, +28.7] | +29.3 [+21.0, +38.6] | 66% |
| 64 | 65.1% | 81.8% | 83.6% | +1.8 [−2.8, +6.7] | +16.7 [+9.7, +24.5] | +18.6 [+12.3, +25.7] | 90% |
| 1024 | 74.5% | 88.6% | 86.6% | **−2.0 [−6.5, +2.2]** | +14.1 [+8.1, +21.0] | +12.1 [+7.2, +17.9] | 117% |

- **Per seed, RL − CPT** at pass@1 / pass@1024:
  - s0: +16.1 / +1.3;
  - s1: +16.5 / −3.4;
  - s2: +16.5 / −4.0.
- **Problems solved at k1024 by one arm only**, pooled over 3 × 1,024 attempts per arm:
  - RL not CPT: 0;
  - CPT not RL: 4;
  - RL not base: 9, and all 9 are also solved by CPT;
  - CPT not base: 13.
- **Pre-registered readings:**

  | reading | outcome |
  |---|---|
  | RL's gain survives (RL − CPT above 0 at pass@1024) | **no** |
  | A large part of the gap is matched by CPT (≥ half of RL − base) | **yes, at every k** |
  | The control explains little | **no** |

**Interpretation** (kept apart from the results above).
- **The large-k gap.** On Charles149, the paper's large-k base-vs-RL gap, the "capability"
  signal, is reached by matched-step continued pretraining on the base's own data, with no RL.
- **What RL adds.** RL's distinctive advantage over this baseline is sampling efficiency: about
  +16 points at pass@1 and +10 at pass@5. It is not coverage.
- **What the paper should do.** Its base-vs-RL comparisons need CPT as the baseline.
- **Why CPT gains is untested.** The Stage-1 schedule ends un-annealed, at a cosine floor of 30%
  of peak. CPT and GRPO both add a low-LR AdamW phase. capability-defs J7 (2026-10-05) is a
  precedent against "it is just the anneal": it annealed a similarly un-annealed base for far
  longer, and the result was lateral. Contamination is ruled out: no G or S problem shares a
  renaming class with any Stage-1 record.

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

## Caveats

- One evaluation set (Charles149) for the three-seed result.
- CPT uses 0.64× RL's full FLOPs. A CPT run at matched full FLOPs would be about 1.6× longer.
- `LEAN_GATE_WORKERS` was 8, not the paper's 16. Lean verdicts do not depend on it.
