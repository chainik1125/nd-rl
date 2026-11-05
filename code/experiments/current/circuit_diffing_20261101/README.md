---
written_on: 2026-11-01
written_by: agent:claude
---

# Circuit diffing: does GRPO create features the Stage-1 base lacks?

Status (2026-11-01): Stages 0–2 are done for the seed-0 primary pair, base `S1g3_s0` against
the L3 end. Replicates are queued behind a GPU slot: seeds 1 and 2, the CPT crosscoder, the
crosscoder-seed replicate, layer-5 Stage 2 and the R sweep. The human-readable summary is
[`experiment-summaries/2026-11-01-circuit-diffing-rl-new-features`](../../../../experiment-summaries/2026-11-01-circuit-diffing-rl-new-features/README.md).
Pre-registration is in [`PREREG.md`](PREREG.md), with the sha256 and time of each version in
[`log.md`](log.md).

**Findings, observed.**
- Crosscoders at layers 1, 3 and 5 find 0 RL-only and 0 base-only latents between the base and
  the L3 end. A planted control recovers 4 of 6 synthetic RL-only features, with no false
  positives.
- RL rotates shared latents' write directions: energy-weighted decoder cosine 0.945, against 1.000
  for the null.
- Causally, at layer 3, magnitude reweighting does nothing (0% of the G gap either way).
  Reverting directions removes 41% [34, 47] of the RL-gained gap.
- Weight-diff statistics are the same for RL and for continued pretraining (CPT).

Leon's question (2026-11-01): do new circuits appear in post-training that were absent from the
pretrained base? That would be a mechanistic sign of capability gain rather than elicitation.
Under elicitation, no new circuits are made and existing ones are only upweighted.

## Files

| file | what |
|---|---|
| `common.py` | Checkpoint registry with R (the RL / Stage-1 FLOP ratio), model loading, residual capture, GPU guard |
| `stage0_weights.py` | Stage 0: weight diffing → `stage0.json`, `stage0.md` |
| `rollouts.py` | Crosscoder data: each model's own rollouts on the L2 and L3 pools, plus a pretraining sample |
| `crosscoder.py` | Stage 1: BatchTopK crosscoder for one layer of a model pair, with evaluation and Latent Scaling |
| `stage1_analyze.py` | Stage 1: latent classes and counts → `stage1.json`, `stage1.md` |
| `cpt.py` | Weight-drift control: continued Stage-1 training at matched optimizer and steps |
| `problems.py` | Stage 2 problem categories (G = RL-gained, S = shared) → `stage2_problems.json` |
| `patching.py`, `stage2.py` | Stage 2: crosscoder patching inside the sampler (ablation and steering) |
| `interpret.py` | Top activations of RL-only, exploratory and rotated latents → `interpret_*.md` |
| `stage2_analyze.py` | Stage 2 solve rates with paired bootstrap → `stage2.json`, `stage2.md` |
| `figures.py` | Figures in `figs/` |
| `queue.sh`, `queue2.sh`, `gpu_gate.sh` | Sequential GPU job queues; `queue2.sh` waits for a slot (fewer than 3 GPU jobs on serrano) |
| `traces.py` | Stage 3: the L3 end's own Lean-accepted G and S attempts with traces → `stage3/seqs_L3.npz` |
| `edges.py` | Stage 3: node attribution, layer 3 → 5 EAP edges (forward-mode JVP through each model's blocks), teacher-forced path patching in L3 |
| `stage3_analyze.py` | Stage 3: drift-control revert (part A), edge counts against nulls and patching (part B) → `stage3.json`, `stage3.md`, `figs/stage3_*` |
| `queue3.sh`, `gpu_guard.sh` | Stage 3 job queue on ynez (one reserved GPU, `CIRCUITS_RUNS` mirror) and a guard that reports foreign processes on that GPU |

Artifacts (sequences, crosscoders, checkpoint copies) are in `/scratch/cowifornia/nd-rl/circuits/`.
