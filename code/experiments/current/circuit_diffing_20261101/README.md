---
written_on: 2026-11-01
written_by: agent:claude
---

# Circuit diffing: does GRPO create features the Stage-1 base lacks?

Status: in progress. Pre-registration is in [`PREREG.md`](PREREG.md) and the running record in
[`log.md`](log.md).

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
| `figures.py` | Figures in `figs/` |
| `queue.sh` | Runs crosscoder jobs in sequence on one GPU |

Artifacts (sequences, crosscoders, checkpoint copies) are in `/scratch/cowifornia/nd-rl/circuits/`.
