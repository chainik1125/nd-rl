---
written_on: 2026-11-01
written_by: agent:claude
---

# Log: circuit diffing, base vs GRPO

Branch `leon/circuit-diffing` (from `leon/high-rl-paper` at `1abc0ee`), worktree
`/scratch/cowifornia/nd-rl/worktrees/circuit-diffing`. Artifacts go to
`/scratch/cowifornia/nd-rl/circuits/`. Times are PT.

## 2026-11-01

- **11:00 am.** Set up the worktree and the artifacts directory.
  - GPU state: another user's process holds GPU 2. GPUs 0, 1, 3 and 4 are free, so this
    experiment uses at most 2 GPUs.
  - Copied 12 low-RL checkpoints read-only from ynez with rsync (439 MB), to
    `/scratch/cowifornia/nd-rl/circuits/ckpts/lowrl/`: `kp16` and `kdense` at
    `st{1,2,4,8,16,32}`.
  - The base sha256 is the same on ynez and serrano (`8012222f…`).
- **11:06 AM.** PREREG.md with the Stage 0 section, before Stage 0 ran: sha256 `69273f6971354f5d13a270656dfdaf8825450a9297ffacf202b974fc52b1731a`.
- **11:16 AM.** PREREG.md with the Stage 1 section added, before any crosscoder was trained: sha256 `a22483bb6acba8b1892ce58227bb25cb9df116cd5212b6e810b2dc2bd59f0856`.
- **11:07 AM.** Stage 0 ran on CPU in 40 s. Results are in [`stage0.md`](stage0.md) and
  `stage0.json`.
- **11:12 AM – 11:33 AM.** Rollouts ran on GPUs 0 and 1 (`rollouts.py`, 1,200 problems × 4
  attempts per model): 19 models plus the pretraining sample.
- **11:33 AM.** Another user (`despair8`) started vLLM jobs on GPUs 3 and 4, about 11:16 am. With
  `cipeng`'s job on GPU 2, that makes 3 GPU jobs from other users, the serrano PSU limit. I
  stopped my remaining GPU job (the GPU 1 rollout queue). Only `L2_s2` is missing; it was
  killed mid-run. No GPU work runs until fewer than 3 of their jobs remain.
