---
written_on: 2026-09-24
---
# w4 report (GEAR run 2, ab2)

(Saved by the monitor from w4's session hand-back; the session's own file write was refused.)

**Frontier (L12-14, 1,000-theorem subset, pass@64).** Every checkpoint started at 0. My best single-seed frontier
counts were 215 (n088) and 208 (n082); n082 also had the run's first L14 solve. Neither held up on a second seed.

**BEST at 14:30:** n078 (w2), two-seed score 1175 (dev 383, seeds 378 and 388; frontier 198, seeds 205 and 191).
Checkpoints: `/scratch/cowifornia/nd-rl/ab2/w2/runs/e12_n071_idx/final.pt` and the auditor's `rep_n078_s1`.
Also in the frontier: LEAN n092 with a two-seed mean of 1212.5, and PENDING n096 at 1311 on one seed.

**My best checkpoint that held up:** n049, score 754 (dev 318, frontier 109):
`/scratch/cowifornia/nd-rl/ab2/w4/runs/x_idx_schema_s0/final.pt`. My highest single seed was n088 at 1235, but its
replicate gave a mean of 933 and it was discarded.

## What moved the scores
- Line-verified sampling (w1's checker) in my crossovers: n029 (lv plus n006's hot line heads) dev 285 / frontier 64
  against 202.5 / 9 for n006; n022 (lv plus prefix branching) dev 227.
- Easy-to-hard ordering of the first pass (E2H): n042 597 against 541 for its parent; n056 +22 dev, lost frontier.
- Schema coverage on top of the index cap: n049 754 against 716.5.
- Staged temperature on top of the index cap: n075 1183 on one seed, two-seed mean 1011.5.
- A larger per-run subset (max_targets 9000): single seeds 1218 (n082) and 1235 (n088); both second seeds ran only
  20-21 rounds on slower GPUs and dropped, giving means of 1026.5 and 933. Refuted as a fixed setting; the gain seems
  to need about 30 rounds to cover the first pass. 12000 targets (n095) scored 1104, 16 below its parent.

## What did not help
| idea | node | score |
|---|---|---|
| knapsack k by length | n001 / n015 | 179 dev (baseline 185) / 114 dev (-70 vs parent) |
| annealed decoding (EAD) | n007 | 107 dev |
| backtracking combined with branching | n035 | 154 (13 rounds only) |
| schema coverage on the ar3po line | n068 | -215 |
| ar3po two-stage adaptive rollout | n062 | +19.5, noise |

## Controller steps (all passed dry-run and test_controller, 27/27 at v13 and v14)
- v12: when a new BEST is confirmed, pending nodes whose single seed no longer clears the margin are demoted without
  a replicate.
- v13: p2 and the forced partner of a fresh elite can no longer be a code ancestor or descendant.
- v14: margins raised to 40 / 80, as the monitor asked.

## Sources tested
Knapsack RL arXiv 2509.25849; EAD arXiv 2510.05251; E2H arXiv 2506.06632; AR3PO arXiv 2509.25808; BFS-Prover line
checks (via w1) arXiv 2502.03438; POLARIS (via n067) https://hkunlp.github.io/blog/2025/Polaris.

## Noise
One seed moves the score by about +-200. Much of that is throughput: the GPUs are power-capped at 150 W, and a run
gets 20 to 30 rounds depending on the GPU.

## Next
Replicate n096. Tie the subset size to the measured round rate, especially for the 7,200 s champion. Record the
round count as a covariate when comparing seeds.
