---
written_on: 2026-11-01
written_by: agent:claude
---

# Weight-change SVD by RL round, and SAE features of the RL change

Robbie's overnight run for the SPAR midterm, 2026-11-01, on seeds of Dan's trajectory ladder (`la_T1_best12`; r1-r8
from `trajectory`, r9-r16 from `rl-continue`), scored with Dan's `state_eval` protocol (T = 1, 48 steps, 256 tokens per
action, sampling seed 0, Lean 4.34.1 alone judges) on 255 of Charles's textbook problems (v0, v1, batch3, lean_v2; k = 32).

- **SVD** (`run_svd.py`): keep the top n singular components of each matrix's change. x = n, y = pass@k, one line per round.
  Written before the sweep: 1-D parameters move to round r (`full`). Holding them at the pretrain (`base`) was the first
  choice and failed its check against Charles's Figure 4: r16 n = 4 / 8 fell below the pretrain, with grammar failures.
- **SAE** (`run_sae.py`): see its docstring. Expected from Charles's steering null and Leon's crosscoders: steering
  recovers little (< 0.2 of the pass@32 gap); removal costs more than steering gains.

`launch.py` holds the pods; `analyze.py` writes `numbers.md` and `figures/`. Fork commit `4dcd0e4f`.

## Results (2026-11-01, 3 seeds; full tables in `numbers.md`)

Observed, share of the pass@1 gap recovered, mean of seeds 0/1/2 (per-seed values are within about 0.1 of the mean):

| round | n = 1 | n = 4 | n = 16 | n = 64 | energy kept at n = 64 |
|---|---|---|---|---|---|
| 1 | +0.18 | +0.36 | +0.67 | +0.95 | 0.89 |
| 2 | +0.14 | +0.25 | +0.59 | +0.95 | 0.87 |
| 4 | +0.09 | +0.20 | +0.49 | +0.94 | 0.84 |
| 8 | +0.03 | +0.12 | +0.43 | +0.92 | 0.81 |
| 16 | -0.22 | -0.05 | +0.27 | +0.87 | 0.76 |

SAE, round 16, block 4, share of the pass@32 gap, seeds 0/1/2: mean-difference steering +0.03/-0.01/-0.01; best steering
arm (`steer_map_n256`) -0.01/+0.17/+0.07; removal `ablate_map_n256` +0.16/+0.11/+0.13 against random 1024 at
+0.04/-0.05/+0.03; removal of all latents +0.38/+0.32/+0.47, which has no random control. At rounds 2 and 4 the random
1024-latent removal itself loses 0.15, so removal there is barely above control. Seed 0's `steer_map_n1024` recovers 0.10-0.18 at every round, against -0.10 to +0.03 for random 1024. Expectations written
above: removal did cost more than steering gained at round 16; steering stayed under 0.2 except `steer_map_n8192` at round 1
(+0.29 [0.12, 0.46], one seed, small gap, no control possible).
