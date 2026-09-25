# Run 2 digest (auditor), newest section first

## 14:45 UTC final (search closed 14:30; all seven workers DONE; all GPUs free; no pending nodes)

**For the monitor**: final two-seed ranking for the same-card comparison: n092 (LEAN, 1286, 1139 -> 1212.5;
dev 372.5, front 210, heldout 0.936) and n078 (BEST, 1198, 1152 -> 1175; dev 383, front 198, L14 0.5, heldout
0.923). n092 leads by 37.5, under the margin 40, so BEST stayed n078 by stickiness. Checkpoints:
w2's run dirs in nodes/n092/meta.json and nodes/n078/meta.json (seed 0), audit/runs/rep_n092_s1 and
audit/runs/rep_n078_s1 (seed 1).

| node | role | score (seeds) | dev | front (L12/L13/L14) | hg | idea (worker) |
|---|---|---|---|---|---|---|
| n078 | BEST* | 1175 (1198, 1152) | 383 | 198 (156/41.5/0.5) | 0.923 | index cap on POLARIS-ramp graph-LV n071 (w2) |
| n092 | LEAN* | 1212.5 (1286, 1139) | 372.5 | 210 (164/46/0) | 0.936 | last-3 weight average onto n065 (w2) |
| n090 | DIVERSE | 1179 | 359 | 205 (159/46/0) | 0.917 | last-5 weight average (w3) |
| n096 | DIVERSE* | 1170.5 (1311, 1030) | 360.5 | 202.5 (161/41/0.5) | 0.927 | max_targets 9,000 on n090 (w3) |

**Last replicates**: n082 (1218 -> 835), n088 (1235 -> 631) and n096 (1311 -> 1030) all fell on their second
seed: max_targets 9,000 at 20 minutes is not confirmed. n092 held (1139).

**Two-seed rule over the run**: 16 replicates, mean second-seed change -154 score points, 13 of 16 lower; 10
confirmed BEST, 6 stopped (n010, n075, n082, n088, n092, n096); v12 demoted 6 more without a replicate.
Same-code pairs on seed 0 differ by 34-234 points (six pairs).

**Limiting factor at the end (survey s6)**: exploration-bound on the per-run subset at 20 minutes (L11 new solves
per round fall to single digits by the last rounds; ~40% of subset targets never solved), not sampling-bound
(sampling share ~0.55, 26-30 rounds on fast cards). The 2-hour full-pool champion (dev 499, front 322, L14 5)
shows that new targets plus time still pay; L14 at 0-1 in 20-minute runs is the open question.

## 13:00 UTC update: champion 2, 20 minutes vs 2 hours on the full pool (n060)

| run | seed | gpu | dev | L8/L9/L10/L11 | frontier | L12/L13/L14 | score | heldout | rounds | samples |
|---|---|---|---|---|---|---|---|---|---|---|
| 1,200 s (w5) | 0 | 1 | 360 | 112/97/75/76 | 176 | 137/39/0 | 1,064 | 0.934 | 26 | 333k |
| 1,200 s (auditor) | 1 | 4 | 384 | 131/105/76/72 | 187 | 149/38/0 | 1,132 | 0.911 | 30 | 376k |
| **7,200 s, max_targets 15,738** | 11 | 1 | **499** | 150/127/110/112 | **322** | 247/70/**5** | **1,787** | 0.918 | 110 | 1,400k |

Champion 1 (n024, 6,000-target subset) gained +97 dev and x3.2 frontier and stopped gaining after 50 minutes;
champion 2 on the full pool gained +127 dev and x1.8 frontier over a much stronger 20-minute base, kept solving
new pool targets to the end (10,564 of 15,738), and produced the run's first real L14 solves (5 of 100).
BEST moved to n078 at 12:51 (1198, 1152 -> 1175); n082 (1218, one seed) is queued for its replicate.

## Earlier sections, condensed
- 12:25: BEST n060 1098; 20-minute runs moved from sampling-bound to exploration-bound on their subset.
- 10:20: BEST n046 950.5; replicates ran on slower cards in 6 of 7 pairs (power cap, card-dependent clocks).
- 08:50: champion 1 (n024, 6,000-target subset): 2 h dev 410 / front 220 vs 20 min 313 / 69.5; its 50-minute
  checkpoint already had 397 / 216 (subset saturated).
- 08:20: BEST n038 769; exploration-bound on the fixed subset at 2 h.
- 06:20: metric switched to dev + 4 x frontier at 05:31 (v11); BEST n024 591; sampling-bound (LV sampler 282/s).
- 04:25: BEST n006 202.5 (dev only); exploration-bound at L11+; n012 NSR ruled compliant.
