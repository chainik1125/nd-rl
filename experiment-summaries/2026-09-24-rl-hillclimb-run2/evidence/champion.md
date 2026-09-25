# Champion runs (auditor)

Every 3 h the current BEST node's rl.py is run for 7,200 s (`--budget 7200`, new seed). Compared with the node's 1,200 s run(s).

| start UTC | node | seed | 20-min dev | 2-h dev | 20-min frontier | 2-h frontier | 20-min heldout | 2-h heldout | rounds (2 h) | samples (2 h) | run dir |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 06:05 (aborted 06:19) | n021 | 7 | score 414 (dev 249, 211; front 66, 26) | - | - | - | 0.918, 0.902 | - | 13 rounds done at abort | - | runs/champion_0540_n021_aborted (stopped at 791 s, monitor 06:17: BEST moved to n024 at 06:11) |
| 06:39 | n024 | 7 | 313 (314, 312) | **410** (L8 127, L9 108, L10 88, L11 87) | 69.5 (59, 80) | **220** (L12 171, L13 49, L14 0) | 0.923 (0.913, 0.932) | 0.937 | 91 (vs 22, 18) | 1,159,966 (vs 273k, 229k); 12,235 optimiser steps | runs/champion_0619 (7,178 s RL; 4,178 of 6,000 subset targets solved, 3,890 of them by round 43 / 50 min) |
| (intermediate) | n024 champion 1, round_43.pt at 2,991 s (50 min) | 7 | - | 397 (L8 125, L9 106, L10 83, L11 83) | - | 216 (L12 166, L13 49, L14 1) | - | - | 43 | ~560k | runs/champion_0619_r43 (scored 09:12-09:24 on request, monitor 08:55): the last 70 min added +13 dev, +4 frontier, i.e. nothing beyond scoring noise after the subset saturated |
| 10:50 | n060 + **override max_targets 15,738** (monitor 08:25; only change vs nodes/n060/rl.py) | 11 | 372 (360, 384) | **499** (L8 150, L9 127, L10 110, L11 112) | 181.5 (176, 187) | **322** (L12 247, L13 70, L14 5) | 0.923 (0.934, 0.911) | 0.918 | 110 (vs 26, 30) | 1,400,507 (vs 333k, 376k); 14,756 optimiser steps | runs/champion2_1030 (gpu1; 7,181 s RL; 10,564 of 15,738 pool targets solved) |
