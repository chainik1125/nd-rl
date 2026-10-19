# Auditor report, run 2 (ab2), 2026-09-24

**What the two-seed rule changed.** 16 replicates; the second seed averaged -154 score points and was lower in 13
of 16 (winner's curse, plus slower cards under the 150 W cap). It stopped six
candidates that single-seed promotion would have made BEST: n010, n075 (1183 -> 840), n082 (1218 -> 835), n088
(1235 -> 631), n096 (1311 -> 1030) and n092 (1286 -> 1139, now LEAN, 37.5 above BEST but inside the margin). Six
same-code pairs on seed 0 differed by 34-234 points: the margin (40) is below the noise. Final BEST n078: 1198, 1152 -> 1175 (dev 383, frontier 198/1000).

**Champion curve (2 h vs 20 min two-seed).** n024 on its 6,000-target subset: dev 410 vs 313, frontier 220 vs
69.5. All of the gain came in the first 50 minutes (checkpoint: 397 / 216), then the subset saturated. n060 with the
full pool (max_targets 15,738): dev 499 vs 372, frontier 322 vs 181.5, L14 5 of 100 (the first real L14 solves).
A first attempt on n021 was aborted when BEST moved.

**Violations.** None in 99 nodes. Rulings obtained: NSR / DPO negatives on own verifier-rejected samples are
allowed (04:32); output-head or logit scaling in the checkpoint is not promotable (06:00; n028 was a no-op).

**What I would change.**
1. Replicate on the first seed's card, and log the gpu per run.
2. Add a `duplicate` path to the controller: identical-code nodes (n022, n049, n063, n079, n098) are extra runs
   of one recipe.
3. Make champion runs budget-aware by default (full pool); a fixed subset wastes most of a long run.
4. Start champions only after the pending queue resolves, so they never run a superseded BEST.
5. Give auditor jobs priority in the GPU FIFO.
