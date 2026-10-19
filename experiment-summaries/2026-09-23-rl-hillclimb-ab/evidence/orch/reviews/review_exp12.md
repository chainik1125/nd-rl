# Review exp12 (per-target weighting: ~12 records per solved target split over its ≤ 4 proofs; on exp10 code; seed 0)

## Recount (written before reading preregistration / numbers.md / log.md for this run)

### Recount of `/work/exec/runs/exp12_pertarget_s0` (tools/recount.py)

- status `ok`, seed 0, budget 600, gpu `1:NVIDIA GeForce RTX 3090`, started_utc 2026-09-23T02:18:39Z, rl_secs 572.5, smoke False
- frozen hashes: 18 checked, mismatched none, not in SHA256SUMS none, frozen files absent from summary none; source/rl.py sha == summary rl.py sha: True
- summary metric 268, by_len {'7': 71, '8': 87, '9': 110}, heldout_greedy 0.899
- score.json: solved 268, by_len {'7': 71, '8': 87, '9': 110}, k 32, seed 0, official True, smoke False; final.pt sha matches: True; score == summary metric: True; by_len sum 268
- metrics.jsonl: 43 records {'sample': 6, 'train': 35, 'summary': 1, 'eval': 1}; step monotone True; sample rounds [1, 2, 3, 4, 5, 6]; sum(samples) 139291; last targets_solved_cum 627
  - vs rl_stats: rounds 6 (=), total_samples 139291 (=), targets_solved_cum 627 (=), total_train_steps 1750 (max step 1751)
  - harness eval heldout_greedy 0.899 == summary: True
  - r1: n_prompts 1500 k 16 T 0.8 samples 24000 accept 0.0232 solved_round 103 cum 103 by_len {'7': 102, '8': 1, '9': 0} new 121
  - r2: n_prompts 1499 k 16 T 0.8 samples 23984 accept 0.1262 solved_round 296 cum 297 by_len {'7': 230, '8': 66, '9': 1} new 218
  - r3: n_prompts 1498 k 16 T 0.8 samples 23968 accept 0.2543 solved_round 482 cum 484 by_len {'7': 332, '8': 133, '9': 19} new 212
  - r4: n_prompts 1498 k 16 T 0.8 samples 23968 accept 0.3233 solved_round 564 cum 566 by_len {'7': 374, '8': 161, '9': 31} new 94
  - r5: n_prompts 1496 k 16 T 0.8 samples 23936 accept 0.3703 solved_round 598 cum 602 by_len {'7': 395, '8': 169, '9': 38} new 42
  - r6: n_prompts 1495 k 13 T 0.8 samples 19435 accept 0.394 solved_round 622 cum 627 by_len {'7': 410, '8': 177, '9': 40} new 28
  - train records 35: loss first 0.0579 last 0.0308; n_records last 15524
- found_targets.jsonl: 715 rows, 715 raw-distinct, 715 distinct by my normaliser; 627 targets solved, by L_true {7: 410, 8: 177, 9: 40}; names not in target pools: 0; thm field matches pool: 715/715
  - re-verified with nd_verify against the pool prompt: 150/150 accepted
- other files: ['round_01.pt', 'round_02.pt', 'round_03.pt', 'round_04.pt', 'round_05.pt', 'round_06.pt']

Per-round secs / sample / ft / steps / RL records / mix: 93/51/40/300/1,236/9,236; 96/50/44/300/3,564/11,564; 98/50/45/300/5,808/13,808; 100/51/46/300/6,792/14,792; 99/50/46/300/7,224/15,224; r6 shrunk (k 13) 85/43/39/250/7,524/15,524. RL share of the mix 13 % → 48 % (exp10: 6 % → 27 %; exp11 at retain 2k: 19 % → 58 %).

Dev per length: **L7 71** (the highest; exp11 62; all others 33–49), L8 87, L9 110. heldout_greedy 0.899.

### rl.py static checks (/work/exec/runs/exp12_pertarget_s0/source/rl.py, 340 lines, diff vs frozen seed: 117 changed lines)
```
-- forbidden paths / sealed / other workspaces:
   none
```

**Hard constraints:** build_mix repetition only (records per target = round(12 / n_proofs), so a target with 1 proof gets 12 copies vs 4 before). Pass. **Gate 1:** pass. GPU1 02:18–02:31; no reviewer rescore overlapped.

Seed-1 rescore: queued first (268 is +19 over the incumbent, the first gain past the SPEC's 15 threshold).

## Compare (after reading preregistration/exp12.md, numbers.md, log.md)

Gate 0: prereg commit 47b6469 at 2026-09-23T02:15:53Z; started_utc 02:18:39Z. **Pass.**

| claim (source) | reviewer's independent value | verdict |
|---|---|---|
| dev 268, 71/87/110; heldout 0.899 | 268, 71/87/110; 0.899 | reproduces |
| prediction: dev = exp10 + 15 = 267, falsified if ≤ 257: "HIT on the point estimate, single seed" | 268 vs 267 | reproduces, and the single-seed caveat is stated |
| (prereg) "L9 up most", "speed the frontier toward L8/L9" | L9 110 vs 125 (exp10), −15; L8 87 = 87; L7 71 vs 40, +31 | **mechanism missed**: the executor's log says so ("the gain is all in L7; L9 fell"), but the prediction is scored as a HIT. It hit the total for a different reason than predicted. Report it as "total hit, mechanism miss" |
| exp11 (retain 2k) also raised L7 (62) | exp11 L7 62 | reproduces. Common factor I see: both raise the RL share of the fine-tune mix (exp11 19–58 %, exp12 13–48 %, vs 6–27 % for exp10). Per-target weighting itself is not isolated |
| seed-1 replicate queued | not yet run | pending |

Wording: "frontier" should not be used for this result. The gain is on the easiest dev length (L7), and on the training targets the frontier length (L9) solved count is 40, the same as exp10.

## Verdict

**REJECT.** Sound (gates 0–2 pass), but not better than the incumbent: exp13 (324) was adopted before this verdict was written. Against the old incumbent (249) it was +19 on a single seed, above the SPEC's 15 but inside my near-control spread (~30). The L7 gain overlaps with what exp11 showed and is dwarfed by exp13's L7 109. The seed-1 rescore stays queued, behind exp13's.
