# Review exp11 (retain 2,000 cap-6 records on exp10 code; seed 0)

## Recount (written before reading preregistration / numbers.md / log.md for this run)

### Recount of `/work/exec/runs/exp11_retain2k_s0` (tools/recount.py)

- status `ok`, seed 0, budget 600, gpu `0:NVIDIA GeForce RTX 3090`, started_utc 2026-09-23T02:15:00Z, rl_secs 575.4, smoke False
- frozen hashes: 18 checked, mismatched none, not in SHA256SUMS none, frozen files absent from summary none; source/rl.py sha == summary rl.py sha: True
- summary metric 255, by_len {'7': 62, '8': 85, '9': 108}, heldout_greedy 0.871
- score.json: solved 255, by_len {'7': 62, '8': 85, '9': 108}, k 32, seed 0, official True, smoke False; final.pt sha matches: True; score == summary metric: True; by_len sum 255
- metrics.jsonl: 48 records {'sample': 7, 'train': 39, 'summary': 1, 'eval': 1}; step monotone True; sample rounds [1, 2, 3, 4, 5, 6, 7]; sum(samples) 152876; last targets_solved_cum 611
  - vs rl_stats: rounds 7 (=), total_samples 152876 (=), targets_solved_cum 611 (=), total_train_steps 1925 (max step 1926)
  - harness eval heldout_greedy 0.871 == summary: True
  - r1: n_prompts 1500 k 16 T 0.8 samples 24000 accept 0.0232 solved_round 103 cum 103 by_len {'7': 102, '8': 1, '9': 0} new 121
  - r2: n_prompts 1499 k 16 T 0.8 samples 23984 accept 0.1255 solved_round 307 cum 308 by_len {'7': 253, '8': 55, '9': 0} new 225
  - r3: n_prompts 1499 k 16 T 0.8 samples 23984 accept 0.2649 solved_round 484 cum 485 by_len {'7': 330, '8': 135, '9': 20} new 193
  - r4: n_prompts 1499 k 16 T 0.8 samples 23984 accept 0.3306 solved_round 547 cum 548 by_len {'7': 358, '8': 154, '9': 36} new 72
  - r5: n_prompts 1498 k 16 T 0.8 samples 23968 accept 0.3651 solved_round 583 cum 585 by_len {'7': 380, '8': 165, '9': 40} new 43
  - r6: n_prompts 1498 k 16 T 0.8 samples 23968 accept 0.3871 solved_round 602 cum 604 by_len {'7': 391, '8': 170, '9': 43} new 25
  - r7: n_prompts 1498 k 6 T 0.8 samples 8988 accept 0.3983 solved_round 609 cum 611 by_len {'7': 395, '8': 172, '9': 44} new 8
  - train records 39: loss first 0.0511 last 0.0298; n_records last 4748
- found_targets.jsonl: 687 rows, 687 raw-distinct, 687 distinct by my normaliser; 611 targets solved, by L_true {7: 395, 8: 172, 9: 44}; names not in target pools: 0; thm field matches pool: 687/687
  - re-verified with nd_verify against the pool prompt: 150/150 accepted
- other files: ['round_01.pt', 'round_02.pt', 'round_03.pt', 'round_04.pt', 'round_05.pt', 'round_06.pt', 'round_07.pt']

Per-round secs / sample / ft / steps / RL records / mix size: 89/52/35/300/484/2,484; 86/48/36/300/1,384/3,384; 89/48/39/300/2,156/4,156; 90/47/40/300/2,444/4,444; 92/48/41/300/2,616/4,616; 91/47/41/300/2,716/4,716; r7 shrunk (tail-fill) 38/19/17/125/2,748/4,748 at k 6. RL share of the mix 19 % → 58 %, vs 6 % → 27 % at retain 8k (exp10) and 2.4 % → 11 % at 20k (exp01).

Dev per length: L7 62 (other runs 33–49), L8 85, L9 108. heldout_greedy 0.871, the lowest of any run (others 0.902–0.927).

### rl.py static checks (/work/exec/runs/exp11_retain2k_s0/source/rl.py, 337 lines, diff vs frozen seed: 110 changed lines)
```
-- forbidden paths / sealed / other workspaces:
   none
```

**Hard constraints:** one knob (retain 8000 → 2000). Pass. **Gate 1:** pass.

Seed-1 rescore: queued (new top raw score, 255).

## Compare (after reading preregistration/exp11.md, numbers.md, log.md)

Gate 0: prereg commit 7cd7de7 at 2026-09-23T02:06:41Z; started_utc 02:15:00Z. **Pass.**

| claim (source) | reviewer's independent value | verdict |
|---|---|---|
| dev 255, 62/85/108; heldout 0.871 | 255, 62/85/108; 0.871 | reproduces |
| prediction (270, L9 ≥ 130; falsified ≤ 262): MISSED | 255, L9 108 | reproduces; honest miss. The heldout expectation (0.85–0.89) was met |
| L7 rose (62 vs 33–49), L9 fell (108), heldout fell 0.04 | L7 62 vs 33–49 in the other 10 runs; L9 108 vs 125 in exp10; heldout 0.910 → 0.871 (−0.039) | reproduces. L7 +13 over the previous max is the largest per-length move so far, but it is single-seed |
| new proofs +72/+43/+25/+8 in rounds 4–7; cum 611/1500 | 72 / 43 / 25 / 8; 611 | reproduces. Round 7 was a shrunk round (k 6), so its +8 is not comparable to the others |
| "training-target frontier saturates" | the cumulative solved count flattens (548 → 585 → 604 → 611). With skip_saturated only the unsaturated targets are resampled, so this reflects the remaining unsolved ~890 targets being hard for the model at k 16 | fair as a description of this run |
| (prereg) "runs with smaller retain (exp03, exp08) had the highest L9" | at prereg time: exp08 126 (retain 8k), exp03 111 (8k) vs 109/112 (20k). exp03's 111 is not the highest. exp10 (8k) 125 came later | half supported: true for exp08 only |

Note: the executor smoke-ran exp13's variant on GPU0 at 02:19 during this run (5 s). Negligible, but it is GPU sharing.

## Verdict

**REJECT** (a single-seed difference too small to count: +6 over 249; heldout 0.871, −0.049 vs the incumbent, is clearly worse). Gates 0–2 pass. Seed-1 rescore queued; it will be appended. If the executor wants to build on retain 2k, a second training seed has to show the L7 jump (62) is real before the heldout cost is worth paying.

## Seed-1 rescore (04:13–04:16 UTC, GPU1)

250/1610 (57/86/107) vs 255 at seed 0 (−5). The REJECT stands.
