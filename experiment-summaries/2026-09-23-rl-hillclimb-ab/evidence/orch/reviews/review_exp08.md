# Review exp08 (fast iteration on exp06 code: k16, ft300, retain 8k, ≤12 rounds, tail-fill, skip saturated; seed 0)

## Recount (written before reading preregistration / numbers.md / log.md for this run)

### Recount of `/work/exec/runs/exp08_fastiter2_s0` (tools/recount.py)

- status `ok`, seed 0, budget 600, gpu `1:NVIDIA GeForce RTX 3090`, started_utc 2026-09-23T01:54:01Z, rl_secs 568.1, smoke False
- frozen hashes: 18 checked, mismatched none, not in SHA256SUMS none, frozen files absent from summary none; source/rl.py sha == summary rl.py sha: True
- summary metric 252, by_len {'7': 39, '8': 87, '9': 126}, heldout_greedy 0.902
- score.json: solved 252, by_len {'7': 39, '8': 87, '9': 126}, k 32, seed 0, official True, smoke False; final.pt sha matches: True; score == summary metric: True; by_len sum 252
- metrics.jsonl: 37 records {'sample': 5, 'train': 30, 'summary': 1, 'eval': 1}; step monotone True; sample rounds [1, 2, 3, 4, 5]; sum(samples) 119696; last targets_solved_cum 617
  - vs rl_stats: rounds 5 (=), total_samples 119696 (=), targets_solved_cum 617 (=), total_train_steps 1500 (max step 1501)
  - harness eval heldout_greedy 0.902 == summary: True
  - r1: n_prompts 1500 k 16 T 0.8 samples 24000 accept 0.023 solved_round 100 cum 100 by_len {'7': 99, '8': 1, '9': 0} new 118
  - r2: n_prompts 1499 k 16 T 0.8 samples 23984 accept 0.1062 solved_round 299 cum 301 by_len {'7': 238, '8': 63, '9': 0} new 228
  - r3: n_prompts 1497 k 16 T 0.8 samples 23952 accept 0.2424 solved_round 498 cum 501 by_len {'7': 345, '8': 142, '9': 14} new 236
  - r4: n_prompts 1493 k 16 T 0.8 samples 23888 accept 0.3165 solved_round 570 cum 577 by_len {'7': 376, '8': 167, '9': 34} new 98
  - r5: n_prompts 1492 k 16 T 0.8 samples 23872 accept 0.3541 solved_round 609 cum 617 by_len {'7': 393, '8': 182, '9': 42} new 49
  - train records 30: loss first 0.0689 last 0.0422; n_records last 10904
- found_targets.jsonl: 729 rows, 729 raw-distinct, 729 distinct by my normaliser; 617 targets solved, by L_true {7: 393, 8: 182, 9: 42}; names not in target pools: 0; thm field matches pool: 729/729
  - re-verified with nd_verify against the pool prompt: 150/150 accepted
- other files: ['round_01.pt', 'round_02.pt', 'round_03.pt', 'round_04.pt', 'round_05.pt']

Per-round secs / sample / ft: 105/65/38, 114/70/42, 115/69/44, 116/68/46, 117/69/46; 5 full rounds (tail-fill did not fire). Sampling 342–369 samples/s on GPU1. rl_secs 568.1.

### rl.py static checks (/work/exec/runs/exp08_fastiter2_s0/source/rl.py, 265 lines, diff vs frozen seed: 34 changed lines)
```
-- forbidden paths / sealed / other workspaces:
   none
```

**Hard constraints:** knob changes on exp06 code (reviewed clean). Pass. **Gate 1:** pass. No reviewer rescore overlapped it (GPU1, 01:54–02:06).

Seed-1 rescore: queued. This is the top raw score (252 vs 249), so it counts as a candidate. Waiting for a GPU not listed in gpu_in_use.txt.

## Compare (after reading preregistration/exp08.md, numbers.md, log.md)

Gate 0: prereg commit e1febdf at 2026-09-23T01:44:23Z; started_utc 01:54:01Z. **Pass.**

| claim (source) | reviewer's independent value | verdict |
|---|---|---|
| dev 252, 39/87/126; heldout 0.902 | 252, 39/87/126; 0.902 | reproduces |
| 5 rounds, 65–70 s sampling + ~45 s ft each | 5 rounds; sampling 65–70 s; ft 38–46 s | reproduces |
| prediction (6–7 rounds, 285; falsified ≤ 255): MISSED | 5 rounds, 252 | reproduces; honest miss. The round-count expectation was also missed (5, not 6–7) |
| "best L9 so far (126 vs 109/112)" | L9 across runs: 126 (exp08), 112 (exp05), 111 (exp03), 109 (exp01), 104, 92, 91, 70 | reproduces. +14 over the best near-control is one seed and at the noise floor; a lead to replicate, not a finding |
| "total +3 vs control is noise" | +3 vs incumbent 249 | agrees |
| heldout ≥ 0.89 (prereg) | 0.902 | met; the lowest of any run so far (smaller retain, 8k) |

Of note: at 64 samples per target it has fewer samples than the control (119,696 vs 144,000), and it still matches it. With exp03 (same knobs, confounded) this gives two runs of the fast-iteration knobs at 238 and 252.

## Verdict

**REJECT** (a single-seed difference too small to count: +3 over the incumbent's 249, far inside the ~15–30 noise floor; heldout 0.902 < 0.920). Gates 0–2 pass. The seed-1 rescore is still queued (no free GPU) and will be appended. It cannot change this verdict, only the noise estimate. Lead worth a second training seed: L9 126.
