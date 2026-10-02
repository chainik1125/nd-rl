# Review exp14 (frontier focus: from round 4 sample only targets with < 2 proofs, k 48, T 1.0; on exp10 code; seed 0)

## Recount (written before reading preregistration / numbers.md / log.md for this run)

### Recount of `/work/exec/runs/exp14_focus_s0` (tools/recount.py)

- status `ok`, seed 0, budget 600, gpu `1:NVIDIA GeForce RTX 3090`, started_utc 2026-09-23T02:30:51Z, rl_secs 569.4, smoke False
- frozen hashes: 18 checked, mismatched none, not in SHA256SUMS none, frozen files absent from summary none; source/rl.py sha == summary rl.py sha: True
- summary metric 269, by_len {'7': 44, '8': 94, '9': 131}, heldout_greedy 0.897
- score.json: solved 269, by_len {'7': 44, '8': 94, '9': 131}, k 32, seed 0, official True, smoke False; final.pt sha matches: True; score == summary metric: True; by_len sum 269
- metrics.jsonl: 35 records {'sample': 5, 'train': 28, 'summary': 1, 'eval': 1}; step monotone True; sample rounds [1, 2, 3, 4, 5]; sum(samples) 175032; last targets_solved_cum 639
  - vs rl_stats: rounds 5 (=), total_samples 175032 (=), targets_solved_cum 639 (=), total_train_steps 1352 (max step 1353)
  - harness eval heldout_greedy 0.897 == summary: True
  - r1: n_prompts 1500 k 16 T 0.8 samples 24000 accept 0.0232 solved_round 103 cum 103 by_len {'7': 102, '8': 1, '9': 0} new 121
  - r2: n_prompts 1499 k 16 T 0.8 samples 23984 accept 0.0985 solved_round 261 cum 262 by_len {'7': 218, '8': 44, '9': 0} new 178
  - r3: n_prompts 1498 k 16 T 0.8 samples 23968 accept 0.2237 solved_round 456 cum 458 by_len {'7': 329, '8': 122, '9': 7} new 232
  - r4: n_prompts 1444 k 48 T 1.0 samples 69312 accept 0.266 solved_round 532 cum 588 by_len {'7': 390, '8': 168, '9': 30} new 169
  - r5: n_prompts 1407 k 24 T 1.0 samples 33768 accept 0.3153 solved_round 546 cum 639 by_len {'7': 408, '8': 188, '9': 43} new 69
  - train records 28: loss first 0.0698 last 0.0457; n_records last 11064
- found_targets.jsonl: 769 rows, 769 raw-distinct, 769 distinct by my normaliser; 639 targets solved, by L_true {7: 408, 8: 188, 9: 43}; names not in target pools: 0; thm field matches pool: 769/769
  - re-verified with nd_verify against the pool prompt: 150/150 accepted
- other files: ['round_01.pt', 'round_02.pt', 'round_03.pt', 'round_04.pt', 'round_05.pt']

Per-round secs / sample / ft / steps: 89/49/38/300, 93/50/42/300, 95/50/43/300, r4 (focus, k 48) 192/142/44/300, r5 (focus, shrunk k 24) 98/72/23/152. Textbook-source targets solved: 2.

### rl.py static checks (/work/exec/runs/exp14_focus_s0/source/rl.py, 345 lines, diff vs frozen seed: 120 changed lines)
```
-- forbidden paths / sealed / other workspaces:
   none
```

**Hard constraints:** sampling-allocation change only. Pass. **Gate 1:** pass.

Seed-1 rescore: not queued. 269 is far below the incumbent's 324.

## Compare (after reading preregistration/exp14.md, numbers.md, log.md)

Gate 0: prereg commit 0f4c9fb at 2026-09-23T02:27:25Z; started_utc 02:30:51Z. **Pass.**

| claim (source) | reviewer's independent value | verdict |
|---|---|---|
| dev 269, 44/94/131; heldout 0.897 | 269, 44/94/131; 0.897 | reproduces |
| cum solved ≥ 680: MISSED (639) | 639 | reproduces |
| (prereg) L9 training targets solved ≥ 60 (exp10: 40) | 43 | **missed; not mentioned in the log.** Should be reported as a miss |
| "falsified overall per the 'or' rule" | correct reading of its own criterion | reproduces |
| "Best L9 (131)" | dev L9 131 vs 129 (exp13), 126, 125 | reproduces; +2 over exp13 is noise |
| focus rounds costly: r4 69k samples, 142 s; 5 rounds | r4 69,312 samples, 142 s sampling; 5 rounds | reproduces |

## Verdict

**REJECT.** Sound, but not better than the incumbent (269 vs 324). Against exp10 (252), its own comparison, it is +17 on one seed, borderline noise. The dev gain is in L8/L9 (+7/+6 over exp10), which is where the focus was aimed, but training-target L9 solves did not reach the predicted 60.
