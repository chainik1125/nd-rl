# Review exp10 (fast iteration knobs + custom fast sampler + tail-fill + skip saturated; seed 0)

## Recount (written before reading preregistration / numbers.md / log.md for this run)

### Recount of `/work/exec/runs/exp10_fastiter_fastgen_s0` (tools/recount.py)

- status `ok`, seed 0, budget 600, gpu `1:NVIDIA GeForce RTX 3090`, started_utc 2026-09-23T02:06:26Z, rl_secs 572.7, smoke False
- frozen hashes: 18 checked, mismatched none, not in SHA256SUMS none, frozen files absent from summary none; source/rl.py sha == summary rl.py sha: True
- summary metric 252, by_len {'7': 40, '8': 87, '9': 125}, heldout_greedy 0.91
- score.json: solved 252, by_len {'7': 40, '8': 87, '9': 125}, k 32, seed 0, official True, smoke False; final.pt sha matches: True; score == summary metric: True; by_len sum 252
- metrics.jsonl: 44 records {'sample': 6, 'train': 36, 'summary': 1, 'eval': 1}; step monotone True; sample rounds [1, 2, 3, 4, 5, 6]; sum(samples) 143808; last targets_solved_cum 614
  - vs rl_stats: rounds 6 (=), total_samples 143808 (=), targets_solved_cum 614 (=), total_train_steps 1800 (max step 1801)
  - harness eval heldout_greedy 0.91 == summary: True
  - r1: n_prompts 1500 k 16 T 0.8 samples 24000 accept 0.0232 solved_round 103 cum 103 by_len {'7': 102, '8': 1, '9': 0} new 121
  - r2: n_prompts 1499 k 16 T 0.8 samples 23984 accept 0.1006 solved_round 268 cum 269 by_len {'7': 230, '8': 39, '9': 0} new 185
  - r3: n_prompts 1498 k 16 T 0.8 samples 23968 accept 0.2273 solved_round 453 cum 455 by_len {'7': 317, '8': 128, '9': 10} new 222
  - r4: n_prompts 1497 k 16 T 0.8 samples 23952 accept 0.2986 solved_round 546 cum 549 by_len {'7': 357, '8': 162, '9': 30} new 108
  - r5: n_prompts 1497 k 16 T 0.8 samples 23952 accept 0.3377 solved_round 586 cum 589 by_len {'7': 379, '8': 173, '9': 37} new 52
  - r6: n_prompts 1497 k 16 T 0.8 samples 23952 accept 0.3669 solved_round 611 cum 614 by_len {'7': 391, '8': 183, '9': 40} new 35
  - train records 36: loss first 0.0698 last 0.0418; n_records last 10880
- found_targets.jsonl: 723 rows, 723 raw-distinct, 723 distinct by my normaliser; 614 targets solved, by L_true {7: 391, 8: 183, 9: 40}; names not in target pools: 0; thm field matches pool: 723/723
  - re-verified with nd_verify against the pool prompt: 150/150 accepted
- other files: ['round_01.pt', 'round_02.pt', 'round_03.pt', 'round_04.pt', 'round_05.pt', 'round_06.pt']

Per-round secs / sample / ft: 88/49/37, 91/49/41, 98/51/45, 97/49/46, 98/50/45, 98/49/46; 6 full rounds; sampling 467–491 samples/s on GPU1 (exp08 with lib sampler, same GPU: 342–369). rl_secs 572.7. Total samples 143,808, about equal to the control's 144,000.

Same total as exp08 (252; by_len 40/87/125 vs 39/87/126). I checked that it is not the same checkpoint: final.pt sha cfc5f3f3… vs cbf4cadd…, score.json ckpt_sha256 match their own files, and the found target sets overlap 578 of 617/614. The equality is a coincidence.

### rl.py static checks (/work/exec/runs/exp10_fastiter_fastgen_s0/source/rl.py, 337 lines, diff vs frozen seed: 110 changed lines)
```
-- forbidden paths / sealed / other workspaces:
   none
```

**Hard constraints:** exp09's sampler + exp08's knobs (both reviewed clean). Pass. **Gate 1:** pass. No reviewer rescore overlapped it (GPU1 02:06–02:19).

Seed-1 rescore: queued after exp08's (top-tied raw score).

## Compare (after reading preregistration/exp10.md, numbers.md, log.md)

Gate 0: prereg commit 3417dd7 at 2026-09-23T02:06:25Z; started_utc 02:06:26Z. **Pass** (1 s).

| claim (source) | reviewer's independent value | verdict |
|---|---|---|
| dev 252, 40/87/125; heldout 0.910 | 252, 40/87/125; 0.91 | reproduces |
| 6 rounds, ~49 s sampling each at ~490/s | 6 rounds; 49–51 s; 467–491 samples/s | reproduces |
| prediction (6 rounds, 265, L9 ≥ 125; falsified ≤ 250): "counted as a miss" | 6 rounds ✓, L9 125 ✓, 252 > 250 (not formally falsified) but 13 below the point estimate | the executor's self-assessment is stricter than its own criterion. Fair and honest |
| "extra rounds past ~5 add nothing" | exp08 (5 rounds) 252 vs exp10 (6 rounds) 252; one seed each, and the runs also differ in sampler and total samples (119,696 vs 143,808) | **reword**: "a 6th round did not measurably help in this pair (n = 1 each)". "Nothing" cannot be resolved at a ±15 noise floor |
| "the plateau is not compute" | two configurations at 252 are not evidence of a plateau. The near-control spread (219–249) is as wide as all the "gains". Adding rounds and samples by ~20 % did not move the score, and that is all these runs show | **not supported** as worded; rephrase as a hypothesis |
| heldout ~0.90 (prereg) | 0.910 | met |

Across the two fast-iteration runs (exp08, exp10; both seed 0) L9 is 126 / 125, against 92–112 for the seed-knob runs. It is the only per-length difference so far that shows up in two runs, but both runs use training seed 0, so it is not a two-seed result.

## Verdict

**REJECT** (a single-seed difference too small to count: +3 over 249; heldout 0.910 < 0.920). Gates 0–2 pass. Seed-1 rescore queued behind exp08's; it will be appended.

## Seed-1 rescore (04:16–04:18 UTC, GPU1)

249/1610 (38/89/122) vs 252 at seed 0 (−3). The REJECT stands.
