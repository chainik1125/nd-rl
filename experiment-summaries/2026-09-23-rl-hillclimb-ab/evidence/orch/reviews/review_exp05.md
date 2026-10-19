# Review exp05 (sample_batch 1024, rounds ≤ 8; seed 0)

## Recount (written before reading preregistration / numbers.md / log.md for this run)

### Recount of `/work/exec/runs/exp05_batch1024_s0` (tools/recount.py)

- status `ok`, seed 0, budget 600, gpu `0:NVIDIA GeForce RTX 3090`, started_utc 2026-09-23T01:40:34Z, rl_secs 540.6, smoke False
- frozen hashes: 18 checked, mismatched none, not in SHA256SUMS none, frozen files absent from summary none; source/rl.py sha == summary rl.py sha: True
- summary metric 242, by_len {'7': 41, '8': 89, '9': 112}, heldout_greedy 0.915
- score.json: solved 242, by_len {'7': 41, '8': 89, '9': 112}, k 32, seed 0, official True, smoke False; final.pt sha matches: True; score == summary metric: True; by_len sum 242
- metrics.jsonl: 41 records {'sample': 3, 'train': 36, 'summary': 1, 'eval': 1}; step monotone True; sample rounds [1, 2, 3]; sum(samples) 144000; last targets_solved_cum 536
  - vs rl_stats: rounds 3 (=), total_samples 144000 (=), targets_solved_cum 536 (=), total_train_steps 1800 (max step 1801)
  - harness eval heldout_greedy 0.915 == summary: True
  - r1: n_prompts 1500 k 32 T 0.8 samples 48000 accept 0.0229 solved_round 124 cum 124 by_len {'7': 121, '8': 3, '9': 0} new 144
  - r2: n_prompts 1500 k 32 T 0.8 samples 48000 accept 0.1095 solved_round 298 cum 298 by_len {'7': 266, '8': 32, '9': 0} new 216
  - r3: n_prompts 1500 k 32 T 0.8 samples 48000 accept 0.2574 solved_round 536 cum 536 by_len {'7': 371, '8': 143, '9': 22} new 291
  - train records 36: loss first 0.0644 last 0.0520; n_records last 22580
- found_targets.jsonl: 651 rows, 651 raw-distinct, 651 distinct by my normaliser; 536 targets solved, by L_true {7: 371, 8: 143, 9: 22}; names not in target pools: 0; thm field matches pool: 651/651
  - re-verified with nd_verify against the pool prompt: 150/150 accepted
- other files: ['round_01.pt', 'round_02.pt', 'round_03.pt']

Per-round secs / sample / ft: r1 176/109/63, r2 174/103/68, r3 189/113/72. Sampling 440/466/425 samples/s (exp01 at batch 384: 441 in r1). rl_secs 540.6, 3 rounds.

### rl.py static checks (/work/exec/runs/exp05_batch1024_s0/source/rl.py, 255 lines, diff vs frozen seed: 4 changed lines)
```
-- forbidden paths / sealed / other workspaces:
   none
```

**Hard constraints:** knob changes only (rounds 8, sample_batch 1024). Pass. **Gate 1:** pass. GPU0, 01:40–01:52; no reviewer rescore overlapped.

Seed-1 rescore: not run. 242 is below the incumbent's 249.

## Compare (after reading preregistration/exp05.md, numbers.md, log.md)

Gate 0: prereg commit 594f10b at 2026-09-23T01:31:26Z; started_utc 01:40:34Z. **Pass.**

| claim (source) | reviewer's independent value | verdict |
|---|---|---|
| dev 242, 41/89/112; heldout 0.915 | 242, 41/89/112; 0.915 | reproduces |
| throughput 440/466/425 at batch 1024 vs 441 at 384 | 440/466/425; exp01 r1 441 | reproduces |
| prediction (≥ 700/s, 4 rounds, 280; falsified < 550/s or ≤ 250): MISSED | 3 rounds, ≤ 466/s, 242 | reproduces; honest miss |
| "sampling is compute-bound at the 150 W cap, not launch-bound" | the data show only that batch size does not raise throughput. The power cap as the cause is not measured here | plausible interpretation; reword as a hypothesis |
| "in effect a seed-0 control replicate" | identical pipeline apart from batch size (batch changes the per-call RNG streams). Same 3 rounds and 144k samples as exp01, ft 600 × 3 | fair |
| control-like runs 249 / 242 / 219 / 238 | reproduce 249, 242, 219 (238 is the brief's reference, not in this arm) | reproduces. Mean 237, SD ≈ 13, range 30 over n = 4 (one of them the exp02 variant) |

Minor: the executor smoke-ran variants/exp07.py on GPU0 at 01:43 while exp05 was sampling there (4 s, --no-score). Negligible, but it is the same kind of GPU sharing that confounded exp03.

## Verdict

**REJECT.** Sound, not better (242 vs 249, within noise). A useful negative: the sampling batch size does not buy rounds. Together with the others, a near-control spread of ~30 theorems is now the working noise estimate.
