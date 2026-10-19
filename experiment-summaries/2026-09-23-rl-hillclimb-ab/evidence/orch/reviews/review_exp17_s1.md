# Review exp17_s1 (exp13 recipe with the textbook stream from round 1; seed 1)

## Recount (written before reading preregistration / numbers.md / log.md for this run)

### Recount of `/work/exec/runs/exp17_tbr1_s1` (tools/recount.py)

- status `ok`, seed 1, budget 600, gpu `1:NVIDIA GeForce RTX 3090`, started_utc 2026-09-23T02:55:28Z, rl_secs 572.5, smoke False
- frozen hashes: 18 checked, mismatched none, not in SHA256SUMS none, frozen files absent from summary none; source/rl.py sha == summary rl.py sha: True
- summary metric 331, by_len {'7': 115, '8': 94, '9': 122}, heldout_greedy 0.891
- score.json: solved 331, by_len {'7': 115, '8': 94, '9': 122}, k 32, seed 0, official True, smoke False; final.pt sha matches: True; score == summary metric: True; by_len sum 331
- metrics.jsonl: 33 records {'sample': 5, 'train': 26, 'summary': 1, 'eval': 1}; step monotone True; sample rounds [1, 2, 3, 4, 5]; sum(samples) 203704; last targets_solved_cum 763
  - vs rl_stats: rounds 5 (=), total_samples 203704 (=), targets_solved_cum 763 (=), total_train_steps 1291 (max step 1292)
  - harness eval heldout_greedy 0.891 == summary: True
  - r1: n_prompts 1336 k 16 T 0.8 samples 47936 accept 0.0119 solved_round 97 cum 97 by_len {'7': 96, '8': 1, '9': 0} new 107
  - r2: n_prompts 1336 k 16 T 0.8 samples 47936 accept 0.0505 solved_round 268 cum 269 by_len {'7': 222, '8': 47, '9': 0} new 180
  - r3: n_prompts 1336 k 16 T 0.8 samples 47936 accept 0.1232 solved_round 503 cum 503 by_len {'7': 356, '8': 128, '9': 19} new 259
  - r4: n_prompts 1335 k 16 T 0.8 samples 47920 accept 0.1909 solved_round 655 cum 656 by_len {'7': 466, '8': 153, '9': 37} new 168
  - r5: n_prompts 1334 k 4 T 0.8 samples 11976 accept 0.2795 solved_round 760 cum 763 by_len {'7': 561, '8': 161, '9': 41} new 110
  - train records 26: loss first 0.0594 last 0.0439; n_records last 11296
- found_targets.jsonl: 824 rows, 824 raw-distinct, 824 distinct by my normaliser; 763 targets solved, by L_true {7: 561, 8: 161, 9: 41}; names not in target pools: 0; thm field matches pool: 824/824
  - re-verified with nd_verify against the pool prompt: 150/150 accepted
- other files: ['round_01.pt', 'round_02.pt', 'round_03.pt', 'round_04.pt', 'round_05.pt']

Per-round secs / sample / ft / steps: 130/89/37/300, 131/87/42/300, 136/89/44/300, 135/87/45/300, 38/23/14/91 (r5 shrunk hard, k 4). Textbook solved cum (own count) 0/1/26/100/181; schemas contraposition 96, export 84, **disjunctive_syllogism 1** (the first third schema in any run).

Paired with exp13_textbook_s1 (same seed 1, stream from round 2): 331 vs 269; textbook 181 vs 13.

### rl.py static checks (/work/exec/runs/exp17_tbr1_s1/source/rl.py, 353 lines, diff vs frozen seed: 130 changed lines)
```
-- forbidden paths / sealed / other workspaces:
   none
```

**Hard constraints:** exp13 code with tb_start_round 1. Pass. **Gate 1:** pass (GPU1 02:55–03:08).

Seed-1 rescore: queued second, after exp13 (top raw score 331).

## Compare (after reading preregistration/exp17.md, numbers.md, log.md)

Gate 0: prereg commit 8c03c5e at 2026-09-23T02:55:26Z; started_utc 02:55:28Z. **Pass** (2 s).

| claim (source) | reviewer's independent value | verdict |
|---|---|---|
| dev 331, 115/94/122; heldout 0.891 | 331, 115/94/122; 0.891 | reproduces |
| textbook solved cum 0/1/26/100/181 | 0/1/26/100/181 | reproduces |
| exp13 s1 at the same seed: 0/0/1/13 | 0/0/0/1/13 (5 rounds) | reproduces (the log drops round 1) |
| prediction (305; falsified if ≤ 297 or textbook < 30): HIT | 331, 181 | reproduces. First clean hit on both criteria |
| "seed-matched vs exp13 s1: +62" | 331 − 269 = 62. "Seed-matched" means the same training seed, not the same trajectory: the extra round-1 textbook sampling changes every later RNG draw. One pair | arithmetic reproduces; one pair, and ignition is itself stochastic, so this is suggestive, not established |

Not mentioned by the executor: a third schema (disjunctive_syllogism) cracked, but only 1 instance.

## Verdict

**REJECT** as incumbent for now (a single-seed difference too small to count: +7 over 324, heldout 0.891 vs 0.896). The recipe looks like the most reliable igniter so far. If exp17_s0 also lands ≥ 324 + 15, a two-seed mean above the incumbent would justify adopting the better of the two checkpoints. The seed-1 rescore is queued.

## Addendum (03:22 UTC)

The seed-0 replicate scored 236 with no ignition (review_exp17_s0.md). Two-seed mean 283.5, below the incumbent. REJECT stands. "Starting at round 1 makes ignition earlier and less seed-dependent" is **not supported**.

## Seed-1 rescore (04:16–04:18 UTC, GPU0)

327/1610 (111/93/123) vs 331 at seed 0 (−4). The REJECT stands.
