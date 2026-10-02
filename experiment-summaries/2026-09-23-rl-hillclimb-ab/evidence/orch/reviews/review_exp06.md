# Review exp06 (tail-fill: shrunk last round instead of stopping; skip saturated; rounds ≤ 8; seed 0)

## Recount (written before reading preregistration / numbers.md / log.md for this run)

### Recount of `/work/exec/runs/exp06_tailfill_s0` (tools/recount.py)

- status `ok`, seed 0, budget 600, gpu `1:NVIDIA GeForce RTX 3090`, started_utc 2026-09-23T01:41:59Z, rl_secs 566.5, smoke False
- frozen hashes: 18 checked, mismatched none, not in SHA256SUMS none, frozen files absent from summary none; source/rl.py sha == summary rl.py sha: True
- summary metric 234, by_len {'7': 40, '8': 90, '9': 104}, heldout_greedy 0.927
- score.json: solved 234, by_len {'7': 40, '8': 90, '9': 104}, k 32, seed 0, official True, smoke False; final.pt sha matches: True; score == summary metric: True; by_len sum 234
- metrics.jsonl: 38 records {'sample': 3, 'train': 33, 'summary': 1, 'eval': 1}; step monotone True; sample rounds [1, 2, 3]; sum(samples) 128956; last targets_solved_cum 516
  - vs rl_stats: rounds 3 (=), total_samples 128956 (=), targets_solved_cum 516 (=), total_train_steps 1628 (max step 1629)
  - harness eval heldout_greedy 0.927 == summary: True
  - r1: n_prompts 1500 k 32 T 0.8 samples 48000 accept 0.023 solved_round 114 cum 114 by_len {'7': 112, '8': 2, '9': 0} new 132
  - r2: n_prompts 1500 k 32 T 0.8 samples 48000 accept 0.1066 solved_round 301 cum 301 by_len {'7': 262, '8': 39, '9': 0} new 227
  - r3: n_prompts 1498 k 22 T 0.8 samples 32956 accept 0.2535 solved_round 514 cum 516 by_len {'7': 366, '8': 134, '9': 16} new 270
  - train records 33: loss first 0.0691 last 0.0554; n_records last 22488
- found_targets.jsonl: 629 rows, 629 raw-distinct, 629 distinct by my normaliser; 516 targets solved, by L_true {7: 366, 8: 134, 9: 16}; names not in target pools: 0; thm field matches pool: 629/629
  - re-verified with nd_verify against the pool prompt: 150/150 accepted
- other files: ['round_01.pt', 'round_02.pt', 'round_03.pt']

Per-round secs / sample / ft / ft_steps: r1 197/122/72/600, r2 213/128/81/600, r3 155/92/60/428 (shrunk: k 22, 1,498 prompts). Sampling 394/374/359 samples/s, vs exp01 441 and exp05 440–466. rl_secs 566.5.

Tail-fill fired on round 3, not as an extra round 4. Rounds 1–2 were slower than in exp01 (197/213 s vs 169/181 s), so a full round 3 no longer fit. The score-request log shows no reviewer rescore or other scoring on GPU1 during 01:42–01:51 (exp04's harness scoring was requested 01:38:07 and exp04 was DONE by 01:41). The lower throughput is unexplained from the files I can see.

### rl.py static checks (/work/exec/runs/exp06_tailfill_s0/source/rl.py, 265 lines, diff vs frozen seed: 28 changed lines)
```
-- forbidden paths / sealed / other workspaces:
   none
```

**Hard constraints:** changes to the round loop and skip-saturated only. Pass. **Gate 1:** pass.

Seed-1 rescore: not run. 234 is below the incumbent's 249.

## Compare (after reading preregistration/exp06.md, numbers.md, log.md)

Gate 0: prereg commit 5e83ca6 at 2026-09-23T01:41:58Z; started_utc 01:41:59Z. Precedes by 1 s. **Pass.** (The file's header says "~01:44 UTC", but the commit time is what counts.)

| claim (source) | reviewer's independent value | verdict |
|---|---|---|
| dev 234, 40/90/104; heldout 0.927 | 234, 40/90/104; 0.927 | reproduces |
| round 3 shrunk to 0.79 (k 22, 428 steps); no 4th round | r3 k 22, 1,498 prompts, 428 ft steps; 3 rounds | reproduces (22/32 ≈ 0.9 × 0.76; consistent) |
| GPU1 ran at ~375–390 samples/s; GPU1 ~10–15 % slower than GPU0 in exp02/04/06 | uncontended rounds: GPU0 422–466 (exp01, exp05, exp03 r3–4); GPU1 416–423 (exp02), 355–376 (exp04, a different prompt mix), 359–394 (exp06) | roughly supported: ~5 % (exp02) to ~13 % (exp06). The cause is not established |
| prediction (4 rounds, 270; falsified ≤ 245): MISSED | 3 rounds, 234 | reproduces; honest miss |
| prereg premise: "the round count is what drives the score (exp04: 2 → 193; exp01: 3 → 249)" | carried over from exp04's single confounded run (see review_exp04) | premise not established; do not treat it as known |
| "Net: no gain over the control" | 234 vs 249 / 242 / 219 near-controls | reproduces |

Note: 128,956 samples vs 144,000 in the control. Tail-fill in this run reduced total sampling rather than adding it, because it is triggered whenever round 3 does not fit fully. The rule only helps when the tail is large relative to a round.

## Verdict

**REJECT.** Sound, not better (234 vs 249; inside the near-control spread). Heldout 0.927 is the highest so far, but +0.007 over the incumbent is within noise, and the dev score is lower, so the "equal score + better heldout" clause does not apply.
