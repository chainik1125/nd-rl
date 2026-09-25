# Review exp22 (exp19 with tb_k 64 on uncracked schemas, gen k 12; seed 0)

## Recount (written before reading preregistration / numbers.md / log.md for this run)

### Recount of `/work/exec/runs/exp22_tbbig_s0` (tools/recount.py)

- status `ok`, seed 0, budget 600, gpu `0:NVIDIA GeForce RTX 3090`, started_utc 2026-09-23T03:38:27Z, rl_secs 554.5, smoke False
- frozen hashes: 18 checked, mismatched none, not in SHA256SUMS none, frozen files absent from summary none; source/rl.py sha == summary rl.py sha: True
- summary metric 290, by_len {'7': 107, '8': 86, '9': 97}, heldout_greedy 0.902
- score.json: solved 290, by_len {'7': 107, '8': 86, '9': 97}, k 32, seed 0, official True, smoke False; final.pt sha matches: True; score == summary metric: True; by_len sum 290
- metrics.jsonl: 29 records {'sample': 4, 'train': 23, 'summary': 1, 'eval': 1}; step monotone True; sample rounds [1, 2, 3, 4]; sum(samples) 225925; last targets_solved_cum 679
  - vs rl_stats: rounds 4 (=), total_samples 225925 (=), targets_solved_cum 679 (=), total_train_steps 1143 (max step 1144)
  - harness eval heldout_greedy 0.902 == summary: True
  - r1: n_prompts 1321 k 12 T 0.8 samples 68972 accept 0.0058 solved_round 91 cum 91 by_len {'7': 91, '8': 0, '9': 0} new 99
  - r2: n_prompts 1321 k 12 T 0.8 samples 57588 accept 0.0246 solved_round 203 cum 205 by_len {'7': 192, '8': 13, '9': 0} new 121
  - r3: n_prompts 1321 k 12 T 0.8 samples 57500 accept 0.0667 solved_round 484 cum 497 by_len {'7': 379, '8': 109, '9': 9} new 313
  - r4: n_prompts 1321 k 9 T 0.8 samples 41865 accept 0.1012 solved_round 603 cum 679 by_len {'7': 500, '8': 152, '9': 27} new 195
  - train records 23: loss first 0.0701 last 0.0464; n_records last 10912
- found_targets.jsonl: 728 rows, 728 raw-distinct, 728 distinct by my normaliser; 679 targets solved, by L_true {7: 500, 8: 152, 9: 27}; names not in target pools: 0; thm field matches pool: 728/728
  - re-verified with nd_verify against the pool prompt: 150/150 accepted
- other files: ['round_01.pt', 'round_02.pt', 'round_03.pt', 'round_04.pt']
textbook solved cum by round [2, 13, 76, 149] schemas [('contraposition', 104), ('export', 44), ('demorgan_nand_to_or', 1)]

Per-round secs / sample / ft / steps: 162/126/32/300, 134/98/34/300, 149/108/37/300, 108/74/32/243 (r4 shrunk); 4 rounds. Dev L9 97, the lowest of any textbook-family run.

### rl.py static checks (/work/exec/runs/exp22_tbbig_s0/source/rl.py, 364 lines, diff vs frozen seed: 141 changed lines)
```
-- forbidden paths / sealed / other workspaces:
   none
```

**Hard constraints:** knob changes on exp19. Pass. **Gate 1:** pass.

## Compare (after reading preregistration/exp22.md, numbers.md, log.md)

Gate 0: prereg commit d547e56 at 2026-09-23T03:32:21Z; started_utc 03:38:27Z. **Pass.**

| claim (source) | reviewer's independent value | verdict |
|---|---|---|
| dev 290, 107/86/97; heldout 0.902 | 290, 107/86/97; 0.902 | reproduces |
| textbook 149 (contraposition 104, export 44, demorgan_nand_to_or 1); 4 rounds | 2/13/76/149; same schema counts; 4 rounds | reproduces |
| "a 3rd schema appeared with 1 instance … did not crack anything new in a way that spreads"; "counted as falsified" | 1 instance of demorgan_nand_to_or. By the literal criterion ("only the usual 2 schemas crack") it is not falsified. The executor's reading (one non-spreading instance, as in exp18/19/21) is the substantive one and is disclosed | reasonable; say "falsified in substance, not by the letter" |
| "fewer gen samples and rounds cost L9" | L9 97 vs 122–130 for exp19/exp21. Gen samples 225,925 total, most of them textbook; 4 rounds vs 5 | plausible single-run explanation (n = 1) |

## Verdict

**REJECT** (290 < 324). Doubling uncracked-schema samples did not crack a schema, and the trade cost the generated-target frontier.
