# Review exp01 (control: seed rl.py, seed 1)

## Recount (written before reading preregistration / numbers.md / log.md)

### Recount of `/work/exec/runs/exp01_control_s1` (tools/recount.py)

- status `ok`, seed 1, budget 600, gpu `0:NVIDIA GeForce RTX 3090`, started_utc 2026-09-23T01:17:52Z, rl_secs 540.0, smoke False
- frozen hashes: 18 checked, mismatched none, not in SHA256SUMS none, frozen files absent from summary none; source/rl.py sha == summary rl.py sha: True
- summary metric 249, by_len {'7': 49, '8': 91, '9': 109}, heldout_greedy 0.92
- score.json: solved 249, by_len {'7': 49, '8': 91, '9': 109}, k 32, seed 0, official True, smoke False; final.pt sha matches: True; score == summary metric: True; by_len sum 249
- metrics.jsonl: 41 records {'sample': 3, 'train': 36, 'summary': 1, 'eval': 1}; step monotone True; sample rounds [1, 2, 3]; sum(samples) 144000; last targets_solved_cum 537
  - vs rl_stats: rounds 3 (=), total_samples 144000 (=), targets_solved_cum 537 (=), total_train_steps 1800 (max step 1801)
  - harness eval heldout_greedy 0.92 == summary: True
  - r1: n_prompts 1500 k 32 T 0.8 samples 48000 accept 0.0246 solved_round 111 cum 111 by_len {'7': 110, '8': 1, '9': 0} new 124
  - r2: n_prompts 1500 k 32 T 0.8 samples 48000 accept 0.1179 solved_round 351 cum 351 by_len {'7': 279, '8': 72, '9': 0} new 268
  - r3: n_prompts 1500 k 32 T 0.8 samples 48000 accept 0.2649 solved_round 537 cum 537 by_len {'7': 363, '8': 146, '9': 28} new 224
  - train records 36: loss first 0.0695 last 0.0540; n_records last 22456
- found_targets.jsonl: 616 rows, 616 raw-distinct, 616 distinct by my normaliser; 537 targets solved, by L_true {7: 363, 8: 146, 9: 28}; names not in target pools: 0; thm field matches pool: 616/616
  - re-verified with nd_verify against the pool prompt: 150/150 accepted
- other files: ['round_01.pt', 'round_02.pt', 'round_03.pt']

### rl.py static checks (/work/exec/runs/exp01_control_s1/source/rl.py, 255 lines, diff vs frozen seed: 0 changed lines)
```
-- forbidden paths / sealed / other workspaces:
   none
-- file access (open / read_jsonl / paths[...] / os / subprocess / glob / torch.load):
64:def read_jsonl(fn, limit=None):
66:    with open(fn) as f:
193:    cfg = dict(CONFIG, **(SMOKE if paths.get('smoke') else {}))
197:    retained = read_jsonl(paths['train_cap6'], limit=cfg['retain'] * 4 if paths.get('smoke') else None)
230:        paths['save_checkpoint'](model, tok, f'{paths["out_dir"]}/round_{r:02d}.pt')
233:            hg = paths['heldout_greedy'](model, tok, n=cfg['eval_n'])
245:    with open(f'{paths["out_dir"]}/found_targets.jsonl', 'w') as f:
-- heldout usage:
16:  paths        dict: targets_extra, train_cap6, heldout_cap6, out_dir (files you may read / write under),
17:               save_checkpoint(model, tok, path), heldout_greedy(model, tok, n=1000) -> float, smoke (bool)
23:premises) and (c) records from train_cap6.jsonl. heldout_cap6.jsonl is for tracking only. No proofs
233:            hg = paths['heldout_greedy'](model, tok, n=cfg['eval_n'])
234:            metrics.log(phase='eval', step=state['step'], round=r, heldout_greedy=hg, secs=time.time() - te)
-- literal proof-like strings (': PR', 'QED', 'N1 '):
   none
-- lines longer than 300 chars:
```

**Hard constraints:** source/rl.py is byte-identical to frozen/rl.py (the seed). Reads only train_cap6 via paths; heldout used only via paths['heldout_greedy'] (disabled: eval_each_round False); trains on own verified samples + cap-6. Pass.

**Gate 1:** status ok, hashes match SHA256SUMS, score.json k=32 seed=0 official, final.pt sha matches score.json, metrics.jsonl sample rounds/samples/solved-cum equal rl_stats. Pass. (Max step 1801 = 1800 train steps + the harness eval record's internal counter.)

Observation: dev by_len L7 49/300 (16%) < L8 91/300 (30%); on the RL targets L7 is solved far more (363 vs 146). Not an error in counting; the sealed L7 slice may differ in composition. Open question.

Seed-1 rescore: pending (copy of final.pt scored from /work/review/rescore/ so the executor's score.json is not overwritten).

Seed-1 rescore (reviewer, GPU0, copy of final.pt, sha identical): **254/1610** (L7 53, L8 91, L9 110). Seed-0 official 249. Sampling-seed spread 5 theorems.

## Compare (after reading preregistration/exp01.md, numbers.md, log.md)

Gate 0: prereg commit f6f480d at 2026-09-23T01:17:50Z; run started_utc 01:17:52Z. Precedes (by 2 s). **Pass.**

| claim (source) | reviewer's independent value | verdict |
|---|---|---|
| dev pass@32 249, L7/L8/L9 49/91/109 (numbers.md) | 249, 49/91/109 (score.json, final.pt sha checked) | reproduces |
| heldout_greedy 0.920 (numbers.md) | 0.92 (harness eval record) | reproduces |
| 3 rounds; cum targets solved 111 → 351 → 537 (log.md) | 111 → 351 → 537; found_targets.jsonl gives 537 solved, 616 distinct proofs (own normaliser), 150/150 re-verify | reproduces |
| accepted-sample rate 2.5 % → 12 % → 26 % (log.md) | 2.46 % → 11.79 % → 26.49 % | reproduces |
| sample ~110 s + ft ~60 s per round (log.md) | sample 109/112/114 s; ft 57/66/69 s (600 steps each) | reproduces |
| failures are mid-proof rule-check errors, qed_rate ~1.0 (log.md) | qed_rate 1.000 all rounds; top fail reasons rule check ANDI / IMPI / IMPE | reproduces |
| expected 238 ± 30, falsified outside 190-290 (prereg) | 249 (s0), 254 (s1 rescore): inside the band | expectation met |
| "run-to-run noise" (prereg question) | one training seed only; this run vs the brief's seed-0 reference (238, not re-derivable by the reviewer: no run dir in this arm) is a +11 gap, under the 15 noise floor. Sampling-seed noise on a fixed checkpoint: 5 | partially answered; training-seed noise needs n ≥ 3 |

Wording: no over-claims found. The log's relabelling remark concerns exp02 and is checked in review_exp02.md.

## Verdict

**ADOPT** as the first incumbent. Gates 0–2 pass; the seed rl.py is unmodified; score 249 (seed-1 rescore 254) against the pretrain checkpoint's ≈ 7. This is the control. Later runs have to beat it by more than about 15 theorems (by more than 5 on sampling noise alone, and training-seed noise is still unmeasured) or repeat the gain across seeds.
