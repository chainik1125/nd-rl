# Review exp17_s0 (exp17 recipe, seed 0; replicate of exp17_s1)

## Recount (written before reading preregistration / numbers.md / log.md for this run)

### Recount of `/work/exec/runs/exp17_tbr1_s0` (tools/recount.py)

- status `ok`, seed 0, budget 600, gpu `1:NVIDIA GeForce RTX 3090`, started_utc 2026-09-23T03:07:55Z, rl_secs 573.3, smoke False
- frozen hashes: 18 checked, mismatched none, not in SHA256SUMS none, frozen files absent from summary none; source/rl.py sha == summary rl.py sha: True
- summary metric 236, by_len {'7': 34, '8': 90, '9': 112}, heldout_greedy 0.886
- score.json: solved 236, by_len {'7': 34, '8': 90, '9': 112}, k 32, seed 0, official True, smoke False; final.pt sha matches: True; score == summary metric: True; by_len sum 236
- metrics.jsonl: 33 records {'sample': 5, 'train': 26, 'summary': 1, 'eval': 1}; step monotone True; sample rounds [1, 2, 3, 4, 5]; sum(samples) 199616; last targets_solved_cum 593
  - vs rl_stats: rounds 5 (=), total_samples 199616 (=), targets_solved_cum 593 (=), total_train_steps 1267 (max step 1268)
  - harness eval heldout_greedy 0.886 == summary: True
  - r1: n_prompts 1321 k 16 T 0.8 samples 47696 accept 0.0112 solved_round 97 cum 97 by_len {'7': 95, '8': 2, '9': 0} new 109
  - r2: n_prompts 1320 k 16 T 0.8 samples 47680 accept 0.0515 solved_round 270 cum 271 by_len {'7': 220, '8': 51, '9': 0} new 196
  - r3: n_prompts 1319 k 16 T 0.8 samples 47664 accept 0.1188 solved_round 491 cum 493 by_len {'7': 335, '8': 142, '9': 16} new 257
  - r4: n_prompts 1318 k 16 T 0.8 samples 47648 accept 0.1603 solved_round 578 cum 581 by_len {'7': 371, '8': 175, '9': 35} new 109
  - r5: n_prompts 1316 k 3 T 0.8 samples 8928 accept 0.1766 solved_round 579 cum 593 by_len {'7': 376, '8': 179, '9': 38} new 13
  - train records 26: loss first 0.0686 last 0.0491; n_records last 10720
- found_targets.jsonl: 684 rows, 684 raw-distinct, 684 distinct by my normaliser; 593 targets solved, by L_true {7: 376, 8: 179, 9: 38}; names not in target pools: 0; thm field matches pool: 684/684
  - re-verified with nd_verify against the pool prompt: 150/150 accepted
- other files: ['round_01.pt', 'round_02.pt', 'round_03.pt', 'round_04.pt', 'round_05.pt']
textbook solved cum by round [0, 0, 0, 0, 1] schemas [('contraposition', 1)]

source/rl.py byte-identical to exp17_tbr1_s1. Per-round secs / sample / ft / steps: 133/92/38/300, 131/86/43/300, 135/86/45/300, 143/94/46/300, 29/18/10/67 (r5 tiny, k 3). **Textbook ignition failed: 1 solve in 5 rounds** (exp17_s1: 181).

**Hard constraints:** identical code to exp17_s1. Pass. **Gate 1:** pass.

## Compare (after reading preregistration/exp17.md, numbers.md, log.md)

Gate 0: preregistration/exp17.md (commit 8c03c5e, 02:55:26Z) names "1 first, then 0 if ≥ 297". Seed 1 gave 331 ≥ 297, so this seed-0 run follows the stop rule. started_utc 03:07:55Z. **Pass.**

| claim (source) | reviewer's independent value | verdict |
|---|---|---|
| dev 236, 34/90/112; heldout 0.886 | 236, 34/90/112; 0.886 | reproduces |
| textbook solved cum 1 at the end (no ignition) | 0/0/0/0/1 (contraposition 1) | reproduces |
| "exp17's two seeds 331 / 236; seed-1 hit, seed 0 misses badly; value hinges on whether and when ignition happens" | 331 / 236 | reproduces; honest |
| exp20 withdrawn before launch, replaced by exp21 | no exp20 run dir; STATUS shows WITHDRAWN | fine. A withdrawn prereg with no run is not a dropped result. Note that the choice was informed by exp19's live round-2 log, which is legitimate (training-side information) |

Combined reading for the exp13-family (textbook stream at T 1.0), dev by run: 324 (exp13 s0, 145 solves), 269 (exp13 s1, 13), 303 (exp15, 150), 324 (exp16, 130), 331 (exp17 s1, 181), 236 (exp17 s0, 1), 300 (exp18, 53). **Dev tracks textbook solves** (≥ 130 solves → 303–331; ≤ 53 → 236–300). Whether ignition happens looks like a coin flip per run with these settings, about 4 of 7.

## Verdict

**REJECT** (236 < 324). This also settles exp17_s1's open question: starting the stream at round 1 does not make ignition reliable (1 of 2 seeds ignited).
