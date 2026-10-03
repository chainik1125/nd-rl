# eval_scale_pretrain: results

Pre-registered in `proposal.md` (commit 2c5d69c). Scorer `code/textbook_problem/eval_ckpt.py`.
Every number here is from `run_<ckpt>/metrics.jsonl`; every judged sample is in
`run_<ckpt>/<set>.jsonl` (the textbook 72 dump is in the gitignored
`docs/textbook-problem/textbook72/`). One seed per checkpoint: signs of life, no comparisons.

Protocol: k = 32 samples per problem, temperature 1.0, max_new 400 lean_seq tokens, seed 0, batch
256, CPU (M1 Pro). Judge: the fork's `lean_tok.inverse` decodes each sample to an ND body, then the
pinned nd_verify through strict_verify. "Parse-fail" = the sample is not in the lean_seq grammar
(the fork's inverse raises), counted as a failure. Avg solve rate = mean over problems of the
verified fraction of 32 samples; the interval is 1.96 × sd / √n over problems.

## scale-d384-p1200_s0 / pretrain.pt (9.56M params, lean_seq, dev 702/1500 in Robbie's harness)

Run 2026-10-02 15:15–15:26 UTC, `run_d384_s0/`.

| set | n | avg solve rate | ±95% | any-of-32 | never solved | parse-fail | secs |
|---|---|---|---|---|---|---|---|
| v0 (forall x, Carnap) | 104 | 0.111 | 0.052 | 22 (21%) | 82 | 0.29 | 115 |
| v1 (Roy, Carr) | 80 | 0.050 | 0.040 | 11 (14%) | 69 | 0.28 | 82 |
| hand14 (unsolved set) | 14 | 0.000 | 0.000 | 0 (0%) | 14 | 0.52 | 31 |
| batch3 (Pelletier, Oxford, R³) | 21 | 0.000 | 0.000 | 0 (0%) | 21 | 0.35 | 34 |
| lean_v2 (other session's Lean-sourced set, 47 + 3) | 50 | 0.088 | 0.067 | 11 (22%) | 39 | 0.33 | 74 |
| derived (18 feedable nuances) | 202 | 0.038 | 0.022 | 20 (10%) | 182 | 0.44 | 258 |
| textbook72 (sealed; aggregate only) | 72 | 0.104 | 0.064 | 12 (17%) | 60 | 0.32 | 73 |

Derived set by nuance (n = 10 each, hint_pairs 30, distractor_scaling 12):

| nuance | avg solve rate | any-of-32 |
|---|---|---|
| completion_closing | 0.203 | 4 |
| permutation | 0.134 | 2 |
| forward_control | 0.116 | 3 |
| extend | 0.087 | 2 |
| curry | 0.078 | 1 |
| no_dn | 0.066 | 1 |
| completion_contradiction | 0.025 | 1 |
| hint_pairs (right 0.025 / wrong 0.000 / none 0.016) | 0.013 | 2 |
| demorgan, line_budget, near_miss, substitution | 0.003 each | 1 each |
| conjoined, distractor_scaling, force_ore, negated_conjunction, wrappers, wrong_turn | 0.000 | 0 |

### Against the pre-registered bands

| set | expected avg | got | expected any-of-32 | got |
|---|---|---|---|---|
| v0 | 0.05–0.15 | 0.111 in band | 30–50% | 21% below |
| v1 | 0.02–0.08 | 0.050 in band | 15–35% | 14% just below |
| hand14 | ≈ 0 | 0.000 | 0–2 | 0 |
| batch3 | 0.02–0.10 | 0.000 below | 20–40% | 0% below |
| lean_v2 | 0.05–0.15 | 0.088 in band | 30–50% | 22% below |
| textbook72 | 0.05–0.15 | 0.104 in band | 30–50% | 17% below |
| derived prefix nuances ≥ base + 0.10 | | only completion_closing (0.20) | | |
| forward_control highest derived family | | third (0.116), behind completion_closing and permutation | | |

Reading: average solve rates landed where predicted, but the any-of-32 counts are below the bands
on every set, which means the model's successes are concentrated. On v0 the 22 solved problems have
per-problem rates spread from 0.03 to 0.97 (median 0.6) and are shorter than the unsolved ones
(mean prompt 18.7 vs 28.4 tokens). The pre-registered conclusion for "within the bands" applies:
textbook forms are a different distribution from the fork's pools for this pretrain, and the set is a
hard eval with 82 of 104 v0 problems, 69 of 80 v1 problems and all of hand14 and batch3 never solved.

### Sanity checks and the stop rule

- Before the run (proposal): fork heldout (lengths 2–6) avg 0.977, any-of-8 32/32; rl_targets
  (7–14) at temp 0.8 avg 0.223, any-of-8 11/32; consistent with the run's own dev metric.
- The stop rule ("parse-fail > 0.5 on any set") was crossed by hand14 (0.52). Investigated while the
  run continued: parse failures are dominated by `unbound` (a cited name never introduced), the same
  class that dominates Robbie's own lean_gate logs (16% of his samples); only 58 of 448 hand14
  samples hit the 400-token cap. Model behaviour, not a scoring fault, so the run was allowed to
  finish. Reasons over all failures on v0: unbound 685, IMPE rule check 479, ANDE2 330, ANDI 232,
  IMPI 227, name 163, ANDE1 149, DN 109.
- 20 raw samples read from `run_d384_s0/v0.jsonl` (10 verified, 10 failed, random): the verified
  proofs are correct and 6–7 lines; the failures are genuine logical errors (a premise restated
  wrongly, ∧E on the wrong conjunct, a reductio closed on a non-contradiction).

### Caveats

- One seed, one checkpoint size per table: no comparison claims.
- Tokens are lean_seq, so max_new 400 is a different budget from 400 ND tokens; a lean_seq proof
  of the same ND proof is longer (box headers restate goals). Long problems are cap-limited.
- Prefix nuances seed the decoder with the lean_seq prefix, which includes the headers of open
  boxes (the box's conclusion), so the prefix tasks reveal slightly more than their ND form.
- Repair and the two clozes were not run (not encodable). Three constraint nuances score default
  habit, not instruction following.
- The derived set's bases include problems the other session's curry-aware audit later dropped;
  this run does not depend on that, but any release decision does.

## scale-d768-p1200_s0 / pretrain.pt (37.99M params)

Run 2026-10-02 15:27–15:50 UTC, `run_d768_s0/` (first launch failed at load because the run's source had not been downloaded; relaunched at 15:27 with the same command).

| set | n | avg solve rate | ±95% | any-of-32 | never solved | parse-fail | secs |
|---|---|---|---|---|---|---|---|
| v0 | 104 | 0.086 | 0.045 | 17 (16%) | 87 | 0.34 | 200 |
| v1 | 80 | 0.058 | 0.045 | 11 (14%) | 69 | 0.37 | 173 |
| hand14 | 14 | 0.000 | 0.000 | 0 (0%) | 14 | 0.69 | 59 |
| batch3 | 21 | 0.002 | 0.003 | 1 (5%) | 20 | 0.43 | 97 |
| lean_v2 | 50 | 0.091 | 0.064 | 9 (18%) | 41 | 0.44 | 110 |
| derived | 202 | 0.036 | 0.022 | 17 (8%) | 185 | 0.52 | 561 |
| textbook72 | 72 | 0.129 | 0.073 | 11 (15%) | 61 | 0.38 | 161 |

Derived set by nuance:

| nuance | avg solve rate | any-of-32 |
|---|---|---|
| extend | 0.163 | 2 |
| completion_closing | 0.100 | 1 |
| curry | 0.094 | 1 |
| permutation | 0.087 | 1 |
| forward_control | 0.084 | 2 |
| distractor_scaling | 0.078 | 1 |
| no_dn | 0.041 | 1 |
| completion_contradiction | 0.016 | 2 |
| line_budget | 0.016 | 2 |
| wrappers | 0.013 | 1 |
| demorgan | 0.006 | 1 |
| substitution | 0.006 | 2 |
| conjoined | 0.000 | 0 |
| force_ore | 0.000 | 0 |
| hint_pairs | 0.000 | 0 |
| near_miss | 0.000 | 0 |
| negated_conjunction | 0.000 | 0 |
| wrong_turn | 0.000 | 0 |

Hint pairs by variant: right 0.000 (0 solved), wrong 0.000 (0 solved), none 0.000 (0 solved).

v0 problems solved at least once: d384 22, d768 17, both 16, either 23.

Reading: the same picture as d384 on one seed each: average rates inside the pre-registered bands on
v0, v1, lean_v2 and textbook72, any-of-32 counts below them, hand14 at zero, batch3 at one solve.
The larger model is not better on these sets at this budget (v0 0.086 vs 0.111; textbook72 0.129
vs 0.104): with one seed per size these are within each other's intervals and no size claim is made.
Its parse-fail fractions are higher on every set (0.34–0.69 vs 0.28–0.52), the stop rule crossed on
hand14 and derived; same `unbound` class, same decision to let it finish. 20 raw samples read from
`run_d768_s0/v0.jsonl`: same character as d384 (short correct proofs; logical slips on the rest).

## Conclusion

For both newest pretrains, the textbook-problem sets are a hard, mostly unsolved eval: at k = 32
and temperature 1.0, four fifths of candidate_v0, seven eighths of candidate_v1 and all of the
hand-proved 14 are never solved, and the derived nuances sit at or near zero except
completion_closing, permutation and forward_control. The pre-registered "within the bands"
conclusion stands: these forms are a different distribution from the fork's pools, and this is the
baseline an RL checkpoint has to move. Next: the same scorer on the RL-stage checkpoints of these
runs (`final.pt`) to see whether RL moves any of it; three seeds before any comparison.
