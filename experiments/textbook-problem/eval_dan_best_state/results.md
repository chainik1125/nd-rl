# eval_dan_best_state: results

Pre-registered in `proposal.md` (commit 6898cdc). Dan's `state_eval.py` (branch `dan`, 0c492902)
samples in his step environment with Lean 4.34.1 as the in-loop judge; `code/textbook_problem/dan_metrics.py`
re-judges every accepted proof with the pinned nd_verify through strict_verify and computes the
plan's metrics from Dan's per-sample counts. Every row is in `run_best12_r8/s<seed>/<set>.jsonl`
(the 72 in the gitignored `docs/textbook-problem/textbook72/dan_eval/`). Checkpoint hashes in
`run_best12_r8/ckpt_sha256.txt`. Leak audit in `run_best12_r8/leak_audit.md`.

Protocol: k = 32, temperature 1.0, sampling seed 0, batch 256, 48 environment steps, 256 tokens per
action. Avg solve rate = mean over problems of `n_ok / n_tried` as Lean counted; "nd≠Lean" = distinct
Lean-accepted proofs that the pinned verifier rejects (Dan's known loose ⊥E rendering class), with the
rows they belong to. Those rows are not removed; the count is the caveat.

## best-state cap-12 T1 r8, seed 0 (`la_T1_best12_s0_r8.pt`, 9.56M params, lean_staten)

Run 2026-10-02 16:01–16:58 UTC on CPU.

| set | n | avg solve rate | ±95% | any-of-32 | never solved | nd≠Lean proofs (rows) |
|---|---|---|---|---|---|---|
| v0 (forall x, Carnap) | 104 | 0.341 | 0.071 | 76 (73%) | 28 | 0 |
| v1 (Roy, Carr) | 80 | 0.356 | 0.082 | 49 (61%) | 31 | 0 |
| hand14 (unsolved set) | 14 | 0.000 | 0.000 | 0 | 14 | 0 |
| batch3 (Pelletier, Oxford, R³) | 21 | 0.298 | 0.158 | 10 (48%) | 11 | 0 |
| lean_v2 (other session's Lean-sourced, 47 + 3) | 50 | 0.401 | 0.106 | 33 (66%) | 17 | 1 (1) |
| derived_plain (13 non-prefix nuances) | 132 | 0.264 | 0.059 | 69 (52%) | 63 | 0 |
| textbook72 (sealed; aggregate only) | 72 | 0.371 | 0.088 | 49 (68%) | 23 | 1 (1) |

Derived plain set by nuance (n = 10, distractor_scaling 12):

| nuance | avg solve rate | any-of-32 |
|---|---|---|
| substitution | 0.456 | 7 |
| line_budget | 0.434 | 6 |
| curry | 0.419 | 9 |
| no_dn | 0.397 | 8 |
| force_ore | 0.322 | 6 |
| permutation | 0.319 | 5 |
| wrappers | 0.309 | 6 |
| forward_control | 0.291 | 6 |
| near_miss | 0.159 | 4 |
| demorgan | 0.156 | 4 |
| distractor_scaling | 0.102 | 3 |
| conjoined | 0.075 | 3 |
| negated_conjunction | 0.025 | 2 |

### Against the pre-registered bands (seed 0)

| set | expected avg | got | expected any-of-32 | got |
|---|---|---|---|---|
| v0 | 0.30–0.50 | 0.341 in band | 50–70 % | 73 % above |
| v1 | 0.15–0.35 | 0.356 just above | 35–55 % | 61 % above |
| hand14 | 0.00–0.05 | 0.000 | 0–3 | 0 |
| batch3 | 0.05–0.20 | 0.298 above | 20–45 % | 48 % just above |
| lean_v2 | 0.25–0.45 | 0.401 in band | 45–65 % | 66 % at the edge |
| derived_plain | 0.15–0.35 | 0.264 in band | 35–55 % | 52 % in band |
| textbook72 | 0.25–0.45 | 0.371 in band | 30–45 of 72 | 49 above |

"Above the bands" fired on v0, v1, batch3 and textbook72, so the proposal's leak rule was applied
before anything else (`leak_audit.md`): the Stage-1 pool (K12, md5 verified against Dan's record),
Dan's rl_targets, rl_targets_v1, transfer, transfer_v1 and heldout were audited by renaming class
against every set; at most 3 problems in any set (2 of 104 on v0, 1 of 72) sit in a class those
pools contain. The results stand as measured. Average solve rates landed in their bands everywhere
except batch3; what the bands under-predicted is the breadth: more problems are solved at least once,
at lower per-problem rates (v0 median per-problem rate 0.47 among solved; 28 solved problems below
0.25, 24 at 0.75 or above).

### What the samples look like (20 read from `s0/v0.jsonl`, 10 solved and 10 unsolved, random)

Accepted proofs are correct but padded: mean written length 13.4 lines against reference minima of
5–10, with unused ∨I lines such as `( P v ( ( Q & Q ) v ( R & S ) ) ) : ORI1` inserted mid-proof, a
habit the ladder's reward (any accepted proof) does not penalise. Failures split between Lean rejecting
a parsed proof (832 of the v0 samples) and the environment rejecting the action syntax (unbound names
396, binder mismatch 344, Or.elim on a non-disjunction 264, premise out of position 96). Unsolved v0
problems are the longer ones (reference lengths 9–29 in the 10 read).

### Caveats

- Seed 0 only in this section; seeds 1 and 2 follow below when done. Still no comparison claim
  against Robbie's pretrains (one seed each, different protocol and token budget).
- Judge of record is Lean in the loop; the pinned verifier disagrees on 2 accepted proofs (one lean_v2
  row, one textbook72 row). The derived prefix nuances (8) and the three marker-word nuances were not
  run; the derived numbers here are the 13 plain-sequent nuances.
- The 13-line mean on accepted proofs means a 400-ND-token cap would not bind here, but it also means
  line_budget's 0.434 is from the 6 problems whose distinct accepted proofs met the budget.

## Seeds 0 and 1 (seed 2 cancelled)

Seed 1 (`la_T1_best12_s1_r8.pt`) ran 2026-10-02 16:58–18:02 UTC. Seed 2 was started and stopped at
Charles's request at 18:10 UTC before it had written any set, then cancelled outright (`status.md`);
the experiment ends with two seeds. Values are the mean over seeds with the min–max in brackets; avg solve rate is per the pinned
re-judging in `dan_metrics.py`.

| set | n | avg solve rate (s0–s1) | any-of-32 (s0–s1) |
|---|---|---|---|
| v0 | 104 | 0.343 (0.341–0.344) | 74.5 (73–76) |
| v1 | 80 | 0.359 (0.356–0.362) | 49.5 (49–50) |
| hand14 | 14 | 0.000 | 0 |
| batch3 | 21 | 0.312 (0.298–0.327) | 11.0 (10–12) |
| lean_v2 | 50 | 0.392 (0.383–0.401) | 32.0 (31–33) |
| derived_plain | 132 | 0.276 (0.264–0.287) | 67.5 (66–69) |
| textbook72 | 72 | 0.375 (0.371–0.378) | 48.0 (47–49) |

Seed-to-seed spread is within ±3 problems on every set and within 0.03 in avg solve rate, inside the
pre-registered ±5 expectation for v0. Two seeds support "signs of life" only; a comparison claim would need a third, which was not run.
