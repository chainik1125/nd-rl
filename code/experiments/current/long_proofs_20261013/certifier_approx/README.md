---
written_on: 2026-10-13
---

# Approximate certifier for "minimal proof needs >= `floor` body lines"

Deliverable: `estimate(prompt, n_prem, floor, secs)` in `estimate.py`, and a CLI
`python estimate.py IN.jsonl OUT.jsonl --floor 20 --secs 30 --procs 2` (input needs `prompt`, `n_prem`;
`certify.py` records work as is). Python: `~/repos/nd-takehome/.venv/bin/python`.

Returns `{'claim_ge': floor or None, 'upper': body length of a verified proof or None, 'p_false': ..., 'secs': ...,
'exact_exh': E, 'covered_ge': ..., 'reached': {...}, 'stages': [...], 'proof': ...}`. Every returned proof is
re-verified with nd_verify (atoms P Q R S T U W X Y Z) before it is reported.

## How it works

The minimal body length m is bracketed from both sides:

1. **Exact half (lower bound).** Dan's minlen DP over S (subformulas, negations, double negations) with the
   semantic pruning of `approach_d_theorem_first/minlen_sem.py`, iterative deepening, truncated by a memo cap
   (1.5M entries, ~300 MB) and a time share. Exhausting level E means m > E (within S, same caveat as minlen).
   If it reaches floor-1 the result is a minlen-equivalent certificate (`exact_cert`).
2. **Hunters (upper bound).** The same DP in restricted spaces, deepened from E+1 to floor-1, cheapest first.
   A find is a sound refutation (the proof verifies), and the only question is how often a hunter *misses*
   a proof under the floor that exists:
   - `none_d2`: no lemma rule, boxes at most 2 deep. Finds a sub-floor proof in ~0.5 s when one exists but it is
     often non-minimal (excess >= 1 line in 78% of finds, excess >= 3 in 11%).
   - `none`: no lemma rule (tree-shaped proofs, no sharing of derived lines). Memo ~5-10x smaller than the full
     search per level. Never missed in calibration (0 of 529); excess over the exact minimum <= 3, >= 1 in 6.4%.
   - `lit`: lemmas restricted to literals. Excess >= 1 in 1.3%.
   - `sub`: lemmas restricted to subformulas and their single negations (drops only `~~X` lemmas, which a
     reductio template replaces at equal cost). Found the exact minimum in every calibrated run that reached
     the level (152/152 at m >= 12); its only misses were time-outs.
3. **Claim and p_false.** `claim_ge = floor` iff nothing found a proof < floor AND every level in (E, floor) was
   reached (exhausted) by at least one *reliable* hunter (`none`/`lit`/`sub`). For each such level m, the miss
   bound is the smallest calibrated miss rate among the hunters that reached m, keyed by slack g = floor - m
   (how many lines a non-minimal proof may waste and still refute). `p_false` = max over m. Levels reached by no
   reliable hunter are uncovered: no claim at `floor`; `covered_ge` says which floor the run does support.
   The miss tables are Wilson 95% upper bounds from `calib.json` (built by `analyze.py --write-calib`).

Stage order, time shares and memo caps (`STAGES` in `estimate.py`): none_d2 5% / 1M, none 35% / 10M, exact 25% / 3M,
lit 15% / 8M, sub 20% / 6M (entries cost ~120 B; stages run one after another, so peak RSS ~1.2 GB).
Dropped after measuring: randomised capped full searches (`rand300k_s1/s2`, shuffled candidate order with a
300k memo cap) never found anything the truncated exact half did not; with a memoised DP the order barely
matters, the cap bites before the interesting level.

## Ground truth

533 distinct theorems (by `certify.theorem_key`) from `approach_d_theorem_first/{r2,r1,t1}_cert20,
t1_rest_cert16, hunt_cert18.jsonl`; 510 had exact minima (2..17). `exact_label.py` turned 20 of the 23 minlen
time-outs into exact labels (14..18) by exhausting levels 1..L-1 with the semantic-pruned full search, where L
is the hunter's find (up to 121 s and 12M memo entries each; `logs/exact_labels.jsonl`). Total: 529 exact labels,
histogram at the top: 13: 63, 14: 28, 15: 11, 16: 14, 17: 6, 18: 1. |S| ranges 13-43, 1-6 premises; the
distribution is whatever Robbie's backward generator and the small-S hunt produce, not a uniform sample of
theorems.

## Results (all measured; logs in `logs/`)

### Hunter miss rates vs exact labels (`calibrate.py`, 15 s per hunter, levels 1..19; `analyze.py`)

Per true minimum m (miss = found nothing <= 19; "long" = found a non-minimal proof):

```
  m    n     none_d2           none              lit              sub      ANY-hunter miss
 13   63   1/63 long 48    0/63 long 2      0/63 long 0      0/63 long 0        0/63
 14   28   3/28 long 21    0/28 long 4      0/28 long 0      0/28 long 0        0/28
 15   11   4/11 long 3     0/11 long 0      0/11 long 0      0/11 long 0        0/11
 16   14  12/14 long 1     0/14 long 1      0/14 long 1      2/14 long 0        0/14   (sub: 2 time-outs)
 17    6   4/6  long 2     0/6  long 1      0/6  long 1      2/6  long 0        0/6    (sub: 2 time-outs)
 18    1   1/1  long 0     0/1  long 0      0/1  long 0      1/1  long 0        0/1    (sub: time-out)
```
(m <= 12: 0 misses for every hunter, n = 416.)

Per slack g = floor - m, pooled over floors 11..19, runs that reached floor-1 only (misses/n, Wilson-95% ub):

```
            g=1              g=2             g=3             g=4             g=5          g=6..8
none_d2  152/285=0.590   73/369=0.242   43/392=0.145   18/435=0.064   11/471=0.041   <=5/464 ~0.025
none      25/285=0.126    4/369=0.028    3/392=0.022    3/435=0.020    2/471=0.015    0/4xx ~0.009
lit        2/285=0.025    0/369=0.010    0/392=0.010    0/435=0.009    0/471=0.008    0/4xx ~0.009
sub        0/280=0.014    0/365=0.010    0/390=0.010    0/435=0.009    0/471=0.008    0/4xx ~0.009
```
Excess (found - exact min) at m >= 12: none {0: 147, 1: 7, 2: 2, 3: 1}; lit {0: 155, 1: 2}; sub {0: 152}.

### Per-floor confusion of the simulated pipeline (exact half capped at 1M memo, then the four hunters)

Truth: long iff exact m >= floor (or cert_ge >= floor). 529 records.

```
floor   TP   FP   FN   TN   false-cert rate   recall   of which exact certificates
  10   285    0    0  244        0.000         1.000        285
  12   157    0    0  372        0.000         1.000        157
  13   123    0    0  406        0.000         1.000        122
  14    60    0    0  469        0.000         1.000         54
  15    32    0    0  497        0.000         1.000         19
  16    21    0    0  508        0.000         1.000          3
  17     7    0    0  522        0.000         1.000          1
  18     1    0    0  528        0.000         1.000          0
```
From floor 15 up most claims rest on the hunters, not on an exact certificate, and none of them is false.

### The real CLI on the 533 ground-truth records

`estimate.py --floor 20 --secs 30` (`logs/est_gt_f20.jsonl`): 0 claims, 0 false certificates on the 529 labelled
records (all truly < 20), 530/533 refuted with verified proofs (500 by `none_d2`, 30 by `none`), upper - exact
histogram {0: 388, 1: 92, 2: 29, 3: 14, 4: 3, 5: 3}. Mean 0.31 s per theorem (11,500 theorems per core-hour on
this mix); the 3 unrefuted ones took the full 30 s and were covered only to level 17-18 (`covered_ge` 18, 19, 19).

`estimate.py --floor 16 --secs 30` (`logs/est_gt_f16.jsonl`), a floor where both classes exist: TP 21, FP 0, FN 0,
TN 508 (false-certificate rate 0.000, recall 1.000). p_false attached to the 21 claims: 0.0 x3 (exact certificates),
0.0135 x15 (`sub` reached 15), 0.0252 x2 (`lit`), 0.1263 x1 (only `none`). 508 verified proofs returned (474 by
`none_d2`, 32 by `none`, 2 by the exact half). Mean 0.68 s per theorem (5,300 per core-hour); the 25 unrefuted
ones averaged 13.4 s.

The 3 theorems nothing refuted (|S| 36-37, 5 premises each, minlen itself timed out on them at floor 20):

- `--floor 22 --secs 150` with the original caps (`logs/est_unknown3_f22_s150.jsonl`): every stage stopped on its
  memo cap, not the clock (`none` 3M entries at level 17-19), so no claim; `covered_ge` 18, 19, 20.
- Reach probe with a 20M-entry cap (`reach_probe.py`, `logs/reach_none.jsonl`, `logs/reach_lit.jsonl`): memo and
  time grow ~2x per level for both hunters (`none`: level 18 3.8M / 7.6 s, 19 7.3M / 14.8 s, 20 14M / 31 s; `lit`
  about 2x those); 20M entries = 2.4 GB RSS (`none`) / 2.6 GB (`lit`). Outcomes: theorem 2, `none` and `lit` both
  found a verified **19**-line proof (so it is < 20, and the earlier runs had simply not reached level 19); theorem 3,
  both found a verified **20**-line proof after exhausting 19; theorem 1, `none` exhausted level 20, `lit` 18.
- `--floor 20 --secs 120` with the raised caps (`logs/est_cand_f20_s120.jsonl`, `logs/est_cand1_f20_s120.jsonl`):
  theorems 3 and 1 are **claimed >= 20** with `p_false` 0.126 (only `none` reached 19 within the budget; `lit`
  reached 16-17, `sub` 15-17; exact half 13-14). With the 20M-cap probe `lit` also exhausted 19 on theorem 3,
  which would price that claim at 0.025. Theorem 2 is refuted (`upper` 19) in 17 s.

## Can floor 20-30 claims be trusted?

What the evidence supports:

- The failure mode of a restricted hunter is specific: it misses only when every proof under the floor needs
  sharing (a lemma) the restriction forbids. The slack tables measure exactly that, and they are flat or
  improving in m (no miss of `none` at any m from 2 to 18, 529 records; `sub` finds the exact minimum
  whenever it reaches the level). Slack is the right key because a floor-20 claim with true m = 17 only needs the
  hunter to find *some* proof <= 19, which is what the g=3 column measures (`none`: 3/392).
- So the per-claim `p_false` at floor 20 is honest *conditional on the hunter reaching floor-1*: 0.126 when only
  `none` reached 19 (its g=1 bound; point estimate 0.088), 0.028 when it reached 19 and the true m could be 18,
  0.025 if `lit` reached 19, 0.014 if `sub` did. It is a bound on the conditional miss rate, not a posterior:
  with these generators the base rate of true m >= 20 is tiny (the exact-min histogram roughly halves per level
  above 13), so among claims the fraction that are false could be much higher than p_false. A posterior needs a
  prior over m for the generator in use; `p_false_for` is where to plug it in.

What it does not support:

- Reach, not miss rate, is the binding constraint. Memo per level grows ~2x for every hunter, so the budget that
  gets `none` to level 19 in 30 s on |S| ~ 36 gets it to ~22-23 with 4 GB. A floor-25 or floor-30 claim with this
  code is only supported when `reached['none'] >= floor-1`, which for |S| >= 30 will not happen within memory;
  the CLI then reports `covered_ge` instead of a claim. Nothing here is a certifier at floor 30.
- Calibration covers m <= 18 and this generator family. Theorems whose short proofs rely heavily on shared
  lemmas (e.g. a derived formula used in many ORE branches) are exactly where `none` is weakest; `lit`/`sub`
  cover those but reach 3-5 levels less in the same time.
- Inherited from minlen: everything is relative to S. A proof using formulas outside S is invisible to the exact
  half and to every hunter.

Practical recommendation for the 10,000-theorem goal: use this as the *screen* (it refutes 99.4% of candidates
in under a second with a verified short proof), spend the budget on `none` and `lit` for the survivors, and
treat `claim_ge = 20` with `p_false <= 0.03` (i.e. `lit` or `sub` reached 19, or `none` reached 19 and the
exact half reached 18) as the acceptance bar.

## Files

- `hunter.py`: `Hunter` (minlen_sem subclass; `lem`, `max_depth`, `memo_cap`, `seed`), `levels()`, `hunt()`.
- `estimate.py`: `estimate()`, `p_false_for()`, CLI.
- `calibrate.py`: run variants over ground truth -> `logs/cal_cheap.jsonl`, `logs/cal_full.jsonl`.
- `analyze.py`: miss tables, simulated per-floor confusion, writes `calib.json`.
- `exact_label.py`: exact labels for minlen time-outs from hunter upper bounds -> `logs/exact_labels.jsonl`.
- `score_est.py`: confusion/throughput for an `estimate.py` output.
- `gt_in.jsonl`, `extra_gt.jsonl`, `unknown3.jsonl`: inputs. `logs/`: everything measured above.
