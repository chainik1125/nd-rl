---
written_on: 2026-10-06
written_by: agent:claude (Worker B)
---

# Worker B phase 4: headline figure, self-judging mechanism, ancestor judges

Branch `dmitry/selection-sprint` (uncommitted). The ancestor-judge analysis was pre-registered in `../logs/worker_B.md`
("Phase 4", 04:49 UTC) before the checkpoints were downloaded. Code: `code/phase4.py` and `code/phase4_figs.py`.

## Summary

- **Headline (`figures/B4_headline.png`).** Results are averaged over three generator seeds, at most 16 attempts.
  - **What the sibling judge buys.** On holdout250, count-stop with sibling judging gives 0.924 at about 3.9
    attempts. That compares with:
    - 0.914 for the model judging itself;
    - 0.892 for "stop at the first finished proof" (at most 1.6 attempts);
    - 0.883 for a fixed 4 attempts with a finished pick;
    - 0.927 for oracle pass@4.
  - **Same comparison on textbook72, at about 7.6 attempts.** Sibling judging gives 0.587 against 0.568 self-judged
    and 0.607 for oracle pass@8.
  - **What stays out of reach.** Oracle stop-at-first-valid (0.95 at 2 attempts on holdout250) shows that much of the
    remaining headroom is in *stopping*, not *picking*.
- **Mechanism.** The self penalty is mainly that **invalid proofs look too likely to their own generator**.
  - **Self vs sibling likelihood.** The generator gives all its own finished proofs higher masked mean log-likelihood
    than the siblings do. The excess is about 2–3 times larger for invalid proofs (+0.071 to +0.107 nats per token)
    than for valid ones (+0.033 to +0.040), in all 6 (set, generator) cells.
  - **Within a theorem.** After subtracting each theorem's mean, invalid proofs sit only −0.011 to −0.016 below the
    theorem mean under self, against −0.032 to −0.049 under siblings. Valid proofs sit at +0.001 to +0.004 under self,
    against +0.004 to +0.023 under siblings.
  - **Separation.** The valid−invalid gap within a theorem is 3–4 times smaller under self (0.012–0.018) than under
    siblings (0.040–0.063).
  - **Where it sits.** The excess is concentrated in **atomic `have` steps**: the self-minus-sibling step log p gap is
    +0.15 to +0.21 on invalid proofs against +0.05 to +0.07 on valid ones. Box and `Or.elim` openings show a smaller
    version (+0.04 to +0.10 against +0.02 to +0.07). `exact` and premise steps show essentially none.
  - **No consistent length pattern.** The AUROC gap (siblings − self) is positive in most length terciles but not
    systematically larger for long proofs.
- **Ancestor judges.** The pre-registered prediction is **not supported**.
  - The r8 ancestor of the same seed is as bad a judge as the r16 generator itself: within-theorem AUROC 0.793 vs
    0.788 on holdout250 (+0.005 [−0.022, 0.030]) and 0.679 vs 0.687 on textbook72 (−0.008 [−0.095, 0.073]).
  - It is significantly worse than the r16 siblings: −0.036 [−0.072, −0.002] and −0.103 [−0.177, −0.036].
  - Stage-1 (SFT only) is the weakest judge: 0.691 and 0.605.
  - Interpretation, untested: the blind spot is shared by the whole s0 lineage, from at least round 8. It is not
    created by the last eight rounds of expert iteration. Expert iteration up to round 8 does make a much better judge
    than stage-1.

## 1. Headline numbers behind the figure (`results/B4_headline_curves.json`)

Pools: holdout250 and textbook72 at k = 64 for generators s0, s1 and s2. Each policy uses the first 16 attempts of 50
seeded orderings. Cells give accuracy as the generator mean [min–max over the three generators] at the generator-mean
attempt count. Policy parameters (k, N_max, c) are fixed across generators, and we take the point nearest m.

| set | policy | m ≈ 2 | m ≈ 4 | m ≈ 8 |
|---|---|---|---|---|
| holdout250 | pick one proof at random (fixed k) | 0.829 [0.818-0.836] @ 2.00 | 0.829 [0.818-0.836] @ 4.00 | 0.829 [0.818-0.836] @ 8.00 |
| holdout250 | pick a proof the environment marked finished (fixed k) | 0.870 [0.856-0.877] @ 2.00 | 0.883 [0.864-0.893] @ 4.00 | 0.889 [0.870-0.900] @ 8.00 |
| holdout250 | stop at the first finished proof | 0.892 [0.872-0.904] @ 1.59 (closest reachable) | 0.892 [0.872-0.904] @ 1.59 (closest reachable) | 0.892 [0.872-0.904] @ 1.59 (closest reachable) |
| holdout250 | stop after c finished; the model keeps its own most-likely one | 0.892 [0.872-0.904] @ 1.59 (closest reachable) | 0.914 [0.901-0.921] @ 3.92 | 0.921 [0.911-0.928] @ 8.11 |
| holdout250 | stop after c finished; two sibling checkpoints judge | 0.892 [0.872-0.904] @ 1.59 (closest reachable) | 0.924 [0.912-0.932] @ 3.92 | 0.934 [0.926-0.942] @ 8.11 |
| holdout250 | oracle: any valid proof among k (pass@k) | 0.898 [0.889-0.904] @ 2.00 | 0.927 [0.919-0.934] @ 4.00 | 0.943 [0.935-0.955] @ 8.00 |
| holdout250 | oracle: stop at the first valid proof | 0.952 [0.941-0.966] @ 1.98 | 0.953 [0.941-0.968] @ 2.03 (closest reachable) | 0.953 [0.941-0.968] @ 2.03 (closest reachable) |
| textbook72 | pick one proof at random (fixed k) | 0.427 [0.407-0.439] @ 2.00 | 0.427 [0.407-0.439] @ 4.00 | 0.427 [0.407-0.439] @ 8.00 |
| textbook72 | pick a proof the environment marked finished (fixed k) | 0.471 [0.456-0.488] @ 2.00 | 0.501 [0.493-0.516] @ 4.00 | 0.517 [0.508-0.531] @ 8.00 |
| textbook72 | stop at the first finished proof | 0.501 [0.493-0.516] @ 2.04 | 0.527 [0.518-0.537] @ 3.96 | 0.527 [0.518-0.537] @ 3.96 (closest reachable) |
| textbook72 | stop after c finished; the model keeps its own most-likely one | 0.527 [0.518-0.537] @ 3.96 (closest reachable) | 0.527 [0.518-0.537] @ 3.96 | 0.568 [0.553-0.592] @ 7.60 |
| textbook72 | stop after c finished; two sibling checkpoints judge | 0.527 [0.518-0.537] @ 3.96 (closest reachable) | 0.527 [0.518-0.537] @ 3.96 | 0.587 [0.571-0.612] @ 7.60 |
| textbook72 | oracle: any valid proof among k (pass@k) | 0.498 [0.477-0.523] @ 2.00 | 0.558 [0.539-0.595] @ 4.00 | 0.607 [0.577-0.653] @ 8.00 |
| textbook72 | oracle: stop at the first valid proof | 0.534 [0.515-0.566] @ 2.07 | 0.599 [0.572-0.643] @ 3.82 | 0.652 [0.620-0.703] @ 7.18 |

"Closest reachable" means the policy cannot spend m attempts:

- first-finished is limited by its ceiling (at most 16 attempts);
- count-stop cannot go below its c = 1 point;
- the oracle stop is limited by its ceiling.

Count-stop with c = 1 *is* first-finished.

## 2. Mechanism (`results/B4_mechanism.json`, `results/B4_mechanism_headline.txt`)

Values are mean masked mean-log-likelihood per token (nats) over finished attempts. "Centred" means minus the mean
over that theorem's finished attempts, using only theorems with both classes. Each row is one (set, generator).

| set / generator | self: valid, invalid (centred) | siblings: valid, invalid (centred) | centred separation self / siblings | self − sibling shift: valid / invalid |
|---|---|---|---|---|
| holdout250 / s0 | +0.001, −0.011 | +0.004, −0.035 | 0.012 / 0.040 | +0.033 / +0.071 |
| holdout250 / s1 | +0.001, −0.013 | +0.007, −0.044 | 0.014 / 0.051 | +0.036 / +0.080 |
| holdout250 / s2 | +0.001, −0.013 | +0.007, −0.047 | 0.014 / 0.053 | +0.034 / +0.083 |
| textbook72 / s0 | +0.002, −0.012 | +0.014, −0.041 | 0.014 / 0.055 | +0.040 / +0.088 |
| textbook72 / s1 | +0.002, −0.016 | +0.016, −0.047 | 0.018 / 0.063 | +0.034 / +0.106 |
| textbook72 / s2 | +0.004, −0.013 | +0.022, −0.042 | 0.017 / 0.063 | +0.039 / +0.107 |

Reading this table:

- Under its own likelihood, an invalid finished proof is almost as typical as a valid one.
- Under a sibling's likelihood, invalid proofs are clearly less likely.
- Both the valid and the invalid columns move, but the invalid column moves about three times more. So the answer is
  "invalid proofs look too likely", not "valid proofs look too unlikely".

**Step types.** Self minus sibling mean log p per step, finished attempts, valid / invalid:

| set / generator | atomic `have` | box opening | `Or.elim` opening | `exact` | premise |
|---|---|---|---|---|---|
| holdout250 / s0 | +0.071 / +0.164 | +0.022 / +0.046 | +0.046 / +0.074 | −0.004 / +0.004 | 0.000 / 0.001 |
| holdout250 / s1 | +0.069 / +0.184 | +0.022 / +0.043 | +0.073 / +0.103 | +0.007 / +0.004 | 0.000 / 0.009 |
| holdout250 / s2 | +0.064 / +0.193 | +0.023 / +0.059 | +0.061 / +0.092 | +0.008 / +0.004 | 0.000 / 0.000 |
| textbook72 / s0 | +0.069 / +0.147 | +0.040 / +0.092 | +0.030 / +0.073 | −0.002 / +0.000 | 0.000 / 0.000 |
| textbook72 / s1 | +0.048 / +0.206 | +0.031 / +0.058 | +0.058 / +0.087 | +0.007 / +0.003 | 0.000 / 0.000 |
| textbook72 / s2 | +0.056 / +0.206 | +0.056 / +0.080 | +0.058 / +0.096 | +0.012 / +0.010 | 0.000 / 0.000 |

The over-confidence sits mostly in atomic `have` steps: the formula and term of an ordinary inference line, which is
where a wrong rule application (accepted by the environment, rejected by Lean) is written.

**By proof length.** Within-theorem AUROC, self vs siblings, in terciles of steps (n theorems with both classes in
that tercile):

| set / generator | short | mid | long |
|---|---|---|---|
| holdout250 / s0 | 0.747 vs 0.858 (35) | 0.771 vs 0.769 (59) | 0.745 vs 0.770 (78) |
| holdout250 / s1 | 0.679 vs 0.763 | 0.670 vs 0.785 | 0.756 vs 0.786 |
| holdout250 / s2 | 0.822 vs 0.866 | 0.774 vs 0.823 | 0.768 vs 0.836 |
| textbook72 / s0 | 0.757 vs 0.927 (8) | 0.737 vs 0.856 | 0.637 vs 0.751 |
| textbook72 / s1 | 0.802 vs 0.808 | 0.743 vs 0.910 | 0.753 vs 0.878 |
| textbook72 / s2 | 0.689 vs 0.771 | 0.726 vs 0.866 | 0.771 vs 0.757 |

There is no consistent concentration in long proofs.

## 3. Ancestor judges (`results/B4_ancestors.{json,txt}`)

Checkpoints were downloaded from `hf://buckets/dan-pandori/nd-rl/trajectory/ckpts/tj/` (`ladder/la_T1_best12_s0_r8.pt`
and `stage1_best12_s0_b1200.pt`) to `~/runs/proof-selection/ckpts/`. Both are `lean_staten` with the same
architecture. They scored the s0_r16 test pools with the same masking and checks (0 errors / mismatches).

| set | judge | within-theorem AUROC (finished) | − self [95 % CI] | k=8 selection | − self [95 % CI] |
|---|---|---|---|---|---|
| holdout250 | self s0_r16 | 0.788 | | 0.917 | |
| holdout250 | ancestor s0_r8 | 0.793 | +0.005 [−0.022, 0.030] | 0.917 | +0.000 [−0.005, 0.004] |
| holdout250 | stage-1 s0 | 0.691 | −0.097 [−0.140, −0.053] | 0.909 | −0.008 [−0.017, −0.001] |
| holdout250 | sibling s1_r16 | 0.828 | +0.040 [0.002, 0.076] | 0.930 | +0.013 [0.004, 0.023] |
| holdout250 | sibling s2_r16 | 0.786 | −0.002 [−0.039, 0.036] | 0.923 | +0.006 [−0.005, 0.016] |
| holdout250 | siblings mean | 0.829 | +0.041 [0.007, 0.074] | 0.928 | +0.010 [0.000, 0.021] |
| textbook72 | self s0_r16 | 0.687 | | 0.543 | |
| textbook72 | ancestor s0_r8 | 0.679 | −0.008 [−0.095, 0.073] | 0.538 | −0.005 [−0.025, 0.010] |
| textbook72 | stage-1 s0 | 0.605 | −0.082 [−0.181, 0.008] | 0.527 | −0.016 [−0.030, −0.001] |
| textbook72 | sibling s1_r16 | 0.783 | +0.095 [0.020, 0.170] | 0.573 | +0.030 [0.007, 0.059] |
| textbook72 | sibling s2_r16 | 0.737 | +0.050 [−0.046, 0.138] | 0.558 | +0.015 [0.003, 0.030] |
| textbook72 | siblings mean | 0.782 | +0.095 [0.020, 0.174] | 0.573 | +0.030 [0.010, 0.055] |

Ancestor r8 minus siblings, within-theorem AUROC: holdout250 −0.036 [−0.072, −0.002]; textbook72 −0.103 [−0.177,
−0.036].

## 4. Limitations

- **Ancestors of one seed only.** We do not know whether s1_r8 is equally blind to s1_r16's proofs.
- **What r8 ≈ r16 can and cannot rule out.** It means EI rounds 9–16 do not add the blind spot. It does not
  distinguish "inherited from earlier EI rounds" from "inherited from the seed's stage-1 training": stage-1 is a
  weaker judge overall, so its own penalty cannot be read off this table.
- **The mechanism numbers are descriptive.** Means of masked per-token log p; step-type gaps pool steps from all
  finished attempts; there are no CIs on the mechanism table.
- **Headline figure.**
  - Policies are capped at 16 attempts.
  - Count-stop parameters were fixed in phase 2 on dev, and the figure sweeps c = 1–8.
  - Bands are min–max over three generator seeds with one pool each.
- **Test sets.** These sets have been read in phases 1–3.
