---
written_on: 2026-11-02
written_by: agent:claude
---

# Caption: RL needed to double a strategy's share of correct proofs

This figure is for the group report's section "How much RL is needed to recover rare or fully
excluded proof strategies?". It is laid out like Dan's knockout chart, but here the strategies are
naturally rare, not knocked out.
- **Figure.** [`figs/strategy_rl_needed.png`](figs/strategy_rl_needed.png), 300 dpi, with PDF and
  SVG beside it.
- **Script.** [`rare_strategies_fig.py`](rare_strategies_fig.py). It is CPU only and runs in about
  5 s.
- **Numbers.** Every number is in [`rare_strategies_numbers.json`](rare_strategies_numbers.json).
  The script recomputes Bpool from the per-problem read tables and asserts that it equals
  `numbers.json` → `q2_charles149` for every key and checkpoint.

## Caption

> **RL needed to double a strategy's share of correct proofs, by how much of it pretraining
> contained.** Each point is one proof strategy in Dan's taxonomy. x is its share of the 154,630
> S1g3 pretraining proofs. y is the RL ÷ pretraining FLOPs at which its share of the model's
> Lean-accepted attempts on Charles149 first reaches twice the base's (seed 0, k = 1,024). y is
> interpolated in log R between checkpoints of the main run, with 95% problem-bootstrap bars.
> This is natural prevalence, not a knockout, and 2× is our choice of bar. Only proof by
> contradiction doubles, from the base to the L3 end:
> - working, 7.6 → 23.7% of correct proofs (3.1×; at the bar at R ≈ 0.037);
> - idle only, 0.5 → 13.5% (27×; already at the bar at the first checkpoint, R 6.5 × 10⁻³);
> - any classical step, 12.9 → 39.8% (3.1×; R ≈ 0.043).
>
> The other strategies stay below 2× up to the largest budget (R 0.45):
> - modus tollens (0.13% of pretraining proofs), 5.0 → 5.7% (1.15×);
> - disjunctive syllogism, 18.3 → 19.9% (1.09×);
> - working proof of a negation, 22.3 → 27.5% (1.23×);
> - genuine proof by cases, 29.1 → 35.6% (1.23×);
> - ex falso, 25.0 → 27.5% (1.10×);
> - double-negation elimination falls, 4.9 → 2.7% (0.56×).
>
> Excluded middle is never used, before or after RL. The GRPO training pools are 40–45%
> classical-only theorems, which need a reductio.

The full-width figure also carries its own small-print caption (seed, set, bar, main run, low-RL
arms).

## Takeaways (to paste under the graph)

- **Growth does not track pretraining prevalence.**
  - Proof by contradiction (0.9–1.5% of pretraining proofs) is the only strategy whose share of
    correct proofs doubles.
  - Disjunctive syllogism is about as common in pretraining (1.19%) but stays flat (1.09×).
  - Modus tollens, the rarest strategy the model uses (0.13%), grows only 1.15×.
  - Commoner strategies (4–21%) grow 0.6–1.2×.
- **"Rare in pretraining" does not mean rare in the base's proofs.**
  - Before any RL, modus tollens is 5.0% of the base's correct proofs (37× its pretraining share),
    and disjunctive syllogism is 18.3% (15×).
  - The problems set the strategy, not the corpus mix. So RL has no excluded strategy to recover
    here.
- **Where on the R axis.**
  - Idle proof by contradiction doubles first: by the main run's first checkpoint (R 6.5 × 10⁻³).
    In the 16-prompt low-RL arm it already doubles at 3.7 × 10⁻⁴.
  - Working proof by contradiction doubles between R 0.030 and 0.15 (interpolated ≈ 0.037).
  - Nothing else doubles by R 0.45.
  - Every crossing is at least 3 × Yue et al.'s median RL budget (1.2 × 10⁻⁴), and the
    main-run crossings are 50–360 × it.
- **The shift is a habit toward contradiction, much of it idle.**
  - Idle-only reductios, which do no logical work, grow 27×; working ones grow 3.1×.
  - The likely source is the GRPO pools, which are 40–45% classical-only theorems. RL is not
    discovering rare strategies.

## Caveats

- **Seed 0 only.**
  - The crossing Rs are uncertain by up to an order of magnitude: working contradiction's 95%
    interval is 0.010–0.27.
  - The base's working reductios are concentrated: 10 problems (7 of them classical-only) hold 89%
    of its accepted working-reductio attempts, and the 30 classical-only problems hold 88%.
- **Naturally rare, not knocked out.** Nothing was removed from the S1g3 corpus. This is not the
  same experiment as Dan's knockout, and the y axis does not measure recovery from zero.
- **What the unit measures.**
  - Bpool is the share of the model's correct proofs, so it measures strategy choice; overall
    success itself rises 1.8× (39.2 → 70.3% of attempts).
  - But it is pooled over all accepted attempts. It therefore also moves with which problems get
    solved: RL solves more classical-only problems, which force a reductio.
  - On the 117 problems that both the base and the L3 end solve, working contradiction rises by
    only +5.3 points (README, B-matched).
  - Unit A (correct and using the strategy, per attempt) would mix in competence as well. It was
    not used here.
- **The 2× bar is relative.**
  - It is easy for a strategy with a tiny base share (idle reductio, 0.5%).
  - It is hard or impossible for a common one. Genuine cases (29%) can rise at most 3.4×, and
    direct reasoning (86%) cannot double.
- **The GRPO training pools are 40–45% classical-only theorems.** Their generator proofs use
  working contradiction in 40–47%, which may favour contradiction over every other strategy.
- **The main run mixes runs.**
  - Its first three checkpoints come from the mini L2 run: the same L2 command on a 1,000-row
    pool.
  - The rest come from the full run: L2 end, then the L3 end, then L3 × 2 (ki78).
  - The interpolation between it8 (0.030) and the L2 end (0.15) spans 0.7 decades.
- **Points are not independent.** Free contradiction is a subset of working; any classical step
  contains every contradiction.

## How y is computed

- **Bar.** Bpool(checkpoint) ≥ 2 × Bpool(base). Bpool is the pooled share of all Lean-accepted
  attempts that use the strategy (`analyze.unit_Bpool`). Both are on Charles149 at k = 1,024,
  seed 0.
- **Main run.** In R order: mini L2 it2 (6.5 × 10⁻³), it4 (0.014), it8 (0.030), L2 end (0.15), L3
  end (0.29), L3 × 2 (0.45).
  - y is the first of these at the bar, interpolated linearly in log10 R on the fold change
    between it and the checkpoint before it.
  - A strategy at the bar by the first checkpoint is drawn there as a down-triangle, meaning "at
    or below".
  - A strategy below the bar at L3 × 2 is an up-triangle above the axis.
- **The two low-RL arms are not used for y.** They are kp16 (16 prompts per step; steps 1–32 read)
  and kdense (the mini L2 command with checkpoints at optimizer steps 1, 2, 4 and 8), R 7.6 × 10⁻⁵
  to 6.6 × 10⁻³. Only idle contradiction meets the
  bar in them:
  - kp16 from step 8 (R 3.7 × 10⁻⁴, 2.1×), and 4.4× / 5.1× at steps 16 / 32;
  - kdense at step 8 (R 6.6 × 10⁻³, 2.2×), the twin of mini L2 it2.
- **Bars.** A 95% bootstrap over the 149 problems. Each draw resamples the problems jointly in
  every read and recomputes the base and checkpoint Bpools and the crossing; there are 4,000
  draws.
- **Dodge.** Free (1.45%) and working (1.47%) contradiction are drawn 2.2 pt either side of their
  x, so that both bars show.

## Table

Shares are % of the S1g3 pretraining proofs, and % of all Lean-accepted attempts (Bpool), with 95%
problem-bootstrap intervals. "R at 2×" is RL ÷ pretraining FLOPs; "–" means not reached by 0.45.
"Max fold" is the largest fold on the main run.

| strategy σ | key | pretraining % | base Bpool | L3-end Bpool | fold | R at 2× [95%] | max fold |
|---|---|---:|---:|---:|---:|---|---:|
| proof by contradiction, working | `work_red` | 1.474 | 7.56 [3.0, 13.2] | 23.66 [16.9, 31.1] | 3.13 | 0.037 [0.010, 0.27] | 3.21 |
| proof by contradiction, free (working, no ¬¬ scaffold) | `free_red` | 1.450 | 6.76 [2.5, 12.4] | 22.16 [15.7, 29.2] | 3.28 | 0.031 [0.0083, 0.24] | 3.39 |
| proof by contradiction, idle only | `templ_red` | 0.918 | 0.50 [0.2, 1.0] | 13.50 [8.8, 18.9] | 26.8 | ≤ 0.0065 (first checkpoint; 94% of draws) | 31.1 |
| any classical step (#10–#12) | `any_dn` | 6.284 | 12.92 [6.2, 20.9] | 39.80 [31.7, 48.0] | 3.08 | 0.043 [0.015, 0.14] | 3.27 |
| modus tollens | `mt` | 0.135 | 4.99 [1.0, 10.1] | 5.72 [2.1, 10.2] | 1.15 | – (reached in 12% of draws) | 1.30 |
| disjunctive syllogism | `ds` | 1.185 | 18.29 [10.3, 27.1] | 19.87 [13.4, 26.7] | 1.09 | – | 1.09 |
| proof of a negation, working | `work_neg` | 5.562 | 22.33 [13.8, 31.7] | 27.53 [19.8, 35.9] | 1.23 | – | 1.28 |
| proof by cases, genuine | `gen_cases` | 6.036 | 29.06 [19.0, 39.5] | 35.64 [26.9, 44.8] | 1.23 | – | 1.26 |
| double-negation elimination | `dne` | 3.940 | 4.86 [0.7, 10.2] | 2.71 [0.0, 6.0] | 0.56 | – | 0.96 |
| ex falso | `efq` | 21.440 | 24.99 [15.7, 35.1] | 27.46 [20.1, 35.6] | 1.10 | – | 1.15 |
| excluded middle | `lem` | 0.062 | 0.00 | 0.00 | – | never used (textbooks 0%) | – |

The other distinct taxonomy keys are in the table only, not the figure:

| strategy σ | key | pretraining % | base Bpool | L3-end Bpool | fold | R at 2× [95%] | max fold |
|---|---|---:|---:|---:|---:|---|---:|
| proof by contradiction, any (working or idle) | `u:contradiction` | 2.392 | 8.06 [3.5, 13.7] | 37.17 [29.5, 45.7] | 4.61 | 0.022 [0.0087, 0.051] | 4.95 |
| proof by cases, any | `u:cases` | 6.566 | 29.29 | 36.70 | 1.25 | – | 1.28 |
| proof of a negation, any | `u:neg_intro` | 7.454 | 22.96 | 30.59 | 1.33 | – | 1.36 |
| lemma reuse (#13) | `u:lemma_reuse` | 2.510 | 7.57 | 8.97 | 1.18 | – (3% of draws) | 1.30 |
| disjunction introduction (#4) | `u:or_intro` | 34.195 | 18.68 | 30.40 | 1.63 | – (15% of draws) | 1.65 |
| conjunction introduction (#2) | `u:and_intro` | 24.143 | 41.94 | 42.44 | 1.01 | – | 1.08 |
| conditional proof (#3) | `u:cond_proof` | 39.850 | 51.18 | 49.44 | 0.97 | – | 1.00 |
| direct reasoning (#1) | `u:direct` | 43.221 | 85.87 | 83.66 | 0.97 | – | 1.01 |

- **Keys left out.** `u:disj_syll`, `u:mod_tollens`, `u:ex_falso`, `u:dne` and `u:lem` equal `ds`,
  `mt`, `efq`, `dne` and `lem` on every problem of every read; the script asserts this.
- **Nesting.** `free_red` ⊂ `work_red`; `u:contradiction` = `work_red` ∪ `templ_red`; `any_dn` ⊇
  `u:contradiction`.
- **`free_red` is not idle.** It is Dan's kind `free`: a working reductio whose theorem supplies no
  ¬¬G or ¬G → ⊥ scaffold and whose goal is not a negation (TAXONOMY.md #10). The idle category,
  "template-only" in the README (every reductio vacuous, pointless or an idle scaffold), is
  `templ_red`.
- **Pass@1** (attempt success, % of attempts): base 39.2, it2 44.8, it4 48.6, it8 55.5, L2 end
  68.3, L3 end 70.3, L3 × 2 69.4.
