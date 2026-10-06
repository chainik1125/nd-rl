# Pre-registration — run `noise-floor` (proposal 11, run 2: what difference can this project resolve?)

Written 2026-09-24, **before any `nf-*` pod exists**. Executor: agent:claude. Branch `dan_noise-floor`
(worktree from `origin/dan_ds-generator`). Brief: run brief `noise-floor` (proposal 11,
`~/nd-rl/docs/proposals/2026-09-24-length-horizon-and-noise-floor.md`). Policy: `AGENT_POLICY.md`.
Sibling run on this host: `cap-horizon` (own pods `ch-*`, own budget; mine are `nf-*`).

## Question

Four raw pools drawn independently from the **control's generator settings**, assembled to the
**control's composition**, trained with the **control's schedule**, differing only in generator RNG
draw and Stage-1 seed. Every difference between them is noise by construction. **How big is it, per
reported quantity, and what is the smallest difference an n = 2 comparison can resolve?**

## What is actually run

### The pools

`P1`–`P4`: `make_coverage_sets.py gen` with **no generator knob flags** (the unmodified `gen.py`
path: `ore_steps` 1, no `ore_boxes`, 45 % goal / 55 % forward, `strict` off, `bot_p` 0.02,
contradictory premises kept), `--tries 6000000` total split over the pod's workers, seeds
**21000 / 22000 / 23000 / 24000**, then `merge` (renaming-class dedup, validation-36 dropped), then
`dsg_assemble.py` (seed 0): flat **31,000 per pruned length 2–6 = 155,000**, depth-3 excluded in
written **and** pruned form, cap 6, every evaluation/ladder pool's classes excluded. This is
byte-identical to `pod/dsg/gen.sh`'s G1 path with the two knob flags removed.

**Two disclosed deviations from the control's own historical generation command**, both stated
before the run and both applied identically to all four pools (so they cannot create a difference
*between* P1–P4, which is what this run measures):

1. **`--cap_np 1000000000 --cap_pat 1000000000` (no per-worker output cap).** The control pool was
   built with `--cap_np 1500 --cap_pat 3000` *per worker at 90 workers*; my pods have ~32 workers,
   so those caps cannot be reproduced and would themselves change composition. `ds-generator`'s G1
   used the uncapped path and its shape table matched the control's to sampling noise, so the
   uncapped path is the one with a demonstrated match. It also makes G1 directly poolable as a
   fifth draw (see gap-closer 1).
2. **The local control pool is `pool_cap6_recon.jsonl`, a reconstruction** (723,534 classes) of a
   pool that was never pulled back. P1–P4 are fresh generator output, not subsamples of it. The
   control's *set* (`data/p2/train_depth3_f0_a1.jsonl`) is unchanged and its published numbers are
   used only as a marked reference value on figures, never as a ninth cell.

**Premise check, run before any cell is treated as a replicate** (`dsg_shape.py`, `dsg_overlap.py`):
per-length rule shares, box-depth distribution, reductio and derived-`ORE` shares, mean premises,
mean term size, contradictory-premise share, mean ND/Lean tokens, and the pairwise renaming-class
overlap between P1–P4. **If any pool's shape table differs from the other three by more than
sampling noise, it is reported as an arm, not as a replicate**, and the variance decomposition is
re-run without it (and with that stated).

### The models

Eight Stage-1 runs: `train.py --data data/nf/train_p<i>.jsonl --heldout data/p2/heldout.jsonl
--mode lean_seq --steps 6000 --bs 128 --cap 6 --seed {0,1}`. 3.3 M-parameter from-scratch GPT,
`lean_seq` Lean surface format, cap 6 — **the same model, format and schedule as `ds-generator`'s
C0/G1/G2 and `ds-composition`'s C0/A1**, which is what makes the floor quotable by those runs.

### The measurements (batch sizes fixed across every cell; `ND_SAMPLE_COMPACT=0`)

1. **Held-out greedy**, `eval_set.py --in data/p2/heldout.jsonl --k 1 --temperature 0 --batch 512`
   (5,000 theorems): overall, per length bin 2–6, and the **depth-3 slice** (the 500 held-out
   theorems with `pat.depth3` true — `ds-rendering`'s bimodal cell).
2. **Coverage pass@2,000**, `coverage.py --k 2000 --temperature 0.8 --seed 0 --batch 1000 --procs 4`
   on `data/p2/targets_reductio_req.jsonl` (300), `data/r3_1/depth3_req.jsonl` (300, required@8),
   `data/p2/targets_depth3.jsonl` (1,000).
3. **Frozen ladder only**, `ladder_ei.py --no_train --rounds 8 --k 32 --batch 512 --max_new 512`
   on `data/ladder/` (targets 4,495, transfer 2,285): transfer solved and transfer `L*`. No EI on
   the P-pools — the frozen ladder is the quantity with the largest observed spread at half the cost.

**Acceptance is Lean ∧ `nd_verify`.** `nd2lean.py`, `nd_verify`, `train.py`, `eval_set.py`,
`coverage.py`, `ladder_ei.py`, `expert_iter.py` unmodified (diff against `origin/dan_ds-generator`
recorded in `log.md`). Every counted proof goes through `nd2lean.py --check`; the agreement count
and the in-loop gate's Lean-vs-`nd_verify` disagreement rate are both reported (the latter was
unreported in `ds-generator` and the review asked for it).

### Two n = 1 gap-closers (existing bucket checkpoints, no retraining)

- **`la_frozen_g1_s0`** — `ds-generator`'s missing cell. Checkpoint
  `hf://…/ds-generator/ckpts/dsg/stage1_g1_s0.pt`. Same ladder command as above. Labelled as a
  **different generator flag set** (`--ore_steps 3 --ore_boxes`, shown by that run to match the
  control's shape table), i.e. a fifth independent draw of the control's *distribution*, **not** a
  fifth `P` pool. Reported both in and out of the pooled variance estimate.
- **`ds-composition` C0 and A1, ladder T1 + frozen, both Stage-1 seeds.** Checkpoints
  `hf://…/lean-format/ckpts/lf/stage1_a1_seq_s{0,1}.pt` (C0) and
  `hf://…/ds-composition/ckpts/dsc/stage1_a1_s{0,1}.pt` (A1), training sets from the same buckets.
  The brief asks only for seed 1; I run **all four cells on my own two pods** so the 2 × 2 is
  internally consistent on one hardware/sampler configuration, and I separately report the
  inherited seed-0 values (C0 T1 856 / frozen 158; A1 T1 965 / frozen 205) as a *same-checkpoint,
  different-host* reference point. If the budget binds, this drops first (see stop rule).

## Analysis — this is the deliverable

Per quantity, over the eight P-cells:

1. The eight values, and max/min and (max − min).
2. **Two-way decomposition, pools (4) × seeds (2), one observation per cell**: fit
   `y_ij = μ + a_i + b_j + e_ij`; report `MS_pool`, `MS_seed`, `MS_resid` and the variance
   components `σ²_pool = (MS_pool − MS_resid)/2`, `σ²_seed = (MS_seed − MS_resid)/4`,
   `σ²_resid = MS_resid` (negative estimates reported as such and truncated to 0 only when used).
   With one observation per cell the pool × seed interaction is confounded with error; stated.
3. The **pooled sd** = sd of the eight values (the quantity a future run's two seeds are drawn from).
4. The **smallest difference an n = 2 comparison has 80 % power to detect**, two-sided α = 0.05:
   `MDD = (t_{0.975,2} + t_{0.80,2}) · s · √(1/2 + 1/2) = 5.364 · s`, reported in absolute units and
   as a ratio to the mean. Reported alongside the fact that a **two-sided exact permutation test at
   2 vs 2 has minimum attainable p = 1/3**, so no n = 2 comparison in this project can ever be
   significant non-parametrically — the MDD is the parametric best case.
5. **Bimodality is checked, not assumed**, per quantity: the eight values, the gap between the two
   largest adjacent order-statistic gaps, and the bimodality coefficient
   `b = (γ₁² + 1)/γ₂` (`b > 5/9` ⇒ bimodal-consistent). Where a quantity is bimodal, the summary is
   **the proportion of models in the high mode with a Wilson 95 % interval**, not mean ± sd;
   for the depth-3 held-out slice the mode boundary is `ds-rendering`'s (> 0.44 high, < 0.11 low)
   and I report which of my eight fall in the gap.
6. **`NOISE_FLOOR.md`** at the repo root: one line per quantity giving the smallest difference worth
   reporting, written to be quoted verbatim by future runs and reviewers.

## Pre-registered expectations

| # | quantity | expected pooled sd / spread across the 8 cells |
|---|---|---|
| E1 | held-out greedy, overall | sd ≤ 0.02; max − min ≤ 5 pp |
| E2 | held-out greedy, 6-line bin | sd 0.04–0.10 |
| E3 | held-out greedy, 2-line bin | sd ≤ 0.005 (saturated) |
| E4 | held-out, depth-3 slice (n = 500) | **bimodal**; 5–8 of 8 in the high mode (> 0.44); sd ≥ 0.20 |
| E5 | `targets_reductio_req` solved (of 300) | max/min ratio **1.5–3×** |
| E6 | `r3_1/depth3_req` required@8 solved (of 300) | max/min ratio **1.5–3×** |
| E7 | `targets_depth3` solved (of 1,000) | max/min ratio 1.2–2× |
| E8 | frozen ladder transfer solved (of 2,285) | max/min ratio **1.4–2.5×**; sd ≥ 25 % of the mean |
| E9 | frozen ladder transfer `L*` | spread 9–11, i.e. **2 points of pure noise** |
| E10 | smallest n = 2 resolvable difference, frozen ladder | **± 40–60 %** of the mean |
| E11 | pool shape tables (premise check) | all four agree; `ORE` share within 1.2–1.8 %, box depth 0/1/2 within ±2 pp of C0's 53.2 / 35.4 / 11.4 |
| E12 | between-pool vs between-seed variance | **σ²_seed ≥ σ²_pool** on the frozen ladder — the Stage-1 draw, not the data draw, is the larger term |
| E13 | Lean vs `nd_verify` on counted proofs | **0 disagreements**; in-loop gate rate 0–1,000 per million, one-directional (Lean-only) |
| E14 | inter-pool renaming-class overlap | 30–70 % pairwise (the ≤ 6-line class universe is finite; stated so the pools are not mistaken for disjoint) |

**Falsifier.** If every quantity's spread across the eight null cells is **below** these bands — in
particular **if the frozen ladder's max/min is under 1.2×** — then `ds-generator`'s G1-vs-C0 gap was
not noise, it was a real effect of `--ore_steps 3 --ore_boxes`, and the shape account this project
has just declared dead is reopened. That is the most surprising outcome available here, it inverts
proposal 11's premise, and it will be stated as the headline if it happens.

**Second falsifier (E12).** If σ²_pool ≫ σ²_seed on the frozen ladder, then re-seeding Stage-1 is
*not* a valid error bar for a dataset-style run and every such run needs independent pools, not
just seeds — a more expensive standard than anything currently in force.

## Standing findings I expect this to invalidate

Checked by name in the write-up, with the original number beside the measured floor.

| standing finding | run | original number | my prediction |
|---|---|---|---|
| "G2's frozen ladder is marginally above C0 on seed 1" | `ds-generator` | 125 vs 114 (+10 %) | **inside the floor** — not a finding |
| "G1 is above both C0 seeds on the frozen ladder" | `ds-generator` | 170 vs 158 / 114 | **inside the floor** (it is the null arm; this is the premise) |
| "G1 s1 required@8 is 2.3× C0 s1" | `ds-generator` | 259 vs 113 | **inside the floor** |
| "A1's frozen ladder beats C0" | `ds-composition` | 205 vs 158 (+30 %), n = 1 | **inside the floor** — and directly retested here at n = 2 |
| "A1's T1 ladder beats C0" | `ds-composition` | 965 vs 856 (+13 %), n = 1 | **inside the floor** — retested here |
| "A1's frozen `L*` is 10 vs C0's 9" | `ds-composition` | 10 vs 9, n = 1 | **inside the floor** (E9 predicts 2 points of pure noise) |
| "A2's frozen ladder is below C0" | `ds-composition` | 111 vs 158 (−30 %), n = 1 | **inside the floor** |
| "A1's overall held-out gain" | `ds-composition` | +0.8 / +4.6 pp | **inside the floor** on s0, outside on s1 — i.e. unresolved |
| "A1's 6-line no-pattern gain" | `ds-composition` | +7.3 / +6.5 pp | **borderline** — same sign both seeds, but ≈ 1 sd of E2; survives only as "consistent in sign" |
| "`lean_seq` beats the token format on held-out" | `lean-format` | 0.909/0.896 vs 0.883/0.883 | **inside the floor** (E1) |
| "`lean_seq` `L*` 11 vs token 10" | `lean-format` | 11/11 vs 10/10 | **inside the floor** (E9) |
| "R2 − C0 depth-3 held-out gap" | `ds-rendering` | +0.129 | **inside the floor** (already self-reported as needing n ≈ 87) |
| **"A3 (cap 8) moves the horizon"** | `ds-composition` | `redreq` 54/72 vs 28/26; frozen ladder 976 vs 158; `L*` 11 vs 9 | **survives** — 2.2× / 6.2× is well outside every band above. This is the one I expect to stand, and it is the premise of the sibling run `cap-horizon`. |
| "G2's `redreq` coverage is 0 on both seeds vs C0's 31/27" | `ds-generator` | 0/0 vs 31/27 | **survives** (zero on both seeds is outside any multiplicative floor) |
| "EI's round-4 depth-3 endpoint lands in 0.38–0.44 whatever the base rate" | three runs | 0.387–0.436 | **survives** — a convergence claim, not a difference; not tested here (no EI on the P-pools) but the floor applies to any *difference* read off it |

## Budget, pods, stop rule

- **Pod budget $18 / 36 pod-hours**, registered with `podbudget noise-floor --set 36 18` **before the
  first pod**. RunPod balance at write time **$177.81**; proposal 11's hard floor is **$130**.
- **Two pods**, cheapest 24 GB+ card in stock; the class **and the real billed rate** are recorded in
  `log.md` (`podbudget`'s dollar column falls back to $0.50/h and is wrong for A6000 and 4090).
  Four jobs per pod at `CUDA_MEM_FRACTION=0.21` (`ds-generator`'s proven configuration).
- **Planned**: generation 4 × ≈ 0.1 h CPU; Stage-1 8 × 0.15 h; held-out 8 × 0.1 h; coverage 8 × 2.5 h;
  frozen ladder 8 × 1.4 h; gap-closers ≈ 8 h → ≈ 47 job-hours / 8 concurrent slots ≈ 6 h wall
  ≈ **12 pod-hours ≈ $7**, against 36 h / $18.
- **Stop rule.** Ordered drops if the ceiling would bind, in this order (the brief's order, with my
  reason for each): (1) the `ds-composition` C0/A1 seed-**0** ladder cells (the inherited values
  cover them); (2) the `ds-composition` ladder entirely; (3) **P4** entirely — the decomposition
  degrades to 3 × 2 and is reported as such; (4) the `targets_depth3` coverage (the least
  informative of the three pools: its solved counts are the least variable). Hard stop **30 h wall**
  from the first pod. Pods are deleted as soon as their work is pulled.
- `pod_budget_watch` warns at 80 % and deletes at 100 %; I watch for
  `~/runs/noise-floor/BUDGET_WARNING`.

## Deliverables

`run_noise_floor.md` (≤ 400 words + two figures), **`NOISE_FLOOR.md`** at the repo root,
`numbers.md` § noise-floor (every entry names its source file and the model it was measured on),
`log.md` dated, `artifacts/nf/summary.json` (one row per pool × seed), `STATUS.md`
`NOISE-FLOOR DONE <UTC>`, bucket `hf://buckets/dan-pandori/nd-rl/noise-floor/{ckpts,artifacts,data}`,
pods deleted, `~/runs/noise-floor/executor.done`. Questions for Dan in `QUESTIONS.md` with defaults.

---

## Addendum 1 — a third Stage-1 seed (2026-09-24 16:31 UTC)

Written **before any outcome of this run exists**: at the time of this commit the eight Stage-1 runs
are at step ~1,800 of 6,000 and no held-out, coverage or ladder number has been produced. Nothing
below is chosen in the light of a result.

**What changes.** Stage-1 **seed 2** is added on all four pools, giving a **4 × 3 = 12-cell**
design. The seed-2 cells get held-out greedy, the frozen ladder, and coverage on
`targets_reductio_req` and `depth3_req` — **not** `targets_depth3` (the brief's own least-informative
pool, and the first coverage drop in my stop rule). Commands, batch sizes, pools and checkers are
unchanged.

**Why.** The deliverable is an error bar, and the pre-registered design estimates the *seed*
variance component with **1 degree of freedom**, which is too few to answer E12 ("is the Stage-1
draw or the data draw the larger term?") at all. A third seed takes the decomposition to
df = 3 (pool) / 2 (seed) / 6 (residual) and the pooled sd to 11 df from 7. Three seeds on four pools
beats six pools on two seeds for the same cell count, because the seed axis is the one with 1 df.

**Cost.** 4 × (Stage-1 0.15 h + held-out 0.1 h + frozen ladder 1.4 h + two coverage runs 1.0 h)
≈ 11 job-hours ≈ 2.7 h wall on four slots ≈ **5.4 pod-hours ≈ $2.7**, against a projection of
≈ 13 pod-hours of the 36 h / $18 ceiling.

**Reporting.** Every floor states the number of cells behind it: held-out, frozen ladder,
`redreq` and `required@8` are **n = 12**; `targets_depth3` coverage stays **n = 8**. The
pre-registered expectations E1–E14 are scored against the 8-cell design they were written for
**and** against all 12, and both are shown where they differ.

---

## Addendum 2 — 40 more Stage-1 seeds, held-out only (2026-09-25 03:35 UTC)

`nf-2` finished its assigned work at 03:32 and would otherwise be deleted with ≈ 19 pod-hours of the
run's 42-hour ceiling unused. This addendum spends ≈ 4.3 of them on the cheapest and most-asked-for
measurement available.

**What changes.** Stage-1 seeds **3–12** on each of P1–P4 — **40 more models**, `train.py` exactly as
before, **held-out greedy only** (`eval_set.py --k 1 --temperature 0 --batch 512` on
`data/p2/heldout.jsonl`). No ladder, no coverage. Held-out therefore goes to **4 pools × 13 seeds =
52 cells**; the frozen-ladder and coverage floors stay at the 8 cells already reported and are
**not** changed by this.

**Why.** The held-out depth-3 slice / 6-line bin is the one quantity this run found to be *bimodal*
rather than merely wide, and for a bimodal quantity the right estimand is **the probability that a
training run lands in the high mode**, not a mean ± sd — `ds-rendering`'s reviewer said so and
costed it at ≈ 87 seeds per arm. At 12 cells my high-mode proportion is 3/8 with a Wilson 95 %
interval of **[0.137, 0.694]**, which is almost no information. At 52 cells the interval is roughly
±0.13 instead of ±0.28. It is also the only way to give the **pool** variance component more than
3 degrees of freedom, which is what E12 needs: with 13 seeds per pool the pool means are estimated
well enough to say whether re-drawing the data set does anything at all.

**Cost.** Measured on these pods: Stage-1 6,000 steps ≈ 21 min and held-out greedy ≈ 5 min, both at
four concurrent jobs, so ≈ 6.5 pod-minutes per model → **≈ 4.3 pod-hours ≈ $2.1**. Run as four
chains of ten models on `nf-2`.

**Pre-registered expectations for the 52-cell held-out set** (written now, before any seed-3+ model
exists):

| # | quantity | expectation |
|---|---|---|
| E15 | high-mode proportion of the depth-3 slice (> 0.44) | **0.30–0.55**, Wilson half-width ≤ 0.15 |
| E16 | cells landing in the gap 0.11–0.44 | **≤ 25 %** of 52 — the cell is genuinely bimodal, not merely wide |
| E17 | pooled sd, held-out overall, at 52 cells | **0.030–0.050** (it is 0.040 at 11) |
| E18 | var_pool for held-out overall, now with 12 df | **≤ 20 %** of var_resid — re-drawing the pool still does nothing |
| E19 | the four pools' high-mode proportions | all four within **±0.20** of each other |

**Reporting.** Every held-out floor states 52 cells; every ladder and coverage floor states 8. The
n = 8 held-out figures already published in `numbers.md` § N2 are kept and shown beside the n = 52
ones, so nothing already quoted by the sibling run `cap-horizon` moves silently.
