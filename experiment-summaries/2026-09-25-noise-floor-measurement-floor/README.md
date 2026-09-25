---
written_on: 2026-09-25
written_by: agent:claude
---

# noise-floor: what difference can this project resolve? (proposal 11, run 2)

Status: completed 2026-09-25 07:22 UTC (autonomous executor, plus two pre-registered addenda) and
independently reviewed 2026-09-25 07:30–08:02 UTC. Review verdict: **accept — no hard-constraint
violation, nothing quarantined**. The reviewer re-derived all thirteen floor-table quantities with
its own tooling (own `nd_verify` re-run on every counted proof in all 52 held-out and 11 ladder
cells — ≈ 270,000 proofs, zero rejections; own pooled-sd, variance-decomposition, MDD and
bimodality-coefficient code; own renaming-class splits checker; a 2,105-proof Lean recheck through
three independent translators/checkers) and reports **"the recount found no error in the floor
table it exists to produce."** Its findings are eleven wording/scope corrections, not arithmetic —
this is, in the reviewer's words, **"the most carefully executed run I have reviewed in this
repository."** Numbers and wording below are the reviewer's, including the reviewer's own words
where they differ from the executor's. Run id `noise-floor`.

Review:
[`review_noise-floor.md`](https://github.com/dan-pandori/nd-takehome/blob/3c0c361/review_noise-floor.md)
(commits [`eab2f11`](https://github.com/dan-pandori/nd-takehome/commit/eab2f11) phase-1 blind
recount, [`3c0c361`](https://github.com/dan-pandori/nd-takehome/commit/3c0c361) phase-2 compare and
verdict; reviewer code under `review_nf_*.py` at that commit). Executor write-up:
[`run_noise_floor.md`](https://github.com/dan-pandori/nd-takehome/blob/3c0c361/run_noise_floor.md),
standing reference
[`NOISE_FLOOR.md`](https://github.com/dan-pandori/nd-takehome/blob/3c0c361/NOISE_FLOOR.md) (quote
this, with the corrections below applied — it was not edited after review), numbers `numbers.md` §
noise-floor (§§ N1–N11), log `log.md`, open questions `QUESTIONS.md`, all on branch
`dan_noise-floor` of `dan-pandori/nd-takehome`. Pre-registration:
[`preregistration/noise-floor.md`](https://github.com/dan-pandori/nd-takehome/blob/3c0c361/preregistration/noise-floor.md)
(`e8f4c3b`, plus two pre-registered addenda, each committed before any outcome it extends).
Proposal:
[`docs/proposals/2026-09-24-length-horizon-and-noise-floor.md`](../../docs/proposals/2026-09-24-length-horizon-and-noise-floor.md)
(run 2 of two; run 1 is the sibling `cap-horizon`).

Question: four raw pools drawn independently from the control's own generator settings, assembled
to the control's own composition (flat 31,000-per-length 155,000-record cap-6 `lean_seq` set,
depth-3 excluded), trained with the control's own schedule, × up to thirteen Stage-1 seeds — every
difference between those cells is noise by construction. **How big is it, per reported quantity,
and what is the smallest difference an n = 2 comparison (this project's standard) can resolve?**

## Configuration

- **Design.** Pools P1–P4: `gen.py` with the control's exact generator settings (no knob flags),
  seeds 21000–24000, assembled by `dsg_assemble.py` to the control's exact composition. Eight
  Stage-1 runs (P1–P4 × seeds 0–1), extended by two pre-registered addenda, each committed before
  any outcome it covers existed: **addendum 1** adds Stage-1 seed 2 on all four pools (12-cell
  design for held-out, frozen ladder and two of three coverage pools, to lift the seed variance
  component off 1 degree of freedom); **addendum 2** adds Stage-1 seeds 3–12 on all four pools,
  held-out only (52-cell design, to give the bimodal depth-3 slice's high-mode proportion a usable
  Wilson interval). All 52 checkpoints, read by the reviewer directly from the pickled weights, are
  the **same configuration**: a **3,214,336-parameter from-scratch GPT**, `lean_seq` surface
  format, cap 6, `--steps 6000 --bs 128 --lr 1e-3`, 4 layers, d 256, 8 heads — no duplicates among
  the 52 `(pool, seed)` pairs. Every held-out cell uses `k = 1, T = 0` on the same 5,000 theorems;
  every frozen-ladder cell uses `rounds 8, k 32, temperature 0.8, batch 512, no_train true`
  identically across all 11 runs; every coverage cell uses `k 2000, T 0.8, seed 0, batch 1000` with
  600,000 samples confirmed identical in all 16 cells. No EI — the frozen ladder only.
- **Two n = 1 gap-closers**, existing checkpoints, no retraining: `ds-generator`'s missing
  `la_frozen_g1_s0` cell (62 solved, `L*` 9, 612 targets) and `ds-composition`'s C0/A1 ladder at
  Stage-1 seed 1 (A1 s1: T1 847, `L*` 11, frozen 194, frozen `L*` 9).
- **Premise check, reviewer's own recount over all 4 × 155,000 records.** The four pools agree to
  ≤ 0.19 pp on box-depth shares, `ORE` share 1.42–1.49 %, `AS` share 46.70–46.89 % — **replicates,
  not arms.** Zero exact or renaming-class collisions against all nine evaluation pools in any of
  the four sets; a stricter premise-order-only class shows a real but pool-uniform residue of
  0.18–0.30 % of the held-out pool. Pairwise inter-pool renaming-class overlap is 5.82–5.95 %
  (**an order of magnitude below the pre-registered 30–70 % band** — E14, a favourable miss: the
  ≤ 6-line class universe is finite but the four draws are far more distinct than expected).
- **Acceptance and the checker, reviewer's own recheck.** `nd_verify` re-run on **every counted
  proof in all 52 held-out cells and all 11 ladder cells (≈ 270,000 proofs): zero rejections, zero
  disagreements with the recorded flag.** A 2,105-proof Lean recheck (≥ 200 per arm) through the
  run's own `nd2lean.py --check`, through the *fixed* `nd2lean.py`, and through proposal 9's
  `lean_check.py` (allowlist; axioms ⊆ {`propext`, `Classical.choice`, `Quot.sound`}; no `sorryAx`):
  **2,105 / 2,105 accepted on all three, zero disagreements.** In-loop gate, de-duplicated by the
  reviewer over 12,016,388 distinct-checked samples (the published 808/12,033,100 double-counts a
  self-test and three repeated `gate_heldout` re-runs): **801 Lean-only acceptances, strictly
  one-directional, 66.7 per million** (published: 808 / 67.1 per million; § N9 separately says 789 —
  internally inconsistent, corrected here). Term size tracks line count (ratio 0.52–0.67) and is the
  *same size in every ladder cell* while the solve count moves 4.3× — the spread is a count, not a
  change in what the proofs look like.
- **Hard constraints, all pass**: `nd_verify` blob-identical to `origin/main`; `TEST_RUN_DONE`
  unchanged; `--heldout` in `train.py` used only for a printed validation loss, never in the
  gradient or checkpoint selection; `nd2lean.py`/`train.py`/`eval_set.py`/`coverage.py`/
  `ladder_ei.py`/`expert_iter.py` unmodified against `origin/dan_ds-generator`; every training
  record from `gen.py` via the assembler, none hand- or LLM-written; cap 6 exact on all 52
  checkpoints (flat 31,000 × lengths 2–6, zero at length 7+). **No quarantine condition.**
- **Gate 0, pods, cost.** Pre-registration `e8f4c3b` committed 2026-09-24 16:05:09 UTC, first pod
  ≈ 16:06 (**36 s later**); addendum 1 (16:27:35) and addendum 2 (2026-09-25 03:33:31) each precede
  every outcome they extend. Two **A40, $0.49/h** pods (`nf-1`, `nf-2`), deleted after use: **28.32
  pod-hours, $13.88** of a $21 budget / 42 h ceiling (raised mid-run from $18/36 h, disclosed and
  reasoned in `QUESTIONS.md`, once measured per-job costs — a ladder round 1.94 pod-hours, not the
  brief's 1.25 — showed the original design needed ≈ 60 pod-hours against a 36 h ceiling). One
  arithmetic slip in addendum 2's own justification text (an 8-cell figure mislabelled as the
  12-cell one) does not affect the argument it supports. Artifacts at
  `hf://buckets/dan-pandori/nd-rl/noise-floor/{ckpts,artifacts,data}`, reviewer-verified present.

## Observations

All values are the reviewer's independent recount; every row of the floor table it checked equals
the executor's, to the last decimal.

### The floor table — what stands

| quantity | cells | range | pooled sd | MDD at n = 2 | as a ratio |
|---|---|---|---|---|---|
| frozen ladder, transfer solved (of 2,285) | 8 | 62 – 265 (4.27×) | 74.06 | **±397 (235 % of mean)** | 3.35× |
| frozen ladder, transfer `L*` | 8 | 9 – 10 (1.11×) | 0.354 | **±1.9 (21 % of mean)** | 1.21× |
| frozen ladder, RL targets solved (of 4,495) | 8 | 714 – 1,765 (2.47×) | 391.2 | **±2,098** | 2.67× |
| `targets_reductio_req` pass@2,000 (of 300) | 8 | 6 – 46 (7.67×) | 13.23 | **±71** | 3.32× |
| `depth3_req` required@8 (of 300) | 8 | 61 – 242 (3.97×) | 73.27 | **±393** | 3.57× |
| held-out greedy, overall (5,000) | 52 | 0.8420 – 0.9630 | 0.0304 | **±16.3 pp** | 1.18× |
| held-out greedy, 6-line bin (1,000) | 52 | 0.4210 – 0.9200 | 0.1523 | **±81.7 pp** | 2.22× |
| held-out greedy, depth-3 slice (500) | 52 | 0.0080 – 0.9180 (114.75×) | 0.3053 | **not resolvable** | 4.72× |
| held-out greedy, 2/3/4/5-line bins | 52 | — | 0.0031–0.0151 | **±1.7 / ±2.6 / ±8.1 / ±8.1 pp** | 1.02–1.09× |

**The falsifier did not fire.** It required the frozen ladder's max/min under 1.2×; measured **4.27×
(3.6× past the threshold)**. Proposal 11's premise stands: the floor is far larger than the project
assumed, and every frozen-ladder value proposal 10 ever reported sits inside the null range.
**Pre-registered expectations, the reviewer's own independent scoring**: 7 met (E3, E11, E12, E13,
E15, E17, E18), 2 partly (E4, E8), 9 missed — all in the direction "the floor is bigger than
predicted" (E1, E2, E5, E6, E9 narrower not wider, E10, E14 an order of magnitude, E16, E19
narrowly), 1 not measurable (E7, `targets_depth3` coverage was dropped for budget before it ran) —
**exactly the executor's own partition.**

### What must be reworded (the reviewer's corrections to `NOISE_FLOOR.md`/`run_noise_floor.md`)

1. **"≈ 99 % of the variance is the individual training run" and "the pool variance component is
   negative on **every** quantity"** generalise a three-quantity table to all eight. At the 4 × 13
   decomposition, σ²_pool is negative on **seven** of eight held-out quantities (the 3-line bin is
   **+3.67e−7**, 1.5 % of variance, p = 0.30 — still zero for practical purposes, but not "every"),
   and the residual share is **95.8 %** on held-out overall, 94.3 % on the 6-line bin, 92.7 % on
   depth-3, but only **66–75 %** on the four precise bins and the 6-line no-pattern slice. Corrected
   wording: "**≈ 95 % on the wide quantities.**"
2. **"The seed component is a few per cent of the residual"** is true for the three tabulated
   quantities (4.4–7.9 %) but the seed main effect is **31–51 % of the residual** on the 4-line,
   5-line, 6-line no-pattern and 3-line bins (F-test p = 0.005–0.030) — because `--seed j` shares an
   initialisation across all four pools, so on the precise bins seeds behave as a **block**. This is
   operationally useful, not just a correction: **an arm-vs-control comparison that reuses the same
   seeds in both arms cuts the MDD by 13–17 %** on those bins, a free precision gain the current
   text argues against.
3. **"G1's own two seeds bracket C0's 158/114 and G2's 44/125 entirely"** — false for one value: G1's
   bracket is [62, 170], which does not contain G2 seed 0's 44. Corrected: "brackets C0 entirely and
   G2's seed 1."
4. **`NOISE_FLOOR.md`'s "at n = 13 to `1.183 · sd`"** → **`1.146`** (n = 12 gives 1.197; no n gives
   1.183). This constant is written to be quoted verbatim by future runs.
5. **The gate total** (§ N8/`summary.json` say 808/67.1 per million; § N9 says 789/66.1 — internally
   inconsistent). De-duplicated correctly: **801 / 12,016,388 = 66.7 per million.**
6. **"about p = 0.22 one-sided"** (§ N10) — 2/(n+1) = 0.222 is the **two-sided** probability;
   one-sided is **0.111**.
7. **Three of the eight "bimodal" flags in `NOISE_FLOOR.md`'s second table are artefacts**: transfer
   `L*` (b = 0.818, a 7:1 split of two values), the 2-line bin (b = 0.648, discreteness at 6 distinct
   values near saturation), and held-out overall (b = 0.5566, clearing the 0.5556 threshold by
   0.001). Only the 6-line bin and the depth-3 slice are meaningfully bimodal, and they are the same
   cell seen twice.
8. **"Of twenty standing findings, two survive"** — both survivors are **n = 1 vs n = 1**, scored
   against an n = 2 MDD, and both are the cap-8 arm, which `NOISE_FLOOR.md`'s own scope note says
   this floor is not claimed for. The distribution-free statement is stronger and honest: 976 is
   3.7× the null maximum of 265, and `L*` 11 is one point above the null maximum of 10.
9. **"The sampler and the checker contribute ≈ 0" is inferred, not measured in this run** — it rests
   on two inherited n = 1 comparisons, and this run's own sampling seed is tied to the Stage-1 seed
   on the frozen ladder, so its σ²_seed and residual both carry unseparated sampler variance.
10. **Name the Lean surface format precisely.** The gate rate is measured pre-BOTE-fix; re-checking
    the 139 distinct proofs behind the 808 raw Lean-only acceptances with the *fixed* renderer/
    checker rejects **116 (83 %)**, leaving 23 genuine cases — the published **67 per million becomes
    ≈ 16 per million post-fix**. No counted proof is affected either way.
11. **`numbers.md` should record that addendum 1's 12-cell design was cut to 8** for the ladder and
    two coverage pools (disclosed in `log.md`, not in `numbers.md`/`run_noise_floor.md`), and that
    E12 ("σ²_seed ≥ σ²_pool on the frozen ladder") is met at exactly the 1-df design addendum 1 was
    written to fix — the general question is answered at 12 df by the held-out decomposition, not by
    the ladder's own.

**Nothing in the floor table itself is unsupported**, and no standing-finding verdict changes under
the recount.

## Limitations and next step

- The four items above the fold (1, 2, 8, 9) are the ones that change how a reader should use the
  floor, not just its footnotes: the variance story is weaker on precise bins than headlined, seed
  pairing is a free win the write-up currently discourages, and both "clears the floor" survivors
  and the sampler/checker null are thinner (n = 1) than the confident prose suggests.
- The depth-3 held-out slice and the 6-line bin remain **not resolvable at n = 2 at all**; the
  reviewer's highest-value, zero-GPU follow-up is checking whether the held-out validation loss
  `train.py` already logs every 200 steps predicts which mode a run lands in — if so, this floor's
  least tractable quantity becomes usable as a covariate rather than an unresolvable bimodal cell.
- Ranked next measurements (reviewer): (1) one more frozen-ladder seed on an existing checkpoint
  (≈ $0.95) to give the sampler/checker-null claim more than n = 1; (2) four more frozen ladders
  (seed 2 on P1–P4, ≈ $3.80) to take the ladder's seed component off 1 df, closing what addendum 1
  set out to do; (3) re-gate existing samples with the fixed `lean_tok`/`nd2lean` (no GPU) and
  publish both gate-rate figures labelled by translator version; (4) the zero-GPU loss-curve check
  above; (5) adopt paired seeds as the project's standard for precise-bin comparisons.
- Open in `QUESTIONS.md`: whether to keep the `pod_budget_watch` fix (a dead-code bug that could
  have mis-attributed a budget warning and deleted the wrong run's pods — sibling `cap-horizon`'s in
  this case) as applied, and whether the executor's choice to run all four `ds-composition`
  gap-closer cells (not just the requested seed 1) was the right call.
