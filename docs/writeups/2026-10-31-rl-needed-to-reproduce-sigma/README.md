---
written_on: 2026-10-31
updated_on: 2026-11-01 (run coe-dose01: the 0.1 % σ dose arm; knockout s1 to round 32)
written_by: agent:claude
---

# How much RL does it take to bring back proof by contradiction?

A figure for the midterm write-up: create-or-elicit's 15 expert-iteration ladders plus coe-dose01's three 0.1 % σ dose
ladders and knockout s1's rounds 25–32, with compute measured the way rl-compute-share measured it.

![RL needed to reach the full bases' level](rl_needed.png)

## Summary

The capability is solving held-out theorems whose every proof needs σ (classical proof by contradiction). These are the
106 strict σ-covered members of create-or-elicit's classical families; no ladder trains on them. The bar is what the
full-exposure bases solve **before any RL update**: a median of 14 of 106 (range 2–19 across 6 seeds) in their
round-1 samples at k 32.

| pretraining | RL ÷ pretraining compute needed to reach 14 of 106 | seeds |
|---|---|---|
| full exposure (σ in 10.9 % of K12) | none: it is their median before RL | 6 |
| 1 % σ dose | **0.39–0.73×** (rounds 2–3) | 3 of 3 |
| 0.1 % σ dose | **0.69–2.58×** (rounds 3, 5, 8; median 1.37×) | 3 of 3 |
| knockout (σ removed) | **not reached by 5.6–13.2×**; the best seed (s1, 32 rounds) solved 6 of 106 | 0 of 6 |

- **Observed:** a little σ in pretraining makes it cheap for RL to recover the capability. With none, 16 rounds of
  this RL (32 for knockout s1) leave all six knockout seeds far below the bar. So without σ the RL needed is at least
  about 8× what a 1 % dose needs (5.6× against 0.73×; about 15× seed against seed at the medians).
- **Observed (coe-dose01): a graded response.** With σ in 0.1 % of pretraining (155 of 155,000 proofs) the bases barely
  use σ in their own round-1 samples (0–2 free positive reductios; knockout 0, 1 % 170–219; 0, 0 and 3 of 106 held-out
  members), yet RL crosses the bar on all three seeds, at a median 1.37× (≈ 3× the 1 % arm's median 0.41×; s2's 0.69×
  overlaps the 1 % range).
  At round 16 they solve 30 / 28 / 19 of 106 (1 %: 58 / 34 / 30). Crossing 3 / 3 against the knockout's 0 / 6: one-sided
  Boschloo p = 0.0033. The 0.1 % vs 1 % differences are descriptive at n = 3.
- **Interpretation, not tested here:** the knockout row is a lower bound. Knockout s1's free positive reductios grew from
  11 to 123 per round over rounds 16–24 (create-or-elicit, Addendum 6) and then levelled off at ≈ 140 per round over
  rounds 25–32 (coe-dose01), with y 5 → 6; more RL of this kind looks unlikely to get there soon. Between the 0.1 % dose
  and none the requirement jumps from ≈ 1.4× to more than 13× (one seed); whether 0.01 % would fall in between is untested.

**Source of the new rows:** run `coe-dose01` ([proposal 33](../../proposals/2026-10-31-coe-dose01.md)), fork branch
`dan_coe-dose01`: `run_coe_dose01.md`, `numbers.md` § coe-dose01, pre-registration `preregistration/coe-dose01.md`.

## How it is computed

- **y (panel a):** `heldout_cum_by_round` in `artifacts/coe/rounds/<ladder>.json` (create-or-elicit's on fork branch
  `dan_create-or-elicit`; `la_dose01_s*.json` and knockout s1's `la_ko_s1_r32.json` on `dan_coe-dose01`), summed over families. A held-out theorem counts once any round's k 32 sample solves it,
  under the Lean-alone judge. Every arm gets the same samples per round.
- **x:** cumulative RL compute after each round's sampling, over the ladder's Stage-1 compute, as defined by
  rl-compute-share's reviewer (`review_rcs/ourruns.py`, fork branch `dan_rl-compute-share`):
  - sampling counts 2N per decoded token and per re-read state token;
  - fine-tuning counts 6N per trained token, including the 20,000 replayed pretraining records per round (excluding
    replay lowers x by about 14 %);
  - Stage-1 counts 6N·D, and N cancels.

  Round 1 is the base's own samples, so it sits at about 0.1×.
- **Check:** each ladder's total, including its last fine-tune, agrees with rl-compute-share's published ratio to two
  decimals (largest gap 0.0012×, 0.021 %: the extraction splits the state re-prefill per round, `ourruns.py` per ladder) on all 14
  ladders it counted (for example knockout s1 9.78× at r24, dose1 s0 5.97×). coe-dose01 re-ran the unchanged extraction
  and reproduced the published `rl_curve.json` exactly before adding its ladders.
- **Ladders:**
  - full s0–s2 are the trajectory (r1–8) + rl-continue (r9–16) ladders, as in create-or-elicit's own figure;
  - full s3–s5, knockout s0–s5 and dose1 s0–s2 are create-or-elicit's;
  - dose01 s0–s2 and knockout s1's rounds 25–32 are coe-dose01's (registry `coe-dose01`; the knockout s1 curve joins
    create-or-elicit's rows for rounds 1–24 with coe-dose01's for 25–32).
- **Files:**
  - `extract_rl_curve.py` ran read-only on agent2 and wrote `rl_curve.json` (re-run 2026-11-01 by coe-dose01);
  - `make_figure.py` draws `rl_needed.png` / `.pdf` from it. Run it with any Python that has matplotlib.

## Caveats

- **One organism:** 9.56M ALiBiGPT bases, T1 expert iteration, one family set. GRPO or a larger model could need more or
  less RL.
- **The bar is one choice:** the full bases' median before RL. A lower bar (5 of 106) would let knockout s1 count, at 7.3×
  (round 19). It is the only knockout seed above 1 of 106.
- **The full s5 ladder's compute rows are about 30 % below its siblings'** from round 1 on, and rl-compute-share left it
  out of its count. It is plotted dashed. It does not affect any number in the table.
- **Knockout s1's rounds 17–32 were exploratory** (17–24 create-or-elicit, Addendum 6; 25–32 coe-dose01, the same
  procedure) and are plotted dotted.
- **The 0.1 % arm has three seeds.** Replay carries σ at the arm's own rate (≈ 20 records per round at 0.1 %), so part of
  its recovery may be replayed exposure rather than RL on the model's own proofs.
- **The review of create-or-elicit holds** that the families "need" σ only within 16 rounds of this EI, not beyond. This
  figure states the same thing as a compute bound.

## Sources

- [create-or-elicit summary](../../../experiment-summaries/2026-10-12-create-or-elicit-sigma-knockout-not-acquired-in-16-ei-rounds-needs-exposure-1pct-dose-restores/README.md)
  and its round figure `coe_rounds.png`, which shows the same data against EI rounds.
- [rl-compute-share summary](../../../experiment-summaries/2026-10-19-rl-compute-share-yue-rl-is-1e-4-of-pretraining-no-computable-tested-run-above-2e-2-hypothesis-supported/README.md),
  Q5 and its "scale curve" recommendation.
- [create-or-elicit next steps](../../papers/create-or-elicit/next_steps.md), item 3 (the 0.1 % and 0.01 % dose arms)
  and item 6 (longer knockout ladders).
