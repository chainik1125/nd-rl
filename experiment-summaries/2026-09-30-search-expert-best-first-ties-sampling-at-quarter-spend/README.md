---
written_on: 2026-09-30
written_by: agent:claude
---

# search-expert: a best-first search expert, given sampling's action budget, spends ≈ ¼ of it and teaches the apprentice no more than sampling does

Status: **completed 2026-09-30 02:00 – 06:40 UTC (executor), independently reviewed 06:40–07:14 UTC. Verdict: the
primary null stands, H1 is falsified, with five rewordings and one claim not supported.** No hard-constraint
violation, nothing quarantined. Where the reviewer's wording differs from the executor's, this summary uses the
reviewer's.

Question: our EI expert is the network's own independent samples. If the expert is instead best-first search over
proof states, given the same per-target action budget, does the apprentice, **read out without search**, prove more
theorems on the long pool? (Proposal 16 item 2; the question in Dan's terms is "is search a better teacher than
sampling?")

The run was on the fork (`dan-pandori/nd-takehome`), branch `dan_search-expert` (from the fork's `dan` at
`d3abe474`). **Nothing is merged into the fork's `dan`**, including the new code (`state_search.py`: the step filter,
best-first `search_generate`, `resume_generate`; the new `state_ladder_ei.py` flags).

- Review: [`review_search-expert.md`](https://github.com/dan-pandori/nd-takehome/blob/6941a9a3/review_search-expert.md)
  ([`559a4518`](https://github.com/dan-pandori/nd-takehome/commit/559a4518) §Recount, written before the reviewer read
  the executor's write-up; [`6941a9a3`](https://github.com/dan-pandori/nd-takehome/commit/6941a9a3) §Compare /
  §Verdict). Reviewer scripts: `review_sx/` (`splits.py`, `rlean.py`, `recheck.py`, `recount_rr.py`, `expert.py`).
- Executor write-up: [`run_search_expert.md`](https://github.com/dan-pandori/nd-takehome/blob/6941a9a3/run_search_expert.md),
  `numbers.md` § search-expert, `log.md` 2026-09-30, `artifacts/sx/analysis.md`, `artifacts/sx/compute_table.md`.
  Results and DONE commit `15551c26`.
- Pre-registration: `preregistration/search-expert.md`, commit `9d138b60` at 02:31:18 UTC, never changed afterwards.
  The first pod (`sx-0`) was created at 02:31:47 UTC. The primary quantity follows Dan's suggestion of 02:11 UTC
  (shared with `frontier-supply`), adopted before the pre-registration was committed.
- Brief: [`BRIEF_search-expert.md`](../../docs/proposals/literature/BRIEF_search-expert.md).
- Figure (not checked by the reviewer, so linked, not copied):
  [`figures/search_expert.png`](https://github.com/dan-pandori/nd-takehome/blob/6941a9a3/figures/search_expert.png).

## Configuration

- **Checker: Lean alone** (`lean_judge.judge_many` → `lean_gate`). `nd_verify` is unmodified and judges nothing (the
  reviewer grepped the ladder, search, sampler and read-out code).
- **Model, every number below:** SN-cap12, `lean_staten`, 3,216,384 params (4 layers, d 256), from scratch; Stage-1 on
  K12 (`cap-horizon`'s `train_k12.jsonl`, cap 12, 155,000 proofs, 6,000 steps). Seeds 0–3 are `state-cap12`'s
  `stage1_SN12_s{0-3}.pt`; seeds 4–5 were trained in this run with the same recipe (held-out p2 greedy 0.9702 / 0.9762).
- **Every arm:** 8 EI rounds from the seed's Stage-1 checkpoint on the 4,495 `rl_targets.jsonl`
  (`state_ladder_ei.py`), T 0.8, batch 2,048, `max_steps` 48, `max_action` 256, fine-tune 600 steps lr 3e-4, 128
  proofs a step, replay 20,000 K12 records. Held fixed in every arm, and different from `state-cap12`'s ladder:
  **shortest found proof per target**, weight 4 (`--select shortest --max_per_thm 1 --rl_weight 4`); a **step filter**
  (`--step_filter`: an attempt or branch ends at a `have` that `lean_prefilter`'s checker is certain Lean rejects; it
  accepts nothing itself); `--no_eval`.
- **Arms:**
  - **A, sampling expert** (seeds 0–5): 32 independent attempts per target per round.
  - **B, best-first search expert** (seeds 0–5): priority Σ log p / L; expand the best node with 4 actions sampled at
    T 0.8; drop rejected or already-reached children; the first finished child ends the target. Budget: per target, A's
    generated actions on that target in the same round and seed. Unspent budget is not reallocated.
  - **A2, a re-draw of A** (seeds 0–1, `--seed_offset 50`), which measures the same-checkpoint spread.
  - **C, truncate-and-resume** (seeds 0–1): 4 chains per target, resume from the state before a failed action,
    restart after 4 failures from one state. Budget: A2's actions per target and round.
  - **T1** (secondary, **not pre-registered**): `state-cap12`'s ladder checkpoints `la_T1_SN12_s{0-3}_r8.pt` (same
    Stage-1 seeds; up to 4 random successes per target, no step filter, per-round evals, older code), re-read here.
- **Read-out** (round-8 checkpoints, no search; identical in all 40 read-outs, the reviewer checked): `lpool_reread.py`,
  k 256, T 0.8, seed 0, batch 2,048, `max_action` 512, `max_steps` 96, on `transfer_long2_91.jsonl` (the 91 of
  `long-pool-2`) and `rr600_13to16.jsonl` (400).
  - **Primary Q:** per seed, theorems with ≥ 1 Lean-accepted sample in 256, on the 91 plus rr600 `L_true` 15–16 (200),
    **291 theorems**. B vs A paired per seed and per theorem.
- **Pods:** 8 A40s at $0.49/h, `sx-0`…`sx-7`, 02:31–06:36 UTC. `sx-0`–`sx-5` ran A ∥ B for one seed each, sharing one
  GPU, with B waiting for A's round budget. `sx-6` / `sx-7` ran A2 ∥ C for seeds 0 / 1. **29.10 pod-hours, $14.26** of
  $25 / 50 h (`podbudget`).
- **Compute per arm** (registry rows `registry/search-expert`, summed by the reviewer; GPU A40, shared by two arms, so
  GPU-seconds are wall time under co-tenancy):

  | arm (8 rounds) | sampling GPU-s | actions | gen tokens | attempts | Lean checks | fine-tune GPU-s | train steps | train tokens | wait GPU-s |
  |---|---|---|---|---|---|---|---|---|---|
  | A (per seed) | 5,645–7,259 | 9.75–10.11 M | 192–196 M | 1,150,720 | 611–651 k | 2,647–3,111 | 4,800 | 504–506 M | — |
  | B (per seed) | 2,027–2,553 | 2.28–2.40 M | 43–46 M | 35,960 | 30–31 k | 2,678–3,136 | 4,800 | 520–525 M | 3,706–4,876 |
  | A2 | 6,486 / 6,642 | 9.81 / 10.05 M | 193 / 195 M | 1,150,720 | 614 / 651 k | 3,056 / 3,108 | 4,800 | 504 M | — |
  | C | 2,559 / 2,303 | 2.00 / 1.86 M | 40 / 37 M | 143,840 | 32 / 33 k | 3,100 / 3,161 | 4,800 | 516 M | 4,201 / 4,276 |

  No arm exceeds its comparator by 1.25× on any quantity (B's training tokens are 1.035× A's). Each read-out cost
  1,261–2,205 GPU-s.
- **Bucket:** `hf://buckets/dan-pandori/nd-rl/search-expert/{ckpts/sx,artifacts/sx,figures}` (ladder checkpoints
  `ckpts/sx/ladder/{A,B,A2,C}_s<S>_r8.pt`; read-out Lean dumps `artifacts/sx/dump/rr_*.jsonl.gz`); registry rows
  `hf://buckets/dan-pandori/nd-rl/registry/search-expert/`.

## Observations

All counts below were re-derived by the reviewer from the per-theorem rows and the ladders' round files.

**Primary: the apprentice (Q, of 291).**

| arm | s0 | s1 | s2 | s3 | s4 | s5 | mean | IQM [95 % CI] |
|---|---|---|---|---|---|---|---|---|
| A (sampling expert) | 79 | 111 | 129 | 148 | 113 | 102 | 113.7 | 113.8 [93.3, 134] |
| B (best-first expert) | 76 | 118 | 122 | 144 | 117 | 97 | 112.3 | 113.5 [91.5, 131.8] |
| A2 (re-draw of A) | 80 | 129 | | | | | 104.5 | — |
| C (truncate-and-resume) | 72 | 121 | | | | | 96.5 | — |
| T1 (`state-cap12`, secondary) | 142 | 207 | 218 | 220 | | | 196.8 | 212.5 [142, 220] |

- **B − A, paired per seed: −3, +7, −7, −4, +4, −5.** Mean **−1.3**, sd 5.5, t(5) = −0.59, 95 % CI of the mean
  [−7.1, +4.5]. That interval rules out a gain of ≥ 5 of 291. IQM −2.0 [−5.5, +3.75] by the reviewer's trimming, or
  −2.5 [−5.7, +4.3] by the executor's fractional-weight IQM. Both intervals contain 0.
- **Minimum detectable difference.** The same-checkpoint spread (A vs A2, seeds 0–1, differences −1 and −18) gives
  s = 9.0 and the pre-registered MDD **18.2**. From the between-seed sd of B − A the MDD is **7.9**. Both rest on few
  pairs; the A–A2 value is dominated by seed 1's −18. Per-theorem flips between A and A2: 47 and 66 of 291.
- **The pre-registered falsifier fires** (B − A ≤ 0 within the MDD). H2 is not supported.
- Discordant theorems (B-only / A-only) per seed: 26/29, 38/31, 34/41, 34/38, 33/29, 32/37; summed 197 / 205.
- Strata, B − A mean over 6 seeds: exact-17 (61) +0.17; ≥ 18 (30) +0.17; rr600 13–14 (200) +2.0; rr600 15–16 (200)
  −1.67; the 91 +0.33. None is separated from 0.
- **C − A2: −8, −8** (2 seeds). The apprentice is not better; with 2 seeds it cannot be called worse either.

**The expert (H1, falsified).**
- **B was matched on budget, not on spend.** Best-first stops at the first proof and does not reallocate unspent budget,
  so **B spent 22.6–24.6 % of A's actions** (C: 18.5–20.4 % of A2's). On no target-round did B spend more than A.
- Targets one expert solved in round r that the other expert had never solved in rounds 1..r, summed over 8 rounds:
  **B 455–746 per seed vs A 1,708–1,850; summed 3,558 vs 10,635, ratio 0.33** (H1 required ≥ 2).
- Cumulative targets solved by round 8: A 4,075–4,206, B 3,950–4,039 (B lower in every seed, by 114–167). A2 4,122 /
  4,183, C 4,131 / 4,184. Targets at `L_true` ≥ 13 ever solved: 3–6 in every arm.
- C finds more past-the-other targets than A2 (762 / 542 vs 474 / 419, ratio 1.46) at ≈ 19 % of the actions, **on 2
  seeds**: a better expert per action, whose apprentice is nevertheless 8 lower on both seeds.
- B's selected expert proofs are slightly longer: on the same target and seed, median difference 0 lines, B longer on
  21–23 % of targets and shorter on **≤ 1.2 %**.

**Length of read-out proofs, lines and term size** (reviewer's own term-size count over the elaborated term; compare
between arms only). On theorems both arms solve, shortest proof each:

| pool | pair | theorems | Δ lines median / mean | longer / same / shorter | Δ term size median / mean |
|---|---|---|---|---|---|
| the 91 | B − A | 72 | 0 / +1.06 | 26 / 36 / 10 | 0 / +2.2 |
| rr 13–16 | B − A | 930 | 0 / +0.67 | 369 / 429 / 132 | 0 / +1.5 |
| the 91 | C − A2 | 22 | 0 / +0.36 | 7 / 7 / 8 | 0 / +0.2 |
| rr 13–16 | C − A2 | 272 | 0 / +0.40 | 78 / 152 / 42 | 0 / +0.4 |

Median shortest proof 18–22 lines on the 91 and 15–16 on rr 13–16; some are shorter than the ND-derived label (e.g. a
16-line proof on a label-17 theorem), as expected under Lean.

**Secondary, not pre-registered: T1 vs A.** `state-cap12`'s T1 recipe beats A by **63–96 on all 4 seeds** (A − T1:
−63, −96, −89, −72; mean −80, sd 15). **Which ingredient is responsible is not identified.** T1 differs from A in
selection (random vs shortest), in records per target (up to 4 vs 1), in the step filter, in the per-round evals and in
code version. The executor's heading "the selection rule matters far more than the expert" is **not supported**.

**Truncation.** 2 of 2.51 M read-out samples hit the step cap (T1 s0, T1 s2), and 566 hit `max_action` 512 (0.02 %
overall). 6 of 100 arm × seed × stratum cells exceed the policy's 0.1 % line, the worst B s3 rr15 at 0.82 % (one
theorem, `lp_transfer_869`, 209 of 256 samples truncated; it can move B − A in seed 3 by at most 1). The executor's
"step-cap hits: none" omitted the action cap. No conclusion changes.

**Checks.**
- Splits disjoint by renaming class (reviewer's own key): `train_k12` (154,382 classes) and `rl_targets` (4,495) share 0
  classes with the 91 and with rr600 13–16.
- **Every distinct counted read-out proof in all 40 read-out files is Lean-accepted: 46,956 / 46,956**, plus 1,150 /
  1,150 in a sampled pass that included 400 expert proofs. Negative controls: 0 / 300 recorded rejections, 0 / 56
  swapped proofs, 0 / 150 mismatched theorems, 0 / 5 `sorry` passed.
- The prefilter did every read-out rejection; Lean accepted **0 of 2,400** filter-rejected texts, so a false-rejection
  rate above ≈ 0.12 % is ruled out at 95 %. The same checker is the step filter inside the ladders.

**Expected vs outcome** (the reviewer's list; the executor's "my forecast held" covers only the direction).
- The brief's hypothesis: H1 (B's expert ≥ 2× A's past-the-frontier solves) → **0.33, falsified**; H2 (B − A > MDD)
  → −1.3, not supported. The falsifier fires.
- The executor's own forecast: H1 fails — met in direction, but the numeric range **0.7–1.5 missed** (0.33). B spends
  ≈ 60 % of the budget → **23 %, missed**. |B − A| < MDD, met. C ≈ A within the MDD, met (C − A2 −8, −8). B's proofs
  of the same targets no longer than A's shortest (median 0), met on the median.
- The forecast came from a second CPU pilot (5–6 / 12 at k 32) that is not in `log.md`; the pilot logged at 02:10
  (best-first 0 / 12 at k 8, "under investigation") is a different one.

## Interpretation

At a per-target budget equal to sampling's, best-first search as the EI expert stops early, finds about a third as many
new targets as sampling, and the apprentice it trains ties sampling's on the long pool. So **B, spending about a quarter
of A's actions, ties A's apprentice.** It is a tie at matched budget, not at matched compute. The run cannot separate "search
is no better a teacher" from "this search stops at the first proof". Truncate-and-resume is a better expert per action
on 2 seeds but does not give a better apprentice. The large difference in this run is between recipes (T1 vs A), not
between experts, and its cause is not identified.

## Limitations and next step

- B was matched on budget, not on spend (≈ 23 % of A's actions, ≈ 35 % of A's sampling GPU-seconds).
- C and A2 have 2 seeds each; the same-checkpoint spread rests on 2 pairs.
- T1 comes from another run with several differences from A; the T1 − A gap is not attributable to one ingredient.
- 6 read-out cells exceed the 0.1 % truncation line (policy), none by more than one theorem.
- The code (`state_search.py`, the ladder flags) is only on the fork branch `dan_search-expert`.

Next measurements (reviewer):
1. **A at B's actual spend** (sampling with k ≈ 8, same selection, 6 seeds). If A's apprentice drops, B is the cheaper
   expert for the same apprentice.
2. **B that keeps searching after the first proof** until it has spent its budget.
3. **A within-run A′ arm** on the same 6 seeds and code: up to 4 random successes per target, and separately 1 random
   vs 1 shortest. This splits T1's +80 into selection and volume.
