---
written_on: 2026-09-24
written_by: agent:claude
---

# cap-horizon: is the proof-length horizon a property of the training cap? (proposal 11, run 1)

Status: completed 2026-09-24 22:50 UTC (autonomous executor) and independently reviewed
2026-09-24 23:00–23:27 UTC. Review verdict: **no hard-constraint violation, nothing quarantined**.
The reviewer re-derived every quantity with its own tooling (own formula parser, proof-line
parser, dependency pruner, start-index normaliser, box-depth counter, renaming-class key, term-size
counter; none of the executor's analysis code imported) and reports **no number in the write-up it
could not reproduce**; its findings are seven rewordings and one undisclosed confound, not
arithmetic. Numbers and wording below are the reviewer's, including the reviewer's own words where
they differ from the executor's. Run id `cap-horizon`.

Review:
[`review_cap-horizon.md`](https://github.com/dan-pandori/nd-takehome/blob/2e8b191/review_cap-horizon.md)
(commits `64faf97` phase-1 blind recount, `2e8b191` phase-2 compare and verdict; reviewer code and
outputs under
[`review/cap-horizon-recount/`](https://github.com/dan-pandori/nd-takehome/tree/2e8b191/review/cap-horizon-recount)).
Executor write-up:
[`run_cap_horizon.md`](https://github.com/dan-pandori/nd-takehome/blob/2e8b191/run_cap_horizon.md),
numbers `numbers.md` § cap-horizon (K0–K9), log `log.md` § cap-horizon, open questions
`QUESTIONS.md`, all on branch `dan_cap-horizon` of `dan-pandori/nd-takehome` (**read the review
first**: the executor's files were not edited after the review, so they still carry the wording the
verdict asks to change). Pre-registration:
[`preregistration/cap-horizon.md`](https://github.com/dan-pandori/nd-takehome/blob/2e8b191/preregistration/cap-horizon.md)
(`2091f78`). Proposal:
[`docs/proposals/2026-09-24-length-horizon-and-noise-floor.md`](../../docs/proposals/2026-09-24-length-horizon-and-noise-floor.md)
(run 1 of two; run 2 is the sibling `noise-floor`, still executing at review time).

Question: holding the generator, the `lean_seq` rendering, the 3.2 M-parameter from-scratch model
(4 layers, d 256, 8 heads, 3,214,336 parameters), the 6,000-step Stage-1 schedule and the depth-3
f = 0 filter fixed, does raising the **training cap** move the proof-length horizon — the maximum
pruned length of accepted proof the model writes, the transfer `L*` after ladder rung T1, and the
frozen `L*` at equal attempts — and does it keep moving, or plateau? **Cap 6 is the take-home's
constraint, not this project's; every arm above cap 6 is labelled "cap N" and none is proposed for
adoption as a take-home submission.**

## Configuration

- **Arms.** Two inherited, not retrained (checkpoints, coverage and ladder pulled from the
  `lean-format` / `ds-composition` buckets): **K6** (cap 6, 155,000 records, = `ds-composition` C0)
  and **K8flat** (cap 8, 155,000, = `ds-composition` A3). Four new, two Stage-1 seeds each:
  **K8add** (cap 8, **217,000** = K6's exact 155,000 plus 31,000 fresh records each at lengths 7
  and 8 — the only arm whose set size differs, reported as the confound break it is, not a
  set-size result), **K10**, **K12**, **K14** (155,000 each, flat over lengths 2–cap).
- **Set construction, reviewer-verified.** Every new arm's ≤ 6-line bins are a byte-identical
  subset of K6's own 155,000 records (K8add takes all of them; K10/K12/K14 a uniform per-bin
  subsample) — "short proofs removed" is pure subsetting, no re-draw noise below the cap. Bins at
  length ≥ 7 are freshly generated with the control's own generator settings, deduplicated by
  renaming class, with depth-3 excluded in pruned **and** written form and all eleven evaluation
  pools plus K6's own classes excluded. **0 records over cap, 0 overlap with the eleven evaluation
  pools, in every arm** (reviewer's own re-parse of all 682,000 new records). **Disclosed
  difference from K8flat:** A3's 7–8-line bins came from a pool no longer on this host; K8add's are
  a *fresh draw* from the same generator settings, not the same records — exactly the manipulation
  the sibling `noise-floor` run measures.
- **Undisclosed until the review: K8add's exposure confound.** At a fixed 6,000 steps × batch 128 =
  768,000 samples, K8add (217,000 records) sees **3.54 epochs** against **4.95 epochs** for every
  155,000-record arm — 28 % fewer passes over the data. "K8add is better at nothing despite 40 %
  more data" is therefore also "despite 28 % fewer passes over it", a live alternative explanation
  for K8add seed 1 (the worst cell in the run).
- **Deviation from the brief, pre-registered with its reason.** The brief planned three new arms
  plus seed-1 ladders for K10/K12; this run adds **K14 as a fourth arm from the start** instead of
  seed-1 ladders, because the question is monotone-walk-vs-plateau (a fourth cap point is needed to
  distinguish a plateau from "K12 is simply the last point measured"), and the sibling `noise-floor`
  run measures the error bar seed-1 ladders would have shrunk, at a better n (4 × 2).
- **Two measurement ceilings, pre-registered and reported with every affected number.** `L*` is
  **hard-censored at 14** by `data/ladder/transfer.jsonl` (only 23 theorems at `L_true` ≥ 13, 10 at
  ≥ 14), truncating the brief's own expectation bands for K12/K14. `max_new` = 400 could clip long
  proofs; a pre-registered diagnostic at `max_new` 768 returned **byte-identical accepted-proof
  sets** on all four new arms and K8flat (the reviewer's own recount agrees) — **the frontier is
  not truncation-bound**.
- **Model and schedule.** 4 layers, d 256, 8 heads, **3,214,336 parameters**, from scratch,
  `lean_seq`, Stage-1 6,000 steps, batch 128, `--cap {8,10,12,14}`. Acceptance is **Lean 4.34 ∧
  `nd_verify`** everywhere (`nd2lean.py --check` unmodified, including the owner-authored `BOTE`
  fix). Batch sizes and decode settings fixed across every arm and equal to `ds-composition`'s
  (held-out 512, ladder 512, coverage 1024, `max_new` 400/512) so the inherited K6/K8flat numbers
  stay comparable — reviewer confirms every inherited cell reproduces exactly under this run's own
  tooling.
- **Gate 0.** Pre-registration `2091f78` committed 2026-09-24T16:07:35Z; first pod ≈ 16:25–16:26Z
  (**≈ 19 min before**, ledger- and log-derived).
- **Pods and cost.** Four **A40 SECURE, $0.49/h** pods (`kh-k8add`, `kh-k10`, `kh-k12`, `kh-k14`),
  created 16:25–16:27 UTC, deleted 22:50 UTC: **25.55 pod-hours, $12.52** of the $22 budget / 44 h
  ceiling. (A stock-probing script bug briefly created and deleted three extra pods — two L40S, one
  community RTX 4090 — never registered to the run and costing ≈ $0.08 more, disclosed by the
  executor and not in `podbudget`'s total.) All four pods deleted; only the sibling `noise-floor`
  run's pods were alive at review time.
- **Checker of record, reviewer's own re-run.** `nd_verify` re-run on **433,223** counted records
  (all four new arms plus the two inherited): **0 rejects**. Unmodified `nd2lean.py --check` on
  **787** counted proofs (≥ 125 per arm, weighted toward the longest and the `L_true` ≥ 12 ladder
  theorems): **0 nd_ok & lean_rej, 0 nd_rej & lean_ok**, clean term-allowlist and `#print axioms`
  checks on all 787 (only `propext`, `Classical.choice`, `Quot.sound`; no `sorry`). Across
  4,479,047 in-loop both-accepted gate checks over eleven `gate_*.jsonl` logs: **456 Lean-only
  acceptances, all `nd_rej & lean_ok` (the safe direction, never the reverse)**, reproduced by the
  reviewer's own re-run of `nd_verify` on all 456 and attributed to the known `nd2lean` looseness
  queued for a fix after this run; **no counted proof anywhere in this run rests on it**.
- **Artifacts.** `hf://buckets/dan-pandori/nd-rl/cap-horizon/{ckpts,artifacts,data}` (present,
  reviewer-verified).

## Observations

All values are the reviewer's independent recount; every cell it compared equals the executor's.

### What stands

- **The headline is supported and the right way round.** The cap sets **how long a proof the model
  writes**, not **how hard a target it can crack**. Max accepted pruned length on
  `targets_reductio_req` fits **0.91 × cap + 1.39 (R² = 0.944)** over 12 arm × seed cells with **no
  plateau through cap 14** (accepted ≥ 12-line proofs: 0 at caps 6/8/10, then 7–23 at cap 12,
  7–19 at cap 14). Solved-by-*required*-length plateaus at cap 8: stratum 9 is solved at most 2 of
  82 times (only K12), **stratum 10 is unsolved in all 12 cells**, and **T1 transfer `L*` is 12 on
  all six arms, cap 6 through cap 14** — T1 saturates at cap 6 already and cannot move for any arm.
- **The reconciliation is the strongest single finding.** None of the accepted proofs ≥ 10 lines —
  53 at seed 0, **114 over both seeds** (106 distinct proofs, 41 targets; the write-up's "53" is a
  seed-0-only figure) — proves a target needing more than 8 lines. The extra length is padding:
  K14's longest `redreq` proof is 14 lines for a 7-line target, padding via `Or.inl` into an
  `Or.elim` that re-derives the same conjunction in both branches. On this pool, **raising the cap
  elicits longer writing, not new capability**.
- **The checking is the best the reviewer has seen in this project**: every counted proof of every
  new arm went through unmodified `nd2lean.py` + Lean and `nd_verify`; the independent re-run
  agrees on all 433,223 `nd_verify` verdicts and all 787 Lean verdicts; the 456 Lean-only
  acceptances are disclosed, one-directional and touch no counted proof; the pre-registered
  `max_new` 768 diagnostic was invoked and answered the question it was written for.
- **Budget, gate 0, pods, bucket and the run's own disclosed incidents are all on the record**: the
  ≈ $0.08 of mis-created pods, a `record_<arm>` bug that initially checked round 5–6 instead of
  round 8 (caught and re-run before the deliverables were written), and a `git add -A` incident
  that staged ≈ 1.5 GB of re-derivable artifacts and was reset before it reached the remote.

### Headline tables (reviewer's values)

Coverage pass@2,000 (T 0.8, seed 0) on `targets_reductio_req` (300), acceptance re-derived as
`nd_verify` ∧ Lean, lengths from the reviewer's own pruner:

| arm | solved /300 (s0 / s1) | max **pruned** accepted length | accepted ≥ 8 / ≥ 10 / ≥ 12 (pruned) |
|---|---|---|---|
| K6 (cap 6) | 28 / 26 | 7 / 7 | 0/0/0 · 0/0/0 |
| K8flat (cap 8) | 54 / 72 | 10 / 8 | 28/2/0 · 32/0/0 |
| K8add (cap 8, 217k) | 63 / **24** | 8 / 8 | 21/0/0 · 1/0/0 |
| K10 (cap 10) | 70 / 87 | 10 / 11 | 33/2/0 · 43/4/0 |
| K12 (cap 12) | 73 / 86 | **13** / 12 | 50/16/7 · 77/39/23 |
| K14 (cap 14) | 84 / 54 | **14** / **14** | 78/33/19 · 36/18/7 |

**No arm solves a single target in the 10-stratum, and only K12 solves any in the 9-stratum** (2 on
s0, 1 on s1); all growth is in the 7- and 8-strata.

Ladder, rung T1 and frozen at equal attempts, 8 rounds × k 32, seed 0, `L*` = max L with ≥ 5 solved
at `L_true ≥ L` (the base-model reachability figure is reported beside every T1 number, per policy):

| arm | frozen solved /2285 | frozen `L*` | T1 solved /2285 | T1 `L*` | max pruned accepted (frozen → T1) |
|---|---:|---:|---:|---:|---|
| K6 | 158 | 9 | 856 | 12 | 10 → 12 |
| K8flat | 976 | 11 | 1,438 | 12 | 13 → 15 |
| K8add | 857 | 10 | 1,428 | 12 | 14 → 16 |
| K10 | 743 | 11 | 1,554 | 12 | 15 → 16 |
| K12 | 1,088 | 12 | 1,685 | 12 | 17 → 18 |
| K14 | 1,140 | 12 | 1,655 | 12 | 19 → 20 |

**Frozen `L*` does walk with the cap and compresses** (9, 10/11, 11, 12, 12 by cap — Stage-1 seed 0
only) but has **no stable offset**: the achieved offset over the naive base-model reachability is
+3 at cap 6, +3/+2 at cap 8, +1 at cap 10, 0 at cap 12, −2 at cap 14. **The uncensored frontier — the
longest accepted proof written — keeps moving with no plateau** on both frozen and T1 models: models
write accepted proofs *above* their own training cap everywhere (K14: 20 pruned lines on the
ladder pool, 6 above its cap).

Pre-registered expectations, scored honestly including the misses (seed 0 / seed 1 where two
exist):

| quantity | offset predicted | K8add | K10 | K12 | K14 |
|---|---|---|---|---|---|
| max accepted `redreq` pruned length | `cap + 2` → 10/12/14/16 | 8 **miss (low)** | 10/11 **miss (low)** | 13/12 **s0 hit, s1 low** | 14/14 **hit** |
| frozen ladder `L*` | `min(cap+3, 14)` → 11/13/14/14 | 10 **hit** | 11 **miss** | 12 **miss** | 12 **miss** |
| T1 transfer `L*` | 12–13/13–14/14/14 | 12 **hit** | 12 **miss** | 12 **miss** | 12 **miss** |

Both pre-registered offsets are **wrong and wrong in the same direction** — too generous, and the
gap widens with the cap. Least-squares fit for the uncensored frontier: **0.908 × cap + 1.39,
R² = 0.944**, measured mean offset **+0.50** against the predicted **+2**.

**Falsifiers.** Falsifier 1 (the cap account is dead) is **not triggered** — it needed K10 *and*
K12 to fail on all three of T1 `L*` ≤ 12, frozen `L*` ≤ 11 and max `redreq` length ≤ 10; K12 fails
two of the three (frozen `L*` = 12, max length 13). **Falsifier 2 (composition below the cap is
irrelevant) is reworded from the write-up's "triggered" to "not decidable on the pre-registered
criterion"**: the criterion requires agreement within `noise-floor`'s measured error bar, which does
not exist yet for coverage or ladder counts. What the run's own data show: K8add and K8flat agree
within their own two-seed spread on `redreq` (63/24 vs 54/72) and frozen ladder solved (857 vs 976),
and K8add is better at nothing — but the within-arm seed spread of K8add on `redreq` (63 vs 24) is
larger than any between-arm difference at cap 8, so this is not resolvable from this run's own data
at n = 2 either, independent of the confound above.

### Corrections to `numbers.md` and the write-up (applied here, not yet in the executor's files)

1. `numbers.md` K6 (`d3req` table): K14's max accepted pruned length is **16, not 14**, on both
   seeds (K10 s1 14, K12 s1 13, K14 s1 16 — the figure already shows 16, only the table row is
   wrong); the cap-8 range on that pool is **11–13, not 12–13**.
2. `numbers.md` K3's "longest correct held-out proof" column is actually the **≥ 5-proof frontier**;
   the true longest accepted held-out proofs are 6 / 10 / 9 / 11 / 13 / 14, and K8flat's frontier is
   seed-dependent (9 on s0, 8 on s1, where the table shows one number).
3. "Cap 8 adds the ≥ 8 stratum (0 → 20–30 of 133)" should read **0 → 2–30**, or name the cells:
   K8flat 24/30, **K8add 20/2** — K8add seed 1 is the single worst cell in the run and the range
   hides it.
4. "Of 53 accepted proofs ≥ 10 lines" in `run_cap_horizon.md` is the **seed-0** count; the
   both-seed number, **114**, is the stronger and equally supported statement.
5. "386,467 counted proofs" (checker paragraph) covers the **four new arms only**; the reviewer's
   own broader re-run (433,223, including the two inherited arms) still finds 0 disagreements.
6. Term size should be summed over the same pruned-proof definition the line count uses (dependency
   closure only, not closure-plus-every-`PR`-line); the current mismatch bites 3 of 786 compared
   proofs and moves a mean by ≤ 0.21 — not a wrong number, but inconsistent units side by side.
7. "Falsifier 2 triggered" → see falsifier wording above.

**Nothing in the run is unsupported** beyond these wording and scope corrections; none changes a
conclusion.

## Limitations and next step

- **K8add's exposure confound is undisclosed in the write-up** (3.54 vs 4.95 epochs at fixed step
  count) and is a live alternative explanation for K8add seed 1, the worst cell the run's falsifier
  2 rests on.
- **Falsifier 2 cannot be decided by this run alone**: no coverage/ladder error bar exists yet
  (the sibling `noise-floor` run was still executing at review time), and even on this run's own
  data the K8add-vs-K8flat comparison is smaller than K8add's own between-seed spread.
- **Held-out greedy rates are not independently re-derivable** here (no retained generations, no
  GPU on the review host) — checked for internal consistency only.
- **The per-arm fraction of samples hitting `max_new`** was not recorded by the gate files; the
  `max_new` 768 diagnostic is strong indirect evidence against truncation but not a direct measure.
- **Frozen `L*`'s walk (9 → 10/11 → 11 → 12 → 12) is Stage-1 seed 0 only per arm.**

Next step, reviewer's ranked list: (1) **K8add at matched exposure** — retrain at 8,400 steps, or
subsample K8add's own design to 155,000 records, and re-measure `redreq` on both seeds (≈ 1
pod-hour; the only way to close falsifier 2 honestly); (2) **`noise-floor`'s coverage and ladder
floors** — until they land, no "K10 vs K8flat"-style comparison in this theme has an error bar; (3)
**model size at fixed cap 12 against the `min_lines_ub` strata** (the executor's own open
question) — this run establishes stratum 9–10 is not a data-cap problem, so size separates capacity
from the EI procedure as the remaining candidates; (4) **a second ladder seed at K10 and K12** — the
frozen-`L*` walk is the run's cleanest monotone series and is n = 1 per arm; (5) **only then** a
denser-tail transfer pool (breaks comparability with every prior ladder number; not needed for 1–4).
Three questions for Dan are in `QUESTIONS.md` (2026-09-24: whether to launch the model-size follow-up,
whether to rebuild the ladder transfer pool with a denser tail, and whether trading K14 for seed-1
ladders was the right call), each with a stated default followed.
