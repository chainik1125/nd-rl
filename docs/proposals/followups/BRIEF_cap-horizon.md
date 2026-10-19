# Run brief: cap-horizon — is the proof-length horizon a property of the training cap?

Run id `cap-horizon`. Repository `~/work/cap-horizon` (branch `dan_cap-horizon`, a worktree from
`origin/dan_ds-composition`, which already carries the `dsc_*` assembler and the cap-8 arm's
tooling). Role: executor. `AGENT_POLICY.md` governs; the designs below are suggestions and the
question is what matters. Pod budget **$22**, ceiling **44 pod-hours**, hard stop 30 h.

Read first: `~/nd-rl/docs/proposals/2026-09-24-length-horizon-and-noise-floor.md` (proposal 11,
this run's motivation), the `ds-composition` summary in `~/nd-rl/experiment-summaries/` (its A3 arm
is one of your controls and its confound is what you are breaking), `gen.py` (the `--cap` path and
how `prune` interacts with it), `dsc_assemble.py`, `data/ladder/POOLS.md`.

Sibling run `noise-floor` runs at the same time on this host with its own pods, budget and bucket
directory. It is measuring the error bar that your claims will be read against — check its
`numbers.md` before you call any difference of yours real, and say so if it is not yet available.

## Question

Holding the generator, the `lean_seq` rendering, the 3.2 M-parameter from-scratch model, the
6,000-step schedule and the depth-3 f = 0 filter fixed, does raising the **training cap** move the
proof-length horizon — the maximum length of accepted proof the model will write, the transfer
`L*` after ladder rung T1, and the frozen `L*` at equal attempts — and does it keep moving it, or
does it plateau?

## Why it matters

Eleven dataset-style variants across proposal 10 failed to move RL readiness. The one manipulation
that moved it was the cap: on `targets_reductio_req`, every cap-6 arm wrote **zero** accepted
proofs of ≥ 8 lines on both seeds at 2,000 attempts per target, and cap 8 wrote 29 / 33 and lifted
the frozen ladder `L*` from 9 to 11. Separately, `L_true` ≥ 13 was solved **0 times in all eleven
ladder runs** of `ds-generator`, T1 and frozen alike. So either the cap sets the horizon — in
which case cap 10 and cap 12 should walk it up, and the SPAR question's answer is that RL is not
creating length capability but reading out what the data's cap permits — or `L*` plateaus and the
wall is the model or the EI procedure, which redirects the project. Both answers are worth more
than another composition arm.

A3 changed two things at once (it added 7–8-line proofs **and** removed 8,857 proofs per ≤ 6-line
bin to hold the total at 155,000). Arm **K8add** separates them.

## Arms

Reuse, do not retrain, the two arms that already exist — pull their Stage-1 checkpoints and ladder
outputs from `hf://buckets/dan-pandori/nd-rl/ds-composition/`:

- **K6** = `ds-composition`'s C0 (= lean-format's a1 set), flat 31,000 per pruned length 2–6, cap 6,
  155,000 records. Checkpoints `ckpts/lf/stage1_a1_seq_s{0,1}.pt`.
- **K8flat** = `ds-composition`'s A3, flat over lengths 2–8 at 155,000 total, cap 8.

New sets (two Stage-1 seeds each; same generator settings as the control, only `--cap` and the
assembly histogram change; depth-3 excluded in written **and** pruned form, as every arm in
proposal 10):

- **K8add** — the control's **exact** 31,000 per length 2–6 (byte-identical bins, not a re-draw),
  **plus** 31,000 each at lengths 7 and 8. 217,000 records, cap 8. Against K8flat this isolates
  "long proofs added" from "short proofs removed"; against K6 it isolates "long proofs added" with
  the short bins held fixed. Note it is the only arm whose set size differs — say so in every
  table, and report the K8add-vs-K8flat comparison as the confound break it is, not as a set-size
  result.
- **K10** — flat over lengths 2–10 at 155,000 total (≈ 17,222 per length), cap 10.
- **K12** — flat over lengths 2–12 at 155,000 total (≈ 14,091 per length), cap 12.

If the generator's yield cannot fill a long bin inside the generation budget (2 pod-hours), **do
not** substitute from a shorter bin silently: report the achieved per-length counts, fill the
deficit proportionally from the bins that did fill, and disclose the fraction per bin in
`data/kh/README.md` — `ds-generator` had to do exactly this for G2 and the disclosure is what made
its result readable. If a cap-12 set cannot be filled at all, say so and run cap 10 with both
seeds instead of cap 12 with one; that is a better use of the budget than a set you cannot
describe.

## Measurements (per arm × seed unless stated)

Drop the depth-3 dial entirely. Three runs now agree its round-4 endpoint is base-rate-insensitive
(0.38–0.44, 0.396–0.436, 0.387–0.420 across frozen rates 0.032–0.322); another cell buys nothing.
Spend the budget on the horizon instead.

1. **Held-out greedy** on `data/p2/heldout.jsonl` (5,000, k = 1, T = 0). Note prominently that this
   pool is the **cap-6 distribution**, so every arm above cap 6 is measured out of distribution on
   it; it is a comparability check, not this run's headline.
2. **Coverage pass@2,000** (T 0.8) on `targets_reductio_req` (300) — the run's sharp quantity.
   Report, both seeds: targets solved; **the maximum pruned line count of any accepted proof**;
   the count of accepted proofs at ≥ 8, ≥ 10 and ≥ 12 lines; and the per-stratum breakdown.
   Also run `r3_1/depth3_req` (300) for continuity with proposal 10.
3. **Ladder T1 and frozen at equal attempts**, 8 rounds × k 32, read out on
   `data/ladder/transfer.jsonl` (2,285) labelled by true minimal length: solved and `L*`.
   **Seed 0 only** for the three new arms (budget); K6's and K8flat's ladders come from the bucket.
   Report the frozen number beside every T1 number — `AGENT_POLICY.md` requires a base-model
   reachability figure with every "RL solved X", and `ds-rendering` was pulled up for omitting it.
4. **Term size beside every line count** (formula nodes over the pruned proof). Proposal 10 asked
   for this and no run has reported it; `ds-composition`'s reviewer found the training-set mean
   term size tracked A4's held-out damage better than the length histogram did.

Batch sizes **fixed across every arm** (the `efficiency` run's caveat: a batch change reshuffles
accepted proofs about as much as an RNG re-draw). Say which batch. Acceptance is **Lean ∧
`nd_verify`** everywhere, `nd2lean.py` unmodified — the looseness fix is queued for after this run
precisely so your arms stay comparable to proposal 10's.

## Pre-registered expectations (in `preregistration/cap-horizon.md` before the first pod)

State these as numbers before you run anything, and score them honestly afterwards including the
misses. Proposal 10's runs missed several of their own control bands by factors of 2–7; a missed
band that is reported is worth more than a band that was never written down.

| quantity | K6 (known) | K8flat (known) | K8add | K10 | K12 |
|---|---|---|---|---|---|
| max accepted reductio length | 7 | 10 | 9–10 | 11–13 | 13–15 |
| accepted reductio proofs ≥ 8 lines | 0 / 0 | 29 / 33 | 20–45 | 30–70 | 30–70 |
| `redreq` solved (of 300) | 28 / 26 | 54 / 72 | 45–80 | 60–110 | 60–110 |
| frozen ladder `L*` | 9 | 11 | 10–11 | 11–13 | 12–14 |
| T1 transfer `L*` | 12 | 12 | 12–13 | 13–15 | 14–16 |
| held-out greedy (cap-6 pool) | 0.909 / 0.896 | 0.951 / 0.954 | ≥ K8flat | −3 to +3 pp of K8flat | −5 to +3 pp of K8flat |

**Falsifiers, both to be checked explicitly.**

- **The cap-sets-the-horizon account is dead** if K10 **and** K12 both leave T1 `L*` ≤ 12 and
  frozen `L*` ≤ 11 — i.e. no advance over cap 8 — while their training sets demonstrably contain
  proofs at those lengths. Then the wall is the model or the EI procedure, not the data, and that
  is the headline.
- **The "A3 removed short proofs" account is dead** if K8add and K8flat agree within the
  `noise-floor` run's measured error bar on `redreq` solved and frozen ladder solved. Then adding
  long proofs is the whole effect and set composition below the cap is irrelevant, consistent with
  everything else proposal 10 found.

Also pre-register: what you expect the **relationship** to be. If `L*` ≈ cap + c, say what you
predict c is and score it. A monotone walk with a stable offset is a much stronger result than
three unordered points, and it is the form that would let the project predict the cap needed for
any target length.

## Necessity of the pools

`targets_reductio_req.jsonl` — reviewer-certified (run 5), every target needs a derived `DN`,
strata by `min_lines_ub`. `data/r3_1/depth3_req.jsonl` — required@8 (round3-run1). Ladder pools
byte-identical to ladder-A's (`e0524d0`, `69233bc`) — verify with `git hash-object` and say so.
**Check split disjointness by renaming class** between each new training set and all evaluation
pools, and report the numbers; `ds-rendering`'s reviewer found an inherited 7-theorem overlap
between `data/transfer.jsonl` and the training set that every earlier run using that pool also
has, so do not assume.

## Design (suggestion)

1. Worktree `~/work/cap-horizon`, data hard-linked from `~/nd-takehome/data`; write new files,
   never edit shared ones in place. Pull K6 and K8flat checkpoints, ladder outputs and round files
   from `hf://buckets/dan-pandori/nd-rl/ds-composition/`. **Disk: this host has ~54 GB free and the
   sibling run is using it too — check `df -h` between stages and upload to the bucket as you go
   (`hf buckets sync`), not only at the end.** A host cleanup destroyed unuploaded artifacts once.
2. Generation on the first GPU pod's cores or a CPU pod (`make_coverage_sets.py gen` with workers;
   the ladder's long pools took ≈ 15 min at 32 workers × 40,000 tries). Shape, overlap and
   per-length tables per set, committed with the pre-registration — gate-0 material.
3. Pods: three, one per new arm, cheapest 24 GB+ card in stock (3090 $0.50/h preferred; A40 $0.49;
   A6000 $0.53; 4090 $0.74 — record the class **and** the real billed rate, `podbudget`'s dollar
   column has no A6000 or 4090 case and silently falls back to $0.50/h, which understated
   `ds-generator`'s resume by $0.18 and its session 1 by 14×). Stage-1 both seeds → held-out →
   coverage → ladder seed 0. Start ladders as soon as Stage-1 lands; they dominate.
4. **Register the pod budget before the first pod**: `podbudget cap-horizon --set 44 22`. Check for
   `~/runs/cap-horizon/BUDGET_WARNING` between stages. You may extend within the declared budget
   with `podbudget cap-horizon --extend <hours> "reason"`; if refused, ask in `QUESTIONS.md` and
   proceed on a stated default.
5. Budget arithmetic: generation ≈ 2 pod-hours; 6 Stage-1 ≈ 6; held-out + coverage ≈ 4; three
   ladders (T1 + frozen) ≈ 12 ≈ **24 pod-hours ≈ $13**, against 44 h / $22. Spend the headroom, in
   this order, on: K12 ladder seed 1; then K10 ladder seed 1; then a K14 arm if K12 has not
   plateaued. Drop order if the budget binds: K12's ladder, then K12 entirely, then K8add's ladder.

## Deliverables

`preregistration/cap-horizon.md` (before the first pod; gate 0); `run_cap_horizon.md` (≤ 400 words
+ two figures: max accepted proof length and `L*` against cap, with the frozen curve beside the T1
curve; the readiness panel per arm); `numbers.md` § cap-horizon (every entry names its source
file, **and every table names the model it was measured on** — checkpoint, parameter count,
format, from-scratch, training set); `log.md` dated; `artifacts/kh/summary.json` one row per arm ×
seed; `data/kh/README.md` with per-length achieved counts, fill disclosures, shape and overlap
tables; `STATUS.md` line `CAP-HORIZON DONE <UTC>`; bucket
`hf://buckets/dan-pandori/nd-rl/cap-horizon/{ckpts,artifacts,data}`; pods deleted;
`touch ~/runs/cap-horizon/executor.done`. Questions for Dan in `QUESTIONS.md` with the default you
will follow.
