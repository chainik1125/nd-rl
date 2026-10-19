# Run brief: noise-floor — what difference can this project actually resolve?

Run id `noise-floor`. Repository `~/work/noise-floor` (branch `dan_noise-floor`, a worktree from
`origin/dan_ds-generator`, which carries the generator flags and the `dsg_*` shape tooling). Role:
executor. `AGENT_POLICY.md` governs; the design below is a suggestion and the question is what
matters. Pod budget **$18**, ceiling **36 pod-hours**, hard stop 30 h.

Read first: `~/nd-rl/docs/proposals/2026-09-24-length-horizon-and-noise-floor.md` (proposal 11),
the `ds-generator` summary in `~/nd-rl/experiment-summaries/` (its §"the measurement noise is this
run's real headline constraint" is the observation this run turns into a number), and the
`ds-rendering` summary's Finding R-2 and its bimodality note.

Sibling run `cap-horizon` runs at the same time on this host with its own pods and budget. Your
output is the yardstick its claims will be read against, so **publish your interim numbers to
`numbers.md` as each stage lands** rather than holding them to the end — it is running now and
will read them.

## Question

Four raw pools drawn independently from the **same generator settings as the control**, assembled
to the **same composition**, trained with the **same schedule**, differing only in RNG draw and
Stage-1 seed. Every difference between them is noise by construction. How big is it, per reported
quantity?

## Why it matters

No run in this project has ever measured its own error bar, and every run has reported two-seed
differences as findings. `ds-generator` found this out by accident: its G1 arm was a **designed
null** — the control's distribution by construction, shape table matching to sampling noise — and
it differed from the control by **+49 % on the frozen ladder** (170 vs 114 on the same seed) and
**1.9× on required@8 coverage**. That is larger than most of the effects proposal 10 reported as
results. Its reviewer's conclusion: *"until a real error bar exists no dataset-style run at n = 2
can resolve anything smaller than ±50 %."* `ds-rendering` found the depth-3 held-out cell is not
merely noisy but **bimodal** — 20 of 26 models land above 0.44 or below 0.11, pooled within-arm sd
0.302 — so a mean ± sd is the wrong summary for it and the right estimand is the probability of
landing in the high mode.

Until this number exists, the project cannot tell a finding from a draw, and several standing
findings may be draws. This is the cheapest run available and the one that decides how to read
every other one.

## Design

**Four pools × two Stage-1 seeds = eight models.** Pools `P1`–`P4`: `gen.py` with the control's
exact settings (**no flags changed** — verify with a diff against the control's generation command
in `data/p2/README.md` or `lean-format`'s log, and paste the command you used), four different
generator RNG seeds. Assemble each to the control's exact composition: flat 31,000 per pruned
length 2–6, depth-3 excluded in written and pruned form, cap 6, 155,000 records. Stage-1 seeds 0
and 1 on each: `train.py --mode lean_seq --steps 6000 --bs 128 --cap 6 --seed {0,1}`.

Confirm the pools really are the same distribution before treating them as a null: per-length rule
shares, box-depth distribution, reductio and derived-`ORE` shares, mean premises, mean term size —
these should agree across P1–P4 to sampling noise, and if one does not, **say so and treat it as
an arm, not as a replicate**. This check is the premise of the whole run.

**Measurements, per pool × seed:**

1. Held-out greedy on `data/p2/heldout.jsonl` (5,000, k = 1, T = 0): overall **and per length bin**,
   plus the depth-3 slice (the 500 theorems whose reference proof is box depth ≥ 3) — that is
   `ds-rendering`'s bimodal cell and the one most in need of an error bar.
2. Coverage pass@2,000 (T 0.8) on `targets_reductio_req` (300), `r3_1/depth3_req` (300),
   `targets_depth3` (1,000).
3. **Frozen ladder only**, 8 rounds × k 32, read out on `data/ladder/transfer.jsonl`: solved and
   `L*`. No EI — the frozen ladder is the quantity with the largest observed spread and half the
   cost. If budget allows after everything else, add T1 on P1 and P2 only.

Batch sizes fixed across every cell; say which. Acceptance is **Lean ∧ `nd_verify`**,
`nd2lean.py` unmodified.

**The analysis is the deliverable.** For each quantity report: the eight values; the
**between-pool** and **between-seed** variance components (a two-way decomposition, pools × seeds,
or state plainly why you used something else); the pooled sd; and **the smallest difference an
n = 2 comparison has 80 % power to detect**. Where the distribution is bimodal rather than normal
— check each quantity, do not assume — report the proportion of models in the high mode with a
Wilson interval instead of a mean ± sd, per `ds-rendering`'s reviewer.

## Two n = 1 gaps to close while the pods are up

Both are cheap, both patch a specific hole proposal 10 left, and both use checkpoints already in
the bucket — no retraining:

- **`ds-generator`'s `la_frozen_g1_s0`** was never run (a disclosed budget drop), leaving its most
  informative noise-floor cell at one seed. Run it: checkpoint in
  `hf://buckets/dan-pandori/nd-rl/ds-generator/ckpts/`, ≈ 2 pod-hours. It is a fifth independent
  draw of the control's distribution and belongs in your variance estimate — label it as a
  different generator flag set (`--ore_steps 3 --ore_boxes`) that the run showed matches the
  control's shape table, not as a fifth `P` pool.
- **`ds-composition`'s ladder is entirely Stage-1 seed 0** for every arm, including the three
  headline claims about A1. Run **C0 and A1, ladder T1 + frozen, seed 1** — checkpoints in
  `hf://buckets/dan-pandori/nd-rl/ds-composition/ckpts/`, ≈ 4 pod-hours. The partial seed-1 records
  already favour A1 (equal-attempts frozen round 1: C0 56 vs A1 139; seed 0: 76 vs 117), so this
  either confirms a two-seed result or kills it.

## Pre-registered expectations (in `preregistration/noise-floor.md` before the first pod)

| quantity | expected pooled sd / spread across the 8 cells |
|---|---|
| held-out greedy, overall | sd ≤ 0.02; max − min ≤ 5 pp |
| held-out greedy, 6-line bin | sd 0.04–0.10 (C0's own two seeds differ by 10.3 pp) |
| held-out, depth-3 slice | **bimodal**; 5–8 of 8 in the high mode; sd ≥ 0.20 |
| `redreq` solved (of 300) | max / min ratio **1.5–3×** |
| `depth3_req` required@8 | max / min ratio **1.5–3×** |
| frozen ladder solved | max / min ratio **1.4–2.5×**; sd ≥ 25 % of the mean |
| frozen ladder `L*` | spread 9–11, i.e. **2 points of pure noise** |
| smallest n = 2 resolvable difference, frozen ladder | **± 40–60 %** |

**Falsifier.** If every quantity's spread across eight null cells is **below** these bands — in
particular if the frozen ladder's max/min is under 1.2× — then `ds-generator`'s G1-vs-C0 gap was
not noise after all, it was a real effect of the `--ore_steps 3 --ore_boxes` flags, and that
reopens the shape account this project has just declared dead. Say so loudly if it happens; it
would be the most surprising outcome available here and it inverts proposal 11's premise.

State also, before running: **which standing findings you expect this to invalidate.** Then check
them against the measured floor in the write-up, by name, with the original number beside the
floor. That retrospective table is the most useful thing this run can produce and it is the reason
it is being run before anything else.

## Design notes

1. Worktree `~/work/noise-floor`, data hard-linked from `~/nd-takehome/data`; write new files.
   **Check `df -h` between stages** — the sibling `cap-horizon` run shares this host's disk, and a
   cleanup once destroyed unuploaded artifacts. `hf buckets sync` after each stage.
2. Pods: two, cheapest 24 GB+ card in stock; record the class **and** the real billed rate
   (`podbudget`'s dollar column falls back to $0.50/h for A6000 and 4090 and is wrong for both).
   Four pools × 2 seeds is 8 Stage-1 runs ≈ 8 pod-hours; generation ≈ 2; held-out + coverage ≈ 5;
   8 frozen ladders ≈ 10; the two gap-closers ≈ 6 ≈ **31 pod-hours ≈ $16**, against 36 h / $18.
3. **Register the pod budget before the first pod**: `podbudget noise-floor --set 36 18`. Watch for
   `~/runs/noise-floor/BUDGET_WARNING`. Drop order if it binds: the `ds-composition` seed-1 ladder,
   then P4 entirely, then the `targets_depth3` coverage (the least informative pool).

## Deliverables

`preregistration/noise-floor.md` (before the first pod; gate 0); `run_noise_floor.md` (≤ 400 words
+ two figures: the eight cells per quantity as a strip plot with the control's published value
marked; the resolvable-difference table as a chart); **`NOISE_FLOOR.md` at the repo root** — a
one-page standing reference, per quantity, of the smallest difference worth reporting, written to
be quoted by future runs and by their reviewers; `numbers.md` § noise-floor (every entry names its
source file and the model it was measured on); `log.md` dated; `artifacts/nf/summary.json` one row
per pool × seed; `STATUS.md` line `NOISE-FLOOR DONE <UTC>`; bucket
`hf://buckets/dan-pandori/nd-rl/noise-floor/{ckpts,artifacts,data}`; pods deleted;
`touch ~/runs/noise-floor/executor.done`. Questions for Dan in `QUESTIONS.md` with the default you
will follow.

A note on framing for the write-up: a large measured floor is **not** a bad result and must not be
written as one. It is the number that makes every future claim in this project interpretable, and
it is worth more than another arm that cannot be read.
