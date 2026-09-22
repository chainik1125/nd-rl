# Run brief: ds-composition — the training set's composition (histogram, rule quotas, cap 8)

Run id `ds-composition`. Repository `~/work/ds-composition` (branch `dan_ds-composition`, a
worktree from `origin/dan_lean_format`, which has the proposal-8 code: `lean_tok.py`,
`lean_gate.py`, `ladder_ei.py`, `pod/lf/`). Role: executor. `AGENT_POLICY.md` governs; designs
are suggestions. Read first: `~/nd-rl/docs/proposals/2026-09-22-dataset-styles.md` (proposal 9;
its "Protocol shared by every arm" section is the protocol for this run),
`preregistration/lean-format.md`, `run_lean_format.md`, `review_lean-format.md`,
`~/nd-rl/experiment-summaries/2026-09-21-lean-format-training-format/README.md`. Pod ceiling
**$22**; hard stop 30 h. Sibling run `ds-generator` runs at the same time on this host with
its own pods, budget and bucket directory; `ds-rendering` starts when one of you finishes.

## Question

With the generator, the Lean rendering, the model, the schedule, the cap (6 on the ND record),
the set size (155,000) and the depth-3 f = 0 filter all held fixed, does the **composition** of
the pretraining set — the per-length histogram, per-rule quotas, or (as a yardstick, outside
the take-home rule) the cap itself — change the `lean_seq` model's held-out accuracy by length
and its RL readiness (base rates of depth-3 and required reductio at pass@2,000, EI − frozen
after 4 dial rounds, transfer `L*` after ladder T1)?

## Why it matters

The control's histogram is flat (31,000 per length 2–6) by a `--per_len` choice nobody tested;
the class-deduplicated pool is 10 / 12 / 13 / 20 / 46 % by length, so flattening under-samples
6-line proofs 2.3×, and the 6-line bin is the held-out deficit in every format (0.852). Every
pre-RL pattern hit in 72 token draws was a 7-line proof (cap + 1); "the cap sets the horizon"
is the standing finding, and cap 8 is the one lever known to move it. `ORE`, `ANDE` and `BOTE`
occur in 1–2 % of proofs; 17 of 19 textbook schemata (De Morgan, dilemmas, distribution —
the `ORE`-needing ones) stay at ≈ 0 under every RL technique, and Lean T1 arms solve the
textbook-dominated `L_true` = 7 bin *less* often than token arms (55–118 vs 139–140). A
composition arm that recovers a large fraction of cap 8's gain kills finding 1; a quota arm
that moves the dead schemata kills the "generator shape" account of finding 2.

## Arms (each one change to the control; two Stage-1 seeds each)

All sets are assembled from `data/p2/pool_cap6_recon.jsonl` (723,534 classes, the unchanged
generator; generate more with `make_coverage_sets.py gen` **without changing generator knobs**
if a quota or a length bin cannot be filled — say so) with `make_coverage_sets.py assemble` or
a small assembler of your own that reuses its exclusion and f = 0 logic. Depth-3 excluded
everywhere (pruned and written form).

- **C0 control.** `data/p2/train_depth3_f0_a1.jsonl` and the two Stage-1 checkpoints
  `hf://buckets/dan-pandori/nd-rl/lean-format/ckpts/lf/stage1_a1_seq_s{0,1}.pt` (do not
  retrain). Measure what is not on file: held-out by length, pass@2,000 on the three pools,
  ladder T1 + frozen; dial round-4 values from
  `lean-format/artifacts/lf/{ei,frozen}_d3_seq_s{0,1}/round_4.json`. Report the a1 set's
  overlap with the ladder and reductio pools (never checked).
- **A1 natural histogram.** Per-length shares 10 / 12 / 13 / 20 / 46 % (the pool's own;
  measure the exact shares on the full pool and use those).
- **A2 rule quotas.** Flat histogram; ≥ 10 % of proofs contain `ORE`, ≥ 8 % `ANDE1`/`ANDE2`,
  ≥ 5 % `BOTE` (a proof can count toward several); the remainder drawn as the control draws.
  Report the achieved shares and the shares of every other rule.
- **A3 cap-8 yardstick.** Lengths 2–8, flat (≈ 22,143 per length; `train.py --cap 8`).
  **Outside the take-home's cap-6 rule** — label every A3 number "cap 8" in every table;
  it is the calibration arm, not a candidate for adoption. Its held-out numbers are on the same
  ≤ 6-line held-out set; the dial's 7–8-line targets are in-distribution length for it, so its
  dial and base-rate numbers are not f = 0-at-length measurements (say so).
- **A4 cap-heavy dose.** Per-length shares 0 / 5 / 15 / 30 / 50 %. Drop first if short.

## Pre-registered expectations (write them in `preregistration/ds-composition.md` before the first pod; both seeds unless stated)

Control values to be measured here; on-file references: held-out 0.909 / 0.896; frozen depth-3
0.206 / 0.134 at 8 × 32; pass@16 7 / 8 / ≥ 9-line proofs 266 / 130 / 18 (full-set model).

| arm | held-out (overall; 6-bin; 2–3-bins) | depth-3 pass@2,000 (1,000 pool; req8) | reductio req pass@2,000 (7-line stratum of 52; ≥ 8-line) | dial EI − frozen at round 4 | ladder frozen solved, `L*`; T1 `L*` | textbook |
|---|---|---|---|---|---|---|
| C0 | measured; 6-bin ≈ 0.85 | 0.35–0.55; 0.05–0.25 | 5–25 targets; 0–5 | ≈ +0.20 to +0.30 | 220–320, 9–10; 10–11 | ≤ 2 schemata beyond contraposition / export at ≥ 5 |
| A1 | −1 to +1 pp; **+2 to +5 pp**; ≥ 0.98 | +5 to +15 pp; +5 to +15 pp | ×1–2; ≥ C0 | within ±0.05 of C0 (no larger) | **+15 to +40 %**, +0–1; 11–12 | unchanged |
| A2 | −1 to +0.5 pp; ±1 pp; ≥ 0.98 | within ×0.5–2 of C0 | within ×0.5–2 | within ±0.05 | ±10 %, unchanged; 11 | **≥ 2 new schemata at ≥ 5 T1 solves; `L_true` = 7 bin +20 to +60** |
| A3 (cap 8) | −1 to +1 pp; **+3 to +6 pp**; ≥ 0.98 | ≥ 0.6 (not f = 0-at-length) | ×2–5; ≥ 5 | not comparable (labelled) | **≥ 1,000**, 11–12; 12–13 | +1–3 schemata |
| A4 | −1 to +1 pp; ≥ A1's 6-bin; **2-bin ≥ 0.95** | ≥ A1 | ≥ A1 | within ±0.05 | ≥ A1 | unchanged |

**Falsifiers.** Finding 1 ("the cap sets the horizon") is dead if A1 or A4 reaches ≥ 70 % of
A3's gain in frozen ladder solves and matches its frozen `L*`. The histogram is *not* a lever if
A1's 6-bin is within ±1 pp of C0 *and* its frozen ladder solves are within ±10 %. Finding 2's
rule-mix version is dead if A2 moves ≥ 2 dead schemata to ≥ 5; it stands if 17 / 19 stay ≤ 2.
Finding 3 (RL amplifies what the base does) gets its first counter-example if any arm has a
lower base rate than C0 and a larger EI − frozen on both seeds.

## Design (suggestion)

1. Worktree is ready (`~/work/ds-composition`, data hard-linked from `~/nd-takehome/data`; do
   not edit shared data files in place — write new files). `git show
   origin/dan_round3-run1:data/r3_1/depth3_req.jsonl > data/r3_1/depth3_req.jsonl` (and
   `depth3_req_transfer.jsonl`) for the required@8 pool. Pull the control checkpoints and
   round-4 files from the bucket (`hf buckets` … `lean-format/`).
2. Build the four sets on the VPS (CPU; the assembler is fast; generation if needed on a CPU
   pod or the first GPU pod's cores). Shape table + overlap table + render check per set
   (proposal 9, protocol) — commit these with the pre-registration or right after; they are
   gate-0 material.
3. Pods: four RTX 3090 ($0.50/h) as lean-format did (`pod/lf/` harness; `LEAN_GATE_WORKERS=12`),
   one arm per pod; Stage-1 (two seeds) → held-out greedy → coverage on the three pools →
   dial EI + frozen (4 rounds) → ladder T1 + frozen (8 rounds). The ladder arms dominate
   (≈ 4 pod-hours per style); start them as soon as Stage-1 is done and run coverage / dial on
   the same pod under them. Control measurements on whichever pod is free first.
4. Reward and counting: Lean ∧ `nd_verify` (`lean_gate.py`, unchanged); `nd2lean.py --check`
   on every counted proof at the end; the BOTE fix is being done by `lean-seed2` on the other
   host — do not wait for it, do not modify `nd2lean.py` yourself; store literal texts.
5. Budget arithmetic: 4 styles × ≈ 7.5 pod-hours + control ≈ 4 pod-hours ≈ 34 pod-hours ≈ $17;
   ceiling $22. Stop at $20 spent: drop A4's ladder first, then A4, then A2's ladder.

## Necessity of the pools

`targets_reductio_req.jsonl`: reviewer-certified (run 5) — every target needs a derived `DN`;
strata by `min_lines_ub`. `data/r3_1/depth3_req.jsonl`: required@8 (restricted search fails at
bound 8, 0 timeouts; round3-run1). `targets_depth3.jsonl` (1,000): the dial's pool, kept for
comparability with every on-file dial number; it is pattern-optional for a minority (report
required@8 separately, so the readiness number is on a pool that requires the pattern).

## Deliverables

`preregistration/ds-composition.md` (committed before the first pod; Gate 0);
`run_ds_composition.md` (≤ 400 words + two figures: held-out by length per arm; the readiness
panel per arm with C0 and the cap-8 yardstick marked); `numbers.md` § ds-composition (every
entry names its file); `log.md` dated; `artifacts/dsc/summary.json` with one row per arm ×
seed; shape and overlap tables in `data/dsc/README.md`; `STATUS.md` line `DS-COMPOSITION DONE
<UTC>`; bucket `hf://buckets/dan-pandori/nd-rl/ds-composition/{ckpts,artifacts,data}`; pods
deleted; `touch ~/runs/ds-composition/executor.done`. Questions for Dan go to `QUESTIONS.md`
with the default you follow.
