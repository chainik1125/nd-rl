# Run brief: ds-generator — the generator's proof-shape distribution (ORE shape; goal-directed strict)

Run id `ds-generator`. Repository `~/work/ds-generator` (branch `dan_ds-generator`, a worktree
from `origin/dan_lean_format`). Role: executor. `AGENT_POLICY.md` governs; designs are
suggestions. Read first: `~/nd-rl/docs/proposals/2026-09-22-dataset-styles.md` (proposal 9; its
"Protocol shared by every arm" section is this run's protocol), `gen.py` (docstring, `reach`,
`step`, `step_local`, `open_box`, `sample_one`, `finish`), `make_coverage_sets.py gen`,
`data/ladder/POOLS.md` (how the strict long generator built the ladder pools),
`preregistration/lean-format.md`, `review_lean-format.md`. Pod ceiling **$20**; hard stop
30 h. Sibling run `ds-composition` runs at the same time on this host with its own pods,
budget and bucket directory.

## Question

With composition (flat 31,000 per length 2–6), rendering, model, schedule, cap 6, set size and
the depth-3 f = 0 filter held fixed, does changing the **shape distribution of the generated
proofs** — `ORE` branches that take steps and open boxes (G1), and on top of that fully
goal-directed, strict generation with no lazy premises (G2) — change held-out accuracy by
length and RL readiness (base rates of depth-3 and required reductio at pass@2,000, EI − frozen
after 4 dial rounds, transfer `L*` after ladder T1)?

## Why it matters

`STATE.md` attributes the absence of textbook transfer in two independent runs to "the
generator's theorem-shape distribution", and no run has changed that distribution: every
pretraining set to date comes from `gen.py` with the same knobs (cap-6 pools: `ORE` branches
0–1 step and no nested box; lazy one-step premises; 45 % goal mode with the `( Z > G )`
fallback at p 0.85). The ladder's transfer pool, by contrast, is the *strict long* generator's
output (no lazy fallback, no `F`, final rule an elimination or discharge). If the shape
distribution is the wall, G2's base model should solve far more of the ladder transfer pool
frozen and G1 should open the `ORE`-needing textbook schemata; if neither moves, the
"generator shape" account of the textbook wall is dead and the length horizon (run
`ds-composition`) is the remaining lever.

## Arms (two Stage-1 seeds each; every set 155,000, flat histogram, cap 6, depth-3 excluded)

- **C0 control.** As in `BRIEF_ds-composition.md`: the a1 set and its two bucket checkpoints;
  measure held-out by length, pass@2,000 on the three pools, ladder T1 + frozen; dial round-4
  from the lean-format round files. (Both runs measure the control; that is intended.)
- **G1 ORE shape.** `gen.py` with `ore_steps = 3` and `step_local` / `step_no_box` no longer
  clamping `max_depth` to the branch's depth (boxes allowed inside `ORE` branches, still under
  the global depth cap 3). Nothing else changed. Target for the set's shape table: `ORE` in
  ≥ 5 % of proofs (1.9 % today) and boxes inside `ORE` in ≥ 1 % (0 today) — if the natural
  yield is below that at cap 6, say so and report what it is; do **not** add quotas (that is
  `ds-composition`'s A2).
- **G2 goal-directed strict.** G1's knobs plus: 100 % goal mode, `strict = True` (no
  `( Z > G )` / `G` lazy fallback, no `F` in the sequent, conclusion not by an intro rule at
  the top level if that is what `finish` enforces in strict mode — read it and say what you
  used), `reach` budgets `choice([2,3,4])`, goal formula depth `choice([2,2,3])`. Yield of
  ≤ 6-line pruned proofs will be lower; generate until the flat histogram is filled or report
  the bin that cannot be filled and fill it from G1's pool (disclose the fraction).

Every set: shape table (length, box-depth, per-proof rule shares, boxes-inside-`ORE`, premise
count, lazy-premise count if you can label it, contradictory-premise share, mean tokens),
overlap table against all pools, render check (proposal 9, protocol). The generator is still
the generator: verified output only, no hand-written or LLM-written proofs; the knob changes
are disclosed in `run_ds_generator.md` and in the set's README.

## Pre-registered expectations (in `preregistration/ds-generator.md` before the first pod; both seeds)

| arm | held-out (overall; 6-bin) | depth-3 pass@2,000 (1,000 pool; req8) | reductio req pass@2,000 (7-line stratum; ≥ 8-line) | dial EI − frozen, round 4 | ladder frozen solved, `L*`; T1 `L*` | textbook per schema |
|---|---|---|---|---|---|---|
| C0 | measured (≈ 0.90; ≈ 0.85) | 0.35–0.55; 0.05–0.25 | 5–25; 0–5 | +0.20 to +0.30 | 220–320, 9–10; 10–11 | ≤ 2 schemata beyond contraposition / export at ≥ 5 |
| G1 | ±1 pp; ±1 pp | within ×0.5–2 of C0 | within ×0.5–2 | within ±0.05 | ±15 %, unchanged; 11 | **≥ 1 `ORE`-needing schema (dilemma, De Morgan, distribution families) at ≥ 5 T1 solves**; `L_true` = 7 bin +10 to +40 |
| G2 | **−2 to +1 pp** (the held-out is the old generator's distribution); −1 to +2 pp | +5 to +15 pp; +5 to +15 pp | ×1–3; ≥ C0 | within ±0.05 (no larger) | **+30 to +80 %**, 10–11; 11–12 | ≥ 2 schemata at ≥ 5; `L_true` = 7 bin +20 to +60 |

**Falsifiers.** The shape-distribution account of the transfer wall is dead if G2's frozen
ladder solves are within ±15 % of C0 on both seeds. The rule-shape account of the textbook wall
is dead if G1 and G2 leave 17 / 19 schemata at ≤ 2. Finding 3 (RL amplifies what the base
does) gets a counter-example if an arm has a lower base rate than C0 and a larger EI − frozen
on both seeds. A large G2 gain with a large held-out loss is a real trade-off, not a win:
report both.

## Design (suggestion)

1. Worktree ready (`~/work/ds-generator`, data hard-linked from `~/nd-takehome/data`; write
   new files, never edit shared ones in place). `git show
   origin/dan_round3-run1:data/r3_1/depth3_req.jsonl > data/r3_1/depth3_req.jsonl` (and
   `_transfer`). Pull the control checkpoints and round-4 files from the bucket.
2. Generator changes behind flags (`--ore_steps`, `--ore_boxes`, `--goal_only`, `--strict`)
   so the control path is byte-identical; `gen.py --test` / `patterns.py --test` still pass;
   commit before generating. Generate on a CPU pod or the first GPU pod's cores
   (`make_coverage_sets.py gen` with workers; the ladder's strict long generation was 32
   workers × 40,000 tries in ≈ 15 min); merge, assemble flat 31,000 per length with the
   exclusions; shape and overlap tables; render check. Commit the tables with the
   pre-registration or right after (gate-0 material).
3. Pods: three RTX 3090 ($0.50/h), one per arm (`pod/lf/` harness, `LEAN_GATE_WORKERS=12`):
   Stage-1 (two seeds) → held-out greedy → coverage on the three pools → dial EI + frozen
   (4 rounds) → ladder T1 + frozen (8 rounds). Control measurements on the third pod.
4. Reward and counting: Lean ∧ `nd_verify` (`lean_gate.py`, unchanged); `nd2lean.py --check`
   on every counted proof; `nd2lean.py` untouched (the BOTE fix is `lean-seed2`'s job on the
   other host); literal texts stored.
5. Budget: 2 styles × ≈ 7.5 pod-hours + control ≈ 4 + generation ≈ 1 ≈ 20 pod-hours ≈ $10;
   ceiling $20. If G2's yield makes generation exceed 2 pod-hours, stop generating, fill
   from G1's pool and disclose.

## Necessity of the pools

As in `BRIEF_ds-composition.md`: `targets_reductio_req.jsonl` (run 5, reviewer-certified,
derived `DN` required), `data/r3_1/depth3_req.jsonl` (required@8, round3-run1),
`targets_depth3.jsonl` (the dial's pool; pattern-optional for a minority; kept for
comparability). The ladder pools are byte-identical to ladder-A's (`e0524d0`, `69233bc`).

## Deliverables

`preregistration/ds-generator.md`; `run_ds_generator.md` (≤ 400 words + two figures: the
shape tables as a bar chart per arm; the readiness panel per arm with C0); `numbers.md`
§ ds-generator; `log.md` dated; `artifacts/dsg/summary.json` (one row per arm × seed);
`data/dsg/README.md` with the knob diff, shape and overlap tables; `STATUS.md` line
`DS-GENERATOR DONE <UTC>`; bucket `hf://buckets/dan-pandori/nd-rl/ds-generator/{ckpts,artifacts,data}`;
pods deleted; `touch ~/runs/ds-generator/executor.done`. Questions for Dan in `QUESTIONS.md`
with the default you follow.

---

## Resume 2026-09-23 (Dan lifted the pause; $50 across the three dataset-style runs)

Your earlier session ended when the account ran out of Fable credits (2026-09-22 ≈ 10:33 UTC),
and the balance floor then deleted every pod. Nothing was wrong with the work. Resume it.

**What changed while you were stopped — use all of it.**

1. **The sampler is ~2× faster.** Use `sample.generate(..., path="fast", early="eos",
   compact=True, rowrng=True, batch=4096, max_new=288)`; peak memory ≈ 11 GB, so **at most two
   sampling jobs per 24 GB GPU**, and ≤ 3 concurrent jobs per GPU in total (four was measured
   counterproductive). Read
   `~/nd-rl/experiment-summaries/2026-09-23-efficiency-sampler-and-checker-throughput/README.md`
   before you set batch sizes. Note the caveat recorded there: a batch-size change reshuffles
   which proofs are accepted about as much as an RNG re-draw does, so hold the batch **fixed
   across every arm you compare**, and say which batch it was.
2. **Every number must name the model it was measured on** — checkpoint, parameter count,
   format (`lean_seq` / token), from-scratch or pretrained, and its training set — in the
   write-up and in each `numbers.md` table. An inherited number carries its label. This is now
   in `AGENT_POLICY.md`, and the reviewer treats an unlabelled number as a finding.
3. **Pod-hour ceiling.** `podbudget <run-id>` shows it. At 80 % a file
   `~/runs/<run-id>/BUDGET_WARNING` appears — check for it between stages. You may extend
   yourself within the declared dollar budget: `podbudget <run-id> --extend <hours> "reason"`;
   it is refused if the projection exceeds the budget, and then you ask Dan in `QUESTIONS.md`
   and proceed on a stated default.
4. **Upload to the bucket as you go, not only at the end**
   (`hf buckets sync artifacts hf://buckets/dan-pandori/nd-rl/<run-id>/artifacts` after each
   stage). A host cleanup deleted local `artifacts/` and `ckpts/` for the runs that had not
   uploaded; `data/` survived. Treat local disk as scratch.

**Specific to this run.** Your `artifacts/` were uploaded to
`hf://buckets/dan-pandori/nd-rl/ds-generator/` before the cleanup — pull what you need back
rather than re-measuring. `ckpts/` and the local `artifacts/` are gone from disk; `data/`
survived. G2's held-out collapse (0.626 / 0.606 against the control's 0.909 / 0.896) is already
a clear negative: do not spend the budget re-confirming it. Spend it on G1 and on the ladder and
textbook numbers that are missing. Budget $14, ceiling 28 pod-hours.
