# Brief: proposals round 2 — five runs in order (5 → 2 → 3 → 1 → 4)

Read `AGENT_POLICY.md` first; it governs this brief. Then `ignition.md`, `review_ignition.md`,
`followup.md`, `review_followup.md`, and the proposal document in the main project repo:
`~/nd-rl/docs/proposals/2026-09-17-proposals-round-2.md` (branch `dan_proposals`; read only —
you work in `~/nd-takehome`, branch `dan_novelty`). Dan approved all five proposals in the order
5, 2, 3, 1, 4. Each proposal is **its own run** with its own $50 pod budget, its own
`STATUS.md` section, its own `run<N>.md` write-up (≤ 400 words + figures), pre-registered
expectations committed before its first pod job, and its own bucket upload at the end
(`hf buckets sync … hf://buckets/dan-pandori/nd-rl/round2/run<N>/…`). Re-arm the pod kill
switch for each run with `crontab` (`killswitch` at a UTC time ≤ 30 h ahead). Questions for
Dan go in `QUESTIONS.md` with the default you will follow.

The designs below are the proposals' designs, restated briefly; they are suggestions, and the
question in each is what matters.

## Run 5 — target pools that require a pattern

Extend `minlen.py` with rule restrictions (a `--forbid` option: e.g. forbid `DN`; forbid `ORE`
whose disjunction line is not `PR`). A theorem *requires* a pattern if the unrestricted search
finds a proof within bound 10 (or the theorem is a known classical-only schema) and the
restricted search fails within bound 10 with no timeout. Build 300-target pools per pattern
(reductio via "requires `DN` with no `~~` in the sequent"; derived-ORE via "requires an `ORE`
on a derived line", cap-8 setting) from the existing candidate files; hand-check ten. Rerun the
two dials with two training seeds at f = 0 and f_max, frozen controls, base pass@10⁴ on the
f = 0 models. Report acquisition on the *required* pools next to the campaign's numbers.

## Run 2 — six new patterns, classes pre-registered

Write `patterns.py` predicates and verifier-checked tests for: **structural** — box depth 4
(from data capped at depth 3), `IMPE` chain length ≥ 5 (from data with chains ≤ 3), a
three-way nested `ORE`; **rule sequences** — `IMPI` whose box contains an `ORE`, `NEGI` whose
box applies `ANDE` to the hypothesis, `ORI` immediately consumed by `ORE`. Commit the class
assignments and expected f = 0 outcomes before assembling any set. For each: an f = 0 set (two
training seeds), targets that use the pattern (and, where run 5's necessity check applies,
require it), identical EI, frozen controls, base reachability. Add the ignition review's
measurement: for zero-hit draws that ignite, the pattern's base rate on the round-1…4
checkpoints (pass@2,000 on 300 targets) — does the prior drift before the first proof?

## Run 3 — minimum outside data for ignition

From the saved round-4 checkpoints of the non-igniting arms (depth-3 s4, s5, s7; reductio
zero-rate seeds), inject and take one training step, then continue EI to round 8:
(a) 1 / 4 / 16 sibling proofs; (b) 4 generator-made ≤ 6-line proofs that contain the pattern on
unrelated easy theorems (`gen.py` with a pattern filter; verify; cap 6); (c) 4 proofs of a
*different* pattern; (d) 4 strings with the pattern's surface tokens but invalid structure
(control; these are not training proofs — if `train.py` rejects invalid proofs, report that the
control is unrunnable rather than weakening the check). Report ignition and plateau per
condition and per arm.

## Run 1 — ND → Lean, and novelty by scale

1. `nd2lean.py`: deterministic translation of spec.md proofs into Lean 4 terms (atoms as `Prop`
   variables, premises as hypotheses, rules as `And.intro`, application, `fun h => …`,
   `Or.elim`, `False.elim`, `Classical.byContradiction`/`not_not`). Install Lean via `elan` on
   the VPS (core only, no Mathlib); check every proof in the take-home pools: `nd_verify`
   accepts ⇔ Lean accepts. Report disagreements with examples. Commit the translator and the
   agreement table before step 2.
2. In-context v0 with **Qwen3-Coder-30B-A3B-Instruct served locally with vLLM on one A100
   pod** (no hosted API): 20 worked examples in context, one fresh theorem, greedy + 8 samples
   at T = 0.7; three surface forms — Lean 4, the take-home token format, plain-English rule
   names (control); validation-36 + 200 transfer theorems stratified by length; 5 example-set
   draws. Verify Lean outputs with Lean and token outputs with `nd_verify`; make the comparison
   fair (same theorems, same examples translated). Report accuracy by form and length with the
   paired Lean − tokens delta and its interval.
3. Novelty by scale: for the campaign's RL-found proof classes (depth-3 from f = 0, reductio
   from f = 0, 9-line transfer proofs; ≥ 30 theorems per class), the smallest Qwen3 model
   (0.6B, 1.7B, 4B, 8B, 14B, 32B — instruct variants, same prompt, k = 16) that proves the
   theorem in Lean. Compare the ordering with the base-model log-probabilities from Phase 1.
   A100 for the 30B/32B models; a smaller card is fine below 8B.

## Run 4 — GRPO vs expert iteration at f = 0

Implement a minimal GRPO (`grpo.py`: groups of 8 and 32 per prompt, binary verifier reward,
group-mean baseline, fixed loss divisor, no KL, on-policy, one update per batch) on the
depth-3 f = 0 sets `a1`–`a3`, two seeds each, at the same total sample budget as EI's 8 rounds;
record per round the fraction of groups with reward variance and the pattern acquisition.
Skip the 85M / relative-codec arm (no access to that code); note it as not run.

## Deliverables per run

`run<N>.md`, `figures/run<N>_*.png`, sections in `numbers.md` and `log.md`, bucket upload,
`STATUS.md` section ending in `RUN<N> DONE <UTC>`. Final line of `STATUS.md` when all five are
done or the time is up: `DONE <UTC>`. Pods deleted before that line. Keep every count
reproducible from files pulled back; a reviewer session will re-derive them.

---

## Update 2026-09-17 22:50 UTC (from Dan, relayed): runs 1 and 4 moved to parallel executors

Proposals 1 (Lean / Qwen) and 4 (GRPO) are now being executed by two other sessions on a second
host, on their own branches (`dan_run1_lean`, `dan_run4_grpo`). **Do not run them here.** Finish
run 2 and run 3, then end `STATUS.md` with `DONE <UTC>`. If you have already started work on
run 1 or 4, stop it, note what exists in `STATUS.md`, and leave it on a clearly named branch;
the parallel executors own those runs.
