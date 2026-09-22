# Run brief: ds-rendering — how the same proofs are written in Lean (premise re-statement, formula-free `have`s, `intro` boxes)

Run id `ds-rendering`. Repository `~/work/ds-rendering` (branch `dan_ds-rendering`, a worktree
from `origin/dan_lean_format`). Role: executor. `AGENT_POLICY.md` governs; designs are
suggestions. Read first: `~/nd-rl/docs/proposals/2026-09-22-dataset-styles.md` (proposal 9; its
"Protocol shared by every arm" section is this run's protocol), `lean_tok.py` (render, `decode`,
`inverse`, the naming schemes), `nd2lean.py` (`translate`, `box_term`), `lean_gate.py`,
`preregistration/lean-format.md`, `review_lean-format.md` (rows on `--no_shift`, wrong-name
citations, the `L_true` = 7 bin). Pod ceiling **$18**; hard stop 30 h. This run was queued
behind `ds-composition` / `ds-generator`; whichever of them is still running shares the host
with you but not pods, budget or bucket directory. Their results, if any are in, are not
inputs to your pre-registration — write yours from the proposal.

## Question

With the proofs (the a1 set: same 155,000 ND proofs, cap 6, depth-3 removed), model, schedule
and pools held fixed, does the **Lean rendering** the model is trained on — whether premises
are re-stated as `have` lines, whether `have`s carry their formula, and whether boxes are
lambdas or `intro` tactics — change held-out accuracy by length and RL readiness (base rates of
depth-3 and required reductio at pass@2,000, EI − frozen after 4 dial rounds, transfer `L*`
after ladder T1)? The sharp sub-question: is the cap + 1 length horizon a property of the
*text* length or of the ND proof's structure?

## Why it matters

Lean-format's mechanism test is the only rendering experiment so far, and it found the
surface form matters a lot (7 / 8 / ≥ 9-line proofs at pass@16: `lean_seq` 266 / 130 / 18 vs
token 108 / 1 / 0; frozen depth-3 0.13–0.28 vs 0.005) without saying why (the "nested `fun`
looks the same at every depth" hypothesis is untested). Today a premise is a hypothesis of
the statement *and* a `have n1 : F1 := h1` line that counts toward the ND cap, so a 7-line
ND proof with two premises is a 7-`have` text the model has never seen at that length; if
citing `h1` directly makes it a 5-`have` text, the "cap + 1 horizon" should move by the
number of premises — or not, which would locate the horizon in the ND structure. Every
`have` carries its formula, which is what a from-scratch model has to write and Lean could
infer for `IMPE`, `ANDE`, `NEGE` and `R` — a 20–30 % token saving if it costs nothing.

## Arms (two Stage-1 seeds each; the a1 set re-rendered; `lean_seq` naming with offset everywhere)

- **C0 control.** As in `BRIEF_ds-composition.md`: the a1 set and its two bucket checkpoints;
  measure held-out by length, pass@2,000 on the three pools, ladder T1 + frozen; dial round-4
  from the lean-format round files. (If `ds-composition`'s control ladder is already in the
  bucket when you start, you may reuse it and say so; the coverage numbers are cheap either way.)
- **R1 no premise re-statement.** Premise lines are not rendered; later lines cite `h_k`
  directly (`have n3 : Q := h1 n2`). The ND record is unchanged (cap 6 counts premise lines as
  before); the text has `n_lines − n_prem` `have`s. `inverse` re-inserts the premise lines
  from the statement. Report the text-length histogram beside the ND one.
- **R3 formula-free `have`s.** `have n3 := n1 n2` (no `: F`) for `IMPE`, `ANDE1` / `ANDE2`
  (`.1` / `.2`), `NEGE` and `R`; annotations kept for `ANDI`, `ORI`, `IMPI` / `NEGI` (the binder
  type stays), `ORE`, `DN`, `BOTE`, and for every `PR` line. `inverse` recomputes the dropped
  formulas by one rule application each (deterministic; `nd_verify` is then the check that it
  recomputed them right). Confirm Lean accepts 1,000 rendered proofs before training —
  unannotated `have`s with `¬` arguments are where elaboration could differ.
- **R2 `intro`-tactic boxes.** `have n5 : ( P → Q ) := by intro n2 ; have … ; exact n4` instead
  of `( fun ( n2 : P ) => by … exact n4 )`; `NEGI` likewise (`intro n2 ; … ; exact n4`). Predicted
  null; **drop first** if short.

Each variant is a rendering mode of `lean_tok.py` (`lean_seq_noprem`, `lean_seq_nofml`,
`lean_seq_intro`) behind `train.py --mode`, stored in the checkpoint like the existing modes,
with the strict grammar extended so anything outside it still yields `LEANPARSE`. Render check
per variant (3,000 render → inverse identical; Lean accepts 1,000 literal texts; 300
theorem-swapped negatives rejected) committed before the first pod. Literal sampled texts stored
in every `found_*.jsonl`. Vocabulary sizes reported.

## Pre-registered expectations (in `preregistration/ds-rendering.md` before the first pod; both seeds)

| arm | tokens per proof vs C0 | held-out (overall; 6-bin) | pass@16 on `data/transfer.jsonl`: 7 / 8-line | depth-3 pass@2,000 (1,000; req8) | reductio req pass@2,000 (7-line; ≥ 8) | dial EI − frozen, round 4 | ladder frozen solved, `L*`; T1 `L*` |
|---|---|---|---|---|---|---|---|
| C0 | 1.0 | measured (≈ 0.90; ≈ 0.85) | 266 / 130 (full-set model; measure the a1 models) | 0.35–0.55; 0.05–0.25 | 5–25; 0–5 | +0.20 to +0.30 | 220–320, 9–10; 10–11 |
| R1 | 0.75–0.85 | **+0.5 to +2 pp; +2 to +4 pp** | **×2–3 / ×1.5–2.5** | **+10 to +20 pp; +10 to +20 pp** | ×1.5–3; ≥ C0 | within ±0.05 (no larger) | +20 to +50 %, **+1**; 11–12 |
| R3 | 0.70–0.80 | ±1 pp; +0 to +3 pp | ×0.8–1.5 | within ×0.7–1.5 | within ×0.7–1.5 | within ±0.05 | ±15 %, unchanged; 11 |
| R2 | 1.0–1.05 | ±1 pp; ±1 pp | ×0.8–1.2 | **within ±5 pp** (null) | within ×0.7–1.5 | within ±0.05 | ±15 %, unchanged; 11 |

**Falsifiers.** "The horizon is text length" is dead if R1's 7-line pass@16 count is within
±25 % of C0 on both seeds (then the cap + 1 horizon lives in the ND structure and the
`--per_len` / cap arms of `ds-composition` are the only levers). The nested-lambda mechanism
for the Lean base's depth-3 composition is supported if R2's frozen depth-3 rate halves
(pre-registered as *not* expected). Finding 3 (RL amplifies what the base does) gets a
counter-example if an arm has a lower base rate than C0 and a larger EI − frozen on both
seeds. R3 is a performance arm: it "qualifies" (proposal 9's decision rule) if held-out is
within 1 pp and no readiness number is worse.

## Design (suggestion)

1. Worktree ready (`~/work/ds-rendering`, data hard-linked; write new files only). Pull the
   control checkpoints and round-4 files from the bucket; `git show
   origin/dan_round3-run1:data/r3_1/depth3_req.jsonl > data/r3_1/depth3_req.jsonl` (and
   `_transfer`).
2. Implement the three modes in `lean_tok.py` first, on the VPS, with the render check and a
   `--selftest`; commit. This is the run's main effort; if R2 is not done within the first
   8 h, drop it and say so.
3. Pods: three RTX 3090 ($0.50/h), one per arm (`pod/lf/` harness, `LEAN_GATE_WORKERS=12`):
   Stage-1 (two seeds) → held-out greedy → `eval_set.py --k 16` on `data/transfer.jsonl`
   (the mechanism test; distinct proofs by written length, start-index-normalised) → coverage
   on the three pools → dial EI + frozen (4 rounds) → ladder T1 + frozen (8 rounds). Control
   on whichever pod frees first.
4. Reward and counting: Lean ∧ `nd_verify` through `lean_gate.py` on each variant's literal
   text; `nd2lean.py --check` on every counted *denoted* proof (the checker of record renders
   the ND proof its own way — say that in the write-up); `nd2lean.py` untouched.
5. Budget: 3 styles × ≈ 7.5 pod-hours + control ≈ 4 ≈ 27 pod-hours ≈ $14; ceiling $18. Stop
   at $16: drop R2's ladder, then R2, then R3's ladder.

## Necessity of the pools

As in `BRIEF_ds-composition.md`. Note for R1: the pools' `n_lines` / `min_lines_ub` count
premise lines; report R1's results on the ND scale (the pool's) *and* note the text length,
so "8-line" means the same thing in every table.

## Deliverables

`preregistration/ds-rendering.md`; `run_ds_rendering.md` (≤ 400 words + two figures: the
7 / 8 / ≥ 9-line pass@16 counts per arm; the readiness panel per arm with C0); one rendered
example of the same proof in all four modes; `numbers.md` § ds-rendering; `log.md` dated;
`artifacts/dsr/summary.json`; `STATUS.md` line `DS-RENDERING DONE <UTC>`; bucket
`hf://buckets/dan-pandori/nd-rl/ds-rendering/{ckpts,artifacts,data}`; pods deleted;
`touch ~/runs/ds-rendering/executor.done`. Questions for Dan in `QUESTIONS.md` with the
default you follow.
