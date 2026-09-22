---
written_on: 2026-09-22
written_by: agent:claude
---

# Proposal 10: pretraining dataset styles for the `lean_seq` model — three runs, $60

Written by the proposer role (run id `dataset-styles-proposer`, 2026-09-22 05:30–06:15 UTC),
with authority to launch. Numbered 10: Dan's [Lean-only checking and term size](2026-09-22-lean-only-checking-and-term-size.md)
took number 9 concurrently; the brief copies committed in the run worktrees at launch still say
"proposal 9" and mean this document. Those runs report lengths in lines only (no `term_size`
yet); their reward is Lean ∧ `nd_verify`, as the 2026-09-22 (later) note keeps until `lean_check` is validated. Dan's brief for the week: *"try different dataset generation formats
and styles. The goal is to get good performance on the pre-trained model and also set the model
up well for the RL stage. $60 in total."* Evidence base: `docs/STATE.md`; the
[Lean-default decision](../project_strategy/2026-09-20-lean-default.md) (updated 2026-09-22:
`lean_seq` is the training format; reward is Lean ∧ `nd_verify` until the `BOTE` fix is
verified); the [lean-format summary](../../experiment-summaries/2026-09-21-lean-format-training-format/README.md)
and its review; the campaign, round-2/3 and ladder summaries; the reviews on the fork; and
`gen.py`, `make_coverage_sets.py`, `nd2lean.py`, `lean_tok.py` on `origin/dan_lean_format`.
Executor briefs are in [`dataset-styles/`](dataset-styles/).

## What the data looks like today (facts every arm is measured against)

The control is the cap-6 `lean_seq` a1 set (`data/p2/train_depth3_f0_a1.jsonl`, 155,000
proofs, depth-3 removed) and its two Stage-1 seeds from lean-format
(`hf://buckets/dan-pandori/nd-rl/lean-format/ckpts/lf/stage1_a1_seq_s{0,1}.pt`). Its style,
none of which was ever chosen against an alternative:

- **Generator** (`gen.py`): proof-first, no search, no backtracking; 45 % goal mode / 55 %
  forward; box depth ≤ 3; in cap-6 pools each `ORE` branch gets 0–1 step and **no nested box**;
  when a rule needs a formula it does not have, a *lazy premise* is added on the spot (in goal
  mode the one-step fallback `( Z > G )` at p 0.85). Consequences on file: depth ≥ 4, `IMPE`
  chains ≥ 4 and nested `ORE` are unreachable; 97 % of cap-6 derived-`ORE`s are `( X v X )`;
  generator "reductio" is classical-only in 7 of 19,099 cases; 70 % of "7–16-line" generator
  theorems have a ≤ 6-line proof.
- **Composition**: the per-length histogram is **flattened** to 31,000 per written length
  2–6 (`--per_len`). The class-deduplicated raw pool (`pool_cap6_recon.jsonl`, 723,534 classes)
  is not flat: lengths 2 / 3 / 4 / 5 / 6 = 10 / 12 / 13 / 20 / **46 %** (200k-record sample,
  measured 2026-09-22). Flattening therefore *under*-samples 6-line proofs 2.3×. Per-proof rule
  shares in that pool: `AS` 70 %, `IMPI` 46, `IMPE` 34, `NEGI` 24, `DN` 22, `NEGE` 15, `ANDI` 13,
  `ORI` 11 + 11, `R` 2.0, **`ORE` 1.9, `ANDE` 1.5 + 1.5, `BOTE` 1.2**. Kept in the set: 6.1 %
  contradictory-premise theorems, 0.9 % `A |- A v A`-type trivia.
- **Rendering** (`nd2lean.py` → `lean_tok.py`, one symbol per token, 107 tokens): premises are
  hypotheses of the statement and are **re-stated** as `have n1 : F1 := h1` lines (they count
  toward the ND cap); every `have` carries its formula; a box is a term lambda
  `( fun ( n2 : P ) => by … exact n4 )` written inside the rule that discharges it; names by
  first appearance plus a random offset.

Where the control stands (lean-format, reviewed): held-out greedy 0.909 / 0.896 (a1 seeds;
full set by length 1.000 / 0.988 / 0.950 / 0.888 / **0.852**, the 6-line bin is the deficit in
every format); frozen depth-3 acquisition 0.206 / 0.134 at 256 attempts (EI 0.476 / 0.479);
Stage-1 pass@16 on 1,638 transfer theorems writes 266 / 130 / 18 proofs of 7 / 8 / ≥ 9 lines;
ladder frozen `L*` 10, T1 `L*` 11 (full-set model); Lean T1 arms solve the textbook-dominated
`L_true` = 7 bin *less* often than token arms (55–118 vs 139–140 of 300) and 17 of 19 textbook
schemata stay at ≈ 0 under every technique. The required-reductio base rate of a `lean_seq`
model has never been measured (digest open question 3).

## The question, and the rule of thumb each arm could kill

**Question.** Holding the model (3.2M, 4 × 256), schedule (6,000 steps, bs 128), cap (6 on the
ND record), set size (155,000) and the depth-3 f = 0 filter fixed, which change to the
pretraining set's *style* — composition, generator shape, or Lean rendering — most improves
(a) in-distribution held-out greedy by length and (b) **RL readiness**: the base rates of
depth-3 and required reductio at pass@2,000, EI − frozen after 4 rounds on the depth-3 dial,
and transfer `L*` after ladder rung T1, with in-distribution retention after RL?

**Standing findings at stake.**
1. *The cap sets the horizon.* Every pre-RL pattern hit in 72 token draws is a 7-line proof
   (cap + 1); no cap-6 draw wrote an 8-line pattern proof in 600k–2M samples (round3-run3,
   reviewer). `lean_seq` already moved this (130 eight-line proofs at pass@16), and the
   candidate explanation is *text* length: with premise re-statement, a 7-line ND proof with
   two premises is a 7-`have` text. Arms A1, A4 (histogram), A3 (cap 8) and R1 (no premise
   re-statement) attack this from three sides; if a cap-6 set matches cap 8 on the 8-line
   counts and `L*`, "the cap sets the horizon" is dead.
2. *Textbook transfer is confined to schemata the base assigns ≥ 10⁻⁴* (campaign; ladder: 17
   of 19 at ≈ 0). `STATE.md` attributes this to the generator's theorem-shape distribution but
   no run has changed that distribution. Arms A2 (rule quotas — `ORE`, `ANDE`, `BOTE` are at
   1–2 %) and G1 / G2 (generator shape) test it; if De Morgan, dilemma and distribution
   schemata stay at ≤ 2 solves under every arm, the wall is not the rule mix or the box
   shape and the "derived disjunction / derived contradiction" account stands.
3. *RL amplifies what the base already does.* Read here as: the style that raises the base
   rate raises EI − frozen by less, not more. Every arm reports EI − frozen at round 4 and
   `L*` − `L*`_frozen; an arm with a lower base rate and a larger EI increment would be the
   first data point against.

## Candidate styles, ranked by expected value per dollar

Each arm is one change to the control; two Stage-1 seeds; the full metric panel below.
Costs are pod dollars at RTX 3090 $0.50/h from lean-format's measured times (Stage-1 4–9
min; pass@2,000 coverage ≈ 40 + 12 + 12 min per model; 4-round dial EI + frozen ≈ 30 min;
ladder T1 + frozen ≈ 2 pod-hours per seed): **≈ $4 per style**, ≈ $2 for the control's
missing measurements.

| rank | arm | run | change to the control | what it tests | pre-registered expectation (both seeds) | falsifier of the standing finding |
|---|---|---|---|---|---|---|
| 1 | **A1 natural histogram** | 1 | per-length shares 10 / 12 / 13 / 20 / 46 % (the pool's own, unflattened) | histogram as the horizon lever | held-out 6-bin +2 to +5 pp, 2–3-bins ≥ 0.98; pass@16 8-line ×1.3–2; depth-3 pass@2,000 +5 to +15 pp of targets; ladder frozen solved +15 to +40 %, `L*` frozen +0–1, T1 11–12 | 6-bin within ±1 pp *and* frozen ladder solved within ±10 % ⇒ histogram is not a lever |
| 2 | **R1 no premise re-statement** | 3 | cite `h_k` directly; no `have n_i : F := h_k` lines | is the cap+1 horizon *text* length? | pass@16 7-line ×2–3 and 8-line ×1.5–2.5 (a 7-line ND proof with ≥ 1 premise is ≤ 6 `have`s); held-out +0.5 to +2 pp; frozen `L*` +1; depth-3 pass@2,000 +10 to +20 pp | 7-line count within ±25 % ⇒ the horizon is ND structure, not text |
| 3 | **A2 rule quotas** | 1 | flat histogram, but ≥ 10 % of proofs contain `ORE`, ≥ 8 % `ANDE`, ≥ 5 % `BOTE` (filtering the unchanged generator's pool) | rule mix as the textbook wall | ≥ 2 schemata beyond contraposition / export reach ≥ 5 T1 solves (0–2 today); `L_true` = 7 bin +20 to +60; held-out −1 to +0.5 pp; `L*` unchanged | 17 / 19 schemata still ≤ 2 ⇒ not the rule mix |
| 4 | **G2 goal-directed strict** | 2 | 100 % goal mode, no lazy fallback, no `F` in the sequent, final rule an elimination or discharge, `ORE` branches with steps and boxes | shape distribution as the transfer wall | ladder frozen solved +30 to +80 %, frozen `L*` 10–11, T1 11–12; pass@16 8-line ×1.5–2.5; held-out −2 to +1 pp (the held-out is the old generator's distribution) | frozen ladder solved within ±15 % ⇒ shape distribution does not limit transfer |
| 5 | **G1 ORE shape** | 2 | only: `ORE` branches get up to 3 steps and may open boxes | separates G2's box-shape change from its goal-directedness | set: `ORE` share ≥ 5 %, boxes inside `ORE` ≥ 1 % (0 today); ≥ 1 `ORE`-needing textbook schema at ≥ 5 solves; held-out ±1 pp; `L*` 11 | dilemma / De Morgan families still ≤ 2 |
| 6 | **A3 cap-8 yardstick** | 1 | lengths 2–8 flat (**outside the take-home's cap-6 rule; labelled**) | the known lever, for calibration: what fraction of the cap-8 gain does a cap-6 style recover? | 6-bin +3 to +6 pp; pass@16 8-line ≥ ×3; frozen `L*` 11–12, T1 12–13, solved ≥ 1,000; depth-3 pass@2,000 ≥ 0.6 (7–8-line targets are now in-distribution length — not an f = 0-at-length measurement) | A1 or R1 ≥ 70 % of A3's ladder gain kills finding 1 |
| 7 | **R3 formula-free `have`s** | 3 | `have n3 := n1 n2` for `IMPE`, `ANDE`, `NEGE`, `R` (Lean infers the type; annotations kept where Lean needs them) | Lean-native brevity | tokens −20 to −30 %; held-out ±1 pp; 6-bin +0 to +3; readiness within ×0.7–1.5 | — (performance arm) |
| 8 | **A4 cap-heavy dose** | 1 | per-length shares 0 / 5 / 15 / 30 / 50 % | dose–response for A1 | monotone with A1 on the 6-bin and 8-line counts; 2-bin ≥ 0.95; drop first if short | A1 up and A4 down ⇒ an optimum, not a lever |
| 9 | **R2 `intro`-tactic boxes** | 3 | `by intro n2 ; …` instead of `( fun ( n2 : P ) => by … )` | whether the nested-lambda form is what lets the Lean base compose depth 3 | predicted null: depth-3 frozen ±5 pp, `L*` unchanged; drop first if short | frozen depth-3 rate halves ⇒ the lambda form is the mechanism |
| — | textbook-schema instances at a controlled fraction | not run | — | the f-dial for schemata; decorative (it would show elicitation of what was put in) | — | — |
| — | necessity-oracle labels, renaming-class dedup, trivial removal | not run | already in the pipeline (`prune.py`, class dedup, `trivial` filter); the 1.9 % premise-order overlap is a reporting item, not a style | — | — | — |

## The three runs (within $60; two executors at once on this host)

| run id | arms | executor effort | pod ceiling | launch |
|---|---|---|---|---|
| `ds-composition` | C0 (control measurements) + A1, A2, A3, A4 | data-only: assembly from the existing pool with the unchanged generator | **$22** | now |
| `ds-generator` | C0 + G1, G2 | `gen.py` knobs, new raw pool, assembly | **$20** | now |
| `ds-rendering` | C0 + R1, R3, R2 | three `lean_tok.py` variants with inverses and Lean checks | **$18** | queued: starts when the first of the two above writes `executor.done` |

Expected spend ≈ $30–35 (lean-format used $6.2 of $30 with the same components); the
ceilings sum to $60. Each run measures its own control (the a1 checkpoints from the bucket,
≈ $2), so no run waits on another; the three control measurements are a replication of the
control's ladder and coverage numbers across EI / sampling seeds.

## Protocol shared by every arm (the briefs reference this section)

- **Set.** 155,000 records; ND cap 6 asserted on the record and re-asserted on the denoted
  proof of every rendered record (`train.py --cap 6`); depth-3 proofs excluded in the pruned
  *and* written form (independent counter, as the lean-format review did); classes of
  `data/p2/heldout.jsonl`, `data/p2/{targets,transfer}_depth3.jsonl`,
  `data/p2/{targets,transfer}_reductio_req.jsonl`, `data/r3_1/depth3_req{,_transfer}.jsonl`,
  `data/ladder/{rl_targets,transfer}.jsonl` and validation-36 excluded (`--exclude`); the
  overlap table reported both order-sensitive and premise-order-insensitive, for the control
  too (the a1 set was never checked against the ladder or reductio pools — disclose whatever
  it is). A **shape table** per set: length histogram, box-depth histogram, per-proof rule
  shares, boxes-inside-`ORE` count, premise count, contradictory-premise share, mean tokens.
- **Render check.** 3,000 records render → `inverse` → identical ND proof; Lean accepts 1,000
  literal texts; 300 theorem-swapped negatives rejected. Literal sampled texts are stored beside
  the denoted proof in every `found_*.jsonl` (the lean-format review's request).
- **Stage-1.** `train.py --mode lean_seq --steps 6000 --bs 128 --cap 6 --seed {0,1}`,
  defaults otherwise; held-out greedy on `data/p2/heldout.jsonl` (5,000; 1,000 per length 2–6)
  by length and split into theorems whose generator proof uses a pattern vs not. No tuning: a
  style that scores < 0.80 is reported as such and still measured.
- **Readiness 1 — base rates.** `coverage.py --k 2000 --temperature 0.8 --seed 0` on
  `data/p2/targets_depth3.jsonl` (1,000; the dial's pool, pattern-optional for a minority —
  round3-run1), `data/r3_1/depth3_req.jsonl` (300; required@8, restricted search fails at
  bound 8, 0 timeouts) and `data/p2/targets_reductio_req.jsonl` (300; every target needs a
  derived `DN`, reviewer-certified in run 5; strata 7 / 8 / 9 / 10 lines). A sample counts iff
  `nd_verify` **and** Lean accept (`nd_verify` first, Lean on the distinct `nd_verify`-accepted
  proofs, and `nd2lean.py --check` at the end); report targets hit, per-sample rate, hits by
  `min_lines_ub` stratum, and the count of distinct ≥ 8-line pattern proofs.
- **Readiness 2 — dial.** `expert_iter.py --rounds 4 --k 32 --temperature 0.8` EI and
  `--no_train` frozen on the 1,000-target depth-3 pool with the arm's own set as `--train`,
  reward Lean ∧ `nd_verify` through `lean_gate.py`; acquisition at round 4, EI − frozen,
  held-out greedy at round 4. Control round-4 values are read from
  `lean-format/artifacts/lf/{ei,frozen}_d3_seq_s{0,1}/round_4.json`.
- **Readiness 3 — ladder.** `ladder_ei.py` T1 and frozen, 8 × 32, pools byte-identical to
  ladder-A's; `L*`, solved, by `L_true` bin, **textbook solves per schema**, held-out greedy
  at round 8. Control: run on the a1 checkpoints (not on file; the full-set model's 794 / 839,
  `L*` 11, frozen 304 / 309, `L*` 10 are the reference).
- **Controls and seeds.** Two Stage-1 seeds per arm (EI seed = Stage-1 seed; sampling seed 0);
  frozen at equal attempts everywhere; differences quoted only when both seeds agree in sign.
- **Reporting.** One table, arms as rows: held-out by length; pass@2,000 targets hit on the
  three pools with the 8-line counts; dial EI / frozen / difference; ladder frozen and T1
  `L*` and solved; held-out retention after RL; set shape summary. Proofs shown in Lean.

## Decision rule (for the librarian and the next proposer)

An arm **replaces the control set** for subsequent runs iff, on both seeds, held-out greedy is
within 1 pp of the control or better *and* at least two of {depth-3 pass@2,000, reductio
pass@2,000, ladder T1 `L*` or solved} improve with none worse. Cap-8 (A3) is excluded from the
decision (outside the rule) and serves only as the yardstick. If no arm qualifies, the control
stands and the shape table tells the next proposer which axis had no effect.

## Budget, stop rules, and what runs when

Per-run ceilings above; `rpbalance` before each pod; pods deleted when pulled; hard stop 30 h
per executor; kill switch armed. If a run is short on time, arms are dropped in the order
listed in its brief (A4 and R2 first). No test-file run; `nd_verify` unmodified; every training
proof is generator output rendered deterministically. The pause of 2026-09-18 is lifted for
this theme by Dan's brief.
