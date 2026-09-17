---
written_on: 2026-09-18
written_by: agent:claude
---

# Proposals, round 3: four runs that could kill or sharpen the rule of thumb

Written by the proposer role (run id `round3-proposer`, 2026-09-17/18 UTC). Evidence base, in
the order read: `docs/STATE.md`; the [novelty campaign summary](../../experiment-summaries/2026-09-15-novelty-campaign/README.md)
with its follow-up and ignition study; the four reviews on the fork (`review_campaign.md`,
`review_followup.md`, `review_ignition.md`, `review_round2-run5.md`); the
[round-2 proposals](2026-09-17-proposals-round-2.md); the
[run-5 summary](../../experiment-summaries/2026-09-17-round2-run5-required-pools/README.md) and the
[week-38 digest](../digests/2026-38.md) (both on branch `dan_round2-run5`). Executor-ready briefs
with the `preregistration/<run-id>.md` gate-0 convention are in [`round3/`](round3/).

**Standing rule of thumb (what these runs attack).** (1) Expert iteration crosses zero
pretraining coverage for a structural repetition of a learned move (a third box from data
capped at two). (2) For rule sequences it amplifies what the base model already generalised
to, or nothing (reductio: zero-hit Stage-1 draws end at 0 / 300 on a pool where every target
needs the pattern; non-zero draws are elicited 13× their frozen control). (3) Ignition is
predictable from the pre-RL base rate for rule sequences (14 / 14 draws so far) and not for
structural patterns (3 of 5 zero-hit depth-3 draws ignited anyway).

**Already running — not proposed again.** Round 2: run 2 (six new patterns, the intermediate-
checkpoint drift measurement, the cap-8 depth-4-free addendum), run 3 (minimum outside data
for ignition) on this host; run 1 (ND → Lean, novelty by scale with Qwen) and run 4 (GRPO vs
EI) on the second host. Run 5 is done and reviewed.

## The confound this round is built around

Two facts from files already on the fork, neither stated in any summary:

- *Zero-rate reductio arms never trained.* On the 606 and the required 300 pools every target
  needs `DN`; a Stage-1 draw with no reductio-shaped sample finds nothing, so expert iteration
  never takes a gradient step (ignition study: "155,136 attempts each, 0 pattern hits, no
  training"; run 5: s1, s2 at 0 / 300). Their "ignition is the base rate" is, under expert
  iteration, close to a tautology: with nothing to train on, only sampling luck can move the
  policy.
- *Zero-rate depth-3 arms trained on the pattern-free stratum first.* The depth-3 pool is
  mostly optional: 673 of its 1,000 targets have a depth-≤ 2 proof within 8 lines
  (`data/p2/targets_depth3_maxdepth2.jsonl`), 142 have a 7–8-line proof but no depth-≤ 2 proof
  within 8 lines, 185 have no ≤ 8-line proof found. Recounting the ignition arms by stratum
  (`artifacts/p2/ei_depth3_f0_a1_s*/found_r.jsonl`, proposer's recount): the three zero-hit
  draws that ignited solved 99 (s2, round 1), 203 (s3, rounds 1–2) and 189 (s8, rounds 1–2)
  optional targets with depth-≤ 2 proofs *before their first depth-3 proof*; the two that never
  ignited (s4, s5) trained on 323 and 357 optional targets for eight rounds. The ignition
  study's own sentence — "decided during training on depth-≤ 2 proofs, not by sampling luck" —
  says the same thing.

So clause (3)'s asymmetry may be a property of the two pools, not of the two pattern classes:
one pool paid for neighbours of the pattern and the other paid for nothing. Proposal 1 tests
that directly. A second gap: the base-level class difference (10 / 16 depth-3 draws vs 4 / 11
reductio draws emit the pattern in 600k samples) has overlapping 95 % intervals (0.35–0.85 vs
0.11–0.69), so "structural patterns are generalised more readily by the base" is not
established either; proposal 3 fixes n. Proposal 2 is the two next measurements the run-5
review asked for. Proposal 4 asks whether "or nothing" survives model size.

A necessity fact that shapes proposals 1 and 4: depth 3 is almost never *required* within
bound 10. Of the 142 targets with no depth-≤ 2 proof within 8 lines, 137 have a 9-line
depth-≤ 2 proof, 1 a 10-line one and 4 none (`minlen.py --max_depth 2 --bound 10`, 0
timeouts, run by the proposer on the VPS; the same "sequential boxes with reiteration"
finding run 2 made for depth 4). "Requires depth 3" therefore means "the pattern-free route
is ≥ 9 lines", which is beyond the written frontier of cap-6 models at round 0 but not in
principle. Pools below say so, and a target solved by the 9-line flat route is reported as a
route choice, not an oracle error.

## 1. Pool composition or pattern class? (`round3-run1`, brief `round3/BRIEF_pool-composition.md`)

**Question.** Does the structural-vs-rule-sequence asymmetry in ignition survive when both
patterns are tested on pools that require them (no neighbour reward) and on pools that mix
required targets with pattern-free neighbours (neighbour reward)? And does training on
pattern-free neighbours alone, with every pattern proof excluded from training, raise a
zero-rate draw's pattern rate — for depth 3 and for reductio alike?

**Why it matters.** If zero-rate depth-3 draws stop igniting on a required pool, and zero-rate
reductio draws start igniting on a mixed pool, clauses (1) and (3) collapse into one rule
about the reward landscape: RL composes an absent pattern only where it can climb through
rewarded neighbours, whatever the pattern's class. If instead depth-3 ignites without
neighbours and reductio does not ignite with them, the class difference is real and has
survived its fairest test. Either outcome is worth more than any decoration of the current
wording.

**Design (suggestion).** Depth-3: 300 *required@8* targets (a 7–8-line depth-3 proof, no
depth-≤ 2 proof within 8 lines; the 142 existing plus ~160 more labelled from
`pool_long_minlen.jsonl`), the bound-10 flat alternative recorded per target; 300 *optional*
neighbours (depth-3-generated, depth-≤ 2 proof at 7–8 lines). Reductio: run 5's 300 required
targets; 300 neighbours = 7–8-line generator theorems whose conclusion is a double negation
`( ~ ( ~ X ) )`, provable without `DN` (`--forbid DN` finds a ≤ 10-line proof, `intuit.py`
agrees), so their proofs end with the first half of the reductio template (`AS ( ~ X ) … F ;
NEGI`) and never the second (167 such candidates exist at 7–8 lines). Eight fresh Stage-1
draws per pattern on the f = 0 sets (the ignition study's checkpoints were not uploaded),
classified zero / non-zero by a pre-RL pass@2,000 over *all* required targets. Per draw three
arms: required-only, mixed (required + neighbours), and neighbours-only with
`--exclude_pattern` (a ~10-line addition to `expert_iter.py` that drops pattern proofs from
the training mix) plus pass@1,000 on the required pool at rounds 2, 4, 6, 8. Frozen control =
the pre-RL sample at 256 attempts. k = 32, 8 rounds, T = 0.8.

**Pre-registered expectations.** E1: ≥ 250 depth-3 required@8 targets, ≥ 250 reductio
neighbours, 0 oracle inconsistencies; the mixed pools' neighbour strata are solved ≥ 40 % by
every draw (reward exists). E2 (depth-3, required-only): zero-rate draws ignite (≥ 20 pattern
theorems) in **0 of n₀** (n₀ ≈ 3–5 expected) — they never train; non-zero draws ignite in
≥ 80 %. E3 (depth-3, mixed): zero-rate draws ignite in ≥ 40 % (the ignition study's 3 / 5).
E4 (reductio, mixed): my expectation is **1–2 of n₀** zero-rate draws ignite by round 8
(n₀ ≈ 4 expected); the standing rule says 0. E5 (drift-only): depth-3 rate on the required
pool rises from 0 to ≥ 10⁻⁵ by round 8 in ≥ 2 of n₀ arms; reductio rate stays ≤ 1 hit in
300k in every arm. Solved-without-pattern on the required strata ≤ 5 % of solved targets.

**Falsifiers.** Clause (3) dies if E2 holds (structural ignition is base-rate-predictable
once the pool requires the pattern). Clause (2) dies if ≥ 2 zero-rate reductio draws ignite
on the mixed pool or drift above 10⁻⁵ in E5. The standing rule survives intact only if
zero-rate depth-3 draws ignite on the required-only pool at the mixed-pool rate *and* no
zero-rate reductio draw moves.

**Necessity.** Depth-3 required@8: restricted search (`--max_depth 2`, bound 8) fails, 0
timeouts, unrestricted finds 7–8 lines; the 9–10-line alternative recorded. Reductio: run 5's
pool (all 300 classically valid, G4ip-unprovable; `DN` required unboundedly). Neighbours: the
opposite certificate (provable without the pattern within bound 10, or within depth 2 at
≤ 8 lines). Class-disjoint from every training set, held-out and validation-36.

**Controls and seeds.** 8 Stage-1 seeds per pattern (16 Stage-1 models); frozen at matched
attempts; solved-without-pattern reported per stratum. **Budget** ≈ $12–15 on three RTX 3090s
(~24 EI arms of 300–600 targets, 16 pre-RL samples of 600k, 8 drift arms with 4 × 300k
samples); ceiling $50. **Stop** when every arm has 8 rounds or at $40. **Deliverables** per
the executor brief: `run1.md`, per-stratum tables, ignition-vs-base-rate figure with pool as
the marker, bucket upload.

## 2. The 8-line reductio stratum and the derived-ORE base-rate dependence (`round3-run2`, `round3/BRIEF_stratum-ignition.md`)

**Question.** After a rule sequence ignites, how far does RL carry it? Run 5's igniting arms
solved 51–53 of the 52 seven-line targets and ≤ 1 of the 248 longer ones in 8 rounds, while
the same checkpoints acquired 25–31 eight-line targets on the 606 pool. What is the per-run
probability that the 8-line stratum ignites, and does it depend on rounds, on samples per
round, or on f? Second: is strict derived-ORE amplification proportional to base
reachability and monotone in f (three points), as run 5 could not test?

**Why it matters.** Clause (2) says "amplifies what the base generalised to". The base emits
reductio only on `nand_neg` (all pre-RL hits, every draw); RL already carried it to
`negimp_to_pos` (7 lines) and, on the 606 pool, to `neg_both` and `chain_neg` (8 lines). If
the 8-line stratum ignites with probability well above zero given enough rounds, the honest
clause is "the base rate gates entry; after entry the sequence generalises across schemata
and lengths", which is composition of a kind — RL producing proofs of schemata the base never
emits. If it never ignites in 6 arms × 16 rounds, the elicitation reading tightens.

**Design (suggestion).** Reductio: run 5's required pool plus a *6-line stratum* (the 45
six-line required candidates from `data/p2/run5_reductio_nec.jsonl`, `--forbid DN` fails for all
945). Arms: f = 0 s0, f = 0.1 s0, f = 0.1 s1 (Stage-1 checkpoints exist) × 2 sampling seeds
× 16 rounds at k = 32; the same three models × 1 sampling seed × 8 rounds at k = 64 (same
attempts, half the rounds); zero-rate draws s1, s2 × 8 rounds on the pool with the 6-line
stratum plus pass@10⁴ on the 45 six-liners. Report acquisition per stratum (6 / 7 / 8 / 9 /
10 lines) and per schema, per round. Derived-ORE (cap 8, non-submission, flagged): f = 10⁻³
s0 (exists) and s1 (train), plus two fresh f = 0 cap-8 draws with pass@10⁴ on the required
pool, keeping the one whose strict reachability differs most from 27 / 300; EI + frozen.

**Pre-registered expectations.** E1: the 8-line stratum (133 targets) ignites (≥ 10 targets)
in **3–5 of the 6** sixteen-round arms, first 8-line proof at rounds 6–12; the 9–10-line
strata stay ≤ 3 targets in every arm. E2: k = 64 × 8 rounds ignites the 8-line stratum in
≤ 1 of 3 (rounds matter more than samples). E3: zero-rate s1, s2 have 0 hits on the six-line
targets at 10⁴ and stay 0 / 345. E4: derived-ORE f = 10⁻³ acquisition 0.08–0.16 (between
f = 0's 0.08 and 10⁻²'s 0.16–0.21); the fresh f = 0 draw's EI acquisition is within 1.5× of
its own base reachability at 10⁴ and EI ≈ 2× frozen.

**Falsifiers.** E1 at 0 of 6 supports a real length wall for elicited rule sequences (kills
the "spreads across lengths" reading); E3 with any hit on the six-liners followed by
ignition means the wall was length-gated, not pattern-gated, and clause (2) must say "base
rate at the shortest instance". E4 outside 0.5–2× of base reachability kills "amplification
in proportion".

**Necessity.** The reductio pool is run 5's (reviewer-certified); the six-line stratum inherits
`--forbid DN` failure and must be G4ip-checked with `review_run5_logic.py`. Derived-ORE
required pool is run 5's (0 pattern-free solves in 18 arms).

**Controls and seeds.** Frozen at equal attempts for every arm; 2 sampling seeds per model.
**Budget** ≈ $8 on two 3090s; ceiling $50. **Stop** at 16 rounds or $30. **Deliverables**
`run2.md`, per-stratum per-round curves, the 6 × 16 ignition table, bucket upload.

## 3. Base generalisation by pattern class, many seeds, no RL (`round3-run3`, `round3/BRIEF_base-generalisation.md`)

**Question.** With what probability does a cap-6 Stage-1 draw at f = 0 emit each pattern at
all, and at what per-sample rate, on the full required pools? Does the class difference
(structural vs rule sequence) exist at the base level with non-overlapping intervals? What
about the draw predicts it?

**Why it matters.** Clause (3) makes the base rate the predictor of ignition; the follow-up
review asked for this quantity "before it is quoted as a rate"; the ignition review asked
for the pre-RL sample over all targets. The two class rates now have overlapping intervals,
so the strongest surviving statement about pattern classes rests on n = 16 and n = 11.

**Design (suggestion).** 24 fresh Stage-1 seeds on each of: depth-3 f = 0 set `a1`, the
reductio f = 0 set, and (cap 6, in-rule) the derived-ORE f = 0 set with the strict pattern.
For each draw a pass@2,000 sample over every target of the required pool (depth-3 required@8
from proposal 1, or the 300 shortest campaign targets if run 1 has not built it; reductio
run 5's 300; derived-ORE: the 300 shortest strict targets of `targets_derived_ore.jsonl`,
labelled with `--forbid ORE_DERIVED`). Record per-draw hit counts, per-target hit counts,
Stage-1 held-out loss and per-rule loss, and the frequency of the pattern's two halves in
the draw's own samples (e.g. `NEGI` of a negated hypothesis; `DN` inside a box). No RL.

**Pre-registered expectations.** E1: depth-3 generalisation in 12–18 of 24 draws (point
estimate 0.62 from 10 / 16), reductio 5–11 of 24 (0.36 from 4 / 11), strict derived-ORE
18–24 of 24 (base p ≈ 10⁻³ in the campaign). E2: the depth-3 vs reductio 95 % intervals
separate (lower bound of depth-3 above the upper bound of reductio) — the class difference
becomes a measured fact; if they still overlap at n = 24 each, "structural patterns are
generalised more readily" is dropped. E3: among draws, held-out loss does not predict
generalisation (|Spearman ρ| < 0.3); the rate of the pattern's first half in the draw's own
samples does (ρ > 0.5).

**Falsifiers.** E2 failing kills the base-level class claim. E1 with reductio ≥ 14 / 24
changes clause (2)'s practical content: most draws would generalise the sequence and "or
nothing" would be the minority case.

**Necessity.** Pools inherit their certificates from runs 5 and 1; the derived-ORE cap-6 pool
is the only new one and needs `necessity.py --pattern ORE_DERIVED` (0 inconsistencies, ten
hand-checked). **Controls and seeds** are the 72 draws themselves. **Budget** ≈ $10 on two
3090s (72 Stage-1 × 10 min ≈ 12 h; 72 × 600k samples ≈ 8 h); ceiling $50. **Stop** at
72 draws or $35. **Deliverables** `run3.md`, per-class Clopper–Pearson table, rate
histograms, predictor table, bucket upload.

## 4. Does "or nothing" survive model size? (`round3-run4a` reductio and `round3-run4b` depth-3; second host, two executors; `round3/BRIEF_scale.md`)

**Question.** At 8 layers / d = 512 (≈ 25M parameters) and 12 layers / d = 768 (≈ 85M, the
sprint's size) on the same cap-6 f = 0 sets: how often does a Stage-1 draw generalise each
pattern before RL, and how much does RL add beyond the base's reach (targets acquired by EI
that the base does not reach at 10⁴) — does that fraction grow or shrink with size?

**Why it matters.** The campaign's own next-step list names scale as the untested axis.
Clause (2)'s "or nothing" is only ever observed on zero-rate draws; if larger bases are
never zero-rate, the clause stops applying and the SPAR question's answer becomes "RL
elicits, and bigger bases have more to elicit". If RL's addition beyond base reach grows with
size, that is the first evidence here of RL composing more as the base gets stronger.

**Design (suggestion).** Per size 3 Stage-1 seeds per pattern (`train.py --n_layer 8 --d 512`,
`--n_layer 12 --d 768`, lr 3e-4 for the larger, steps chosen to match the 3.2M model's
held-out greedy ≥ 0.86, gated), pre-RL pass@2,000 on the required pools (run 5's reductio
300; depth-3 required@8 from run 1 or the campaign's 300 shortest), EI k = 32 × 8 rounds,
frozen controls, pass@10⁴ on igniting draws' base for the EI-only count. Executor A: reductio
at both sizes; executor B: depth-3 at both sizes; each its own run id and $50. A100 80 GB
pods ($1.59/h).

**Pre-registered expectations.** E1: the fraction of non-zero draws rises with size for both
patterns — reductio 1–2 / 3 at 25M and 3 / 3 at 85M (from 4 / 11 at 3M); depth-3 3 / 3 at
both. E2: per-sample base rates rise ≥ 10× per size step. E3: zero-rate draws at any size stay
at 0 on the required pool (they never train). E4: the EI-only fraction (acquired targets not
base-reachable at 10⁴) *falls* with size for reductio (run 5: 45 / 51 at 3M) to ≤ 50 % at 85M;
for depth-3 the plateau acquisition rises above 0.36. E5: held-out greedy ≥ 0.93 at 85M.

**Falsifiers.** E4 reversed (EI-only fraction grows with size) kills the elicitation framing
at scale. E1 with 85M draws still zero-rate at ≥ 1 / 3 for reductio keeps "or nothing"
alive at the sprint's size. **Necessity** inherited from runs 5 and 1. **Controls** frozen at
equal attempts; 3 seeds per cell. **Budget** ≈ $20–30 per executor on A100s (Stage-1 85M
≈ 25 min each, pre-RL 600k samples ≈ 45 min each at 85M); ceiling $50 each. **Stop** at $45
per executor or when the 85M cell has 3 draws through 8 rounds. **Deliverables** `run4a.md`,
`run4b.md`, size × pattern tables, bucket upload.

## Order and dependencies

1 first (it decides how 3 and 4 name their depth-3 pool; if it is late, 3 and 4 fall back to
the campaign's 300 shortest depth-3 targets, disclosed). 2 is independent and cheap. 3 and 4
can run at once; 4 is sized for the second host with two executors.

## Questions for Dan (defaults stated; copied to the fork's `QUESTIONS.md` by the executor at gate 0)

- **2026-09-18 (round 3).** Proposal 2's derived-ORE half is cap 8 and outside the take-home's
  cap-6 rule (digest item 2). Default: run it, labelled non-submission as in run 5. Say if
  cap-8 compute should stop.
- **2026-09-18 (round 3).** Proposal 4 uses two executors on the second host, each with the
  $50 per-run budget. Default: proceed; ceiling $50 each, no request for more.
- **2026-09-18 (round 3).** Proposal 1 adds a training-mix filter (`--exclude_pattern`) to
  `expert_iter.py` for the drift-only arms. Default: add it as an opt-in flag, off in every
  other arm, with a verifier-checked test.
