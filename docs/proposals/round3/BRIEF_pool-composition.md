# Run brief: round3-run1 — pool composition or pattern class?

Run id `round3-run1`. Role: executor. Fork `~/nd-takehome`, branch `dan_round3-run1` from
`origin/dan_novelty` (or its merged successor). `AGENT_POLICY.md` governs; the design below is
a suggestion, the question is what matters. Read first: `ignition.md`, `review_ignition.md`,
`run5.md`, `review_round2-run5.md`, and §1 of
`~/nd-rl/docs/proposals/2026-09-18-proposals-round-3.md`.

## Question

Zero-rate reductio draws never ignite; three of five zero-rate depth-3 draws did. The two
pools differ in one thing besides the pattern: the depth-3 pool paid reward for depth-≤ 2
proofs of 673 of its 1,000 targets, and every zero-rate draw that ignited had trained on 99–203
such proofs before its first depth-3 proof, while zero-rate reductio arms never took a
training step (every target needed `DN`; 155k attempts, 0 hits). Is the asymmetry in clause (3)
of the rule of thumb a property of the pattern class, or of whether the pool rewards
neighbours of the pattern? Three conditions per Stage-1 draw and per pattern: a pool of
required targets only, a mixed pool of required targets plus pattern-free neighbours, and a
neighbours-only pool with every pattern proof excluded from training while the pattern's rate on
the required targets is sampled each two rounds.

## Why it matters

If zero-rate depth-3 draws stop igniting without neighbours and zero-rate reductio draws
start igniting with them, clauses (1) and (3) become one statement about reward landscapes:
expert iteration composes an absent pattern only where it can climb through rewarded
neighbours. If depth-3 ignites without neighbours and reductio does not ignite with them, the
class difference is real and has passed its fairest test. The drift-only arms say which
mechanism it is: a prior that moves under training on neighbours (rate rises before any
pattern proof is rewarded) or selection of a lucky sample.

## Gate 0: `preregistration/round3-run1.md`

Before the first pod, write and commit that file with the question, the design you will run,
and the expectations below (edit numbers only with a dated reason in the same file). Gate 0
compares the commit time with the first pod's creation in `~/pods.log`.

### Pre-registered expectations (numbers)

Let n₀ be the number of zero-rate draws per pattern (pre-RL hits = 0 over all required
targets at k = 2,000); expected n₀ ≈ 3–5 of 8 for depth-3 (5 of 10 `a1` draws in the ignition
study on 300 targets) and ≈ 4–5 of 8 for reductio (7 of 11). If n₀ < 3 for a pattern, train
4 more seeds (disclose).

- **E1 pools.** ≥ 250 depth-3 required@8 targets (restricted search `--max_depth 2 --bound 8`
  fails, 0 timeouts, unrestricted ≤ 8 lines); for every one the bound-10 depth-≤ 2 alternative
  is recorded (expected: ≥ 90 % have a 9-line alternative, ≤ 5 % none — the proposer found
  137 / 1 / 4 of 142). ≥ 250 reductio neighbours (conclusion `( ~ ( ~ X ) )`, `--forbid DN`
  finds a ≤ 10-line proof, `intuit.py` says provable, min length 7–8). 0 oracle
  inconsistencies. Every draw solves ≥ 40 % of the neighbour stratum of its mixed pool by
  round 8 (the neighbours are rewarding).
- **E2 depth-3, required-only pool.** Zero-rate draws ignite (≥ 20 required targets with a
  depth-3 proof, cumulative) in **0 of n₀**; they solve ≤ 5 targets in 8 rounds. Non-zero
  draws ignite in ≥ 80 % by round 5.
- **E3 depth-3, mixed pool.** Zero-rate draws ignite on the required stratum in ≥ 40 % (3 of
  5 in the ignition study), after ≥ 1 round of neighbour-only successes.
- **E4 reductio, mixed pool.** Proposer's expectation: **1–2 of n₀** zero-rate draws ignite
  (≥ 12 required targets with a strict reductio proof) by round 8. The standing rule predicts
  0. Non-zero draws (e.g. the run-5 s0 kind) ignite on both pools as in run 5.
- **E5 drift-only (neighbours only, pattern proofs excluded from training).** Depth-3: rate on
  the required pool rises from 0 to ≥ 10⁻⁵ (≥ 3 hits in 300k) by round 8 in ≥ 2 of n₀ arms.
  Reductio: ≤ 1 hit in 300k at every checkpoint in every zero-rate arm.
- **E6 route choice.** Required@8 depth-3 targets solved without a depth-3 proof (the 9-line
  flat route) ≤ 5 % of solved required targets in every arm (the ignition arms show 0).
  Reductio required targets solved without the strict pattern: 0 (reviewer-certified pool).

### What would falsify the standing rule

- Clause (3) ("not predictable for structural patterns") dies if E2 holds: with no
  neighbours, structural ignition is exactly as base-rate-gated as reductio.
- Clause (2) ("or nothing") dies if ≥ 2 zero-rate reductio draws ignite on the mixed pool, or
  if any zero-rate reductio arm drifts to ≥ 10⁻⁵ in E5.
- The rule survives unchanged only if zero-rate depth-3 draws ignite on the required-only
  pool at the mixed-pool rate and no zero-rate reductio draw moves in either condition.

## Design (suggestion)

**Pools.** Depth-3 required@8: start from the 142 in
`data/p2/targets_depth3_maxdepth2.jsonl` (`min_lines_ub` None) ∩ `targets_depth3.jsonl`
(`min_lines_ub` 7–8); add candidates from `data/p2/pool_long_minlen.jsonl` whose generating
proof has depth 3 and `min_lines_ub` ∈ {7, 8}, labelled with `minlen.py --max_depth 2
--bound 8 --time 20`; then `--max_depth 2 --bound 10 --time 40` on the pool for the
alternative-length field; keep 300 + 100 transfer. Depth-3 optional neighbours: 300 from the
673 with a depth-≤ 2 proof at 7–8 lines (`targets_depth3_maxdepth2.jsonl`), disjoint from
the required set. Reductio required: `data/p2/targets_reductio_req.jsonl` (300) and
`transfer_reductio_req.jsonl`. Reductio neighbours: from `pool_long_minlen.jsonl` (167
candidates with a `( ~ ( ~` conclusion at 7–8 lines) and, if short, a fresh
`make_coverage_sets.py gen --long` batch filtered on the conclusion; label with
`minlen.py --forbid DN --bound 10` (must succeed) and `intuit.py`; keep 300. Every pool
class-disjoint from `train_depth3_f0_a1.jsonl`, `train_reductio_f0.jsonl`, both held-out
sets and `validation_36.jsonl` (renaming key). Add a `stratum` field (`required` /
`neighbour`) to every record; `expert_iter.py` ignores it. Hand-check ten required@8 targets
(read the 9-line alternative and the 7–8-line depth-3 proof) and ten neighbours before the
arms; log them.

**Stage-1 draws.** `train.py --mode abs --steps 6000 --bs 128 --cap 6` on
`train_depth3_f0_a1.jsonl` seeds 20–27 and `train_reductio_f0.jsonl` seeds 20–27 (the
ignition study's seeds 2–10 were never uploaded; reductio s1, s2 exist and may be added as
two more zero-rate draws). Pre-RL sample: `coverage.py --k 2000 --temperature 0.8 --batch
2000` over **all** required targets (300 × 2,000 = 600k per draw), per-proof hit counts kept.
Classify each draw zero / non-zero.

**Arms** (k = 32, 8 rounds, T = 0.8, retain 20k, min-round rule), per draw and pattern:
1. `req` — required pool only (300 targets).
2. `mix` — required + neighbours (600 targets; report the two strata separately).
3. `drift` — neighbours only, with `--exclude_pattern <name>` (new opt-in flag in
   `expert_iter.py`: drop any found proof whose dependency-pruned form satisfies
   `patterns.depth3` / `patterns.reductio` from the training mix; log the count dropped; a
   verifier-checked test in `patterns.py --test` style); after rounds 2, 4, 6, 8 run
   `coverage.py --k 1000` on the 300 required targets from the saved round checkpoint. For
   non-zero draws run `drift` on at most two, as a reference.
Frozen control = the pre-RL sample restricted to the first 256 attempts per target (as in the
ignition study), reported next to each `req` and `mix` arm. If pod time is short, drop `drift`
for non-zero draws first, then `mix` for non-zero draws.

**Acquisition** = required targets with a verified proof whose pruned form has the pattern;
ignition thresholds 20 (depth-3) and 12 (reductio) targets, cumulative, and the round.
Solved-without-pattern per stratum. Every count from `found_r.jsonl` after start-index
normalisation; `nd_verify` re-run on every counted pattern proof.

## Controls and seeds

8 Stage-1 seeds per pattern (16 models), two sampling seeds are not needed because draws are
the unit; frozen at matched attempts; solved-without-pattern per stratum; held-out greedy per
round (models intact). Report n₀ and the per-draw pre-RL rate table first.

## Budget and stop rule

Three RTX 3090 pods at $0.50/h. Estimate: 16 Stage-1 × 12 min ≈ 3.2 h; 16 pre-RL samples of
600k ≈ 16 × 0.4 h ≈ 6.5 h; ≈ 40 EI arms × 0.25–0.4 h ≈ 13 h; drift sampling 8 arms × 4 × 300k
≈ 3 h; total ≈ 26 pod-hours ≈ **$13**; ceiling $50. Never co-schedule a coverage run with
another job on a 24 GB card (run-5 OOM rule). Stop when every arm has 8 rounds, or at $40
spent, whichever first; at $40 finish the `req` and `mix` arms of zero-rate draws before
anything else. Re-arm the kill switch (`killswitch`, ≤ 30 h). Delete pods when pulled.

## Deliverables

`preregistration/round3-run1.md` (gate 0); `run1.md` (≤ 400 words + figures: ignition
vs pre-RL rate with pool as the marker, per pattern; drift curves); tables in `numbers.md`
§Round 3 run 1: per-draw pre-RL rate, ignition round and acquisition per arm and stratum,
route-choice counts, drift rates per checkpoint; `artifacts/r3_1/summary.json`; pools with
oracle fields in `data/r3_1/`; `STATUS.md` section ending `RUN3-1 DONE <UTC>`; bucket
`hf://buckets/dan-pandori/nd-rl/round3-run1/{artifacts,ckpts,data}` (all 16 Stage-1
checkpoints — later runs reuse them); `touch ~/runs/round3-run1/executor.done`. Deviations
dated in `log.md`; questions to `QUESTIONS.md` with defaults.
