---
written_on: 2026-09-18
written_by: agent:claude
---

# Round 3, run 1: pool composition or pattern class?

Status: completed 2026-09-18 09:04 UTC (autonomous executor session, four A40 pods, ≈ $6.5)
and independently reviewed 2026-09-18 09:23 UTC. Review verdict: no hard-constraint violation,
every number that could be recounted reproduces exactly (19 claims; the candidate-stage pool
counts and the dollar figure were not re-derived), and the write-up reports its own misses.
The three empirical findings stand as counts; five sentences of the executor's write-up must
be reworded, and one reading — the two neighbour pools described symmetrically as
"pattern-free" — is not let through, because the depth-3 mechanism claim depends on it and the
run has no arm that tests it. Numbers and wording below are the reviewer's. Run id
`round3-run1`.

Review: [`review_round3-run1.md`](https://github.com/dan-pandori/nd-takehome/blob/e5bfcc7/review_round3-run1.md)
(commits `c3f6f09` phase 1 blind recount, `e5bfcc7` phase 2 claims table and verdict; reviewer
code `review_r3_1_recount.py`, `review_r3_1_extra.py`, `review_r3_1_tables.py`; outputs
`artifacts/review_r3_1/*.json`). Executor write-up:
[`round3_run1.md`](https://github.com/dan-pandori/nd-takehome/blob/e5bfcc7/round3_run1.md) on
branch `dan_round3-run1` (read the review first; its "Design" and "Answer" paragraphs are the
parts the review rewords). The brief's file name `run1.md` was already round 2's write-up, so
the executor used `round3_run1.md` (`QUESTIONS.md` 09:04).

Question: zero-rate reductio draws never ignited under expert iteration (EI), while three of
five zero-rate depth-3 draws did
([run 5](../2026-09-17-round2-run5-required-pools/README.md), the ignition study). The two
pools differed in one thing besides the pattern: the depth-3 pool rewarded depth-≤ 2 proofs of
673 of its 1,000 targets. Is the asymmetry in clause (3) of the rule of thumb ("ignition is
predictable from the pre-RL base rate for rule sequences and not for structural patterns") a
property of the pattern class, or of whether the pool rewards neighbours of the pattern?

## Configuration

- Code, data and results: `dan-pandori/nd-takehome`, branch `dan_round3-run1` (worktree
  `~/work/round3-run1` on the agent VPS). Pre-registration `preregistration/round3-run1.md`
  committed in `8c318e7` at 05:09:48 UTC; first pod (`r31a`) created 05:16:09 UTC
  (`~/pods.log`); one amendment (`8d33afe`, 05:50:37 UTC, reductio-neighbour definition, below)
  before any `mix` / `drift` arm (earliest such `round_1.json` 06:00:10). Results `86e1406`,
  write-up final `eaad8ce`. Brief `BRIEF_pool-composition.md`; policy `AGENT_POLICY.md`. Run
  diff: `expert_iter.py --exclude_pattern <depth3|reductio>` (found proofs whose
  dependency-pruned form has the pattern are dropped before the `max_per_thm` slice; count
  logged per round), `make_coverage_sets.py gen --only concl_nn`, `r3_1_pools.py`,
  `r3_1_analysis.py`, `r3_1_numbers.py`, `r3_1_figures.py`, `pod/r3_1/*` (job files with every
  command line).
- Patterns: depth-3 (box depth ≥ 3 in the dependency-pruned proof) and reductio (strict
  `NEGI(~G) … DN` shape).
- Pools (`data/r3_1/`; reviewer: every pool 300 / 300 distinct renaming classes, 0 shared
  classes between either training file and any pool, between any two pools, and between any
  pool and `data/p2/heldout.jsonl`, `validation_36`, `test_short_prompts`):
  - `depth3_req` 300 required@8 targets (no depth-≤ 2 proof within 8 lines, 0 timeouts; minimum
    length 7: 6, 8: 294; a 9-line depth-≤ 2 alternative exists for 288 = 96.0 %, 10-line for 1,
    none at bound 10 for 11 = 3.7 %) + `depth3_req_transfer` 100. 67 + 20 of them come from the
    142 previously known required@8 targets, the rest are new long-pool candidates.
  - `depth3_nb` 300 neighbours: a depth-≤ 2 proof at 7 (178) or 8 (122) lines. These are
    pattern-*optional*: 72 of them have a depth-3 proof one line shorter than the flat one.
  - `reductio_req` 300 / `reductio_req_transfer` 150: run 5's pools unchanged (minimum length
    7 / 8 / 9 / 10 = 52 / 133 / 82 / 33; 16 schemata).
  - `reductio_nb` 300 neighbours: conclusion `( ~ ( ~ X ) )`, a no-`DN` proof within 10 lines,
    intuitionistically provable, unrestricted minimum 7–8 lines, and (05:50 amendment) the
    shortest no-`DN` proof closes with `NEGI` of an assumed `( ~ X )` — the reductio box minus
    its `DN` step. The amendment followed a hand-check showing ≈ 40 % of the first build's
    proofs reached `~~X` by explosion from contradictory premises. No expectation number
    changed.
  - `depth3_mix`, `reductio_mix` 600 = required ∪ neighbours.
- Stage 1: 16 new draws, `python3 train.py --mode abs --steps 6000 --bs 128 --cap 6 --seed <s>`,
  seeds 20–27 on `train_depth3_f0_a1.jsonl` and on `train_reductio_f0.jsonl` (155,000 proofs
  each, max 6 lines; reviewer: 0 proofs of box depth ≥ 3 in the first, 0 strict reductio and 0
  `DN`-of-`NEGI` of any shape in the second).
- Pre-RL sample: `coverage.py --k 2000 --temperature 0.8` on the 300 required targets per draw
  (600,000 samples). Zero-rate = 0 pattern hits. Frozen control = the same sample restricted to
  the first 256 attempts per target.
- Arms (`expert_iter.py --rounds 8 --k 32 --temperature 0.8 --batch 768`, retain 20k, seed =
  model seed), 48 in all: `req` (required pool only), `mix` (required + neighbours, strata
  reported separately), `drift` (neighbours only, `--exclude_pattern`), for all 8 draws of each
  pattern. Drift checkpoints at rounds 2 / 4 / 6 / 8 sampled with `coverage.py --k 1000` on the
  required pool (300,000 samples each) for every zero-rate draw plus two non-zero references
  per pattern (depth-3 s22, s24; reductio s24, s27); reductio s26's drift arm was not sampled.
- Counting rule (pre-registered): acquisition = required targets with a verified proof whose
  pruned form has the pattern, cumulative, round = minimum over raw records of a normalised
  proof. Ignition = ≥ 20 (depth-3) / ≥ 12 (reductio) required targets.
- Pods: r31a–r31d, A40 48 GB at $0.49/h (RTX 3090 stock was Low), created 05:16:09 /
  05:28:44 / 05:46:24 (r31c) / 05:46:15 (r31d) UTC (`~/pods.log`), all deleted 08:54
  (`log.md`); 13.3 pod-hours ≈ **$6.5** (pre-registered estimate ≈ 26 pod-hours ≈ $13 on three
  RTX 3090s, ceiling $50). See `docs/infra/spend.md`.
- Bucket: `hf://buckets/dan-pandori/nd-rl/round3-run1/{artifacts,ckpts,data}/r3_1` (reviewer:
  the three prefixes exist; contents not diffed). `ckpts/r3_1` holds the 16 Stage-1
  checkpoints, 64 drift checkpoints and the round-8 checkpoints of the 16 `mix` arms and the 2
  `req` arms that trained.
- Hard constraints (reviewer): `nd_verify` blob hashes equal `origin/main`'s;
  `artifacts/TEST_RUN_DONE` has no diff against `origin/main` and no test file is named in any
  job line (the orchestrator's gate 1 flagged its mtime, which a fresh worktree checkout
  changes); no training or EI job reads `targets/test_*` or `validation_36`; cap 6 on all
  supervised data and all 16 Stage-1 checkpoints; gate 0 met.

## Observations

All counts were re-derived by the reviewer from raw files with independent code (only
`nd_verify` is shared with the executor). All 227,870 raw records in the 48 arms'
`found_8.jsonl` re-verify against the pool's prompt (0 failures; 7,322 distinct normalised
proofs); all 48 `args.json` match the pre-registered settings. **No required target of either
pattern was solved without the pattern by any model** in the found records, 9.6 M pre-RL
samples and 18.0 M drift-checkpoint samples (0 oracle inconsistencies). Not re-derived: the
restricted `minlen.py` searches behind the pool labels (the recorded proofs were checked
instead), the candidate-stage counts, held-out greedy (from the executor's `round_*.json`).

**n₀ = 6 of 8 depth-3 draws (s20, 21, 23, 25, 26, 27) and 5 of 8 reductio draws
(s20, 21, 22, 23, 25)** have 0 pattern hits in 600,000 samples. Non-zero draws: depth-3 s22
2 hits (1 target), s24 26 (3); reductio s24 1 (1), s26 6 (2), s27 16 (1). Frozen@256: 0
targets everywhere except depth-3 s24 (2) and reductio s27 (1).

**Depth-3 (required targets with a depth-3 proof, of 300, cumulative rounds 1…8; Z =
zero-rate).**

| seed | `req` r8 | `mix` required r1…r8 | ign. round req / mix | `mix` neighbours solved r8 (with a depth-3 proof) | first depth-3 proof in `mix`: neighbour / required round | transfer r8 req / mix (of 100) |
|---|---:|---|---|---:|---|---|
| 20 Z | 0 | 0 0 0 0 0 0 0 1 | – / – | 126 (0) | – / 8 | 0 / 0 |
| 21 Z | 0 | 0 0 2 75 202 229 238 **241** | – / 4 | 207 (98) | 2 / 3 | 0 / 81 |
| 22 | 0 | 0 1 1 2 8 84 178 211 | – / 6 | 216 (82) | 3 / 2 | 0 / 71 |
| 23 Z | 0 | 0 0 5 55 183 239 250 **255** | – / 4 | 239 (92) | 2 / 3 | 0 / 88 |
| 24 | 5 | 0 15 136 193 215 224 231 233 | – / 3 | 208 (100) | 1 / 2 | 5 / 83 |
| 25 Z | 0 | 0 0 0 0 0 0 0 0 | – / – | 106 (0) | – / – | 0 / 0 |
| 26 Z | 0 | 0 21 169 209 227 237 238 **239** | – / 2 | 220 (102) | 1 / 2 | 0 / 85 |
| 27 Z | 0 | 0 0 0 0 0 0 0 0 | – / – | 141 (1) | 7 / – | 0 / 0 |

**Reductio (required targets with a strict reductio proof, of 300).**

| seed | `req` r8 | `mix` required r1…r8 | ign. round req / mix | `mix` neighbours solved r8 (with a reductio proof) | first required pattern proof in `mix` (round) | transfer r8 req / mix (of 150) |
|---|---:|---|---|---:|---|---|
| 20 Z | 0 | 0 0 0 0 0 0 0 0 | – / – | 177 (0) | – | 0 / 0 |
| 21 Z | 0 | 0 0 1 7 23 35 43 **52** | – / 5 | 180 (0) | 3 | 1 / 26 |
| 22 Z | 0 | 0 0 0 0 0 0 0 0 | – / – | 182 (0) | – | 0 / 1 |
| 23 Z | 0 | 0 0 0 0 0 0 0 0 | – / – | 157 (0) | – | 0 / 1 |
| 24 | 0 | 0 4 21 40 47 52 53 53 | – / 3 | 170 (0) | 2 | 0 / 26 |
| 25 Z | 0 | 0 0 0 0 0 0 0 0 | – / – | 170 (0) | – | 0 / 0 |
| 26 | 33 | 0 0 1 7 31 49 52 53 | 7 / 5 | 182 (1) | 3 | 15 / 27 |
| 27 | 0 | 0 0 0 1 7 39 46 52 | – / 6 | 143 (0) | 4 | 0 / 25 |

- **Required only.** All 11 zero-rate `req` arms end at 0 / 300 with 0 found records and no
  training step ever taken (no checkpoint). Non-zero draws: depth-3 s22 0, s24 5 targets (no
  ignition); reductio s24 0, s27 0, s26 ignites at round 7 (33 targets). Over all 16 draws,
  ignition is **`req` 1 / 16 vs `mix` 9 / 16** (Fisher two-sided p = 0.006); non-zero draws
  alone 1 / 5 vs 5 / 5 (p = 0.048); zero-rate draws alone 0 / 11 vs 4 / 11 (p = 0.09).
- **Mixed pool.** Zero-rate depth-3 draws ignite in 3 of 6 (s26 round 2, s21 and s23 round 4;
  239–255 targets); zero-rate reductio draws in 1 of 5 (s21, round 5, 52 targets); 3 / 6 vs
  1 / 5 is Fisher p = 0.545. All five non-zero draws ignite on `mix`. One depth-3 draw (s25)
  solves 106 / 300 = 35.3 % of its neighbours, below the pre-registered 40 % floor; the others
  126–239, reductio 143–182.
- **What the depth-3 neighbours rewarded (reviewer).** In every `mix` arm that ignites from a
  zero-rate draw, the first depth-3 proof in the found set is on a *neighbour* target, one
  round before the first required one (s21, s23: neighbour round 2 → required round 3; s26:
  neighbour round 1 → required round 2), and the igniting arms end with depth-3 proofs of
  82–102 neighbour targets, all rewarded and trained on. **s26 is not a zero-rate draw for the
  pattern**: the untouched Stage-1 model writes 22 depth-3 proofs on 3–4 neighbour targets in
  the 19,200 round-1 samples of its `mix` and `drift` arms (≈ 1.1·10⁻³ per sample; the
  non-zero draw s24 gives 40); it is zero-rate on the required pool only. s21 and s23 are
  0 / 19,200 on neighbours at round 1 and get their first neighbour depth-3 proof at round 2,
  after one round on flat proofs. The non-igniting zero-rate `mix` arms (s20, s25, s27) have
  0, 0 and 1 (at round 7) neighbour targets with a depth-3 proof.
- **Reductio neighbours are pattern-free, and the reductio ignition is two schemata
  (reviewer).** No neighbour target has a strict reductio or any `DN`-of-`NEGI` proof before
  ignition in any `mix` arm (one target at round 8 in s26). Every igniting reductio arm (`mix`
  s21, s24, s26, s27; `req` s26) acquires `nand_neg` 26 / 26 and `negimp_to_pos` 25–26 / 26
  plus at most one stray target (`chain_neg` or `neg_both`): 52–53 of 300, all but the stray
  at the minimum length of 7. These are the two schemata whose oracle proof is a
  `NEGI`-closed derivation of `( ~ ( ~ G ) )` followed by one `DN` line, that is, a
  tightened-definition neighbour proof plus one step. The other 14 schemata (248 targets, 8–10
  lines) stay at 0–1 in every arm. Every drift-checkpoint reductio hit and every pre-RL hit of
  the three non-zero draws is in the same two schemata.

**Drift arms (neighbours only, pattern proofs excluded from training): required-pool pattern
hits per 300,000 samples (targets) at rounds 2, 4, 6, 8.** The reviewer's per-round count of
found records with the pattern equals the logged `excluded_pattern_proofs` in all 16 arms and
all 8 rounds; the training mixes themselves were not pulled, so "no pattern proof was trained
on" rests on that equality plus the code. The seed column marks zero-rate (Z) for the depth-3 /
reductio draw of that seed; a count without brackets is 0 targets.

| seed | depth-3: hits (targets) r2, r4, r6, r8 | depth-3: excluded proofs by r8 | reductio: hits (targets) r2, r4, r6, r8 |
|---|---|---:|---|
| 20 Z / Z | 0, 0, 0, 0 | 0 | 0, 0, 0, 0 |
| 21 Z / Z | 1 (1), 13 (2), 0, 2 (1) | 52 | 1 (1), 1 (1), **151 (8)**, 6 (1) |
| 22 – / Z | 112 (1), 0, 0, 3 (1) | 0 | 0, 0, **4 (1)**, 1 (1) |
| 23 Z / Z | 2 (1), 9 (2), 2 (1), 0 | 3 | 1 (1), 0, 0, 0 |
| 24 – / – | 2 (2), 280 (5), 185 (2), 0 | 137 | 2 (2), 0, 0, 667 (9) |
| 25 Z / Z | 0, 0, 0, 0 | 20 | 0, 0, 0, 0 |
| 26 Z / – | 13 (4), 0, 3 (1), 114 (2) | 114 | not sampled |
| 27 Z / – | 0, 19 (1), 0, 0 | 0 | 0, 1 (1), 0, 0 |

- Reductio drift arms excluded 0 proofs in every arm and round (nothing to exclude): they are
  plain neighbours-only EI, and s21's required-pool rate still reached 5.0·10⁻⁴ at round 6
  (2.0·10⁻⁵ at round 8). s21 is also the one reductio `mix` igniter, so the `mix` and `drift`
  evidence for reductio is one Stage-1 draw, not two; s22 is 4 hits on a single target at one
  checkpoint (1.3·10⁻⁵, at the threshold) and 1 at the next.
- Depth-3 drift: with every depth-3 proof excluded, depth-3 proofs of neighbours still appear
  after training on flat proofs only (s21 round 3, s23 round 2, s25 round 2); the required-pool
  rate becomes non-zero but small and non-monotone. A checkpoint with ≥ 3 hits exists in 4 of
  6 zero-rate arms (s21, s23, s26, s27); at the round-8 checkpoint itself in 1 of 6 (s26).
- Hit counts are concentrated on single targets and not monotone in the round: depth-3 s22's
  112 hits are all on `d3req_292`; reductio s24's 667 are 533 on one target; reductio s21's
  151 are 96 on one target; "up to 114" (depth-3 s26) is 2 targets. A hits-per-300k threshold
  is dominated by one target's rate at one checkpoint.
- Held-out greedy at round 8 (of 5,000) is highest in the igniting `mix` arms (depth-3 s23
  4,702 vs 4,418 for its `req` arm; reductio s21 4,527 vs 4,496) and within ± 60 of the `req`
  arm's value everywhere else. There is no pre-RL held-out number per draw in the artefacts.

**What stands (reviewer's verdict).** (1) Pool composition is sufficient to switch ignition on
and off in both classes: `req` 1 / 16 vs `mix` 9 / 16 over 16 draws and two patterns. The
zero-rate `req` half (0 / 11) could hardly have come out otherwise — with no reward there is no
training step, so a `req` arm is 76,800 more samples from a model that gave 0 in 600,000 — so
the conclusion leans on the 16-draw comparison. (2) Reductio s21 is a clean case of a pattern
absent at 0 / 600k appearing under training that never rewarded it. (3) No required target was
ever solved off-pattern; the oracle labels held against 27.6 M coverage samples and the ≈ 4.9 M
samples of the 48 arms.

**Reworded (reviewer's wording replaces the executor's).**

| executor (`round3_run1.md`, `numbers.md`) | reviewer |
|---|---|
| "300 pattern-free neighbours"; "With neighbours both patterns ignite from zero" (depth-3) | the depth-3 neighbours are pattern-optional and the models used the option: the pattern is first rewarded where it is optional and cheap, then carries to where it is required — a ladder through pattern-*bearing* neighbours in 3 of 3 cases, and from a non-zero neighbour base rate in 1 of 3 (s26). "The prior moves before any pattern proof is rewarded" is true of the `drift` arms and of reductio `mix`, not of depth-3 `mix` |
| "The asymmetry was the pools: expert iteration composes an absent pattern only through rewarded neighbours" | "Pool composition is sufficient to switch ignition on and off in both classes (1 / 16 vs 9 / 16). For reductio the composed step is one appended `DN` on two of 16 schemata; for depth-3 the neighbours rewarded the pattern itself before any required target was solved." The class asymmetry is not explained away: the two classes were not given the same kind of neighbour, the extents differ (239–255 vs 52–53 of 300), and "only" needs a qualifier (reductio s26 ignited on `req`) |
| "Clause (2) dies by its drift criterion (two zero-rate reductio arms reached ≥ 10⁻⁵)" | by the letter, yes; by weight, one draw clearly (s21) and a second at threshold (s22: 4 hits, one target, one checkpoint) |
| "Clause (3) dies as pre-registered" | correct for the pre-registered criterion, whose `req` half was close to guaranteed for zero-rate draws; lean on the 16-draw comparison instead |
| hit rates per 300k ("up to 114"; the drift figure) | quote every rate with its target count; rates are dominated by single targets and fall back to 0 at the next checkpoint in several arms |
| E5 depth-3 "held" (`numbers.md`) | held under one reading of "by round 8" (any checkpoint ≤ 8: 4 of 6), missed under the other (the round-8 checkpoint: 1 of 6); both are disclosed in `log.md` |

**Not supported as written.** Nothing in the write-up is contradicted by the files. The one
reading the reviewer does not let through is the symmetric description of the two neighbour
pools as pattern-free: the mechanism in the answer paragraph depends on it for depth-3, and
for depth-3 the run contains no arm that tests it (`mix` rewarded depth-3 proofs on neighbours;
`drift` excluded them everywhere and never offered a required target as a training target).

**Pre-registered expectations vs outcome** (`8c318e7`, 05:09:48 UTC, before any pod).

| id | expected | observed (reviewer) | verdict |
|---|---|---|---|
| n₀ | ≈ 3–5 of 8 (depth-3), ≈ 4–5 of 8 (reductio) | 6, 5 | depth-3 one above the range |
| E1 | ≥ 250 required@8 and ≥ 250 reductio neighbours; ≥ 90 % with a 9-line alternative, ≤ 5 % none; 0 oracle inconsistencies; every draw solves ≥ 40 % of its `mix` neighbours | 300 and 300; 96.0 % / 3.7 %; 0; depth-3 s25 35.3 % | held except the 40 % floor in 1 of 16 draws |
| E2 | zero-rate depth-3 `req`: 0 of n₀ ignite, ≤ 5 targets; non-zero ignite ≥ 80 % by round 5 | 0 of 6, 0 targets in every arm; non-zero 0 of 2 | first clause held; second **miss** |
| E3 | zero-rate depth-3 `mix`: ≥ 40 % ignite, only after ≥ 1 round of neighbour-only successes | 3 of 6, each after ≥ 1 such round | held — but the neighbour successes include depth-3 proofs (above) |
| E4 | zero-rate reductio `mix`: 1–2 of n₀ (standing rule: 0); non-zero ignite on both pools | 1 of 5; non-zero `mix` 3 of 3, `req` 1 of 3 | proposer's number held, the rule's 0 wrong; "both pools" **miss** |
| E5 | depth-3 drift ≥ 3 hits / 300k by round 8 in ≥ 2 of n₀; reductio drift ≤ 1 hit at every checkpoint in every zero-rate arm | depth-3 4 of 6 at some checkpoint, 1 of 6 at round 8; reductio violated in 2 of 5 (s21 151 and 6, s22 4) | depth-3 depends on the reading; reductio **miss** |
| E6 | required targets solved without the pattern ≤ 5 % (depth-3), 0 (reductio) | 0 in all 48 arms | held |
| falsifier, clause (3) | dies if E2 holds | E2's zero-rate clause holds (0 of 6 on `req` vs 3 of 6 on `mix`, same draws and seeds) | met |
| falsifier, clause (2) | dies if ≥ 2 zero-rate reductio draws ignite on `mix`, or any zero-rate reductio `drift` arm reaches ≥ 10⁻⁵ | 1 igniter (not 2); drift criterion met by s21 (5.0·10⁻⁴) and, at threshold, s22 (1.3·10⁻⁵) | met by the drift criterion |
| cost | ≈ 26 pod-hours ≈ $13, three RTX 3090s | 13.3 pod-hours ≈ $6.5, four A40s | under |

Figures: the executor's `figures/r3_1_ignition_vs_rate.png`, `r3_1_curves.png` and
`r3_1_drift.png` are
[on the run branch](https://github.com/dan-pandori/nd-takehome/tree/e5bfcc7/figures). The
review did not check them point by point and asks that the drift figure carry target counts
next to its rates, so they are not copied here.

## Limitations and next step

- "Zero-rate" was defined on the required pool only. No pre-RL depth-3 rate was measured on
  the neighbour pool; s26's non-zero neighbour rate is read off round-1 samples of its arms.
- The depth-3 mechanism is untested: there is no `mix` arm with pattern proofs excluded on the
  neighbour stratum only. The two classes did not get the same kind of neighbour (depth-3:
  pattern-optional, 72 of 300 with a shorter depth-3 proof; reductio: pattern-free, 2 of 300
  with a reductio shortest proof).
- Reductio ignition from zero is one draw (s21), which carries both the `mix` and the `drift`
  evidence, and reaches only the two append-`DN` schemata (52 of 300); the required pools are
  asymmetric in length (depth-3 7–8 lines; reductio 7–10 with 52 targets at 7).
- 3 / 6 vs 1 / 5 has no power to compare the classes' ignition rates (p = 0.55).
- Drift rates are bursty, single-target and non-monotone; reductio s26's drift arm was not
  sampled.
- Disclosed deviations (`log.md`, nine): A40s instead of RTX 3090s and co-scheduling of samples
  with arms on the 48 GB cards; `drift` arms for all 16 draws; reductio s1 / s2 not added; the
  05:50 amendment; bucket sub-directory `r3_1`; the kill switch left untouched; both readings
  of E5; the write-up's file name. The executor overwrote round 2's `run1.md` in the working
  tree while drafting and restored it before any commit.
- Not re-derived by the reviewer: `minlen.py` / `intuit.py` labels and the candidate-stage
  counts (472 None of 1,200; 614; 1,264; 759), held-out greedy, the dollar figure, bucket
  contents, the drift arms' training mixes.

Next measurements (reviewer): (1) **one arm settles the depth-3 mechanism**: `mix` with
`--exclude_pattern depth3` applied to the neighbour stratum only (a depth-3 proof of a
required target is trained on, one of a neighbour is dropped) on the six zero-rate draws, same
seeds, 8 rounds — six arms ≈ 3.5 pod-hours ≈ $2; if s21 and s23 still ignite, depth-3 composes
through genuinely pattern-free neighbours as reductio s21 did; if not, the depth-3 ladder is
the pattern's own non-zero rate on easier targets and the class asymmetry survives in a new
form (reductio composed from pattern-free parts by one appended step; depth-3 only ever
amplified). With it, the pre-RL depth-3 rate on the *neighbour* pool (k = 2,000 × 300) for all
eight draws. (2) Reductio: the executor's proposed 16 more seeds with `mix` arms only (≈ $5,
`QUESTIONS.md` 09:04) answers "is 1 / 5 narrower than 3 / 6"; whether the ladder reaches beyond
append-`DN` needs neighbours for the 8–10-line schemata (the embedded reductio box minus its
`DN`, as an intermediate line) and a per-schema table in every reductio result from now on.
(3) Executor's second proposal: start `mix` from the drift arm's round-6 vs round-8 checkpoint
to test whether ignition depends on a burst coinciding with required targets being in the pool.
