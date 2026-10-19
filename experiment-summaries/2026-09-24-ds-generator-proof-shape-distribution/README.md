---
written_on: 2026-09-24
written_by: agent:claude
---

# ds-generator: does the generator's proof-*shape* distribution buy RL readiness or textbook transfer? (proposal 10, run 2)

Status: completed 2026-09-24 09:17 UTC (autonomous executor, two sessions: 2026-09-22 and a
2026-09-23 resume after a Claude usage-credit exhaustion) and independently reviewed
2026-09-24 09:17 UTC. Review verdict: **no hard-constraint violation, nothing quarantined.** The
reviewer re-derived every quantity with its own tooling (own parser, start-index normaliser,
pattern predicates, renaming-class canonicaliser, ladder/coverage/dial recounters, term-size
counter) and found **zero differences** from `artifacts/dsg/summary.json`, `numbers.md` and
`data/dsg/README.md` across 18 coverage runs, 12 dial cells, 11 ladder runs, 6 held-out
evaluations and 465,000 training records. Its findings are about **wording, an unreported gate
rate, and a materially wrong cost figure — not arithmetic.** Numbers and wording below are the
reviewer's where they differ from the executor's. Run id `ds-generator`.

Review:
[`review_ds-generator.md`](https://github.com/dan-pandori/nd-takehome/blob/c969edd/review_ds-generator.md)
(commits `65bef8e` phase-1 blind recount, `c969edd` phase-2 compare and verdict). Executor
write-up:
[`run_ds_generator.md`](https://github.com/dan-pandori/nd-takehome/blob/474e322/run_ds_generator.md),
numbers `numbers.md` § ds-generator, log `log.md` §§ 2026-09-22 / 2026-09-23–24, open questions
`QUESTIONS.md`, all on branch `dan_ds-generator` of `dan-pandori/nd-takehome`
(**read the review first**: the executor's files were not edited after the review, so they still
carry the wording the verdict asks to change). Pre-registration:
[`preregistration/ds-generator.md`](https://github.com/dan-pandori/nd-takehome/blob/474e322/preregistration/ds-generator.md)
(`5a6acd8`, resume addendum `2733373`). Proposal:
[`docs/proposals/2026-09-22-dataset-styles.md`](../../docs/proposals/2026-09-22-dataset-styles.md);
brief
[`dataset-styles/BRIEF_ds-generator.md`](../../docs/proposals/dataset-styles/BRIEF_ds-generator.md).

Question: holding composition (flat 31,000 per pruned length 2–6), rendering (`lean_seq`), model,
schedule, cap 6, set size (155,000) and the depth-3 f = 0 filter fixed, does changing the **shape
distribution** of the generated proofs — `ORE` branches that take steps and open boxes (G1), and
on top of that fully goal-directed strict generation with no lazy premises (G2) — change held-out
accuracy by length and RL readiness (depth-3 / required-reductio base rates at pass@2,000,
EI − frozen after 4 dial rounds, ladder T1 `L*`, textbook solves per schema)?

## Configuration

- **Arms** (two Stage-1 seeds each; every set 155,000 proofs, flat histogram, cap 6, depth-3
  excluded in written and pruned form): **C0 control** — the unmodified lean-format a1 set and its
  bucket checkpoints (`ckpts/lf/stage1_a1_seq_s{0,1}.pt`, md5 `9bde44c0…` / `fc27e52d…`, reviewer-
  verified byte-identical, **not retrained**). **G1** — `gen.py --ore_steps 3 --ore_boxes` (boxes
  allowed inside `ORE` branches under the global depth cap 3), raw pool, uniform draw per length,
  0 % fill from any other pool. Pre-registered as a **null manipulation**: at cap 6 the knob
  *lowers* `ORE` share (1.48 % → 0.98 % in the pre-run probe) because an `ORE` whose first branch
  takes a step or opens a box is ≥ 7 lines after pruning and is cut by the cap — decided and
  disclosed before any pod. **G2** — `gen.py --ladder --drop_contra --ore_boxes`, i.e. the same
  generator settings that built the ladder's transfer pool (50 % goal / 50 % forward, strict, no
  lazy `( Z > G )` fallback, `ore_steps` 3, contradictory-premise theorems dropped), restricted to
  ≤ 6 pruned lines, plus G1's boxes-in-`ORE` knob; **24.3 % of G2 (37,716 records, all at lengths
  2–3) is filled from G1's own pool** because the strict generator's yield at those lengths is too
  low to fill a flat 31,000/length set inside the 2-pod-hour generation budget (disclosed per bin).
  The brief's *literal* G2 (100 % goal mode, no lazy fallback at all) was found degenerate in the
  pre-run probe (84 % reductio, 30–160 proofs per 80,000 tries in the short bins) and run only as a
  discarded probe; both deviations from the brief's *design* (not its question) were pre-registered
  and put to Dan in `QUESTIONS.md` with a stated default.
- **Sets, as re-derived by the reviewer from all 465,000 records with independent code**: every
  shape, overlap and render-check quantity reproduces `data/dsg/README.md` exactly. Split
  disjointness by renaming class against all eight evaluation pools plus the ladder pools: **0**
  exact-`thm` and **0** renaming-class collisions for all three training sets. Ladder pools
  (`data/ladder/{transfer,rl_targets}`) byte-identical to ladder-A's (`git hash-object` match).
- **Model and schedule.** 3.3 M-parameter from-scratch GPT, `lean_seq` Lean surface format;
  `train.py --mode lean_seq --steps 6000 --bs 128 --cap 6 --seed {0,1}`.
- **Checker.** A proof counts only if **Lean 4.34 and `nd_verify` both accept** it
  (`nd2lean.py --check`, byte-identical to `origin/dan_lean_format`); `nd_verify`,
  `lean_gate.py`, `train.py`, `expert_iter.py`, `ladder_ei.py` all unchanged (verified by diff).
- **Commands.** Held-out greedy: `eval_set.py --k 1 --temperature 0 --batch 512` on
  `data/p2/heldout.jsonl` (5,000). Coverage, pass@2,000: `coverage.py --k 2000 --temperature 0.8
  --seed 0 --batch 1000` on `targets_depth3` (1,000), `r3_1/depth3_req` (300),
  `targets_reductio_req` (300). Dial: `expert_iter.py --rounds 4 --k 32 --temperature 0.8 --batch
  768` (EI and `--no_train` frozen). Ladder: `ladder_ei.py` T1 and `--no_train` frozen, 8 rounds ×
  k 32, `--batch 512 --max_new 512` — the resume addendum's proposed 4,096/384 was tested by
  `dsg_regress.py` (128 rows, token-identical) and **reverted to 512/512** because compaction
  flipped 1 row of 128; the batch was held fixed across all six models and both seeds either way,
  which is what the addendum's requirement was for.
- **Gate 0.** Pre-registration `5a6acd8` committed 2026-09-22 06:09:12 UTC, first `dsg-*` pod
  06:10:32 (78 s). Resume addendum `2733373` committed 2026-09-23 21:06:00 UTC, first resume pod
  21:07:18 (78 s). **Both hold.**
- **Pods and cost — corrected by the reviewer.** *Session 1* (2026-09-22): `dsg-1`/`dsg-2`/`dsg-3`,
  RTX 3090 $0.50/h. The executor's session was killed by a **Claude usage-credit exhaustion at
  ≈ 10:33 UTC** (the same incident that stopped the sibling run `ds-composition` three minutes
  later); the three pods then billed **unattended for ≈ 19 h** until the account's balance floor
  caught them at ≈ 05:20 on 2026-09-23. RunPod's billing API (not `podbudget`, which has no
  data before 2026-09-23) gives **$34.79** for this phase (≈ 23 pod-hours each at $0.50/h), of
  which **≈ $6.6 is attended, ≈ $29 is idle**. `numbers.md`'s figure for this phase, "≈ $2.5 of its
  own $20 ceiling", is **not supported, by a factor of ≈ 14** (§Finding A of the review). Their
  local `artifacts/` and `ckpts/` (except the uploaded `artifacts/dsg/` tree and the two data sets)
  were lost to a host cleanup; `ckpts/dsg/stage1_g{1,2}_s{0,1}.pt` were never uploaded and are
  gone. *Session 2* (resume, 2026-09-23 21:07 – 2026-09-24): two pods, one seed each, all three
  arms per pod (3090/A40 out of stock at every tier) — `dsg-1` **RTX A6000 48 GB**, 11.56 h,
  **$5.93** billed (not the $5.78 `podbudget` prints from its no-A6000 fallback of $0.50/h; the
  real rate is $0.53/h, confirmed in `log.md` and by RunPod's `list-pod-billing`), and `dsg-2`
  **RTX 4090 24 GB**, 9.78 h, **$7.27** billed ($0.74/h). Resume-phase total **21.34 pod-hours,
  $13.20** of the $14 budget (not the executor's $13.02) — still inside the ceiling. **Run total,
  both phases: ≈ $48**, against the $20 + $14 = $34 declared across the two pre-registrations; see
  the [spend ledger](../../docs/infra/spend.md) for the derivation and the
  [week-39 digest](../../docs/digests/2026-39.md) for the standing decision this and
  `ds-composition`'s matching incident raise. All pods deleted; no `dsg-*` pod at the review's
  `list-pods`.
- **Artifacts.** `hf://buckets/dan-pandori/nd-rl/ds-generator/{ckpts,artifacts,data}` (present and
  verified by the reviewer).

## Observations

All values are the reviewer's independent recount unless noted; **every one reproduces the
executor's** (18 coverage runs × all fields and strata, 12 dial cells, 11 ladder runs × solves /
`L*` / per-bin / per-schema / distinct proofs, 6 held-out evaluations — 0 differences from
`artifacts/dsg/summary.json`). The reviewer's own re-check of 450 counted proofs (150 per arm)
through the unmodified `nd2lean.py --check`: 450 / 450 agree, 60 / 60 theorem-swapped negatives
rejected, 0 `sorry`/`decide`/`simp`-class tokens in 150 translated sources and 5,091 stored Lean
texts. The executor's own pass over 57,013 counted proofs: 57,013 both-accept, 0 disagreements.

### What stands

- **The transfer wall is not the shape distribution.** G2 *is* the ladder transfer pool's own
  generator at cap 6, so if shape were the wall its base model should read that pool far better.
  Frozen ladder solves of 2,285 (8 × 32, equal attempts), pre-registered at 450–760: **C0 158 /
  114, G2 44 / 125** — one seed 72 % below the control, one 10 % above, against a predicted +50 to
  +150 % on both. G1 (the pre-registered null, whose shape table matches the control's up to
  sampling noise: `ORE` 0.89 vs 1.46 %) has a **one-seed** frozen ladder of 170 (`la_frozen_g1_s0`
  was a deliberate, dated, disclosed drop for budget — see Limitations) — larger than either C0
  seed, which is itself inside this run's measured noise floor (below).
- **The textbook wall is not the rule shape.** Of 19 schemata × 40 T1 solves, 10 are at 0 in every
  arm and seed; **the pre-registered falsifier fires on every seed of both arms**: G1 and G2 leave
  17, 18, 18, 19 of 19 schemata at ≤ 2 solves. The brief's predicted `ORE`-needing openings
  (constructive dilemma, De Morgan, distribution families) are **0 in all six arm-seeds**.
  Disjunctive syllogism (also `ORE`-needing) reaches ≥ 5 in C0 s1, G1 s0/s1 and G2 s0 alike — and
  the reviewer confirmed **every** DS solution in those cells uses `ORE` (21/21, 58/58, 38/38), so
  the control already has the capability; it is a draw effect, not a knob effect.
- **Finding 3 (RL amplifies what the base does) has a clean two-seed counter-example.** G2's
  frozen depth-3 acquisition is far below C0's (0.032 / 0.034 vs 0.163 / 0.116) yet its
  EI − frozen is larger on both seeds (+0.383 / +0.362 vs +0.255 / +0.292) — the pre-registered
  counter-example condition, met on both seeds.
- **The measurement noise is this run's real headline constraint.** Retraining Stage-1 from the
  same (set, seed) on different hardware moves held-out greedy by **−4.44 to +5.96 pp**, almost all
  in the 6-line bin; the byte-identical C0 checkpoint, new GPU, new decode path moves **+0.00 /
  +0.08 pp** on the same measurement — so the retrain spread is training-run variance, not a
  recovery or sampler artefact. G1's two Stage-1 draws of the control's *own* distribution differ
  by **1.9×** on required@8 coverage and **+49 %** on the frozen ladder (170 vs C0's 114 on the
  same seed) — a larger deviation than G2's +10 %, from a pool that is by construction the same
  distribution as the control. **Any claim in this run smaller than these margins (≈ ±50 % on the
  frozen ladder, ≈ 2× on coverage) is not supported by n = 2.**

### Headline tables (reviewer's values; every field reproduces `artifacts/dsg/summary.json`)

Held-out greedy overall (5,000 theorems; C0 not retrained, G1/G2 are 2026-09-23/24 retrains — the
2026-09-22 checkpoints were lost to the session-1 cleanup):

| model | overall | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| C0 s0 / s1 | 0.9088 / 0.8968 | 0.994 / 0.997 | 0.989 / 0.992 | 0.951 / 0.967 | 0.924 / 0.944 | 0.686 / 0.584 |
| G1 s0 / s1 (retrain) | 0.8606 / 0.8956 | 0.998 / 0.992 | 0.984 / 0.975 | 0.947 / 0.915 | 0.923 / 0.878 | 0.451 / 0.718 |
| G2 s0 / s1 (retrain) | 0.6078 / 0.6652 | 0.996 / 0.997 | 0.963 / 0.970 | 0.525 / 0.533 | 0.417 / 0.401 | 0.138 / 0.425 |

Coverage, pass@2,000 (T 0.8, k = 2,000; a sample counts iff `nd_verify` **and** Lean accept):

| pool (n) | C0 s0 | C0 s1 | G1 s0 | G1 s1 | G2 s0 | G2 s1 |
|---|---|---|---|---|---|---|
| `targets_depth3` (1,000) | 506 | 423 | 456 | 603 | 288 | 266 |
| `r3_1/depth3_req` required@8 (300) | 165 | 113 | 138 | **259** | 38 | 23 |
| `targets_reductio_req` (300) | 31 | 27 | 38 | 31 | **0** | **0** |

Depth-3 dial round 4 (EI vs frozen at equal attempts, k 32 × 4 rounds):

| model | EI acq | frozen acq | EI − frozen |
|---|---|---|---|
| C0 s0 / s1 | 0.418 / 0.408 | 0.163 / 0.116 | +0.255 / +0.292 |
| G1 s0 / s1 | 0.403 / 0.436 | 0.165 / **0.322** | +0.238 / +0.114 |
| G2 s0 / s1 | 0.415 / 0.396 | **0.032 / 0.034** | **+0.383 / +0.362** |

Ladder (8 rounds × k 32, `data/ladder/`, identical batch for all six models):

| model | T1 transfer / 2,285 | T1 `L*` | frozen transfer | frozen `L*` |
|---|---|---|---|---|
| C0 s0 / s1 | 890 / 965 | 12 / 11 | 158 / 114 | 9 / 9 |
| G1 s0 / s1 | 916 / 635 | 12 / 10 | — / **170 (s1 only)** | — / 9 |
| G2 s0 / s1 | 632 / 607 | 11 / 11 | 44 / 125 | 9 / 9 |

Textbook schemata at T1 reaching ≥ 5 solves (of 19; only non-zero schemata shown): contraposition
19/30/39/0/0/0 (C0 s0, C0 s1, G1 s0, G1 s1, G2 s0, G2 s1), disjunctive syllogism 4/21/38/23/38/0,
export 5/33/0/0/1/0; **schemata left at ≤ 2: 15, 16, 17, 18, 18, 19**.

### Rewordings the verdict requires

1. **Headline.** `run_ds_generator.md`'s "the manipulation is smaller than the noise" is right for
   the *readiness differences* and wrong for G2's manipulation itself, which is large (box depth
   23/45/32 % vs C0's 53/35/11 %, `IMPI`-final 70 % vs 34 %, mean premises 0.78 vs 1.41) and whose
   held-out effect (−24 to −30 pp) is 5× any noise measured here. **Corrected form:** "Both
   accounts are dead: the rule-shape account by its falsifier, the shape-distribution account by
   its prediction — G2 changes the training distribution profoundly and the transfer readiness not
   at all, and what differences remain are inside a between-pool noise floor the null arm measures
   at ±50 %."
2. **"G2 is at or below C0 on both seeds"** (frozen ladder) → "44 vs 158 on seed 0 and 125 vs 114
   on seed 1 — one far below, one marginally above, against a pre-registered +50 to +150 %."
3. **The ±15 % falsifier is "dead on the prediction, not on the equivalence."** Read literally
   ("dead if G2's frozen ladder is within ±15 % of C0 on both seeds") it does not fire — seed 0 is
   72 % below (outside ±15 %, wrong direction), seed 1 is +10 % (inside) — but the *substantive*
   conclusion (no support for the shape-distribution account) holds regardless; `log.md` already
   states the distinction, `run_ds_generator.md` does not.
4. **Label G1's frozen ladder (170) as the one-seed estimate it is** wherever it carries weight,
   including "G1 … is above both C0 seeds."
5. **Cost.** Correct the first-phase figure (§Finding A above, ≈ $34.79 not ≈ $2.5), the "$0.50/h
   billed" A6000 rate (§ Configuration above, $0.53/h, resume total $13.20 not $13.02), and
   `numbers.md`'s `git diff origin/main` command for `nd_verify` / `nd2lean.py` (not empty — the
   correct base is `origin/dan_lean_format`, which *is* empty; substance unaffected, `log.md`
   already uses the right base).
6. **Score the pre-registered held-out-retention row as the miss it is.** Round-1 → round-8 greedy
   retention was pre-registered at ±1 pp (C0, G1) / ±3 pp (G2); measured **+1.62 to +9.38 pp** on
   all six, in the favourable direction — never called a miss in the write-up.

### One repository-level finding, not specific to this run's counts

**The in-loop gate's Lean-vs-`nd_verify` disagreement rate is not reported, and it rose 22× since
the last run that measured it.** Of 19,775,560 gated samples (12,877,375 distinct checked by
both), **9,249 "Lean accepts, `nd_verify` rejects", 0 the other way** — 718 per million, against
`lean-format`'s 33 per million on a workload with fewer premises. The reviewer re-verified all
9,249 and classified them: **8,594 (92.9 %)** the model does not restate every `PR` line (the Lean
rendering binds premises as hypotheses and does not enforce restatement, `nd_verify` does), and
**655 (7.1 %)** `¬A ≡ A → False` definitional-unfolding kinds (`DN`, `NEGE`, `BOTE`/`Not.elim`,
`NEGI`, `ORI1`, `R`) — exactly the two looseness classes `QUESTIONS.md` flagged on 2026-09-21.
**No counted proof is affected** (both checkers are required; confirmed by the executor's 57,013
and the reviewer's own 450), but the run's own standard is "report agreement with `nd_verify`",
and this rate belongs in `numbers.md`'s checker section. Relevant to proposal 9 (retiring
`nd_verify` for a Lean-only allowlist check): "Lean is the checker of record" should be read as
"Lean *and* `nd_verify`" until `nd2lean.py`'s `BOTE` rendering and premise-block handling close
these two classes.

### Minor

`run_ds_generator.md` is 528 words against the brief's ≤ 400-word limit (the two required figures
are present). There is no render check on file for C0 — its set is the unmodified lean-format a1
set, inherited rather than missing, but `data/dsg/README.md`'s render-check table should mark the
row "inherited" instead of leaving it absent.

## Limitations and next step

- **`la_frozen_g1_s0` was never run** (`args.json` only, no output) — a dated, reasoned budget drop
  disclosed in `QUESTIONS.md`, but not the pre-registration's stated drop order. Consequence: the
  run's most informative noise-floor cell (G1's frozen ladder) is a one-seed estimate.
- **Session 1's cost overrun (≈ $29 idle, §Finding A) is the same failure mode as `ds-composition`'s
  same-day incident**: a Claude usage-credit exhaustion killed the executor session and
  `pod_watchdog` did not delete the pods; the account's balance floor, hours later, did. The
  `pod_budget_watch` guard introduced the next day (`efficiency` run, 2026-09-23) would have caught
  this at ≈ 6 pod-hours instead of ≈ 69; see the week-39 digest for the standing decision.
- **The checker looseness rate rose 22× and is undocumented** — not a correctness issue for this
  run (both checkers are required for every count) but a live input to proposal 9's Lean-only
  checker design.
- Model labelling is otherwise exemplary: every table names checkpoint, parameter count, format,
  training set and retrain status, and the executor's own C0-ladder-band mislabelling (traced to a
  different model, `stage1_full_seq_s0` on `data/train.jsonl`) was self-caught and corrected in
  `log.md` before the reviewer found it independently.

Next step (reviewer's ranked list): (1) **make the noise floor a measured quantity** — four
independent raw pools from the same generator × two Stage-1 seeds, frozen ladder only (no EI),
≈ 8 cheap jobs — this run's most valuable number (C0 vs G1's +49 % / 1.9× spread) was accidental,
and until a real error bar exists no dataset-style run at n = 2 can resolve anything smaller than
±50 %; (2) **finish G1's frozen ladder, seed 0** (≈ $1.3, ≈ 2 h) — turns the noise-floor estimate
from one draw into two; (3) **the length horizon, not the shape, is the remaining lever named in
the brief** — both walls survived a shape manipulation that moved the training distribution as far
as the cap allows; `ds-composition`'s cap axis and the `L_true` ≥ 13 result (0 solves in all eleven
ladder runs here, T1 and frozen alike) say where the ceiling sits; (4) **fix the translator's two
looseness classes** (`False.elim na` for `BOTE`; mirror the "every premise restated" rule in the
inverse grammar) before treating "Lean is the checker of record" literally.
