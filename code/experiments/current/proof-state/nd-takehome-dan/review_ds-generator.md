# Review of run `ds-generator` — reviewer (independent session)

Reviewer: agent:claude, role reviewer, 2026-09-24. Policy `AGENT_POLICY.md`; role brief as given.
Phase 1 was done **blind** in `~/review/ds-generator/` (a copy of the run's repository with the executor's
write-ups — `run*.md`, `*_summary*.md`, `campaign.md`, `followup.md`, `ignition.md`, `phase*.md`, `numbers.md`,
`STATUS*.md` — removed). My recount code is in `~/review/work/` (`rnorm.py` parser + start-index normaliser,
`rpat.py` pattern predicates + dependency pruner, `rkey.py` renaming-class canonicaliser, `ladder_recount.py`,
`cov_recount.py`, `dial_recount.py`, `heldout_recount.py`, `setcheck.py`, `termsize.py`); none of it is the
executor's. Inputs: `BRIEF_ds-generator.md`, `preregistration/ds-generator.md`, the code, `data/`, `artifacts/dsg/`,
`artifacts/lf_control/`, and `data/dsg/train_g{1,2}.jsonl` re-fetched from the run's bucket.

---

# §Recount

## 0. Model labels (every number below)

| tag | model |
|---|---|
| **C0** | 3.3 M-parameter from-scratch GPT, `lean_seq` Lean surface format, Stage-1 `train.py --mode lean_seq --steps 6000 --bs 128 --cap 6`, set `data/p2/train_depth3_f0_a1.jsonl` (155,000, flat 31k/length 2–6, depth-3 f = 0), seeds 0/1. Checkpoints `ckpts/lf/stage1_a1_seq_s{0,1}.pt`, md5 `9bde44c0b6c7580951656bf57aec3e43` / `fc27e52d017e5a361b3232fcd13f4ddc` — **verified by me**, byte-identical to the values the pre-registration records. Not retrained. |
| **G1** | same architecture / format / schedule / seeds, set `data/dsg/train_g1.jsonl` (155,000). Checkpoints `ckpts/dsg/stage1_g1_s{0,1}.pt` are **2026-09-23/24 retrains**, not the 2026-09-22 originals. |
| **G2** | same, set `data/dsg/train_g2.jsonl` (155,000). Checkpoints likewise retrained. |

Consequence I verified and carry through the tables: **held-out and ladder** numbers for G1 / G2 are on the
**retrained** checkpoints; **coverage (pass@2,000) and the depth-3 dial** are on the **2026-09-22 originals**.
They are different training draws of the same (set, seed) — see §5.

## 1. Hard constraints

| constraint | result |
|---|---|
| `nd_verify` hash equals `origin/main`'s | **PASS.** `nd_verify/__init__.py` md5 `4d48a3f00672a5547af4f41a238ebe0a`, `nd_verify/verify.py` md5 `321932cdf4591d6ec7b2b48f2f793cac`; identical to `origin/main` (`c55f7987`). `git diff origin/main...dan_ds-generator -- nd_verify/` is empty. |
| `artifacts/TEST_RUN_DONE` unchanged | **PASS.** md5 `e2eea349c5320ccb597c2279a0391991` on both `origin/main` and `dan_ds-generator`; last touched by `ca93f83` (2026-09-15), no commit of this run touches it. No test-file read anywhere in the repo (`grep -rn 'test\.jsonl'` → nothing). |
| no evaluation file read in training code | **PASS.** `train.py` opens `--heldout` only to compute a validation loss (lines 93, 116–120); it is never batched into the optimiser. `expert_iter.py:98–99` and `ladder_ei.py:176–180` build `eval_keys` from transfer ∪ held-out ∪ targets ∪ `targets/validation_36.jsonl` and *exclude* those classes from the RL training data (`:153`, `:238`). `grpo.py` was not used in this run. |
| `nd2lean.py`, `lean_gate.py`, `train.py`, `expert_iter.py`, `ladder_ei.py` unchanged | **PASS.** `git diff origin/dan_lean_format...dan_ds-generator` touches only `coverage.py`, `gen.py`, `make_coverage_sets.py`, `model.py`, `sample.py`, the new `dsg_*.py` / `pod/dsg/*` — none of the five. |
| cap 6 on supervised data | **PASS.** My own parser over all 465,000 records of the three sets: **0** records with written length > 6 or pruned length > 6. |
| depth-3 excluded, pruned *and* written | **PASS.** 0 depth-3 records in all three sets under my own depth counter, on the written proof and on the dependency-pruned proof. |
| no hand- or LLM-written training proofs | **PASS** as far as files can show: every record carries generator provenance (`src`, `gen_last_rule`), and my independent render check (§6) round-trips and re-verifies them. |
| ladder pools byte-identical to ladder-A's | **PASS.** `git hash-object data/ladder/transfer.jsonl` = `e0524d0a84…`, `rl_targets.jsonl` = `69233bcaa2…`; identical blobs to `origin/dan_ladder_a`. |
| pre-registration before the first pod (gate 0) | **PASS, both phases.** Prereg committed `5a6acd8` 2026-09-22 06:09:12 UTC; first `dsg-*` pod in `~/pods.log` 06:10:32 UTC. Resume addendum `2733373` 2026-09-23 21:06:00 UTC; first resume pod 21:07:18 UTC. |
| pods deleted, spend | **PASS.** `runpodctl pod list` is empty. `podbudget ds-generator` → 21.34 h, $13.02 against a $14 / 28 h ceiling. (`QUESTIONS.md` records that `podbudget`'s dollar column understates real billed rates; the executor's own tracked figure was $8.51 at 08:25 with a $13.0 projection, consistent.) |

**No quarantine.**

## 2. The sets (my own code, all 465,000 records)

Every quantity below is mine, from `~/review/work/setcheck.py`, and every one **reproduces** `data/dsg/README.md`.

| quantity | C0 (a1) | G1 | G2 |
|---|---|---|---|
| n; pruned-length histogram 2/3/4/5/6 | 155,000; 31,000 × 5 | 155,000; 31,000 × 5 | 155,000; 31,000 × 5 |
| records over cap 6 (written / pruned) | 0 / 0 | 0 / 0 | 0 / 0 |
| depth-3 (written / pruned) | 0 / 0 | 0 / 0 | 0 / 0 |
| box depth 0 / 1 / 2 / 3 (%) | 53.2 / 35.4 / 11.4 / 0 | 52.2 / 34.7 / 13.1 / 0 | 23.0 / 45.4 / 31.6 / 0 |
| proofs with `ORE` (n; %) | 2,256; **1.46** | 1,374; **0.89** | 524; **0.34** |
| box inside an `ORE` branch (n; %) | 49; 0.032 | 52; 0.034 | 23; 0.015 |
| premises 0/1/2/3 (%); mean | 12.3/39.6/42.6/5.5; 1.412 | 12.6/40.3/42.1/4.9; 1.394 | 35.5/50.7/13.7/0.0; 0.782 |
| final rule IMPI / IMPE / DN / ORI / ANDI (%) | 34.0 / 16.2 / 6.4 / 24.9 / 12.3 | 38.1 / 15.8 / 5.6 / 24.3 / 10.8 | 70.0 / 7.2 / 5.8 / 13.3 / 1.8 |
| pattern proofs reductio / derived-`ORE` / depth-3 | 10,547 / 88 / 0 | 9,374 / 57 / 0 | 8,956 / 34 / 0 |
| mean lazy premises; contradictory-premise share (%) | not labelled; 6.06 (executor's) | 0.677; 6.12 | 0.294; 0.15 |
| `src` | – | 155,000 `main` | 117,284 `main`, **37,716 `fill` (24.3 %)** |

Two things the shape table says plainly and that no later claim may ignore:

1. **G1's `ORE` share is *lower* than the control's** (0.89 % vs 1.46 %) and its boxes-inside-`ORE` count is
   statistically the same (52 vs 49). The brief's G1 shape targets (`ORE` ≥ 5 %, box-in-`ORE` ≥ 1 %) are **missed by
   a factor of 5 and 30**. The pre-registration predicted exactly this and reclassified G1 as a null manipulation.
2. **G2 is 24.3 % G1-pool fill** — its 2-line bin is 72.8 % fill and its 3-line bin 48.8 % (`assemble_g2.json`,
   which I re-derived from the `src` field: 22,579 + 15,137 = 37,716). "G2" is the strict long generator only above
   length 3.

**Split disjointness by renaming class** (my own canonicaliser, cross-checked against the stored `key` field on
3,000 records: 3,000/3,000 agreement). Against **every** evaluation pool — `data/p2/heldout`, `targets_depth3`,
`transfer_depth3`, `targets_reductio_req`, `transfer_reductio_req`, `data/r3_1/depth3_req{,_transfer}`,
`data/ladder/{rl_targets,transfer}` — all three sets show **0 exact-`thm` and 0 renaming-class collisions**.
Disjointness holds. (The third, premise-order-insensitive diagnostic column differs from the executor's by 1–6
records per cell — e.g. G1 vs held-out 16 or 22 by my two variants against their 21 — because "premise-order
insensitive" can be defined as rename-then-sort or sort-then-rename. It is a diagnostic, not the criterion;
the criterion is 0 in both.)

## 3. Held-out greedy (`eval_set.py --k 1 -T 0`, 5,000 records; my own recount from the per-record `.jsonl`)

All six **reproduce** `artifacts/dsg/heldout2_*.json` exactly.

| model | overall | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| C0 s0 | 0.9088 | 0.994 | 0.989 | 0.951 | 0.924 | 0.686 |
| C0 s1 | 0.8968 | 0.997 | 0.992 | 0.967 | 0.944 | 0.584 |
| G1 s0 (retrain) | 0.8606 | 0.998 | 0.984 | 0.947 | 0.923 | 0.451 |
| G1 s1 (retrain) | 0.8956 | 0.992 | 0.975 | 0.915 | 0.878 | 0.718 |
| G2 s0 (retrain) | 0.6078 | 0.996 | 0.963 | 0.525 | 0.417 | 0.138 |
| G2 s1 (retrain) | 0.6652 | 0.997 | 0.970 | 0.533 | 0.401 | 0.425 |

**Retrain reproduction (pre-registered at ±0.5 pp): MISSED, by up to 12× the band.**
Original → retrain: G1 s0 0.8750 → 0.8606 (**−1.44 pp**), G1 s1 0.9400 → 0.8956 (**−4.44 pp**),
G2 s0 0.6256 → 0.6078 (−1.78 pp), G2 s1 0.6056 → 0.6652 (**+5.96 pp**). Almost all of it is the 6-line bin
(G1 s1 0.849 → 0.718; G2 s1 0.101 → 0.425). The same comparison on the **byte-identical** C0 checkpoints moves
+0.00 / +0.08 pp, so this is Stage-1 training-run variance, not a recovery or sampler artefact. This is disclosed
in `data/dsg/README.md`. Its consequence is the single most important thing in this run: **the seed-to-seed
(and retrain-to-retrain) noise on the 6-line bin is ±5 pp, and on the downstream RL numbers it is much larger.**

## 4. RL readiness

### 4a. Coverage, pass@2,000 (T 0.8, k = 2,000; count requires `nd_verify` **and** Lean)

Re-derived from the per-target `cov_*.s0.jsonl` with my own pattern predicates and my own stratification on
`min_lines_ub`. **Every field reproduces `artifacts/dsg/summary.json`: targets hit, hits-with-pattern, sample
counts, distinct proofs, distinct ≥ 8-line, distinct ≥ 8-line-with-pattern, distinct 7-line-with-pattern,
written-length histograms, all strata, `lean_checked` / `lean_rejected` — 0 differences.**

| pool (n) | C0 s0 | C0 s1 | G1 s0 | G1 s1 | G2 s0 | G2 s1 |
|---|---|---|---|---|---|---|
| `targets_depth3` (1,000) | 506 | 423 | 456 | **603** | 288 | 266 |
| `r3_1/depth3_req` required@8 (300) | 165 | 113 | 138 | **259** | 38 | 23 |
| `targets_reductio_req` (300) | 31 | 27 | 38 | 31 | **0** | **0** |
| — its 7-line stratum (52) | 31 | 27 | 38 | 31 | 0 | 0 |
| — its ≥ 8-line strata (248) | 0 | 0 | 0 | 0 | 0 | 0 |
| distinct ≥ 8-line depth-3 proofs | 179 | 96 | 153 | 384 | 42 | 24 |
| Lean rejections among `nd_verify` accepts | 0 | 0 | 0 | 0 | 0 | 0 |

Read on the seed axis: **G1's two seeds differ by 1.3× (d3) and 1.9× (req8)** on identically-distributed sets —
that is the noise floor for any arm comparison here. G2 is below C0 on both seeds on all three pools, and is at
**0 / 300 on required reductio on both seeds** (C0: 31, 27).

### 4b. Depth-3 dial, round 4 (EI vs frozen at equal attempts, `k` 32 × 4 rounds)

Acquisition re-derived by me as (theorems with ≥ 1 depth-3 proof in `found_4.jsonl`) / n, on start-index-normalised
proofs with my own depth counter. **All 36 fields reproduce `summary.json` exactly.**

| model | EI acq | frozen acq (base rate) | EI − frozen | EI targets solved | frozen targets solved |
|---|---|---|---|---|---|
| C0 s0 | 0.418 | 0.163 | **+0.255** | 650 | 358 |
| C0 s1 | 0.408 | 0.116 | **+0.292** | 646 | 295 |
| G1 s0 | 0.403 | 0.165 | +0.238 | 620 | 346 |
| G1 s1 | 0.436 | **0.322** | +0.114 | 662 | 497 |
| G2 s0 | 0.415 | **0.032** | **+0.383** | 640 | 228 |
| G2 s1 | 0.396 | **0.034** | **+0.362** | 600 | 202 |

**The pre-registered counter-example condition for Finding 3 is met, on both seeds.** The falsifier reads:
"Finding 3 gets a counter-example if an arm has a lower base rate than C0 **and** a larger EI − frozen on both
seeds." G2's frozen acquisition is 0.032 / 0.034 against C0's 0.163 / 0.116 (lower, both seeds) and its
EI − frozen is +0.383 / +0.362 against C0's +0.255 / +0.292 (larger, both seeds). The pre-registration expected
the opposite (G2's EI − frozen *smaller* than C0's by 0.03–0.15). **Miss, and an informative one.**
Note also that G1's two seeds bracket the whole effect (frozen acq 0.165 vs 0.322, EI − frozen +0.238 vs +0.114),
so the *magnitude* is not well determined by n = 2; the *sign* is, and it agrees on both seeds.

### 4c. Ladder (8 rounds × k 32, `data/ladder/`, batch 512, `max_new` 512 — identical for all six models)

My own counter: distinct solved transfer names from `found_transfer_8.jsonl`, min round per start-index-normalised
proof, `L*` = max L with ≥ 5 solved theorems at `L_true` ≥ L, textbook schemata joined from the pool file.
**Every field reproduces `summary.json`**, including the distinct-proof counts.

| model | T1 transfer / 2,285 | T1 `L*` | T1 targets / 4,495 | frozen transfer | frozen `L*` | frozen targets |
|---|---|---|---|---|---|---|
| C0 s0 | 890 | 12 | 2,830 | **158** | 9 | 1,214 |
| C0 s1 | 965 | 11 | 2,924 | **114** | 9 | 1,061 |
| G1 s0 | 916 | 12 | 2,832 | — *not run* | — | — |
| G1 s1 | 635 | 10 | 2,346 | **170** | 9 | 1,278 |
| G2 s0 | 632 | 11 | 2,282 | **44** | 9 | 561 |
| G2 s1 | 607 | 11 | 2,259 | **125** | 9 | 1,158 |

**`la_frozen_g1_s0/` contains `args.json` and nothing else — the job was never run.** This is a deliberate,
dated, reasoned drop recorded in `QUESTIONS.md` (2026-09-24 08:35 UTC: spend projection $14.2 with both seeds,
$13.0 with one). It is not in the pre-registration's stated drop order ("G1's second-seed ladder, then G2's
second-seed ladder, then G1 entirely"), and its consequence is structural: **G1's frozen ladder exists on one
seed only, so by this run's own two-seed rule no difference may be claimed from it.**

**The pre-registered falsifier, read literally, does not fire.** It says the shape-distribution account is dead
"if G2's frozen ladder solves are within ±15 % of C0 on both seeds". Measured: s0 **44 vs 158 = −72 %**
(far outside ±15 %, in the *wrong* direction), s1 **125 vs 114 = +10 %** (inside). So the literal condition is
not met — but only because one seed is far *below* the control, which is the opposite of what the account
predicts. The account predicted +50 to +150 % (450–760 solves). Nothing resembling that was measured on either
seed. **The substantive conclusion — no support whatever for the shape-distribution account of the transfer wall —
holds; the sentence "the falsifier fired" would not.**

**And the noise floor swallows the comparison anyway.** G1 is pre-registered as a null manipulation whose ≤ 6-line
output distribution equals the control's up to sampling noise (§2 confirms it: `ORE` 0.89 vs 1.46 %, everything
else within a point). Its one frozen seed solves **170 vs C0's 114 on the same seed = +49 %** — a *larger*
deviation than G2's +10 %, from a set that is by construction the same distribution. At this n, a between-pool
frozen-ladder difference of ±50 % is noise.

**Held-out retention across T1 (pre-registered at ±1 pp for C0/G1, ±3 pp for G2): MISSED on all six**, in the
favourable direction — round-1 → round-8 greedy: C0 +5.02 / +7.20 pp, G1 +9.38 / +3.58 pp, G2 +8.60 / +1.62 pp.

### 4d. Textbook schemata at T1 (760 textbook theorems, 40 per schema, 19 schemata)

Only schemata with a non-zero solve anywhere are shown; the other 10 are 0 everywhere.

| schema | C0 s0 | C0 s1 | G1 s0 | G1 s1 | G2 s0 | G2 s1 |
|---|---|---|---|---|---|---|
| contraposition | 19 | 30 | 39 | 0 | 0 | 0 |
| disjunctive_syllogism | 4 | 21 | 38 | 23 | 38 | 0 |
| export | 5 | 33 | 0 | 0 | 1 | 0 |
| demorgan_nand_to_or | 4 | 2 | 2 | 0 | 0 | 0 |
| contraposition_conv / import / peirce / peirce_sequent / dist_or_over_and_conv | ≤ 1 | ≤ 1 | ≤ 1 | 0 | 0 | 0 |
| **schemata at ≥ 5** | 2 | 3 | 2 | 1 | 1 | 0 |
| **schemata left at ≤ 2** | 15 | 16 | **17** | **18** | **18** | **19** |

**The rule-shape falsifier fires cleanly.** "The rule-shape account of the textbook wall is dead if G1 and G2
leave 17 / 19 schemata at ≤ 2": G1 leaves 17 and 18, G2 leaves 18 and 19. On every seed of both arms.

**G1's brief expectation ("≥ 1 `ORE`-needing schema — dilemma, De Morgan, distribution families — at ≥ 5 T1
solves") is a MISS**: constructive dilemma 0, all four De Morgan ≤ 4, all four distribution ≤ 1, excluded middle 0,
in every arm and seed. One caveat the write-up should not be allowed to exploit: **disjunctive syllogism is also
`ORE`-needing** and does reach ≥ 5 — but it reaches ≥ 5 in the **control** too (C0 s1: 21), and I checked that
*every* DS solution in C0 s1 (21/21), G1 s0 (58/58) and G2 s0 (38/38) uses `ORE`. So DS is not a capability the
generator change opened; the control already had it.

## 5. Which model each number is on

I checked this specifically, because the pre-registration gets it wrong once. The prereg's bracketed on-file
C0 ladder references — "frozen 220–320 **[304 / 309]**", "T1 700–900 **[794 / 839]**", "`L*` **[10]** / **[11]**",
"`L_true` = 7 bin **[75 / 118]**" — are **not** C0's. I pulled `origin/dan_lean_format:artifacts/lf/la_*_seq_s*/args.json`
and re-counted the found files myself: those numbers were measured on **`ckpts/lf/stage1_full_seq_s{0,1}.pt`,
trained on `data/train.jsonl`** (the take-home's own 155k set) — a different model on a different training set
from `stage1_a1_seq_s{0,1}.pt` on `train_depth3_f0_a1.jsonl`, which is C0 here. My recount of those lean-format
files gives exactly 794 / 839 (T1) and 304 / 309 (frozen), confirming the identification.

So **C0's pre-registered ladder band (220–320 frozen, 700–900 T1) was anchored to a different model**, and C0's
measured 158 / 114 frozen is outside it for that reason and not because anything went wrong. Commit `1cd1c18`
(2026-09-24 01:05 UTC), *"C0 ladder band was anchored to a different model; G2 retrain differs"*, shows the
executor found this during the run. Phase 2 must check that the write-up says so.

The other label I checked: the resume phase's `dsg_regress.py` verdict is **PASS** (`regress.json`: 128 rows,
token streams and decoded texts identical between the base path and the adopted default), with compaction
flipping 1 row of 128 — which is why `pod/dsg/jobs2.py` sets `ND_SAMPLE_COMPACT=0` and **keeps the originally
pre-registered `--batch 512 --max_new 512`** rather than the addendum's 4,096 / 384. I confirmed all twenty
`args.json` files: batch 512 and `max_new` 512 for every ladder job, 768 for every dial job, across all arms and
seeds. **Comparability is preserved** (the addendum's requirement was that the setting be held fixed across arms,
and it is), but the addendum's stated setting was not the one used; the reason is documented in `jobs2.py`.

## 6. The checker of record

- **My own re-check.** 150 counted proofs per arm (450 total), sampled uniformly at random across every counted
  source of that arm — ladder T1 and frozen `found_8` / `found_transfer_8` both seeds, dial EI and frozen
  `found_4` / `found_transfer_4` both seeds, and all three coverage pools both seeds — put through the
  **unmodified** `nd2lean.py --check`: **450 / 450 `nd_ok` ∧ `lean_ok`, 0 disagreements.**
- **Adversarial control.** 60 of those proofs re-paired with a different theorem statement: **60 / 60 rejected by
  both checkers.** I also scanned all 150 C0 translated Lean sources and all 5,091 stored literal `lean_text`
  strings for `sorry` / `admit` / `native_decide` / `decide` / `exact?` / `simp` / `tauto` / `omega`: **0 hits.**
  (`lean_gate.py`'s error regex would not see a `sorry` *warning*, so this scan is the substitute for the axiom
  check proposal 9 asks for; proposal 9's own `lean_check` with an allowlist and `#print axioms` does not exist in
  this repository yet, so I could not run it. The only classical axiom the translation can reach is
  `Classical.choice`, via `Classical.byContradiction` in the `DN` rendering.)
- **Executor's pass reproduces.** `pod/dsg/record.py` ran the same unmodified translator over **57,013** counted
  proofs (6 arm × seed files): 57,013 both-accept, **0** disagreements, 0 missing sources. Independently
  aggregated by me from `record_*.json`.
- **Coverage gate.** 0 Lean rejections among `nd_verify` accepts in all 18 coverage runs (re-derived by me).

**One real disagreement finding, which does not touch any count.** The in-loop gate logs
(`artifacts/dsg/gate_*.jsonl`, 19,775,560 samples, 12,877,375 distinct checked) record **9,249 samples where Lean
accepted and `nd_verify` rejected**, and **0** the other way. I re-verified all 9,249 with `nd_verify` myself and
classified the reasons: **8,594 (92.9 %) "premise block does not match declared premises"** — the model restates,
reorders or omits `PR` lines, which `nd_verify` enforces exactly and the Lean rendering (premises as hypotheses
`h1…hk`) does not — and **655 (7.1 %) rule-check failures**, all the `¬A ≡ A → False` definitional-unfolding kind
(`DN` of a `( ~ X ) > F` premise 131, `NEGE` 90, `BOTE`/`Not.elim` 241 across lines, `NEGI` 59, `ORI1` 61, `R` 27).
These are exactly the two looseness classes `QUESTIONS.md` flagged on 2026-09-21. **Because every count in this run
requires both checkers, no counted proof is affected** — which is confirmed twice over, by the executor's 57,013
and by my 450. The finding is about the *checker*, not the run: **on this workload Lean alone is looser than
`nd_verify` by ≈ 0.28 % of its accepts (9,249 / 3,278,149 both-accepted)**, and "Lean is the checker of record"
should be read as "Lean *and* `nd_verify`" until `nd2lean.py`'s BOTE rendering and premise-block handling are fixed.

## 7. Term sizes as well as line counts

Because "lines" is the run's only size axis, I re-derived a size measure that a model cannot game by
re-indentation: total formula-node count (atoms and connectives) over the dependency-pruned proof.

| model | T1 counted transfer proofs | mean term size | median | max | mean written / pruned lines |
|---|---|---|---|---|---|
| C0 s0 / s1 | 2,108 / 2,175 | 61.7 / 60.5 | 60 / 59 | 131 / 142 | 9.12 / 9.09 ; 9.01 / 8.95 |
| G1 s0 / s1 | 2,263 / 1,089 | 62.4 / 60.7 | 61 / 58 | 154 / 122 | 9.13 / 8.85 ; 9.02 / 8.81 |
| G2 s0 / s1 | 1,042 / 1,020 | 62.5 / 64.4 | 62 / 62 | 130 / 122 | 8.83 / 8.94 ; 8.79 / 8.90 |
| frozen (C0 / G1 / G2, all seeds) | 203 / 136 / 215 / 59 / 157 | 46–54 | 43–54 | 80–99 | 7.85–8.20 |

Term size tracks line count monotonically in every arm; no arm is producing structurally larger proofs at equal
line count, and no arm's advantage at any `L` is an artefact of the line metric.

## 8. Render check

`dsg_render_check.py` (proposal 9 protocol) re-run by me at a **different seed** (20260924, n = 1,000 / 500 Lean
positives / 200 theorem-swapped negatives) on both sets fetched from the bucket:

| set | round-trip identical | denoted verified at cap 6 | Lean positives accepted | Lean negatives rejected |
|---|---|---|---|---|
| G1 | 1,000 / 1,000 | 1,000 / 1,000 | 500 / 500 | 200 / 200 |
| G2 | 1,000 / 1,000 | 1,000 / 1,000 | 500 / 500 | 200 / 200 |

Reproduces `render_g{1,2}.json` (3,000 / 1,000 / 300 at seed 0). There is **no** render check on file for C0;
C0's set is the unmodified lean-format a1 set, so this is inherited rather than missing, but it should be labelled
as inherited.

## 9. Recount summary — what the files support

1. **Every number in `artifacts/dsg/summary.json` that I could re-derive, re-derives exactly** with independent
   code: 18 coverage runs (all fields, all strata), 12 dial cells, 11 ladder runs (solves, `L*`, per-bin,
   per-schema, distinct proofs), 6 held-out evaluations. **0 differences.**
2. **The generator-shape account of the transfer wall gets no support.** G2 — the transfer pool's own generator at
   the training cap — solves *fewer* of the transfer pool frozen than the control on one seed and 10 % more on the
   other, against a pre-registered +50 to +150 %. Its held-out collapses to 0.61 / 0.67 and its required-reductio
   base rate is 0 / 300 on both seeds.
3. **The rule-shape account of the textbook wall is dead** by the pre-registered falsifier: 17, 18, 18, 19 of 19
   schemata left at ≤ 2, and no `ORE`-needing schema in the brief's named families reached ≥ 5 anywhere.
4. **Finding 3 has a two-seed counter-example** on the depth-3 dial: G2 has a lower base rate *and* a larger
   EI − frozen than C0 on both seeds.
5. **The measurement noise is the headline constraint.** Retrained checkpoints of the same (set, seed) move
   held-out by up to 5.96 pp; G1's two seeds differ by 1.9× on required@8 coverage; G1's null pool differs from
   C0 by +49 % on the frozen ladder. Any claim in this run smaller than those margins is not supported by n = 2.
6. **One structural gap:** `la_frozen_g1_s0` was never run, so the noise-floor arm is a one-seed estimate.
7. **One checker finding:** 9,249 Lean-accepts-`nd_verify`-rejects in the gate stream, 0 the other way, none
   reaching a count.

*Phase 2 follows in this file, written after reading the executor's `run_ds_generator.md`, `numbers.md` and `log.md`.*

---

# §Phase 2 — claims vs the recount

Read after §Recount was committed (`65bef8e`): `run_ds_generator.md`, `numbers.md` § ds-generator (G1–G8),
`log.md` 2026-09-22 → 2026-09-24, `STATUS.md`, `data/dsg/README.md`, `QUESTIONS.md`.

## Claim-by-claim

| # | claim (source) | my independent value | verdict |
|---|---|---|---|
| 1 | Held-out greedy C0 0.909 / 0.897, G1 0.861 / 0.896, G2 0.608 / 0.665 (`run`, G1 table) | identical, all six, and every length bin and pattern cell | **reproduces** |
| 2 | Coverage pass@2,000, G2 table, 6 rows × 6 columns incl. per-sample rates and ≥ 8-line pattern proofs | identical in every field, plus every stratum and the written-length histograms | **reproduces** |
| 3 | Depth-3 dial G3: EI, frozen, EI − frozen, acq-by-round, held-out r4 (36 numbers) | identical, including the four-element acq-by-round vectors, which I re-derived from min-round bookkeeping in `found_4.jsonl` | **reproduces** |
| 4 | Ladder G4: 11 rows × (`L*`, solved, 8 `L_true` bins, textbook, schemata ≥ 5, held-out) | identical in every cell, including `c0 s1` frozen "export 7" and all 88 bin counts | **reproduces** |
| 5 | Textbook table, 19 schemata × 6 arm-seeds | identical; the 10 all-zero schemata are the 10 I list | **reproduces** |
| 6 | Retrain deltas G5: −1.44 / −4.44 / −1.78 / +5.96 pp; control +0.00 / +0.08 | identical, and I confirmed the C0 checkpoint md5s myself | **reproduces** |
| 7 | Checker of record G6: 57,013 counted proofs, 57,013 both-accept, 0 disagreements; 0 Lean rejections in all six coverage runs | identical by my own aggregation; plus my own independent 450-proof re-check (150/arm) through the unmodified `nd2lean.py --check`: 450/450, 0 disagreements; 60/60 theorem-swapped negatives rejected; 0 `sorry`/`decide`/`simp`-class tokens in 150 translated sources and 5,091 stored Lean texts | **reproduces, and independently confirmed** |
| 8 | Sampler equivalence G7: base vs adopted path token-identical, PASS; compaction flips 1 row of 128 | `regress.json` says exactly this; I confirmed all 20 `args.json` use batch 512 / `max_new` 512 (ladder) and 768 (dial), the same for every arm and seed | **reproduces**; comparability holds |
| 9 | "`L_true` ≥ 13 is 0 solves in all eleven ladder runs" (`run`) | 0 in bins 13 and 14 in all eleven | **reproduces** |
| 10 | "10 of 19 schemata are at 0 solves in every arm and seed" (`run`) | exactly 10 | **reproduces** |
| 11 | "G1 and G2 leave 17–19 of 19 at ≤ 2 on every seed — the pre-registered falsifier" (`run`) | 17, 18, 18, 19 | **reproduces**; the falsifier fires |
| 12 | "The brief predicted G1 would open an `ORE`-needing schema from the dilemma / De Morgan / distribution families: those are 0 in all six arm-seeds" (`run`) | constructive dilemma 0, De Morgan ≤ 4, distribution ≤ 1, excluded middle 0, everywhere | **reproduces**; miss correctly reported as a miss |
| 13 | "Disjunctive syllogism (which does need `ORE`) reaches ≥ 5 in C0 s1, G1 s0/s1 and G2 s0 alike — a draw effect, not a knob effect" (`run`) | correct, and I checked further: **every** DS solution uses `ORE` (21/21 C0 s1, 58/58 G1 s0, 38/38 G2 s0) — so the control already has the `ORE`-needing capability | **reproduces**, and the strongest form of the point is available |
| 14 | "Finding 3 keeps its counter-example: G2's base rate far below C0's yet EI − frozen larger on both seeds" (`run`) | frozen acq 0.032 / 0.034 vs 0.163 / 0.116; EI − frozen +0.383 / +0.362 vs +0.255 / +0.292. The pre-registered counter-example condition is met on both seeds | **reproduces** |
| 15 | "My own C0 ladder band was wrong: it quoted `lean-format` figures measured on `stage1_full_seq_s0`, a model trained on a different set" (`run`, `log` 01:05) | **confirmed independently.** I pulled `origin/dan_lean_format:artifacts/lf/la_*_seq_s*/args.json` and re-counted the found files: `init ckpts/lf/stage1_full_seq_s{0,1}.pt`, `train data/train.jsonl`, giving exactly 794 / 839 (T1) and 304 / 309 (frozen) | **reproduces**; a self-caught label error, correctly disclosed |
| 16 | "G1's own two seeds, 916 and 635, bracket the entire C0–G2 gap" (`run`) | G1 T1 range 635–916 (spread 281); C0 mean − G2 mean = 927.5 − 619.5 = 308 | **reproduces** as an order-of-magnitude statement; "bracket" is loose (G1's range lies inside the C0–G2 span rather than straddling it) but the substantive point — the null arm's seed spread ≈ the arm difference — is right |
| 17 | "G2 is at or below C0 on both seeds" (frozen ladder, `run`) | s0 44 vs 158 (−72 %); **s1 125 vs 114 (+9.6 %, above)** | **differs — rewording needed.** `log.md` 06:55–08:20 states it correctly; the compressed write-up does not |
| 18 | "Both accounts … are dead, for the same reason: the manipulation is smaller than the noise" (`run`, headline) | G2's *manipulation* is not small: box depth 23/45/32 vs C0's 53/35/11 %, `IMPI`-final 70 % vs 34 %, mean premises 0.78 vs 1.41, and its held-out effect (−24 to −30 pp) is 5× any noise measured here. What is inside the noise is the **frozen-ladder difference** | **differs — rewording needed** (see §Rewording 1) |
| 19 | "11 of 12 ladder jobs (`la_frozen_g1_s0` dropped, `QUESTIONS.md`)" (`run`, G4) | confirmed: `la_frozen_g1_s0/` holds `args.json` only; the drop is dated and reasoned in `QUESTIONS.md` 2026-09-24 08:35 | **reproduces**; disclosed |
| 20 | "`git diff origin/main HEAD -- nd_verify/ nd2lean.py` empty" (`numbers.md` G6) | **not empty**: `nd2lean.py` is 249 added lines against `origin/main` (it was born on `dan_lean_format`). The correct comparison, which `log.md` 08:20 uses, is `git diff origin/dan_lean_format HEAD`, and that **is** empty for all six files | **differs** — the *substance* (unchanged in this run) holds; the command quoted does not |
| 21 | Cost G8: "21.34 pod-hours, $13.02"; "dsg-1 RTX A6000, 11.56 h, $5.78 (billed $0.50/h); dsg-2 RTX 4090, 9.78 h, $7.24 ($0.74/h)"; both deleted | `podbudget` agrees, but `podbudget`'s rate table has **no A6000 entry** and falls through to a `$0.50` default — so "$0.50/h billed" is the tool's fallback, and `log.md` 23:05 and `QUESTIONS.md` both say the A6000 bills **$0.53/h**. RunPod `list-pod-billing`: dsg-1 `9pb5x3wfjj4ue6` **$5.93**, dsg-2 `uixjxcwkx01rb3` **$7.27**, total **$13.20** | **differs by $0.18** (still inside $14); the "$0.50/h billed" label is wrong and contradicts the run's own log |
| 22 | "The 2026-09-22 phase spent ≈ $2.5 of its own $20 ceiling" (`numbers.md` G8) | **Not supported, by a factor of ≈ 14.** RunPod billing for the three first-phase pods (`cjwj3tobkfxcua`, `lfjig3g5f1efpb`, `i8ijzvzj0gplcj`, all in `~/pods.log` as `dsg-1/2/3`): $8.96 + $8.93 + $8.92 on 2026-09-22 and $2.66 each on 2026-09-23 = **$34.79**. Hourly records show each ran continuously at $0.50/h from ≈ 06:10 on 09-22 to ≈ 05:20 on 09-23 — ≈ 23 h each, of which ≈ 19 h after the session died at ≈ 10:33 | **not supported — see §Finding A** |
| 23 | Pods deleted; bucket carries `{artifacts/dsg, ckpts/dsg, ckpts/ladder, data/dsg}` | `runpodctl pod list` empty; bucket has all four trees (185 artifact entries, 4 Stage-1 checkpoints, both 155k sets) | **reproduces** |
| 24 | Set shape, overlap, assembly, render-check tables (`data/dsg/README.md`) | every cell reproduces with my own code over all 465,000 records; my independent render check at a fresh seed is 1,000/1,000 round-trips, 500/500 Lean accepts, 200/200 negatives rejected on both sets | **reproduces** |

## Wording against n

- **"Both accounts … are dead."** The *textbook/rule-shape* account is dead by a pre-registered falsifier that
  fires on every seed of both arms — that wording is earned. The *shape-distribution/transfer* account is dead
  **on its prediction, not on its equivalence test**, and `log.md` 06:55–08:20 says exactly that
  ("within ±15 % of C0 on seed 1 (125 vs 114) but −72 % on seed 0 … dead on the prediction, not on the
  equivalence"). That distinction does not survive into `run_ds_generator.md`. It should.
- **Two seeds.** Every arm difference the write-up asserts is quoted with both seeds, and the one-seed cell
  (`G1 frozen 170`) is explicitly labelled "(s1)". **Correct throughout.** One consequence the write-up leans on
  should be stated as the limit it is: G1's frozen ladder is a *one-seed* noise-floor estimate, so
  "G1 … is *above* both C0 seeds" is a single draw, not a replicated finding.
- **"never" / "wall" / "bistable".** Not used loosely. "`L_true` ≥ 13 is 0 solves in all eleven ladder runs" is
  a count, not a claim about reachability, and is correct. "10 are at 0 solves in every arm and seed" is a count.
- **Pre-registered retention row missed and not scored.** The prereg predicts held-out retention across T1 at
  ±1 pp (C0, G1) and ±3 pp (G2). Measured (round 1 → round 8 greedy): **+5.02, +7.20, +9.38, +3.58, +8.60,
  +1.62 pp** — outside the band on all six, in the favourable direction. The *numbers* are in `numbers.md` G4's
  "held-out" column; the **miss is never called a miss**. Everything else in the expectations table is scored.

## Model labelling

**Exemplary, and better than the pre-registration it came from.** `numbers.md` opens § ds-generator with the model
for every arm (3.3 M from-scratch GPT, `lean_seq`, Stage-1 command line, training set, checkpoint path and md5 or
retrain provenance); every table names its source files and its measurement date; the coverage and dial tables
carry an explicit "measured on the **original** checkpoints, the ladder rows are on retrained replicas" note; and
the one inherited number that was mislabelled (the C0 ladder band) was found by the executor during the run,
traced to `pod/lf/jobs.py`, corrected, and the correction put in the write-up. I found no unlabelled number in
this run's deliverables.

## Gate 0

Pre-registration `5a6acd8` at 2026-09-22 06:09:12 UTC, first `dsg-*` pod 06:10:32 UTC. Addendum `2733373` at
2026-09-23 21:06:00 UTC, first resume pod 21:07:18 UTC. **Both phases pass**, and both deviations from the brief's
design (G1 as a null manipulation, G2 as the strict long generator rather than the brief's literal G2) were
pre-registered with their reasons and put to Dan in `QUESTIONS.md` with a stated default. The one deviation *from
the addendum* — batch 512 / `max_new` 512 instead of 4,096 / 384 — is documented in `pod/dsg/jobs2.py` and
`log.md` 21:42 with the equivalence test that forced it, and it was applied identically to all six models, which
is what the addendum's requirement was actually for.

## §Findings

**A. The first phase's pod spend is understated by ≈ 14×, and its $20 ceiling was exceeded (≈ $34.8).**
`numbers.md` G8 says "The 2026-09-22 phase spent ≈ $2.5 of its own $20 ceiling." RunPod's billing API says the
three `dsg-*` pods of that phase billed **$34.79** — they ran ≈ 23 h each, from 06:10 on 2026-09-22 to ≈ 05:20 on
2026-09-23, at $0.50/h. The session died at ≈ 10:33 on 09-22 when the account ran out of Fable credits, so
≈ 19 h per pod (≈ $29) is unattended idle time after the executor could no longer act. The sibling run
`ds-composition`'s five pods show the same pattern on the same day (≈ $8.8 each). Run total, both phases:
**≈ $48**, against $20 + $14 declared.
*This is not an accusation of overspending in session* — the in-session part is ≈ $6.6 — **but the number in the
deliverable is wrong and must be corrected**, and the mechanism (a dead session leaves pods running until a
balance floor or cron catches them) is worth Dan's attention because it hit two runs simultaneously and is not
something the executor can fix from inside a dead session. `podhours.log` only begins 2026-09-23, which is why
`podbudget` cannot see the first phase at all.

**B. The in-loop gate's Lean-vs-`nd_verify` disagreements are not reported, and they moved 22× since the last
run.** `artifacts/dsg/gate_*.jsonl`: 19,775,560 samples, 12,877,375 distinct checked by both, **9,249 "Lean
accepts, `nd_verify` rejects", 0 the other way** — 718 per million, against `lean-format`'s 460 in 13,887,708
(33 per million), which **that** run did report in `numbers.md` § lean-format. I re-verified all 9,249 with
`nd_verify` and classified them: **8,594 (92.9 %) "premise block does not match declared premises"** (the model
does not restate every `PR` line; the Lean rendering binds premises as hypotheses and does not care) and **655
(7.1 %) `¬A ≡ A → False` definitional unfolding** (`DN` 131, `NEGE` 90, `BOTE`/`Not.elim` 241, `NEGI` 59, `ORI1`
61, `R` 27) — exactly the two classes `QUESTIONS.md` flagged on 2026-09-21. **No counted proof is affected**
(both checkers are required, confirmed by the executor's 57,013 and my 450). But the run's own standard is
"report agreement with `nd_verify`", and § ds-generator reports only the counted-proof agreement. The rate should
be in G6.

**C. `numbers.md` G8's "$0.50/h billed" for the A6000 is `podbudget`'s fallback, not a billed rate.** `podbudget`
has no A6000 case and defaults to $0.50; `log.md` 23:05 and `QUESTIONS.md` both say $0.53/h; the API says the pod
billed $5.93, not $5.78. Resume-phase total $13.20, not $13.02 — still inside $14. The rate table should grow an
A6000 entry so the next run's ceiling is not computed from a default.

**D. `numbers.md` G6 quotes a diff command that does not return empty.** `git diff origin/main HEAD --
nd_verify/ nd2lean.py` shows `nd2lean.py` as 249 added lines, because it does not exist on `origin/main`.
`log.md` 08:20 uses the right base (`origin/dan_lean_format`) and is correct. Substance unaffected.

**E. Minor.** `run_ds_generator.md` is **528 words** against the brief's "≤ 400 words + two figures" (the two
figures exist). There is **no render check on file for C0** — its set is the unmodified lean-format a1 set, so
this is inherited rather than missing, but `data/dsg/README.md`'s render-check table should say "inherited"
rather than leave the row absent.

## §Verdict

**What stands.** Everything measured. I re-derived, with code I wrote myself, every quantity the pre-registration
promised — per-arm counts, rates, frontiers, acquisition, base reachability, `L*`, per-schema textbook solves,
set shape, overlap, disjointness, term sizes — across 18 coverage runs, 12 dial cells, 11 ladder runs, 6 held-out
evaluations and 465,000 training records, and found **zero differences** from `artifacts/dsg/summary.json`,
`numbers.md` and `data/dsg/README.md`. Every hard constraint passes. The checker of record agrees with `nd_verify`
on 57,013 counted proofs (executor) and on my independent 450, with adversarial negatives rejected. Gate 0 passes
in both phases. Every number names its model. The two headline conclusions are supported:

1. **The generator's proof-shape distribution is not the transfer wall.** G2 is the ladder transfer pool's own
   generator at cap 6 and its frozen ladder reads that pool *worse or no better* than the control (44 vs 158,
   125 vs 114) against a pre-registered 450–760 — while its held-out collapses to 0.61 / 0.67 and its
   required-reductio base rate is 0/300 on both seeds.
2. **The generator's rule shape is not the textbook wall.** The pre-registered falsifier fires on every seed of
   both arms (17, 18, 18, 19 of 19 schemata left at ≤ 2), and no `ORE`-needing schema in the brief's named
   families is reached anywhere — while disjunctive syllogism, which *is* `ORE`-needing, is reached by the
   control too, with `ORE`, in 100 % of its solutions.

And a third that the run earned by accident and reports honestly: **Finding 3 has a clean two-seed
counter-example.** G2 has a far lower depth-3 base rate than C0 *and* a larger EI − frozen, on both seeds.

**What must be reworded.**
1. `run_ds_generator.md`'s headline "the manipulation is smaller than the noise" is right for the *readiness
   differences* and wrong for G2's manipulation, which is large and whose held-out effect is large. Suggest:
   *"Both accounts are dead: the rule-shape account by its falsifier, the shape-distribution account by its
   prediction — G2 changes the training distribution profoundly and the transfer readiness not at all, and what
   differences remain are inside a between-pool noise floor the null arm measures at ±50 %."*
2. "G2 is at or below C0 on both seeds" → "44 vs 158 on seed 0 and 125 vs 114 on seed 1 — one far below, one
   marginally above, against a pre-registered +50 to +150 %." `log.md` already has the right form.
3. Carry `log.md`'s "dead on the prediction, not on the equivalence" reading of the ±15 % falsifier into the
   write-up; a reader of `run_ds_generator.md` alone cannot tell which way the falsifier was scored.
4. Label G1's frozen ladder (170) as the one-seed estimate it is wherever it carries weight, including in
   "G1 … is *above* both C0 seeds".
5. Correct `numbers.md` G8's first-phase cost (§Finding A), its "$0.50/h billed" (§Finding C), and G6's diff
   command (§Finding D).
6. Score the pre-registered held-out-retention row as the miss it is (+1.6 to +9.4 pp against ±1 / ±3 pp).

**What is not supported.** Only the first-phase cost sentence (§Finding A). Nothing scientific in this run rests
on it.

**Next measurements that would settle what is left open.**
1. **Make the noise floor a measured quantity rather than a by-product.** This run's most valuable number is
   accidental: two independent raw pools from the *same* generator (C0 and G1) differ by +49 % on the frozen
   ladder and 1.9× on required@8 coverage. Four independent pools × two Stage-1 seeds at cap 6, measured frozen
   on the ladder only (no EI), is ≈ 8 cheap jobs and would give the first real error bar for every "arm X beats
   the control" claim in this campaign. Until it exists, no dataset-style run with n = 2 can resolve anything
   smaller than ±50 %.
2. **Finish G1's frozen ladder on seed 0** (≈ $1.3, ≈ 2 h). It is the one dropped cell, and it is the cell that
   turns the noise-floor claim from one draw into two.
3. **The length horizon, not the shape.** Both walls survived a shape manipulation that moved the training
   distribution as far as the cap allows. `ds-composition`'s length axis is now the only remaining lever named in
   the brief, and the `L_true` ≥ 13 result (0 solves in all eleven runs, T1 and frozen alike) says where the
   ceiling sits.
4. **Fix the translator's two looseness classes** (`False.elim na` for `BOTE`; mirror the "every premise
   restated" rule in the inverse grammar). 718 disagreements per million is a 22× rise over `lean-format`'s rate
   on a workload with more premises, and "Lean is the checker of record" cannot be taken literally until both are
   closed. No count in either run is affected, because both runs required `nd_verify` as well.

**No hard-constraint violation. No quarantine. The run's numbers stand as written, subject to the six rewordings
above and the correction of the first-phase cost.**
