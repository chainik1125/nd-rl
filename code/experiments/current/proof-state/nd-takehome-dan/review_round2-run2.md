# Review of round2-run2 (six new patterns, classes pre-registered)

Reviewer session, independent of the executor. Phase 1 (this section) was written from the run brief
(`BRIEF_ROUND2.md` §Run 2), the pre-registration as committed (log.md in 5b3b257, 19:45:07 UTC, before any set was
assembled; the cap-8 addendum in 73cd145, 22:51:24 UTC), the code, the data and the raw artefacts only — no executor
write-up (`run2.md`, `numbers.md` §Run 2, `STATUS.md`, `artifacts/r2/summary*.json`, `analysis_*.log`, figures) was
read before this section was committed. Extracting the two pre-registration entries from the committed log did expose
me to the executor's two adjacent progress lines (22:12 "first arms 0 / 500", 22:52 "controls"); every number below is
nevertheless my own recount. All counts come from `review_run2_recount.py` (own proof parser, own dependency pruning,
own start-index normaliser, own atom-renaming key — the reviewer's module from the run-5 review — and own predicates
for the six patterns, written from the pre-registration's one-line definitions and self-tested against verifier-checked
hand-built proofs). Only `nd_verify` is shared with the executor. Machine-readable outputs: `artifacts/review_r2/`
(`sets.json`, `pools.json`, `arms_ei_.json`, `arms_frozen_.json`, `cov.json`, `ckpts.json`).

## Recount

### Hard constraints

| check | result |
|---|---|
| `nd_verify` unmodified | tree hash of `nd_verify/` at HEAD = `origin/main` (9437bb7); sha256 of both files identical in the repo and in the review copy |
| `artifacts/TEST_RUN_DONE` | byte-identical to the Sep 15 07:38 commit (one test run) |
| cap 6 on supervised data | the four cap-6 Stage-1 sets (`struct`, `impi_ore_f0`, `negi_ande_hyp_f0`, `ori_ore_f0`): 155,000 proofs each, 31,000 per written length 2–6, max 6, 0 violations, 0 unparsable; 811 random records per set re-verified, 0 failures. Two **cap-8** sets are used by control arms only (`train_r2_c8_depth4_f0`: 154,994 = 22,142 × lengths 2–8, max 8; the campaign's `train_derived_ore_strict_f0_c8` for the `c8ctl` arms): outside the take-home's cap-6 rule, labelled as controls, not submission models. EI rounds train on 7–11-line RL proofs under the standard `--cap 0` recipe |
| hand-written / LLM-written training proofs | every training record is a generator-pool proof (`pool_cap6_recon`, `pool_cap8`); the two schema pools (`chain_pool.py`, `ore_pool.py`) feed **target** files only, and no arm ever found a proof there. Nothing outside the generator enters any mix |
| evaluation files in training code | `expert_iter.py` reads targets / transfer / held-out / val-36 for sampling, tracking and the (unused) relabel exclusion; the found target proofs are what EI trains on, by design. `coverage.py` and `necessity.py` write no training data |
| pre-registration before the first pod job | classes + expectations committed 19:45:07 (5b3b257); first target-generation log written 19:57; sets assembled 21:19–21:21; first Stage-1 checkpoint 21:44:51 (file mtimes preserved by the pull). **Met.** Cap-8 addendum committed 22:51:24 (73cd145); its set written 00:02, its Stage-1 checkpoints 00:13:00 / 00:13:05. **Met.** The four `c8ctl` / `natctl` control arms were queued ≈ 22:11 (job-file mtimes) and disclosed as "outside the pre-registration" in be90708 at 22:13:47, before their first checkpoint (22:26:18) — not pre-registered, correctly labelled |
| generator "unchanged" | `gen.py` is unchanged; `make_coverage_sets.py` gained a `--gen_max_depth` knob (default 3 = the take-home generator) used only for the depth-4 **target** pool (`--gen_max_depth 5`, `pod/r2/targets2.sh`); every Stage-1 set was drawn from the pre-existing pools. Matches the default stated in `QUESTIONS.md` |

### Split disjointness (my renaming key, atoms relabelled by first appearance)

| training set | classes | key field = my key | ∩ any run-2 targets / transfer (12 files) | ∩ heldout | ∩ heldout_c8 | ∩ val-36 |
|---|---:|---|---:|---:|---:|---:|
| train_r2_struct | 155,000 | ✓ | **0** | 0 | 134 | 0 |
| train_r2_impi_ore_f0 | 155,000 | ✓ | **0** | 0 | 138 | 0 |
| train_r2_negi_ande_hyp_f0 | 155,000 | ✓ | **0** | 0 | 122 | 0 |
| train_r2_ori_ore_f0 | 155,000 | ✓ | **0** | 0 | 133 | 0 |
| train_r2_c8_depth4_f0 (cap 8) | 154,994 | ✓ | **0** | 142 | 0 | 0 |
| p2/train_derived_ore_strict_f0_c8 (cap 8, `c8ctl` parent, built Sep 16) | 154,994 | ✓ | depth-4 pools **0**; impi_ore 7 + 3, negi 1, ori_ore 1 + 1 | 101 | 0 | 0 |

The non-zero cells are harmless: each arm's greedy held-out set is the one its own training set excludes (`heldout`
for cap-6 arms, `heldout_c8` for cap-8 arms), and the `c8ctl` models were run on the depth-4 pool only. Cross-pool:
`targets_impi_ore` ∩ `targets_ori_ore` = 1 class (and 1 each in two transfer pairs), different arms; all six
targets ∩ transfer pairs = 0.

### Pattern frequency in the Stage-1 sets (my predicates, dependency-pruned; written form gives the same counts)

| set | depth4 | impe_chain4 | nested_ore | impi_ore | negi_ande_hyp | ori_ore |
|---|---:|---:|---:|---:|---:|---:|
| train_r2_struct (natural draw) | **0** | **0** | **0** | 130 (0.084 %) | 11 (0.007 %) | 36 (0.023 %) |
| train_r2_impi_ore_f0 | 0 | 0 | 0 | **0** | 14 | 62 |
| train_r2_negi_ande_hyp_f0 | 0 | 0 | 0 | 128 | **0** | 43 |
| train_r2_ori_ore_f0 | 0 | 0 | 0 | 118 | 19 | **0** |
| train_r2_c8_depth4_f0 (cap 8) | **0** | 0 | 0 | 5,327 | 35 | 2,882 |
| p2/train_derived_ore_strict_f0_c8 (cap 8) | **0** | 0 | 0 | 4,919 | 33 | 72 |

So f = 0 holds exactly for each arm's own pattern, and the executor's assembly report reproduces to the proof. One
consequence for the controls: the campaign's cap-8 set contains **no** depth-4 proof either (the generator's box cap is
3, so no cap-8 pool proof has a fourth box). The `c8ctl` arms, described when queued as a "natural-rate" cap-8 control,
are therefore a second pair of **f = 0 depth-4 arms at cap 8**, the same condition as the addendum's `c8f0` arms.

### Target and transfer pools (own key, own predicate, every oracle proof re-verified)

| pool | targets / transfer | source | min-proof lines | required / uses-only (targets) | oracle proof verifies, has pattern | notes |
|---|---:|---|---|---|---|---|
| depth4 | 500 / 200 | generator, box cap raised to 5 | 8: 269, 9: 53, 10: 178 | 176 / 324 | 500 / 500, 500 / 500 | `build_pools.sh` says `--mode requires`; the file's `mode` field is `uses` and 324 targets have a depth-≤ 3 proof within 10 lines. 473 / 500 have no premise; conclusions are right-nested implications of depth 4–6 and 377 / 500 end in an `( X > X )` identity: the theorems are "open four boxes, close four boxes" |
| impe_chain4 | 500 / 200 | 7 schemata (`chain_pool.py`) | 8: 110, 9: 246, 10: 144 | – / 500 (no restriction exists) | 500 / 500, 500 / 500 | `major_conj2` 110, `major_conj3` 82, `mixed` 82, `mixed2` 82, `minor_chain2` 80, `minor_chain` 60, `major_conj3b` 4 |
| nested_ore | **300 / 41** | 5 schemata (`ore_pool.py`) | 11: 113, 12: 172, 13: 15 | 300 / 0 (all required, bound 13) | 300 / 300, 300 / 300 | 343 required of 800 candidates, so the pool is 300 + 41 rather than the pre-registered 500 + 200 |
| impi_ore | 500 / 200 | generator | 7: 144, 8: 171, 9: 132, 10: 53 | 109 / 391 | 500 / 500, 500 / 500 | |
| negi_ande_hyp | **400 / 94** | generator (49 M tries for 807 candidates) | 7: 213, 8: 86, 9: 83, 10: 18 | 97 / 303 | 400 / 400, 400 / 400 | |
| ori_ore | **400 / 160** | generator, mode `gen` | 7: 310, 8: 81, 9: 8, 10: 1 | 0 / 0 | 400 / 400, **0** / 400 | the *generating* proof has the pattern (400 / 400 verify and have it); the shortest proof never does, by construction — a decoration control |

Every pool: my key = stored key and `thm` = prompt for every record; `n_lines` = `min_lines_ub` = verified length;
every record is present in its oracle file with the same labels. Oracle files (`nec_*.jsonl`): 0 timeouts in all six;
every `requires` record has the pattern in its unrestricted proof (my predicate) and every restricted proof lacks it;
all 80,618 oracle / restricted proofs verify. Base-reachability sampling used the **first 300 records of each shuffled
pool** (`coverage.py --limit 300`; length mix proportional to the pool), not "the 300 shortest" as pre-registered —
immaterial for the zero-rate arms, and the c8 arms' 300 include all three lengths.

### Arms (identical EI: 8 rounds, k = 32, T = 0.8, retain 20k, `max_per_thm` 4 × weight 4, 600 steps at 3·10⁻⁴)

Per target: pattern theorem = ≥ 1 verified proof whose dependency-pruned form has the pattern (my predicate; the
unpruned count is identical in every cell); round = first per-round file the (name, normalised proof) appears in.
Every distinct proof in every arm re-verified against its prompt (3,913 target and 1,688 transfer proofs, 0 failures;
0 unparsable), as were the 1,072 distinct coverage proofs (0 failures); per-round files nested, `round` field = first appearance everywhere; my cumulative solved count equals
the executor's `round_<r>.json` in every round of all 39 arms with files. Ignition = ≥ 2 % of targets with a pattern
proof (10 / 500, 8 / 400, 6 / 300), as pre-registered. Base = my recount of `cov_*.jsonl` (pass@2,000 on 300 targets
of the Stage-1 draw): theorems with a verified pattern proof / pattern hits per sample.

**Pre-registered arms (cap-6 Stage-1, f = 0 for the arm's pattern).**

| pattern | seed | base pass@2,000: pattern theorems / 300; rate per sample | EI cum. pattern theorems r1…r8 | EI r8 acq. | EI r8 solved | frozen r8 solved (pattern) | transfer r8 solved (pattern) / n | held-out greedy r8 |
|---|---|---|---|---:|---:|---:|---:|---:|
| depth4 | 0 | 0; 0 | 0 0 0 0 0 0 0 0 | 0.000 | 0 / 500 | 0 (0) | 0 (0) / 200 | 0.957 |
| depth4 | 1 | 0; 0 | 0 … 0 | 0.000 | 0 | 0 (0) | 0 / 200 | 0.959 |
| impe_chain4 | 0 | 0; 0 (**file incomplete: 96 / 300 theorems**) | 0 … 0 | 0.000 | 0 / 500 | 0 (0) | 0 / 200 | 0.957 |
| impe_chain4 | 1 | 0; 0 | 0 … 0 | 0.000 | 0 | 0 (0) | 0 / 200 | 0.959 |
| nested_ore | 0 | 0; 0 (269 / 300 theorems at k = 2,000) | 0 … 0 | 0.000 | 0 / 300 | 0 (0) | 0 / 41 | 0.957 |
| nested_ore | 1 | 0; 0 (**100 theorems at k = 500**, re-queued smaller) | 0 … 0 | 0.000 | 0 | 0 (0) | 0 / 41 | 0.959 |
| impi_ore | 0 | 0; 0 | 0 … 0 | 0.000 | 0 / 500 | 0 (0) | 0 / 200 | 0.946 |
| impi_ore | 1 | 0; 0 | 0 … 0 | 0.000 | 0 | 0 (0) | 0 / 200 | 0.961 |
| negi_ande_hyp | 0 | 0; 0 | 0 … 0 | 0.000 | 0 / 400 | 0 (0) | 0 / 94 | 0.944 |
| negi_ande_hyp | 1 | **2; 5.0·10⁻⁶** (3 hits in 600k) | 0 0 0 **1** 15 40 49 59 | **0.147** | 59 / 400 | 2 (2) | 12 (12) / 94 | 0.970 |
| ori_ore | 0 | 125 solved, **0 with pattern**; 0.12 per sample | 0 … 0 | 0.000 | 217 / 400 | 148 (0) | 94 (0) / 160 | 0.967 |
| ori_ore | 1 | 148 solved, 0 with pattern; 0.16 | 0 … 0 | 0.000 | 219 / 400 | 173 (0) | 97 (0) / 160 | 0.969 |

Nine of the twelve pre-registered arms found no verified proof of any target in 256 attempts × 500 targets (the
generator-pool models' written frontier is 7–8 lines; the depth-4, chain and nested-ORE targets are 8–13 lines and
the impi_ore / negi targets 7–10). The one igniting arm, `negi_ande_hyp` s1, is the one draw with a non-zero pre-RL
rate (2 theorems, 3 hits in 600,000 samples); its frozen control at the same 256 attempts holds 2 theorems, EI reaches
59 (ignition round 5, first proof round 4), all 7–8-line proofs of **uses-only** targets (0 of the 97 required ones),
and the pattern reaches 12 of the 94 never-trained transfer theorems. `ori_ore` behaves as the decoration control it
was pre-registered to be: 0 pattern proofs anywhere, while EI raises plain solving from 148 / 173 (frozen) to 217 / 219.

**Controls outside the pre-registration (queued 22:11, disclosed 22:13) and the pre-registered addendum (`c8f0`).**

| arm | seed | Stage-1 set (own pattern count) | base pass@2,000: pattern theorems / 300; rate | EI cum. pattern theorems r1…r8 | EI r8 acq. (solved) | frozen r8 (pattern) | required stratum solved / 176 (all with pattern) | transfer r8 (pattern) / 200 | held-out greedy r8 |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| depth4 `c8ctl` | 0 | cap-8 derived-ORE set (depth4 **0**) | **110; 0.132** | 123 312 344 381 453 479 484 485 | **0.970** (485) | 148 (150) | 162 | 187 (187) | 0.924 |
| depth4 `c8ctl` | 1 | same | 134; 0.115 | 147 280 321 348 455 475 479 481 | 0.962 (481) | 201 (202) | 162 | 189 (189) | 0.934 |
| depth4 `c8f0` | 0 | cap-8 set, depth-4 "removed" (0 → 0) | 131; 0.195 | 174 297 331 417 469 474 479 481 | 0.962 (481) | **no files** (`args.json` only) | 161 | 187 (187) | 0.923 |
| depth4 `c8f0` | 1 | same | 128; 0.106 | 131 297 332 338 348 414 474 482 | 0.964 (482) | 180 (180) | 161 | 187 (187) | 0.932 |
| impi_ore `natctl` | 0 | struct (impi_ore 130 = 0.084 %) | 12; 6.2·10⁻⁴ | 5 30 51 78 88 92 92 95 | 0.190 (95) | 10 (10) | 4 / 109 | 34 (34) | 0.966 |
| impi_ore `natctl` | 1 | struct | 26; 1.3·10⁻³ | 9 37 53 64 89 100 105 111 | 0.222 (111) | 17 (17) | 4 / 109 | 45 (45) | 0.963 |
| negi_ande_hyp `natctl` | 0 | struct (negi 11 = 0.007 %) | 0; 0 | 0 … 0 | 0.000 (0) | 0 (0) | 0 / 97 | 0 / 94 | 0.957 |
| negi_ande_hyp `natctl` | 1 | struct | 4; 1.2·10⁻⁵ | 0 0 0 1 1 3 11 11 | 0.028 (11) | 3 (3) | 0 / 97 | 3 (3) / 94 | 0.961 |

Reading the cap-8 rows: all four cap-8 Stage-1 draws were trained on sets with **zero** depth-4 proofs, yet each writes
verified depth-4 proofs at 11–20 % per sample before any RL (110–134 of 300 targets in 2,000 samples; frozen controls
150–202 of 500 at 256 attempts), and EI lifts that to 0.96–0.97 with every solved target solved *with* the pattern
(2 and 1 of the frozen controls' targets are solved by a depth-3 proof, all in the uses-only stratum; 0 in EI). The
written lengths are 8–11 lines. The same pattern at cap 6 is 0 in 600,000 pre-RL samples and 0 after RL in both
draws. The `natctl` rows are elicitation of a pattern present in the data at 130 / 155k (impi_ore: base 12–26 theorems
→ 95–111 with EI, ignition round 2) and near-absence at 11 / 155k (negi: one draw 0 → 0, the other 4 → 11).

**Drift measurement (pre-registered for zero-hit draws that ignite).** No zero-hit draw ignited. The only drift file
on disk is `drift_negi_ande_hyp_s1_r4` (28 pattern theorems, 8.6·10⁻³ per sample); the round-4 checkpoint was trained
on the arm's first pattern proof (found in round 4; `mix_4` holds 4 RL records), so this measures the state *after*
the first proof, not before it. The queued r1–r3 measurements (`pod/r2/drift_p2.txt`) have no output files and the
r1–r3 checkpoints of that arm were not pulled (`ckpts/r2/` holds `ei_negi_ande_hyp_s1_r4…r8` only).

### Pre-registered expectations (5b3b257 19:45; addendum 73cd145 22:51) against my values

| id | expectation | my value | verdict |
|---|---|---|---|
| depth4 — STRUCTURAL: acquired 2 / 2, ≥ 0.20 at r8, ignition by round 4; base ≈ 0 | base 0 / 0 (600k samples each); EI **0 / 500 and 0 / 500**; frozen 0 | **wrong** — not acquired at cap 6 from either draw |
| impe_chain4 — STRUCTURAL: 2 / 2 ≥ 0.20; base small non-zero | base 0 (s0's coverage file stopped at 96 theorems); EI 0 / 500 both; no proof of any chain target in 512k attempts | **wrong** |
| nested_ore — ≤ 0.02 both, base 0 (length-limited) | 0 / 300 both, base 0 | met (as a length limit; 11–13-line targets) |
| impi_ore — RULE SEQUENCE: ≈ 0 (≤ 0.05) unless pre-RL rate > 0 | base 0 / 0; EI 0 / 500 both | met, but uninformative: neither draw produced any valid proof of any target; the natural-rate control (`natctl`) elicits 0.19–0.22 from a base of 12–26 theorems |
| negi_ande_hyp — ≈ 0 from zero-rate draws; ≥ 1 of 2 draws with pre-RL rate > 0 → elicitation | s0: base 0 → 0; s1: base 3 hits / 600k → 0.147 (frozen 0.005) | **met exactly** |
| ori_ore — decoration control, ≤ 0.02, frozen ≈ EI | 0 pattern proofs in EI, frozen, base and transfer; EI solves 217 / 219 vs frozen 148 / 173 | met for the pattern; solving itself is not "frozen ≈ EI" |
| drift — zero-hit draws that ignite show rate > 0 on r2–r4 checkpoints before the first proof | no zero-hit draw ignited; the one drift file is post-first-proof | not testable from the files |
| addendum `c8f0` — cap 8 with depth-4 removed acquires depth 4 (≥ 0.20 in ≥ 1 of 2) if "structural repetition" is the rule; ≈ 0 with base 0 if a length margin is needed | 0.962 / 0.964, but **base 0.11–0.20 per sample and frozen 0.36** before any RL; the "removed" set had nothing to remove (cap-8 pool: 0 depth-4) | the stated outcome occurred, but its premise "base rate 0" did not: the fourth box is already generalised by Stage-1 at cap 8, so this is elicitation of a Stage-1 generalisation, not RL acquisition from zero |
| pools 500 + 200 per pattern | 500 + 200 for depth4, impe_chain4, impi_ore; 400 + 94, 300 + 41, 400 + 160 for negi, nested_ore, ori_ore | short, disclosed by the file sizes |
| budget ≈ $8, ceiling $50 | not derivable from files | – |

### Reproducibility

`python3 review_run2_recount.py` (sub-commands `sets`, `pools`, `arms`, `cov`, `ckpts`) in the repository root
regenerates every number above from `data/r2/`, `data/p2/`, `artifacts/r2/`, `ckpts/r2/` and `targets/`; outputs in
`artifacts/review_r2/`. Checkpoint provenance was read from the 24 `ckpts/r2/*.pt` files' stored training arguments
(without torch): every Stage-1 model from its named set at cap 6 (cap 8 for the two `c8` sets), 6,000 steps, seeds 0 / 1;
every `_r8` from its own `_r7` on `mix_8.jsonl` at `--cap 0`, seed s·1000 + 8; the two `c8ctl` parents are the
campaign's Sep 16 05:03 checkpoints trained on `train_derived_ore_strict_f0_c8`. Only round-8 checkpoints were pulled for
most arms (plus r4–r8 of `negi_ande_hyp` s1); the `mix_*.jsonl` files were not pulled, so the mixes are known only
through the `round_<r>.json` counts (RL records 0 in every round of the nine empty arms). Not re-derived: pod spend,
kill-switch timing, the executor's "hand check of ten targets per pattern" (I spot-checked two oracle proofs per
pattern by eye; all six patterns present as claimed), and the bucket contents beyond the existence of
`round2/run2/{artifacts,ckpts,data}` (uploaded 01:27 UTC). At 02:10 UTC no `p1`–`p4` pods exist on the account
(the ten running pods are `r4-*`, `r1-a100`, `la-*` — other runs' executors).

## Compare (phase 2: `run2.md`, `numbers.md` §Round 2 — Run 2, `log.md` 19:45–01:30, `STATUS.md` §Run 2, `artifacts/r2/summary.json`, `analysis_final.log`, `figures/run2_acq.png`, `run2_curves.png`)

Gate 0: classes and expectations committed 19:45:07, first target-generation job 19:57, first set 21:19 (met); the
cap-8 addendum committed 22:51:24 before its set (00:02) (met); the four `c8ctl` / `natctl` controls disclosed as
outside the pre-registration two minutes after queueing. The results-vs-expectations entry (log 01:30) reports depth4
and impe_chain4 as **wrong** and the rest as held — honest. One correction to my phase 1: the round-1…3 checkpoints of
`negi_ande_hyp` s1 were never written (EI skips the fine-tune when a round has no accepted proof; `mix_rl_records` is
0 in rounds 1–3), so the three missing drift files could not have been produced; the executor's reading — "nothing is
trained before the first proof, so nothing can drift" — is right for this arm.

| claim (executor) | my independent value | verdict |
|---|---|---|
| `numbers.md` per-arm lines (solved, pattern theorems, per-round lists, required stratum, violations, ignition round, transfer, held-out) for all 39 arms with files; pre-RL lines for 20 coverage files; the drift line | identical element for element (own predicates, own normaliser, first-appearance round) | reproduces |
| `run2.md` table: 0 / 0 (depth4, chain, nested, impi_ore, ori_ore), 0 / 0.147 (negi); base rates 0 and 5·10⁻⁶; cap-8 0.97 / 0.96 and 0.96 / 0.96, base 0.11–0.20; natctl 0.19 / 0.22 and 0 / 0.03 | identical (0.028 for the last cell) | reproduces |
| cap-8 sets "zero depth-4 proofs" (run2.md, log 00:36) | 0 in both cap-8 sets by my predicate, pruned and written; and 0 in every cap-6 set | reproduces |
| "all 485 with a depth-≥ 4 proof … 0 violations" (log 22:52; numbers.md `viol` 0 everywhere) | EI cap-8: every solved target has a depth-4 proof (485 / 481 / 481 / 482), required stratum 162 / 162 / 161 / 161 all with the pattern; the only pattern-free solutions anywhere are 2 + 1 uses-only targets in two frozen controls | reproduces |
| `run2.md` table column "f > 0 control" and both figures' legend "f > 0 controls (orange)" for `c8ctl` | the `c8ctl` Stage-1 sets hold 0 depth-4 proofs, as the executor found at 00:36 and states in the same table cell ("cap-8 models with 0 depth-4 proofs") | **relabel**: four f = 0 cap-8 arms, none is an f > 0 control; only `natctl` is |
| "pre-RL pass@2,000 on the 300 shortest targets" (numbers.md, log 22:25, 00:40) | `coverage.py --limit 300` takes the first 300 records of the shuffled pool: all lengths in proportion (depth-4: 8 / 9 / 10 lines 156 / 33 / 111) | reword: "300 random targets" |
| "the cap-6 base never writes [a fourth box]"; "the structural negatives are length, not structure" (run2.md) | 0 depth-4 proofs in 1.2 M pre-RL samples and 256 k EI attempts per draw — but the same cap-6 draws do write 8-line proofs (negi s1: 7 targets at 8 lines; ori_ore arms: 32 distinct 8-line proofs each), and 269 of the 500 depth-4 targets have 8-line pattern proofs that were never produced | the zero reproduces; "length, not structure" is an interpretation the run does not separate — the cap-6 models reach 8 lines, they do not reach 8 lines *with four boxes*. The supported form is the one in the restated rule: the pattern's minimal instance (8 lines) lies outside the cap |
| "at cap 8 the base models … already write a fourth box in 11–20 % of samples" | 0.132 / 0.115 / 0.195 / 0.106 per sample; 110–134 of 300 targets | reproduces; add the pool's shape: 473 / 500 targets have no premise and 377 / 500 are right-nested implication chains ending in `( X > X )`, so the proof is dictated by the conclusion's syntax (open a box per antecedent, close them in order). The 11–20 % says the cap-8 models extend "one box per antecedent" from three to four on such theorems; it is not comparable in difficulty to the campaign's depth-3 pool (premised generator theorems, base ≈ 10⁻⁶ at cap 6) |
| "nothing at f = 0 crossed zero except negi_ande_hyp seed 1, whose base already emitted the shape at 5·10⁻⁶ — elicitation" | s1 base 2 theorems / 3 hits in 600 k; frozen 2 at 256 attempts; EI 59 (0.147), first proof round 4, all uses-only targets, transfer 12 / 94 | reproduces; "elicitation" is the right word, and the 30× gain over the frozen control at equal attempts is the run's one RL effect on a cap-6 f = 0 draw |
| "The decoration control solves 217 / 219 targets and never writes the pointless ORI-then-ORE" | 217 / 219, 0 pattern proofs in EI, frozen, base and transfer | reproduces |
| pre-registration: pools of "500 targets + 200 transfer" per pattern; "ten targets per pattern hand-checked in this log" | 500 + 200 for three pools, 400 + 94 / 300 + 41 / 400 + 160 for the others (sizes given in numbers.md); no run-2 hand check appears in the log (the two logged hand checks are run 5's) | disclose the missing hand check; my spot check of two oracle proofs per pattern found all six patterns as labelled |
| numbers.md: sets assembled with "3,151 run-2 pool classes … excluded" | `sets.log`: 49,337 classes excluded (every candidate file + held-out); the pools' 3,495 classes are among them | reword; disjointness holds either way (0 overlap, above) |
| `frozen_depth4_c8f0_s0` | directory holds `args.json` only (no round or found files); the other three cap-8 frozen controls exist. Not mentioned in run2.md / numbers.md / log | disclose (one frozen control of four lost; conclusions unaffected: the three others are 0.30–0.40) |
| impe_chain4 s0 coverage "96 targets, the run was cut"; nested_ore s1 "100 at k = 500" (numbers.md) | 96 / 300 and 100 at k = 500, 0 hits in both | disclosed; fine |
| held-out intact | greedy 0.944–0.973 (cap 6), 0.918–0.934 (cap 8) in all arms | reproduces (not claimed; worth one clause) |
| `run2.md` ≤ 400 words | 545 words including the figure lines | over the brief's length |
| bucket `round2/run2/{artifacts,ckpts,data}`; pods p1–p3 ≈ $6 | the three directories exist (01:27 UTC); spend not derivable | reproduces / not derivable |

## Verdict

**What stands.** Every number in `run2.md`, `numbers.md` §Run 2 and `artifacts/r2/summary.json` reproduces from
the raw per-round and coverage files with independent code and predicates; every counted proof re-verifies (5,601
arm proofs, 1,072 coverage proofs, 80,618 oracle proofs, 0 failures); the five Stage-1 sets are exactly f = 0 for
their patterns and class-disjoint from every pool; the checkpoints were trained from the stated sets on the stated
mixes; the classes and expectations were committed before any set existed and the executor scored two of them as
wrong. The clean results are: (i) at cap 6, none of the six pre-registered patterns was acquired from a draw with a
zero pre-RL rate — two structural and three rule-sequence patterns stayed at 0 / 500 through 256 attempts per target
with frozen controls at 0, and the one arm that moved (negi s1, 0.005 → 0.147) is the one draw that emitted the
pattern before RL; (ii) the natural-rate controls show the same thing from the other side (impi_ore at 130 / 155k:
base 12–26 theorems → 0.19–0.22; negi at 11 / 155k: 0 → 0 and 4 → 11); (iii) at cap 8, four Stage-1 draws trained
on **no** depth-4 proof write depth-4 proofs at 11–20 % per sample before any RL, and EI takes them to 0.96–0.97 with
every solved target solved with the pattern. Together with the ignition study, the run supports the executor's
restated rule — the pre-registered syntactic classes predicted nothing that the base rate did not — and the one
genuine zero-to-something event in the run happened in Stage-1 pretraining (cap 8, depth 4), not in RL.

**What must be reworded.** (1) The `c8ctl` arms are f = 0 arms, not "f > 0 controls": the table column and both
figure legends contradict the text and the 00:36 log entry. (2) "300 shortest targets" is "300 random targets".
(3) The depth-4 pool's shape should be stated next to the 11–20 %: no-premise identity chains whose proof the
conclusion dictates; the number measures a narrow generalisation and is not on the same scale as the campaign's
depth-3 base rates. (4) "Length, not structure" is a hypothesis: the cap-6 draws write 8-line proofs of other
shapes and never an 8-line four-box proof; what the run shows is that a pattern whose minimal instance exceeds the
cap has base rate 0 and is not acquired. (5) Disclose the lost `frozen_depth4_c8f0_s0`, the absent per-pattern hand
check, and the 49,337-class exclusion. (6) `run2.md` is 545 words against 400.

**Flags, not quarantine.** The two cap-8 Stage-1 sets and their eight arms lie outside the take-home's cap-6 rule
(labelled controls; not submission models). No hand-written or LLM-written proof entered any training set.

**Next measurement.** (1) The length-margin clause directly: cap-7 f = 0 sets (two seeds) on the same depth-4 pool —
the clause predicts base rate ≈ 0 at cap 7 (minimal instance 8 lines) and the observed 11–20 % at cap 8; a cap-7 rate
between them would say the margin is soft. (2) A depth-4 pool that is not identity chains — premised generator
theorems whose depth-4 proof is 9–10 lines and required — at cap 8, to see whether the 11–20 % survives when the
conclusion no longer dictates the proof. (3) The run-3 protocol on a rule-sequence pattern with base rate 0: inject
1 / 4 / 16 sibling or generator `impi_ore` proofs into the cap-6 f = 0 draws (7-line targets, within the frontier) and
continue EI; if the pattern ignites like depth-3 did, the rule-sequence negatives are base-rate zeros, not a harder
kind of novelty; if not, the class distinction has content after all. (4) Two more `negi_ande_hyp` draws to see how
often a cap-6 f = 0 draw emits the pattern at all (1 of 4 so far, counting the `natctl` pair).
