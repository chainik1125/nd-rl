# Review of round2-run3 (minimum outside data for ignition)

Reviewer session, independent of the executor. Phase 1 (this section) was written from the run brief
(`BRIEF_ROUND2.md` §Run 3), the pre-registration (log.md as committed in 734d51f, 22:14:24 UTC, *before* the
first run-3 pod job), the code, the data and the raw artefacts only — no executor write-up (`run3.md`,
`numbers.md` §Run 3, `STATUS.md`, `artifacts/r3/summary*.json`, figures) was read before it was committed.
All counts below come from my own code: `review_run3_recount.py` and `review_run3_splits.py` (own proof
parser, own dependency pruning, own start-index normaliser, own box-depth counter and reductio predicate, own
atom-renaming key — the reviewer's module from the run-5 review, `review_run5_recount.py`). Only `nd_verify`
is shared with the executor. Machine-readable outputs: `artifacts/review_r3/recount.json`, `splits.json`.

## Recount

### Hard constraints

| check | result |
|---|---|
| `nd_verify` unmodified | tree hash of `nd_verify/` at HEAD = `origin/main` (9437bb7); sha256 of both files identical in the repo and in the review copy |
| `artifacts/TEST_RUN_DONE` | byte-identical to `origin/main` (Sep 15 07:38, one test run) |
| evaluation files in training code | `train.py`: only `--data`. `expert_iter.py` reads the target / transfer / held-out pools and `validation_36` only for sampling, tracking and the (unused) relabel exclusion set. `run3_inject.py` reads one outside file per condition: for `sib*` a *sibling arm's `found_4.jsonl`*, i.e. verified RL proofs of theorems of the **same target pool** (see "Injected records"); for `gen4` / `other4` / `inv4` the four-record files in `data/r3/`. Nothing else from an evaluation file enters training. **OK by the brief's design** (sibling proofs are the brief's condition (a)); the fact that the sibling theorems are targets must be stated in the write-up |
| cap 6 on supervised data | both Stage-1 sets: 155,000 proofs, lengths 2–6 exactly 31,000 each, max 6, 0 violations. The injection step (`train.py --cap 0`, the EI round recipe) trains on 7–8-line sibling proofs and, in `inv4`, on 4 verifier-rejected strings (×4) — see flag below |
| hand-written / LLM-written training proofs | `gen4` / `other4`: generator output (`src: generator_*`), all 8 records verify, my key = stored key. `inv4`: 8 generator proofs with **one citation index edited** (`src: invalid_*`, verifier reason stored and reproduced by me); they are not proofs. The code that produced `data/r3/*.jsonl` is not in the repository (commit 3ef50a4 contains only the four files and `peek_samples.py`); the files themselves are committed and are 16 records in total |
| pre-registration before the first pod job | expectations R3-E1…E4 committed 22:14:24 UTC (734d51f); the first two injection manifests were written at 22:14:41 and 22:14:46, the first injected-step checkpoints at 22:15:29 / 22:15:37 (file mtimes preserved by the pull). Injection files committed 22:09:25 (3ef50a4). **Met, by 17 s** |

**Flag, not a quarantine.** The five `inv4` checkpoints (`ckpts/r3/*_inv4_*`) were fine-tuned on invalid strings.
The brief prescribes this control and the standard EI recipe already runs `train.py --cap 0` (no verification),
so nothing was weakened to make it runnable; but these models must not be used as submission models or quoted
as take-home results. The 25 other arms trained only on verifier-accepted proofs.

### Split disjointness (my renaming key, atoms relabelled by first appearance)

| file | classes | my key = stored `key` | ∩ targets_depth3 | ∩ transfer_depth3 | ∩ targets_reductio2 | ∩ transfer_reductio2 | ∩ heldout | ∩ val-36 |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| train_depth3_f0_a1 | 155,000 | – | 0 | 0 | 0 | 0 | 0 | 0 |
| train_reductio_f0 | 155,000 | – | 0 | 0 | 0 | 0 | 0 | 0 |
| targets_depth3 | 1,000 | ✓ | – | 0 | 0 | 0 | 0 | 0 |
| transfer_depth3 | 500 | ✓ | | – | 0 | 0 | 0 | 0 |
| targets_reductio2 | 606 | ✓ | | | – | 0 | 0 | 0 |
| transfer_reductio2 | 300 | ✓ | | | | – | 0 | 0 |

### Pattern frequency in the Stage-1 sets (my predicates, dependency-pruned)

| set | depth ≥ 3 | reductio |
|---|---:|---:|
| train_depth3_f0_a1 | **0** | 10,547 (6.8 %) |
| train_reductio_f0 | 5,687 (3.7 %) | **0** |

So f = 0 holds exactly for each arm's own pattern. Note for the `other4` control: the "other" pattern is
*not* novel for these models (the depth-3 set holds 10,547 reductio proofs; the reductio set 5,687 depth-3
proofs). `other4` therefore controls for "one extra fine-tuning step on four valid outside proofs", not for
"a second unseen pattern".

### Injected records (all 30 manifests; my verifier run, my predicates, my keys)

| condition | n | verifier-valid | has the arm's pattern | theorems are **targets** of the arm's pool | in transfer / held-out / val-36 | class in the arm's Stage-1 set | pattern proof in the mix | notes |
|---|---:|---:|---:|---:|---:|---:|---|---|
| sib1 / sib4 / sib16 (depth-3, from a1 s2 `found_4`) | 1 / 4 / 16 | all | all | **all** (1 / 4 / 16) | 0 | 0 | 7–8-line RL proofs | executor's manifest flags agree |
| sib1 / sib4 / sib16 (reductio, from s7 `found_4`) | 1 / 4 / 16 | all | all | **all** (1 / 4 / 16) | 0 | 0 | 7–8-line RL proofs | same |
| gen4 (depth-3 / reductio) | 4 | 4 | 4 | 0 | 0 | 0 | 6-line generator proofs | `data/r3/inject_*_gen4.jsonl` |
| other4 (depth-3 arms ← reductio gen4; reductio arms ← depth-3 gen4) | 4 | 4 | 0 (4 of the other pattern) | 0 | 0 | 1 of 4 (depth-3 arms) / 0 | | |
| inv4 (depth-3) | 4 | **0** | 4 (three box bars, IMPI chain intact) | 0 | 0 | 0 | | reasons: bad box cite ×3, ref out of range ×1 |
| inv4 (reductio) | 4 | **0** | 4 (AS ~G … NEGI … DN intact) | 0 | 0 | 2 of 4 | | reasons: bad box cite ×1, IMPE ×1, NEGE ×2 |

Two facts about the injections that the pre-registration states differently:

1. **The sibling theorems are targets.** The pre-registration says "the injected theorems are not targets".
   That is true of `gen4` / `other4` / `inv4` (class-disjoint from targets, transfer, held-out and val-36) and
   false of every `sib*` arm: the sibling files are `found_4.jsonl` of arms run on the *same* pool, so each
   `sib*` arm receives 1 / 4 / 16 verified proofs of its own targets. Every injected theorem is then re-found
   by the arm (injected acquired = 1 / 4 / 16 in all 15 sibling arms). The tables below therefore also give
   the counts **excluding the injected theorems**; the largest correction is 16 of 344 (depth-3) and 16 of
   80–95 (reductio).
2. **The invalid strings are one token from a valid proof.** Each of the 8 `inv4` records has exactly one
   single-token substitution of a citation index that makes it verify (my exhaustive search over all
   citation tokens × all line indices: 1 repair each; e.g. `IMPI N2 N5` → `IMPI N1 N5`). Box bars, formulas,
   rule sequence and 5 of 6 citations are those of a correct pattern proof. The brief asked for "the
   pattern's surface tokens but invalid structure"; what was injected is valid structure with an invalid
   citation. The control therefore tests whether the *verifier's acceptance* of the injected example matters,
   not whether its *structure* matters.

Sibling sources: a1 s2 `found_4.jsonl` = 10,202 records = 602 distinct start-index-normalised proofs, 284 with
depth ≥ 3 on 239 theorems (the pre-registration's "3,099 depth-3 proofs" counts start-index variants; 239
theorems agrees); reductio s7 `found_4.jsonl` = 1,902 records = 51 distinct proofs, all reductio, 51 theorems.

### Parents (the round-4 states) and their base rates

| arm | Stage-1 pre-RL pattern hits in 600k samples on 300 targets (ignition study's coverage files, as reproduced by the ignition review) | parent's cumulative pattern theorems, rounds 1…8 (my recount) | round-4 state |
|---|---|---|---|
| depth3 a1 s4 | 1 (1.7·10⁻⁶) | 0 0 0 0 0 0 1 4 | 0 pattern theorems; 232 targets solved (910 proofs in the mix) |
| depth3 a1 s5 | 0 | 0 0 0 0 0 0 0 0 | 0; 247 solved (968) |
| depth3 a1 s3 | 0 | 0 0 **1 3** 28 166 265 295 | **3 pattern theorems already found and trained on** at round 4; 233 solved (916) |
| reductio s1 | 0 (0 valid proofs of any kind) | 0 … 0 | Stage-1 model, empty found set |
| reductio s2 | 0 (0 valid proofs of any kind) | 0 … 0 | Stage-1 model, empty found set |

The pre-registration describes s3 as "0 pattern theorems at round 4 but self-ignited at round 5". Its round-4
found file holds 3 pattern theorems (first at round 3) and its round-4 fine-tune included them, so s3's
round-4 state is not a zero-pattern state: every one of its six conditions, including `other4` and `inv4`,
ignites at round 5 (42 / 45 pattern theorems against the parent's 28). s3 is a reference, as pre-registered,
not a fifth test of injection. The tests are s4, s5, s1, s2.

### Arms × conditions (identical continuation: rounds 5–8, k = 32, T = 0.8, retain 20k, resumed own found set)

Per target: pattern theorem = ≥ 1 verified proof whose dependency-pruned form has the pattern (my predicate);
round = first per-round file the (name, normalised proof) appears in (resumed round-≤ 4 proofs keep the
parent's round). Round 4 = the parent's resumed `found_4`. Ignition = cumulative ≥ 20 (depth-3) / ≥ 12
(reductio), as pre-registered. Every distinct pattern proof re-verified against its prompt with `nd_verify`;
200 random non-pattern proofs per arm likewise (0 failures anywhere). Per-round files are nested
(found_5 ⊆ … ⊆ found_8), the resumed records equal the parent's `found_4` exactly, the `round` field agrees
with first appearance for every proof, 0 unparsable proofs, and my cumulative solved counts equal the
executor's `round_<r>.json` at every round in all 30 arms.

| arm | condition | injected: n / valid / pattern / are targets | cum. pattern theorems r4 → r5 r6 r7 r8 | r8 excl. injected | ignition round (thr 20 / 12) | acq. r8 | solved r8 (exec.) | distinct pattern proofs verified / fail | transfer pattern r8 / n | held-out greedy min |
|---|---|---|---|---:|---|---:|---|---:|---:|---:|
| depth3_f0_a1_s4 | sib1 | 1 / 1 / 1 / 1 | 0 → 8 91 195 264 | 263 | 6 | 0.264 | 555 (✓) | 308 / 0 | 137 / 500 | 0.891 |
| depth3_f0_a1_s4 | sib4 | 4 / 4 / 4 / 4 | 0 → 105 250 308 322 | 318 | 5 | 0.322 | 613 (✓) | 389 / 0 | 170 / 500 | 0.899 |
| depth3_f0_a1_s4 | sib16 | 16 / 16 / 16 / 16 | 0 → 236 303 336 344 | 328 | 5 | 0.344 | 627 (✓) | 407 / 0 | 181 / 500 | 0.905 |
| depth3_f0_a1_s4 | gen4 | 4 / 4 / 4 / 0 | 0 → 1 27 98 167 | 167 | 6 | 0.167 | 480 (✓) | 200 / 0 | 88 / 500 | 0.902 |
| depth3_f0_a1_s4 | other4 | 4 / 4 / 0 / 0 | 0 → 0 0 0 0 | 0 | – | 0.000 | 333 (✓) | 0 / 0 | 0 / 500 | 0.881 |
| depth3_f0_a1_s4 | inv4 | 4 / 0 / 4 / 0 | 0 → 4 56 173 252 | 252 | 6 | 0.252 | 537 (✓) | 309 / 0 | 127 / 500 | 0.896 |
| depth3_f0_a1_s5 | sib1 | 1 / 1 / 1 / 1 | 0 → 23 135 198 262 | 261 | 5 | 0.262 | 607 (✓) | 311 / 0 | 133 / 500 | 0.897 |
| depth3_f0_a1_s5 | sib4 | 4 / 4 / 4 / 4 | 0 → 91 241 291 316 | 312 | 5 | 0.316 | 661 (✓) | 376 / 0 | 167 / 500 | 0.904 |
| depth3_f0_a1_s5 | sib16 | 16 / 16 / 16 / 16 | 0 → 143 250 294 312 | 296 | 5 | 0.312 | 625 (✓) | 365 / 0 | 172 / 500 | 0.902 |
| depth3_f0_a1_s5 | gen4 | 4 / 4 / 4 / 0 | 0 → 2 54 170 235 | 235 | 6 | 0.235 | 583 (✓) | 280 / 0 | 118 / 500 | 0.899 |
| depth3_f0_a1_s5 | other4 | 4 / 4 / 0 / 0 | 0 → 0 0 0 0 | 0 | – | 0.000 | 346 (✓) | 0 / 0 | 0 / 500 | 0.887 |
| depth3_f0_a1_s5 | inv4 | 4 / 0 / 4 / 0 | 0 → 0 0 0 0 | 0 | – | 0.000 | 339 (✓) | 0 / 0 | 0 / 500 | 0.889 |
| depth3_f0_a1_s3 | sib1 | 1 / 1 / 1 / 1 | 3 → 69 228 286 322 | 321 | 5 | 0.322 | 599 (✓) | 399 / 0 | 176 / 500 | 0.899 |
| depth3_f0_a1_s3 | sib4 | 4 / 4 / 4 / 4 | 3 → 101 258 298 326 | 322 | 5 | 0.326 | 595 (✓) | 404 / 0 | 176 / 500 | 0.890 |
| depth3_f0_a1_s3 | sib16 | 16 / 16 / 16 / 16 | 3 → 201 284 327 344 | 328 | 5 | 0.344 | 610 (✓) | 433 / 0 | 186 / 500 | 0.890 |
| depth3_f0_a1_s3 | gen4 | 4 / 4 / 4 / 0 | 3 → 84 238 290 322 | 322 | 5 | 0.322 | 589 (✓) | 397 / 0 | 168 / 500 | 0.891 |
| depth3_f0_a1_s3 | other4 | 4 / 4 / 0 / 0 | 3 → 42 197 274 294 | 294 | 5 | 0.294 | 602 (✓) | 369 / 0 | 158 / 500 | 0.888 |
| depth3_f0_a1_s3 | inv4 | 4 / 0 / 4 / 0 | 3 → 45 183 267 305 | 305 | 5 | 0.305 | 579 (✓) | 379 / 0 | 164 / 500 | 0.890 |
| reductio_f0_s1_t2 | sib1 | 1 / 1 / 1 / 1 | 0 → 6 24 31 44 | 43 | 6 | 0.073 | 44 (✓) | 45 / 0 | 17 / 300 | 0.874 |
| reductio_f0_s1_t2 | sib4 | 4 / 4 / 4 / 4 | 0 → 21 36 41 43 | 39 | 5 | 0.071 | 43 (✓) | 43 / 0 | 18 / 300 | 0.871 |
| reductio_f0_s1_t2 | sib16 | 16 / 16 / 16 / 16 | 0 → 51 62 75 80 | 64 | 5 | 0.132 | 80 (✓) | 80 / 0 | 39 / 300 | 0.875 |
| reductio_f0_s1_t2 | gen4 | 4 / 4 / 4 / 0 | 0 → 0 0 0 0 | 0 | – | 0.000 | 0 (✓) | 0 / 0 | 0 / 300 | 0.869 |
| reductio_f0_s1_t2 | other4 | 4 / 4 / 0 / 0 | 0 → 0 0 0 0 | 0 | – | 0.000 | 0 (✓) | 0 / 0 | 0 / 300 | 0.875 |
| reductio_f0_s1_t2 | inv4 | 4 / 0 / 4 / 0 | 0 → 0 0 0 0 | 0 | – | 0.000 | 0 (✓) | 0 / 0 | 0 / 300 | 0.870 |
| reductio_f0_s2_t2 | sib1 | 1 / 1 / 1 / 1 | 0 → 17 43 61 65 | 64 | 5 | 0.107 | 65 (✓) | 65 / 0 | 30 / 300 | 0.891 |
| reductio_f0_s2_t2 | sib4 | 4 / 4 / 4 / 4 | 0 → 32 57 65 65 | 61 | 5 | 0.107 | 65 (✓) | 65 / 0 | 32 / 300 | 0.896 |
| reductio_f0_s2_t2 | sib16 | 16 / 16 / 16 / 16 | 0 → 50 66 72 95 | 79 | 5 | 0.157 | 95 (✓) | 98 / 0 | 46 / 300 | 0.897 |
| reductio_f0_s2_t2 | gen4 | 4 / 4 / 4 / 0 | 0 → 5 29 53 61 | 61 | 6 | 0.101 | 61 (✓) | 62 / 0 | 30 / 300 | 0.894 |
| reductio_f0_s2_t2 | other4 | 4 / 4 / 0 / 0 | 0 → 0 0 0 0 | 0 | – | 0.000 | 0 (✓) | 0 / 0 | 0 / 300 | 0.902 |
| reductio_f0_s2_t2 | inv4 | 4 / 0 / 4 / 0 | 0 → 0 1 4 14 | 14 | 8 | 0.023 | 14 (✓) | 14 / 0 | 6 / 300 | 0.904 |

Notes on the table. Every reductio proof found is a pattern proof (solved = pattern theorems in all 12
reductio arms), as in every earlier reductio run. Held-out greedy stays at the parents' level (0.870–0.902)
in all arms, including the `inv4` ones — training on four invalid strings did not damage the models.
Ignition rounds are insensitive to the threshold: at 10 / 20 / 40 (depth-3) they change in two cells by one
round (s4 gen4 → 7 at 40; s5 sib1 → 6 at 40); at 6 / 12 / 24 (reductio) s1 sib1 is 5 / 6 / 6 and s2 inv4
is 8 / 8 / never. Excluding the injected theorems moves no ignition round except s1 sib1 at threshold 6.

**Comparison with the ignition study's full-sibling injection (ivS: the sibling's whole `found_4`, 239 / 51
theorems; my recount of those files).** Depth-3 s4 / s5 / s3: ivS 286 / 300 / 297 pattern theorems at round 5
and 330 / 335 / 336 at round 8. So: 16 proofs give 236 / 143 / 201 at round 5 and 344 / 312 / 344 at round 8
(the plateau, reached one round later); 4 proofs 105 / 91 / 101 → 322 / 316 / 326; 1 proof 8 / 23 / 69 →
264 / 262 / 322 (still rising at round 8). Reductio: 16 proofs 80 / 95 at round 8, 64 / 79 excluding the
injected 16, against 0.11–0.18 for the full injection in the ignition study.

**Transfer (never trained on).** The pattern generalises to the transfer pool in every igniting arm:
depth-3 sib1 137 / 133 of 500 at round 8, gen4 88 / 118, inv4 (s4) 127; reductio sib1 17 / 30 of 300,
gen4 (s2) 30, inv4 (s2) 6. 0 in every non-igniting arm.

**Reductio s2 `inv4` (the one marginal cell).** 14 / 606 at round 8, first proof at round 6, threshold
crossed at round 8 exactly. The executor's log (00:20) records that the first attempt of this very job
(same command, same seed; killed in the pod's OOM window and deleted) had 6 / 47 pattern theorems at rounds
5–6, against 0 / 1 in the surviving rerun. Two runs of the same seed thus differ by 46 theorems at round 6;
the surviving one is the one on file. The condition is at the edge of igniting for s2, and 0 for s1.

### Pre-registered expectations (734d51f, 22:14 UTC) against my values

| id | expectation | my value | verdict |
|---|---|---|---|
| R3-E1 | `sib16` ignites 5 / 5 by round 5 or 6 and reaches the plateau (depth-3 ≥ 0.30, reductio ≥ 0.10); `sib4` ignites ≥ 4 / 5; `sib1` ≥ 2 / 5 | sib16: 5 / 5 at round 5; r8 0.344 / 0.312 / 0.344 and 0.132 / 0.157 (0.328 / 0.296 / 0.328 and 0.106 / 0.130 excluding the injected 16) ✓; sib4 5 / 5 (all round 5) ✓; sib1 5 / 5 (rounds 6, 5, 5, 6, 5) | **met**, sib1 stronger than allowed |
| R3-E2 | `gen4` ignites ≥ 3 / 5; depth-3 more likely than reductio | depth-3 s4 (r6), s5 (r6), s3 (r5, reference); reductio s2 (r6), s1 never → 4 / 5 (3 / 4 tests) ✓; depth-3 2 / 2 vs reductio 1 / 2 ✓ | **met** |
| R3-E3 | `other4` ignites 0 / 5 (s3 excepted) | s4, s5, s1, s2: 0 / 0 / 0 / 0 pattern theorems through round 8; s3 ignites (294) | **met** |
| R3-E4 | `inv4` ignites 0 / 5; "if it ignites, what transfers is token statistics, not valid structure" | s4 ignites (252, r6), s5 never, s1 never, s2 14 at round 8 (marginal, same-seed replicate disagreed), s3 reference. 1 clear + 1 marginal of 4 tests | **wrong** for depth-3 s4; the pre-registered interpretation does not follow — the strings carry valid structure with one bad citation (above) |

### Reproducibility

`python3 review_run3_recount.py` and `python3 review_run3_splits.py` in the repository root regenerate every
number above from `artifacts/r3/`, `artifacts/p2/` (manifests, sibling and parent found files), `data/r3/`,
`data/p2/` and `targets/`; outputs in `artifacts/review_r3/`. Checkpoint provenance was read from the 56
`ckpts/r3/*.pt` files' stored training arguments (without torch): every `_r4t` was trained from the
parent's round-4 checkpoint (reductio: the Stage-1 model) on its manifest's mix for 600 steps at lr 3·10⁻⁴,
`--cap 0`, seed s·1000 + 4; every `_r8` from its own `_r7` on `mix_8.jsonl`. The 12 reductio mix files on disk
contain exactly the manifest's records ×4 plus 20,000 Stage-1 records (the 18 depth-3 mix files were not
pulled). Not re-derived: the pre-RL coverage counts (taken from the ignition study's summary, which the
ignition review reproduced), pod spend, kill-switch timing, and the bucket contents beyond the existence of
`round2/run3/{artifacts,ckpts,data}` (01:05 UTC).

## Compare (phase 2: `run3.md`, `numbers.md` §Round 2 — Run 3, `log.md`, `STATUS.md` §Run 3, `artifacts/r3/summary.json`, `figures/run3_ignition.png`)

Gate 0: expectations R3-E1…E4 committed 22:14:24 UTC, first injection step 22:14:41 (met). The
results-vs-expectations entry (log 01:06) reports R3-E4 as wrong for depth-3 and marginal for reductio, and
lists the counted-in reference s3 separately ("3 / 4 without it") — honest. The two OOM re-runs (depth-3 s5
`inv4` round 8; reductio s2 `inv4` whole arm) are disclosed in `numbers.md` together with the killed
attempt's 6 / 47, and the rsync nesting incident (gen4 briefly run twice, outputs deleted) is logged at 22:47;
the files on disk are single clean runs (checkpoint chains and mtimes consistent).

| claim (executor) | my independent value | verdict |
|---|---|---|
| `artifacts/r3/summary.json`: per-round pattern theorems, solved, ignition round, round-8 acquisition for 30 arms; parents' per-round lists | identical element for element in all 30 arms and 5 parents (own predicate, own normaliser, first-appearance round) | reproduces |
| `run3.md` table (24 cells, s4 / s5 / s1 / s2) | identical | reproduces |
| all injected records "verifier-valid, with the pattern" (sib, gen4); `inv4` "rejected by the verifier" | sib 1 / 4 / 16 valid and pattern in all 15 arms; gen4 4 / 4; inv4 0 / 4 valid, reasons as stored | reproduces |
| `numbers.md`: other4 "injected 4 (valid 4, with pattern 4)" | 4 valid, **0** with the arm's pattern (4 with the *other* pattern; the job passed `--pattern <other>` so the manifest flag records the other pattern) | reword |
| "sibling pattern proofs of distinct theorems" (numbers.md); pre-registration: "the injected theorems are not targets" | the sibling theorems **are targets** of the arm's own pool in every `sib*` arm (1 / 4 / 16); every one is re-found by the arm. Excluding them: sib16 r8 328 / 296 / 328 (depth-3), 64 / 79 (reductio); sib1 263 / 261, 43 / 64; no ignition round changes at the pre-registered thresholds | must be disclosed; conclusions unchanged |
| "A single verifier-valid sibling proof ignites 4 / 4 arms (0.26 depth-3, 0.07–0.11 reductio by round 8)" | 0.264 / 0.262; 0.073 / 0.107 (0.263 / 0.261; 0.071 / 0.106 excluding the injected theorem) | reproduces |
| "16 proofs reach the plateau a round earlier" | sib16 236 / 143 / 201 at round 5 vs sib1 8 / 23 / 69; sib16 within 10 of the full-sibling ivS plateau (330–336) by round 7–8, sib1 still rising at round 8 | reproduces |
| "It must be the right pattern: the other pattern ignites 0 / 4" | 0 pattern theorems through round 8 in s4, s5, s1, s2; but the "other" pattern is present in each arm's Stage-1 set (10,547 reductio proofs in the depth-3 set, 5,687 depth-3 proofs in the reductio set), so `other4` shows that the extra fine-tuning step on four valid proofs without the pattern does not ignite — it does not test a second *novel* pattern | reproduces; add the caveat |
| "Four short generator instances of the shape ignite 3 / 4" | s4 (r6), s5 (r6), s2 (r6); s1 never (0 valid proofs of any kind) | reproduces |
| "four *invalid* strings that merely carry three box bars ignite depth-3 s4 (0.252, not s5) … for the structural pattern the seed can be a token statistic, not a proof" | s4 252 (r6), s5 0, s3 305 (reference); s2 14 at r8 (killed same-seed attempt 47 at r6), s1 0. The strings do not "merely carry three box bars": each is a complete 6-line depth-3 (or reductio) proof with **one** citation index changed — exactly one single-token repair makes each verify. Box structure, formulas and rule sequence are those of a correct pattern proof | numbers reproduce; **interpretation not supported** — reword to "a structurally correct pattern example that the verifier rejects for one bad citation ignites one of the two depth-3 test draws; the verifier's stamp is not what transfers, the structure is" |
| "these are not proofs" (inv4) | correct; and the 5 `inv4` checkpoints were fine-tuned on them | reproduces; label the checkpoints |
| "parents: 0–4 throughout" (table caption) | true of s4, s5, s1, s2 (the table's columns); s3 (not in the table) reaches 295 | reproduces |
| pre-registration: s3 "0 pattern theorems at round 4" | 3 (first at round 3), trained on at round 4 | wrong in the pre-registration; s3 is correctly excluded from the tallies |
| held-out intact | greedy 0.869–0.905 in all 30 arms, parents 0.870–0.902 | reproduces (not claimed in run3.md; worth one clause) |
| transfer generalisation | not reported by the executor: depth-3 sib1 137 / 133 of 500, gen4 88 / 118, inv4 s4 127; reductio sib1 17 / 30 of 300 | add: the pattern reaches never-trained theorems in every igniting arm |
| STATUS "invalid strings with the box tokens ignite one depth-3 draw" | s4 yes, s5 no (s3 reference also) | reproduces |
| `run3.md` ≤ 400 words | 489 words excluding the figure line | over the brief's length |
| figure `run3_ignition.png` | every curve matches my table | reproduces |
| bucket `round2/run3/{artifacts,ckpts,data}`; pods ≈ $1.6 | the three directories exist (01:05 UTC); spend not derivable | reproduces / not derivable |

## Verdict

**What stands.** Every number in `run3.md`, `numbers.md` §Run 3 and `artifacts/r3/summary.json` reproduces
from the raw per-round files with independent code, every counted pattern proof re-verifies (5,000+ distinct
proofs, 0 failures), the per-round files are internally consistent, the checkpoints were trained from the
stated states on the stated mixes, the two Stage-1 sets are exactly f = 0, and every pool is class-disjoint
from every training set. The three clean results are: (i) one verifier-valid sibling proof, repeated four
times inside one 600-step fine-tune against 20,000 retained records, ignites all four test arms (two depth-3
draws with 0–1 pattern hits in 600k pre-RL samples, two reductio draws with 0 valid proofs of any kind), and
four proofs reach the full-sibling plateau within four rounds; (ii) the same step with four valid proofs
that lack the pattern ignites nothing (0 / 4), so it is the injected pattern and not the extra training that
ignites; (iii) four 6-line generator instances of the pattern on unrelated theorems ignite 3 / 4 — the
pattern transfers from short, easy, class-disjoint examples to 7–12-line targets and to the never-trained
transfer pool.

**What must be reworded.** The `inv4` sentence. The injected strings are not "strings that merely carry
three box bars": they are complete pattern proofs one citation token away from verifying, so the control
did not separate "surface tokens" from "valid structure" — it separated *verifier acceptance* from
structure. The supported statement is that a structurally correct example need not be verifier-valid to
seed the pattern (1 clear of 2 depth-3 draws; 0 and a marginal 14 / 606 for reductio, where the killed
same-seed attempt had reached 47 at round 6). The pre-registered inference "token statistics, not valid
structure" does not follow from this design. Two disclosures are missing: the sibling-injected theorems
are targets of the arm's own pool (all are re-found; the effect on every count is ≤ 16 theorems and no
ignition round moves), and `other4`'s pattern is not novel to the models (it is in each Stage-1 set at
4–7 %). Minor: `numbers.md`'s "with pattern 4" for `other4` means the other pattern; the pre-registration's
"s3: 0 pattern theorems at round 4" is 3; `run3.md` is 489 words against the brief's 400.

**Flags, not quarantine.** The five `inv4` checkpoints were fine-tuned on verifier-rejected strings (the
brief's control, run under the standard `--cap 0` EI recipe) and must not be quoted as take-home models. The
code that built the four injection files is not in the repository; the files are.

**Next measurement.** (1) Repeat `inv4` with strings whose *structure* is broken while the token statistics
are kept — e.g. the same three box bars with the IMPI chain scrambled or the closing formulas randomised —
on s4, s5 and the two reductio draws, two sampling seeds each; this is the control the pre-registration
described, and the same-seed disagreement on reductio s2 (47 vs 1 at round 6) says single runs of marginal
cells are not enough. (2) Find the floor below one proof: one sibling proof at weight 1 instead of 4, or at
a tenth of the learning rate, to measure how much gradient the ignition needs. (3) Inject sibling proofs of
non-target theorems (the sibling's transfer finds are evaluation data, so use gen4-style class-disjoint
theorems at 7–9 lines) to make the sibling condition free of the target leak.
