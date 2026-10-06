# Review of run `lean-format` (proposal 8: Lean as the training format for the from-scratch model)

Reviewer: agent:claude (reviewer role, separate session from the executor). Started 2026-09-21 08:20 UTC.

## Recount (phase 1 — written and committed before reading `run_lean_format.md`, `numbers.md`, `log.md`, `STATUS.md`)

Workspace `~/review/lean-format` (write-ups removed). Everything below comes from my own code —
`review_lean_format_recount.py` (own ND line parser, start-index normaliser, dependency pruner, box-depth counter, `L*`),
`review_lean_format_splits.py` (own renaming-class canonicaliser: minimum over the 24 atom permutations, premises
order-insensitive; f = 0 check; sequence lengths), `review_lean_format_lean_recheck.py` (own Lean batching harness around the
unmodified `nd2lean.translate`). Outputs: `artifacts/review_lf/`. I did not import or read `lean_format_analysis.py`,
`analysis_stdout.txt`, `summary.json`, `seqlen.json`, `disagreement_kinds.json` or the `record_*` files.

### Hard constraints

| check | result |
|---|---|
| `nd_verify` hash = `origin/main` | **ok** — `__init__.py` dfa3bc3, `verify.py` 1cfed53 in the worktree, the review copy and `origin/main` |
| `nd2lean.py` unmodified by this run | ok — blob 1d5cf06 = the pre-run commit d3dfc93 = `origin/dan_ladder_a` |
| `artifacts/TEST_RUN_DONE` unchanged | ok — blob 1667609, last touched by ca93f83 (the take-home's single test run) |
| evaluation / test files in training code | ok — no `test_*` path in `lean_tok.py`, `lean_gate.py`, `train.py`, `sample.py`, `expert_iter.py`, `ladder_ei.py`, `pod/lf/*`; transfer and held-out pools are only sampled and judged (`found_t` never enters a mix). Note: `pod/lf/sync.sh` copies the whole `targets/` directory (which holds `test_*_prompts.jsonl`) to the pods; nothing reads it except `validation_36.jsonl` for the exclusion keys |
| cap 6 on supervised data | ok — both Stage-1 files: ND line histogram 2–6 only (31,000 / ≈ 30,998 per length); my render → `inverse` round trip returns the identical ND proof on 155,000 / 155,000 and 154,990 / 154,990 records, so the Lean text denotes exactly the cap-checked proof |
| f = 0 really zero | ok — `train_depth3_f0_a1.jsonl`: ND max box depth {0: 82,393, 1: 54,883, 2: 17,724}, **0 at depth 3**; the nested-`fun`-box depth of the Lean rendering has the same histogram (0 at depth 3). (`data/train.jsonl`, used for the ladder, has 5,664 depth-3 proofs, as before.) |
| same model, same schedule | ok — read from the checkpoints without torch: every Lean Stage-1 is 4 layers / d 256 / 8 heads, vocab 107, 3,214,336 parameters (token: vocab 99, 3,210,240; difference = 8 rows × 256 × 2); args 6,000 steps, bs 128, lr 1e-3 → 1e-4, warm-up 200, wd 0.1, cap 6, seeds as registered; `seqfixed` has `no_shift: True`; EI fine-tunes 600 steps at 3e-4 |
| pre-registration before first pod | ok — commit c2dcd0a 03:00:36 UTC; first pod `lf-1` 03:04:23 UTC (`~/pods.log`). **Four** pods were created (`lf-1` … `lf-4`); the pre-registration said two, a third only if wall-clock required it |
| pods deleted | ok — `podls` lists none at 08:35 UTC; balance $121.52 |

### Split disjointness (own canonicaliser)

Premise-order-**sensitive** classes (the repository's definition): 0 overlaps on all 14 pairs checked (both Stage-1 files × their
targets / transfer / held-out pools; targets × transfer; ladder pools × take-home held-out / transfer).
Premise-order-**insensitive** classes (stricter than the repository's): `train_a1` × `p2/heldout` 21 classes, `train` ×
`heldout` 95, `train` × take-home `transfer` 2, ladder `rl_targets` × ladder `transfer` 2 (`la_transfer_307`, `la_transfer_842`,
both `L_true` 9); every other pair 0. These files pre-date this run and are shared with the token comparators, so they cannot
create a Lean-vs-token difference; 95 / 5,000 = 1.9 % of the held-out set is not negligible next to a 2 pp band, though, and
is the same for both formats. No effect on any `L*` (two `L_true` 9 theorems).

### Every counted proof re-checked: `nd_verify` and Lean

All cumulative round-8 found files of the 16 arms (both pools) and all mechanism-test proofs: **58,201 proofs, `nd_verify`
accepts 58,201, Lean 4.34 (through the unmodified `nd2lean.translate`, my harness, flagged theorems re-run alone) accepts
58,201, 0 translation errors, agreement 58,201 / 58,201.** Every proof's prompt equals its pool record's prompt. A 150-proof sample
per arm and pool re-rendered in the training surface form (`lean_tok`) also checks in Lean: 4,800 / 4,800. Harness negative
control (theorem-swapped): 0 / 138 accepted by either checker. My normaliser finds raw = normalised-distinct in every Lean-arm
file (the denoted ND proof is numbered from N1 by construction).

**Not reproducible from the pulled files:** the literal sampled Lean text is not saved for accepted proofs (found files hold
the denoted ND proof only), so acceptance condition (ii) of the pre-registration — "Lean accepts the literal sampled text" —
can be audited only through the gate logs' counts and the disagreement files, not re-run.

### P1 / P1b — Stage-1 held-out greedy (from the frozen arms' `round_*.json`, identical in all 8 rounds; no per-sample file for the a1 models, no torch on the VPS, so aggregates only)

| model | held-out greedy | band | |
|---|---|---|---|
| token a1 s0 / s1 (on file) | 4,413 / 4,415 of 5,000 = 0.883 / 0.883 | — | reproduces |
| `lean_rand` a1 s0 / s1 | 4,408 / **3,949** = 0.882 / **0.790** | ≥ 0.863 | **s1 fails by 7.3 pp** |
| `lean_seq` a1 s0 / s1 | 4,543 / 4,480 = 0.909 / 0.896 | ≥ 0.863 | passes both |
| token full (on file) | 4,739 = 0.948 | — | reproduces |
| `lean_rand` full | 4,149 = **0.830** (per-sample jsonl agrees; all 4,149 re-verified) | ≥ 0.928 | **fails by 9.8 pp** |
| `lean_seq` full / `seq --no_shift` | 4,678 = 0.936 / 4,696 = 0.939 | ≥ 0.928 | passes |

### P2 — depth-3 f = 0 dial (acquisition = fraction of the 1,000 targets with an accepted proof whose dependency-pruned form has box depth ≥ 3; written-depth gives identical numbers)

| arm | solved / 1,000 | acquisition | **frozen control, same Stage-1, equal attempts** | EI − frozen | transfer acq (500): EI / frozen |
|---|---|---|---|---|---|
| token s0 (on file, my counter) | 645 | 0.335 | 0.005 (5 targets; solved 175) | 0.330 | 0.372 / 0.004 |
| token s1 | 698 | 0.364 | 0.005 (solved 175) | 0.359 | 0.366 / 0.004 |
| `lean_rand` s0 | 663 | **0.433** | **0.202** (solved 351) | 0.231 | 0.472 / 0.198 |
| `lean_rand` s1 | 673 | **0.462** | **0.280** (solved 423) | 0.182 | 0.476 / 0.302 |
| `lean_seq` s0 | 715 | **0.476** | **0.206** (solved 409) | 0.270 | 0.494 / 0.184 |
| `lean_seq` s1 | 736 | **0.479** | **0.134** (solved 326) | 0.345 | 0.512 / 0.156 |

All four Lean arms are above 0.27 and above 0.40. **The finding that changes the reading: the Lean Stage-1 models write depth-3
proofs before any RL.** First depth-3 proof is in round 1 in every arm, frozen included (round-1 acquisition 0.085–0.172 at
k = 32); the frozen controls reach 0.13–0.28 at the full 256 attempts, against 0.005 for the token format (the
pre-registration expected ≤ 0.01 and wrote "frozen 0" for the token value; it is 5 / 1,000). The training file has no depth-3
proof in either rendering (above), so this is composition by the base model, not a leak. Consequence: in the Lean format the
"f = 0 dial" is not at zero behaviourally, and the EI-attributable part of acquisition (EI − frozen, 0.18–0.35) is **not
larger** than the token format's (0.33 / 0.36). Almost every frozen-acquired target is also EI-acquired (frozen-only 0–2).

### P3 — ladder T1 (`L*` = largest L with ≥ 5 distinct solved theorems of `L_true` ≥ L, cumulative 8 × 32)

| arm | transfer solved / 2,285 | transfer `L*` | solved at `L_true` ≥ 10 / ≥ 11 | frozen: solved, `L*` | `L*` − `L*`_frozen | targets `L*` (T1 / frozen) |
|---|---|---|---|---|---|---|
| token s0 / s1 (from `origin/dan_ladder_a`, my code) | 612 / 623 | 10 / 10 | 35 / 1, 34 / 0 | 22 / 24, `L*` 7 / 7 | **3 / 3** | — |
| `lean_rand` s0 / s1 | 583 / 657 | 10 / 10 | 46 / 2, 44 / 4 | 229 / 222, `L*` **9 / 9** | 1 / 1 | 10 / 9 |
| `lean_seq` s0 / s1 | 794 / 839 | **11 / 11** | 95 / 12, 94 / 13 | 304 / 309, `L*` **10 / 10** | 1 / 1 | 11 / 10 |

`L*` by round: `lean_rand` 9, 9, 10 … (both seeds); `lean_seq` 10, 10, 10, 10, 11 … (both seeds); frozen flat. Pool files are
byte-identical to the token ladder's (`transfer.jsonl` e0524d0, `rl_targets.jsonl` 69233bc). T1-solved transfer theorems not
solved by either frozen seed: `lean_rand` 348 / 415, `lean_seq` 464 / 509 (at `L_true` ≥ 11: 2 / 4 and 12 / 13 — i.e. every
`L_true` ≥ 11 solve is EI-only). As with P2, most of the Lean-vs-token difference is already in the base model: the Lean
Stage-1 models solve 222–309 transfer theorems frozen, the token model 22–24.

### P4 — mechanism test (full-train Stage-1, pass@16 on `data/transfer.jsonl`, distinct proofs per theorem after my normaliser)

| model | solved / 1,638 | distinct | written 7 | written 8 | ≥ 9 | pruned 7 / 8 / 9 (own pruner) |
|---|---|---|---|---|---|---|
| token `abs` / `rel` / `abs-fixed` (aggregate json on file; no per-sample file) | 732 / 664 / 631 | 926 / 817 / 774 | 108 / 79 / 0 | 1 / 0 / 0 | 0 | 51 / 0 / 0 (`abs`) |
| `lean_rand` | 859 | 1,067 | **228** | **86** | 7 | 226 / 84 / 7 |
| `lean_seq` | 935 | 1,225 | **266** | **130** | 18 | 265 / 120 / 18 |
| `lean_seq --no_shift` | 466 | 546 | **37** | 0 | 0 | 35 / 0 / 0 |

Against the pre-registration: `lean_rand` / `lean_seq` expected 50–150 — both above the range (miss, in the favourable
direction); `--no_shift` expected ≤ 2 — **37, a miss**: the barrier is weakened, not reproduced (the ablation model also
solves half as many theorems as `lean_seq`, 466 vs 935, while scoring slightly *higher* in distribution, 0.939 vs 0.936).

### P5 — in-loop gate, Lean (literal text) vs `nd_verify` (denoted proof), summed over all 24 gate logs

18,426,794 samples; 3,515,138 outside the grammar (19.1 %; frozen arms 18–44 %, mostly `unbound` names and `expected )`);
13,889,708 distinct (prompt, text) pairs checked: both accept 4,047,945, both reject 9,841,303, **`nd_verify`-only 0,
Lean-only 460** (61 distinct denoted proofs) = 33 per million checked. Disagreement-file rows = logged counts in every log. My
classification of the 460 rows (re-running `nd_verify` on each): 206 `BOTE` where `.elim` was applied to a non-`False` term
(`Not.elim`: `n : ¬A ⊢ n.elim : A → B`, so Lean accepts a step that is not ⊥-elimination); 117 premise lines omitted or out of
order (Lean does not need them restated); 137 `¬A` ≡ `A → False` (NEGE 67, ORI2 36, IMPI-for-NEGI 30, ANDE1 2, ORI1 1, ANDI 1).
All are valid Lean, none is a counted proof (acceptance needs both), and none is a soundness problem for counts. Against the
pre-registration (≤ 10 per million, all of the `¬A` kind): **rate missed (33 per million) and two unanticipated kinds**.

### P6 — like-for-like single frozen round on the same pod (`secs` in `round_1.json`)

| pair | token | Lean | ratio |
|---|---|---|---|
| depth-3, `rand` pod | 59 s | 105 s | 1.78 |
| depth-3, `seq` pod | 64 s | 141 s | 2.20 |
| ladder, `rand` pod | 406 s | 594 s | 1.46 |
| ladder, `seq` pod | 402 s | 657 s | 1.63 |

All below the 3× stop; one of four (2.20) is above the expected 1.2–1.8×. One round each, n = 1. Gate throughput from the logs:
Lean 53–222 checked / process-second (pre-registered ≈ 250 / s / core), 580–2,070 / wall-second with 12 workers; `nd_verify`
8,000–27,000 / s — Lean is ≈ 100–200× slower per core than `nd_verify`, hidden by parallelism.

### P7 — sequence lengths (prompt + proof tokens incl. end token; own count through both tokenizers)

| file | Lean mean | token mean | ratio of totals |
|---|---|---|---|
| `train_depth3_f0_a1` | 54.2 + 84.3 | 39.1 + 73.6 | 1.229 |
| `train` | 51.8 + 82.8 | 36.9 + 72.1 | 1.233 |
| `p2/heldout` | 54.1 + 92.2 | 39.7 + 81.3 | 1.210 |
| `heldout` | 51.8 + 83.2 | 37.0 + 72.6 | 1.232 |

Ratio ≈ 1.23 as expected (1.24). The pre-registration's absolute proof means (65 Lean / 57 token on "3,000 held-out proofs")
do not match the full held-out file (83 / 73); the prompts match. Vocabulary 107 vs 99 confirmed.

### Decision rule, applied to my numbers

| scheme | P1 ≥ 0.863 both a1 seeds | P2 ≥ 0.27 both seeds | P3 `L*` ≥ 10 both seeds | verdict by the registered rule |
|---|---|---|---|---|
| `lean_rand` (**primary**) | **no** (0.882 / 0.790) | yes (0.433 / 0.462) | yes (10 / 10) | **worse** (fails P1; P1b also fails, 0.830) |
| `lean_seq` (secondary) | yes (0.909 / 0.896) | yes (0.476 / 0.479) | yes (11 / 11) | **not worse** |

"Lean not worse" overall (at least one scheme passes all three) holds through the secondary scheme only — two tries at a fixed
band, as the pre-registration says. What I will look for in phase 2: that the write-up (i) names the primary scheme's failure,
(ii) reports the frozen controls (0.13–0.28 acquisition, `L*` 9–10) beside every EI number rather than comparing Lean EI with
token EI alone, (iii) reports the P4 `--no_shift`, P5 and frozen-control misses as misses, and (iv) does not call `lean_seq`'s
`L*` 11 an RL gain over the token format without the `L*` − `L*`_frozen = 1 vs 3 comparison.

## Compare (phase 2 — after reading `run_lean_format.md`, `numbers.md` § lean-format, `log.md` § lean-format, `STATUS.md`, `QUESTIONS.md`)

Extra checks made for this phase are in `artifacts/review_lf/phase2_checks_stdout.txt`.

| # | executor's claim | my independent value | verdict |
|---|---|---|---|
| 1 | P1 a1 held-out: token 0.883 / 0.883, `lean_seq` 0.909 / 0.896, `lean_rand` 0.882 / 0.790; full set 0.948 / 0.936 / 0.830; `--no_shift` 0.939 | 4,413 / 4,415; 4,543 / 4,480; 4,408 / 3,949; 4,739 / 4,678 / 4,149; 4,696 of 5,000 | reproduces (a1 values from round-file aggregates only — no per-sample file, not re-runnable on the VPS) |
| 2 | `lean_rand` full set by length 0.999 / 0.979 / 0.864 / 0.645 / 0.662; 851 failures = 813 rejected by both + 38 outside grammar | identical | reproduces |
| 3 | "`lean_rand` failures are mostly wrong-name citations" | tested directly: the first failing line becomes valid when a different earlier line is cited in **451 of 851** failures (53 %); 347 are not fixable by re-citing (wrong rule or formula). `lean_seq`: 69 of 322 | partly — about half of the failures, ≈ 72 % of the *excess* over `lean_seq` (382 of 529). "Mostly" holds for the excess, not for the total |
| 4 | P2 acquisition EI 0.476 / 0.479, 0.433 / 0.462; frozen 0.206 / 0.134, 0.202 / 0.280; token 0.335 / 0.364, frozen 0.005; solved, depth-3 proof counts, transfer columns, first round r1, by-round curves (Lean s0 and token s0) | identical in every cell (own pruner and depth counter; token by-round from the `round` field of `found_8`) | reproduces |
| 5 | "frozen Lean models solve 8.5–17.2 % of targets with a depth-3 proof in 32 attempts, 13.4–28.0 % in 256"; EI adds +0.27 / +0.35, +0.23 / +0.18 (token +0.33 / +0.36) | 85–172 / 1,000; 134–280; 0.270 / 0.345, 0.231 / 0.182; 0.330 / 0.359 | reproduces |
| 6 | P3 transfer `L*` 11 / 11 (`lean_seq`), 10 / 10 (`lean_rand`), frozen 10 / 10 and 9 / 9; token 10 / 10, frozen 7 / 7; solved 794 / 839, 583 / 657, 304 / 309, 229 / 222, 612 / 623, 22 / 24; by-`L_true` rows; targets `L*` and solved; `L*` = 11 from round 5 | identical (token arms recounted from `origin/dan_ladder_a`; pool files byte-identical) | reproduces |
| 7 | 12 / 13 theorems at `L_true` ≥ 11 (token 1 / 0); ≥ 12: 3 / 4; no label contradicted by a shorter accepted proof | 12 / 13, 3 / 4; 0 theorems in any Lean arm (either pool) have an accepted proof shorter than `L_true`; shortest written proofs at `L_true` ≥ 11 are 11–13 lines | reproduces |
| 8 | `L_true` = 7 bin solved less often by Lean T1 arms (55–118 of 300 vs 139–140) | 55, 114, 75, 118 vs 140, 139 | reproduces; reported as unexplained, which is accurate |
| 9 | ladder held-out r1 → r8: 0.936 → 0.944 / 0.943; 0.830 → 0.923 / 0.924 | 4,678 → 4,718 / 4,713; 4,149 → 4,617 / 4,619 | reproduces |
| 10 | P4: pass@16 935 / 859 / 466; 7- / 8- / ≥ 9-line 266 / 130 / 18, 228 / 86 / 7, 37 / 0 / 0; token 108 / 1, 79 / 0, 0 / 0 | identical; token values only from the aggregate json on file | reproduces |
| 11 | log 05:50: `--no_shift` reaches 37 because "the model re-uses a name whose binding is out of scope — shadowing" | literal texts of accepted proofs were not saved | **not derivable** |
| 12 | P5: 18,424,794 samples, 13,887,708 checked, both accept 4,046,345, both reject 9,840,903, 0 / 460; 33 per million; kinds 206 / 137 / 117; parse-fail 19.1 % (`rand` 20.7 %, `seq` 17.4 %) | mine are larger by exactly the pod self-test log (2,000 samples, 1,600 + 400), which the executor excludes; excluding it, identical. Kinds 206 / 137 (67 + 37 + 30 + 3) / 117 from my own classifier; 0.207 / 0.174 | reproduces |
| 13 | checker of record 41,840 / 41,840 | 41,840 of my 58,201 are the 16 arms' found files (the other 16,361 are mechanism-test proofs); all accepted by `nd_verify` and by Lean through my own harness | reproduces, and extends to the mechanism-test proofs |
| 14 | Lean 85.5 / process-s vs `nd_verify` 14,497 / s ≈ 170× | 85.5; 14,497; 169× | reproduces |
| 15 | solo round ratios 1.8× / 2.2× / 1.46× / 1.64× | 1.78 / 2.20 / 1.46 / 1.63 | reproduces (one round per cell) |
| 16 | sequence-length table, ratio 1.21–1.23 | identical means and ratios | reproduces |
| 17 | cost ≈ $6.2; pods deleted; bucket uploaded | balance $121.52 at 08:35; `podls` empty; bucket has `artifacts`, `ckpts`, `data`, including the per-round `found_1…7` files that git ignores | reproduces |
| 18 | gate change at 03:33 was diagnostic only | diff 63f94b2 → c544d90: only the `LEANREJ` prefix on both-reject samples and a parse-reason field changed; acceptance (Lean ∧ `nd_verify`) is the same. The two mechanism evaluations that ran under the old gate (03:17–03:19) are unaffected — I re-verified all their proofs | reproduces |

Every number in `run_lean_format.md` and `numbers.md` § lean-format reproduces. No hard-constraint violation; no quarantine.

### Expectations and misses

Pre-registration committed 03:00:36, first pod 03:04:23 — before the run. Misses the executor reports as misses: `L*` "stays
10" (wrong, 11 for `lean_seq`); `lean_rand` in-distribution loss (expected ≤ 5 pp); disagreement rate (33 vs ≤ 10 per
million) and its two unanticipated kinds; frozen depth-3 acquisition (expected ≤ 0.01, found 0.13–0.28 — made the headline);
`--no_shift` ≤ 2 (found 37; called wrong in `log.md` 05:50). Deviations logged: four pods instead of two; timing rounds run
solo instead of co-tenant; the grammar does not mirror "every premise restated". Not flagged, all minor: the `--no_shift`
miss is in the log but not in the write-up, which presents 37 / 0 only as confirmation; the pre-registration's token
"frozen 0" is 0.005; P6 `lean_seq` depth-3 ratio 2.2× is above the expected 1.2–1.8×; P7's pre-registered absolute proof
lengths (65 / 57) were measured on the first 3,000 held-out records (lengths 2–4; log 03:00) and differ from the full-file
83 / 73 without a note; frozen ladder `L*` was expected 7 and is 9–10 (reported, not labelled a miss); the write-up is 588
words against the brief's 400.

### Wording against n

- **"On depth-3 and length generalisation Lean is *better* than tokens."** Holds for `lean_seq`. For `lean_rand` it holds on
  the depth-3 dial only: ladder `L*` is equal (10 / 10) and transfer solved is 583 / 657 against 612 / 623.
- **"`lean_seq` reaches `L*` = 11 in both seeds."** The two seeds are EI seeds from **one** Stage-1 model (the caveat is in
  the write-up's last line, not beside the claim). `lean_rand`'s two a1 Stage-1 seeds differ by 9 pp in distribution and by
  0.08 in frozen acquisition, so Stage-1 seed variance in this format is large; a format-level "`L*` = 11" needs a second
  Stage-1 seed. The same applies to the token comparator.
- **`L*` − `L*`_frozen** is the ladder's registered headline and is not stated: it is **1 / 1** for both Lean schemes against
  **3 / 3** for the token format. The write-up gives the frozen `L*` values, so the information is there, but the sentence
  "the proposal's result that would matter" should carry it: the `L*` = 11 is a base-model gain plus the same-or-smaller EI
  step, not a larger RL effect.
- **"`lean_rand` loses 9–12 pp in distribution."** a1 s0 loses 0.1 pp, a1 s1 9.3 pp, the full set (one seed) 11.8 pp. It is
  "0 to 12 pp, strongly seed-dependent".
- **"The name-offset barrier is format-independent: `lean_seq` without offsets drops to 37 / 0."** One Stage-1 seed; the
  token `abs-fixed` value is 0 and the pre-registered expectation was ≤ 2, so the measured statement is "present but weaker
  in Lean (37 vs 0 seven-line proofs, 0 vs 0 eight-line)". The shadowing explanation in the log cannot be checked from the
  pulled files.
- **"So part of the token format's 'new capability from RL' was a surface-form barrier."** Supported as worded ("part of"):
  two Stage-1 seeds per scheme on the depth-3 dial, all four frozen controls at 0.13–0.28 against 0.005, training file
  verified to hold no depth-3 proof in either rendering. The unary-prefix explanation is labelled untested, correctly.
- **Verdict line.** Matches the registered rule applied to my numbers (`lean_seq` not worse; `lean_rand`, the primary,
  worse on P1). The pre-registration promised to say plainly that "not worse overall" rests on two tries at a fixed band;
  the write-up names both schemes and which was primary but does not use those words.
- No "never", "wall" or "bistable" claims.

## Verdict

**What stands.** All counts, rates, frontiers, timings and costs. The per-scheme decision-rule verdict. 58,201 / 58,201
counted proofs accepted by `nd_verify` and by Lean; 0 "`nd_verify` yes, Lean no" among 13.9 M gate checks. The run's most
important result for the project question stands and is robust at n = 4 Stage-1 models: **a Lean-format base model trained on
zero depth-3 proofs writes depth-3 proofs for 13–28 % of targets at 256 attempts, where the token-format base writes them
for 0.5 %**, and solves ≈ 10× more ladder transfer theorems frozen (222–309 vs 22–24). The EI increment over the frozen
control is no larger in Lean than in tokens (acquisition +0.18 to +0.35 vs +0.33 / +0.36; `L*` +1 vs +3).

**What must be reworded.** (1) "Lean is better on length generalisation" → `lean_seq` only. (2) Put `L*` − `L*`_frozen
(1 vs 3) and "one Stage-1 model" beside the `L*` = 11 claim. (3) "loses 9–12 pp" → 0 / 9 / 12 pp by model. (4) The
`--no_shift` result as a weakened barrier and a missed prediction, one seed. (5) "mostly wrong-name citations" → about half
of all failures, about 70 % of the excess over `lean_seq`. (6) Add the "two tries at a fixed band" sentence.

**What is not supported.** Nothing numerical. Two explanations are untested and should stay labelled as such: unary depth
prefix vs nested `fun` (write-up, labelled) and name shadowing behind the 37 (log, not labelled).

**Consequences the write-up does not draw, for whoever acts on the decision rule.**
- If `lean_seq` becomes the training format, the campaign's depth-3 f = 0 dial no longer measures "RL crossing f = 0": the
  pattern is base-reachable (0.085–0.172 of targets in the first 32 attempts). Depth-3 results in Lean must be read as
  EI − frozen, and the creation-vs-elicitation question needs a pattern the Lean base does not reach.
- Acceptance condition (ii), Lean on the literal sampled text, is not auditable after the fact because accepted literal
  texts are not stored. Future Lean-format runs should write the literal text beside the denoted proof in `found_*.jsonl`.
- `nd2lean.py`'s BOTE rendering is loose (`.elim` resolves to `Not.elim` on a negation; 206 of the 460). This run is
  protected by requiring `nd_verify` too, but any pipeline that uses Lean through `nd2lean` alone is not. The executor's
  question in `QUESTIONS.md` (render BOTE as `False.elim na`) is the right one and is still open.
- Held-out sets overlap their Stage-1 files in 21 / 5,000 and 95 / 5,000 classes once premise order is ignored. Same for
  both formats, so P1 comparisons are fair, but 1.9 % is the size of the P1b margin that `lean_seq` passed by (0.8 pp inside
  the band).

**Next measurements that would settle what is open.**
1. Two more Stage-1 seeds per scheme on the full set, each with a frozen ladder control and one T1 arm (≈ 6 Stage-1 + 12
   ladder arms, ≈ $8 at this run's rates): decides whether `L*` 11 / frozen 10 is the format or the seed, and whether
   `lean_rand`'s in-distribution loss is typical. The same for the token model if the 10-vs-11 comparison is to carry weight.
2. A 2 × 2 of Stage-1 + frozen controls only (no EI; ≈ $3): {Fitch order, goal-first order} × {unary `|` depth prefix, no
   depth prefix}, two seeds each, on the a1 set, scored by frozen depth-3 acquisition. This separates the two candidate
   mechanisms for the base-model jump — the conclusion stated before the box vs the unseen `| | |` prefix — which the
   current data cannot.
3. For the project question in the Lean format: find a pattern with frozen acquisition ≈ 0 at 256 attempts in `lean_seq`
   (candidates: depth-4 from a depth ≤ 2 set, the reductio pattern) before running any further f = 0 dial in it.
4. `--no_shift` with a second seed and stored literal texts, to check the shadowing explanation.

Reviewer finished 2026-09-21 UTC. No `QUARANTINE`.
