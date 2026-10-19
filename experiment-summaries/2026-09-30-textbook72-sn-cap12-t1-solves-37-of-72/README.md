---
written_on: 2026-09-30
written_by: agent:claude
---

# textbook72: SN-cap12 T1 solves 36–38 of the group's 72 textbook problems at pass@256, +9 over its frozen Stage-1

Status: **completed 2026-09-30 14:48 – 15:17 UTC (executor), independently reviewed 15:20–15:47 UTC. Verdict: the
counts stand, with three rewordings; one comparison (SN-v2 cap-6 vs SN-cap12 as an effect of Stage-1 cap) is
descriptive only.** No hard-constraint violation, nothing quarantined. No training: sampling read-outs of existing
checkpoints only. Where the reviewer's wording differs from the executor's, this summary uses the reviewer's.

Question: our textbook numbers were on 760 generated schema instances. Robbie's "combined model" post (2026-09-29)
scores the group's **72 textbook problems** (dev58 + train14, from Dmitry's cap-comparison eval data) at pass@256: naive
pipeline 13 / 72, combined model 31 / 72. How many do our proof-state models solve on the same 72 at the same k and T?

The run was on the fork (`dan-pandori/nd-takehome`), branch `dan_textbook72` (from the fork's `dan` at `d7411772`).
**Nothing is merged into the fork's `dan`.** The new code is analysis only: `tb72_analysis.py`, `tb72_contam.py`,
`tb72_compute.py` and `pod/tb72/*.sh`.

- Review: [`review_textbook72.md`](https://github.com/dan-pandori/nd-takehome/blob/9ffa9d31/review_textbook72.md)
  ([`5239bf0c`](https://github.com/dan-pandori/nd-takehome/commit/5239bf0c) §Recount, written blind from a workspace
  without the executor's write-ups; [`9ffa9d31`](https://github.com/dan-pandori/nd-takehome/commit/9ffa9d31)
  §Compare / §Verdict). Reviewer scripts and outputs: `review_tb72/` (none import the executor's `tb72_*.py`).
- Executor write-up: [`run_textbook72.md`](https://github.com/dan-pandori/nd-takehome/blob/9ffa9d31/run_textbook72.md),
  `numbers.md` § textbook72, `log.md` § textbook72, `artifacts/textbook72/{summary.json,analysis_stdout.txt,long_proofs.md}`.
  Results and DONE commit `ba2a3f1c`.
- Pre-registration: `preregistration/textbook72.md`, committed `51c97089` at 14:51:47 UTC, before the pod (14:53:47) and
  the first read-out (14:55:03); not edited afterwards. (Its header's "written ~15:10 UTC" is a typo; the commit time
  counts.)
- Brief: [`BRIEF_textbook72.md`](../../docs/proposals/state-env/BRIEF_textbook72.md).

## Configuration

- **Checker: Lean alone** (`state_eval` → `lean_gate`, `"judge": "lean-only"` in every gate record). `nd_verify` is
  unmodified (tree equal to `origin/main`'s). It is used only for the pre-registered, labelled secondary count
  "Lean ∧ `nd_verify`" (Robbie's reward), which judges nothing.
- **Problems:** `data/eval_only/textbook72/` on the fork, copied from nd-rl `origin/robbie-experiments:
  code/experiments/current/factorial_20260929/textbook72/`; both sha256 match `MANIFEST.json`. Eval only (manifest rule:
  never trained or tuned on). dev58 by `reference_lines`: 1–5: 5, 6–10: 24, 11–15: 16, 16+: 13 (max 33); train14 has no
  `reference_lines` and is its own bin. ND reference lengths are upper bounds under Lean.
- **Models** (all 12 checkpoints: 3,216,384 params, `lean_staten`, from scratch; md5s checked by the reviewer against
  the registry):

  | arm | checkpoints | Stage-1 data | RL |
  |---|---|---|---|
  | SN-cap12 T1 | `state-cap12/ckpts/sc12/ladder/la_T1_SN12_s{0..3}_r8.pt` | K12 `kh/train_k12.jsonl` (cap 12, 155 k) | 8 ladder EI rounds |
  | SN-cap12 frozen | `state-cap12/ckpts/sc12/stage1_SN12_s{0..3}.pt` | same | none |
  | SN-v2 cap-6 T1 | `state-env/ckpts/se/ladder/la_T1_SN_s{0,1}_r8.pt` | `p2/train_depth3_f0_a1.jsonl` (cap 6, 155 k) | 8 ladder EI rounds |
  | SN-v2 cap-6 frozen | `state-env/ckpts/se/stage1_SN_s{0,1}.pt` | same | none |

  "SN-v2" = SN checkpoints sampled with `Env(canon=True, assign=True)`.
- **Read-out** (identical for all 12): `state_eval.py --k 256 --temperature 0.8 --seed 0 --max_steps 96 --max_action 512
  --batch 4096 --lenfield reference_lines`, one NVIDIA A40. Solved = ≥ 1 of 256 attempts Lean-accepted. Peak
  `torch.cuda.max_memory_allocated` 20.4–28.2 GB at batch 4,096.
- **Pod:** `tb72-1`, one A40 at $0.49/h, 14:53:47 – ≈ 15:15 UTC: **0.36 pod-hours, $0.18** of $3 / 6 h
  (`podbudget textbook72`, `~/podhours.log`).
- **Compute** (registry rows written by `state_eval.py`; `artifacts/textbook72/compute.tsv`; reproduced by the
  reviewer). No training (0 train steps / tokens).

  | arm (per checkpoint) | GPU-s (A40) | attempts | actions | gen tokens | Lean checks |
  |---|---|---|---|---|---|
  | SN-cap12 T1 | 101.0–108.6 | 18,432 | 229–245 k | 3.07–3.26 M | 2,118–2,898 |
  | SN-cap12 frozen | 73.2–87.0 | 18,432 | 170–176 k | 2.32–2.40 M | 1,024–1,273 |
  | SN-v2 cap-6 T1 | 50.8–57.9 | 18,432 | 136–144 k | 1.84–1.98 M | 627–1,051 |
  | SN-v2 cap-6 frozen | 43.4–48.8 | 18,432 | 108–119 k | 1.54–1.65 M | 325–368 |

  12 read-outs: 936.9 GPU-s, 29.1 M generated tokens, 16,654 Lean checks; with the `max_action` 1,024 diagnostic,
  1,053.8 GPU-s (≈ 0.29 GPU-h). Matched on attempts. The T1 arms use ≈ 1.3× the frozen arms' GPU-s and tokens (and
  ≈ 2× the Lean checks) at equal attempts, because RL'd policies take more steps per attempt; this is an evaluation at
  matched attempts, so the > 1.25× flag is informational. The registry `seed` of T1 rows is 8 / 1008 / 2008 / 3008
  (`record.model_seed` of a ladder checkpoint), not the Stage-1 seed; join on `ckpt`, not `seed`.
- **Bucket:** `hf://buckets/dan-pandori/nd-rl/textbook72/artifacts/textbook72/` (77 files, listed by the reviewer).

## Observations

The reviewer re-derived every count below with their own scripts; all 12 per-checkpoint rows reproduce exactly.

**Solved per checkpoint (Lean alone, pass@256).**

| checkpoint | all 72 | dev58 | train14 | 1–5 /5 | 6–10 /24 | 11–15 /16 | 16+ /13 | distinct accepted proofs | action-cap % |
|---|---|---|---|---|---|---|---|---|---|
| T1_SN12_s0 | **37** | 30 | 7 | 5 | 18 | 5 | 2 | 878 | **0.157** |
| T1_SN12_s1 | **38** | 32 | 6 | 5 | 18 | 7 | 2 | 1,033 | 0.000 |
| T1_SN12_s2 | **36** | 30 | 6 | 5 | 18 | 4 | 3 | 1,387 | 0.011 |
| T1_SN12_s3 | **38** | 32 | 6 | 5 | 19 | 6 | 2 | 941 | 0.005 |
| Fz_SN12_s0 | **26** | 20 | 6 | 5 | 12 | 2 | 1 | 371 | 0.087 |
| Fz_SN12_s1 | **29** | 23 | 6 | 5 | 14 | 4 | 0 | 487 | 0.092 |
| Fz_SN12_s2 | **32** | 26 | 6 | 5 | 17 | 4 | 0 | 367 | 0.049 |
| Fz_SN12_s3 | **26** | 21 | 5 | 5 | 13 | 3 | 0 | 392 | 0.000 |
| T1_SN6_s0 | **22** | 18 | 4 | 5 | 12 | 1 | 0 | 152 | 0.005 |
| T1_SN6_s1 | **16** | 12 | 4 | 5 | 7 | 0 | 0 | 52 | 0.005 |
| Fz_SN6_s0 | **16** | 13 | 3 | 5 | 8 | 0 | 0 | 37 | 0.005 |
| Fz_SN6_s1 | **14** | 12 | 2 | 5 | 7 | 0 | 0 | 31 | 0.043 |

**By arm.**

| arm | per seed (72) | IQM [95 % CI] | union 72 (dev58) | union by bin 1–5 / 6–10 / 11–15 / 16+ / train14 |
|---|---|---|---|---|
| SN-cap12 T1 | 37 / 38 / 36 / 38 | 37.5 [36, 38] | 46 (38) | 5 / 21 / 9 / 3 / 8 |
| SN-cap12 frozen | 26 / 29 / 32 / 26 | 27.5 [26, 32] | 36 (30) | 5 / 17 / 7 / 1 / 6 |
| SN-v2 cap-6 T1 (n = 2) | 22 / 16 | — | 22 (18) | 5 / 12 / 1 / 0 / 4 |
| SN-v2 cap-6 frozen (n = 2) | 16 / 14 | — | 18 (15) | 5 / 10 / 0 / 0 / 3 |

- **RL gain, SN-cap12 T1 − frozen, paired by seed: +11, +9, +4, +12** (mean **+9.0**, sd 3.56, paired t 5.06, df 3,
  p ≈ 0.015). The pre-registered MDD was ≈ 8; at the observed sd the n = 4 paired MDD is ≈ 7.4. +9 clears it. It is
  positive on every seed but not ≥ +6 on every seed (s2 +4), as pre-registered and reported.
- Only one problem some SN-cap12 frozen seed solves is solved by no T1 seed (`textbook_8244c4b1…`); 11 are solved by
  some T1 seed and by no frozen seed.
- SN-v2 cap-6 paired T1 − frozen: +6, +2 (n = 2, descriptive). **Any ranking of SN-v2 cap-6 against SN-cap12 as an
  effect of Stage-1 cap is not supported**: n = 2, and the two differ in Stage-1 data *and* in their ladder runs.
- **By length (reviewer's wording):** the solve rate falls steeply with reference length — all of 1–5, ~¾ of 6–10
  (T1 18–19 / 24 per seed), ~⅓ of 11–15 (4–7 / 16), 2–3 of 13 at 16+. This replaces the executor's "the length wall at
  `L*` ≈ 12 again": solved proofs reach 16–19 lines, a frozen checkpoint (Fz_SN12_s0) solves the reference-18
  problem, and `L*` belongs to the 760-pool read-out, not to this set.
- **All 12 checkpoints together solve 47 / 72. 25 are solved by none: 19 dev58 + 6 train14** (reviewer; the executor's
  `numbers.md` says "20 dev58, 5 train14", which mis-states the split; its own `analysis_stdout.txt` has 19 + 6). By
  bin: 10 of 13 at 16+, 6 of 16 at 11–15, 3 of 24 at 6–10 (`0824150e`, `5758428c`, `a104fab3`), and 6 of train14.
- **Secondary, labelled (Lean ∧ `nd_verify`, judges nothing):** 0 of the 6,128 distinct Lean-accepted proofs are
  rejected by `nd_verify`, so every count above is unchanged under Robbie's checker.

**Robbie's combined model (descriptive only: different model, format and checker).** Read by the reviewer directly
from nd-rl `origin/robbie-experiments:…/2026-09-28-combined-model/charts/passk.csv` (n = 256 per problem, same problem
names).
- Combined model `fact-lean-best-leon_s{0,1,2}`: **32 / 30 / 32** (mean 31.3), union 36. Robbie's checker is Lean ∧
  `nd_verify`. His model is described as "pretrained 6×384 `lean_seq` + Leon EI, T 0.8"; that is from his summary and
  not derivable from the csv.
- "Naive 13": which of the file's four naive arms this is cannot be read from it. `fact-abs-naive-ei_s{0,1,2}` gives
  7 / 11 / 9 (mean 9 = 12.5 %), union 13; no naive arm has a per-seed mean of 13 problems.
- Per problem: our checkpoints solve **34 of his 36**; he alone solves 2 (`0824150e` ref 6, `3d573ac4` ref 11), which
  no checkpoint of ours solves; we alone solve 13 (12 from the SN-cap12 T1 union), including all three solved problems
  with reference ≥ 16. His union contains none of the 16+ problems.

**Contamination** (reviewer's own keys, reproducing the executor's). With premise order kept: **0** overlaps with the
cap-6 control set, K12, ladder and old `rl_targets`, and all 48 EI replay mixes. Ignoring premise order: **1** —
`textbook_3ed45280…` (`(P ∨ Q), ¬P ⊢ Q`, disjunctive syllogism, reference 7), a premise-order permutation of a K12
Stage-1 theorem, also in 4 SN-cap12 replay mixes. All 12 checkpoints solve it. So every SN-cap12 count includes at most
one contaminated problem; without it, each drops by 1.

**Proof length.** The longest-reference solved problem is `textbook_b98931f2…`, `¬(P → Q) ⊢ P ∧ ¬Q` (reference 18
lines), solved by 5 checkpoints including frozen Fz_SN12_s0. Its shortest accepted proof is 16 lines (Lean as
rendered and accepted):

```lean
theorem tb_b98931 (P Q R S : Prop) (h1 : ¬ (P → Q)) : P ∧ ¬ Q := by
  have n1 : ¬ (P → Q) := h1
  have n8 : ¬ ¬ P := fun (n2 : ¬ P) => by
    have n6 : P → Q := fun (n3 : P) => by have n4 : False := n2 n3; have n5 : Q := n4.elim; exact n5
    have n7 : False := n1 n6; exact n7
  have n9 : P := Classical.byContradiction (fun hh => n8 hh)
  have n15 : ¬ Q := fun (n10 : Q) => by
    have n13 : P → Q := fun (n11 : P) => by have n12 : Q := n10; exact n12
    have n14 : False := n1 n13; exact n14
  exact ⟨n9, n15⟩
```

- The other two solved 16+ problems: `418e4b67` (ref 16, 16 lines) and `48e30865` (ref 16, 14 lines).
- Term size: `lean_check` elaborated-term size (executor) 12 / 11 / 11 for these three; the reviewer's own measure
  (ND inference nodes reachable from the conclusion) 14 / 15 / 12. The two are different definitions and are not
  comparable. Line counts reproduce.
- Of the 39 solved dev problems, the shortest accepted proof is shorter than `reference_lines` on 7, equal on 28,
  longer on 4 (reviewer). Accepted proofs often pad with `X ∨ X` / `X ∧ X` detours, which inflate lines, not counts.

**Checks.**
- **Lean re-check, all 6,128 distinct accepted proofs** (not a sample), Lean 4.34.1 core with an axiom check, under
  two independent renderings (reviewer's own ND → Lean translator and the `lean_tok` `lean_seq` rendering):
  **6,128 / 6,128 accepted** under each. Controls: proof under a different prompt 0 / 300; final `exact` → `sorry`
  0 / 100; ORI1↔ORI2 / ANDE1↔ANDE2 swap 14 / 300, all on symmetric formulas where the swap stays valid.
- The accepted `lean_seq` texts are not stored (only the ND rendering and the rejected texts), so "re-checked from the
  literal text" means from the rendering. Both renderings pass.
- **Caps:** step cap (96) never hit (longest attempt 76 steps). Action cap (512) above 0.1 % only on T1_SN12_s0
  (29 attempts, 0.157 %). A diagnostic re-run at `max_action` 1,024 (not pre-registered, same seed) cut truncations to
  6 (0.033 %); the 23 freed attempts all ended in syntax errors, and the solved set (37), distinct proofs (878) and
  accepted samples (3,317) are identical. The cap cost nothing.

**Expected vs outcome** (pre-registration `51c97089`; the reviewer adds the two qualitative misses the executor's table
omitted).

| pre-registered | outcome |
|---|---|
| SN-cap12 T1 36 / 72 per seed (28–44), union 42 | 36–38, mean 37.25: **in range**; union 46 (above) |
| SN-cap12 frozen 24 (16–32), union 30 | 26–32, mean 28.25: **in range**; union 36 (above) |
| SN-v2 cap-6 T1 32 (24–40) | 22, 16: **miss (below)** |
| SN-v2 cap-6 frozen 18 (10–26) | 16, 14: in range |
| T1: "near-everything ≤ 10 reference lines and train14" | ≤ 10: 23–24 / 29 per seed, yes; **train14 6–7 / 14 per seed (union 8): miss** |
| T1: about a third of 11–15 | 4–7 / 16 (25–44 %): roughly yes |
| T1: ≤ 2 of 16+ | 2, 2, **3**, 2: **s2 misses by one** |
| (a) T1 − frozen ≥ +6 on each SN-cap12 seed | +11, +9, **+4**, +12: **miss on s2** (the mean +9 holds) |
| (b) no checkpoint > 3 of the 13 at 16+ | max 3: holds |
| (c) SN-cap12 T1 median seed ≥ 31 | 37.5: holds |
| (d) step- and action-cap each ≤ 0.1 % | step 0; action **0.157 % on T1_SN12_s0: miss** (no effect on solved) |
| (e) contamination 0–5 overlaps, all short | 1 (reference 7): holds |

## Interpretation

On the group's own 72 problems, at Robbie's k and T, our 3.2 M from-scratch proof-state RL model (SN-cap12 T1) solves
36–38 per seed, against 30–32 for Robbie's pretrained combined model under his checker. The comparison crosses model,
format and checker, and is descriptive; under his checker our counts do not change. RL adds a paired +9 problems over
the same checkpoints' Stage-1 models, above the MDD. This run does not say whether that +9 is new capability or
elicitation of rare behaviour: the frozen models were read at 256 attempts only. Past 15 reference lines neither our
models nor Robbie's reach much: 10 of 13 such problems are solved by no checkpoint.

## Limitations and next step

- The comparison with Robbie is across different models, formats and checkers, and his "naive 13" cannot be matched
  to one arm of his file.
- SN-v2 cap-6 has n = 2 and differs from SN-cap12 in both Stage-1 data and ladder; it ranks nothing.
- The frozen arms were read at pass@256 only, so the RL gain is not split into elicitation vs acquisition.
- One problem (ref 7) is a premise-order permutation of a K12 training theorem; counts include it.
- Accepted `lean_seq` texts are not stored; the registry `seed` of T1 rows is not the Stage-1 seed.
- The analysis scripts are only on the fork branch `dan_textbook72`.

Next measurement (reviewer): for the 11 problems some T1 seed solves and no frozen seed solves (and the 25 no
checkpoint solves), read frozen SN-cap12 at **pass@4,096** (same T, seed-paired; ≈ 16× the attempts on 36 problems,
< 1 A40-hour). If the frozen models reach most of the 11, the RL gain here is elicitation of rare behaviour; if not,
it is acquisition on textbook problems. A seed-matched n ≥ 4 for SN-v2 cap-6 would settle the Stage-1-cap comparison,
if that matters.
