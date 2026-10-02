---
written_on: 2026-10-02
written_by: agent:claude
---

> **Reading copy (librarian, 2026-10-02).** This is the claim-audit run's ledger `audit/CLAIMS.md` (fork branch
> `dan_claim-audit`, DONE `04e9be83`) with the reviewer's five rewordings applied; each change is marked *[review]*.
> Verdict **stands**, nothing quarantined, no claim *not supported*. Review:
> [`review_claim-audit.md`](https://github.com/dan-pandori/nd-takehome/blob/ad9116a5/review_claim-audit.md).
> Summary: [experiment-summaries/2026-10-02-claim-audit-…](../../experiment-summaries/2026-10-02-claim-audit-headline-counts-reproduce-three-claims-weaker-than-stated/README.md).
> Paths such as `audit/C1.md` are relative to the fork branch.

# Claim audit: the project's headline claims, red-teamed from raw artefacts (2026-10-02)

Run `claim-audit` (fork branch `dan_claim-audit`; pre-registration `preregistration/claim-audit.md`, `eb12ce49`).
Claims are taken from nd-rl `docs/project_strategy/2026-10-01-week-in-review.md` § "What seems important" and the
run summaries behind it. Every number below was re-derived from raw run files (bucket `hf://buckets/dan-pandori/nd-rl/<run>/`
or pulled run worktrees) by new code in `audit/scripts/`, not by the runs' analysis scripts. Per-claim detail, sources
and md5s: `audit/C1.md` … `audit/C7.md`; tables in `audit/out/`. Checker everywhere: **Lean alone** (post-2026-09-27).

**Bottom line.** No headline number is wrong: every stated count re-derived exactly. No claim is unsupported.
One claim is solid as stated (the state base's 28 / 29). The others need narrower wording. The usual problems are:
- a null result ("level", "no naming effect", "no fall-off") stated as a positive fact;
- a result from one model, seed or cell stated as general;
- an uncontrolled confound (model size, the step interface) folded into the label.

## Ledger

| # | claim (as shared) | rating | strongest objection | suggested wording |
|---|---|---|---|---|
| C1 | EI solves 29 theorems its base never solved in 400k attempts; an 8× base does not reach them | **holds with caveats** | "Never" was one base seed (s0). This audit's R3 gave base s1 200,000 T 0.8 attempts on each: it reaches **3** of the 29 and 26 stay at 0. The 8× base reaches 0 and 2, and it is not a better prover (worse on 6-line held-out). EI's p̂ reproduces on fresh draws (R2: 29 / 29 ≥ 0.01). Selection, leakage, truncation and the checker all pass. | "After 8 EI rounds one model solves 29 transfer theorems (p̂ ≥ 0.02 on fresh draws) that its base seed solved 0 times in 400,000 attempts; a second base seed reaches 3 of them in 200,000 and 26 not at all; a 7.9× larger base on the same data reaches 0 and 2." |
| C2a | A proof-state base reaches 28 of the 29 with no RL | **solid** (as a *reach* statement) | "Reaches" means ≥ 1 Lean-accepted proof within ≤ 400k attempts. At EI's own bar for "solves" (p̂ ≥ 0.01) the SN base clears **21 (s0) / 14 (s1)** of 29. | Keep "reaches 28 of 29 per seed (23–26 within 10,000 attempts)". Do not set it against "EI solves 29" without saying the two use different bars. |
| C2b | It is seeing the state, not the environment's naming | **weaker than stated** | "State" is confounded with the whole step interface. The environment writes the box-closing and Or.elim-branch tokens, and a failed action ends the attempt. No arm has the steps without the state. "Not naming" is a null on reach at n = 2 per arm, and naming lowers per-theorem rates (S/SN median p̂ ratio 0.16 on s0, 0.69 on s1). | *[review]* "A 3.2 M base trained from scratch on the cap-6 set (`lean_staten`, SN, 2 seeds) and acting through the proof-state step interface reaches 28/29; a variant that writes its own names (S, `lean_state`, same size and data) reaches 28 and 25. No naming effect on reach is detected at 2 seeds, though its per-theorem rates are lower. The state and the step interface are not separated." |
| C3 | The proof state and cap 12 compound; the best model solves nearly all of the long pools | **holds with caveats** | "Compound" holds as "together far beyond either alone": 233–320 of 380 vs ≤ 102, every seed. Super-additivity is scale-dependent: +153 on counts (p 0.09), −0.63 on logit (p 0.55). Three of the four cells have n = 2 and the ladders are unmatched. *[review]* The MDD (≈ 113–119) comes from the observed spread of 4 vs 2 seeds; `NOISE_FLOOR.md` has no row for these models. If the floor's hard-pool CV (≈ 0.45) transferred, the MDD would be ≈ 250 and the +216 gap would sit inside it, so "compound" rests on complete separation in 4 vs 2 seeds, not on a measured floor. The best model was read on 2 of the 3 long pools (not the ≥ 17 file). "No fall-off past 17" has power 0.72 for a 20 pp drop, and no theorem is labelled > 18. | "With the proof state and cap-12 training together (SN-cap12 T1: 3.2 M `lean_staten`, from scratch on K12, 4 seeds) the model solves 233–320 of 380 long theorems, against ≤ 102 for either lever alone (K12 T1: 3.2 M `lean_seq`, 2 seeds; SN6 T1: 3.2 M `lean_staten` cap 6). *[review]* Whether the levers interact is unresolved. The best model saturates rr600 and long2 (it was not read on the ≥ 17 pool)." |
| C4a | Robbie's recipe beats ours after RL at both caps | **holds with caveats** | It is beyond the observed-spread MDD at both caps (+18.3 cap 6, +14.4 cap 12) and replicates in trajectory's independent training runs (+21.7, +13.1; permutation p 0.036, 0.005). But "recipe" includes **3× the parameters**. compute-match (2026-10-02) gave our cap-12 ladder best12's A40-seconds and the gap stayed, so extra *ladder* compute is ruled out at cap 12 only. Model size and architecture are not separated. "Level frozen at cap 12" is unresolved (+0.4, MDD 7.7). | "After the same ladder (equal attempts), Robbie's 9.6 M recipe solves 51–52 (cap 12) and 33–40 (cap 6) of 72, against our 3.2 M model's 36–38 and 16–22; replicated in fresh training runs; matching ladder GPU time does not close the cap-12 gap; model size is not separated from the recipe." |
| C4b | textbook72: SN-cap12 T1 solves 37 of 72 (and the 36–38 / 51–52 ranges) | **solid** | "37" is seed 0; the per-seed counts are 37 / 38 / 36 / 38 (union 46). All 289 reads across 4 runs use the identical 72 problems at k 256 and T 0.8. One problem (`textbook_3ed45280`, P ∨ Q, ¬P ⊢ Q) is a premise-order copy of a K12 training theorem; it affects both cap-12 arms equally. *[review]* The dev set (1,108 records) shares 1 class with K12 and 1 with `rl_targets`; this does not change +292 / +104. | Say "per seed". Footnote the 1/72 overlap. |
| C5a | trajectory: the eventual proof's worst step climbs in pretraining and RL (cap 12), mainly in RL (cap 6) | **holds with caveats** | The cap-6 vs cap-12 contrast far exceeds the seed spread (t 11.6, df 4) and holds on cross-seed and reference targets and on matched theorems. Within cap 12, the per-seed CIs of Δ_RL − Δ_PT all span 0. Confounds: training set, ≈ 1.4–1.5× ladder compute at cap 12, and less headroom (cap 12's B starts at −6 nats). | As run summaries, naming the models (best-cap12 / best-cap6: 9.56 M `ALiBiGPT`, `lean_staten`, from scratch, 3 seeds) *[review]*, and adding "cap 12 has less room to climb". |
| C5b | (week-in-review) RL-only theorems were within one improbable step (≈ 1-in-400) after pretraining; RL concentrated on the model's own proofs; unsolved ones have several hard steps | **weaker than stated** | 1-in-400 is a median of a very wide spread: 1/245–1/724 by seed, IQR within a seed ≈ 1/180–1/17,000, and ≈ 1/22,000 at cap 6. The rest of the proof costs 7–9 nats, more than the worst step, and 40 % have a second step below 1/55, so it is not "one step". "Own proofs" is partly argmax selection: other seeds' RL-found proofs also gain ≈ 1 nat more than references, but ≈ 2–2.4 nats (depending on aggregation) of the own-proof premium cannot be separated from selection. *[review]* On other seeds' proofs, RL's lift (≈ 2.6 nats, paired median) is about 1.7× the reference's (1.55), not 3×, and smaller than the pretraining lift (3.19) on the same proofs. "Several hard steps" is a post-hoc restatement of the pre-registered expectation's miss. | "After pretraining (best-cap12, 9.56 M `lean_staten`, from scratch on K12, 3 seeds *[review]*), the median RL-only theorem's hardest step had probability about 1/250–1/700 and the rest of the proof cost about as much again. RL raised RL-found proofs (from any seed) more than the shortest known proofs." |
| C6 | rl-from-ckpt: from step 5k, RL ends level with the end arm; no early start reaches beyond it | **holds with caveats** | "Level" is a non-rejection at n = 3. Pooled over both pools (322 theorems) every early start ends **≈ 4–6 below** (≈ 5–6 on the x1 read, per-seed means *[review]*) the end arm in 17 of 18 seed × draw comparisons (theorem-bootstrap CIs exclude 0), with ≈ 1/3 fewer total training tokens. "No reach beyond" can only see gains of ≈ 10+ theorems per seed. | "For best-cap12 (9.56 M `lean_staten`, seeds 0–2) *[review]*, from step 5,000 on, RL ends within ≈ 2 % of the end arm (≈ 4–6 of 322 lower in nearly every seed; ≈ 5–6 on the x1 read *[review]*) with ≈ 1/3 less total training; no early start solves anything the end arm demonstrably cannot (detects only ≳ 10-theorem gains)." |
| C7 | GPU nondeterminism is about half the seed-to-seed spread ("much of our noise") | **weaker than stated** | The "half" is a variance ratio (0.54, bootstrap 95 % [0.02, 1.08]); in SD terms it is 0.74. It comes from 9 runs of one high-mode cell on one pod and is driven by 1–2 mode flips (dropping the lowest run gives 0.18). On val loss the replicates spread *more* than the grid (1.56×). It is one 3.2 M Stage-1 recipe, generalised to "much of our noise". | "On one 3.2 M Stage-1 recipe, 9 identical-seed runs of one cell put 1 run in the low depth-3 mode and 1 at the boundary: same-pod GPU nondeterminism alone can flip modes, and could account for anywhere from a small part to all of the seed variance." |

The week-in-review's point 1 ("'RL created it' depends on the interface") survives the audit. It should be read
with C2a's bar difference and C2b's state ≡ step-interface confound. Point 2's "the best model solves nearly all of
the long pools" covers 2 of 3 pools (C3).

## Cross-run consistency (same checkpoint × same pool, scored twice)

| checkpoint / pool | runs | result |
|---|---|---|
| WP base s1, 383 transfer | support-curves s3 vs support-followups A | 37 vs 37 solved, 377 / 383 agree; \|z\| > 3 on 1 / 40 (batch / `max_new` differ) |
| WP base s0, 6 longest survivors | support-curves stage 2 vs support-followups B | 0 vs 1 in 1.67 M; consistent |
| SN base s0, 29 survivors | support-state H vs S1 (different seeds) | \|z\| > 3 on 0, > 2 on 1 |
| SN / S bases, 29 survivors | state-env 256-attempt reads vs support-state / state-readouts p̂ | 21 vs 22.7 ± 0.9, 18 vs 17.3 ± 1.0, 16 vs 16.5 ± 1.3, 15 vs 16.2 ± 0.7 |
| SN-cap12 / SN-v2 / K12, long pools | state-cap12, long-pool, long-pool-2 (38 pairs) | Q differences ≤ 7 (\|z\| ≤ 1.31); ≥ 17 file: 1 / 24 at z 2.6 (s3, 46 vs 41) |
| 66 checkpoints, textbook72 | trajectory x0 vs x1 draws (133 pairs) | ≤ 4 problems apart; 7 / 133 at \|z\| > 1.96 (6.6 expected); sd(z) 1.11 |
| best12 / best6 recipes, textbook72 | best-state vs trajectory (fresh trainings) | +1.3 and −3.3; inside the spread |
| SN base s0, 29 survivors (**this audit**, fresh seed) | support-state H vs claim-audit R1 | 26 vs 26 reached within 10k; 21 vs 21 at p̂ ≥ 0.01; \|z\| > 3 on 1 / 29 |
| EI s0, 29 survivors (**this audit**, fresh seed) | support-curves stage 1 vs claim-audit R2 | 29 vs 29 at p̂ ≥ 0.01; \|z\| > 2 on 1 / 29 |
| WP base s1, 29 survivors | support-curves s3 / support-followups A vs claim-audit R3 | 1 and 2 reached in 10k each vs 3 in 200k; consistent |

**No disagreement beyond what multiplicity predicts.** One caution: several "agreements" are **deterministic replays**,
not independent checks: the same checkpoint and the same sampling seed give bit-identical reads. These are
rl-from-ckpt vs trajectory (322 / 322 identical), long-pool pass 1 vs pass 2, and textbook72 a512 vs a1024. They show
reproducibility of the pipeline, not the size of the sampling spread. Do not cite them as replications.

## Definitions checked

- **Checker.** Lean alone in every audited number. One exception: C0's ladder (whole-proof cap-6 cell of C3) *trained*
  under Lean ∧ `nd_verify`. It is read under Lean alone.
- **"Solved".** n_ok > 0 of 256 at T 0.8 for textbook72, the long pools, trajectory and rl-from-ckpt; consistent.
  C1 / C2 use different bars (C1 "EI solves" p̂ ≥ 0.01; C2 "reaches" ≥ 1 of ≤ 400k). See C2a.
- **Pools.** textbook72 is identical in all 289 reads (hash `8789589d21`); rr600 `fc85baaf`; transfer 383 identical
  across the support runs.
- **`L_true`** is an ND-derived upper bound. Re-binning C3 by Lean term size keeps the arm ordering in every bin.

## Leakage (own canonicaliser, invariant to atom renaming and premise order; `F` = falsum)

- 0 of the 29 survivors (0 of 383 transfer) share a class with the Stage-1 set, `rl_targets`, or any of the 272,632
  EI-s0 training records. The positive control matched 383 / 383.
- 0 shared classes between K12, the cap-6 set or `rl_targets` and rr600, the ≥ 17 file, long2-new21 or calib70.
- **1** of the 322 trajectory / textbook evaluation theorems (`textbook_3ed45280`) is a premise-order copy of a K12
  training theorem. `gen.canon_key` is premise-order sensitive and would miss it.

## Lean spot-checks (own drivers; any error or `sorry` rejects)

| family | accepted proofs re-checked | negative controls rejected |
|---|---|---|
| C1 / C2 (survivors: EI, base, big, SN, S) | 82 / 82 | 82 / 82 |
| C3 / C4 (long pools, textbook72; 48 are `nd2lean` renderings of stored ND) | 240 / 240 | 960 / 960 (4 kinds × 240) |
| C5–C7 (trajectory, rl-from-ckpt, M2) | 122 / 122 | 366 / 366 |
| this audit's re-samples (R1–R3) | 27 / 27 | 27 / 27 |
| **total** | **471 / 471** | **1,435 / 1,435** |

The harness is valid: every control kind was rejected. *[review]* The reviewer's own drivers (own statement builder, Lean 4.34.1, axiom allowlist) re-checked **2,782 / 2,782** proofs, including all 216 re-sampled ones, and rejected **981 / 981** controls of four kinds. The controls were wrong theorem, negated goal, final step
dropped, `exact sorry`, swapped names and truncated text.

## Re-sampling (pod `ca1`, RTX A6000, $0.53/h billed; 2.0 h, $1.06; fresh sampling seeds, no early stopping)

Script `audit/scripts/ca_resample.py`; raw rows `audit/raw/ca/` (bucket `claim-audit/audit/raw/ca/`); job scripts
`audit/pod/`. Code = `git archive origin/dan_support-state` (`support.py`, `ss_support.py`, unchanged).

| job | model | setting | result | on file |
|---|---|---|---|---|
| R2 | support-curves EI s0 `la_T1_sc_s0_r8.pt` (5cebd7ec; 3.2 M, `lean_seq`, base s0 + 8 EI rounds) | 29 survivors, T 0.8, k 2,000, seed 9002 | **29 / 29 at p̂ ≥ 0.01**, min 0.0245, median 0.57; \|z\| > 2 on 1 / 29 | 29 / 29, min 0.022 |
| R1 | SN base s0 `stage1_SN_s0.pt` (ec3888d9; 3.2 M, `lean_staten`, no RL) | 29 survivors, T 0.8, k 10,000, seed 9001 | **26 / 29 reached**, 21 at p̂ ≥ 0.01; \|z\| > 3 on 1 / 29 (la_transfer_149, 0.55 vs 0.52) | 26 within 10,000; 21 |
| R3 | WP base **s1** `stage1_a1_seq_s1.pt` (fc27e52d; 3.2 M, `lean_seq`, no RL) | 29 survivors, T 0.8, k 200,000, `stop_at` 1, seed 9003 | **3 / 29 reached** (1645 at 654, 1759 at 7,058, 543 at 27,493); 26 at 0 / 200,000 | (2 / 29 in 20,000) |

Lean (own driver `c12_lean.py`): 27 / 27 re-sampled accepted proofs accepted (all 3 R3 hits, 12 R2, 12 R1), 27 / 27
wrong-theorem controls rejected (`audit/out/ca_resample_lean.tsv`).
Reading: C1's EI side and C2a reproduce on fresh draws. Pre-registered expectation for R3 was 3–8 reached, with ≤ 4 meaning
"never" is close to seed-independent. Outcome 3: **26 of the 29 are unreached by both base seeds** at ≥ 200,000 T 0.8 attempts
each. C1 stays *holds with caveats*; the caveat narrows from "one seed" to "3 of the 29 fall to the other seed".

## Multiple comparisons, forking paths, seeds

- Pre-registered and passing: C1's falsifier, C3's "exceeds both by > MDD", C4's T1 comparisons (best-state's "any 1 of
  8" rule is lenient, but both T1 headlines pass even at the larger observed-sd MDDs), and C5's w1 ≤ −5 at r0.
- Post hoc: C5's "several hard steps" and "own proofs" readings, C6's reach comparison, and C7's "half" ratio. C7's
  pre-registered quantity was replicate sd ≥ 0.10.
- Seed counts: C2b and C3's cells are n = 2 per arm. `NOISE_FLOOR.md` notes that 2 vs 2 can never reach p < 0.05.
  `NOISE_FLOOR.md` has no row for the state models, textbook72 or long pools, so the MDDs here come from observed seed
  spread.

## Pre-registered expectations vs outcome (`preregistration/claim-audit.md`)

- E1 (≥ 6 of 7 re-derive exactly): **hit**. All 7 re-derived exactly. One sub-auditor sentence was wrong and is corrected
  in `C1.md`: base s1's two hits are not both in both draws.
- E2 (≥ 1 cross-run pair outside the sampling spread): **missed**. The extremes (1 / 24 at z 2.6, 1 / 40 and 1 / 29 at
  \|z\| > 3) are what multiplicity predicts.
- E3 (0 survivors leak): **hit**. There is 1 textbook72 premise-order overlap, outside the survivors.
- E4 (0 Lean rejections; all controls rejected): **hit**.
- E5 (ratings): C1, C3, C6 and C7 as predicted. C2 split: the 28 / 29 itself is *solid* (I predicted caveats), and the
  naming half is weaker. C4 came out *holds with caveats*, not the predicted *weaker*: it replicates in independent runs
  and compute-match removed the ladder-compute objection. C5's run findings hold with caveats; the week-in-review
  sentence is weaker, as predicted. Addendum R3 (3–8 expected): 3.
