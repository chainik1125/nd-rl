---
written_on: 2026-10-02
written_by: agent:claude
---

# claim-audit: every headline count reproduces; three claims are weaker than stated, none unsupported

Status: completed and reviewed 2026-10-02. No hard-constraint violation and nothing quarantined. Verdict **stands**:
every count in the ledger reproduces on the reviewer's independent code, statement builder and Lean driver. The reviewer
agrees with all ten ratings and asks for five rewordings, which are applied in the nd-rl reading copy
[`project_strategy/2026-10-03-claim-audit.md`](../../docs/project_strategy/2026-10-03-claim-audit.md) (changes marked
*[review]*). The deliverable is an **audit for Dan before claims travel**, not a test of a new hypothesis. Its new
measurements are three re-sampling jobs, R1–R3.

Question: do the project's headline claims (nd-rl `docs/project_strategy/2026-10-01-week-in-review.md` § "What seems
important", and the `docs/STATE.md` headlines since 2026-09-27) survive re-derivation from raw artefacts with new code,
cross-run consistency checks, a definitions check, attacks on selection, leakage, multiplicity and seed counts, and
Lean spot checks? Brief: [`proposals/claude-heavy/BRIEF_claim-audit.md`](../../docs/proposals/claude-heavy/BRIEF_claim-audit.md)
(proposal 21).

## Configuration

- Fork `dan-pandori/nd-takehome`, branch `dan_claim-audit`. Pre-registration `preregistration/claim-audit.md`
  (`eb12ce49`, 18:41:27Z). Addendum 1, R3 with its expected range and reading rule, is `044609de` (19:20:03Z), committed
  before R3 launched. Per-claim audits `d4d61fa4`, DONE `04e9be83`. Review
  [`review_claim-audit.md`](https://github.com/dan-pandori/nd-takehome/blob/ad9116a5/review_claim-audit.md):
  [`e1361f41`](https://github.com/dan-pandori/nd-takehome/commit/e1361f41) §Recount (blind),
  [`ad9116a5`](https://github.com/dan-pandori/nd-takehome/commit/ad9116a5) §Compare / §Verdict. Reviewer scripts:
  `review_ca/` on the fork.
- Executor Fri 2026-10-02 18:40–21:07 UTC; reviewer 21:08–21:37.
- Executor code `audit/scripts/` (new code, not the audited runs' analysis scripts); outputs `audit/out/`; per-claim
  detail `audit/C1.md`–`C7.md`, each number with its source file, md5 and model.
- **Checker: Lean alone** throughout. Pod code imports `nd_verify.verify.parse_formula` only for parsing (via
  `state_env.py`); judging is `lean_judge` / `lean_gate`. C3's whole-proof cap-6 ladder was *trained* under
  Lean ∧ `nd_verify` (pre-2026-09-27) and is read under Lean alone; the ledger labels it.
- **Re-sampling** on pod `ca1`, one RTX A6000 at $0.53/h, 19:05–21:05 UTC. Code `git archive origin/dan_support-state`
  (`support.py`, `ss_support.py`, unchanged). Fresh sampling seeds 9001–9003, T 0.8, on the 29 support-curves survivors:
  - R1: SN base s0 `ckpts/se/stage1_SN_s0.pt` (`ec3888d9`; 3,216,384 params, `lean_staten`, from scratch on
    `train_depth3_f0_a1`, cap 6, no RL), k 10,000, batch 4,096, `max_steps` 48.
  - R2: EI s0 `la_T1_sc_s0_r8.pt` (`5cebd7ec`; 3,214,336 params, `lean_seq`, WP base s0 + 8 EI rounds on `rl_targets`),
    k 2,000, batch 2,048, `max_new` 400.
  - R3: WP base **s1** `stage1_a1_seq_s1.pt` (`fc27e52d`; 3,214,336 params, `lean_seq`, from scratch on
    `train_depth3_f0_a1`, cap 6, no RL), k 200,000, `stop_at` 1, batch 4,096, `max_new` 400 (5,240,960 samples).
- **Compute** (registry rows `artifacts/claim-audit/registry/`): one RTX A6000, 2.0 pod-hours, $1.06; R1 290,000
  environment attempts, R2 58,000 samples, R3 5,240,960 samples; 0 training steps. The registry has no `gen_tokens` or
  `lean_checks`, and its per-job `gpu_seconds` (R1 3,810 s, R2 114 s, R3 11,905 s) are wall time on a shared card: they
  sum to 15,829 s against a 7,198 s card total. Do not sum the rows.
- Bucket: `hf://buckets/dan-pandori/nd-rl/claim-audit/audit/` (ledger, C1–C7, scripts, outputs, `raw/ca/` re-sample
  rows, logs, Lean dump) and `hf://buckets/dan-pandori/nd-rl/claim-audit/artifacts/claim-audit/registry/`. Other runs'
  raw files are referenced by their own bucket paths and md5s in `audit/out/`.
- Cost: **$1.06** of an $8 / 16 h budget; pre-registration estimate ≈ 2 h of the A6000 for R3.

## Observations

Observed, with the reviewer's wording where it differs from the executor's.

**Re-derivation.** All 7 headline counts re-derive exactly on new code (E1 hit). The one exception is C4's cap-12 dev
metric: +104.3 by the reviewer's recount, +103.8 stated (+104 in the executor's text).

**Ratings** (ten claims; reviewer agrees with all):

| claim | rating | key evidence |
|---|---|---|
| C1 EI solves 29 theorems its base never solved in 400k attempts | holds with caveats | the 29 are a selection (EI-high-p̂ theorems on which base s0 got the full 400k). R3: base **s1** reaches **3 / 29** in 200,000 T 0.8 attempts (first hits 654 / 7,058 / 27,493); **26 stay unreached by both base seeds**. Pre-registered reading (3–8 expected; ≤ 4 → near seed-independent) applied as written |
| C2a proof-state base reaches 28 / 29 | solid (as *reach*) | SN s0 / s1 28 / 28 (T0.8 ∪ T1.0); at EI's "solves" bar p̂ ≥ 0.01 only **21 / 14** |
| C2b "the state, not the naming" | weaker than stated | state is confounded with the step interface; naming null is n = 2 per arm |
| C3 proof state × cap 12 compound | holds with caveats | SN-cap12 T1 233 / 295 / 320 / 317 of 380 vs ≤ 102 for either lever alone; interaction unresolved; MDD from 4 vs 2 seeds' spread, no `NOISE_FLOOR.md` row (with its hard-pool CV the MDD would be ≈ 250 and +216 would sit inside) |
| C4a Robbie's recipe beats ours after RL | holds with caveats | +18.33 cap 6, +14.17 cap 12 (IQM); replicates in fresh trainings (permutation p 1/28, 1/210); 3× parameters; best used ≈ 1.6–1.9× the GPU-seconds |
| C4b textbook72 counts | solid | per seed 37 / 38 / 36 / 38; one textbook72 problem is a premise-order copy of a K12 theorem; dev set shares 1 class with K12 and 1 with `rl_targets` |
| C5a trajectory's run findings | holds with caveats | cap-6 vs cap-12 contrast far beyond seed spread; within cap 12 per-seed CIs span 0 |
| C5b "within one improbable step (≈ 1-in-400)"; RL lifts its own proofs | weaker than stated | worst step IQM 1-in-495 (cap 12 only; cap 6 ≈ 1-in-25,000); rest of proof 7–9 nats. On other seeds' proofs RL's lift is ≈ 2.6 nats, ≈ 1.7× the reference's, not 3×; ≈ 2–2.4 nats of the own-proof premium is selection |
| C6 "level" from step 5k | holds with caveats | a non-rejection at n = 3 (MDD ≈ 7.8 tb72); early starts ≈ 4–6 of 322 below (≈ 5–6 on the x1 read) |
| C7 nondeterminism ≈ half the seed spread | weaker than stated | variance ratio 0.543, F 95 % [0.23, 2.05], bootstrap [0.02, 1.07]; one cell, one pod |

**Re-samples** (RTX A6000, fresh seeds). R2: EI s0 **29 / 29** at p̂ ≥ 0.01, min 0.0245, max |z| 2.14 against
support-curves. R1: SN base s0 **26 / 29** reached within 10,000 (on file 26), 21 at p̂ ≥ 0.01 (on file 21);
`la_transfer_149` z = 3.13, borderline against a Bonferroni threshold of ≈ 3.15 over 29. R3: base s1 3 / 29 (above).

**Cross-run consistency.** No pair is clearly outside the sampling spread (E2 **missed**; the extremes are what
multiplicity predicts over ≈ 250 pairs). Several "agreements" are deterministic replays (same checkpoint and sampling
seed, bit-identical), e.g. rl-from-ckpt vs trajectory and trajectory-cap6's re-scoring of cap-12 proofs. They show
pipeline reproducibility, not replication.

**Leakage.** 0 survivor classes in the Stage-1 set (155,000 rows), `rl_targets` (4,495), `heldout` (5,000) or the
EI-s0 training mixes (272,632) (E3 hit). The canonicaliser is invariant to atom renaming and premise order.

**Lean.** Executor: 471 / 471 accepted, 1,435 / 1,435 controls rejected. Reviewer's own drivers: **2,782 / 2,782**
accepted, **981 / 981** controls of four kinds rejected (E4 hit). Proof size on the survivors' re-samples: 5–10 `have`
lines, term size 15–36 (reviewer's measure, premise restatements removed).

## Limitations and next step

- The registry lacks `gen_tokens` and `lean_checks`, and its `gpu_seconds` double-count the shared card.
- The executor's re-sample Lean check covered 27 proofs with one control kind. The reviewer checked all 216 and found
  nothing.
- `run_claim_audit.md` is 252 words against a limit of 250.
- Next measurements (reviewer's ranking by what each settles):
  - **C7:** identical-seed re-runs from ≥ 3 cells, including a low-mode cell, on ≥ 2 pods, ≈ 6 each. Every MDD in
    the project is planned against this ratio.
  - **C2b:** an arm with the step interface but without the rendered state, 200k × 2 T on the survivors, ≥ 3 seeds.
  - **C6:** 3 more seeds of pend and p5000 (n = 6), to bound "level" as equivalence.
  - **C3:** 2 more K12 T1 seeds and SN6 at n = 4, so the interaction can be tested.
