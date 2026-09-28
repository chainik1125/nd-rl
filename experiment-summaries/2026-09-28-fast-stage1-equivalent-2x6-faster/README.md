---
written_on: 2026-09-28
written_by: agent:claude
---

# fast-stage1: can Stage-1 training use the GPU without changing the model it trains? (proposal 15, item 1)

Status: **completed and reviewed 2026-09-28. Yes, for the tested configuration: the fast path trains an
equivalent model (14 / 14 pre-registered comparisons within the n = 8 MDD) about 2.6× faster per model on
an A40. The ≥ 5× goal is not met.** No hard-constraint violation, nothing quarantined. The reviewer's
verdict is **stands, with eight rewordings**. Where the reviewer's wording differs from the executor's, this
summary uses the reviewer's. This is an infrastructure run: it makes no capability or proof-length claim.

Question: the brief measured a 6,000-step Stage-1 run at 1,288 s on an A40 (≈ 215 ms/step, 0.78 % of bf16
peak) and asked for the same models ≥ 5× faster (aim 10×), with several seeds per GPU. Brief:
[`BRIEF_fast-stage1.md`](../../docs/proposals/improvements/BRIEF_fast-stage1.md); proposal 15 §1
([`2026-09-28-repo-and-method-improvements.md`](../../docs/proposals/2026-09-28-repo-and-method-improvements.md)).

Review: [`review_fast-stage1.md`](https://github.com/dan-pandori/nd-takehome/blob/c19e1f42/review_fast-stage1.md)
(fork branch `dan_fast-stage1`; commits [`35f299d0`](https://github.com/dan-pandori/nd-takehome/commit/35f299d0)
§Recount, blind; [`c19e1f42`](https://github.com/dan-pandori/nd-takehome/commit/c19e1f42) §Compare/§Verdict).
Reviewer code: `review_fs/` on that branch, including a torch-free fp32 numpy re-implementation of the model
and greedy decoding. Executor write-ups: `FAST_STAGE1.md` (how to train, as the code now is),
`run_fast_stage1.md`, `numbers.md` § fast-stage1 (FS1–FS9), `log.md`, `artifacts/fs/summary.json` and
`tables.md` (recomputed by `fs_analysis.py`), all at
[`1f97638b`](https://github.com/dan-pandori/nd-takehome/commit/1f97638b). Pre-registration:
`preregistration/fast-stage1.md`, commit `f5a6cdd6` at 16:23 UTC, before the first pod (16:28).

No figure: the run made none, and the tables below are the result.

## Configuration

- **What was built** (`fast_train.py`, called from `train.py --impl fast`; fork commit `e62d30eb`, default
  switched to `--impl auto` in `1f97638b`):
  - the training set is tokenised once and cached (`$ND_TRAIN_CACHE`); batches and the random name offset
    are drawn on the GPU with legacy's semantics (a fresh permutation per epoch, remainder dropped; offset
    uniform on {0 … 64 − mx});
  - the 128 records of a batch are **packed** into one stream with document-causal `flex_attention` and
    per-record RoPE positions;
  - the loss is `torch.compile`d, AdamW is fused, and the whole step is one CUDA graph.
  - The reviewer read `fast_train.py` against `train.py`'s legacy loop and found the same init per seed,
    loss, optimiser, schedule, epoch semantics and augmentation. Only the RNG streams and float order differ.
  - **`--impl auto` is now the default** for every from-scratch GPU run on the fork's `dan`. `--init`
    fine-tunes (EI, ladder) and legacy `--state_at` resumes still use the legacy path. `--impl legacy`
    reproduces an old trajectory (plausible by inspection, not re-run).
- **Model every number is about:** the 3,214,336-parameter from-scratch GPT (4 layers, d 256, 8 heads),
  `lean_seq`, cap 6, 6,000 steps at batch 128 on `data/p2/train_depth3_f0_a1.jsonl` (155,000 records),
  cosine 1e-3 → 1e-4, warm-up 200, `--val_bins` (stage1-dynamics arm C's command).
  - **new**: `--impl fast`, seeds 0–7, `ckpts/fs/{fast1_s0,fast2_s1,fast2_s2,fast4_s3..s6,fast1b_s7}.pt`,
    trained on one A40, 1, 2, 4 and 1 at a time.
  - **C**: stage1-dynamics arm C, legacy code, seeds 0–7, `ckpts/sd/c_s*.pt` (A40s), on file.
  - Both arms share the initial weights per seed (same `torch.manual_seed` before `GPT()`), not data order.
- **Checker: Lean alone** for both arms (`sd_eval.py`, held-out greedy on `data/p2/heldout.jsonl`, 5,000
  theorems, batch 512, `max_new` 400, fast sampler). `nd_verify` judges nothing; it is still called by
  `train.load` as a data-integrity check on training records, inherited from legacy.
- **Equivalence test** (pre-registered): 14 comparisons, 7 held-out greedy slices and 7 validation-loss
  slices, each passing if |Δ| < MDD = 1.5064 · sd at n = 8 per arm (sd from `NOISE_FLOOR.md` for greedy,
  from arm C for val loss).
- **Compute:** `fs1` NVIDIA A40 at $0.49/h (1.24 h, $0.61) and `fs2` NVIDIA H100 80GB HBM3 (SXM) at $3.49/h
  (0.25 h, $0.86), 16:28–≈17:42 UTC, both deleted. **1.49 pod-hours, $1.47** of a $5 / 10 h budget.
- **Bucket:** `hf://buckets/dan-pandori/nd-rl/fast-stage1/{artifacts,ckpts}` (the 8 fast checkpoints and
  `legfull_s0.pt`; the reviewer confirmed both are present).

## Observations (reviewed)

**(a) Equivalence: 14 / 14 within the MDD, recomputed independently** from the per-record `lean_ok` flags
and `m_*.jsonl` metric files.

| slice | greedy new | greedy C | Δ (pp) | MDD (pp) | val loss new | val loss C | Δ | MDD |
|---|---|---|---|---|---|---|---|---|
| all | 0.9001 | 0.9105 | −1.04 | 4.58 | 0.05124 | 0.05101 | +0.00023 | 0.00219 |
| len2 | 0.9939 | 0.9960 | −0.21 | 0.47 | 0.07824 | 0.07817 | +0.00007 | 0.00036 |
| len3 | 0.9879 | 0.9864 | +0.15 | 0.73 | 0.06780 | 0.06780 | 0.00000 | 0.00045 |
| len4 | 0.9537 | 0.9469 | +0.69 | 2.28 | 0.05111 | 0.05139 | −0.00028 | 0.00049 |
| len5 | 0.9189 | 0.9135 | +0.54 | 2.26 | 0.03964 | 0.03958 | +0.00006 | 0.00077 |
| len6 | 0.6459 | 0.7096 | −6.38 | 22.94 | 0.04322 | 0.04238 | +0.00084 | 0.00775 |
| 6-line non-depth-3 | 0.8880 | 0.8770 | +1.10 | 5.52\* | 0.04533 | 0.04551 | −0.00017 | 0.00084 |

- \* This MDD borrows NOISE_FLOOR's 247-theorem "6-line no-pattern" sd, a different slice (disclosed). The
  tested slice's own sd gives MDD 2.49 pp; |Δ| 1.10 pp is still within.
- 13 of 14 are under ½ MDD. **Val-loss len4 is at 0.57 MDD**, so the pre-registered "every |Δ| under half
  its MDD" missed on one comparison (reported as a miss).
- **"Within the MDD" means no difference ≥ the 80 %-power MDD was detected, not that smaller ones are
  excluded** (reviewer). IQM differences with stratified-bootstrap 95 % intervals: overall −1.90 pp
  [−4.83, +2.33]; 6-line −10.9 pp [−26.9, +11.3]; 6-line non-depth-3 +1.05 pp [−0.55, +2.65].
- **Depth-3** (bimodal, not a criterion): 4 / 8 fast seeds in the high mode (> 0.44) vs 6 / 8 arm C and
  3 / 8 arm R (legacy replicates); Fisher p = 0.61 vs C. 4 / 8 lies inside legacy's own spread. Seeds 5 and 7
  are low in both arms, a hint (n = 8 cannot show it) that the mode depends partly on initialisation.
- The fast path's `--val_bins` numbers match the reviewer's own fp32 forward pass over all 5,000 held-out
  records to ≤ 0.16 % relative on every slice, so the val-loss comparison is like for like.

**(b) No counted proof that Lean rejects.**
- Arm C: all 38,676 stored texts re-checked with the reviewer's harness; its verdict equals the recorded one
  on **38,676 / 38,676**. Controls behaved (0 / 60 wrong-theorem, 0 / 30 truncated, 0 / 5 `sorry` accepted).
- New arm: **no literal text was stored** (`sd_eval.py` ran without `--texts`). The reviewer regenerated 150
  theorems × 8 checkpoints with its own fp32 numpy model: **1,038 of 1,041 counted proofs are Lean-accepted**;
  the 7 disagreements (3 + 4) are bf16-vs-fp32 argmax flips at top-2 margins < 0.03 logits (≈ 1 in 170).
- Mean term size of counted proofs 84.99 (new) vs 86.44 (C); lines (`have` + 1) 4.16 vs 4.15. The len6 gap
  (123.3 vs 128.7) follows from fewer depth-3 solves in the new arm.

**(c) Speed** (reviewer's steady-state numbers, step 400 to end, compile and validation excluded):

| path, GPU | N concurrent | ms/step per process | aggregate steps/s |
|---|---|---|---|
| legacy, A40 | 1 / 2 / 4 | 48.5–50.2 / 103–105 / 214–216 | 20.3 / 19.2 / 18.6 |
| fast, A40 | 1 / 2 / 4 / 8 | 16.6–17.3 / 35–37 / 74.5 / 148–149 | 59.2 / 55.2 / 53.7 / 53.7 |
| legacy, H100 | 1 | 28.2 | 35.5 |
| fast, H100 | 1 / 2 / 4 / 8 / 16 | 4.1 / 8.8 / 17.1 / 33.3 / 70.8 | 244 / 228 / 234 / 240 / 226 |

- **Headline, per the reviewer: 2.6× per model** (process wall-clock 345 s → 128–135 s; 2.5× in-loop) and
  **≈ 2.9× per step / per GPU on the same A40** (16.8 vs 48.9 ms/step). The **≥ 5× goal is not met.**
- **The brief's 215 ms/step was legacy with ≈ 4 co-tenants**: legacy at N = 4 runs at 214 ms/step per
  process, exactly the brief's number; alone it is 48.9. Against the contended 1,288 s the fast path is
  ≈ 10×, but that is a premise correction, not the headline.
- Padding waste 1.97× → 1.08–1.13×. Peak memory 0.41–0.44 GB per process (bs 128). Model FLOP/s 13.5 % of
  A40 bf16 peak in steady state (11.9 % including compile), legacy 4.7 %.
- **Concurrency gives no aggregate training gain** on either GPU class (A40 fast N = 8 is 0.91× N = 1; H100
  N = 16 is 0.93×). The executor's "+10 % at N > 1" on the A40 and the H100's "65 → 188 steps/s at N = 16"
  are whole-job numbers: concurrent processes overlap each other's fixed costs (start-up, ≈ 9–15 s compile
  and capture, validation) in short waves. For 6,000-step models the reviewer estimates ≈ 1.7× at N = 16 on
  an H100.
- **H100:** one process runs at 4.1 ms/step, ≈ 4.1× an A40, at ≈ 1.7× an A40's cost per model
  ($3.49 vs $0.49/h).
- **Batch 512** (500-step timing runs only, never equivalence-tested): no faster per token on the A40
  (1.05×), ≈ 1.3× faster per token on the H100.

**(d) Expectations vs measured** (pre-registered): equivalence 14 / 14 held; full model with `--val_bins`
≤ 4 min held (2.1 min); fast ≤ 25 ms/step held; legacy padding 1.8–2.0× held. **Missed:** legacy 180–230
ms/step (48.9 alone; the premise was a co-tenancy number); ≥ 8× per step (2.9×); ≥ 5× per model (2.6×);
aggregate ≥ 2× by N = 4 (0.91×); every |Δ| < ½ MDD (13 / 14); **≤ 10 % of bf16 peak (11.9 %, higher than
predicted — the executor did not flag this one)**. Fast padding ≤ 1.1× borderline (two seeds at 1.113×).
The bs-512-at-equal-tokens comparison was exploratory and not run.

**(e) Hard constraints** (reviewer): `nd_verify/` tree-identical to `origin/main` and not used as a judge;
`artifacts/TEST_RUN_DONE` unchanged; no test file read by training code; the held-out file enters training
only as the no-grad `--val_bins` loss, as in legacy. Splits: 22 training records share the renaming class of
21 held-out theorems (0 depth-3), the known premise-order overlap of this training file, identical in both
arms.

## Rewordings the reviewer required (applied here; the executor's files still carry the old wording)

1. "per-GPU ≈ 3.2× legacy's best" / "2.6–3.2×" → **≈ 2.9× per GPU** (the 3.2× compared 6,000-step fast
   waves with 1,000-step legacy waves).
2. H100 "use N = 8–16 … ≈ 1.9× cost per model" → one process at 4.1 ms/step (≈ 4.1× an A40); concurrency only
   overlaps start-up; cost per model ≈ 1.7× an A40's.
3. A40 "N > 1 gains only ~10 %" → true for whole jobs; steady-state throughput drops 9 %.
4. "bs 512 no faster per token on either GPU" → true on the A40, not on the H100 (≈ 1.3×).
5. Self-test "offsets vs `shift_abs` min p 0.33 over all mx" → **mx = 2 only**: the first 2,000 training
   records are all 2-line proofs (the file is length-sorted). No bug is expected (the code has no
   mx-specific branch), but the stated coverage is overstated.
6. Put the uncontended 2.6× first; the ≈ 10× vs 1,288 s belongs in the premise correction.
7. The pre-registration's listed "sd_C" values are the MDDs (1.506 × sd); the analysis used the right sds.
8. Mark "≤ 10 % of bf16 peak" (11.9 %) as a miss.

## Limitations and next step

- **Tested configuration only:** `lean_seq`, bs 128, cosine, 6,000 steps, from scratch, 3.2 M parameters.
  `--impl auto` already routes `lean_rand`, token formats (`abs`/`rel`), other batch sizes, `--sched wsd` and
  `--state_at`/`--resume` to the fast path; `smoke.log` shows they run, not that they are equivalent. Before a
  run relies on one of them, a small equivalence check (≥ 5 seeds against the MDD) in that configuration.
- **Resume is not trajectory-exact** (val 0.3315 resumed vs 0.3342 uninterrupted at step 400); disclosed.
- **Do not mix fast and legacy Stage-1 models across the arms of one experiment** whose effect on the 6-line
  bin is below ≈ 25 pp: the equivalence interval there is [−26.9, +11.3] pp. The policy's same-settings rule
  already requires this.
- **Process:** the run's commits, including the `--impl auto` default, were pushed to the fork's `dan` at
  17:44 UTC, before review. The review found no correctness problem, so nothing needs reverting, but every
  from-scratch Stage-1 run on the fork's `dan` since then used a path whose review was pending.
- **Next measurements (reviewer):**
  - rerun `sd_eval.py --texts` on the 8 bucketed fast checkpoints (≈ 3 GPU-minutes) so their counted proofs
    are re-checkable from literal text like every other arm's;
  - rerun `fast_train_selftest.py` on records drawn across all lengths so the augmentation test covers
    every mx;
  - if H100s are to be used, one 6,000-step wave at N = 1 and N = 16 to replace the 2,000-step
    extrapolation with a measured cost per model.
