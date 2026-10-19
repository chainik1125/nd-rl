---
written_on: 2026-09-30
written_by: agent:claude
---

# lit-measures: the state base's one bad step, identical-seed re-run noise, and mode shares by EI round

Status: **completed 2026-09-29 23:59 – 2026-09-30 02:53 UTC (executor; a first session was killed at ≈ 16:05 UTC on
09-29, see below), independently reviewed 03:00–03:51 UTC. Verdict: M1 stands; M2 stands; M3 stands qualitatively,
but its round3-run4b per-round counts are not supported (start-index variants counted as distinct proofs).** Three
rewording groups. No hard-constraint violation, nothing quarantined. Where the reviewer's wording differs from the
executor's, this summary uses the reviewer's.

Question (proposal 16 item M, from `lit-review` §(c)). Three cheap measurements:
- **M1.** Are the state policy's EI-only proofs improbable under its base at one step (as part D found for the
  whole-proof models), and by how much more than controls?
- **M2.** How much of the Stage-1 depth-3 lottery comes from the data-order seed, as opposed to the init seed?
- **M3.** Across existing EI ladders, do seeds' style modes (box depth) separate by round 2?

The run was on the fork (`dan-pandori/nd-takehome`), branch `dan_lit-measures` (from the fork's `dan`). **Nothing is
merged into the fork's `dan` yet.** The reviewer calls the `--data_seed` trainer change (`train.py`, `fast_train.py`,
`tests/test_data_seed.py`) "fit to merge".

- Review: [`review_lit-measures.md`](https://github.com/dan-pandori/nd-takehome/blob/fb9c9579/review_lit-measures.md)
  ([`bf9644a0`](https://github.com/dan-pandori/nd-takehome/commit/bf9644a0) §Recount, written before the reviewer read
  the executor's write-up; [`fb9c9579`](https://github.com/dan-pandori/nd-takehome/commit/fb9c9579) §Compare /
  §Verdict). Reviewer scripts and outputs: `artifacts/lit-measures/review/rv_*.py`, `m1_analysis.txt`,
  `m2_recount.json`, `m3_analysis.txt` (the corrected round3-run4b values).
- Executor write-up: [`run_lit_measures.md`](https://github.com/dan-pandori/nd-takehome/blob/fb9c9579/run_lit_measures.md),
  `numbers.md` § lit-measures, `log.md` § lit-measures. Results commit `a3b394bd`, wording fix `115e9574`.
- Pre-registration: `preregistration/lit-measures.md`, commit `bd3b330e` at 00:04:53 UTC. The first pod (`lm1`) was
  created at 00:06:39 UTC, so the pre-registration came first. Amendment 1 (identical-seed replicates, E2.4) was
  committed at 00:40:07 UTC while the first grid cells were running. It adds a read-out and changes no hypothesis.
- Brief: [`BRIEF_lit-measures.md`](../../docs/proposals/literature/BRIEF_lit-measures.md).

## Configuration

- **Checker: Lean alone** (Lean 4.34.0 core). `nd_verify` is unmodified and on no path. Logs are natural (nats).
- **M1** (VPS CPU, forward passes only, no sampling):
  - Scored under **SN base s0**: state-env `ckpts/se/stage1_SN_s0.pt`, md5 `ec3888d9`, 3,216,384 params,
    `lean_staten`, from scratch on `data/p2/train_depth3_f0_a1.jsonl`, cap 6.
  - The proofs scored are **SN EI s0**'s (`la_T1_SN_s0_r8.pt`, `fb448247`), taken from support-state's records.
  - Sets:
    - **S**: the 7 theorems (`1108 1185 1352 198 2089 394 988`) the base never reaches at 200,000 attempts per
      temperature; 91 distinct proofs.
    - **C1**: 35 length-matched theorems the base does reach (pre-registered, seed 0); 466 proofs.
    - **C2**: the base's own proofs of the C1 theorems; 86 proofs.
    - **C1x**: 35 forward-crux theorems the base reaches only in S2 (post hoc, labelled).
  - Actions are scored in `Env(canon=True, assign=True)` at T 0.8 (EI's temperature, the primary) and at T 1.0 (part
    D's primary). The unit is the best proof per theorem.
- **M2** (pods): 72 Stage-1 GPTs, 3,214,336 params, `lean_seq`, from scratch, cap 6.
  - Recipe: `train.py --impl fast`, 6,000 × 128 on noise-floor's `data/nf/train_p1.jsonl` (md5 `e80aaa0c`).
  - Seeds: init 0–7 × data order 100–107 (8 × 8, not the brief's 3 × 3: a pre-registered simulation showed 3 × 3 gives
    a degenerate interval). Plus 8 identical-seed re-runs of (0, 100).
  - Read-outs: held-out greedy on `data/p2/heldout.jsonl` (5,000; depth-3 slice 500), and final validation loss.
  - Analysis: two-way random-effects variance shares, with a two-way bootstrap.
- **M3** (files only): the per-round `found_*.jsonl` of 111 ladders (19 EI arms, **44** seeds, 10 run families; 17
  frozen arms as the no-RL reference). Style is the maximum box depth of each found proof. Each ladder keeps its own
  run's model label. Note that round3-run4b's `ei_depth3_*` arms are **25 M / 85 M-parameter models**, not 3.2 M.
- **Pods:** `lm1` RTX 3090 community $0.22/h (00:06–00:13, no working CUDA, deleted); `lm2` RTX 3090 community $0.22/h
  (00:07–01:22, lost its GPU mid-run after 16 grid cells); `lm5`, `lm6` RTX 3090 secure $0.50/h (01:23–02:43). `lm3`
  and `lm4` were create attempts that failed or hung, ≈ $0.05, outside `podbudget`.
- **Spend:** `podbudget` 3.75 pod-hours / **$1.50** + ≈ $0.05 unregistered ≈ **$1.55**. Budget $3 / 6 h.
- **Compute** (`artifacts/lit-measures/compute.tsv`, registry rows under `artifacts/lit-measures/registry/`):

  | arm | GPU-s (RTX 3090) | train steps | Lean-judged attempts |
  |---|---|---|---|
  | M2 grid (64 cells) | 8,113 | 384,000 | 320,000 |
  | M2 replicates (8) | 995 | 48,000 | 40,000 |
  | M2 default-reproduction check | 338 | — | — |

  Train tokens per cell ≈ 1.066 × 10⁸ (`useful_tokens`). Not recorded: eval GPU-seconds, generated tokens, M1 CPU
  time, M3 pod time (reviewer). The write-up has no per-arm table (it is in `numbers.md` / `compute.tsv`).
- **Bucket:** `hf://buckets/dan-pandori/nd-rl/lit-measures/` (`ckpts/lm/`, 73 `.pt`; `artifacts/lit-measures/`; bulk
  files in `artifacts/MANIFEST.jsonl`).
- **Deviation: the first session was killed.** At ≈ 16:05 UTC on 09-29 the first executor session started 160
  parallel `hf buckets cp` processes. That exhausted the VPS's memory and killed both sessions on the host for ≈ 8 h.
  No commit and no pod was lost. `AGENT_POLICY.md` now caps the VPS at 4 concurrent processes.

## Observations

All numbers below were re-derived by the reviewer with independent code unless marked otherwise.

**M1 — the state base's EI-only proofs have one very bad step (stands).** Model: SN base s0 (`ec3888d9`).

| quantity (best EI proof per theorem) | T 0.8 | T 1.0 |
|---|---|---|
| S labelled concentrated (part D rule), E1.1 ≥ 5/7 | **7 / 7** ✓ | 6 / 7 |
| median worst step w1: S / C1 / C1x | **−14.71** / −1.56 / −10.59 | −12.06 / −1.65 / −8.51 |
| one-sided permutation p, S vs C1 (E1.2) | **0.0002** ✓ | 0.0002 |
| w1 < ln(3/400,000) = −11.80: S / C1 / C1x (E1.3 ≥ 4/7) | **5/7** ✓ / 2/35 / 9/35 | 4/7 / 2/35 / 3/35 |

- In S, the second-worst step has median **−4.6** nats, and all other steps together **−1.5** (reviewer's wording;
  the executor's "remaining steps −1.5" left out w2).
- The worst step is 13 nats below C1. It is 4.1 nats below C1x (post hoc, p 0.003; the reviewer did not re-derive this
  p), and 31/35 of C1x are also labelled concentrated. So the new-theorem proofs differ from rarely-reached controls
  in degree, as in part D, and the ln(3/N) cut is not sharp: 9/35 of C1x fall below it.
- The worst step of each S proof opens a box: 4 `Or.elim` openers, 2 `Classical.byContradiction` lines, and one `.2`
  projection.
- Part D recount (whole-proof base `stage1_a1_seq_s0.pt`, `9bde44c0`, `lean_seq`, 3,214,336 params): 19/29 below
  −11.80, 22/29 below −11.11.
- **Deviation, not labelled by the executor, immaterial:** the pre-registration said "no name marginalisation needed".
  The executor summed over name bases 0…64 − max name, while the sampler draws the base uniformly from [0, 32]. The
  reviewer re-scored with four attributions (the executor's; the sampler mixture; the mixture with name tokens scored;
  base 0 only). Every E1 verdict is the same under all four.
- Lean: 643 / 643 scored proofs accepted (literal text and via `nd2lean`), 200 / 200 corrupted controls rejected.
  Term size (lean_check inference nodes) median S 10, C1 6, C2 5. The `term_size` field in support-state's records is a
  different quantity and must not be quoted as term size.

**M2 — init seed vs data order: the data order explains almost none of it, and re-runs at identical seeds explain
about half (stands).** Models: 3.2 M `lean_seq` Stage-1, fast path, `train_p1`.

| quantity | mean | sd | range | init share [95 %] | data share [95 %] | residual | F-test p (init / data) |
|---|---|---|---|---|---|---|---|
| depth-3 slice | 0.496 | 0.305 | 0.006–0.900 | 0.17 [0.00, 0.59] | **0.02** [0.00, 0.41] | 0.81 | 0.020 / 0.34 |
| held-out overall | 0.913 | 0.033 | 0.852–0.959 | 0.21 [0.00, 0.61] | 0.03 [0.00, 0.41] | 0.75 | 0.007 / 0.24 |
| final val loss | 0.07262 | 0.00020 | 0.0722–0.0731 | 0.21 [0.01, 0.55] | 0.32 [0.00, 0.67] | 0.48 | 0.001 / < 0.001 |

(Reviewer's bootstrap, 4,000 draws; the executor's intervals agree to ±0.01.)

- **E2.1** (data order ≥ 30 % of depth-3 variance): **not supported, not falsified**. The point estimate is 0.02,
  and the upper bound 0.41 does not reach the pre-registered falsifier (< 0.30).
- **E2.2** (data share > init share): **missed** (0.02 < 0.17; the intervals overlap, so the falsifier is not met). The
  write-up did not report this; the reviewer requires it.
- The **init seed has a detectable effect on the continuous depth-3 rate** (F p = 0.020), and the data order has none
  (p = 0.34). The binary high mode (depth-3 ≥ 0.5, 36 / 64 cells) follows neither (permutation p: rows 0.16,
  columns 0.61).
- **Identical-seed re-runs** (9 runs of init 0 / data 100, the grid cell included): depth-3 0.900 0.502 0.222 0.726
  0.796 0.832 0.850 0.850 0.874, sd **0.225** (E2.4 ≥ 0.10 ✓), 8/9 in the high mode. Replicate variance is **0.55 ×**
  the grid's depth-3 variance. GPU non-determinism alone produces about half the spread on this recipe. The reviewer
  calls this the most useful finding of the run: choosing seeds cannot control this variance, only more runs average
  it down.
- **E2.3** holds: the fast path reproduces noise-floor's bimodality (sd 0.305, range ⊇ [0.1, 0.8]).
- **Pod as a nuisance factor** (from the log; the reviewer asks for it in the write-up). By pod, depth-3 is lm5 0.433
  vs lm2/lm6 0.560, permutation p 0.09. The cells were laid out in a checkerboard by pod, so this sits in the residual
  and not on either factor.
- `--data_seed`: the default is bit-identical to the old trainer on CPU (reviewer: 20 steps, legacy). On GPU, same-code
  re-runs drift as much as old vs new code. The 6,000-step legacy `--seed 0` run ends at val 0.0724 vs noise-floor's
  0.0729. The reviewer did not test the fast path for bit-identity.
- Lean: 7,200 counted held-out proofs from 72 cells re-checked, all accepted; 200 / 200 controls rejected. Term size
  median 3 on depth-3 proofs, 2 on the rest.
- **Policy miss (unreported, immaterial):** 79 of 36,000 depth-3-slice samples ended without `<eos>` (0.22 %, worst
  cell 4 / 500), above the ≈ 0.1 % line. That moves one cell's depth-3 rate by at most 0.008. Eval peak memory is not
  recorded (training peak 0.39 GiB).

**M3 — mode shares by round (stands qualitatively; round3-run4b counts not supported).**
- **E3.1** (pairwise agreement with the final seed ordering at round 2 ≥ 0.75, cumulative depth-≥ 3 share): **19.5 /
  26 = 0.75, met on the letter only.** It reproduces after deduplication. Next to it:
  - with ties not credited, 0.69;
  - the new-proof share of round 2 alone, 0.56;
  - disjoint sets (rounds 1–2 vs rounds 3–8), **0.52**, which is chance;
  - **frozen ladders with no training, 0.89**.

  The cumulative round-2 share is part of the final share, so agreement is partly built in. The frozen reference agrees
  more than the trained arms, so the statistic measures persistence of the base's style, not RL separating modes.
- **Most arms have no modes to separate.** In 13 of 15 whole-proof and state arms, seeds' final depth-≥ 3 shares differ
  by ≤ 0.03. The only bimodal arms are round3-run4b's `25Mr_mix` (0.766 / 0.770 / 0.044) and `85Mr_mix` (0.009 /
  0.231 / 0.760), which are 25 M / 85 M models. In 25Mr the low seed is separated by round 1–2. In 85Mr all three seeds
  are at 0.000 at round 2 and separate only at rounds 5–7. So the brief's question gets one "by round 2" and one "only
  later", from 3 seeds each.
- **E3.2** (new-proof depth-≥ 3 share monotone in at least half the ladders): **missed**, 4–5 / 44 (4 after dedup, 5 by
  the executor). The write-up did not say so.
- **Convergence (post hoc; ds-generator g1 / g2 only, two arms of one family).** EI brings the seeds together: g1 seed
  spread 0.56 → 0.03 (frozen 0.56 → 0.50), g2 0.47 → 0.01 (frozen 0.41). The low seed jumps from 0.05 to 0.62 (g1) and
  from 0.17 to 0.68 (g2) in one EI round. The other arms' seeds start within ≈ 0.03 of each other, so they have nothing
  to converge.
- **Not supported: round3-run4b's per-round counts and depth shares in `per_round.tsv` / `numbers.md`.** Those found
  files store proofs at raw start indices, and the executor counted each start-index variant as a separate proof (e.g.
  `ei_depth3_85M_s1_mix`: 4,377 rows = 115 distinct proofs). This is the start-index-normalisation bug the policy
  lists. After deduplication the end gaps are 25Mr **0.726** and 85Mr **0.751** (executor: 0.65 / 0.54). The E3.1
  headline does not move. The other 1,364 of 1,488 (ladder, round) cells are identical to the reviewer's count.
- M3 adds no new counted proofs; the earlier runs' reviews cover their Lean checks.

## Limitations and next step

- M1 has n = 7 new theorems, from one seed pair (SN base s0 / SN EI s0).
- M2 is one recipe (3.2 M `lean_seq`, fast path, one GPU class). Its intervals are wide: at 8 × 8 an init or data
  share up to ≈ 0.4–0.6 cannot be excluded. The replicate estimate comes from one cell.
- M3's only informative arms are two bimodal arms of 25 M / 85 M models with 3 seeds each.
- Reporting gaps (reviewer): E2.2 and E3.2 misses, the pod nuisance, the 0.22 % truncation, the name-base deviation and
  the 41 → 44 seed count were missing or wrong in the write-up. They are corrected here. `run_lit_measures.md` is
  352 words against a 350 limit.
- The fork's `dan` does not have `--data_seed` yet.

Next measurements (reviewer):
1. **Is the depth-3 mode set by early training noise?** Run 16–32 identical-seed replicates of 2–3 cells, one per GPU
   class. Then fork replicates from a common step-k checkpoint (k = 500, 1,500, 3,000) to find when the mode is decided.
2. **For M3:** more seeds of round3-run4b's `85Mr_mix` / `25Mr_mix` (3 each now). Or a 3.2 M analogue started from
   bases in the low depth-3 mode, since ds-generator shows EI lifts a low-mode seed in one round.
