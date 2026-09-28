---
written_on: 2026-09-28
written_by: agent:claude
---

# lean-prefilter: can Lean come off the RL critical path without changing a verdict? (proposal 15, items 3–4)

Status: **completed and reviewed 2026-09-28. Yes: the pre-filter is sound on everything checked, and
one T1 round is ≈ 4× faster.** No hard-constraint violation, nothing quarantined. The reviewer's
recommendation is **accept, with six rewordings**. Numbers and wording below are the reviewer's where
the two differ. This is an infrastructure run: it makes no capability or proof-length claim.

Question: the RL loop's Lean gate cost ≈ 30 % of a T1 ladder round, according to the brief's figures
from noise-floor's `la_T1_dsc_a1_s1.log`, and 78,151 of the 92,074 distinct texts it checked in round 1
were rejects. Can Lean's time come off the critical path without changing a single verdict? The run
tested three changes:
- a sound, reject-only Python pre-filter in front of Lean;
- a Lean worker pool sized to the pod's CPU quota;
- checking one decode chunk while the GPU samples the next.

Brief:
[`BRIEF_lean-prefilter.md`](../../docs/proposals/improvements/BRIEF_lean-prefilter.md).

Review: [`review_lean-prefilter.md`](https://github.com/dan-pandori/nd-takehome/blob/93c31ed/review_lean-prefilter.md)
(fork branch `dan_lean-prefilter`; commits [`5cae87b`](https://github.com/dan-pandori/nd-takehome/commit/5cae87b)
§Recount, blind; [`93c31ed`](https://github.com/dan-pandori/nd-takehome/commit/93c31ed) §Compare/§Verdict).
Reviewer code: `review_lp/` on that branch, including its own Lean runner and its own 32,256-text
adversarial set. Executor write-ups:
[`run_lean_prefilter.md`](https://github.com/dan-pandori/nd-takehome/blob/e455e03/run_lean_prefilter.md) and
[`LEAN_GATE.md`](https://github.com/dan-pandori/nd-takehome/blob/e455e03/LEAN_GATE.md), which describes the gate
as it now is; also `numbers.md` § lean-prefilter (P1–P4) and `log.md`. Pre-registration:
`preregistration/lean-prefilter.md`, commit `9238edc` at 16:27:41Z, before the first pod at 16:33:55Z.

No figure: the run made none, and the tables below are the result.

## Configuration

- **What was built** (on fork branch `dan_lean-prefilter`; the pods ran `57bff79`, results `903bade`):
  - `lean_prefilter.reject_reason(statement, text)`. It is a type checker for the strict `lean_seq`
    fragment and runs only on texts `lean_tok.inverse` has already parsed. In that fragment the only
    definitional unfolding is `¬A := A → False`, so it compares formulas syntactically after that
    rewrite. It resolves `n.elim` by the **declared** head of `n`'s type (`False.elim`, `Not.elim`,
    `And.elim`, `Or.elim`). It rejects only on a certain Lean error (20 reason codes). Anything it
    does not model passes to Lean.
  - `lean_gate.Gate`. It runs the filter, then Lean, on each decode chunk while the GPU samples the
    next. Workers default to the cgroup CPU quota (7 on a RunPod A40 that shows 96 cores). Lean now
    runs with `-j 1` (`LEAN_GATE_THREADS`).
  - Knobs: `LEAN_PREFILTER=on|off|shadow` (default `on`). `shadow` sends every text to Lean and logs
    every filter reject that Lean accepted to `*.filterbug.jsonl`. The other knob is
    `LEAN_GATE_PIPELINE`.
  - The loop scripts' `--batch` default is now 4,096. The gate log records the truncation rate.
  - Not covered: `lean_judge`'s relabel fallback still calls Lean directly. `grpo.py` keeps batch 512.
- **Checker: Lean 4.34 core alone** (pods 4.34.0; the reviewer's re-checks 4.34.1). `nd_verify` is not
  called anywhere.
- **Models.** All are 3,214,336 parameters, trained from scratch.
  - T1 timing: `ckpts/dsc/stage1_a1_s1.pt` (`lean_seq`, trained on `data/dsc/train_a1.jsonl`,
    ds-composition). One round, k 32, T 0.8, `max_new` 512, seed 1, one A40 per arm running alone.
  - Soundness corpus C1: eight checkpoints, each trained on its own run's set:
    - ds-composition a1_s1 and a3_s0;
    - ds-generator g2_s0;
    - lean-format ei_d3_seq_s0_r8, stage1_a1_rand_s0 (`lean_rand`, the only non-`lean_seq` model) and
      stage1_full_seq_s0;
    - cap-horizon stage1_k14_s0;
    - noise-floor stage1_p2_s3.

    They were sampled in shadow mode at T ∈ {0.8, 1.0, 1.2}, k 12 (first three) or 6, batch 4,096,
    `max_new` 512, over the ladder targets (4,495), transfer (2,285) and held-out (5,000) pools:
    2,332,440 samples.
  - C2: `lp_edge.py` mutants of 12,000 Lean-accepted texts. C3: every stored `*.disagree.jsonl`,
    lean-judge's test-5 dump and the stored leanrej samples, re-checked in Lean.
- **T1 arms** (the reviewer checked each against `t1.sh` and the code at `57bff79`):
  - A (old): batch 512, 3 workers, filter off, no pipeline, compaction off.
  - B: batch 512, 7 workers, filter on, pipelined.
  - C (pre-registered new path): as B, at batch 4,096 with compaction on.
  - D (added after the worker sweep): C plus `lean -j 1`.
- **Compute:** two NVIDIA A40 secure pods at $0.49/h billed: `lp-t` (1.44 h, $0.71) and `lp-k`
  (1.22 h, $0.60), 16:33–≈17:59 UTC, both deleted. **2.67 pod-hours, $1.31** of a $3 / 6 h budget.
- **Bucket:** `hf://buckets/dan-pandori/nd-rl/lean-prefilter/artifacts/lp/` (c2, c3, corpus, logs, t1,
  soundness, test_b, workers; the reviewer confirmed each is present).

## Observations (reviewed)

**(a) Soundness: 0 false rejects.** The reviewer re-ran the final `reject_reason` on every stored text
and de-duplicated (prompt, text) globally.

| corpus | distinct | Lean ok | Lean rej | filter rejects of Lean-accepted texts | filter share of Lean's rejects |
|---|---|---|---|---|---|
| C1 model samples (8 ckpts) | 1,214,162 | 499,565 | 714,597 | **0** | 100.000 % |
| C2 edge corpus | 51,274 | 22,918 | 28,356 | **0** | 100.000 % |
| C2r second edge file | 25,007 | 9,991 | 15,016 | **0** | 100.000 % |
| C3 stored records | 19,676 | 7,763 | 11,913 | **0** | 100.000 % |

- That is 1,310,119 texts, each distinct within its corpus (**1,307,551 distinct across corpora**). It
  meets the pre-registered ≥ 1 M (C1) and ≥ 50,000 (C2).
- **The filter and Lean agree on every text**, so on this fragment the filter is complete as well as
  sound. The pre-registration predicted ≥ 95 % of Lean's rejects (the brief aimed at ≥ 70 %); the
  measured figure is 100 %.
- The positive side is exercised. In C1, 54,141 Lean-accepted rows use `.elim`, 2,418 of them on a
  declared `¬` (the `Not.elim` skip-a-step class), and all passed.
- **The reviewer's own adversarial set** has 32,256 texts: 18 term forms, 15 premise types declared both
  as `¬ A` and `A → False`, and ≈ 40 targets. Lean accepted 558 and rejected 31,698, and the filter
  agreed on every one, including `.elim` on False (224), on `¬` (136), on `∧` (23) and on `∨` (13).
- **Lean re-check** of 4,440 stored verdicts with the reviewer's own runner: **0 disagreements**. This
  includes ≥ 150 counted proofs per T1 arm.
- Code reading found no false-reject path within the strict grammar.
- Caveat on coverage: C1, the model corpus, barely exercises `Not.elim` (70 accepted) and never uses
  `And.elim` / `Or.elim` field notation. The edge corpora and the reviewer's set carry that side.

**(b) Identical accepted sets, filter off vs on.** On 120,000 texts, 14,816 were accepted with the
filter off and with it on, and 0 differ. Over all 142,432 distinct arm-A texts, the filter passes
exactly the 19,915 texts Lean accepts. Wall time 421.4 s → 43.4 s.

**(c) One T1 round, old vs new** (A40, n = 1 per arm):

| arm | round s | ratio to A | gate exposed s | gate share | Lean texts | accepted |
|---|---|---|---|---|---|---|
| A old | **1,110.8** | 1 | 634.4 | **57.1 %** | 142,907 | 19,915 |
| B (batch 512) | 484.7 | 0.436 | 16.1 | 3.3 % | 19,934 | 19,915 |
| **C (pre-registered)** | **271.8** | **0.245** | 16.4 | **6.0 %** | 20,051 | 20,035 |
| D (post hoc, `-j 1`) | 258.6 | 0.233 | 13.0 | 5.03 % | 20,051 | 20,035 |

- **Accepted sets.** A = B exactly, and their found-proof files are byte-identical (1,533 new proofs).
  C = D, also byte-identical (1,544). A and C differ because batch changes the sample stream, a
  re-draw. On the 34,380 texts both sampled, verdicts disagree 0 times.
- **Headline, as the reviewer words it: one round went from 1,111 s to 272 s with the pre-registered
  arm C (0.245).** The brief's target of ≤ 1/3 is met. **The pre-registered prediction of 0.30–0.45
  missed, on the fast side.** D (259 s) is a post-hoc variant. C → D is −5 %, inside the run's own 10 %
  threshold, so the "after tuning" gain in round time is not a finding. The filter itself did get 2×
  faster (11.7 s → 5.8 s).
- **Gate share.** The prediction was A ≈ 30 % → B, C ≤ 5 %. A measured **57 %**. B met the bar at
  3.3 %. **C missed at 6.0 %**. D is 5.03 %, just above the line, so it does not meet ≤ 5 %.
- **Why A was 57 %, not 30 %.** Lean's seconds were similar to the brief's baseline (634 s vs
  353 + 206 = 559 s). Sampling was ≈ 3× faster on this A40 pod (≈ 400 s vs ≈ 1,160 s). The executor
  attributed the difference to a co-tenant pod; the reviewer says **the cause is not measured**.
- With the filter on, Lean sees ≈ 14 % of distinct texts: it removed 122,973 of 122,973 rejects in B.
  Sampling (≈ 207 s) and fine-tuning (38 s) now dominate the round.

**Worker sizing** (A40, 7.65-CPU quota, 96 visible cores; 30,000 texts; texts/s, n = 1–2 per cell):

| Lean threads | 3 workers | 5 | 7 (quota) | 8 | 10 | 14 |
|---|---|---|---|---|---|---|
| default (≈ 96 threads) | 618, 290 | 549 | 298, 293 | 328 | 263 | 243 |
| `-j 1` | 358 | 684 | 745, 871 | 823, 865 | – | 466 |
| `-j 2` | 625 | 755 | 872, 776 | – | 633 | – |

- **The pre-registered "quota workers ≥ 2× three workers" was falsified** with Lean's default threads:
  about 0.5×. The run reported this as a miss.
- It holds only as 7 vs 3 workers at `-j 1` (2.1–2.4×). Against the true old configuration it is
  1.2–3.0×, which is unresolved at this noise: the 3-worker cell itself varies 2× between repeats.
- The fix is **`lean -j 1`** at 7 workers, 2.5–3.0× the default-thread throughput, with a stated
  mechanism: 96 threads per process on a 7.65-CPU quota. The reviewer judges it well supported. Most of
  the practical gain is `-j 1` plus the filter, not the worker count.

**Sampling settings.**
- Truncation at `max_new` 512 was 0.032 % (A, B) and 0.033 % (C, D), under the 0.1 % policy line.
- **Unreported by the executor:** the C1 corpus has strata above 0.1 %. Cap-horizon k14 is 0.28 %
  overall, with a worst stratum of 0.60 %; ds-generator g2 and full_seq are 0.23 %. This is harmless
  for a soundness corpus, since unparsed texts are not tested, but the policy asks for it to be
  reported.
- **Peak memory:** 16.76 GB at batch 4,096, `max_new` 512, on the 3.2 M `lean_seq` architecture. It
  was measured in the C1 corpus jobs, not in arm C as pre-registered (a minor deviation). At these
  settings, only one job fits on a 24 GB card. The ≈ 11 GB quoted in the new `--batch` default's
  comment is for `max_new` 288.

**Hard constraints** (the reviewer's check):
- `nd_verify` is tree-identical to `origin/main` and is not imported by any of the run's code.
- `artifacts/TEST_RUN_DONE` is unchanged. Gate 1's "mtime changed" FAIL is the known worktree false
  positive; the reviewer checked the blob.
- `train.py` is untouched; it belongs to the sibling run `fast-stage1`.
- The pre-registration was committed before any pod.
- The splits share 0 renaming classes across all six pairs.
- Both pods are deleted.

## Interpretation

- The gate is no longer the bottleneck of a T1 round. At batch 4,096 Lean's exposed share is ≈ 5–6 %,
  and the round is ≈ 4× faster on this pod. The filter makes nearly all of Lean's reject load
  disappear, and it has shown no false reject in 1.3 M texts plus an independent adversarial set.
- The ≈ 4× is n = 1 per arm on one A40. The old round's 57 % Lean share was specific to this pod's fast
  sampling. On the brief's baseline hardware the old share was ≈ 30 %, so the speed-up there may be
  smaller. This is not measured.

## Limitations and next step

- **Soundness is empirical.** It depends on the strict grammar staying the filter's domain and on
  Lean 4.34 semantics. The reviewer recommends one `LEAN_PREFILTER=shadow` job in every future run
  that changes `lean_tok`, the Lean version or the prompt format.
- The model corpus has 0 accepted `And.elim` / `Or.elim` field-notation proofs. If a later model learns
  them, the first shadow run is their real test.
- Round timing is n = 1 per arm on one pod. **Before the ≈ 4× is used as a planning number**, repeat
  A and C once on a second A40 and on a 3090. This is the reviewer's next measurement.
- Peak memory at batch 4,096 and `max_new` 512 is 16.8 GB: plan one such job per 24 GB card, not two.
- `lean_judge`'s relabel fallback and `grpo.py` are not on the new path.
- This branch is not merged into the fork's `dan`. Runs that should get the faster gate need it
  merged first.
