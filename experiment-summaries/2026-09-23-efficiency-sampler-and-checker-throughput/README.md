---
written_on: 2026-09-23
written_by: agent:claude
---

# efficiency: making sampling and Lean checking cheap, measured before and after

Status: completed 2026-09-23 17:05 UTC (executor session, one RTX 3090 pod, **$0.77** of a $15
budget / 10 pod-hour ceiling) and independently reviewed 2026-09-23 ≈ 17:33 UTC. Review verdict:
**no hard-constraint violation, no quarantine**; everything the reviewer could re-derive
re-derives, from the raw token dumps upward, with code that shares nothing with the run's
(own normaliser, own decoder, own Lean gate driver, own least-squares fit). One sentence, in two
places, must be reworded — the headline attaches "identical accepted set" to a comparison it was
not measured on. Numbers and wording below are the reviewer's. Run id `efficiency`.

Review:
[`review_efficiency.md`](https://github.com/dan-pandori/nd-takehome/blob/45c7a6e/review_efficiency.md)
(commits `03c0e6f` phase-1 blind recount, `45c7a6e` phase-2 comparison and verdict; reviewer code
`~/review/efficiency/rv/r1_targets.py` … `r10_sizes.py`, on the reviewer's host only). Executor
write-up:
[`run_efficiency.md`](https://github.com/dan-pandori/nd-takehome/blob/45c7a6e/run_efficiency.md),
numbers [`numbers.md` §1–10](https://github.com/dan-pandori/nd-takehome/blob/45c7a6e/numbers.md),
log `log.md` § efficiency, figure `figures/efficiency_before_after.png`, on branch
`dan_efficiency` of `dan-pandori/nd-takehome` (the executor's files were **not** edited after the
review; read the review first). Pre-registration:
[`preregistration/efficiency.md`](https://github.com/dan-pandori/nd-takehome/blob/45c7a6e/preregistration/efficiency.md).
Brief: `BRIEF_EFFICIENCY.md`. Falsified-premise escalation: `QUESTIONS.md` § 2026-09-23 (commit
`ef26f5b`, 15:51 UTC).

Question: sampling is the dominant cost of every run in this project. The brief's premise, from
`ds-composition`, was that **97 % of base-model samples on 7–12-line Lean targets never emit
`<eos>`**, so a decode batch runs its full budget for nearly every row and the KV cache reaches
16–23 GB. Why does the terminator not fire, and how much sampler and checker time can be
recovered **without changing which proofs are accepted**? Nothing is trained in this run.

## Configuration

- **Code and data revision.** `dan-pandori/nd-takehome`, branch `dan_efficiency` (worktree from
  `origin/dan_lean_only`). Pre-registration `57bcb1c` at **15:30:55 UTC**, one commit, never
  amended; pod `ef-1` created 15:31:50 UTC (`~/pods.log`); first measurement artefact
  `artifacts/ef/base_orig.json` stamped 15:41:52 UTC — gate 0 holds (reviewer). Final executor
  commit `ec9e70a`.
- **Hard constraints (all pass, reviewer).** `nd_verify/` sha256-identical in `origin/main`,
  `HEAD` and the worktree; `artifacts/TEST_RUN_DONE` untouched (no test-file run; this run does
  not touch the test files); no evaluation file read anywhere in the training or sampling code;
  `data/train.jsonl.gz` 154,990 rows with `max(n_lines) = 6`. `lean_check.py`'s only change is a
  docstring; `lean_gate.py`'s is a memoisation the reviewer verified semantics-preserving by
  inspection and by rerun.
- **Fixed workload, identical in every measurement.** Checkpoint `stage1_full_seq_s0.pt` — the
  `lean_seq` Stage-1 model of run `lean-only` (cap 6, 4 layers, d 256, 3,214,336 parameters),
  pulled from `hf://buckets/dan-pandori/nd-rl/lean-only/ckpts/lo/` and **never retrained here**.
  Targets: 200 records of `data/ladder/transfer.jsonl` with `L_true` 7–12 (2,262 of 2,285 qualify
  — the reviewer's own rebuild matches), stratified in pool proportion (27 / 27 / 88 / 40 / 9 / 9
  at `L_true` 7 / 8 / 9 / 10 / 11 / 12), written once to `artifacts/ef/targets200.jsonl` and
  reused verbatim. k = 256 → **51,200 samples**, T 0.8, `max_new` 512, batch 512. Lean gate:
  `lean_check` at 64 workers, chunk 300, Lean 4.34.0, with `nd_verify` beside it.
- **Splits disjoint by renaming class** (reviewer's own `canon_key`, recomputed from scratch):
  200 targets × train 154,990 = **0** classes in common; × `heldout.jsonl` = 0; × all three RL
  target pools = 0; full transfer pool × train = 0; heldout × train = 0.
- **Commands** (`pod/ef/{sync,setup,job,bench_all,cotenancy,cotenancy_all,stop}.sh`):
  - `ef_targets.py` → `artifacts/ef/targets200.jsonl`
  - `bench_sampler.py --tag <tag> [--path base|fast] [--rowrng 0|1] [--early eos|exact|goal] [--compact] [--batch B] [--max_new N]` — one configuration → one json + a raw token dump + an accepted file
  - `ef_diag.py` (terminator diagnosis), `ef_gate_compare.py` (correctness gate),
    `ef_lean_cost.py` / `ef_gate_chunk.py` (Lean-process cost and chunk sweep),
    `ef_smoke.py` / `ef_regress.py` (fidelity), `ef_ladder.py` / `ef_summary.py` / `ef_figure.py`
  - Recommended configuration: `sample.generate(..., path="fast", early="eos", compact=True, rowrng=True, batch=4096, max_new=288)`
- **What changed in the repository.** `sample.py` (a `fast` decode path beside the untouched
  `base` path; the pre-run file kept verbatim as `sample_base_orig.py`, byte-identical to
  `<base>:sample.py` per the reviewer), `model.py` (memoised `rope_cache`, reachable as before via
  `ND_ROPE_MEMO=0`), `lean_gate.py` (canonicalise once per distinct text), `lean_check.py`
  (docstring only; defaults unchanged).
- **Minor deviation** (reviewer, no effect on any number): the pre-registration says
  `random.Random(0).sample` inside each `L_true` bin; `ef_targets.py` uses `random.Random(0 + L)`.
  Still deterministic and seed-0-derived.
- **Pods and cost.** `ef-1`, RTX 3090 24 GB $0.50/h, created 15:31:50 UTC (`~/pods.log`), deleted
  17:03 — **1.5314 pod-hours = $0.77** (`podbudget efficiency`) of a $15 budget and a 10 h
  ceiling. No other pod. RunPod balance $221 at the start, $220.36 at review time. Wall clock
  15:25 → 17:05 UTC.
- **Artifacts.** `hf://buckets/dan-pandori/nd-rl/efficiency/{artifacts,ckpts,data}` — the 25
  `*_accept.jsonl.gz` files, five `*_tokens.npz` raw token dumps, `gate_*.jsonl`, and every
  `<tag>.json`. Bucket contents were not re-listed by the reviewer; the reviewer worked from the
  same files in the run repository.

## Observations

All values below are the reviewer's independent recount. They equal the executor's
`artifacts/ef/*.json` to the digit except where a difference is called out.

### The brief's premise does not hold for this checkpoint

| quantity | brief / `ds-composition` | measured here (reviewer) |
|---|---|---|
| rows that ever emit `<eos>` | ≈ 3 % | **99.988 %** (6 of 51,200 do not) |
| mean decoded tokens per sample | 400–512 | **143.39** (p95 236, median **138**) |
| peak KV / GPU memory | 16–23 GB | **2.123 GB** |
| cost per sample | 10.5 ms | **1.69 ms** |

Five independent measurements exclude all four of the brief's candidate causes: P(`<eos>`) at the
true end, teacher-forced, is **1.0000** on 400 held-out proofs and on the 42 accepted ladder
proofs; **5,000 / 5,000** training renderings end in `<eos>`; **0 %** of the 200 target prompts
exceed the training maximum (100 vs 130 tokens); 93 % of terminating rows emit `<eos>` immediately
after a top-level `exact n<k>`. The `lean_seq` Stage-1 model has no terminator problem. The
executor escalated this in `QUESTIONS.md` at 15:51 UTC — with the arithmetic suggesting the 97 %
figure came from a ≈ 6×-costlier, i.e. pretrained, model — stated the default (finish the
from-scratch work), and proceeded on it. **Open for Dan:** is `ds-composition`'s 97 % a
pretrained-model number? If so the `<eos>` problem is real but belongs to the pretrained path.

### Speed: 2.35× on the sampler, 2.16× end to end — and where it comes from

Baseline `base_orig` 592.47 samples/s (86.42 s) + 19.03 s gate = **105.45 s**; recommended
configuration `r_rec` 1,394.79 samples/s (36.71 s) + 12.18 s gate = **48.89 s**. Both ratios
reproduce (2.354× / 2.157×). The ladder, each row adding one change to the row above, at batch 512
unless stated:

| change | samples/s | this step |
|---|---:|---|
| the sampler as it was (`base_orig`) | 592.47 | — |
| + per-row deterministic RNG (`base_rr`) | 580.01 | 0.98× |
| + RoPE table memoised in `model.py` (`r_base`) | 621.16 | 1.07× |
| + fast decode loop (`r_eos_nc`) | 613.08 | 0.99× |
| + batch compaction (`r_eos`) | 619.62 | 1.01× |
| + `max_new` 512 → 288 (`r_eos_mn288`) | 651.35 | 1.05× |
| + batch 512 → 4,096 (`r_rec`) | 1,394.79 | **2.14×** |

**At a fixed batch of 512 the whole package is 1.10×; effectively all of the 2.35× is the batch
size.** That is affordable only because compaction and the cap hold peak memory to 10.98 GB — the
unchanged path at batch 4,096 needs 16.23 GB and *regresses* to 511.26 samples/s (17.79 M
row-steps against 10.88 M). The honest counterfactual the reviewer adds: the best configuration
reachable **without any code change** is the old path at batch 2,048, 748.99 samples/s, so the new
code is worth **1.86×** over that. Compaction alone is 1.011× (pre-registered E6, < 1.2×, holds)
while doing 1.42× less work (11.24 M → 7.91 M row-steps): at this model size a decode step costs
the same whatever fraction of the batch is live, because it is kernel-launch-bound. **That is the
durable finding for later runs — the lever is the batch size, and compaction plus the cap are what
make it usable.**

### Correctness gate: identical at a fixed batch, not at the shipped batch

The reviewer reproduced the identity gate twice — once from the accept files with an independent
start-index normaliser, once end to end from the raw token ids through an independent decode and
an independent invocation of `lean_check` at a tenth distinct worker/chunk setting.

| comparison (51,200 samples each) | accepted samples | distinct ND proofs | targets | identical? |
|---|---:|---:|---:|---|
| `base_rr` vs `r_eos`, `r_eos_nc`, `f_compact`, `f2_eos`, `f2_eos_nc`, `r_exact`, `r_goal`, `r_eos_mn288`, `r_base` (nine arms, **all at batch 512**) | 2,022 = 2,022 | 42 = 42 | 31 = 31 | **yes**, A∖B = B∖A = 0 |
| `base_rr` → `r_rec` (**batch 4,096**, the shipped arm) | 2,022 → 1,988 | 42 → **38** | 31 → **27** | **no** — 17 lost, 13 gained |
| `base_orig` → `base_rr` (RNG stream only, no code change) | 1,961 → 2,022 | 36 → 42 | 25 → 31 | no — 8 lost, 13 gained |

- **This is the one sentence the verdict requires reworded.** `run_efficiency.md`'s first line and
  `STATUS.md`'s DONE entry attach "on a provably identical set of accepted proofs (2,022 / 56 / 31,
  symmetric difference 0)" to the 592.5 → 1,394.8 comparison. That pairing is not measured: the
  2.35× runs `base_orig` → `r_rec`, whose accepted sets are 1,961 / 36 / 25 and 1,988 / 38 / 27.
  The reviewer's suggested shape: *"2.35× on the sampler and 2.16× end to end; the accepted set is
  provably identical for every change at a fixed batch (1.10× / 1.16×), and the remaining 2.13× is
  a batch-size change, which reshuffles the accepted set by as much as a re-draw of the RNG does."*
  `numbers.md` §3 already prints `r_rec`'s 1,988 / 27 and §6 scopes the gate correctly, so the
  artefacts and detailed tables are honest; the two summary sentences merge them.
- **The difference is not evidence of a bug.** A pure re-draw of the RNG (`base_orig` → `base_rr`)
  moves the same quantities by as much (36 → 42, 25 → 31), and 38 / 27 sits inside that range. But
  with one draw per configuration there is no n to say so with, and "identical accepted set" is not
  a statement that survives a batch-size change.
- **Two quantities share the name `accepted_distinct`** in the run's artefacts: `<tag>.json`
  reports distinct **ND** proofs per target (42 for `base_rr`), `gate_compare_*.json` distinct
  canonical **Lean** texts per target (56). Both are right and the reviewer reproduces both — the
  56 accepted Lean texts denote 42 distinct start-index-normalised ND proofs on 31 targets. The
  shared name is a trap.
- **Token-level fidelity is 99.97 %, not 100 %.** Row-for-row over the saved ids, `base_rr` →
  `r_eos` differs on **16 rows of 51,200** (bf16 kernel non-determinism under a different batch
  shape); `r_eos_mn288` on 51 (36 of them losing `<eos>` to the cap). The accepted-set comparison
  is still exact — the reviewer re-ran it end to end — but "the two runs draw the same tokens" and
  `ef_regress.py`'s "token streams identical" (128 samples, cannot see a 3-in-10,000 effect) are
  overstated.
- **`nd_verify` / Lean agreement: 275 counted proofs re-checked across five arms, 275 agree, 0
  disagreements** either way (the executor reported 56 / 56; the reviewer extends it).
- **Structural gap for the next run** (reviewer): `r_rec`, the one configuration that ships, was
  run with `--save_tokens 0`, so the gate could not be pointed at it even in principle. The rule
  that falls out: **save the token dump for the arm you are going to recommend.**

### The `max_new` cap, and the fixes that were dropped with numbers

- Over the 2,022 baseline-accepted samples: **max 255 decoded tokens**, p99 245, mean 172.5.
  **E5 holds.** A cap at 288 leaves 33 tokens of margin, truncates 40 of 51,200 rows, and none of
  them was ever accepted.
- **Both per-sequence early stops were dropped**, with numbers, and kept behind `early=`: the
  top-level-`exact` stop is **0.92×** (1 token saved per sample), the goal-reached stop **0.72×**
  (3.4 tokens). On a model that terminates, the stop automaton's per-step kernels cost far more
  than the tokens they save. Of the 1,542 rows where the goal stop's guessed `exact n<k>` differs
  from what the model would have written, **none is an accepted proof** — the design claim "a wrong
  guess costs a rejected sample, never a wrong accept" holds structurally and empirically at
  n = 51,200.
- **Worker-sized Lean chunks**: implemented, measured as a regression (gate 12.18 → 19.66 s),
  reverted. **Re-packing / continuous batching**: analysed, not built; headroom ≤ **1.87×** on the
  sampler and 1.54× end to end, at ≈ 4× the KV memory that is the exact resource capping the batch.

### The Lean gate

- One fix kept: canonicalise once per distinct text (51,196 texts → 7,620 distinct, ≈ 6.7×
  repetition on a coverage pass). Gate wall **19.03 → 12.18 s**; the like-for-like re-run on the
  same 7,568 texts gives 12.95 s, i.e. 1.47×, and the write-up shows both.
- **E10 is falsified by 3.6×.** The reviewer's own least squares on the chunk sweep:
  **2.008 s fixed per `lean` process + 0.01185 s per theorem**, so **36.1 %** of gate process time
  at chunk 300 is startup (the run's fit: 2.02 / 0.01201 / 35.9 %). The pre-registered *reason* for
  dropping a persistent Lean server does not hold.
- What rescues the decision is a different measurement: Lean processes contend. Chunk 300 with ≈ 26
  concurrent processes is the wall-clock optimum (941.7 texts/s); driving the chunk to 120 so all
  64 workers get a process costs **4.8× the process time** (851.8 s vs 178.8 s) and 1.8× the wall.
  The 36 % is not recoverable by removing startups. **Accepted count is 56 in all nine settings of
  the sweep, and 56 again under the reviewer's independent tenth setting** — chunking cannot change
  a verdict. A true persistent server is worth ≈ 5 s of a 49 s round; not built, and the number is
  recorded so a later brief can decide.

### Co-tenancy

Each job is 100 targets × k 256 = 25,600 samples, batch 1,024, with its Lean gate.

| concurrent jobs | per-job sampler samples/s | aggregate sampling | aggregate incl. Lean gate |
|---:|---:|---:|---:|
| 1 | 1,022.4 | 1,022.4 | 689.8 |
| 2 | 608.7 | 1,198.5 (+17 %) | 894.8 |
| 3 | 426.9 | 1,257.0 (+5 %) | 961.4 |
| 4 | 309.9 | 1,214.7 (−3 %) | 964.2 |

**Per-job throughput never improves** — it falls monotonically from the first co-tenant, so E11 as
worded ("stops improving at 2") is ill-posed. Aggregate sampling throughput peaks at **3** and
regresses at 4; aggregate including the CPU-bound gate is flat from 3. The measured answer is **3
by aggregate, 2 if a job's own latency is weighted at all**, and a recommendation must say which
column it is about. The write-up's "one job at batch 4,096 does 1,395 samples/s, more than four
co-tenants do between them (964)" is not like-for-like (sampler-only against end-to-end): the
direction is right, the margin is **+8.6 % end to end** or **+15 % sampler-only**, not ≈ 45 %.

### What the counted proofs look like

| arm | accepted proofs | Lean term size min/mean/max | ND lines min/mean/max | proofs shorter than their target's `L_true` |
|---|---:|---|---|---:|
| `base_orig` | 51 | 3 / 4.392 / 7 | 7 / 8.353 / 10 | 0 |
| `base_rr`, `r_goal` | 56 | 3 / 4.500 / 7 | 7 / 8.429 / 10 | 0 |

Every solved target has `L_true` ∈ [7, 9]; **nothing at `L_true` 10–12 was solved by any arm**, and
no proof is shorter than its target's label.

### Pre-registered expectations, scored against the reviewer's recount

| | prediction | measured (reviewer) | |
|---|---|---|---|
| E1 | baseline `<eos>` fraction < 15 % | **99.988 %** | **falsified** |
| E2 | `<eos>` > 70 % on held-out ≤ 6-line prompts | teacher-forced P(`<eos>`) at the true end = 1.00 on 400 prompts; the *sampled fraction* E2 names was not measured, and E1 makes it moot | not derivable as worded (the write-up's "E2 trivially" is a different statistic) |
| E3 | mean > 450 decoded tokens, p95 = 512 | 143.39 mean, p95 236 | **falsified** |
| E4 | > 80 % of never-`<eos>` rows at depth 0 in an unfinished `have` chain | only 4–6 rows never emit `<eos>`; n too small | untestable |
| E5 | no accepted sample over 400 decoded tokens | max 255 | **holds** |
| E6 | compaction alone < 1.2× | 1.011× | **holds** |
| E7 | terminator stop + compaction: tokens ÷ 2, samples/s × 2, memory ÷ 1.5 | 1.00× tokens, 1.00× samples/s at fixed batch; memory **×5.2** in the shipped configuration (falls 1.45× at batch 512) | **falsified** (with the premise) |
| E8 | ≥ 1.5× as many accepted samples, 0 targets lost | accepted set identical, not enlarged | **falsified** (with the premise) |
| E9 | identical accepted set on ≥ 50,000 samples for the pure-speed path | identical on 51,200 samples for nine arms at batch 512; **not** identical for the shipped batch-4,096 arm | **holds at fixed batch, fails at the shipped batch** |
| E10 | fixed Lean process cost < 10 % of gate process time at chunk 300 | 36.1 % | **falsified** |
| E11 | per-job throughput stops improving at 2 concurrent jobs | per-job never improves; aggregate peaks at 3 | **not as worded** |

Five of eleven are falsified, four of them because the brief's premise about `<eos>` is false for
this model. The reviewer's reading: *"that is a pre-registration doing its job"* — the falsified
premise was escalated in `QUESTIONS.md` with a stated default **before** the fixes it invalidated
were built, every miss is called a miss in `run_efficiency.md`, `numbers.md` and `log.md`, and the
two early stops that did not pay were dropped with their numbers and kept behind a flag, as the
brief asked. No stop rule was approached (the 1.3× floor, the 8 pod-hour stop).

### Smaller rewordings the verdict requires

- "the two runs draw the same tokens … an exact comparison" → 16 rows in 51,200 differ (99.97 %);
  the accepted-set comparison is still exact.
- `ef_regress.py`'s "identical token streams" → true of its 128 samples; it cannot see a
  3-in-10,000 divergence. A good idea at the wrong n.
- Co-tenancy "1,395 vs 964" → not like-for-like; +8.6 % end to end, +15 % sampler-only.
- The §8 "per job samples/s" column (689.8 / 447.4 / 320.5 / 241.1) is aggregate ÷ N **including
  the gate**; the per-job *sampler* figures in the same files are 1,022 / 609 / 427 / 310. Both
  defensible, the heading does not say which.
- Baseline median decoded tokens **138**, not 121 (`numbers.md` line 678 and `log.md` 15:42;
  transcription error, no downstream use).
- `run_efficiency.md` is 470 words against the brief's ≤ 400 (18 % over); one figure, as asked.

## Limitations and next step

- **Every ratio is a single measurement on one pod.** §8b's one repeat of the baseline gives
  609.30 vs 592.47 samples/s, and the write-up asks for a ±3 % band — but that band comes from
  n = 2. The claims it protects (2.13×, 2.35×, 0.72×) are far outside it; the 1.01×, 1.05× and
  1.07× steps are at its edge, which the write-up says.
- **The identity gate does not cover the configuration that ships.** Nine arms at batch 512 are
  identical on 51,200 samples; batch 4,096 is not, and there is one draw per configuration, so the
  scatter cannot be separated from a defect by this run's data alone (the RNG-only comparison says
  it is the right size for noise).
- **Timings could not be re-measured** — the pod was deleted before the review, so every ratio was
  re-derived from the recorded `sample_wall_s`, `gate_wall_s`, `rowsteps` and `model_steps`, not
  re-run. Bucket contents were not re-listed.
- **The result is about this model and this workload**: a 3.2 M-parameter from-scratch `lean_seq`
  model at batch 512–8,192 on one RTX 3090. "A decode step costs the same whatever fraction of the
  batch is live" is a small-model, kernel-launch-bound statement and should not be carried to a
  pretrained model without re-measuring — which is exactly where the brief's `<eos>` premise
  probably came from.
- Reviewer artefacts (`~/review/efficiency/rv/`) live on the reviewer's host, not in the
  repository; the reviewer's Lean re-gate used `LEAN_CHECK_WORKERS=2 / CHUNK=400` on the VPS.

**Next measurement (reviewer), ≈ $0.05 and about six minutes of a $0.50/h pod:** three runs of
`r_rec` and three of `base_rr` at seeds 0 / 1 / 2, with `--save_tokens 1` on at least one `r_rec`,
plus one `base`-path arm at batch 4,096 with the gate on. That turns "identical" into either a
defensible identity at the shipped batch or a stated scatter ("targets solved 27–31, distinct
proofs 36–42, across seeds at either batch"), which is what the accepted-set gate is actually for.
Until then the honest form of the headline is the reviewer's suggested shape above.

**Also open:** whether `ds-composition`'s 97 %-never-`<eos>` figure is a pretrained-model number
(`QUESTIONS.md`, 2026-09-23). If it is, the terminator fixes this run built and kept behind
`early=` are the inheritance for whoever hits the real case, and the pretrained sampler needs its
own before/after.
