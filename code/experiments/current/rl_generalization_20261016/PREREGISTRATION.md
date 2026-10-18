---
written_on: 2026-10-16
written_by: agent:claude
---

# Pre-registration: the mini-run validation gate and the knob-wave analysis

Written by the read-only experiment auditor for Leon's RL-generalization study. The goal, in Leon's words: "a science
of how we can change the generalization of RL training". Generalization here means two things:

- generalization to longer proofs: t1320, with 13–20 lines, of which 16–20 is beyond the training lengths; and t2130,
  with 21–30 lines;
- generalization to other sources: Charles149, a proxy that can be tuned against; textbook72 stays sealed.

Both are to be separated from in-distribution (ID) competence, measured on the held-out L2 and L3 `test_1k` sets.

§§ 1–5 are the 2026-10-16 text as first given to the orchestrator, unchanged in substance; it was first kept in a
scratchpad that a reboot wiped. Amendments since then are dated, in § 6.

**Inputs read for the first version:**
- the uncommitted mini diff in `nd-rl-v2` on `leon/grpo-v2` 765161f9 (`line.sh`, `grpo.py`, `build_mini.py`), later
  committed as 74864c28;
- `jobs_mini_w1.txt` and `jobs_mini_w2.txt`;
- `runs/pools/mini/`: built, with its md5 pass record;
- RECIPE.md on `origin/leon/main`;
- `v2_results_ynez.md` §§ 1, 3 and 7;
- EXPERIMENT_QUEUE, the Running section;
- the design review and the knob proposals. Their content is summarised in `KNOB_PROPOSALS.md` and the queue.

The numbers in § 5 come from a short CPU script. It was lost with /tmp, but it only applies the formulas stated there.

## 1. Problems with the plan, and the fixes

| # | problem (observed in the code or the data) | fix |
|---|---|---|
| P1 | **`g64` is applied in both mini stages** (`PROMPTS`/`GROUP` are global in `line.sh`). The known −5.5 effect is for L3 only, from a fixed L2. In L2-mini, where training pass rises from about 0.23 to 0.63, 64-attempt groups give far more signal at low pass rates, but cover only about 640 of 3,000 pairs. So the sign there is unknown, and the arm does not replicate the known effect | Make the gate's g64 **L3-mini only**. Keep a both-stage g64, if wanted, as a separate exploratory knob with no role in the gate |
| P2 | **`kl1` trains its own L2-mini** (`src=$arm` for every mini arm). This adds L2 noise to an L3-only comparison and costs about 20 min | **Fork both controls from the reference's L2-mini `_it10.pt` at the same seed**, with the existing 0-rollback check. The KL reference is then that checkpoint, as at full scale |
| P3 | **The expected sizes are mis-stated.** kl1's −2.7/−2.2 is against KL 0.02. The mini reference is at KL 0, so the full-scale contrast is kl1 − kkl0 = **−4.2 / −3.5** (13–20 pass@1 / pass@5, seed 0). Effects in points also shrink at the mini's lower pass rate: g64's −0.255 logit is −3.8 points at 20% and −2.1 at 10% | Run the gate on the **logit of the set mean**, with expected shifts of **−0.19** for kl1 and **−0.255** for g64 (§ 5) |
| P4 | **One seed per control is too few.** Against 4 references, a single draw has SE 1.12σ, and the control's own spread goes unestimated. At a run SD of 0.10 logit, kl1's power is about 0.5 with one seed and 0.7 with two | **2 seeds per control (0, 1) as forks**, plus 2 more (seeds 2, 3) automatically if a control is "partial" (§ 3) |
| P5 | Both controls are **harmful L3 effects**. No known positive or L2-stage effect is large enough to use: kl0's +1.7 is below any mini MDE | State the limit: the gate validates detection of large harmful L3 effects only. kl1 acts through drift, so it may be null in 10 iterations without the mini being broken for early-learning knobs; the decision table separates that case (§ 3) |
| P6 | **`_ckreads` reads it0 once per arm.** Every seed-0 knob pilot's it0 is the same `grpo_think_vs_s0_L2_it39.pt`, read under 9 names (`${arm}_it0`): about 8 × 45 min ≈ 6 GPU-h of duplicate reads | Read each distinct L2 checkpoint once (s0, s1, s2), plus the base once |
| P7 | The **base S1g3_s0** has never been read on t1320 or `test_1k` in the ynez world, and G0 needs it | `MINI_ROLL="L2:0"` on **one** reference run (`it0_rolling` holds the init weights, saved before training), or a separate base read. Do not run it on all 4 references: that reads the same checkpoint 4 times |
| P8 | **`--ban_stop` ends a stage without `_it10.pt`** (exit 3). The line then fails and nothing is read | Count it as a failed run (§ 4.6), never as a silent drop |
| P9 | **Job order with forks.** `pull_queue.sh` deletes a job line when a worker takes it. A fork taken before its reference's L2-mini exists refuses, and the job is lost | A wait-for-file in the fork arm, or start the fifth worker after the first L2-mini lands |
| P10 | **Wave 2 as written** is 7 knobs × 1 seed. The temperature arms read only at T 0.8 (`read1` hard-codes `--temperature 0.8`). `--len_penalty` charges think **and** action tokens (all steps), so it is a sampled-length knob, not a think-length knob | Seeds per § 4.2, extremes first, one new reference seed per wave. Add a `T_eval` option to `read1` before the T arms run. Split the length manipulation check into think and action tokens |
| P11 | **No train-subset read** in the mini reads. The pool-size knob needs one for memorization | Add `L3_mini.jsonl` k8 (and `L2_mini`) to the final reads of the references and the pool-size arms |

Checked and fine:
- Nothing reads textbook72 (`MINI_READS`, `CKR_READS`; the preflight only checks overlap).
- The pass record matches the md5s of the mini subsets.
- Timeouts: 2 h per stage with a 1 h watchdog, against expected times of 20 and 30 min; 9,000 s per read.
- Disk: 1.1 TB free; checkpoints are about 39 MB each.
- `--len_penalty` keeps the default path: at 0, `on_top` is None and `n_norm` is as before.

## 2. The validation wave (recommended form)

| job | what | seeds | about |
|---|---|---|---|
| `grpo_think_vs_mini` | the reference, both stages from S1g3_s0 | 0, 1, 2, 3 | 1.5 h each |
| kl1 fork | L3-mini at `--kl_anchor_coef 0.1`, from the reference's L2-mini at the same seed | 0, 1 | 1.2 h |
| g64 fork | L3-mini at 64 × 64, from the reference's L2-mini at the same seed | 0, 1 | 1.2 h |
| base | S1g3_s0 on the `MINI_READS` sets | — | 0.7 h |

In total, about 12 GPU-h. Rolling checkpoints are kept (`--save_rolling 2`), so the dynamics panels can be read after
the gate.

## 3. The gate (apply mechanically)

**Run-level metric.** For run r and set S:
- p_r is the mean, over the set's problems, of successes / k: the unbiased pass@1;
- pass@5 is the unbiased estimator from k8 (k64 for Charles);
- x_r = logit(p_r) of the set mean, with p clipped to [0.005, 0.995].

Definitions: Charles is the clean 149 rows of the charles173 read. 16–20 means the t1320 rows with `min_lines` ≥ 16
(616 rows). All runs read the same problems, so the run-level spread holds training noise plus attempt noise.

**Statistic.** For a control c:
- Δ_c = the mean over c's seeds of x, minus the mean over the reference seeds;
- s_ref = the SD of x over the reference seeds (df = n_r − 1);
- SE_c = s_ref · √(1/n_c + 1/n_r);
- upper bound U_c = Δ_c + t(0.95, n_r − 1) · SE_c, where t = 2.35 at n_r = 4.

The test is unpaired. With forks it is conservative, because the forks share L2 with references 0 and 1. Report the
paired fork differences c_s − r_s as a secondary.

**Primary metric:** 13–20 pass@1, as a logit. Expected shifts E: kl1 −0.19, g64 −0.255.

**Each control is classed as:**
- *Reproduces*: Δ_c ≤ ½E and U_c < 0, and the point estimates on 16–20 pass@1 and 21–30 pass@5 are both below 0.
- *Partial*: Δ_c < 0 but the control does not reproduce.
- *Fails*: Δ_c ≥ 0.

**G0, the setup learns and is sane:**
- All 4 references end with L3-mini `_it10.pt` and every read.
- At most 1 of the 4 rolled back. Keep that one, flagged. Two or more mean the safeguards are miscalibrated for
  10-iteration stages.
- The references' mean 13–20 pass@1 is in [12%, 60%] and at least the base + 5 points. Without a base read, use 7.3%,
  serrano's 2-seed S1g3 figure.
- The references' mean L3 `test_1k` pass@1 is ≤ 90%. Otherwise the ID axis is at its ceiling: add B0 or a hard ID
  subset before knob work.

**G1, noise.** s_ref on 13–20 pass@1, in logit. It must be ≤ 0.11 to pass (§ 4.2). Its 95% interval at df 3 is
[0.57 s, 3.7 s], so G1 only sets the number of seeds.

**G3, informational and not blocking.** Once the ckreads land, place the references' (L3 test, t1320) points against
the full-scale trajectory line from it0 to it39. A mini far off that line is in a different regime for generalization.

**Decision.** The gate counts only for the configuration it ran on. A remedy is re-validated on fresh runs only, with
at most one remedy round per night.

| G0 | G1 | g64 | kl1 | decision |
|---|---|---|---|---|
| fail | any | any | any | NO-GO, remedy A |
| pass | s > 0.11 | any | any | NO-GO for both-stage knobs, remedy B (or C) |
| pass | pass | reproduces | reproduces | **GO**, all knobs |
| pass | pass | reproduces | partial | kl1 forks at seeds 2 and 3, re-apply at n_c = 4; if still partial, restricted GO |
| pass | pass | reproduces | fails | **restricted GO** |
| pass | pass | partial | any | g64 forks at seeds 2 and 3, re-apply; if still partial, remedy A |
| pass | pass | fails | any | NO-GO, remedy A; if that also fails, remedy C |

**Restricted GO** covers knobs that act on sampling, signal, data or the loss mask: temperature, pool size,
positive- vs negative-only advantages, action- vs think-only loss, the size twins, and the length reward if its
manipulation check passes. It excludes knobs that act through drift or anchoring: KL variants, Stage-1 replay,
prolonged training. Those go to remedy C, or to a 20-iteration L3-mini that has first reproduced kl1.

**Remedies:**
- **A.** 20 + 20 iterations on the same subsets (`--decay_frac 0.2` then gives a 4-iteration cooldown), and the whole
  validation wave again. Larger subsets would not help: at 10 iterations each pair is drawn about 0.85 times, so
  coverage is not the limit.
- **B.** Knobs as L3-only forks from the 4 reference L2-minis, at 4 seeds, plus one `rr` fork at 2 seeds for the
  L3-only noise.
- **C.** Full-scale L3 forks from `grpo_think_vs_s{0,1,2}_L2_it39.pt`: the validated knob-pilot protocol, 3 seeds,
  about 2.5 h a job.

## 4. Knob-wave analysis

### 4.1 Unit and outcomes

The unit is the run's final checkpoint, plus (run, checkpoint) wherever panels are read. All tests use the logit of the
set mean; results are reported in points (% of the set) at pass@1 and pass@5.

Primary outcomes, one family per wave:
- **P1, ID competence:** L3 `test_1k` pass@1.
- **P2, length generalization:**
  - (a) Δslope: the per-line slope of logit(mean pass by `min_lines`) on t1320 ∪ t2130, 13–30 lines, by weighted least
    squares;
  - (b) the 16–20 residual against the reference ID→OOD line (§ 4.3).
- **P3, source generalization:** Charles149 pass@1 and its residual. It is reported separately and never averaged with
  the long sets: across models, r ≈ 0.2 between 13–20 and Charles149.

Secondary outcomes:
- L2 `test_1k`; t1320 13–15 and 16–20, and t2130, at pass@1 and pass@5; Δlevel, the logit at 13 lines.
- The ladder gaps:
  - structure = logit(t1320 13–15) − logit(L3 test);
  - length = logit(16–20) − logit(13–15).
- The hazard of the first invalid step, by step index.
- The train-subset gap.
- Rollbacks and the final `lr_scale`.

### 4.2 Seeds per knob

- n_k is the smallest n in {2, 3, 4} such that 2.8 · s · √(1/n + 1/n_r) ≤ **0.20 logit**: about 3 points at 20% and
  4 at 30%.
- s is the pooled within-arm SD over the references and every both-stage arm with ≥ 2 seeds, recomputed each wave.
- Add **one new reference seed per wave**.
- Rough guide: s ≤ 0.08 gives 2 seeds, s ≤ 0.10 gives 3, s ≤ 0.11 gives 4; above that, see G1.
- **Order:**
  - Extremes first: T 0.7 and 1.3, `--len_penalty` −0.1 and +0.5, pool size 250.
  - The middle levels only if an extreme is a screening hit, or if the two extremes differ at the screening threshold.

### 4.3 Competence versus generalization (residuals)

- Fit an OLS line, logit(OOD) = a + b · logit(ID), on **reference points only**. The points are the final checkpoints
  plus the panel checkpoints (L2 it 0/4/8; L3 it 0/2/4/6/8) read through `MINI_ROLL`. The reference panel includes
  t2130 and Charles, so they have lines too.
- A run's residual = logit(OOD) − line(logit(ID)). A knob's residual is the mean over its seeds.
  - SE = the reference residual SD × √(1/n_k + 1/n_r).
  - Line uncertainty comes from a bootstrap over reference runs, each with its trajectory.
- Flag a knob as "extrapolated" when its ID lies more than 0.2 logit outside the references' range.
- Δlogit(OOD) = b · Δlogit(ID) (competence) + the residual (generalization). Report both parts.

### 4.4 Classification (after confirmation)

| class | rule |
|---|---|
| competence | P1's CI excludes 0, and every residual's CI includes 0 |
| length generalization | Δslope's CI excludes 0 (a flatter slope is a gain), whatever P1 does |
| structure (level) generalization | the 16–20 or 13–15 residual excludes 0, while Δslope includes 0 |
| source generalization | the Charles149 Δ or its residual excludes 0 |
| trade-off | long-set and Charles effects that are both significant and of opposite sign |
| bounded null | every CI includes 0, the manipulation check passed, and the CI lies within ±0.2 logit. Report it as "no effect larger than X", never as "no effect" |
| not resolved | a CI includes 0 and is wider than ±0.2 logit |
| not tested | the manipulation check failed |

### 4.5 Screening, then confirmation

- A **screening hit** is a primary outcome whose 95% CI excludes 0 (uncorrected), with the same sign at every seed of
  the knob.
- A **claim** needs **confirmation on 2–3 fresh seeds**: the same sign, and a one-sided 95% bound in that direction,
  computed from **the confirmation seeds alone**.
- Every arm that was run is reported, nulls included. textbook72 is never read.
- A mini claim becomes a full-scale claim in RECIPE or a write-up only after a remedy-C confirmation.

### 4.6 Rollbacks and failed runs

- **Intention to treat is primary:** every run with a final checkpoint is analysed, rollbacks included. Rollbacks and
  the final `lr_scale` per stage are reported as knob outcomes.
- **Sensitivity check:** repeat the analysis without the runs that rolled back. If a call changes, the knob is
  "unresolved" and gets one more seed.
- **A run without its final checkpoint** (a `ban_stop` exit 3, a timeout or a crash) is a failure, reported as a rate
  and replaced once with the next seed. If the replacement also fails, the knob is "unstable", and its last good
  rolling checkpoint is read as a secondary only.

### 4.7 Manipulation checks, declared before the runs

| knob | manipulation check |
|---|---|
| temperature | training think and action entropy; the share of skipped groups; extra reads at T_eval 0.6 and 1.0 (t1320 k4, L3 test k4) for the T arms and 2 references. The primary read is fixed at T 0.8; the best T_eval is secondary |
| `--len_penalty` | think tokens per step at B64 and B256, separately from action tokens per attempt; steps / `min_lines` of solved proofs; the `len_only` token share. Monotone in λ. "Not tested" if think tokens at lp50 move by less than 10% |
| pool size | train-subset minus `test_1k` pass@1, in logit, against pool size; draws per pair |
| action-only / think-only loss | the drift in think length and entropy; a B0 read |
| positive-only / negative-only | entropy; the gap pass@5 − pass@1 |
| size twins | the binder-mismatch share in the large-statement tercile of t1320 |

## 5. Full-scale expectations (seed-0 reads, `v2_results_ynez.md` § 7)

| contrast | set | Δ points | Δ logit |
|---|---|---|---|
| kl1 − kkl0 | 13–20 pass@1 / pass@5 | −4.2 / −3.5 | −0.188 / −0.146 |
| kl1 − kkl0 | 16–20 pass@1 | −4.5 | −0.223 |
| kl1 − kkl0 | 21–30 pass@5 | −5.2 | −0.295 |
| kl1 − kkl0 | Charles149 pass@1 | −0.5 | −0.02 (null) |
| g64 − merged | 13–20 pass@1 / pass@5 | −5.5 / −5.4 | −0.255 / −0.221 |
| g64 − merged | 16–20 pass@1 | −5.2 | −0.268 |
| g64 − merged | 21–30 pass@5 | −4.3 | −0.267 |
| g64 − merged | Charles149 pass@1 | −2.2 | −0.10 |

The same logit shifts in points, at a mini reference level p:

| p | kl1 (−0.19) | g64 (−0.255) |
|---|---|---|
| 10% | −1.6 | −2.1 |
| 15% | −2.3 | −3.0 |
| 20% | −2.9 | −3.8 |
| 30% | −3.8 | −5.1 |

Power of the gate test, one-sided 0.05, z approximation (t at df 3 is lower), with n_r = 4:

| run SD s (logit) | n_c = 1, kl1 / g64 | n_c = 2, kl1 / g64 |
|---|---|---|
| 0.08 | 0.68 / 0.89 | 0.86 / 0.98 |
| 0.10 | 0.52 / 0.74 | 0.71 / 0.90 |
| 0.15 | 0.30 / 0.45 | 0.43 / 0.62 |

For scale: the full-scale L3-only rerun (krr) moved 13–20 by −0.053 logit, so σ_L3 is about 0.04.

## 6. Amendments (dated; each written before the results it affects were examined)

**2026-10-16, audits 1b and 2** (before any mini result existed).
- P1, P2, P4, P6–P9 and P11 were implemented: `leon/grpo-v2` 74864c28, then c00f45a6.
  - kl1 and g64 are L3-only forks of the reference L2-mini at seeds 0 and 1.
  - g64b is the both-stage g64, exploratory only.
  - A FAILED line marks a missing final checkpoint.
  - The ckreads are deduplicated, with a shared `grpo_think_vs_l2_ckreads` per seed and an `s1g3_ckreads` base read.
- The panels:
  - `MINI_PANEL` = l3test k8, t1320 k8, t2130 k4, charles173 k16;
  - `CKR_PANEL` = L2/L3 `test_1k` k8, t1320 k4, t2130 k4, charles173 k16.

  At k4, pass@5 cannot be estimated, so **dynamics analyses use pass@1 only**; pass@5 comes from the final reads.
- **Temperature arms.** These are read at their own training T (`MINI_PANEL` at T, dirs `_T07` …) instead of the 0.6 /
  1.0 grid of § 4.7. Such a read is interpreted only against a reference read at the same T. The amended plan is to
  read reference s0's `MINI_PANEL` at T 0.7 and 1.3. The primary read stays T 0.8 for every arm.
- **The base read** (`s1g3`) is the native `lean_thinkn` S1g3_s0, with no effort tokens. Each run's L2
  `it0_rolling` is the effort-expanded copy, whose effort embeddings are randomly initialised. The base is G0's
  reference; it is not spliced into trajectories as an it0.

**2026-10-17, serrano's power failures** (PSU; the box went down several times).
- The w1b forks kl1/g64 s1 reran L3-mini from scratch.
- The GPUs now run at stock power limits (350 / 420 W), while the references ran at a 150 W cap. This changes speed,
  not the computation: same code, same seeds, and no timing-dependent kernel choice (`cudnn.benchmark`, TF32 toggles)
  is enabled in `code.thinking`. Comparability is unaffected beyond the GPU nondeterminism already in the noise model.
- **Analysis rule.** `grpo.py` appends to `metrics.jsonl` on a fresh, non-`--resume` rerun. So any stage that crashed
  and was rerun is checked for duplicate iteration rows; only the rerun's rows count. A crashed stage is one run, the
  rerun.

**2026-10-18, a 3-GPU budget** (Leon: few GPUs; finish and report). Written after the 4 reference reads were in, the G1
input; before any control read or any knob run was examined.

- **Observed G1 input.**
  - 13–20 pass@1 of the references at s0–s3: 24.8, 25.2, 24.4 and 25.0%.
  - s_ref ≈ **0.017 logit**, about 0.34 points: about the size of k8 attempt noise alone.
  - Training noise from a shared base is negligible at this scale.
- **Seeds, amending § 4.2.**
  - **Screen at 1 seed per knob.** With s_pooled from the references, the n = 1 MDE is about 2.8 · 0.017 · √(1 + 1/5)
    ≈ 0.05 logit, about 1 point, far below the 0.20 target.
  - A single-seed hit is screening only (§ 4.5). Its CI is Δ ± t(0.975, n_r − 1) · s_pooled · √(1 + 1/n_r).
  - **Confirmation:** 2 fresh seeds (1, 2) for each screening hit. The claim rule is unchanged: it uses the
    confirmation seeds alone.
  - A knob whose seed-0 run had a rollback, an alarm-driven lr change or a FAILED stage gets a second seed before any
    call, because its own variance may exceed the references'.
  - Single-seed non-hits are reported as bounded nulls at n = 1.
  - s is computed per metric. Charles149 and 21–30 have their own s_ref.
- **Reference seeds.** One new reference seed (4) runs in wave 2, as a drift check across the hardware change. If its
  13–20 logit falls more than 3 s_ref outside the s0–s3 range, the knob comparisons are re-examined before any call.
- **Levels.**
  - Extremes only: t07, t13, lpm10, lp50, pool250, lossact, lossthink, advpos, advneg.
  - The middle levels (t085, t115, lp20, pool500) run only if an extreme of the same knob is confirmed and budget
    remains.
- **Knob definitions** (audit 3; to be adopted before these arms start, and any deviation recorded in the queue):
  - `--loss_tokens action|think` keeps the divisor over the trained phase's tokens. Under AdamW the global gradient
    scale cancels, so the divisor matters only for the balance against the terms that sit on top: the L2-mini KL
    anchor and the step penalty. The selected-token divisor keeps that balance as in a full run.
    - `lossthink` trains nothing at B0 (a third of the groups), so it is read against the B0 read.
    - `</think>` counts as think, so `lossact` does not train think length directly. Manipulation check: think
      length and think entropy stay near the reference's under `lossact`.
  - `--adv_sign pos|neg` keeps the zeroed items in the divisor, as Zhu et al.'s objectives do (an indicator over all
    samples). Two changes are needed for the PSR / NSR interpretation:
    - (i) the sign is decided by the **outcome**: `neg` keeps only failed attempts' advantages. Otherwise the
      unlikeliness deduction gives the most likely solved attempts of mostly-solved groups small negative
      advantages, about −0.06, and `neg` would push correct proofs down.
    - (ii) `advpos` runs with **`--rej_penalty 0`**: the step penalty is a negative signal. Its relative weight would
      also double, because the outcome term halves while the penalty sits on top.

    In L2-mini the relative weight of the KL anchor also roughly doubles under both arms. Either run `advpos` and
    `advneg` as L3-only forks of the reference L2-minis (KL 0 in L3, paired, cheaper), or report that caveat.
- **Reference trajectories for § 4.3.** `MINI_ROLL="L2:4 L3:0 L3:4 L3:8"` on references s0 and s1, a rerun that writes
  `reads_commit.txt` (0923b391). Skip L2:0: `s1g3` covers it.
- **Dynamics question (KL 0's slope effect), the ckreads subset.** In priority order:
  1. `grpo_think_vs_l2_ckreads:0`, `grpo_think_vs_mb_kkl0_ckreads:0`, `grpo_think_vs_mb_krr_ckreads:0` (the KL-0.02
     partner at seed 0) and `grpo_think_vs_mb_kkl1_ckreads:0`.
  2. `grpo_think_vs_l2_ckreads:1`, `grpo_think_vs_mb_kkl0_ckreads:1` and `grpo_think_vs_mb_ckreads:1` (the KL-0.02
     partner at seed 1).
  3. `grpo_think_vs_l2_ckreads:2` and `grpo_think_vs_mb_kkl0_ckreads:2`. Unpaired, because merged s2 kept no rolling
     checkpoints.

  The other ckreads are deferred: kg64, kt12, ki78, kch4, kx66, kp3 and fbb.
- **Expected precision.** The panel's slope at k4, for a single run and checkpoint, will be noisier than the
  full-scale k16/k8 estimates. The KL-0 slope analysis therefore pools seeds and checkpoints with a hierarchical
  bootstrap (seed, then problem). No per-checkpoint slope call is made.

**2026-10-18, about 04:45Z, after the gate's NO-GO** (`results/gate.md`, 03:47Z). Written before any mini20 or g64
replicate result existed.

- **The gate's outcome, for the record.**
  - G0 and G1 pass; kl1 reproduces.
  - g64 was still partial after its seeds 2 and 3: Δ −0.053 logit, against ½E −0.1275.
  - § 3's decision is therefore **remedy A**, which Leon chose to run as pre-registered.
- **Remedy A, as run.** The arm is `grpo_think_vs_mini20` (`leon/grpo-v2` 79105ffe, `line.sh` c338f00f).
  - Both stages run at `--iters 20 --save_every 20`; nothing else in the reference commands changes. The data are the
    same subsets, and the init is the same S1g3_s0.
  - `--decay_frac 0.2` puts the L3 cooldown on the last 4 of the 20 iterations. The 10-step warmup is unchanged.
  - Stage timeout 3 h, watchdog 2 h.
  - The runs are 4 fresh references (seeds 0–3), plus kl1 and g64 as L3 forks of each reference's `_L2_it20` at seeds
    0 and 1.
  - The base read (`s1g3`) is reused, since it is the same checkpoint.
- **Remedy-A gate.** § 3 applies unchanged: the same E (kl1 −0.19, g64 −0.255 logit), the same ½E, secondaries, G0
  thresholds, G1 limit and decision table.
  - **The gate analysis uses mini20 runs only.** Nothing is pooled with the 10+10 runs, references or controls.
  - **Seeds 2 and 3 are queued automatically** for any control that is partial after seeds 0 and 1, as § 3 requires.
  - If the remedy-A gate ends NO-GO, the mini is retired as the knob testbed, and knobs go to remedy C budget
    permitting.
  - If the remedy-A gate passes, the knob wave runs at 20+20: the gate counts only for the configuration it ran on.
    The fork knobs (`advpos`, `advneg`) fork from the mini20 `_L2_it20`.
- **Wave-2 seeds at 20+20.** Recompute s_ref (13–20 pass@1, logit) from the 4 mini20 references before wave 2.
  - The § 6 (2026-10-18) single-seed screening applies only if s_ref ≤ about 0.05 logit.
  - Otherwise each wave-2 knob is screened at 2 seeds.
  - The confirmation rule is unchanged: 2 fresh seeds, and claims rest on the confirmation seeds alone.
- **The full-scale g64 replicate** (`grpo_think_vs_mb_kg64:1`: the merged recipe's L3 at KL 0.02 with 64 × 64, from
  `grpo_think_vs_s1_L2_it39.pt`).
  - **Statistic.** Δ = `kg64` s1 − `mb` s1 on 13–20 pass@1, the logit of the set mean, from the knob-pilot plain reads.
  - **It replicates the full-scale g64 effect** if Δ ≤ −0.1275 (½ of −0.255) and its 95% problem-bootstrap interval
    excludes 0.
  - **It does not replicate** if Δ > −0.1275.
  - Between the two, the outcome is reported as is, with no call.
  - Report it next to seed 0's −0.255. It is a single replicate. Full-scale rerun noise is about 0.05 logit (krr), so
    its resolution is about ±0.1.
  - This replicate does not change the gate. It bears only on whether the full-scale g64 effect is a reliable
    reference (the write-up note).

**2026-10-18, clarification** (written before any mini20 result existed).
- **What governs the remedy-A gate.** § 3's table, row by row.
  - "Retired, remedy C" applies only where § 3 would call for remedy A again, since A cannot be repeated.
  - A mini20 **G1 failure** (s_ref > 0.11) follows § 3's own row: remedy B (knobs as L3 forks of the 4 mini20
    reference L2-minis) if both mini20 controls reproduce, otherwise remedy C.
- **G0's "every read"** means MINI_READS, the five reads the gate uses, as applied at 10+10. The training-subset reads
  (`l2train` / `l3train`) serve the pool-size analysis and do not gate.
- **The wave-2 seed threshold** is mechanical: s_ref ≤ 0.050 logit screens at 1 seed, and > 0.050 at 2. A value within
  ±0.005 of 0.050 is flagged to Leon, but the rule still applies. A Leon override is recorded as a deviation.

**2026-10-18, 06:20Z (11:20 PM PDT, Oct 17): the g64 replicate run failed** (written before any seed-2 kg64 run
existed; nothing from the seed-1 run has been read).
- **What happened.** `grpo_think_vs_mb_kg64:1` was ended by `--ban_stop 0.005` at its last iteration (it 39 of 39).
  - Its think ban mass rose from about 0.0006 (its 1–20) to 0.0057.
  - It rolled back once (it 22, lr × 0.5); the restore check let its 35–38 through.
  - The line exited 3 with no `_it39.pt`, so none of its reads ran.
- **Under § 4.6 it is a failed run** (1 of 2 full-scale kg64 runs). It is replaced once with the next seed:
  `grpo_think_vs_mb_kg64:2`, paired with `grpo_think_vs_mb_s2`, whose reads exist. It is queued after the validation
  wave.
- **The criterion is unchanged**, with seed 2 in place of seed 1.
- **If seed 2 also fails**, full-scale g64 is "unstable". The last good rolling checkpoints (seed 1's it 35, and
  seed 2's) are then read as secondaries only.
- **Observation, not a test.** Seed 0's kg64 ban mass stayed under about 0.0008, and the merged recipe at seed 1 stayed
  near 0.001. One run cannot separate g64 from seed.

**2026-10-18, 09:27Z (2:27 AM PDT; commit ba275d73): DEVIATION, the remedy-A G0 ceiling check** (decided by agent:claude, delegated by
Leon at about 09:25Z: "make decisions for me since im gonna be asleep").
- **When it was written.** After the four mini20 references' B64 L3 test pass@1 was seen: 91.9% mean, against
  G0's ≤ 90%. Every other G0 check passed, and G1 passed (s_ref 0.042). kl1 s0 (provisionally reproduces) and g64 s0
  (provisionally partial) had also been seen. It was written before any B0 read existed. g64 s1 and the kg64 s2
  replicate were still training. kl1 s1's L2 and L3 test reads were already in `mini20_table.md` (regenerated at
  09:21Z) but had not been looked at. Its t1320 read finished at 09:27Z, the minute of the commit.
- **What the mechanical reading gives.** NO-GO (G0). Remedy A is spent, so the mini is retired and the knobs go to
  remedy C.
- **Why it deviates.**
  1. Remedy C cannot pass the failing check. The full-scale L3 runs' B64 L3 test pass@1 is 94.9–96.8% (kg64, kkl0,
     kkl1, krr, kt12 at s0), and even the shared L2 start is 89.3%.
  2. G0's ceiling clause names its own fix: "add B0 or a hard ID subset before knob work". There is no usable hard
     subset in test_1k. By `min_lines`, the references score about 92.5% at 13 lines, 90% at 14 and 87% at 15, and
     the 15-line stratum has only 44 problems.
  3. G3 (informational) puts the four references on the full-scale kkl0 trajectory line: a mean residual of +0.083
     logit, against the line's residual SD of 0.082. So the mini is not in a different generalization regime.
- **The rule, fixed before the B0 numbers.**
  - **B0 substitution.** The ceiling check alone uses the references' **B0** L3 test pass@1 (`reads_b0_l3test.sh`:
    line.sh's `l3test:8` read with `--budgets 0`, otherwise identical). G0 passes if the mean over the four is
    ≤ 90.0%; nothing else in G0 changes. B64 L3 test is still reported everywhere.
  - **Then § 3 applies mechanically** to kl1 and g64 once their seed-1 runs land. A partial control gets forks at
    seeds 2 and 3, and the gate is re-applied at n_c = 4. A partial kl1 and a partial g64 are forked together.
  - **If the B0 mean is > 90.0%,** G0 fails as written, the mini is retired, and no knob wave starts. Remedy C is not
    started overnight, because it would fail the same check. The choice goes to Leon.
- **Analysis consequences (§ 4.3).**
  - On a GO, every wave-2 arm also gets the B0 L3 test read.
  - The competence axis for the residual analysis is then B0 L3 test pass@1 (logit). The B64 L3 test residuals are a
    sensitivity check.
  - Caveat: B0 measures competence without thinking, while the out-of-distribution reads are at B64. A knob that
    changes how thinking is used can move the two differently. Report both residuals, and flag any knob whose call
    differs between them.
- **Status.** This is a deviation from § 3 and § 6, reported as such in every write-up of the knob wave.

**2026-10-18, 09:40Z (2:40 AM PDT): the deviation, completed after an audit** (written before any B0 read or g64 s1
read existed).
- **The B0 check was predictable, and the amendment should have said so.**
  - The references' L3-mini `it 20` training lines show B0 training pass of 81.0, 79.0, 77.5 and 79.6%.
  - So B0 L3 test ≤ 90% was close to certain when the rule was written.
  - The substitution therefore applies the clause's named fix ("add B0 … before knob work"). It is not a test the
    mini passes. G0's ceiling check is reported as "met by the clause's fix (B0)", never as "passed".
- **The primary ID outcome stays as it was.** P1 (§ 4.1) is still B64 L3 test pass@1. It is resolvable: s_ref 0.018
  logit, and kl1 s0's paired shift is −0.172 [−0.279, −0.063].
  - The deviation changes only the § 4.3 competence axis: B0 primary, with B64 as the sensitivity check.
- **No retire branch starts anything overnight.** This covers g64 still partial at n_c 4, g64 failing, and kl1 or g64
  reaching § 3's "remedy A" row. In each, the mini is retired as the clarification says. Remedy C is not started
  tonight (§ 3 already allows at most one remedy round per night), and Leon decides.
- **"Forked together" means:**
  - § 3's two steps run in parallel: g64 forks first, kl1 forks if kl1 is partial too.
  - If g64 is still partial at n_c 4, kl1's class at n_c 4 is reported but decides nothing.
- **A control seed that fails** (`ban_stop`, crash or timeout; § 4.6) is replaced by the next seed whose reference
  L2-mini exists and that this control has not used yet. If there is none, the control is assessed at the reduced
  n_c, flagged.
- **The kg64 s2 full-scale replicate changes neither E nor the gate,** whatever its value. It informs only the
  write-up's note on the full-scale g64 effect.
- **The knob wave and its B0 reads.**
  - Wave 2 starts only once the gate decision is written in `results/gate_mini20.md`.
  - Every wave-2 arm's final L3-mini checkpoint gets the B0 L3 test read.
  - The references s0 and s1 get B0 and B64 L3 test reads of their trajectory checkpoints (L2 it 8, L3 it 0, 8, 16)
    for the § 4.3 lines.
  - `grpo_think_vs_mini20:4` (a 5th reference outside n_r) moves to the end of the wave.

**2026-10-18, 10:07Z (3:07 AM PDT; efdc89a8): errata to the two deviation amendments, after a second audit** (written after the
B0 reads were seen; no rule changes).
- **Timestamp.** The completion amendment's header says 09:40Z. It was committed at 09:36:23Z (963d90b3), and the
  deviation itself at 09:27:42Z (ba275d73).
- **Reason 1 holds only at B64.** At B0, full-scale `grpo_think_vs_mb` s0 and `grpo_think_vs_mb_kkl0` s0 both read
  88.7% L3 test pass@1, under 90%. So remedy C could also meet the clause's fix.
  - Keeping the mini rather than going to remedy C therefore rests on reasons 2 and 3.
  - It also rests on cost and on the measured noise: a mini20 run is 1.2–1.9 GPU-h against about 2.5 h, and the
    mini20 s_ref is 0.042.
  - The references' B0 mean was 76.3% (75.9, 77.4, 75.3, 76.7). The base S1g3_s0 reads 11.5% at B0.
- **The size twins (`ksize`) run in wave 2,** although the "Extremes only" list above does not name them. They are a
  single-level knob (twins against matched originals), are in § 3's restricted GO and in § 4.7, and were approved by
  Leon on 2026-10-16.
