---
written_on: 2026-09-19
written_by: agent:claude
---

# Round 3, run 4b: does "or nothing" survive model size? (depth-3, required pool and neighbours, 25M / 85M)

Status: completed 2026-09-19 01:01 UTC (autonomous executor session, nine pods — seven A40,
two RTX A6000 — ≈ $23.5 of a $50 ceiling, all deleted) and independently reviewed 2026-09-19
01:23 UTC. Review verdict: no hard-constraint violation, no quarantine; every number in the
executor's `numbers.md` section and `run4b.md` reproduces from the pulled files with independent
code (22 claim rows; 14,580 + 25,309 counted proofs and 60,000 greedy outputs re-verified, 0
failures). **The readings do not all survive: one of the executor's conclusions is not supported
("gated by Stage-1 quality") and four are reworded below ("larger bases are zero-rate more
often", "ignition follows the schedule, not the size", "flat in size", the expectations line).**
Numbers and wording below are the reviewer's. Run id `round3-run4b`.

Review: [`review_round3-run4b.md`](https://github.com/dan-pandori/nd-takehome/blob/450c0b2/review_round3-run4b.md)
(commits `75e3232` phase 1 blind recount, `450c0b2` phase 2 claims table and verdict; reviewer
code `review_run4b_lib.py`, `review_run4b_recount.py`, `review_run4b_splits.py`; outputs
`artifacts/r3_4b/review/`). Executor write-up:
[`run4b.md`](https://github.com/dan-pandori/nd-takehome/blob/450c0b2/run4b.md), numbers
`numbers.md` §Round 3 run 4b, on branch `dan_round3-run4b` of `dan-pandori/nd-takehome` (read
the review first: the executor's files were **not** edited after the review, so they still carry
the wording the verdict rejects or asks to change). Brief: `BRIEF_scale.md` (this run is its
**4b** half; the reductio half is
[run 4a](../2026-09-19-round3-run4a-reductio-vs-model-size/README.md)).

Question: every depth-3 result so far is on a 3.2M-parameter model. At ≈ 25M and ≈ 85M, trained
on the same cap-6 f = 0 depth-3 set (no depth-3 proof in pretraining): (1) how often does a
Stage-1 draw emit depth-3 proofs before any RL, and at what rate; (2) do zero-rate draws still
stay at 0 under expert iteration on a required pool ("or nothing"), and does adding rewarded
depth-≤ 2 neighbours ignite them as at 3.2M; (3) what fraction of what EI acquires is outside
the base's reach at pass@10⁴, and does it change with size?

## Configuration

- **Code and data revision.** `dan-pandori/nd-takehome`, branch `dan_round3-run4b`, branch point
  `e1d6a3b`; `expert_iter.py`, `patterns.py`, `coverage.py`,
  `r3_1_analysis.py` and the three pools are blob-identical to `origin/dan_round3-run1`
  (reviewer); the run adds `pod/r3_4b/*`, `r3_4b_*.py`. Pre-registration
  [`preregistration/round3-run4b.md`](https://github.com/dan-pandori/nd-takehome/blob/450c0b2/preregistration/round3-run4b.md)
  committed `c1ad029` 17:54:40Z; first pod of this run `r34b-1` 17:56:28Z (`~/pods.log`). The
  `gate0` script printed FAIL because the host-wide `~/pods.log` has `la-6` (ladder-A's pod,
  17:53:21Z) between the executor's start and the commit; the reviewer confirms the expectations
  were written before any pod of this run. Amendments **A1 / A2** (`dd10040`, 18:57:53Z:
  optional-pool pre-RL sample; a lower private guess for E6, the registered E6 is what is
  scored) and **A3** (`69631a5`, 20:59:57Z: EI-only fraction moved to the `mix` arm, pass@10⁴
  for `mix` igniters) were each committed before the jobs they concern and each states what had
  already been seen — "informed guesses, correctly labelled" (reviewer). Final executor commit
  `00d3a82`. `nd_verify` tree `9437bb7` = `origin/main`; `artifacts/TEST_RUN_DONE` blob
  `1d5cf06` untouched; no evaluation or test file read by training code; no test-file run.
- **Training set.** `data/p2/train_depth3_f0_a1.jsonl`: 155,000 cap-6 records, 31,000 per length
  2–6, written box depth 0 / 1 / 2 = 82,393 / 54,883 / 17,724, **0 proofs at depth ≥ 3**
  (reviewer). Renaming-class overlap with the required pool, required transfer, neighbours and
  optional pool: 0 (premise-ordered and premise-sorted key). The file is git-ignored and, in the
  worktree, a symlink into another run's worktree (`~/work/run4-grpo/…`); **the bucket copy
  (133,670,628 bytes) is the durable one.**
- **Models** (`train.py --mode abs --cap 6 --bs 128`, seeds 0–2 per cell): **25M** (25,321,472;
  `--n_layer 8 --d 512 --n_head 8`), **85M** (85,208,064; `--n_layer 12 --d 768 --n_head 12`).
  First schedule `--steps 6000 --lr 3e-4 --min_lr 3e-5 --warmup 500` (tags `25M`, `85M`); both
  sizes missed the pre-registered held-out gate, so the pre-registered retry `--steps 12000 --lr
  1e-4 --min_lr 1e-5` (tags `25Mr`, `85Mr`) was run on the same seeds and both sets go through
  the whole pipeline: 12 draws. The 3.2M reference row (8 draws, seeds 20–27, Stage-1 lr 1e-3) is
  run 1's, recounted by the reviewer from run 1's raw files with the same code.
- **Pools** (run 1's, unchanged). Required@8 `data/r3_1/depth3_req.jsonl` (300 targets; shortest
  proof 7 / 8 lines: 6 / 294; depth-≤ 2 alternative at 9 / 10 lines / none: 288 / 1 / 11);
  `depth3_mix.jsonl` = the 300 required + 300 depth-≤ 2 neighbours; `depth3_req_transfer.jsonl`
  (100). Optional pool (A1) = first 300 of `data/p2/targets_depth3.jsonl`, the ignition study's
  pool — the brief's 3.2M reference numbers (10 / 16 non-zero, plateau 0.30–0.36) are on *that*
  pool, not the required one. The 130 classes shared between the EI pool and the optional pool
  are harmless: the optional pool is only ever sampled from Stage-1 checkpoints (reviewer).
- **Per draw.** Pre-RL: `coverage.py --k 2000 --temperature 0.8 --seed 0` on the 300 required
  targets (600,000 samples) and on the optional pool. `req` arm: `expert_iter.py --rounds 8
  --k 32 --temperature 0.8 --retain 20000 --ft_steps 600`, seed = model seed, **`--ft_lr` 1e-4
  at 25M, 3e-5 at 85M** (3.2M: 3e-4). Frozen control (`--no_train`, 256 attempts per target):
  run for 25M ×3 and 25Mr s1 only; the other 8 were skipped because a `req` arm that never
  trains is the frozen control by construction (checked byte-equal on 25Mr s1). `mix` arm: the
  same EI on `depth3_mix.jsonl`, all 12 draws. pass@10⁴ (`coverage.py --k 10000 --seed 1`,
  3·10⁶ samples on the 300 required targets) from the Stage-1 checkpoint of each `mix` igniter.
  Pattern = some line in the dependency cone of the last line at box depth ≥ 3;
  start-index-normalised distinct proofs; min-round rule; *ignition* = first round with ≥ 20 of
  300 required targets holding a pattern proof (the ignition study's ≥ 6 also reported);
  EI-only = acquired targets with no pattern proof in the base's 10⁴ samples ÷ acquired.
- **Pods.** `r34b-1`…`-4` A40 48 GB $0.49/h (created 17:56:28 / 17:59:59 / 18:01:08 / 18:02:26
  per `~/pods.log`), `r34b-5`, `-6` RTX A6000 48 GB $0.53/h (18:04:58 / 18:05:53; A40 stock ran
  out — the pre-registered fallback), `r34b-7`, `-8`, `-9` A40 (20:44:03 / 20:45:00 / 22:12:34,
  added after out-of-memory failures on shared 85M cards); deleted ≈ 23:47 (`-2`, `-3`), 00:05,
  00:13, 00:14, 00:20, 00:42, 00:44, 00:48 per `log.md`. 46.8 pod-hours = **$23.47** (reviewer:
  24.38 h + 9.10 h A40 × $0.49 + 13.32 h A6000 × $0.53). Estimate ≈ $21 + ≈ $5 for `mix` arms on
  up to six A40s, ceiling $50, stop $45; the pre-registration deviates from the brief's two
  A100s, decided before any pod.
- **Artifacts.** `hf://buckets/dan-pandori/nd-rl/round3-run4b/{artifacts,ckpts,data}` (1,012 /
  25 / 8 entries, listing checked by the reviewer; token scan of everything under
  `artifacts/r3_4b`, including the uploaded pod job logs `q/*.log`: 0 matches): 12 Stage-1
  checkpoints, the round-8 checkpoint of every arm that trained, the training set, held-out set,
  optional pool and assemble report, all coverage / `found_<r>` / `round_<r>` files and the
  per-theorem held-out greedy outputs. Figures (executor's, linked not copied):
  [`r3_4b_fractions.png`](https://github.com/dan-pandori/nd-takehome/blob/450c0b2/figures/r3_4b_fractions.png),
  [`r3_4b_mix_curves.png`](https://github.com/dan-pandori/nd-takehome/blob/450c0b2/figures/r3_4b_mix_curves.png),
  [`r3_4b_writing.png`](https://github.com/dan-pandori/nd-takehome/blob/450c0b2/figures/r3_4b_writing.png).

## Observations

Observed results first; the reviewer's readings and what is not supported follow. "r" = the
retry schedule. **Every 85M row is "gate-missing"** (untrained by the pre-registered rule at both
schedules).

**Held-out gate (E5; all 12 × 5,000 greedy outputs re-verified, the reviewer's verdict equals
the file's flag on all 60,000).**

| draw | solved / 5,000 | rate | 6-line bin |
|---|---|---|---|
| 25M s0 / s1 / s2 | 4435 / 4424 / 4456 | 0.8870 / 0.8848 / 0.8912 | 0.489 / 0.485 / 0.495 |
| 25Mr s0 / s1 / s2 | 4579 / 4492 / 4554 | 0.9158 / 0.8984 / 0.9108 | 0.619 / 0.541 / 0.590 |
| 85M s0 / s1 / s2 | 4442 / 4453 / 4431 | 0.8884 / 0.8906 / 0.8862 | 0.495 / 0.495 / 0.485 |
| 85Mr s0 / s1 / s2 | 4485 / 4518 / 4462 | 0.8970 / 0.9036 / 0.8924 | 0.516 / 0.544 / 0.487 |

Medians: 25M 0.8870 (**miss**, gate 0.90) → retry 0.9108 (**pass**; s1 alone is below 0.90); 85M
0.8884 (**miss**, gate 0.93) → retry 0.8970 (**miss**). All of the deficit is in the 6-line bin
(2–5-line bins ≥ 0.967 everywhere); final validation loss 0.0819–0.0822 in all twelve; 3.2M
draws 0.8708–0.8934 with 6-line bin 0.466–0.579 (run 1's files, not re-verified here). 21 of the
5,000 held-out theorems are premise-order permutations of a training class (inherited split);
removing them moves no rate by more than 0.0005 and no gate verdict.

**Pre-RL samples (600,000 per draw; every stored proof re-verified).**

| draws | required pool: pattern hits (rate); targets hit | optional pool: pattern hits (rate); targets hit |
|---|---|---|
| 25M s0, s1, s2 | 0, 0, 0 | 0, 0, 0 |
| 25Mr s0, s1, s2 | 0, **9** (1.5·10⁻⁵; 1 target, `d3req_204`, one distinct proof), 0 | 75 (1.25·10⁻⁴; 6), **7,961** (1.33·10⁻²; 24), 0 |
| 85M s0, s1, s2 | 0, 0, 0 | 0, 0, 0 |
| 85Mr s0, s1, s2 | 0, 0, 0 | 0, 1 (1.7·10⁻⁶; 1), 0 |

Non-zero draws on the required pool: **0 / 3, 1 / 3, 0 / 3, 0 / 3** (9 hits in 7.2·10⁶ samples;
3.2M 2 / 8). A zero in 600,000 bounds the per-sample rate at ≈ 5·10⁻⁶ (95 %). No valid
non-pattern proof of a required target in any sample. Optional pool: 0 / 3, 2 / 3, 0 / 3, 1 / 3
(3.2M "10 / 16" is the ignition study's count, not recounted in this review). Writing diagnostic
(60 targets × 64 samples, summary files only — the raw samples were not pulled, so not
recountable): fraction of samples opening a third box 3.2M 0.001–0.088, 25M 0.0005–0.021, 25Mr
0.005–0.080, 85M 0–0.0005, 85Mr 0–0.002 (no file for 85Mr s0).

**`req` arms (300 required targets only, 8 × 32).** 0 required targets throughout and **no
training step in 11 of 11 zero-rate draws**; the one non-zero draw (25Mr s1) gets 1 target
(`d3req_204`) in round 8, so no sample ever came from a trained model. Ignition on `req`: **0 of
12** (3.2M: 0 / 8). Required targets solved *without* the pattern: 0 in every arm, `req` and
`mix`.

**`mix` arms (300 required + 300 neighbours; all 8 rounds trained in every arm).**

| draw | required targets with a pattern proof, rounds 1–8 | ignition (≥ 20) | neighbours solved r8 | transfer (100) with pattern | distinct / counted proofs |
|---|---|---|---|---|---|
| 25M s0 / s1 / s2 | 0 ×8 each | – | 129 / 121 / 145 | 0 | 0 |
| 25Mr s0 | 0, 2, 15, 106, 205, 224, 226, **230** | r4 | 214 | 81 | 259 / 4,795 |
| 25Mr s1 | 0, 13, 113, 194, 219, 229, 230, **231** | r3 | 207 | 82 | 253 / 6,035 |
| 25Mr s2 | 0 ×7, **3** | – | 131 | 0 | 3 / 10 |
| 85M s0 / s1 / s2 | 0 ×8 each (s0 rounds 5–8 resumed from `_r4.pt` after an OOM, batch 384) | – | 103 / 110 / 119 | 0 | 0 |
| 85Mr s0 | 0, 0, 1, 1, 1, 1, 1, **1** | – | 112 | 0 | 1 / 27 |
| 85Mr s1 | 0, 0, 0, 0, 0, 2, 4, **11** | – (≥ 6 at r8) | 146 | 11 | 11 / 138 |
| 85Mr s2 | 0, 0, 1, 1, 3, 16, 130, **206** | r7 | 195 | 76 | 228 / 3,573 |

Ignition on `mix`: first schedule **0 / 3 and 0 / 3**; retry **2 / 3 (25Mr) and 1 / 3 (85Mr)**;
3.2M reference **5 / 8** (3 of 6 zero-rate, 2 of 2 non-zero; round-8 values 0, 0, 1, 211, 233,
239, 241, 255 — reproduces from run 1's files). Two 85Mr curves are still rising at round 8
(s1 2 → 4 → 11; s2 130 → 206): **the 8-round window censors the 85M retry cell.** There is no
frozen control on the `mix` pool; the equal-attempts control for the required stratum is the
never-trained `req` arm of the same draw (0, 1, 0 targets for the three igniters).

**pass@10⁴ from the Stage-1 checkpoint of each igniter (A3; 3·10⁶ samples each).**

| draw | pattern hits | base-reachable targets | acquired by `mix` | EI-only | fraction |
|---|---|---|---|---|---|
| 25Mr s0 | 0 | 0 | 230 | 230 | 1.000 |
| 25Mr s1 | 29 (`d3req_204` only) | 1 | 231 | 230 | 0.996 |
| 85Mr s2 (gate-missing) | 0 | 0 | 206 | 206 | 1.000 |

Adding the independent 600k sample changes nothing. 3.2M analogue at 600k samples only:
241 / 241, 210 / 211, 255 / 255, 230 / 233, 239 / 239.

**Pre-registered expectations (reviewer's scoring).** Missed: **E1** (≥ 2 / 3 non-zero per size
written, brief 3 / 3; observed 0 / 3, 0 / 3, retry 1 / 3, 0 / 3), **E2** (medians ≥ 10⁻⁵ at 25M
and ≥ 10⁻⁴ at 85M written; medians 0 at every size and schedule), **E4a's counts** (1–2 of 3 85M
and ≥ 1 of 3 25M `req` ignitions written; 0 — the "iff rate ≥ 10⁻⁴" half is trivially true, no
draw reaches 10⁻⁴), **E5** (25M 0.90–0.93 written, 0.887 on the first schedule, 0.911 only after
the retry; 85M 0.91–0.95 written, 0.888 → 0.897, below the predicted range), **E6** (≥ half of
non-igniting draws ignite on `mix` written; 0 / 6 first schedule, 3 / 6 retry, 3 / 12 overall),
**A1 on the retry draws** (optional pool ≤ 1 / 3 at 25M and 0 / 3 at 85M with rates
< 2.5·10⁻⁴ written; retry 2 / 3 with 1.3·10⁻² and 1 / 3 at 85Mr — it holds for the
first-schedule draws it was written for). Held: **E3** (11 of 11), **E4d** (0 non-pattern
solutions), **A3** (base-reachable 0, 1, 0 targets; fractions 1.000, 0.996, 1.000; n = 2 draws at
25Mr, 1 at 85Mr). Vacuous: E4b / E4c on `req`. The executor's one-line list in `run4b.md` omits
E4a's counts, E5 at 25M on the first schedule and A1 on the retry draws; its `numbers.md` table
is complete. Cost: ≈ $26 written, $23.5.

**What the reviewer does not accept, and the wording that replaces it** (the executor's files
still have the left-hand side):

1. **"…beyond base reach at 10⁴, gated by Stage-1 quality" — not supported.** Held-out greedy
   does not order ignition: 85Mr s2 ignites at 0.892 with a 6-line bin of 0.487 (the *lowest*
   retry value, equal to first-schedule draws), 25Mr s2 does not at 0.911 (0.590), 85Mr s1 does
   not by round 8 at 0.904 (0.544). Across cells, 3.2M draws at greedy 0.871–0.893 ignite 5 / 8
   while first-schedule 25M / 85M draws at the same greedy (0.885–0.891) ignite 0 / 6 (Fisher
   two-sided **0.03** — the largest between-cell contrast in the run, and it runs against a
   quality gate). Something separates those cells, but this run cannot say whether it is size,
   the Stage-1 learning rate (1e-3 vs 3e-4 vs 1e-4), the fine-tune learning rate, or chance in
   which draws open a well-formed third box.
2. "**Larger bases are zero-rate more often, not less**" / "these bigger bases had *less* to
   elicit" → "larger bases were *not more often* non-zero (E1 / E2 failed at both sizes); on the
   optional pool the first-schedule draws and both 85M cells were less often non-zero than 3.2M,
   the 25M retry draws were not." Required pool 2 / 8 vs 1 / 12: Fisher two-sided 0.54 — **no
   difference**. Optional pool 10 / 16 vs first-schedule 0 / 6 (p = 0.015) is a difference, but
   the same seeds at 25M on the retry schedule are 2 / 3 non-zero at 1.3·10⁻², seven times the
   largest 3.2M rate the brief quotes. "Less to elicit" is true of 9 of the 12 draws, not of
   "bigger bases".
3. "**With neighbours, ignition follows the schedule, not the size**" → "all three igniters are
   retry-schedule draws (3 / 6 vs 0 / 6; Fisher one-sided 0.09, two-sided 0.18; not significant
   at n = 6, post hoc)". The heading states a cause; the data are a suggestion.
4. "EI-only fraction ≈ 1 at every size … **flat in size**" → "≈ 1 in all three igniting draws;
   no fall; one 85M igniter (gate-missing), so no trend is called" — the pre-registration itself
   requires two igniters per size — "and a rise could not be seen on a required pool": the
   fraction is at its ceiling at the smallest size, so the brief's falsifier (fraction growing
   with size) is untestable here.
5. "Samples that open a third box: … 0–2 at 85M" → "0–2 (first schedule), 0–6 with the retry
   draws" (85Mr s2 = 6; summary-level only).
6. `run4b.md`'s Design line lists "frozen" without saying 8 of 12 frozen arms were not run (the
   substitution is valid); the caveats must add the **`ft_lr` confound** (3e-4 / 1e-4 / 3e-5 by
   size, and Stage-1 lr 1e-3 at 3.2M) and the **8-round censoring of 85Mr**.

Standing as written: "size bought nothing" as a statement about held-out greedy on this
155k-record set ("data-limited, not capacity-limited" is the accurate version); "'or nothing'
holds at every size on the required pool" — by construction an arm with no accepted proof cannot
train, so the content is "zero-rate at 600k ⇒ nothing at 256 attempts × 300", which held 11 / 11;
"scale does not retire the clause" for these twelve draws, with the 85M label and the confounds.

**What stands** (reviewer): (1) at 25M and 85M on the cap-6 f = 0 depth-3 set, no first-schedule
draw and 1 of 6 retry draws emits a depth-3 proof on the required pool in 600k samples (9 hits,
one target); the brief's E1 / E2 and the executor's fail at both sizes; (2) zero-rate ⇒ nothing
on the required pool, 11 of 11, no training step; the one non-zero draw acquires 1 target; no
`req` arm ignites at any size (0 / 12; 3.2M 0 / 8); (3) **with neighbours in the pool, 3 of 12 new
draws ignite (230, 231, 206 of 300 required targets; 76–82 of 100 transfer targets), all through
the pattern, and what they acquire is not in the base's 10⁴ reach: 230 / 230, 230 / 231,
206 / 206**; (4) held-out greedy does not improve with size on this data; 25M passes the gate
only on the retry schedule; 85M is gate-missing on both. **For the project's question**
(reviewer): the run adds nothing against the elicitation-or-nothing reading and one clean point
for the other half — with rewarded neighbours, EI reaches required depth-3 targets that the base
does not reach in 3·10⁶ samples, at 25M as at 3.2M. It does not show how that depends on size.

## Limitations and next step

- Three seeds per cell; the size axis is confounded with the Stage-1 schedule and learning rate
  (1e-3 / 3e-4 / 1e-4), with `ft_lr` (3e-4 / 1e-4 / 3e-5) and, at 85M, with an 8-round window
  that cuts two rising curves. 85M is untrained by the run's own gate at both schedules.
- The schedule contrast (3 / 6 vs 0 / 6) is post hoc and not significant; the proposed mediator
  (held-out greedy, 6-line bin) does not order the igniters.
- The EI-only fraction is pinned at its ceiling on a required pool, so this design cannot show a
  rise with size; one igniter at 85M. The 3.2M row has no pass@10⁴ sample (600k analogue only).
- The pool needs 7–8 lines and the cap is 6: depth and length are confounded, as in run 1 and
  round-3 run 3.
- Only 4 of 12 frozen arms exist as files and none on the `mix` pool (nothing is lost: the
  never-trained `req` arm is an exact equal-attempts control for the required stratum). The
  writing diagnostic's raw samples were not pulled.
- Process: interrupted samplers (the executor's repeated out-of-memory mistakes from doubling jobs on one card, all in `log.md`)
  were resumed with a re-seeded generator, and forward / reverse halves overlap on 6–110 targets;
  merged files keep one record per target and every affected cell is 0 hits or a single target,
  so no count depends on the merge rule. The training set in the worktree is a symlink into
  `~/work/run4-grpo`; a clean-up there breaks this checkout (bucket copy is durable). Spend
  rests on `~/pods.log` creations and `log.md` deletions. `gate0` / `gate1` failures are harness
  artefacts (`la-6` is not this run's pod; marker blob identical to `origin/main`; the
  done-regex rejects the hyphen in `RUN3-4b`). The review's phase-2 header says "≈ 01:45 UTC";
  its commit is 01:23:14Z.

Next measurements (reviewer, cheapest first): (1) *is 85Mr 1 / 3 or more?* — continue
`ei_depth3_85Mr_s1_mix` and `_s0_mix` from their round-8 checkpoints (in the bucket) to round 16
(`--start_round 9 --resume_found`), ≈ 2 × 1.5 A40-hours ≈ $1.5; (2) *schedule or chance?* — 25M,
seeds 3–8, both schedules, `mix` arm only, with the writing diagnostic's raw samples kept, ≈ $9
(0 / 9 vs ≥ 5 / 9 makes the schedule effect real at p < 0.03; one first-schedule igniter removes
it), alongside the executor's proposed 3.2M two-schedule test (8 + 8 seeds, ≈ $3, `QUESTIONS.md`
01:05), with `ft_lr` held at the same ratio to the Stage-1 peak in every cell; (3) *can the
EI-only fraction move with size at all?* — pass@10⁴ and a `mix`-style arm on the **optional**
pool for 25Mr s1 (base rate 1.3·10⁻²), 25Mr s0 and two 3.2M igniters; only there can "falls" or
"grows with size" be observed.
