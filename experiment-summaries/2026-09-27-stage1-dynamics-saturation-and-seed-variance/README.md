---
written_on: 2026-09-27
written_by: agent:claude
---

# stage1-dynamics: does Stage 1 saturate at 6,000 steps, and is "seed variance" a property of the seed?

Status: completed 2026-09-27 18:05–21:13 UTC (executor) and independently reviewed same day. Review
verdict: **accept, with twelve rewordings and one gap to close.** No hard-constraint violation,
nothing quarantined. Numbers and wording below are the reviewer's, including the reviewer's own
words where they differ from the executor's. Run id `stage1-dynamics`. This run happened on the
fork (`dan-pandori/nd-takehome`), branch `dan_stage1-dynamics` (a worktree from the fork's own
`origin/dan`); like `cap-horizon` before it, this branch has not been fast-forwarded into the fork's
`dan` — the commits below are reachable only from `dan_stage1-dynamics`.

Review: [`review_stage1-dynamics.md`](https://github.com/dan-pandori/nd-takehome/blob/48170ed/review_stage1-dynamics.md)
(commits [`8f47cb9`](https://github.com/dan-pandori/nd-takehome/commit/8f47cb9) §Recount,
[`48170ed`](https://github.com/dan-pandori/nd-takehome/commit/48170ed) §Compare/§Verdict). Executor
write-up: [`run_stage1_dynamics.md`](https://github.com/dan-pandori/nd-takehome/blob/48170ed/run_stage1_dynamics.md),
numbers `numbers.md` § stage1-dynamics (S1–S11), log `log.md`, `QUESTIONS.md`, all on
`dan_stage1-dynamics`. Pre-registration:
[`preregistration/stage1-dynamics.md`](https://github.com/dan-pandori/nd-takehome/blob/48170ed/preregistration/stage1-dynamics.md)
(`5fddcc7`, committed 18:05:40Z, 10 min before the first pod), with two disclosed addenda
(`2ea21a8` arm R at 18:30:46Z, `680dca3`/`05c6dd0` pass@8 at 20:17:39Z, both before the outcomes they
cover). Brief: [`docs/proposals/followups/BRIEF_stage1-dynamics.md`](../../docs/proposals/followups/BRIEF_stage1-dynamics.md),
following from Dan's reading of the noise-floor training curves.

Question: `train.py`'s validation loss only ever measured 2- and 3-line proofs, uncorrelated with
the 6-line and depth-3 accuracy that actually varies across runs. Past 6,000 steps, does long-proof
accuracy keep rising within a run? Is a run's depth-3 "mode" decided early or does it cross over?
Does WSD differ from cosine at equal steps? Is a training gain from steps or from repeating the same
155,000 proofs ~20 times? Is per-length validation loss a usable proxy for per-length accuracy?

## Configuration

- **Model, every arm.** 3,214,336-parameter from-scratch GPT (4 layers, d 256, 8 heads),
  `lean_seq`, cap 6, `--bs 128 --lr 1e-3 --min_lr 1e-4 --warmup 200`. Arms **C** (cosine, 6k), **W**
  (WSD, stable to 24k with decay branches **W-6k**/**W-12k**/**W-24k**), **R** (addendum 1, ten extra
  same-command replicates) train on `data/p2/train_depth3_f0_a1.jsonl` (155,000 records, 0
  depth-3); arm **F** (WSD, 24k) trains on the four noise-floor pools de-duplicated into
  `data/sd/train_fresh.jsonl` (572,759 records, 0 depth-3). Seeds 0–7 in C/W, 0–3 in F. `train.py`
  gained opt-in per-length/depth-3 validation logging, a `--sched wsd` flag and `--resume`
  (weights + optimiser + step + rng); default behaviour is bit-identical, verified by the reviewer
  by reading the code.
- **Judge: Lean alone** (policy 2026-09-27). `sd_eval.py` never calls `nd_verify`. Cross-checked
  against two `noise-floor` checkpoints published under Lean ∧ `nd_verify` (S1: +200 to +400 per
  million on held-out greedy, both directions predicted) and, once run `lean-judge` landed, against
  a second independent Lean route on 600 stored proofs (0/600 disagreements, S1b).
- **Measurements.** Greedy held-out (5,000 theorems, k=1, T=0) per length bin and on the depth-3
  slice, at every 1,000-step checkpoint of W's stable trajectory (24/seed) and every arm's final
  checkpoint; validation loss every 200 steps; pass@8 at T=0.8 on the depth-3 slice at four
  loss-selected checkpoint pairs (addendum 2).
- **Budget and cost.** `podbudget stage1-dynamics --set 30 15` registered before the first pod
  (briefly raised to 60 h/$30 mid-run, then reset). Actual: **8.74 pod-hours, $4.29** of 30 h/$15,
  four **NVIDIA A40, real billed $0.49/h**, all deleted. RunPod balance $144.04 after (floor $100).
  Bucket: `hf://buckets/dan-pandori/nd-rl/stage1-dynamics/{ckpts/sd,artifacts/sd,data/sd}` — 290
  checkpoints, 274 evaluations, 49 metric files.

## Observations

**The blind spot was real.** `data/p2/heldout.jsonl` is length-sorted and its first 2,000 records
are exactly 1,000 two-line and 1,000 three-line proofs — every validation curve in this project
before this run measured only proofs every model already solves at 0.99+. Verified directly by the
reviewer.

**Stage 1 is not near saturation at 6,000 steps, and the headline holds with room to spare.**
Within-seed paired 95 % intervals (reviewer's own, from the per-seed values): overall **+5.3 pp
[+2.6, +7.9]**, 5-line **+5.4 pp [+4.4, +6.3]**, 6-line non-depth-3 **+6.9 pp [+5.5, +8.4]** —
all clear zero with room. The 6-line bin and depth-3 slice the write-up leads with move further
(+16.8 pp and +26.6 pp) but far less precisely (±13 pp and ±26 pp); the reviewer's point: **Q1's
strongest evidence is the slices that do not oscillate**, not the ones the write-up headlines with.
E1's falsifier needed ≥ 6 of 8 seeds flat; **0 of 8** were.

**The depth-3 "mode" is not a property of a seed, a pool, or even a run — it is where a checkpoint
happens to land.** All **12** of the 24,000-step runs cross the 0.44 cut along their own trajectory
(1–10 times, min 0.004–0.038, max 0.488–0.934); mean fraction of checkpoints in the high mode across
the twelve runs is **0.37**, against `NOISE_FLOOR.md`'s 0.462 across 52 *separate* runs — the
cross-seed bimodality is this same oscillation sampled once per run. Arm R (the identical command
line run five times at two seeds) spans depth-3 **0.064–0.736** and 6-line **0.490–0.802**; three
same-command replicates at 24,000 steps give sds statistically indistinguishable from the 52-cell
floor (all three replicate groups' 95 % CIs on sd contain `NOISE_FLOOR.md`'s 0.1523) — the write-up's
"reproduces nearly the whole floor" is reworded to that (R4), since n = 3 cannot claim more.

**The oscillation is capability, not decoding.** pass@8 at T=0.8 on 500 depth-3 theorems moves
**−50.0 pp** and **+85.8 pp** with greedy across the two within-run pairs the addendum chose by
*loss*, not accuracy; the low-greedy checkpoints stay at 0.392 and 0.026, both below 0.60, and
pass@8 > greedy at all 4/4 checkpoints. Every bimodality this project has reported on greedy readouts
is a real capability difference, not a greedy-decode artefact.

**Per-length validation loss ranks runs; `val2k` does not.** Across 8 seeds at 24,000 steps, Spearman
ρ = **−0.905** (6-line) and **−0.934** (depth-3) against **−0.310** for `val2k`, which moved
0.0015–0.0027 all run while the 6-line bin moved up to 45 pp. Pooled trajectory ρ = −0.885/−0.893
over 192 checkpoints.

**Q3 ("schedules indistinguishable") and Q4 ("steps, not data") both need rewording.** Overall,
WSD and cosine at equal steps are indistinguishable (−1.0 pp [−3.3, +1.2]), but on the 6-line bin
it is a **non-detection at ±12 pp** (−8.8 pp [−20.7, +3.2]) and the pre-registered falsifier (>5 pp)
fired (R2). Q4's positive claim, "steps, not data," **is not supported**: the pre-registered endpoint
estimator failed on its own terms (E11's steps-vs-repetition ratio ≈ 0, E12 median −14.8 pp) and,
read against arm R's own measured sd, the endpoint design's resolution is **±21 pp** on the 6-line
bin against a pre-registered band of +2…+8 pp — **4× underpowered before the first pod** (R1). The
post-hoc trajectory estimators are genuinely flat (best depth-3 loss 0.0286 vs 0.0283, best rate
0.823 vs 0.820), which is the honest finding: *fresh data bought nothing detectable, and this
contrast needs a different readout.*

**The judge is sound where it was checked, but arm R — the arm the whole run's cross-run licence
rests on — cannot be checked.** 733 counted proofs across the five arms that store literal sampled
text were independently re-verified in Lean, 0 rejections, after 197 negative controls confirmed
the reviewer's harness rejects what it should. But `pod/sd/jobs.py` evaluates every arm-R checkpoint
(and all 228 trajectory evaluations) **without `--texts`**, so none of arm R's counted proofs
retains the literal text a reviewer must re-check. **This is the run's one open gap**, not a
suspected error — remedy is cheap and needs no training (`sd_eval.py --texts` on the ten arm-R
checkpoints, ≈ 10 GPU-minutes, ≈ $0.10) and should be done before anything else is built on this
run's floor measurement.

**Smaller findings, all bounded and none changing a conclusion:** the pre-registered "0 records
collide with a held-out class" holds only under the project's premise-order-sensitive `canon_key`
(a true canonical form finds 21 and 66 held-out collisions in the control and fresh sets; impact
≤ +0.14 pp on any rate, 0 on the depth-3 slice); two pre-registered noise-floor baselines are quoted
over the wrong population (the "6-line non-depth-3" baseline and the "≈ 67 per million" gate
boundary); `mean_written_lines` undercounts by omitting box-assumption lines (≈ 4.0 written lines for
every 6-line depth-3 proof); S9's proof-length table has a transcription error in four of five rows
(the analysis, `summary.json`, and `sd_tables.py` are all correct); S8's "arm W" table is actually
arm C at step 6,000.

## Limitations and next step

- **Close the gap before building on this run's floor measurement**: `sd_eval.py --texts
  --ckpts 'ckpts/sd/r6*.pt' 'ckpts/sd/r24*.pt'`, ten checkpoints, ≈ $0.10, no training. Add `--texts`
  to trajectory evaluations in any future run — storing text costs ≈ 1.6 MB per 5,000-theorem
  evaluation uncompressed and is what makes a run reviewable at all.
- **The next measurement that would settle what the oscillation is**: average the weights of the
  last k stable-phase trajectory checkpoints, not the readouts, and re-evaluate. If the averaged
  checkpoint's depth-3 rate sits near the high end of its run's own oscillation, the oscillation is
  a loss-basin wobble that weight averaging removes for free and "seed variance" stops being a
  property of Stage 1; if it lands near the mean, the oscillation is a genuine alternation between
  distinct solutions and checkpoint selection on per-length loss (shown here to work, ρ = −0.905) is
  the right response. The run's own data already shows the trajectory-mean readout is 2× quieter
  (between-seed sd 0.158 → 0.073 on the 6-line bin, 0.315 → 0.147 on depth-3) — ≈ 1 GPU-hour, ≈ $0.50.
  If Q4 is to be answered properly on that readout (sd 0.073), a 4-seed × 3-replicate cell per arm
  gives an MDD of ≈ ±6 pp against the pre-registered +2…+8 pp band: 24 runs of 24,000 steps,
  ≈ 35 GPU-hours, ≈ $17 — worth its own pre-registration.
- Apply the twelve rewordings in `run_stage1_dynamics.md` and `numbers.md` (full list in the
  review's §Verdict); none change a conclusion, but the 400-word write-up currently reads Q4's
  negative result as a positive demonstration and understates two of its own findings (E7's 24k mode
  count, "7 of 8 seeds" where the run's own `summary.json` already has 8 of 8).
