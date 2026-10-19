---
written_on: 2026-09-28
written_by: agent:claude
---

# run4-grpo-review: does on-policy GRPO ignite the depth-3 pattern from a zero base rate? (review of the salvaged replication)

Status: **review only, no new training.** The run under review, `run4-grpo` (agent2, 2026-09-17
23:00 – 09-18 ≈ 03:00 UTC, fork branch `dan_run4_grpo`), was an independent replication of
[round 2's run 4](../2026-09-18-round2-run4-grpo-vs-ei/README.md) (GRPO vs expert iteration at
depth-3 f = 0). Its session was ended by the 2026-09-18 authentication failure. The results were
salvaged in fork commit `25295bc` with an unfinished write-up (`run4.md` still contains `XXX`) and no
`numbers.md` § Run 4, and **nobody reviewed it**. Proposal 14 item 7
([`2026-09-28-after-state-env.md`](../../docs/proposals/2026-09-28-after-state-env.md)) commissioned a
no-GPU review, which ran 2026-09-28 04:16–04:42 UTC. Verdict: **the zero-base-rate ignition result
stands on two draws, scoped as below.** Several of the run's counts have to be reworded, and
everything about the extension draws, the lr grid and the ablations is **not supported by surviving
files**. No hard-constraint violation, nothing quarantined. Numbers and wording below are the
reviewer's.

Review: [`review_run4-grpo-review.md`](https://github.com/dan-pandori/nd-takehome/blob/f4b2188/review_run4-grpo-review.md)
(fork branch `dan_run4-grpo-review`; commits [`ad5c10e`](https://github.com/dan-pandori/nd-takehome/commit/ad5c10e)
§Recount (blind), [`f4b2188`](https://github.com/dan-pandori/nd-takehome/commit/f4b2188) §Compare/§Verdict).
Reviewer code and outputs: `artifacts/run4-grpo-review/` on that branch (`rvlib.py`, `base_rate.py`,
`arms.py`, `lean_recheck.py`, `splits.py`, `ckpt_identity.py`, `shape.py` and their `.json` outputs).
Reviewed material: [`run4.md`](https://github.com/dan-pandori/nd-takehome/blob/25295bc/run4.md),
`log.md` § Run 4 (including the three entries the salvage commit deleted, read from its diff),
`preregistration/run4-grpo.md` (`49ecc8a`, 2026-09-17 23:00:11Z) and amendments 1–3 (`2e69d57`,
`d0d5ed0`, `d9002fd`, 23:10–23:26Z). Brief:
[`BRIEF_run4-grpo-review.md`](../../docs/proposals/followups/BRIEF_run4-grpo-review.md).

Question: at an equal sample budget, does on-policy GRPO acquire the depth-3 box-nesting pattern on
Stage-1 draws whose base rate is zero, where expert iteration (EI) often does not? If so, that goes
past the project's standing "RL elicits rare-but-known behaviour" reading: there is nothing rare to
elicit.

## Configuration

- **Model** (every number here). Stage-1 draws s20–s25 of `run4-grpo`: a 3,210,240-parameter
  transformer (`train.py` defaults, 4 layers, d 256) in the ND **token** format (`--mode abs`, shift
  on), trained **from scratch** for 6,000 steps × batch 128 on `data/p2/train_depth3_f0_a1.jsonl`.
  That set has 155,000 records at cap 6 and **0** depth-3 proofs by the reviewer's own counter.
  Checkpoints were `ckpts/r4/stage1_depth3_f0_a1_s<s>.pt`, and **none survive**: no md5 was recorded
  and none was uploaded. The `round3-run1` bucket files with the same names are different trainings
  (`ckpt_identity.py`).
- **Checker.** Every count in the run was produced under `nd_verify`, as the rules of 2026-09-17
  required. The reviewer translated counted proofs with `nd2lean.py` and re-checked them in Lean 4 core
  (`lean_judge.py`'s fallback path, batched): **13,317 of 13,317** distinct (prompt, proof) pairs
  were accepted, 0 rejected and 0 untranslatable. That covers every counted depth-3 proof on the
  zero-rate draws (7,127), every depth-3 proof in the six coverage files, and a sample elsewhere. The
  counts below are therefore "`nd_verify`-counted, Lean-confirmed". The reverse direction (proofs Lean
  would have accepted that `nd_verify` rejected) cannot be checked, because the run never stored
  rejected samples.
- **Design** (pre-registered). GRPO (`grpo.py`): on-policy, groups G ∈ {8, 32}, binary reward,
  group-mean baseline, no KL, 800 samples per update, 320 updates = 256,000 samples = EI's 8 rounds
  × 32 per target, over 1,000 targets. lr 1e-4 and 3e-5, two seeds per draw. The controls were paired
  EI (two seeds) and a frozen base at 256 attempts per target. The base rate is from 2,000 samples
  per target on the first 300 targets (600,000 per draw) at T 0.8. Ignition means ≥ 20 target
  theorems solved with a depth-3 proof (cumulative). Acquisition is depth-3 theorems / 1,000 at the
  last round. Truncation: up to 6.4 % of a GRPO update's samples hit `max_new` 400 (G = 32). Depth-3
  proofs are ≈ 7.5 lines, so the depth-3 counts are not length-biased, but plain solve rates may be
  slightly low.
- **What survives**: coverage files for s20–s25; all 32 main GRPO arms, 8 EI arms and 4 frozen arms on
  s22–s25 (the arms `g32 s24 lr3e-5 e2` and `g32 s25 lr3e-5 e2` were cut short at r5 and r6, and EI
  s24 e2 and s25 e2 at r4 and r6); per-round novelty files for the primary-seed arms on s20 and s21;
  sprint-sized arms on s22–s25. **Nothing survives from s26–s29, the lr 1e-5 / 3e-4 grid, the
  `noretain` / `posonly` ablations, or the s20/s21 second seeds.**
- **Splits.** Using the reviewer's own canonicaliser, the Stage-1 set overlaps targets **0** and
  transfer **0**, and overlaps held-out in **21** of 5,000 classes. All 21 differ only in premise order
  (order-sensitive overlap is 0) and can move held-out greedy by at most 0.004. `nd_verify/` and
  `artifacts/TEST_RUN_DONE` are unchanged from `origin/main`. `grpo.py` trains only on `--targets`.
- **Raw data**: `hf://buckets/dan-pandori/nd-rl/run4-grpo/artifacts/r4/` (public). No new bucket
  upload for the review; its outputs are committed on the fork branch.
- **Cost of the review**: **$0**, no pod. Registered as `podbudget run4-grpo-review` 1 h / $0.50 as a
  guard only, used 0.00 h. The original run's six RTX 3090 pods (`r4-1`…`r4-6`, $0.50/h) ran on the
  second host and are not in the ledger (see [spend](../../docs/infra/spend.md)).

## Observations

**Base rates (the claim everything rests on).** Counts are depth-3 samples in 600,000 per draw,
under the reviewer's own predicate (box depth of the dependency-pruned proof). The reviewer's labels
agree with the files' own flags on every proof.

| draw | targets solved (of 300) | depth-3 samples | depth-3 theorems | rate |
|---|---|---|---|---|
| s20 | 7 | 19 | 3 | 3.2·10⁻⁵ |
| **s21** | 4 | **0** | 0 | **0** |
| s22 | 20 | 7 | 5 | 1.2·10⁻⁵ |
| **s23** | 5 | **0** | 0 | **0** |
| s24 | 37 | 1,442 | 35 | 2.4·10⁻³ |
| s25 | 12 | 10 | 1 | 1.7·10⁻⁵ |

Separately, the frozen control (256 attempts × all 1,000 targets) found **0** depth-3 proofs on
both s21 and s23. The 600,000 samples cover only the hard first 300 targets, so the frozen control is
what covers all 1,000.

**GRPO ignition.** GRPO ignites in **40 of 40 arms with surviving data** (32 on s22–s25 including both
seeds, plus 4 primary-seed arms each on s20 and s21). That includes **12 of 12 on the zero-rate draws**
(s21: 4 arms, s23: all 8). On s23, ignition came at update 23–160, i.e. **18,400–128,000 samples**. That
is inside the 256,000 samples in which the frozen base found none. Round-8 acquisition is 0.361–0.476
on complete s22–s25 arms and 0.413–0.505 on s20/s21. Over the main arms, the first depth-3 training
sample appears at update 1–24. Sprint-sized arms (3,200 samples) reach 27 / 1 / 24 / 35 depth-3
theorems on s22–s25 at lr 1e-4 and 9 / 0 / 10 / 1 at lr 1e-5.

**Paired EI.** EI ignites in **6 of 8** arms on s22–s25, or **7 of 10** counting the s20/s21 primary
seeds (s20 ignites late, at r8 with 0.083). On the zero-rate draws it ignites in **1 of 3** arms with
data: s21 no, s23 seed 1 at r6 (0.255), s23 seed 2 no. The two EI seeds of a draw disagree on s22 and
s23.

**Held-out greedy** (5,000 in-distribution theorems). The Stage-1 draws score 0.872–0.891. GRPO at lr
1e-4 falls to **0.580–0.715**. At lr 3e-5 it holds **0.745–0.865** and still ignites all 20 surviving
arms. EI ends at 0.881–0.935. Only 5 of 16 lr 3e-5 arms stay within 0.05 of Stage-1.

**Base reachability.** This comes from the executor's stored scores; the lost checkpoints mean it
cannot be re-scored. Every distinct GRPO depth-3 proof on s20–s23 has base log p < log(1/256). On s25
the share is 99.9 % (one theorem is above), and on s24 it is 97.6–98.0 % (10 theorems above). The
theorem-level maximum is e^−13.75 on s21 and e^−14.86 on s23.

**What "depth-3" is here** (a finding the reviewer added). On s21/s23, **67–71 % of GRPO's depth-3
proofs open a third box whose assumption nothing cites.** For example, in `S > (S > (Q > P))` the third
box is a vacuous `IMPI`. EI's depth-3 proofs on s23 are the same (69 %). The base proves these targets
at depth 2 by re-ordering boxes. What gets crossed from zero is a **box-nesting habit**, not a new
inference rule. The two methods acquire the same thing. On the zero-rate draws, depth-3 proofs average
7.5–7.8 pruned lines and 7.1–7.3 term nodes, against 6.1–6.2 lines and 5.3–5.4 nodes for the same arms'
other proofs.

### What stands (the reviewer's quotable paragraph)

> In an independent replication (run4-grpo, 2026-09-17/18; 3.2 M-parameter ND-token transformers trained from scratch on
> `train_depth3_f0_a1`, a set with no depth-3 proof), two of six Stage-1 draws (s21, s23) produced no depth-3 proof in
> 600,000 base samples on the 300 hardest targets and none in 256,000 samples over all 1,000 targets (`nd_verify` at run
> time; the checkpoints are lost, so this cannot be re-measured under Lean). On-policy GRPO (groups of 8 or 32, binary
> reward, group-mean baseline, no KL, lr 1e-4 or 3e-5, 256,000 samples) ignited the depth-3 pattern (≥ 20 of 1,000 target
> theorems) in all 12 arms with surviving files on those two draws, after 18,400–128,000 samples on s23, reaching
> 0.37–0.48 acquisition; paired expert iteration ignited in 1 of 3 arms there. Across all six draws, 40 of 40 surviving
> GRPO arms and 6 of 8 paired EI arms on s22–s25 ignited. lr 1e-4 cost held-out greedy (0.58–0.72 vs 0.87–0.89 at
> Stage-1); lr 3e-5 ignited every arm at 0.75–0.87. Every depth-3 proof checked (7,127 on the zero-rate draws) is accepted
> by Lean 4. About 70 % of these depth-3 proofs nest a third, unused assumption: the acquired pattern is a box-nesting
> habit.

### Corrections to the executor's write-up (reviewer's V3)

| `run4.md` says | reviewed value |
|---|---|
| GRPO ignited in "30 of 30 arms" | 40 of 40 arms whose files survive, 12 of 12 on s21/s23. The count 30 cannot be derived from any combination of arms |
| paired EI ignited in "6 of 10" | 6 of 8 on s22–s25; 7 of 10 with the s20/s21 primary seeds |
| ignition at update 5–110 (4,000–88,000 samples) | **5–160 (4,000–128,000)**; G = 32 range 6–160, not 11–110 |
| EI acquisition 0.25–0.38 where it ignited | 0.08–0.38 (s20 ignites at r8 with 0.083) |
| base log p < 1/256 for "100 % of 4,300+ proofs on the five low-rate draws" | 100 % on s20–s23, 99.9 % on s25; 12,570 proofs over the four primary settings. "4,300+" is not derivable |
| 7–32 % of groups carry variance at update 1 | 4–32 % |
| variance "rises to 0.3–0.5 within 20 updates on every draw" | true on every draw except three s23 G = 8 arms (max 0.22–0.28) |
| E5 held | **E5 failed.** Variance > 0.30 by the ignition round fails in 13 of 32 arms; "G = 32 above G = 8 at every update" fails at 406 of 4,963 paired updates |
| E7 held at 3e-5 "for 5 of 8 arms" | 5 of 16 |
| "the sprint's null result reproduces on budget and lr alone" | the sprint-sized budget stays below the ignition line at lr 1e-5 (0–10 theorems); it is not a null |
| E9: crossing zero is "a property of RL against a verifier" | on-policy GRPO too, with one model size, one pretraining set and one pattern |
| lr 1e-4 held-out 0.54–0.72 | 0.58–0.72 on surviving arms. The 0.54 low end (s20/s21) is not derivable |

The write-up also never names its model or checker (reviewer's V5), and it lacks the claim-by-claim
comparison with round 2's executor that Dan's brief asked for.

**Not supported by surviving files** (V4). Nothing about s26–s29 has any surviving evidence, including
"s27, s29: 0 in 600,000, GRPO ignites", which is half of the deleted STATUS line's "four zero-rate
draws". The same goes for the lr 1e-5 / 3e-4 grid ("lr 3e-4 collapses to 0.17"), the `ei_noretain` /
`grpo_posonly` ablations (and hence E9's sentence about EI's retained slice), the s20/s21 second seeds,
per-update traces and held-out values, and the "4,300+" count. All of these rest on log entries only,
and the salvage commit deleted three of those entries.

**Pre-registration.** The commit times agree with the claimed order (49ecc8a at 23:00:11, then the
amendments at 23:10–23:26, each before the arms it governs). Amendment 2 was written after the round-1
results and says so. The first-pod time (23:01:07 per the run) cannot be checked from this host's
`pods.log`, because the run used the second host.

## Relation to the reviewed round-2 run 4

[Round 2's run 4](../2026-09-18-round2-run4-grpo-vs-ei/README.md) compared GRPO with EI at equal
samples on six earlier draws, none of which was zero-rate. This replication adds the case the original
lacked: **two draws with 0 depth-3 samples in 600,000, where GRPO ignites every surviving arm and EI
ignites in 1 of 3.** On those draws there is no rare base behaviour to elicit, so read literally, this
is the first reviewed instance in the project of RL crossing a measured zero. Three scopes limit it:
the "zero" is 0 in 8.56 × 10⁵ `nd_verify`-accepted samples; the pattern crossed is a layout habit
(≈ 70 % vacuous third box); and n = 2 draws, all at 3.2 M parameters, token format, one pretraining set.

## Limitations and next step

- **Closed, not extensible (V1).** No checkpoint of this run survives, so the zero base rates cannot be
  re-measured under the Lean-only judge or re-sampled at all, with or without a GPU. They also cannot
  be recomputed Lean-only from files, because rejected samples were not stored.
- n = 2 zero-rate draws. The GRPO seeds on s21 are primary-only (the second seeds were lost). No
  per-update data for s21.
- ND token format (`abs`), not the project's current `lean_seq` training format. The run predates the
  2026-09-22 and 2026-09-27 format and judge decisions.
- The coverage base rate uses the first 300 (hard) targets, not the 1,000 GRPO trains on. The frozen
  control covers the 1,000 at only 256 per target.
- `run4.md`'s figures (`figures/run4_{acquisition,heldout,variance}.png` on `dan_run4_grpo`) were not
  checked by the reviewer and are not copied here.

**Next measurement** (reviewer). Screen fresh `a1` draws for a zero rate **with Lean as the judge and
every sample stored**, and upload the checkpoints with md5s. Then run GRPO (G = 8, lr 3e-5, 2 seeds) and
EI (2 seeds) on ≥ 3 zero-rate draws, plus the `posonly` / `noretain` ablations on one of them. Estimate:
≈ 8 screens + 15 arms ≈ 15 RTX 3090 GPU-hours ≈ $8–12. Given `NOISE_FLOOR.md` (which postdates the
run), the seed count should be planned against its minimum detectable difference before launch. In a
redesign, the ≈ 70 % vacuous-box share argues for a depth predicate that counts only **cited**
assumptions, so that "crossing zero" means a pattern and not padding.
