---
written_on: 2026-09-24
written_by: agent:claude
---

# ds-composition: does the pretraining set's *composition* buy what the cap buys? (proposal 10, run 1)

Status: completed 2026-09-24 04:20 UTC (autonomous executor, two sessions; five pods in session 2)
and independently reviewed 2026-09-24 04:39 UTC. Review verdict: **no hard-constraint violation,
nothing quarantined**. The reviewer re-derived every number with its own tooling (own tokeniser,
proof parser, start-index normaliser, depth counter, pattern predicates, renaming-class key and
term-size counter; none of the executor's analysis scripts imported) and found **no number in the
write-up it could not reproduce**, and no disagreement between `nd_verify` and Lean anywhere in
the run. Its findings are about **wording and scope, not arithmetic**: eight rewordings, four
"support thinner than the wording implies" notes, and seven things it checked that the write-up
does not claim and should. Numbers and wording below are the reviewer's. Run id `ds-composition`.

Review:
[`review_ds-composition.md`](https://github.com/dan-pandori/nd-takehome/blob/6b99c8b/review_ds-composition.md)
(commits `4ebc003` phase-1 blind recount, `6b99c8b` phase-2 compare and verdict; reviewer code and
outputs under
[`review/ds-composition-recount/`](https://github.com/dan-pandori/nd-takehome/tree/6b99c8b/review/ds-composition-recount)).
Executor write-up:
[`run_ds_composition.md`](https://github.com/dan-pandori/nd-takehome/blob/6b99c8b/run_ds_composition.md),
numbers `numbers.md` § ds-composition (D1–D7), log `log.md` § ds-composition (both sessions),
open questions `QUESTIONS.md`, all on branch `dan_ds-composition` of `dan-pandori/nd-takehome`
(**read the review first**: the executor's files were not edited after the review, so they still
carry the wording the verdict asks to change). Pre-registration:
[`preregistration/ds-composition.md`](https://github.com/dan-pandori/nd-takehome/blob/6b99c8b/preregistration/ds-composition.md)
(`56c42bb`, plus three amendments). Proposal:
[`docs/proposals/2026-09-22-dataset-styles.md`](../../docs/proposals/2026-09-22-dataset-styles.md);
brief [`dataset-styles/BRIEF_ds-composition.md`](../../docs/proposals/dataset-styles/BRIEF_ds-composition.md).

Question: holding the generator, the `lean_seq` Lean rendering, the 3.2M from-scratch model, the
6,000-step schedule, the set size (155,000) and the depth-3 f = 0 filter fixed, does the
**composition** of the pretraining set change (a) held-out greedy accuracy by length and (b) RL
readiness — base rates of depth-3 and required-reductio proofs at pass@2,000, EI − frozen after
four dial rounds, and transfer `L*` after ladder rung T1? Decision rule (proposal 10): an arm
replaces the control iff, on both seeds, held-out greedy is within 1 pp of the control or better
**and** at least two of {depth-3 pass@2,000, reductio pass@2,000, ladder T1 `L*` or solved}
improve with **none worse**.

## Configuration

- **Arms** (one change each to the control `C0`, everything else fixed; 155,000 proofs each,
  depth-3 excluded in the pruned *and* the written form, cap 6 on the ND record except A3):
  **A1** the pool's own length histogram (10.42 / 12.63 / 15.53 / 20.60 / 40.82 % vs the control's
  flat 20 %); **A2** rule quotas (`ORE` ≥ 10 %, `ANDE1` ∪ `ANDE2` ≥ 8 %, `BOTE` ≥ 5 %, flat
  histogram); **A3** cap 8, lengths 2–8 flat (a **yardstick outside the take-home's cap-6 rule**,
  labelled "cap 8" in every row, not a candidate for adoption); **A4** cap-heavy dose
  0 / 5 / 15 / 30 / 50 %. Two Stage-1 seeds per arm.
- **Sets, as re-counted by the reviewer from every one of the 775,000 training records.** Every
  per-length count equals the pre-registration's table exactly; every set is 155,000 **distinct
  renaming classes** with **0 duplicate rendered texts** and **0 records over cap**. A2's achieved
  quotas: `ORE` 15,500 = 10.00 %, `ANDE1` ∪ `ANDE2` 12,464 = 8.04 %, `BOTE` 7,750 = 5.00 %.
- **Amendment 1's confound fix verified** (the single most important thing in this run): A1 and A4
  differ from the control in the **length histogram only** — per-length reductio and derived-`ORE`
  shares are bit-for-bit the control's (reductio 17.34 / 16.68 % and derived-`ORE` 0.09 / 0.20 % at
  lengths 5 / 6 in C0, A1 **and** A4). A3's 7 / 8 bins carry `pool_cap8`'s own rates (5.79 / 5.18
  and 3.43 / 13.59).
- **Split disjointness.** 0 renaming-class overlap between **every** one of the five training sets
  and **all eleven** evaluation pools (held-out, `targets_depth3`, `d3sub250`, `transfer_depth3`,
  `depth3_req`, `d3req_transfer`, `reductio_req`, `red_req_transfer`, ladder targets, ladder
  transfer, val36) — including the brief's specific open question (the control's a1 set against the
  ladder and reductio pools: **0**).
- **Model and schedule.** 4 layers, d 256, 8 heads, **3,214,336 parameters** (3.2M, not the write-up's
  "3.3M"); from scratch; `lean_seq` Lean surface form; Stage-1 6,000 steps, bs 128, `--cap 6`
  (A3 `--cap 8`). The control is **not retrained**: its Stage-1 checkpoints are lean-format's
  `ckpts/lf/stage1_a1_seq_s{0,1}.pt`, and its held-out values reproduce the on-file lean-format
  numbers to four decimals.
- **Reward / checker.** A proof counts only if **Lean 4.34 and `nd_verify` both accept** it
  (`nd2lean.py --check`, unmodified apart from `dan_lean_seed2`'s owner-authored `BOTE` fix;
  `lean_gate.py` and `nd_verify` byte-identical to their upstreams).
- **Commands** (`pod/dsc/jobs.py <arm>`; one pod per arm, ≤ 3 then ≤ 5 concurrent jobs per GPU
  through `pod/dsc/runner.sh`). Batch sizes **held fixed across arms**: held-out 512, dial 768,
  ladder 512, coverage 1024; decode through `sample.py`'s fast path (run `efficiency`).
  - Stage-1, seeds 0 and 1: `train.py --data <set> --heldout data/p2/heldout.jsonl --mode lean_seq --steps 6000 --bs 128 --cap {6|8} --seed <s>`
  - Held-out greedy: `eval_set.py --in data/p2/heldout.jsonl --temperature 0 --batch 512` (5,000 theorems)
  - Depth-3 dial, EI and frozen control (`--no_train`), per Stage-1 seed:
    `expert_iter.py --init <ckpt> --targets data/p2/targets_depth3.jsonl --transfer data/p2/transfer_depth3.jsonl --train <set> --rounds 4 --k 32 --temperature 0.8 --batch 768 --seed <s>`
  - Ladder rung T1 and frozen: `ladder_ei.py --init <ckpt> --name la_{T1,frozen}_<arm>_s<s> --train <set> --batch 512 --seed <s> [--no_train]` (8 rounds × 32)
  - Coverage pass@2,000: `coverage_lean.py --ckpt <ckpt> --in <pool> --k 2000 --temperature 0.8 --seed 0 --batch 1024` on `targets_reductio_req` (300), `depth3_req` (300) and `targets_depth3_sub250` (250; amendment 2)
  - Checker of record: `pod/dsc/record.py <arm>` — unmodified `nd2lean.py --check` on every counted proof.
- **Gate 0 (expectations committed before the pods they govern).** Pre-registration `56c42bb`
  2026-09-22T06:08:58Z, first `dsc-*` pod 06:09:23Z (25 s); amendment 3 `b836eef`
  2026-09-23T21:11:47Z, first session-2 pod 21:12:35Z (48 s). **Holds both times.**
- **Pods and cost.** Session 2 (the session every reported number comes from): `dsc-c0`, `dsc-a1`,
  `dsc-a3` on **A40 $0.49/h** and `dsc-a2`, `dsc-a4` on **RTX A6000 $0.53/h** (no RTX 3090 in
  stock at any tier — logged deviation), created 2026-09-23 21:12–21:22 UTC, deleted 2026-09-24
  01:05 / 01:19 / 02:10 / 02:59 / 04:10; **25.70 pod-hours, $12.84** of the $18 budget and 36-hour
  ceiling. Session 1 (2026-09-22, superseded) cost a further **≈ 115 pod-hours ≈ $57**, of which
  ≈ 93 pod-hours ≈ $47 were **idle pods after the executor was stopped by a Claude usage-credit
  exhaustion at 10:33 UTC** — see the [spend ledger](../../docs/infra/spend.md) for the derivation
  and the [week-39 digest](../../docs/digests/2026-39.md) for what needs deciding about it.
  Pre-registered: ≈ 35 pod-hours ≈ $18, ceiling $22.
- **Artifacts.** `hf://buckets/dan-pandori/nd-rl/ds-composition/{artifacts,ckpts,data}` (present
  and verified by the reviewer at 04:19–04:20 UTC). No `dsc-*` pod alive at the review's
  `list-pods`.

## Observations

All values are the reviewer's independent recount; they equal the executor's in every cell it
compared (24 claim rows, plus every per-length count, rule share and `L*`). **124,952 counted
proofs, 124,952 accepted by both checkers, 0 disagreements in either direction**; the reviewer
independently re-ran `nd2lean.py --check` on a 1,037-proof sample spanning all five arms and every
measurement family: 1,037 / 1,037 agree, and a negative control is rejected by Lean.

### What stands

- **The histogram changes in-distribution accuracy at length 6, on both seeds.** A1's 6-line
  **no-pattern** bin is **+7.3 / +6.5 pp** over the control (z ≈ 2.4 and 2.2 per seed), with the
  within-bin rule mix held bit-identical to the control's. This is the run's cleanest positive
  result and it is two-seed. (It is *not* the pre-registered quantity — see below.)
- **Cap 6 cannot write a long reductio proof, and cap 8 can — on both seeds.** On a pool where
  every target requires a derived `DN`, C0, A1, A2 and A4 solve **zero** targets outside the
  7-line stratum and write **zero** accepted reductio proofs of ≥ 8 lines, across both seeds and
  2,000 attempts per target; A3 (cap 8) solves 24 and 30 stratum-8 targets and writes 29 / 33 such
  proofs. **This — not the single-seed ladder — is the two-seed evidence for "the cap sets the
  horizon", and it should lead that paragraph.**
- **No arm replaces the control.** A1 is better in distribution, on required@8 depth-3 and on the
  ladder, but its required-reductio pass@2,000 is below the control on **both** seeds (16 / 18 vs
  28 / 26), so the decision rule's "none worse" clause fails. Two-seed, correctly applied.
- **Falsifier 3 is not triggered.** No arm has both a lower frozen base rate than C0 and a larger
  EI − frozen on both seeds. The sharper, supported form: **EI's round-4 acquisition lands at
  0.38–0.44 in every cap-6 arm × seed whatever base rate it starts from** (frozen 0.117–0.270),
  across eight cells and two seeds.
- **A4 is a clear negative.** Removing 2-line training proofs costs 25 / 30 pp on the 2-line
  held-out bin (0.746 / 0.696 against a pre-registered ≥ 0.95) and 9.2 / 7.2 pp overall, on both
  seeds.

### Headline tables (reviewer's values)

Held-out greedy, 5,000 theorems, both Stage-1 seeds; all 50,000 arm × seed × theorem verdicts
re-decided by the reviewer with **0 disagreements**:

| arm | overall s0 / s1 | 2-line | 6-line | **6-line no-pattern** | Δ vs C0 (no-pattern) |
|---|---|---|---|---|---|
| C0 control | 0.9088 / 0.8962 | 0.994 / 0.997 | 0.686 / 0.583 | **0.818 / 0.834** | — |
| A1 natural histogram | 0.9172 / 0.9424 | 0.996 / 0.993 | 0.695 / 0.840 | **0.891 / 0.899** | **+7.3 / +6.5 pp** |
| A2 rule quotas | 0.9186 / 0.9402 | 0.999 / 0.997 | 0.713 / 0.825 | **0.777 / 0.773** | **−4.1 / −6.1 pp** |
| A3 **cap 8** | 0.9508 / 0.9536 | 0.998 / 0.997 | 0.877 / 0.869 | **0.858 / 0.895** | +4.1 / +6.1 pp |
| A4 cap-heavy | 0.8166 / 0.8238 | **0.746 / 0.696** | 0.512 / 0.611 | **0.846 / 0.826** | +2.8 / −0.8 pp |

Base rates at pass@2,000 (T = 0.8, sampling seed 0, on each arm's Stage-1 model; every stored proof
re-verified, 0 `nd_verify` rejects):

| arm | `redreq` solved (of 300) | ≥ 8-line reductio proofs | stratum 7 / 8 solved | `d3req` with depth-3 proof (of 300) | `d3sub` with depth-3 proof (of 250) |
|---|---|---|---|---|---|
| C0 | 28 / 26 | **0 / 0** | 28/0 ; 26/0 | 170 / 108 | 71 / 40 |
| A1 | **16 / 18** | 0 / 0 | 16/0 ; 18/0 | 174 / 197 | 70 / 76 |
| A2 | 23 / **4** | 0 / 0 | 23/0 ; 4/0 | 132 / 214 | 48 / 84 |
| A3 **cap 8** | **54 / 72** | **29 / 33** | 30/**24** ; 42/**30** | **276 / 289** | **149 / 163** |
| A4 | 17 / **0** | 0 / 0 | 17/0 ; 0/0 | 167 / 106 | 58 / 49 |

Depth-3 dial (1,000 targets, 4 rounds × 32, frozen control at equal attempts) and ladder rung T1
(**Stage-1 seed 0 only**; `L*` = largest L with ≥ 5 solved transfer targets at `L_true` ≥ L):

| arm | dial EI s0 / s1 | dial frozen | EI − frozen | T1 transfer solved (of 2,285) | T1 `L*` | frozen transfer solved | frozen `L*` |
|---|---|---|---|---|---|---|---|
| C0 | 0.432 / 0.419 | 0.170 / 0.117 | +0.262 / +0.302 | 856 | 12 | 158 | 9 |
| A1 | 0.437 / 0.408 | 0.212 / 0.270 | +0.225 / **+0.138** | 965 | 11 | 205 | 10 |
| A2 | 0.408 / 0.406 | 0.135 / 0.262 | +0.273 / **+0.144** | 977 | 11 | 111 | 9 |
| A3 **cap 8** | 0.656 / 0.698 | 0.553 / 0.604 | +0.103 / +0.094 | **1,438** | 12 | **976** | **11** |
| A4 | 0.403 / 0.383 | 0.155 / 0.118 | +0.248 / +0.265 | 966 | 12 | 156 | 10 |

### Rewordings the verdict requires

1. **Label every ladder claim "seed 0 only" in the sentence that makes it** (B1). The ladder is
   70 % of the run's compute and the entire ladder is Stage-1 seed 0; seed 1 was cut at rounds 1–3
   under the pre-registered stop rule. `numbers.md` D4 and `QUESTIONS.md` say so, but
   `run_ds_composition.md` mentions it only in its last line, after three headline claims that rest
   on it, and the readiness figure's ladder panels show one bar per arm under a caption saying
   "left bar s0, right bar s1". The standing rule is that a difference needs two seeds; the D5
   rows for A1's ladder should read "met, n = 1". The one two-seed ladder statement recoverable
   from the partial seed-1 records is an equal-attempts frozen round 1: **C0 56 vs A1 139**
   (seed 0: 76 vs 117) — A1's base ladder reachability beats the control on both seeds at round 1.
2. **The pre-registered 6-line quantity is the whole bin, not the no-pattern sub-bin** (B2). On the
   whole 6-line bin against its pre-registered +2 to +6 pp band, A1 is **+0.9 pp (s0) / +25.7 pp
   (s1)** — the sign agrees but **neither seed lands in the band**, and the control's own seed
   spread on that bin (0.686 vs 0.583, 10.3 pp) is larger than the s0 effect. The no-pattern
   sub-bin substitution is well motivated and was announced in `log.md` at 06:24 before any arm
   result, but it is **post hoc** and D5 scores it against the whole-bin band. D5 needs two rows
   per arm.
3. "Above the cap-8 yardstick's +4.0 / +6.1" → **"above on s0, level on s1"** (B3): A1 +7.29 / +6.48
   against A3's +4.05 / +6.07, a 0.4 pp margin on seed 1 on a 247-theorem bin.
4. Fix D2's prose "(29 / 30 of them)" to agree with the "29 / 33" in its own table, and **state the
   counting convention**: 29 / 33 by *written* length, 28 / 32 by *pruned* length (B4).
5. **"The cap sets the horizon" is a reductio-depth horizon, not a line-length ceiling** (B5). On
   the required-reductio pool the cap-6 arms stop dead at 7 pruned lines while A3 reaches 10 — sharp
   and clean; but on the depth-3 pools the same cap-6 models routinely write accepted 8-, 9-, 10-
   and (A2) 11-line proofs. The unqualified version is contradicted by this run's own coverage files.
6. D5's A3 reductio row: measured ×1.93 / ×2.77 against a pre-registered ×2–5, so **"s0 just below
   band, s1 inside"**, not "met" (B6). Parameter count **3.2M**, not 3.3M (B7).
7. Present finding 3 as **one** observation — EI's round-4 endpoint is base-rate-insensitive — with
   its arithmetic consequence, not as two findings (B8).
8. Narrow "finding 2's rule-mix account stands" to what a single-seed table can carry (C1): counting
   schemata reaching ≥ 5 T1 solves beyond the three baseline schemata, **C0 0, A1 2, A2 1, A3 4,
   A4 2**. A2's falsifier needed 2 and A2 moved 1, so **by the letter finding 2 is not falsified** —
   but A1, a *pure histogram* change with the control's exact within-bin rule mix, moved 2, and A4
   moved 2. What the run supports is "A2's particular quotas did not move two schemata, while a
   pure histogram change did", on one seed; **not** "rule mix is not the lever, generator shape is".

### Support thinner than the wording implies

- **A1's reductio deficit** (16 / 18 vs 28 / 26) is correct and is what blocks adoption; pooled over
  seeds the two-proportion z is **2.21** (p ≈ 0.027), per seed 1.88 and 1.25. The "n ≈ 20 events per
  cell" caveat must travel with the claim: seed spread on that pool is enormous (A2 23 → 4,
  A4 17 → 0).
- The decision rule says "depth-3 pass@2,000" without naming which of the two depth-3 pools. A1
  improves on `d3req` on both seeds but on `d3sub` seed 0 is 70 vs 71 — marginally worse. The
  conclusion is unaffected; the ambiguity should be resolved in the text.
- **Three of the control's own pre-registered bands were missed**: frozen ladder 158 vs 220–320,
  reductio 28 / 26 vs 5–25, required@8 0.567 / 0.360 vs 0.05–0.25 (a factor 2–7 miss on the
  baseline). The arm bands were stated as offsets from those C0 values, so a mis-estimated baseline
  explains several "above band" verdicts.
- C0's and A4's T1 `L*` = 12 each rest on exactly 5 solved targets at `L_true` ≥ 12 — the threshold
  itself. The `L*` column is knife-edge at the top; solved counts are the robust statistic.

### Checked by the reviewer, absent from the write-up, and worth carrying forward

1. **The histogram arms are clean** (per-length reductio and derived-`ORE` shares identical to the
   control's to four decimals). This is the premise of every A1 conclusion and is stated nowhere.
2. **A2 is not a pure rule-quota arm**: 14,775 of its 15,500 `ORE` proofs are 6-line, so 47.7 % of
   A2's 6-line bin is an `ORE` proof. A2 changes the rule mix *and* the within-bin shape at length 6.
3. **Split disjointness is perfect** (0 overlap, eleven pools × five sets) and answers the brief's
   open question. Not reported.
4. **The evaluation pools overlap each other** — `targets_depth3 × depth3_req` 67,
   `targets_reductio_req × ladder_rl_targets` 16, `× ladder_transfer` 15,
   `transfer_reductio_req × ladder_*` 7 / 6, `targets_depth3 × ladder_rl_targets` 6. Not a training
   leak, but two of the run's "independent" readiness axes share theorems.
5. **Term size is not reported at all**, though proposal 10 asks for it. Median term size (formula
   nodes over the pruned proof) / maximum pruned line count on the required-reductio pool: C0 28 / **7**,
   A1 26.5 / **7**, A2 29 / **7**, A4 28 / **7**, A3 (cap 8) 33 / **10**. Training-set **mean** term
   size orders A2 25.1 < C0 26.0 < A1 28.6 < A3 29.4 < A4 30.9 — and A4, with the largest mean term
   size, has the worst held-out accuracy. That is a candidate explanation for A4's collapse which the
   length histogram alone does not give, and it should be excluded or adopted before "the histogram
   is a lever" is carried into the next run's design.
6. **The sampler change is quantified and benign**: on the *same* control checkpoints the new fast
   decode path gives dial acquisition 0.432 / 0.419 against the on-file 0.418 / 0.408 (+1.4 / +1.1 pp)
   and identical greedy held-out. That is the size of "sampler noise" and the right yardstick for
   reading the dial's pre-registered ±0.05 bands.
7. **Gate totals across the run**: 26,715,560 samples gated, 5,752,957 both-accepted,
   `nd_ok & lean_rej` **0**, `nd_rej & lean_ok` **1,611** — all the known `BOTE` looseness
   (`n.elim` on `n : ¬A` resolving as `Not.elim`), which amendment 3 predicted and accepted by
   leaving `lean_tok.py` unchanged so the control's checkpoints stay comparable. Because the gate is
   Lean **∧** `nd_verify`, none was counted. Gate self-test: 2,000 samples, 1,600 accepted,
   0 disagreements.

### Expectations that were wrong

A4's 2-line bin ≥ 0.95 → **0.746 / 0.696**; A4's overall within ±1 pp of C0 → **9.2 / 7.2 pp below**.
A1's whole 6-line bin +2 to +6 pp → **+0.9 / +25.7 pp** (sign right, band missed on both seeds).
A1 and A2's dial EI − frozen within ±0.05 of C0 → **−0.164** and **−0.158** on seed 1. A3's frozen
ladder ≥ 1,000 → 976; A3's reductio ×2–5 → ×1.93 on s0. On the control itself: frozen ladder
220–320 → **158**; reductio 5–25 → **28 / 26**; required@8 0.05–0.25 → **0.567 / 0.360**; and the
d3sub depth-3 rate 0.35–0.55 → **0.284 / 0.160**. A2's textbook schemata at ≥ 5: expected ≥ 2,
measured **1**.

## Limitations and next step

- **The ladder is n = 1.** Every ladder number — including the three headline claims and A1's
  "+30 % frozen solves, `L*` +1" — is Stage-1 seed 0. The standing two-seed rule is not met there.
- **The pre-registered 6-line comparison was replaced by a post-hoc sub-bin.** The substitution is
  defensible and was announced before any arm result, but the pre-registered quantity misses its
  band on both seeds.
- **The schema table (19 schemata × 5 arms) is single-seed**, and the arm that moved the most
  schemata is the pure-histogram arm, not the rule-quota arm.
- **A2 confounds two changes** (rule quotas and the within-bin shape at length 6).
- **Session 1's checkpoints and proof-level files were lost** to a host cleanup after the run was
  stopped by a Claude usage-credit exhaustion; everything reported is re-measured on fresh Stage-1
  draws. The one number that shows why the lost session could not simply be quoted: A4's frozen
  depth-3 base rate was 0.282 pre-cleanup and 0.155 on the fresh draw — the frozen base rate is a
  property of the *draw*, not only of the set.
- **Proposal 10's `lean_check`** (term allowlist + `#print axioms`) does not exist in this
  repository, so the allowlist and axiom checks could not be run; `nd2lean.py`'s translation is
  core-Lean only by construction.
- Two job failures in the whole run (one missing synced target file, one CUDA OOM), both re-run;
  the session-1 orphan-duplicate coverage incident did not recur.

Next step (the reviewer's ranked list; nothing runs until Dan lifts the 2026-09-18 pause and names
it): (1) **two ladder seeds, four rounds, C0 and A1 only** (≈ 4 pod-hours, ≈ $2) — every headline
ladder claim is n = 1, and the partial seed-1 records already favour A1, which is why it is cheap
insurance; (2) **two more Stage-1 seeds for C0 and A1 on the `redreq` pool only** (≈ 1 pod-hour) —
A1's reductio deficit is the single finding that blocks adoption and sits at z ≈ 2.2 pooled over
≈ 20 events per cell; (3) **cap 8 at an equal ≤ 6-line count** — A3 changes two things at once (it
adds 7–8-line proofs *and* removes 8,857 proofs per ≤ 6-line bin), and a cap-8 arm with the
control's 31,000 per length 2–6 plus 7–8-line proofs on top would separate "longer proofs in the
set" from "fewer short ones"; (4) **report term size** — this run has the files to produce it
retrospectively, and the training-set mean term-size ordering tracks A4's held-out damage better
than the histogram does. The executor's three open questions in `QUESTIONS.md` (cap-8 regime,
ladder seeds, A1's reductio deficit) propose (3), (1) and (2) respectively and await Dan.
