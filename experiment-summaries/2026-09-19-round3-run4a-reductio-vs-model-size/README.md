---
written_on: 2026-09-19
written_by: agent:claude
---

# Round 3, run 4a: does "or nothing" survive model size? (reductio, required pool, 3.2M / 25M / 85M)

Status: completed 2026-09-19 00:57 UTC (autonomous executor session, six A100 80 GB pods,
≈ $33.8 of a $50 ceiling, all deleted) and independently reviewed 2026-09-19 01:13 UTC. Review
verdict: no hard-constraint violation, no quarantine; every number in the executor's
`numbers.md` section and `run4a.md` reproduces from the pulled files with independent code (27
claim rows), apart from one quoted range and one denominator. **Three of the executor's readings
are reworded below: "'or nothing' applies to ≈ 1 draw in 10 instead of 2 in 3" / "bigger bases
emit the pattern more often", "at most one of 133 eight-line targets", and "'or nothing'
survives" without its attempt budget.** Numbers and wording below are the reviewer's. Run id
`round3-run4a`.

Review: [`review_round3-run4a.md`](https://github.com/dan-pandori/nd-takehome/blob/f9e1ce2/review_round3-run4a.md)
(commits `69d9dfc` phase 1 blind recount, `cdcb8f0` phase 2 claims table and verdict, `f9e1ce2`
timing details; reviewer code `review_r3_4a_lib.py`, `review_r3_4a_recount.py`,
`review_r3_4a_splits.py`; outputs `artifacts/review_r3_4a/{recount,splits}.json`). Executor
write-up: [`run4a.md`](https://github.com/dan-pandori/nd-takehome/blob/f9e1ce2/run4a.md), numbers
`numbers.md` §Round 3 — Run 4a, on branch `dan_round3-run4a` of `dan-pandori/nd-takehome` (read
the review first: the executor's files were **not** edited after the review, so they still carry
the wording the verdict asks to change). Brief: `BRIEF_scale.md` (this run is its **4a** half;
4b, the depth-3 / neighbour half, is a separate run still in review and is not used here).

Question: every reductio result so far is on a 3.2M-parameter model. At ≈ 25M and ≈ 85M, trained
on a cap-6 set with zero strict-reductio proofs: (1) how often does a Stage-1 draw emit the
strict `AS ( ~ G ) … NEGI → DN` shape before any RL, and at what rate; (2) do zero-rate draws
still stay at 0 under expert iteration ("or nothing"); (3) what fraction of the targets EI
acquires is unreachable by the base at pass@10⁴ (the EI-only fraction), and does it change with
size?

## Configuration

- **Code and data revision.** `dan-pandori/nd-takehome`, branch `dan_round3-run4a`, merge-base
  `e1d6a3b`; the run changed no core code (adds only `pod/r3_4a/*`, `r3_4a_*.py`, `run4a_*.py`).
  Pre-registration
  [`preregistration/round3-run4a.md`](https://github.com/dan-pandori/nd-takehome/blob/f9e1ce2/preregistration/round3-run4a.md)
  committed 17:55:56Z (`da5bdd8`; clock-text fix `defddcb` 17:56:06Z); first pod `r4a-1` created
  17:56:44Z (`~/pods.log`) — gate 0 holds, by 48 s (reviewer). Final executor commit `a39c9b2`.
  `nd_verify/verify.py` blob `1cfed53b…` = `origin/main`; `artifacts/TEST_RUN_DONE` blob
  `1d5cf064…` untouched; no evaluation or test file read by training code; no test-file run.
- **Training set (deviation, decided before any pod; `QUESTIONS.md` 17:55).** Run 5's
  `train_reductio_f0.jsonl` exists only on the first host. Rebuilt with the same assembler
  (`make_coverage_sets.py assemble --patterns reductio --freqs 0`, seed 41) from five
  generator-made cap-6 sets (`train_depth3_f0_a1` + bucket `round2/run2/data/r2/train_r2_{struct,
  impi_ore_f0,negi_ande_hyp_f0,ori_ore_f0}`; pool script `r3_4a_pool.py`):
  `data/r3_4a/train_reductio_f0_b1.jsonl`, 155,000 = 31,000 per length 2–6. Reviewer: 0 verifier
  failures on all 155,000, max 6 lines, every `(prompt, proof)` pair found verbatim in a
  generator set (unaccounted 0), strict predicate fires on **0 / 155,000** (pruned and written
  form), renaming-class overlap **0** with targets, transfer, held-out, validation-36 and every
  earlier reductio pool. Because the set differs from run 5's, three **same-set 3.2M draws**
  were added so size is not confounded with set.
- **Models** (`train.py --mode abs --cap 6 --bs 128`, seeds 0–2 per cell): **3.2M**
  (3,210,240; 4 × 256, campaign command, lr 1e-3, 6,000 steps; tags `m3_*`); **25M**
  (25,321,472; `--n_layer 8 --d 512 --n_head 8`); **85M** (85,208,064; `--n_layer 12 --d 768
  --n_head 12`). Config **A** = `--steps 6000 --lr 3e-4 --min_lr 3e-5 --warmup 500` (`m25_*`,
  `m85_*`); config **B** = the pre-registered retry, lr 1e-4 / 12,000 steps (`m25B_*`, `m85B_*`).
  Both configurations were run in full at both sizes (3 draws each); by the pre-registered rule
  (better seed-0 held-out greedy) **B is the pre-registered cell at both sizes** and A is a
  second complete cell. Four more 85M config-A draws (s3–s6) are **exploratory, coverage only,
  not pre-registered**, added at 22:40 after A was known to be 3 / 3. 19 checkpoints in all.
- **Per draw.** Pre-RL: `coverage.py --k 2000 --temperature 0.8 --seed 0` on the 300 targets of
  `data/p2/targets_reductio_req.jsonl` (600,000 samples; strata 7 / 8 / 9 / 10 lines = 52 / 133 /
  82 / 33). EI: `expert_iter.py --k 32 --rounds 8 --temperature 0.8 --retain 20000 --max_per_thm
  4 --ft_steps 600 --seed <draw seed>`, `--ft_lr` 3e-4 (3.2M) / 1e-4 (25M, 85M; the brief
  suggested 3e-5 at 85M, `QUESTIONS.md` 17:55), transfer `transfer_reductio_req.jsonl` (150).
  Frozen twin: same command with `--no_train` (256 attempts per target). Base reachability:
  `coverage.py --k 10000 --seed 1` on the EI-acquired targets only (economy deviation from the
  brief's 3·10⁶, pre-registered). *Acquired* = ≥ 1 verified proof whose start-index-normalised,
  dependency-pruned form is strict reductio; *ignition round* = first round with ≥ 6 / 300.
  Exploratory arms (expectations committed before their results, `8ccd6cb`, `9663b4f`):
  `x_ei_m85_s2_r9-16` (rounds 9–16 on a draw at 0 after 8) and `x_ei_m85_s0_lr3e-5`.
- **Mid-run amendment.** The rule "B replaces A if seed-0 held-out improves by > 0.01" is not in
  the pre-registration; it was committed (`ae7f662`, 18:54:56Z) before any B result existed and
  selects the same cell as the registered rule. No effect on any number (reviewer).
- **Pods.** Six A100-SXM4-80GB at $1.59/h (pre-registered: two, a third if needed; the extra
  pods are logged with reasons — co-tenancy OOMs, the B cells): `r4a-1` 17:57–21:51, `r4a-2`
  18:07–22:31, `r4a-3` 19:11–00:18, `r4a-4` 20:09–22:30, `r4a-5` 20:56–22:31, `r4a-6`
  20:57–00:53 UTC = 21.28 pod-hours = **$33.84** (reviewer's sum; creation times match
  `~/pods.log`, deletion times are in `log.md` only; `podls` empty at 01:10Z). Estimate ≈ $29
  (≈ 18 A100-hours), ceiling $50, stop $45.
- **Artifacts.** `hf://buckets/dan-pandori/nd-rl/round3-run4a/{artifacts/r3_4a,ckpts/r3_4a,data/r3_4a}`
  (1,084 / 29 / 18 files, listing checked by the reviewer; token scan of everything uploaded
  clean): 19 Stage-1 checkpoints `stage1_reductio_f0_b1_<draw>.pt`, the round-8 checkpoint of
  every arm that trained, the rebuilt set and its reports, `acq_<draw>.jsonl`, all coverage /
  `found_<r>` / `round_<r>` files and per-theorem held-out greedy outputs. The per-round `mix_*`
  files were not pulled except `ei_m85_s0`'s (checked: RL rows are target prompts only, all
  verify). Figures (executor's, linked not copied):
  [`run4a_size.png`](https://github.com/dan-pandori/nd-takehome/blob/f9e1ce2/figures/run4a_size.png),
  [`run4a_curves.png`](https://github.com/dan-pandori/nd-takehome/blob/f9e1ce2/figures/run4a_curves.png).

## Observations

Observed results first; the reviewer's readings and what is not supported follow.

**Per draw (reviewer's counts; identical to the executor's in every cell).** Rate = strict hits
÷ 600,000. EI-only = acquired targets with 0 verified samples in 10⁴ fresh base samples ÷
acquired. **Every 85M row is "gate missed"** (0.93 held-out not reached in either configuration).

| draw | held-out greedy | pre-RL strict hits (rate) | targets hit | EI acquired, rounds 1…8 | ign. round | 7 / 8 / 9 / 10 | frozen | base-reach@10⁴ | EI-only |
|---|---|---|---|---|---|---|---|---|---|
| m3_s0 | 0.8668 | 0 | 0 | 0 ×8 | – | 0/0/0/0 | 0 | – | undefined |
| m3_s1 | 0.8770 | 14 (2.3e-5) | 3 | 0,1,3,7,8,11,21,32 | 4 | 32/0/0/0 | 2 | 5 | 27/32 = 0.844 |
| m3_s2 | 0.8712 | 0 | 0 | 0 ×8 | – | 0/0/0/0 | 0 | – | undefined |
| m25_s0 (A) | 0.8926 | 1,872 (3.1e-3) | 11 | 4,13,32,38,40,45,45,45 | 2 | 44/1†/0/0 | 7 | 15 | 30/45 = 0.667 |
| m25_s1 (A) | 0.8972 | 830 (1.4e-3) | 10 | 4,20,37,49,52,52,52,52 | 2 | 52/0/0/0 | 7 | 11 | 41/52 = 0.788 |
| m25_s2 (A) | 0.9078 | 14 (2.3e-5) | 3 | 0,1,15,29,36,38,39,40 | 3 | 39/1†/0/0 | 1 | 6 | 34/40 = 0.850 |
| m25B_s0 | 0.9104 | 3 (5.0e-6) | 2 | 0 ×8 (never trained) | – | 0/0/0/0 | 0 | – | undefined |
| m25B_s1 | 0.8972 | 0 | 0 | 0 ×8 | – | 0/0/0/0 | 0 | – | undefined |
| m25B_s2 | 0.9104 | 131 (2.2e-4) | 5 | 0,1,9,31,39,43,44,48 | 3 | 48/0/0/0 | 2 | 10 | 38/48 = 0.792 |
| m85_s0 (A) | 0.8988 | 37 (6.2e-5) | 3 | 1,6,30,39,42,53,53,53 | 2 | 52/1†/0/0 | 2 | 4 | 49/53 = 0.925 |
| m85_s1 (A) | 0.9134 | 49 (8.2e-5) | 2 | 1,11,28,33,42,46,51,52 | 2 | 52/0/0/0 | 2 | 5 | 47/52 = 0.904 |
| m85_s2 (A) | 0.8940 | 6 (1.0e-5) | 3 | 0 ×8 (never trained) | – | 0/0/0/0 | 0 | – | undefined |
| m85B_s0 | 0.9160 | 0 | 0 | 0 ×8 | – | 0/0/0/0 | 0 | – | undefined |
| m85B_s1 | 0.9152 | 4 (6.7e-6) | 3 | 0 ×8 (never trained) | – | 0/0/0/0 | 0 | – | undefined |
| m85B_s2 | 0.9148 | 1,548 (2.6e-3) | 12 | 4,16,30,50,52,52,52,52 | 2 | 52/0/0/0 | 7 | 13 | 39/52 = 0.750 |
| m85_s3…s6 (A, exploratory, pre-RL only) | 0.8962 / 0.9008 / 0.8956 / 0.9022 | 13 / 2 / 60 / 8 (2.2e-5 / 3.3e-6 / 1.0e-4 / 1.3e-5) | 2 / 1 / 7 / 5 | – | – | – | – | – | – |

† Always the same target, `targets_reductio_req_42`, a `negimp_to_pos` instance with an **unused
premise**: its proof is the 7-line shape plus the mandatory extra `PR` line (the pod-side
`pruned` field says 7). **Genuine 8-line acquisition is 0 of 132 in every arm at every size.**
Final validation loss 0.0820–0.0829 in all 19 checkpoints. Run 5's original-set 3.2M row,
quoted from reviewed files and reproduced: s0 113 / 3·10⁶, EI 51, frozen 4, EI-only 45 / 51 =
0.88; s1 and s2 EI 0.

- **Every pre-RL hit, at every size, is on a 7-line target of schema `nand_neg` or
  `negimp_to_pos`** (100 % of 4,591 hits over 19 checkpoints). 0 hits on the 248 targets at 8–10
  lines in **9.42M** samples on those targets (19 × 248 × 2,000; the executor's "11.4M" is the
  total including 7-line targets).
- **EI saturates the 7-line stratum and stops.** Igniting arms (all 8 draws with rate
  ≥ 2.3·10⁻⁵): ignition round 2–4, 32–53 targets, 39–52 seven-liners at 25M / 85M; 9–10-line 0
  everywhere; transfer 18–27 of the 27 seven-line theorems, **0 of 123 longer**. Frozen twins
  0–7 targets, all 7-line, always a subset of what the base also reaches in the pre-RL + b10k
  samples. One 85M arm through 16 rounds is also 49 / 0 / 0 / 0.
- **Zero-rate draws** (0 hits in 600k): `m3_s0`, `m3_s2`, `m25B_s1`, `m85B_s0` — all EI 0 / 300,
  frozen 0 / 300, no accepted proof, so `expert_iter.py` never takes a training step. At 85M
  this is **one draw** (in cell B; vacuous in cell A).
- **Three non-zero but rare draws also end at 0 / 300 with no trained round** (`m25B_s0` 3 hits,
  `m85B_s1` 4, `m85_s2` 6 per 600k): with 8 × 32 × 300 = 76,800 EI attempts the expected hits are
  0.38 / 0.51 / 0.77, so staying at 0 is what the base rate predicts, not an additional finding.
  **Exploratory continuation of `m85_s2`: first accepted proof in round 13, then 1 → 13 → 29 →
  49 targets in rounds 13–16. A draw that read "0 / 300 after 8 rounds" ignites with more
  attempts.**
- **EI-only fraction tracks the draw's base rate, not the size.** 3.2M 0.844 (run 5: 0.88); 25M
  0.667 / 0.788 / 0.850 (A), 0.792 (B); 85M 0.925 / 0.904 (A), 0.750 (B). Over the 8 igniting
  draws Spearman ρ(pre-RL rate, EI-only) = **−0.80** (p = 0.017); the numerator is capped by the
  52-target ceiling, so the fraction is essentially 1 − (base-reachable 7-line targets) / 52 —
  instance-level spread inside one stratum. Adding the 2,000 pre-RL and 256 frozen samples to
  the 10⁴ changes it by at most 2 targets per draw.
- **No rise of the rate with size.** Per-draw rates span three orders of magnitude within each
  size and the ranges overlap almost entirely (0–2.3e-5, 0–3.1e-3, 0–2.6e-3); medians of
  non-zero draws 2.3e-5 (n = 1) / 2.2e-4 (n = 5) / 2.2e-5 (n = 9, configs and extras pooled):
  the ordering is 3.2M ≈ 85M < 25M.
- **Quality gate.** Held-out greedy 25M 0.8926–0.9104 (B s0, s2 pass 0.90; s1 0.8972 does not);
  85M 0.8940–0.9160, **0.93 missed in both configurations**. 679 of the 5,000 held-out theorems
  are reductio-labelled and f = 0 models solve 178–308 of them; non-reductio held-out greedy is
  **0.980–0.992** at 25M / 85M (executor's quoted "0.982–0.988" is wrong; its per-draw table is
  right), 3.2M 0.957–0.971. Best case from these numbers ≈ 0.919, so "unreachable for f = 0
  models" stands as "not reached in 16 large draws and ≈ 0.92 at best".
- **Distinct proofs** (start-index-normalised): one per acquired target in nearly every arm
  (`ei_m85_s0` 2,193 raw records → 53 distinct). Raw record counts must not be quoted as proof
  counts. Oracle: 0 verified-without-pattern samples, 0 pattern-free solves anywhere.
- **ft-lr check (exploratory, one draw):** `m85_s0` at 3e-5: 1,9,28,33,40,46,49,51 (50 + the
  target-42 case) against 53 at 1e-4 — a single-seed observation.
- **Not in the executor's write-up (reviewer).** "Zero-rate" is a property of the draw *on this
  pool*, not absence of the shape from the model: on the 5,000 greedy held-out outputs, 14 of 19
  checkpoints — including zero-rate `m25B_s1` and `m85B_s0` — emit ≥ 1 verified strict
  `AS ( ~ G ) … NEGI → DN` proof (0–11 each), all the degenerate 5–6-line case where a premise
  is literally `( ~ ( ~ G ) )`. What separates zero-rate from non-zero draws is whether that
  composition extends to a 7-line context. "An f = 0 base has never produced the pattern" would
  be a wrong paraphrase.
- **Splits, stricter key.** Identifying premise reorderings as well, 25 held-out and 2
  validation-36 theorems (`modus_ponens`, `modus_tollens`) have a premise-permuted twin in the
  training set (targets and transfer still 0); inherited from the generator's key, moves the
  held-out gate by at most 0.005.

**Pre-registered expectations** (the reviewer's scoring matches the executor's line by line;
"the expectations paragraph is accurate and complete"). Held: P1 (non-zero draws 3.2M 1 / 3;
25M B 2 / 3, A 3 / 3; 85M B 2 / 3, A 3 / 3), P3 (zero-rate ⇒ 0 / 300; n = 1 at 85M), P6, P7, and
"no 10× per step" against the brief's E2. Wrong: **P4's 8-line prediction missed outright**
(≥ 5 eight-line targets in ≥ 2 igniting 85M draws written; genuine 8-line 0 in every arm), its
"≥ 45 of 52 seven-liners" met by 5 of 9 draws (32, 44, 39, 0 miss), and `m85_s2` at exactly
1.0e-5 did not ignite in 8 rounds; **P5 / brief's E4** (85M EI-only 0.3–0.7, median ≤ 0.6
written; 0.750–0.925 observed, all three above 0.7; 25M held); P2's rate band (25M-A median
1.4e-3 and 85M-B 1.3e-3 above 10⁻³). P5's "reversed" criterion is nominally met on cell A
(median 0.914, n = 2) and **not** on the pre-registered cell B (0.750, n = 1); given ρ = −0.80
with base rate the reviewer reads it as no size effect in either direction, and the executor
correctly claims none. Cost: ≈ $29 written, $33.84.

**Rewordings the verdict requires** (the executor's files still have the left-hand side):

1. "'Or nothing' … applies to ≈ 1 draw in 10 instead of 2 in 3", `STATUS.md` "less often",
   "Bigger bases emit the pattern more often" → "zero-rate draws were 2 / 3 at 3.2M, 1 / 6 at
   25M and 1 / 6 at 85M (1 / 10 with four exploratory draws; B 1 / 3 vs A 0 / 7, so it depends
   on the Stage-1 recipe); the direction favours fewer zero-rate draws at scale but is not
   established (Fisher p = 0.11 pooled, 0.23 on pre-registered draws). Counting outcomes instead
   — no accepted proof in 8 × 32 — 'nothing' happened in 2 / 3, 2 / 6 and 3 / 6 draws." The
   1 / 10 pools two configurations with four draws added to the non-headline one after it was
   known to be 3 / 3.
2. "'Or nothing' survives" → say what the data bound: a 0-in-600k draw has rate < 5·10⁻⁶
   (95 %), for which 76,800 attempts give < 0.4 expected hits; three non-zero draws also got
   nothing and one ignited in rounds 13–16. Supported: **"nothing within the attempt budget
   unless the base emits the shape on the pool at ≳ 2·10⁻⁵"**; no zero-rate draw has been run
   long enough to call it "never".
3. "At most one of 133 eight-line targets" (and P4's scoring) → "0 genuine 8-line targets (the
   one counted is a 7-line proof of `targets_reductio_req_42`, which has an unused premise)" —
   the true statement is stronger than the one made.
4. `numbers.md`: non-reductio held-out 0.982–0.988 → 0.980–0.992; "0 hits … in 11.4M" → 9.4M
   samples on the 248 longer targets.
5. "No sensitivity" to ft-lr and "size did not move the wall" carry their n: one draw; 8 rounds
   × 32, this pool, ≥ 2 igniting seeds per new size (nothing here tests more rounds after
   saturation).

**What stands** (reviewer): (1) at 25M and 85M as at 3.2M, every pre-RL hit and every EI
acquisition lies in the 52-target 7-line stratum of two schemata; EI saturates it and acquires
no genuine 8–10-line target at any size; (2) zero-rate draws exist at every size including 85M
(n = 1) and, never training, stay at 0 / 300 within 8 × 32; (3) the EI-only fraction does not
fall with size and is governed by the base rate; (4) no ≥ 10× rate increase per size step;
(5) the 0.93 gate is missed at 85M for the stated structural reason. **Not supported:** any
quantitative size trend in the non-zero fraction or the median rate. Nothing else — the
write-up does not overclaim on EI-only, ignition or the wall.

## Limitations and next step

- Three pre-registered draws per cell; two Stage-1 configurations that differ in their zero-rate
  fraction at 85M (B 1 / 3 vs A 0 / 7, Fisher p = 0.30), so recipe and size are not separable;
  the four extra draws are coverage-only and were added to the cell already known to be 3 / 3.
- The training set is a rebuild, not run 5's; the same-set 3.2M control (1 / 3 non-zero, EI-only
  0.84) agrees with run 5's row (1 / 3, 0.88) at n = 3.
- 85M never clears the brief's 0.93 gate; every 85M row is "gate missed (structural)".
- Eight rounds × 32; saturated arms had only 2–3 rounds after saturation; base reachability at
  10⁴ was measured on acquired targets only, not all 300.
- Six pods against a pre-registered two–three, with three co-tenancy OOM incidents and one
  stale-pull overwrite (25M-A s0 / s2 coverage files, restored; all 27 coverage files complete
  and consistent with the surviving pod logs). Spend is not derivable from repository files
  (deletion times are log-only). Splits are disjoint with premise order fixed, not up to
  premise permutation.

Next measurements (reviewer, in order): (1) *is "nothing" only a budget effect?* — base
sampling on the **52 seven-line targets only**, 10⁵–10⁶ per target, on the four zero-rate and
two remaining low-rate checkpoints (in the bucket; 5.2M samples ≈ 1 A100-hour at 85M), then
pre-register the first-hit round from the measured rate and test it on one such draw at
k = 256; (2) *is the 7-line wall a rounds effect?* — continue one 25M and one 85M saturated arm
to round 32 with the 52 saturated targets removed from sampling, frozen control at equal
attempts; (3) *recipe vs size for the non-zero fraction* — ≈ 10 coverage-only draws per cell in
one fixed configuration at each size (≈ $1 per 85M draw), only if the fraction matters to the
SPAR answer.
