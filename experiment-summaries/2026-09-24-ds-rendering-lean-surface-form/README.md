---
written_on: 2026-09-24
written_by: agent:claude
---

# ds-rendering: does the Lean surface form change what a from-scratch model learns?

Status: completed 2026-09-24 04:17:54 UTC (executor session, six A40 pods plus one aborted RTX
3090, **32.69 of 36 pod-hours, $16.01 of an $18 budget**) and independently reviewed 2026-09-24
04:20–04:42 UTC. Review verdict: **no hard-constraint violation, no quarantine**. The reviewer
re-derived every number in `numbers.md` with code they wrote themselves — their own renaming-class
canonicaliser, box-depth counter, start-index normaliser, `L*` definition, depth-3 acquisition
predicate, and a 1,250-proof re-verification through `nd_verify`, the unmodified `nd2lean.py` and
Lean on the literal sampled text — and reports **"there is not one arithmetic disagreement in the
run"**. The findings are about **scope and inference, not arithmetic**: six statements in the
write-up outrun the evidence and are reworded below, one pre-registered experiment (E9, the frozen
ladder) was cut for budget and has no data in any arm, and the committed `summary.json` asserts a
hypothesis the run itself falsified. Numbers and wording below are the reviewer's wherever they
differ from the executor's. Run id `ds-rendering`; proposal 10 (dataset styles).

Review:
[`review_ds-rendering.md`](https://github.com/dan-pandori/nd-takehome/blob/f9b22a3/review_ds-rendering.md)
(commits `6122434` phase-1 blind recount, `f9b22a3` phase-2 comparison and verdict; reviewer code
`review_dsr/rv_*.py`, committed beside the review). Executor write-up:
[`run_ds_rendering.md`](https://github.com/dan-pandori/nd-takehome/blob/f9b22a3/run_ds_rendering.md)
(≈ 330 words, two figures), numbers
[`numbers.md` § ds-rendering §§ 1–9](https://github.com/dan-pandori/nd-takehome/blob/f9b22a3/numbers.md),
log `log.md` § ds-rendering, figures `figures/ds_rendering_lengths.png` and
`figures/ds_rendering_readiness.png`, rendered example `examples/ds_rendering_four_modes.md`
(five modes of the same proof), all on branch `dan_ds-rendering` of `dan-pandori/nd-takehome`
(the executor's files were **not** edited after the review; read the review first).
Pre-registration:
[`preregistration/ds-rendering.md`](https://github.com/dan-pandori/nd-takehome/blob/f9b22a3/preregistration/ds-rendering.md)
(`bcb0d06` 21:15:54 UTC; addendum 1, the R4 arm, `3a6b390` 21:45:02; addendum 2, the seed sweep,
`14327f9` 21:58:38). Brief: `BRIEF_ds-rendering.md`. Questions for Dan: `QUESTIONS.md`, three
entries dated 2026-09-23.

Question: proofs, model, training schedule, data and evaluation pools all fixed; **only the Lean
rendering varies**. Does that change held-out accuracy by length, and does it change *RL
readiness* — the base rate on required pools, the expert-iteration increment over a frozen control
at equal attempts, and how far up a true-length ladder eight rounds of EI reach? And is the
project's standing **"cap + 1" length horizon a property of the text the model writes, or of the
ND proof's structure?**

## Configuration

- **Five arms, one change each, everything else identical.** All five are **3.2 M parameters**
  (4 layers, d 256, 8 heads), trained **from scratch**, 6,000 steps, bs 128, lr 1e-3 → 1e-4,
  cap 6, on the **same 155,000-record ND set** `data/p2/train_depth3_f0_a1.jsonl` (depth-3 proofs
  removed, f = 0), in a `lean_seq`-family Lean surface form with a random first-appearance name
  offset. The arms differ **only** in the rendering mode of `lean_tok.py`:

  | arm | mode | what changes | mean proof tokens, ratio to C0 | vocab |
  |---|---|---|---:|---:|
  | **C0** control | `lean_seq` | — | 82.6, **1.000** | 107 |
  | **R1** | `lean_seq_noprem` | no premise re-statement lines | 58.3, **0.706** | 107 |
  | **R3** | `lean_seq_nofml` | formula-free `have`s on `IMPE ANDE1 ANDE2 NEGE R` | 78.9, **0.955** | 107 |
  | **R2** | `lean_seq_intro` | boxes as `by intro` instead of a lambda | 75.4, **0.913** | 108 |
  | **R4** | `lean_seq_funbare` | `fun` kept, the binder's type dropped (addendum 1) | 75.4, **0.913** | 107 |

  R4 is **exactly** token-matched to R2 and vocabulary-matched to C0, as addendum 1 claims
  (reviewer's own 3,000-record re-run). C0 seeds 0/1 are lean-format's two Stage-1 checkpoints
  (`stage1_a1_seq_s{0,1}.pt`), re-measured here; **every other model was trained in this run**.
- **Seeds.** Seeds 0 and 1 through the full plan on all five arms; **seeds 2–5 through Stage-1 and
  held-out greedy** on C0, R2, R3, R4 (addendum 2; R1 excluded, its two seeds agree to 0.01 and it
  is worse than C0 everywhere) — **26 models in all, n = 6 per arm** in the depth-3 cell.
- **Acceptance is the conjunction, everywhere.** A sample counts only if **`nd_verify` accepts the
  denoted ND proof and Lean 4.34 accepts the literal sampled text**, through `lean_gate.py`, at
  every stage. **83,513** distinct counted proofs (C0 19,004; R4 16,958; R3 16,743; R1 15,783;
  R2 15,025).
- **Sampler, fixed in every arm and every stage:** `path=fast, early=eos, compact=True,
  rowrng=True`, **batch 2048**, `max_new` 400, T 0.8 — the configuration the `efficiency` run
  recommended.
- **Stages per arm** (run in the order 1 → 2 → 3 → 4 → 5, so a budget cut loses the ladder first
  and never a base rate): Stage-1 training → held-out greedy (`data/p2/heldout.jsonl`, 5,000,
  k = 1, T = 0) → mechanism pass@16 (`data/transfer.jsonl`, 1,638) → coverage pass@2,000 on three
  pools (`p2/targets_depth3` 1,000; `r3_1/depth3_req` 300; `p2/targets_reductio_req` 300) → dial
  `expert_iter.py --rounds 4 --k 32 --temperature 0.8 --batch 2048` with a `--no_train` frozen
  control at equal attempts → ladder `ladder_ei.py` 8 rounds × k 32 on `ladder/rl_targets.jsonl`
  (4,495) read out on `ladder/transfer.jsonl` (2,285).
- **Hard constraints (all pass, reviewer's own checks).** `nd_verify` tree
  `9437bb72…` identical to `origin/main`; `artifacts/TEST_RUN_DONE` blob `1d5cf064…` identical to
  `origin/main` (still the 2026-09-15 single test run); `nd2lean.py` blob `16676091…` identical to
  the branch point `origin/dan_lean_format`; cap 6; **no evaluation file read for gradient or
  model selection** — `--heldout data/p2/heldout.jsonl` is passed but `train.py:93,116-121` uses it
  only for a logged validation loss on the first 2,000 records, and the last step is saved
  unconditionally (noted, not a violation).
- **Gate 0 passes three times** (reviewer, against `~/pods.log`): main pre-registration 21:15:54Z
  → first pod `dsr-c0` 21:19:20Z (+3 m 26 s); addendum 1 21:45:02Z → `dsr-r4` 21:45:53Z (+51 s);
  addendum 2 21:58:38Z → `dsr-s` 21:59:27Z (+49 s).
- **Splits, recomputed by renaming class** with the reviewer's own canonicaliser (which separates
  all 155,000 training records into 155,000 distinct classes, so it is not over-merging): training
  set × `heldout`, × `targets_depth3`, × `depth3_req`, × `targets_reductio_req`, × `transfer_depth3`,
  × `ladder/transfer` = **0** shared classes each; training set × `data/transfer` = **7** (see
  Limitations); `ladder/rl_targets` × `targets_depth3` / `depth3_req` / `targets_reductio_req` =
  6 / 2 / 16 (not checked by the run's own `dsr_splits.py`; they affect no number in this run,
  every coverage number here being on a Stage-1 model).
- **Deviations, each logged with a reason before acting.** (a) **A40 (46 GB, $0.49/h), not
  RTX 3090** — RunPod had no 3090 capacity once the two sibling dataset-styles runs had taken
  theirs; the first `dsr-c0` was created on a 3090 at 21:19, deleted at 21:26 and recreated on an
  A40 (0.0375 h, $0.02) so that all arms including the control share one GPU class. (b) **A fifth
  arm, R4**, because R2 changes two things at once. (c) **A Stage-1 seed sweep to n = 6**, because
  the depth-3 cell has a seed spread of 0.55 on the control. (d) **The frozen ladder (E9) cut from
  all five arms** for budget — see Limitations. (b), (c) and (d) are all in `QUESTIONS.md` or
  `log.md` with the arithmetic and the alternative considered.
- **Pods and cost.** Six A40 48 GB $0.49/h — `dsr-c0` 6.2119 h, `dsr-r1` 6.0619, `dsr-r3` 6.2333,
  `dsr-r2` 6.8819, `dsr-r4` 5.4628, `dsr-s` (sweep) 1.8033 — plus the aborted RTX 3090 `dsr-c0`
  0.0375 h: **32.69 pod-hours, $16.01** (`podbudget ds-rendering`, `~/podhours.log`), against a
  $18 budget and a 36 h ceiling. `~/runs/ds-rendering/BUDGET_WARNING` fired at 28.84 h. All
  `dsr-*` pods deleted (`podls` shows only the sibling `ds-generator` pods). Wall clock
  2026-09-23 21:01 → 2026-09-24 04:18 UTC.
- **Artifacts.** `hf://buckets/dan-pandori/nd-rl/ds-rendering/{artifacts,ckpts,data,figures}` plus
  `numbers.md`, `preregistration.md`, `run_ds_rendering.md`. **`numbers.md` § 9 Cost does not
  record the bucket paths** — a missing deliverable the reviewer flags, one line to fix.

## Observations

All values are the reviewer's independent recount; they equal the executor's `numbers.md` to the
digit everywhere, so only differences in *wording and scope* are called out.

### What stands

**1. Rendering does almost nothing in distribution.** On the 4,500 held-out theorems whose
reference proof has box depth ≤ 2, the five renderings span **2.28 pp** (C0 0.9615, R2 0.9563,
R4 0.9526, R3 0.9387 at n = 6; R1 0.9462 at n = 2). Solid at n = 2 and still true at n = 6. (It
narrowly misses pre-registered E19's 2 pp band; C0/R2/R4 are within 0.9 pp of each other and R3 is
the outlier.)

**2. The whole visible difference between renderings is zero-shot depth-3 composition — and that
difference is bookkeeping, not logic.** `data/p2/heldout.jsonl` has 500 theorems whose reference
proof is box depth ≥ 3, and the a1 training set contains **zero** depth-3 proofs, so that slice is
f = 0 composition. Across the 24 sweep models, Spearman ρ between an arm-seed's depth-3 rate and
its count of **`LEANPARSE` grammar violations at depth 3** is **−0.933** (−0.943 over all 26).
Pre-registered E20 holds. **This is the run's most robust mechanistic finding and it is under-sold
in the write-up.**

**3. The control wins the ladder.** Eight rounds of EI on the ladder pools, read out on 2,285
never-trained transfer theorems labelled by true minimal length:

| arm | seed 0 solved (`L*`) | seed 1 solved (`L*`) | mean solved |
|---|---|---|---:|
| **C0** control | **923** (**12**) | **1003** (11) | **963** |
| R1 | 837 (11) | 945 (11) | 891 |
| R3 | 898 (11) | 698 (11) | 798 |
| R4 | 771 (11) | 896 (11) | 833 |
| R2 | 658 (**10**) | 800 (11) | 729 |

C0 is **first of five on both seeds** — under arm exchangeability, permuting arm labels within
each seed, p ≈ **0.040** — and owns the only `L*` = 12. **No rendering variant buys RL readiness.**
This is the run's headline and it survives. The paired ranking is what carries it: the ANOVA is not
significant (F(4,5) = 1.69) and the within-arm seed spread (R3 698 vs 898, R2 658 vs 800) is larger
than every between-arm difference, so the honest summary is *"no arm beats C0, and none is shown to
differ from it."*

**4. The arm that leads at Stage 1 need not lead after RL.** R2 has the best frozen dial (0.406),
the best Stage-1 depth-3 mean (0.661 at n = 6) — and the **worst ladder mean** (729, ×0.757 of C0).
Neither endpoint is individually significant at n = 2, but the *reversal* is a real warning and the
reviewer calls it **the most transferable thing the run produced**: a rendering picked on a Stage-1
number would have been picked exactly wrong.

**5. Lean alone is not a sufficient checker for this ND rule set.** Over **16,300,481** distinct
(theorem, Lean text) pairs gated across all six pods: `nd_verify`-accepts-and-Lean-rejects **0**;
Lean-accepts-and-`nd_verify`-rejects **2,208** (99.9865 % agreement). Disagreement is entirely
one-directional — **Lean never once caught what `nd_verify` missed**. Named mechanism: Lean's `¬a`
is *definitionally* `a → False`, so `nd2lean.py:135`'s `BOTE` → `nA.elim` resolves to
`Not.elim : ¬a → a → b` rather than `False.elim` whenever the goal happens to be `a → …`
(1,645 cases, 74.5 %; then 315 fewer-`PR`-lines cases, 179 `NEGE`, 69 unclassified). The rate is
**rendering-dependent**, per 100k pairs: R1 **1.9** < C0 **4.5** < R4 **7.8** < R2 **15.0** < R3
**39.6** — R3 is 8.8× the control, because dropping the stated formula leaves the decoder nothing
to contradict when the model misapplies a rule. Acceptance here is the conjunction, so **no counted
proof in this run is affected**; but this is a finding about the **repository**, correctly raised in
`QUESTIONS.md`, and the reviewer's conclusion is that **any past or future claim in this project
that rests on a Lean pass alone is not safe.**

**6. Checker of record: clean.** Pre-registered E11 (≤ 100 `nd_verify`-ok/Lean-rejected per million)
holds at **0.0 per million in every arm**, against lean-format's 33. All 83,513 counted proofs were
re-checked on the pod before deletion — 100 % accepted by the unmodified `nd2lean.py --check` and
by Lean on the literal text, zero rejections. The reviewer re-drew **250 counted proofs per arm
(1,250 total)** across every counting stage with their own seed: 1,250 / 1,250 `nd_verify`-ok,
1,250 / 1,250 Lean-ok via `nd2lean.py`, 1,250 / 1,250 literal texts Lean-ok, 0 translation
failures, 0 disagreements; on 40 translated proofs, no `sorry`, zero Lean errors, and `#print
axioms` reports either none or `[propext, Classical.choice, Quot.sound]` — the classical axioms
`DN` legitimately needs.

**7. No arm satisfies proposal 10's decision rule, and the control set is not replaced.** The rule:
*an arm replaces the control iff, on both seeds, held-out greedy is within 1 pp of C0 or better and
at least two of {depth-3 pass@2,000, reductio pass@2,000, ladder `L*` or solved} improve with none
worse.* R1 fails on held-out (−3.2 / −2.5 pp); R3 fails (−6.1 pp on s1); R4 fails (ladder solved
worse on both seeds); **R2 passes held-out and depth-3 but fails "none worse"** — reductio
pass@2,000 ×0.42 / ×0.73 and ladder solved 658 vs 923 and 800 vs 1003. R3's separate "qualifies"
clause (falsifier 4) also fails. The reviewer agrees with the run's conclusion.

### The rendering effect the run was built to measure is not established

**Finding R-2 (major).** At n = 6, **no arm's depth-3 held-out slice is distinguishable from C0's**.

| arm | n | depth-3 slice, per seed | mean | across-seed **sd** |
|---|---:|---|---:|---:|
| C0 | 6 | 0.486 0.274 0.824 0.502 0.848 0.262 | 0.533 | **0.256** |
| R2 | 6 | 0.822 0.746 0.742 0.846 **0.034** 0.778 | **0.661** | **0.310** |
| R3 | 6 | 0.674 0.024 0.788 0.038 0.808 0.632 | 0.494 | **0.365** |
| R4 | 6 | 0.788 0.440 0.808 0.754 0.480 0.104 | 0.562 | **0.276** |
| R1 | 2 | 0.232 0.218 | 0.225 | 0.010 |

Exact two-sided permutation tests on the six seeds per arm (all 924 splits), pooled within-arm sd
**0.302**:

| comparison | difference in means | exact p |
|---|---:|---:|
| R2 − C0 | +0.129 | **0.452** |
| R2 − R4 | +0.099 | 0.617 |
| R4 − C0 | +0.030 | 0.829 |
| R2 − R3 | +0.167 | 0.272 |
| R1 − C0 (n = 2 vs 6) | **−0.308** | 0.072 |

- **E17 missed by half**: the pre-registered ordering R2 > R4 > C0 ≈ R3 does hold
  (0.661 > 0.562 > 0.533 > 0.494) but the required gap mean(R2) − mean(C0) ≥ 0.25 came out at
  **0.129**.
- **E18 falsified**: R2's sd was predicted to be the smallest of the four and < 0.10; it is
  **0.310, the second largest**, and seed 4 collapses to 0.034. *"The hypothesis that `by intro` is
  the rendering that removes the depth-3 seed variance is dead."* Every rendering in this family,
  `intro` included, is **bimodal across Stage-1 seeds**: 20 of the 26 models land either above 0.44
  or below 0.11.
- Addendum 2's decision rule (|mean(R2) − mean(R4)| ≤ s ⇒ the binder type is what matters) does
  fire, 0.099 ≤ 0.302 — **but so would a rule comparing R2 to C0, which is the comparison the run
  exists to make.** The reviewer's reading: *"the sweep was launched to settle the mechanism and its
  honest answer is that six seeds are still not enough."* The only ordering with any support is that
  **R1 is worse than C0**, and even that is p = 0.072.
- E13/E14 (R4's two disjoint bands) land one seed in each band — **unresolved, and correctly
  reported as unresolved**. E16 (R4 within ×0.8–1.25 of R2, or of C0) is satisfied by *both*
  alternatives on both seeds: **the test as designed has no power**, because R2 and C0 are
  themselves within 1.22× of each other.

### The horizon result: R1 moved it the wrong way

Mechanism test, pass@16 on 1,638 transfer theorems, "distinct" = distinct start-index-normalised ND
proof, binned by the ND length the model **wrote**:

| arm | theorems solved (s0 / s1) | distinct **7-line** | distinct **8-line** | ≥ 9-line | mean Lean tokens |
|---|---|---|---|---|---:|
| **C0** | 906 / 922 | **234 / 265** | **77 / 67** | 7 / 1 | 103.1 / 103.3 |
| R1 | 681 / 855 | 164 (×0.70) / 206 (×0.78) | 48 (×0.62) / 59 (×0.88) | 1 / 4 | 80.8 / 79.8 |
| R3 | 808 / 656 | 187 (×0.80) / 118 (×0.45) | 78 / 30 | 5 / 2 | 104.8 / 101.5 |
| R2 | 891 / 861 | 247 (×1.06) / 206 (×0.78) | 79 / 47 | 2 / 3 | 95.8 / 93.2 |
| R4 | 831 / 903 | 193 (×0.82) / 246 (×0.93) | 46 / 84 | 0 / 7 | 91.9 / 95.2 |

C0 reproduces its pre-registered bands (E3 7-line 130–270 → 234 / 265; E4 8-line 50–140 → 77 / 67).
**R1 missed E3 and E4 badly and in the wrong direction** — ×0.70 / ×0.78 on 7-line against a
pre-registered ×2–3, ×0.62 / ×0.88 on 8-line against ×1.5–2.5 — and solves far fewer theorems
overall. R3, R2 and R4 are within noise of C0 on 7-line except R3 seed 1. **≥ 9-line counts are 0–7
and flip sign between seeds; nothing is derivable from them.** E12 (R1's frontier moves right by
1–2 lines) is **missed**: C0's frontier is 9 / 8, R1's is 8 / 8 — it moves **left or not at all**.

**Term size, re-derived by the reviewer** (the brief asks for term sizes beside line counts). Mean
literal Lean tokens of accepted proofs, by written ND length, both seeds:

| arm | 5-line | 6-line | **7-line** | 8-line |
|---|---|---|---|---|
| C0 | 89.3 / 89.0 | 107.2 / 108.1 | **127.4 / 127.7** | 156.2 / 160.1 |
| R1 | 68.0 / 68.1 | 84.7 / 82.2 | **104.9 / 106.0** | 146.9 / 133.6 |
| R3 | 91.7 / 93.5 | 106.3 / 107.7 | **135.7 / 132.9** | 166.0 / 168.7 |
| R2 | 81.6 / 81.6 | 97.2 / 97.7 | **119.7 / 116.2** | 149.5 / 144.8 |
| R4 | 83.0 / 82.1 | 95.2 / 95.5 | **121.1 / 117.9** | 139.0 / 149.8 |

**R1's token saving is 29 % on the training distribution but only 18 % at the 7-line frontier**
(105 vs 127), because long proofs have proportionally fewer premise lines. *The manipulation the
run's main question turns on is weakest exactly where it was supposed to act.*

### Base rates and the dial

Base rates, Stage-1 models, pass@2,000, T 0.8:

| pool | C0 (s0 / s1) | R1 | R3 | R2 | R4 |
|---|---|---|---|---|---|
| `targets_depth3` (1,000) | **512 / 411** | 393 / 443 | 513 / 355 | 544 / 502 | 496 / 499 |
| `depth3_req` (300) | **175 / 111** | 129 / 113 | 208 / 146 | 218 / 175 | 178 / 162 |
| `targets_reductio_req` (300) | **31 / 26** | 17 / 36 | 24 / 17 | 13 / 19 | 37 / 46 |

- **Large pre-registration misses on the control**, all to be reported as misses: **E5** 0.25–0.50
  → 0.512 (just above) / 0.411; **E6** 0.05–0.25 → **0.583 / 0.370**, a factor of 2–7 above the
  band (`depth3_req` is far easier for these models than the pre-registration expected); **E7**
  5–25 hits → 31 / 26. E6's arm predictions are all *relative* to C0, so the comparisons survive.
- **E5/E6/E7 for R1 all missed with the wrong sign** (−11.9 / +3.2 pp; −15.3 / +0.7 pp;
  ×0.55 / ×1.38 against a pre-registered ×1.5–3). **E5's declared null for R2 fails on one seed**
  (+9.1 pp against ±5 pp) — and the direction is *up*, so **falsifier 2 does not fire and could not
  have**.
- Reviewer's own depth predicate: on `targets_depth3` only 193–350 of the 355–544 hits actually
  carry a depth-≥ 3 proof, i.e. **30–45 % of "depth-3 target" solves are shallow proofs of a
  theorem that happens to admit one**. A claim about depth-3 *composition* should quote
  `depth3_req` or the depth-filtered count. On `targets_reductio_req`, **zero** accepted proofs
  anywhere reach box depth 3.

Dial, EI − frozen at round 4 on 1,000 depth-3 targets, at **equal attempts** (k = 32 × 4):

| arm | EI − frozen (s0 / s1) | vs C0 | E8 (±0.05 of C0) |
|---|---|---|---|
| **C0** | **+0.293 / +0.375** | — | s0 inside the pre-registered +0.20–0.35, **s1 above it** |
| R1 | +0.266 / +0.356 | −0.027 / −0.019 | holds |
| R3 | +0.255 / +0.329 | −0.038 / −0.046 | holds |
| R2 | +0.177 / +0.240 | **−0.116 / −0.135** | **fails both seeds** |
| R4 | +0.240 / +0.304 | −0.053 / −0.071 | **fails both seeds**, marginally |

The mechanism is consistent: R2's and R4's **frozen** rates are the highest (0.444 / 0.369 and
0.377 / 0.348 vs C0's 0.356 / 0.291) and all arms converge to a similar EI endpoint (0.55–0.67), so
the gap closes from below. The reviewer reads this as **Finding 3 ("RL amplifies what the base
does") behaving exactly as stated, not a counter-example**. **Falsifier 3 does not fire**: it needs
an arm with a *lower* base rate than C0 **and** a *larger* EI − frozen on both seeds; R2/R4 have
higher base rates and smaller gaps, R1 has a lower base rate on s0 and a smaller gap. No
counter-example to Finding 3 in this run.

### Where the write-up's wording outruns the evidence (all reworded below)

| write-up claim | reviewer's finding | the form that is supported |
|---|---|---|
| "Expert iteration **compensates**: corr(frozen, EI − frozen) = −0.871" (the headline of the mechanism paragraph and of `log.md` 00:10) | **An artefact of the shared term.** If the EI endpoint were *statistically independent* of the frozen rate, the expected correlation would be **−0.886** (analytic) and the permutation null mean is **−0.904**; the observed −0.871 is *less* negative than the null, permutation **p = 0.70**. The data are exactly what independence predicts. | Drop it. Use the executor's *other* statement of the same fact — **corr(frozen, EI endpoint) = +0.158**, "where a rendering starts predicts almost nothing about where it ends" — and the **acquisition** spread, which is in the artefacts and reproduces exactly: arm means span **12.3 pp frozen** (0.117–0.241, F(4,5) = 2.10) and **3.3 pp after four EI rounds** (0.387–0.420, F = 0.32, between-arm sd 0.014 against a within-arm seed sd of 0.025). After EI the five renderings are *demonstrably* indistinguishable; before it they merely are not distinguished. |
| "At the base, **untyped-binder arms beat the control**" | Arm means are right, but the between-arm spread is **not distinguishable from seed noise**: within-arm seed sd ≈ **0.047**, between-arm **0.045**, one-way **F(4,5) = 1.32** (5 % critical value 5.19); on the EI side F = 0.59. | "R2 and R4 have the highest frozen means; with two seeds per arm the ordering is not resolvable." |
| "the arms converge (sd 0.045 → 0.024)" | True as arithmetic, but **both numbers are at or below the within-arm seed sd** (0.047 frozen, 0.035 EI). | Use the acquisition spread above, which supports the second half of the claim outright. |
| "**The horizon belongs to the ND proof's structure**" | What is shown is that R1 is **worse everywhere**. *"The equally consistent reading is that the premise `have` lines are actively useful — they give the model an explicit, citable copy of each premise — which is a statement about the text, not about the ND proof."* The disjunction was **not tested**. | "Removing the premise re-statement shortens the text and makes the model worse on every measure; the cap + 1 horizon is **not** moved right by shortening the text. Whether that is because length is not the lever, or because the premise `have` lines are themselves useful, **this run cannot say**." |
| "R1 writes the shortest text (**−21 %**)" | **Not derivable from any committed artefact**; "−21 %" names no population. | Name the population: **−29 %** on the training records, **−22 %** on accepted mechanism proofs, **−18 % at the 7-line frontier** — and the frontier figure is the one the argument actually needs. |
| "R2 … is **worst**, ×0.757" | R2 is worst on the mean (729) and on seed 0 (658, last of five); on **seed 1 it is fourth** — R3's 698 is lower. | "Worst on the mean and on seed 0; fourth of five on seed 1." |
| Falsifier 1 ("dead if R1's distinct 7-line count is within ±25 % of C0 on both seeds") | Seed 0 is **−30 %** (outside), seed 1 −22 % (inside), so the falsifier **does not fire — because R1 is below the window, not inside it**. | Do not report falsifier 1 as "not triggered". The pre-registered prediction (×2–3) is **refuted far more strongly** than the null the falsifier was written to catch: R1 moved the frontier in the *opposite* direction. |
| "E8 holds on both clauses / C0's mean is +0.334, inside the band" (`log.md` 23:50) | The pre-registration scores E8 **per seed**, and C0 s1 is **+0.375**, above the band. | Partly self-corrected already (`log.md` 00:10 scores R2/R4 correctly, `numbers.md` § 6 prints both seeds); the write-up does not mention E8 at all. |

### Two things a reader of the committed artefacts alone would get wrong

**Finding R-5 (important, reproducibility) — must fix before anyone else reads the run.**
`numbers.md` opens with *"Every number is re-derived by `python3 dsr_analysis.py` →
`artifacts/dsr/summary.json`"*. The reviewer re-ran it: its `seed_sweep` block reports
**`n_seeds: 2` for every arm**, because `dsr_analysis.py:29` looks for `held_*.jsonl` in
`artifacts/dsr/<arm's pod>/` and `artifacts/dsr/` while the sweep's 16 files are in
`artifacts/dsr/dsr-s/`.

| arm | committed `summary.json` (n = 2) | the truth at n = 6 |
|---|---|---|
| C0 | depth-3 mean 0.380, **sd 0.150** | 0.533, **sd 0.256** |
| R2 | 0.784, **sd 0.054** | 0.661, **sd 0.310** |
| R3 | 0.349, sd 0.460 | 0.494, sd 0.365 |
| R4 | 0.614, sd 0.246 | 0.562, sd 0.276 |

At n = 2 R2's sd (0.054) **is** the smallest and below 0.10 — so **the committed `summary.json`
says E18 holds, the exact opposite of the run's own finding and of the write-up's "What I got
wrong."** Worse, **`numbers.md` has no seed-sweep section at all**: the 24-model sweep that
addendum 2 pre-registered (E17–E20) and that *changed the run's conclusion* appears only in
`log.md` and in one clause of the write-up. Fix: add the n = 6 table to `numbers.md` as its own
section, and add `artifacts/dsr/dsr-s` to `dsr_analysis.py`'s search path.

**Finding R-6 (important).** The depth-3 panel of `figures/ds_rendering_readiness.png` — the panel
carrying the run's most eye-catching contrast (R2 0.822 / 0.746 against C0 0.486 / 0.274) — plots
**only seeds 0 and 1**, i.e. precisely the n = 2 picture the run has already shown to be a seed
lottery. It should add the four sweep seeds or carry the n = 6 mean ± sd beside it.

### Reporting discipline

The reviewer calls the run's protocol discipline **"the best I have reviewed in this repository"**
— one sampler batch everywhere, the literal Lean text stored beside every counted proof, the
checker of record re-run on every pod before deletion — and the reporting of misses **exemplary**:
`log.md` 23:31 scores E17 "half right" and **E18 "falsified, and it was my leading hypothesis"**;
00:10 corrects an over-generalisation the executor had made 20 minutes earlier; 00:35 and 04:20
correct a per-arm multiplier twice (18× → 11.8× → 8.8×) and wire the figure to regenerate; the
write-up carries a "What I got wrong" section. Every number in `numbers.md` is sourced to a file
and every file says what the table says.

## Limitations and next step

- **Finding R-4 (major, scope): the frozen ladder was not run for any arm, so E9 has no data and
  no ladder claim has an equal-attempts frozen control.** All ten `la_frz_*` directories contain
  only `args.json`. The pre-registered drop order was "R2's ladder, then R4's ladder, then R2, then
  R3's ladder"; what was actually dropped is the **frozen half of every arm's ladder**, which is not
  on that list and is the half that carries the baseline. The executor handled the cut well —
  `log.md` 00:20 and 01:20 record the deviation **before acting**, with the measured round time
  (1,376 → 1,704 → 1,717 s, plateauing ≈ 28.5 min, ≈ 25 % above the estimate the first cut was made
  on), the arithmetic, the alternative considered (truncate `la_T1` to 6 rounds) and the reason for
  rejecting it — and `numbers.md` § 7 prints `– / –`. The remaining gap is in the **write-up**,
  which mentions neither the cut nor any base number beside "ladder, 8 rounds (/2,285): C0 963",
  while `AGENT_POLICY.md` requires *"a base-model reachability number with every 'RL solved X'"*.
  The nearest available base number, which the reviewer computed: **round 1 of each T1 ladder
  samples the Stage-1 checkpoint before any training and solves 59–146 of 2,285 (2.6–6.4 %) at
  `L*` = 9, in every arm** — at 32 attempts, not the ladder's 8 × 32. So the ladder gain is large
  and real, but it is **not** the attempt-matched comparison the policy requires, and every
  "the ladder reached `L*` 12" sentence must be published with this caveat and with the base number
  beside it.
- **Finding R-1 (minor, inherited): `data/transfer.jsonl` is not fully disjoint from the training
  set.** 7 of its 1,638 theorems (`transfer_{81,102,493,703,900,1115,1590}`) are training records up
  to renaming, and the training copies carry **shorter** proofs (2–6 lines) than the pool's labels
  (7–16). That is 0.43 %, and the mechanism counts bin by the length the model *writes*, so the
  effect on E3/E4 is at most a few proofs — but the pool is described in the pre-registration as
  held out and it is not entirely. The overlap is **inherited**: `data/transfer.jsonl` and the
  ladder pools are byte-identical to lean-format's and ladder-A's (the sha256s the pre-registration
  names), so this run did not introduce it and **the same theorems are in every earlier run that
  used this pool**. Separately, `ladder/rl_targets` shares 6 / 2 / 16 renaming classes with
  `targets_depth3` / `depth3_req` / `targets_reductio_req` — not checked by `dsr_splits.py`,
  harmless here (every coverage number in this run is on a Stage-1 model), and **worth adding to
  the standing splits check for future runs**.
- **Two cheap must-fixes, neither of which changes a conclusion** (reviewer): (1) add
  `artifacts/dsr/dsr-s` to `dsr_analysis.py:29`'s search path so `summary.json`'s `seed_sweep` block
  reports n = 6 and stops asserting the falsified E18, and add the n = 6 table to `numbers.md` as
  its own section; (2) record the bucket paths in `numbers.md` § 9.
- **Everything that separates the arms rests on a bimodal, high-variance cell.** The pooled
  within-arm sd of the depth-3 held-out slice is 0.302, and 20 of 26 models land either above 0.44
  or below 0.11. **What is not supported and should not be repeated:** any claim that R2's `by
  intro` rendering is more seed-stable, or that any arm's depth-3 held-out rate differs from the
  control's.
- **R4 is the only arm that isolates one variable cleanly**, and its own mechanism question
  (binder-type repetition vs `fun` syntax) is **unresolved**, one seed in each pre-registered band
  at n = 2 and not separable at all at n = 6.
- **Every ladder and dial number is two seeds.** The n = 6 sweep covers Stage-1 and held-out greedy
  only.

**Next measurement (reviewer), ≈ $8.50 and ≈ 17 pod-hours.** The run's own question — does the
rendering change RL readiness — is answered **no** by the ladder, and that answer is safe because it
is a paired ranking over five arms. What is *not* settled is the mechanism behind the depth-3
held-out gap, and this run shows exactly why: **that cell needs far more seeds, not more arms.** At
the measured pooled within-arm sd of 0.302, a two-sample test has 80 % power to detect a difference
of **0.489 at 6 seeds per arm**, 0.268 at 20, 0.189 at 40, and **0.129 — the observed R2 − C0 gap —
only at ≈ 87**. So the honest options are **{C0, R2} × ≈ 85 seeds, Stage-1 and held-out only** — 170
models, ≈ 17 pod-hours, ≈ $8.50 at the sweep's measured $0.05 per model, nothing else in the design
changing — **or to stop asking**. Note also that because the distribution is bimodal rather than
normal, the cleaner estimand is **the probability that a seed lands in the high mode** — C0 4/6,
R2 5/6, R4 4/6, R3 3/6 — reported as a proportion with a Wilson interval rather than a mean.

**Second and larger, ≈ $2.50 for one arm × two seeds: the ND-structure-vs-text disjunction that R1
could not separate.** R1 removed the premise lines *and* whatever value those lines carry as
citable, formula-bearing intermediate steps. The separating arm keeps **a premise line per premise
that carries no information the statement does not already give** — e.g. `have n1 := h1` with no
formula, R3's treatment applied to `PR` lines only. If it matches C0, the premise **line** is what
helps and length is irrelevant; if it matches R1, the premise **formula** is what helps. That makes
R1's negative result interpretable instead of merely negative.

**Also open, and larger than this run:** `nd2lean.py`'s `BOTE` → `.elim` makes a Lean pass
insufficient on its own for this ND rule set (2,208 Lean-only acceptances in 16.3 M, 0 the other
way). Raised in `QUESTIONS.md`; it bears directly on proposal 9's plan to retire `nd_verify` for a
Lean term-allowlist check, whose allowlist must close these kinds itself.
