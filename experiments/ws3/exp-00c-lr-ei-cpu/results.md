# exp-00c learning-rate sweep and expert iteration on CPU: results

**Status: closed. Review verdict PASS WITH CORRECTIONS, not quarantined** (`review.md`, reviewer
commits `54d2790` recount, `61887dc` claims, `0ca481e` verdict). Every number below is from the
reviewer's Recount (`review.md` and `review/`), which is the source of record. It agrees with the
executor's `analysis.md` on every point estimate. All seven corrections in the verdict are
applied here. Pre-registration: `proposal.md`, commit `487f43d` (2026-09-27T23:36:25Z), before
the pool filter sampled anything and before any `metrics.jsonl` line (gate A passes). Branch
`charles/exp-00c-lr-ei-cpu`.

History that must be stated in every write-up (plan section 1a): the 72 textbook problems are
Dmitry's set from `origin/dmitry/cap-comparison-textbook-eval-data`. The 14 `textbook_train` were
RL training prompts in the cap-comparison sprint and the 58 `textbook_dev` its dev benchmark.
Nothing here trained on any of them.

## What ran

All four arms start from the fixed checkpoint (sha256 `418af3f9…`) with seed 0 and train on the
same 198-prompt pool. The pool comes from the **train half** of `rl_targets` (131 at length 7, 4
at 8) plus 63 length-6 prompts from `train.jsonl`. It holds only prompts the frozen model solved
1 to 7 times out of 8 in the pre-registered filter. Every arm is evaluated on 200 prompts from
the **eval half** (160 at length 7, 40 at 8), 32 samples each at temperature 1.0. The two halves
share no renaming class; the reviewer found 0 overlap under `canon_key`, `canonical_theorem`,
exact text, and an atom-erased skeleton.

| arm | training | samples | wall-clock (train + eval) |
| --- | --- | --- | --- |
| lr3e-5 | GRPO (TRL), 48 steps x 16 prompts x group 16, KL 0.01 | 12,288 | 531 s + 255 s |
| lr1e-4 | same, lr 1e-4 | 12,288 | 488 s + 250 s |
| lr3e-4 | same, lr 3e-4 | 12,288 | 576 s + 260 s |
| ei | expert iteration, 3 rounds x 21 samples per prompt, SFT (lr 1e-4) on verified proofs | 12,474 (+1.5%: k rounds up) | 238 s + 269 s |

Shared costs, not charged to any arm: the pool filter (13,296 frozen samples, 242 s) and the
frozen baseline (202 s). Total wall-clock was 56 minutes of laptop CPU; no GPU and no spend. One
design change from the chat plan was made before the commit: lr 1e-3 was replaced by 3e-5,
because a smoke test showed one step at 1e-3 breaks the model (`proposal.md`).

## Headline

CIs are 95% intervals over problems (the plan section 5 method, from the Recount).

| held-out | frozen | lr3e-5 | lr1e-4 | lr3e-4 | ei |
| --- | --- | --- | --- | --- | --- |
| L7 avg solve rate [95% CI] | 0.122 [0.079, 0.166] | 0.430 [0.365, 0.495] | 0.391 [0.327, 0.455] | 0.333 [0.270, 0.397] | 0.545 [0.479, 0.611] |
| L7 solved at least once (of 160) | 39 | 97 | 97 | 85 | 118 |
| L7 new capability (frozen 0/32, trained >= 0.5) | - | 42 | 30 | 26 | 62 |
| **pre-registered:** L7 mean paired diff on the 39 frozen-solved [bootstrap CI] | - | +0.273 [0.18, 0.37] | **+0.325** [0.23, 0.43] | +0.163 [0.02, 0.31] | +0.245 [0.14, 0.36] |
| same 39: up / down / same (sign test p) | - | 31 / 4 / 4 (3.5e-6) | 30 / 3 / 6 (1.4e-6) | 22 / 17 / 0 (0.52) | 29 / 8 / 2 (7.5e-4) |
| L7 mean paired diff, all 160 | - | +0.307 | +0.268 | +0.211 | +0.423 |
| L8 avg solve rate [95% CI] | 0.000 | 0.005 | 0.013 | 0.011 | 0.266 [0.151, 0.381] |
| L8 solved at least once (of 40); new capability | 0 | 3; 0 | 5; 0 | 2; 0 | 19; 12 |
| greedy heldout, lengths 2 to 6 | 0.978 | 0.982 | 0.978 | **0.881** (106 lost) | 0.978 |
| textbook 72 avg | 0.101 | 0.113 | 0.119 | 0.084 | 0.108 |
| textbook, reference length >= 7, solved at least once (of 43) | 0 | 1 | 1 | 0 | 1 |
| final / max KL | - | 0.010 / 0.268 | 0.028 / 0.045 | 0.019 / 0.164 | n/a |

The textbook long solve is the same problem in every arm that has one: `textbook_c18068…`,
`|- (P v Q) > (Q v P)`, at 2 to 4 of 32 samples. At the plan's bar that is noise; the previous
best was 0 to 1 of 43.

![per-problem, length 7](analysis_len7.png)

**Selection rule (pre-registered): lr 1e-4.** lr3e-4 is excluded because greedy heldout dropped
0.097, past the 0.02 limit. lr 1e-4 has the higher mean paired difference on the 39
frozen-solved problems (+0.325 against +0.273). The rule selects on the evaluation set (plan T5
allows this; it carries a small winner's-curse bias). On one seed it is not evidence that 1e-4
beats 3e-5: 3e-5 is ahead on L7 avg, reliable count, new capability and the all-problem
difference.

## Against the pre-registration

| expected | observed | verdict |
| --- | --- | --- |
| pool 150 to 300; 90 to 170 at L7; 5 to 25 at L8 | 198; 131; 4 | L8 slightly below |
| frozen eval L7 avg 0.08 to 0.14, 30 to 40 solved; L8 0 to 2 | 0.122, 39; 0 | as expected |
| lr3e-5 max KL 0.001 to 0.01, L7 about neutral | max KL 0.268 (one spike, typical 0.01); L7 avg +0.31 | missed: far larger lift |
| lr1e-4 paired diff 0 to +0.08, sign test maybe not significant | +0.325, p = 1.4e-6 | missed: about 4x larger |
| lr3e-4 50/50 between like-1e-4 and degraded | degraded (greedy -0.097); still +0.21 L7 avg | degraded branch |
| EI within about 0.03 of the best GRPO arm (pre-registered metric: paired diff on frozen-solved L7) | **EI +0.245 is 0.080 below lr1e-4** | missed, in the other direction on this metric. EI is ahead only on metrics that were not pre-registered: L7 avg (+0.115), all-problem difference (+0.423 against +0.307), new capability (62 against 42) and L8 (0.266 against 0.013). One seed |
| new capability 0 to 1 per arm; L8 0 to 2 new | 26 to 62 at L7; EI 12 at L8 | missed by far |
| wall-clock 60 to 75 minutes | 56 minutes | as expected |

## Is it real?

Two independent checks found no reward bug, no verifier loophole and no leak:
- a verifier session with its own code, covering lr3e-5 and EI (`verify_lr3e-5/`,
  `verify_ei/`);
- the reviewer, covering all arms (`review.md`).

- **Re-verification.** The reviewer re-verified 115,127 rows with `strict_verify.safe_verify`,
  with 0 disagreements. That covers every eval dump of every arm, the filter, all training
  completions, periodic eval, and SFT data. (The earlier verifier covered the three long-pool
  dumps of the frozen model, lr3e-5 and EI, plus the lr3e-5 and EI training rewards.)
- **No loophole.**
  - No accepted proof has the conclusion or `F` among its premises, a wrong final line, or text
    after `QED`.
  - One lr1e-4 proof is shorter than its `L_true` (7 lines against 8). It is sound: its `ORE` uses
    one subproof for both halves of `Q v Q`.
  - The rule mix is ordinary.
  - Renumbering lines to start at `N1` changes no verdict among parseable rows. Across all rows,
    it turns 1 (lr3e-5) and 9 (EI) rejections into accepts. That shows the checker is strict
    about numbering, not lenient.
- **No leak.**
  - 0 overlap between anything trained on and any evaluation set, under all four keys, including
    the eval half, the 72 textbook problems, and the test prompt files. Test verdicts were never
    read.
  - 0 of 3,130 accepted EI eval proofs appear in EI's SFT data, even after renaming and
    renumbering.
  - 6 eval prompts share a renamed conclusion with a pool prompt. They improve less than the rest
    (EI 0.05 to 0.21, against 0.13 to 0.56 for the other 154 length-7 problems), so they don't
    drive the gain.
- **What changed.** The reviewer read all 167 distinct accepted proofs at length 8 or above, plus
  100 random samples. The gain is one repaired failure mode across the generator's
  nested-implication templates. The frozen model closed the innermost `IMPI` by citing the
  assumption itself (for example `IMPI N34 N34`) instead of first deriving the consequent from a
  premise. EI's length-8 solves (90% one shape: three nested assumptions, one premise-derived
  inner step, three `IMPI`s) combine two shapes present in its SFT data. **Templates that need a
  case split or a contradiction stay at 0 in every arm.**

## How to read it

1. **In-distribution only.** The lift holds on held-out problems from the same generator, and
   transfers across renaming classes and skeletons within it. It does not reach the textbook set
   (0.101 to 0.084–0.119, inside the CIs; 0 to 1 of 43 long problems, always the same one). It
   also misses the generator's case-split and contradiction templates.
2. **"New capability" meets the plan's definition, but should not be read as emergence.** The
   counts are nonzero (42 / 30 / 26 / 62 at length 7, 12 at length 8 for EI). But they come from
   one seed, and they are one repaired step in one template family. The plan's emergence bar is 3
   seeds.
3. **EI vs GRPO is unresolved.** On the pre-registered metric, GRPO at 1e-4 leads EI by 0.080. On
   L7 avg, new capability and length 8, EI leads, in half the training time and with 1.5% more
   samples. The length-8 gap is the only EI-vs-GRPO difference whose intervals don't overlap.
   Everything is one seed.
4. **Learning rate, on one seed:** 3e-5 and 1e-4 both work at KL 0.01. 3e-4 forgets short proofs
   (greedy lengths 2 to 6 fell 0.097), and 1e-3 breaks the model in one step (smoke). exp-00 and
   exp-00b's 1e-5 on easy prompts moved nothing, but that was the prompts as much as the rate.
5. **Diversity collapses.** Most newly solved problems end up with one distinct accepted proof.

## 20 raw samples per arm

The executor read 20 random held-out samples per arm (seed 0), 80 in total. 8, 8, 7 and 12 of
the 20 were accepted (executor count; the reviewer's own 100 samples and 167 long proofs are in
`review/`). All accepted proofs are genuine. Failures are real rule errors:
- `DN` on the wrong formula;
- `IMPI` citing the assumption as its own consequent;
- wrong `ORE` box citations;
- malformed rule names.

lr3e-4 shows more malformed output (a stray `|` inside a formula, `PRF` emitted mid-proof), which
fits its forgetting. On problems 3445 and 2484, GRPO attempts the vacuous `IMPI N23 N23` and
fails, while EI derives the formula from a premise inside the innermost box (`ANDE1 N20`) and
succeeds.

## Caveats

- **One seed per arm.**
- **The pool filter selects prompts the frozen model sometimes solves** (train half only). It is
  not a leak, but no arm can show solves from nothing on training prompts, only on held-out ones.
- **EI used 1.5% more samples** than the GRPO arms.
- **EI's dedup compares raw text,** so the same proof with different line numbers counts twice.
  Only 297 of about 2,260 SFT records are truly distinct.
- **The lr rule selects on the eval set.**
- **Unverifiable:** the executor's 20-sample counts, the lr 1e-3 smoke, the line-index-start
  check and the timings are the executor's own and were not independently recomputed.

## Conclusion

On the fixed Muon checkpoint, with dataset and pre-training unchanged, both TRL GRPO (lr 3e-5 to
1e-4, KL 0.01, 48 steps) and expert iteration in TRL gave a large, verified, single-seed lift on
held-out `rl_targets` problems at length 7:
- the average solve rate went from 0.12 to between 0.39 and 0.55;
- EI also moved length 8 from 0 to 0.27;
- short-proof ability was intact, except at lr 3e-4.

The lift is one repaired step in the generator's nested-implication templates. It doesn't reach
the textbook set or the case-split and contradiction templates. It is not an emergence claim, and
the EI-vs-GRPO ranking depends on the metric. The pre-registered rule carries lr 1e-4 forward.

**Next:**
1. 3 seeds of EI and of GRPO at 1e-4 (and 3e-5), to settle the claims above;
2. EI and GRPO on an unfiltered pool, to see whether the filter or the algorithm drives the
   result;
3. a probe of whether longer training or lengths 9 to 10 open the case-split and contradiction
   templates, or anything on the textbook set.

Open for Charles (from the log and the review):
- whether to report "new capability" as defined, given it is mostly one-step fixes;
- whether the pool filter is acceptable as standard practice;
- the process flag: the queued arms ran on during the stop-and-check after the lr3e-5 result. It
  cost only laptop CPU, and the checks were clean.
