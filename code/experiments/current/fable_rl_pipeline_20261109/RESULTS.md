---
written_on: 2026-11-10
written_by: agent:claude
---

# fable_rl pipeline: results

**Headline (dev: 473 y statements held out from RL, 5 attempts each, Lean-judged).** Full mode,
three seeds: pass@1 94.0 / 95.6 / 93.0%,
pass@5 95.3 / 97.5 / 95.8%, at
447 / 492 / 528 generated tokens per attempt.
Plain mode (one rollout, temperature 0.8): pass@1 86.5 / 88.4 / 84.1%,
pass@5 92.2 / 94.1 / 91.3%, at about 190 tokens.
The pretrained bases in plain mode: 59.5 / 60.7 / 55.9% pass@1;
Dan's full-corpus base with his sampler on the same split: 48.9% pass@1, 69.1% pass@5.
Of the 6,466 full-mode attempts the solver called verified, 3 were wrong.
These are dev numbers; the test numbers are the orchestrator's.

Design: [`README.md`](README.md). Plan and amendments: [`PREREG.md`](PREREG.md). Figures:
`experiment-summaries/2026-11-10-fable-rl-pipeline/`. Everything below is observed on dev unless a
sentence says "interpretation" or "guess".

## What is handed back

| what | where (serrano) | md5 |
|---|---|---|
| final checkpoint, seed 0 (policy r80 + its two verifier networks + solver settings) | `/scratch/cowifornia/nd-rl/fable_rl/final/final_s0.pt` | `a7c62400dbee535931bcf145a6315549` |
| final checkpoint, seed 1 (policy r42) | `/scratch/cowifornia/nd-rl/fable_rl/final/final_s1.pt` | `2d78b6073920bbe8b92d52861ebaa546` |
| final checkpoint, seed 2 (policy r42) | `/scratch/cowifornia/nd-rl/fable_rl/final/final_s2.pt` | `a3eb2bb7af0d66f201c93a95e1d7e4c6` |
| ablation checkpoint (plain expert iteration, policy r39, no verifier: plain mode only) | `/scratch/cowifornia/nd-rl/fable_rl/final/final_abl.pt` | `b279001efcd78b110cffbd87556dd0ee` |
| solver folder (a copy of this folder's `solve.py`, `pack.py`, `fr/`, `vendor_dan/` at the commit named in `/scratch/cowifornia/nd-rl/fable_rl/final/solver/COMMIT`) | `/scratch/cowifornia/nd-rl/fable_rl/final/solver/` | |
| dev rows and logs of every final model | `/scratch/cowifornia/nd-rl/fable_rl/final/<tag>/` | |
| pod ledger | `/scratch/cowifornia/nd-rl/fable_rl/pods.tsv`, `/scratch/cowifornia/nd-rl/fable_rl/final/ledger.json` | |

Run, from the solver folder, with the lab environment:

    python solve.py --ckpt /scratch/cowifornia/nd-rl/fable_rl/final/final_s0.pt --in SET.in.jsonl --out SET.rows.jsonl --k 32 --seed S
    SOLVER_MODE=plain python solve.py ...        # one rollout per attempt, no verifier, nothing solver-written

Full mode is the default and the headline. Each call prints one summary line (attempts, proofs
called verified, generated tokens per attempt, premise lines written by the solver, attempts cut
by the time budget, peak memory). `SOLVER_MAX_SECONDS` sets the budget of one call (default 0.06 s
per attempt, at least 900 s; 0 for none).

## Dev results

### Dev, pass@1 / pass@5 (% of the dev theorems in the column), by lines in the reference proof

| model (mode) | all (n=473) | 7 lines (n=180) | 8 lines (n=100) | 9 lines (n=101) | 10 lines (n=45) | 11 lines (n=20) | 12 lines (n=21) | 13 lines (n=3) | 14 lines (n=3) |
|---|---|---|---|---|---|---|---|---|---|
| Dan's full-corpus base, his sampler (plain) | 48.9 / 69.1 | 58.4 / 78.3 | 58.8 / 78.0 | 52.3 / 78.2 | 16.4 / 33.3 | 31.0 / 55.0 | 1.9 / 9.5 | 13.3 / 33.3 | 0.0 / 0.0 |
| pretrained base, seed 0 (plain) | 59.5 / 75.3 | 72.0 / 88.3 | 72.6 / 86.0 | 60.6 / 80.2 | 24.4 / 42.2 | 31.0 / 45.0 | 2.9 / 4.8 | 13.3 / 33.3 | 0.0 / 0.0 |
| pretrained base, seed 1 (plain) | 60.7 / 74.2 | 75.3 / 88.3 | 69.2 / 82.0 | 61.6 / 78.2 | 27.1 / 44.4 | 33.0 / 45.0 | 6.7 / 9.5 | 0.0 / 0.0 | 0.0 / 0.0 |
| pretrained base, seed 2 (plain) | 55.9 / 70.8 | 67.7 / 82.2 | 68.0 / 81.0 | 56.4 / 73.3 | 22.2 / 35.6 | 27.0 / 55.0 | 11.4 / 23.8 | 0.0 / 0.0 | 0.0 / 0.0 |
| ablation: plain expert iteration from the seed-0 base (plain) | 84.4 / 91.5 | 91.3 / 98.3 | 86.6 / 91.0 | 88.3 / 93.1 | 70.7 / 84.4 | 65.0 / 80.0 | 62.9 / 76.2 | 33.3 / 33.3 | 0.0 / 0.0 |
| after RL, seed 0 (plain) | 86.5 / 92.2 | 91.4 / 96.1 | 86.0 / 92.0 | 93.9 / 97.0 | 81.8 / 91.1 | 76.0 / 80.0 | 51.4 / 71.4 | 33.3 / 33.3 | 0.0 / 0.0 |
| after RL, seed 0 (full) | 94.0 / 95.3 | 97.1 / 97.8 | 94.8 / 95.0 | 98.4 / 99.0 | 93.3 / 95.6 | 84.0 / 90.0 | 71.4 / 81.0 | 33.3 / 33.3 | 33.3 / 33.3 |
| after RL, seed 1 (plain) | 88.4 / 94.1 | 92.4 / 96.7 | 88.8 / 95.0 | 90.7 / 95.0 | 84.4 / 93.3 | 81.0 / 85.0 | 72.4 / 85.7 | 53.3 / 66.7 | 13.3 / 33.3 |
| after RL, seed 1 (full) | 95.6 / 97.5 | 97.8 / 98.9 | 96.4 / 97.0 | 98.4 / 99.0 | 96.0 / 100.0 | 90.0 / 95.0 | 76.2 / 85.7 | 66.7 / 100.0 | 33.3 / 33.3 |
| after RL, seed 2 (plain) | 84.1 / 91.3 | 89.3 / 95.6 | 88.0 / 92.0 | 86.5 / 93.1 | 77.3 / 88.9 | 69.0 / 85.0 | 53.3 / 66.7 | 46.7 / 66.7 | 6.7 / 33.3 |
| after RL, seed 2 (full) | 93.0 / 95.8 | 96.7 / 98.9 | 94.2 / 96.0 | 96.4 / 98.0 | 91.1 / 97.8 | 81.0 / 85.0 | 66.7 / 71.4 | 73.3 / 100.0 | 33.3 / 33.3 |

### Dev by source, tokens and verdicts

| model (mode) | generated (n=384) | textbook (n=89) | generated tokens per attempt | attempts the solver said yes to | of those wrong | premise lines written by the solver per attempt |
|---|---|---|---|---|---|---|
| Dan's full-corpus base, his sampler (plain) | 58.1 / 81.0 | 9.0 / 18.0 | 180 |  |  |  |
| pretrained base, seed 0 (plain) | 71.3 / 87.5 | 9.0 / 22.5 | 189 | 90.9% | 742 of 2150 | 0.00 |
| pretrained base, seed 1 (plain) | 72.2 / 85.7 | 11.0 / 24.7 | 188 | 89.3% | 677 of 2113 | 0.00 |
| pretrained base, seed 2 (plain) | 66.6 / 82.6 | 9.9 / 20.2 | 181 | 91.7% | 846 of 2169 | 0.00 |
| ablation: plain expert iteration from the seed-0 base (plain) | 89.1 / 95.6 | 64.0 / 74.2 | 183 | 96.6% | 289 of 2285 | 0.00 |
| after RL, seed 0 (plain) | 90.7 / 95.6 | 68.5 / 77.5 | 189 | 92.9% | 150 of 2196 | 0.00 |
| after RL, seed 0 (full) | 97.8 / 98.2 | 78.0 / 83.1 | 447 | 91.7% | 0 of 2169 | 1.22 |
| after RL, seed 1 (plain) | 90.5 / 95.3 | 79.6 / 88.8 | 191 | 93.6% | 123 of 2214 | 0.00 |
| after RL, seed 1 (full) | 98.0 / 99.2 | 85.2 / 89.9 | 492 | 91.8% | 1 of 2171 | 1.22 |
| after RL, seed 2 (plain) | 88.3 / 94.3 | 65.6 / 78.7 | 190 | 92.6% | 203 of 2191 | 0.00 |
| after RL, seed 2 (full) | 97.4 / 99.0 | 74.2 / 82.0 | 528 | 89.9% | 2 of 2126 | 1.22 |

All plain-mode rows of my models in these tables are at temperature 0.8, the frozen setting, so
the base rows differ from the temperature-1.0 numbers quoted in the amendments and in the sections
below (57.4 / 57.9 / 52.5% pass@1 for the three bases).
Tokens are tokens the policy generated, refused samples and restarts included. The verifier
networks generate nothing; they read the steps counted in the last table of this section. Dan's
base is a 9.56M-parameter model in his format, sampled by his code at temperature 0.8 on serrano
CPU. The baseline recipe itself (GRPO with a thinking scratchpad) was not run on this split.

### Runs, checkpoint rule, training side

| run | RL rounds in 11 h | checkpoint picked by the rule (dev plain at temperature 1.0, pass@1 / pass@5) | training theorems with a kept proof | textbook | 12 lines | 13 lines | 14 lines |
|---|---|---|---|---|---|---|---|
| seed 0 | 80 | r80 (85.3 / 93.9) | 4,022 of 4,022 | 671/671 | 85/85 | 13/13 | 11/11 |
| seed 1 | 55 | r42 (87.0 / 94.7) | 4,021 of 4,022 | 670/671 | 85/85 | 13/13 | 10/11 |
| seed 2 | 42 | r42 (81.9 / 92.4) | 4,015 of 4,022 | 664/671 | 85/85 | 13/13 | 6/11 |
| ablation | 50 | r39 (82.9 / 90.7) | 3,932 of 4,022 | 592/671 | 78/85 | 3/13 | 1/11 |

The rule (amendment 2): the scored round with the highest dev pass@1 in plain mode, ties to the
later round. Dev theorems no seed solved in full mode in 5 attempts: 6 of 473 (by seed: 22, 12, 20); by reference length: 7 lines 1, 8 lines 2, 9 lines 1, 14 lines 2; 4 of them are textbook theorems.

### Verifier networks and full-mode verdicts, per seed

| seed | one-logit judge: training steps | its held-out refused caught / accepted kept at 0.5 | masked slot model: training slots | its held-out accepted steps kept at 0.001 | attempts called verified | of those wrong | unscreened last-try proofs handed in | verifier-read steps per attempt | premise lines written by the solver per attempt |
|---|---|---|---|---|---|---|---|---|---|
| seed 0 | 6,625,256 | 98.7% / 99.7% | 8,981,336 | 99.91% | 2,169 of 2,365 | 0 | 157 (55 correct) | 8 | 1.22 |
| seed 1 | 8,100,206 | 98.8% / 98.9% | 7,166,979 | 99.93% | 2,171 of 2,365 | 1 | 149 (90 correct) | 8 | 1.22 |
| seed 2 | 7,704,047 | 98.2% / 98.1% | 5,745,320 | 99.95% | 2,126 of 2,365 | 2 | 189 (76 correct) | 9 | 1.22 |

The held-out steps here come from the same rounds as the training steps (a 2% split), so these
are fit checks; the read on rounds no verifier saw is in PREREG amendment 5 (seed 1, mid-run).

The ablation's policy under seed 0's two verifier networks and the same solver (full mode): 92.4 / 94.5 at 385 tokens, against seed 0's 94.0 / 95.3.

## What earned the result, and how I know

Three parts, measured on dev. The first two are measured cleanly; the third has one ablation run.

**1. A verifier-screened solver (the largest part).** With the same two verifier networks and the
frozen solver settings, seed 0:

| policy | plain pass@1 / pass@5 | full pass@1 / pass@5 | full, textbook part | full, generated part |
|---|---|---|---|---|
| pretrained base | 57.4 / 76.5 | 83.9 / 87.9 | 36.0 / 46.1 | 95.1 / 97.7 |
| after 1 RL round | 63.8 / 82.2 | 88.4 / 91.8 | 51.9 / 62.9 | 96.9 / 98.4 |
| after 6 rounds | 74.1 / 88.8 | 91.8 / 94.3 | 67.4 / 76.4 | 97.5 / 98.4 |
| after 12 rounds | 79.1 / 91.1 | 94.2 / 97.7 | 76.9 / 88.8 | 98.3 / 99.7 |
| after 18 rounds | 79.5 / 90.3 | 94.7 / 96.8 | 77.3 / 84.3 | 98.7 / 99.7 |
| after 24 rounds | 78.6 / 91.1 | 94.6 / 97.9 | 77.8 / 91.0 | 98.5 / 99.5 |
| after 30 rounds | 81.5 / 92.4 | 93.9 / 96.8 | 76.9 / 88.8 | 97.8 / 98.7 |

(Plain column at temperature 1.0, from the runs' own dev checks. The verifier networks in this
table were trained on steps from seed 0's rounds 0 to 35, so the first row is not a system without
RL.) Observed: the solver takes the pretrained base from 57 to 84% pass@1; on generated theorems it
alone reaches 95%. Interpretation: most plain-mode failures are single ill-typed steps, which a
step verifier with resampling removes.

**2. RL on y (the textbook part).** Same table, read down: full mode goes from 83.9 to about 94.5,
and nearly all of that is the textbook theorems (36 to 77% pass@1). In plain mode RL adds 25
points by round 30 and keeps adding slowly (seed 0: 81.5 at round 30, 85.3 at round 80, both at
temperature 1.0). Full mode stops improving after about round 12: the final seed-0 model, 50
rounds later, is at 94.0 / 95.3.

**3. The checker inside the sampling loop during RL (one ablation run: plain expert iteration,
same base as seed 0, same 11 h, same 512 attempts for unsolved theorems).**

| | seed 0 (checker in the loop) | ablation (no checker in the loop) | seeds 1 and 2, for the spread |
|---|---|---|---|
| training theorems with a kept proof (of 4,022) | 4,022 | 3,932 | 4,021 and 4,015 |
| of the 11 training theorems with a 14-line reference | 11 | 1 | 10 and 6 |
| dev plain, final (temperature 0.8) | 86.5 / 92.2 | 84.4 / 91.5 | 88.4 / 94.1 and 84.1 / 91.3 |
| dev full, final, under seed 0's verifier networks | 94.0 / 95.3 | 92.4 / 94.5 | (own verifiers) 95.6 / 97.5 and 93.0 / 95.8 |
| dev full, textbook part | 78.0 / 83.1 | 72.8 / 79.8 | 85.2 / 89.9 and 74.2 / 82.0 |

Observed: the ablation is 2.1 points below seed 0 in plain mode and 1.6 below in full mode, and
both differences are smaller than the spread between my three seeds (4.3 and 2.6 points). So on
dev I cannot call the checker in the loop established as a gain. Where it is clear is the
training side: 90 training theorems never got a proof without it, and the 14-line family opened
in all three seeds with it (first new proof at round 22, 39 and 29) and not in the ablation.
Interpretation: it matters for the hardest and longest theorems, of which dev has few; whether
that carries to the long test sets is exactly what the test will say. One run; not a measured
effect size.

**The verifier itself** (what replaced the one-logit judge and why) is in PREREG amendment 5: a
generative slot model catches the near misses a discriminative judge lets through, and the two
must both pass. On dev, with seed 1's 18-round policy: one-logit judge alone 112 of 2,207 verified
attempts wrong (at 6 rounds); both networks 0 of about 2,230, and 96.1 / 97.3. With the final
models, 3 of the 6,466 attempts the solver called verified were wrong (seeds 0, 1, 2: 0, 1, 2): two
`Or.inr` steps and one pairing whose stated formula did not match the cited line, passed by both networks.

**Tries.** On dev the first try produces 2,123 of 2,365 proofs and later tries little; on the long
hand-written statements later tries produce a third of the proofs (first try 214, second 99, third 5
of 318 under a time budget). Dev under-weights hard theorems, so the choice of 4 tries, made on dev
and on time, is likely too low for long theorems when time allows more.

## Solver cost, and long statements

Measured on one A40 with the process capped at 22 GB, frozen settings; seed 0's round-30 policy
and its mid-run verifier networks unless the row names another checkpoint:

| set | attempts | full mode | plain mode | peak memory allocated |
|---|---|---|---|---|
| 473 dev theorems x 32, seed 0's round-30 policy | 15,136 | 409 s, 438 tokens per attempt | 62 s, 186 tokens | 9.7 GB |
| the same with the final seed-0 checkpoint (93.9 / 96.4 full, 87.0 / 93.4 plain at 32 attempts) | 15,136 | 486 s (431 s in the attempt loop), 447 tokens | 109 s (60 s in the loop), 188 tokens | 9.9 GB |
| first 832 training theorems x 32 (easy for the model: a size check) | 26,624 | 211 s, 190 tokens | 102 s, 193 tokens | 9.3 GB |
| 832 theorems of dev difficulty x 32 (projected from the first row) | 26,624 | about 12 minutes | about 2 minutes | |

Time on an easy set is set by its few longest-running attempts; on a hard set every attempt is
long. Four long statements that I wrote by hand (never trained on, used only to run the solver;
seed 1's 18-round policy and verifier networks):

| statement | full mode, no time budget: Lean-correct of 512 attempts | plain mode: Lean-correct of 512 | plain mode: attempts the environment called complete |
|---|---|---|---|
| 4 premises with a disjunction (proofs found: 15 to 25 lines) | 324 (317 verified, 7 from the unscreened last try) | 18 | 354 |
| implication chain (14 to 20 lines) | 9 | 0 | 134 |
| long conjunction | 0 | 0 | 111 |
| largest (about 25 lines) | 0 | 0 | 0 |

- In plain mode the policy finishes long proofs that Lean rejects (581 of 599 complete ones). Each
  of the 326 attempts full mode called verified is Lean-correct.
- A hard attempt in full mode costs about 3,100 generated tokens (plain: about 250) and 0.6 s with
  other jobs on the GPU. With a 300 s budget for 4,096 such attempts (0.07 s each) the solver
  stopped at 303 s, wrote all rows, cut 3,640 attempts (91 of them still in their first try), and
  7.8% of attempts were correct against 16.4% without a budget. So full mode on long theorems is
  as good as the time it is given: the default budget of 0.06 s per attempt is about one to two
  tries per attempt there.
- Interpretation: the 63% on one statement shows that the solver can carry a pattern the policy
  knows to 25 lines; the zeros show that the policy does not know the other patterns. It is not a
  measurement of the long test sets.

## Against what I wrote down beforehand

- Original plan (before any RL): plain pass@1 after RL 50 to 70%, full ("search") 80 to 90%, full
  pass@5 above 90%, 40 to 60% at 13 and 14 lines in full mode; the base under 10%. Observed: plain
  84 to 88% (at temperature 0.8), full 93 to 96%, pass@5 95 to 98%, full mode at 13 lines 33, 67
  and 73% and at 14 lines 33% for every seed (3 theorems each); the base 52 to 58%. Too low on
  everything except the long end, where the guess was about right and the sample is 6 theorems.
- Amendment 5 (after the verifier was settled): full mode 94 to 97% pass@1 and 96 to 98% pass@5 for
  every seed at 400 to 600 tokens. Observed: 94.0, 95.6, 93.0 and 95.3, 97.5, 95.8 at 447 to 528
  tokens. Seed 2's pass@1 and two of the pass@5 values fall just below the range.
- The mid-run read behind amendment 5 had 0 of 2,217 verified attempts wrong; the final models
  have 3 of 6,466.

## What failed or was dropped

Observed, in the order it happened:

- **Unlikelihood loss on refused steps** (one RL round from the seed-0 base, dev plain): 42.0 / 68.3
  against 49.3 / 73.8 without it. It taught the policy to drop the closing `;` (328 of 670 failed
  rollouts ended that way). Removed.
- **Verifier and value losses on the policy's own trunk**: 49.3 / 73.8 with them, 60.2 / 80.8
  without (same round, same base; the base itself is 57.4 / 76.5). Removed; the verifier became
  its own network.
- **A verifier trained only on the policy's refused and accepted steps** passes near misses
  (`.1` for `.2` on two almost equal conjuncts, an `Or.inl` whose stated formula has the right atoms
  in the wrong shape): 112 of 2,207 attempts it called verified were wrong. A larger network,
  three rounds of mining its own mistakes and judging each step under several name offsets did not
  fix this (2.2 to 2.8% of unseen refused steps still passed).
- **One-symbol edits of accepted steps, labelled by the training-time checker**, did fix most of it
  (95.3 / 97.3 on dev with seed 1's 18-round policy, 20 of 2,239 verified attempts wrong), but they
  are synthetic steps, so they were not used.
- **The unmasked slot model** (write each formula of a step from the rest) passes 0.03% of unseen
  refused steps and never called a wrong dev attempt verified, but it has habits about the parts of
  a formula that a rule leaves free and refused 2% of the valid steps on unseen theorems,
  blocking whole theorem families (94.9 / 96.8).
- **The masked slot model alone** fixes that and passes 3.2% of unseen refused steps: with part of
  the formula in view it explains gross mismatches away. Together with the one-logit judge, which
  catches exactly those, 0.01% pass.
- **Three worker processes sharing the GPU** in the solver: 459 s against 469 s for one process.
- **Resampling a refused step from the state's cached encoding** (instead of encoding the state
  again in the next decoding round): same dev result, slower (531 s against 327 s on 512 long
  attempts). Left in the code, switched off.
- **The subgoal procedure at test time** (a refused step's formula becomes a subgoal): on seed 0
  94.5 / 97.5 at 590 tokens against 93.9 / 96.8 at 444 without it, and 94.7 / 97.5 at 701 against
  94.8 / 97.0 at 564 with 8 tries: no gain in pass@1 beyond what more tries give at fewer tokens.
  It is on during RL sampling (amendment 4), where its effect was not isolated.
- **My first guess about the base** (under 10% on dev) was wrong by a factor of six: 57.4%.

## What not to trust

- **Dev is not the test.** Dev is 473 y statements held out from RL, from the same generator and
  the same textbook schemata as y train. Every setting (verifier design, thresholds, tries,
  temperature, checkpoint choice) was chosen on it. The numbers above are upper estimates for
  anything out of that distribution.
- **Length.** Dev has 3 theorems at 13 lines and 3 at 14; nothing longer. The only look past 14
  lines is four statements I wrote by hand as a mechanical check. They are not a sample of
  anything, and full mode's 63% on one of them says only that the solver can produce 15- to
  25-line proofs when the policy knows the pattern.
- **Seeds differ where it matters for length.** The 14-line training family opened at different
  times: seed 0 in rounds 22 to 28 (11 of 11), seed 1 in rounds 39 to 42 (9, then 10 at round 55),
  seed 2 in rounds 29 to 37 (6 of 11). The checkpoint rule picked round 80 for seed 0 and round 42
  for seeds 1 and 2, so seeds 1 and 2 were picked within a few rounds of first seeing those proofs.
  On dev (3 theorems each) full mode is 33, 67 and 73% pass@1 at 13 lines and 33% for every seed
  at 14 lines. Expect seed spread on the long sets.
- **Plan changes during the runs.** Amendments 3 and 4 changed the kept runs in flight (128, then
  512 attempts for unsolved theorems; subgoals in sampling), by restart with `--resume`. They
  were written before they were applied, but they were made after looking at the runs.
- **Verifier and solver were designed on seed 1's mid-run checkpoints**, then the solver settings
  were chosen on seed 0's round-30 checkpoint, not on the final checkpoints.
- **Solver-written lines.** In full mode the premise lines are written by the solver (orchestrator's
  ruling A), and after the screened tries one unscreened rollout hands in a proof with verdict
  "no". Both are counted in the rows (`scaffold_lines`, `fallback`). Plain mode has neither.
- **Hand-written statements.** Five statements of mine exist: four long ones and one with `F` and
  four premises. They were used only to run the solver (never for training, never for a choice
  of setting) and they were copied to pod 5 for that.
- **Running time on long theorems.** A hard attempt in full mode costs about 3,100 generated
  tokens and 0.4 to 0.6 s on an A40. The solver's wall-clock budget cuts attempts that are still
  running; under the default budget a set of long theorems gets roughly one screened try per
  attempt. The summary line of each call says how many attempts were cut.
- **Not ablated:** hindsight lemmas, subgoal variants, the mixture weights, the model size, the
  512 attempts for unsolved theorems. Each is in the recipe on argument, not on a measured gain.

## With ten times the budget

1. The missing ablations, three seeds each: no hindsight, no subgoal variants, no x replay, plain
   expert iteration with the same verifier at test time. Today's single ablation run cannot
   separate seed noise from effect below about 3 points.
2. Length, directly: let the model state its own longer targets (conjunctions and chains of lemmas
   it has already proved), which the rules allow, and measure dev by length out to 30 lines before
   any test. Today nothing in training is longer than 14 lines.
3. Share what one seed finds: the 14-line family shows that one found proof opens a family; a
   population of policies exchanging checker-accepted proofs would remove that luck.
4. A real search in place of independent tries: the verifier's verdicts were wrong for 3 of 6,466
   verified dev attempts, so best-first search over verified partial proofs with a value network
   is nearly safe, and the tries after the first add little on dev as they are.
5. Solver speed: reuse the encoded prefix of a state across steps (a step appends one line), and
   pool the few long actions of a decoding round; both were out of reach today and long theorems
   are where the time goes.
6. A larger policy: dev in full mode was flat from about round 12 on at 14M parameters, and the
   theorems left unsolved are ones the policy cannot prove.

## Ledger

| pod | created (UTC) | deleted (UTC) | hours | USD at $0.59/h |
|---|---|---|---|---|
| fable-rl-1 | 2026-11-10T04:18:58Z | 2026-11-10T19:07:53Z | 14.82 | 8.74 |
| fable-rl-2 | 2026-11-10T04:56:13Z | 2026-11-10T19:10:14Z | 14.23 | 8.40 |
| fable-rl-3 | 2026-11-10T04:56:14Z | 2026-11-10T19:08:45Z | 14.21 | 8.38 |
| fable-rl-4 | 2026-11-10T05:47:27Z | 2026-11-10T18:43:03Z | 12.93 | 7.63 |
| fable-rl-5 | 2026-11-10T07:21:25Z | 2026-11-10T16:17:19Z | 8.93 | 5.27 |

Total 65.1 pod-hours on A40s, **$38.42** of GPU time, plus about $0.63 for disks and
volumes at list price (not read from billing). Budget $60; no run was started after $54.

GPU-hours by phase (five A40 pods; from job logs):

| phase | pod-hours |
|---|---|
| pod setup, smoke tests, pretraining of the three bases (pods 1 to 4, until each base was ready) | 3.3 |
| pilot RL runs and the one-round A/Bs behind amendments 1 and 2 (pods 1 to 4, until 07:00Z) | 4.7 |
| kept RL runs, seeds 0 to 2 (pods 1 to 3, 07:00Z to RL_DONE; 11 h of run clock each plus dev checks and two restarts) | 34.0 |
| ablation run, plain expert iteration (pod 4, 07:00Z to RL_DONE) | 11.4 |
| verifier and solver development, tuning, dev scoring, timing and memory tests (pod 5, whole life) | 8.9 |
| finalisation: verifier networks per seed, packing, final dev scoring, copies and checks (pods 1 to 4, RL_DONE to deletion) | 2.7 |
| **total** | **65.1** |

No pretraining was run on the knockout corpus (the correction arrived before the first run), so
nothing was discarded for that reason.
