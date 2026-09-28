# exp-00c learning-rate sweep and expert iteration on CPU: proposal

Pre-registration (plan section 6, gate A). Committed before `make_pool.py` samples anything and
before the first `metrics.jsonl` line. Executor: one Claude Code session on branch
`charles/exp-00c-lr-ei-cpu` (worktree `../nd-rl-exp-00c`). Laptop CPU only (Apple M1 Pro), about
one hour. No pod, no GPU, no Tinker, no spend.

## Why

exp-00 and exp-00b ran GRPO at lr 1e-5 on lengths 3 to 5, and in both the policy barely moved
(KL 0.0017 and 0.0003). For the Muon checkpoint those lengths are solved: 90% of groups were
all-correct. The RL question is at lengths 7 and 8 (exp-00b: length-7 avg solve rate 0.113,
length 8 1 of 44). Before any pod run, two things are needed: a learning rate that moves this
3M-parameter model without breaking it, and a first read on GRPO against rejection-sampling
fine-tuning (expert iteration, EI) at the same number of samples.

## Hypotheses

1. There is a learning rate between 3e-5 and 3e-4 at which 48 GRPO steps on frontier prompts
   (lengths 6 to 8) raise the solve rate on the training prompts, keep greedy lengths 2 to 6
   intact, and move held-out length-7 problems up more often than down.
2. EI at the same 12,288 samples lifts held-out length 7 by a similar amount (Robbie's EI result;
   RAFT++ reports EI roughly matching GRPO).

## Design

**Split (done before this commit, no model involved).** `make_split.py` splits `rl_targets` by
renaming class (sha256 of the key, even = train). The train half has 2,223 prompts (864 at L_true
7, 498 at 8) and the eval half 2,272 (936 at 7, 502 at 8); the halves share no class. The eval
set `eval_set.jsonl` is a stride of the eval half: 160 at length 7 and 40 at length 8. Training
candidates (`candidates.jsonl.gz`, 1,662) are every train-half prompt at lengths 7 and 8, plus 300
length-6 prompts from `train.jsonl` (seed 0). All candidates are checked against heldout,
transfer, validation_36, the 72 textbook problems, the test prompt files (their renaming classes
are now banned from RL prompts, per the questions.md default) and the eval half.

**Pool filter (pre-registered rule, `make_pool.py`).** Take 8 frozen-model samples per candidate
(temperature 1.0, 400 new tokens, seed 0). Keep prompts with 1 to 7 verified out of 8. Then cap
the length-6 survivors at half the number of length-7 survivors (random, seed 0). This touches
only the training half. It makes most GRPO groups mixed, a manual version of DAPO's dynamic
sampling. What it implies: at this scale the experiment can only show sharpening of prompts the
model sometimes solves, not solves from nothing. The filter's 13,296 samples are a shared cost of
all four arms and are not counted in any arm's budget.

**Arms**, all from the fixed checkpoint (sha256 `418af3f9…`), seed 0, each writing
`run_<arm>/`:

| arm | algorithm | settings | samples |
| --- | --- | --- | --- |
| `lr3e-5` | TRL GRPO (`WS3GRPOTrainer`, `advantage: trl`) | lr 3e-5, KL beta 0.01, clip 0.2, temp 1.0, 16 prompts x group 16 per step, 48 steps, cap 400 tokens, `loss_type: grpo`, `scale_rewards: group` | 12,288 |
| `lr1e-4` | same | lr 1e-4 | 12,288 |
| `lr3e-4` | same | lr 3e-4 | 12,288 |
| `ei` | expert iteration, TRL `SFTTrainer` (`loss_type: nll`) | 3 rounds, k = round(12,288 / (3 x pool size)) per prompt per round, keep every distinct verified proof (max 4 per prompt), SFT lr 1e-4, 1 epoch, batch 64, continue from the previous round | 12,288 |

**Change from the design discussed in chat: 1e-3 is replaced by 3e-5.** In the pre-run plumbing
smoke (tiny batch, scratch folder, not an experiment), a single GRPO optimizer step at lr 1e-3
dropped greedy heldout from 10/10 to 2/10. The EI smoke, trained at 1e-4, stayed at 10/10, so
this is not a save/load bug. Adam's first update moves every weight by about +/- lr whatever the
batch size, so the full-size arm would break the same way. 3e-5 fills the gap between exp-00b's
1e-5 and 1e-4 instead.

**Evaluation**, identical for the frozen model and each arm (same k, temperature, seed):
- `eval_set.jsonl` at k=32, temperature 1.0 (6,400 samples);
- greedy on the 1,000-record heldout stride (lengths 2 to 6);
- textbook 72 at k=32, which is reported but never used for any choice (plan T5).

The periodic eval every 12 steps runs greedy on 50 heldout records and 40 eval-set records. The
frozen baseline is computed once in `run_lr3e-5/baseline/` and copied into the other arms.

**Analysis** (`analyze.py`, written before the run). The headline is on eval-set length 7, among
problems the frozen model solves at least once:
- counts of problems that went up, down, or stayed the same;
- a two-sided exact sign test;
- the mean paired difference with a 95% bootstrap CI (seed 0);
- the same mean over all length-7 problems;
- new-capability counts at lengths 7 and 8.

Secondary: avg solve rate with CIs; greedy lost and gained; KL, entropy, mixed-group fraction,
and reward curves.

**Selection rule (the learning rate carried forward).** Among the three GRPO arms, pick the one
with the highest mean paired difference on length-7 problems the frozen model solves at least
once. Exclude any arm whose greedy heldout avg drops by more than 0.02, or whose final KL is
above 1.0. If none qualifies, report "none qualifies". Textbook numbers never enter the choice.

## Expected result (numeric)

- **Pool:** 90 to 170 length-7 prompts kept (roughly 10 to 20% of 864), 5 to 25 at length 8,
  and length 6 at the cap. Pool 150 to 300 in total, so EI's k comes out between 14 and 27.
- **Frozen on the eval set:** length-7 avg solve rate 0.08 to 0.14 (exp-00b stride 0.113), with
  about 30 to 40 of the 160 solved at least once; length 8: 0 to 2 of 40.
- **lr3e-5:** max KL 0.001 to 0.01. Training reward rises a little, from about 0.4. Eval
  length 7 is roughly neutral, with up and down within a few problems.
- **lr1e-4:** max KL 0.01 to 0.2. Training reward rises clearly, by 0.1 or more. On eval
  length 7, more problems go up than down, with a mean paired difference over solved problems of
  0 to +0.08; the sign test may not reach p < 0.05 on one seed. Greedy 2 to 6 drops by no more
  than 0.02.
- **lr3e-4:** this is the risky arm. Either it behaves like 1e-4 with a larger KL, or it
  degrades, with entropy or length drifting and greedy dropping by more than 0.02 (and so being
  excluded). I put 50% on each.
- **EI:** the verified fraction on the pool rises from round 1 to round 3. On eval length 7 it
  is within about 0.03 of the best GRPO arm's mean paired difference, and greedy is unchanged.
- **New capability** (frozen 0 of 32, trained at least 0.5): 0 to 1 per arm, which is noise.
  Length 8: 0 to 2 new solves per arm.
- **Wall-clock:** pool filter 4 to 5 minutes, about 12 minutes per GRPO arm, 5 minutes for EI,
  about 3 minutes per evaluation, 60 to 75 minutes in total.

## What we conclude either way

| outcome | conclusion | next |
| --- | --- | --- |
| an arm qualifies and its held-out length-7 difference is positive | the carried-forward lr; first sign that GRPO transfers at the frontier (one seed) | 3 seeds of that arm (and EI), then exp-02 on a pod at that lr |
| training reward rises in some arm but no held-out difference | sharpening on training prompts without transfer | more prompts and steps (pod), or EI-first |
| no arm's training reward rises | an lr or implementation problem, not "GRPO fails" | debug with the overfit test before any pod |
| every GRPO arm degrades greedy | even 3e-5 is too large for Adam on this model at KL 0.01 | lower lr (1e-5 to 3e-5), a higher KL, or a smaller Adam epsilon or beta2 |
| EI beats every GRPO arm by more than the noise | on this budget, rejection sampling is the stronger algorithm | GRPO vs EI is the main comparison for the week |
| held-out length 7 falls in most arms | the exp-00b drop generalises: RL on this pool trades length-7 transfer for training fit | replay and KL experiments before scaling |
| surprisingly good (new capability of 3 or more, or length-8 avg above 0.05) | check for a reward bug and a train/eval leak first (CLAUDE.md) | reviewer recount before anything else |

## Budget and stop rule

About one hour of laptop CPU, with a hard stop at 100 minutes. If an arm runs over 20 minutes,
let it finish, drop the last arm still pending (EI last), and record the change in
`status.md`. Quarantine if training code reads any evaluation file, the eval half, or the
textbook problems, or if a fixed component changes. A separate reviewer session recounts the
dumps before `results.md` carries a conclusion.
