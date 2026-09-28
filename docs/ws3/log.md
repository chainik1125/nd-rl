# Workstream 3 log

Newest entry last. One entry per session: date, what ran, what was learned, what is next, open questions.

## 2026-09-22 RunPod setup

**What ran.** Followed `https://docs.runpod.io/agent-setup.md`: added the `runpod` plugin
marketplace and installed `runpod@runpod` 1.2.0 (user scope); Charles reloaded plugins and
signed in to the hosted MCP server (`https://mcp.getrunpod.io/`) with OAuth. `list-pods`
returned an empty list, so the connection works. No `RUNPOD_API_KEY` is stored yet; OAuth
covers the MCP tools only, so `runpodctl` (file transfer, SSH helpers) is still unauthenticated.

Copied Dan's pod setup (`origin/dan:docs/infra/runpod.md`, written 2026-09-14) into Charles's
account:

- SSH key: the laptop's `~/.ssh/id_ed25519.pub` is the only registered key. `update-ssh-keys`
  replaces the whole list, so read `get-ssh-keys` first before adding a second machine.
- Template `nd-rl-dev-a40`, id `142gumkku0`: image
  `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`, 30 GB container disk, 20 GB persistent
  `/workspace`, port `22/tcp` only, SSH on, Jupyter off, `allowedCudaVersions` 13.0 and 13.2.
  The CUDA 13 host constraint is needed: `code/uv.lock` pins torch 2.14.0 on Linux, which is a
  cu130 wheel, the same reason Dan gives.
- Pod: `create-pod` with that template, `NVIDIA A40` x1, Secure Cloud, $0.49/hr, **failed with
  402 Payment Required**: the account has no balance. `list-pods` confirmed nothing was created.

**What was learned.** Dan's default is 1x A40 48 GB on Secure Cloud ($0.49/hr), with RTX 3090
for cheap sweeps and A100 80 GB only when A40s are out or memory demands it. At the time of
the check, Secure A40 stock was HIGH (EU-SE-1 HIGH, CA-MTL-1 MEDIUM) with CUDA 12.8, 13.0 and
13.2 hosts all available. Dan's SSH gotchas: registered keys reach a new pod two to seven
minutes after boot, and the external SSH port changes on every stop/start.

**Next.** Charles adds credit to the RunPod account, then re-run the pod create (startup.md
step 12), SSH in, confirm `nvidia-smi` and `torch.cuda.is_available()`, and record throughput
on the CPU smoke test. Steps 7 to 11 of `startup.md` (evaluator, HF wrapper, stage scripts,
orchestration, CPU smoke test) need no GPU and can proceed meanwhile.

**Open questions.** Whether to also store a `RUNPOD_API_KEY` in `.env` so `runpodctl` and the
orchestration scripts (`pod_watchdog`, `killswitch`, `rpbalance`) can authenticate; Dan's setup
uses a key, not OAuth, for that reason.

## 2026-09-23 exp-00 local check (executor session)

**What ran.** Branches: `charles` renamed to `charles-pre-ws3` (git cannot hold `charles` and
`charles/main` at once; noted in `questions.md`), `charles/main` created with the setup commit
(CLAUDE.md, plan, startup, fork_pin, textbook72, uv env), then `charles/exp-00-local-check` with
one commit per build step: `code/ws3/eval/` (binary reward through `strict_verify`, evaluator
with Wilson CIs, textbook slices, T1 next-token accuracy; 13 tests), `code/ws3/hf_wrapper.py`
(the fork's KV cache wired into HF `generate` via a model-managed cache; logits within 1e-4 of
the fork model; TRL `GRPOTrainer` takes it as policy and reference; 10 tests), and
`code/ws3/run.py` with stages `baseline`, `train`, `evaluate`, `report` (smoke pipeline test,
24 ws3 tests in all). Then exp-00 itself in `experiments/ws3/exp-00-local-check/`, all on the
laptop CPU in 14 minutes: proposal committed first (gate A), `numbers.md`, `status.md` ending
`DONE`, `results.md` with 20 raw samples. No pod, no GPU, no spend.

**What was learned.**
- Without a KV cache the wrapper generated 2 completions/s on CPU; with the fork's cache inside
  HF `generate` it is 50 to 108/s for evaluation and 32/s for GRPO including the optimizer step.
  That single change is what made exp-00 fit the hour. transformers 5 lets a model manage its
  own cache when `_supports_default_dynamic_cache()` is False; no `Cache` subclass was needed.
- T1 gate passes: next-token accuracy 98.35%, loss 0.067 on all 5,000 heldout records. Greedy
  by length 100 / 99 / 95 / 87 / 72.5% for lengths 2 to 6. Textbook 10 of 72 at k=64 and 0 of
  43 at reference length 7 or more (the project's previous best). Long pool 14 of 82 at true
  length 7, none above.
- 50 GRPO steps (32 prompts x group 8, lengths 3 to 5, lr 1e-5, KL 0.01) moved mean reward from
  0.876 to 0.923 with max KL 0.0017: the chain works, but the policy barely moved and no
  evaluation changed beyond noise. Two thirds of the groups had zero reward variance, so
  lengths 3 to 5 are too easy to carry signal; exp-01/02 should weight lengths 5 to 7 and
  expect to need a larger learning rate than 1e-5.
- TRL 1.13 defaults `gradient_checkpointing=True`; the wrapper declines it, so every config sets
  it False (done in `run.py` defaults). TRL's reward `log_metric` hook works for the
  unparseable/invalid fractions.
- The Claude Code background runner kills commands at 10 minutes; long runs are launched with
  `nohup ... & disown` (macOS has no `setsid`) and watched through `stdout.log`.

**What is next.** A separate reviewer session writes `review.md` (recount from the gzipped dumps,
leak check, reward sanity) before anything merges into `charles/main`. Then startup.md step 12:
fund RunPod, launch the pod, rerun the smoke test and exp-00's config on CUDA, and write exp-01's
proposal with a harder prompt mix.

**Open questions.** In `docs/ws3/questions.md`: keep or delete `charles-pre-ws3`; whether
committing gzipped proof dumps (about 1.1 MB per experiment) is acceptable or they should go to
the HF bucket; whether exp-01 may deviate from the startup.md row's lr 1e-5 and lengths 3 to 5
given the zero-variance fraction seen here.

## 2026-09-23 exp-00 review (reviewer session)

**What ran.** A fresh agent session recounted exp-00 from the gzipped dumps with its own code
(`experiments/ws3/exp-00-local-check/review/`), then read `numbers.md` and `results.md` and
checked all 43 claims. Commits `58ae5fb` (Recount) and `fd897c4` (Claims, Verdict), author
"ws3 reviewer".

**What was learned.** Verdict: PASS, not quarantined. 36,836 proofs re-verified with zero
disagreements; every reported number reproduces; gates A and B pass; the control model is
bit-identical to the pinned checkpoint. Two wording errors in `results.md` must not be copied:
the textbook problem "lost" after RL is `textbook_eec1f557…` (not `b5920d…`, which both models
solve), and evaluation sampling ran at 45.6 to 103.4 samples/s (not "50 to 108"). Flag: the
fixed `train.jsonl.gz` shares 110 of 256 `test_short` renaming classes and 6 of 531 `test_long`
classes, so 2 of exp-00's 1,600 RL prompts were in a test-file class. Inherited from the fixed
pre-training data, the test files were never scored, and no number depends on it, but the
quarantine rule's literal wording is met and the decision is Charles's.

**What is next.** Charles reads `review.md`, decides on the test-file overlap (see
`questions.md`), then merges `charles/exp-00-local-check` into `charles/main`. Then the pod.

## 2026-09-23 fixed checkpoint switched to Robbie's Muon export

**What changed.** Charles decided to replace `share_pretrain_4l_seed0/pretrain.pt` with
`runs/039_muon-adamw-finish-5pct_s1/pretrain.pt` from `robbiethompson2018/nd-rl-checkpoints`
(sha256 `418af3f9…`, seed 1). `CLAUDE.md`, `plan.md` section 1a, and `startup.md` updated. The
separate repo `robbiethompson2018/nd-rl-autoresearch-034-muon` returned 401 with our token; the
same runs are mirrored in `nd-rl-checkpoints`.

**Why.** Same architecture (3,210,240 params), tokenizer, data, fork pin, and 300 s budget; only
the optimizer differs (Muon on block matrices, AdamW finish for the last 5%). On Robbie's harness
(functionally identical across the s20 and Muon runs), dev solved after his 4-round EI is
264 / 291 / 254 for 039 vs 122 / 133 / 76 for the old recipe, and the untouched pretrain's
round-1 sample accuracy is 0.034 to 0.048 vs 0.009 to 0.027. Seed 1 is the best 039 seed on that
metric. Not yet evaluated on the 72 textbook problems.

**What is next.** Not done yet, needs a code change: `code/ws3/fork_pin/MANIFEST.json` (checkpoint
path, URL, sha256), `fork_pin/checkpoint_source/` (summary, `pretrain.py`, run README), and
`fork_pin/README.md` still pin the old checkpoint, so `fetch.sh --verify` and gate B will fail
once `ckpt/pretrain.pt` is replaced. Then rerun exp-00's baseline parts (T1 gate, heldout greedy,
textbook 72, `rl_targets` pass@32) on the new checkpoint; exp-00's numbers are for the old one.

**Open questions.** Is the dev-metric seed pick acceptable, or should the textbook baseline be run
on all three 039 seeds first?

## 2026-09-24 fork_pin repinned to the Muon checkpoint

**What ran.** Commit `71b4879`: `fork_pin/MANIFEST.json`, `checkpoint_source/`, and
`fork_pin/README.md` now pin `runs/039_muon-adamw-finish-5pct_s1/pretrain.pt`. The old
`ckpt/pretrain.pt` was replaced through `fetch.sh`; `fetch.sh --verify` passes (13/13) and all 24
ws3 tests pass on the new weights. Also: `plan.md` rewritten shorter, `rl_methods_brainstorm.md`
added, empty `notes.md` and `spend.md` created.

**What is next.** Rerun the baseline parts of exp-00 (T1 gate, heldout greedy, textbook 72,
`rl_targets` pass@32) on the new checkpoint as a new experiment; exp-00's numbers are for the old one.

## 2026-09-25 RL methods coded, not run

**What ran.** No experiments. Code only, on `charles/main`'s working tree (not committed):
`code/ws3/rl/` (advantage transforms, adaptive prompt sampler, `WS3GRPOTrainer`, expert
iteration driver), `code/ws3/configs/presets/` (one YAML per method, combined with the new
`include:` key in `run.py`), a pass@k curve and distinct-proof count in the evaluator, and an
`unparseable_reward` option in `ProofReward`. `tests/test_ws3_rl.py` checks the transforms
against brute-force enumeration and runs CPU plumbing smokes of GRPO (virtual_positive, unlikely,
curriculum) and a 2-round EI; all 44 ws3 tests pass. Also: `rl_methods_brainstorm.md` got a
"why not PPO" section and a literature and Tinker scan (sections K, L).

**What was learned.** TRL 1.13 computes GRPO advantages inline with no hook, so the transforms
replace `advantages` after `_generate_and_score_completions`. TRL 1.13's SFT default
`loss_type="chunked_nll"` needs an HF backbone our wrapper lacks; EI uses `loss_type="nll"`.
MaxRL and W-REINFORCE formulas were checked against the papers.

**What is next.** Rerun the exp-00 baseline on the new checkpoint, now with `pass_at_k` (set
textbook and long-pool k to 256 if the budget allows). Then each method needs a proposal.md
before it runs.

**Open questions.** Commit this to `charles/main` or an `exp-` branch? Hint/prefix methods and
the Tinker spend (see brainstorm, end of section K/L) still need Charles.

## 2026-09-27 headline metric switched to avg solve rate (pass@1)

**What changed.** Charles decided pass@k at large k cannot carry a capability claim. The headline is
now the avg solve rate: each problem's fraction of verified samples, averaged over problems.
Added to `code/ws3/eval/`: `metrics.py` (avg CI, reliable fraction, frontier, k50, AUROC,
per-problem base vs RL with new-capability / sharpened / lost counts), `progress.py`
(`valid_prefix`), `variants.py` (renamings and premise orders, consistency). `evaluator.py`
reports all of these; `run.py` evaluates textbook variants (`eval.variants`), scores known proofs
under both models, and its report leads with the avg. Plan section 5 rewritten around these.

**Checks.** `valid_prefix` agrees with the verifier on every one of the 36,816 exp-00 attempts:
all 13,891 accepted proofs fully valid, all 19,870 rule-stage rejections on the same line, all
187 structural rejections on the same or an earlier line. 59 ws3 tests pass, including the smoke
run through all four stages.

**Open questions.** Variants default to m=4 renamings/premise orders at k=16; revisit the cost once
pod throughput is known. Uncommitted work from 2026-09-25 (`ws3/rl/`, presets, reward and run.py
changes) sits in the same working tree and was built on, not reviewed.

**Later the same day: review of the 2026-09-25 RL code.** `rl/`, presets, EI, and
`unparseable_reward` reviewed; solid, smoke-tested only. Fixed: expert iteration now has its own
plan section 4 metric set (it had failed gate B on 8 GRPO-only metrics), and the report checks the
set matching `run.algo`; the curriculum's fallback no longer samples lengths the ladder has not
unlocked; EI refuses `unparseable_reward` instead of ignoring it. Still open: banning the test-file
renaming classes before exp-01, and whether exp-01 keeps `loss_type: grpo` or moves to `dapo`.
60 ws3 tests pass.

## 2026-09-27 exp-00c: lr sweep and expert iteration on CPU

**What ran.** `experiments/ws3/exp-00c-lr-ei-cpu/`, branch `charles/exp-00c-lr-ei-cpu` (worktree
`../nd-rl-exp-00c`), 56 minutes of laptop CPU. The design:
- `rl_targets` split by renaming class into train and eval halves;
- a training pool of 198 prompts the frozen model solves 1 to 7 of 8 times;
- 3 GRPO arms (lr 3e-5, 1e-4, 3e-4; the planned 1e-3 was replaced after a smoke showed one step
  breaks the model) and expert iteration, each at about 12,300 samples;
- evaluation on 200 eval-half prompts (L7 and L8) at 32 samples.

New code: prompt and eval files with leak checks (`data.py`), EI `samples_total`, and
`analyze.py`. The test-file renaming classes are now banned from file-based RL prompts.

**What was learned (executor numbers, independently re-verified, reviewer recount pending).**
- **Large in-distribution lift.** Held-out L7 avg solve rate went from 0.122 (frozen) to 0.430
  (GRPO 3e-5), 0.391 (1e-4), 0.333 (3e-4) and **0.545 (EI)**. L8 went from 0 to 0.266 for EI
  and at most 0.013 for GRPO.
- **Selection rule: lr 1e-4.** 3e-4 is excluded for forgetting (greedy 2 to 6 dropped 0.097).
- **Textbook flat.** 0.101 to 0.084–0.119 across arms.
- **The verifier session found no bug, loophole or leak.** The gains come from fixing one step
  the frozen model almost had. Diversity collapses to about one proof per problem.

**What is next.** Reviewer recount; 3 seeds of EI and GRPO at 1e-4 and 3e-5; whether any of
this reaches the textbook set; whether the pool filter or the algorithm drives EI's lead.

**Open questions.**
- Report "new capability" as defined (frozen 0/32), given that it is mostly one-step fixes of
  near misses?
- Is the pool filter acceptable as standard practice (it selects on frozen-model success in the
  training half only)?

## 2026-09-28 exp-00c closed: review PASS WITH CORRECTIONS

**What ran.** A separate reviewer session recounted exp-00c from the raw files with its own code
and committed each phase: Recount `54d2790`, Claims `61887dc`, Verdict `0ca481e`. It
re-verified 115,127 rows with 0 disagreements. Gates A and B pass, no quarantine. `results.md`
was rewritten from the Recount with all seven corrections applied, and the branch was merged
into `charles/main`.

**What was learned.**
- **The numbers reproduce exactly.** Held-out L7 avg went from 0.122 to 0.430, 0.391, 0.333 and
  0.545 (lr3e-5, lr1e-4, lr3e-4, EI); EI L8 went from 0 to 0.266. There is no leak under four
  keys, no reward bug, and no loophole: the one lr1e-4 proof shorter than `L_true` is sound.
- **The main correction.** On the pre-registered metric (paired diff on the 39 frozen-solved L7
  problems), EI (+0.245) is below GRPO lr1e-4 (+0.325) and lr3e-5 (+0.273). EI leads only on
  metrics that were not pre-registered. EI vs GRPO is unresolved on one seed.
- **What the lift is.** One repaired step (derive the innermost consequent instead of a vacuous
  `IMPI`) across the generator's nested-implication templates. Case-split and contradiction
  templates stay at 0; the textbook set is flat (0 to 1 of 43, one problem).
- **CIs.** The eval package's `mean_ci` is Wilson-type with n = problems, not the plan's CI over
  problems. `results.md` now uses the over-problems intervals. The evaluator/report should switch
  or label them before the Saturday graphs.

**What is next.** Seeds (EI, GRPO 1e-4 and 3e-5); an unfiltered-pool comparison; a probe of the
case-split and contradiction templates and the textbook set.

**Open questions.**
- "New capability" wording for one-step fixes.
- The pool filter as standard practice.
- The `mean_ci` method in the evaluator.
- The reviewer's process flag: the queued arms ran during a stop-and-check.
