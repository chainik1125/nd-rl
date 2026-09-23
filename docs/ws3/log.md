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
