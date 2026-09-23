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
