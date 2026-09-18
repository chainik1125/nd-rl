---
written_on: 2026-09-14
written_by: agent:claude
---

# RunPod setup for independent experiments

## Summary

How to set up a personal RunPod account so that you, or an agent acting for you, can start
right-sized GPU pods on demand. It records the decisions made, the costs expected for this
project's model sizes, and what has been checked so far.

**Status (2026-09-14): validated on a GPU.** On a Secure Cloud A40 pod with a CUDA 13.0 host,
`make -C code install-dev` succeeded. The project's locked `torch 2.14.0+cu130` ran CUDA work,
and all 53 R1 tests passed with GPU tests enabled
(`R1_TEST_CUDA=1 R1_REVIEW_DEVICE=cuda`). An earlier RTX 4090 pod on a CUDA 12.8 host passed the
51 CPU tests but failed CUDA initialization; see the CUDA requirement under step 5. Both test
pods were deleted afterwards.

## Decisions

- **Provider: RunPod.** The project already uses it ([infra overview](README.md)). It bills
  by the second, and pods can be controlled by API, CLI or MCP, so an agent can start a pod,
  run a job and remove the pod without a person present. Vast.ai is cheaper (roughly 40–60%)
  for interruptible jobs and is a reasonable backup for throwaway sweeps. Lambda, AWS and GCP
  are aimed at multi-GPU clusters, which this project does not need.
- **Default GPU: 1× A40 (48 GB) on Secure Cloud, $0.49/hr.** The models are 8M and 85M
  parameters, so an H100 buys faster wall-clock time, not more capability. The A40 is the card
  the environment was validated on. It was available with the CUDA 13.0+ host the project
  requires (see step 5), and it costs less than an RTX 4090, whose CUDA 13 hosts were scarce.
  An RTX 4090 ($0.74/hr) is an acceptable fallback when a CUDA 13 host is free. Move to an
  A100 80GB or larger only if memory or throughput becomes a real constraint.
- **Secure vs Community Cloud.** Use Community Cloud (about half the price) for diagnostic runs
  that can tolerate an interruption. Use Secure Cloud for runs that cannot be interrupted, such
  as a frozen evaluation.
- **Keep the always-on agent separate from GPUs.** Run the long-lived Claude Code or Codex
  session on a small, cheap VPS, and have it start GPU pods only for training and evaluation.
  This way you don't pay for an idle GPU while the agent reads, plans or waits.

## Cost estimates

Live RunPod catalog prices, read on 2026-09-14 (per GPU-hour, one GPU; prices change):

| GPU | Secure | Community | Availability (Secure) |
|---|---:|---:|---|
| RTX 4090 24GB | $0.74 | $0.34 | Medium (best: EU-RO-1); CUDA 13 hosts scarce |
| A40 48GB | $0.49 | $0.35 | Low–medium; CUDA 13 hosts available |
| A100 SXM 80GB | $1.59 | $1.39 | Low |
| A100 PCIe 80GB | $1.59 | $1.19 | None at time of reading |
| H100 SXM 80GB | $3.49 | $2.69 | High |

The estimate is based on the R1-CoT sprint's allocation of two GPUs for a 10-hour window
([strategy](../../experiment-summaries/2026-09-08-r1-cot/strategy.md)), which is about
**20 GPU-hours per sprint**. That is the allocation, not metered usage, so actual use was
probably lower. At five sprints a week:

| GPU (Secure) | Per sprint | Per week | 13 weeks, if sustained |
|---|---:|---:|---:|
| **A40 (default)** | ~$10 | ~$49 | ~$640 |
| RTX 4090 | ~$15 | ~$74 | ~$960 |
| A100 80GB | ~$32 | ~$160 | ~$2,070 |
| H100 SXM | ~$70 | ~$350 | ~$4,540 |

Treat the 13-week totals as upper bounds. Real usage comes in bursts, with little compute
during analysis and writing weeks. Other costs:

- **Always-on agent host:** a 2–4 vCPU VPS costs $5–20 a month. RunPod does not publish fixed
  prices for CPU pods, which makes them harder to budget.
- **Storage:** network volumes cost $0.07/GB-month (under 1 TB) and are billed even while no
  pod is running. Push durable artifacts to HuggingFace, as the infra overview asks.

## Setup steps

1. **Account and billing.** Create a RunPod account, add a payment method, **add credit**
   (pods cannot start with a zero balance), and set a low-balance alert. Before spending
   personal money, ask Dmitry whether the group uses a shared RunPod org or reimburses compute.
2. **Claude Code plugin.** In a terminal where the `claude` CLI is installed
   (`brew install --cask claude-code` on macOS; the VS Code extension does not put the CLI on
   `PATH`), run:
   ```bash
   claude plugin marketplace add runpod/runpod-plugins-official
   claude plugin install runpod@runpod
   ```
   This is RunPod's official plugin ([repository](https://github.com/runpod/runpod-plugins-official)).
3. **API key and MCP connection.** Create an API key in the RunPod console (Settings → API
   Keys). Store it only as the environment variable `RUNPOD_API_KEY`, never in the repo or in a
   chat. Connecting with the key authenticates both the MCP tools and `runpodctl`; OAuth sign-in
   authenticates only the MCP tools.
   ```bash
   claude mcp add --transport http runpod -s user https://mcp.getrunpod.io/ \
     --header "Authorization: Bearer $RUNPOD_API_KEY"
   ```
   Then run `/mcp` in Claude Code and confirm that `runpod` shows **Connected**, not
   *Needs authentication*.
4. **SSH key, before creating any pod.** RunPod copies registered public keys into a pod only
   when it boots. Register a key first, either through the MCP `update-ssh-keys` tool or with
   `runpodctl ssh add-key --key-file ~/.ssh/id_ed25519.pub`. That tool replaces the whole key
   list, so include every key you want to keep.
5. **Pod template.** A private template is a reusable pod configuration. The template created
   for this setup:

   | Field | Value |
   |---|---|
   | Name | `nd-rl-dev-a40` |
   | GPU | Not stored in a template; request `NVIDIA A40` when creating the pod |
   | Image | `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` (PyTorch 2.8, CUDA 12.8) |
   | Container disk | 30 GB (ephemeral) |
   | `/workspace` | 20 GB persistent mount |
   | Ports | `22/tcp` only; SSH on, Jupyter off |
   | CUDA hosts | **13.0 or newer** (required; see below) |

   Private templates belong to one account, so each person creates their own. Only
   `/workspace` survives a pod stop; anything else, including installed packages, is wiped.

   **The host must support CUDA 13.0 or newer.** `code/uv.lock` pins `torch 2.14.0+cu130`.
   On a host whose NVIDIA driver supports CUDA 12.8 at most (driver 570), the project's
   PyTorch cannot initialize CUDA, even though the image's own PyTorch 2.8 works. Set
   `allowedCudaVersions` (template) or `gpu.minCudaVersion` (pod) to 13.0.
6. **Test pod, then automation.** Start one pod by hand and check it end to end before letting
   an agent start pods:
   - SSH in.
   - Run `nvidia-smi` and confirm `torch.cuda.is_available()`.
   - Clone `nd-rl` into `/workspace`.
   - Run `make -C code install-dev` and the CPU tests in `code/tests/test_r1_*.py`.
   - Terminate the pod and confirm billing stopped.

   For throwaway test pods, use `runpodctl pod create ... --terminate-after <time>` as a cost
   guard. For a dev box you will come back to, use `--stop-after` instead; a stopped pod keeps
   its disk and still pays for it.

## Always-on agent host (interim)

A small VPS runs long agent sessions so they don't depend on a laptop. Dmitry may later provide
a standard host; until then, this is the working setup (set up 2026-09-14).

- **Provider and size:** DigitalOcean droplet, Ubuntu 24.04, 2 vCPU / 2 GB RAM plus a 2 GB swap
  file. 2 vCPU / 4 GB (about $24/month) is the safer size if several Claude Code sessions run
  at once. Hetzner's EU CX33 (4 vCPU / 8 GB, about €8.50/month) is cheaper if its signup ID
  check isn't a problem. Its US regions got more expensive in 2026.
- **Hardening:**
  - Log in as a regular user (`dan`) with passwordless sudo and SSH keys only.
  - Root login and password login are disabled (`/etc/ssh/sshd_config.d/99-hardening.conf`).
  - The `ufw` firewall allows inbound SSH only.
  - Ubuntu's automatic security updates are on by default.
  - Before disabling root login, confirm the new user can log in and use sudo.
- **Tools:** `git`, `tmux`, `jq`, `gh` (GitHub's apt repository), `uv` (Astral's installer),
  `runpodctl` (`curl -sSL https://cli.runpod.net | sudo bash`), Claude Code
  (`curl -fsSL https://claude.ai/install.sh | bash`), and the RunPod Claude Code plugin.
- **Credentials (separate from your laptop's; each can be revoked on its own):**
  - **Pod SSH:** `~/.ssh/runpod_ed25519` has no passphrase and is set as the default key in
    `~/.ssh/config`. Its public key is registered on RunPod next to the laptop key. The
    key-update call replaces the whole list, so read the current keys first.
  - **RunPod:** use a separate API key for the host, stored as `RUNPOD_API_KEY` in
    `~/.config/nd-rl/env` (chmod 600, sourced from `~/.bashrc`).
  - **GitHub:** first try a fine-grained token with Contents and Pull requests set to
    read/write. If `gh repo view chainik1125/nd-rl` then fails, fall back to a classic token
    with `repo` scope and a short expiry; fine-grained tokens may not reach repositories you
    collaborate on but don't own. Log in with `gh auth login --with-token`.
- **Working pattern:** run each long session in `tmux` so it survives disconnects. The agent
  starts pods only for GPU work and checks for leftover pods at the end of the session.
- **Cost guard:** `runpodctl` 2.14.0 has no `--terminate-after` flag. Wrap pod work in a
  script whose `trap ... EXIT` runs `runpodctl pod remove`, and start it with `nohup` so a
  dropped SSH connection doesn't skip the cleanup. `~/pod_smoke_test.sh` on the host
  follows this pattern.
- **Checked 2026-09-14:** from the host, the smoke-test script created a Secure Cloud A40 pod
  from the `nd-rl-dev-a40` template (CUDA 13.0 host). The host's own key was accepted after
  about 30 seconds, `nvidia-smi` and PyTorch saw the GPU, and the script deleted the pod.
  Total run time was about 40 seconds.
- **MCP pitfall:** run `claude mcp add ... --header "Authorization: Bearer $RUNPOD_API_KEY"`
  only after the key is loaded into the shell. Otherwise it registers an empty `Bearer`
  header and `claude mcp list` shows HTTP 401.

## Gotchas

- **GPU stock filtered by CUDA version is thin.** On 2026-09-14, no Secure Cloud RTX 4090 with
  a CUDA 13 host could be created, although the catalog briefly listed one. An A40 (48 GB,
  $0.49/hr Secure) with a CUDA 13 host was available and is a good, cheaper default for these
  model sizes. Check `list-gpu-types` with `include=AVAILABILITY`, `product=POD` and
  `minCudaVersion=13.0` before choosing a GPU.
- **SSH keys reach the pod a few minutes after boot.** sshd accepted connections soon after
  start but rejected the registered key for about two minutes on the RTX 4090 host and about
  seven minutes on the A40 host.
  Retry before assuming the key is wrong. `ssh -v` shows `Server accepts key` once the key is
  installed.
- **Passphrase-protected keys need `ssh-agent` for unattended use.** An agent running SSH
  without a terminal cannot type a passphrase. Load the key once in your own terminal with
  `ssh-add --apple-use-keychain ~/.ssh/id_ed25519` (macOS), and never give the passphrase to
  an agent.
- **Getting the private repo onto a pod.** Stream a local checkout over SSH rather than
  putting a GitHub token on the pod:
  `tar czf - --exclude='nd-rl/code/.venv' nd-rl | ssh <pod> 'tar xzf - -C /workspace'`.
  Warnings that ownership can't be changed on `/workspace` are harmless; then run
  `git config --global --add safe.directory /workspace/nd-rl`. On the first pod,
  `make -C code install-dev` took about 6.5 minutes and the R1 CPU tests about 2 minutes.
- A network volume is tied to one data center, and the pod must run in that same data center.
  Check GPU availability there before creating the volume.
- A pod showing "Running" does not mean SSH is ready yet. Retry for up to about 90 seconds.
  The external SSH port changes on every stop and start, so read it again each time.
- An `isError` response from a delete tool may still be a success, because RunPod returns
  204 No Content. Confirm with a follow-up `get-` or `list-` call.
- Agents must stop or delete only pods they created. A pod name is not proof of ownership.

## Next steps

1. Ask Dmitry whether an existing image or environment for the `simplex1` code should replace
   the stock PyTorch image. The `r1_cot` code expects vendored modules on `PYTHONPATH`; see
   [its README](../../code/experiments/current/r1_cot/README.md).
2. Swap in Dmitry's image and standard agent-host setup when they're available, and
   update this note.
