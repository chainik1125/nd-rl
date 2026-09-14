---
written_on: 2026-09-14
written_by: agent:claude
---

# RunPod setup for independent experiments

## Summary

How to set up a personal RunPod account so that you, or an agent acting for you, can start
right-sized GPU pods on demand. It records the decisions made, the costs expected for this
project's model sizes, and what has been checked so far.

**Status:** account, MCP connection, SSH key and pod template are set up. The first test pod
has **not** started: RunPod refused it because the account had no balance (HTTP 402). The
environment is not yet validated on a GPU.

## Decisions

- **Provider: RunPod.** The project already uses it ([infra overview](README.md)). It bills
  by the second, and pods can be controlled by API, CLI or MCP, so an agent can start a pod,
  run a job and remove the pod without a person present. Vast.ai is cheaper (roughly 40–60%)
  for interruptible jobs and is a reasonable backup for throwaway sweeps. Lambda, AWS and GCP
  are aimed at multi-GPU clusters, which this project does not need.
- **Default GPU: 1× RTX 4090 (24 GB) on Secure Cloud.** The models are 8M and 85M parameters,
  so an H100 buys faster wall-clock time, not more capability. Move to an A100 80GB or larger
  only if memory or throughput becomes a real constraint.
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
| RTX 4090 24GB | $0.74 | $0.34 | Medium (best: EU-RO-1) |
| A100 SXM 80GB | $1.59 | $1.39 | Low |
| A100 PCIe 80GB | $1.59 | $1.19 | None at time of reading |
| H100 SXM 80GB | $3.49 | $2.69 | High |

The estimate is based on the R1-CoT sprint's allocation of two GPUs for a 10-hour window
([strategy](../../experiment-summaries/2026-09-08-r1-cot/strategy.md)), which is about
**20 GPU-hours per sprint**. That is the allocation, not metered usage, so actual use was
probably lower. At five sprints a week:

| GPU (Secure) | Per sprint | Per week | 13 weeks, if sustained |
|---|---:|---:|---:|
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
   | Name | `nd-rl-dev-4090` |
   | Image | `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` (PyTorch 2.8, CUDA 12.8) |
   | Container disk | 30 GB (ephemeral) |
   | `/workspace` | 20 GB persistent mount |
   | Ports | `22/tcp` only; SSH on, Jupyter off |
   | CUDA hosts | 12.8–13.3 |

   Private templates belong to one account, so each person creates their own. RunPod's
   dev-pod guide tested this image on an RTX 4090. Only `/workspace` survives a pod stop;
   anything else, including installed packages, is wiped.
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

## Gotchas

- A network volume is tied to one data center, and the pod must run in that same data center.
  Check GPU availability there before creating the volume.
- A pod showing "Running" does not mean SSH is ready yet. Retry for up to about 90 seconds.
  The external SSH port changes on every stop and start, so read it again each time.
- An `isError` response from a delete tool may still be a success, because RunPod returns
  204 No Content. Confirm with a follow-up `get-` or `list-` call.
- Agents must stop or delete only pods they created. A pod name is not proof of ownership.

## Next steps

1. Add credit to the RunPod account, then start the test pod from `nd-rl-dev-4090`
   (RTX 4090, Secure Cloud, EU-RO-1, $0.74/hr) and run the checks in step 6.
2. Ask Dmitry whether an existing image or environment for the `simplex1` code should replace
   the stock PyTorch image. The `r1_cot` code expects vendored modules on `PYTHONPATH`; see
   [its README](../../code/experiments/current/r1_cot/README.md).
3. Set up the always-on agent host, and give it `RUNPOD_API_KEY` as an environment variable.
