# Orchestration tools

Shell tools for the role loop described in `docs/proposals/orchestration/README.md`. Install
on the agent host with `bash install.sh` (adds them to `~/bin`, a `~/pods.log` line to
`podnew`, and a 10-minute `orchestrator` cron tick).

| tool | does |
|---|---|
| `runrole <role> <run-id> [--repo] [--brief] [--stop] [--model] [--done] [--done-grep]` | starts one role session with the resume loop; the model is read per invocation from `~/.config/nd-rl/model.<role>`, else `~/.config/nd-rl/model` (default `claude-opus-5-5`; since 2026-09-28 every role runs Opus 5.5, pinned by explicit id so a CLI update cannot silently change it), unless `--model` is given, so a running loop follows a change; stops and notifies when the account is out of credits for that model or authentication fails — judged from the CLI's own error result only, never from text the agent read; policy + role brief + run brief as the first prompt; per-role git author; transcripts and driver log under `~/runs/<run-id>/`; messages queued by `runmsg` go at the top of the next prompt, and the driver wakes early from its gap when one arrives |
| `runmsg <run-id> <role> [--now] [message …]` | message a running session (stdin if no words). Without `--now` it is delivered at the agent's next invocation (within ~10 s if the driver is waiting); `--now` stops the invocation in progress — the conversation is kept — and the driver resumes it at once with the message, not counted as a failure. Added 2026-09-29 after a note put in `STATUS.md` for a running executor was committed but never acted on |
| `gate0 <repo> <run-id>` | `preregistration/<run-id>.md` committed before the first pod in `~/pods.log` |
| `gate1 <repo> <run-id> [--status] [--done-grep] [--strict-pods]` | done line present; not quarantined; pods (warn or fail); `nd_verify` hash equals main; test marker unchanged; `numbers.md` sources exist |
| `gate2 <repo> <run-id>` | `review_<run-id>.md` committed with Recount, claims table, Verdict; not quarantined |
| `review_workspace <run-id> [repo]` | copy of the run repo with the executor's write-ups removed, for the reviewer's phase 1 |
| `orchestrator` | cron tick over `~/orchestrator.conf` (pipe-separated `<run-id>|<repo>|<done-regex>|[brief]`): done line → gate1 → workspace → reviewer; `reviewer.done` → gate2 → librarian |

| `notify "<title>" "<body>" [priority]` | push to Dan via ntfy (`NTFY_TOPIC`) and email via Resend if `RESEND_API_KEY` is set; used by the driver (done, limit, hard stop), the orchestrator, the kill switch and the balance alert |

| `pod_watchdog` | cron: registered pods but no live `claude -p` process for 30 min → pull, delete this host's pods, notify. Liveness is the Claude process itself, never a tmux name or a wrapper command line |
| `podtoken <pod>` | install the HF token on a pod via stdin |

The driver stops (and notifies) on an authentication failure or three consecutive failed invocations instead of spinning.

| `podreconcile` | drops registry entries for pods that no longer exist on the account; called first by `pod_watchdog` and `killswitch` so a pod deleted elsewhere cannot look like an orphan for ever |
| `podbudget [<run>] [--set H USD | --extend H "reason"]` | per-run pod-hour ledger and ceiling; extensions are logged, notified, and refused when the projected spend exceeds the run's declared dollar budget |
| `pod_budget_watch` | cron: warns a run at 80 % of its ceiling (`~/runs/<run>/BUDGET_WARNING`), pulls and deletes that run's pods at 100 % |

Pods are attributed to runs by `ND_RUN_ID` (exported by `runrole`, recorded by `podnew`); `podrm` writes hours and dollars to `~/podhours.log`.

Gate 3 (summary only from reviewed numbers) is the librarian's brief, not a script yet.

## Starting a run, and keeping the disks from filling (2026-09-27)

- `newrun <run-id> --brief FILE [--base dan] [--hours H --usd D] [--stop "…"]` — one step instead of
  six: refuses if the host has < 10 GB free (and shows what `worktree_gc` could reclaim), creates the
  worktree as a **sparse checkout** without earlier runs' `artifacts/<dir>/` trees (253 MB instead of
  4.1 GB on `dan`), registers the pod budget, adds the orchestrator entry (reviewer and librarian
  follow automatically), and launches the executor in tmux `ex-<run-id>`.
- `worktree_gc` — reports each `~/work/<run>` worktree and each `~/review/<run>` reviewer copy as SAFE
  or KEEP with reasons, and how much removing it actually frees (files with no other hard link;
  `du` on one directory overstates it). Reviewer copies were the largest consumer on VPS1: 16.5 GB. SAFE needs: nothing
  running, librarian done, no modified tracked file, nothing unpushed, and every untracked file either
  in the HF bucket (same path and size) or byte-identical in `~/nd-takehome`. `worktree_gc --delete`
  removes only SAFE worktrees; `--min-age-hours N` also requires the run to have finished ≥ N hours
  ago.
- `disk_watch` (cron, hourly) — at ≥ 85 % disk use, runs `worktree_gc --delete --min-age-hours 24`
  and logs to `~/disk_watch.log`; if usage is still ≥ 85 %, writes one line a day to `~/ALERTS.md` for
  agents. It does not notify Dan (his choice, 2026-09-27, when he also authorised the automatic
  deletion of SAFE worktrees).
- `install.sh` installs every tool in this directory and rewrites the managed cron lines
  (orchestrator, pod_watchdog, pod_budget_watch, balance_floor, disk_watch); it drops one-off dated
  lines such as a leftover `killswitch`.

## Pod tools: fixes of 2026-09-28

- `podnew` never exits silently and never leaves an unregistered pod: a failed create prints RunPod's error
  and removes any pod the call half-created (only pods that did not exist before it); a failed SSH wait
  removes the pod it registered. `POD_CLOUD` (SECURE | COMMUNITY) selects the cloud. The billed rate RunPod
  reports is recorded as `RATE` in the registry and used by `podrm` and `podbudget` (the class table is only
  a fallback — community 3090s bill $0.22/h, not the table's $0.50).
- `podpush` / `podpull` copy into the parent directory, so an existing directory is updated rather than
  nested (`data/r3` no longer becomes `data/r3/r3`). `POD_LOCAL_ROOT` sets the local side (default
  `~/nd-takehome`), so a run can push from and pull into its own worktree.
- `podbg` shell-quotes the command (`printf %q`), so commands containing quotes work.


## `podjob`: compute that releases itself (2026-09-28, proposal 15 item 7)

`podjob <run-id> [--gpu CLASS[,CLASS…]] [--cloud SECURE|COMMUNITY] [--pack N] [--sync] [--push PATH]… [--out PATH]…
[--no-bucket] [--no-pull] [--name NAME] -- <cmd> [';' <cmd> …]` ties a pod's lifetime to its jobs. It:

1. creates a pod named `pj-<run>-<random>`, registered to the run (`ND_RUN_ID`), trying the `--gpu` classes in order;
2. optionally runs `podsync` / `podpush`, then starts each command detached on the pod (`setsid nohup`, in
   `/workspace/nd-takehome`), at most `--pack N` at once, logging to `artifacts/podjob/<name>/job<i>.log`;
3. polls every 15 s; when all jobs have ended (or failed) it syncs the log directory and every `--out` path
   to `hf://buckets/dan-pandori/nd-rl/<run-id>/<path>` from the pod (from the host if that fails), pulls
   them into `POD_LOCAL_ROOT` (default `~/nd-takehome`), and deletes the pod with `podrm` (hours →
   `~/podhours.log`); events go to `~/podjob.log`.

The same salvage-and-delete runs from a host-side `trap` on EXIT / INT / TERM / HUP; a second signal
during cleanup is ignored, and after `podrm` any pod still carrying the job's unique name is removed
(a signal during creation). The RunPod key never goes to the pod, so a `kill -9` of `podjob` is left to
`pod_watchdog` and `pod_budget_watch`, which see the registered pod as before. Exit code: the first non-zero
job exit code, else 0; 128 + signal when interrupted. Stop a backgrounded `podjob` with SIGTERM: bash ignores
SIGINT in commands that a non-interactive script starts with `&`, and cannot trap it there.

Example — three sampling jobs packed on one pod, outputs to the bucket, pod gone when the last one ends:

    nohup podjob myrun --gpu "NVIDIA A40,NVIDIA RTX A6000" --sync --pack 3 --out artifacts/myrun -- \
      python sample.py --seed 0 ';' python sample.py --seed 1 ';' python sample.py --seed 2 > podjob.out 2>&1 &

Use `podnew` / `podrun` / `podrm` for interactive or multi-stage work that needs a pod to persist between steps.

`podbg` fix (same day): `a && setsid … & disown` backgrounded the whole `&&` list, and that subshell held the
ssh channel open until the job ended or `timeout 60` fired (rc 124). Only the job is backgrounded now.

**Notifications (2026-09-30).** Dan gets one `<run>: recorded` message per run when its librarian finishes, naming the
summary; a run's own new questions (lines merged in from `dan` are ignored); and every problem: gate-2 failure or
quarantine, hard stop, out of credits, auth failure, three failed invocations, usage limit, pod watchdog, kill switch,
budget enforcement, balance floor. Session starts, executor and reviewer completions, and the 80 % budget warning are
logged, not sent.
