# Orchestration tools

Shell tools for the role loop described in `docs/proposals/orchestration/README.md`. Install
on the agent host with `bash install.sh` (adds them to `~/bin`, a `~/pods.log` line to
`podnew`, and a 10-minute `orchestrator` cron tick).

| tool | does |
|---|---|
| `runrole <role> <run-id> [--repo] [--brief] [--stop] [--model] [--done] [--done-grep]` | starts one role session with the resume loop; the model is read per invocation from `~/.config/nd-rl/model.<role>`, else `~/.config/nd-rl/model` (default `claude-opus-5-5`; since 2026-09-28 every role runs Opus 5.5, pinned by explicit id so a CLI update cannot silently change it), unless `--model` is given, so a running loop follows a change; stops and notifies when the account is out of credits for that model or authentication fails — judged from the CLI's own error result only, never from text the agent read; policy + role brief + run brief as the first prompt; per-role git author; transcripts and driver log under `~/runs/<run-id>/` |
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

