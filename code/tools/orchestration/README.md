# Orchestration tools

Shell tools for the role loop described in `docs/proposals/orchestration/README.md`. Install
on the agent host with `bash install.sh` (adds them to `~/bin`, a `~/pods.log` line to
`podnew`, and a 10-minute `orchestrator` cron tick).

| tool | does |
|---|---|
| `runrole <role> <run-id> [--repo] [--brief] [--stop] [--model] [--done] [--done-grep]` | starts one role session with the resume loop; the model is read per invocation from `~/.config/nd-rl/model` (default `opus`) unless `--model` is given, so a running loop follows a change; stops and notifies when the account is out of credits for that model; policy + role brief + run brief as the first prompt; per-role git author; transcripts and driver log under `~/runs/<run-id>/` |
| `gate0 <repo> <run-id>` | `preregistration/<run-id>.md` committed before the first pod in `~/pods.log` |
| `gate1 <repo> <run-id> [--status] [--done-grep] [--strict-pods]` | done line present; not quarantined; pods (warn or fail); `nd_verify` hash equals main; test marker unchanged; `numbers.md` sources exist |
| `gate2 <repo> <run-id>` | `review_<run-id>.md` committed with Recount, claims table, Verdict; not quarantined |
| `review_workspace <run-id> [repo]` | copy of the run repo with the executor's write-ups removed, for the reviewer's phase 1 |
| `orchestrator` | cron tick over `~/orchestrator.conf` (pipe-separated `<run-id>|<repo>|<done-regex>|[brief]`): done line → gate1 → workspace → reviewer; `reviewer.done` → gate2 → librarian |

| `notify "<title>" "<body>" [priority]` | push to Dan via ntfy (`NTFY_TOPIC`) and email via Resend if `RESEND_API_KEY` is set; used by the driver (done, limit, hard stop), the orchestrator, the kill switch and the balance alert |

| `pod_watchdog` | cron: registered pods but no live `claude -p` process for 30 min → pull, delete this host's pods, notify. Liveness is the Claude process itself, never a tmux name or a wrapper command line |
| `podtoken <pod>` | install the HF token on a pod via stdin |

The driver stops (and notifies) on an authentication failure or three consecutive failed invocations instead of spinning.

| `podbudget [<run>] [--set H USD | --extend H "reason"]` | per-run pod-hour ledger and ceiling; extensions are logged, notified, and refused when the projected spend exceeds the run's declared dollar budget |
| `pod_budget_watch` | cron: warns a run at 80 % of its ceiling (`~/runs/<run>/BUDGET_WARNING`), pulls and deletes that run's pods at 100 % |

Pods are attributed to runs by `ND_RUN_ID` (exported by `runrole`, recorded by `podnew`); `podrm` writes hours and dollars to `~/podhours.log`.

Gate 3 (summary only from reviewed numbers) is the librarian's brief, not a script yet.
