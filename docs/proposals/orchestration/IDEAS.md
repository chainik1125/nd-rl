# Orchestration ideas, not yet adopted

**Principle (Dan, 2026-09-30): don't scar on the first cut.** A single incident gets a fix to what broke and
an entry here, not new machinery or new policy. An idea is adopted once a second incident shows it would
have helped. Each entry names its trigger, so the second incident is easy to recognise.

## Session watchdog

- **Idea.** The orchestrator's cron tick relaunches a role session whose driver died mid-invocation, once
  per session. It sends a `runmsg` explaining the restart and preserves the old transcript.
- **Rule for "died mid-invocation":** the role has `<role>.started` but no `<role>.done`; there is no live
  `runrole <role> <run>` process; and the driver log's last line is an `invocation N` start with no
  `exited` line after it.
- **Leave alone** anything the driver ended on purpose: `loop finished`, hard stop, auth failure, out of
  credits.
- **First incident, 2026-09-29.** At ≈ 16:05 UTC `lit-measures` started 160 parallel `hf` downloads on
  agent2. The VPS ran out of memory, and the tmux sessions of `lit-measures` (executor) and `repo-hygiene-2`
  (reviewer) died without a driver log line. Nothing noticed for 8 hours; the orchestrator relaunched both
  by hand at 23:59 UTC. The fix was a policy sentence: at most 4 concurrent processes on the VPS.
- **Adopt when** a second session dies without its driver logging an exit, and nobody notices within an
  hour.
