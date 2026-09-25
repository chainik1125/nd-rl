# Quiet Codex / Claude loop

Two loops run in parallel, one per agent, each from its own full checkout (see the
[checkouts note](../../../../.claude/notes/autoresearch-checkouts.md)), both pushing to
`autoresearch-pretrain`. Since 2026-09-24 (Lean protocol, from 098) only the Claude loop runs;
Robbie paused Codex. The `nd-rl` checkout is still at 056 and must `git pull` before Codex restarts.
From each checkout's root, source the RunPod credentials and run:

```bash
source .envrc.local
# nd-rl: Codex, gpt-6-astra at reasoning effort max
code/.venv/bin/python code/experiments/current/autoresearch/loop.py --agent codex -- --sandbox danger-full-access
# nd-rl-3: Claude, claude-opus-5-5 at effort max, --dangerously-skip-permissions
code/.venv/bin/python code/experiments/current/autoresearch/loop.py --agent claude
```

Claude runs as `claude --print` on the logged-in claude.ai subscription: the launcher strips
`ANTHROPIC_API_KEY` and a parent Claude Code session's `CLAUDE*` variables from its environment.
Models and effort live in `MODELS` and `agent_argv` in `loop.py`. Each loop refuses to start
an iteration when the RunPod balance is below `MIN_BALANCE`.

The explicit full-access option allows unattended Git commits/pushes and SSH/RunPod calls.
Use it only in a trusted, dedicated research checkout/environment: Codex can execute commands
with your account's filesystem and network access. The launcher does **not** silently disable
the sandbox. You can instead pass your own Codex permission profile after `--`; options are
forwarded unchanged. Ordinary workspace-write protects `.git` and cannot perform this entire
commit-based workflow unattended. See [Codex permissions](https://learn.chatgpt.com/docs/agent-approvals-security).

Codex must already be installed and logged in; Git push authentication and the SSH key expected
by `pod.py` must work.

Each fresh `codex exec` receives [iteration.md](iteration.md), completes one experiment using
[program.md](program.md), and exits. A committed and pushed keep/discard/crash/timeout allows the next
iteration. A CLI error or missing new result stops the loop; there is no automatic retry,
lock, experiment cap, or deadline. Do not run two agents in the same checkout. A loop resumes
only its own unfinished experiment (by the id's agent suffix; unsuffixed ids are Codex's).

For simultaneous tooling work, use the [separate development checkout](../../../../.claude/notes/autoresearch-checkouts.md).

## Restarting an interrupted experiment

Run the same launcher command again. If exactly one preregistration is unfinished, the launcher
validates its committed files, candidate hash, frozen files, and registered seeds before starting
Codex. It also checks for a surviving runner or matching RunPod pods. A valid pending candidate
is expected dirty work, not a blocker: the fresh session is instructed to finish it unchanged.
Uncommitted/mismatched registrations, mismatched candidate code, multiple unfinished experiments,
or surviving attempts stop the restart instead of overwriting work or launching duplicates.

This restarts training for all registered seeds from scratch, not from a checkpoint. Prior
experiment/seed artifact directories move into `artifacts/autoresearch/previous_attempts/` before
launch so stale summaries cannot contaminate the new result. Completed experiments cannot rerun.
The read-only check, which neither commits nor rents pods, is:

```bash
code/.venv/bin/python code/experiments/current/autoresearch/run_idea.py 015_beta2-099 --seeds 0,1,2 --check-only
```

This is a startup safety check, not a lock: do not start competing launchers simultaneously.

## Script-owned commits

The agent chooses the hypothesis and keep/discard decision; scripts own the Git steps:

- `run_idea.py` validates, commits, and pushes only the experiment's `params.json` and
  `pretrain.patch` before renting pods. Committed preregistrations cannot be edited.
- `record.py` records the outcome, regenerates `RESULTS.md`, keeps the candidate or restores
  its registered baseline, then commits/pushes only `results.json`, `RESULTS.md`, and `pretrain.py`.

Neither uses `git add -A` or includes unrelated staged changes. Normal Git hooks remain enabled.
Commit/push failures stop the workflow; a completed result file alone does not advance the loop.
The recorder writes a session-local `completed.json` receipt only after publication succeeds.
Retrying the same command can finish a failed push without rewriting results or an empty commit.
Do not restore the candidate or change the recorded decision/observation before retrying.

## Terminal output

Each session immediately prints `Starting Codex | checking experiment state` and its
diagnostic-log directory, before validation or launching Codex. A pending experiment also prints
`Resuming experiment` with its name and seeds. Then these research milestones are displayed,
with Pacific timestamps and experiment/seed labels:

1. Experiment started (after preregistration, when the runner launches).
2. Pretrain started (the harness has loaded its pools and is entering pretraining).
3. RL started (after the pretrain greedy check).
4. RL round finished: cumulative targets solved and new proofs.
5. Experiment finished: recorded seed scores, aggregate versus the registered baseline,
   keep/discard/crash/timeout, and **NEW BEST** only for a kept, strictly higher score.

After the startup message, proposal/edit/preregistration can be quiet for several minutes.
RL solved counts are
cumulative discoveries, not fresh dev-evaluation scores. No Codex commentary, reasoning events,
tool dumps, or per-step loss logs reach the terminal. Errors and cancellation/cleanup notices
are also displayed. If the agent exits without launching an experiment or recording its result,
the loop prints `BLOCKED` plus its last public message (collapsed to one line and capped at
360 characters). CLI failures remain `ERROR`; they also include that message when available.
Neither case automatically retries or changes the experiment's validation requirements.

Diagnostic logs live in ignored `artifacts/autoresearch/loop/session-*/`: `<agent>.jsonl`,
`<agent>.stderr`, launch manifests, publication receipt, per-attempt seed driver logs, and
(when needed) `cleanup.log`.
The session directory is private to your OS account. Ordinary training artifacts still go to
`artifacts/autoresearch/<experiment>_s<seed>/`.

Ctrl+C or SIGTERM stops the current local Codex/runner process groups and terminates the
session's uniquely named pods, including pods found remotely before their local state was saved.
It does not wait for an experiment to finish. Failed cleanup is reported, not silently treated
as success. SIGKILL, host failure, or an unavailable RunPod API cannot guarantee cleanup.

## Tests (no Codex model calls or GPUs)

```bash
code/.venv/bin/python -m pytest -c code/pyproject.toml code/tests/test_autoresearch_{loop,registration}.py
```
