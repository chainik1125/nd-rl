# Orchestration: role briefs and the loop

Each role is a separate Claude Code session started by `runrole` (see
`code/tools/orchestration/README.md`) from the fork's `AGENT_POLICY.md`, the role brief in this
directory, and a run-specific brief. Roles never share a session. The loop:

```
question → proposer → brief → gate0 → executor → gate1 → reviewer → gate2 → librarian → gate3
```

| role | brief | starts when | ends by |
|---|---|---|---|
| proposer | `proposer.md` | Dan or the librarian names an open question | committing a run brief with pre-registered expectations |
| executor | `executor.md` + the run brief | gate0 passes on the brief | `RUN<id> DONE <UTC>` (or `DONE <UTC>`) in `STATUS.md`; `~/runs/<run>/executor.done` |
| reviewer | `reviewer.md` | `orchestrator` sees the executor's done line and gate1 passes | `review_<run>.md` committed; `~/runs/<run>/reviewer.done` |
| librarian | `librarian.md` | `reviewer.done` exists and gate2 passes | summary, STATE line, spend row, digest committed to `nd-rl` and merged into the `dan` branch; `~/runs/<run>/librarian.done` |

Conventions every role follows: `AGENT_POLICY.md` in the fork governs; commits carry the
trailers `Edited-by: agent:claude`, `Agent-role: <role>`, `Run-id: <run>`; questions for Dan go
to `QUESTIONS.md` with the default that will be followed; no other human is ever contacted.
