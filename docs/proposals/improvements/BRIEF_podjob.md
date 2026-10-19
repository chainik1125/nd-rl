# Run brief: podjob — compute that releases itself (proposal 15, item 7)

Run id `podjob`. The work is in **nd-rl**, not the fork: make your own worktree
(`git -C ~/nd-rl worktree add -b dan_podjob ~/work/podjob-nd-rl origin/dan`) and work in
`code/tools/orchestration/`; the fork worktree `newrun` made for you is unused. Role: executor.
`AGENT_POLICY.md` governs. Budget **$1**, ceiling **2 pod-hours** (tests on the cheapest pod). Authorised by
Dan 2026-09-28 (proposal 15's queue). Read first: proposal 15 §7, the orchestration `README.md`, `podnew`,
`podrun`, `podpull`, `podrm`, `pod_watchdog`, `pod_budget_watch`, `podbudget`.

## Why

Idle pods cost ≈ $180 before the current guards: a pod outlives the agent that wanted it. The guards
(`pod_budget_watch`, the fixed `pod_watchdog`) cap the damage but not the cause.

## What to build

`podjob <run-id> [--gpu CLASS] [--pack N] -- <command> [; <command> …]`: create a pod registered to the run,
run the job(s) — up to N concurrently on one pod — pull or sync the outputs to the bucket, and delete the
pod when the job ends, fails or is interrupted (a `trap` on EXIT/INT/TERM on the host side), so a pod's
lifetime is tied to its job. Log each job's pod-hours to `~/podhours.log` as today. **Do not put the RunPod
API key on a pod** (a pod-side self-destruct would need it); rely on the host-side trap plus the existing
watchdogs. Backward compatible: the existing `pod*` tools keep their behaviour.

## Acceptance (pre-register)

On the cheapest pod: a normal job (pod gone within 2 minutes of the job ending, outputs in the bucket); a
failing job (same); a job killed from the host with SIGTERM (same); `--pack 3` runs three jobs on one pod.
Install on the hosts with `install.sh` only after the tests pass **and** when no pods are live on that host.

## Deliverables

Merged into nd-rl `dan` after acceptance (the reviewer follows); the README section; proposed policy text
("prefer `podjob` for one-shot jobs") in `QUESTIONS.md` for Dan's session to adopt;
`run_podjob.md` in the fork worktree (≤ 250 words) with `STATUS.md` line `PODJOB DONE <UTC>` there (the
orchestrator watches that file); pods deleted; `touch ~/runs/podjob/executor.done`.
