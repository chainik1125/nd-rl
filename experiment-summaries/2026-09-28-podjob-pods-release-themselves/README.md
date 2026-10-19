---
written_on: 2026-09-28
written_by: agent:claude
---

# podjob: can a host-side wrapper tie a pod's lifetime to its job? (proposal 15, item 7)

Status: completed and reviewed 2026-09-28. No hard-constraint violation, nothing quarantined; the
results stand, with three rewordings (applied below). This is an infrastructure run with no model, no
proofs and no training data, so no checker, model label or seed count applies. Code merged into nd-rl
`dan` by this librarian commit; **not installed** on the VPS (see Limitations).

Question: can a host-side wrapper tie a pod's lifetime to its job, so that a pod never outlives the
process that wanted it, without putting the RunPod API key on the pod? *Why:* idle pods had cost
≈ $180 before the current guards. `pod_budget_watch` and `pod_watchdog` cap the damage but do not
remove the cause.

## Configuration

- **Brief:** [`BRIEF_podjob.md`](../../docs/proposals/improvements/BRIEF_podjob.md). Budget $1 and
  2 pod-hours, registered at 19:00:06 UTC, before the first pod.
- **Code:** nd-rl branch `dan_podjob`, commit `4334bfb`:
  [`code/tools/orchestration/podjob`](../../code/tools/orchestration/podjob) plus a README section in
  [`code/tools/orchestration/README.md`](../../code/tools/orchestration/README.md). It also fixes
  `podbg` and `podnew` (see Observations).
  Usage: `podjob <run> [--gpu A,B,…] [--cloud …] [--pack N] [--sync] [--push P]… [--out P]… -- cmd ';' cmd`.
  - It creates a pod `pj-<run>-<rand>` registered to the run and starts each job detached on it
    (`setsid nohup`), at most N at once.
  - It polls every 15 s. It syncs outputs to `hf://buckets/dan-pandori/nd-rl/<run>/` and pulls them,
    then deletes the pod with `podrm`.
  - A host-side trap on EXIT/INT/TERM/HUP runs the same salvage and delete. After that it removes any
    pod that still carries the job's unique name, which covers a signal during creation.
  - No RunPod key goes to the pod.
- **Pre-registration:** fork `preregistration/podjob.md` (commit `94a5ae8`, 19:01:18 UTC). The first
  `podjob START` was at 19:06:48 and the first pod at 19:09:30. The file's text says "written 19:10",
  which is later than its own commit; the reviewer calls this cosmetic.
- **Tests:** T1–T5, one secure pod per test on the cheapest classes that booted the template:
  RTX A4000, RTX 2000 Ada and RTX A5000, template `uhf9wr47j6`, min CUDA 12.4.
  - Raw files are in `~/runs/podjob/tests/` (per-test `.txt`, `.listing.json`, `.bucket.txt` and the
    pulled tree), with host events in `~/podjob.log`.
  - Every "gone" time is an **upper bound**, because the account-listing poll starts only after podjob
    exits.
- **Fork:** worktree `~/work/podjob`, branch `dan_podjob` of `dan-pandori/nd-takehome`, holding the
  pre-registration, `run_podjob.md`, `numbers.md` § podjob and `log.md`.
- **Review:** [`review_podjob.md`](https://github.com/dan-pandori/nd-takehome/blob/fe8978f/review_podjob.md),
  with §Recount in [`414513b`](https://github.com/dan-pandori/nd-takehome/commit/414513b) and
  §Compare/§Verdict in [`fe8978f`](https://github.com/dan-pandori/nd-takehome/commit/fe8978f). The
  reviewer's own recount script is `artifacts/podjob/review/review_recount.py`, which reads the raw
  test files and not the executor's `summarize.sh`.
- **Bucket:** `hf://buckets/dan-pandori/nd-rl/podjob/artifacts/podjob/` holds the tests, summary, logs
  and six job directories. More test outputs are under `hf://buckets/dan-pandori/nd-rl/podjob/artifacts/pjtest/`.
- **Cost:** 7 pods, 0.13 pod-h and **$0.04** by `podhours.log` / `podbudget`, of $1. START→DELETED in
  `~/podjob.log` over the 7 pods is 0.39 h, which is ≈ $0.10 at ≈ $0.25/h.
  - The billed rates the executor gives ($0.25 / $0.24 / $0.27 per hour) are not derivable: the
    registries were deleted and RunPod had posted no billing records yet.
  - The T5-SIGTERM pod `x1qujwzx9ckgq0` (`UNDONE`) has no `podhours.log` entry.
  - T2's create call waited ≈ 11 min (19:16:20 → 19:27:23) before `podnew`'s `CREATED` fix was in
    effect, so its 55 s entry leaves that wait out. Whether the wait was billed is not verified.

## Observations

Recount by the reviewer (pre-registered limit: the pod is gone ≤ 120 s after the job ends or the signal):

| test | podjob exit (pre-reg) | pod gone ≤ (s) | bucket files | verdict |
|---|---|---|---|---|
| T1 normal job | 0 (0) | 15 | 1, pulled ✓ | pass |
| T2 failing job (`exit 3`) | 3 (3) | 48 | 1, pulled ✓ (pod-side `hf` missing, host fallback used) | pass |
| T3 SIGTERM mid-job | 143 (143) | 26 after the kill | 1 (partial) | pass |
| T4 `--pack 3`, three 60-s jobs | 0 (0) | 33 | 3 | pass on **substitute evidence**: wall 65 s ≤ 90, intervals overlap, one host `456e10895074`, one pod id. The registered condition "same `RUNPOD_POD_ID`" was **not met**: every job printed `pod=?` because the variable is not in the job's environment |
| T5 as registered: **SIGINT** during creation | **0** | 79 | – | **fails as registered.** The signal was ignored, and podjob created the pod, ran the job and exited 0. bash ignores SIGINT in `&`-started children of a non-interactive script, and it cannot be trapped there. The pod was still removed, but only by the normal path |
| T5 re-run with **SIGTERM** during creation | 143 | 74 (bounded by podjob's 60 s create-grace loop) | 0 (no job ran) | pass with the substituted signal. The half-made pod was removed 2 s after the signal and was already absent from a listing at +27 s |

In the reviewer's words: **T1–T4 pass; T5 as registered (SIGINT) failed, and passed with SIGTERM.**
The executor's "all pass" is superseded by this.

- **No leftover pods.** No saved listing has a `pj-` pod beyond the tests running at that moment. The
  reviewer's listing at 19:4x UTC and the librarian's at 19:5x have none.
- **Design conditions:** checked by reading the code.
  - No RunPod key reaches the pod.
  - Signal waits use `sleep & wait`.
  - Cleanup is guarded against re-entry and ignores INT/TERM/HUP while it runs.
  - SIGKILL is left to `pod_watchdog` / `pod_budget_watch`, as the header states.
- **Two bugs in existing tools were found and fixed.** Both are deviations from the pre-registration,
  which said "existing `pod*` tools unchanged". The executor did not flag them as deviations; the
  reviewer judges both fixes correct and minimal.
  - `podbg`: `a && setsid … & disown` backgrounded the whole `&&` list, so ssh was held until the job
    ended. Measured for a 40-s job: 42 s before the fix, 4 s after. The first T1 run on v1 of `podjob`
    showed the same defect: the host logged the job start 30 s late. The T1 re-run agrees with the pod.
  - `podnew`: it now stamps `CREATED` at the create call rather than after its wait of up to 12 min,
    so that wait is now logged as pod time.

*Interpretation (the executor's, which the reviewer accepts):* a pod's lifetime can be tied to its
job from the host side alone. SIGTERM, a job failure and a normal exit all clean up well inside 2
minutes, without the API key on the pod.

## Limitations and next step

- **SIGINT from an interactive terminal is untested.** The code header claims a foreground Ctrl-C is
  trapped, but only the backgrounded case was tried, and it failed. Stop a backgrounded podjob with
  SIGTERM (the README says so).
- **The T4 `RUNPOD_POD_ID` condition** needs either the variable exported into the job's environment
  from the registry, or the condition dropped from the acceptance wording.
- **Code notes from the reviewer (not blocking):**
  1. With ≥ 10 jobs the exit code comes from lexical job order (`job10` before `job2`).
  2. If `runpodctl pod remove` keeps failing, the unregistered-pod loop never ends.
  3. The pod-side `hf` install runs in the background and races the first sync. It failed on T2 and
     T5, and the host fallback covered both.
- **Not installed.** The brief allows install only when no pods are live on the host, and other runs'
  pods (`lp-*`, `sf-*`, `ss*`) were up at the end of the run and at 19:5x. Until
  `bash code/tools/orchestration/install.sh` runs on a pod-free host, the `podbg` and `podnew` fixes are
  merged but not in `~/bin`.
- **Policy text** ("prefer `podjob` for one-shot jobs") is proposed in the fork's `QUESTIONS.md`
  (2026-09-28 19:40 UTC) for Dan to adopt. It is not in `AGENT_POLICY.md`.
- **Next:**
  1. Install on a pod-free host.
  2. Run one foreground Ctrl-C test and export `RUNPOD_POD_ID` (one cheap pod).
  3. Check RunPod billing for pods `82q4i3bttq5co3` and `x1qujwzx9ckgq0` once it posts, to settle the
     creation-wait accounting.
