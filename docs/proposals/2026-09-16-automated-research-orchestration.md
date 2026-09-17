---
written_on: 2026-09-16
written_by: agent:claude
---

# Automated research orchestration for nd-rl: roles, gates, constraints

## Summary

Five autonomous runs on Dan's fork (`nd-takehome`, branch `dan_novelty`; the fifth, "round 2",
is in progress) have shown a stable pattern: an agent launched from a written brief does the mechanical work reliably (pools,
Stage-1 training, expert iteration on A40 pods, pulling results, writing `numbers.md` as it
goes), and its **interpretations and summaries needed an independent pass every time**. Each
of those passes used the same technique, recomputing the load-bearing numbers from the raw
files with code the reviewer wrote itself. This note proposes turning that pattern into a
system of four agent roles with explicit gates, modelled loosely on Anthropic's automated
researchers write-up (a research agent in a fixed loop; a separate monitor that reviews
proposals before execution and transcripts after, enforcing hard constraints). Human
involvement is reduced to setting the question and the budget, reading reviewed summaries, and
deciding what leaves the group. Nothing in this note is a result; it is a design.

*Revised 2026-09-17.* The first version of this note was written by the executor of the
ignition study before that study ran. This revision records what has since been built or
decided (the 4 GB VPS, self-service droplet control, the HF bucket, the standing policy file,
the spend rules) and what the ignition study showed about the roles in practice. Sections
that changed: evidence base, where roles run, budget mechanics, infrastructure status, the
first workload, failure modes, next steps.

## Evidence base

| run | wall clock, pod cost | what the run got right | what an independent pass had to fix |
|---|---|---|---|
| take-home (2026-09-14/15) | 4.5 h, ≈ $5 | pipeline, single test run with marker | "distinct proofs" inflated ≈ 13× by counting start-index variants; transfer set used for two selection decisions; a 45 % renaming overlap between test-short and training was unreported |
| campaign (2026-09-15/16) | 5.5 h, three pods, ≈ $8 | depth-3 headline reproduced exactly from raw files | a 609-vs-170 two-seed spread called "bistable"; reductio called "a wall" although no target required the pattern; first-round bookkeeping off by one (in-place shuffle) |
| follow-up (2026-09-16) | 4.3 h, ≈ $6 | every count reproduced; bookkeeping bug found and fixed | interpretation caution: one seed's "base generalisation" quoted as if it were a rate |
| ignition study (2026-09-16, run by the session that wrote this note's first version) | 3.7 h, three pods (a 3090 and two A100s, chosen when no A40 was in stock), ≈ $9 | pre-registration committed before the first pod; all 21 arms' counts reproduced exactly; an orphan pod deleted within two minutes | "k = 128 / T = 1.0 ignited the arms that would have ignited anyway" refined to "no detectable effect in either direction" (the intervention arms' trajectories sit next to the uninterrupted ones); 300-target pre-RL sample under-covers some draws |

Two further facts shape the design. The agent's pre-registered expectations were wrong three
times (Phase 1 "mostly elicitation", block B "≈ 0", block C "10–35 %") and each miss was
reported honestly, so pre-registration works as a discipline even when the prediction fails.
And compute has never been the constraint: about $24 of pods for four runs against 18 h of
wall clock, four Claude usage-limit pauses of 30 min each, and several hours of human review.
The system below spends more on review and less on human time.

A third fact, from Dan's reading of the briefs: the briefs were over-prescriptive. Budget
lines, pod counts, deadlines and suggested designs were written with the same force as the
rules learned from bugs, and the executor optimised the brief rather than the question. The
fix is now in place as `AGENT_POLICY.md` in the fork, which every brief is read together with:
**one hard rule** (no other humans are contacted), a short list of **standards with reasons**
(the take-home's own rules; normalised counts; frozen control; base reachability; two seeds;
pre-registration; class-disjoint splits), and everything else marked **suggestion**. It also
opens a channel to Dan: `QUESTIONS.md`, dated, with the default the agent will follow if
unanswered, so an agent never blocks on a question and never proceeds silently either.

## Roles

Each role is a separate Claude Code session started from its own brief, with its own git
author name and a role trailer, so `git log --author` and the trailers separate who wrote
what. The `Edited-by: agent:claude` trailer from `AGENTS.md` stays; a second trailer
`Agent-role: <role>` and a third `Run-id: <run>` are added.

| role | input | output | may touch | concurrency |
|---|---|---|---|---|
| **proposer** | an open question from `docs/STATE.md` or Dan, the reviewed summaries | a brief: question, design, pre-registered expected results with numbers, budget, stop rule | `docs/proposals/` only | runs between runs; light |
| **executor** | the brief (current run pattern: `claude -p` driver, pods via `podrun`/`podbg`/`podpull`) | raw artefacts (`found_*.jsonl`, `round_*.json`, coverage shards, checkpoints), `numbers.md` entries with sources, `log.md`, `STATUS.md` | the fork's run branch; pods; never `nd-rl` | one per run; holds the pod budget |
| **reviewer** | the pulled raw files and the run's write-up; **never the executor's session** | `review_<run>.md`: every headline count re-derived with its own code, disjointness recomputed, verdict per claim | read-only on the run branch, writes only `review_<run>.md` and its own scripts | starts when `STATUS.md` says `DONE`; can overlap the next executor |
| **librarian** | `review_<run>.md` plus the run's files, only after the reviewer's verdict | `docs/STATE.md`, `experiment-summaries/<run>/`, `numbers.md` in `nd-rl`, `docs/infra/spend.md`, the weekly digest | `nd-rl` on a run branch, draft PR, no reviewers | short; after review |

The driver loop (a bash script per role, the current `run_<run>.sh` pattern: invoke
`claude -p`, detect usage-limit text, sleep 30 min, resume with `--continue`, stop at a hard
stop time or at `DONE`) is not an agent; it is the scheduler and the only process that starts
sessions. Where roles run: the VPS now has **4 GB RAM / 2 vCPU** (resized 2026-09-17), which
holds an executor and a reviewer at once, the steady state this design targets. More VPSs can
be created without a human: `newvps <name>` on the existing VPS creates and hardens a droplet
(the DigitalOcean token lives on the VPS and on Dan's laptop; `doctl` is installed on both).
One lesson from the resize is now a rule: **power actions on a machine are issued from a
different machine**, because the first attempt cut the SSH session that was issuing the
resize and left the droplet off until Dan restarted it by hand.

What has been practiced so far, by role: the executor role four times (four runs); the
reviewer role four times, always as a separate session from the executor (Dan's laptop
session), with the recompute-from-raw-files method each time; the proposer role once by hand
(the five proposals of 2026-09-17, `docs/proposals/2026-09-17-proposals-round-2.md`, which
round 2 is now executing in the order 5 → 2 → 3 → 1 → 4); the librarian role by the reviewer
session (summaries and `STATE.md` lines after each review). No role has yet run as its own
briefed session on the VPS.

## The loop and its gates

```
question → [proposer] brief → GATE 0 → [executor] run → GATE 1 → [reviewer] review_<run>.md
        → GATE 2 → [librarian] STATE / summary / numbers / spend → GATE 3 → next question
```

| gate | who checks | artefacts required | concrete checks |
|---|---|---|---|
| **0 pre-registration** | reviewer session (a short one), before any pod starts | the brief committed on its own branch; `log.md` "expected results" entry | expectations are numeric or falsifiable; commit timestamp of the expectations precedes the first pod creation (from `~/.config/nd-rl/pods/*` mtime and the pod log); budget and stop rule stated; every count the brief promises names its normalisation and its base-reachability companion |
| **1 execution done** | driver + reviewer | `STATUS.md` ends with `DONE <UTC>`; every pod deleted; raw files pulled and listed in `log.md` with sizes; `numbers.md` entries carry a source file each | pod list empty; every file cited in `numbers.md` exists on the VPS; test-file marker unchanged (`artifacts/TEST_RUN_DONE` mtime); `nd_verify` unchanged (hash) |
| **2 review** | reviewer | `review_<run>.md` with a table "claim / independent check / result" and a verdict | each headline count re-derived from the raw files with code the reviewer wrote (own normaliser, own pattern predicates); a ≥ 100-proof sample per arm re-run through `nd_verify`; split disjointness by renaming class recomputed; frozen control present at equal attempts; ≥ 2 seeds behind every difference claimed; expectations written before the run (gate 0 record); wording checked against n (e.g. "bistable" with n = 2 is rejected) |
| **3 summary** | librarian, then Dan | `experiment-summaries/<run>/README.md` per `TEMPLATE.md`; one line in `docs/STATE.md`; `numbers.md` rows tagged with the review that cleared them; `docs/infra/spend.md` row | only numbers that appear in `review_<run>.md` with a passing check are copied; every negative result appears as such; the summary links the review |

An executor cannot open a PR against `nd-rl`; only the librarian can, and only with a review
on file. A number without a review is a number in the fork, not in the project.

## Hard constraints the reviewer enforces

These are the project's equivalents of the post's "no capability degradation, no distillation
shortcuts, no evaluation tampering". All were learned the hard way and are already rules in the
fork's briefs.

| constraint | how it is checked | on violation |
|---|---|---|
| test files run at most once per submission and never inspected | marker file mtime; grep of the transcript (`*_claude_*.jsonl`) for the test paths outside the single run script | quarantine |
| no training on validation-36, the test files, transfer pools, or their renaming classes | class-intersection of every training file against every evaluation pool, recomputed | quarantine |
| no hand-written proofs as training data; pretraining verifier length ≤ cap | assert at load (already in `train.py`); reviewer re-runs the length histogram on the written file | quarantine |
| `nd_verify` unmodified | hash against `main` | quarantine |
| distinct-proof counts normalised for start index | reviewer's own normaliser reproduces the count | number rejected until fixed |
| base-model reachability with every "RL solved X" | presence of the base pass@k or log-prob file for the same targets | number rejected |
| frozen-resampling control at equal attempts | control dir exists, attempts match | number rejected |
| ≥ 2 seeds before a difference is a difference | count of seeds behind each claimed difference | wording rejected |
| expected results written before the run | commit timestamps | run marked "not pre-registered" in the summary |

*Quarantine* means: the run branch is kept, `STATUS.md` gets a `QUARANTINE <UTC> <reason>`
line, nothing from the run enters `nd-rl`, the librarian writes one line in `docs/STATE.md`
saying a run was quarantined and why, and Dan reads it in the next digest. No automatic retry:
a quarantined run needs a new brief.

## Human oversight points

Dan does three things and nothing else is required of him:

1. **Sets the question and the budget.** A one-paragraph question plus a dollar and a
   pod-count ceiling, written into a proposer brief or directly into `docs/STATE.md`.
2. **Reads the reviewed summary.** The librarian's `experiment-summaries/<run>/README.md`,
   which links the review; never the executor's own write-up first.
3. **Decides what is shared.** The system never adds reviewers to a PR, never messages anyone,
   never posts results anywhere a third person reads. Draft PRs with no reviewers are the
   hand-over surface; Dan promotes them.

**Weekly digest** (one file, `docs/digests/YYYY-WW.md`, written by the librarian, ≤ 1 page):
runs started / finished / quarantined this week with one line each; every headline number that
cleared review, with its review link; expectations that were wrong and by how much; spend this
week and cumulative against the ceiling; usage-limit pauses; open questions the proposer
suggests next, ranked; anything that needs Dan's decision, marked as such at the top.

## Budget and safety mechanics

- **Per-run pod budget: $50** unless Dan authorises more for that run (his decision of
  2026-09-17); the prepaid RunPod balance is the only hard cap. No limit on concurrent pods and
  no fixed GPU class: the executor chooses per experiment (a 3090 for Stage-1 sweeps, A100s
  when A40s are out of stock or a brief needs the memory). `rpbalance` reads the balance; a
  daily cron writes an alert line to `~/ALERTS.md` when it falls below $50, and Dan tops up.
  Spend so far ≈ $24; balance ≈ $121. The librarian's `docs/infra/spend.md` ledger is still to
  be created; until then `~/balance.log` on the VPS is the record.
- **Kill switch, implemented.** Every pod created by `podnew` is registered under
  `~/.config/nd-rl/pods/<name>`; `killswitch`, armed by cron at each run's hard stop, pulls the
  result directories from every registered pod, deletes every pod the RunPod account lists (not
  only the registered ones), and ends the agent's tmux session. The prepaid balance is the
  second, independent safeguard.
- **Usage limits.** The driver detects the limit text, sleeps 30 min, and resumes with the
  instruction to record `LIMIT_HIT <UTC>` in `STATUS.md`. Pods keep running during the pause,
  so a pause costs pod-hours, not correctness; four pauses so far across four runs. **Dan's
  decision (2026-09-17): roles share the Max 20x subscription for now.** If an agent judges
  that limits are slowing progress it writes `LIMIT_SLOWING <UTC> <cost>` in `STATUS.md`; on the
  second such flag Dan moves the agents to API keys. The paragraph below is therefore the case
  for that switch, kept for when it is triggered.
- **Subscription vs API.** The Max subscription suffices for one executor at a time. The
  reviewer and executor running concurrently, or two executors, would benefit from per-role API
  keys: they remove the shared limit, give per-run cost accounting, and allow effort settings
  per role (reviewer high, librarian low). Token volumes of the three runs, from the driver transcripts:
  40 M / 63 M / 48 M cache-read tokens, 0.35 M / 6 M / 0.42 M cache-write tokens, 0.20 M /
  0.33 M / 0.22 M output tokens. At list prices for the model that ran them (cached table
  2026-06-24; cache reads $0.25 per M, output $50 per M) that is roughly $25 / $110 / $30 per
  run, mostly cache traffic; a review session is about a third of an execution. Opus 5 is
  about half that. These are estimates from usage fields, not invoices.

## Infrastructure: status of the requests (2026-09-17)

The first version of this note listed six requests. Dan answered all of them; four are done or
decided, two are deferred with a trigger.

| request | status | what exists now |
|---|---|---|
| VPS 4 GB / 2 vCPU | **done** (resized 2026-09-17, ≈ $24 / month) | `doctl` on the VPS and on Dan's laptop with a DigitalOcean token in `~/.config/nd-rl/env` on each; `newvps <name>` creates and hardens a droplet; only Claude login, `gh` token and the RunPod plugin remain manual on a new box |
| API keys per role | **deferred**; trigger: a second `LIMIT_SLOWING` flag in any `STATUS.md` | roles share Max 20x; the resume loop handles pauses |
| more concurrent pods | **no limit** (authorisation, not a change) | the executor decides; the account has shown four pods at once |
| cheaper GPU for sweeps | **agent's choice** | used already: a 3090 for Stage-1 in the ignition study |
| larger GPU when needed | **agent's choice** | used already: two A100s when A40s were out of stock; run 1 of round 2 serves Qwen3-Coder-30B on an A100 |
| shared HuggingFace bucket | **done**: public bucket `dan-pandori/nd-rl` | `hf` CLI logged in on the VPS with a fine-grained write token; `podrun` passes `HF_TOKEN` to pods; `AGENT_POLICY.md` requires `hf buckets sync` of `ckpts/`, `artifacts/`, `data/` at `DONE` and forbids uploading anything that could hold a token (the bucket is public) |

Still open on the system side, none of it needing Dan: per-role git trailers, the
`docs/infra/spend.md` ledger and the weekly digest (librarian), the role briefs as files, and
the gate-0 timestamp check as a script rather than a reviewer's habit.

## First workload: the ignition study, as it actually went through the roles

The question: expert iteration on hard targets is seed-driven; what determines whether and
when an arm ignites, and is it predictable from the Stage-1 model before any RL?

1. **Proposer** (the laptop session, in the hand-off brief): 8 new Stage-1 seeds on the
   depth-3 f = 0 set and 8 on the reductio f = 0 set; a pre-RL base rate per model; identical
   expert iteration; interventions from the round-4 checkpoint of non-igniters. Expectations
   were written and committed at 18:37 UTC; the first pod was created at 18:41 — gate 0 held,
   checked by timestamps.
2. **Executor** (the VPS session): found no A40 in stock, used a 3090 and two A100s and wrote
   the deviation down; deleted an orphan pod within two minutes; found and fixed a bookkeeping
   bug (the round of a normalised proof is the minimum over its raw records); wrote
   `ignition.md`, the `numbers.md` and `log.md` sections, pulled every file, deleted the pods,
   ended `STATUS.md` with `DONE` after 3.7 h and ≈ $9.
3. **Reviewer** (the laptop session, `review_ignition.md`): re-derived the pre-RL hit counts
   for all 21 draws (matching once the per-proof `count` field was used), the per-round
   pattern-theorem counts for all 21 arms element for element, the ignition rounds and final
   acquisitions, and the 11/11 sibling-transfer ignitions; re-ran `nd_verify` on samples. It
   changed one sentence: the k = 128 and T = 1.0 interventions did not "ignite arms that would
   have ignited anyway" — their trajectories sit next to the uninterrupted arms', so the
   accurate statement is "no detectable effect in either direction". It listed two next
   measurements (the pattern's base rate on intermediate checkpoints; a pre-RL sample over all
   targets).
4. **Librarian** (the same laptop session): appended the reviewed results to
   `experiment-summaries/2026-09-15-novelty-campaign/README.md`. No digest, no spend row.

What the gates caught in practice: nothing in the counts (the executor was right), one
over-claim in wording, and one sampling caveat the executor had already flagged itself. What
they did not need to catch: the executor's honesty about its own three wrong expectations was
intact without a gate. The design's cost is one reviewer session per run; its value on this
run was a sentence — which is the expected steady state, and the reason the reviewer stays.

## Failure modes and how the design catches them

| failure mode (seen or expected) | where it shows up | catch |
|---|---|---|
| counting variants of one proof as distinct (take-home, 13×) | the executor's summary | gate 2: reviewer's own normaliser; constraint "counts normalised" |
| selection on the transfer set; unreported overlap (take-home) | `log.md` reasoning, absent from the write-up | gate 2: disjointness recomputed; transcript grep for evaluation files in selection code |
| over-interpreting n = 2 ("bistable"), naming a wall the targets never tested (campaign) | wording in the summary | gate 2: seeds-behind-each-claim count; "does any target require the pattern" check is now part of the target-pool review |
| off-by-one bookkeeping from an in-place shuffle (campaign) | per-round counts | gate 2: min-round rule; per-round counts re-derived, not read from the executor's json |
| wrong pre-registered expectation reported as confirmed | the write-up | gate 0 record compared with gate 2 numbers by the reviewer, not by the executor |
| cherry-picking seeds or rounds; quietly dropping an arm | transcript and file inventory | gate 1: file list in `log.md` against the brief's arm list; reviewer counts arms, not results |
| test-file access, verifier edits, training on evaluation pools | transcript, hashes, class intersections | hard constraints; quarantine |
| pods left running, budget overrun | `podls`, billing | gate 1 pod-list-empty check; kill switch on every pod; prepaid balance |
| usage-limit pause mid-run | driver log | resume loop; `LIMIT_HIT` line; pods keep running |
| over-prescriptive briefs: the agent optimises the brief instead of the question (seen, corrected by Dan) | briefs that state budgets, designs and deadlines with the force of rules; no questions ever asked | `AGENT_POLICY.md`: one hard rule, standards with reasons, everything else a suggestion; `QUESTIONS.md` with a stated default |
| control-plane credential only on the controlled machine (seen: the resize powered off the VPS mid-command) | any power/resize action issued from the VPS to itself | the DigitalOcean token also lives on Dan's laptop; power actions are issued from elsewhere; `newvps` runs from a machine that is not the target |
| the reviewer agreeing with the executor because it read the write-up first | review text mirrors the summary | the reviewer brief gives the raw files and the brief, and withholds the executor's summary until the recount is committed |

The post observed that misbehaviour tends to show up in the agent's reasoning, and the
transcripts here (`*_claude_*.jsonl`) are kept for exactly that reason: the reviewer's
transcript checks are greps over them. **The one this design cannot catch:** a correct count of
a wrong measurement. If the executor and the reviewer both compute a pattern predicate that
does not mean what the brief meant (the campaign's reductio dial: every number reproduced, and
the targets never required the pattern), independent recomputation agrees and the gate passes.
The follow-up caught that case only because a human read the review and asked what the targets
required. The mitigation is the proposer's checklist ("does any target require the pattern;
how do we know") and Dan's reading of the summary; it is not a gate.

## Next steps

1. Write the four role briefs as files under `docs/proposals/orchestration/` on this branch,
   from `AGENT_POLICY.md`, `BRIEF_ROUND2.md` and the reviewer method in `review_*.md`; the
   reviewer brief withholds the executor's summary until the recount is committed.
2. Run the reviewer of round 2's run 5 as its own VPS session (the 4 GB box now allows it),
   concurrently with the executor of run 2, as the first two-session test.
3. Librarian: create `docs/infra/spend.md` from `~/balance.log` and the pod logs; write the
   first weekly digest at `docs/digests/2026-38.md`.
4. Automate the proposer: a session that reads the reviewed summaries and the open questions in
   `docs/STATE.md` and writes the next brief with pre-registered expectations, replacing the
   hand-written proposals of 2026-09-17.
5. Switch to per-role API keys only when a second `LIMIT_SLOWING` flag appears.
