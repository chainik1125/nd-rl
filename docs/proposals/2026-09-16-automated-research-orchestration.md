---
written_on: 2026-09-16
written_by: agent:claude
---

# Automated research orchestration for nd-rl: roles, gates, constraints

## Summary

Three autonomous runs on Dan's fork (`nd-takehome`, branch `dan_novelty`) have shown a stable
pattern: an agent launched from a written brief does the mechanical work reliably (pools,
Stage-1 training, expert iteration on A40 pods, pulling results, writing `numbers.md` as it
goes), and its **interpretations and summaries needed an independent pass every time**. Each
of those passes used the same technique, recomputing the load-bearing numbers from the raw
files with code the reviewer wrote itself. This note proposes turning that pattern into a
system of four agent roles with explicit gates, modelled loosely on Anthropic's automated
researchers write-up (a research agent in a fixed loop; a separate monitor that reviews
proposals before execution and transcripts after, enforcing hard constraints). Human
involvement is reduced to setting the question and the budget, reading reviewed summaries, and
deciding what leaves the group. Nothing in this note is a result; it is a design.

## Evidence base

| run | wall clock, pod cost | what the run got right | what an independent pass had to fix |
|---|---|---|---|
| take-home (2026-09-14/15) | 4.5 h, ≈ $5 | pipeline, single test run with marker | "distinct proofs" inflated ≈ 13× by counting start-index variants; transfer set used for two selection decisions; a 45 % renaming overlap between test-short and training was unreported |
| campaign (2026-09-15/16) | 5.5 h, three pods, ≈ $8 | depth-3 headline reproduced exactly from raw files | a 609-vs-170 two-seed spread called "bistable"; reductio called "a wall" although no target required the pattern; first-round bookkeeping off by one (in-place shuffle) |
| follow-up (2026-09-16) | 4.3 h, ≈ $6 | every count reproduced; bookkeeping bug found and fixed | interpretation caution: one seed's "base generalisation" quoted as if it were a rate |

Two further facts shape the design. The agent's pre-registered expectations were wrong three
times (Phase 1 "mostly elicitation", block B "≈ 0", block C "10–35 %") and each miss was
reported honestly, so pre-registration works as a discipline even when the prediction fails.
And compute has never been the constraint: about $19 of pods for three runs against 14 h of
wall clock, two Claude usage-limit pauses of 30 min each, and several hours of human review.
The system below spends more on review and less on human time.

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
sessions. Where roles run: the present VPS has 2 GB RAM and 2 CPUs, which holds one Claude
Code session comfortably and two only if no data job runs beside them. Until the VPS is
upgraded (below), the executor and the reviewer alternate on it; the proposer and librarian
are short sessions that fit between. With a 4 GB VPS the reviewer of run *n* runs while the
executor of run *n+1* runs, which is the steady state this design targets.

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

- **Per-run pod budget** is in the brief (dollars and maximum concurrent pods). The executor
  writes the pod creation time into `log.md`; the librarian computes the cost from `runpodctl`
  billing at the end and reconciles it against the estimate in `docs/infra/spend.md` (columns:
  run, pods, pod-hours, dollars, estimate, notes). Cumulative spend against the ceiling appears
  in every digest.
- **Kill switch generalised.** Today one cron entry deletes one pod at a fixed time. The
  system's version: every pod created by `podnew` is registered under
  `~/.config/nd-rl/pods/<name>`; a cron job at the run's hard stop pulls `artifacts/` and
  `ckpts/` from every registered pod and then deletes every pod the RunPod account lists, not
  only the registered ones. A second, independent safeguard is the RunPod balance itself, which
  is prepaid.
- **Usage limits.** The driver already detects the limit text, sleeps 30 min, and resumes with
  the instruction to record `LIMIT_HIT <UTC>` in `STATUS.md`. Pods keep running during the
  pause, so a pause costs pod-hours, not correctness; the follow-up run hit it twice for about
  an hour total. With several roles on one subscription the limit will be hit more often, which
  is the main argument for API access below.
- **Subscription vs API.** The Max subscription suffices for one executor at a time. The
  reviewer and executor running concurrently, or two executors, need per-role API keys: they
  remove the shared limit, give per-run cost accounting, and allow effort settings per role
  (reviewer high, librarian low). Token volumes of the three runs, from the driver transcripts:
  40 M / 63 M / 48 M cache-read tokens, 0.35 M / 6 M / 0.42 M cache-write tokens, 0.20 M /
  0.33 M / 0.22 M output tokens. At list prices for the model that ran them (cached table
  2026-06-24; cache reads $0.25 per M, output $50 per M) that is roughly $25 / $110 / $30 per
  run, mostly cache traffic; a review session is about a third of an execution. Opus 5 is
  about half that. These are estimates from usage fields, not invoices.

## Infrastructure changes to request from Dan

| change | reason | rough cost |
|---|---|---|
| VPS 4 GB / 2 vCPU (or two 2 GB VPSs, one for executor, one for reviewer + librarian) | 2 GB holds one Claude Code session; the reviewer must overlap the next executor; data jobs already had to move to pods for RAM | about $24 / month for 4 GB on DigitalOcean, or $12 / month per extra 2 GB droplet |
| API keys per role (executor, reviewer, librarian, proposer), each with its own spend cap | concurrency without a shared usage limit; per-role cost accounting; role-specific effort | $30–110 per run at list price; a cap of $300 / month covers a run every two days |
| more concurrent A40s (limit 6 instead of 3), same template | seed-heavy designs (the ignition study below is 16 arms) are wall-clock bound at 3 pods, not budget bound; A40 at $0.49 / h is already oversized for a 4-layer d = 256 model | none beyond usage; ≈ $12 / day per extra pod when busy |
| cheaper GPU type for Stage-1 sweeps (RTX A5000 / 3090 class, CUDA 13 host) | Stage-1 is 5 min per model and does not need 46 GB | ≈ $0.2–0.3 / h; saves little today, matters at 50+ seeds |
| larger GPU (A100 / H100 class) only when a brief needs it | none of the current experiments do; a cap-16 or 12-layer d = 768 pretraining on the 723 k-class pool would be the first | ≈ $2–3 / h, per brief, not standing |
| shared HuggingFace bucket for checkpoints and coverage shards, per `docs/infra/README.md` | the fork holds 1.5 GB of checkpoints (13 MB each) and 4.9 GB of artifacts outside git; the reviewer needs the executor's exact files, and the librarian needs a stable location to cite; automatic upload at `DONE` is the rule the infra note already states | free tier suffices at this size; one token per role, write for executor, read for reviewer |

## First workload: the ignition study, through the roles

The question: expert iteration on hard targets is seed-driven (one f = 0.1 depth-3 arm ignited
at round 7 and reached 0.122 while its sibling had 86 depth-3 theorems at round 2; reductio
f = 0 solved 58 / 0 / 0 across seeds). What determines whether and when an arm ignites, and is
it predictable from the Stage-1 model before any RL?

1. **Proposer** writes the brief: 8 new Stage-1 seeds on the depth-3 f = 0 set `a1` and 8 on
   the reductio f = 0 set; for each model a pre-RL base rate of the pattern on 300 targets at
   pass@2,000 with per-proof hit counts; identical expert iteration (k = 32, 8 rounds) on all
   16; interventions from the round-4 checkpoint of non-igniting arms (k = 128 for a round,
   temperature 1.0 for a round, one round of transfer from a sibling's found proofs, labelled
   as outside data). Pre-registered expectations: ignition round ≈ 1 / (r · k · N) rounds for
   base rate r; round-8 acquisition depends only on ignition before ≈ round 5; base
   generalisation of the reductio shape in a stated fraction of draws. Budget: three A40s,
   ≈ $25, hard stop. Gate 0: the expectations commit precedes the first `podnew`.
2. **Executor** runs it on the fork, writes `ignition.md` (≤ 400 words, four figures),
   `numbers.md` and `log.md` sections, pulls every file, deletes the pods, ends `STATUS.md`
   with `DONE`.
3. **Reviewer** re-derives: per-arm pattern-theorem counts per round with its own predicates
   and min-round-per-normalised-proof bookkeeping (the bug the follow-up fixed); the base rates
   from the hit-count files; the binomial interval on "fraction of draws that generalise";
   the disjointness of `a1` and the reductio set from the target pools; that the intervention
   arms started from the round-4 checkpoint and not from a later one; that the "transfer from
   a sibling" arm is labelled as outside data everywhere it appears. Verdict per claim.
4. **Librarian** writes `experiment-summaries/2026-09-17-ignition/README.md` from the review,
   adds a line to `docs/STATE.md`, the spend row, and the digest entry. If the base rate
   predicts nothing, the summary says so as the result.

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

1. Dan decides on the VPS upgrade and per-role API keys (the two items that unblock
   concurrency).
2. Write the four role briefs as files under `docs/proposals/orchestration/` on this branch,
   starting from the current `HANDOFF_BRIEF.md`, `CAMPAIGN_BRIEF.md` and the reviewer's method
   in `review_campaign.md`.
3. Run the ignition study through the four roles as the first end-to-end test, with the
   reviewer as a separate session from the executor, and report in the first digest what the
   gates caught.
