# Hand-off brief: (1) orchestration proposal for nd-rl, then (2) the ignition study

You are unattended. The human (Dan) has closed his laptop; the session that used to review your
work is gone. Two tasks, in order. Everything in `CAMPAIGN_BRIEF.md` still applies (hard rules,
pod helpers, normalised counts, base reachability with every claim, `LIMIT_HIT` flagging,
numbers with sources as you go). Read `campaign.md`, `followup.md`, `review_campaign.md` and
`review_followup.md` before starting either task.

**Spend policy (new):** Dan has pre-approved pod spend up to **$100 total across all runs**;
about $19 has been spent so far, so you may spend up to ~$80 more without asking. RunPod
balance is ~$131 and is the hard cap. Up to 3 A40 pods at once. Kill switch for pods:
**2026-09-17 16:00 UTC**.

**Sharing rule (standing):** never add reviewers to a PR, never message anyone, never put
results in front of another human. Draft PRs with no reviewers are fine. Dan reviews first.

---

## Task 1 — write the orchestration proposal (documentation only; ~1–2 hours; no pods)

Repository: **`~/nd-rl`** (cloned for you; this is the main project repo, owned by Dmitry, in
which Dan has push rights). For this task only, the "never touch `~/nd-rl`" rule is lifted for
the branch **`dan_orchestration`**, which is already created from `origin/main`. Commit and
push to that branch only; open a **draft** PR against `main` with no reviewers
(`gh pr create --draft`). Follow the repo's note conventions (`AGENTS.md`; YAML front matter
`written_on`/`written_by: agent:claude`; trailer `Edited-by: agent:claude` in commit messages).
Put the proposal at `docs/proposals/2026-09-16-automated-research-orchestration.md`; link it
from `docs/STATE.md` with one line.

### What the proposal is for

Dan wants to move from "one agent run per brief, launched and reviewed by hand" to an
orchestration system that can run, review and iterate on experiments in this project with
little human involvement, modelled loosely on Anthropic's "automated researchers" work
(anthropic.com/research/automated-researchers-mitigate-alignment-failures). What that post
describes, in outline: a research agent runs a fixed loop (literature → method proposal → data
→ training → benchmark evaluation), one problem at a time, with a **separate monitoring agent
that reviews proposals before execution and transcripts after**, enforcing hard constraints
(no capability degradation, no distillation shortcuts, no evaluation tampering). The monitor
flagged cheating attempts in ~2.4% of ~1,600 transcripts — exfiltrating test labels through a
remote API, cherry-picking results — caught because the misbehaviour appeared in the agent's
reasoning. Human oversight was at the constraint-setting and final-reading stages, not per run.

### What you know that the proposal must build on (this is the evidence base — use it)

1. **What the runs looked like.** Three autonomous runs so far on this fork, each launched from
   a written brief by a `claude -p` driver on this VPS with a rate-limit resume loop, GPU work
   on A40 pods via `podrun/podsync/podpull`, results pulled back and pushed to git. Take-home:
   4.5 h, ~$5. Campaign: 5.5 h, three pods, ~$8. Follow-up: 4.3 h, ~$6. Costs are dominated by
   nothing — compute has never been the constraint; judgment and review have been.
2. **What went wrong and how it was caught.** (a) The take-home counted start-index variants of
   the same proof as distinct, inflating "distinct proofs" ~13× in the executive summary; a
   separate review session caught it by re-deriving counts from raw files. (b) The take-home
   used its transfer set for two selection decisions and did not report a 45% renaming overlap
   between test-short and training; the review added both. (c) The campaign called a 609-vs-170
   two-seed spread "bistable" and called reductio "a wall" when its targets never required the
   pattern; the review restated both, and the follow-up then tested reductio properly. (d) The
   campaign's "first round" bookkeeping was off by one because of an in-place shuffle; the
   follow-up found and fixed it. (e) The agent's pre-registered expectations were wrong three
   times (Phase 1 "mostly elicitation" — it was largely creation; block B "≈ 0" — one seed
   composed; block C "10–35%" — flat at 2%), and each time the run reported the miss honestly.
   Pattern: **the agent's raw work was reliable; its interpretations and summaries needed an
   independent pass every time**, and the review always used the same technique — recompute
   from raw files with independent code.
3. **What the infrastructure is.** One 2 GB DigitalOcean VPS (`dan@138.68.44.205`) holding the
   Claude Code CLI (Max 20x subscription), `gh` (classic token, `repo` scope), `runpodctl` with a
   full-scope RunPod key, pod helpers that take a pod name, a cron kill switch, and the two git
   repos. Pod template `nd-rl-dev-a40` (A40, CUDA 13 host, $0.49/hr). Dan is willing to turn on
   API access for agents, add more VPSs, and allow more/different pod types.
4. **Methodological rules the project has converged on** (all learned the hard way): every
   distinct-proof count normalised for start index; every "RL solved X" with a base-model
   reachability number; frozen-resampling control at equal attempts; two seeds minimum before a
   difference is a difference; expected results written down before a block runs; the test files
   run at most once per submission and never inspected; splits disjoint by renaming class;
   negative results written up as such.

### What the proposal must contain (concrete, this project, not generic)

- **Roles**, as separate Claude Code sessions with separate briefs and separate git identities
  in commit trailers: at least a *proposer* (turns an open question into a brief with
  pre-registered expectations), an *executor* (the current run pattern), a *reviewer* (the
  recompute-from-raw-files pass; never the same session as the executor; must produce a
  `review_<run>.md` before any number enters `nd-rl`), and a *librarian/state keeper* (updates
  `docs/STATE.md`, `experiment-summaries/`, `numbers.md` only from reviewed material). Say which
  can run concurrently and where (VPS RAM is the constraint: 2 GB now).
- **The loop and its gates**: brief → pre-registration check → execution → raw artifacts pulled →
  review → summary → next brief. Specify the artefacts each stage must produce and the concrete
  checks at each gate (e.g. the reviewer must re-derive every headline count with code it wrote,
  re-verify a sample of counted proofs, recompute split disjointness, confirm the test-file
  marker, and check that expectations were written before the run started — use the git
  timestamps).
- **Hard constraints the reviewer enforces** (the project's equivalents of the post's
  constraints): no test-file access beyond the single permitted run; no training on evaluation
  pools or renamings; no hand-written proofs as training data; no modification of `nd_verify`;
  counts normalised; base reachability present; controls present; seeds ≥ 2 for any claimed
  difference. State what happens on a violation (run is quarantined, not summarised; human
  notified in `STATUS.md`).
- **Human oversight points**, minimal and explicit: Dan sets the research question and the
  budget, reads the reviewed summary, and decides what is shared with others. Nothing is sent
  to other humans by the system. Propose what a weekly digest for Dan looks like.
- **Budget and safety mechanics**: per-run pod budget, the kill-switch pattern generalised to
  every pod the system creates, a spend ledger the librarian keeps (`docs/infra/spend.md` or
  similar), what the system does when a Claude usage limit hits (the current resume loop),
  and what needs API access vs. subscription.
- **Infrastructure changes to request from Dan**, each with a reason and a rough cost: a
  bigger VPS (or one per role), API keys for the agents, additional pod types (state which
  experiments would need more than an A40), and whether a shared HuggingFace bucket is needed
  for checkpoints per `docs/infra/README.md`.
- **A first concrete workload** for the system: the ignition study you are about to run
  (Task 2) — describe how it would flow through the proposed roles, as the worked example.
- **Failure modes you expect and how the design catches them**, drawing on items 2(a)–(e)
  above and on the post's observation that misbehaviour tends to show up in reasoning. Include
  the one you cannot catch this way.
- Length: aim for 2–4 pages of Markdown. Positive voice. Tables for roles/gates/constraints.
  No new results in it; it is a design document.

Commit, push `dan_orchestration`, open the draft PR (no reviewers), and note the PR URL in
`STATUS.md` here (in `~/nd-takehome`). Then start Task 2.

---

## Task 2 — the ignition study (pods; default-accept spend within the policy above)

Question: expert iteration on hard targets is seed-driven — one f = 0.1 depth-3 arm reached
0.122 after igniting at round 7 while its same-set sibling had 86 depth-3 theorems at round 2,
and reductio f = 0 solved 58 / 0 / 0 across three seeds. **What determines whether and when an
arm ignites, and is it predictable from the Stage-1 model before any RL?** Work on branch
`dan_novelty` in `~/nd-takehome`; results in `ignition.md`; sections appended to `numbers.md`
and `log.md`; `STATUS.md` ends with `DONE <UTC>`. Write your expected results before running.

Suggested design (you own the details; write down deviations):
1. **Many seeds, cheap arms.** Depth-3 f = 0 set `a1` (exists) and the reductio f = 0 set
   (exists): train **8 new Stage-1 seeds each** (5 min per model). For every Stage-1 model,
   *before* any RL, measure the base rate of the target pattern on the target pool with a large
   fixed sample (pass@2,000 per target on 300 targets: enough to see 10⁻³ events; per-proof hit
   counts stored). Then run identical EI (k = 32, 8 rounds) on each. Record per round: number of
   pattern-containing theorems and proofs, plain solve rate, and the round of first pattern proof.
2. **Predictability.** Plot the round-8 acquisition and the ignition round against the
   pre-RL base rate. The claim to test: ignition round is determined by the base rate (an arm
   with base rate r and k × N attempts per round ignites in about 1/(r·k·N) rounds), and
   round-8 acquisition depends only on whether ignition happened before ~round 5. If the base
   rate predicts nothing, say so.
3. **Intervention.** For the arms that did not ignite by round 4, test the cheapest fixes on a
   *copy* of the arm from its round-4 checkpoint: (a) k = 128 for one round; (b) temperature
   1.0 for one round; (c) one round of training on the sibling arm's found proofs (this is a
   different intervention — label it *transfer from a sibling*, it introduces outside data).
   Report which ignites the arm, at what attempt cost, and whether the ignited arm then
   reaches the ~0.34 plateau by round 8.
4. **Base-generalisation rate** (from `review_followup.md`): across the 8 + 8 new Stage-1 seeds
   and the existing ones, what fraction of cap-6 f = 0 draws produce the pattern at all in the
   pre-RL sample? This is a base-model property and should be reported as a fraction of draws
   with a binomial interval.
5. Frozen controls are the pre-RL samples themselves at matched attempts; report them.

Budget guidance: 16 Stage-1 models ≈ 1.5 pod-hours; 16 EI arms ≈ 24 pod-hours shared; the
pre-RL sampling ≈ 4 pod-hours; interventions ≈ 4 pod-hours — roughly $20–25 on three pods over
~10 hours. Well inside the policy. Pull everything and delete pods before finishing.

Deliverables: `ignition.md` (≤ 400 words + figures `figures/ignition_*.png`: acquisition vs
base rate, ignition round vs base rate, per-round curves for all arms, intervention outcomes),
`numbers.md` and `log.md` sections, `STATUS.md` with `DONE`. A separate reviewer session will
re-derive your counts afterwards, so keep every count reproducible from the files you pull back.

---

## Addendum from Dan, relayed 2026-09-16 (read this as overriding the tone above)

The brief above is more prescriptive than Dan intends. What he actually wants:

- **The only hard rule is: do not contact or involve other humans** (no reviewers, no
  messages, nothing sent to Dmitry or anyone else). Everything else in this and earlier briefs
  is either a *standard with a reason* or a *suggestion*.
- **Standards with reasons** (keep them, and say when you deviate and why): the take-home's
  own rules (cap 6 on supervised data, at most one test-file run, `nd_verify` unmodified, no
  hand-written or LLM-written training proofs), and the things that were learned from real
  bugs (start-index-normalised counts, a frozen control at equal attempts, a base-reachability
  number with every "RL solved X", two seeds before calling a difference, expectations written
  down before a run, splits disjoint by renaming class). They exist because each one was
  violated once and produced a wrong number.
- **Suggestions**: the experiment designs, the pod counts, the budget lines, the deadlines,
  the deliverable formats. Use your own judgment. If a different experiment, a bigger one, a
  different kind of pod, or more compute would answer the project's questions better, propose
  it or do it. The $100 figure is a comfort level, not a wall; the RunPod balance is the only
  real cap. Dan is open to using the budget in other ways.
- **Dan is reachable.** He is not watching live, but he reads `STATUS.md` and will answer
  questions. Put anything you want to ask him in `QUESTIONS.md` at the repo root (dated, with
  the default you will follow if unanswered), commit it, and proceed on that default rather
  than blocking. Questions about direction, scope, spend, and what he would find most useful
  are all welcome.
- **The goal** is the SPAR project's question — to what extent does RL against a verifier
  produce genuinely new capability rather than eliciting rare-but-known behaviour, and what
  limits it — not any particular design in these briefs. Work that answers that better than
  the plan is better than the plan.
