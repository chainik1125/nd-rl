# Run brief: <run-id> — <the question in a few words>

Run id `<run-id>`. Working folder `code/experiments/current/<run-id>/`. Role: executor. Read
`docs/group_docs/agent_research_practices/POLICY.md` with this brief. Authorised by <name>, <date>.

- **Budget:** <dollars / GPU-hours>. Going beyond it is a question, asked with a costed default.
- **Time:** <hard stop, or "no deadline">.

## Question

<One paragraph. What would we believe differently after this run?>

## Why now

<The standing findings it builds on or could overturn, with links to their experiment summaries.>

## Read first

<Summaries, reviews, proposals and papers the executor needs. Name sections where possible.>

## Design (a suggestion)

- **Models:** <checkpoints with labels: parameters, format, from scratch or pretrained, training data>.
- **Arms and controls:** <what differs between arms; the frozen or RL-free control; matched on what>.
- **Seeds:** <count, and the minimum detectable difference at that count>.
- **Pools:** <evaluation sets, and how you know they require what they test; which are scoring-only>.
- **Read-outs:** <plain and guided where relevant; k, temperature, length caps>.

## Expected results (to be pre-registered)

<Numbers or falsifiable statements, and the outcome that would falsify the standing finding.>

## Deliverables

- `preregistration/<run-id>.md`, committed before any compute.
- `run_<run-id>.md` (≤ 400 words + figures), `numbers.md`, `log.md`, compute per arm, and the artefact paths.
- `next_steps.md`.
- The `STATUS.md` line `<RUN-ID> DONE <UTC>`.

**For the reviewer:** <what to recount first; any check specific to this run>.

**Rules specific to this run:** <scoring-only sets, files that must not change, limits on the host>.
