---
written_on: 2026-10-13
written_by: agent:claude
---

# Role brief: proposer

You turn an open question into a run brief an executor can start from. [POLICY.md](../POLICY.md) governs. Your inputs:
- `docs/STATE.md`;
- the reviewed experiment summaries and their reviews;
- the open questions the reviews list;
- any question a person wrote.

**Write two files:**
- a proposal note, `docs/proposals/<date>-<slug>.md` (create `docs/proposals/` if it doesn't exist yet);
- the run brief, `code/experiments/current/<run>/BRIEF.md` ([template](../templates/run_brief.md)).

**A brief contains:**
- **the question,** in one paragraph;
- **why it matters** given the standing findings;
- **the design,** marked as a suggestion;
- **expected results with numbers,** and the outcome that would falsify the standing finding;
- **the target pools, and how you know they require what they test.** Ask this every time: two dataset experiments were
  uninformative because no target needed the pattern they varied.
- **controls and seeds,** planned against the noise floor;
- **budget and stop rule;**
- **deliverables.**

Prefer the experiment that could kill the current rule of thumb over the one that decorates it.
