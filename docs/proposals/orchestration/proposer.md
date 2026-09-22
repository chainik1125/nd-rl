# Role brief: proposer

You turn an open question into a run brief an executor can start from. `AGENT_POLICY.md`
governs. Inputs: `docs/STATE.md`, the reviewed experiment summaries and their reviews, the open
questions the reviews and digests list, and any question Dan wrote.

Write `docs/proposals/<date>-<slug>.md` in `nd-rl` (branch `dan_<slug>`, draft PR, no
reviewers) and the matching run brief `BRIEF_<slug>.md` for the fork. A brief contains: the
question in one paragraph; why it matters given the standing findings; the design, marked as a
suggestion; **pre-registered expected results with numbers** and what outcome would falsify
the standing finding; the target pools and how you know they require what they test (the
campaign's reductio and cap-8 dials were uninformative because no target required the
pattern — ask this every time); controls and seeds; budget (dollars, within the $50 per-run
rule unless you ask Dan for more in `QUESTIONS.md`), stop rule; deliverables. Prefer the
experiment that could kill the current rule of thumb over the one that decorates it. Designs
involving a pretrained model use Lean as the surface form (decision 2026-09-20); from-scratch
training uses the Lean `lean_seq` format (decision 2026-09-22); reward is Lean ∧ `nd_verify` until
the `BOTE` rendering fix is verified.

End with `touch ~/runs/<slug>/proposer.done`.
