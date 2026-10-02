# Run brief: radical-scoping — departures from the current path that could answer the capability-emergence question

Run id `radical-scoping`. Repository `~/work/radical-scoping` (branch `dan_radical-scoping` from the fork's `origin/dan`, made
by `newrun`). Role: executor. `AGENT_POLICY.md` governs. Pod budget **$3** / 6 h: CPU feasibility pilots only, and none is
required. Claude time is expected to be heavy. Authorised by Dan 2026-10-02 (proposal 21). Dan meets the team on
2026-10-03, so finish the write-up as early as you can.

## Why

Over two weeks the project built a model organism that solves most of its propositional natural-deduction benchmarks,
and found RL mostly completing proofs that pretraining had brought within reach (`trajectory`, `trajectory-cap6`,
`rl-from-ckpt`). Dan's view (2026-10-02): propositional logic "just isn't that deep of a space"; longer proofs are "not
that interesting"; at some point it makes sense to move to a richer domain. Proposal 20 (exploration) is drafted and
unqueued. The task is to scope **radical departures**, still aimed at the question: to what extent does RL against a
verifier create genuinely new capability rather than elicit rare-but-known behaviour, and what limits it?

Read first:
- nd-rl `docs/project_strategy/2026-10-01-week-in-review.md`, `docs/STATE.md`.
- Proposals 16–20.
- `docs/literature/` (both reviews).
- The June FOL sprint, `experiment-summaries/2026-06-11-fol-sprint`.
- The `trajectory`, `trajectory-cap6` and `rl-from-ckpt` summaries.

## What to do

Scope **4–6 candidate departures.** For each:
- the hypothesis;
- why the current path cannot answer it;
- the setup and domain;
- the **smallest decisive pilot**, with an expected result and a falsifier;
- what it reuses (proof-state environment, Lean judge, trajectory analysis, orchestration);
- cost in pod dollars and Claude time;
- the main risks.

Then rank them by expected information per dollar and per week, and recommend 1–2.

Starting candidates, not a constraint. Add your own and drop weak ones:
1. **First-order logic.** Choosing which term to instantiate a quantifier with is open-ended, and the logic is
   undecidable. Reuse the June FOL sprint's generator and verifier. Check that they still run (CPU), and that Lean
   core checks FOL proofs rendered from them.
2. **Skill knockout.** Remove every pretraining proof that uses one move (for example proof by contradiction, or
   `Or.elim`) and test whether RL, with and without exploration, can acquire it. This is the cleanest in-domain
   creation test. Check feasibility: how many proofs use each move, and whether the generator can avoid it.
3. **Lemma invention and library learning** (DreamCoder- or LILO-style). Here creation means new reusable
   abstractions that shorten proofs.
4. **A different formal domain:** arithmetic with induction in Lean core, or program synthesis with tests.
5. **A different measurement:** a compression or description-length measure of creation (the excess description
   length paper is in `references.bib`).
6. **Self-play conjecturing at our scale.**

Feasibility pilots run on CPU (the VPS or one small CPU pod) and stay small, e.g. generate 1,000 FOL theorems and check
200 in Lean. **Never train a model here.**

## Deliverables

- nd-rl-ready doc `radical_scoping/SCOPING.md` (≤ 2,500 words): a ranked table plus one section per candidate. Mark it
  as a draft for Dan. The librarian copies it to nd-rl `docs/proposals/2026-10-03-radical-departures.md`.
- `run_radical_scoping.md` (≤ 200 words), `log.md`, `STATUS.md` line `RADICAL-SCOPING DONE <UTC>`, and
  `touch ~/runs/radical-scoping/executor.done`.

Do not contact anyone. Do not open issues or PRs. Never run `test_run_once.sh`. At most 4 concurrent processes on the VPS.
