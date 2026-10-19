# Run brief: lit-review-2 — create vs elicit since 2025, exploration that reaches outside support, and model-organism domains

Run id `lit-review-2`. Repository `~/work/lit-review-2` (branch `dan_lit-review-2` from the fork's `origin/dan`, made by
`newrun`). Role: executor. `AGENT_POLICY.md` governs. **No pods.** A $0.50 / 1 h pod budget is registered as a guard
only. Claude time is expected to be heavy. Authorised by Dan 2026-10-02 (proposal 21).

Read first:
- nd-rl `docs/literature/2026-09-29-lit-review/REVIEW.md` and its notes: 58 papers already screened. **Extend this
  review; don't redo it.**
- `docs/literature/README.md` (conventions; never edit `references.bib`).
- Proposal 20 (exploration), which already records AlphaProof's exploration from its Methods.
- The `trajectory`, `trajectory-cap6` and `rl-from-ckpt` summaries.

## Questions

1. **Create vs elicit, 2025–2026.** What is the current evidence and the measurement standard? Follow the threads
   from Yue et al. (support and pass@k) through:
   - ProRL and its successors;
   - "RL squeezes, SFT expands"-type results;
   - Invisible Leash;
   - negative-sample reinforcement;
   - RL composing skills;
   - the pretraining / mid-training / RL interplay studies;
   - power sampling;
   - excess description length.

   Which designs separate creation from elicitation convincingly, and where do our `trajectory` and `rl-from-ckpt`
   designs fall short of them?
2. **Exploration.** Which methods demonstrably produce behaviour **outside the base model's support**, not just
   higher pass@1? Cover AlphaZero / MuZero noise, AlphaProof's search and variants, RMaxTS, outcome-based and
   count-based exploration, entropy methods, curiosity and novelty, and Go-Explore-style approaches. For each:
   - What is the evidence that it reaches out of support?
   - Does it need a pretrained prior?
3. **Model-organism domains for capability emergence.** Look at Physics of Language Models, Othello-GPT,
   grokking, synthetic CFG and graph tasks, program synthesis, FOL and arithmetic provers, and games. Which ones
   allow exact control of pretraining coverage *and* an RL phase, and what could each show?

## Rules

- **Verify every cited claim against the source text.** Fetch it into files and grep it, with the location,
  as in `lit-review`. Mark anything unverified.
- **Prefer 2025–2026 papers.**
- **Screen 25–40 new papers and read 8–12 in depth.**

## Deliverables

- `lit_review_2/REVIEW.md` (≤ 2,500 words): answers to Q1–Q3, an "implications for our project" section ranked by
  value, and the screened-paper table.
- `lit_review_2/notes/*.md` (TEMPLATE format).
- `lit_review_2/to_add_to_zotero.md`.
- The librarian copies them to nd-rl `docs/literature/2026-10-02-lit-review-2/` and links them from
  `docs/literature/README.md`.
- `run_lit_review_2.md` (≤ 200 words), `log.md`, `STATUS.md` line `LIT-REVIEW-2 DONE <UTC>`, and
  `touch ~/runs/lit-review-2/executor.done`.

Do not contact anyone. Do not edit `references.bib` or run `code/tools/literature.py`. At most 4 concurrent processes on
the VPS.
