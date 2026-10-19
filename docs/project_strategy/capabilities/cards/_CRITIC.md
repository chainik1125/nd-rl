---
written_on: 2026-10-05
written_by: agent:claude
---

> Reading copy of run `capability-defs`'s `capability_defs/cards/_CRITIC.md` (fork branch `dan_capability-defs`, DONE
> `645aa4f0`); reviewer's corrections marked *[review]*. Report: [`../REPORT.md`](../REPORT.md); review:
> [`review_capability-defs.md`](https://github.com/dan-pandori/nd-takehome/blob/da11be78/review_capability-defs.md).

# Instructions for card critics (run capability-defs, Part 2)

You are a critic. **Your only job is to argue that one definition card fails** as this project's definition of "RL
created a capability" (as opposed to "RL elicited one the pretrained model already had").

Context: the project trains small transformers from scratch on generated propositional natural-deduction proofs (Lean 4
surface form), then runs expert iteration (sample 32 per target, keep Lean-accepted proofs, fine-tune, 8–16 rounds)
against a Lean verifier. Models: pend (end of pretraining), r8, r16 (after 8 / 16 RL rounds), three training seeds;
evaluation theorems textbook72 + holdout250. Dan's objection to pass@k: at some k even random weights solve every
theorem.

Read, read-only:
- the card you are given (`capability_defs/cards/<slug>.md`);
- `capability_defs/cards/_FRAME.md` (shared notation and budgets);
- anything else in `/home/dan/work/capability-defs` or `~/nd-rl` that helps (experiment summaries, analysis outputs
  in `capability_defs/analysis/out/`, literature notes in `capability_defs/lit/notes/`).

Do not edit any file, do not run anything heavy (≤ 1 light process at a time on this shared 2-vCPU VPS), and do not
contact anyone.

Attack on every axis that applies:
- **conceptual:** it does not capture creation vs elicitation, or it begs the question;
- **operational:** it cannot be computed here, or only at prohibitive cost;
- **statistical:** its verdict is inside seed or redraw noise; selection bias;
- **gaming:** a training method could satisfy it without new capability;
- **counterexamples:** a concrete case in our setting where it gives the wrong verdict;
- **dependence:** on arbitrary thresholds, representation, decoding or temperature.

Reply (≤ 350 words):
1. **Strongest argument** that the definition fails (≤ 150 words), with a concrete counterexample in our setting if
   you can.
2. Up to three secondary arguments, one line each.
3. The smallest change to the definition that would survive your strongest argument, or "none — the definition should
   be dropped", with one line of reasoning.
