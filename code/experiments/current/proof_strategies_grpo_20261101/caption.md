---
written_on: 2026-11-01
written_by: agent:claude
---

# Draft caption: Figure 1, "capability emergence"

File: [`docs/papers/high-rl-regime/figures/fig1_emergence.pdf`](../../../../docs/papers/high-rl-regime/figures/fig1_emergence.pdf)
(PNG beside it). Every number below is in [`fig1_numbers.json`](fig1_numbers.json),
[`numbers.md`](numbers.md) or `high_rl_passk_20261030/seeds3.json`, and was checked against those files.

## Caption (draft)

> **Figure 1: Capability emergence under high-RL GRPO.**
> **(a)** pass@k on Charles149 (149 textbook problems) for the Stage-1 base and the RL end, as means
> over three seeds with 95 % problem × seed bootstrap bands. RL leads at every k: 70.5 % vs 32.6 % at
> pass@1, and 86.6 % vs 74.5 % at pass@1024.
> **(b)** Two Lean-accepted RL proofs, each identical to the textbook's own proof. The base
> succeeds on these problems in 0 and 4 of 3,072 attempts; RL succeeds in 34 % and 52 % of its own
> 3,072. The first is a proof by contradiction that opens from the negated goal and uses modus
> tollens. The second uses two disjunctive syllogisms. Each of these strategies occurs in at most
> 1.5 % of the Stage-1 corpus's proofs.
> **(c)** Two curves against RL / pretraining FLOPs, with the base drawn left of the axis break:
> RL's per-attempt solve rate on the 32 Charles149 problems the base never solves in 1,024
> attempts (orange), and the share of attempts that are accepted 13–20-line proofs using a working
> proof by contradiction (blue). Both stay at
> the base level through Yue et al.'s median ratio (dotted), then rise in an S-shape to 27.5 % and
> 13.1 %; the logistic midpoints sit at ratios of 0.05 and 0.08.
> **(d)** Pending.

## Notes for the integrator (not caption text)

- **Seeds.** (a) uses seeds 0–2. In (b), RL and base rates are both pooled over seeds 0–2. Per
  seed of 1,024 attempts:
  - first problem: RL 592 / 3 / 451, base 0 / 0 / 0;
  - second problem: RL 984 / 145 / 478, base 0 / 4 / 0.

  Each shown proof is the shortest one without dead lines across the three RL seeds, ties going to
  the most frequent. The first comes from RL seed 2 (41 attempts), the second from seed 1 (96).
  (c) uses the seed-0 checkpoints.
- **Problems in (b).**
  - `textbook_49b7d961bb8d378750b8` (Carnap, carnap.io/book/7, ex. 8.8) is classical-only. It is
    one of the 9 Charles149 problems that no base seed solves in 3,072 attempts. Its reductio is
    working but "scaffolded" in Dan's terms, because ¬¬Q occurs in the premises. Lines 5–8 are a
    modus tollens.
  - `textbook_5b38b6ab0a04f4c4bb28` (*forall x: Calgary*, Additional rules) is intuitionistic.
  - Neither is in a textbook72 renaming class.
  - Each displayed proof is one of the model's Lean-accepted texts; `fig1.py` re-checks three Lean
    texts per proof (6 / 6 accepted).
- **Why the second example is not base-never.** Of the 9 problems no base seed solves:
  - 7 are too long: 5fdd, 905b, 9443, 96a5, c88e, f31c, fd05.
    - Their shortest RL proofs have 15–55 pruned lines.
    - Their shortest proofs with a rare strategy have 16–55; 5fdd has none, and 96a5's 15-line
      proofs use none.
  - d337's 11-line proof uses only case analysis (6 % of the corpus, not rare); its shortest
    proof with a rare strategy has 16 lines.
  - 49b7 is used.
  - None failed the textbook72 guard.
- **The "at most 1.5 %" in (b).** In the Stage-1 corpus, a working proof by contradiction occurs
  in 1.47 % of proofs, modus tollens in 0.13 % and disjunctive syllogism in 1.19 %
  ([`numbers.md`](numbers.md) Q1).
- **The two curves in (c) come from different sets.** The pre-registered rule for choosing
  panel (c)'s strategy picks modus tollens on Charles149, whose share of proofs does not rise, and
  a working proof by contradiction on the 13–20-line set. The panel therefore shows the
  pre-registered fallback quantity on Charles149, the base-unsolved solve rate, next to the
  rule's 13–20-line pick. See the README, "Deviations".
- **What could weaken the story** (README, Q2):
  - On problems both solve, the base already uses every textbook strategy at textbook rates.
  - RL's biggest shift in strategy choice is toward template (idle) reductios.
  - The GRPO training pools are 40–45 % classical-only.
