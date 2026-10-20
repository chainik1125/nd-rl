---
written_on: 2026-10-20
written_by: agent:claude
---

> **Reading copy (librarian, 2026-10-20).** This is the plain-language summary of run `kalmar-pretrain` (fork
> `dan-pandori/nd-takehome`, branch `dan_kalmar-pretrain`, DONE
> [`9c997fe8`](https://github.com/dan-pandori/nd-takehome/commit/9c997fe8), file `run_kalmar_pretrain_plain.md`),
> written by the executor, with the corrections from its
> [review](https://github.com/dan-pandori/nd-takehome/blob/3813b53c/review_kalmar-pretrain.md) applied. Each change is
> marked *[review]*:
> - the error findings are from one of the three models (KAL-D s2). The one in six is 671 of the 4,002 wrong steps
>   that could be checked under the run's scoring rule (16.8 %). Under the reviewer's stricter rule it is 667 of 3,309
>   (20 %), and it is at least 9.7 % of all 6,903 wrong steps (review rewordings (c) and (e));
> - all four release sets are given, not one (review (g)). Greedy, three seeds' means, "annealed" model (KAL-D+A)
>   against the standard model (control): candidate_v0 37.9 vs 15.7 %, batch3 40.5 vs 14.3 % (14 problems),
>   candidate_v1 16.7 vs 13.0 %, hand_proved_14 0 vs 0;
> - the reinforcement-learning comparison rests on one seed (s0) over one round (review (d) and its claim table);
> - "annealing" stays in quotes (review, "Not supported, or not derivable"): the extra phase changed the learning rate,
>   the batch (≈ 150× larger), the optimizer and the data volume (+13.5 %) at once, and the run did not separate them.
>
> The corrections take it from 799 to about 850 words, against the brief's 800. The executor's text, byte for byte, is
> this file at commit `c78f460`. See the [summary](README.md) for the reviewed numbers.

# Can a small model learn a proof algorithm from examples? (kalmar-pretrain, plain-language summary)

## The setting

This project trains small language models (about 10 million parameters) to prove simple logic theorems built from four
letters with "and", "or", "not" and "implies". The model sees the current proof state and writes one step at a time;
Lean, a proof checker, decides whether a finished proof is correct.

This run asks whether ordinary training on examples can teach a model a complete *method*.

## The method

Every theorem in this logic can be proved by one mechanical recipe, the truth-table argument behind Kalmár's
completeness proof:
1. Split into cases for every letter, true or false: up to 16 cases with four letters.
2. In each case, prove each formula, or its negation, by a fixed recursion.
3. Close at once any case in which an assumption is false.

Each step depends only on the current state, so a model that learned the recipe could prove every theorem. The catch
is length: recipe proofs average 109 steps, against 8 for the project's usual training proofs.

## What we did

We generated recipe proofs for all 155,000 theorems of the project's standard training set (K12); Lean accepted every
one. We trained the project's best small model on them from scratch, three times with different seeds, making the same
3.7 passes over the data as the standard model, which trains on K12's own short proofs (so about 27 times more text).
Before any training, we wrote down what we expected and what would count as success.

## What we found

**The planned test said no.**
- Shown each state of the recipe's proof of an unseen theorem, the model chose the recipe's next step
  94.5 % of the time.
- Writing whole proofs alone, it proved only a third of unseen theorems correctly.
- With 32 tries it solved far fewer held-out problems than the standard model: 33 against 165 of 250.
- Its mistakes were mostly bookkeeping: it lost its place in the long recursion, or miscopied a long formula. Of the
  wrong steps that could be checked, about one in six asserted something false. Both findings come from one of the
  three trained models. *[review]*

**A surprise: a short extra phase of training fixed much of the failure.**
- We trained each model a little further on recipe proofs it had already seen, differently: 600 large steps, each
  averaged over 128 whole proofs, with the learning rate falling ten-fold (13.5 % more text).
- After this "annealing" the models chose the recipe's step 98.7 % of the time. They wrote correct proofs for 71–79 %
  of unseen theorems, and followed the recipe exactly in 54–60 %.
- The "annealed" *[review]* models then roughly matched the standard model on ordinary held-out problems: 37 % against
  38 % solved at the first try. With 32 tries the standard model stays ahead (66 % against 55 %).
- They clearly beat it on theorems that need classical reasoning the standard data almost never shows (35 % against
  7 %). On the four sets of externally collected problems, at the first try, they scored 38 % against 16 %, about
  40 % against 14 % (a set of only 14 problems), 17 % against 13 %, and 0 against 0. *[review]*
- They still miss our advance definition of having learned the method: exact recipe-following on small problems is
  about two thirds, against 90 %.
- Which change mattered (learning rate, batch size, optimizer) is open; more training in the original style helps far
  less (its last quarter added 10 points of valid proofs, against 42 here).
- This part was not planned in advance: a strong lead, not a settled result.

**Reinforcement learning, mostly explained.** In a few rounds of "expert iteration" the model is trained on its own
Lean-accepted proofs.
- After one round, the number of target problems solved jumped from 135 to 572 of 1,000.
- But each round's training step replays thousands of recipe proofs in that same style.
- "Annealing" alone, without the model's own proofs, solves 578: in this one round, tested on one of the three
  models, reinforcement learning added nothing measurable on top. *[review]*
- By our advance tests it neither kept to the method nor avoided shorter proofs, but the untrained model's samples
  score the same; its greedy proofs follow the recipe more often.

**A side finding.** Models trained on K12 were known to fail problems with four or more assumptions.
- Proofs begin by restating each assumption; K12 never has more than three, and with more the standard model
  restates them wrongly.
- When the environment restates them instead, it solves 177–191 of 200 such problems with 32 tries each.

## What it means

Imitating an algorithm's long proofs gets much further at this scale than the planned test showed, though not all the
way. The pattern-like parts are learned easily. Keeping track of a long computation improved most with a short extra
phase of training. Large apparent gains from reinforcement learning can come from such a phase.
