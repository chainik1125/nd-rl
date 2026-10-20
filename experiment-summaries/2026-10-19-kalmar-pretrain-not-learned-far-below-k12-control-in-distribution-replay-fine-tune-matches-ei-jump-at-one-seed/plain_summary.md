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
- Its mistakes were mostly bookkeeping: it lost its place in the long recursion, or miscopied a long formula. About
  one wrong step in six asserted something false.

**A surprise: a short extra phase of training fixed much of the failure.**
- We trained each model a little further on recipe proofs it had already seen, differently: 600 large steps, each
  averaged over 128 whole proofs, with the learning rate falling ten-fold (13.5 % more text).
- After this "annealing" the models chose the recipe's step 98.7 % of the time. They wrote correct proofs for 71–79 %
  of unseen theorems, and followed the recipe exactly in 54–60 %.
- The annealed models then roughly matched the standard model on ordinary held-out problems: 37 % against 38 % solved
  at the first try. With 32 tries the standard model stays ahead (66 % against 55 %).
- They clearly beat it on theorems that need classical reasoning the standard data almost never shows (35 % against
  7 %), and on one set of externally collected problems (38 % against 16 %).
- They still miss our advance definition of having learned the method: exact recipe-following on small problems is
  about two thirds, against 90 %.
- Which change mattered (learning rate, batch size, optimizer) is open; more training in the original style helps far
  less (its last quarter added 10 points of valid proofs, against 42 here).
- This part was not planned in advance: a strong lead, not a settled result.

**Reinforcement learning, mostly explained.** In a few rounds of "expert iteration" the model is trained on its own
Lean-accepted proofs.
- After one round, the number of target problems solved jumped from 135 to 572 of 1,000.
- But each round's training step replays thousands of recipe proofs in that same style.
- Annealing alone, without the model's own proofs, solves 578: in this one round, reinforcement learning added
  nothing measurable on top.
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
