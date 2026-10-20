# How much is there to learn in a proof-writing algorithm? (run kalmar-complexity, plain summary; UNREVIEWED)

**The setting.** In propositional logic, a 1930s proof by László Kalmár gives a mechanical recipe that proves every
valid statement:
- split into cases on each variable being true or false;
- in each case, prove every part of the statement, or its negation, from the inside out.

The previous run trained small language models (9.6 million parameters, from scratch) to write proofs this way, step
by step. They agreed with the recipe on 95 % of steps but followed it through a whole proof only 14 % of the time. Dan
asked how many bits a model must absorb to learn the recipe, and whether the shortest program's length measures it.

**What we measured.**
1. **The shortest programs we could write for the recipe**, each exact on 1,000 test statements and fresh ones: in
   compressed Python; in binary lambda calculus (a minimal language with exact bit lengths); in RASP-L (a language
   for what a transformer computes in one pass); and as a decision tree over proof-state features.
2. **The same for simpler algorithms** of known difficulty: copying, reversing, sorting, parity, addition,
   evaluating a formula, checking a truth table.
3. **The bits the networks actually absorbed**, by "prequential" accounting: each training example first tests the
   model, then trains it. The cost of those first guesses, minus what the finished model still pays, is the
   information taken in. Sources: the earlier run's logs, and new small training runs.
4. **Which measure predicts the kinds of step the models get wrong.**

**What we found.**
- **The recipe is short:** about 5,400 bits (680 bytes) in lambda calculus, 860 bytes of compressed Python;
  7,000–12,000 bits with parsing and the exact output format.
- **It is the largest algorithm we compared, in every language.** The languages agree on the ordering;
  truth-table checking comes next, 13 times smaller.
- **Length does not order learnability.**
  - The style of the earlier run's comparison data is learned easily, yet its generator is about twice as long.
  - Evaluating a formula takes under 300 bits, yet none of our small models got it right after 130,000 examples
    (one did after 260,000).
- **The networks absorbed about a thousand times the program's length.** In one pass over 155,000 training proofs, a
  model took in about 6.6 million bits, about 0.7 bits per parameter. Smaller models trained on a tenth of the data
  absorbed 1.1–1.5 million bits and ended just as good.
- **What predicts failure is the work a step requires, not the length of its instructions.**
  - Some steps need almost nothing (open a case split, close a sub-proof). Others require searching the formula for
    the next part not yet proved, checking which pieces are available, and knowing which parts are true here.
  - The models fail on exactly those steps: the RASP-L count of sequential lookups a step needs correlates with
    the failure rate at about 0.85; the code length per kind of step does not, once frequency is allowed for.
  - The commonest mistake, working on the wrong part of the formula, tracks search demand most closely.
  - The RASP-L program needs about 40 sequential attention layers in the worst case; the models have 6.
- **We then tested this directly,** training small models on two versions of the same data. In one, each proof
  state also displays the part of the formula the recipe works on next — the result of the search. Nothing else
  changed: same recipe, same steps, same proofs. On the steps that need the search, errors fell from 32 % to 2 % in
  all three paired runs, and the link between search demand and failure vanished.

**What this means.** Counting bits is the right instinct, but the bits to count are not the program's.
- A program length of a few thousand bits is a floor no realistic learner comes near.
- How far above it a learner lands depends on how well its per-step computation matches what each step demands.
- A useful statement of "how much there is to learn" has two parts:
  - the bits a specific learner absorbs under a stated protocol (here about 6.6 million);
  - a profile of what each step demands, compared with the model's depth.
- The follow-up question (how many bits pretraining and reinforcement learning each supply) can use the same
  accounting, split by kind of step and demand.

**How strong is the evidence?** Strong for this kind of model: three trained models, fifteen kinds of step, the same
pattern within each kind, and an intervention that worked as predicted. But it is one architecture and one program per
language, whose constants are as large as the programs. Whether more depth helps is untested (`next_steps.md`).
