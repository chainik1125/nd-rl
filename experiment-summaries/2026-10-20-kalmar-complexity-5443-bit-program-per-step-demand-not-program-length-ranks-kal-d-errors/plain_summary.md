---
written_on: 2026-10-20
written_by: agent:claude
---

> **Reading copy (librarian, 2026-10-20).** This is the plain-language summary of run `kalmar-complexity` (fork
> `dan-pandori/nd-takehome`, branch `dan_kalmar-complexity`, DONE
> [`a9cc4a1b`](https://github.com/dan-pandori/nd-takehome/commit/a9cc4a1b), file `run_kalmar_complexity_plain.md`),
> written by the executor, with the corrections from its
> [review](https://github.com/dan-pandori/nd-takehome/blob/0fe457ee/review_kalmar-complexity.md) applied. Each change is
> marked *[review]*; the numbers in parentheses are the review's claim numbers:
> - "13 times smaller" is one language's ratio (binary lambda calculus), and the recipe's second version sits between
>   (20);
> - the generator comparison is between two mechanically minified upper bounds (21);
> - formula evaluation's slowness is a code length, not an accuracy, with one model per size (22);
> - absorbed bits say what they are counted against (23, and "Must be reworded"). More than half of the small models'
>   bits are the cost of the first 64 proofs, coded before any training, and only the mid-sized model ends "just as
>   good", on another set of theorems (24, "Not supported as worded");
> - code length *anti*-predicts errors among the formula-proving steps ("Must be reworded: Program length");
> - the error classes come from one of the three models (26);
> - the RASP-L depth is what one hand-written program uses, with operations beyond strict RASP-L (27);
> - the displayed part of the formula is also the formula the step must write; the other steps improved too; training
>   stopped at its step limit (28, and "Must be reworded");
> - the intervention's second prediction failed, and it cannot separate search from copying (29, and "Not supported as
>   worded").
>
> The corrections take it from 800 to about 1,050 words, against the brief's 800. The executor's text, byte for byte, is
> this file at commit `b502105`. See the [summary](README.md) for the reviewed numbers.

# How much is there to learn in a proof-writing algorithm? (run kalmar-complexity, plain summary, with the review's corrections) *[review]*

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
- **It is the largest algorithm we compared, in every language.** The languages agree on the ordering. The
  largest simpler algorithm, truth-table checking, is 13 times smaller in lambda calculus but only 4 times in
  compressed Python and 8 in RASP-L; a second version of the recipe sits between them in lambda calculus and
  Python. *[review]*
- **Length does not order learnability.**
  - The style of the earlier run's comparison data is learned easily, yet its generator is about twice as long
    (comparing two mechanically shrunk upper bounds). *[review]*
  - Evaluating a formula takes under 300 bits, yet after 130,000 examples our small models still paid about half a
    bit per answer (a code length, not an accuracy); only the mid-sized one neared certainty, after 260,000. Each
    size was trained once. *[review]*
- **The networks absorbed 50 to 1,300 times the program's length, depending on how it is counted.** *[review]*
  - In one pass over 155,000 training proofs, a model took in about 2 million bits, counted against the model it had
    become by the end of that pass. Against the fully trained model (3.7 passes), the first pass's excess is about
    6.6 million, 0.7 bits per parameter. *[review]*
  - Smaller models trained on a tenth of the data absorbed 1.1–1.5 million bits, but more than half of that is the
    cost of the first 64 proofs, coded before any training. Only the mid-sized one ended about as good as the big
    model, measured on another set of theorems; the smallest ended 3 times worse. *[review]*
- **What predicts failure is the work a step requires, not the length of its instructions.**
  - Some steps need almost nothing (open a case split, close a sub-proof). Others require searching the formula for
    the next part not yet proved, checking which pieces are available, and knowing which parts are true here.
  - The models fail on exactly those steps: the RASP-L count of sequential lookups a step needs correlates with
    the failure rate at about 0.85; the code length per kind of step does not, once frequency is allowed for. Among
    the steps that prove a part of the formula, shorter code even goes with more errors. *[review]*
  - The commonest mistake, working on the wrong part of the formula, tracks search demand most closely (in the one
    model whose mistakes were classified). *[review]*
  - Our one hand-written RASP-L program, with some operations beyond strict RASP-L, uses 41 sequential attention
    layers in the worst case (33 on the steps we sampled); the models have 6. It is an upper bound, not a proven
    need. *[review]*
- **We then tested this directly,** training small models on two versions of the same data. In one, each proof
  state also displays the part of the formula the recipe works on next — the result of the search. It is also, up to
  a leading negation, the very formula the step must write, so the display hands over text to copy as well as the
  search result. The recipe, steps and proofs are the same. *[review]* On the steps that need the search, errors fell
  from 32 % to 2 % in all three paired runs, and the link between search demand and failure vanished. Other steps'
  errors fell by about a third too; all six models were still improving when training hit its step limit.
  *[review]*

**What this means.** Counting bits is the right instinct, but the bits to count are not the program's.
- A program length of a few thousand bits is a floor no realistic learner comes near.
- How far above it a learner lands depends on how well its per-step computation matches what each step demands.
- A useful statement of "how much there is to learn" has two parts:
  - the bits a specific learner absorbs under a stated protocol, saying what they are counted against (here about
    6.6 million against the fully trained model, 2 million within the first pass); *[review]*
  - a profile of what each step demands, compared with the model's depth.
- The follow-up question (how many bits pretraining and reinforcement learning each supply) can use the same
  accounting, split by kind of step and demand.

**How strong is the evidence?** Strong for this kind of model: three trained models, fifteen kinds of step, the same
pattern within each kind, and an intervention whose main prediction held, though its second (the other steps barely
change) failed, and it cannot separate finding the next part of the formula from copying its text. *[review]* But it
is one architecture and one program per language, whose constants are as large as the programs. Whether more depth
helps is untested (`next_steps.md`).
