# Plain summary — olmo-think32b

**The question.** When a language model is trained with reinforcement learning (RL) on maths problems whose answers a
program can check, does it learn to solve problems it could not solve before, or does it mostly get more reliable at
problems it could already solve now and then? We asked this of AI2's Olmo 3.1 Think 32B. It came out of one long RL run
(2,300 steps, about 26 days on 224 GPUs) that started from a checkpoint called Olmo-3-32B-Think-DPO. We compared that
starting checkpoint with three points along the run (steps 100, 750 and 2,300) on the 59 AIME 2025 and 2026
competition problems.

**How.** Every model ran exactly as in RL training (same chat format, randomness and 32,768-token limit). Answers were checked three ways:
- by the grading program the RL training used;
- by the answer the model states at the end;
- by hand, for every solution that decided an outcome. Blinded readers, not told which model wrote a solution,
  labelled each one sound, right through a flawed step, or a guess, and I confirmed each label.

The starting model got 16 tries per problem and each RL checkpoint 8. A problem RL solved and the start did not was
given to the start again, up to 256 tries, to see whether it could solve it after all. The same was done in reverse.
All of this was written down in advance.

**What we found.**
- RL moves the averages only a little here. On a single try, the final model solves 75 % of the problems and the start
  72 %. With 8 tries, both solve about 86 %.
- Seven problems defeated every model. Only one problem, AIME 2025 II problem 13, was solved by the RL checkpoints and
  not by the start in its 16 tries.
- **On that problem the picture is mixed.** In 256 tries the starting model reached the right answer (248) seven times
  by the RL grader, but stated it at the end only once. It more often settled on 208 or ran out of room. The final
  model states 248 in about a quarter of its tries.
- By hand, the starting model's solutions mostly skip the crucial step, showing that a certain number stays divisible
  by 3 at every stage. Its one stated solution argues that step informally, which my pre-written rule accepts and a
  strict reading does not. The RL models produced fully rigorous solutions.
- Nothing was lost: the one problem the final model first missed, it solved within 56 tries.
- The starting model finds the RL model's correct solutions about as likely as its own good ones, even slightly more.

**The verdict.** By the rule fixed in advance, the result is **inconclusive**:
- "RL reaches new problems" needed at least three such problems; there is one.
- "No reach" needed the starting model to state the answer, soundly, at least twice in 256 tries; it did once.
- By the RL grader alone, the answer is "no reach at 256".

**What it means.** RL here used roughly 1 % of pre-training's compute by one estimate, and 14 % of its GPU time. At
that scale it mostly made the model more reliable at what it could already do. The one open case is a problem where it
turned a rare, shaky success into a routine one. A follow-up of about $27–40 would settle that case: 1,024 tries of the
starting model on that one problem.

**Cost.** $116 of the $150 budget, on rented H200 GPUs.
