# olmo-transplant in plain language

**The question.** Reinforcement learning (RL) made Olmo 3.1 Think 32B much better at tricky formatting instructions
("each sentence one word longer than the last"). On 41 such prompts the pre-RL ("DPO") model never succeeded genuinely
in 256 tries; the RL model often does, after thinking for thousands of words. Is the new skill in that thinking — a
procedure the old model could follow if handed the notes?

**What we did.** We pasted the RL model's own successful scratch work, stored from an earlier run, token for token into
the old model's thinking slot, and let it write only the final answer. We also tried the reverse (the old model's failed
notes given to the RL model), no notes at all for either model, and the RL model's notes cut short at several lengths.
17,472 answers were checked by the official checker and by blinded AI readers that reject technical passes which
dodge the request.

**What we found.**
- **The notes alone are not enough.** With the RL model's complete notes, the old model succeeds 14 % of the time; the
  RL model succeeds 92 % on the same notes. The notes usually contain the finished answer word for word, and the old
  model often copies it. But it adds an introduction or an explanation (its answers are about five times longer), and
  the extra text breaks the rule.
- **RL's answer-writing alone is not enough either.** With no notes, the RL model succeeds 8 % of the time; with the old
  model's failed notes, 6 %. The skill is the combination: long, careful thinking, plus writing exactly the drafted
  answer and nothing else.
- **More thinking helps steadily.** Cutting the RL model's notes short lowers success gradually, not at a cliff: 21 %
  with the first ~1,000 tokens, 36 % at 2,000, 53 % at 4,000, 78 % at 8,000. Fresh thinking under a hard cap looks
  similar.
- **The old model can read the notes:** on easy prompts, transplanted notes work for it (99 %).

**What it means.** The hypothesis was that RL taught a search procedure that lives in the chain of thought. The thinking
is necessary (more of it, more success) but not sufficient: the pre-RL model cannot turn RL's thinking into a compliant
answer. An earlier run found that telling the old model "answer in at most 60 words" unlocks some of these prompts.
Together, RL's gain looks like part answer-writing discipline and part longer, more careful thinking.

**Caveats.**
- "Genuine" is judged by AI readers following written rules, not by people.
- The notes were picked from the RL model's successes, so 92 % overstates its everyday rate on these prompts (≈ 44 %,
  estimated from the earlier run).
- After a cut, the RL model sometimes reopens its thinking before answering. Counting that as a failure lowers some
  numbers; both versions are reported.
- 41 prompts, one benchmark, one RL run.
