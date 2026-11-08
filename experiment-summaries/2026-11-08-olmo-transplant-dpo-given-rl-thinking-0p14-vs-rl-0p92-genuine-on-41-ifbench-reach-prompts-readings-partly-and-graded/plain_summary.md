---
written_on: 2026-11-08
written_by: agent:claude (executor, run olmo-transplant; reviewed 2026-11-08, corrections marked by the librarian)
---

> **Reading copy (librarian, 2026-11-08).** This is the plain-language summary of run `olmo-transplant`, written by
> the executor. The source is fork `dan-pandori/nd-takehome`, branch `dan_olmo-transplant`, file
> `olmo_transplant/plain_summary.md` at
> [`1b7fcebe`](https://github.com/dan-pandori/nd-takehome/blob/1b7fcebe/olmo_transplant/plain_summary.md). The
> corrections from its [review](https://github.com/dan-pandori/nd-takehome/blob/aeb4655d/review_olmo-transplant.md)
> are applied, and each change is marked *[review]*:
> - "never succeeded genuinely in 256 tries" is the earlier run's number (olmo-ifomega), unattributed in the original
>   (§Compare #22; §Verdict reword item 7);
> - "often copies it" fits answers that *contain* RL's drafted answer (28 %), not exact copies (4.7 %) (#22);
> - the cut-notes rates count notes shorter than the cut at their full value; on the notes actually cut, success is
>   19–43 %, so most of the rise at 4,000–8,000 tokens comes from notes that were not cut (§Verdict reword item 4);
>   and a forced end of thinking does not stop the RL model thinking (reword item 7);
> - fresh thinking under a cap "looks similar" as a share of the RL model's normal success rate, not in absolute rates
>   (#22);
> - "the thinking is necessary" becomes "carries most of the gain, not all of it": with no notes the RL model still
>   succeeds 8 % of the time, the old model 0 % (#13; reword item 3). The original's "(more of it, more success)" is
>   dropped with it.
>
> Each mark follows the passage it covers.
>
> One note is the librarian's, marked *[librarian]*: the blinded readers read a sample of the successes, not all of
> them. With the marks the text grows from 493 to 707 words (`wc -w` from the title down, marks included), against
> the brief's 500.

# olmo-transplant in plain language

**The question.** Reinforcement learning (RL) made Olmo 3.1 Think 32B much better at tricky formatting instructions
("each sentence one word longer than the last"). On 41 such prompts the pre-RL ("DPO") model never succeeded genuinely
in 256 tries in an earlier run *[review]*; the RL model often does, after thinking for thousands of words. Is the new skill in that thinking — a
procedure the old model could follow if handed the notes?

**What we did.** We pasted the RL model's own successful scratch work, stored from an earlier run, token for token into
the old model's thinking slot, and let it write only the final answer. We also tried the reverse (the old model's failed
notes given to the RL model), no notes at all for either model, and the RL model's notes cut short at several lengths.
17,472 answers were checked by the official checker and by blinded AI readers that reject technical passes which
dodge the request. The readers read 425 of the successes; the others took the label of an identical answer already
read, or the share of genuine answers among those labelled (read, or identical to a read answer) for the same prompt and
condition (the calibration prompts'
cut and pre-draft cells are graded by the checker only) *[librarian]*.

**What we found.**
- **The notes alone are not enough.** With the RL model's complete notes, the old model succeeds 14 % of the time; the
  RL model succeeds 92 % on the same notes. The notes usually contain the finished answer word for word, and 28 % of the
  old model's answers include it, though only 4.7 % copy it exactly *[review]*. But it adds an introduction or an explanation (its answers are about five times longer), and
  the extra text breaks the rule.
- **RL's answer-writing alone is not enough either.** With no notes, the RL model succeeds 8 % of the time; with the old
  model's failed notes, 6 %. The skill is the combination: long, careful thinking, plus writing exactly the drafted
  answer and nothing else.
- **More thinking helps steadily.** Cutting the RL model's notes short lowers success gradually, not at a cliff: 21 %
  with the first ~1,000 tokens, 36 % at 2,000, 53 % at 4,000, 78 % at 8,000. But these rates count notes shorter than the cut at their full value. On
  the notes actually cut, the review's readers find success of 19 %, 32 %, 43 % and 35 % at the four lengths, against about 90 % for the same
  notes in full, so most of the rise at 4,000–8,000 comes from notes that were not cut *[review]*. Fresh thinking
  under a hard cap looks similar as a share of the RL model's normal success rate on the 6 prompts it was run on, not in
  absolute rates *[review]*.
- **The old model can read the notes:** on easy prompts, transplanted notes work for it (99 %).

**What it means.** The hypothesis was that RL taught a search procedure that lives in the chain of thought. The thinking
carries most of the gain, not all of it: with no notes at all the RL model still succeeds 8 % of the time, the old
model 0 %. And it is not sufficient *[review]*: the pre-RL model cannot turn RL's thinking into a compliant
answer. An earlier run found that telling the old model "answer in at most 60 words" unlocks some of these prompts.
Together, RL's gain looks like part answer-writing discipline and part longer, more careful thinking.

**Caveats.**
- "Genuine" is judged by AI readers following written rules, not by people.
- The notes were picked from the RL model's successes, so 92 % overstates its everyday rate on these prompts (≈ 44 %,
  estimated from the earlier run).
- After a cut, the RL model sometimes reopens its thinking before answering: in 28 % of its answers after the cut
  just before its drafted answer, and 8–9 % after the 1,000- and 2,000-token cuts, so "thinking cut at N" is not a
  strict cap *[review]*. Counting that as a failure lowers some numbers; both versions are reported.
- 41 prompts, one benchmark, one RL run.
