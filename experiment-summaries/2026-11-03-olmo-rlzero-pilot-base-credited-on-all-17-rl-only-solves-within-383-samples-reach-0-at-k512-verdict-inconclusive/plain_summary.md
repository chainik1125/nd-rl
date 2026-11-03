---
written_on: 2026-11-03
written_by: agent:claude (executor, run olmo-rlzero-pilot; reviewed 2026-11-03, corrections marked by the librarian)
---

> **Reading copy (librarian, 2026-11-03).** This is the plain-language summary of run `olmo-rlzero-pilot`, written by
> the executor. The source is fork `dan-pandori/nd-takehome`, branch `dan_olmo-rlzero-pilot`, file
> `olmo_pilot/plain_summary.md` at [`046d217a`](https://github.com/dan-pandori/nd-takehome/blob/046d217a/olmo_pilot/plain_summary.md).
> The corrections from its [review](https://github.com/dan-pandori/nd-takehome/blob/525e2e0e/review_olmo-rlzero-pilot.md)
> are applied, and each change is marked *[review]*:
> - "with complete worked solutions, not lucky guesses" is replaced by the hand-check (review, Verdict item 1);
> - the reverse problems get the aime2025_14 caveat (item 2);
> - "every correct RL solution falls inside the range" becomes every *scored* solution, on the pre-registered
>   per-token measure, with the whole-solution gap (item 3);
> - "never" becomes "generally does not" (item 6), and "genuinely new problems never appeared" gets its budget and its
>   checker (the review's note on strong words);
> - "solved" is said to mean "credited by RL's checker" (the review's "Not supported" note);
> - the costs are called projections, and get the review's additions to the 32B design (next measurement 2).
>
> The corrections take it from about 600 to about 880 words (`wc -w`, marks included), against the brief's 600. See
> the [summary](README.md) for the reviewed numbers.

# Does reinforcement learning teach Olmo new maths, or make it use what it already knew? — a plain summary

**The question.** AI2 trained a 7-billion-parameter language model, Olmo 3, on text, then ran reinforcement learning
(RL) on maths problems: the model writes a solution, a checker says whether the final answer is right, and the model
is nudged towards answers that were right. AI2 published snapshots along the way. We asked whether RL lets the model
solve problems it *could not* solve before, or mostly makes it solve, more often, problems it could already solve now
and then.

**What we did.** We took 83 problems from the 2024–2026 American Invitational Mathematics Examination (AIME), all newer
than the model's training text, after removing 7 that (or close variants of which) appear in its RL or training data.
We sampled solutions from the model before RL (the "base") and after 500 and 2,800 RL steps, with exactly RL's prompt,
randomness and length limit, and graded them with RL's own checker. *[review]* That checker reads a boxed answer, a
"final answer is" phrase or the response's last formula (failing those, the whole response), never the "Answer:" line
the prompt asks for, so "solved" below means "credited by RL's checker". Every problem an RL snapshot solved and the
base had not went back to the base, attempt after attempt (up to 512), until it solved it — and the same the other way.

**What we found.**

1. *RL makes the model much better at first tries.* On one attempt the base solves a problem 9 % of the time; after
   500 RL steps 30 %, after 2,800 steps 42 %. With 8 attempts: 36 % against 63 %.

2. *But everything RL "newly" solved, the base could solve too, with more attempts.* Seventeen problems were solved
   by an RL snapshot and by the base in none of its first 16 attempts. Re-sampled, the base solved all 17, after 18 to
   383 attempts (typically about 70). *[review]* Read by hand, 10 of the base's 17 solutions are sound, 6 reach the
   right answer through a flawed step, and 1 is a guess, as are all 5 of RL's correct solutions on that problem. Two
   of the base's 17 were cut off at the length limit after giving their answer, and on one more problem all 5 of RL's
   correct solutions share the base's flaw. The 3 problems the base solved and the final RL model had missed, the RL
   model solved with more attempts. *[review]* The final model's solution on one of them is a guess (it computes 34,
   then writes 60), and the base's only credit there is probably the checker accepting a 60° angle restated from the
   problem. On the evidence neither model solves it: the tally stays the same, but not because the final model can
   solve it. At these budgets,
   *[review]* by RL's checker, RL added no problem the base could not do; it made rare correct solutions common.

3. *[review]* *Word by word, RL's correct solutions look like things the base would write.* Scored word by word by
   the base, *[review]* every scored correct RL solution (213 of 718) falls, per word and by the test we fixed in
   advance, inside the range of the base's own correct solutions, including those on the 2023 trial problems. The
   final model's are slightly less typical for the base (about 5 % less probable per token) than for itself, with no
   part unusually surprising to it. *[review]* Over a whole solution of about 12,000 tokens, though, that adds up: the
   base is a median e^731 times less likely than the final model to write its solutions on the 17 problems, so "inside
   the range" is weak evidence.

4. *How much RL is that?* About a thousandth of the compute used to pretrain the model (1.5–3.2 × 10⁻³ at the last
   snapshot). Over that range the gains grew; genuinely new problems *[review]* did not appear, at up to about 500
   attempts on these 83 problems and by RL's checker.

**What this does and does not show.** One model, one RL run, 83 problems, at most about 500 attempts per problem: a
problem the base fails 500 times might still fall at 5,000. Our decision rule, written in advance, came out
"inconclusive" only because one clause asked the base to match RL at 8 attempts, which sharpening alone *[review]*
generally does not. RL this small may be exactly the regime where it only sharpens; Olmo 3.1 Think 32B spent far more
of its compute on RL (about 14 % of pretraining in GPU-hours), so the next test is planned there.

**What it cost.** About $39 of rented GPUs. Sampling is expensive — about half a dollar per million tokens —
because each solution is long (6,000 tokens for the base, 12,000 after RL) and the model's attention stores a lot per
token. The decision memo (`olmo_pilot/decision_32b.md`) prices the 32B test at about $110 for this design and $205–320
for one that pushes the starting model to 512–2,048 attempts. *[review]* These are projections. The review also asks
the 32B test for a grade that reads only the final stated answer, a hand-check of every deciding solution, and the
same confirmation rule for the base as for RL.
