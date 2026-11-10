---
written_on: 2026-11-10
written_by: agent:claude (executor, run olmo-ifladder; reviewed 2026-11-09/10, corrections marked by the librarian)
---

> **Reading copy (librarian, 2026-11-10).** This is the plain-language summary of run `olmo-ifladder`, written by
> the executor. The source is fork `dan-pandori/nd-takehome`, branch `dan_olmo-ifladder`, file
> `olmo_ifladder/plain_summary.md` at
> [`fdb4c02b`](https://github.com/dan-pandori/nd-takehome/blob/fdb4c02b/olmo_ifladder/plain_summary.md). The
> corrections from its [review](https://github.com/dan-pandori/nd-takehome/blob/6ff22000/review_olmo-ifladder.md)
> are applied, and each change is marked *[review]* (the numbers are the review's §Verdict "must be reworded" items):
> - the step counts need the detection limit fixed in advance: differences of up to 4 prompts are not findings, so
>   the 1 and the 3 are within it and the 5 just outside (item 3);
> - one borderline answer at step 4 decides between 25 and 26 prompts reached by instructions, and so between 16 and
>   15 reached only by the fine-tune (item 6);
> - "40 of the 48" is 40–41: the run's sentence-splitting screen hid one real answer (item 4);
> - "about one pass in five still gamed the checker" is the figure of the procedure fixed in advance; readers told
>   that thin answers which serve the request count as real find 7 % (item 1);
> - "on two of them … every added instruction also trips up the RL model, so only the fine-tune test informs" is not
>   supported (item 2);
> - each fine-tune was trained once (item 5);
> - "34 % of tries pass" counts automatic passes, not real answers (§Compare).
>
> Two notes are the librarian's, marked *[librarian]*: the 256 tries are an earlier run's (olmo-ifomega), and RunPod's
> billed total ($62.77, with disk). Each mark follows the passage it covers. With the marks the text grows from 500 to 715 words (`wc -w` from the title down,
> marks included), against the brief's 500.

# olmo-ifladder in plain language

**The question.** Olmo 3.1 Think 32B is a model trained on examples and preferences (`dpo`), then further with
reinforcement learning (RL). The RL model (`s2300`) follows some tricky writing rules that `dpo` did not manage in 256
ordinary tries in an earlier run *[librarian]*: 48 prompts such as "every word must contain a consonant cluster". Did RL teach a new skill, or unlock
one `dpo` already had?

**What we tried.** First, better instructions, in four steps: (1) "Answer in at most 60 words"; (2) add a reminder to
check each rule against the draft; (3) add a short method for each rule; (4) add two example answers by the RL model
to similar prompts. Second, light extra training: we fine-tuned `dpo` on 16 good answers by the RL model (reasoning
included) to other prompts of the same kind. As a control, we fine-tuned `dpo` the same way on 16 good answers of its
own. Each version tried each prompt 64 times (steps 3–4 only where earlier steps failed); 2 real answers counted as
reaching it. Blinded readers checked passing answers until two real ones were found, since the automatic checker can
be gamed.

**What we found.**
- Better instructions reached 25 of the 48 prompts: 16 with the 60-word line, then 1, 5 and 3 more with the later
  steps. By the rule fixed in advance, differences of up to 4 prompts are within chance at 64 tries, so the 1 and the
  3 are not findings and the 5 is just outside; step 2, which keeps the 60-word line, re-reached only 11–12 of step
  1's 16, a sign of how noisy the two-in-64 line is *[review]*. One borderline answer at step 4, whose sentence types balance only by the checker's own count,
  decides a 26th prompt *[review]*.
- **Sixteen examples of the RL model's work reached 40 of the 48 prompts with no special instructions** (41 once a
  blind spot in the run's sentence-splitting screen is fixed *[review]*), including 16 (15 if that borderline answer
  counts *[review]*) that no instruction reached and rule types the fine-tune never saw. The control, trained on `dpo`'s own answers,
  reached none.
- The fine-tuned model also took on the RL model's habits: on a typical prompt it thinks about three times as long as
  `dpo` and answers briefly (70 words instead of 231; the RL model: 32). About one pass in five still gamed the
  checker by the procedure fixed in advance; readers told that thin answers which serve the request count as real
  find 7 % *[review]*; its automatic pass rate on ordinary prompts was as high as `dpo`'s on average.
- Nothing we tried reached seven prompts: limits on a letter or on vowels, an emoji after every sentence, equal-length
  sentences, sentences growing by three words, copying an exact span, exact keyword counts. The run found that on two of them (equal
  and growing sentences) every added instruction also trips up the RL model, so that only the fine-tune test informs;
  the review does not support this: with the screen fixed, the RL model succeeds at step 3 on the growing-sentence
  prompt, and on the equal-sentence prompt the review's readers find 2 real answers in 16 at steps 2 and 4
  *[review]*.

**What it means.** On these prompts, most of what RL reached can also be reached by fine-tuning on a handful of RL
examples, though less reliably (34 % of tries pass the automatic checker, against the RL model's 66 %; these are not
counts of real answers *[review]*). The copied habits suggest a
teachable way of working; whether knowledge transfers too was not tested. The seven holdouts are the best candidates
for a skill RL itself created.

**Limits.** With 64 tries per prompt, a prompt whose real success rate is 1–5 % can land on either side. The control
changed `dpo` much less than the main fine-tune did (its held-out likelihoods barely moved), so a stronger control
might reach some prompts. Each fine-tune was trained only once, on one set of 16 examples with one seed, so how much
a second training run would differ is unknown *[review]*. Readers disagree on borderline answers. Only one model family was tested.

Cost: $61.85 of $80 ($62.77 as billed, with disk *[librarian]*).
