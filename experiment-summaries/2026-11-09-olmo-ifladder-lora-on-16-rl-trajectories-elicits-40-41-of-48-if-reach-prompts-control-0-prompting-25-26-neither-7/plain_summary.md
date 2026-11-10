# olmo-ifladder in plain language

**The question.** Olmo 3.1 Think 32B is a model trained on examples and preferences (`dpo`), then further with
reinforcement learning (RL). The RL model (`s2300`) follows some tricky writing rules that `dpo` did not manage in 256
ordinary tries: 48 prompts such as "every word must contain a consonant cluster". Did RL teach a new skill, or unlock
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
  steps.
- **Sixteen examples of the RL model's work reached 40 of the 48 prompts with no special instructions**, including 16
  that no instruction reached and rule types the fine-tune never saw. The control, trained on `dpo`'s own answers,
  reached none.
- The fine-tuned model also took on the RL model's habits: on a typical prompt it thinks about three times as long as
  `dpo` and answers briefly (70 words instead of 231; the RL model: 32). About one pass in five still gamed the
  checker; its automatic pass rate on ordinary prompts was as high as `dpo`'s on average.
- Nothing we tried reached seven prompts: limits on a letter or on vowels, an emoji after every sentence, equal-length
  sentences, sentences growing by three words, copying an exact span, exact keyword counts. On two of them (equal and
  growing sentences) every added instruction also trips up the RL model, so only the fine-tune test informs.

**What it means.** On these prompts, most of what RL reached can also be reached by fine-tuning on a handful of RL
examples, though less reliably (34 % of tries pass, against the RL model's 66 %). The copied habits suggest a
teachable way of working; whether knowledge transfers too was not tested. The seven holdouts are the best candidates
for a skill RL itself created.

**Limits.** With 64 tries per prompt, a prompt whose real success rate is 1–5 % can land on either side. The control
changed `dpo` much less than the main fine-tune did (its held-out likelihoods barely moved), so a stronger control
might reach some prompts. Readers disagree on borderline answers. Only one model family was tested.

Cost: $61.85 of $80.
