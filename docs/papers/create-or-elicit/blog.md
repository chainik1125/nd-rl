# Did RL create it? We deleted one reasoning move from a small prover's training data

*Draft for Dan, 2026-10-13. Not for circulation.*

*Epistemic status: one small model (9.6 million parameters), one kind of logic, one simple RL algorithm, six training
runs per main condition and three for one extra condition. The main test was pre-registered, and someone who had not
written the experiment's code recounted its numbers from the raw files. Some analyses here were added later and are
labelled so. I would not generalise this to large language models without more work.*

**The short version.** We trained a tiny theorem prover from scratch, twice: once on data that included one classical
reasoning move, and once with every example of that move deleted and the data topped back up to the same size. Then we
ran the same simple form of reinforcement learning (RL), expert iteration, on both, six independent runs ("seeds")
each. With the move in its data, five of six runs learned a new family of classical theorems during RL. ("Learning a
family" means proving at least half of its unseen instances.) Without it, no run learned any family in 16 rounds. The
best proved 3 of 106 unseen classical theorems in any test, and 5 after we let it run longer. Putting the move back
into 1% of the data restored the learning on three runs out of three, in a smaller, exploratory comparison.

The models trained without the move were not completely unable to make it. Given about a million tries, most of them
stumbled onto it a handful of times, and during RL they used it routinely on easy problems. RL rarely carried it over to
the problems that require it. In this small organism, this form of RL amplified what pretraining had exposed, and did
not, within 16 rounds, learn the classical families without it.

**Why it matters.** A model that fails a task in 1,000 tries may still succeed once in a million, so claims of absence
need far more attempts or likelihood estimates. Deleting a skill from pretraining kept this RL from learning it within 16
rounds, but putting 1% back undid the deletion. One caveat matters a lot: a closely related study found the opposite
with another RL algorithm (GRPO), so our result may be specific to ours.

## 1. The question

Reinforcement learning against a verifier is how today's reasoning models get much of their skill. A model tries
problems, an automatic checker says which attempts are right, and the model is trained to make the right attempts more
likely. A live debate asks whether this *creates* abilities the original ("base") model never had, or only *elicits*
abilities it already had, by making rare good behaviour common.

The evidence is hard to read for a reason that sounds pedantic and isn't. Suppose the base model solves a problem
once in a million attempts. Sample it 1,000 times and you will almost certainly see nothing, and conclude "RL created
this". Sample it 10 million times and you'll see about ten successes. So "the base can't" always means "the base can't
within some number of attempts", and that number should be stated.

You also can rarely retrain a frontier model without some piece of knowledge to see if RL rediscovers it. A *model
organism*, a small system we fully control, lets us.

## 2. A prover small enough to edit

Our model is a 9.6M-parameter transformer that proves theorems of propositional logic: statements built from atoms
like P and Q with "and", "or", "implies" (→) and "not" (¬). A theorem looks like "from ¬S → ¬P and P, conclude
R → S". The model works step by step: at each step it sees what it knows and what it still has to prove, and writes one
inference. A proof counts only if Lean, a widely used proof checker, accepts it.

We pretrain it from scratch on 155,000 machine-generated proofs. Then we run *expert iteration*, a simple form of RL.
Each round, the model attempts each of 4,495 training theorems 32 times, the proofs Lean accepts are added to its
training data, and it is fine-tuned on them.

**The move we deleted.** Classical logic allows a kind of proof by contradiction that weaker ("intuitionistic") logic
does not. To prove S, assume "not S", derive a contradiction, conclude "not not S", then drop the double negation. We
call that combination σ (sigma). Here is one in Lean, written by one of our models pretrained without σ:

```lean
theorem t (P R S : Prop) (h1 : ¬S → ¬P) (h2 : P) : R → S := by
  have n9 : R → S := (fun (n3 : R) => by
    have n7 : ¬¬S := (fun (n4 : ¬S) => by      -- assume ¬S ...
      have n5 : ¬P := h1 n4
      exact n5 h2)                              -- ... contradiction, so ¬¬S
    exact Classical.byContradiction (fun hh => n7 hh))   -- drop the double negation: σ
  exact n9
```

*(Simplified by hand from the model's output, which also restates the premises; Lean accepts both versions and rejects
the same proof of the opposite claim.)*

σ shows up in two situations. Sometimes the problem already hands you a double negation, or asks you to prove a "not"
statement; that use is easy. The genuinely classical use proves a positive statement with no double negation supplied.
That step, or a roundabout variant of it, is the only way to prove the theorems we test.

About 11% of the original training proofs use σ or a close variant. We made three versions of the training data:
- **Full:** the original 155,000 proofs.
- **Knockout:** every proof using σ or a variant removed (16,961) and replaced by fresh proofs from the same generator.
  "Assume and refute" and "drop a double negation" both still appear separately, though each became three to four times
  rarer.
- **1% dose:** the knockout data with 1,550 σ proofs put back.

To test learning we use six families of classical theorems, such as excluded middle ("P or not P"), converse
contraposition and Peirce's law. We hold out 105 of their instances, plus one similar theorem, that can only be proved
with a genuinely classical step, and never train on them.

## 3. What happened

![Held-out classical theorems solved, per seed](figures/blog_fig1_endpoint.png)

*Figure 1. Each line is one training run. The y-axis counts the 106 held-out classical theorems proved, given 256
attempts each, before RL and after 8, 16 and (one knockout run) 24 rounds. Blue: full data. Green: 1% dose. Orange:
knockout.*

- **Full data:** during RL the six runs gained 8 to 69 of the 106, and five of six learned a new family. One family
  (converse contraposition) was already learned *before* RL on five of the six, so the cleanest comparison is the
  during-RL one.
- **Knockout:** the six runs gained 0 to 2, and none learned any family.
- **Statistics:** the pre-registered test (at least one family by round 16: six of six against zero of six) gives
  p = 0.00024, but it is mostly decided before RL. The during-RL comparisons were added afterwards. The gains do not
  overlap (p = 0.0011), new families give five of six against zero of six (p = 0.0032), and on the three full runs with the
  knockout runs' hardware the gains still separate (p = 0.012).
- **1% dose:** all three runs learned families during RL and proved 30 to 58 of the 106, inside the full runs' range.

One complication: RL also replays old pretraining data, so the full runs kept seeing σ during RL and the knockout runs
did not. We can't tell whether exposure mattered in pretraining or in that replay.

## 4. "Absent" is not quite right

**They could make the move, rarely.** We sampled each knockout model 1.1 to 1.5 million times across the held-out
theorems. Five of six proved some of them with a genuinely classical σ, one to seven times, almost all in one family
(converse contraposition). On the other five families, the six models together managed one in 6.7 million tries. With
guided redraws (section 6), four of the six proved one within 256 attempts. Only one
model had zero successes in 1.5 million tries, enough to certify statistically that its average rate is below what RL's
compute could find, but only for plain sampling and one way of counting the budget.

**They practised it on easy problems.** From the first round, every knockout run used σ or a variant where the problem
handed it the double negation: thousands of accepted proofs over 16 rounds. They almost never used the genuinely classical form. That
fits a back-of-envelope calculation: RL spends about 3,900 attempts per round on the 123 training
theorems that require such a step, and at a few successes per million or less, that yields at most about 0.03 per
round. In a check added afterwards, three of the six runs did better than this arithmetic predicts, so RL probably nudged the rate up.
None got as far as learning a family.

**Likelihoods.** Because the model is small, we can compute how likely it is to write any given proof (using one of 33
equivalent spellings; the true value is at most 3.5 units lower, and may be higher). For the best known σ proof of each
held-out theorem, the full models before RL give a median log-probability of −15 to −21, the knockout models −26 to −36. That is roughly a hundred
to a billion times less likely. During RL, one full run and one dose run moved strongly toward these proofs; the rest, from all three
kinds of run, ended between 8 units lower and 4 units higher.

![How likely each model is to write the classical proofs, along RL](figures/blog_fig2_logp.png)

*Figure 2. Median log-probability (higher = more likely; natural-log units) that each model writes the best known σ
proof of a held-out classical theorem, from the end of pretraining through 16 rounds of RL.*

**The one knockout run that moved.** One knockout run kept writing more genuinely classical proofs, a few per round from
round 9 and 123 per round by round 24 (we let it run past the planned 16 rounds). It never learned a family. At rounds 20
and 24 it proved the same five held-out theorems, just more reliably. A first, early-stopped sample had found four of
those five before RL. A fresh, larger sample (18,432 tries each) found only two, and none of the three that improved
most. We had committed in advance to dropping our "it mostly amplified theorems its pre-RL model could already
prove" reading if that happened, so we drop it. Another knockout run whose model had proved three such theorems before RL learned none of them. Whether this
counts as eliciting a rare behaviour or creating a new one is, honestly, unclear.

**Seeding didn't help, at least briefly.** One extra RL round seeded with one of these rare proofs changed nothing
measurable; in an earlier, smaller model, one injected proof had got a similar pattern going within a round.

## 5. Why "created" depends on how you count

The usual test gives the base model and the RL model the same number of attempts, say 256, and calls anything the RL
model solves and the base doesn't "created". On three earlier full-data runs, that labels 51 to 60 of 322 evaluation
theorems as created by RL.

But RL itself spent a lot of compute. Give the base model that compute as attempts, thousands per theorem, and the list
shrinks. How much depends on how you convert compute into attempts. Across five defensible conversions the count falls
to between 14 and 34 per run, and only 14 to 21 count as created under all five. On nine hand-picked theorem–run pairs
sampled over a million times, four survive every control at one of those budgets.

![The "created" count depends on the base model's budget](figures/blog_fig3_budget.png)

*Figure 3. Theorems the RL model solves that its base can't reach within a budget of K attempts per theorem, for three
runs. Dots mark five reasonable conversions of RL's compute into base attempts; crosses mark equal attempts (256), where
51–60 look "created".*

Our recommendation for anyone claiming RL created something: report the result as a curve over the base's budget,
include a control that gets RL's training without RL's proofs, and say which number of attempts "never" means.

## 6. Thinking harder at test time

If RL mostly makes good-but-rare proofs common, then making the *base* model try harder at test time should recover
part of RL's gain. We tried *guided redraws*, where a checker rejects a logically wrong step and the model tries again,
and *tree search* over partial proofs. Both helped the base model more than the RL model:
- Guided redraws halved the gap between the base and the RL model on one held-out set, though they use about twice the
  compute per attempt.
- Tree search, at equal time, recovered 14% to 44% of what eight RL rounds add, depending on the problem set, for the
  base. For the RL model it was within ±3 theorems of plain sampling everywhere.

![Test-time compute helps the base model more than the RL model](figures/blog_fig4_ttc.png)

*Figure 4. Extra theorems solved by guided redraws (left) and tree search (right) over plain sampling, for the model
before RL (light) and after 8 RL rounds (dark). Dots are individual runs. After RL the model already solves 236–240 of
the 250 held-out problems, so the textbook problems are the fairer comparison.*

## 7. What we did not show

- **One small model on propositional logic.** Large models may behave differently.
- **One RL algorithm, 16 rounds.** Ours is the simplest kind. In the closest related
  study GRPO built behaviour pretraining lacked while methods like ours stalled, and in our earlier, smaller models GRPO
  started a different pattern from zero. But here, from full data, plain GRPO solved fewer theorems than our method
  (which also re-read old training data), and a variant that rewards coverage did about as well. GRPO from the
  knockout models has not been run.
- **Six runs per main condition.** Enough for the main comparison, not for most single-family tests.
- **The knockout isn't perfectly clean.** It made σ's ingredients rarer and cost a few non-classical theorems. The 1%
  dose, which changes the data far less, restored the learning.
- **A null we can't interpret.** Our implementation of "power sampling", which sharpens the base model without training,
  found none of RL's new solves. It also did worse than plain sampling on theorems the base solves rarely, so we count it
  as a failure of our sampler.

## 8. What's next

- GRPO and more exploratory RL from the knockout models.
- σ examples added during RL instead of pretraining, and a 0.1% dose.
- More runs, for the single-family tests.

Code, data and model checkpoints are in public repositories; the paper is a draft.
