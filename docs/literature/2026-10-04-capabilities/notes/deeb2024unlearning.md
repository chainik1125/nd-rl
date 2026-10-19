---
written_on: 2026-10-05
written_by: agent:claude
papers:
  - deeb2024unlearning
---

# Do unlearning methods remove information from LM weights? (retrain on T, test on V)

Paper: [@deeb2024unlearning] (Deeb, Roger 2024)
Source: arXiv 2410.08827v3 (HTML rendering read: abstract, Sec. 1-8, App. A-H; Table 1 of Sec. 2)

## Learnings

- **The question is exactly ours, for knowledge.** "Historically, it has been unclear whether unlearning
  techniques are removing information from the model weights or just making it harder to access" (Abstract).
- **Information criterion (Sec. 3.1).** With Y the answer to a question and θ the weights after training (a
  random variable depending on Y), an unlearning process U "fully removes the information about q from the
  weights if and only if the mutual information between U(θ) and Y is 0" (Sec. 3.1). Restriction: "Our
  formalization only applies to questions that are practically impossible to guess" (Sec. 3.1).
- **Estimator: Retraining on T (RTT), Sec. 3.2.** Take a set of independent facts with negligible mutual
  information given the rest of the data, split into T and V; the attacker fine-tunes on T and is scored on V:
  "Any facts V that the attacker recovers indicate that these facts were hidden, rather than removed" (Sec.
  3.2). Because T and V are independent, "we do not need to worry about "reteaching" the model the facts, which
  is a concern if we perform attacks that use access to V, like relearning time" (Sec. 3.2). Mechanism: "Training
  on T might reveal information that was hidden by increasing the model's propensity to output the unlearned
  facts without teaching the model the facts again" (Sec. 3.2). Validity condition: "training the model on T
  should not increase accuracy on V for a model that was not trained on either T or V" (Sec. 3.2).
- **Why not relearning time / sample count.** "The main problem with this approach is the lack of a reliable
  baseline for comparison"; after recovery in some number of steps or samples "we cannot determine if this
  number of time steps or samples implies that the information was hidden or removed" (Sec. 2).
- **Metric.** Recovery Rate = (accuracy on V of the unlearned model after RTT) / (accuracy on V of the original
  model after RTT) (Sec. 5.1). For pretrained information "recovery rates … were greater than 88%" (Sec. 5.1);
  for fine-tuned information (Random Birthdays) "a recovery rate less than 35% for GD and RIA" (Sec. 5.2).
- **Negative control run.** RTT on a model never fine-tuned on Random Birthdays: "The highest accuracy we are
  able to get is 31.2%" (App. E; 4-way MCQ), i.e. T does not teach V.
- **Budget.** Splits of 157 items, T = 628 items (Sec. 4.3); 6 epochs, a 6-value learning-rate sweep, 2 T/V
  splits, max over epochs reported (App. A, Table 2).
- **Stress test.** A model built to hide facts at high granularity (layers frozen so the information stays in
  the weights) drops to 35 % forget accuracy; RTT brings it "back to 83% quickly" (Sec. 6.1).
- **Limitations stated.** "Our evaluation does not guarantee that information is removed from the weights;
  rather, it sets a higher bar than previous evaluation methods for unlearning" (Sec. 7.1). Low-leakage data
  are needed, so the method does not cover capabilities: "if the goal is to unlearn the capability of coding,
  it's hard to construct T and V with low leakage" (Sec. 7.1).

## Evidence and limitations

- Llama 3 8B mainly; MCQ evaluation after plain-text unlearning (Sec. 4). Uncertainty bound 2σ ≤ 0.055 from 157
  items × 2 splits (App. A). Learning rate chosen by highest validation accuracy and max over 6 epochs reported
  (Sec. 4.3, App. A): optimistic for the attacker, which is the conservative direction for an unlearning claim.
- The design needs facts with negligible mutual information; proofs are the opposite (every proof follows from
  shared rules), which the authors flag as the capability case they cannot handle (Sec. 7.1).
- Not checked: figures (images), App. I-K.

## Connections and questions

- **Definition offered:** information about Y is "in the weights" iff I(θ; Y) > 0; estimated by the accuracy on
  held-out items V after fine-tuning on independent items T, normalised by the same procedure on the original
  model (Recovery Rate).
- **New vs better access:** the cleanest operational rule in this thread. Access = what training on
  *information-disjoint* data (T) can unlock on V; creation (or re-teaching) = what requires training on V
  itself. The rule needs (i) T ⟂ V informationally and (ii) a negative control showing that a model without the
  information gains nothing on V from T (App. E). It explicitly rejects a raw step/sample budget as
  uninterpretable without such a baseline (Sec. 2) — the same complaint as Dan's about k.
- **Null / floor:** handled by the negative control: a model that never held the information stays near chance
  after RTT (31.2 %), so "eventually recoverable" is ruled out by design rather than by a budget. But the
  mutual-information criterion explicitly excludes guessable answers (Sec. 3.1), and theorem proving is all
  "guessable" by search: a proof carries no information beyond the rules, so for us the criterion must become
  resource-bounded (information usable by a bounded learner/sampler), which is where a budget re-enters.
- **Transfer to our setting:** an RTT analogue is natural because RL already trains on `rl_targets` (4,495) and
  is scored on the held-out 322. Add (a) the negative control: the same RL or SFT on T applied to a model that
  lacks the capability (an early Stage-1 checkpoint such as step 1,600, or a base trained on data with a rule
  family removed), and (b) the positive reference: RL on T from pend. Define a recovery-rate-like ratio on V
  (pass@k or per-theorem log p̂): RL gain on V that the negative control also achieves is taught by T; gain only
  pend achieves is pend's pre-existing information made accessible. Strongest version: build T and V from
  disjoint schemata/rule mixes and verify low leakage with the negative control first. Cost: one ladder per arm
  (hours on an A40) or minutes for SFT-on-T; the rl-from-ckpt ladders (starts at steps 0/1,600/5,000/…) already
  supply part of (a). Main failure mode: leakage — proof skills transfer between T and V, so the negative
  control will gain too and the difference-in-gains is all that remains; a better base also learns faster from
  T, which mimics "pre-existing information".
- Related notes: greenblatt2024passwordlocked.md and hofstatter2025elicitation.md (fine-tuning as elicitation;
  same author F. Roger), harding2024capability.md.
