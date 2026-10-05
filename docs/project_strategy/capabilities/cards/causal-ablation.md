---
written_on: 2026-10-05
written_by: agent:claude
---

> Reading copy of run `capability-defs`'s `capability_defs/cards/causal-ablation.md` (fork branch `dan_capability-defs`, DONE
> `645aa4f0`); reviewer's corrections marked *[review]*. Report: [`../REPORT.md`](../REPORT.md); review:
> [`review_capability-defs.md`](https://github.com/dan-pandori/nd-takehome/blob/da11be78/review_capability-defs.md).

# Card: causal ablation — remove a skill from pretraining, then ask whether RL brings it back

Family X (intervention). Slug `causal-ablation`. Notation: `_FRAME.md`.

## 1. Definition, formally

The only design in the literature that *intervenes* on what the base can have:
- Pretrain the base on data D⁻ from which a skill σ is removed. For us, σ is a rule (DN, the only classical rule), a
  schema (premise-free A ∨ ¬A), or a length band above a cap.
- Verify that the base has no σ: zero success on σ-requiring theorems at a large budget, and the σ-move outside its
  step support.
- Run the same RL as usual.
- **RL created σ** if the RL model solves held-out σ-requiring theorems that the σ-free base cannot reach at the budget.
- **Dose-response version:** vary the share of σ in pretraining (0, 0.1, 1, 10 %; the Interplay design) and measure how
  RL's success on σ depends on it. A threshold dose below which RL cannot acquire σ is "the elicitation-to-creation
  crossover" the group named on 20-09.

## 2. Decision rule

| verdict | rule |
|---|---|
| **created** | RL from the σ-free base reaches the σ family at the bar (held-out rate ≥ 0.5) on ≥ 2 / 3 seeds, while the σ-free base is at 0 within the budget |
| **elicited** (σ needed in pretraining) | RL reaches σ only from bases that saw σ, above some dose |
| **neither** | RL never reaches σ |

## 3. Null or floor

- The σ-free base is the null by construction. Its reach on σ is measured, not assumed: for a rule that never occurs in
  its data, the action tokens are in the vocabulary but untrained.
- The random-weights objection becomes "RL from a σ-free base is a model with random weights *on σ*" — the right null.

## 4. How to compute it here

- **Part done in this run (J6, J6b):** best-recipe pretraining on K12 minus every DN record, three seeds, then the J4
  demonstration fine-tunes on it (J6b with DN-free replay). This measures **teachability without the skill**
  (elicit-finetune's never-had-it control), not RL. Results (`out/lem.txt`): the knockout's held-out greedy is 0.06 /
  0.09 / 0.10 below pend's; it solves 0 of the 39 classical-only A ∨ ¬A instances (1 / 0 / 1 of 40, the one being
  intuitionistically provable) and 0.88 / 0.80 / 0.85× pend's holdout250 count; 16 demonstrations take it to 34–36 of
  39 with DN-free replay, as fast as pend (gain differences +0.05 / −0.01 / −0.01).
- **The full design (not run):** an 8–16-round EI ladder from each no-DN base. Cost per seed: 8 rounds ≈ 9 A40-hours
  (`rl-continue`: 1.1–1.2 h per round), so ≈ $13 for three seeds at 8 rounds; the excluded-middle burst took 13–14 rounds
  on s1, so ≈ $26 for 16 rounds. A dose sweep multiplies that.
- **Cheap variant (also not run):** remove the A ∨ ¬A schema only. It is already absent premise-free; this run shows K12
  has 0 such theorems, so the existing ladders *are* this ablation at the schema level, with DN present.

## 5. Sensitivity

- **What is removed:** the rule (DN) vs the schema vs examples near σ. Removing DN also removes 13.7 % of the data and
  every classical theorem.
- **RL recipe:** EI with k 32 may never sample a DN step from a base that never saw one (p ≈ 0). GRPO with exploration
  bonuses, or a guided decoder, may.
- **Seeds:** creation events were 1 / 3 even with DN present (s1's LEM), so several seeds are needed.
- **Temperature, representation:** as for the ladder.

## 6. Failure modes

- **Leakage of σ through other data:** e.g. reductio-shaped NEGI proofs may teach parts of DN's pattern.
- **Removal changes general ability** (less data), confounding "can't acquire σ" with "weaker base". The
  general-ability readouts (held-out greedy, holdout250) are needed for the σ-free base. J6 records them.
- **Negative results are uninformative:** EI is known to fail without a first success. The design tests "can this RL
  create σ", not "can RL in principle".

## 7. Relations

- The interventional version of `out-of-data-novelty`.
- The never-had-it control for `elicit-finetune` and `latent-probe-steer`.
- Its dose curve is the most direct test of the "crossover" hypothesis (big-picture notes, sub-question 2).

## 8. Literature anchor

- **Abdulsalam, Patel & Saxe (2607.07646v1;** earlier review): pretraining that excludes shortcuts, "even at pass@1024,
  the pretrained policy gets 0% on Buckets 4–5, while RL later solves them at pass@16".
- **Interplay (2512.07783;** earlier review): RL transfers only when the context made up ≥ 1 % of pretraining.
- **Tsilivis et al. (2510.11495;** earlier review): the p_cot dial.
- **Jain et al. 2023 (2311.12786):** the never-had-it control (L6 note).
- **Deeb & Roger 2024 (2410.08827v3):** "training the model on T should not increase accuracy on V for a model that was
  not trained on either T or V" (Sec. 3.2).

## 9. Critic's verdict

**Strongest argument (critic): with σ = DN the card tests whether an ingredient was *necessary*, not whether pend
*had* the capability, and the verdict is fixed before RL runs.**
- DN is a single token (`Classical.byContradiction`). In the knockout, its output row only ever receives push-down, so
  the base is *anti*-σ, not "random weights on σ": 0 DN steps in 2,997 distinct accepted holdout250 proofs (J6 s0).
- EI reinforces only what it samples, so "created" is unreachable.
- Every success from pend then becomes "elicited (σ needed in pretraining)", the same label pure sharpening gets.
- Secondary arguments:
  - Two of the 80 A ∨ ¬A instances need no DN (G4ip), and the knockout solves one.
  - The positive control meets the bar on at most one seed.
  - Usual RL replay re-injects ≈ 2,700 DN records per round.

**My answer: accepted. The design is changed to ablate a *composition*, not a primitive:**
- drop only the K12 records where DN is applied to a NEGI line (16,703), keeping DN elsewhere (≈ 3 % of records);
- restrict the family to classical-only instances (G4ip);
- draw replay from the ablated corpus;
- rename the negative verdict "not acquired from this base".

J6 / J6b remain valid as the never-had-it control for `elicit-finetune` (teachability), not as this card's test. The
full design (pretrain the composition-ablated base, then 16 EI rounds, 3 seeds) is proposed with a cost of ≈ $26 and
was not run here.
