---
written_on: 2026-10-05
written_by: agent:claude
---

> Reading copy of run `capability-defs`'s `capability_defs/cards/elicit-finetune.md` (fork branch `dan_capability-defs`, DONE
> `645aa4f0`); reviewer's corrections marked *[review]*. Report: [`../REPORT.md`](../REPORT.md); review:
> [`review_capability-defs.md`](https://github.com/dan-pandori/nd-takehome/blob/da11be78/review_capability-defs.md).

# Card: elicitable by small fine-tuning (sample complexity, description length)

Family E. Slug `elicit-finetune`. Notation: `_FRAME.md`.

## 1. Definition, formally

For a capability defined over a family 𝒯 (e.g. the 40 held-out transfer A ∨ ¬A instances), and a fine-tuning recipe
𝓕 (here: one EI-round fine-tune, 600 steps, lr 3e-4 → 3e-5, with 20,000 pretraining replay records):

- **Sample complexity to elicit:** N*_B(𝒯) = the smallest number of demonstrations D (proofs of *other* family
  members, never of 𝒯's own instances) such that 𝓕(B, D) reaches mean pass@256 ≥ τ on 𝒯.
- **Description-length version (prequential / EDL).** The online code length of the demonstrations under 𝓕 starting
  from B, Σᵢ −log π_{θ_{i−1}}(yᵢ), minus the final model's code length. Compare it with the same from a control start
  (init, or a model not pretrained on the family's moves): elicitation is cheap in bits, teaching expensive.
- **The Deeb–Roger variant** (information already in the weights): fine-tune on demonstrations T, test on held-out V,
  where T and V share the skill but no instance. Include a control model that never had the skill. Recovery on V
  beyond the control means the skill was present.

## 2. Decision rule

*Revised after the critic pass (§9).* Compare the gain demonstrations give the base with the gain they give a
**never-had-it control**: a model pretrained by the same recipe on data without the family's key step (here the no-DN
knockout), both fine-tuned identically with replay drawn from the knockout's corpus.

gain(M) = held-out pass@256 after N family demonstrations − held-out pass@256 after N matched non-family proofs.

| verdict | rule |
|---|---|
| **latent in the base** | gain(base) − gain(knockout) ≥ 0.3 on ≥ 2 / 3 seeds |
| **teachable from scratch** | |gain(base) − gain(knockout)| ≤ 0.2 on ≥ 2 / 3 seeds |
| **between** | otherwise |

This card no longer decides "created vs elicited" by itself: fine-tuning on verifier-checked proofs is RL's own
mechanism (`capability-vs-propensity`), so it measures how cheaply a capability can be *installed*.

## 3. Null or floor

- **The never-had-it control is the floor**: the no-DN knockout, with replay from its own (DN-free) corpus so the
  fine-tune cannot re-teach the move. init was dropped as a control after the critic: it learns nothing from 16
  demonstrations for reasons unrelated to the skill.
- **Matched non-family demonstrations** (C16n: 16 length-matched non-LEM proofs) separate "demonstrations of the skill"
  from "any fine-tune".

## 4. How to compute it here

- **J4** (pre-registered): from pend s0 / s1 / s2, mixes A0 (replay only), A4 / A16 (4 / 16 A ∨ ¬A demonstrations from
  s1's r16 found set on `rl_targets`), C16 (16 length-matched non-LEM proofs). Read the 40 held-out transfer A ∨ ¬A
  instances at k 256 (decisions on the 39 that are not intuitionistically provable). About 0.5 A40-hours per seed.
- An init-start control and a prequential sweep (N = 1, 2, 4, 8, 16) would cost ≈ 1 more GPU-hour. They are not run
  in J4; this is stated as a limitation.
- **J6 / J6b** (pre-registered): the no-DN knockout (pend's recipe on K12 minus every DN record) and, after the critic,
  fine-tunes of pend and of the knockout on knockout-corpus replay (A16n / C16n), two fine-tune seeds each.
- **Results** (`out/lem.txt`; the 39 classical-only held-out instances, k 256, solved of 39 with mean pass@1):
  - pend, pend + A0 (replay only), pend + C16: 0 on every seed. RL r16: 0 / 36 / 13.
  - pend + A4: 36 (0.66) / 33 (0.35) / 36 (0.64). pend + A16: 37 (0.79) / 34 (0.68) / 37 (0.83).
  - Knockout: 0; knockout + A16 (K12 replay): 35 / 37 / 37.
  - J6b, DN-free replay: pend + A16n 36, 36 / 36, 35 / 35, 36; knockout + A16n 34, 34 / 36, 36 / 36, 36; both + C16n 0.
  - **Decision (pre-registered: gain = pass@256(A16n) − pass@256(C16n), pend vs knockout):** differences +0.05 /
    −0.01 / −0.01 → **teachable from scratch** on 3 / 3 seeds (latent would need ≥ 0.3 on ≥ 2 / 3). Secondary, mean
    pass@1: +0.05 / +0.04 / +0.12 (the base slightly ahead).
  - A16 also raises holdout250 solved@64 by +12 / +15 / +5 over A0.

## 5. Sensitivity

- **The recipe 𝓕:** learning rate, steps and replay fraction change N* by large factors. EDL "depends on the training
  algorithm" (Donoway et al., earlier review).
- **Demonstration quality:** demos the model "can't imitate well" give lower performance "no matter the number of
  demonstrations" (Greenblatt et al.). Ours are RL-found proofs, i.e. in-distribution for a sibling model.
- **Representation:** the fine-tune trains (state, action) pairs. A whole-proof format would teach differently.
- **Renaming:** test instances must not be renamings of demonstrations. This holds: the 40 transfer instances are a
  disjoint generation, checked for renaming-class overlap with textbook72.
- **Noise:** one fine-tune per arm per seed; the fine-tune seed adds variance. Report all three seeds.

## 6. Failure modes

- **Teaching vs eliciting is a matter of degree:** "the difference between tuning and training proper is a matter of
  degree" (Harding & Sharadin, fn. 38). The rule needs the control, not an absolute N.
- **Leakage:** T and V share sub-skills by design. Deeb & Roger: "if the goal is to unlearn the capability of coding,
  it's hard to construct T and V with low leakage".
- **The fine-tune also pretrains:** 20,000 replay records are more pretraining. A0 isolates that.
- **One family only (LEM)** in J4. Generalising needs more families.

## 7. Relations

- Extends `capability-vs-propensity` to weight changes.
- Its bits version is the training-side counterpart of `bits-over-null`.
- `schema-acquisition` supplies the families.
- `causal-ablation` is the stronger, interventional cousin.

## 8. Literature anchor

- **Greenblatt et al. 2024 (2405.19550v1).**
  - "fine-tuning on a single demonstration is often enough to recover most of the "hidden" performance" (Sec. 1).
  - 18 M-parameter models "locked from scratch require 2-4x the amount of data to unlock" (Sec. 7).
  - "it is not necessarily clear if fine-tuning is eliciting hidden capabilities or re-introducing them" (Sec. 3).
- **Deeb & Roger 2024 (2410.08827v3).**
  - The T / V design (Sec. 3.2): "training the model on T should not increase accuracy on V for a model that was not
    trained on either T or V".
  - Relearning time lacks "a reliable baseline for comparison" (Sec. 3.2).
- **Harding & Sharadin 2024:** "if a model can be fine-tuned with very little compute … this does provide indirect
  evidence that it already had the ability" (Sec. 5.2.2).
- **Whitney et al. 2020 (2009.07368v2):** examples needed and surplus description length to reach ε loss (Sec. 4).
- **Voita & Titov 2020 (2003.12298v1):** online code length with control tasks (Sec. 3–4).
- **Blier & Ollivier 2018 (1802.07044v5):** prequential coding.
- **Donoway et al. 2026, EDL** (earlier review).
- Verified in `_claims_L2.md` / `_claims_L3.md`.

## 9. Critic's verdict

**Strongest argument (critic): the control decides the verdict, not pend.**
- **Init** can never write proofs, so "the control never reaches τ" holds for every family, and the rule collapses to
  the bare "≤ 16 demonstrations".
- **The pre-registered knockout (J6)** is re-taught the missing move by the fine-tune itself: its 20,000 K12 replay
  records include 2,764 with DN, 2,178 of which apply DN to a NEGI line (the template's closing step). That is ≈ 8,300
  DN presentations against ≈ 245 for the demonstrations.
- Same pend, same demonstrations, opposite verdicts, depending on the control.
- Secondary arguments:
  - Leakage by design: 11 of the 16 demonstrations share one 9-line skeleton, and lem40 only asks for a new X in the
    same template.
  - "Absorbing ≠ having": the rule ignores R (s0 kept one LEM proof in every fine-tune, yet held-out LEM stayed at
    0.03).
  - N* is never measured: there are only N ∈ {0, 4, 16}, one fine-tune seed, and no τ.

**My answer: accepted, and the design is corrected.** J6b (pre-registered in `log.md` before launch):
- Both arms' replay is drawn from the knockout corpus (K12 minus DN), for pend and for the knockout alike.
- Arms A16n (16 demonstrations) and C16n (16 L_true-matched non-LEM proofs), two fine-tune seeds each, for all three
  training seeds.
- The decision quantity is the gain (A16n − C16n) on held-out A ∨ ¬A, pend vs knockout.
- **Revised rule:**
  - "latent in pend" (elicited) if pend's gain exceeds the knockout's by ≥ 0.3 on ≥ 2 / 3 seeds;
  - "teachable from scratch" if the gains are within 0.2;
  - init is dropped as a control.
- The template-leakage caveat stays: the test measures schema completion with new formulas, not open-ended discovery.

