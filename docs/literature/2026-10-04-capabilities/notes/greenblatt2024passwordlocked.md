---
written_on: 2026-10-05
written_by: agent:claude
papers:
  - greenblatt2024passwordlocked
---

# Stress-testing capability elicitation with password-locked models

Paper: [@greenblatt2024passwordlocked] (Greenblatt, Roger, Krasheninnikov, Krueger 2024)
Source: arXiv 2405.19550v1 (HTML rendering read: abstract, Sec. 1-9, App. B, D, F, G; App. C and E skimmed)

## Learnings

- **Working definition (the paper says it is informal: "we are not aware of any rigorous accounts", Sec. 2).**
  "We say an AI system exhibits a capability to perform a task if it produces high-quality responses across a
  set of inputs representative of that task" (Sec. 2). "A model can be said to possess a capability if it
  exhibits that capability for some "reasonable" prompt, i.e. one that doesn't itself effectively provide the
  capability in question" (Sec. 2). Example: a translator persona prompt is reasonable, "a prompt which
  includes a French -> English dictionary would not" (Sec. 2). A capability is **hidden** (from a party) if an
  eliciting prompt exists but that party does not know it (Sec. 2).
- **Scope of the definition, as stated by the authors.** It covers what API misuse could exploit "but not
  capabilities that fine-tuning could easily elicit and that could be exploited by users with weights or
  fine-tuning access" (Sec. 2); "it is difficult to show that a model does not possess a particular
  capability" (Sec. 2). Fine-tuning-based elicitation does not cover "very large fine-tuning runs that teach
  LLMs new capabilities" (Sec. 1). On natural models: "it is not necessarily clear if fine-tuning is
  eliciting hidden capabilities or re-introducing them" (Sec. 3) — the reason for constructing models whose
  hidden capability is known by design.
- **Construction (Eq. 1, Sec. 2).** Supervised training on D_lock = {(x, y ~ π_weak(x))} ∪ {(pwd+x+pwd,
  y ~ π_strong(x))}. Unlocking = recover the with-password performance without the password. Metric:
  "the proportion of problems for which the temperature 0 generation (so, greedy sampling) is correct"
  (Sec. 4.2).
- **SFT elicitation is very sample-efficient** on generation tasks: "fine-tuning on a single demonstration is
  often enough to recover most of the "hidden" performance"; MMLU without CoT "requires a few hundred samples"
  (Sec. 1); "fine-tuning on 512 samples isn't much better than fine-tuning on 4" (Sec. 5.1, all tasks except
  MMLU). Demonstrations that are "often incorrect or which the password-locked model can't imitate well" give
  lower performance "no matter the number of demonstrations" (Sec. 1). Demonstrations from one sub-domain
  unlock other sub-domains (Sec. 5.2, Fig. 4).
- **RL elicitation depends on the starting policy's hit rate.** "starting from a worse policy results in poor
  exploration, which slows down and sometimes entirely prevents learning" (Sec. 6.2). From fixedrdm (a fixed
  random token distribution) "RL is unable to explore and reinforce good behaviors when starting from such weak
  policy" (Sec. 6.2). Budget used: "50 iterations for expert iteration and filtering (each generating 640
  samples and fine-tuning on the best 64 ones)" (Sec. 6.1). Expert iteration "is able to pick up on a few lucky
  examples sampled from π_weak and gradually bootstrap from that to π_strong" (App. D.2).
- **From-scratch locking (closest to our scale).** 18 M-parameter transformers on a 32-function composition task
  (Sec. 7): "models locked from scratch require 2-4x the amount of data to unlock to the same extent as models
  locked via SFT", but "even models locked from scratch do eventually get unlocked, and the number of
  demonstrations required is small overall" (Sec. 7). Motivation: "previous works find fine-tuning yields only
  superficial changes which can easily be undone" (Sec. 7, citing Jain et al. 2023).
- **Limitations stated.** Password-locked models "are only intended to assess whether an elicitation method
  reveals those capabilities which could be accessed via prompting" (Sec. 8); fine-tuning and tooling "can
  expand the capabilities" (Sec. 8). "Hidden capabilities may be less "salient" in realistic models" (Sec. 8,
  item 3); "sample efficiency may be much lower when good answers are less salient" (App. D.2).

## Evidence and limitations

- Ground truth is by construction: the locked model was trained on π_strong, so "possesses" is not inferred
  but known. This is the paper's strength and the reason it cannot by itself give a rule for natural models.
- No negative control of the kind we need: there is no run that applies the same few-demonstration SFT
  budget to a model that never had the capability (to measure how much the budget *teaches*). The closest
  are the weak-sample sanity check (fine-tuning on π_weak samples raises accuracy on 2/4 tasks, "spurious
  unlocking", App. B.2) and the fixedrdm RL start (Sec. 6.2).
- Epoch selection uses "the highest average test correctness among our 3 seeds" (App. G.3) — mild selection
  on the evaluation set; results are 3-seed means.
- Not checked: per-figure numbers (Figs. 3-6 are images in the HTML rendering; I quote only text claims).

## Connections and questions

- **Definition offered:** possessing a capability = high-quality responses on representative inputs under
  *some reasonable elicitation*, where reasonable means the elicitation does not itself carry the capability
  (the "dictionary in the prompt" exclusion). Operationally: performance after the best elicitation method
  that is known not to inject the answer; the paper validates methods by whether they recover a known,
  deliberately hidden capability.
- **New vs better access:** decided by construction (the capability was inserted), not by a rule. The
  conceptual criterion is the information-content exclusion: an elicitation counts only if it "doesn't itself
  effectively provide the capability". The paper concedes that large fine-tuning teaches (Sec. 1, 8) but
  gives no threshold. A quantitative rule would be a **calibrated budget**: the elicitation budget B is
  admissible if it recovers a known hidden capability (positive control: password-locked model) and does not
  produce the capability in a model known to lack it (negative control). The paper supplies only the
  positive half.
- **Null / floor:** the fixedrdm result is the elicitation version of Dan's objection: with a fixed budget
  (here 50 × 640 samples), RL from a random policy gets nowhere, so "random weights eventually solve it" does
  not translate into "random weights possess it" once the elicitation budget is fixed. The paper does not
  formalise the budget.
- **Transfer to our setting:** cheap and direct. (1) Positive control: train a password-locked model from
  scratch (as in Sec. 7, at our 3-10 M scale): with a prefix token, imitate reference proofs (or pend's own
  accepted proofs); without it, imitate a weaker policy (e.g. a cap-6 base, or pend with a rule such as
  `Or.elim` or `Classical.byContradiction` suppressed). Then measure the SFT demonstrations and EI rounds
  (k = 32, our ladder) needed to unlock. (2) Negative control: same budget on a model whose pretraining data
  never contained the rule. A budget that unlocks (1) but leaves (2) at floor is a defensible "elicitation
  budget"; RL gains on pend achieved within it count as elicitation, gains beyond it as creation relative to
  that budget. Cost: one Stage-1 run per locked model (≈ 20 min on an A40 at our scale) plus minutes of SFT;
  EI unlocking is a ladder (hours). Main failure mode: salience — a locked capability trained on exact
  reference proofs is more "salient" than a capability pend has only diffusely (the authors' own caveat,
  Sec. 8), so unlock budgets measured on locked models may understate what natural elicitation needs.
- Related notes: vanderweij2024sandbagging.md (password-locking as sandbagging), hofstatter2025elicitation.md
  (password-locked + circuit-broken organisms; SFT as the reliable elicitor), deeb2024unlearning.md
  (recoverability through fine-tuning as the information criterion), harding2024capability.md (conditional
  analysis: "if tried" ↔ "reasonable prompt").
