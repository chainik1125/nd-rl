---
written_on: 2026-10-05
written_by: agent:claude
papers:
  - wu2023counterfactual
---

# Reasoning or reciting? The default-vs-counterfactual gap as a test of a transferable procedure

Paper: [@wu2023counterfactual] (Wu, Qiu, Ross, Akyürek, Chen, Wang, Kim, Andreas, Kim, NAACL 2024)
Source: arXiv 2307.02477v3 (HTML rendering read: abstract, Sec. 1-2.1, 3.4, 4, 5.1-5.5, 6, 7; per-task
appendices not read).

## Learnings

- **Claim.** Across 11 tasks, "we observe nontrivial performance on the counterfactual variants, but
  nevertheless find that performance substantially and consistently degrades compared to the default
  conditions" (Abstract); hence benchmark success "should not be considered as sufficient evidence for
  their possession of full general capacity for the target task" (Sec. 1).
- **Design.** A task is f_w: X → Y under a world model w (conditions such as the number base). A
  counterfactual task keeps the procedure but changes the mapping: "The general reasoning procedure for
  these tasks remains the same under the new conditions, but the specific input-output mapping functions are
  changed"; "If models implement a general and transferable task-solving procedure, we expect comparable
  performance on counterfactual and default tasks" (Sec. 1). Attribution needs matched difficulty: "If we
  control f_w(x) to be similarly hard between w^default and w^cf, we can attribute the performance
  difference to an LM overfitting to the default instantiation of the task" (Sec. 2).
- **Control.** "counterfactual comprehension checks (CCCs) that test an LM's surface understanding of the
  specified counterfactual world" — a simpler task g_w that distinguishes the two worlds (Sec. 2.1).
  Counterfactual worlds are not guaranteed unseen: "Nor do we aim to guarantee that counterfactual world
  models are unobserved in a pretraining corpus" (Sec. 2).
- **Graded, frequency-linked.** Results "point to a memorization-like effect where the models perform
  better under more common conditions" (bases 8 and 16 beat 9 and 11; Sec. 5.1); performance falls with
  distance from the default (Sec. 5.2). Default and counterfactual accuracy correlate across tasks,
  instances and models, so the question "is not a dichotomy, but rather they can co-exist in a continuum";
  a memorisation signature is that "the base-10 performance decreases much more slowly than the other
  bases" as digits increase (Sec. 5.3, Fig. 4a). Few-shot demonstrations shrink but do not close the gap
  (Sec. 5.5).
- **Budget.** Discussing humans: competence to generalise "even though it may sometimes require sufficient
  execution budget to realize it as robust performance" (Sec. 6).
- **Limitation that matters for us.** Superficial perturbations "may admit a shortcut where the model first
  figures out a simple mapping of the input back to the default conditions and performs the task" — the
  logic task's word replacements are the example (Sec. 7.2).

## Evidence and limitations

- Closed models (GPT-4, GPT-3.5, Claude v1.3, PaLM-2), 0-shot with and without CoT, single greedy-style
  decoding (argmax with approximate decoding, Sec. 2). Difficulty matching is imperfect (Sec. 7.1:
  "an objective difficulty measure may not even exist"); the pretraining corpus is unknown (Sec. 7.2).
- Per-task numbers in Fig. 2-3 / App. C were not checked.

## Connections and questions

- **Definition offered:** a general (abstract, transferable) task-solving capability is one whose accuracy
  is invariant to a change of the task's conditions that preserves the procedure; the default-minus-
  counterfactual gap, controlled by a comprehension check and matched difficulty, measures how much of the
  observed performance is condition-specific (recitation).
- **New vs better access:** not about training stages, but the gap is a ready-made rule for *what kind*
  of capability RL produced: RL that raises accuracy on default instances while the counterfactual gap
  widens has added condition-specific behaviour (closer to memorised instances); RL that raises default and
  counterfactual accuracy together, on held-out instances, has added a transferable procedure.
- **Null / floor:** chance-level baselines per task; no sampling budget. The commonness analysis is a
  useful null: if counterfactual accuracy tracks the pretraining frequency of the counterfactual condition,
  frequency (exposure), not reasoning, explains it.
- **Transfer to our setting:** we know pretraining frequencies exactly, which removes the paper's main
  confound (Sec. 7.2). Default vs counterfactual conditions available here: frequent vs rare atom letters,
  canonical vs permuted premise order, curried vs uncurried hypotheses, frequent vs rare formula depth. The
  atom-renaming case is exactly the "superficial perturbation" the authors warn can be undone by mapping
  back, so renaming invariance is a necessary but weak test for our models; it should be paired with a
  condition that changes the proof itself (e.g. premises presented in a different but equivalent logical
  form). CCC analogue: the model's solve rate on trivial theorems (one-step proofs) in the same surface
  form, to separate "cannot read the format" from "cannot do the reasoning". Cost: generator variants plus
  J2-type sampling at k = 256 for pend, r8, r16. Failure modes: tokenizer and name-base effects make some
  "superficial" changes non-superficial for a 3-10 M parameter model; difficulty matching by proof length
  and term size must be checked per variant.
- Related: `hupkes2020compositionality.md` (substitutivity consistency), `firestone2020performance.md`
  (competence vs performance), `harding2024capability.md` (reliability across conditions), screened here:
  GSM-Symbolic (name/number templates), McCoy et al. 2023 (Embers of autoregression).
