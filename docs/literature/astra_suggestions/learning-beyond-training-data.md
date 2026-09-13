---
written_on: 2026-09-13
written_by: agent:codex
title: Learning beyond training data — a reading list for natural-deduction RL
papers:
  - Silver_2018
  - guoDeepSeekR1IncentivizesReasoning2025
  - yueDoesReinforcementLearning2025
  - ellis2020dreamcodergrowinggeneralizableinterpretable
  - Hubert_2025
  - liuProRLProlongedReinforcement2025
  - donoway2026excessdescriptionlengthlearning
  - zomboriFindingLongerProofs2021
  - lake2018generalizationsystematicitycompositionalskills
  - suttonBartoReinforcementLearning2018
---

# TL;DR

- **[AlphaZero — Silver and colleagues, 2018](https://doi.org/10.1126/science.aar6404).** This is the clearest starting point for understanding how search and environmental feedback can produce expertise without demonstrations of expert behavior. Read it to understand the loop that discovers better decisions, trains a network on them, and uses that network to make further discovery possible.
- **[DeepSeek-R1 — DeepSeek-AI, 2025](https://arxiv.org/abs/2501.12948).** This is the essential demonstration that verifiable rewards can substantially improve a pretrained language model’s reasoning performance without supplying a human reasoning trace for each training problem. Its contrast between R1-Zero and the full R1 pipeline forces us to account separately for pretraining, supervised initialization, RL, and distillation.
- **[Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model? — Yue and colleagues, 2025](https://arxiv.org/abs/2504.13837).** This is the essential skeptical companion to R1: higher one-attempt accuracy can coexist with narrower coverage when many answers are sampled. Read it to design evidence for learning new capabilities, while treating its finite-sampling results as observations about particular training regimes rather than a universal limit on RL.
- **[DreamCoder — Ellis and colleagues, 2020](https://arxiv.org/abs/2006.08381).** This supplies a concrete meaning of “new knowledge”: reusable abstractions that make previously impractical problems tractable. Its separation of a learned concept library from a learned search procedure suggests unusually clear experiments on whether our models acquire transferable proof ideas.
- **[AlphaProof — Hubert and colleagues, 2025/2026](https://www.nature.com/articles/s41586-025-09833-y).** This is the most direct bridge from AlphaZero to learning mathematics through a formal verifier. It shows why the proof-state interface, the source of training problems, and the distinction between search and test-time learning deserve as much attention as the RL optimizer.

## The top five, in detail

### 1. AlphaZero: learning from the consequences of search

The mechanism to study is the interaction between the policy, value function, and Monte Carlo tree search: search produces an improved move distribution, self-play supplies outcomes, and the network learns from both. Starting from random initialization, separate agents learn chess, shogi, and Go without expert game demonstrations; the environment, representations, and search algorithm still supply substantial structure. [@Silver_2018] For our question, the useful interpretation is that computation over known rules creates experience from which transferable heuristics can be learned: self-generated experience becomes training data, so “beyond training data” must specify which initial data are excluded. Read the algorithm and training loss, not just the match results; also distinguish improvements in the network from improvements contributed by search at evaluation. **Suggested experiment:** compare the same short-proof initialization under outcome-only RL and iterative search-generated proof training, evaluating both frozen policies and both policies with identical search budgets on withheld proof families.

### 2. DeepSeek-R1: what outcome rewards can teach an initialized model

Read the original report’s sections 2.2–2.4 and 4.1: R1-Zero applies GRPO to DeepSeek-V3-Base using correctness and formatting rewards, while full R1 adds supervised stages and the distillation experiments transfer generated reasoning to smaller models. [@guoDeepSeekR1IncentivizesReasoning2025] These are different sources of learning signal, and none starts from an untrained language model. The paper makes outcome-based improvement compelling, but longer traces and examples of self-correction do not isolate whether RL discovers new algorithms, activates existing ones, or improves their reliability. This matters especially because our [current implementation](../code/experiments/current/r1_cot/README.md) uses a small symbolic model and supervised scratchpad targets: success or failure there need not reproduce the pretrained-language-model regime. **Suggested experiment:** cross direct-proof versus scratchpad generation with RL versus no RL from matched supervised parents, controlling training compute and inference budgets and measuring transfer to structurally withheld theorems rather than treating increased scratchpad length as learning.

### 3. Yue and colleagues: making the capability claim testable

The paper compares base and RL-trained models across sampling budgets and finds settings where RL raises pass@1 but lowers large-k pass@k, the probability that at least one of k attempts succeeds. [@yueDoesReinforcementLearning2025] Read the coverage analyses and temperature/entropy controls: aggregate accuracy can conceal both gains and losses in the set of solvable problems. However, failure in a finite sample does not establish zero success probability, and nonzero likelihood under a base model is a weak definition of possessing a skill. Our stronger operational question is whether training produces reliable, transferable solutions at a declared resource budget; the contrasting ProRL results below make this a hypothesis to investigate. **Suggested experiment:** freeze base and RL checkpoints, tune decoding only on development data, then compare proof-verified pass@k and coverage versus total inference compute on untouched structural splits, reporting per-theorem gains, losses, and uncertainty across training seeds.

### 4. DreamCoder: learning concepts that change what is feasible

DreamCoder alternates program search, abstraction learning that compresses discovered solutions into a library, and training a neural search guide using generated and replayed tasks. Its resulting library is explicit, inspectable, and reusable; the key distinction is between having a solution expressible in the initial language and being able to find it with feasible computation. [@ellis2020dreamcodergrowinggeneralizableinterpretable] Read the wake–sleep mechanism and the learned program hierarchies, especially the list-processing example. For natural deduction, the analogous knowledge could be parameterized lemmas or proof procedures whose usefulness transfers to new combinations of assumptions and goals. DreamCoder is a program-induction system with designed primitives and abstraction machinery, so its success does not show that ordinary policy-gradient RL will spontaneously learn these structures. **Suggested experiment:** mine verified reusable proof procedures from training-only discoveries, compare learned-library and fixed-library systems at matched total compute, and ablate individual procedures to test their causal contribution to held-out compositional transfer.

### 5. AlphaProof: turning formal reasoning into an environment for learning

Read the training and inference sections alongside the Methods description of the Lean environment and AND–OR search, where all subgoals must be discharged. AlphaProof combines pretraining, human-proof supervision, search over a large auto-formalized problem corpus, and learning from verified discoveries; its harder-problem results also use adaptation on generated problem variants at test time. [@Hubert_2025] This makes it a valuable pipeline reference, but neither a demonstration of learning from no human knowledge nor a clean measurement of frozen-model generalization. Its central lesson for us is that choosing what to attempt and exposing useful intermediate states can determine whether the model encounters learnable successes. **Suggested experiment:** on a new adaptation benchmark, compare a frozen prover given additional search with a copy trained on automatically generated target-related variants, match total compute including variant generation and updates, and test both target success and transfer to separately sealed sibling problems.

## What “beyond the training data” should mean here

**My recommendation is to study the acquisition of reusable reasoning procedures under bounded compute.** A newly generated proof string is weak evidence: an old procedure can produce infinitely many new strings. An unseen theorem solved reliably using a newly acquired procedure, with transfer to a withheld family and a causal ablation of the procedure, is a much stronger result.

Keep four claims separate:

| Claim | Evidence we should seek |
| --- | --- |
| Better elicitation or reliability | Higher success probability on problems the starting model can already solve at larger budgets |
| More capable search | More problems solved by a frozen model plus an external search procedure |
| Learning that transfers | A trained, frozen model improves on withheld structures at matched inference budgets |
| Reusable knowledge acquisition | A learned lemma, procedure, or representation supports new families, with interventions showing that it matters |

These are complementary outcomes; reliable access to a formerly rare solution can itself be valuable learning. None establishes novelty relative to **all** training experience: RL trajectories, verifier responses, generated problems, teacher outputs, and model-selection feedback are also information sources. “New to the model,” “absent from its initial demonstration corpus,” and “new to human mathematics” are different claims.

**Natural deduction is a good microscope for some of these questions.** We can control initial exposure, inspect complete proof objects, and verify derivations instead of checking only a guessed final answer. But logical deductions establish consequences of supplied premises and rules; they do not establish contingent facts about the outside world. The setting primarily studies computational access to consequences, and any inference from that to empirical scientific discovery needs another environment.

There are three particularly important limitations of the present setting:

- **Length mixes several kinds of difficulty.** A long proof can repeat one familiar pattern, and a shorter proof may require a difficult lemma or search decision. Reference proofs and shortest-found proofs provide upper bounds on minimum length; use bounded exhaustive search where feasible, and otherwise report length, nesting, branching, rule composition, and search effort separately.
- **The generator defines the generalization problem.** The [local sampler](../code/experiments/current/r1_cot/vendor/nd_data.py) constructs forward derivations over four atoms and takes a derived formula as its target. Under classical Boolean semantics, four atoms have only 16 valuations: syntactic novelty can greatly exceed semantic novelty, although producing formal proofs can still be difficult. Test across independent generators, premise arrangements, withheld rule compositions, and eventually larger vocabularies or richer theories; random train/test splits alone do not establish this transfer.
- **The interface can supply much of the reasoning.** A system choosing among valid rule applications receives different help from a system emitting raw proof tokens; likewise, a tactic can conceal substantial symbolic search. Measure verifier and search costs, compare with a competent symbolic prover, and expand learned macros into primitive verified steps when assessing proof complexity.

The [first-pair audit](../experiment-summaries/2026-09-08-r1-cot/pair1_review.md) is instructive: modest RL coverage gains occurred without observed extension beyond six reference lines, at the tested budgets. Its inherited-model exposure, one seed per arm, and finite sampling prevent a general conclusion about whether RL can learn longer reasoning. **The next concrete step I recommend is to specify a fresh structural-transfer benchmark and its information/compute accounting before choosing the next training recipe**; existing development results can guide that specification, while the existing final-only split should retain its recorded restrictions.

## Five additional readings

### 6. ProRL — Liu and colleagues, 2025

[ProRL: Prolonged Reinforcement Learning Expands Reasoning Boundaries in Large Language Models](https://arxiv.org/abs/2505.24864) is the required counterpoint to recommendation 3. Extended RL with measures to maintain exploration produces coverage gains in settings where the starting model performs poorly, including large-k evaluations. [@liuProRLProlongedReinforcement2025] Read sections 2 and 4; the starting model is already R1-distilled, and finite sampling still cannot establish absolute impossibility for it. This suggests testing whether training duration and sustained exploration change transfer, while holding starting checkpoints, task exposure, and total compute fixed across recipe comparisons.

### 7. Excess Description Length — Donoway and colleagues, 2026

[Excess Description Length of Learning Generalizable Predictors](https://arxiv.org/abs/2601.04728) offers an information-theoretic framework for distinguishing elicitation from learning predictive structure, using sequential label-encoding cost relative to final held-out loss. [@donoway2026excessdescriptionlengthlearning] Read the definitions and toy examples, but retain their assumptions: its guarantees do not automatically cover adaptive RL trajectories or distribution-shifted theorem families. I would first use it as a diagnostic in controlled supervised experiments, comparing mere notation changes with learning a previously withheld proof procedure, rather than declare it a validated scalar measure of new reasoning knowledge.

### 8. FLoP — Zombori and colleagues, 2019; revised 2021

[Towards Finding Longer Proofs](https://arxiv.org/abs/1905.13100) is unusually close to our length-generalization question: RL proof guidance generalizes from very little training data on structured theorem families. [@zomboriFindingLongerProofs2021] Read the backward curriculum in section 3.3 and the different datasets: some emphasize long repetitive calculations, others require more search and composition. That contrast helps expose what a length result actually means. Try curricula starting near the ends of training-only proofs, while evaluating repetitive extension and novel proof composition separately and recording the proof information supplied by those starting states.

### 9. Generalization without Systematicity — Lake and Baroni, ICML 2018

[Generalization without Systematicity: On the Compositional Skills of Sequence-to-Sequence Recurrent Networks](https://proceedings.mlr.press/v80/lake18a.html) introduces SCAN and demonstrates how success on familiar combinations can coexist with failure on systematic recombination. [@lake2018generalizationsystematicitycompositionalskills] Its lasting contribution here is experimental design, not a claim that contemporary transformers must share the tested RNNs’ limitations. Translate its split logic into deduction: expose every primitive rule during training, withhold selected combinations or scope structures, and test whether the learned procedure composes them without having seen that pattern in supervised data or RL experience.

### 10. Reinforcement Learning: An Introduction — Sutton and Barto, second edition, 2018

Read selected parts of [this textbook](https://mitpress.mit.edu/9780262039246/reinforcement-learning/): chapters 3–4 for the environment and policy improvement, chapter 8 for planning versus learning, and chapter 13 for policy gradients. [@suttonBartoReinforcementLearning2018] This is the best conceptual preparation for deciding what the state, action, reward, and learned object actually are in our pipeline. Its framework suggests a tiny, bounded proof environment with an exact planning baseline, used to distinguish exploration failure, weak credit assignment, and insufficient model representation before interpreting a failed neural run as an inability to acquire knowledge.

## Sources and reading locations

1. David Silver and colleagues. *A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play*. **Science**, 2018. [Author-hosted full text](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/alphazero-shedding-new-light-on-chess-shogi-and-go/alphazero_preprint.pdf), especially the algorithm, loss, and supplementary pseudocode.
2. DeepSeek-AI / Daya Guo and colleagues. *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*. [Original report, January 2025](https://arxiv.org/html/2501.12948v1), sections 2.2–2.4 and 4.1; [expanded Nature publication, September 2025](https://www.nature.com/articles/s41586-025-09422-z), which is the version indexed by the project citekey.
3. Yang Yue and colleagues. *Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?* [Paper and version history](https://arxiv.org/abs/2504.13837), 2025, especially sections 2–4 and the coverage and decoding appendices.
4. Kevin Ellis and colleagues. *DreamCoder: Growing generalizable, interpretable knowledge with wake-sleep Bayesian program learning*. [2020 report](https://arxiv.org/pdf/2006.08381), especially Figures 1–3 and the wake, abstraction, and dreaming descriptions.
5. Thomas Hubert and colleagues. *Olympiad-level formal mathematical reasoning with reinforcement learning*. [Nature](https://www.nature.com/articles/s41586-025-09833-y), online November 2025; volume 651, pages 607–613, March 2026. Read Training, Inference, and Methods.
6. Mingjie Liu and colleagues. *ProRL: Prolonged Reinforcement Learning Expands Reasoning Boundaries in Large Language Models*. [2025 paper](https://arxiv.org/abs/2505.24864), sections 2–4 and training limitations.
7. Elizabeth Donoway, Hailey Joren, Fabien Roger, and Jan Leike. *Excess Description Length of Learning Generalizable Predictors*. [2026 preprint](https://arxiv.org/html/2601.04728v1), sections 3–5 and the assumptions attached to the results.
8. Zsolt Zombori and colleagues. *Towards Finding Longer Proofs*. [2019 preprint, revised 2021](https://arxiv.org/abs/1905.13100), sections 3.3–5.
9. Brenden Lake and Marco Baroni. *Generalization without Systematicity: On the Compositional Skills of Sequence-to-Sequence Recurrent Networks*. [ICML 2018 paper](https://proceedings.mlr.press/v80/lake18a/lake18a.pdf), particularly the contrasting generalization experiments.
10. Richard S. Sutton and Andrew G. Barto. *Reinforcement Learning: An Introduction*, second edition. [MIT Press, 2018](https://mitpress.mit.edu/9780262039246/reinforcement-learning/); the publisher’s [contents](https://mitp-content-server.mit.edu/books/content/sectbyfn/books_pres_0/10094/Toc.pdf?dl=1) locate the selected chapters.
