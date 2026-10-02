---
written_on: 2026-10-02
written_by: agent:claude
papers:
  - ecoffet2021goexplore
---

# Go-Explore: first return, then explore

Paper: [@ecoffet2021goexplore]
Source: https://arxiv.org/abs/2004.12919v6 (v6 reviewed; html text; published in Nature 2021)

## Learnings
- Diagnosis: exploration fails through "detachment" (forgetting how to reach previously visited states) and "derailment" (failing to first return to a state before exploring from it) (Abstract; Introduction).
- Algorithm (Fig. 1 caption): keep an archive of cells; "(b) Return to the selected state, such as by restoring simulator state or by running a goal-conditioned policy. (c) Explore from that state by taking random actions or sampling from a policy."
- In the main Atari experiments, "the 'explore' step happens through random actions" with no policy, and the state-to-cell mapping "does not require any game-specific domain knowledge" (main text, Atari section).
- Exploration is decoupled from learning: brittle trajectories found by search are turned into robust policies in a "robustification phase", a modified "backward algorithm" (learning from demonstrations) (main text, Atari section).
- Outcome: with domain knowledge it discovered "all 255 rooms of Pitfall" and solved Montezuma's Revenge to the end of level 3. The robustified policies average 102,571 on Pitfall and 1,731,645 on Montezuma (main text, domain-knowledge results). Superhuman and above the previous state of the art "in all eleven games" studied (except Freeway, which is tied at the maximum) (main text, Fig. 2 discussion).
- "Policy-based Go-Explore" returns with a goal-conditioned policy instead of restoring simulator state (main-text section "Policy-based Go-Explore").

## Evidence and limitations
- The out-of-support evidence is strong in the RL sense: rooms and scores that earlier intrinsic-motivation agents never reached, measured directly (rooms visited, score) under sticky actions. It needs **no pretrained prior**: the explore step is uniform random.
- It depends on two things a generic RL loop lacks. The first is cheap return to an archived state (simulator restore). The second is a hand-designed cell representation; domain-knowledge cells help a lot.
- The search is not learning. New behaviour enters the policy only through robustification, a demonstration-learning step on the found trajectories.

## Connections and questions
- **Our interpretation.** Lean proof states are restorable for free: the state-env can start an episode from any stored partial-proof state. That is Go-Explore's "go" step. Our ladder currently restarts every attempt from the theorem statement, which is detachment in Go-Explore's terms. A proof-state archive keyed by (goal, hypothesis set), with expansion from the least-visited or closest-to-goal states, is a cheap E1/E2 variant.
- **E2 (ε-random valid actions).** Go-Explore's own explore step is random actions from a returned state, not ε-mixing along the whole rollout. The paper notes that earlier methods "mix in exploration throughout an episode, usually by adding random actions a fraction of the time", and argues that this causes derailment. This suggests restricting E2's ε to steps after a return to a promising archived state, rather than applying it from the root.
- **Group C.** Robustification uses the backward algorithm: start episodes near the end of a demonstration and move the start point back. We have reference proofs for group C, so starting the state-env at reference-proof prefixes is a direct analogue. It also matches POPE's oracle-prefix guidance (2601.18779v1; see Q2.md).
- **E3.** Go-Explore shows that a random explorer plus an archive can find sparse rewards with no prior. The ND action space (rules × existing lines × subformulas) is small enough for this to be plausible for short theorems.
- Earlier notes: `exit-anthony2017.md` (search as expert, learner as apprentice: the same split as explore/robustify) and `deepseek-prover-v15.md` (RMaxTS, whose intrinsic reward for new tactic states is a count-style novelty bonus).
