---
written_on: 2026-10-04
written_by: agent:claude
status: approved by Dan 2026-10-04 (launched as `capability-defs`)
---

# Proposal 26: what "capability" should mean for this project, and how to measure it

Dan (2026-10-04) wants a Claude-heavy run on the notion of capability. In his words: "I'm happy with the agents really
thinking about many different ways to define 'capabilities', as long as there is some way to make them
quantifiable." He set a $50 pod budget, to be used only if the agents find pod experiments useful, and asked that
they "really take their time" at maximum thinking effort.

| part | what |
|---|---|
| 1. Literature | How ML, RL, psychometrics, information theory and evaluation practice define a capability, and tell a new one from better access to an old one. ≥ 60 new papers screened, ≥ 20 read, every claim checked against its source. Extends `lit-review` and `lit-review-2`. |
| 2. Catalogue | ≥ 15 definitions, each with a formal statement, a decision rule (created / elicited / neither), a null answering "random weights eventually solve anything", sensitivity, failure modes and a critic subagent's case against it. Dan's two notions (teacher-forced probability of specific proofs; pass@k at large k) are made strict. |
| 3. Quantify | ≥ 6 definitions computed on the same comparisons (cap-12 and cap-6, pretraining end vs r8 vs r16), with the theorems each one says RL "created", noise floors, and an agreement matrix across definitions. |
| 4. Recommend | 2–4 definitions as the project standard with exact protocols, what our results already show under each, a plain glossary, and open questions for Dan. |

**Run:** `capability-defs` on VPS1. Every role runs at `--effort max`, set through a per-run effort file that
`runrole` now reads. Budget $50 / 100 pod-hours; hard stop 72 h. Brief:
[claude-heavy/BRIEF_capability-defs.md](claude-heavy/BRIEF_capability-defs.md).
