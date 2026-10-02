---
written_on: 2026-10-02
written_by: agent:claude
status: approved by Dan 2026-10-02 (queued)
---

# Proposal 21: five Claude-heavy, compute-light runs

Dan (2026-10-02): "We currently have a fair amount of free Claude quota, so I'm soliciting proposals that are
Claude heavy & compute light … 2-5 proposals, each of which should have a compute budget under $10." His
suggestions were more literature review, deeper analysis and figures of the group's experiments, and scoping
radical departures. All three are here, plus two of my own. **Every output is a draft for Dan.** Nothing is
sent to, or posted for, anyone else, and other people's branches are read-only.

| run | question | pod budget | brief |
|---|---|---|---|
| `radical-scoping` | Which departures from the current path best answer the capability-emergence question, with CPU feasibility checks? Ready for Dan's team meeting. | ≤ $3 | [brief](claude-heavy/BRIEF_radical-scoping.md) |
| `lit-review-2` | Create-vs-elicit evidence since 2025; exploration that demonstrably reaches outside support; model-organism domains for capability emergence. | $0 | [brief](claude-heavy/BRIEF_lit-review-2.md) |
| `evidence-atlas` | One harmonized map of what the whole project has measured (Dan's runs and others'), with figures and protocol differences. | ≤ $5 | [brief](claude-heavy/BRIEF_evidence-atlas.md) |
| `organism-analysis` | From existing data: what predicts RL success, what RL learns step by step, entropy and diversity, a gallery in Lean. | ≤ $8 | [brief](claude-heavy/BRIEF_organism-analysis.md) |
| `claim-audit` | Red-team the project's headline claims from raw artefacts, before Dan shares them more widely. | ≤ $8 | [brief](claude-heavy/BRIEF_claim-audit.md) |

**Hosts.**
- VPS1: `radical-scoping` and `lit-review-2`, which are reading and writing. They start first, so they are ready
  before the team meeting.
- agent2: `evidence-atlas`, `organism-analysis` and `claim-audit`, which are data-heavy. `claim-audit` waits for a
  free slot.
