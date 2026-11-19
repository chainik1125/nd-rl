---
written_on: 2026-11-18
written_by: agent:claude
---

# Other ways to split proofs into strategies

Four research agents looked at this on 2026-11-18, after the rule knockouts in the parent folder showed that
"one rule = one strategy" gives too few strategies. Each note is the agent's own write-up. Their counts on our data
were not independently re-checked.

| note | angle |
| --- | --- |
| [proof_theory.md](proof_theory.md) | decompositions from the proof-theory literature: structural flags (reuse or ignore an assumption), the logic each theorem needs, boxes opened to feed a hypothesis, detours |
| [motifs.md](motifs.md) | multi-step patterns mined from the pretraining proofs, the RL targets and the control runs' proofs, with how much of pretraining each knockout deletes |
| [prior_work.md](prior_work.md) | what Dan's taxonomy and knockouts, Leon's directions and Robbie's long-RL shapes already say, and which of their strategies are the best next arms |
| [other_domains.md](other_domains.md) | systems outside natural deduction where strategies are independent: equational reasoning, implication-only logic with structural switches, Horn rules, Hilbert systems |

The summary written for the group is the shared doc "A generator basis for proof strategies":
https://claude.ai/artifact/LTdzaX9RjPZxGZsYeV4w5F. [generator_basis.md](generator_basis.md) is a markdown export of it.

`code/` holds the agents' scratch scripts and small outputs, one folder per note. They are unreviewed and exempt from
lint. Paths inside the notes point at the temporary directory the agents worked in; the same files are under `code/`.
Large intermediate files (pickles, the n-gram table, per-target labels) were not kept; the scripts regenerate them.
