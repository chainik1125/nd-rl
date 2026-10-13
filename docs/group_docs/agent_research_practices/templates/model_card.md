---
written_on: YYYY-MM-DD
written_by: <human:Name or agent:claude>
---

# Model card: `<model-name>`

Named by <name>, <date>. <One sentence on what it is and why it has a name.> Once published, this definition does not
change. A variant gets a new name and its own card.

## What it is

| part | setting | from |
| --- | --- | --- |
| network | <architecture, layers × width, **parameter count**> | <run or recipe> |
| pretraining | <optimiser, objective, duration and hardware> | |
| interface | <whole proof / proof state; tokenizer> | |
| data | <training set, size, length range, filters> | |
| RL | <algorithm, rounds, attempts, temperature, targets, replay> | |
| judge | Lean 4 core alone | |

## Checkpoints

<Paths in the external store, one per seed.>

## Headline numbers

Give these per seed, saying plain or guided, with k and temperature.

| set | metric | s0 / s1 / s2 | source |
| --- | --- | --- | --- |

## Known limits

<Saturation, overlap between training and evaluation data, settings outside the policy, compute relative to
comparators.>

## Experiments that used it

<Run ids, with links to their summaries.>
