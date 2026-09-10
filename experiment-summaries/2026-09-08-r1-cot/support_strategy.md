---
written_on: 2026-09-08
written_by: agent:codex
---

# Creating useful RL support without increasing the initial cap

**First repair the classical precursor skills, then sample near-frontier targets with larger groups.** Overall reward is misleading when easy targets always succeed and hard targets always fail. In the allowed training-log snapshot, direct RL has 214/256 updates with zero mixed groups, despite mean reward 0.626; its 256 updates contain only 44 mixed groups. CoT's incomplete 219-update snapshot has 106 zero-mixed updates and 164 mixed groups, with mean reward 0.607. Both report a maximum found proof length of seven. These are training diagnostics, not textbook evaluation or an equal-compute endpoint comparison.

I inspected only the 14 authorized `textbook_train.jsonl` tasks and the exact `pair1_8m_{direct,cot}_rl/events.jsonl` logs. The training tasks emphasize classical negation, implication discharge, and composing biconditional halves. No development/final textbook statement or prediction was read.

The separate [precursor builder](../../code/experiments/current/r1_cot/prepare_precursors.py) produced **2,974 canonical synthetic labels**, all strictly verified and at most six lines, from 25 short subformula fragments of the permitted training tasks:

| Precursor family | Labels | Lines |
|---|---:|---:|
| Negated disjunction to a negated component | 535 | 5 |
| ¬(¬A∨¬B) ⊢ A | 537 | 6 |
| ¬(A∧B), A ⊢ ¬B | 536 | 6 |
| A→¬¬B ⊢ A→B | 537 | 5 |
| Forward half of a conjunction encoding equivalence through ¬¬B | 537 | 6 |
| ¬A ⊢ A→B by contradiction and explosion | 292 | 5 |

The builder rejected 338 candidate instances through held-out canonical/context/skeleton guards and 438 canonical duplicates. None of the retained labels exactly matches a textbook class; 64 already occur in the current short pool. Both current synthetic development populations are also protected. The permitted textbook population remains exactly 14 classes. These labels are **synthetic supervised repair**, not model discoveries or longer proof labels.

Data and hashes are at `simplex1:/data/users/dmitry/nd-rl/r1-cot-20260909/data/precursors_cap6/{precursors_cap6.jsonl,manifest.json}`. Eighteen independently constructed synthetic template/substitution checks also passed the real strict verifier and the dependency-plan vocabulary check. The [saved manifest](precursors_cap6_manifest.json) records lineage and counts.

Recommended next run:

1. Continue matched direct/CoT parents with approximately 25% precursor records and 75% retained short data for a bounded 500–1,000 updates; compare to an equal-update old-data continuation. Keep the generator-label cap at six and record token differences. The dependency-goal scratch format is a useful separate condition because these proofs have explicit assumption/subgoal structure.
2. Run a quarantined support probe on **training-only** seven-plus-line targets and the same permitted 14 textbook prompts. Compare temperatures 0.8, 1.0, and 1.15 with 32 samples. Measure distinct accepted targets and mixed groups, not overall reward. These diagnostic samples must not silently become the next training batch.
3. Use groups of 16–32 after repair, with enough distinct prompts per update and a curriculum favoring near-frontier cases that actually have nonzero success. At success probability 1%, a group of eight has about 7.7% probability of mixing success/failure; a group of 32 has about 27.5%. Larger groups cannot repair truly absent sampling support. Maintain short retention through supervised updates instead of spending half the rollout budget on all-success targets.
4. If longer final proofs remain absent, first test a larger response allowance on the same training support panel, preserving room for the final certificate after scratch. Then consider harvesting strictly verified **closed** prefixes as hindsight goals, preserving original premises and applying every held-out guard after relabeling. Do not train on active-assumption fragments as if they were closed proofs.

Synthetic failed-attempt repair traces are lower priority. Without useful feedback, they can teach a model to ignore noisy scratch and copy a final answer. The immediate evidence favors missing composition support over a need for a more elaborate reward: maintain binary final correctness initially. End each declared RL run before textbook development evaluation; keep final-only tasks sealed until frozen final selection.
