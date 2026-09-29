---
written_on: 2026-09-20
written_by: agent:codex
---

# Synthetic proof-length frontiers after cap-six task training

## Summary

This note compares the longest verified synthetic certificates observed when the experiment's
supplied task-training proofs or demonstrations had at most six natural-deduction lines. This is a
task-specific supervision constraint, not a claim about a foundation model's original pretraining.

Among models that generate complete textual proofs, the strongest observed result was **eight
lines**: the June offset-format expert-iteration run produced seven distinct dependency-pruned
eight-line certificates. Most later textual-proof procedures reached seven lines. The specialized
FLoP legal-action system is not directly comparable: full-start PPO reached eleven lines, behavior
cloning reached 65, and backward-reset curriculum PPO reached 129 on a narrow repeated-projection
challenge.

## Textual proof generators

| Study and procedure | Longest verified synthetic proof | Population and qualification |
| --- | ---: | --- |
| June offset SFT, before expert iteration | 7 | Synthetic target sampling; maximum written length |
| June offset expert iteration, three rounds | **8** | Seven distinct certificates; all remain eight lines after dependency pruning |
| R1 pair-1 direct RL | 7 | Synthetic training-rollout diagnostic, not held-out transfer |
| R1 pair-1 symbolic-CoT RL | 7 | Synthetic training-rollout diagnostic, not held-out transfer |
| Later Direct cap-6 RL | 7 | Raw training maximum; the above-cap certificate was on an SFT-exposed short theorem and there were no successes on the long-synthetic RL pool |
| Later symbolic-CoT cap-6 RL | 7 | 186 RL-only synthetic theorems had compressed certificates above six, all topping out at seven |
| OLMo-32B ICL | 7 | Held-out greedy pilot; same seven-step proof as every trained endpoint |
| OLMo-32B SFT | 7 | Held-out greedy pilot |
| OLMo-32B SFT+DPO | 7 | Held-out greedy pilot |
| OLMo-32B SFT+DPO+verifier RL | 7 | Held-out greedy pilot |
| Proof-fragment adapted start | 7 | Held-out sampled final population; best observed dependency-pruned proof |
| Proof-fragment selected/full RL | 7 | Held-out sampled final population |
| Proof-fragment selected/reset RL | 7 | Held-out sampled final population |
| Proof-fragment uniform/full RL | 7 | Held-out sampled final population |
| Proof-fragment uniform/reset RL | 7 | Held-out sampled final population |
| FLoP token LM, relative reward/full | 7 | Held-out sampled primary evaluation |
| FLoP token LM, relative reward/reset | 7 | Held-out sampled primary evaluation |
| FLoP token LM, PPO/full | 7 | Held-out sampled primary evaluation |
| FLoP token LM, PPO/reset | 7 | One seed reached seven; the other reached six |

The proof-fragment selector read losses on reference certificates as long as twelve lines, so that
study's information regime is more precisely “cap-six direct training with a cap-twelve selection
oracle.” Its supplied primary training and reset-source certificates remained capped at six.

## Specialized FLoP legal-action policies

These policies choose legal deductions while the environment stores and serializes the proof.
Their lengths should not be pooled with free-form token-generation results.

| Procedure trained from cap-six sources | Longest dependency-pruned proof | Qualification |
| --- | ---: | --- |
| Minimal full-start PPO | **11** | One sampled primary-final success in seed 4401; zero successes on the separate 33/65/129-line challenge |
| Minimal behavior cloning | 65 | Repeated atomic projection on the later depth challenge |
| Minimal backward-reset curriculum PPO | **129** | Repeated atomic projection; 1/16 and 2/16 challenge targets across the two seeds |

## Exact key witnesses

### Seven eight-line expert-iteration certificates

The [exact seven discovery records](2026-06-09-nd-bootstrapping-sprint/results/cap6_eight_line_witnesses.json)
were selected from the cap-6 round-three discovery buffer at source commit `e8766d2`. The copied
ledger records source SHA-256
`9a2f6d7b60c478b4eab5ab323fbd717aabb8ddec654643939585a2373749d164`.
The independent [post-hoc proof-length audit](2026-06-09-nd-bootstrapping-sprint/proof-length-audit.md)
replayed and reverified the records after dependency/reiteration pruning; all seven remain eight
lines.

### FLoP full-start PPO eleven-line certificate

The [exact eleven-line witness](2026-09-10-flop-adaptation/reports/witness_full_start_11line_v1.json)
has source ID `a27ab5a57cab223bf8f282ca6726be1464a7c038ce9efff47e9ec09072bf8d2d`.
It has 23 raw lines and eleven dependency-pruned lines. Its supplied reference witness has ten
lines. The file includes the theorem, raw certificate, reverified pruned certificate, rule counts,
source-analysis hash, and selection criterion.

### FLoP backward-reset 129-line certificate

The [exact 129-line witness](2026-09-10-flop-adaptation/reports/witness_length_challenge_v1.json)
has source ID `094705b1433304132cbb256317d13203386573db20ea603555a1e604d1ba3589`.
It has 130 raw lines and 129 compressed and dependency-pruned lines. The
[complete sub-1MB challenge analysis](2026-09-10-flop-adaptation/reports/analysis_length_challenge_v1.json)
records every policy, seed, challenge length, replay result, and source hash. This deepest success
is an atomic-projection theorem concluding the demonstrated atom `P`; compound-distractor cases
failed.

## Evidence and interpretation limits

- [June proof-length audit](2026-06-09-nd-bootstrapping-sprint/proof-length-audit.md)
- [Cap-6 Direct training-growth audit](2026-09-10-proof-cap-comparison/reports/cap6_direct_s20260910_training_growth.json)
- [Cap-6 symbolic-CoT training-growth audit](2026-09-10-proof-cap-comparison/reports/cap6_cot_s20260910_training_growth.json)
- [Proof-fragment per-model certificate-length ledger](2026-09-10-proof-fragment-selection/reports/final_certificate_lengths_v1.json)
- [Compact FLoP synthetic proof-length ledger](2026-09-10-flop-adaptation/reports/synthetic_proof_length_ledger_v1.json)
- [OLMo five-method report](2026-09-10-olmo-32b-pilot/icl-extension-20260910-1754/summary.md)
- [R1 training diagnostic](2026-09-08-r1-cot/support_strategy.md)

Every reported maximum is an observed verified certificate length, not a lower bound on theorem
difficulty. Longer certificates may contain avoidable steps. “Held out,” “training rollout,” and
“later challenge” are kept explicit because those populations support different generalization
claims.
