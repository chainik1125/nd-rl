---
written_on: 2026-10-06
written_by: agent:codex
status: accepted_with_wording_corrections
---

# Final reader-report adversarial review

## Summary

The report's three claims accurately separate independent-pool recognition and confidence thresholds, unsuccessful joint retry allocation, and harmful native categorical continuation on the registered forced-state distribution. Headline estimates and intervals agree with the completed independent audits. No model work, new statistical analysis, or frozen-source change was performed for this review.

Reviewed [reader report](../../../../../experiment-summaries/2026-10-06-learned-compute-allocation/summary.md), SHA256 `e8dd2e7ede1a8d5f84f454fc5d88b68c8ae04cbd6640140f618f1ab08028bd7b`, on branch `dmitry/main`, starting commit `315f36266bb4dd7feeff87228cb2662f8a36badc`. The report remains `final_review`; this is a context-aware adversarial review, not an independent fresh-context reading.

## Actionable wording corrections

1. Replace “analyses were frozen before final inference” with “original analysis plans were frozen before final inference,” explicitly referring to the two disclosed implementation amendments below. The estimands and original plans were frozen, but the numerical-domain and historical-role successor implementations were approved later. The later disclosure is accurate; the earlier blanket statement should agree with it.
2. Add the boundary-only control when explaining the initial history-interface collapse: three boundary tokens alone yielded 10/192 valid proofs, whereas three EOS tokens yielded 177/192, against 180/192 fresh and 5/192 with one prior attempt. The original collapse therefore does not require previous proof content or a long history. The separately reported held-out memoryless intervention still supports a limitation of the deployed history interface, without identifying a unique internal mechanism.
3. Describe historical syntax-plus-likelihood as the highest observed accuracy or point estimate, rather than saying it “beats all learned selectors.” Its 59.72% is indeed highest, but the sentence should not imply that every pairwise superiority claim was established by a reported interval.

Optional attribution clarification: the historical 576-proof direct Lean check was executed by the selection researcher; the reviewer independently audited its measurement/provenance and historical calculations. The reviewer directly reran all 3,200 MAIN independent proofs and the prespecified 36 live proofs. The current passive wording is true, but separating the actors would avoid reading the preceding “Astra independently” as applying to every direct checker invocation.

## Evidence and accepted scope

- [MAIN selection review](main_selection_review.md), [staged review](main_staged_aggregate_review.md), and [frozen-head review](main_frozen_head_review.md): the strongest MAIN feature comparator, neural deficits, own-scorer development-fitted mixtures, TEST work residuals, and lack of a noninferiority result are represented correctly. Staged confidence thresholds are not presented as jointly learned stopping or meta-learning.
- [MAIN joint review](joint_live_main_review.md): all headline accuracies, attempts, work, tiny favorable count/mixture contrasts, and stronger same-model first-proof comparison are accurate. Controls' one seed and selected update ages are disclosed. Greedy visited-path first-proof selection is correctly distinguished from the 22 unreachable prefix exceptions and native categorical probability spreading.
- [Causal review](causal_test_aggregate_review.md): the primary equal count/checkpoint weights, old/new valid mass decomposition, oracle increase, memoryless intervention and conditional uncertainty are accurately scoped. Acceptance associations remain descriptive; conditioning on verifier-derived incumbent validity is an analysis stratification, not deployed information. No internal-representation damage or greedy harm is claimed.
- [Historical selection](historical_selection_review.md), [staged](historical_staged_review.md), and [joint](joint_live_historical_review.md) reviews: role reuse, unchanged choices, unequal transferred work, and absence of established historical generator improvement are disclosed. The 58/14 source roles are not misrepresented as known exposure counts.
- [Resource accounting](../allocation/resource_accounting_review.md): the new table matches the audited inventories and preserves different timing scopes. The 102 processes and 12.31 summed process hours are not called kernel-active GPU hours, exact total sprint work, or deployment latency. The table avoids double-counting the overlapping final allocation ledger.
- [Next-study review](final_next_experiment_review.md): the exact frozen-head DEV reranking diagnostic remains proposed, related full-critic pilots are acknowledged, and the subsequent small-controller study is distinct from LM joint learning or task-level adaptation. The idealized iid reservation-threshold argument is appropriately limited to known payoffs and positive remaining horizon.

The report appropriately retains the narrow skeleton split, training/interface limitations, differing selection supervision/context burdens, omitted proposal mechanisms, analysis/operational amendments, unavailable clean-machine replication, and unrun next study. Improving selection relative to random or an untrained readout and failing the strongest-baseline usefulness criterion remain distinct claims.

## Checks and next step

This review compared prose with already audited artifacts and the resource note; it did not rerun inference, bootstrap estimates, or regression tests. The three wording changes above are sufficient; no additional experimental endpoint is requested. Final status should change only when root finishes its reserved writing and artifact checks.

## Terminal-probability wording check

The final source-only completeness check accepts reporting 5.73% compulsory-terminal probability for correctness-only evaluation and below 0.001% for cost-aware evaluation. In `evaluate_live_path.integrate`, a prefix is compulsory if its saved decision is forced or it is the final recorded prefix; its hazard becomes one and all surviving reach mass contributes to `forced_termination_probability`. Collection sets `fixed_count=max_attempts`. In `train_live.episode`, this path ends at the attempt cap or when context reservation disallows another attempt. “Protocol-imposed terminal prefix (attempt cap or context limit)” is therefore the clearest phrase, although “hard limit” is also correct for these complete paths.

This mass reaches the limit instead of demonstrating an earlier learned stop. It is not the fraction that would necessarily continue if the limit were removed: the integration assigns all reach mass to compulsory termination, including any voluntary acceptance the unmasked head might otherwise have proposed there. No new statistic was calculated for this wording check.
