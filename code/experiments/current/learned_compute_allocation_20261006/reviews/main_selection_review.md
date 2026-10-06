---
written_on: 2026-10-06
written_by: agent:codex
---

# MAIN learned-selection audit

Accepted `analysis/final_main/selection`. The [independent audit](audit_main_selection.py) and [results](main_selection_independent_audit.json) verify all 27 trained prediction/config/checkpoint/source bindings against the sealed registry and inventory, including objective, architecture, optimizer-seed identity and evaluation permutation. All 10,800 theorem readouts have finite canonical probabilities, matching argmax tie sets and exact selected-proof identities/labels. Primary three-seed theorem means, paired 5,000-draw family-bootstrap intervals, supplementary order means and the all-complete mixed-validity stratum reproduce with maximum discrepancy 1.11e-16. No neural inference or tuning was performed.

| Method | Primary selected accuracy | Difference from public features | Paired 95% interval |
|---|---:|---:|---:|
| Pooled, primary | 0.816667 | −0.058333 | [−0.077500, −0.040000] |
| Factorized marginal, control | 0.861250 | −0.013750 | [−0.024583, −0.003333] |
| Binary critic, secondary | 0.847917 | −0.027083 | [−0.044167, −0.010833] |

Public features achieve 0.875, random selection 0.7159375, and exact-pool oracle availability 0.895. Pooled-minus-random is +0.100729, interval [+0.079268, +0.122812]. All declared trained methods remain below the strong feature comparator; the proposal's strongest-baseline usefulness criterion is not met. Do not substitute the better factorized result for the pooled primary endpoint.

Separately checked the registered primary-order untrained readouts and their source hashes: pooled 0.6975; factorized and binary-factorized readouts 0.76375. [Saved untrained-readout checks](main_selection_untrained_readout_audit.json) bind their prediction bytes. These establish lower untrained point estimates on the same benchmark/interface. They support distinguishing training improvement from failure to beat the strongest comparator; this review did not add an unregistered paired significance test for trained-minus-untrained.

There are 67 pools with eight syntactically complete candidates but mixed semantic validity. Their random accuracy is 0.766791; pooled, factorized and BCE means are 0.825871, 0.920398 and 0.900498, while features reach 0.955224. Completion discrimination alone cannot explain performance within this stratum. The learned models' above-random point estimates do not show an absence of semantic signal, while their remaining deficit to features prevents claiming superior semantic recognition. Features include likelihood/entropy and other public signals, not just completion. This stratum is a descriptive breakdown, not a new independently selected headline or significance test.

The pooled primary has one registered overflow fallback per seed, retained in all-theorem results. Pooled three-order means range 0.811667–0.822917; factorized and BCE accuracies are identical across the three declared orders. These controls narrow order sensitivity but do not eliminate the pooled model's new long-context/interface confound.

The saved two-panel figure is readable, has a zero-based accuracy axis, clearly identifies the pooled primary and controls, and faithfully shows negative paired differences. Add a final-report caption specifying 95% theorem bootstrap within the two families, averaging three fixed trained checkpoints within each theorem and conditioning on saved pools/models. Oracle availability is an upper bound, not a deployed baseline. Historical outcomes were not read. Ruff passes for the new audit helper.
