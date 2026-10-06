---
written_on: 2026-10-06
written_by: agent:codex
status: preliminary_review
---

# Frozen-pool threshold baseline audit

## Mathematical validation

The exact expectations in `fixed_pool_policy` are correct. With `m > 0` candidates meeting the threshold among `n`, each qualifying candidate is selected with probability `1/m`. Its inclusion probability is also `1/m`; a nonqualifier precedes all qualifiers with probability `1/(m+1)`. Expected attempts are `(n+1)/(m+1)`. If none qualify, the policy sees the entire pool and chooses among the highest-scoring ties. Fixed-count selection averages uniformly over all subsets and uniformly over tied maxima.

An independent enumerator checked **891 cases**: all 81 four-candidate score vectors over `{0, 0.5, 1}`, each under four fixed counts and seven thresholds, averaging all 24 orderings explicitly. Maximum discrepancy in inclusion probabilities, selection probabilities, or expected attempts was `1.78e-15`. The existing ten selection-metric tests also passed. This validation used synthetic data only.

## Information and evaluation boundaries

The ten features in `analyze_candidates.py` use public generation statistics and deterministic syntax replay. They do not inspect Lean labels. Private candidate outcomes are joined by candidate ID for supervised feature fitting or final scoring. Feature fitting enforces training-only pools; feature mean/scale are estimated there. Evaluation rejects theorem overlap with the feature-training IDs. Development-only threshold calibration chooses reward-maximizing thresholds separately for each declared attempt cost, with explicit tie handling.

Frozen-policy evaluation verifies the feature-model hash and retains development-selected choices. However, it computes and writes metrics for **all** threshold candidates. A headline test comparison must use the saved `choices[cost]`, never the best test threshold. Other points may appear as predeclared descriptive curves, clearly distinguished from a deployable selected policy. The written-protocol guard checks file existence; substantive adherence still requires the frozen document and human-readable report to agree with the actual settings.

Recommended provenance hardening: bind the feature-name order/schema and evaluation implementation to the frozen policy, in addition to weights and data hashes. Evaluator source snapshots preserve auditability but do not themselves prevent a changed feature transform between calibration and final evaluation.

## Compute and scope

Expected additive generation tokens and generator state-prefix tokens follow directly from candidate inclusion probabilities. They are valid accounting proxies for independently generated candidates. They are not measured GPU time, batching cost, or FLOPs. **The existing feature critic requires the collector's likelihood/entropy replay**, so that replay must be charged in its total inference work. Calling it excluded diagnostic work is incorrect for this comparator. Cached-logit extraction could avoid a second model pass, but is a different, unvalidated numerical implementation. See the [required-replay audit](feature_required_replay_audit.md) for measured development counters and the distinction between unpadded proxies and executed batching costs.

These are finite-pool, independent-generation policies. They do not estimate what a history-conditioned generator would produce after a different stopping trajectory. Compare them to the frozen-pool controller directly; compare to joint live generation only with the protocol and compute differences disclosed.

## Evidence

[Independent enumeration](threshold_math_independent_audit.json), [reproduction script](audit_threshold_math.py), `code/shared_code/selection_metrics.py`, `analysis/threshold_baselines.py`, and `analysis/analyze_candidates.py`. No final-test outcomes were opened.
