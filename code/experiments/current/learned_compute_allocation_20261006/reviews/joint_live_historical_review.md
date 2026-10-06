---
written_on: 2026-10-06
written_by: agent:codex
---

# Independent historical joint-live audit

Accepted aggregate SHA256 `2bf1f8f55513dc71fbf706f8d2b9e84a2ce3405eb5067edeedb66a66e86f9309` after explicit root release09:28:05UTC. Original historical registry `d228e09e...` and approved numerical sidecar `c41fba4e...` remain unchanged. No model fitting, inference, policy recalibration or new endpoint was introduced.

[Raw/aggregate replay](audit_joint_live_historical.py) and [results](joint_live_historical_independent_audit.json) independently reconstruct18,864 policy rows for all nine models, every registered categorical/greedy/fixed/count/context/feasible-mixture readout, all per-seed means, and4953 aggregate metric/interval checks. Combined72, textbook_dev58 and rl_train14 cohorts reproduce; maximum relative interval discrepancy6.33e−16 and point discrepancy0. The bootstrap preserves observed role counts, uses10,000 draws/seed102006, and averages two paths then primary three seeds within theorem.

[Binding helper](audit_historical_live_bindings.py) and [evidence](historical_live_bindings_audit.json) verify164 source/snapshot/model/fit bindings, every released launch-ledger hash, and exact equality of historical versus MAIN model/inference/DEV-policy identities. Every mixture retains its MAIN-DEV feasibility and reward-optimal weights; independent vertex enumeration verifies those constraints. No historical work match was refitted. The fixed1 control remains selected at100 updates, primary/fixed4 at200, and controls have one seed. All helpers pass Ruff and syntax checks.

## Combined results and comparison limits

Cost0.03 native correctness is .41402709 at1.0388868 attempts; no-cost is .40951014 at2.3719609 attempts. Both arms' own fixed1 correctness is .41435185. Learned-minus-first accuracy is −.03248 percentage points[−.04381,−.02165] for cost and −.48417 points[−.67779,−.31684] for no-cost. Required forward work increases389.39 tokens[339.13,443.55] and6126.91[5082.87,7302.00]. Thus the own-first baseline remains stronger and cheaper overall.

Cost versus count-only accuracy is −.00280 points with an interval crossing zero[−.00660,.00140]; native reward is lower by .05165 points[−.06183,−.04206]. No-cost is below count-only accuracy by .09516 points[−.15932,−.04908], and uses1342.16 more forward tokens. This historical readout does not reproduce the small positive MAIN count-only accuracy contrasts.

Cost exceeds its frozen DEV-forward-work mixture in correctness by .11040 points[.07949,.14237] and native reward by .22818 points[.19115,.26607], but uses144.66 more historical tokens[109.94,179.73]. It is not an equal-work gain. No-cost is below its analogous mixture by .13501 correctness points[−.19062,−.08396], while using447.87 more tokens[165.09,683.87]. Report these historical budget mismatches rather than importing the original DEV match claim.

## Roles, controls and interpretation

The58-textbook_dev cohort has cost/no-cost native correctness .4450446/.4399411, versus .4454023 own first; the14-rl_train cohort has .2855259/.2834390, versus .2857143 own first. Small role-specific intervals and differences are descriptive conditional-model comparisons; the cost-minus-first accuracy interval in the14-theorem cohort barely includes zero. Avoid saying every role-specific accuracy contrast is statistically negative merely because the pooled contrast is.

Format-only first-proof accuracy is .41666667. Each primary arm's first-proof contrast versus format-only is −.00231481 with broad intervals crossing zero (no-cost[−.03009,.02315], cost[−.03241,.02778]). The substantial fresh-MAIN generation improvement therefore does not establish historical transfer improvement. Cost-minus-no-cost overall correctness .00451695 has interval[−.00822789,.01710992], and both learned-versus-fixed4-control accuracy intervals cross zero. No historical superiority, equivalence or broad training-seed stability follows.

Historical roles identify original source use, not verified r16 training membership. The benchmark has known reuse and at least one canonical pretraining exposure. Same bounded MAIN caps and eight-attempt paths differ from upstream published k256/max_action512/max_steps96 curves. These results are descriptive retention/transfer measurements, not a new fresh-generalization benchmark or a reproduction of the original oracle curves. Expected deployment work remains distinct from forced-path measurement expense and wall time.

No additional historical Lean sweep was required: the completed MAIN audit already performed36 model-specific direct transcript checks, and this audit independently recomputes all historical policy arithmetic from saved verdicts without claiming those verdicts were newly rechecked.
