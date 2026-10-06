---
written_on: 2026-10-06
written_by: agent:codex
---

# Independent MAIN joint-live result audit

Accepted final aggregate SHA256 `7a21f093c5433348a99c424782875243a977b15313fe900e2e8d5b488d070614` following root's09:03:42UTC all-nine release. The original MAIN registry remains `bd289d62...`; the approved numerical-admission sidecar is `c41fba4e0dcd63feb26892184dc2e71aac21ed9c38ee610d9404d8738aefc313`. No model fitting, neural inference, protocol edits or historical outcome reads were performed during this audit.

## Evidence and checks

- [Main replay helper](audit_joint_live_main.py) and [results](joint_live_main_independent_audit.json): all104,800 integrated policy rows independently reconstructed from public path decisions, candidate work and private verdicts. Every categorical/greedy/fixed/count/context/feasible-mixture readout agrees. The independent scalar work reconstruction uses reached-prefix generator work and the appropriate learned versus terminal-only controller charges.
- All nine model inventories,400 theorems×two draws per model, equal primary three-seed weighting, per-seed point estimates and sample standard deviations, finite-path oracle tables, and every published paired contrast were checked. All1839 metric/interval checks reproduce with maximum relative discrepancy5.87e−15; point estimates reproduce exactly. Bootstrap uses the registered10,000 family-stratified theorem draws, seed72006.
- Independent basic-feasible-vertex enumeration verifies every registered DEV mixture's feasibility, matching constraints and reward optimum. Frozen weights are applied unchanged to TEST. Joint matching is infeasible for every cost checkpoint and for one no-cost checkpoint, so no primary arm-level joint mixture is available; the two feasible no-cost per-seed rows are retained without silently forming a partial-seed primary average.
- [Binding and exception helper](audit_joint_live_bindings_and_exceptions.py) and [results](joint_live_bindings_and_exceptions_audit.json):164 exact current/snapshot source, base, checkpoint and fit bindings pass. The original numerical guard correction remains admission-only: no clipping, renormalization or changed estimates.
- [Identity/verdict helper](audit_joint_live_readout_and_lean.py), [sample plan](joint_live_lean_sample_plan.json), and [results](joint_live_readout_and_lean_audit.json): direct Lean checks of36 prospectively hash-selected first/last public transcripts across nine models match all recorded verdicts. Sample theorem IDs follow the previously specified model-ID/theorem-ID/repeat0 hash rule, rather than outcome selection. This is a bounded spot-check, not another full verifier sweep.

All new helpers pass Ruff and syntax checks. Arithmetic and the36 Lean checks were actually executed; formatting does not constitute additional numerical evidence.

## Primary findings

| Three-seed arm | Native correctness | Attempts | Own fixed1 correctness | Total forward tokens |
|---|---:|---:|---:|---:|
| Cost0.03 | .8531942 |1.0213346|.8533333|2616.8254|
| Cost0 | .8397385 |2.3249369|.8433333|14666.3891|

Both learned policies are below their own fixed1 correctness and use more work. Cost0.03 minus fixed1 accuracy is −.01392 percentage points,95% interval[−.01695,−.01069]; its native reward difference is −.07792 points[−.08322,−.07294]. Cost0 minus fixed1 accuracy is −.35949 points[−.40479,−.31714]. Fixed1 requires2136.40 and2263.83 forward tokens respectively, versus learned2616.83 and14666.39. These paired small intervals describe stable within-theorem differences conditional on these trained models, not precision over a population of training runs.

The positive comparisons must also be reported. Cost0.03 beats its count-only rule by .01040 correctness points[.00901,.01198], and native reward by .01420 points[.00900,.01888], while consuming366.80 additional forward tokens. Cost0 beats count-only by .01871 correctness points[.01012,.02622], with .09644 more attempts and2805.95 more forward tokens. The latter difference is not consistent across all three seeds: no-cost seed103 is below its own count-only accuracy. These tiny average gains demonstrate a limited advantage over that specific count-only comparator, not useful allocation relative to the stronger and cheaper own fixed1 rule.

Against its own DEV-forward-work mixture, cost0.03 improves correctness by .11988 points[.10711,.13248] and native reward by .29031 points[.27568,.30441]. Its TEST work mismatch is −3.89 tokens[−19.78,12.94], so do not claim a significant work reduction or exact matching. No-cost is below its DEV-forward-work mixture by .06202 correctness points[−.08484,−.03503] while using836.97 more TEST forward tokens[665.11,1009.16]. Its attempt-matched mixture is also more accurate and cheaper on TEST. These are applications of DEV-frozen policies, not TEST-refitted efficiency frontiers.

## Generation, controls and seed variation

First-proof correctness improves from format-only .7275 to .8433333 for no-cost and .8533333 for cost, with paired gains11.58 points[9.21,14.13] and12.58 points[9.96,15.29]. This supports improved clean first-proof generation after the joint training procedure; it does not require a successful stopping mechanism. The fixed1 training control reaches .8325 and fixed4 control's native fixed4 readout reaches .8182902.

All six primary checkpoints and the fixed4 control were selected at200 updates. The fixed1 control was selected at100, despite completing the same maximum training budget. Report selected checkpoint age as well as completed training expense. Fixed controls each have one training seed, whereas primary arms average three; theorem-bootstrap comparisons do not establish replication stability of those controls.

No-cost attempts vary sharply across seeds:1.6950,1.1855 and4.0943. Their learned accuracies are .85799,.84958 and.81164. Cost attempts are1.0149,1.0168 and1.0323, with accuracies .85744,.84976 and.85238. This variation matters alongside the narrow conditional theorem intervals. The cost-versus-no-cost accuracy advantage1.34557 points[.01716,2.66023] combines differently trained generators and policies, and cannot isolate stopping alone.

## Greedy interpretation

Of38,400 saved primary prefixes,38,378 have the first proof as the unique argmax. There are22 later-proof argmax exceptions:13 for cost102,5 for cost103 and4 for no-cost102, all at counts4–8. Some substitute an invalid later proof for a valid first proof at very small native stopping mass. Thus native-stop/greedy-index correctness is approximately, not bitwise, equal to first-proof correctness.

All22 exceptions are unreachable under fully greedy stopping. On every visited path the fully greedy policy therefore selects the first proof; this narrower statement is directly supported by identities, not just aggregate accuracy. Fully greedy cost stops after one attempt, while fully greedy no-cost averages3.3925 attempts without changing selected-proof correctness. Extra generation can be wasted even when greedy selection avoids the categorical dilution seen in the intervention study. Untrained fixed1/format-only selector heads have exact index ties and must not be described as learned first-proof preferences.

## Claim boundary

The study demonstrates training-related generation improvement and small paired advantages over particular count-only/work-matched policies. It does not demonstrate useful state-sensitive stopping against its strongest same-checkpoint baseline: stopping after the first proof is more accurate and cheaper for both native primary policies. This is not a claim of no learning, no state dependence, or universally harmful additional reasoning. Frozen-pool recognition and the causal categorical-selection findings remain separate experiments.

Model-section timing excludes setup, parser, checker and diagnostics; foreign-GPU colocation further precludes isolated deployment-latency claims. Expected deployment work differs from the expense of generating all forced paths for measurement. Work accounting and timing scopes should remain explicit.
