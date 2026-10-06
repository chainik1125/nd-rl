---
written_on: 2026-10-06
written_by: agent:codex
---

# Independent historical frozen-head secondary review

Accepted the unaffected original registered frozen-head paired and threshold outputs. [Replay helper](audit_historical_frozen_head.py) and [results](historical_frozen_head_independent_audit.json) check original head/base/source/prediction/DEV-policy bindings, unchanged generator parameter digest, CPU linear scoring of cached vectors, role-stratified paired intervals, and all3312 exact staged policy rows. Cached-logit discrepancy is at most7.11e−15; maximum absolute policy arithmetic error1.16e−10 is in large work proxies. No backbone inference or fitting was rerun.

Head accuracy is41/72=.56944444. Against matched BCE seed0 the difference is .01388889 with95% interval[0,.04166667]; versus the feature model it is .01388889 [−.02777778,.06944444]. The separately labeled comparison to the three-BCE-seed mean is .02314815 [0,.06018519]. Intervals touching zero do not establish strictly positive superiority, and none of these comparisons establishes equivalence. All pooled estimates preserve58 textbook_dev/14 rl_train weighting and historical exposure limitations.

The original development threshold0.65 remains unchanged. At cost0.03 it yields .565625 correctness,4.32331349 attempts,11162.23342 logical input tokens and .46592560 reward. Its required critic scoring is1592.09466 tokens; generation contributes9570.13876 and no confidence replay is charged. These point estimates are not an equal-work comparison to another scorer. Padding, deployment overhead and isolated latency remain outside the logical lower bound.

The frozen original trunk/LM parameter digest remains `594173e7d67a64f5c4ffa9ac5176e4390770e44ce7d307590474bed3957c8eb5`. This supports preserved generation parameters under an unchanged interface, not universal immunity to new prompting interfaces or a full generation diagnostic for a fine-tuned critic. The separate historical selector role correction does not alter this method or its original analysis source.
