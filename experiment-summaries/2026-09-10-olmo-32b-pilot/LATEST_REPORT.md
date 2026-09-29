# Completed five-method OLMo-32B pilot

Read the [final five-method report](icl-extension-20260910-1754/summary.md), finalized 2026-09-10 after the authorized ICL completion.

All five endpoints have 24 matched greedy attempts. ICL, SFT, SFT+DPO and SFT+DPO+RL solve the same single longer target; short counts are 3/8, 4/8, 2/8 and 5/8, respectively. Zero-shot scores 0/24 with severe output-protocol confounding. The small panel does not establish a reliable training advantage. The original 120-target and sampled-K4 evaluations remain unperformed.

The [machine-readable handoff](icl-extension-20260910-1754/reports/final_status_v2.json) links results, checkpoints, tests, independent audits and resource accounting. No sprint GPU job remains active. The earlier [root summary](summary.md) is preserved as the historical trained-subset handoff, not the latest result.
