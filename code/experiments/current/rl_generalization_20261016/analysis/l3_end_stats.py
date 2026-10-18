"""End-of-L3 training statistics per run: means over the last 5 logged iterations (DESIGN_REVIEW.md E5).

    python3 l3_end_stats.py

Training pass@1, think / action entropy (at the training temperature), think-ban mass, share of groups skipped,
think tokens per attempt, and the final rollback count and lr scale. CPU only.
"""

import json
import statistics as st
from pathlib import Path

ROOT = Path('/scratch/cowifornia/nd-rl/think/runs/grpo_v2')
RUNS = {
    'mb_s0': 'grpo_think_vs_mb_s0_L3', 'mb_s1': 'grpo_think_vs_mb_s1_L3', 'mb_s2': 'grpo_think_vs_mb_s2_L3',
    'kkl0_s0': 'grpo_think_vs_mb_kkl0_s0_L3', 'kkl0_s1': 'grpo_think_vs_mb_kkl0_s1_L3',
    'kkl0_s2': 'grpo_think_vs_mb_kkl0_s2_L3',
    'krr': 'grpo_think_vs_mb_krr_s0_L3', 'kkl1': 'grpo_think_vs_mb_kkl1_s0_L3', 'kg64': 'grpo_think_vs_mb_kg64_s0_L3',
    'kch4': 'grpo_think_vs_mb_kch4_s0_L3', 'kx66': 'grpo_think_vs_mb_kx66_s0_L3', 'kp3': 'grpo_think_vs_mb_kp3_s0_L3',
    'kt12': 'grpo_think_vs_mb_kt12_s0_L3', 'ki78': 'grpo_think_vs_mb_ki78_s0_L3',
}

for k, d in RUNS.items():
    rows = [json.loads(l) for l in open(ROOT / d / 'metrics.jsonl')]
    last = {}
    for x in rows:  # a rolled-back iteration can appear twice; keep the last row
        last[x['iter']] = x
    its = sorted(last)
    final = last[its[-1]]
    tail = [last[i] for i in its[-5:]]

    def m(key):
        return st.mean(t.get(key, 0) or 0 for t in tail)

    print(f"{k:8s} lastit {its[-1]} pass {m('reward_pass1'):.3f} Hth {m('entropy_think'):.3f} "
          f"Hact {m('entropy_action'):.3f} ban {m('ban_think'):.5f} skip {m('groups_skipped'):.2f} "
          f"think {m('think_per_attempt'):.0f} rb {final.get('rollbacks')} lr_scale {final.get('lr_scale')}")
