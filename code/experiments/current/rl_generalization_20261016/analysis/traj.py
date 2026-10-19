"""Training trajectories of GRPO v2 runs from their metrics.jsonl (DESIGN_REVIEW.md E4, E5).

    python3 traj.py grpo_think_vs_s0_L2 grpo_think_vs_mb_krr_s0_L3 ...
    python3 traj.py --late grpo_think_vs_mb_ki78_s0_L3      # iterations 40-78 of the 78-iteration run

Prints per selected iteration: training pass@1 (all, main pool, 3-premise extra pool, per budget), the share of
groups skipped (no outcome signal), entropy (all / think / action, at the training temperature), think and total
tokens per attempt, the KL anchor value, the lr and the rollback count. A rolled-back iteration number can appear
twice in metrics.jsonl; the last row is used. CPU only.
"""

import json
import sys
from pathlib import Path

ROOT = Path('/scratch/cowifornia/nd-rl/think/runs/grpo_v2')
NAN = float('nan')


def main(argv):
    late = '--late' in argv
    runs = [a for a in argv if a != '--late']
    its = (40, 45, 50, 55, 60, 65, 70, 75, 78) if late else (1, 2, 5, 10, 15, 20, 25, 30, 35, 39)
    for r in runs:
        rows = [json.loads(l) for l in open(ROOT / r / 'metrics.jsonl')]
        print('==', r, len(rows))
        for it in its:
            rs = [x for x in rows if x['iter'] == it]
            if not rs:
                continue
            x = rs[-1]
            if late:
                print(it, round(x['reward_pass1'], 3), 'H', round(x['entropy'], 4), round(x['entropy_think'], 4),
                      'think', round(x['think_per_attempt']), 'lr', x['lr'], 'rb', x.get('rollbacks'),
                      'ban', x.get('ban_think'))
                continue
            pb = x.get('pass1_by_budget', {})
            print(f"it{it:>2} pass {x.get('reward_pass1', 0):.3f} main {x.get('main_pass1', NAN):.3f} "
                  f"extra {x.get('extra_pass1', NAN):.3f} B0 {pb.get('0', NAN):.3f} B64 {pb.get('64', NAN):.3f} "
                  f"B256 {pb.get('256', NAN):.3f} skip {x.get('groups_skipped', 0):.2f} "
                  f"H {x.get('entropy', 0):.4f} Hth {x.get('entropy_think', 0):.4f} "
                  f"Hact {x.get('entropy_action', 0):.4f} think {x.get('think_per_attempt', 0):.0f} "
                  f"tok {x.get('tokens_per_attempt', 0):.0f} kl {x.get('kl_anchor', 0):.5f} lr {x.get('lr', 0):.2e} "
                  f"rb {x.get('rollbacks', 0)} seen {x.get('curriculum', {}).get('seen')}")


if __name__ == '__main__':
    main(sys.argv[1:])
