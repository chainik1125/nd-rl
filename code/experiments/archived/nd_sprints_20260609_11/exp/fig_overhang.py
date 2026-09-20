"""Overhang curve: training cap vs discovered-proof-length frontier.

For each cap-L chain (offset format), measure from the discovery buffers:
- round-1 frontier: lengths discovered by sampling the ROUND-0 (SFT-only)
  model — the SFT overhang;
- round-3 frontier: lengths in the buffer after two retraining rounds — the
  RL overhang (3 rounds for caps 4/6; round-3 buffer used for L8 too, for
  comparability).
Frontier metrics: max length, and "robust max" = longest length with >=5
discovered theorems.

Usage: python -m exp.fig_overhang --runs '4:/path/ei_cap4,6:/path/ei_cap6,8:/path/ei_rl_off' --out fig_overhang.png
"""
import argparse
import json
import os
from collections import Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_lens(run_dir, rnd):
    p = os.path.join(run_dir, f'round{rnd}', 'discoveries.jsonl')
    if not os.path.exists(p):
        return None
    lens = []
    with open(p) as f:
        for line in f:
            if line.strip():
                lens.append(json.loads(line)['n_lines'])
    return Counter(lens)


def frontier(c):
    if not c:
        return None, None
    mx = max(c)
    robust = max((k for k in c if sum(v for kk, v in c.items() if kk >= k) >= 5),
                 default=mx)
    return mx, robust


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', required=True,
                    help='comma list cap:run_dir')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    rows = []
    for spec in args.runs.split(','):
        cap, run_dir = spec.split(':', 1)
        cap = int(cap)
        c1 = load_lens(run_dir, 1)
        c3 = load_lens(run_dir, 3)
        m1, r1 = frontier(c1)
        m3, r3 = frontier(c3)
        rows.append((cap, m1, r1, m3, r3, c1, c3))
        print(f'cap {cap}: round-1 max {m1} robust {r1} | round-3 max {m3} '
              f'robust {r3}')
        print(f'  r1 tail: {dict(sorted((k, v) for k, v in (c1 or {}).items() if k > cap))}')
        print(f'  r3 tail: {dict(sorted((k, v) for k, v in (c3 or {}).items() if k > cap))}')

    caps = [r[0] for r in rows]
    plt.figure(figsize=(6.5, 4.5))
    plt.plot(caps, caps, 'k--', alpha=0.5, label='training cap (y = x)')
    plt.plot(caps, [r[2] for r in rows], 'o-', color='tab:blue',
             label='SFT-only frontier (round-1 finds, robust max)')
    plt.plot(caps, [r[4] for r in rows], 's-', color='tab:orange',
             label='after RL rounds (round-3 buffer, robust max)')
    plt.plot(caps, [r[1] for r in rows], 'o', color='tab:blue', alpha=0.3)
    plt.plot(caps, [r[3] for r in rows], 's', color='tab:orange', alpha=0.3,
             label='faint markers: absolute max')
    plt.xlabel('training length cap L (lines)')
    plt.ylabel('longest discovered proof (lines)')
    plt.title('The overhang: how far past its training cap\n'
              'can the model prove (offset format)', fontsize=11)
    plt.xticks(caps)
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out, dpi=160)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
