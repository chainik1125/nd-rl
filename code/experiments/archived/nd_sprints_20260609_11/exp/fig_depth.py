"""Depth sweep figure: capability and frontier vs number of layers.

Usage: python fig_depth.py <out.png> <d:histjson:discjsonl> ...
Each arg: layers : path-to-ei-history.json : path-to-round3-discoveries.jsonl
(discoveries path '-' if none).
"""
import json
import sys
from collections import Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main(out, *specs):
    rows = []
    for spec in specs:
        d, hist_p, disc_p = spec.split(':', 2)
        d = int(d)
        h = json.load(open(hist_p))
        r0 = h[0]
        rN = h[min(3, len(h) - 1)]
        mx, robust = 0, 0
        if disc_p != '-':
            lens = [json.loads(l)['n_lines'] for l in open(disc_p) if l.strip()]
            if lens:
                c = Counter(lens)
                mx = max(c)
                robust = max((k for k in c
                              if sum(v for kk, v in c.items() if kk >= k) >= 5),
                             default=mx)
        rows.append((d, r0.get('iid_pass1', 0), r0['transfer_passk'],
                     rN['transfer_passk'], mx, robust,
                     rN.get('buffer_size', 0)))
    rows.sort()
    ds = [r[0] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    axes[0].plot(ds, [r[1] for r in rows], 'o-', color='tab:gray',
                 label='in-distribution accuracy (SFT, before RL)')
    axes[0].plot(ds, [r[2] for r in rows], 's-', color='tab:blue',
                 label='transfer pass@8 (SFT, before RL)')
    axes[0].plot(ds, [r[3] for r in rows], '^-', color='tab:orange',
                 label='transfer pass@8 after 3 hindsight-EI rounds')
    axes[0].set_xlabel('layers (width fixed at d=256, same data/steps)')
    axes[0].set_ylabel('solve rate')
    axes[0].set_xscale('log', base=2)
    axes[0].set_xticks(ds)
    axes[0].set_xticklabels([str(d) for d in ds])
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[0].set_title('Capability vs depth', fontsize=11)

    axes[1].axhline(8, color='k', ls='--', alpha=0.6)
    axes[1].text(ds[0], 8.15, 'training cap (8 lines)', fontsize=8)
    axes[1].axhline(12, color='tab:red', ls=':', alpha=0.6)
    axes[1].text(ds[0], 12.15, 'universal 4-layer ceiling (12)', fontsize=8,
                 color='tab:red')
    axes[1].plot(ds, [r[4] for r in rows], 'o-', color='tab:purple',
                 label='longest discovered proof (3 rounds)')
    axes[1].plot(ds, [r[5] for r in rows], 's--', color='tab:purple',
                 alpha=0.5, label='robust max (>=5 theorems)')
    axes[1].set_xlabel('layers')
    axes[1].set_ylabel('discovered proof length (lines)')
    axes[1].set_xscale('log', base=2)
    axes[1].set_xticks(ds)
    axes[1].set_xticklabels([str(d) for d in ds])
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    axes[1].set_title('Frontier vs depth', fontsize=11)

    fig.suptitle('Depth sweep: same cap-8 data, same hindsight-EI protocol',
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    print('wrote', out)


if __name__ == '__main__':
    main(*sys.argv[1:])
