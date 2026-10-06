#!/usr/bin/env python3
"""Main figure of the campaign: pattern acquisition after R rounds vs pretraining pattern frequency f.

  python phase2_figure.py --metrics artifacts/p2/metrics_derived_ore.json artifacts/p2/metrics_reductio.json artifacts/p2/metrics_depth3.json \
      --report data/p2/assemble_report.json --out figures/phase2_acquisition.png

Arm names: ei_<pattern>_f<f>_s<seed> / frozen_<pattern>_f<f>_s<seed>. f = 0 is plotted at the left edge of a
log axis (position F0). Seed points shown; line through the seed mean; frozen control dashed.
"""
import argparse, json, re, collections
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COL = {'derived_ore': '#e74c3c', 'reductio': '#2e86c1', 'depth3': '#27ae60'}
LABEL = {'derived_ore': 'P1 derived-ORE', 'reductio': 'P2 reductio', 'depth3': 'P3 depth-3'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--metrics', nargs='+', required=True)
    ap.add_argument('--report', default=None, help='assemble_report.json for achieved frequencies (x positions)')
    ap.add_argument('--out', default='figures/phase2_acquisition.png')
    ap.add_argument('--round', type=int, default=8)
    ap.add_argument('--pool', default='targets', choices=['targets', 'transfer'])
    a = ap.parse_args()
    rep = json.load(open(a.report)) if a.report else {}
    pts = collections.defaultdict(list)   # (pattern, kind) -> list of (f, acq, seed, n_proofs)
    for fn in a.metrics:
        m = json.load(open(fn))
        for arm, rounds in m.items():
            mm = re.match(r'(ei|frozen)_(.+)_f([0-9.e-]+)_s(\d+)', arm)
            if not mm:
                continue
            kind, pat, f, seed = mm.group(1), mm.group(2), float(mm.group(3)), int(mm.group(4))
            rr = [r for r in rounds if r['round'] <= a.round]
            if not rr:
                continue
            last = rr[-1]
            tag = f'{pat}_f{f:g}'
            f_ach = rep.get(tag, {}).get('achieved_freq', {}).get(pat, f)
            pts[(pat, kind)].append((f, f_ach, last[f'acq_{a.pool}'], seed, last[f'n_pattern_proofs_{a.pool}'], last['round']))
    F0 = 3e-5   # left-edge position for f = 0
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for (pat, kind), rows in sorted(pts.items()):
        xs = [F0 if r[0] == 0 else r[1] for r in rows]
        ys = [r[2] for r in rows]
        by = collections.defaultdict(list)
        for x, y in zip(xs, ys):
            by[x].append(y)
        mx = sorted(by); my = [np.mean(by[x]) for x in mx]
        ax.plot(mx, my, '-' if kind == 'ei' else '--', color=COL[pat], lw=2 if kind == 'ei' else 1.2,
                label=f'{LABEL[pat]} ({"EI" if kind == "ei" else "frozen control"})')
        ax.scatter(xs, ys, color=COL[pat], s=22 if kind == 'ei' else 12, marker='o' if kind == 'ei' else 'x', zorder=3)
    ax.set_xscale('log')
    ax.set_xlim(F0 * 0.7, 0.3)
    ax.set_xticks([F0, 1e-4, 1e-3, 1e-2, 1e-1]); ax.set_xticklabels(['0', '1e-4', '1e-3', '1e-2', '1e-1'])
    ax.axvline(F0 * 1.6, color='gray', lw=0.6, ls=':')
    ax.set_xlabel('pattern frequency f in the 155k pretraining set (f = 0 at the left edge)')
    ax.set_ylabel(f'acquisition after round {a.round}: fraction of {a.pool} theorems\nsolved by a proof that CONTAINS the pattern')
    ax.set_ylim(-0.02, 1.0)
    ax.legend(fontsize=8, loc='upper left')
    ax.set_title('RL (expert iteration, k=32 × 8 rounds) vs pretraining coverage of the pattern', fontsize=10)
    plt.tight_layout(); plt.savefig(a.out, dpi=140); plt.close()
    # table
    print('| pattern | kind | f (target) | f (achieved) | seed | acquisition | pattern proofs | round |')
    print('|---|---|---:|---:|---:|---:|---:|---:|')
    for (pat, kind), rows in sorted(pts.items()):
        for r in sorted(rows):
            print(f'| {pat} | {kind} | {r[0]:g} | {r[1]:.5f} | {r[3]} | {r[2]:.3f} | {r[4]} | {r[5]} |')


if __name__ == '__main__':
    main()
