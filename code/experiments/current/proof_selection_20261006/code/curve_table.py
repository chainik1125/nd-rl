#!/usr/bin/env python3
"""Curve readings (NOT dev-selected operating points): accuracy of each policy interpolated at exact mean attempts
m in {1.5, 2, 4, 8, 16} on one pool's stopping curves.  Adaptive policies use the upper concave frontier of all
(N_max in {2,3,4,6,8,12,16,24,32,48,64}) x (tau) points on THIS pool, i.e. an optimistic best-of-family reading achievable by
randomising between two policies, the completion-stop policy the N_max sweep at tau = 1 (its reachable range ends at its ceiling:
cells beyond it print '-' plus the ceiling), fixed-k uses the adjacent-k mixture.  Also prints the paired bootstrap
of (adaptive - fixed-k same selector) at the nearest grid point from stopping/<stem>__<scorer>.json.

Usage: curve_table.py --prefix ~/runs/proof-selection/stopping/<stem> --scorers completion_loglik,... [--md out.md]
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stopping import fixed_at

MS = (1.5, 2, 4, 8, 16)


def hull(att, acc):
    """upper concave envelope of (attempts, accuracy) points, truncated at its maximum (a frontier is non-decreasing
    in cost); every point on it is achievable by randomising between two policies of the family."""
    pts = sorted(zip(np.asarray(att, float), np.asarray(acc, float)))
    H = []
    for p in pts:
        while len(H) >= 2 and (H[-1][0] - H[-2][0]) * (p[1] - H[-2][1]) - (H[-1][1] - H[-2][1]) * (p[0] - H[-2][0]) >= 0:
            H.pop()
        if H and abs(H[-1][0] - p[0]) < 1e-12:
            if p[1] >= H[-1][1]:
                H[-1] = p
            continue
        H.append(p)
    j = int(np.argmax([h[1] for h in H]))
    x, y = zip(*H[:j + 1])
    return np.array(x), np.array(y)


def family_points(z):
    """all (N_max, tau) points of an adaptive family: per-theorem-mean attempts and accuracy."""
    return z['adaptive_att'].mean(-1).ravel(), z['adaptive_acc'].mean(-1).ravel()


def interp_curve(att, acc, m):
    o = np.argsort(att)
    att, acc = np.asarray(att)[o], np.asarray(acc)[o]
    if m < att[0] - 1e-9 or m > att[-1] + 1e-9:
        return None
    return float(np.interp(m, att, acc))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prefix', required=True)
    ap.add_argument('--scorers', required=True)
    ap.add_argument('--md', default=None)
    a = ap.parse_args()
    L = lambda s: np.load(f'{a.prefix}__{s}.npz')
    J = lambda s: json.load(open(f'{a.prefix}__{s}.json'))
    rows = []
    r = L('random'); o = L('oracle'); c = L('completion')
    rows.append(('fixed-k random pick', [float(fixed_at(r['fixed_acc'], m).mean()) for m in MS]))
    rows.append(('fixed-k completion pick', [float(fixed_at(c['fixed_acc'], m).mean()) for m in MS]))
    natt, nacc = c['nsweep_att'].mean(1), c['nsweep_acc'].mean(1)
    rows.append((f'stop at first finished (ceiling {nacc[-1]:.3f} at {natt[-1]:.2f} att)',
                 [interp_curve(natt, nacc, m) for m in MS]))
    for s in a.scorers.split(','):
        p = f'{a.prefix}__{s}.npz'
        if not os.path.exists(p):
            continue
        z = L(s)
        ni = len(z['Ns']) - 1
        rows.append((f'fixed-k {s} pick', [float(fixed_at(z['fixed_acc'], m).mean()) for m in MS]))
        hx, hy = hull(*family_points(z))
        rows.append((f'adaptive {s}-stop (frontier over N_max 2-64 x tau)', [interp_curve(hx, hy, m) for m in MS]))
    rows.append(('oracle pass@k (fixed-k)', [float(fixed_at(o['fixed_acc'], m).mean()) for m in MS]))
    oatt, oacc = o['nsweep_att'].mean(1), o['nsweep_acc'].mean(1)
    rows.append(('oracle stop at first valid', [interp_curve(oatt, oacc, m) for m in MS]))
    hdr = '| policy | ' + ' | '.join(f'm={m}' for m in MS) + ' |\n|---|' + '---|' * len(MS)
    lines = [hdr] + ['| ' + n + ' | ' + ' | '.join('-' if v is None else f'{v:.3f}' for v in vals) + ' |' for n, vals in rows]
    # bootstrap rows (nearest tau grid point on the N_max=64 curve)
    bl = ['', '| adaptive vs fixed-k same selector (N_max 64, nearest grid point) | m | att | diff [95% CI] |', '|---|---|---|---|']
    for s in ['completion'] + a.scorers.split(','):
        if not os.path.exists(f'{a.prefix}__{s}.json'):
            continue
        for k, b in J(s)['bootstrap'].items():
            if (k.startswith('N64_') or k.startswith('N16_') or k.startswith('nsweep_')) and b['target'] in (2, 4, 8):
                d = b['same_selector']
                bl.append(f'| {s} {k.split("_")[0]} | {b["target"]} | {b["att"]:.2f} | {d["diff"]:+.3f} [{d["lo"]:+.3f}, {d["hi"]:+.3f}] |')
    txt = f'### {os.path.basename(a.prefix)}\n\n' + '\n'.join(lines + bl) + '\n'
    print(txt)
    if a.md:
        open(a.md, 'w').write(txt)


if __name__ == '__main__':
    main()
