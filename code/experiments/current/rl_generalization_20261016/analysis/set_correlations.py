"""Correlations of pass@1 between read sets across (arm, seed) models (DESIGN_REVIEW.md E1).

    python3 set_correlations.py

Parses the per-seed table (§ 2) of matched_eval/v2_results_ynez.md, plain reads only, and prints the Pearson r of
pass@1 for each pair of sets: over all models with both sets, then over GRPO models only (EI arms, the T 1.0 reads
and the feedback pilot left out). CPU only.
"""

import collections
import statistics as st
from pathlib import Path

RESULTS = Path('/scratch/cowifornia/nd-rl/think/nd-rl/code/experiments/current/thinking_traces_20261005/'
               'matched_eval/v2_results_ynez.md')
SETS = ['13–20 lines (986)', '16–20 lines', '21–30 lines (1,000)', 'textbook72', 'Charles149']


def load():
    rows = collections.defaultdict(dict)
    for line in open(RESULTS):
        if not line.startswith('| '):
            continue
        c = [x.strip() for x in line.strip().strip('|').split('|')]
        if len(c) != 7 or c[2] != 'plain':
            continue
        arm, seed, _read, sset, _n, p1, _p5 = c
        try:
            v = float(p1.split()[0])
        except ValueError:
            continue
        rows[(arm, seed)][sset] = v
    return rows


def corr(rows, a, b):
    pts = [(r[a], r[b]) for r in rows.values() if a in r and b in r]
    xs, ys = zip(*pts)
    mx, my = st.mean(xs), st.mean(ys)
    sxy = sum((x - mx) * (y - my) for x, y in pts)
    sx = sum((x - mx) ** 2 for x in xs) ** 0.5
    sy = sum((y - my) ** 2 for y in ys) ** 0.5
    return len(pts), sxy / (sx * sy)


def table(rows, title):
    print(title)
    for i, a in enumerate(SETS):
        for b in SETS[i + 1:]:
            n, r = corr(rows, a, b)
            print(f'  {a} | {b}: n {n}, r {r:+.2f}')


if __name__ == '__main__':
    rows = load()
    table(rows, 'all models')
    grpo = {k: v for k, v in rows.items()
            if not k[0].startswith('EI') and 'T 1.0' not in k[0] and 'feedback' not in k[0]}
    table(grpo, 'GRPO only (excluding T 1.0 reads and the feedback pilot)')
