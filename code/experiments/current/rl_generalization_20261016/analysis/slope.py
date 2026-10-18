"""Level and length-slope of long-proof pass rates, per model (DESIGN_REVIEW.md E2).

    python3 slope.py grpo_think_vs_mb_s0 grpo_think_vs_mb_kkl0_s0 ...

Pools a model's plain B64 reads of t1320 and t2130 (13-30 reference lines). For each reference length L, the mean
per-problem pass rate m_L (with +0.5 / +1 smoothing) gives logit(m_L); a weighted (by problems) least-squares line
logit(m_L) = a + b (L - 13) gives the level a (logit at 13 lines) and the slope b (per extra line). Intervals:
300-draw bootstrap over problems (unpaired), 2.5% / 97.5% points. Exploratory. CPU only.
"""

import collections
import json
import math
import random
import sys
from pathlib import Path

E = Path('/scratch/cowifornia/nd-rl/think/runs/eval/v2')


def rows(setp, arm):
    d = [x for x in E.iterdir() if x.name.startswith(f'{setp}_{arm}_plain')]
    return [json.loads(l) for l in open(d[0] / 'rows_plain_B64.jsonl')] if d else None


def fit(pts):
    """pts: (L, per-problem pass rate). Returns (level a, slope b)."""
    by = collections.defaultdict(list)
    for L, p in pts:
        by[L].append(p)
    xs, ys, ws = [], [], []
    for L, v in by.items():
        m = (sum(v) + 0.5) / (len(v) + 1)
        xs.append(L - 13)
        ys.append(math.log(m / (1 - m)))
        ws.append(len(v))
    W = sum(ws)
    mx = sum(w * x for w, x in zip(ws, xs)) / W
    my = sum(w * y for w, y in zip(ws, ys)) / W
    b = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys)) / sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    return my - b * mx, b


def main(arms):
    for a in arms:
        r1, r2 = rows('t1320', a), rows('t2130', a)
        if r1 is None or r2 is None:
            print('missing', a)
            continue
        pts = [(r['min_lines'], sum(r['ok']) / len(r['ok'])) for r in r1 + r2]
        A, B = fit(pts)
        random.seed(0)
        bs = [fit([random.choice(pts) for _ in pts]) for _ in range(300)]
        sl = sorted(t[1] for t in bs)
        lv = sorted(t[0] for t in bs)
        print(f'{a:28s} logit pass at 13 lines {A:+.2f} [{lv[7]:+.2f},{lv[292]:+.2f}] '
              f'(= {100 / (1 + math.exp(-A)):.1f}%)  slope/line {B:+.3f} [{sl[7]:+.3f},{sl[292]:+.3f}]')


if __name__ == '__main__':
    main(sys.argv[1:])
