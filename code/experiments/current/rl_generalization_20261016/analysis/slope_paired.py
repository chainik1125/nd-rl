"""Paired differences in level and length-slope between two models (DESIGN_REVIEW.md E2 table).

    python3 slope_paired.py A1 B1 [A2 B2 ...]      # prints A − B for each pair

Same fit as slope.py (logit of the smoothed per-line mean pass rate on t1320 + t2130, 13-30 lines), on the problems
both models were read on. Interval: 400-draw paired bootstrap over problems (the same resampled problems for both
models), 2.5% / 97.5% points. Exploratory. CPU only.
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from slope import E, fit  # noqa: E402

import json  # noqa: E402


def rows(setp, arm):
    d = [x for x in E.iterdir() if x.name.startswith(f'{setp}_{arm}_plain')][0]
    return {r['name']: (r['min_lines'], sum(r['ok']) / len(r['ok']))
            for r in (json.loads(l) for l in open(d / 'rows_plain_B64.jsonl'))}


def load(arm):
    d = rows('t1320', arm)
    d.update(rows('t2130', arm))
    return d


def main(argv):
    pairs = [(argv[i], argv[i + 1]) for i in range(0, len(argv), 2)]
    for a, b in pairs:
        A, B = load(a), load(b)
        names = sorted(set(A) & set(B))

        def diff(ns):
            fa = fit([A[n] for n in ns])
            fb = fit([B[n] for n in ns])
            return fa[0] - fb[0], fa[1] - fb[1]

        p = diff(names)
        rng = random.Random(0)
        bs = [diff([rng.choice(names) for _ in names]) for _ in range(400)]
        ia = sorted(x[0] for x in bs)
        ib = sorted(x[1] for x in bs)
        print(f'{a} − {b}: Δintercept {p[0]:+.3f} [{ia[10]:+.3f},{ia[389]:+.3f}]  '
              f'Δslope {p[1]:+.4f} [{ib[10]:+.4f},{ib[389]:+.4f}]')


if __name__ == '__main__':
    main(sys.argv[1:])
