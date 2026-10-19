"""Per-step failure hazard and per-line pass rates from plain B64 eval rows (DESIGN_REVIEW.md E4, E6).

    python3 hazard.py t1320 grpo_think_vs_mb_s0 grpo_think_vs_mb_kkl0_s0 ...

For each model: the hazard of the first invalid step per 1,000 steps reached, in step-index buckets (an attempt
reaches steps 1..fail_step if it failed, else 1..steps; a failure at fail_step counts as a death there), and the
mean per-problem pass rate by reference length (`min_lines`). Pooled over problems, so late buckets hold only the
long problems that got that far (survivorship). CPU only.
"""

import collections
import json
import sys
from pathlib import Path

E = Path('/scratch/cowifornia/nd-rl/think/runs/eval/v2')
BUCKETS = [(1, 5), (6, 10), (11, 15), (16, 20), (21, 25), (26, 30), (31, 40), (41, 60)]


def load(setp, arm):
    d = [x for x in E.iterdir() if x.name.startswith(f'{setp}_{arm}_plain')]
    if not d:
        return None
    return [json.loads(l) for l in open(d[0] / 'rows_plain_B64.jsonl')]


def main(setp, arms):
    print('set', setp)
    for a in arms:
        rows = load(setp, a)
        if rows is None:
            print('missing', a)
            continue
        reach = collections.Counter()
        die = collections.Counter()
        bylen = collections.defaultdict(list)
        for r in rows:
            bylen[r['min_lines']].append(sum(r['ok']) / len(r['ok']))
            for stp, fs, ok in zip(r['steps'], r['fail_step'], r['ok']):
                end = fs if fs is not None else stp
                for t in range(1, end + 1):
                    reach[t] += 1
                if fs is not None and not ok:
                    die[fs] += 1
        hz = []
        for lo, hi in BUCKETS:
            n_reach = sum(reach[t] for t in range(lo, hi + 1))
            n_die = sum(die[t] for t in range(lo, hi + 1))
            hz.append(f'{1000 * n_die / n_reach:5.1f}' if n_reach else '  -  ')
        lp = ' '.join(f'{L}:{100 * sum(v) / len(v):.0f}' for L, v in sorted(bylen.items()))
        print(f"{a:28s} hazard/1000 steps {' '.join(hz)} | pass by line {lp}")


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2:])
