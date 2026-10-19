"""Read noise: paired SE of a pass@1 difference between two models at a given k and set size (DESIGN_REVIEW.md Q3).

    python3 power.py

For each (set, model A, model B, k list): resample k of each problem's recorded attempts (without replacement)
for both models, take the mean per-problem difference in pass rate, and bootstrap problems (400 draws) with fresh
attempt subsamples, so the SE covers problems and attempts. Also a random 300-problem subsample at the smallest k.
Training (run-to-run) noise is not in these numbers. CPU only.
"""

import json
import random
import statistics as st
from pathlib import Path

E = Path('/scratch/cowifornia/nd-rl/think/runs/eval/v2')


def rows(setp, arm):
    d = [x for x in E.iterdir() if x.name.startswith(f'{setp}_{arm}_plain')]
    return {r['name']: r for r in (json.loads(l) for l in open(d[0] / 'rows_plain_B64.jsonl'))}


def paired(setp, a, b, k, sub=None, B=400, filt=None):
    A = rows(setp, a)
    Bm = rows(setp, b)
    names = sorted(set(A) & set(Bm))
    if filt:
        names = [n for n in names if filt(A[n])]
    random.seed(1)
    if sub:
        names = random.sample(names, sub)

    def est(ns, rng):
        d = []
        for n in ns:
            oa, ob = A[n]['ok'], Bm[n]['ok']
            sa = rng.sample(oa, k) if k < len(oa) else oa
            sb = rng.sample(ob, k) if k < len(ob) else ob
            d.append(sum(sa) / len(sa) - sum(sb) / len(sb))
        return 100 * st.mean(d)

    rng = random.Random(2)
    point = est(names, rng)
    bs = [est([rng.choice(names) for _ in names], rng) for _ in range(B)]
    return len(names), point, st.pstdev(bs)


CASES = [
    ('t1320', 'grpo_think_vs_mb_krr_s0', 'grpo_think_vs_mb_s0', [16, 8, 4], None),
    ('t1320', 'grpo_think_vs_mb_krr_s0', 'grpo_think_vs_mb_s0', [16, 8], lambda r: r['min_lines'] >= 16),
    ('t2130', 'grpo_think_vs_mb_krr_s0', 'grpo_think_vs_mb_s0', [8, 4], None),
    ('charles173', 'grpo_think_vs_mb_krr_s0', 'grpo_think_vs_mb_s0', [64, 16, 8], None),
    ('t1320', 'grpo_think_vs_mb_kkl0_s0', 'grpo_think_vs_mb_s0', [16, 8, 4], None),
]

if __name__ == '__main__':
    for setp, a, b, ks, filt in CASES:
        for k in ks:
            n, p, se = paired(setp, a, b, k, filt=filt)
            print(f"{setp:10s} {'16-20' if filt else 'all':5s} k{k:<3d} n{n} Δ {p:+.2f} SE {se:.2f}")
        n, p, se = paired(setp, a, b, ks[-1], sub=min(300, n), filt=filt)
        print(f'   subsample 300 problems, k{ks[-1]}: Δ {p:+.2f} SE {se:.2f}')
