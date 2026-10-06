#!/usr/bin/env python3
"""The retrospective table: every standing finding this run can score, put beside the floor of the
quantity it was read off.  Sources for the originals are named per row; the floors come from
artifacts/nf/summary.json (nf_analysis.py).  Two criteria, both reported, because they can disagree:

  MDD   the smallest difference a two-sample t-test at n seeds per arm has 80 % power to detect,
        (t_{.975,2n-2} + t_{.80,2n-2}) * s * sqrt(2/n) with s the pooled sd over the null cells.
        This is the *parametric best case*: the exact permutation test at 2 vs 2 cannot reach
        p < 0.05 at all (minimum attainable p = 1/3).
  RANGE does the arm's value fall outside the interval spanned by the null cells?  Weaker evidence
        (with 8 null draws, one value outside their range is only about p = 0.22 one-sided) but it
        is the question a reader actually asks, and it is distribution-free.

A finding is reported as surviving only if it clears the MDD; clearing RANGE alone is stated as
"outside the null range, but below the MDD".  python3 nf_retro.py"""
import json, math

F = json.load(open('artifacts/nf/summary.json'))['floors']

# t_{.975,df} and t_{.80,df} for the df this project's runs actually use (n = 2 and n = 6 per arm)
TK = {2: 4.302653 + 1.060660, 6: 2.228139 + 0.879058}


def mdd(f, n):
    """smallest detectable difference at n seeds per arm, 80 % power, two-sided alpha .05"""
    return TK[n] * f['sd'] * math.sqrt(2.0 / n)

# (quantity, run, claim, arm value(s), control value(s), how to score)
#   'ratio' rows are scored against the quantity's max/min over the null cells and its MDD ratio;
#   'pp' rows against the MDD in percentage points.
R = [
 ('ladder_frozen_transfer_solved', 'ds-composition', 'A3 (cap 8) frozen ladder beats C0', [976], [158], 'ratio'),
 ('ladder_frozen_transfer_solved', 'ds-generator', 'G1 frozen ladder is above both C0 seeds', [170], [158, 114], 'ratio'),
 ('ladder_frozen_transfer_solved', 'ds-composition', 'A1 frozen ladder beats C0 (n = 1)', [205], [158], 'ratio'),
 ('ladder_frozen_transfer_solved', 'ds-composition', 'A2 frozen ladder is below C0 (n = 1)', [111], [158], 'ratio'),
 ('ladder_frozen_transfer_solved', 'ds-generator', 'G2 frozen ladder is above C0 on seed 1', [125], [114], 'ratio'),
 ('ladder_frozen_transfer_solved', 'ds-generator', 'G2 frozen ladder is far below C0 on seed 0', [44], [158], 'ratio'),
 ('ladder_frozen_transfer_lstar', 'ds-composition', "A3 (cap 8) frozen L* 11 vs C0 9", [11], [9], 'abs'),
 ('ladder_frozen_transfer_lstar', 'ds-composition', "A1 frozen L* 10 vs C0 9 (n = 1)", [10], [9], 'abs'),
 ('ladder_frozen_transfer_lstar', 'lean-format', "lean_seq ladder L* 11 vs token 10", [11], [10], 'abs'),
 ('cov_red_solved', 'ds-composition', 'A3 (cap 8) solves more `redreq` than C0', [54, 72], [28, 26], 'ratio'),
 ('cov_red_solved', 'ds-composition', "A1's reductio deficit vs C0", [16, 18], [28, 26], 'ratio'),
 ('cov_red_solved', 'ds-generator', 'G2 writes no `redreq` proof at all', [0, 0], [31, 27], 'ratio'),
 ('cov_req8_solved', 'ds-generator', 'G1 s1 required@8 is 2.3x C0 s1', [259], [113], 'ratio'),
 ('heldout_overall', 'lean-format', 'lean_seq held-out beats the token format', [0.909, 0.896], [0.883, 0.883], 'pp'),
 ('heldout_overall', 'ds-composition', 'A1 held-out beats C0', [0.9172, 0.9424], [0.9088, 0.8962], 'pp'),
 ('heldout_overall', 'ds-composition', 'A3 (cap 8) held-out beats C0', [0.9508, 0.9536], [0.9088, 0.8962], 'pp'),
 ('heldout_len6_nopattern', 'ds-composition', "A1's 6-line no-pattern gain over C0", [0.891, 0.899], [0.818, 0.834], 'pp'),
 ('heldout_len6_nopattern', 'ds-composition', "A2's 6-line no-pattern deficit vs C0", [0.777, 0.773], [0.818, 0.834], 'pp'),
 ('heldout_depth3_slice', 'ds-rendering', 'R2 depth-3 held-out beats C0 (n = 6 each)', [0.661], [0.533], 'pp6'),
 ('heldout_len6', 'ds-generator', 'G2 6-line held-out is far below C0', [0.138, 0.425], [0.686, 0.584], 'pp'),
]

def rows():
    """One dict per standing finding: the original numbers, the floor it is scored against and the
    verdict.  Shared by the markdown table below and by nf_figures.py so the two cannot disagree."""
    out = []
    for q, run, claim, arm, ctrl, kind in R:
        f = F[q]
        n = 6 if kind == 'pp6' else 2
        a, c = sum(arm) / len(arm), sum(ctrl) / len(ctrl)
        m = mdd(f, n)
        outside = all(x < f['min'] or x > f['max'] for x in arm)
        eff = abs(a - c)
        if kind in ('pp', 'pp6'):
            orig = f'{100 * a:.1f} vs {100 * c:.1f} ({100 * (a - c):+.1f} pp)'
            rng = f'{100 * f["min"]:.1f}–{100 * f["max"]:.1f}'
            fl = f'±{100 * m:.1f} pp' + (' (n = 6)' if n == 6 else '')
        elif kind == 'abs':
            orig = f'{a:g} vs {c:g} ({a - c:+g})'
            rng = f'{f["min"]:g}–{f["max"]:g}'
            fl = f'±{m:.1f}'
        else:
            ratio = max(a, c) / min(a, c) if min(a, c) else float('inf')
            orig = ('/'.join(f'{x:g}' for x in arm) + ' vs ' + '/'.join(f'{x:g}' for x in ctrl)
                    + (f' ({ratio:.2f}×)' if ratio != float('inf') else ' (∞)'))
            rng = f'{f["min"]:g}–{f["max"]:g}'
            fl = f'±{m:.0f} ({1 + m / f["mean"]:.2f}×)'
        v = ('survives' if eff > m else
             ('outside the null range, but below the MDD' if outside else 'inside the floor'))
        out.append({'q': q, 'run': run, 'claim': claim, 'orig': orig, 'range': rng, 'floor_str': fl,
                    'effect': eff, 'mdd': m, 'n_per_arm': n, 'verdict': v,
                    'effect_pct': 100 * eff / c if c else None, 'mdd_pct': 100 * m / c if c else None})
    return out


if __name__ == '__main__':
    print('| quantity | run | standing finding | original | null cells | MDD | verdict |')
    print('|---|---|---|---|---|---|---|')
    for r in rows():
        mark = '**survives**' if r['verdict'] == 'survives' else (
            '**inside the floor**' if r['verdict'] == 'inside the floor' else r['verdict'])
        print(f'| `{r["q"]}` | {r["run"]} | {r["claim"]} | {r["orig"]} | {r["range"]} | {r["floor_str"]} | {mark} |')
