#!/usr/bin/env python3
"""noise-floor analysis: every number of the write-up from pulled files -> artifacts/nf/summary.json
(one row per pool x Stage-1 seed, plus the gap-closer rows) and a markdown dump (stdout).

Sources (all under artifacts/nf/ unless said):
  held-out   heldout_<pool>_s<k>.json / .jsonl (eval_set greedy on data/p2/heldout.jsonl), joined by name to the
             held-out records' pattern labels for the 500-theorem depth-3 slice
  coverage   cov_<pool>_<p>_s<k>.s0.jsonl (coverage.py k 2,000; a proof counts iff nd_verify AND Lean accept)
  ladder     la_frozen_<p>_s<k>/round_8.json (transfer_cum.solved, transfer_cum.lstar), and the same for the
             gap-closer runs la_frozen_dsg_g1_s0, la_{T1,frozen}_dsc_{c0,a1}_s{0,1}
  sets       shape_<p>.json, overlap_<p>.json, assemble_<p>.json, render_<p>.json
  record     record_*.json (unmodified nd2lean.py --check + nd_verify on every counted proof; gate_*.jsonl rates)

  python3 nf_analysis.py [--out artifacts/nf/summary.json]

MODEL LABEL for every row: 3.3 M-parameter from-scratch GPT, `lean_seq` Lean surface format, cap 6,
train.py --mode lean_seq --steps 6000 --bs 128; trained on the 155,000-record set named in `set` .
"""
import argparse, json, os, sys, glob, collections, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

D = 'artifacts/nf'
POOLS = ('p1', 'p2', 'p3', 'p4')
SEEDS = tuple(range(0, 13))   # 0-1 pre-registered; 2 by addendum 1; 3-12 by addendum 2 (held-out only)
COV = {'d3': ('data/p2/targets_depth3.jsonl', 'depth3'),
       'req8': ('data/r3_1/depth3_req.jsonl', 'depth3'),
       'red': ('data/p2/targets_reductio_req.jsonl', 'derived_dn')}

rd = lambda fn: [json.loads(l) for l in open(fn) if l.strip()]
jl = lambda fn: json.load(open(fn)) if os.path.exists(fn) else None


def heldout(p, s):
    fn = f'{D}/heldout_{p}_s{s}.jsonl'
    if not os.path.exists(fn):
        return None
    rows = rd(fn); summ = jl(fn.replace('.jsonl', '.json'))
    lab = {r['name']: r['pat'] for r in rd('data/p2/heldout.jsonl')}
    by = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        pt = lab[r['name']]
        key = 'pattern' if any(pt.get(k) for k in ('derived_ore', 'reductio', 'depth3')) else 'none'
        by[key][0] += r['solved']; by[key][1] += 1
        if pt.get('depth3'):
            by['depth3'][0] += r['solved']; by['depth3'][1] += 1
        if r['n_lines'] == 6:
            by['len6_' + key][0] += r['solved']; by['len6_' + key][1] += 1
    return {'rate': summ['rate'], 'n': summ['n'], 'by_len': {k: v['rate'] for k, v in summ['by_len'].items()},
            'by_pattern': {k: {'rate': v[0] / v[1], 'solved': v[0], 'n': v[1]} for k, v in by.items()}}


def coverage(p, s, pool):
    fn = f'{D}/cov_{pool}_{p}_s{s}.s0.jsonl'
    if not os.path.exists(fn):
        return None
    rows = rd(fn); pfn, pat = COV[pool]
    meta = {r['name']: r for r in rd(pfn)}
    n = len(rows); hit = sum(1 for r in rows if r['n_ok'] > 0)
    hit_pat = sum(1 for r in rows if any(x['pat'].get(pat) for x in r['proofs']))
    tried = sum(r['n_tried'] for r in rows); ok = sum(r['n_ok'] for r in rows)
    strata = collections.defaultdict(lambda: [0, 0, 0])
    for r in rows:
        st = str(meta[r['name']].get('min_lines_ub'))
        strata[st][1] += 1; strata[st][0] += r['n_ok'] > 0; strata[st][2] += any(x['pat'].get(pat) for x in r['proofs'])
    d8 = sum(1 for r in rows for x in r['proofs'] if x['written'] >= 8)
    return {'n_targets': n, 'targets_hit': hit, 'targets_hit_with_pattern': hit_pat, 'pattern': pat,
            'per_sample_rate': ok / tried if tried else None, 'samples': tried, 'n_ok_samples': ok,
            'by_min_lines_ub': {k: {'n': v[1], 'hit': v[0], 'hit_with_pattern': v[2]} for k, v in sorted(strata.items())},
            'distinct_proofs': sum(len(r['proofs']) for r in rows), 'distinct_ge8': d8,
            'written_hist': dict(sorted(collections.Counter(x['written'] for r in rows for x in r['proofs']).items())),
            'lean_checked': sum(r.get('n_lean_checked', 0) for r in rows),
            'lean_rejected': sum(r.get('n_lean_rejected', 0) for r in rows), 'complete': n == len(meta)}


def ladder(name, need=8):
    """<D>/<name>/round_8.json -> the cumulative transfer solves and L* after all 8 rounds.
    A directory without round_8.json is a run that did not finish and is NOT read: the 2026-09-24
    orphaned-waiter fault left several partial ladder directories behind, and reading the last round
    present would silently report a 3-round ladder as an 8-round one."""
    d = f'{D}/{name}'
    rounds = sorted(int(os.path.basename(f).split('_')[-1][:-5]) for f in glob.glob(f'{d}/round_*.json'))
    if not rounds or rounds[-1] != need or len(rounds) != need:
        return None
    R = need
    j = jl(f'{d}/round_{R}.json')
    tc, gc = j['transfer_cum'], j['targets_cum']
    return {'rounds': R, 'transfer_solved': tc['solved'], 'transfer_n': tc['n'], 'transfer_lstar': tc['lstar'],
            'transfer_ge': tc['ge'], 'targets_solved': gc['solved'], 'targets_n': gc['n'],
            'targets_lstar': gc['lstar'], 'ckpt': j['ckpt'], 'k': j['k'], 'max_new': j['max_new'],
            'secs_total': sum(jl(f'{d}/round_{r}.json')['secs'] for r in rounds)}


# ---------------- noise-floor statistics ----------------

def mean(v):
    return sum(v) / len(v)


def sd(v, ddof=1):
    if len(v) - ddof <= 0:
        return 0.0
    m = mean(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - ddof))


def two_way(cells):
    """cells[(pool, seed)] -> value.  Unreplicated pools x seeds decomposition.
    y_ij = mu + a_i + b_j + e_ij; the pool x seed interaction is confounded with error (stated)."""
    ps = sorted({k[0] for k in cells}); ss = sorted({k[1] for k in cells})
    a, b = len(ps), len(ss)
    if a < 2 or b < 2 or len(cells) != a * b:
        return None
    y = [[cells[(p, s)] for s in ss] for p in ps]
    gm = mean([x for row in y for x in row])
    rm = [mean(row) for row in y]
    cm = [mean([y[i][j] for i in range(a)]) for j in range(b)]
    ss_pool = b * sum((r - gm) ** 2 for r in rm)
    ss_seed = a * sum((c - gm) ** 2 for c in cm)
    ss_tot = sum((y[i][j] - gm) ** 2 for i in range(a) for j in range(b))
    ss_res = ss_tot - ss_pool - ss_seed
    ms_pool = ss_pool / (a - 1); ms_seed = ss_seed / (b - 1); ms_res = ss_res / ((a - 1) * (b - 1))
    return {'levels_pool': ps, 'levels_seed': ss, 'grand_mean': gm,
            'pool_means': dict(zip(ps, rm)), 'seed_means': dict(zip([str(s) for s in ss], cm)),
            'MS_pool': ms_pool, 'MS_seed': ms_seed, 'MS_resid': ms_res,
            'F_pool': ms_pool / ms_res if ms_res else None, 'F_seed': ms_seed / ms_res if ms_res else None,
            'var_pool': (ms_pool - ms_res) / b, 'var_seed': (ms_seed - ms_res) / a, 'var_resid': ms_res,
            'note': 'unreplicated two-way: pool x seed interaction is confounded with error; '
                    'negative variance components are reported as measured, not truncated'}


# two-sided alpha .05, 80 % power, n = 2 per group: (t_{.975,2} + t_{.80,2}) * sqrt(1/2 + 1/2)
MDD_K = (4.302653 + 1.060660) * math.sqrt(1.0)


def floor_stats(cells, integer=False):
    v = list(cells.values())
    m = mean(v); s = sd(v)
    lo, hi = min(v), max(v)
    g = sorted(v)
    gaps = [(g[i + 1] - g[i], i) for i in range(len(g) - 1)]
    gmax = max(gaps) if gaps else (0, 0)
    # bimodality coefficient b = (skew^2 + 1) / kurtosis (sample, excess+3); b > 5/9 is bimodal-consistent
    n = len(v); bc = None
    if n > 3 and s > 0:
        m3 = sum((x - m) ** 3 for x in v) / n / (sum((x - m) ** 2 for x in v) / n) ** 1.5
        m4 = sum((x - m) ** 4 for x in v) / n / (sum((x - m) ** 2 for x in v) / n) ** 2
        g1 = m3 * math.sqrt(n * (n - 1)) / (n - 2)
        g2 = ((n - 1) / ((n - 2) * (n - 3))) * ((n + 1) * (m4 - 3) + 6) + 3
        bc = (g1 ** 2 + 1) / g2 if g2 else None
    return {'n_cells': n, 'values': {f'{k[0]}_s{k[1]}': cells[k] for k in sorted(cells)},
            'mean': m, 'sd': s, 'cv': s / m if m else None, 'min': lo, 'max': hi,
            'range': hi - lo, 'max_over_min': hi / lo if lo else None,
            'largest_gap': gmax[0], 'largest_gap_between': [g[gmax[1]], g[gmax[1] + 1]] if gaps else None,
            'bimodality_coefficient': bc, 'bimodal_consistent': (bc is not None and bc > 5 / 9),
            'mdd_n2_abs': MDD_K * s, 'mdd_n2_frac_of_mean': (MDD_K * s / m) if m else None,
            'two_way': two_way(cells),
            'note_permutation': 'a two-sided exact permutation test at 2 vs 2 has minimum attainable p = 1/3, '
                                'so no n = 2 comparison is significant non-parametrically; the MDD is the '
                                'parametric best case (t-test, alpha .05 two-sided, 80 % power).'}


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def done(c):
    return c if (c and c.get('complete')) else None


QUANTS = [
    ('heldout_overall', lambda r: r['heldout'] and r['heldout']['rate']),
    ('heldout_len2', lambda r: r['heldout'] and r['heldout']['by_len'].get('2')),
    ('heldout_len3', lambda r: r['heldout'] and r['heldout']['by_len'].get('3')),
    ('heldout_len4', lambda r: r['heldout'] and r['heldout']['by_len'].get('4')),
    ('heldout_len5', lambda r: r['heldout'] and r['heldout']['by_len'].get('5')),
    ('heldout_len6', lambda r: r['heldout'] and r['heldout']['by_len'].get('6')),
    ('heldout_depth3_slice', lambda r: r['heldout'] and r['heldout']['by_pattern']['depth3']['rate']),
    ('heldout_len6_nopattern', lambda r: r['heldout'] and r['heldout']['by_pattern']['len6_none']['rate']),
    # only complete coverage runs count: a run still in flight has processed part of its pool
    ('cov_red_solved', lambda r: done(r['coverage']['red']) and r['coverage']['red']['targets_hit']),
    ('cov_req8_solved', lambda r: done(r['coverage']['req8']) and r['coverage']['req8']['targets_hit']),
    ('cov_d3_solved', lambda r: done(r['coverage']['d3']) and r['coverage']['d3']['targets_hit']),
    ('cov_d3_with_depth3_proof', lambda r: done(r['coverage']['d3']) and r['coverage']['d3']['targets_hit_with_pattern']),
    ('ladder_frozen_transfer_solved', lambda r: r['ladder_frozen'] and r['ladder_frozen']['transfer_solved']),
    ('ladder_frozen_transfer_lstar', lambda r: r['ladder_frozen'] and r['ladder_frozen']['transfer_lstar']),
    ('ladder_frozen_targets_solved', lambda r: r['ladder_frozen'] and r['ladder_frozen']['targets_solved']),
]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', default=f'{D}/summary.json'); a = ap.parse_args()
    rows = []
    for p in POOLS:
        for s in SEEDS:
            r = {'pool': p, 'seed': s, 'set': f'data/nf/train_{p}.jsonl',
                 'ckpt': f'ckpts/nf/stage1_{p}_s{s}.pt',
                 'model': '3.3M from-scratch GPT, lean_seq, cap 6, train.py --steps 6000 --bs 128',
                 'heldout': heldout(p, s),
                 'coverage': {k: coverage(p, s, k) for k in COV},
                 'ladder_frozen': ladder(f'la_frozen_{p}_s{s}')}
            rows.append(r)
    gap = {n: ladder(n) for n in ['la_frozen_dsg_g1_s0'] +
           [f'la_{k}_dsc_{arm}_s{s}' for arm in ('c0', 'a1') for k in ('T1', 'frozen') for s in (0, 1)]}
    sets = {p: {'shape': (lambda j: j and list(j.values())[0])(jl(f'{D}/shape_{p}.json')),
                'assemble': jl(f'{D}/assemble_{p}.json'), 'overlap': jl(f'{D}/overlap_{p}.json'),
                'render': jl(f'{D}/render_{p}.json')} for p in POOLS}
    record = {os.path.basename(f)[:-5]: jl(f) for f in sorted(glob.glob(f'{D}/record_*.json'))}
    # the in-loop Lean gate's own agreement rate, over every gate log pulled back from both pods
    # (ds-generator's review asked for this rate; it is one json line per generate() call)
    g = collections.Counter()
    for fn in glob.glob(f'{D}/gate_*.jsonl'):
        if fn.endswith('.disagree.jsonl'):
            continue
        for l in open(fn):
            try:
                r = json.loads(l)
            except Exception:
                continue
            g['calls'] += 1
            for k in ('samples', 'parse_fail', 'distinct_checked', 'both_ok', 'nd_ok_lean_rej', 'nd_rej_lean_ok', 'both_rej'):
                g[k] += r.get(k, 0)
    dis = sum(1 for fn in glob.glob(f'{D}/gate_*.disagree.jsonl') for _ in open(fn))
    gate = dict(g)
    gate['disagree_lines'] = dis
    gate['lean_only_per_million_distinct'] = (1e6 * g['nd_rej_lean_ok'] / g['distinct_checked']) if g['distinct_checked'] else None
    gate['nd_only_per_million_distinct'] = (1e6 * g['nd_ok_lean_rej'] / g['distinct_checked']) if g['distinct_checked'] else None

    floors = {}
    for name, fn in QUANTS:
        cells = {}
        for r in rows:
            try:
                v = fn(r)
            except Exception:
                v = None
            if v is not None:
                cells[(r['pool'], r['seed'])] = v
        if len(cells) >= 4:
            floors[name] = floor_stats(cells)
            # the 8-cell design the expectations E1-E14 were written for, scored separately
            c8 = {k: v for k, v in cells.items() if k[1] in (0, 1)}
            if len(c8) == 8:
                floors[name]['seeds01_only'] = floor_stats(c8)
    # the depth-3 held-out slice: bimodal, so report the high-mode proportion with a Wilson interval
    d3 = floors.get('heldout_depth3_slice')
    if d3:
        v = list(d3['values'].values())
        hi = sum(1 for x in v if x > 0.44); lo = sum(1 for x in v if x < 0.11)
        w = wilson(hi, len(v))
        d3['high_mode'] = {'threshold_high': 0.44, 'threshold_low': 0.11, 'n_high': hi, 'n_low': lo,
                           'n_between': len(v) - hi - lo, 'p_high': hi / len(v), 'wilson95': list(w),
                           'note': "mode boundaries are ds-rendering's (>0.44 high, <0.11 low)"}
    out = {'rows': rows, 'gap_closers': gap, 'sets': sets, 'record': record, 'gate': gate, 'floors': floors}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, 'w'), indent=1)

    print('# noise-floor — the null cells (pool x Stage-1 seed)\n')
    print('| quantity | cells | mean | sd | min | max | max/min | MDD at n = 2 |\n|---|---|---|---|---|---|---|---|')
    for name, _ in QUANTS:
        f = floors.get(name)
        if not f:
            continue
        fmt = (lambda x: f'{x:.4f}') if abs(f['mean']) < 3 else (lambda x: f'{x:g}')
        print(f'| `{name}` | {f["n_cells"]} | {fmt(f["mean"])} | {f["sd"]:.4g} | {fmt(f["min"])} | {fmt(f["max"])} | '
              + (f'{f["max_over_min"]:.2f}x' if f['max_over_min'] else '-')
              + f' | {f["mdd_n2_abs"]:.4g}'
              + (f' ({100 * f["mdd_n2_frac_of_mean"]:.0f} %)' if f['mdd_n2_frac_of_mean'] else '') + ' |')
    print('\n### the eight pre-registered cells (Stage-1 seeds 0 and 1), per cell\n')
    print('| quantity | ' + ' | '.join(f'{p} s{s}' for p in POOLS for s in (0, 1)) +
          ' |\n|---|' + '---|' * (len(POOLS) * 2))
    for name, _ in QUANTS:
        f = floors.get(name)
        if not f:
            continue
        vals = f['values']
        fmt = (lambda x: f'{x:.4f}') if abs(f['mean']) < 3 else (lambda x: f'{x:g}')
        print(f'| `{name}` | ' + ' | '.join(fmt(vals[f'{p}_s{s}']) if f'{p}_s{s}' in vals else '—'
                                            for p in POOLS for s in (0, 1)) + ' |')
    print('\n## variance components (unreplicated pools x seeds)\n')
    print('| quantity | MS_pool | MS_seed | MS_resid | var_pool | var_seed | var_resid | bimodality b |\n|---|' + '---|' * 7)
    for name, _ in QUANTS:
        f = floors.get(name)
        if not f or not f['two_way']:
            continue
        t = f['two_way']
        print(f'| `{name}` | {t["MS_pool"]:.4g} | {t["MS_seed"]:.4g} | {t["MS_resid"]:.4g} | {t["var_pool"]:.4g} | '
              f'{t["var_seed"]:.4g} | {t["var_resid"]:.4g} | '
              + (f'{f["bimodality_coefficient"]:.3f}' + (' **bimodal**' if f['bimodal_consistent'] else '')
                 if f['bimodality_coefficient'] is not None else '-') + ' |')
    print('\n## gap-closers (existing bucket checkpoints; no retraining)\n')
    print('| run | checkpoint | transfer solved / 2,285 | transfer L* | targets solved / 4,495 |\n|---|---|---|---|---|')
    for n, g in gap.items():
        if g:
            print(f'| `{n}` | `{g["ckpt"]}` | {g["transfer_solved"]} | {g["transfer_lstar"]} | {g["targets_solved"]} |')
    print('\n## in-loop Lean gate (both pods)\n')
    print(json.dumps(gate, indent=1))
    print(f'\nwritten {a.out}')


if __name__ == '__main__':
    main()
