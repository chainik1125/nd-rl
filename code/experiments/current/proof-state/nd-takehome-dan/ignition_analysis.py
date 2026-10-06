#!/usr/bin/env python3
"""Ignition study (HANDOFF_BRIEF Task 2): base rates from pre-RL samples, per-round pattern counts of the EI arms,
ignition rounds, interventions, base-generalisation fraction.  All counts start-index normalised via
phase2_metrics.arm_metrics (min-round rule).

  python ignition_analysis.py --out artifacts/ign/summary.json [--figs figures]

Definitions (log.md 2026-09-16 18:37, pre-registered):
  base rate r_P      = sum over the 300 sampled targets of pattern-P verified samples / total samples (2000 x 300),
                       re-derived from the per-proof hit counts in cov_<model>.s0.jsonl (not from hits_by_pattern);
  frozen control     = targets whose first pattern-P sample index <= 256 (the 8 x 32 attempts of an EI arm);
  ignition round     = first round with >= thr cumulative target theorems solved by a pattern-P proof
                       (thr = 2 % of the pool: 20 of 1000 depth-3, 12 of 606 reductio);
  predicted first-proof round = 1 / (r_P * k * N)  with k = 32, N = pool size.
"""
import argparse, json, os, sys, glob, math, re, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase2_metrics import arm_metrics

K = 32
SETS = {
    'depth3': dict(pattern='depth3', n=1000, thr=20, plateau=0.34,
                   cov_re=r'cov_(depth3_f0(?:_a\d)?_s(\d+))\.s0\.jsonl$',
                   arm_re=r'ei_depth3_f0_a1_s(\d+)$', iv_re=r'ei_depth3_f0_a1_s(\d+)_iv([KTS])$',
                   model=lambda s: f'depth3_f0_a1_s{s}', arm=lambda s: f'artifacts/p2/ei_depth3_f0_a1_s{s}'),
    'reductio': dict(pattern='reductio', n=606, thr=12, plateau=None,
                     cov_re=r'cov_(reductio_f0_s(\d+))\.s0\.jsonl$',
                     arm_re=r'ei_reductio_f0_s(\d+)_t2$', iv_re=r'ei_reductio_f0_s(\d+)_t2_iv([KTS])$',
                     model=lambda s: f'reductio_f0_s{s}', arm=lambda s: f'artifacts/p2/ei_reductio_f0_s{s}_t2'),
}


def binom_cdf(k, n, p):
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(0, k + 1))


def clopper_pearson(k, n, alpha=0.05):
    """exact binomial interval by bisection (no scipy on the VPS)."""
    if n == 0:
        return (0.0, 1.0)
    def solve(f, lo, hi):
        for _ in range(60):
            mid = (lo + hi) / 2
            if f(mid): lo = mid
            else: hi = mid
        return (lo + hi) / 2
    lower = 0.0 if k == 0 else solve(lambda p: 1 - binom_cdf(k - 1, n, p) < alpha / 2, 0.0, 1.0)
    upper = 1.0 if k == n else solve(lambda p: binom_cdf(k, n, p) > alpha / 2, 0.0, 1.0)
    return (lower, upper)


def coverage_stats(fn, pattern):
    recs = [json.loads(l) for l in open(fn) if l.strip()]
    n_tried = sum(r['n_tried'] for r in recs)
    hits = 0; n_ok = 0; tw = 0; solved = 0; f256_p = 0; f256_any = 0; per_t = []; n_distinct_p = 0; wh = collections.Counter()
    for r in recs:
        ps = r['proofs']
        assert sum(p['count'] for p in ps) == r['n_ok'], fn
        hp = sum(p['count'] for p in ps if p['pat'][pattern])
        assert hp == r['hits_by_pattern'][pattern], (fn, r['name'])
        hits += hp; n_ok += r['n_ok']; solved += bool(r['n_ok'])
        if hp:
            tw += 1; n_distinct_p += sum(1 for p in ps if p['pat'][pattern])
            first_p = min(p['first'] for p in ps if p['pat'][pattern])
            f256_p += first_p <= 256
            for p in ps:
                if p['pat'][pattern]: wh[p['written']] += 1
            per_t.append({'name': r['name'], 'thm': r['thm'], 'hits': hp, 'n_tried': r['n_tried'], 'first': first_p, 'rate': hp / r['n_tried']})
        if r['first_hit'] is not None and r['first_hit'] <= 256:
            f256_any += 1
    per_t.sort(key=lambda x: -x['hits'])
    return {'file': fn, 'n_targets': len(recs), 'n_tried': n_tried, 'n_ok': n_ok, 'solved_any': solved,
            'hits_pattern': hits, 'rate_pattern': hits / n_tried if n_tried else None,
            'targets_with_pattern': tw, 'distinct_pattern_proofs': n_distinct_p,
            'frozen256_pattern_theorems': f256_p, 'frozen256_solved_any': f256_any,
            'pattern_written_hist': dict(sorted(wh.items())), 'per_target': per_t}


def arm_rows(arm, pattern, thr):
    rows = arm_metrics(arm, pattern)
    if not rows:
        return None
    ign = next((x['round'] for x in rows if x['acq_targets_theorems'] >= thr), None)
    last = rows[-1]
    # rounds before any training happened are frozen samples of the Stage-1 model on ALL targets (expert_iter skips
    # training when nothing was found): count them and the pattern hits they contain -> a full-pool pre-training rate.
    pre_rounds = 0
    for x in rows:
        st = json.load(open(f"{arm}/round_{x['round']}.json"))
        if st.get('mix_rl_records', 0) > 0 or x['targets_solved'] > 0:
            break
        pre_rounds += 1
    pre_samples = pre_rounds * rows[0]['attempts'] * last['targets_n']   # attempts per round = k
    pre_hits = rows[pre_rounds - 1]['n_pattern_proofs_targets'] if pre_rounds else 0
    return {'arm': arm, 'rounds_done': last['round'],
            'per_round_pattern_theorems': [x['acq_targets_theorems'] for x in rows],
            'per_round_pattern_proofs': [x['n_pattern_proofs_targets'] for x in rows],
            'per_round_solved': [x['targets_solved'] for x in rows],
            'per_round_transfer_pattern_theorems': [x['acq_transfer_theorems'] for x in rows],
            'first_pattern_round': last['first_round_pattern_targets'],
            'ignition_round': ign, 'untrained_rounds': pre_rounds, 'untrained_samples': pre_samples, 'untrained_pattern_proofs': pre_hits,
            'final_acq': last['acq_targets'], 'final_pattern_theorems': last['acq_targets_theorems'],
            'final_pattern_proofs': last['n_pattern_proofs_targets'], 'final_solved': last['targets_solved'],
            'final_transfer_acq': last['acq_transfer'], 'heldout_greedy': last['heldout_greedy'],
            'examples': last['pattern_examples_targets'][:5]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='artifacts/ign/summary.json')
    ap.add_argument('--covdir', default='artifacts/ign')
    ap.add_argument('--figs', default=None)
    a = ap.parse_args()
    summary = {}
    for sname, S in SETS.items():
        P = S['pattern']; N = S['n']
        cov = {}
        for fn in sorted(glob.glob(f'{a.covdir}/cov_*.s0.jsonl')):
            m = re.search(S['cov_re'], fn)
            if not m: continue
            st = coverage_stats(fn, P)
            if st['n_targets'] < 300:
                st['INCOMPLETE'] = True
            r = st['rate_pattern']
            st['pred_first_round'] = (1 / (r * K * N)) if r else None
            cov[m.group(1)] = st
        arms = {}
        for d in sorted(glob.glob('artifacts/p2/ei_*')):
            m = re.search(S['arm_re'], os.path.basename(d))
            if not m: continue
            ar = arm_rows(d, P, S['thr'])
            if ar: arms[int(m.group(1))] = ar
        ivs = collections.defaultdict(dict)
        for d in sorted(glob.glob('artifacts/p2/ei_*_iv?')):
            m = re.search(S['iv_re'], os.path.basename(d))
            if not m: continue
            ar = arm_rows(d, P, S['thr'])
            if ar: ivs[int(m.group(1))][m.group(2)] = ar
        # join by seed
        table = []
        for s in sorted(set(arms) | {int(re.search(r'_s(\d+)$', k).group(1)) for k in cov if (S['model'](0)[:-1] in k)}):
            model = S['model'](s)
            c = cov.get(model); ar = arms.get(s)
            table.append({'seed': s, 'model': model, 'base_rate': c and c['rate_pattern'], 'base_hits': c and c['hits_pattern'],
                          'base_targets_with_pattern': c and c['targets_with_pattern'], 'frozen256': c and c['frozen256_pattern_theorems'],
                          'pred_first_round': c and c['pred_first_round'],
                          'first_pattern_round': ar and ar['first_pattern_round'], 'ignition_round': ar and ar['ignition_round'],
                          'rounds_done': ar and ar['rounds_done'], 'final_acq': ar and ar['final_acq'],
                          'final_pattern_theorems': ar and ar['final_pattern_theorems'], 'final_solved': ar and ar['final_solved'],
                          'per_round': ar and ar['per_round_pattern_theorems'],
                          'round1_pattern_theorems': ar and ar['per_round_pattern_theorems'][0],  # round 1 = 32 samples per target from the Stage-1 model, all targets
                          'round1_pattern_proofs': ar and ar['per_round_pattern_proofs'][0],
                          'untrained_rounds': ar and ar['untrained_rounds'], 'untrained_samples': ar and ar['untrained_samples'],
                          'interventions': {k: {'ignition_round': v['ignition_round'], 'per_round': v['per_round_pattern_theorems'],
                                                'final_acq': v['final_acq'], 'rounds_done': v['rounds_done']} for k, v in ivs.get(s, {}).items()}})
        # base generalisation over every coverage draw of this pattern (all sets)
        draws = sorted(cov)
        n_with = sum(1 for k in draws if cov[k]['hits_pattern'] > 0)
        lo, hi = clopper_pearson(n_with, len(draws))
        summary[sname] = {'pattern': P, 'n_targets_pool': N, 'ignition_threshold': S['thr'], 'k': K, 'plateau': S['plateau'],
                          'table': table, 'coverage': cov, 'arms': arms, 'interventions': ivs,
                          'base_generalisation': {'draws': draws, 'n_draws': len(draws), 'n_with_pattern': n_with,
                                                  'fraction': n_with / len(draws) if draws else None, 'ci95': [lo, hi],
                                                  'rates_when_present': {k: cov[k]['rate_pattern'] for k in draws if cov[k]['hits_pattern'] > 0}}}
        print(f'== {sname}: {len(cov)} coverage files, {len(arms)} arms, {sum(len(v) for v in ivs.values())} intervention arms; '
              f'base generalisation {n_with}/{len(draws)} (95% CI {lo:.2f}-{hi:.2f})')
        print('seed  base_rate   hits  tgts  frz256  pred_r1  r1hits  first  ign  r8_acq  per_round')
        for t in table:
            br = t['base_rate']
            print(f"{t['seed']:>4}  {('-' if br is None else f'{br:.2e}'):>9}  {t['base_hits'] if t['base_hits'] is not None else '-':>5}  "
                  f"{t['base_targets_with_pattern'] if t['base_targets_with_pattern'] is not None else '-':>4}  {t['frozen256'] if t['frozen256'] is not None else '-':>6}  "
                  f"{(f'{t['pred_first_round']:.2f}' if t['pred_first_round'] else 'inf') if br is not None else '-':>7}  "
                  f"{t['round1_pattern_theorems'] if t['round1_pattern_theorems'] is not None else '-':>6}  {t['first_pattern_round'] if t['first_pattern_round'] is not None else '-':>5}  {t['ignition_round'] if t['ignition_round'] is not None else '-':>3}  "
                  f"{(f'{t['final_acq']:.3f}' if t['final_acq'] is not None else '-'):>6}  {t['per_round']}  {t['interventions'] and {k: v['per_round'] for k, v in t['interventions'].items()}}")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(summary, open(a.out, 'w'), indent=1)
    write_tables(summary, a.out.replace('.json', '_table.md'), a.out.replace('.json', '_examples.md'))
    if a.figs:
        import ignition_figures
        ignition_figures.make_all(summary, a.figs)



def write_tables(summary, out_md, out_ex):
    """Markdown tables (per seed; interventions) and five printed example proofs per pattern set."""
    L = []
    for sname, S in summary.items():
        L.append(f"\n### {sname} (pool n = {S['n_targets_pool']}, k = {S['k']}, ignition threshold {S['ignition_threshold']} theorems)\n")
        L.append('| seed | pre-RL rate r (600k) | r hits | targets w/ pattern | frozen@256 | 1/(r·k·N) | round-1 hits (32×N) | first pattern round | ignition round | round-8 pattern theorems / proofs | solved r8 | acq r8 |')
        L.append('|---|---|---|---|---|---|---|---|---|---|---|---|')
        for t in S['table']:
            br = t['base_rate']; c = S['coverage'].get(t['model'], {})
            inc = ' (incomplete)' if c.get('INCOMPLETE') else ''
            ar = S['arms'].get(t['seed'])
            L.append(f"| {t['seed']} | {('-' if br is None else f'{br:.1e}') + inc} | {t['base_hits'] if t['base_hits'] is not None else '-'} | {t['base_targets_with_pattern'] if t['base_targets_with_pattern'] is not None else '-'} | {t['frozen256'] if t['frozen256'] is not None else '-'} | "
                     f"{('∞' if not t['pred_first_round'] else f'{t['pred_first_round']:.1f}') if br is not None else '-'} | {t['round1_pattern_theorems'] if t['round1_pattern_theorems'] is not None else '-'} | "
                     f"{t['first_pattern_round'] if t['first_pattern_round'] is not None else 'never'} | {t['ignition_round'] if t['ignition_round'] is not None else 'never'} | "
                     f"{ar['final_pattern_theorems'] if ar else '-'} / {ar['final_pattern_proofs'] if ar else '-'} | {t['final_solved'] if t['final_solved'] is not None else '-'} | {('%.3f' % t['final_acq']) if t['final_acq'] is not None else '-'} |")
        ivrows = [(t, iv, v) for t in S['table'] for iv, v in sorted((t.get('interventions') or {}).items())]
        if ivrows:
            L.append(f"\n**Interventions ({sname})**, from the round-4 state; per-round cumulative pattern theorems (rounds 5–8); attempts per target: parent 4×32 = 128 before, then K: 128 + 3×32, T/S: 4×32.\n")
            L.append('| seed | parent r1–8 | intervention | r5–8 pattern theorems | ignition round | r8 acquisition |')
            L.append('|---|---|---|---|---|---|')
            for t, iv, v in ivrows:
                L.append(f"| {t['seed']} | {t['per_round']} | {iv} | {v['per_round']} | {v['ignition_round'] if v['ignition_round'] is not None else 'never'} | {('%.3f' % v['final_acq']) if v['final_acq'] is not None else '-'} |")
        bg = S['base_generalisation']
        L.append(f"\nBase generalisation ({sname}): {bg['n_with_pattern']} of {bg['n_draws']} Stage-1 draws produce ≥ 1 pattern sample in 600k pre-RL samples; 95 % CI {bg['ci95'][0]:.2f}–{bg['ci95'][1]:.2f}.")
        r1 = [t for t in S['table'] if t['round1_pattern_theorems'] is not None]
        n1 = sum(1 for t in r1 if t['round1_pattern_theorems'] > 0)
        lo, hi = clopper_pearson(n1, len(r1))
        L.append(f"Round-1 (32 × N pre-training samples): {n1} of {len(r1)} arms had ≥ 1 pattern theorem; 95 % CI {lo:.2f}–{hi:.2f}.")
    open(out_md, 'w').write('\n'.join(L) + '\n')
    E = []
    for sname, S in summary.items():
        E.append(f"\n## {sname}: five pattern proofs found by EI (normalised), with round\n")
        n = 0
        for seed, ar in sorted(S['arms'].items()):
            for ex in ar['examples']:
                if n >= 5: break
                E.append(f"- seed {seed}, round {ex['round']}, `{ex['thm']}` ({ex['written']} lines):\n\n```\n{ex['proof']}\n```\n"); n += 1
            if n >= 5: break
        E.append(f"\n## {sname}: pre-RL pattern samples (top targets by hit count, per model)\n")
        for m, c in sorted(S['coverage'].items()):
            for pt in c['per_target'][:2]:
                E.append(f"- {m}: `{pt['thm']}` hits {pt['hits']}/{pt['n_tried']} (first at sample {pt['first']})")
    open(out_ex, 'w').write('\n'.join(E) + '\n')


if __name__ == '__main__':
    main()
