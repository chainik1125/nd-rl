"""Confusion tables and the calibration table from calibrate.py logs.

    python analyze.py logs/cal_cheap.jsonl logs/cal_full.jsonl [--write-calib]

Simulates estimate.py's decision on each ground-truth record: the exact half is `full_cap1m` (found -> exact
minimum, exhausted to E), the hunters are the other variants. At floor f the estimator claims ">= f" iff the
exact half found nothing < f and no hunter found a proof < f. Truth: long iff true_min >= f (or cert_ge >= f).
"""
import argparse
import collections
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
HUNTERS = ['none_d2', 'none', 'lit', 'sub']
EXACT = 'full_cap1m'


def wilson_upper(k, n, z=1.96):
    if n == 0:
        return 1.0
    p = k / n
    den = 1 + z * z / n
    centre = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return min(1.0, (centre + half) / den)


def load(paths, labels=()):
    by = collections.defaultdict(dict)
    meta = {}
    for p in paths:
        for line in open(p):
            d = json.loads(line)
            by[d['key']][d['variant']] = d
            meta[d['key']] = {k: d[k] for k in ('true_min', 'cert_ge', 'src', 'prompt', 'n_prem', 'S')}
    for p in labels:   # exact_label.py output: exact minima for records whose minlen run had timed out
        for line in open(p):
            d = json.loads(line)
            if d.get('min_body') is not None and d['key'] in meta and meta[d['key']]['true_min'] is None:
                meta[d['key']]['true_min'] = d['min_body']
                meta[d['key']]['src'] += '+exact_label'
    return by, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('paths', nargs='+')
    ap.add_argument('--write-calib', action='store_true')
    ap.add_argument('--hunters', nargs='*', default=HUNTERS)
    ap.add_argument('--labels', nargs='*', default=[])
    a = ap.parse_args()
    by, meta = load(a.paths, a.labels)
    hunters = a.hunters
    exact = [k for k in by if meta[k]['true_min'] is not None]
    print(f'{len(by)} records, {len(exact)} with exact minima; variants seen:',
          sorted({v for k in by for v in by[k]}))

    # 1. per-level miss rates of each hunter (did it find ANY proof <= 19 when a proof of length m exists?)
    print('\n== per-level hunter miss rates (true_min = m; miss = found nothing; "long" = found but > m)')
    hdr = f'{"m":>3} {"n":>4} ' + ' '.join(f'{h:>16}' for h in hunters) + f' {"ANY-hunter miss":>16} {"wilson95 ub":>12}'
    print(hdr)
    miss_ub = {}
    for m in sorted({meta[k]['true_min'] for k in exact}):
        ks = [k for k in exact if meta[k]['true_min'] == m]
        cells, any_miss, n_any = [], 0, 0
        for h in hunters:
            runs = [by[k][h] for k in ks if h in by[k]]
            miss = sum(r['found'] is None for r in runs)
            longer = sum(r['found'] is not None and r['found'] > m for r in runs)
            cells.append(f'{miss:>3}/{len(runs):<3} long {longer:<3}')
        for k in ks:
            runs = [by[k][h] for h in hunters if h in by[k]]
            if runs:
                n_any += 1
                any_miss += all(r['found'] is None for r in runs)
        ub = wilson_upper(any_miss, n_any)
        miss_ub[m] = ub
        print(f'{m:>3} {len(ks):>4} ' + ' '.join(f'{c:>16}' for c in cells) + f' {any_miss:>7}/{n_any:<8} {ub:>12.3f}')

    # 2. per-floor confusion of the simulated estimator
    print('\n== simulated estimator per floor (exact half = full_cap1m, then hunters)')
    print(f'{"floor":>5} {"n":>4} {"TP":>4} {"FP":>4} {"FN":>4} {"TN":>4} {"FP rate":>8} {"recall":>7} {"exact-certs":>11}')
    for f in (10, 12, 13, 14, 15, 16, 17, 18):
        tp = fp = fn = tn = ex = 0
        for k in by:
            tm, cg = meta[k]['true_min'], meta[k]['cert_ge']
            if tm is None and not (cg and cg >= f):
                continue                      # truth unknown at this floor
            long = tm is None or tm >= f
            e = by[k].get(EXACT)
            if e is None:
                continue
            if e['found'] is not None and e['found'] < f:
                claim = False
            elif e['exh'] >= f - 1:
                claim = True; ex += 1
            else:
                claim = not any(h in by[k] and by[k][h]['found'] is not None and by[k][h]['found'] < f for h in hunters)
            if claim and long: tp += 1
            elif claim and not long: fp += 1
            elif not claim and long: fn += 1
            else: tn += 1
        n = tp + fp + fn + tn
        print(f'{f:>5} {n:>4} {tp:>4} {fp:>4} {fn:>4} {tn:>4} {fp / max(1, fp + tn):>8.3f} {tp / max(1, tp + fn):>7.3f} {ex:>11}')

    # 3. what the hunters found on records the exact search could not label
    print('\n== records with no exact label (minlen timed out at its floor): hunter upper bounds')
    for k in by:
        if meta[k]['true_min'] is None:
            founds = {h: by[k][h]['found'] for h in hunters if h in by[k]}
            e = by[k].get(EXACT, {})
            print(f"  src={meta[k]['src']:<15} S={meta[k]['S']:>3} exact_exh={e.get('exh')} hunters={founds}")

    # 4. cost
    print('\n== mean secs per variant, by true_min bucket')
    for h in [EXACT] + hunters:
        buckets = collections.defaultdict(list)
        for k in by:
            if h in by[k]:
                tm = meta[k]['true_min']
                buckets['none' if tm is None else ('<10' if tm < 10 else '10-13' if tm < 14 else '14+')].append(by[k][h]['secs'])
        print(f'  {h:<12} ' + '  '.join(f'{b}: {sum(v) / len(v):6.2f}s (n={len(v)})' for b, v in sorted(buckets.items())))

    # 5. miss rate by (floor, slack = floor - m): does any hunter find a proof < floor when m = floor - slack?
    print('\n== ANY-hunter miss by slack g = floor - m (cells: misses/n), pooled column at the right')
    floors = list(range(11, 20))
    pooled = collections.defaultdict(lambda: [0, 0])
    print(f'{"g":>2} ' + ' '.join(f'{"f=" + str(f):>8}' for f in floors) + f' {"pooled":>9} {"wilson ub":>9}')
    miss_slack = {}
    for g in range(1, 9):
        cells = []
        for f in floors:
            m = f - g
            ks = [k for k in exact if meta[k]['true_min'] == m and any(h in by[k] for h in hunters)]
            miss = sum(not any(h in by[k] and by[k][h]['found'] is not None and by[k][h]['found'] < f for h in hunters) for k in ks)
            cells.append(f'{miss}/{len(ks)}' if ks else '-')
            pooled[g][0] += miss; pooled[g][1] += len(ks)
        k_, n_ = pooled[g]
        miss_slack[g] = wilson_upper(k_, n_)
        print(f'{g:>2} ' + ' '.join(f'{c:>8}' for c in cells) + f' {k_:>4}/{n_:<4} {miss_slack[g]:>9.3f}')

    # 5a. per-hunter miss by slack (the hunter alone finds nothing < floor when m = floor - g); pooled over floors
    #     11..19, only counting runs that REACHED level floor-1 (exhausted or found), so budget misses are excluded.
    print('\n== per-hunter miss by slack g (misses/n over runs that reached floor-1), Wilson-95% upper bounds')
    per_h = {}
    for h in hunters:
        per_h[h] = {}
        cells = []
        for g in range(1, 9):
            k_ = n_ = 0
            for f in floors:
                m = f - g
                for k in exact:
                    if meta[k]['true_min'] != m or h not in by[k]:
                        continue
                    r = by[k][h]
                    reached = r['found'] is not None or r['exh'] >= f - 1
                    if not reached:
                        continue
                    n_ += 1
                    k_ += not (r['found'] is not None and r['found'] < f)
            per_h[h][g] = wilson_upper(k_, n_)
            cells.append(f'{k_}/{n_}={per_h[h][g]:.3f}')
        print(f'  {h:<8} ' + ' '.join(f'{c:>14}' for c in cells))

    # 5b. excess length of each hunter's find over the exact minimum (how many lines lemma-sharing saved).
    #     P(excess >= g) is the miss probability of that hunter alone at slack g, if it reaches the level.
    print('\n== excess = found - true_min, per hunter (records with true_min >= 12 where the hunter found a proof)')
    for h in hunters:
        ex = collections.Counter(by[k][h]['found'] - meta[k]['true_min'] for k in exact
                                 if meta[k]['true_min'] >= 12 and h in by[k] and by[k][h]['found'] is not None)
        n = sum(ex.values())
        tail = {g: round(sum(v for e, v in ex.items() if e >= g) / n, 3) for g in range(1, 5)} if n else {}
        print(f'  {h:<8} n={n:<4} hist={dict(sorted(ex.items()))}  P(excess>=g)={tail}')

    # 6. prior: how fast does the exact-minimum distribution decay above 10 (for the posterior p_false)?
    ms = sorted(meta[k]['true_min'] for k in exact if meta[k]['true_min'] >= 11)
    cnt = collections.Counter(ms)
    ratios = [cnt[m + 1] / cnt[m] for m in range(11, 16) if cnt[m]]
    decay = sum(ratios) / len(ratios) if ratios else None
    print(f'\n== exact-min histogram >= 11: {dict(sorted(cnt.items()))}; mean level-to-level ratio 11..16 = {decay}')

    if a.write_calib:
        out = {'miss_ub': {str(m): round(v, 4) for m, v in miss_ub.items()},
               'miss_ub_slack': {str(g): round(v, 4) for g, v in miss_slack.items()},
               'miss_ub_slack_by_hunter': {h: {str(g): round(v, 4) for g, v in t.items()} for h, t in per_h.items()},
               'prior_decay': round(decay, 3) if decay else None,
               'hunters': hunters, 'note': 'Wilson 95% upper bounds of P(all hunters miss), by true min m and by slack floor-m; from analyze.py'}
        json.dump(out, open(os.path.join(HERE, 'calib.json'), 'w'), indent=1)
        print('\nwrote calib.json', out['miss_ub'])


if __name__ == '__main__':
    main()
