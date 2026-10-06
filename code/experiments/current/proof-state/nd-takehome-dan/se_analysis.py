#!/usr/bin/env python3
"""Numbers for run `state-env`: the ladder tables, the textbook slice, term sizes, the env diagnostics.

  python3 se_analysis.py --runs artifacts/se/la_T1_S_s0 artifacts/se/la_frozen_S_s0 ... \
      --c0 artifacts/dsg/la_T1_c0_s0 artifacts/dsg/la_frozen_c0_s0 ... --out artifacts/se/summary.json [--term_size]

Every count is re-derived from the run's own `found_transfer_<r>.jsonl` and `data/ladder/transfer.jsonl`, not from the
round json's summary fields, so the two can be compared (they must agree).  `L*` = max L with >= 5 transfer theorems
solved at `L_true` >= L.  `L_true` is the pool's `n_lines`, which is **ND-derived and an upper bound under Lean**.
`--term_size` runs `lean_check` over the accepted proofs and reports the elaborated term size beside line counts.
"""
import argparse, collections, json, os, subprocess, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def read(fn):
    return [json.loads(l) for l in open(fn) if l.strip()]


def git_json(path):
    return json.loads(subprocess.check_output(['git', 'show', f'HEAD:{path}']))


def git_lines(path):
    return [json.loads(l) for l in subprocess.check_output(['git', 'show', f'HEAD:{path}']).decode().splitlines() if l.strip()]


def lstar(names, recs, need=5):
    solved = [r['n_lines'] for r in recs if r['name'] in names]
    ge = {L: sum(1 for x in solved if x >= L) for L in range(2, 21)}
    return max([L for L, c in ge.items() if c >= need], default=0), ge


def derive(found_rows, recs):
    """found_rows: records of found_transfer_<r>.jsonl -> the counts, re-derived."""
    by_name = collections.defaultdict(list)
    for x in found_rows:
        by_name[x['name']].append(x)
    names = set(by_name)
    ls, ge = lstar(names, recs)
    tb = [r for r in recs if r.get('source') == 'textbook']
    tb_sch = collections.Counter()
    tb_tot = collections.Counter()
    for r in tb:
        tb_tot[r['schema']] += 1
        if r['name'] in names:
            tb_sch[r['schema']] += 1
    bins = collections.defaultdict(lambda: [0, 0])
    for r in recs:
        bins[r['n_lines']][0] += r['name'] in names
        bins[r['n_lines']][1] += 1
    wl = collections.Counter(x['written'] for xs in by_name.values() for x in xs)
    pl = collections.Counter(x['pruned'] for xs in by_name.values() for x in xs)
    ge13 = sorted([(r['name'], r['n_lines'], r.get('schema'), r.get('source')) for r in recs
                   if r['n_lines'] >= 13 and r['name'] in names])
    return {'solved': len(names), 'n': len(recs), 'lstar': ls, 'ge': {str(k): v for k, v in ge.items()},
            'solved_ge13': ge.get(13, 0), 'solved_ge13_names': ge13,
            'by_bin': {str(L): {'solved': k, 'n': n} for L, (k, n) in sorted(bins.items())},
            'distinct_proofs': sum(len(v) for v in by_name.values()),
            'textbook': {'solved': sum(1 for r in tb if r['name'] in names), 'n': len(tb),
                         'schemata_solved': sum(1 for s in tb_tot if tb_sch[s]), 'schemata': len(tb_tot),
                         'per_schema': {s: [tb_sch[s], tb_tot[s]] for s in sorted(tb_tot)}},
            'written_hist': {str(k): v for k, v in sorted(wl.items())},
            'pruned_hist': {str(k): v for k, v in sorted(pl.items())}}


def term_sizes(found_rows, recs, workers, cap=4000):
    """elaborated Lean term size of the accepted transfer proofs (one shortest proof per theorem)."""
    import nd2lean
    from lean_check import check
    L = {r['name']: r['n_lines'] for r in recs}
    best = {}
    for x in found_rows:
        k = x['name']
        if k not in best or x['written'] < best[k]['written']:
            best[k] = x
    items = sorted(best.values(), key=lambda x: -L[x['name']])[:cap]
    srcs, keep = [], []
    for x in items:
        try:
            srcs.append(nd2lean.translate(x['prompt'], x['proof'], require_all_pr=False)); keep.append(x)
        except Exception:
            pass
    res, wall, proc = check(srcs, workers=workers)
    by = collections.defaultdict(list)
    bad = 0
    for x, r in zip(keep, res):
        if r['ok'] and r['size'] is not None:
            by[L[x['name']]].append((r['size'], x['written']))
        else:
            bad += 1
    out = {'n_checked': len(keep), 'rejected_by_lean_check': bad, 'lean_wall_s': wall,
           'by_L_true': {}}
    for Lt in sorted(by):
        sz = sorted(s for s, _ in by[Lt]); ln = sorted(w for _, w in by[Lt])
        out['by_L_true'][str(Lt)] = {'n': len(sz), 'term_size_min': sz[0], 'term_size_median': sz[len(sz) // 2],
                                     'term_size_max': sz[-1], 'lines_min': ln[0], 'lines_median': ln[len(ln) // 2],
                                     'lines_max': ln[-1]}
    return out


def one(run, recs, rounds, from_git, term, workers):
    g = git_lines if from_git else read
    gj = git_json if from_git else (lambda p: json.loads(open(p).read()))
    out = {'dir': run, 'rounds': {}}
    last = None
    for r in range(1, rounds + 1):
        p = f'{run}/round_{r}.json'
        try:
            d = gj(p)
        except Exception:
            continue
        last = r
        row = {'secs': d.get('secs'), 'transfer_sample_acc': d.get('transfer_sample_acc'),
               'target_sample_acc': d.get('target_sample_acc'),
               'transfer_greedy_solved': d['transfer_greedy']['solved'],
               'heldout_greedy_rate': d['heldout_greedy']['rate'],
               'transfer_cum_reported': {'solved': d['transfer_cum']['solved'], 'lstar': d['transfer_cum']['lstar'],
                                         'ge13': d['transfer_cum']['ge'].get('13', d['transfer_cum']['ge'].get(13))},
               'targets_cum_reported': {'solved': d['targets_cum']['solved'], 'lstar': d['targets_cum']['lstar']}}
        k = d.get('k', 32)
        n_tr = d['transfer_round']['n'] if 'transfer_round' in d else 2285
        n_ho = d['heldout_greedy']['n']
        att = d.get('target_samples', 0) + k * n_tr + n_tr + n_ho
        acc = (round(d.get('target_sample_acc', 0) * d.get('target_samples', 0))
               + round(d.get('transfer_sample_acc', 0) * k * n_tr)
               + d['transfer_greedy']['solved'] + d['heldout_greedy']['solved'])
        row['attempts'] = att
        row['accepted_by_lean'] = acc
        if 'env' in d:
            e = d['env']
            row['env'] = {k: e.get(k) for k in ('env_end', 'env_fail_reason', 'env_waves', 'env_wall_s',
                                                'peak_alloc_gb', 'action_declen_mean', 'rows', 'rowsteps')}
            st = e.get('env_steps') or {}
            tot = sum(st.values()) or 1
            row['env']['mean_steps_per_attempt'] = sum(int(k) * v for k, v in st.items()) / tot
            dl = e.get('action_declen_hist') or {}
            n = sum(dl.values()) or 1
            row['env']['action_trunc_frac'] = sum(v for kk, v in dl.items() if int(kk) >= 256) / n
            ee = e.get('env_end') or {}
            tot = sum(ee.values()) or 1
            row['env']['end_frac'] = {kk: v / tot for kk, v in ee.items()}
            row['env']['ended_by_lean_reject'] = ee.get('done', 0) - acc
            row['env']['ended_by_lean_reject_frac'] = (ee.get('done', 0) - acc) / tot
        out['rounds'][str(r)] = row
    if last is None:
        return None
    out['last_round'] = last
    fr = g(f'{run}/found_transfer_{last}.jsonl')
    out['derived'] = derive(fr, recs)
    rep = out['rounds'][str(last)]['transfer_cum_reported']
    out['derived_matches_reported'] = (rep['solved'] == out['derived']['solved'] and rep['lstar'] == out['derived']['lstar'])
    if term:
        out['term_size'] = term_sizes(fr, recs, workers)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='*', default=[])
    ap.add_argument('--c0', nargs='*', default=[])
    ap.add_argument('--transfer', default='data/ladder/transfer.jsonl')
    ap.add_argument('--rounds', type=int, default=8)
    ap.add_argument('--term_size', action='store_true')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--out', default='artifacts/se/summary.json')
    a = ap.parse_args()
    recs = read(a.transfer)
    res = {'utc': time.strftime('%FT%TZ', time.gmtime()), 'transfer_pool': a.transfer, 'n_transfer': len(recs),
           'L_true_note': 'L_true = the pool n_lines, ND-derived; an UPPER BOUND under Lean',
           'runs': {}, 'c0': {}}
    for r in a.runs:
        o = one(r, recs, a.rounds, False, a.term_size, a.workers)
        if o:
            res['runs'][os.path.basename(r)] = o
    for r in a.c0:
        o = one(r, recs, a.rounds, True, a.term_size, a.workers)
        if o:
            res['c0'][os.path.basename(r)] = o
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    json.dump(res, open(a.out, 'w'), indent=1)
    hdr = f'{"run":28} {"last":>4} {"solved":>7} {"L*":>3} {"ge13":>5} {"greedy":>7} {"held":>6} {"tb":>8} {"steps":>6} {"synt%":>6}'
    print(hdr)
    for grp in ('runs', 'c0'):
        for k, o in res[grp].items():
            d = o['derived']; lr = o['rounds'][str(o['last_round'])]
            e = lr.get('env') or {}
            ee = e.get('env_end') or {}
            tot = sum(ee.values()) or 1
            print(f'{k:28} {o["last_round"]:>4} {d["solved"]:>7} {d["lstar"]:>3} {d["solved_ge13"]:>5} '
                  f'{lr["transfer_greedy_solved"]:>7} {lr["heldout_greedy_rate"]:>6.3f} '
                  f'{d["textbook"]["solved"]:>3}/{d["textbook"]["n"]:<4} '
                  f'{e.get("mean_steps_per_attempt", float("nan")):>6.2f} {100*ee.get("syntax",0)/tot:>6.1f}'
                  + ('' if o['derived_matches_reported'] else '   !! derived != reported'))


if __name__ == '__main__':
    main()
