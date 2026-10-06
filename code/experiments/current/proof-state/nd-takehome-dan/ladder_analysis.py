#!/usr/bin/env python3
"""ladder-A leaderboard: re-derives every number from the pulled files.

  python ladder_analysis.py [--arms la_frozen_s0 la_T1_s0 ...] --out ladder.md --json artifacts/ladder/summary.json

Per arm (artifacts/ladder/<arm>/): found_transfer_<R>.jsonl and found_<R>.jsonl (found_union_<R>.jsonl for T6
siblings), novelty_proofs.jsonl (base log p at T = 0.8 under stage1_abs.pt, marginalised over start index).
Counts: distinct proofs after start-index normalisation (normalize.norm; re-applied here), theorems solved per L_true
bin (L_true = the pool record's n_lines) with Wilson 95 % intervals, L* = max L with >= 5 theorems solved at
L_true >= L, base-reachability split of the counted proofs (p >= 1e-5: elicitation; else creation; also the 1/256 line).
"""
import argparse, json, os, sys, glob, collections, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from normalize import norm


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)

RUNGS = {'frozen': 'frozen control', 'T1': 'EI baseline', 'T2': 'EI + difficulty-weighted sampling', 'T3': 'EI + hindsight relabelling',
         'T4': 'EI + moving target window', 'T5': 'EI + precursor injection', 'T6': 'EI + sibling ensemble (2 x k=16)'}
BINS = list(range(7, 15))


def rd(fn):
    return [json.loads(l) for l in open(fn) if l.strip()]


def load_found(fn, pool):
    """-> name -> set(norm proofs); verify every proof string is distinct after normalisation."""
    by = collections.defaultdict(set)
    for x in rd(fn):
        by[x['name']].add(norm(x['proof']))
        SHORTEST[x['name']] = min(SHORTEST.get(x['name'], 99), x['written'])
    return by


SHORTEST = {}   # theorem name -> shortest verifier-accepted written proof seen in any analysed arm (either pool)


def mcnemar(b, c):
    """exact two-sided sign test on discordant pairs (b: arm only, c: reference only)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def lstar_relabelled(solved, pool):
    """L* with every label replaced by min(label, shortest accepted written proof seen anywhere)."""
    ge = {L: sum(1 for r in pool if r['name'] in solved and min(r['n_lines'], SHORTEST.get(r['name'], 99)) >= L) for L in range(7, 16)}
    return max([L for L, c in ge.items() if c >= 5], default=0), ge


def analyse_pool(found, pool, nov=None, ref_solved=None):
    Lt = {r['name']: r['n_lines'] for r in pool}
    solved = {n for n in found if found[n]}
    res = {'n': len(pool), 'solved': len(solved), 'distinct_proofs': sum(len(v) for v in found.values()), '_solved': solved}
    ge = {L: sum(1 for n in solved if Lt[n] >= L) for L in range(7, 16)}
    res['ge'] = ge
    res['lstar'] = max([L for L, c in ge.items() if c >= 5], default=0)
    bins = {}
    for L in BINS:
        n = sum(1 for r in pool if r['n_lines'] == L)
        k = sum(1 for r in pool if r['n_lines'] == L and r['name'] in solved)
        lo, hi = wilson(k, n)
        bins[L] = {'solved': k, 'n': n, 'rate': k / n if n else None, 'ci': [lo, hi]}
    res['bins'] = bins
    # by source
    src = collections.defaultdict(lambda: [0, 0])
    for r in pool:
        src[r.get('source')][0] += r['name'] in solved; src[r.get('source')][1] += 1
    res['by_source'] = {s: {'solved': k, 'n': n} for s, (k, n) in src.items()}
    sch = collections.defaultdict(lambda: [0, 0])
    for r in pool:
        if r.get('schema'):
            sch[r['schema']][0] += r['name'] in solved; sch[r['schema']][1] += 1
    res['by_schema'] = {k: {'solved': v[0], 'n': v[1]} for k, v in sorted(sch.items())}
    # theorems solved at L_true >= 9 by schema
    res['schemas_ge9'] = dict(collections.Counter(r['schema'] for r in pool if r['name'] in solved and r['n_lines'] >= 9 and r.get('schema')))
    res['gen_ge9'] = sum(1 for r in pool if r['name'] in solved and r['n_lines'] >= 9 and r.get('source') == 'gen')
    # generator-only L*
    ge_gen = {L: sum(1 for r in pool if r['name'] in solved and r['n_lines'] >= L and r.get('source') == 'gen') for L in range(7, 16)}
    res['lstar_gen_only'] = max([L for L, c in ge_gen.items() if c >= 5], default=0)
    if ref_solved is not None:
        res['solved_not_in_ref'] = len(solved - ref_solved)
        res['ge8_not_in_ref'] = sum(1 for n in solved - ref_solved if Lt[n] >= 8)
    if nov is not None:
        # base reachability per theorem: max over its counted proofs of base p (T=0.8)
        best = collections.defaultdict(lambda: -1e9)
        for x in nov:
            best[x['name']] = max(best[x['name']], x['base_logp_T08'])
        reach = {}
        for L in (7, 8, 9, 10, 11, 12):
            names = [n for n in solved if Lt[n] >= L]
            scored = [n for n in names if n in best]
            reach[L] = {'theorems': len(names), 'scored': len(scored),
                        'elicit_1e-5': sum(1 for n in scored if best[n] >= math.log(1e-5)),
                        'elicit_1/256': sum(1 for n in scored if best[n] >= math.log(1 / 256))}
        res['reachability'] = reach
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='artifacts/ladder'); ap.add_argument('--round', type=int, default=8)
    ap.add_argument('--arms', nargs='*', default=None); ap.add_argument('--out', default='ladder.md'); ap.add_argument('--json', default='artifacts/ladder/summary.json')
    ap.add_argument('--pools', default='data/ladder')
    a = ap.parse_args()
    transfer, targets = rd(f'{a.pools}/transfer.jsonl'), rd(f'{a.pools}/rl_targets.jsonl')
    arms = a.arms or sorted(os.path.basename(d) for d in glob.glob(f'{a.dir}/la_*') if os.path.isdir(d))
    summ = {}
    frozen_solved = {}
    for arm in arms:
        d = f'{a.dir}/{arm}'
        ft = f'{d}/found_transfer_{a.round}.jsonl'
        if not os.path.exists(ft):
            continue
        fu = f'{d}/found_union_{a.round}.jsonl'
        fg = fu if os.path.exists(fu) else f'{d}/found_{a.round}.jsonl'
        nov = rd(f'{d}/novelty_proofs.jsonl') if os.path.exists(f'{d}/novelty_proofs.jsonl') else None
        nov_t = [x for x in nov if x['src'] == 'transfer'] if nov else None
        nov_g = [x for x in nov if x['src'] == 'targets'] if nov else None
        rung = arm.split('_')[1]; seed = arm.split('_s')[-1]
        ref = frozen_solved.get(seed)
        res = {'rung': rung, 'seed': seed, 'transfer': analyse_pool(load_found(ft, transfer), transfer, nov_t, ref), 'targets': analyse_pool(load_found(fg, targets), targets, nov_g),
               'union': os.path.exists(fu)}
        rs = [json.load(open(f)) for f in sorted(glob.glob(f'{d}/round_*.json'), key=lambda f: int(f.split('_')[-1][:-5]))]
        res['rounds'] = len(rs); res['secs'] = sum(r['secs'] for r in rs); res['heldout_greedy_final'] = rs[-1]['heldout_greedy']['rate'] if rs else None
        res['transfer_greedy_final'] = rs[-1]['transfer_greedy']['rate'] if rs else None
        res['attempts_per_target'] = rs[-1]['targets_cum'].get('attempts_per_theorem') if rs else None
        res['lstar_by_round'] = [(r['round'], r['transfer_cum']['lstar'], r['targets_cum']['lstar']) for r in rs]
        if rung == 'frozen':
            frozen_solved[seed] = {n for n, v in load_found(ft, transfer).items() if v}
        summ[arm] = res
    # ---- T6: the pre-registered arm is the sibling pair (transfer: union of both siblings' 8 x 16 samples = 256 attempts;
    # targets: found_union). The single siblings stay in the json as rung 'T6sib' (128 transfer attempts each).
    for seed in ('0', '1'):
        sa, sb = f'la_T6_s{seed}a', f'la_T6_s{seed}b'
        if sa in summ and sb in summ:
            d = a.dir
            ft = collections.defaultdict(set)
            for arm in (sa, sb):
                for n, v in load_found(f'{d}/{arm}/found_transfer_{a.round}.jsonl', transfer).items():
                    ft[n] |= v
            nov = [x for arm in (sa, sb) if os.path.exists(f'{d}/{arm}/novelty_proofs.jsonl') for x in rd(f'{d}/{arm}/novelty_proofs.jsonl')]
            res = {'rung': 'T6', 'seed': seed, 'union': True,
                   'transfer': analyse_pool(ft, transfer, [x for x in nov if x['src'] == 'transfer'] or None, frozen_solved.get(seed)),
                   'targets': summ[sa]['targets'], 'rounds': min(summ[sa]['rounds'], summ[sb]['rounds']), 'secs': summ[sa]['secs'] + summ[sb]['secs'],
                   'heldout_greedy_final': min(summ[sa]['heldout_greedy_final'], summ[sb]['heldout_greedy_final']),
                   'transfer_greedy_final': None, 'attempts_per_target': summ[sa]['attempts_per_target'] + summ[sb]['attempts_per_target'],
                   'lstar_by_round': []}
            summ[f'la_T6_s{seed}'] = res
            for arm in (sa, sb):
                summ[arm]['rung'] = 'T6sib'
    # ---- robustness: labels contradicted by a shorter accepted proof (any arm, either pool)
    allrec = {r['name']: r for r in transfer + targets}
    contradicted = {n: (allrec[n]['n_lines'], m) for n, m in SHORTEST.items() if m < allrec[n]['n_lines']}
    for arm, r in summ.items():
        for pool_name, pool in (('transfer', transfer), ('targets', targets)):
            ls, ge = lstar_relabelled(r[pool_name]['_solved'], pool)
            r[pool_name]['lstar_relabelled'] = ls; r[pool_name]['ge_relabelled'] = ge
    # ---- paired comparison with T1 (same seed) on transfer theorems
    Ltr = {r['name']: r['n_lines'] for r in transfer}
    for arm, r in summ.items():
        ref = summ.get(f"la_T1_s{r['seed']}")
        if ref is None or r['rung'] in ('T1', 'frozen', 'T6sib'):
            continue
        A, B = r['transfer']['_solved'], ref['transfer']['_solved']
        r['vs_T1'] = {}
        for L in (7, 9, 10):
            b = sum(1 for n in A - B if Ltr[n] >= L); c = sum(1 for n in B - A if Ltr[n] >= L)
            r['vs_T1'][L] = {'arm_only': b, 'T1_only': c, 'p': mcnemar(b, c)}
    if 'la_T1_s0' in summ and 'la_T1_s1' in summ:     # seed-to-seed noise of the reference itself
        A, B = summ['la_T1_s1']['transfer']['_solved'], summ['la_T1_s0']['transfer']['_solved']
        summ['la_T1_s1']['vs_T1'] = {L: {'arm_only': sum(1 for n in A - B if Ltr[n] >= L), 'T1_only': sum(1 for n in B - A if Ltr[n] >= L)} for L in (7, 9, 10)}
        for L in (7, 9, 10):
            v = summ['la_T1_s1']['vs_T1'][L]; v['p'] = mcnemar(v['arm_only'], v['T1_only'])
    for r in summ.values():
        r['transfer'].pop('_solved', None); r['targets'].pop('_solved', None)
    summ = dict(sorted(summ.items(), key=lambda kv: (kv[1]['rung'] != 'frozen', kv[1]['rung'], kv[1]['seed'])))
    os.makedirs(os.path.dirname(a.json), exist_ok=True)
    json.dump({'arms': summ, 'labels_contradicted': contradicted}, open(a.json + '.full', 'w'), indent=1)
    json.dump(summ, open(a.json, 'w'), indent=1)
    # ---- ladder.md
    md = ['# ladder-A leaderboard (Phase A: T1–T6)\n',
          f'Re-derived by `ladder_analysis.py` from `artifacts/ladder/<arm>/found_transfer_{a.round}.jsonl`, `found_{a.round}.jsonl` (`found_union_{a.round}.jsonl` for T6) and `novelty_proofs.jsonl`; pools `data/ladder/` (`POOLS.md`). '
          'Counts are start-index-normalised distinct proofs; solved = ≥ 1 verifier-accepted proof of the prompted sequent in the cumulative attempts (256 per theorem). '
          '`L*` = largest L with ≥ 5 theorems solved at `L_true ≥ L`; `L_true` is `minlen.py`\'s minimal length in its restricted formula space at bound 14 (see POOLS.md caveat). '
          'Base reachability: a solved theorem is "elicitation" if one of its counted proofs has base p(T = 0.8) ≥ 1e-5 under `stage1_abs.pt`, else "creation".\n',
          '| arm | rung | seed | rounds | L* transfer | L* transfer − L*_frozen (same seed) | L* targets | L* transfer gen-only | L* transfer / targets, labels corrected | transfer solved | targets solved | transfer thms at L_true ≥ 8 / 9 / 10 / 11 | targets thms ≥ 8 / 9 / 10 / 11 | held-out greedy | pod time |',
          '|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---|---|---:|---:|']
    for arm, r in summ.items():
        t, g = r['transfer'], r['targets']
        md.append(f"| {arm} | {RUNGS.get(r['rung'], r['rung'])} | {r['seed']} | {r['rounds']} | **{t['lstar']}** | {('+' + str(t['lstar'] - summ['la_frozen_s' + r['seed'][0]]['transfer']['lstar'])) if r['rung'] != 'frozen' and ('la_frozen_s' + r['seed'][0]) in summ else '–'} | {g['lstar']} | {t['lstar_gen_only']} | {t['lstar_relabelled']} / {g['lstar_relabelled']} | {t['solved']}/{t['n']} | {g['solved']}/{g['n']} | "
                  f"{t['ge'][8]} / {t['ge'][9]} / {t['ge'][10]} / {t['ge'][11]} | {g['ge'][8]} / {g['ge'][9]} / {g['ge'][10]} / {g['ge'][11]} | {r['heldout_greedy_final']:.3f} | {r['secs']/60:.0f} min |")
    md.append('\n`L*_frozen` = 7 on transfer (both seeds; 8 on the RL targets). T6 rows `la_T6_s<seed>` are the pre-registered arm (both siblings: transfer = union of 2 × 8 × 16 samples, targets = `found_union`); '
              'rows `T6sib` are the single siblings (128 transfer attempts each). `la_T3_s0`: rounds 1–5 from the first session, rounds 6–8 resumed from `la_T3_s0_r5.pt` after a crash (log.md 2026-09-18 18:00).\n')
    md.append('**Cost** (`~/pods.log` creation times, $0.50/h RTX 3090): la-1…la-5 created 01:15–02:14 UTC, last arm finished 04:42 (≈ 15 pod-hours ≈ $7.5 of pool building and arms); '
              'the pods then stayed up idle until Dan deleted them at 17:20 after the session was cut off (≈ 63 idle pod-hours ≈ $31.6); la-6 17:53–18:22 for the T3 s0 resume ≈ $0.25. Total ≈ **$39** of the $50 budget. '
              'Per arm: the "pod time" column (two to three arms shared one GPU).\n')
    md.append('\n## Transfer solve rate by `L_true` bin (Wilson 95 %)\n')
    md.append('| arm | ' + ' | '.join(f'L_true {L}' for L in BINS) + ' |')
    md.append('|---|' + '---|' * len(BINS))
    for arm, r in summ.items():
        cells = []
        for L in BINS:
            b = r['transfer']['bins'][L]
            cells.append(f"{b['solved']}/{b['n']} ({b['rate']*100:.1f} [{b['ci'][0]*100:.1f},{b['ci'][1]*100:.1f}])" if b['n'] else '–')
        md.append(f'| {arm} | ' + ' | '.join(cells) + ' |')
    md.append('\n## RL-target solve rate by `L_true` bin (Wilson 95 %)\n')
    md.append('| arm | ' + ' | '.join(f'L_true {L}' for L in BINS) + ' |')
    md.append('|---|' + '---|' * len(BINS))
    for arm, r in summ.items():
        cells = []
        for L in BINS:
            b = r['targets']['bins'][L]
            cells.append(f"{b['solved']}/{b['n']} ({b['rate']*100:.1f} [{b['ci'][0]*100:.1f},{b['ci'][1]*100:.1f}])" if b['n'] else '–')
        md.append(f'| {arm} | ' + ' | '.join(cells) + ' |')
    md.append('\n## Base reachability of solved transfer theorems (max base p over counted proofs)\n')
    md.append('| arm | L_true ≥ 7: elicit(1e-5) / elicit(1/256) / scored / solved | ≥ 8 | ≥ 9 | ≥ 10 |')
    md.append('|---|---|---|---|---|')
    for arm, r in summ.items():
        rc = r['transfer'].get('reachability')
        if not rc:
            md.append(f'| {arm} | (novelty not scored) | | | |'); continue
        md.append(f'| {arm} | ' + ' | '.join(f"{rc[L]['elicit_1e-5']} / {rc[L]['elicit_1/256']} / {rc[L]['scored']} / {rc[L]['theorems']}" for L in (7, 8, 9, 10)) + ' |')
    md.append('\n## Transfer theorems solved at `L_true ≥ 9`, by source\n')
    md.append('| arm | generator | schemata |')
    md.append('|---|---:|---|')
    for arm, r in summ.items():
        t = r['transfer']
        md.append(f"| {arm} | {t['gen_ge9']} | {', '.join(f'{k} {v}' for k, v in sorted(t['schemas_ge9'].items()))} |")
    md.append('\n## Transfer solve rate by source (all `L_true`)\n')
    md.append('| arm | generator | textbook schemata | schemata with ≥ 1 instance solved |')
    md.append('|---|---|---|---|')
    for arm, r in summ.items():
        bs = r['transfer']['by_source']; g_, t_ = bs.get('gen', {'solved': 0, 'n': 0}), bs.get('textbook', {'solved': 0, 'n': 0})
        sch = ', '.join(f"{k} {v['solved']}/{v['n']}" for k, v in r['transfer']['by_schema'].items() if v['solved'])
        md.append(f"| {arm} | {g_['solved']}/{g_['n']} ({100*g_['solved']/max(1,g_['n']):.1f} %) | {t_['solved']}/{t_['n']} ({100*t_['solved']/max(1,t_['n']):.1f} %) | {sch} |")
    md.append('\n## Paired comparison with T1 of the same seed (transfer theorems solved by one arm only; exact sign test)\n')
    md.append('`la_T1_s1` is compared with `la_T1_s0`: that row is the seed-to-seed noise of the reference.\n')
    md.append('| arm | all L_true: arm only / T1 only (p) | L_true ≥ 9 | L_true ≥ 10 |')
    md.append('|---|---|---|---|')
    for arm, r in summ.items():
        if 'vs_T1' in r:
            md.append(f'| {arm} | ' + ' | '.join(f"{r['vs_T1'][L]['arm_only']} / {r['vs_T1'][L]['T1_only']} (p = {r['vs_T1'][L]['p']:.3f})" for L in (7, 9, 10)) + ' |')
    md.append('\n## Where the budget went: RL targets with `L_true` 11–13 (from `alloc_<R>.json` and `found_<R>.jsonl`; T6 siblings separately)\n')
    md.append('| arm | targets | samples spent | per target | accepted samples | targets solved |')
    md.append('|---|---:|---:|---:|---:|---:|')
    hard = [r['name'] for r in targets if 11 <= r['n_lines'] <= 13]
    for arm, r in summ.items():
        fa = f'{a.dir}/{arm}/alloc_{a.round}.json'
        if not os.path.exists(fa):
            continue
        al = json.load(open(fa)); sv = {x['name'] for x in rd(f'{a.dir}/{arm}/found_{a.round}.jsonl')}
        tr = sum(al['tried'].get(n, 0) for n in hard)
        md.append(f"| {arm} | {len(hard)} | {tr} | {tr/len(hard):.0f} | {sum(al['accepted'].get(n, 0) for n in hard)} | {sum(n in sv for n in hard)} |")
    md.append('\n## Mechanism checks (from `round_<r>.json`)\n')
    for arm, r in summ.items():
        fr = f'{a.dir}/{arm}/round_{a.round}.json'
        if not os.path.exists(fr):
            continue
        st = json.load(open(fr)); bits = [f"round-{a.round} RL records in mix {st.get('mix_rl_records')}", f"target samples per round {st.get('target_samples')}"]
        if 'relabelled_total' in st:
            bits.append(f"T3 by-product theorems accumulated {st['relabelled_total']}")
        if os.path.exists(f'{a.dir}/{arm}/inject_log.json'):
            bits.append(f"T5 injected records (round 4 only) {json.load(open(f'{a.dir}/{arm}/inject_log.json'))['n']}")
        if st.get('alloc_log'):
            bits.append('allocation ' + json.dumps({k: v for k, v in st['alloc_log'].items() if k != 'weights'} or st['alloc_log']))
        md.append(f'- {arm}: ' + '; '.join(bits))
    md.append('\n## Label robustness\n')
    byL = collections.Counter(v[0] for v in contradicted.values())
    md.append(f"{len(contradicted)} pool theorems (transfer + targets) have an accepted written proof shorter than their `L_true` label "
              f"(by label: {', '.join(f'{L}: {c}' for L, c in sorted(byL.items()))}; largest gap {max([v[0]-v[1] for v in contradicted.values()], default=0)} line(s)); "
              'the proofs cite one box for both cases of an `ORE` on `X v X`, which `minlen.py` does not search. The "labels corrected" column recomputes `L*` with '
              'label = min(label, shortest accepted written proof in any arm); list in `artifacts/ladder/summary.json.full`.\n')
    md.append('\n## L* by round (transfer, targets)\n')
    for arm, r in summ.items():
        if not r['lstar_by_round']:
            continue
        md.append(f"- {arm}: " + ', '.join(f'r{a_}: {b_}/{c_}' for a_, b_, c_ in r['lstar_by_round']))
    open(a.out, 'w').write('\n'.join(md) + '\n')
    print('\n'.join(md[:12]))


if __name__ == '__main__':
    main()
