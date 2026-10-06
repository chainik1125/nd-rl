#!/usr/bin/env python3
"""ds-generator analysis: every number of the write-up from pulled files -> artifacts/dsg/summary.json (one row per arm x seed)
and a markdown dump (stdout). Sources (all under artifacts/dsg/ unless said):
  held-out      heldout_<arm>_s<k>.json / .jsonl (eval_set greedy on data/p2/heldout.jsonl), joined by name to the held-out records' pattern labels
  coverage      cov_<pool>_<arm>_s<k>.s0.jsonl (coverage.py, k 2,000, Lean AND nd_verify), joined by name to the pool file (min_lines_ub strata)
  dial          {ei,frozen}_d3_<arm>_s<k>/ (expert_iter, 4 rounds) via phase2_metrics.arm_metrics; C0 from artifacts/lf_control/ (lean-format bucket)
  ladder        la_{T1,frozen}_<arm>_s<k>/found_transfer_8.jsonl, found_8.jsonl, round_*.json (ladder_ei)
  record        record_<arm>.json (unmodified nd2lean.py --check + nd_verify on every counted proof)
  sets          shape_<tag>.json, overlap_<arm>.json, assemble_<arm>.json, render_<arm>.json
  python3 dsg_analysis.py [--out artifacts/dsg/summary.json]"""
import argparse, json, os, sys, glob, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from normalize import norm
from phase2_metrics import arm_metrics

D = 'artifacts/dsg'
ARMS = ('c0', 'g1', 'g2')
POOLS = {'d3': ('data/p2/targets_depth3.jsonl', 'depth3'), 'req8': ('data/r3_1/depth3_req.jsonl', 'depth3'), 'red': ('data/p2/targets_reductio_req.jsonl', 'derived_dn')}
TEXTBOOK_BASE = ('contraposition', 'contraposition_conv', 'export')


def rd(fn):
    return [json.loads(l) for l in open(fn) if l.strip()]


def jl(fn):
    return json.load(open(fn)) if os.path.exists(fn) else None


def lstar(solved_names, pool, need=5):
    L = [r['n_lines'] for r in pool if r['name'] in solved_names]
    ge = {k: sum(1 for x in L if x >= k) for k in range(7, 15)}
    return max([k for k, c in ge.items() if c >= need], default=0), ge


def heldout(arm, s):
    # resume phase: the 2026-09-22 per-record .jsonl files were lost with the host cleanup (only the .json
    # summaries were committed), so every arm's held-out greedy was re-measured on 2026-09-23 -> heldout2_*.
    fn = f'{D}/heldout2_{arm}_s{s}.jsonl'
    if not os.path.exists(fn):
        fn = f'{D}/heldout_{arm}_s{s}.jsonl'
    if not os.path.exists(fn):
        return None
    rows = rd(fn); summ = jl(fn.replace('.jsonl', '.json'))
    lab = {r['name']: r['pat'] for r in rd('data/p2/heldout.jsonl')}
    by = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        p = lab[r['name']]
        key = 'pattern' if any(p.get(k) for k in ('derived_ore', 'reductio', 'depth3')) else 'none'
        by[key][0] += r['solved']; by[key][1] += 1
        if p.get('depth3'):
            by['depth3'][0] += r['solved']; by['depth3'][1] += 1
        if r['n_lines'] == 6:
            by['len6_' + key][0] += r['solved']; by['len6_' + key][1] += 1
    return {'rate': summ['rate'], 'n': summ['n'], 'by_len': {k: v['rate'] for k, v in summ['by_len'].items()},
            'by_pattern': {k: {'rate': v[0] / v[1], 'solved': v[0], 'n': v[1]} for k, v in by.items()}}


def coverage(arm, s, pool):
    fn = f'{D}/cov_{pool}_{arm}_s{s}.s0.jsonl'
    if not os.path.exists(fn):
        return None
    rows = rd(fn); pfn, pat = POOLS[pool]
    meta = {r['name']: r for r in rd(pfn)}
    n = len(rows); hit = sum(1 for r in rows if r['n_ok'] > 0)
    hit_pat = sum(1 for r in rows if any(p['pat'].get(pat) for p in r['proofs']))
    tried = sum(r['n_tried'] for r in rows); ok = sum(r['n_ok'] for r in rows)
    strata = collections.defaultdict(lambda: [0, 0, 0])
    for r in rows:
        st = str(meta[r['name']].get('min_lines_ub'))
        strata[st][1] += 1; strata[st][0] += r['n_ok'] > 0; strata[st][2] += any(p['pat'].get(pat) for p in r['proofs'])
    d8 = sum(1 for r in rows for p in r['proofs'] if p['written'] >= 8)
    d8p = sum(1 for r in rows for p in r['proofs'] if p['written'] >= 8 and p['pat'].get(pat))
    d7p = sum(1 for r in rows for p in r['proofs'] if p['written'] == 7 and p['pat'].get(pat))
    wh = collections.Counter(p['written'] for r in rows for p in r['proofs'])
    return {'n_targets': n, 'targets_hit': hit, 'targets_hit_with_pattern': hit_pat, 'pattern': pat, 'per_sample_rate': ok / tried if tried else None,
            'samples': tried, 'n_ok_samples': ok, 'by_min_lines_ub': {k: {'n': v[1], 'hit': v[0], 'hit_with_pattern': v[2]} for k, v in sorted(strata.items())},
            'distinct_proofs': sum(len(r['proofs']) for r in rows), 'distinct_ge8': d8, 'distinct_ge8_pattern': d8p, 'distinct_7_pattern': d7p, 'written_hist': dict(sorted(wh.items())),
            'lean_checked': sum(r.get('n_lean_checked', 0) for r in rows), 'lean_rejected': sum(r.get('n_lean_rejected', 0) for r in rows), 'complete': n == len(meta)}


def dial(arm, s):
    base = 'artifacts/lf_control' if arm == 'c0' else D
    tag = 'seq' if arm == 'c0' else arm
    out = {}
    for a in ('ei', 'frozen'):
        d = f'{base}/{a}_d3_{tag}_s{s}'
        if not os.path.exists(f'{d}/found_4.jsonl'):
            return None
        m = arm_metrics(d, 'depth3')
        r4 = [x for x in m if x['round'] == 4]
        if not r4:
            return None
        x = r4[0]
        out[a] = {'acq_round4': x['acq_targets'], 'acq_theorems': x['acq_targets_theorems'], 'n_depth3_proofs': x['n_pattern_proofs_targets'],
                  'targets_solved': x['targets_solved'], 'transfer_acq': x['acq_transfer'], 'heldout_greedy_r4': x['heldout_greedy'],
                  'acq_by_round': [y['acq_targets'] for y in m], 'first_round': x['first_round_pattern_targets']}
    out['ei_minus_frozen'] = out['ei']['acq_round4'] - out['frozen']['acq_round4']
    return out


def retrain_check(arm, s):
    """Resume phase: G1 / G2's Stage-1 checkpoints were deleted by the host cleanup and retrained on 2026-09-23
    from the same set, seed and command line, on a different GPU class (A6000 / 4090, not the 3090 the originals
    were trained on).  Held-out greedy is re-measured with the identical eval_set call and compared.  C0's row is
    the control for the comparison itself: its checkpoint is byte-identical (md5 verified against the
    pre-registration), so its delta is the harness's, not a retrain's."""
    old, new = jl(f'{D}/heldout_{arm}_s{s}.json'), jl(f'{D}/heldout2_{arm}_s{s}.json')
    if not old or not new:
        return None
    return {'heldout_2026_09_22': old['rate'], 'heldout_2026_09_23_retrain': new['rate'],
            'delta_pp': 100 * (new['rate'] - old['rate']), 'n': old.get('n')}


def ladder(arm, s, transfer, targets):
    out = {}
    for k in ('T1', 'frozen'):
        d = f'{D}/la_{k}_{arm}_s{s}'
        rounds = sorted(int(f.split('_')[-1][:-5]) for f in glob.glob(f'{d}/round_*.json'))
        if not rounds or not os.path.exists(f'{d}/found_transfer_{rounds[-1]}.jsonl'):
            out[k] = None; continue
        R = rounds[-1]
        ft = collections.defaultdict(set); schema = collections.Counter(); src = collections.Counter()
        for x in rd(f'{d}/found_transfer_{R}.jsonl'):
            ft[x['name']].add(norm(x['proof']))
        fg = collections.defaultdict(set)
        for x in rd(f'{d}/found_{R}.jsonl'):
            fg[x['name']].add(norm(x['proof']))
        for r in transfer:
            if r['name'] in ft:
                src[r.get('source')] += 1
                if r.get('schema'):
                    schema[r['schema']] += 1
        lt, get = lstar(set(ft), transfer); lg, geg = lstar(set(fg), targets)
        byb = collections.Counter(r['n_lines'] for r in transfer if r['name'] in ft)
        tb_all = collections.Counter(r['schema'] for r in transfer if r.get('schema'))
        rs = [json.load(open(f'{d}/round_{r}.json')) for r in rounds]
        out[k] = {'rounds': R, 'lstar_transfer': lt, 'lstar_targets': lg, 'transfer_solved': len(ft), 'targets_solved': len(fg), 'transfer_n': len(transfer), 'targets_n': len(targets),
                  'transfer_ge': get, 'transfer_by_bin': {str(b): byb.get(b, 0) for b in range(7, 15)}, 'textbook_solved': sum(schema.values()), 'textbook_n': sum(tb_all.values()),
                  'by_schema': {sc: [schema.get(sc, 0), tb_all[sc]] for sc in sorted(tb_all)}, 'schemata_ge5': sorted(sc for sc in tb_all if schema.get(sc, 0) >= 5),
                  'new_schemata_ge5': sorted(sc for sc in tb_all if schema.get(sc, 0) >= 5 and sc not in TEXTBOOK_BASE), 'by_source': dict(src),
                  'distinct_transfer_proofs': sum(len(v) for v in ft.values()), 'heldout_greedy_r1': rs[0]['heldout_greedy']['rate'], 'heldout_greedy_final': rs[-1]['heldout_greedy']['rate'],
                  'lstar_by_round': [(r['round'], r['transfer_cum'].get('lstar')) for r in rs], 'secs_by_round': [r['secs'] for r in rs]}
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', default=f'{D}/summary.json'); a = ap.parse_args()
    transfer, targets = rd('data/ladder/transfer.jsonl'), rd('data/ladder/rl_targets.jsonl')
    S = {'rows': [], 'sets': {}, 'record': {}}
    for arm in ARMS:
        tag = 'c0_a1' if arm == 'c0' else arm
        sh = jl(f'{D}/shape_{tag}.json')
        S['sets'][arm] = {'shape': list(sh.values())[0] if sh else None, 'overlap': jl(f'{D}/overlap_{arm}.json'), 'assemble': jl(f'{D}/assemble_{arm}.json'), 'render': jl(f'{D}/render_{arm}.json')}
        rec = jl(f'{D}/record_{arm}.json') or {}
        for k in (0, 1):                      # resume phase: one seed per pod -> record_<arm>_s<seed>.json
            rec.update(jl(f'{D}/record_{arm}_s{k}.json') or {})
        S['record'][arm] = rec or None
        S['sets'][arm]['retrain_check'] = {f's{k}': retrain_check(arm, k) for k in (0, 1)}
        for s in (0, 1):
            row = {'arm': arm, 'seed': s, 'heldout': heldout(arm, s), 'coverage': {p: coverage(arm, s, p) for p in POOLS}, 'dial': dial(arm, s), 'ladder': ladder(arm, s, transfer, targets)}
            S['rows'].append(row)
    os.makedirs(D, exist_ok=True)
    json.dump(S, open(a.out, 'w'), indent=1)
    # ---- markdown dump
    f = lambda x, d=3: ('–' if x is None else (f'{x:.{d}f}' if isinstance(x, float) else str(x)))
    print('## Held-out greedy (data/p2/heldout.jsonl, 5,000): overall; by length 2–6; pattern-free / pattern (depth-3) theorems')
    print('| arm s | overall | 2 | 3 | 4 | 5 | 6 | none | pattern | depth-3 thms | 6-line none |\n|---|---|---|---|---|---|---|---|---|---|---|')
    for r in S['rows']:
        h = r['heldout']
        if h: print(f"| {r['arm']} s{r['seed']} | {f(h['rate'])} | " + ' | '.join(f(h['by_len'].get(str(L))) for L in range(2, 7)) + f" | {f(h['by_pattern']['none']['rate'])} | {f(h['by_pattern']['pattern']['rate'])} | {f(h['by_pattern'].get('depth3', {}).get('rate'))} | {f(h['by_pattern'].get('len6_none', {}).get('rate'))} |")
    print('\n## Coverage pass@2,000 (Lean ∧ nd_verify): targets hit (with pattern) / n; per-sample rate; distinct ≥ 8-line pattern proofs; Lean-rejected distinct')
    print('| arm s | depth-3 1,000 | req8 300 | reductio-req 300 (7-line stratum; ≥ 8) | per-sample d3 / req8 / red | ≥ 8-line pattern proofs d3 / req8 / red | Lean rej |\n|---|---|---|---|---|---|---|')
    for r in S['rows']:
        c = r['coverage']
        if not any(c.values()): continue
        def hit(p): return f"{c[p]['targets_hit']} ({c[p]['targets_hit_with_pattern']}) / {c[p]['n_targets']}" if c[p] else '–'
        red = c['red']; st = red['by_min_lines_ub'] if red else {}
        s7 = st.get('7', {}).get('hit'); s8 = sum(v['hit'] for k, v in st.items() if k != '7') if st else None
        print(f"| {r['arm']} s{r['seed']} | {hit('d3')} | {hit('req8')} | {hit('red')} ({f(s7)}; {f(s8)}) | " + ' / '.join(f(c[p]['per_sample_rate'], 4) if c[p] else '–' for p in POOLS) + ' | ' + ' / '.join(str(c[p]['distinct_ge8_pattern']) if c[p] else '–' for p in POOLS) + f" | {sum(c[p]['lean_rejected'] for p in POOLS if c[p])} |")
    print('\n## Dial (targets_depth3, 4 rounds × 32): acquisition at round 4 EI / frozen / EI − frozen; held-out greedy at round 4 (EI)')
    print('| arm s | EI | frozen | EI − frozen | EI acq by round | held-out r4 |\n|---|---|---|---|---|---|')
    for r in S['rows']:
        d = r['dial']
        if d: print(f"| {r['arm']} s{r['seed']} | {f(d['ei']['acq_round4'])} | {f(d['frozen']['acq_round4'])} | {d['ei_minus_frozen']:+.3f} | {', '.join(f(x) for x in d['ei']['acq_by_round'])} | {f(d['ei']['heldout_greedy_r4'])} |")
    print('\n## Ladder (8 × 32): transfer L*, solved / 2,285, by L_true bin 7…14, textbook solved / 760, schemata at ≥ 5 solves, held-out r1 → r8')
    print('| arm s | rung | L* | solved | 7 / 8 / 9 / 10 / 11 / 12 / 13 / 14 | textbook | schemata ≥ 5 | held-out |\n|---|---|---|---|---|---|---|---|')
    for r in S['rows']:
        L = r['ladder']
        if not L: continue
        for k in ('frozen', 'T1'):
            x = L.get(k)
            if x: print(f"| {r['arm']} s{r['seed']} | {k} | {x['lstar_transfer']} | {x['transfer_solved']} | " + ' / '.join(str(x['transfer_by_bin'][str(b)]) for b in range(7, 15)) + f" | {x['textbook_solved']} | {', '.join(sc + ' ' + str(x['by_schema'][sc][0]) for sc in x['schemata_ge5'])} | {f(x['heldout_greedy_r1'])} → {f(x['heldout_greedy_final'])} |")
    print('\n## Stage-1 retrain reproduction check (G1 / G2 retrained 2026-09-23; C0 = same checkpoint, so its row is the harness control)')
    print('| arm s | held-out greedy 2026-09-22 (RTX 3090) | held-out greedy 2026-09-23 (A6000 / 4090) | delta pp |\n|---|---|---|---|')
    for arm in ARMS:
        rc = S['sets'][arm].get('retrain_check') or {}
        for k, v in sorted(rc.items()):
            if v: print(f"| {arm} {k} | {f(v['heldout_2026_09_22'])} | {f(v['heldout_2026_09_23_retrain'])} | {v['delta_pp']:+.2f} |")
    print('\n## Checker of record')
    for arm, rec in S['record'].items():
        if rec:
            tot = sum(v.get('n', 0) for v in rec.values()); ba = sum(v.get('both_accept', 0) for v in rec.values()); dis = sum(v.get('nd_ok_lean_rej', 0) + v.get('nd_rej_lean_ok', 0) for v in rec.values())
            print(f'- {arm}: {tot} counted proofs, both accept {ba}, disagreements {dis}')


if __name__ == '__main__':
    main()
