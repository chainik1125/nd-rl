#!/usr/bin/env python3
"""lean-format (proposal 8): every number of the run, re-derived from pulled files -> artifacts/lf/summary.json + printed tables.

  python3 lean_format_analysis.py [--seqlen]        (--seqlen recomputes the sequence-length statistics; needs the training files)

Sources
  Stage-1 held-out greedy : artifacts/lf/frozen_d3_<sch>_s<k>/round_1.json (a1 sets; the frozen arm's round-1 greedy IS the Stage-1 model),
                            artifacts/lf/stage1_full_<sch>_heldout_greedy.json (full set); token: artifacts/p2/ei_depth3_f0_a1_s<k>/round_1.json, numbers.md (0.948)
  depth-3 dial            : phase2_metrics.arm_metrics on artifacts/lf/{ei,frozen}_d3_<sch>_s<k>/ and on the token arms artifacts/p2/{ei,frozen}_depth3_f0_a1_s<k>/
  ladder T1               : artifacts/lf/la_{T1,frozen}_<sch>_s<k>/found_transfer_8.jsonl, found_8.jsonl vs data/ladder pools (L* = max L with >= 5 theorems solved at L_true >= L)
  mechanism               : artifacts/lf/stage1_full_<tag>_transfer2_k16.jsonl (distinct verified proofs by written length); token: artifacts/stage1_{abs,rel,absfixed}_transfer2_k16_norm.json
  Lean gate               : artifacts/lf/gate_*.jsonl (2 x 2 agreement, Lean / nd_verify seconds), gate_*.disagree.jsonl
  checker of record       : artifacts/lf/record_<arm>_{found,found_transfer}.jsonl (nd2lean.py --check on every counted proof)
  round times             : artifacts/lf/timing_*/round_1.json (solo, back to back on the same pod), round_<r>.json of the arms
"""
import argparse, json, os, sys, glob, collections, statistics
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from normalize import norm
from phase2_metrics import arm_metrics

D = 'artifacts/lf'
SCH = ('rand', 'seq')


def rd(fn):
    return [json.loads(l) for l in open(fn) if l.strip()]


def jl(fn):
    return json.load(open(fn)) if os.path.exists(fn) else None


def lstar(solved_names, pool, need=5):
    L = [r['n_lines'] for r in pool if r['name'] in solved_names]
    ge = {k: sum(1 for x in L if x >= k) for k in range(7, 15)}
    return max([k for k, c in ge.items() if c >= need], default=0), ge


def found_names(fn):
    by = collections.defaultdict(set)
    for x in rd(fn):
        by[x['name']].add(norm(x['proof']))
    return by


def seqlen():
    from tokenizer import Tokenizer
    from lean_tok import LeanTokenizer
    ta, tl = Tokenizer('abs'), LeanTokenizer('lean_rand')
    out = {}
    for tag, fn in (('train_full', 'data/train.jsonl'), ('train_depth3_f0_a1', 'data/p2/train_depth3_f0_a1.jsonl'), ('heldout_full', 'data/heldout.jsonl'),
                    ('heldout_p2', 'data/p2/heldout.jsonl'), ('ladder_transfer_prompts', 'data/ladder/transfer.jsonl'), ('depth3_target_prompts', 'data/p2/targets_depth3.jsonl')):
        if not os.path.exists(fn):
            continue
        recs = rd(fn); row = {'n': len(recs)}
        for nm, t in (('token_abs', ta), ('lean', tl)):
            pl = [len(t.encode_prompt(r['prompt'])) for r in recs]
            row[nm] = {'prompt_mean': statistics.mean(pl), 'prompt_max': max(pl)}
            if 'proof' in recs[0] and 'train' in tag or 'heldout' in tag:
                ql = [len(t.encode_proof(r['proof'])) for r in recs]
                tot = [a + b for a, b in zip(pl, ql)]
                row[nm].update({'proof_mean': statistics.mean(ql), 'proof_median': statistics.median(ql), 'proof_max': max(ql), 'total_mean': statistics.mean(tot), 'total_max': max(tot),
                                'total_p95': sorted(tot)[int(0.95 * len(tot))]})
        if 'proof_mean' in row['lean']:
            row['ratio_total_mean'] = row['lean']['total_mean'] / row['token_abs']['total_mean']; row['ratio_proof_mean'] = row['lean']['proof_mean'] / row['token_abs']['proof_mean']
        out[tag] = row
    json.dump(out, open(f'{D}/seqlen.json', 'w'), indent=1)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--seqlen', action='store_true'); a = ap.parse_args()
    S = {}
    if a.seqlen:
        seqlen()
    S['seqlen'] = jl(f'{D}/seqlen.json')
    # ---- P1 held-out greedy of the Stage-1 models
    p1 = {'token_a1': {}, 'token_full': 0.948}
    for s in (0, 1):
        r = jl(f'artifacts/p2/ei_depth3_f0_a1_s{s}/round_1.json'); p1['token_a1'][s] = r['heldout_greedy']['rate'] if r else None
    for sch in SCH:
        p1[sch] = {'a1': {}, 'full': None}
        for s in (0, 1):
            r = jl(f'{D}/frozen_d3_{sch}_s{s}/round_1.json') or jl(f'{D}/ei_d3_{sch}_s{s}/round_1.json')
            p1[sch]['a1'][s] = r['heldout_greedy']['rate'] if r else None
        r = jl(f'{D}/stage1_full_{sch}_heldout_greedy.json'); p1[sch]['full'] = r['rate'] if r else None
        p1[sch]['full_by_len'] = {k: v['rate'] for k, v in r['by_len'].items()} if r else None
    r = jl(f'{D}/stage1_full_seqfixed_heldout_greedy.json'); p1['seqfixed_full'] = r['rate'] if r else None
    S['P1_heldout_greedy'] = p1
    # ---- P2 depth-3 dial
    p2 = {}
    arms = [(f'token_{k}_s{s}', f'artifacts/p2/{k}_depth3_f0_a1_s{s}') for k in ('ei', 'frozen') for s in (0, 1)]
    arms += [(f'{sch}_{k}_s{s}', f'{D}/{k}_d3_{sch}_s{s}') for sch in SCH for k in ('ei', 'frozen') for s in (0, 1)]
    for tag, d in arms:
        if not glob.glob(f'{d}/round_*.json') or not glob.glob(f'{d}/found_*.jsonl'):
            continue
        m = arm_metrics(d, 'depth3')
        last = m[-1]
        p2[tag] = {'dir': d, 'rounds': last['round'], 'attempts': last['attempts'], 'targets_solved': last['targets_solved'], 'acq': last['acq_targets'], 'acq_theorems': last['acq_targets_theorems'],
                   'n_depth3_proofs': last['n_pattern_proofs_targets'], 'first_round': last['first_round_pattern_targets'], 'transfer_solved': last['transfer_solved'],
                   'acq_transfer': last['acq_transfer'], 'heldout_greedy_final': last['heldout_greedy'], 'acq_by_round': [x['acq_targets'] for x in m],
                   'solved_by_round': [x['targets_solved'] for x in m], 'written_hist_targets': last['written_hist_targets'],
                   'secs_by_round': [json.load(open(f'{d}/round_{x["round"]}.json'))['secs'] for x in m], 'example': (last['pattern_examples_targets'] or [None])[0]}
    S['P2_depth3'] = p2
    # ---- P3 ladder
    p3 = {}
    if os.path.exists('data/ladder/transfer.jsonl'):
        transfer, targets = rd('data/ladder/transfer.jsonl'), rd('data/ladder/rl_targets.jsonl')
        for sch in SCH:
            for k in ('T1', 'frozen'):
                for s in (0, 1):
                    d = f'{D}/la_{k}_{sch}_s{s}'
                    rounds = sorted(int(f.split('_')[-1][:-5]) for f in glob.glob(f'{d}/round_*.json'))
                    if not rounds or not os.path.exists(f'{d}/found_transfer_{rounds[-1]}.jsonl'):
                        continue
                    R = rounds[-1]
                    ft, fg = found_names(f'{d}/found_transfer_{R}.jsonl'), found_names(f'{d}/found_{R}.jsonl')
                    lt, get = lstar(set(ft), transfer); lg, geg = lstar(set(fg), targets)
                    byb = collections.Counter(r['n_lines'] for r in transfer if r['name'] in ft)
                    rs = [json.load(open(f'{d}/round_{r}.json')) for r in rounds]
                    p3[f'{sch}_{k}_s{s}'] = {'dir': d, 'rounds': R, 'lstar_transfer': lt, 'lstar_targets': lg, 'transfer_solved': len(ft), 'targets_solved': len(fg), 'transfer_n': len(transfer), 'targets_n': len(targets),
                                             'transfer_ge': get, 'targets_ge': geg, 'transfer_by_bin': dict(sorted(byb.items())), 'distinct_transfer_proofs': sum(len(v) for v in ft.values()),
                                             'heldout_greedy_final': rs[-1]['heldout_greedy']['rate'], 'heldout_greedy_r1': rs[0]['heldout_greedy']['rate'], 'secs_by_round': [r['secs'] for r in rs],
                                             'lstar_by_round': [(r['round'], r['transfer_cum'].get('lstar')) for r in rs]}
    p3['token_on_file'] = {'la_T1_s0': {'lstar_transfer': 10, 'transfer_solved': 612, 'transfer_ge': {'8': 472, '9': 335, '10': 35, '11': 1}, 'heldout_greedy_final': 0.953},
                           'la_T1_s1': {'lstar_transfer': 10, 'transfer_solved': 623, 'transfer_ge': {'8': 484, '9': 346, '10': 34, '11': 0}, 'heldout_greedy_final': 0.957},
                           'la_frozen_s0': {'lstar_transfer': 7, 'transfer_solved': 22}, 'la_frozen_s1': {'lstar_transfer': 7, 'transfer_solved': 24}, 'source': 'ladder.md on branch dan_ladder_a'}
    S['P3_ladder'] = p3
    # ---- P4 mechanism
    p4 = {}
    for tag in ('rand', 'seq', 'seqfixed'):
        fn = f'{D}/stage1_full_{tag}_transfer2_k16.jsonl'
        if not os.path.exists(fn):
            continue
        wh = collections.Counter(); solved = 0; n = 0
        for r in rd(fn):
            n += 1; solved += r['solved']
            seen = set()
            for p, w in zip(r['proofs'], r['written_lens']):
                q = norm(p)
                if q not in seen:
                    seen.add(q); wh[w] += 1
        p4[tag] = {'n': n, 'solved_pass16': solved, 'rate': solved / n, 'written_hist': dict(sorted(wh.items())), 'len7': wh[7], 'len8': wh[8], 'len9plus': sum(c for w, c in wh.items() if w >= 9)}
    for tag in ('abs', 'rel', 'absfixed'):
        r = jl(f'artifacts/stage1_{tag}_transfer2_k16_norm.json')
        if r:
            p4[f'token_{tag}'] = {'n': r['n'], 'solved_pass16': r['solved'], 'rate': r['solved'] / r['n'], 'written_hist': r['written_hist'], 'len7': r['written_hist'].get('7', 0), 'len8': r['written_hist'].get('8', 0)}
    S['P4_mechanism'] = p4
    # ---- P5 / P6 gate agreement and throughput
    g = collections.Counter(); per_job = {}
    for fn in sorted(glob.glob(f'{D}/gate_*.jsonl')):
        if fn.endswith('.disagree.jsonl') or 'selftest' in fn:
            continue
        c = collections.Counter(); pr = collections.Counter()
        for x in rd(fn):
            for k in ('samples', 'parse_fail', 'distinct_checked', 'both_ok', 'nd_ok_lean_rej', 'nd_rej_lean_ok', 'both_rej', 'lean_wall_s', 'lean_proc_s', 'nd_verify_s'):
                c[k] += x[k]
            pr.update(x.get('parse_reasons', {}))
        per_job[os.path.basename(fn)[5:-6]] = {**dict(c), 'parse_reasons': dict(pr.most_common(6))}
        g.update(c)
    dis = [x for fn in sorted(glob.glob(f'{D}/gate_*.disagree.jsonl')) for x in rd(fn)]
    kinds = collections.Counter()
    for x in dis:
        kinds[('elim-on-negation' if '.elim' in x['lean_text'] else 'other') if not x['nd_ok'] else 'ND-OK-LEAN-REJ'] += 1
    S['P5_gate'] = {'total': dict(g), 'disagreements': len(dis), 'disagreement_kinds_rough': dict(kinds), 'per_job': per_job,
                    'lean_proofs_per_proc_second': g['distinct_checked'] / g['lean_proc_s'] if g['lean_proc_s'] else None,
                    'nd_verify_proofs_per_second': g['distinct_checked'] / g['nd_verify_s'] if g['nd_verify_s'] else None}
    rec = {}
    for fn in sorted(glob.glob(f'{D}/record_*.json')):
        rec[os.path.basename(fn)[7:-5]] = json.load(open(fn))
    S['checker_of_record'] = rec
    tm = {}
    for fn in sorted(glob.glob(f'{D}/timing_*/round_1.json')):
        r = json.load(open(fn)); tm[fn.split('/')[-2]] = r['secs']
    S['P6_timing_solo_round_secs'] = tm
    json.dump(S, open(f'{D}/summary.json', 'w'), indent=1, ensure_ascii=False)
    # ---- print
    print('P1 held-out greedy:', json.dumps(p1))
    print('P2 depth-3:')
    for k, v in p2.items():
        print(f"  {k:18s} r{v['rounds']} solved {v['targets_solved']:4d}/1000 acq {v['acq']:.3f} ({v['acq_theorems']} thms / {v['n_depth3_proofs']} proofs / first r{v['first_round']}) transfer acq {v['acq_transfer']:.3f} heldout {v['heldout_greedy_final']:.3f} acq by round {[round(x, 3) for x in v['acq_by_round']]}")
    print('P3 ladder:')
    for k, v in p3.items():
        if k != 'token_on_file':
            print(f"  {k:18s} r{v['rounds']} L* transfer {v['lstar_transfer']} targets {v['lstar_targets']} transfer solved {v['transfer_solved']}/{v['transfer_n']} ge {v['transfer_ge']} targets solved {v['targets_solved']} heldout {v['heldout_greedy_final']:.3f}")
    print('P4 mechanism:', json.dumps({k: (v['rate'], v['len7'], v['len8']) for k, v in p4.items()}))
    print('P5 gate:', json.dumps({k: v for k, v in S['P5_gate'].items() if k != 'per_job'}))
    print('record:', json.dumps(rec)); print('timing:', json.dumps(tm))


if __name__ == '__main__':
    main()
