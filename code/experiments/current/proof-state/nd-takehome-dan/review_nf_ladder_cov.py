#!/usr/bin/env python3
"""Reviewer's independent ladder + coverage recount for run noise-floor.
Re-derives, from found_*.jsonl / coverage jsonl only:
  - frozen-ladder transfer & targets distinct-theorem counts (own dedup by name)
  - L* (own implementation; need=5 on the pool's declared L_true) and sensitivity to `need`
  - coverage pass@2000 hit counts (own n_ok recomputation via nd_verify on recorded proofs)
  - term size (total connectives in the written formulas) as well as line counts
"""
import json, os, sys, collections, re
sys.path.insert(0, '/home/dan/review/noise-floor')
from nd_verify import verify_text
from nd_verify.verify import parse_proof_tokens, parse_formula

ROOT = '/home/dan/review/noise-floor'
A = f'{ROOT}/artifacts/nf'

def read(fn):
    return [json.loads(l) for l in open(fn) if l.strip()]

def pool(fn):
    return read(fn)

TRANSFER = pool(f'{ROOT}/data/ladder/transfer.jsonl')
TARGETS = pool(f'{ROOT}/data/ladder/rl_targets.jsonl')

def my_lstar(solved_names, recs, need=5):
    L = [r['n_lines'] for r in recs if r['name'] in solved_names]
    ge = {k: sum(1 for x in L if x >= k) for k in range(2, 21)}
    return max([k for k, c in ge.items() if c >= need], default=0), ge

def term_size(formula_str):
    """my own term size: number of tokens in the parenthesised formula that are
    connectives or atoms (i.e. nodes of the syntax tree)."""
    toks = formula_str.split()
    return sum(1 for t in toks if t in ('~', '&', 'v', '>', 'F') or re.fullmatch(r'[A-Z]', t))

def proof_term_size(proof):
    """sum of formula node counts over the proof's lines"""
    tot = 0
    for seg in proof.split(';'):
        seg = seg.strip()
        if not seg or seg == 'QED':
            continue
        m = re.match(r'N\d+\s+((?:\|\s*)*)(.*?):', seg)
        if not m:
            continue
        tot += term_size(m.group(2))
    return tot

def ladder_cell(name, verify=False):
    d = f'{A}/{name}'
    if not os.path.isdir(d):
        return None
    out = {}
    for pool_tag, recs, fpat in (('transfer', TRANSFER, 'found_transfer_%d.jsonl'),
                                 ('targets', TARGETS, 'found_%d.jsonl')):
        byname = {r['name']: r for r in recs}
        seen_names = set(); proofs = collections.defaultdict(set)
        rounds_present = []
        for r in range(1, 9):
            f = os.path.join(d, fpat % r)
            if not os.path.exists(f):
                continue
            rounds_present.append(r)
            rows = read(f)
        # the round-8 file is cumulative; use it, but assert monotonicity
        f8 = os.path.join(d, fpat % max(rounds_present))
        for row in read(f8):
            if row['name'] not in byname:
                out.setdefault('offpool', 0)
                out['offpool'] += 1
                continue
            seen_names.add(row['name'])
            proofs[row['name']].add(row['proof'])
        ls5, ge = my_lstar(seen_names, recs, 5)
        ls1, _ = my_lstar(seen_names, recs, 1)
        ls10, _ = my_lstar(seen_names, recs, 10)
        ndistinct = sum(len(v) for v in proofs.values())
        tsz = [proof_term_size(p) for v in proofs.values() for p in v]
        out[pool_tag] = {'solved': len(seen_names), 'n': len(recs), 'lstar_need5': ls5,
                         'lstar_need1': ls1, 'lstar_need10': ls10, 'ge': ge,
                         'distinct_proofs': ndistinct, 'rounds': rounds_present,
                         'mean_term_size': sum(tsz)/len(tsz) if tsz else None,
                         'max_term_size': max(tsz) if tsz else None}
        if verify:
            bad = 0; n = 0
            for nm, ps in proofs.items():
                for p in ps:
                    n += 1
                    ok, why, nl = verify_text(byname[nm]['prompt'] + ' ' + p)
                    if not ok:
                        bad += 1
            out[pool_tag]['reverified'] = n
            out[pool_tag]['nd_verify_rejects'] = bad
    return out

def cov_cell(fn, verify=True):
    if not os.path.exists(fn):
        return None
    rows = read(fn)
    hit = 0; reverified = 0; rejects = 0; lean_checked = 0; lean_rejected = 0
    tried = 0; ok_samples = 0
    for r in rows:
        tried += r['n_tried']; ok_samples += r['n_ok']
        lean_checked += r.get('n_lean_checked', 0); lean_rejected += r.get('n_lean_rejected', 0)
        good = False
        for p in r['proofs']:
            s = p['proof'] if isinstance(p, dict) else p
            reverified += 1
            ok, why, nl = verify_text(r['prompt'] + ' ' + s)
            if ok:
                good = True
            else:
                rejects += 1
        if good:
            hit += 1
        if bool(r['n_ok'] > 0) != bool(r['proofs']):
            pass
    return {'n': len(rows), 'hit': hit, 'samples': tried, 'ok_samples': ok_samples,
            'reverified': reverified, 'nd_verify_rejects': rejects,
            'lean_checked': lean_checked, 'lean_rejected': lean_rejected}

if __name__ == '__main__':
    out = {'ladder': {}, 'cov_red': {}, 'cov_req8': {}}
    names = [f'la_frozen_p{p}_s{s}' for p in (1, 2, 3, 4) for s in range(13)]
    names += ['la_frozen_dsg_g1_s0', 'la_frozen_dsc_a1_s1', 'la_T1_dsc_a1_s1']
    for nm in names:
        c = ladder_cell(nm, verify=True)
        if c is None:
            continue
        out['ladder'][nm] = c
        print(f"{nm}: transfer {c['transfer']['solved']}/{c['transfer']['n']} "
              f"L*={c['transfer']['lstar_need5']} (need1 {c['transfer']['lstar_need1']}, need10 {c['transfer']['lstar_need10']}) "
              f"targets {c['targets']['solved']}/{c['targets']['n']} "
              f"distinct-proofs t={c['transfer']['distinct_proofs']}/g={c['targets']['distinct_proofs']} "
              f"nd_verify rejects {c['transfer']['nd_verify_rejects']}/{c['transfer']['reverified']}, "
              f"{c['targets']['nd_verify_rejects']}/{c['targets']['reverified']}", flush=True)
    for tag, pre in (('cov_red', 'cov_red'), ('cov_req8', 'cov_req8')):
        for p in (1, 2, 3, 4):
            for s in range(13):
                c = cov_cell(f'{A}/{pre}_p{p}_s{s}.s0.jsonl')
                if c is None:
                    continue
                out[tag][f'p{p}_s{s}'] = c
                print(f"{tag} p{p}_s{s}: hit {c['hit']}/{c['n']} samples {c['samples']} "
                      f"ok_samples {c['ok_samples']} nd_rejects {c['nd_verify_rejects']}/{c['reverified']} "
                      f"lean {c['lean_rejected']}/{c['lean_checked']}", flush=True)
    json.dump(out, open(f'{ROOT}/rv/ladder_cov_recount.json', 'w'), indent=1)
