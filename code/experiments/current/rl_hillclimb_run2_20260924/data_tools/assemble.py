#!/usr/bin/env python3
"""ab2 pool assembly: old ladder pools + new generator / textbook labels + cap-6 held-out L6 -> class-disjoint (pkey) files.

  python tools/assemble.py --seed 24005
"""
import argparse, json, os, sys, random, collections, hashlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from keys2 import pkey, canon_key, split_thm

D = '/scratch/cowifornia/nd-rl/ab2/data'
FORK = '/scratch/cowifornia/nd-rl/external/nd-takehome'
CAP6 = '/scratch/cowifornia/nd-rl/ab/common/data'
TBDEV = '/scratch/cowifornia/nd-rl/ab/sealed/textbook_dev.jsonl'
VIS = ('name', 'thm', 'key', 'pkey', 'prompt', 'n_lines', 'L_true', 'source', 'schema', 'n_prem')


def rd(fn):
    return [json.loads(l) for l in open(fn) if l.strip()]


def labels(*fns):
    """name -> final label record; later files (higher bound) override unresolved earlier ones."""
    out = {}
    for fn in fns:
        if not os.path.exists(fn):
            continue
        for x in rd(fn):
            assert 'error' not in x, x
            prev = out.get(x['name'])
            if prev is not None and prev['min_lines_ub'] is not None:
                continue
            out[x['name']] = x
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--seed', type=int, default=24005)
    ap.add_argument('--per_len', type=int, default=2500); ap.add_argument('--dev', type=int, default=300)
    ap.add_argument('--schema_share', type=float, default=0.4)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    st = collections.Counter()

    # ---- exclusion set (key and pkey) ----
    excl_k, excl_p = set(), set()
    for fn in (f'{CAP6}/train_cap6.jsonl', f'{CAP6}/heldout_cap6.jsonl'):
        for r in rd(fn):
            excl_k.add(r['key']); excl_p.add(pkey(r['thm']))
    for l in open(TBDEV):            # textbook dev: only `key` and (for pkey) `thm` are read; nothing else is used or copied
        r = json.loads(l); excl_k.add(r['key']); excl_p.add(pkey(r['thm']))
    for l in open(f'{FORK}/targets/validation_36.jsonl'):
        t = json.loads(l)['thm'].strip(); excl_k.add(canon_key(t)); excl_p.add(pkey(t))
    st['excl_keys'] = len(excl_k); st['excl_pkeys'] = len(excl_p)

    # ---- candidates, in priority order for pkey dedup: old ladder pools first ----
    cands = []
    for pool in ('rl_targets', 'transfer', 'reserve'):
        for r in rd(f'{FORK}/data/ladder/{pool}.jsonl'):
            r = dict(r); r['orig_name'] = r.pop('name'); r['origin'] = f'ladder/{pool}'
            cands.append(r)
    tb = {r['name']: r for r in rd(f'{D}/raw/textbook.jsonl')}
    tbl = labels(f'{D}/labels/tb_b10.jsonl', f'{D}/labels/tb_b16.jsonl')
    gen = {r['name']: r for r in rd(f'{D}/raw/pool_gen.jsonl')}
    genl = labels(f'{D}/labels/gen_b10.jsonl', f'{D}/labels/gen_b16.jsonl')
    for src, recs, lab in (('textbook', tb, tbl), ('gen', gen, genl)):
        st[f'new_{src}_candidates'] = len(recs); st[f'new_{src}_labelled_attempted'] = len(lab)
        for name, x in lab.items():
            L = x['min_lines_ub']
            if L is None:
                st[f'new_{src}_unresolved'] += 1; st[f'new_{src}_unresolved_timeout'] += bool(x.get('timeout'))
                st[f'new_{src}_unresolved_oom'] += bool(x.get('oom')); continue
            r = recs[name]
            rec = {'orig_name': name, 'origin': f'ab2/{src}', 'thm': r['thm'], 'key': r['key'], 'prompt': r['prompt'], 'n_lines': L,
                   'L_true': L, 'minlen_bound': x['bound'], 'source': src, 'schema': r.get('schema'),
                   'gen_lines': r['n_lines'] if src == 'gen' else None, 'n_prem': r['n_prem'], 'rules': r.get('rules'),
                   'minlen_proof': x['proof']}
            if src == 'gen':
                rec['gen_proof'] = r['proof']
            cands.append(rec)
    # cap-6 held-out, generator length 6 (the L6 training bin; exempt from the held-out exclusion by construction)
    hl = labels(f'{D}/labels/heldout6_b6.jsonl')
    held6 = []
    for r in rd(f'{CAP6}/heldout_cap6.jsonl'):
        if r['n_lines'] != 6:
            continue
        x = hl[r['name']]
        L = x['min_lines_ub']
        if L is None:                 # restricted search found no proof <= 6: unresolved, dropped like every unresolved theorem
            st['heldout6_unresolved'] += 1; continue
        held6.append({'orig_name': r['name'], 'origin': 'cap6/heldout_cap6', 'thm': r['thm'], 'key': r['key'], 'pkey': pkey(r['thm']),
                      'prompt': r['prompt'], 'n_lines': L, 'L_true': L, 'minlen_bound': 6, 'source': 'cap6_heldout', 'schema': None,
                      'gen_lines': 6, 'n_prem': r['n_prem']})
    st['heldout6_n'] = len(held6)
    st['heldout6_Ltrue_hist'] = str(dict(sorted(collections.Counter(r['L_true'] for r in held6).items())))
    held6 = [r for r in held6 if r['L_true'] == 6]      # records whose true length is < 6 are not used (the bin is topped up below)
    st['heldout6_used_Ltrue6'] = len(held6)

    pool, seen = [], set(r['pkey'] for r in held6)
    for r in cands:
        o = r['origin'].split('/')[0] + '_' + r['source']
        r['key'] = r.get('key') or canon_key(r['thm'])
        assert r['key'] == canon_key(r['thm'])
        r['pkey'] = pkey(r['thm'])
        if r['key'] in excl_k or r['pkey'] in excl_p:
            st[f'excluded_{o}'] += 1; st[f'excluded_{o}_by_key'] += r['key'] in excl_k; continue
        if r['pkey'] in seen:
            st[f'dup_pkey_{o}'] += 1; continue
        seen.add(r['pkey']); pool.append(r)
    st['pool_after_excl_dedup'] = len(pool)

    by = collections.defaultdict(list)          # L -> records
    for r in pool:
        by[r['L_true']].append(r)
    for L in by:
        rng.shuffle(by[L])

    def draw(L, n):
        """Take n records at length L: textbook share <= schema_share unless the generator runs out; textbook drawn
        round-robin across schemata. Removes them from by[L]."""
        rs = by[L]
        g = [r for r in rs if r['source'] == 'gen']; t = [r for r in rs if r['source'] == 'textbook']
        n = min(n, len(rs))
        n_t = min(len(t), max(int(a.schema_share * n), n - len(g)))
        n_g = n - n_t
        sch = collections.defaultdict(list)
        for r in t:
            sch[r['schema']].append(r)
        tt, names = [], sorted(sch)
        while len(tt) < n_t:
            for s in names:
                if sch[s] and len(tt) < n_t:
                    tt.append(sch[s].pop())
        sel = g[:n_g] + tt
        ids = set(map(id, sel))
        by[L] = [r for r in rs if id(r) not in ids]
        rng.shuffle(sel)
        return sel

    files = {}
    files['dev_L8-11'] = [r for L in (8, 9, 10, 11) for r in draw(L, a.dev)]
    fr, se = [], []
    for L in (12, 13, 14):
        grp = collections.defaultdict(list)
        for r in by[L]:
            grp[r['source']].append(r)
        for s in sorted(grp):
            rs = sorted(grp[s], key=lambda r: (r['schema'] or '', rng.random()))   # schema-grouped, random within
            for i, r in enumerate(rs):
                (fr if i % 2 == 0 else se).append(r)
        by[L] = []
    files['frontier_L12-14'] = fr; files['sealed_L12-14'] = se
    files['sealed_L15-16'] = [r for L in (15, 16) for r in by[L]]
    for L in (15, 16):
        by[L] = []
    tg = list(held6) + draw(6, 1000 - len(held6))      # L6: held-out cap-6 records with L_true 6, topped up to 1,000 from the new pool
    for L in (7, 8, 9, 10):
        tg += draw(L, a.per_len)
    tg += draw(11, len(by[11]))
    files['targets_L6-11'] = tg
    files['reserve'] = [r for L in sorted(by) for r in by[L]]
    for k in files:
        rng.shuffle(files[k])

    # ---- checks ----
    allp = collections.Counter()
    for k, rs in files.items():
        ps = [r['pkey'] for r in rs]
        assert len(ps) == len(set(ps)), k
        allp.update(set(ps))
    assert max(allp.values()) == 1, 'files not pkey-disjoint'
    held6p = {r['pkey'] for r in held6}
    for k, rs in files.items():
        for r in rs:
            if r['pkey'] in held6p:
                assert k == 'targets_L6-11' and r['source'] == 'cap6_heldout'
            else:
                assert r['pkey'] not in excl_p and r['key'] not in excl_k
            assert r['n_lines'] == r['L_true']
    # ---- write ----
    prov = []
    summary = {'stats': dict(st), 'files': {}}
    for k, rs in files.items():
        visible = not (k.startswith('sealed') or k == 'reserve')
        with open(f'{D}/{k}.jsonl', 'w') as f:
            for i, r in enumerate(rs):
                name = f'ab2_{k}_{i}'
                prov.append({'name': name, 'file': k, 'orig_name': r['orig_name'], 'origin': r['origin'], 'minlen_bound': r.get('minlen_bound'), 'gen_lines': r.get('gen_lines')})
                out = {**r, 'name': name}
                if visible:
                    out = {f: out.get(f) for f in VIS}
                f.write(json.dumps(out) + '\n')
        summary['files'][k] = {'n': len(rs), 'by_L_source': {f'{L}_{s}': c for (L, s), c in sorted(collections.Counter((r['L_true'], r['source']) for r in rs).items())}}
    with open(f'{D}/provenance.jsonl', 'w') as f:
        for p in prov:
            f.write(json.dumps(p) + '\n')
    json.dump(summary, open(f'{D}/pools_summary.json', 'w'), indent=1)
    print(json.dumps(summary['stats'], indent=1))
    # table
    Ls = sorted({r['L_true'] for rs in files.values() for r in rs})
    print('| file | source | ' + ' | '.join(map(str, Ls)) + ' | total |')
    for k, rs in files.items():
        for s in sorted({r['source'] for r in rs}):
            c = collections.Counter(r['L_true'] for r in rs if r['source'] == s)
            print(f'| {k} | {s} | ' + ' | '.join(f'{c[L]:,}' if c[L] else '' for L in Ls) + f' | {sum(c.values()):,} |')


if __name__ == '__main__':
    main()
