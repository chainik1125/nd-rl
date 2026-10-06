#!/usr/bin/env python3
"""Theorem-level selector splits, written BEFORE any pool is generated (v2, after the 02:08 design audit item 6).

Equivalence key = atom renaming + premise permutation: for every ordering of the premises, rename atoms in order of
first appearance over (premises..., conclusion), take the lexicographically smallest rendering.  Theorems sharing a
key (within or across sets) form one group; a group never straddles train/dev, and a group containing an rl_targets
theorem is train.  train: all rl_targets + ~70% of dev1108 / rr600 groups; dev: the rest; test (frozen): holdout250,
textbook72.  Asserts no key is shared between train/dev sources and the test sets."""
import itertools, json, os, random, re, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pool_lib as PL

ATOM = re.compile(r'^[A-Z]$')


def split_top(toks):
    out, cur, d = [], [], 0
    for t in toks:
        if t == '(':
            d += 1
        elif t == ')':
            d -= 1
        if t == ',' and d == 0:
            out.append(cur); cur = []
        else:
            cur.append(t)
    if cur:
        out.append(cur)
    return out


def canon_key(prompt):
    toks = prompt.split()
    assert toks[0] == 'THM' and toks[-1] == 'PRF', prompt
    i = toks.index('SEQ')
    prem, concl = split_top(toks[1:i]), toks[i + 1:-1]
    best = None
    perms = itertools.permutations(prem) if len(prem) <= 6 else [sorted(prem)]
    for pp in perms:
        ren = {}
        seq = []
        for f in list(pp) + [['|-'] + concl]:
            g = [ren.setdefault(t, f'a{len(ren)}') if ATOM.match(t) else t for t in f]
            seq.append(' '.join(g))
        s = ' , '.join(seq[:-1]) + ' ' + seq[-1]
        best = s if best is None or s < best else best
    return best


def main():
    sets = {s: PL.load_set(s) for s in PL.SET_FILES}
    keys = {s: [canon_key(r['prompt']) for r in rows] for s, rows in sets.items()}
    rep = {}
    for a, b in itertools.combinations(sets, 2):
        rep[f'{a}~{b}'] = len(set(keys[a]) & set(keys[b]))
    within = {s: len(v) - len(set(v)) for s, v in keys.items()}
    test_keys = set(keys['holdout250']) | set(keys['textbook72'])
    rl_keys = set(keys['rl_targets'])
    rng = random.Random(20261006)
    out = dict(created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), version=2, seed=20261006,
               unit='equivalence group (atom renaming + premise permutation)', key_overlap_between_sets=rep,
               duplicate_keys_within_set=within, train={}, dev={}, test={}, excluded={})
    out['train']['rl_targets'] = [r['name'] for r in sets['rl_targets']]
    for s in ('dev1108', 'rr600'):
        rows = sets[s]
        gkeys = sorted(set(keys[s]) - rl_keys); rng.shuffle(gkeys)
        tr = set(gkeys[:round(0.7 * len(gkeys))]) | rl_keys
        out['excluded'][s] = [r['name'] for r, k in zip(rows, keys[s]) if k in test_keys]
        out['train'][s] = [r['name'] for r, k in zip(rows, keys[s]) if k in tr and k not in test_keys]
        out['dev'][s] = [r['name'] for r, k in zip(rows, keys[s]) if k not in tr and k not in test_keys]
    # a group may span dev1108 and rr600: keep it on one side
    trk = {k for s in ('dev1108', 'rr600') for r, k in zip(sets[s], keys[s]) if r['name'] in set(out['train'][s])} | rl_keys
    dvk = {k for s in ('dev1108', 'rr600') for r, k in zip(sets[s], keys[s]) if r['name'] in set(out['dev'][s])}
    clash = trk & dvk
    for s in ('dev1108', 'rr600'):
        moved = [r['name'] for r, k in zip(sets[s], keys[s]) if k in clash and r['name'] in set(out['dev'][s])]
        out['dev'][s] = [n for n in out['dev'][s] if n not in set(moved)]
        out['train'][s] += moved
    for s in ('holdout250', 'textbook72'):
        out['test'][s] = [r['name'] for r in sets[s]]
    tr_all = {k for s in out['train'] for r, k in zip(sets[s], keys[s]) if r['name'] in set(out['train'][s])}
    dv_all = {k for s in out['dev'] for r, k in zip(sets[s], keys[s]) if r['name'] in set(out['dev'][s])}
    assert not (tr_all & dv_all) and not (tr_all & test_keys) and not (dv_all & test_keys)
    out['counts'] = {sp: {s: len(v) for s, v in out[sp].items()} for sp in ('train', 'dev', 'test', 'excluded')}
    fn = os.path.join(PL.SPRINT, 'results', 'splits.json')
    json.dump(out, open(fn, 'w'), indent=1)
    print(json.dumps({k: out[k] for k in ('counts', 'key_overlap_between_sets', 'duplicate_keys_within_set', 'created_utc')}))


if __name__ == '__main__':
    main()
