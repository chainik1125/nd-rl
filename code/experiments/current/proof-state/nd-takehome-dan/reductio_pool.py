#!/usr/bin/env python3
"""Follow-up block B: a reductio target pool whose double negation must be DERIVED.

Every theorem is (i) classically valid (truth table, asserted), (ii) NOT intuitionistically provable (intuit.py,
asserted), (iii) free of any ( ~ ( ~ X ) ) sub-formula in premises and conclusion, (iv) disjoint by renaming class
from every training set, the held-out set, the old target pools and validation-36, (v) after minlen.py (bound 8),
has min_lines_ub >= 7 or None.  Since the only classical rule is DN and no double negation is given, every proof
must apply DN to a double negation it derived itself (or assumed and discharged).

Two sources, kept in a `source` field:
  schema     : classical-only schemata instantiated with random sub-formulas (as textbook_pool.py), no ( ~ ( ~ ;
  generator  : the unchanged generator's own reductio theorems that pass (ii)-(v) (rare: 7 in 19,099 in the
               first 20M-try run), all of them go into the targets.

  python reductio_pool.py gen --out data/p2/reductio2_cands.jsonl --exclude data/p2/pool_cap6_recon.jsonl ...
  python minlen.py --in data/p2/reductio2_cands.jsonl --out data/p2/reductio2_cands_minlen.jsonl --bound 8 --time 20 --procs 5
  python reductio_pool.py finalize --cands data/p2/reductio2_cands.jsonl --minlen data/p2/reductio2_cands_minlen.jsonl \
      --native data/p2/pool_rnd_co_minlen.jsonl --native_pool data/p2/pool_rnd_co.jsonl --outdir data/p2
"""
import argparse, json, random, sys, os, collections, itertools, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen import fstr, canon_key, ATOMS
from intuit import intuit_provable
from textbook_pool import rformula, N, AND, OR, IMP

# classical-only schemata with no double negation anywhere (A, B, C metavariables)
SCHEMATA = {
    'negimp_to_pos':        lambda A, B, C: ([IMP(N(A), B), N(B)], A),                    # ~A>B, ~B |- A
    'nand_neg':             lambda A, B, C: ([N(AND(N(A), B)), B], A),                    # ~(~A & B), B |- A
    'peirce_seq':           lambda A, B, C: ([IMP(IMP(A, B), A)], A),                     # (A>B)>A |- A
    'peirce':               lambda A, B, C: ([], IMP(IMP(IMP(A, B), A), A)),
    'negcond_ante':         lambda A, B, C: ([N(IMP(A, B))], A),                          # ~(A>B) |- A
    'demorgan_nor_neg':     lambda A, B, C: ([N(AND(N(A), N(B)))], OR(A, B)),             # ~(~A & ~B) |- A v B
    'negimp_to_or':         lambda A, B, C: ([IMP(N(A), B)], OR(A, B)),                   # ~A>B |- A v B
    'imp_to_or':            lambda A, B, C: ([IMP(A, B)], OR(N(A), B)),                   # A>B |- ~A v B
    'excluded_middle':      lambda A, B, C: ([], OR(A, N(A))),
    'demorgan_nand':        lambda A, B, C: ([N(AND(A, B))], OR(N(A), N(B))),             # ~(A&B) |- ~A v ~B
    'contraposition_conv':  lambda A, B, C: ([IMP(N(B), N(A))], IMP(A, B)),               # ~B>~A |- A>B
    'neg_both':             lambda A, B, C: ([IMP(N(A), B), IMP(N(A), N(B))], A),         # ~A>B, ~A>~B |- A
    'neg_to_contra':        lambda A, B, C: ([IMP(N(A), AND(B, N(B)))], A),               # ~A>(B&~B) |- A
    'dummett':              lambda A, B, C: ([], OR(IMP(A, B), IMP(B, A))),
    'chain_neg':            lambda A, B, C: ([IMP(N(A), B), IMP(C, N(B)), C], A),         # ~A>B, C>~B, C |- A
    'impimp_to_or':         lambda A, B, C: ([IMP(IMP(A, B), B)], OR(A, B)),              # (A>B)>B |- A v B
    'cases_neg':            lambda A, B, C: ([IMP(A, B), IMP(N(A), B)], B),               # A>B, ~A>B |- B
    'nand_to_imp':          lambda A, B, C: ([N(AND(A, N(B)))], IMP(A, B)),               # ~(A & ~B) |- A>B
    'nor_neg_ante':         lambda A, B, C: ([N(OR(N(A), B))], A),                        # ~(~A v B) |- A
    'imp_cases_or':         lambda A, B, C: ([IMP(A, B), IMP(N(A), C)], OR(B, C)),        # A>B, ~A>C |- B v C
    'consequentia_cond':    lambda A, B, C: ([IMP(N(A), A), IMP(A, B)], B),               # ~A>A, A>B |- B
}


def ev(f, v):
    t = f[0]
    if t == 'atom': return v[f[1]]
    if t == 'bot': return False
    if t == 'not': return not ev(f[1], v)
    a, b = ev(f[1], v), ev(f[2], v)
    return a and b if t == 'and' else (a or b if t == 'or' else (not a) or b)


def atoms_of(f, acc):
    if f[0] == 'atom': acc.add(f[1])
    elif f[0] != 'bot':
        for x in f[1:]: atoms_of(x, acc)
    return acc


def valid(prem, concl):
    at = sorted(atoms_of(concl, set().union(*[atoms_of(p, set()) for p in prem])) if prem else atoms_of(concl, set()))
    for bits in itertools.product([False, True], repeat=len(at)):
        v = dict(zip(at, bits))
        if all(ev(p, v) for p in prem) and not ev(concl, v):
            return False
    return True


def has_dn(f):
    if f[0] == 'not' and f[1][0] == 'not': return True
    return any(has_dn(x) for x in f[1:] if isinstance(x, tuple))


def cmd_gen(a):
    rng = random.Random(a.seed)
    excl = {canon_key(json.loads(l)['thm'].strip()) for l in open('targets/validation_36.jsonl')}
    for fn in a.exclude:
        for l in open(fn):
            if l.strip():
                excl.add(json.loads(l)['key'])
    print('excluded classes', len(excl), flush=True)
    names = sorted(SCHEMATA)
    out, seen = [], set(); stats = collections.Counter()
    for name in names:
        got = 0; tries = 0
        while got < a.per_schema and tries < 50000:
            tries += 1
            depth = rng.choice([0, 1, 1, 2])
            A, B, C = (rformula(rng, depth) for _ in range(3))
            if len({A, B, C}) < 3:
                stats['degenerate'] += 1; continue
            prem, concl = SCHEMATA[name](A, B, C)
            if concl in prem or len(set(prem)) != len(prem) or any(has_dn(f) for f in prem + [concl]):
                stats['degenerate_or_dn'] += 1; continue
            thm = (' , '.join(fstr(p) for p in prem) + ' |- ' + fstr(concl)).strip()
            assert '( ~ ( ~' not in thm
            key = canon_key(thm)
            if key in excl or key in seen:
                stats['dup_or_excluded'] += 1; continue
            prompt = f'THM {" , ".join(fstr(p) for p in prem)} SEQ {fstr(concl)} PRF' if prem else f'THM SEQ {fstr(concl)} PRF'
            if len(prompt.split()) > a.max_prompt_toks:
                stats['too_long'] += 1; continue
            if not valid(prem, concl):
                stats['INVALID'] += 1; continue      # never expected: schemata are classically valid
            if intuit_provable(prompt):
                stats['intuit_provable'] += 1; continue   # never expected for these schemata, but asserted per instance
            seen.add(key)
            out.append({'name': f'{name}_{got}', 'schema': name, 'source': 'schema', 'thm': thm, 'key': key, 'prompt': prompt,
                        'n_prem': len(prem), 'n_lines': None})
            got += 1
        stats[name] = got
    rng.shuffle(out)
    with open(a.out, 'w') as f:
        for r in out:
            f.write(json.dumps(r) + '\n')
    print(json.dumps({'n': len(out), 'stats': dict(stats)}, indent=1))


def cmd_finalize(a):
    rng = random.Random(a.seed)
    ml = {json.loads(l)['name']: json.loads(l) for l in open(a.minlen)}
    cands = [json.loads(l) for l in open(a.cands)]
    keep = []
    for r in cands:
        m = ml[r['name']]
        r['min_lines_ub'] = m['min_lines_ub']; r['minlen_timeout'] = m.get('timeout')
        if m['min_lines_ub'] is not None and m['min_lines_ub'] < a.min_ub:
            continue
        r['n_lines'] = m['min_lines_ub'] if m['min_lines_ub'] is not None else 9
        keep.append(r)
    print(f'schema candidates {len(cands)} -> {len(keep)} with min_lines_ub >= {a.min_ub} or None')
    by = collections.defaultdict(list)
    for r in keep:
        by[r['schema']].append(r)
    print({s: len(v) for s, v in sorted(by.items())})
    targets, transfer = [], []
    for s, rs in sorted(by.items()):
        rng.shuffle(rs)
        nt = min(len(rs) * 2 // 3, a.per_schema_t)
        targets += rs[:nt]; transfer += rs[nt:nt + a.per_schema_x]
    # generator-native theorems (all into targets)
    native = []
    if a.native and os.path.exists(a.native):
        pool = {json.loads(l)['name']: json.loads(l) for l in open(a.native_pool)} if a.native_pool else {}
        used = {r['key'] for r in targets + transfer}
        for l in open(a.native):
            m = json.loads(l)
            if m['min_lines_ub'] is not None and m['min_lines_ub'] < a.min_ub:
                continue
            p = pool.get(m['name'], {})
            thm = m['thm']; key = canon_key(thm.strip())
            if key in used or '( ~ ( ~' in thm:
                continue
            assert not intuit_provable(m['prompt'])
            used.add(key)
            native.append({'name': f"native_{len(native)}", 'schema': 'generator', 'source': 'generator', 'thm': thm, 'key': key, 'prompt': m['prompt'],
                           'n_prem': p.get('n_prem', m['prompt'].split(' SEQ ')[0].count(' , ') + (0 if m['prompt'].startswith('THM SEQ') else 1)),
                           'n_lines': p.get('n_lines', m.get('gen_lines')) or 9, 'gen_proof': p.get('proof'), 'min_lines_ub': m['min_lines_ub'], 'minlen_timeout': m.get('timeout')})
    targets += native
    rng.shuffle(targets); rng.shuffle(transfer)
    for name, recs in (('targets', targets), ('transfer', transfer)):
        with open(f'{a.outdir}/{name}_reductio2.jsonl', 'w') as f:
            for i, r in enumerate(recs):
                r = dict(r); r['name'] = f'{name}_reductio2_{i}'
                r['pat'] = {'derived_ore': False, 'reductio': True, 'depth3': False}
                f.write(json.dumps(r) + '\n')
        print(name, len(recs), 'native', sum(r['source'] == 'generator' for r in recs),
              'min_lines_ub hist', dict(sorted(collections.Counter(str(r['min_lines_ub']) for r in recs).items())),
              'by schema', dict(sorted(collections.Counter(r['schema'] for r in recs).items())))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    g = sub.add_parser('gen'); g.add_argument('--out', required=True); g.add_argument('--exclude', nargs='*', default=[])
    g.add_argument('--per_schema', type=int, default=60); g.add_argument('--seed', type=int, default=0); g.add_argument('--max_prompt_toks', type=int, default=80)
    f = sub.add_parser('finalize'); f.add_argument('--cands', required=True); f.add_argument('--minlen', required=True)
    f.add_argument('--native', default=None); f.add_argument('--native_pool', default=None); f.add_argument('--outdir', default='data/p2')
    f.add_argument('--min_ub', type=int, default=7); f.add_argument('--per_schema_t', type=int, default=30); f.add_argument('--per_schema_x', type=int, default=15)
    f.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    {'gen': cmd_gen, 'finalize': cmd_finalize}[a.cmd](a)


if __name__ == '__main__':
    main()
