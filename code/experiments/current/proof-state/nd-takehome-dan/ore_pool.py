#!/usr/bin/env python3
"""Run 2: nested-ORE target candidates from schemata (the generator never nests an ORE inside an ORE branch: 0 in 9M long-mode
tries).  Instances use random sub-formulas; class-deduplicated; validation-36 and --exclude classes dropped.  necessity.py
--pattern nested_ore (bound 13, restriction ORE_IN_ORE) then labels requires / uses.
  python ore_pool.py --n 800 --out data/r2/cands_nested_ore.jsonl --exclude data/p2/heldout.jsonl
"""
import argparse, json, random, sys, os, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen import fstr, canon_key
from textbook_pool import rformula, AND, OR, IMP, N

SCHEMATA = {
    'assoc':      lambda A, B, C, D: ([OR(OR(A, B), C)], OR(A, OR(B, C))),                     # 12 lines
    'assoc_rev':  lambda A, B, C, D: ([OR(A, OR(B, C))], OR(OR(A, B), C)),                     # 12
    'or_and':     lambda A, B, C, D: ([OR(A, B), OR(A, C)], OR(A, AND(B, C))),                 # 12
    'or_imp':     lambda A, B, C, D: ([OR(A, B), OR(A, C), IMP(B, IMP(C, D))], OR(A, D)),      # 13
    'swap3':      lambda A, B, C, D: ([OR(OR(A, B), C)], OR(C, OR(B, A))),                     # 12
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=800); ap.add_argument('--out', required=True); ap.add_argument('--exclude', nargs='*', default=[])
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--max_prompt_toks', type=int, default=60)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    excl = {canon_key(json.loads(l)['thm'].strip()) for l in open('targets/validation_36.jsonl')}
    for fn in a.exclude:
        for l in open(fn):
            if l.strip():
                excl.add(json.loads(l)['key'])
    names = sorted(SCHEMATA); per = a.n // len(names) + 1
    out, seen, st = [], set(), collections.Counter()
    for name in names:
        got = tries = 0
        while got < per and tries < 100000:
            tries += 1
            fs = [rformula(rng, rng.choice([0, 0, 1, 1])) for _ in range(4)]
            if len(set(fs)) < 4:
                st['degenerate'] += 1; continue
            prem, concl = SCHEMATA[name](*fs)
            if concl in prem or len(set(prem)) != len(prem):
                st['degenerate'] += 1; continue
            thm = (' , '.join(fstr(p) for p in prem) + ' |- ' + fstr(concl)).strip()
            key = canon_key(thm)
            if key in excl or key in seen:
                st['dup_or_excluded'] += 1; continue
            prompt = f'THM {" , ".join(fstr(p) for p in prem)} SEQ {fstr(concl)} PRF'
            if len(prompt.split()) > a.max_prompt_toks:
                st['too_long'] += 1; continue
            seen.add(key); got += 1
            out.append({'name': f'nore_{name}_{got}', 'schema': name, 'source': 'schema', 'thm': thm, 'key': key, 'prompt': prompt, 'n_prem': len(prem), 'n_lines': None})
        st[name] = got
    rng.shuffle(out)
    with open(a.out, 'w') as f:
        for r in out[:a.n]:
            f.write(json.dumps(r) + '\n')
    print(json.dumps({'n': min(a.n, len(out)), 'stats': dict(st)}))


if __name__ == '__main__':
    main()
