#!/usr/bin/env python3
"""Run 2: IMPE-chain target candidates from schemata (the take-home generator never emits an IMPE chain of length >= 4:
0 in 9.4M tries even with max_prem 6, because its lazy-premise recursion is shallow).  Schemata keep the prompt inside the
generator's form (<= 3 premises) and give 8-10-line proofs; instances use random sub-formulas (textbook_pool.rformula), are
class-deduplicated and exclude validation-36 and --exclude classes.  Necessity (necessity.py --pattern impe_chain4) then
keeps the instances whose SHORTEST found proof has a chain >= 4 ("uses" stratum; chains have no restriction oracle).

  python chain_pool.py --n 1500 --out data/r2/cands_impe_chain4.jsonl --exclude data/p2/heldout.jsonl --seed 0
"""
import argparse, json, random, sys, os, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen import fstr, canon_key
from textbook_pool import rformula, AND, IMP

SCHEMATA = {   # (premises, conclusion) -> chain of 4 IMPE; A..E metavariables
    'major_conj2':  lambda A, B, C, D, E: ([AND(A, B), IMP(A, IMP(B, IMP(A, IMP(B, E))))], E),                    # 8 lines
    'major_conj3':  lambda A, B, C, D, E: ([AND(A, AND(B, C)), IMP(A, IMP(B, IMP(C, IMP(A, E))))], E),            # 10
    'major_conj3b': lambda A, B, C, D, E: ([AND(AND(A, B), C), IMP(A, IMP(B, IMP(C, IMP(D, E)))), D], E),        # 10
    'minor_chain':  lambda A, B, C, D, E: ([AND(A, AND(IMP(A, B), IMP(B, C))), IMP(C, IMP(A, D))], D),           # 10
    'minor_chain2': lambda A, B, C, D, E: ([A, AND(IMP(A, B), IMP(B, C)), IMP(C, IMP(B, D))], D),                # 9
    'mixed':        lambda A, B, C, D, E: ([AND(A, B), IMP(A, IMP(B, C)), IMP(C, IMP(A, E))], E),                # 8
    'mixed2':       lambda A, B, C, D, E: ([A, IMP(A, B), IMP(B, IMP(A, C)), IMP(C, IMP(B, E))], E),             # 9, 4 premises
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=1500); ap.add_argument('--out', required=True); ap.add_argument('--exclude', nargs='*', default=[])
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--max_prompt_toks', type=int, default=70)
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
            fs = [rformula(rng, rng.choice([0, 1, 1, 2])) for _ in range(5)]
            if len(set(fs)) < 5:
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
            out.append({'name': f'chain_{name}_{got}', 'schema': name, 'source': 'schema', 'thm': thm, 'key': key, 'prompt': prompt, 'n_prem': len(prem), 'n_lines': None})
        st[name] = got
    rng.shuffle(out)
    with open(a.out, 'w') as f:
        for r in out[:a.n]:
            f.write(json.dumps(r) + '\n')
    print(json.dumps({'n': min(a.n, len(out)), 'stats': dict(st)}))


if __name__ == '__main__':
    main()
