"""Theorem-first sampler, clause family: premises are small formulas over literals (a v b, a > b, a & b, ~(a & b),
literal), conclusion a literal or a 2-literal connective. Keeps sequents that are valid, have consistent premises,
and where EVERY premise is necessary (dropping any one makes it invalid). Writes {"prompt": ...} jsonl.

    python sample_cnf.py OUT.jsonl N SEED [--atoms 3-7] [--prems 3-10]
"""
import argparse, itertools, json, os, random, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import certify as C  # noqa: E402  (truth_table, fstr, LETTERS)

SHAPES = ['or', 'or', 'imp', 'imp', 'and', 'nand', 'lit']


def lit(rng, atoms):
    a = ('atom', rng.choice(atoms))
    return ('not', a) if rng.random() < 0.4 else a


def premise(rng, atoms, shape):
    a, b = lit(rng, atoms), lit(rng, atoms)
    if shape == 'lit':
        return a
    if a[-1] == b[-1] and a == b:
        return None
    if shape == 'or':
        return ('or', a, b)
    if shape == 'imp':
        return ('imp', a, b)
    if shape == 'and':
        return ('and', a, b)
    return ('not', ('and', a, b))


def conclusion(rng, atoms):
    r = rng.random()
    a, b = lit(rng, atoms), lit(rng, atoms)
    if r < 0.5 or a == b:
        return a
    if r < 0.65:
        return ('and', a, b)
    if r < 0.8:
        return ('or', a, b)
    if r < 0.9:
        return ('imp', a, b)
    return ('not', ('and', a, b))


def sample(rng, atom_range, prem_range):
    k = rng.randint(*atom_range)
    atoms = C.LETTERS[:k]
    m = rng.randint(*prem_range)
    prem = []
    while len(prem) < m:
        p = premise(rng, atoms, rng.choice(SHAPES))
        if p is not None and p not in prem:
            prem.append(p)
    concl = conclusion(rng, atoms)
    if concl in prem:
        return None
    sat, valid = C.truth_table(prem, concl)
    if not (sat and valid):
        return None
    for i in range(m):                       # every premise necessary
        if C.truth_table(prem[:i] + prem[i + 1:], concl)[1]:
            return None
    return prem, concl


def prompt_of(prem, concl):
    return 'THM ' + ' , '.join(C.fstr(p) for p in prem) + ' SEQ ' + C.fstr(concl) + ' PRF'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out'); ap.add_argument('n', type=int); ap.add_argument('seed', type=int)
    ap.add_argument('--atoms', default='3-7'); ap.add_argument('--prems', default='3-10')
    a = ap.parse_args()
    ar = tuple(int(x) for x in a.atoms.split('-')); pr = tuple(int(x) for x in a.prems.split('-'))
    rng = random.Random(a.seed)
    seen, n, tries = set(), 0, 0
    with open(a.out, 'w') as f:
        while n < a.n:
            tries += 1
            s = sample(rng, ar, pr)
            if s is None:
                continue
            key = C.theorem_key(*s)
            if key in seen:
                continue
            seen.add(key)
            f.write(json.dumps({'prompt': prompt_of(*s), 'n_prem': len(s[0])}) + '\n')
            n += 1
    print(f'{n} sequents from {tries} draws')


if __name__ == '__main__':
    main()
