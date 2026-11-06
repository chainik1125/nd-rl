"""Constructive sampler: sequents built from a random tree-like resolution derivation of a literal goal, so that
4-10 premises are all needed (checked by truth table afterwards; redundant ones are rejected).

  D(l)        : derive literal l  = unit premise l | SPLIT (a v b) with D(a -> l) and D(b -> l)
  D(a -> l)   : derive l from a   = one clause linking a to l (encoded as (~a v l), (a > l), (l v ~a) or ~(a & ~l))
                                  | chain: clause linking a to c, then D(c -> l)
                                  | split inside: (c v d) plus D(a&c -> l) ... (kept simple: not used)
Atoms are reused at random, which creates shortcuts; the necessity filter rejects those.

    python sample_tree.py OUT.jsonl N SEED [--atoms 3-6] [--prems 4-10]
"""
import argparse, json, os, random, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import certify as C
import sample_cnf as SC

LETTERS = C.LETTERS


def neg(l):
    return l[1] if l[0] == 'not' else ('not', l)


def link(rng, a, l):
    """One premise from which l follows given a."""
    k = rng.randrange(5)
    if k == 0:
        return ('or', neg(a), l)
    if k == 1:
        return ('or', l, neg(a))
    if k == 2 or k == 3:
        return ('imp', a, l)
    return ('not', ('and', a, neg(l)))


class Builder:
    def __init__(self, rng, atoms, max_prem):
        self.rng, self.atoms, self.max_prem = rng, atoms, max_prem
        self.prem = []

    def lit(self):
        a = ('atom', self.rng.choice(self.atoms))
        return ('not', a) if self.rng.random() < 0.4 else a

    def D(self, l, budget):
        """Add premises deriving l; budget = premises allowed."""
        if budget <= 1 or self.rng.random() < 0.15:
            self.prem.append(l)
            return
        a, b = self.lit(), self.lit()
        self.prem.append(('or', a, b))
        half = (budget - 1) // 2
        self.Dfrom(a, l, half)
        self.Dfrom(b, l, budget - 1 - half)

    def Dfrom(self, a, l, budget):
        if a == l:
            return
        if a == neg(l) or a == neg(a):
            return
        if budget <= 1 or self.rng.random() < 0.5:
            self.prem.append(link(self.rng, a, l))
            return
        c = self.lit()
        self.prem.append(link(self.rng, a, c))
        self.Dfrom(c, l, budget - 1)


def sample(rng, atom_range, prem_range):
    k = rng.randint(*atom_range)
    atoms = LETTERS[:k]
    m = rng.randint(*prem_range)
    b = Builder(rng, atoms, m)
    goal = b.lit()
    b.D(goal, m)
    prem = []
    for p in b.prem:
        if p not in prem:
            prem.append(p)
    if not (prem_range[0] <= len(prem) <= prem_range[1]) or goal in prem:
        return None
    rng.shuffle(prem)
    sat, valid = C.truth_table(prem, goal)
    if not (sat and valid):
        return None
    for i in range(len(prem)):
        if C.truth_table(prem[:i] + prem[i + 1:], goal)[1]:
            return None
    return prem, goal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out'); ap.add_argument('n', type=int); ap.add_argument('seed', type=int)
    ap.add_argument('--atoms', default='3-6'); ap.add_argument('--prems', default='4-10')
    ap.add_argument('--prove', action='store_true', help='also write a proof (prover.py) and emit text records')
    a = ap.parse_args()
    ar = tuple(int(x) for x in a.atoms.split('-')); pr = tuple(int(x) for x in a.prems.split('-'))
    rng = random.Random(a.seed)
    seen, n, tries = set(), 0, 0
    if a.prove:
        import prover
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
            prompt = SC.prompt_of(*s)
            rec = {'prompt': prompt, 'n_prem': len(s[0])}
            if a.prove:
                body = prover.prove_text(prompt)
                if body is None:
                    print('prover failed', prompt, file=sys.stderr); continue
                rec['text'] = prompt + ' ' + body
                rec['gen_body'] = body.count(';') - len(s[0])
            f.write(json.dumps(rec) + '\n')
            n += 1
    print(f'{n} sequents from {tries} draws')


if __name__ == '__main__':
    main()
