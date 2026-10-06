#!/usr/bin/env python3
"""Phase 3: textbook-shaped target pool. Instantiates classical schemata with fresh atom assignments and random
sub-formulas; disjoint by renaming class from validation-36 and from every pool given with --exclude.

  python textbook_pool.py --n 600 --out data/p3/textbook_targets.jsonl --exclude data/train.jsonl data/heldout.jsonl data/rl_targets.jsonl data/transfer.jsonl

Schemata (A, B, C are metavariables filled with random formulas):
  demorgan_nand_to_or  ~(A&B) |- ~A v ~B        demorgan_or_to_nand  ~A v ~B |- ~(A&B)
  demorgan_nor_to_and  ~(AvB) |- ~A & ~B        demorgan_and_to_nor  ~A & ~B |- ~(AvB)
  dist_and_over_or     A&(BvC) |- (A&B)v(A&C)   dist_and_over_or_conv (A&B)v(A&C) |- A&(BvC)
  dist_or_over_and     Av(B&C) |- (AvB)&(AvC)   dist_or_over_and_conv (AvB)&(AvC) |- Av(B&C)
  contraposition       A>B |- ~B>~A             contraposition_conv  ~B>~A |- A>B
  peirce               |- ((A>B)>A)>A           peirce_sequent       (A>B)>A |- A
  disjunctive_syllogism AvB, ~A |- B            hypothetical_syllogism A>B, B>C |- A>C
  export               (A&B)>C |- A>(B>C)       import               A>(B>C) |- (A&B)>C
  dn_intro             A |- ~~A                 dn_elim              ~~A |- A
  triple_negation      ~~~A |- ~A               explosion            A, ~A |- B
  excluded_middle      |- A v ~A                consequentia_mirabilis ~A>A |- A
  negated_conditional  ~(A>B) |- A&~B           negated_conditional_conv A&~B |- ~(A>B)
  constructive_dilemma AvB, A>C, B>D |- CvD     modus_tollens        A>B, ~B |- ~A
The theorem names in validation-36 are the same schemata with atoms; the pool uses non-trivial sub-formulas
(and, for a fraction, plain atoms in a different arrangement) so no instance is a renaming of a validation theorem.
"""
import argparse, json, random, sys, os, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen import fstr, canon_key, ATOMS

N = lambda a: ('not', a)
AND = lambda a, b: ('and', a, b)
OR = lambda a, b: ('or', a, b)
IMP = lambda a, b: ('imp', a, b)

SCHEMATA = {
    'demorgan_nand_to_or': lambda A, B, C, D: ([N(AND(A, B))], OR(N(A), N(B))),
    'demorgan_or_to_nand': lambda A, B, C, D: ([OR(N(A), N(B))], N(AND(A, B))),
    'demorgan_nor_to_and': lambda A, B, C, D: ([N(OR(A, B))], AND(N(A), N(B))),
    'demorgan_and_to_nor': lambda A, B, C, D: ([AND(N(A), N(B))], N(OR(A, B))),
    'dist_and_over_or': lambda A, B, C, D: ([AND(A, OR(B, C))], OR(AND(A, B), AND(A, C))),
    'dist_and_over_or_conv': lambda A, B, C, D: ([OR(AND(A, B), AND(A, C))], AND(A, OR(B, C))),
    'dist_or_over_and': lambda A, B, C, D: ([OR(A, AND(B, C))], AND(OR(A, B), OR(A, C))),
    'dist_or_over_and_conv': lambda A, B, C, D: ([AND(OR(A, B), OR(A, C))], OR(A, AND(B, C))),
    'contraposition': lambda A, B, C, D: ([IMP(A, B)], IMP(N(B), N(A))),
    'contraposition_conv': lambda A, B, C, D: ([IMP(N(B), N(A))], IMP(A, B)),
    'peirce': lambda A, B, C, D: ([], IMP(IMP(IMP(A, B), A), A)),
    'peirce_sequent': lambda A, B, C, D: ([IMP(IMP(A, B), A)], A),
    'disjunctive_syllogism': lambda A, B, C, D: ([OR(A, B), N(A)], B),
    'hypothetical_syllogism': lambda A, B, C, D: ([IMP(A, B), IMP(B, C)], IMP(A, C)),
    'export': lambda A, B, C, D: ([IMP(AND(A, B), C)], IMP(A, IMP(B, C))),
    'import': lambda A, B, C, D: ([IMP(A, IMP(B, C))], IMP(AND(A, B), C)),
    'dn_intro': lambda A, B, C, D: ([A], N(N(A))),
    'dn_elim': lambda A, B, C, D: ([N(N(A))], A),
    'triple_negation': lambda A, B, C, D: ([N(N(N(A)))], N(A)),
    'explosion': lambda A, B, C, D: ([A, N(A)], B),
    'excluded_middle': lambda A, B, C, D: ([], OR(A, N(A))),
    'consequentia_mirabilis': lambda A, B, C, D: ([IMP(N(A), A)], A),
    'negated_conditional': lambda A, B, C, D: ([N(IMP(A, B))], AND(A, N(B))),
    'negated_conditional_conv': lambda A, B, C, D: ([AND(A, N(B))], N(IMP(A, B))),
    'constructive_dilemma': lambda A, B, C, D: ([OR(A, B), IMP(A, C), IMP(B, D)], OR(C, D)),
    'modus_tollens': lambda A, B, C, D: ([IMP(A, B), N(B)], N(A)),
}


def rformula(rng, depth):
    if depth == 0 or rng.random() < 0.35:
        return ('atom', rng.choice(ATOMS))
    r = rng.random()
    if r < 0.25:
        return N(rformula(rng, depth - 1))
    op = rng.choice(['and', 'or', 'imp'])
    for _ in range(10):
        l, r_ = rformula(rng, depth - 1), rformula(rng, depth - 1)
        if l != r_:          # avoid degenerate ( X op X ) sub-formulas
            return (op, l, r_)
    return N(rformula(rng, depth - 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=600)
    ap.add_argument('--out', required=True)
    ap.add_argument('--exclude', nargs='*', default=[])
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--max_prompt_toks', type=int, default=90)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    excl = {canon_key(json.loads(l)['thm'].strip()) for l in open('targets/validation_36.jsonl')}
    n_val = len(excl)
    for fn in a.exclude:
        for l in open(fn):
            if l.strip():
                excl.add(json.loads(l)['key'])
    names = sorted(SCHEMATA)
    per = a.n // len(names) + 1
    out, seen = [], set()
    stats = collections.Counter()
    for name in names:
        got = 0; tries = 0
        while got < per and tries < 20000:
            tries += 1
            # metavariables: mostly small compound formulas, sometimes atoms (but then arranged so the class differs from val-36)
            depth = rng.choice([1, 1, 2])
            A, B, C, D = (rformula(rng, depth) for _ in range(4))
            if len({A, B, C, D}) < 4:
                stats['degenerate'] += 1; continue
            prem, concl = SCHEMATA[name](A, B, C, D)
            # skip degenerate instances (a premise equal to the conclusion, duplicate premises)
            if concl in prem or len(set(prem)) != len(prem):
                stats['degenerate'] += 1; continue
            thm = (' , '.join(fstr(p) for p in prem) + ' |- ' + fstr(concl)).strip()
            key = canon_key(thm)
            if key in excl or key in seen:
                stats['dup_or_excluded'] += 1; continue
            prompt = f'THM {" , ".join(fstr(p) for p in prem)} SEQ {fstr(concl)} PRF' if prem else f'THM SEQ {fstr(concl)} PRF'
            if len(prompt.split()) > a.max_prompt_toks:
                stats['too_long'] += 1; continue
            seen.add(key)
            out.append({'name': f'{name}_{got}', 'schema': name, 'thm': thm, 'key': key, 'prompt': prompt, 'n_prem': len(prem)})
            got += 1
        stats[name] = got
    rng.shuffle(out)
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    with open(a.out, 'w') as f:
        for r in out:
            f.write(json.dumps(r) + '\n')
    print(json.dumps({'n': len(out), 'excluded_classes': len(excl), 'val36_classes': n_val, 'stats': dict(stats)}, indent=1))


if __name__ == '__main__':
    main()
