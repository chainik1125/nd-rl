#!/usr/bin/env python3
"""Phase 3 precursor set: generator-made, verified, <= 6-line proofs whose THEOREM matches a short sub-step of a
classical textbook schema (the "precursor" idea). Nothing here writes proofs: the proofs come from the random
generator's pool (`make_coverage_sets.py gen` output / `data/raw_cap6.jsonl`); this file only recognises which of
those proofs prove a sub-step sequent (unification of the theorem against templates with metavariables A, B, C, D).

  python precursors.py --pool data/p2/pool_cap6.jsonl --out data/p3/precursors.jsonl --exclude data/p3/textbook_targets.jsonl
  python precursors.py --test

Templates are written as (premises, conclusion) over metavariables; premises may appear in any order in the theorem.
A theorem matches if some template unifies with it (metavariables bind to arbitrary formulas, consistently).
"""
import argparse, json, sys, os, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nd_verify import verify_text
from nd_verify.verify import parse_formula
from gen import canon_key

A, B, C, D = ('var', 'A'), ('var', 'B'), ('var', 'C'), ('var', 'D')
BOT = ('bot',)
N = lambda a: ('not', a)
AND = lambda a, b: ('and', a, b)
OR = lambda a, b: ('or', a, b)
IMP = lambda a, b: ('imp', a, b)

# name -> list of (premises, conclusion). Each is a <= 6-line sub-step of the named schema.
TEMPLATES = {
    'demorgan_nand_to_or': [([N(AND(A, B)), A], N(B)), ([N(AND(A, B)), B], N(A)), ([N(A)], OR(N(A), N(B))), ([N(B)], OR(N(A), N(B))),
                            ([N(OR(N(A), N(B))), N(A)], BOT), ([N(OR(N(A), N(B)))], N(N(A))), ([N(OR(N(A), N(B)))], N(N(B)))],
    'demorgan_or_to_nand': [([N(A), AND(A, B)], BOT), ([N(B), AND(A, B)], BOT), ([N(A)], N(AND(A, B))), ([N(B)], N(AND(A, B)))],
    'demorgan_nor_to_and': [([N(OR(A, B))], N(A)), ([N(OR(A, B))], N(B)), ([N(A), N(B)], AND(N(A), N(B)))],
    'demorgan_and_to_nor': [([AND(N(A), N(B)), A], BOT), ([AND(N(A), N(B)), B], BOT), ([N(A), N(B), A], BOT)],
    'dist_and_over_or': [([A, B], OR(AND(A, B), AND(A, C))), ([A, C], OR(AND(A, B), AND(A, C))), ([AND(A, OR(B, C))], OR(B, C))],
    'dist_and_over_or_conv': [([AND(A, B)], AND(A, OR(B, C))), ([AND(A, C)], AND(A, OR(B, C))), ([AND(A, B)], OR(B, C))],
    'dist_or_over_and': [([A], AND(OR(A, B), OR(A, C))), ([AND(B, C)], AND(OR(A, B), OR(A, C))), ([AND(B, C)], OR(A, B))],
    'dist_or_over_and_conv': [([A], OR(A, AND(B, C))), ([B, C], OR(A, AND(B, C))), ([AND(OR(A, B), OR(A, C))], OR(A, B))],
    'contraposition': [([IMP(A, B), N(B)], N(A)), ([IMP(A, B), A], B), ([IMP(A, B), N(B), A], BOT)],
    'contraposition_conv': [([IMP(N(B), N(A)), A], N(N(B))), ([IMP(N(B), N(A)), A, N(B)], BOT), ([N(N(B))], B)],
    'peirce': [([IMP(IMP(A, B), A), N(A)], IMP(A, B)), ([N(A)], IMP(A, B)), ([IMP(IMP(A, B), A), IMP(A, B)], A), ([IMP(IMP(A, B), A), N(A)], BOT)],
    'disjunctive_syllogism': [([A, N(A)], B), ([N(A), A], BOT)],
    'hypothetical_syllogism': [([IMP(A, B), IMP(B, C), A], C), ([IMP(A, B), A], B)],
    'export': [([IMP(AND(A, B), C), A, B], C), ([A, B], AND(A, B)), ([IMP(AND(A, B), C), A], IMP(B, C))],
    'import': [([IMP(A, IMP(B, C)), AND(A, B)], C), ([IMP(A, IMP(B, C)), A, B], C), ([IMP(A, IMP(B, C)), A], IMP(B, C))],
    'dn_forms': [([A, N(A)], BOT), ([A], N(N(A))), ([N(N(A))], A), ([N(N(N(A)))], N(A))],
    'explosion': [([A, N(A)], B), ([N(A)], IMP(A, B))],
    'excluded_middle': [([N(OR(A, N(A))), A], BOT), ([N(OR(A, N(A)))], N(A)), ([N(OR(A, N(A))), N(A)], BOT), ([N(OR(A, N(A)))], N(N(A)))],
    'consequentia_mirabilis': [([IMP(N(A), A), N(A)], BOT), ([IMP(N(A), A)], N(N(A))), ([IMP(N(A), A), N(A)], A)],
    'negated_conditional': [([B], IMP(A, B)), ([N(IMP(A, B)), IMP(A, B)], BOT), ([N(IMP(A, B)), B], BOT), ([N(A)], IMP(A, B)), ([N(IMP(A, B)), N(A)], BOT)],
    'negated_conditional_conv': [([A, N(B), IMP(A, B)], BOT), ([AND(A, N(B)), IMP(A, B)], BOT), ([A, N(B)], N(IMP(A, B)))],
    'constructive_dilemma': [([IMP(A, C), A], OR(C, D)), ([IMP(B, D), B], OR(C, D))],
    'modus_tollens': [([IMP(A, B), A, N(B)], BOT), ([IMP(A, B), N(B)], N(A))],
}


def unify(pat, f, env):
    if pat[0] == 'var':
        if pat[1] in env:
            return env[pat[1]] == f
        env[pat[1]] = f
        return True
    if pat[0] != f[0] or len(pat) != len(f):
        return False
    if pat[0] in ('atom', 'bot'):
        return pat == f
    return all(unify(p, x, env) for p, x in zip(pat[1:], f[1:]))


def match_template(prem_pat, concl_pat, prem, concl):
    """premises may be in any order; every template premise must match a distinct theorem premise and vice versa."""
    if len(prem_pat) != len(prem):
        return None
    import itertools
    for perm in itertools.permutations(range(len(prem))):
        env = {}
        if not unify(concl_pat, concl, env):
            continue
        if all(unify(prem_pat[i], prem[perm[i]], env) for i in range(len(prem))):
            return env
    return None


def parse_thm(prompt):
    toks = prompt.split()
    i = 1
    prem = []
    if toks[i] != 'SEQ':
        while True:
            f, i = parse_formula(toks, i)
            prem.append(f)
            if toks[i] == ',':
                i += 1; continue
            break
    concl, i = parse_formula(toks, i + 1)
    return prem, concl


def classify_thm(prompt):
    """-> list of (schema, template index, env) that the theorem matches."""
    prem, concl = parse_thm(prompt)
    out = []
    for name, temps in TEMPLATES.items():
        for ti, (pp, cp) in enumerate(temps):
            env = match_template(pp, cp, prem, concl)
            if env is not None:
                out.append((name, ti, env))
    return out


def test():
    cases = [('THM ( P > Q ) , ( ~ Q ) SEQ ( ~ P ) PRF', True), ('THM P , ( ~ P ) SEQ Q PRF', True), ('THM ( ~ ( ( P & R ) & Q ) ) , ( P & R ) SEQ ( ~ Q ) PRF', True),
             ('THM ( P > Q ) , P SEQ Q PRF', True), ('THM P , ( P > Q ) SEQ Q PRF', True), ('THM ( P & Q ) SEQ ( Q & P ) PRF', False),
             ('THM ( ~ ( P v ( ~ P ) ) ) SEQ ( ~ P ) PRF', True), ('THM ( ~ ( P v ( ~ Q ) ) ) SEQ ( ~ P ) PRF', True), ('THM SEQ ( P > P ) PRF', False),
             ('THM ( ( P > Q ) > P ) , ( ~ P ) SEQ ( P > Q ) PRF', True), ('THM ( P v Q ) , ( ~ P ) SEQ Q PRF', False)]
    bad = 0
    for prompt, exp in cases:
        m = classify_thm(prompt)
        ok = bool(m) == exp
        bad += not ok
        print('ok  ' if ok else 'FAIL', exp, [(a, b) for a, b, _ in m][:3], prompt)
    print('PRECURSOR TESTS', 'PASS' if not bad else 'FAIL')
    return bad == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--test', action='store_true')
    ap.add_argument('--pool'); ap.add_argument('--out'); ap.add_argument('--exclude', nargs='*', default=[])
    ap.add_argument('--max_per_template', type=int, default=400)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    if a.test:
        sys.exit(0 if test() else 1)
    import random
    rng = random.Random(a.seed)
    excl = {canon_key(json.loads(l)['thm'].strip()) for l in open('targets/validation_36.jsonl')}
    for fn in a.exclude:
        for l in open(fn):
            if l.strip():
                excl.add(json.loads(l)['key'])
    per = collections.defaultdict(list)
    n = 0
    for l in open(a.pool):
        r = json.loads(l); n += 1
        if r['n_lines'] > 6 or r['key'] in excl:
            continue
        m = classify_thm(r['prompt'])
        if not m:
            continue
        ok, reason, nl = verify_text(r['prompt'] + ' ' + r['proof'])
        assert ok and nl == r['n_lines'] <= 6, (reason, r)
        r['precursor_of'] = sorted({s for s, _, _ in m})
        r['templates'] = [f'{s}:{ti}' for s, ti, _ in m]
        per[m[0][0] + ':' + str(m[0][1])].append(r)
    out = []
    for k, rs in per.items():
        rng.shuffle(rs)
        out += rs[:a.max_per_template]
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    with open(a.out, 'w') as f:
        for i, r in enumerate(out):
            f.write(json.dumps({'name': f'precursor_{i}', 'thm': r['thm'], 'key': r['key'], 'prompt': r['prompt'], 'proof': r['proof'],
                                'text': r['prompt'] + ' ' + r['proof'], 'n_lines': r['n_lines'], 'precursor_of': r['precursor_of'], 'templates': r['templates']}) + '\n')
    summ = {k: len(v) for k, v in sorted(per.items())}
    print(json.dumps({'pool': n, 'matched_total': sum(summ.values()), 'written': len(out), 'per_template_available': summ,
                      'by_schema_written': dict(collections.Counter(s for r in out for s in r['precursor_of'])),
                      'by_len_written': dict(sorted(collections.Counter(r['n_lines'] for r in out).items()))}, indent=1))


if __name__ == '__main__':
    main()
