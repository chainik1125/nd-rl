"""Curated textbook sequents (Halbach-style classics) as held-out eval targets.

Defines ~30 classic propositional sequents, validates each with a brute-force
truth-table check (classical semantics, 'bot' always false), tags each with
classical_only (not intuitionistically valid; best-judgment metadata), and
writes data/targets_textbook.jsonl:

  {"thm": theorem_id, "prompt": "THM ... SEQ ... PRF", "name": ..., "classical_only": ...}

Usage:  python code/exp/make_textbook_targets.py
"""
import itertools
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nd.core import (atom, f_not, f_and, f_or, f_imp, BOT, formula_str,
                     canon_premises, prompt_tokens, theorem_id)
from nd.tokenizer import STOI

P, Q, R, S = atom('P'), atom('Q'), atom('R'), atom('S')

# ---------------------------------------------------------------------------
# Sequents: (name, [premises], conclusion, classical_only)
# classical_only = True iff NOT intuitionistically valid (metadata only).
# ---------------------------------------------------------------------------

SEQUENTS = [
    # core inference patterns
    ('modus_ponens',          [P, f_imp(P, Q)],                       Q,                          False),
    ('modus_tollens',         [f_imp(P, Q), f_not(Q)],                f_not(P),                   False),
    ('disjunctive_syllogism', [f_or(P, Q), f_not(P)],                 Q,                          False),
    ('hypothetical_syllogism', [f_imp(P, Q), f_imp(Q, R)],            f_imp(P, R),                False),
    ('constructive_dilemma',  [f_or(P, Q), f_imp(P, R), f_imp(Q, S)], f_or(R, S),                 False),
    ('simple_dilemma',        [f_imp(P, Q), f_imp(f_not(P), Q)],      Q,                          True),
    ('resolution',            [f_or(P, Q), f_or(f_not(P), R)],        f_or(Q, R),                 False),

    # De Morgan, all four directions
    ('demorgan_nand_to_or',   [f_not(f_and(P, Q))],                   f_or(f_not(P), f_not(Q)),   True),
    ('demorgan_or_to_nand',   [f_or(f_not(P), f_not(Q))],             f_not(f_and(P, Q)),         False),
    ('demorgan_nor_to_and',   [f_not(f_or(P, Q))],                    f_and(f_not(P), f_not(Q)),  False),
    ('demorgan_and_to_nor',   [f_and(f_not(P), f_not(Q))],            f_not(f_or(P, Q)),          False),

    # contraposition both directions
    ('contraposition',        [f_imp(P, Q)],                          f_imp(f_not(Q), f_not(P)),  False),
    ('contraposition_conv',   [f_imp(f_not(Q), f_not(P))],            f_imp(P, Q),                True),

    # double negation both directions
    ('dn_intro',              [P],                                    f_not(f_not(P)),            False),
    ('dn_elim',               [f_not(f_not(P))],                      P,                          True),
    ('triple_negation',       [f_not(f_not(f_not(P)))],               f_not(P),                   False),

    # classical theorems (empty premises)
    ('excluded_middle',       [],                                     f_or(P, f_not(P)),          True),
    ('peirce',                [],                                     f_imp(f_imp(f_imp(P, Q), P), P), True),
    ('conditional_ordering',  [],                                     f_or(f_imp(P, Q), f_imp(Q, P)), True),
    ('identity',              [],                                     f_imp(P, P),                False),

    # Peirce as a sequent; consequentia mirabilis
    ('peirce_sequent',        [f_imp(f_imp(P, Q), P)],                P,                          True),
    ('consequentia_mirabilis', [f_imp(f_not(P), P)],                  P,                          True),

    # import / export
    ('import',                [f_imp(P, f_imp(Q, R))],                f_imp(f_and(P, Q), R),      False),
    ('export',                [f_imp(f_and(P, Q), R)],                f_imp(P, f_imp(Q, R)),      False),

    # distribution laws
    ('dist_and_over_or',      [f_and(P, f_or(Q, R))],                 f_or(f_and(P, Q), f_and(P, R)), False),
    ('dist_and_over_or_conv', [f_or(f_and(P, Q), f_and(P, R))],       f_and(P, f_or(Q, R)),       False),
    ('dist_or_over_and',      [f_or(P, f_and(Q, R))],                 f_and(f_or(P, Q), f_or(P, R)), False),
    ('dist_or_over_and_conv', [f_and(f_or(P, Q), f_or(P, R))],        f_or(P, f_and(Q, R)),       False),

    # paradoxes of material implication & explosion
    ('explosion',             [f_and(P, f_not(P))],                   Q,                          False),
    ('negative_paradox',      [f_not(P)],                             f_imp(P, Q),                False),
    ('positive_paradox',      [Q],                                    f_imp(P, Q),                False),

    # negated conditional both directions
    ('negated_conditional',   [f_not(f_imp(P, Q))],                   f_and(P, f_not(Q)),         True),
    ('negated_conditional_conv', [f_and(P, f_not(Q))],                f_not(f_imp(P, Q)),         False),

    # a few more classics
    ('reductio_pair',         [f_imp(P, Q), f_imp(P, f_not(Q))],      f_not(P),                   False),
    ('absorption',            [f_imp(P, Q)],                          f_imp(P, f_and(P, Q)),      False),
    ('frege_distribution',    [f_imp(P, f_imp(Q, R))],                f_imp(f_imp(P, Q), f_imp(P, R)), False),
]


# ---------------------------------------------------------------------------
# Truth-table validation (classical semantics; 'bot' is always false)
# ---------------------------------------------------------------------------

def eval_formula(f, val):
    t = f[0]
    if t == 'atom':
        return val[f[1]]
    if t == 'bot':
        return False
    if t == 'not':
        return not eval_formula(f[1], val)
    if t == 'and':
        return eval_formula(f[1], val) and eval_formula(f[2], val)
    if t == 'or':
        return eval_formula(f[1], val) or eval_formula(f[2], val)
    if t == 'imp':
        return (not eval_formula(f[1], val)) or eval_formula(f[2], val)
    raise ValueError(f)


def atoms_of(f):
    if f[0] == 'atom':
        return {f[1]}
    return set().union(*(atoms_of(sub) for sub in f[1:])) if len(f) > 1 else set()


def is_valid(premises, conclusion):
    names = sorted(set().union(atoms_of(conclusion), *(atoms_of(p) for p in premises)))
    for bits in itertools.product([False, True], repeat=len(names)):
        val = dict(zip(names, bits))
        if all(eval_formula(p, val) for p in premises) and not eval_formula(conclusion, val):
            return False
    return True


def main():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    out_path = os.path.join(root, 'data', 'targets_textbook.jsonl')

    records = []
    seen = set()
    failures = []
    for name, raw_prems, concl, classical_only in SEQUENTS:
        if not is_valid(raw_prems, concl):
            failures.append(name)
            continue
        prems = canon_premises(raw_prems)
        thm = theorem_id(prems, concl)
        if thm in seen:
            raise SystemExit(f"duplicate sequent: {name} ({thm})")
        seen.add(thm)
        prompt = ' '.join(prompt_tokens(prems, concl))
        for tok in prompt.split():
            if tok not in STOI:
                raise SystemExit(f"token {tok!r} not in vocab ({name})")
        records.append({'thm': thm, 'prompt': prompt, 'name': name,
                        'classical_only': classical_only})

    if failures:
        raise SystemExit(f"INVALID sequents (truth-table check failed): {failures}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as fh:
        for rec in records:
            fh.write(json.dumps(rec) + '\n')

    # human-readable table
    w = max(len(r['name']) for r in records)
    print(f"{'#':>2}  {'name':<{w}}  {'cls':<3}  sequent")
    print('-' * (w + 60))
    for i, rec in enumerate(records, 1):
        flag = 'C' if rec['classical_only'] else '.'
        print(f"{i:>2}  {rec['name']:<{w}}  {flag:<3}  {rec['thm']}")
    n_cls = sum(r['classical_only'] for r in records)
    print('-' * (w + 60))
    print(f"{len(records)} sequents ({n_cls} classical-only), all truth-table valid")
    print(f"wrote {out_path}")


if __name__ == '__main__':
    main()
