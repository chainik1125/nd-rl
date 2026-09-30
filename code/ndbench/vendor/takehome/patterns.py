#!/usr/bin/env python3
"""Pattern predicates on dependency-pruned proofs (Phase 2 of the novelty campaign).

  P1 derived_ore : an ORE line whose disjunction line (first ref) is not a PR line.
  P2 reductio    : a NEGI line closing a box whose hypothesis is ( ~ G ), followed by a DN line that
                   cites that NEGI line and yields G (classical proof by contradiction of a non-negated goal).
  P3 depth3      : maximum box depth >= 3.
All predicates are evaluated on the DEPENDENCY-PRUNED proof (only lines the conclusion transitively cites,
box citations keeping the AS line and the box's last line; PR lines always kept so the pruned proof still
verifies against the same prompt).  prune_proof() returns the pruned proof text renumbered from N1.

  python patterns.py --test              # verifier-checked unit tests
  python patterns.py --stats FILE.jsonl  # frequency of each pattern in a pool (records with 'proof' or 'text')
"""
import argparse, json, sys, os, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nd_verify import verify_text
from nd_verify.verify import parse_proof_tokens, ParseError

PATTERNS = ('derived_ore', 'reductio', 'depth3')


def parse(proof):
    toks = proof.split()
    try:
        return parse_proof_tokens(toks)
    except ParseError:
        return None


def prune_lines(lines):
    """Keep PR lines and every line the last line transitively cites. Returns the kept lines (original idx)."""
    if not lines:
        return []
    byidx = {ln['idx']: ln for ln in lines}
    keep = set(ln['idx'] for ln in lines if ln['rule'] == 'PR')
    stack = [lines[-1]['idx']]
    while stack:
        i = stack.pop()
        if i in keep or i not in byidx:
            continue
        keep.add(i)
        stack.extend(byidx[i]['refs'])
    return [ln for ln in lines if ln['idx'] in keep]


def fstr(f):
    if f[0] == 'atom':
        return f[1]
    if f[0] == 'bot':
        return 'F'
    if f[0] == 'not':
        return f'( ~ {fstr(f[1])} )'
    op = {'and': '&', 'or': 'v', 'imp': '>'}[f[0]]
    return f'( {fstr(f[1])} {op} {fstr(f[2])} )'


def lines_to_text(lines):
    """Renumber kept lines from N1 and re-map citations."""
    m = {ln['idx']: i + 1 for i, ln in enumerate(lines)}
    toks = []
    for ln in lines:
        toks += [f"N{m[ln['idx']]}"] + ['|'] * ln['depth'] + [fstr(ln['formula']), ':', ln['rule']] + [f'N{m[r]}' for r in ln['refs']] + [';']
    toks.append('QED')
    return ' '.join(toks)


def prune_proof(proof):
    """-> pruned proof text (renumbered from N1), or None if the proof does not parse."""
    lines = parse(proof)
    if lines is None:
        return None
    return lines_to_text(prune_lines(lines))


def derived_ore(lines):
    byidx = {ln['idx']: ln for ln in lines}
    return any(ln['rule'] == 'ORE' and ln['refs'] and byidx.get(ln['refs'][0], {}).get('rule') != 'PR' for ln in lines)


def derived_ore_strict(lines):
    """Stricter P1 variant (reported alongside): the disjunction line is obtained by a RULE (not PR, not AS) and its two
    disjuncts differ (excludes the degenerate ( X v X ) -> X extraction)."""
    byidx = {ln['idx']: ln for ln in lines}
    for ln in lines:
        if ln['rule'] != 'ORE' or not ln['refs']:
            continue
        d = byidx.get(ln['refs'][0])
        if d and d['rule'] not in ('PR', 'AS') and d['formula'][0] == 'or' and d['formula'][1] != d['formula'][2]:
            return True
    return False


def ore_shape(lines):
    """For every ORE line: (rule of the disjunction line, disjuncts equal?, both boxes one-line?)."""
    byidx = {ln['idx']: ln for ln in lines}
    out = []
    for ln in lines:
        if ln['rule'] == 'ORE' and len(ln['refs']) == 5:
            d = byidx.get(ln['refs'][0]); j, s1, e1, s2, e2 = ln['refs']
            out.append((d['rule'] if d else '?', bool(d and d['formula'][0] == 'or' and d['formula'][1] == d['formula'][2]), s1 == e1 and s2 == e2))
    return out


def reductio(lines):
    byidx = {ln['idx']: ln for ln in lines}
    for ln in lines:
        if ln['rule'] != 'DN' or not ln['refs']:
            continue
        neg = byidx.get(ln['refs'][0])
        if not neg or neg['rule'] != 'NEGI' or len(neg['refs']) != 2:
            continue
        hyp = byidx.get(neg['refs'][0])
        if hyp and hyp['rule'] == 'AS' and hyp['formula'] == ('not', ln['formula']):
            return True
    return False


def derived_dn(lines):
    """Loose reductio predicate (follow-up block B): a DN line whose cited line is neither a premise (PR) nor a
    hypothesis (AS), i.e. the double negation it eliminates was DERIVED by a rule."""
    byidx = {ln['idx']: ln for ln in lines}
    return any(ln['rule'] == 'DN' and ln['refs'] and byidx.get(ln['refs'][0], {}).get('rule') not in ('PR', 'AS', None) for ln in lines)


def depth3(lines):
    return max((ln['depth'] for ln in lines), default=0) >= 3


def classify(proof, pruned=True):
    """-> dict pattern -> bool (on the dependency-pruned proof by default), plus 'pruned_len'; None if unparsable."""
    lines = parse(proof)
    if lines is None:
        return None
    if pruned:
        lines = prune_lines(lines)
    return {'derived_ore': derived_ore(lines), 'reductio': reductio(lines), 'depth3': depth3(lines), 'pruned_len': len(lines),
            'derived_ore_strict': derived_ore_strict(lines), 'ore_shapes': ore_shape(lines), 'derived_dn': derived_dn(lines)}


def test():
    # (prompt, proof, expected {pattern: bool} on the pruned proof)
    cases = [
        # plain modus ponens: nothing
        ('THM ( P > Q ) , P SEQ Q PRF', 'N1 ( P > Q ) : PR ; N2 P : PR ; N3 Q : IMPE N1 N2 ; QED', {}),
        # ORE over a premise disjunction: NOT derived
        ('THM ( P v Q ) , ( ~ P ) SEQ Q PRF',
         'N1 ( P v Q ) : PR ; N2 ( ~ P ) : PR ; N3 | P : AS ; N4 | F : NEGE N3 N2 ; N5 | Q : BOTE N4 ; N6 | Q : AS ; N7 Q : ORE N1 N3 N5 N6 N6 ; QED', {}),
        # ORE over a disjunction obtained by ANDE: derived
        ('THM ( ( P v Q ) & R ) SEQ ( Q v P ) PRF',
         'N1 ( ( P v Q ) & R ) : PR ; N2 ( P v Q ) : ANDE1 N1 ; N3 | P : AS ; N4 | ( Q v P ) : ORI2 N3 ; N5 | Q : AS ; N6 | ( Q v P ) : ORI1 N5 ; N7 ( Q v P ) : ORE N2 N3 N4 N5 N6 ; QED',
         {'derived_ore': True}),
        # ORE over a derived disjunction that is NOT on the dependency path (padding): not counted after pruning
        ('THM ( P v Q ) , ( ( P v Q ) > ( R v S ) ) , ( ~ P ) SEQ Q PRF',
         'N1 ( P v Q ) : PR ; N2 ( ( P v Q ) > ( R v S ) ) : PR ; N3 ( ~ P ) : PR ; N4 ( R v S ) : IMPE N2 N1 ; '
         'N5 | R : AS ; N6 | ( R v S ) : ORI1 N5 ; N7 | S : AS ; N8 | ( R v S ) : ORI2 N7 ; N9 ( R v S ) : ORE N4 N5 N6 N7 N8 ; '
         'N10 | P : AS ; N11 | F : NEGE N10 N3 ; N12 | Q : BOTE N11 ; N13 | Q : AS ; N14 Q : ORE N1 N10 N12 N13 N13 ; QED', {}),
        # classical reductio of a non-negated goal: reductio
        ('THM ( ~ ( ~ P ) ) SEQ P PRF',
         'N1 ( ~ ( ~ P ) ) : PR ; N2 | ( ~ P ) : AS ; N3 | F : NEGE N2 N1 ; N4 ( ~ ( ~ P ) ) : NEGI N2 N3 ; N5 P : DN N4 ; QED',
         {'reductio': True}),
        # NEGI of a positive hypothesis (modus tollens): not reductio
        ('THM ( P > Q ) , ( ~ Q ) SEQ ( ~ P ) PRF',
         'N1 ( P > Q ) : PR ; N2 ( ~ Q ) : PR ; N3 | P : AS ; N4 | Q : IMPE N1 N3 ; N5 | F : NEGE N4 N2 ; N6 ( ~ P ) : NEGI N3 N5 ; QED', {}),
        # DN of a premise (no NEGI box): not reductio
        ('THM ( ~ ( ~ P ) ) SEQ P PRF', 'N1 ( ~ ( ~ P ) ) : PR ; N2 P : DN N1 ; QED', {}),
        # NEGI of ( ~ G ) without the DN: not reductio (the goal is ( ~ ( ~ G ) ))
        ('THM P SEQ ( ~ ( ~ P ) ) PRF', 'N1 P : PR ; N2 | ( ~ P ) : AS ; N3 | F : NEGE N1 N2 ; N4 ( ~ ( ~ P ) ) : NEGI N2 N3 ; QED', {}),
        # depth 3: nested IMPI boxes
        ('THM SEQ ( P > ( Q > ( R > R ) ) ) PRF',
         'N1 | P : AS ; N2 | | Q : AS ; N3 | | | R : AS ; N4 | | ( R > R ) : IMPI N3 N3 ; N5 | ( Q > ( R > R ) ) : IMPI N2 N4 ; N6 ( P > ( Q > ( R > R ) ) ) : IMPI N1 N5 ; QED',
         {'depth3': True}),
        # depth 2 only
        ('THM SEQ ( P > ( Q > P ) ) PRF',
         'N1 | P : AS ; N2 | | Q : AS ; N3 | | P : R N1 ; N4 | ( Q > P ) : IMPI N2 N3 ; N5 ( P > ( Q > P ) ) : IMPI N1 N4 ; QED', {}),
        # depth-3 box that is padding (not cited): not counted after pruning
        ('THM SEQ ( P > ( Q > P ) ) PRF',
         'N1 | P : AS ; N2 | | Q : AS ; N3 | | | R : AS ; N4 | | ( R > R ) : IMPI N3 N3 ; N5 | | P : R N1 ; N6 | ( Q > P ) : IMPI N2 N5 ; N7 ( P > ( Q > P ) ) : IMPI N1 N6 ; QED', {}),
        # reductio inside a box, plus derived ORE: both
        ('THM ( ( P v Q ) & ( ~ ( ~ R ) ) ) SEQ ( R v S ) PRF',
         'N1 ( ( P v Q ) & ( ~ ( ~ R ) ) ) : PR ; N2 ( P v Q ) : ANDE1 N1 ; N3 ( ~ ( ~ R ) ) : ANDE2 N1 ; '
         'N4 | P : AS ; N5 | | ( ~ R ) : AS ; N6 | | F : NEGE N5 N3 ; N7 | ( ~ ( ~ R ) ) : NEGI N5 N6 ; N8 | R : DN N7 ; N9 | ( R v S ) : ORI1 N8 ; '
         'N10 | Q : AS ; N11 | R : DN N3 ; N12 | ( R v S ) : ORI1 N11 ; N13 ( R v S ) : ORE N2 N4 N9 N10 N12 ; QED',
         {'derived_ore': True, 'reductio': True}),
    ]
    # loose reductio (derived_dn) expectations, keyed by case index: DN citing a NEGI line -> True; DN of a PR line -> False
    loose = {4: True, 6: False, 11: True}
    bad = 0
    for ci, (prompt, proof, exp) in enumerate(cases):
        if ci in loose:
            got_l = classify(proof)['derived_dn']
            print(('ok   ' if got_l == loose[ci] else 'FAIL ') + f'derived_dn case {ci}: {got_l}')
            bad += got_l != loose[ci]
        ok, reason, nl = verify_text(prompt + ' ' + proof)
        assert ok, (reason, prompt, proof)
        pr = prune_proof(proof)
        ok2, reason2, nl2 = verify_text(prompt + ' ' + pr)
        assert ok2, ('pruned proof invalid', reason2, pr)
        got = classify(proof)
        want = {p: exp.get(p, False) for p in PATTERNS}
        g = {p: got[p] for p in PATTERNS}
        flag = 'ok' if g == want else 'FAIL'
        bad += g != want
        print(f'{flag:5s} {prompt[:60]:60s} written {nl} pruned {nl2} got {g}')
    print('PATTERN TESTS', 'PASS' if not bad else f'FAIL ({bad})')
    return bad == 0


def stats(fn, limit=None):
    c = collections.Counter(); n = 0
    by_len = collections.defaultdict(collections.Counter)
    for l in open(fn):
        if not l.strip():
            continue
        r = json.loads(l)
        proof = r.get('proof') or r.get('gen_proof') or r['text'].split(' PRF ', 1)[1]
        cl = classify(proof)
        n += 1
        L = r.get('n_lines')
        for p in PATTERNS:
            if cl[p]:
                c[p] += 1; by_len[L][p] += 1
        by_len[L]['n'] += 1
        if limit and n >= limit:
            break
    print(fn, 'n', n, {p: f'{c[p]} ({c[p]/n:.4%})' for p in PATTERNS})
    for L in sorted(by_len, key=lambda x: (x is None, x)):
        d = by_len[L]
        print(f'  len {L}: n {d["n"]:6d}  ' + '  '.join(f'{p} {d[p]:5d} ({d[p]/d["n"]:.3%})' for p in PATTERNS))
    return c, n


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--test', action='store_true')
    ap.add_argument('--stats', nargs='*')
    ap.add_argument('--limit', type=int, default=None)
    a = ap.parse_args()
    if a.test:
        sys.exit(0 if test() else 1)
    for fn in a.stats or []:
        stats(fn, a.limit)
