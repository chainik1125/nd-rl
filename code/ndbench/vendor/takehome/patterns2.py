#!/usr/bin/env python3
"""Run 2 pattern predicates (six new patterns, classes pre-registered in log.md), on the dependency-pruned proof like
patterns.py.  Structural: depth4, impe_chain (longest IMPE-to-IMPE citation chain; impe_chain4 = length >= 4),
nested_ore (an ORE inside an ORE branch; nested_ore3 = three deep).  Rule sequences: impi_ore (an IMPI whose box contains
an ORE), negi_ande_hyp (a NEGI whose box applies ANDE1/2 to the box's own hypothesis), ori_ore (an ORE whose disjunction
line is an ORI1/ORI2 line — never necessary: the ORI's source branch alone proves the goal).

  python patterns2.py --test              # verifier-checked unit tests
  python patterns2.py --stats FILE.jsonl  # frequency of each pattern in a pool
"""
import argparse, json, sys, os, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nd_verify import verify_text
from patterns import parse, prune_lines, prune_proof

PATTERNS2 = ('depth4', 'impe_chain4', 'nested_ore', 'nested_ore3', 'impi_ore', 'negi_ande_hyp', 'ori_ore')
CLASS = {'depth4': 'structural', 'impe_chain4': 'structural', 'nested_ore': 'structural', 'nested_ore3': 'structural',
         'impi_ore': 'rule_sequence', 'negi_ande_hyp': 'rule_sequence', 'ori_ore': 'rule_sequence'}


def box_lines(lines, s, e):
    return [ln for ln in lines if s <= ln['idx'] <= e]


def depth4(lines):
    return max((ln['depth'] for ln in lines), default=0) >= 4


def impe_chain(lines):
    """Longest chain of IMPE lines linked by citation (an IMPE citing an earlier IMPE line, as major or minor)."""
    byidx = {ln['idx']: ln for ln in lines}
    best = {}
    for ln in lines:
        if ln['rule'] != 'IMPE':
            continue
        prev = [best[r] for r in ln['refs'] if r in best]
        best[ln['idx']] = 1 + max(prev, default=0)
    return max(best.values(), default=0)


def impe_chain4(lines):
    return impe_chain(lines) >= 4


def ore_nesting(lines):
    """Maximum nesting of ORE lines: 1 = an ORE, 2 = an ORE inside a branch of another ORE, ..."""
    ores = [ln for ln in lines if ln['rule'] == 'ORE' and len(ln['refs']) == 5]
    memo = {}

    def depth_of(o):
        if o['idx'] in memo:
            return memo[o['idx']]
        _, s1, e1, s2, e2 = o['refs']
        inner = [p for p in ores if p is not o and (s1 <= p['idx'] <= e1 or s2 <= p['idx'] <= e2)]
        memo[o['idx']] = 1 + max((depth_of(p) for p in inner), default=0)
        return memo[o['idx']]
    return max((depth_of(o) for o in ores), default=0)


def nested_ore(lines):
    return ore_nesting(lines) >= 2


def nested_ore3(lines):
    return ore_nesting(lines) >= 3


def impi_ore(lines):
    for ln in lines:
        if ln['rule'] == 'IMPI' and len(ln['refs']) == 2:
            s, e = ln['refs']
            if any(x['rule'] == 'ORE' for x in box_lines(lines, s, e)):
                return True
    return False


def negi_ande_hyp(lines):
    for ln in lines:
        if ln['rule'] == 'NEGI' and len(ln['refs']) == 2:
            s, e = ln['refs']
            if any(x['rule'] in ('ANDE1', 'ANDE2') and x['refs'] and x['refs'][0] == s for x in box_lines(lines, s, e)):
                return True
    return False


def ori_ore(lines):
    byidx = {ln['idx']: ln for ln in lines}
    return any(ln['rule'] == 'ORE' and ln['refs'] and byidx.get(ln['refs'][0], {}).get('rule') in ('ORI1', 'ORI2') for ln in lines)


FUNCS = {'depth4': depth4, 'impe_chain4': impe_chain4, 'nested_ore': nested_ore, 'nested_ore3': nested_ore3, 'impi_ore': impi_ore, 'negi_ande_hyp': negi_ande_hyp, 'ori_ore': ori_ore}


def classify2(proof, pruned=True):
    lines = parse(proof)
    if lines is None:
        return None
    if pruned:
        lines = prune_lines(lines)
    out = {p: FUNCS[p](lines) for p in PATTERNS2}
    out['impe_chain'] = impe_chain(lines); out['ore_nesting'] = ore_nesting(lines); out['max_depth'] = max((ln['depth'] for ln in lines), default=0)
    return out


def test():
    P = 'THM'
    cases = [
        # depth 4 (8 lines) vs depth 3
        ('THM SEQ ( P > ( Q > ( R > ( S > S ) ) ) ) PRF',
         'N1 | P : AS ; N2 | | Q : AS ; N3 | | | R : AS ; N4 | | | | S : AS ; N5 | | | ( S > S ) : IMPI N4 N4 ; N6 | | ( R > ( S > S ) ) : IMPI N3 N5 ; N7 | ( Q > ( R > ( S > S ) ) ) : IMPI N2 N6 ; N8 ( P > ( Q > ( R > ( S > S ) ) ) ) : IMPI N1 N7 ; QED', {'depth4': True}),
        ('THM SEQ ( P > ( Q > ( R > R ) ) ) PRF',
         'N1 | P : AS ; N2 | | Q : AS ; N3 | | | R : AS ; N4 | | ( R > R ) : IMPI N3 N3 ; N5 | ( Q > ( R > R ) ) : IMPI N2 N4 ; N6 ( P > ( Q > ( R > R ) ) ) : IMPI N1 N5 ; QED', {}),
        # IMPE chain 4 (9 lines) vs chain 3 (7 lines) vs two independent IMPEs (chain 1)
        ('THM P , ( P > Q ) , ( Q > R ) , ( R > S ) , ( S > ( P & Q ) ) SEQ ( P & Q ) PRF',
         'N1 P : PR ; N2 ( P > Q ) : PR ; N3 ( Q > R ) : PR ; N4 ( R > S ) : PR ; N5 ( S > ( P & Q ) ) : PR ; N6 Q : IMPE N2 N1 ; N7 R : IMPE N3 N6 ; N8 S : IMPE N4 N7 ; N9 ( P & Q ) : IMPE N5 N8 ; QED', {'impe_chain4': True}),
        ('THM P , ( P > Q ) , ( Q > R ) , ( R > S ) SEQ S PRF',
         'N1 P : PR ; N2 ( P > Q ) : PR ; N3 ( Q > R ) : PR ; N4 ( R > S ) : PR ; N5 Q : IMPE N2 N1 ; N6 R : IMPE N3 N5 ; N7 S : IMPE N4 N6 ; QED', {}),
        ('THM P , ( P > Q ) , ( P > R ) SEQ ( Q & R ) PRF',
         'N1 P : PR ; N2 ( P > Q ) : PR ; N3 ( P > R ) : PR ; N4 Q : IMPE N2 N1 ; N5 R : IMPE N3 N1 ; N6 ( Q & R ) : ANDI N4 N5 ; QED', {}),
        # chain through the MAJOR premise: A, B, C, D, A > ( B > ( C > ( D > E ) ) ) |- E  (chain 4)
        ('THM P , Q , R , S , ( P > ( Q > ( R > ( S > ( P & S ) ) ) ) ) SEQ ( P & S ) PRF',
         'N1 P : PR ; N2 Q : PR ; N3 R : PR ; N4 S : PR ; N5 ( P > ( Q > ( R > ( S > ( P & S ) ) ) ) ) : PR ; N6 ( Q > ( R > ( S > ( P & S ) ) ) ) : IMPE N5 N1 ; N7 ( R > ( S > ( P & S ) ) ) : IMPE N6 N2 ; N8 ( S > ( P & S ) ) : IMPE N7 N3 ; N9 ( P & S ) : IMPE N8 N4 ; QED', {'impe_chain4': True}),
        # nested ORE (associativity, 12 lines): ORE inside an ORE branch
        ('THM ( ( P v Q ) v R ) SEQ ( P v ( Q v R ) ) PRF',
         'N1 ( ( P v Q ) v R ) : PR ; N2 | ( P v Q ) : AS ; N3 | | P : AS ; N4 | | ( P v ( Q v R ) ) : ORI1 N3 ; N5 | | Q : AS ; N6 | | ( Q v R ) : ORI1 N5 ; N7 | | ( P v ( Q v R ) ) : ORI2 N6 ; '
         'N8 | ( P v ( Q v R ) ) : ORE N2 N3 N4 N5 N7 ; N9 | R : AS ; N10 | ( Q v R ) : ORI2 N9 ; N11 | ( P v ( Q v R ) ) : ORI2 N10 ; N12 ( P v ( Q v R ) ) : ORE N1 N2 N8 N9 N11 ; QED',
         {'nested_ore': True}),
        # two sequential (not nested) OREs
        ('THM ( P v Q ) , ( R v S ) SEQ ( ( Q v P ) & ( S v R ) ) PRF',
         'N1 ( P v Q ) : PR ; N2 ( R v S ) : PR ; N3 | P : AS ; N4 | ( Q v P ) : ORI2 N3 ; N5 | Q : AS ; N6 | ( Q v P ) : ORI1 N5 ; N7 ( Q v P ) : ORE N1 N3 N4 N5 N6 ; '
         'N8 | R : AS ; N9 | ( S v R ) : ORI2 N8 ; N10 | S : AS ; N11 | ( S v R ) : ORI1 N10 ; N12 ( S v R ) : ORE N2 N8 N9 N10 N11 ; N13 ( ( Q v P ) & ( S v R ) ) : ANDI N7 N12 ; QED', {}),
        # IMPI whose box contains an ORE (9 lines)
        ('THM ( P v Q ) , ( P > R ) SEQ ( ( Q > R ) > R ) PRF',
         'N1 ( P v Q ) : PR ; N2 ( P > R ) : PR ; N3 | ( Q > R ) : AS ; N4 | | P : AS ; N5 | | R : IMPE N2 N4 ; N6 | | Q : AS ; N7 | | R : IMPE N3 N6 ; N8 | R : ORE N1 N4 N5 N6 N7 ; N9 ( ( Q > R ) > R ) : IMPI N3 N8 ; QED',
         {'impi_ore': True}),
        # ORE whose branches contain IMPI boxes (the other order): not impi_ore
        ('THM ( P v Q ) SEQ ( R > ( P v Q ) ) PRF',
         'N1 ( P v Q ) : PR ; N2 | R : AS ; N3 | ( P v Q ) : R N1 ; N4 ( R > ( P v Q ) ) : IMPI N2 N3 ; QED', {}),
        # NEGI whose box applies ANDE to its hypothesis (5 lines)
        ('THM ( ~ P ) SEQ ( ~ ( P & Q ) ) PRF',
         'N1 ( ~ P ) : PR ; N2 | ( P & Q ) : AS ; N3 | P : ANDE1 N2 ; N4 | F : NEGE N3 N1 ; N5 ( ~ ( P & Q ) ) : NEGI N2 N4 ; QED', {'negi_ande_hyp': True}),
        # NEGI whose box applies ANDE to a PREMISE, not the hypothesis: no
        ('THM ( P & ( ~ Q ) ) SEQ ( ~ Q ) PRF',
         'N1 ( P & ( ~ Q ) ) : PR ; N2 | Q : AS ; N3 | ( ~ Q ) : ANDE2 N1 ; N4 | F : NEGE N2 N3 ; N5 ( ~ Q ) : NEGI N2 N4 ; QED', {}),
        # IMPI (not NEGI) box applying ANDE to its hypothesis: no
        ('THM SEQ ( ( P & Q ) > P ) PRF', 'N1 | ( P & Q ) : AS ; N2 | P : ANDE1 N1 ; N3 ( ( P & Q ) > P ) : IMPI N1 N2 ; QED', {}),
        # ORI immediately consumed by ORE (decoration)
        ('THM P , ( P > Q ) SEQ Q PRF',
         'N1 P : PR ; N2 ( P > Q ) : PR ; N3 ( P v R ) : ORI1 N1 ; N4 | P : AS ; N5 | Q : IMPE N2 N4 ; N6 | R : AS ; N7 | Q : IMPE N2 N1 ; N8 Q : ORE N3 N4 N5 N6 N7 ; QED', {'ori_ore': True}),
        # nested ORE three deep (18 lines): A v B, A v C, A v D |- A v ( B & ( C & D ) )
        ('THM ( P v Q ) , ( P v R ) , ( P v S ) SEQ ( P v ( Q & ( R & S ) ) ) PRF',
         'N1 ( P v Q ) : PR ; N2 ( P v R ) : PR ; N3 ( P v S ) : PR ; N4 | P : AS ; N5 | ( P v ( Q & ( R & S ) ) ) : ORI1 N4 ; N6 | Q : AS ; '
         'N7 | | P : AS ; N8 | | ( P v ( Q & ( R & S ) ) ) : ORI1 N7 ; N9 | | R : AS ; '
         'N10 | | | P : AS ; N11 | | | ( P v ( Q & ( R & S ) ) ) : ORI1 N10 ; N12 | | | S : AS ; N13 | | | ( R & S ) : ANDI N9 N12 ; N14 | | | ( Q & ( R & S ) ) : ANDI N6 N13 ; N15 | | | ( P v ( Q & ( R & S ) ) ) : ORI2 N14 ; '
         'N16 | | ( P v ( Q & ( R & S ) ) ) : ORE N3 N10 N11 N12 N15 ; N17 | ( P v ( Q & ( R & S ) ) ) : ORE N2 N7 N8 N9 N16 ; N18 ( P v ( Q & ( R & S ) ) ) : ORE N1 N4 N5 N6 N17 ; QED',
         {'nested_ore': True, 'nested_ore3': True}),
    ]
    bad = 0
    for prompt, proof, exp in cases:
        ok, reason, nl = verify_text(prompt + ' ' + proof)
        assert ok, (reason, prompt, proof)
        pr = prune_proof(proof); assert verify_text(prompt + ' ' + pr)[0]
        got = classify2(proof); g = {p: got[p] for p in PATTERNS2}
        want = {p: exp.get(p, False) for p in PATTERNS2}
        flag = 'ok' if g == want else 'FAIL'; bad += g != want
        print(f"{flag:5s} {prompt[:58]:58s} written {nl} chain {got['impe_chain']} nest {got['ore_nesting']} depth {got['max_depth']} " + ' '.join(p for p in PATTERNS2 if g[p]))
    print('PATTERN2 TESTS', 'PASS' if not bad else f'FAIL ({bad})')
    return bad == 0


def stats(fn, limit=None):
    c = collections.Counter(); n = 0; by_len = collections.defaultdict(collections.Counter); chain = collections.Counter(); nest = collections.Counter()
    for l in open(fn):
        if not l.strip():
            continue
        r = json.loads(l); proof = r.get('proof') or r.get('gen_proof') or r['text'].split(' PRF ', 1)[1]
        cl = classify2(proof); n += 1; L = r.get('n_lines')
        by_len[L]['n'] += 1; chain[cl['impe_chain']] += 1; nest[cl['ore_nesting']] += 1
        for p in PATTERNS2:
            if cl[p]:
                c[p] += 1; by_len[L][p] += 1
        if limit and n >= limit:
            break
    print(fn, 'n', n, {p: f'{c[p]} ({c[p]/n:.5%})' for p in PATTERNS2})
    print('  impe_chain hist', dict(sorted(chain.items())), 'ore_nesting hist', dict(sorted(nest.items())))
    for L in sorted(by_len, key=lambda x: (x is None, x)):
        d = by_len[L]; print(f'  len {L}: n {d["n"]:6d}  ' + '  '.join(f'{p} {d[p]:5d}' for p in PATTERNS2))
    return c, n


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--test', action='store_true'); ap.add_argument('--stats', nargs='*'); ap.add_argument('--limit', type=int, default=None)
    a = ap.parse_args()
    if a.test:
        sys.exit(0 if test() else 1)
    for fn in a.stats or []:
        stats(fn, a.limit)
