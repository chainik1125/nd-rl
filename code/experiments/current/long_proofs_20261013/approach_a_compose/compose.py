"""Approach A: compose long ND theorems out of individually hard pieces chained by a cut.

A piece takes an input formula X (the previous piece's output; None for the first piece), introduces
fresh atoms and 1-3 premises, and derives an output formula. Pieces are chained: piece k+1's main
premise is guarded by piece k's output (`( X > Y )`), so every premise of every piece is needed to
reach the final conclusion, and atoms are fresh per piece so pieces cannot shortcut each other.

    python compose.py --n 200 --pieces 3 4 --seed 0 --out out/raw.jsonl
"""
import argparse
import json
import random
import sys

LETTERS = ['P', 'Q', 'R', 'S', 'T', 'U', 'W', 'X', 'Y', 'Z']
BOT = ('bot',)


def A(a):
    return ('atom', a)


def N(f):
    return ('not', f)


def I(a, b):
    return ('imp', a, b)


def O(a, b):
    return ('or', a, b)


def C(a, b):
    return ('and', a, b)


def fstr(f):
    if f[0] == 'atom':
        return f[1]
    if f[0] == 'bot':
        return 'F'
    if f[0] == 'not':
        return f'( ~ {fstr(f[1])} )'
    return f'( {fstr(f[1])} ' + {'and': '&', 'or': 'v', 'imp': '>'}[f[0]] + f' {fstr(f[2])} )'


# ---------------------------------------------------------------------------------------------
# Pieces. Each returns (premises, body, out_label). body = list of (depth, formula, rule, refs) where refs are
# labels: 'in' (the input line), 'p<i>' (own premise i), 'b<i>' (own body line i). Depth is relative to 0.
# `main` is how the piece's main premise is obtained: if X is None it is a plain premise; else the premise is
# ( X > main ) and body line 0 derives `main` by IMPE (the cut).
# ---------------------------------------------------------------------------------------------

def _guard(X, main, prems, body):
    """Prefix: premise 0 is `main` (X None) or ( X > main ) followed by an IMPE line. Returns ref label of main."""
    if X is None:
        prems.insert(0, main)
        return 'p0'
    prems.insert(0, I(X, main))
    body.append((0, main, 'IMPE', ['p0', 'in']))
    return 'b0'


def piece_merge(X, a, b, c):
    """( A v B ) , ( A > C ) , ( B > C ) |- C   (5 body lines, +1 with a guard)."""
    a, b, c = A(a), A(b), A(c)
    prems, body = [I(a, c), I(b, c)], []
    m = _guard(X, O(a, b), prems, body)
    k = len(body)
    body += [(1, a, 'AS', []), (1, c, 'IMPE', ['p1', f'b{k}']),
             (1, b, 'AS', []), (1, c, 'IMPE', ['p2', f'b{k+2}']),
             (0, c, 'ORE', [m, f'b{k}', f'b{k+1}', f'b{k+2}', f'b{k+3}'])]
    return prems, body, f'b{k+4}'


def piece_split(X, a, b, c, d):
    """( A v B ) , ( A > C ) , ( B > D ) |- ( C v D )   (7 body lines)."""
    a, b, c, d = A(a), A(b), A(c), A(d)
    prems, body = [I(a, c), I(b, d)], []
    m = _guard(X, O(a, b), prems, body)
    k = len(body)
    body += [(1, a, 'AS', []), (1, c, 'IMPE', ['p1', f'b{k}']), (1, O(c, d), 'ORI1', [f'b{k+1}']),
             (1, b, 'AS', []), (1, d, 'IMPE', ['p2', f'b{k+3}']), (1, O(c, d), 'ORI2', [f'b{k+4}']),
             (0, O(c, d), 'ORE', [m, f'b{k}', f'b{k+2}', f'b{k+3}', f'b{k+5}'])]
    return prems, body, f'b{k+6}'


def piece_mt(X, a, b):
    """( ~ B ) , ( A > B ) |- ( ~ A )   (4 body lines)."""
    a, b = A(a), A(b)
    prems, body = [I(a, b)], []
    m = _guard(X, N(b), prems, body)
    k = len(body)
    body += [(1, a, 'AS', []), (1, b, 'IMPE', ['p1', f'b{k}']), (1, BOT, 'NEGE', [f'b{k+1}', m]),
             (0, N(a), 'NEGI', [f'b{k}', f'b{k+2}'])]
    return prems, body, f'b{k+3}'


def piece_ds(X, a, b):
    """( ~ A ) , ( A v B ) |- B   (5 body lines)."""
    a, b = A(a), A(b)
    prems, body = [O(a, b)], []
    m = _guard(X, N(a), prems, body)
    k = len(body)
    body += [(1, a, 'AS', []), (1, BOT, 'NEGE', [f'b{k}', m]), (1, b, 'BOTE', [f'b{k+1}']),
             (1, b, 'AS', []),
             (0, b, 'ORE', ['p1', f'b{k}', f'b{k+2}', f'b{k+3}', f'b{k+3}'])]
    return prems, body, f'b{k+4}'


def piece_contra(X, a, b):
    """( A > B ) |- ( ( ~ B ) > ( ~ A ) )   (6 body lines)."""
    a, b = A(a), A(b)
    prems, body = [], []
    m = _guard(X, I(a, b), prems, body)
    k = len(body)
    body += [(1, N(b), 'AS', []), (2, a, 'AS', []), (2, b, 'IMPE', [m, f'b{k+1}']),
             (2, BOT, 'NEGE', [f'b{k+2}', f'b{k}']), (1, N(a), 'NEGI', [f'b{k+1}', f'b{k+3}']),
             (0, I(N(b), N(a)), 'IMPI', [f'b{k}', f'b{k+4}'])]
    return prems, body, f'b{k+5}'


def piece_peirce(X, a, b):
    """( ( A > B ) > A ) |- A   (9 body lines, classical)."""
    a, b = A(a), A(b)
    prems, body = [], []
    m = _guard(X, I(I(a, b), a), prems, body)
    k = len(body)
    body += [(1, N(a), 'AS', []), (2, a, 'AS', []), (2, BOT, 'NEGE', [f'b{k+1}', f'b{k}']),
             (2, b, 'BOTE', [f'b{k+2}']), (1, I(a, b), 'IMPI', [f'b{k+1}', f'b{k+3}']),
             (1, a, 'IMPE', [m, f'b{k+4}']), (1, BOT, 'NEGE', [f'b{k+5}', f'b{k}']),
             (0, N(N(a)), 'NEGI', [f'b{k}', f'b{k+6}']), (0, a, 'DN', [f'b{k+7}'])]
    return prems, body, f'b{k+8}'


def piece_dist(X, a, b, c):
    """( A & ( B v C ) ) |- ( ( A & B ) v ( A & C ) )   (9 body lines)."""
    a, b, c = A(a), A(b), A(c)
    prems, body = [], []
    m = _guard(X, C(a, O(b, c)), prems, body)
    k = len(body)
    out = O(C(a, b), C(a, c))
    body += [(0, a, 'ANDE1', [m]), (0, O(b, c), 'ANDE2', [m]),
             (1, b, 'AS', []), (1, C(a, b), 'ANDI', [f'b{k}', f'b{k+2}']), (1, out, 'ORI1', [f'b{k+3}']),
             (1, c, 'AS', []), (1, C(a, c), 'ANDI', [f'b{k}', f'b{k+5}']), (1, out, 'ORI2', [f'b{k+6}']),
             (0, out, 'ORE', [f'b{k+1}', f'b{k+2}', f'b{k+4}', f'b{k+5}', f'b{k+7}'])]
    return prems, body, f'b{k+8}'


def piece_negdisj(X, a, b):
    """( ~ ( A v B ) ) |- ( ( ~ A ) & ( ~ B ) )   (9 body lines)."""
    a, b = A(a), A(b)
    prems, body = [], []
    m = _guard(X, N(O(a, b)), prems, body)
    k = len(body)
    body += [(1, a, 'AS', []), (1, O(a, b), 'ORI1', [f'b{k}']), (1, BOT, 'NEGE', [f'b{k+1}', m]),
             (0, N(a), 'NEGI', [f'b{k}', f'b{k+2}']),
             (1, b, 'AS', []), (1, O(a, b), 'ORI2', [f'b{k+4}']), (1, BOT, 'NEGE', [f'b{k+5}', m]),
             (0, N(b), 'NEGI', [f'b{k+4}', f'b{k+6}']),
             (0, C(N(a), N(b)), 'ANDI', [f'b{k+3}', f'b{k+7}'])]
    return prems, body, f'b{k+8}'


def piece_export(X, a, b, c):
    """( ( A & B ) > C ) |- ( A > ( B > C ) )   (6 body lines)."""
    a, b, c = A(a), A(b), A(c)
    prems, body = [], []
    m = _guard(X, I(C(a, b), c), prems, body)
    k = len(body)
    body += [(1, a, 'AS', []), (2, b, 'AS', []), (2, C(a, b), 'ANDI', [f'b{k}', f'b{k+1}']),
             (2, c, 'IMPE', [m, f'b{k+2}']), (1, I(b, c), 'IMPI', [f'b{k+1}', f'b{k+3}']),
             (0, I(a, I(b, c)), 'IMPI', [f'b{k}', f'b{k+4}'])]
    return prems, body, f'b{k+5}'


def piece_negimp(X, a, b):
    """( ( ~ A ) > B ) , ( ~ B ) |- A   (7 body lines, classical: reductio)."""
    a, b = A(a), A(b)
    prems, body = [N(b)], []
    m = _guard(X, I(N(a), b), prems, body)
    k = len(body)
    body += [(1, N(a), 'AS', []), (1, b, 'IMPE', [m, f'b{k}']), (1, BOT, 'NEGE', [f'b{k+1}', 'p1']),
             (0, N(N(a)), 'NEGI', [f'b{k}', f'b{k+2}']), (0, a, 'DN', [f'b{k+3}'])]
    return prems, body, f'b{k+4}'


# name -> (fn, n_fresh_atoms)
PIECES = {
    'merge': (piece_merge, 3), 'split': (piece_split, 4), 'mt': (piece_mt, 2), 'ds': (piece_ds, 2),
    'contra': (piece_contra, 2), 'peirce': (piece_peirce, 2), 'dist': (piece_dist, 3),
    'negdisj': (piece_negdisj, 2), 'export': (piece_export, 3), 'negimp': (piece_negimp, 2),
}


def compose(names, rng, atoms=LETTERS):
    """Chain the named pieces with fresh atoms. Returns (text, meta) or None if atoms run out."""
    pool = list(atoms)
    rng.shuffle(pool)
    all_prems, lines, X, out_line = [], [], None, None
    for nm in names:
        fn, k = PIECES[nm]
        if k > len(pool):
            return None
        fresh, pool = pool[:k], pool[k:]
        prems, body, out = fn(X, *fresh)
        base_p = len(all_prems)
        all_prems += prems
        # body labels -> absolute 0-based indices are assigned after the whole premise block is known
        lines.append((base_p, body, out, out_line))
        X = None
        for bl, (d, f, r, refs) in enumerate(body):
            if f'b{bl}' == out:
                X = f
        out_line = out
    n_prem = len(all_prems)
    numbered = [(0, p, 'PR', []) for p in all_prems]
    prev_out_abs = None
    for base_p, body, out, _ in lines:
        base_b = len(numbered)
        for d, f, r, refs in body:
            absrefs = []
            for ref in refs:
                if ref == 'in':
                    absrefs.append(prev_out_abs)
                elif ref[0] == 'p':
                    absrefs.append(base_p + int(ref[1:]))
                else:
                    absrefs.append(base_b + int(ref[1:]))
            numbered.append((d, f, r, absrefs))
        prev_out_abs = base_b + int(out[1:])
    concl = numbered[-1][1]
    toks = ['THM'] + ' , '.join(fstr(p) for p in all_prems).split() + ['SEQ'] + fstr(concl).split() + ['PRF']
    for i, (d, f, r, refs) in enumerate(numbered):
        toks += [f'N{i+1}'] + ['|'] * d + fstr(f).split() + [':', r] + [f'N{j+1}' for j in refs] + [';']
    toks.append('QED')
    return ' '.join(toks), {'pieces': list(names), 'n_prem': n_prem, 'n_body': len(numbered) - n_prem,
                            'n_atoms': len(atoms) - len(pool)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=100)
    ap.add_argument('--pieces', type=int, nargs=2, default=[3, 4], help='min max pieces per chain')
    ap.add_argument('--names', nargs='*', default=sorted(PIECES))
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    seen = set()
    with open(a.out, 'w') as fo:
        tries = 0
        while len(seen) < a.n and tries < a.n * 50:
            tries += 1
            k = rng.randint(*a.pieces)
            names = [rng.choice(a.names) for _ in range(k)]
            r = compose(names, rng)
            if r is None or tuple(names) in seen:
                continue
            seen.add(tuple(names))
            text, meta = r
            fo.write(json.dumps({'text': text, **meta}) + '\n')
    print(f'wrote {len(seen)} records to {a.out}', file=sys.stderr)


if __name__ == '__main__':
    main()
