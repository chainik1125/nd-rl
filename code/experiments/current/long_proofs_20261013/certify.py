"""Shared scoreboard for the long-proofs hunt: check a generated proof, then certify its minimal length.

Length = proof lines AFTER the premise block (`n_body`). A record counts toward the goal when
`cert_ge >= 20`: Dan's minlen search (vendor/fork/minlen.py, branch dan/dan of the take-home fork)
exhausted every proof with fewer than 20 body lines and found none. minlen searches a restricted
space (subformulas of the theorem, their negations and double negations), so "certified" means
"no proof under the floor exists in that space", not a formal lower bound. Timeouts certify nothing.

    from certify import check, certify
    c = check(text)                 # None if rejected, else dict with prompt, n_prem, n_body, key, ...
    r = certify(c['prompt'], c['n_prem'], floor=20, secs=60)
    # r['min_body']: exact minimal body length if minlen found one under the floor, else None
    # r['cert_ge']:  floor if the search was exhausted below it, else None

CLI (tags a jsonl of {"text": "THM ... QED"} records, writes the passing ones with labels):
    python certify.py IN.jsonl OUT.jsonl --floor 20 --secs 60 --procs 2
    python certify.py --score OUT.jsonl [more.jsonl ...]   # grid of certified counts
Atoms: P Q R S T U W X Y Z (no V, which reads as `v`).
"""
import argparse
import collections
import itertools
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vendor', 'fork'))
import nd_verify  # noqa: E402

V = sys.modules['nd_verify.verify']
LETTERS = ['P', 'Q', 'R', 'S', 'T', 'U', 'W', 'X', 'Y', 'Z']
V.ATOM_NAMES.update(LETTERS)
from minlen import minlen, parse_thm  # noqa: E402

BOT = ('bot',)


def atoms(f, acc):
    if f[0] == 'atom':
        acc.add(f[1])
    elif f[0] != 'bot':
        for g in f[1:]:
            atoms(g, acc)
    return acc


def ev(f, val):
    t = f[0]
    if t == 'atom':
        return val[f[1]]
    if t == 'bot':
        return False
    if t == 'not':
        return not ev(f[1], val)
    a, b = ev(f[1], val), ev(f[2], val)
    return (a and b) if t == 'and' else (a or b) if t == 'or' else ((not a) or b)


def truth_table(prem, concl):
    """(premises jointly satisfiable, sequent valid) by brute force over <= 10 atoms (1024 rows)."""
    names = sorted(set().union(*[atoms(f, set()) for f in prem + [concl]]))
    sat = False
    for bits in itertools.product((False, True), repeat=len(names)):
        val = dict(zip(names, bits))
        if all(ev(p, val) for p in prem):
            sat = True
            if not ev(concl, val):
                return sat, False
    return sat, True


def fstr(f):
    if f[0] == 'atom':
        return f[1]
    if f[0] == 'bot':
        return 'F'
    if f[0] == 'not':
        return f'( ~ {fstr(f[1])} )'
    return f'( {fstr(f[1])} ' + {'and': '&', 'or': 'v', 'imp': '>'}[f[0]] + f' {fstr(f[2])} )'


def theorem_key(prem, concl):
    """Identity up to atom renaming and premise order (approximate for premises with equal skeletons):
    premises ordered by their atom-blind skeleton, then atoms renamed by first appearance."""
    def skel(f):
        return ' '.join('a' if t in LETTERS else t for t in fstr(f).split())
    order = sorted(prem, key=lambda p: (skel(p), fstr(p)))
    m = {}
    toks = []
    for t in (' , '.join(fstr(p) for p in order) + ' |- ' + fstr(concl)).split():
        toks.append(m.setdefault(t, f'a{len(m)}') if t in LETTERS else t)
    return ' '.join(toks)


def check(text):
    """Structural checks on a full 'THM ... PRF N1 ... QED' record. Returns None if rejected (reason in
    `check.last_reason`), else a dict with prompt, n_prem, n_body, n_atoms, max_depth, key."""
    def no(reason):
        check.last_reason = reason
        return None
    ok, reason, n_lines = V.verify_text(text)
    if not ok:
        return no('invalid: ' + str(reason))
    prompt, body = text.split(' PRF ', 1)
    prompt += ' PRF'
    prem, concl = parse_thm(prompt)
    if len(set(prem)) != len(prem):
        return no('duplicate premise')
    if concl in prem:
        return no('conclusion is a premise')
    if BOT in prem:
        return no('F premise')
    sat, valid = truth_table(prem, concl)
    if not valid:
        return no('truth table says invalid (verifier bug?)')
    if not sat:
        return no('premises inconsistent (ex falso shortcut)')
    lines = V.parse_proof_tokens(body.split())
    cited = {r for ln in lines for r in ln['refs']}
    if any(ln['rule'] == 'PR' and ln['idx'] not in cited for ln in lines):
        return no('a premise is never cited')
    n_prem = len(prem)
    return {'prompt': prompt, 'text': text, 'n_prem': n_prem, 'n_body': n_lines - n_prem,
            'n_atoms': len(set().union(*[atoms(f, set()) for f in prem + [concl]])),
            'max_depth': max(ln['depth'] for ln in lines),
            'key': theorem_key(prem, concl)}


def certify(prompt, n_prem, floor=20, secs=60.0):
    """Search for a proof with fewer than `floor` body lines. Exhausted with none found -> cert_ge = floor."""
    t0 = time.time()
    r = minlen(prompt, bound=n_prem + floor - 1, time_limit=secs)
    out = {'min_body': None, 'cert_ge': None, 'timeout': r['timeout'], 'cert_secs': round(time.time() - t0, 2),
           'floor': floor}
    if r.get('error'):
        out['error'] = r['error']
    elif r['min_lines_ub'] is not None:
        out['min_body'] = r['min_lines_ub'] - n_prem
    elif not r['timeout']:
        out['cert_ge'] = floor
    return out


def _work(args):
    text, floor, secs = args
    c = check(text)
    if c is None:
        return {'rejected': check.last_reason}
    return {**c, **certify(c['prompt'], c['n_prem'], floor, secs)}


def score(paths):
    seen, rs = set(), []
    for p in paths:
        for line in open(p):
            r = json.loads(line)
            if r.get('key') and r['key'] not in seen:
                seen.add(r['key'])
                rs.append(r)
    cert = [r for r in rs if r.get('cert_ge') and r['cert_ge'] >= 20]
    print(f'{len(rs)} distinct records; certified min body >= 20: {len(cert)}')
    print('certified >=20, by written body length bucket:',
          dict(sorted(collections.Counter(min(r['n_body'] // 10 * 10, 80) for r in cert).items())))
    print('certified >=20, by premise count:', dict(sorted(collections.Counter(r['n_prem'] for r in cert).items())))
    exact = [r for r in rs if r.get('min_body') is not None]
    print('exact minimal body length (under floor):',
          dict(sorted(collections.Counter(r['min_body'] for r in exact).items())))
    print('timeouts:', sum(bool(r.get('timeout')) for r in rs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('paths', nargs='+')
    ap.add_argument('--score', action='store_true')
    ap.add_argument('--floor', type=int, default=20)
    ap.add_argument('--secs', type=float, default=60.0)
    ap.add_argument('--procs', type=int, default=2)
    a = ap.parse_args()
    if a.score:
        return score(a.paths)
    src, dst = a.paths
    texts = [json.loads(line)['text'] for line in open(src)]
    rej = collections.Counter()
    seen = set()
    with ProcessPoolExecutor(a.procs) as ex, open(dst, 'w') as out:
        for r in ex.map(_work, [(t, a.floor, a.secs) for t in texts], chunksize=4):
            if 'rejected' in r:
                rej[r['rejected'].split(':')[0]] += 1
            elif r['key'] in seen:
                rej['duplicate theorem'] += 1
            else:
                seen.add(r['key'])
                out.write(json.dumps(r) + '\n')
                out.flush()
    print('rejected:', dict(rej))
    score([dst])


if __name__ == '__main__':
    main()
