"""Small-S hunt: sequents whose subformula set is tiny (minlen's cost is driven by |S|), with long minimal proofs.

Enumerates subformula-closed pools T over the given atoms (|T| <= max_sub), then every sequent whose premises
(0..max_prem of them) and conclusion come from T, keeps the ones that are valid, with consistent premises each
semantically necessary and not implied by the rest, and runs minlen with a cheap bound to find the minimal
length (or "no proof <= bound"). Streams records to OUT.jsonl.

    python tiny_s.py OUT.jsonl --atoms PQ --max_sub 6 --max_prem 3 --bound 13 --secs 3 --procs 2
"""
import argparse
import itertools
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'vendor', 'fork'))
sys.path.insert(0, os.path.join(HERE, '..'))
import nd_verify  # noqa: E402

V = sys.modules['nd_verify.verify']
V.ATOM_NAMES.update('P Q R S T U W X Y Z'.split())
from certify import fstr, truth_table  # noqa: E402
from minlen import minlen, subformulas  # noqa: E402


def closed_pools(atoms, max_sub):
    """All subformula-closed sets containing every atom, of size <= max_sub (as frozensets), grown one formula at a time."""
    base = frozenset(('atom', a) for a in atoms)
    seen = {base}
    frontier = [base]
    while frontier:
        nxt = []
        for T in frontier:
            if len(T) >= max_sub:
                continue
            cands = set()
            for f in T:
                cands.add(('not', f))
                for g in T:
                    for op in ('and', 'or', 'imp'):
                        cands.add((op, f, g))
            for c in cands:
                if c in T:
                    continue
                T2 = T | {c}
                if T2 not in seen:
                    seen.add(T2)
                    nxt.append(T2)
        frontier = nxt
    return seen


def sequents(T, max_prem, min_prem):
    T = sorted(T, key=lambda f: (len(fstr(f)), fstr(f)))
    for c in T:
        if c[0] == 'atom' and min_prem == 0:
            pass
        rest = [f for f in T if f != c]
        for k in range(min_prem, max_prem + 1):
            for prem in itertools.combinations(rest, k):
                yield list(prem), c


def good(prem, c):
    sat, valid = truth_table(prem, c)
    if not (sat and valid):
        return False
    if not prem and c[0] == 'atom':
        return False
    for i in range(len(prem)):
        rest = prem[:i] + prem[i + 1:]
        if truth_table(rest, c)[1] or truth_table(rest, prem[i])[1]:
            return False
    return True


def prompt_of(prem, c):
    return 'THM ' + (' , '.join(fstr(p) for p in prem) + ' ' if prem else '') + 'SEQ ' + fstr(c) + ' PRF'


def work(args):
    prompt, n_prem, bound, secs = args
    t0 = time.time()
    r = minlen(prompt, bound=n_prem + bound, time_limit=secs)
    return {'prompt': prompt, 'n_prem': n_prem, 'min_body': None if r['min_lines_ub'] is None else r['min_lines_ub'] - n_prem,
            'timeout': r['timeout'], 'bound_body': bound, 'secs': round(time.time() - t0, 2), 'proof': r['proof']}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--atoms', default='PQ')
    ap.add_argument('--max_sub', type=int, default=6)
    ap.add_argument('--min_sub', type=int, default=6, help='only sequents whose closure has at least this many formulas')
    ap.add_argument('--max_prem', type=int, default=3)
    ap.add_argument('--min_prem', type=int, default=0)
    ap.add_argument('--bound', type=int, default=13, help='body-line bound for the first minlen pass')
    ap.add_argument('--secs', type=float, default=3.0)
    ap.add_argument('--procs', type=int, default=2)
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    pools = closed_pools(list(a.atoms), a.max_sub)
    print(f'{len(pools)} closed pools of size <= {a.max_sub} over {a.atoms}', file=sys.stderr)
    jobs = []
    for T in pools:
        if len(T) < a.min_sub:
            continue
        for prem, c in sequents(T, a.max_prem, a.min_prem):
            sub = set()
            for f in prem + [c]:
                subformulas(f, sub)
            if len(sub) != len(T):      # each sequent is produced exactly once: by the pool equal to its own closure
                continue
            if good(prem, c):
                jobs.append((prompt_of(prem, c), len(prem), a.bound, a.secs))
    print(f'{len(jobs)} candidate sequents', file=sys.stderr)
    if a.limit:
        jobs = jobs[:a.limit]
    t0 = time.time()
    n_long = 0
    with ProcessPoolExecutor(a.procs) as ex, open(a.out, 'w') as out:
        for i, r in enumerate(ex.map(work, jobs, chunksize=8)):
            out.write(json.dumps(r) + '\n')
            if r['min_body'] is None and not r['timeout']:
                n_long += 1
            if (i + 1) % 500 == 0:
                out.flush()
                print(f'{i + 1}/{len(jobs)} {time.time() - t0:.0f}s  long(>{a.bound}) so far {n_long}', file=sys.stderr, flush=True)
    print(f'done {len(jobs)} in {time.time() - t0:.0f}s; no proof <= {a.bound}: {n_long}', file=sys.stderr)


if __name__ == '__main__':
    main()
