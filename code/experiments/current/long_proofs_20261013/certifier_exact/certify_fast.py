"""Drop-in fast replacement for certify.certify: the same answers as Dan's minlen (vendor/fork/minlen.py), in C.

    from certify_fast import certify_fast
    r = certify_fast(prompt, n_prem, floor=20, secs=60, mem_cap_mb=2048)
    # same keys as certify.certify: min_body, cert_ge, timeout, cert_secs, floor (+ error)
    from certify_fast import run_fast        # more detail: status, body, proof text, search stats
    x = run_fast(prompt, floor=20, secs=60, mem_cap_mb=2048)

CLI (same as certify.py):  python certify_fast.py IN.jsonl OUT.jsonl --floor 20 --secs 60 --procs 2 [--mem 2048] [--proofs]
                           python certify_fast.py --score OUT.jsonl [more.jsonl ...]

The C core (fastmin.c) computes minlen's minimal cost over minlen's proof space; its header lists every difference
from minlen and proves each one cost-preserving (no towers of negations, reduced rules, lemma hoisting).
`mem_cap_mb` caps the memo table (it starts at ~1 MB and doubles; at the cap it evicts, which only costs time).
Any proof found is linearised with minlen's own Search.to_text and checked with nd_verify (atoms widened as in
certify.py); a proof that fails verification is reported as an error, never as a label.
Builds go to build/ on first use (clang). FASTMIN_VARIANT="-DTOWERS" selects minlen's literal rule set (slow;
for cross-checks), FASTMIN_BUILD=DIR pins a prebuilt set of libraries.
"""
import argparse
import ctypes
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import certify as C  # noqa: E402  (widens nd_verify atoms, imports minlen)
from minlen import Search, parse_thm, subformulas  # noqa: E402

V = C.V
KIND = {'atom': 0, 'bot': 1, 'not': 2, 'and': 3, 'or': 4, 'imp': 5}
_LIBS = {}


VARIANT = os.environ.get('FASTMIN_VARIANT', '')   # cross-checks/experiments only: space-separated -D flags


def _lib(nw):
    if nw not in _LIBS:
        flags = VARIANT.split()
        tag = ''.join('_' + f.lstrip('-D').replace('=', '') for f in flags)
        frozen = os.environ.get('FASTMIN_BUILD')   # a frozen build dir: use its libs as they are
        so = os.path.join(frozen or os.path.join(HERE, 'build'), f'libfastmin{nw}{tag}.dylib')
        src = os.path.join(HERE, 'fastmin.c')
        if not os.path.exists(so) or (not frozen and os.path.getmtime(so) < os.path.getmtime(src)):
            os.makedirs(os.path.dirname(so), exist_ok=True)
            subprocess.check_call(['clang', '-O3', '-march=native', '-shared', '-fPIC', f'-DNW={nw}', *flags, src,
                                   '-o', so + f'.{os.getpid()}'])
            os.replace(so + f'.{os.getpid()}', so)
        lib = ctypes.CDLL(so)
        P = ctypes.POINTER
        lib.fm_run.argtypes = [ctypes.c_int, P(ctypes.c_int), P(ctypes.c_int), P(ctypes.c_int), P(ctypes.c_int),
                               ctypes.c_int, ctypes.c_int, P(ctypes.c_int), ctypes.c_int, ctypes.c_int,
                               ctypes.c_double, ctypes.c_double, ctypes.c_char_p, ctypes.c_int, P(ctypes.c_int64)]
        lib.fm_run.restype = ctypes.c_int
        _LIBS[nw] = lib
    return _LIBS[nw]


def _formula_table(prem, concl):
    """minlen's S, as a list with children before parents. -> (formulas, index, atom names)."""
    sub = set()
    for f in prem + [concl]:
        subformulas(f, sub)
    S = set(sub) | {('not', f) for f in sub} | {('not', ('not', f)) for f in sub} | {C.BOT}
    names = sorted(set().union(*[C.atoms(f, set()) for f in prem + [concl]]))
    order = sorted(S, key=lambda f: (len(C.fstr(f)), C.fstr(f)))   # a subformula is strictly shorter
    return order, {f: i for i, f in enumerate(order)}, names


def run_fast(prompt, floor=20, secs=60.0, mem_cap_mb=2048.0, nw=None):
    """-> dict(status in found/exhausted/timeout/error, body, proof text, stats)."""
    prem, concl = parse_thm(prompt)
    order, idx, names = _formula_table(prem, concl)
    if len(names) > 10:
        return {'status': 'error', 'error': 'more than 10 atoms'}
    n = len(order)
    ai = {a: i for i, a in enumerate(names)}
    k = (ctypes.c_int * n)(*[KIND[f[0]] for f in order])
    left = (ctypes.c_int * n)(*[ai[f[1]] if f[0] == 'atom' else idx[f[1]] if f[0] in KIND and f[0] not in ('bot',) else -1
                                for f in order])
    right = (ctypes.c_int * n)(*[idx[f[2]] if f[0] in ('and', 'or', 'imp') else -1 for f in order])
    ins = (ctypes.c_int * n)(*([1] * n))
    pr = (ctypes.c_int * max(1, len(prem)))(*[idx[p] for p in prem])
    buf = ctypes.create_string_buffer(1 << 20)
    stats = (ctypes.c_int64 * 144)()
    t0 = time.time()
    if nw:
        widths = [nw]
    elif '-DTOWERS' in VARIANT.split():
        widths = [2, 4, 8]                    # towers add formulas outside S; retry wider on overflow
    else:
        widths = [1 if n <= 55 else 2 if n <= 111 else 4]   # A is a subset of S; memo records hold |S| goals
    for w in widths:
        left_secs = secs - (time.time() - t0)
        rc = _lib(w).fm_run(n, k, left, right, ins, len(names), len(prem), pr, idx[concl], floor, left_secs,
                            mem_cap_mb, buf, len(buf), stats)
        if rc != -2:
            break
    st = dict(zip(['calls', 'sem_pruned', 'hits', 'stores', 'evictions', 'entries', 'abits', 'nformulas', 'cap', 'values', 'slot_bytes'],
                  list(stats)))
    st['nw'] = w
    st['levels'] = [(b, stats[16 + b] / 1e6, stats[80 + b]) for b in range(1, min(floor, 64)) if stats[16 + b]]
    st['S'] = n
    out = {'stats': st, 'secs': time.time() - t0}
    if rc == -1:
        return {**out, 'status': 'timeout'}
    if rc < 0:
        return {**out, 'status': 'error', 'error': f'fm_run rc={rc}'}
    if rc == 0:
        return {**out, 'status': 'exhausted'}
    toks = buf.value.decode().split()
    F = dict(enumerate(order))
    lib = _lib(w)
    kk, ll, rr = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
    for i in range(n, lib.fm_nf()):     # negation towers created inside C (children precede parents)
        lib.fm_get(i, ctypes.byref(kk), ctypes.byref(ll), ctypes.byref(rr))
        assert kk.value == KIND['not']
        F[i] = ('not', F[ll.value])
    template = _parse_template(toks, F)
    s = Search(prem, concl)
    text = s.to_text(template)
    ok, reason, nl = V.verify_text(prompt + ' ' + text)
    if not ok or nl != len(prem) + rc:
        return {**out, 'status': 'error', 'error': f'verify failed: {reason} nl={nl} body={rc}', 'proof': text}
    return {**out, 'status': 'found', 'body': rc, 'proof': text}


def _parse_template(toks, F):
    pos = [0]

    def form(i):
        return F[int(i)]

    def nxt():
        t = toks[pos[0]]
        pos[0] += 1
        return t

    def node():
        op = nxt()
        if op == 'N':
            return None
        if op in ('R0', 'R'):
            return (op, form(nxt()))
        G = form(nxt())
        if op in ('ANDE1', 'ANDE2', 'DN', 'BOTE', 'ORI1', 'ORI2', 'RAA'):
            return (op, G, node())
        if op in ('ANDI', 'IMPE', 'NEGE'):
            return (op, G, node(), node())
        if op in ('IMPI', 'NEGI'):
            return (op, G, form(nxt()), node())
        if op == 'ORE':
            td = node()
            a = form(nxt()); ta = node()
            c = form(nxt()); tc = node()
            return (op, G, td, a, ta, c, tc)
        if op == 'LEMMA':
            return (op, G, form(nxt()), node(), node())
        raise ValueError(op)

    return node()


def certify_fast(prompt, n_prem, floor=20, secs=60.0, mem_cap_mb=2048.0, want_proof=False):
    """certify.certify's contract (n_prem is implied by the prompt and unused). want_proof=True adds 'min_proof',
    a verified proof of the minimal length, when one was found under the floor."""
    t0 = time.time()
    r = run_fast(prompt, floor, secs, mem_cap_mb)
    out = {'min_body': None, 'cert_ge': None, 'timeout': r['status'] == 'timeout',
           'cert_secs': round(time.time() - t0, 2), 'floor': floor}
    if r['status'] == 'error':
        out['error'] = r['error']
    elif r['status'] == 'found':
        out['min_body'] = r['body']
        if want_proof:
            out['min_proof'] = r['proof']
    elif r['status'] == 'exhausted':
        out['cert_ge'] = floor
    return out


def _work(args):
    text, floor, secs, mem, proofs = args
    c = C.check(text)
    if c is None:
        return {'rejected': C.check.last_reason}
    return {**c, **certify_fast(c['prompt'], c['n_prem'], floor, secs, mem, proofs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('paths', nargs='+')
    ap.add_argument('--score', action='store_true')
    ap.add_argument('--floor', type=int, default=20)
    ap.add_argument('--secs', type=float, default=60.0)
    ap.add_argument('--procs', type=int, default=2)
    ap.add_argument('--mem', type=float, default=2048.0, help='memo cache MB per process')
    ap.add_argument('--proofs', action='store_true', help="also write 'min_proof' (a verified minimal proof)")
    a = ap.parse_args()
    if a.score:
        return C.score(a.paths)
    src, dst = a.paths
    texts = [json.loads(line)['text'] for line in open(src)]
    import collections
    rej = collections.Counter()
    seen = set()
    with ProcessPoolExecutor(a.procs) as ex, open(dst, 'w') as out:
        for r in ex.map(_work, [(t, a.floor, a.secs, a.mem, a.proofs) for t in texts], chunksize=1):
            if 'rejected' in r:
                rej[r['rejected'].split(':')[0]] += 1
            elif r['key'] in seen:
                rej['duplicate theorem'] += 1
            else:
                seen.add(r['key'])
                out.write(json.dumps(r) + '\n')
                out.flush()
    print('rejected:', dict(rej))
    C.score([dst])


if __name__ == '__main__':
    main()
