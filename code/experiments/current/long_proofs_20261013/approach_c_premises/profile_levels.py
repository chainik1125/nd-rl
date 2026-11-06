"""Per-deepening-level cost of minlen on a theorem: seconds, calls and memo size at each total bound.

    python profile_levels.py "THM ... PRF" MAX_BODY SECS
Prints one row per level until a proof is found, the body bound is reached, or the time runs out.
"""
import os
import resource
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'vendor', 'fork'))
import nd_verify  # noqa: E402

V = sys.modules['nd_verify.verify']
V.ATOM_NAMES.update('P Q R S T U W X Y Z'.split())
from minlen import Search, Timeout, parse_thm  # noqa: E402


def profile(prompt, max_body, secs, out=sys.stdout):
    prem, concl = parse_thm(prompt)
    s = Search(prem, concl, deadline=time.time() + secs)
    t0 = time.time()
    rows = []
    for body in range(1, max_body + 1):
        c0 = s.calls
        t1 = time.time()
        try:
            r = s.best(concl, frozenset(prem), body)
        except Timeout:
            print(f'  body {body:2d} TIMEOUT after {time.time() - t1:.1f}s  (total {time.time() - t0:.1f}s)  memo {len(s.memo)}  rss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss >> 20} MB', file=out)
            return rows, None
        dt = time.time() - t1
        rows.append((body, dt, s.calls - c0, len(s.memo)))
        print(f'  body {body:2d} {dt:7.2f}s calls {s.calls - c0:9d} memo {len(s.memo):9d} |S| {len(s.S)}  {"FOUND" if r else ""}', file=out, flush=True)
        if r:
            return rows, body
    return rows, None


if __name__ == '__main__':
    profile(sys.argv[1], int(sys.argv[2]), float(sys.argv[3]))
