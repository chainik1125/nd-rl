"""Timing probe: hand-made hard families; how long does minlen take to exhaust/finish at certify's bound?"""
import sys, time, json, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import certify as C
from minlen import minlen, parse_thm, Search

cases = [
 'THM ( P & ( Q v R ) ) SEQ ( ( P & Q ) v ( P & R ) ) PRF',
 'THM ( ( P v Q ) v R ) SEQ ( P v ( Q v R ) ) PRF',
 'THM ( P v Q ) , ( R v S ) SEQ ( ( ( P & R ) v ( P & S ) ) v ( ( Q & R ) v ( Q & S ) ) ) PRF',
 'THM ( ( P > Q ) & ( Q > R ) ) , ( ( R > S ) & ( S > T ) ) , P SEQ T PRF',
 'THM ( P v Q ) , ( ( ~ P ) v R ) , ( ( ~ Q ) v R ) SEQ R PRF',
 'THM ( P v Q ) , ( P > R ) , ( Q > S ) , ( R > T ) , ( S > T ) SEQ T PRF',
 'THM ( P v Q ) , ( R v S ) , ( P > T ) , ( Q > T ) , ( R > U ) , ( S > U ) SEQ ( T & U ) PRF',
 'THM ( ( P v Q ) & ( R v S ) ) SEQ ( ( P & R ) v ( ( P & S ) v ( ( Q & R ) v ( Q & S ) ) ) ) PRF',
 'THM ( P > Q ) , ( Q > R ) , ( R > S ) , ( S > T ) , ( T > U ) , ( U > W ) , ( W > X ) , ( X > Y ) , ( Y > Z ) SEQ ( P > Z ) PRF',
 'THM ( ( P > Q ) & ( Q > R ) ) , ( ( R > S ) & ( S > T ) ) , ( ( T > U ) & ( U > W ) ) , ( ( W > X ) & ( X > Y ) ) SEQ ( P > Y ) PRF',
]
secs = float(sys.argv[1]) if len(sys.argv) > 1 else 60
for p in cases:
    prem, concl = parse_thm(p)
    s = Search(prem, concl)
    t0 = time.time()
    r = minlen(p, bound=len(prem) + 19, time_limit=secs)
    dt = time.time() - t0
    mb = r['min_lines_ub'] - len(prem) if r['min_lines_ub'] else None
    print(f"|S|={len(s.S):3d} nprem={len(prem)} min_body={mb} timeout={r['timeout']} calls={r['calls']} secs={dt:.1f}  {p[:90]}", flush=True)
