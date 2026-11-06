"""Time minlen exhaustion vs body bound for a given prompt (proof known not to exist below written length)."""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
import certify
from minlen import Search, Timeout, parse_thm
prompt = sys.argv[1]; secs = float(sys.argv[2]); bounds = [int(b) for b in sys.argv[3:]]
prem, concl = parse_thm(prompt)
for b in bounds:
    s = Search(prem, concl, deadline=time.time() + secs); t0 = time.time()
    try:
        r = s.best(concl, frozenset(prem), b); status = f'found {r[0]}' if r else 'exhausted'
    except Timeout:
        status = 'TIMEOUT'
    print(f'{prompt[:60]:60s} body_bound {b:2d} {status:12s} {time.time()-t0:7.1f}s memo {len(s.memo)}', flush=True)
