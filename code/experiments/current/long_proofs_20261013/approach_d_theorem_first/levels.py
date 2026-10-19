"""Per-deepening-level cost of minlen on one prompt: python levels.py 'THM ...' MAXBODY"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import certify as C
from minlen import Search, parse_thm
prompt, maxbody = sys.argv[1], int(sys.argv[2])
prem, concl = parse_thm(prompt)
s = Search(prem, concl)
print('|S|=', len(s.S), flush=True)
for body in range(1, maxbody + 1):
    t0 = time.time(); c0 = s.calls
    r = s.best(concl, frozenset(prem), body)
    print(f'body<={body:2d}: {time.time()-t0:7.2f}s calls+{s.calls-c0:9d} memo={len(s.memo):9d} found={r[0] if r else None}', flush=True)
    if r: break
