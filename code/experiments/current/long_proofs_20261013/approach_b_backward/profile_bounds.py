"""Time minlen's iterative deepening per bound on a few theorems (certification cost profile)."""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import certify
from minlen import Search, parse_thm, Timeout
prompts = sys.argv[1:]
for prompt in prompts:
    prem, concl = parse_thm(prompt)
    print(prompt, '| n_prem', len(prem), flush=True)
    s = Search(prem, concl, deadline=time.time() + 100)
    print('  |S| =', len(s.S), flush=True)
    t0 = time.time()
    try:
        for k in range(1, 40):
            r = s.best(concl, frozenset(prem), k)
            print(f'  body<={k}: {"FOUND " + str(r[0]) if r else "none"} cum {time.time()-t0:.1f}s calls {s.calls} memo {len(s.memo)}', flush=True)
            if r:
                break
    except Timeout:
        print('  TIMEOUT at', k, 'calls', s.calls, 'memo', len(s.memo))
