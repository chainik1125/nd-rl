"""Show generated proof vs minlen's shorter proof for smoke records (diagnosis only)."""
import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import certify
from minlen import minlen
def pp(text):
    body = text.split(' PRF ', 1)[1]
    for ln in body.split(';')[:-1]:
        print('    ', ln.strip())
for l in open(sys.argv[1]):
    r = json.loads(l)
    if r['min_body'] is None: continue
    print('=' * 100); print(r['prompt'], 'gen body', r['n_body'], 'min body', r['min_body'])
    print('  GENERATED:'); pp(r['text'])
    m = minlen(r['prompt'], bound=r['n_prem'] + r['min_body'], time_limit=120)
    print('  MINLEN:'); pp(r['prompt'] + ' ' + m['proof'])
