"""Reach test: run certify_fast at a high floor with a time/memory budget and record, per theorem, the highest
body budget exhausted (= certified floor - 1) or the exact minimum found.
    python reach.py IN.jsonl OUT.jsonl --floor 40 --secs 60 --mem 2048"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from certify_fast import run_fast
ap = argparse.ArgumentParser()
ap.add_argument('inp'); ap.add_argument('out')
ap.add_argument('--floor', type=int, default=40); ap.add_argument('--secs', type=float, default=60)
ap.add_argument('--mem', type=float, default=2048); ap.add_argument('--limit', type=int, default=None)
a = ap.parse_args()
recs = [json.loads(l) for l in open(a.inp)][:a.limit]
with open(a.out, 'w') as fo:
    for r in recs:
        x = run_fast(r['prompt'], a.floor, a.secs, a.mem)
        st = x['stats']
        lv = st.get('levels', [])
        exhausted = max([b for b, t, e in lv if x['status'] != 'found' or b < x.get('body', 99)], default=0)
        out = {'prompt': r['prompt'], 'status': x['status'], 'body': x.get('body'), 'exhausted_body': exhausted,
               'secs': round(x['secs'], 2), 'S': st['S'], 'entries': st['entries'], 'evictions': st['evictions'],
               'levels': lv}
        fo.write(json.dumps(out) + '\n'); fo.flush()
        print(out['status'], out['body'], 'exhausted<=', exhausted, out['secs'], 'S', out['S'], 'entries', out['entries'],
              'evict', out['evictions'], flush=True)
