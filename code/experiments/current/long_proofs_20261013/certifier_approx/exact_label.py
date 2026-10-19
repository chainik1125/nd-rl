"""Turn hunter upper bounds into exact labels: for records whose exact minlen timed out but a hunter found a proof
of L body lines, run the semantic-pruned exact search (lem='all') on levels 1..L-1 with a memo cap; exhausting
them proves min = L (within S).   python exact_label.py CAL.jsonl [...] OUT.jsonl --maxL 16 --secs 150 --cap 9000000
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hunter import hunt

ap = argparse.ArgumentParser()
ap.add_argument('paths', nargs='+')
ap.add_argument('--maxL', type=int, default=16)
ap.add_argument('--secs', type=float, default=150)
ap.add_argument('--cap', type=int, default=9_000_000)
a = ap.parse_args()
out = a.paths.pop()
best = {}
for p in a.paths:
    for line in open(p):
        d = json.loads(line)
        if d['true_min'] is None and d['found'] is not None:
            if d['key'] not in best or d['found'] < best[d['key']]['found']:
                best[d['key']] = d
done = set()
if os.path.exists(out):
    done = {json.loads(l)['key'] for l in open(out)}
with open(out, 'a') as fo:
    for k, d in sorted(best.items(), key=lambda kv: kv[1]['found']):
        if k in done or d['found'] > a.maxL:
            continue
        t0 = time.time()
        r = hunt(d['prompt'], 1, d['found'] - 1, secs=a.secs, lem='all', memo_cap=a.cap)
        row = {'key': k, 'prompt': d['prompt'], 'n_prem': d['n_prem'], 'src': d['src'], 'hunter_found': d['found'],
               'exact_stop': r['stop'], 'exact_exh': r['exh'], 'exact_found': r['found'], 'secs': r['secs'], 'memo': r['memo']}
        if r['found'] is not None:
            row['min_body'] = r['found']; row['ANOMALY'] = 'exact found shorter than hunter'
        elif r['stop'] == 'exhausted':
            row['min_body'] = d['found']
        fo.write(json.dumps(row) + '\n'); fo.flush()
        print(row, flush=True)
