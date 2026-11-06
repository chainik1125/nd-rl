"""Fixed workload timing: python bench.py [bench_set.jsonl]  (env FASTMIN_VARIANT selects compile flags).
Each theorem runs at its recorded floor; the answer must equal the recorded one."""
import json, sys, time, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from certify_fast import run_fast, VARIANT
path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bench_set.jsonl')
mem = float(os.environ.get('BENCH_MEM', 2048))
tot = calls = ents = 0
bad = 0
for line in open(path):
    r = json.loads(line)
    t0 = time.time()
    x = run_fast(r['prompt'], r['floor'], 300, mem)
    dt = time.time() - t0
    got = [x['status'], x.get('body')]
    bad += got != r['got']
    tot += dt; calls += x['stats']['calls']; ents += x['stats']['entries']
    print(f"{dt:7.2f}s {got} {'OK' if got == r['got'] else 'MISMATCH ' + str(r['got'])} nw={x['stats']['nw']} abits={x['stats']['abits']} entries={x['stats']['entries']}", flush=True)
print(f'VARIANT={VARIANT!r} total {tot:.2f}s calls={calls} entries={ents} mismatches={bad}')
