"""High-floor exactness check: the literal-minlen C build (-DTOWERS) vs the default build on the same theorems.
    python crosscheck_towers.py IN.jsonl OUT.jsonl FLOOR SECS"""
import json, os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
inp, out, floor, secs = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
code = '''
import sys, json; sys.path.insert(0, %r)
from certify_fast import run_fast
for line in open(sys.argv[1]):
    p = json.loads(line)['prompt']
    r = run_fast(p, %d, %f, 3000)
    print(json.dumps([p, r['status'], r.get('body'), round(r['secs'], 2), r['stats']['entries']]), flush=True)
''' % (HERE, floor, secs)
res = {}
for var in ['-DTOWERS', '']:
    o = subprocess.run([sys.executable, '-c', code, inp], env=dict(os.environ, FASTMIN_VARIANT=var),
                       capture_output=True, text=True).stdout
    res[var] = [json.loads(l) for l in o.splitlines()]
with open(out, 'w') as f:
    for a, b in zip(res['-DTOWERS'], res['']):
        f.write(json.dumps({'prompt': a[0], 'towers': a[1:], 'default': b[1:]}) + '\n')
        tag = 'same' if a[1:3] == b[1:3] else ('towers timeout' if a[1] == 'timeout' else 'DIFF')
        print(tag, 'towers', a[1:], 'default', b[1:], flush=True)
