"""Compare the literal-minlen build (-DTOWERS) and the default no-tower build on random valid sequents, biased toward the proof's
edge cases (F inside the theorem, double negations in the subformulas).  python fuzz_tower.py N FLOOR SEED OUT [--big]
--big: 3-4 atoms, 1-4 premises of depth 2-4, conclusion depth 2-4 (longer proofs)."""
import json, os, random, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import certify as C  # noqa: E402

def rand_f(rng, atoms, d):
    if d == 0 or rng.random() < 0.25:
        r = rng.random()
        return ('bot',) if r < 0.08 else ('atom', rng.choice(atoms))
    op = rng.choice(['not', 'not', 'and', 'or', 'imp', 'nn'])
    if op == 'nn':
        return ('not', ('not', rand_f(rng, atoms, d - 1)))
    if op == 'not':
        return ('not', rand_f(rng, atoms, d - 1))
    return (op, rand_f(rng, atoms, d - 1), rand_f(rng, atoms, d - 1))

n, floor, seed, out = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
big = '--big' in sys.argv[5:]
rng = random.Random(seed)
prompts = []
while len(prompts) < n:
    if big:
        atoms = rng.sample(C.LETTERS[:5], rng.randint(3, 4))
        prem = [rand_f(rng, atoms, rng.randint(2, 4)) for _ in range(rng.randint(1, 4))]
        concl = rand_f(rng, atoms, rng.randint(2, 4))
    else:
        atoms = rng.sample(C.LETTERS[:4], rng.randint(2, 3))
        prem = [rand_f(rng, atoms, rng.randint(1, 3)) for _ in range(rng.randint(0, 3))]
        concl = rand_f(rng, atoms, rng.randint(1, 3))
    if concl in prem or not C.truth_table(prem, concl)[1]:
        continue
    prompts.append('THM ' + ' , '.join(C.fstr(p) for p in prem) + (' ' if prem else '') + 'SEQ ' + C.fstr(concl) + ' PRF')
code = '''
import sys, json; sys.path.insert(0, %r)
from certify_fast import run_fast
for p in json.load(open(sys.argv[1])):
    r = run_fast(p, %d, 20, 256)
    print(json.dumps([p, r['status'], r.get('body'), r.get('error')]), flush=True)
''' % (HERE, floor)
tmp = out + '.prompts.json'
json.dump(prompts, open(tmp, 'w'))
res = {}
for var in ['-DTOWERS', '']:
    env = dict(os.environ, FASTMIN_VARIANT=var)
    o = subprocess.run([sys.executable, '-c', code, tmp], env=env, capture_output=True, text=True).stdout
    res[var] = [json.loads(l) for l in o.splitlines()]
bad = 0
with open(out, 'w') as f:
    for a, b in zip(res['-DTOWERS'], res['']):
        same = a[1:3] == b[1:3]
        f.write(json.dumps({'prompt': a[0], 'full': a[1:], 'notower': b[1:], 'same': same}) + '\n')
        if not same and 'timeout' not in (a[1], b[1]):
            bad += 1
            print('DIFF', a, b)
import collections
print('n', len(res['-DTOWERS']), 'statuses full', collections.Counter(x[1] for x in res['-DTOWERS']),
      'bodies', sorted(collections.Counter(x[2] for x in res['-DTOWERS'] if x[2]).items()), 'DIFFS', bad)
