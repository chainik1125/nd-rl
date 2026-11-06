"""Hard-theorem pool for reach tests: clause-style premises (2-literal v, >, ~&) over 5-7 atoms, literal conclusion,
valid, consistent, every premise necessary. Then certify_fast at a high floor keeps the long ones.
    python gen_hard.py OUT.jsonl N SEED --prems 5-7 --atoms 5-7 --floor 32 --secs 20 --keep 22"""
import argparse, itertools, json, os, random, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import certify as C
from certify_fast import run_fast

def lit(rng, atoms):
    a = ('atom', rng.choice(atoms))
    return ('not', a) if rng.random() < 0.45 else a

def rows_ok(prem, concl, names):
    return C.truth_table(prem, concl)

ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('n', type=int); ap.add_argument('seed', type=int)
ap.add_argument('--prems', default='5-7'); ap.add_argument('--atoms', default='5-7')
ap.add_argument('--floor', type=int, default=32); ap.add_argument('--secs', type=float, default=20)
ap.add_argument('--keep', type=int, default=22); ap.add_argument('--mem', type=float, default=1024)
a = ap.parse_args()
rng = random.Random(a.seed)
p0, p1 = map(int, a.prems.split('-')); a0, a1 = map(int, a.atoms.split('-'))
kept = 0
seen = set()
with open(a.out, 'w') as fo:
    while kept < a.n:
        atoms = C.LETTERS[:rng.randint(a0, a1)]
        prem = []
        for _ in range(rng.randint(p0, p1)):
            x, y = lit(rng, atoms), lit(rng, atoms)
            if x[-1] == y[-1]:
                continue
            prem.append(rng.choice([('or', x, y), ('or', x, y), ('imp', x, y), ('not', ('and', x, y))]))
        concl = lit(rng, atoms)
        if len(set(prem)) != len(prem) or concl in prem:
            continue
        sat, valid = C.truth_table(prem, concl)
        if not (sat and valid):
            continue
        if any(C.truth_table(prem[:i] + prem[i + 1:], concl)[1] for i in range(len(prem))):
            continue
        prompt = 'THM ' + ' , '.join(C.fstr(p) for p in prem) + ' SEQ ' + C.fstr(concl) + ' PRF'
        key = C.theorem_key(prem, concl)
        if key in seen:
            continue
        seen.add(key)
        r = run_fast(prompt, a.floor, a.secs, a.mem)
        good = r['status'] == 'exhausted' or (r['status'] == 'found' and r['body'] >= a.keep) or r['status'] == 'timeout'
        if good:
            kept += 1
            fo.write(json.dumps({'prompt': prompt, 'status': r['status'], 'body': r.get('body'), 'secs': round(r['secs'], 2),
                                 'S': r['stats']['S'], 'levels': r['stats']['levels']}) + '\n'); fo.flush()
            print(kept, r['status'], r.get('body'), round(r['secs'], 2), 'S', r['stats']['S'], prompt, flush=True)
