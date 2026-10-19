"""Yield of clause-style theorems needing >= FLOOR body lines, certified with certify_fast.
    python yield_probe.py OUT.jsonl N SEED --prems 4-6 --atoms 4-5 --floor 20 --secs 10"""
import argparse, collections, json, os, random, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import certify as C
from certify_fast import certify_fast
ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('n', type=int); ap.add_argument('seed', type=int)
ap.add_argument('--prems', default='4-6'); ap.add_argument('--atoms', default='4-5')
ap.add_argument('--floor', type=int, default=20); ap.add_argument('--secs', type=float, default=10)
a = ap.parse_args()
rng = random.Random(a.seed)
p0, p1 = map(int, a.prems.split('-')); a0, a1 = map(int, a.atoms.split('-'))
def lit(atoms):
    x = ('atom', rng.choice(atoms)); return ('not', x) if rng.random() < 0.45 else x
seen, cnt, rows, draws = set(), collections.Counter(), [], 0
t_cert = 0.0
with open(a.out, 'w') as fo:
    while len(rows) < a.n:
        draws += 1
        atoms = C.LETTERS[:rng.randint(a0, a1)]
        prem = []
        for _ in range(rng.randint(p0, p1)):
            x, y = lit(atoms), lit(atoms)
            if x[-1] != y[-1]:
                prem.append(rng.choice([('or', x, y), ('or', x, y), ('imp', x, y), ('not', ('and', x, y))]))
        concl = lit(atoms)
        if len(set(prem)) != len(prem) or concl in prem:
            continue
        sat, valid = C.truth_table(prem, concl)
        if not (sat and valid) or any(C.truth_table(prem[:i] + prem[i + 1:], concl)[1] for i in range(len(prem))):
            continue
        key = C.theorem_key(prem, concl)
        if key in seen:
            continue
        seen.add(key)
        prompt = 'THM ' + ' , '.join(C.fstr(p) for p in prem) + ' SEQ ' + C.fstr(concl) + ' PRF'
        r = certify_fast(prompt, len(prem), a.floor, a.secs)
        t_cert += r['cert_secs']
        k = 'cert_ge' if r['cert_ge'] else 'timeout' if r['timeout'] else 'error' if r.get('error') else 'min_%d' % r['min_body']
        cnt['cert_ge' if r['cert_ge'] else 'timeout' if r['timeout'] else 'found']+=1
        rows.append(r)
        fo.write(json.dumps({'prompt': prompt, **r}) + '\n')
mins = collections.Counter(r['min_body'] for r in rows if r['min_body'] is not None)
print(f'{len(rows)} distinct valid theorems from {draws} draws; certify time total {t_cert:.1f}s (mean {t_cert/len(rows):.3f}s)')
print('status', dict(cnt), 'min_body hist', dict(sorted(mins.items())))
print('need >= %d (cert_ge):' % a.floor, cnt['cert_ge'], ' (%.1f%%)' % (100 * cnt['cert_ge'] / len(rows)))
