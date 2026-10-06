#!/usr/bin/env python3
"""Render check of a training set (proposal 9 protocol): 3,000 records render (lean_tok, lean_seq) -> inverse -> identical ND
proof; Lean accepts 1,000 literal texts; 300 theorem-swapped negatives rejected. Needs Lean (~/.elan/bin/lean): run on a pod.

  python3 dsg_render_check.py data/dsg/train_g1.jsonl --out artifacts/dsg/render_g1.json [--n 3000 --n_lean 1000 --n_neg 300 --seed 0]"""
import argparse, json, random, sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lean_tok import LeanTokenizer
from lean_gate import lean_check
from nd_verify import verify_text

ap = argparse.ArgumentParser()
ap.add_argument('set'); ap.add_argument('--out', required=True); ap.add_argument('--n', type=int, default=3000)
ap.add_argument('--n_lean', type=int, default=1000); ap.add_argument('--n_neg', type=int, default=300); ap.add_argument('--seed', type=int, default=0)
a = ap.parse_args()
rng = random.Random(a.seed)
recs = [json.loads(l) for l in open(a.set) if l.strip()]
sample = rng.sample(recs, min(a.n, len(recs)))
tk = LeanTokenizer('lean_seq')
n_rt = 0; fails = []; texts = []; n_cap = 0
for r in sample:
    ids = tk.encode_proof(r['proof'])
    ids = tk.shift_abs(ids, rng)                  # a random name offset, as training presents it
    nd = tk.decode(ids)
    if nd == r['proof']:
        n_rt += 1
    else:
        fails.append({'thm': r['thm'], 'got': nd[:200]})
    ok, reason, nl = verify_text(r['prompt'] + ' ' + nd)
    n_cap += bool(ok and nl <= 6)
    texts.append((r['prompt'], tk.last_text))
pos = texts[:a.n_lean]
t0 = time.time()
lok, wall, cpu = lean_check([(tk.statement(p), tx) for p, tx in pos])
neg = [(sample[(k + 7) % len(sample)]['prompt'], tx) for k, (p, tx) in enumerate(texts[:a.n_neg])]
nok, _, _ = lean_check([(tk.statement(p), tx) for p, tx in neg])
out = {'set': a.set, 'n_records': len(recs), 'n_sampled': len(sample), 'roundtrip_identical': n_rt, 'denoted_verified_cap6': n_cap,
       'lean_positive_checked': len(pos), 'lean_positive_accepted': sum(lok), 'lean_negative_checked': len(neg), 'lean_negative_rejected': sum(1 for x in nok if not x),
       'lean_wall_s': wall, 'lean_proc_s': cpu, 'roundtrip_failures': fails[:20]}
os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
json.dump(out, open(a.out, 'w'), indent=1)
print(json.dumps(out, indent=1))
