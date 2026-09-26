#!/usr/bin/env python3
"""Additivity validation sample for 3-part ANDI joins (not an output file): k = 3 joins (atoms 1 + 1 + <= 2, same
construction: build_mid.combine_tree) at written length 11-17, built from parts already used in sft_long / sft_short
(SFT side, so no test part is touched). Exact minlen is run on them; agreement with the written length is reported.
  python tools/validate_k3.py --seed 34101 --per 60"""
import json, sys, random, collections, argparse
sys.path.insert(0, '/scratch/cowifornia/nd-rl/ab3/data_build/tools'); sys.path.insert(0, '/scratch/cowifornia/nd-rl/ab3/midtrain/build/tools')
from comp import Part, check, thm_str, ATOMS, prompt_str
from build_mid import combine_tree
from keys2 import pkey
B = '/scratch/cowifornia/nd-rl/ab3/midtrain/build'
ap = argparse.ArgumentParser(); ap.add_argument('--seed', type=int, default=34101); ap.add_argument('--per', type=int, default=60)
a = ap.parse_args(); rng = random.Random(a.seed)
names = {n for fn in ('sft_long', 'sft_short') for l in open(f'{B}/raw/{fn}.jsonl') for n in json.loads(l)['parts']}
res = {}
for l in open('/scratch/cowifornia/nd-rl/ab2/data/reserve.jsonl'):
    r = json.loads(l)
    if r['orig_name'] in names:
        res[r['orig_name']] = Part(r['orig_name'], r['thm'], r['minlen_proof'], r['L_true'], 'gen')
by = collections.defaultdict(list)
for p in res.values():
    by[(p.L, len(p.atoms))].append(p)
out = []; seen = set()
for L in range(11, 18):
    n = 0; tries = 0
    while n < a.per and tries < 200000:
        tries += 1
        tot = L - 2; l1 = rng.randint(2, tot - 4); l2 = rng.randint(2, tot - l1 - 2); l3 = tot - l1 - l2
        o1, o2, o3 = by.get((l1, 1)), by.get((l2, 1)), by.get((l3, 2), []) + by.get((l3, 1), [])
        if not (o1 and o2 and o3):
            continue
        ch = [rng.choice(o1), rng.choice(o2), rng.choice(o3)]
        if len({c.name for c in ch}) < 3 or sum(len(c.prem) for c in ch) > 5 or sum(len(c.atoms) for c in ch) > 4:
            continue
        at = ATOMS[:]; rng.shuffle(at); rs = []; o = 0
        for c in ch:
            rs.append(c.renamed(at[o:o + len(c.atoms)], rng)); o += len(c.atoms)
        prem, concl, lines = combine_tree(rs, rng)
        if len(prompt_str(prem, concl).split()) > 150 or concl in prem:
            continue
        pk = pkey(thm_str(prem, concl))
        if pk in seen:
            continue
        ok, reason, prompt, txt = check(prem, concl, lines); assert ok
        seen.add(pk); n += 1
        out.append({'name': f'valk3_{L}_{n}', 'thm': thm_str(prem, concl), 'prompt': prompt, 'n_lines': L, 'bound': L, 'proof': txt, 'part_L': [c.L for c in ch]})
with open(f'{B}/raw/validate_k3.jsonl', 'w') as f:
    for r in out:
        f.write(json.dumps(r) + '\n')
print(len(out), collections.Counter(r['n_lines'] for r in out))
