#!/usr/bin/env python3
"""Run 2 pretraining sets: uniform per-length draws from the cap-6 pool, optionally with one run-2 pattern excluded (f = 0).

  python run2_sets.py --pool data/p2/pool_cap6_recon.jsonl --outdir data/r2 --heldout data/p2/heldout.jsonl \
      --exclude data/r2/targets_*.jsonl data/r2/transfer_*.jsonl --sets struct:none,impi_ore_f0:impi_ore,negi_ande_hyp_f0:negi_ande_hyp,ori_ore_f0:ori_ore
Every record is verifier-checked at write time (cap 6 asserted); the written file is re-classified with patterns.py and
patterns2.py (pruned and written form) and the counts stored in data/r2/assemble_report_r2.json; f = 0 asserted.
"""
import argparse, json, random, sys, os, collections, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nd_verify import verify_text
from patterns import classify, PATTERNS
from patterns2 import classify2, PATTERNS2

ap = argparse.ArgumentParser()
ap.add_argument('--pool', required=True); ap.add_argument('--outdir', required=True); ap.add_argument('--heldout', required=True)
ap.add_argument('--exclude', nargs='*', default=[]); ap.add_argument('--sets', required=True); ap.add_argument('--size', type=int, default=155000)
ap.add_argument('--lens', default='2-6'); ap.add_argument('--seed', type=int, default=0)
a = ap.parse_args()
lo, hi = map(int, a.lens.split('-')); LENS = list(range(lo, hi + 1)); quota = a.size // len(LENS)
excl = set()
for pat in a.exclude:
    for fn in glob.glob(pat):
        excl |= {json.loads(l)['key'] for l in open(fn) if l.strip()}
excl |= {json.loads(l)['key'] for l in open(a.heldout) if l.strip()}
print('excluded classes', len(excl), flush=True)
pool = [json.loads(l) for l in open(a.pool) if l.strip()]
n0 = len(pool); pool = [r for r in pool if r['key'] not in excl]
print(f'pool {n0} -> {len(pool)} after exclusion', flush=True)
cl2 = {}
for r in pool:
    c = classify2(r['proof']); cl2[r['name']] = {p: c[p] for p in PATTERNS2}
print('pool run-2 pattern counts', {p: sum(v[p] for v in cl2.values()) for p in PATTERNS2}, flush=True)
os.makedirs(a.outdir, exist_ok=True)
report = {}
for i, spec in enumerate(a.sets.split(',')):
    tag, pat = spec.split(':')
    rng = random.Random(a.seed * 100 + i)
    cands = [r for r in pool if pat == 'none' or not cl2[r['name']][pat]]
    by_len = collections.defaultdict(list)
    for r in cands:
        by_len[r['n_lines']].append(r)
    sel = []
    for L in LENS:
        rs = by_len[L]; rng.shuffle(rs)
        assert len(rs) >= quota, (tag, L, len(rs))
        sel += rs[:quota]
    rng.shuffle(sel)
    fn = f'{a.outdir}/train_r2_{tag}.jsonl'
    cnt = collections.Counter(); wcnt = collections.Counter()
    with open(fn, 'w') as f:
        for j, r in enumerate(sel):
            ok, reason, nl = verify_text(r['prompt'] + ' ' + r['proof'])
            assert ok and nl == r['n_lines'] <= hi, (reason, r)
            c1 = classify(r['proof']); c1w = classify(r['proof'], pruned=False); c2 = classify2(r['proof']); c2w = classify2(r['proof'], pruned=False)
            for p in PATTERNS: cnt[p] += c1[p]; wcnt[p] += c1w[p]
            for p in PATTERNS2: cnt[p] += c2[p]; wcnt[p] += c2w[p]
            rec = {'name': f'train_r2_{tag}_{j}', 'thm': r['thm'], 'key': r['key'], 'prompt': r['prompt'], 'proof': r['proof'], 'text': r['prompt'] + ' ' + r['proof'],
                   'n_lines': r['n_lines'], 'rules': r.get('rules'), 'n_prem': r.get('n_prem'), 'pat': r.get('pat'), 'pat2': cl2[r['name']]}
            f.write(json.dumps(rec) + '\n')
    if pat != 'none':
        assert cnt[pat] == 0 and wcnt[pat] == 0, (tag, cnt[pat], wcnt[pat])
    report[tag] = {'n': len(sel), 'excluded_pattern': pat, 'per_len': quota, 'counts_pruned': dict(cnt), 'counts_written': dict(wcnt), 'seed': a.seed * 100 + i}
    print(tag, len(sel), 'pruned', dict(cnt), 'written', dict(wcnt), flush=True)
json.dump(report, open(f'{a.outdir}/assemble_report_r2.json', 'w'), indent=1)
