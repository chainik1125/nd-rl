"""Read-only look at Dan's r16 k=256 pools (textbook72, holdout250): how often is a *finished* proof Lean-valid?

The pool rows keep only valid proof texts, but `reasons` lists every failure; 'lean rejected' = finished but invalid,
everything else = never finished.  So n_finished = n_ok + #'lean rejected'.
"""
import json, collections, sys

rows = []
for s in (0, 1, 2):
    for name, fn in [('tb72', f'pools/s{s}_r16__tb72_x1.jsonl'), ('h250', f'pools/s{s}_r16__h250_x1.jsonl')]:
        for l in open(fn):
            r = json.loads(l)
            n_rej = sum(1 for x in r['reasons'] if x == 'lean rejected')
            rows.append(dict(seed=s, set=name, name=r['name'], n_ok=r['n_ok'], n_rej=n_rej, n_fin=r['n_ok'] + n_rej,
                             n=r['n_tried'], ref=r.get('reference_lines')))


def agg(sel, label):
    ok = sum(r['n_ok'] for r in sel); fin = sum(r['n_fin'] for r in sel); n = sum(r['n'] for r in sel)
    if not n:
        return
    print(f"{label:40s} thms {len(sel):4d} attempts {n:7d} finished {fin/n:.3f} valid {ok/n:.3f} "
          f"P(valid|fin) {ok/max(fin,1):.3f} P(fin|invalid) {(fin-ok)/max(n-ok,1):.3f}")


for name in ('tb72', 'h250'):
    for s in (0, 1, 2):
        agg([r for r in rows if r['set'] == name and r['seed'] == s], f'{name} seed{s} all')
    sel = [r for r in rows if r['set'] == name]
    agg(sel, f'{name} pooled')
    for lo, hi in [(0, 0.001), (0.001, 0.05), (0.05, 0.25), (0.25, 0.5), (0.5, 0.9), (0.9, 1.01)]:
        agg([r for r in sel if lo <= r['n_ok'] / r['n'] < hi], f'  {name} pass-rate in [{lo},{hi})')
sel = [r for r in rows if r['set'] == 'tb72']
for lo, hi in [(0, 7), (7, 10), (10, 13), (13, 40)]:
    agg([r for r in sel if lo <= (r['ref'] or 0) < hi], f'  tb72 ref lines [{lo},{hi})')
bad = [r for r in rows if r['n_fin'] >= 8 and r['n_ok'] / r['n_fin'] < 0.8]
print('theorem-rows with >=8 finished and P(valid|finished)<0.8:', len(bad), 'of', len(rows))
for r in sorted(bad, key=lambda r: r['n_ok'] / r['n_fin'])[:15]:
    print('  ', r['set'], r['seed'], r['name'][:28], 'ok', r['n_ok'], 'fin', r['n_fin'], 'ref', r['ref'])
# how much of the pass@k gap is "first finished proof" already closing?  (single-theorem, i.i.d. approx)
print('\nper-theorem: P(valid|finished) distribution (rows with >=1 finished):')
hist = collections.Counter()
for r in rows:
    if r['n_fin']:
        hist[round(r['n_ok'] / r['n_fin'], 1)] += 1
print(sorted(hist.items()))
