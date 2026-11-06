import pickle, glob, collections
rows = [o for f in sorted(glob.glob('alt_final_*.pkl')) for o in pickle.load(open(f, 'rb'))]
print(len(rows))
agg = collections.defaultdict(lambda: [0, 0, 0, 0, 0])  # n events, alt_ok samples, gold_ok samples, samples, events alt any ok
for o in rows:
    for key in [(o['k'], o['solved']), ('ALL', o['solved'])]:
        a = agg[key]; a[0] += 1; a[1] += o['alt_ok']; a[2] += o['gold_ok']; a[3] += o['n']; a[4] += o['alt_ok'] > 0
print(f"{'class':12s} {'solved':6s} {'n':>3s} {'P(ok|alt)':>9s} {'P(ok|gold)':>10s} {'alt any ok':>10s}")
for (k, s), a in sorted(agg.items()):
    print(f'{k:12s} {str(s):6s} {a[0]:3d} {a[1]/a[3]:9.2f} {a[2]/a[3]:10.2f} {a[4]/a[0]:10.2f}')
