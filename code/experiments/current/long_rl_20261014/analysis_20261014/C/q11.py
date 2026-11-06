import pandas as pd, json, sys
sys.argv = ['x']
src = open('q8.py').read(); exec(src[:src.index('tab = ')])
d = pd.read_pickle('events_df.pkl')
recs = json.load(open('recs.json'))
f = d[(d.ck == 'final') & d.band.isin(['10-14', '15-19', '20-24'])].copy()
st = {i: ('NEST1' if a else 'nestbox' if b else 'flat') for i in f.i.unique() for a, b, _ in [feats(recs[i]['proof'])]}
f['struct'] = f.i.map(st)
for k in ['HOW:BOTE', 'WHAT:BOTE', 'WHAT:ANDI', 'HOW:NEGI', 'CLOSE:NEGI', 'HYP:ORE', 'HOW:ORE', 'HOW:IMPE', 'CLOSE:IMPI']:
    x = f[f.k == k]
    print(k, x.groupby(['struct', 'solved']).surp.agg(['mean', 'size']).round(3).to_dict('index'))
