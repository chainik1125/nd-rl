import pandas as pd, numpy as np, json
import lemma
pd.set_option('display.width', 250)
d = pd.read_pickle('events_df.pkl')
recs = json.load(open('recs.json'))
x = d[d.rule == 'ORE'].copy()
kinds = {}
for i in x.i.unique(): kinds[i] = lemma.ore_kinds(recs[i]['proof'])
x['ok'] = [kinds[i].get(o, '?') for i, o in zip(x.i, x.own)]
print(x[x.ck=='final'].groupby(['kind', 'ok']).surp.agg(['mean', 'size']).round(3))
print(x[(x.pos0<300)].groupby(['kind', 'ok', 'ck']).surp.mean().unstack().round(3))
w = x[(x.ck=='final') & (x.kind=='WHAT') & x.band.isin(['10-14','15-19','20-24'])]
print(w.groupby(['band','ok','solved']).surp.agg(['mean','size']).round(3))
# per-proof: has lemma split -> solve rate by band & src
pr = pd.DataFrame([dict(i=i, lem=any(v == 'LEMMA' for v in kinds[i].values()), band=recs[i]['band'], src=recs[i]['src'].split('.')[0]) for i in kinds])
pr['solved'] = d[d.ck=='final'].groupby('i').solved.first().reindex(pr.i).values
print(pr.groupby(['band','src','lem']).solved.agg(['mean','size']).round(3).query('size>=15'))
