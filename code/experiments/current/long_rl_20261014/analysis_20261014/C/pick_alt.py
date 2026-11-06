import pandas as pd, numpy as np, sys, pickle
d = pd.read_pickle('events_df.pkl')
K = ['WHAT:IMPE','WHAT:ANDE1','WHAT:ANDE2','WHAT:ORE','WHAT:ANDI','WHAT:BOTE','WHAT:NEGI','HOW:IMPE','HOW:ORE','HOW:BOTE','HYP:ORE','CLOSE:ORE','CLOSE:IMPI','CLOSE:NEGI','HOW:NEGI']
x = d[(d.ck == 'final') & d.surp & d.band.isin(['7-9', '10-14', '15-19']) & d.k.isin(K)].copy()
x['gk'] = x.k + x.solved.astype(str)
parts = []
for (k, s), g in x.groupby(['k', 'solved']):
    parts.append(g.sample(min(len(g), 20 if s else 15), random_state=1))
x = pd.concat(parts).sample(frac=1, random_state=2)
print(x.groupby(['k','solved']).size().unstack())
pickle.dump(x.to_dict('records'), open('cands_alt.pkl', 'wb')); print(len(x))
