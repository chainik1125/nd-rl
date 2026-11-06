import pandas as pd, numpy as np, sys, pickle
d = pd.read_pickle('events_df.pkl')
ck, N = sys.argv[1], int(sys.argv[2])
x = d[(d.ck == ck) & (d.kind == 'WHAT') & d.surp]
if ck == 'pretrain': x = x[x.pos0 < 300]
x = x.groupby('rule', group_keys=False).apply(lambda g: g.sample(min(len(g), N), random_state=0))
pickle.dump(x.to_dict('records'), open(f'cands_{ck}.pkl', 'wb')); print(len(x))
