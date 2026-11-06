import pandas as pd, numpy as np
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 300)
d = pd.read_pickle('events_df.pkl')
f = d[(d.ck == 'final') & (d.band == '15-19')]
t = f.groupby(['k', 'src']).surp.mean().unstack().round(3)
n = f.groupby(['k', 'src']).size().unstack()
t = t[[c for c in ['c_f20', 's2_f20', 's_f20', 'd_f20']]]
print(t.loc[n['c_f20'].sort_values(ascending=False).index[:16]])
# per proof bits for c vs s2
p = f.groupby(['i', 'src']).bits.sum().reset_index().groupby('src').bits.mean(); print(p)
