import pandas as pd, numpy as np
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 300); pd.set_option('display.max_columns', 20)
d = pd.read_pickle('events_df.pkl')
e = d[(d.pos0 < 300)]
t = e.pivot_table(index='k', columns='ck', values='surp', aggfunc='mean').round(3)
t['n'] = e[e.ck=='final'].groupby('k').size()
print('pos<300'); print(t.sort_values('n', ascending=False))
f = d[(d.ck=='final') & d.surp]
for k in ['WHAT:ORE','WHAT:IMPE','WHAT:ANDE1','WHAT:ANDE2','HOW:IMPE','HYP:ORE','CLOSE:ORE','HOW:ORE','WHAT:NEGI','CLOSE:IMPI','WHAT:BOTE','WHAT:ANDI']:
    x = f[f.k==k]
    print(k, len(x), 'role', dict(x.role.value_counts().head(4)), 'gold->am', dict((x.gold+'->'+x.am).value_counts().head(8)))
