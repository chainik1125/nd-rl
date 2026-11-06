import pickle, pandas as pd, numpy as np
d = pd.DataFrame(pickle.load(open('events.pkl','rb')))
d['surp'] = d.minlp < np.log(0.2)
d['bits'] = -d.sumlp
d['k'] = d.kind + ':' + d.rule.str.replace('CLOSE_','')
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 200)
t = d.pivot_table(index='k', columns='ck', values='surp', aggfunc='mean')
t['n'] = d[d.ck=='final'].groupby('k').size()
b = d.pivot_table(index='k', columns='ck', values='bits', aggfunc='sum')
t['bits_pre'] = b.pretrain / d[d.ck=='pretrain'].bits.sum()
t['bits_fin'] = b.final / d[d.ck=='final'].bits.sum()
t['bitsper_pre'] = b.pretrain / t.n; t['bitsper_fin'] = b.final / t.n
print(t.sort_values('n', ascending=False).round(3))
print('total bits/proof', d.groupby('ck').bits.sum() / d.i.nunique())
d.to_pickle('events_df.pkl')
