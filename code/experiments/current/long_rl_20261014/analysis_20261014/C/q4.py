import pandas as pd, numpy as np
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 300); pd.set_option('display.max_columns', 30)
d = pd.read_pickle('events_df.pkl')
f = d[(d.ck=='final') & (d.kind!='PREM')]
f = f.assign(cell=f.band + '|' + f.src)
# per proof stats
p = f.groupby('i').agg(solved=('solved','first'), cell=('cell','first'), band=('band','first'), src=('src','first'), body=('body','first'),
                       nsurp=('surp','sum'), bits=('bits','sum'), nev=('surp','size'))
print('solved share by band'); print(p.groupby('band').solved.agg(['mean','size']))
# matched comparison: within cells with >=10 solved and >=10 unsolved
ok = p.groupby('cell').solved.agg(lambda s: s.sum() >= 10 and (~s).sum() >= 10)
cells = ok[ok].index
print('cells', list(cells))
# per kind surprise rate solved vs unsolved within cell, then average over cells weighted by n
g = f[f.cell.isin(cells)].groupby(['cell','k','solved']).surp.agg(['mean','size']).unstack('solved')
g = g.dropna()
g['diff'] = g[('mean', False)] - g[('mean', True)]
g['w'] = g[('size', False)] + g[('size', True)]
res = g.groupby('k').apply(lambda x: pd.Series({'sol': np.average(x[('mean',True)], weights=x.w), 'uns': np.average(x[('mean',False)], weights=x.w),
                                               'diff': np.average(x['diff'], weights=x.w), 'ncell': len(x), 'n': x.w.sum()}))
print(res.sort_values('n', ascending=False).round(3))
# per-proof: number of surprised events per body line, solved vs unsolved within cell
p['rate'] = p.nsurp / p.body
print(p[p.cell.isin(cells)].groupby(['cell','solved']).rate.mean().unstack().round(3))
p.to_pickle('proofs_df.pkl')
