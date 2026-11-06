import pickle, numpy as np, json
S = pickle.load(open('scores.pkl','rb'))
import collections
# token-level surprise rate (p<0.2) by absolute token position (prompt+proof) bucket, per ckpt
import sys
sys.path.insert(0,'..')
pl = collections.defaultdict(lambda: collections.defaultdict(list))
for i, r in enumerate(S['recs']):
    np_ = len(r['prompt'].split())  # rough
    for ck in ('pretrain','final'):
        g = S[ck][i][0]
        for j, x in enumerate(g):
            pl[ck][min(j // 100, 10)].append(x < np.log(0.2))
for ck in pl:
    print(ck, [(k*100, round(float(np.mean(v)),3), len(v)) for k, v in sorted(pl[ck].items())])
# per band mean bits per proof token
bb = collections.defaultdict(lambda: collections.defaultdict(list))
for i, r in enumerate(S['recs']):
    for ck in ('pretrain','final'):
        bb[r['band']][ck].append(-S[ck][i][0].mean())
for b in sorted(bb): print(b, {ck: round(float(np.mean(v)),3) for ck, v in bb[b].items()}, len(bb[b]['final']))
