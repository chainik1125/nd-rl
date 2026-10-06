import json, math, hashlib, collections
from pathlib import Path
import numpy as np
S=Path('/data/users/dmitry/ndrl-selection-sprint/code/experiments/current/proof_selection_20261006')
R=Path.home()/'runs/proof-selection'
O=Path('/tmp/proof_selection_20261006_review')
def rows(p): return [json.loads(x) for x in p.open()]
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def auc(s,y):
    # Pairwise wins with half-credit ties, independent of production rank implementation.
    s=np.asarray(s); y=np.asarray(y,bool); p=s[y]; n=np.sort(s[~y])
    if not len(p) or not len(n): return None
    return float((np.searchsorted(n,p,'left')+np.searchsorted(n,p,'right')).sum()/(2*len(n)*len(p)))
results={}
freeze=json.load((S/'results/test_freeze_scores_sha256.json').open())
checks={f:sha(R/f if '/' in f else R/'scores'/f)==h for f,h in freeze['files'].items()}
results['freeze']={'n':len(checks),'bad':[f for f,v in checks.items() if not v],'frozen_utc':freeze['frozen_utc']}
man=json.load((R/'pools_v2/MANIFEST.json').open())
checks={f:sha(R/'pools_v2'/f)==h for p in man['pools'].values() for f,h in [(p['file'],p['sha256']),(p['labels'],p['labels_sha256'])]}
checks['splits']=sha(R/'pools_v2'/man['splits'])==man['splits_sha256']
checks['generator']=sha(Path(man['generator']))==man['generator_sha256']
results['manifest']={'n':len(checks),'bad':[f for f,v in checks.items() if not v]}
perms=[]
for d in sorted((R/'logs').glob('permtest_*')):
    for pool in (d/'pools').glob('*.jsonl'):
        if '.labels.' in pool.name:continue
        lp=pool.with_name(pool.stem+'.labels.jsonl')
        if not lp.exists():continue
        old=rows(R/'pools_v2'/lp.name); new=rows(lp)
        pairs=lambda rr:[(bool(v),r) for row in rr for v,r in zip(row['valid'],row['reason'])]
        sc={f.name:sha(f)==sha(R/'scores'/f.name) for f in (d/'scores').glob('*.jsonl')}
        perms.append({'folder':d.name,'pool_equal':sha(pool)==sha(R/'pools_v2'/pool.name),'labels_changed':sha(lp)!=sha(R/'pools_v2'/lp.name),'joint_histogram_equal':collections.Counter(pairs(old))==collections.Counter(pairs(new)),'scores_equal':sc})
results['permutation_artifacts']=perms
for name in ['holdout250','textbook72']:
    stem=f'{name}__s0_r16__k64'; pool=rows(R/'pools_v2'/f'{stem}.jsonl')
    lab={r['name']:r['valid'] for r in rows(R/'pools_v2'/f'{stem}.labels.jsonl')}
    names=[r['name'] for r in pool]; V=np.array([lab[n] for n in names],bool); D=np.array([[a['done'] for a in r['attempts']] for r in pool])
    scorers=['completion_loglik','verifier_ens','completion','random','loglik_mean','verifier_hist_ens']
    scores={}
    for sc in scorers:
        f={r['name']:r['scores'] for r in rows(R/'scores'/f'{stem}__{sc}.jsonl')}
        assert set(f)==set(names)
        scores[sc]=np.array([f[n] for n in names]); assert scores[sc].shape==V.shape
    assert np.array_equal(scores['completion'],D)
    assert np.array_equal(scores['completion_loglik'],np.where(D,scores['loglik_mean'],-np.inf))
    acc={s:[] for s in scorers}; acc['oracle']=[];acc['random_expectation']=[]
    exact={s:[] for s in scorers}; exact['oracle']=[]
    for ti in range(len(pool)):
        # Same registered random subsets, independent choice code using sorted candidate indices.
        sub=np.argsort(np.random.default_rng(ti*101+8).random((200,64)),axis=1)[:,:8]
        sub=np.sort(sub,axis=1)
        v=V[ti]; acc['oracle'].append(v[sub].any(axis=1).mean());acc['random_expectation'].append(v[sub].mean())
        for s in scorers:
            sc=scores[s][ti]; best=sub[np.arange(200),np.argmax(sc[sub],axis=1)]
            acc[s].append(v[best].mean())
            order=sorted(range(64),key=lambda j:(-sc[j],j))
            exact[s].append(sum(v[j]*math.comb(63-rank,7)/math.comb(64,8) for rank,j in enumerate(order) if rank<=56))
        exact['oracle'].append(1-(math.comb(64-int(v.sum()),8)/math.comb(64,8) if v.sum()<=56 else 0))
    av={s:float(np.mean(v)) for s,v in acc.items()};boot=np.random.default_rng(7).integers(0,len(pool),(1000,len(pool)))
    comparisons={}
    for s in ['verifier_ens','completion']:
        d=np.array(acc['completion_loglik'])-acc[s]
        comparisons[s]={'difference':float(d.mean()),'ci':np.quantile(d[boot].mean(1),[.025,.975]).tolist()}
    aus={}
    for s in ['completion_loglik','verifier_ens','verifier_hist_ens']:
        a=[auc(scores[s][i,D[i]],V[i,D[i]]) for i in range(len(pool))]; a=[x for x in a if x is not None]
        aus[s]={'pooled':auc(scores[s].ravel(),V.ravel()),'finished':auc(scores[s][D],V[D]),'within_finished':float(np.mean(a)),'n':len(a)}
    nd_counts=collections.Counter((a['done'],a['nd'] is not None,bool(lab[r['name']][a['i']])) for r in pool for a in r['attempts'])
    results[name]={'selection':av,'gap_recovered':(av['completion_loglik']-av['random_expectation'])/(av['oracle']-av['random_expectation']),'paired':comparisons,'exact_all_subsets':{s:float(np.mean(v)) for s,v in exact.items()},'exact_random':float(V.mean()),'auroc':aus,'nd_counts':{str(k):v for k,v in nd_counts.items()},'n_nan_scores':sum(np.isnan(s).sum().item() for s in scores.values())}
json.dump(results,(O/'recomputed.json').open('w'),indent=2)
print(json.dumps(results,indent=2))
