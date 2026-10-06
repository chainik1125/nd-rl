import json, math
from pathlib import Path
import numpy as np
R=Path.home()/'runs/proof-selection';S=Path('/data/users/dmitry/ndrl-selection-sprint/code/experiments/current/proof_selection_20261006');O=Path('/tmp/proof_selection_20261006_review')
def rows(p):return [json.loads(x) for x in p.open()]
def mix(F,m):
    m=np.clip(m,1,len(F));k=int(m);w=m-k
    return F[k-1]*(1-w)+F[min(k,len(F)-1)]*w
H=json.load((S/'results/stopping_headline.json').open());result={}
for name in ['holdout250','textbook72']:
    stem=f'{name}__s0_r16__k64'; pool=rows(R/'pools_v2'/f'{stem}.jsonl'); lab={r['name']:r['valid'] for r in rows(R/'pools_v2'/f'{stem}.labels.jsonl')}
    nt=len(pool); ff_a=np.zeros((64,nt));ff_n=np.zeros((64,nt));scs=['completion_loglik','verifier_ens']
    for sc in scs:
        ss={r['name']:r['scores'] for r in rows(R/'scores'/f'{stem}__{sc}.jsonl')}
        fixed=np.zeros((64,nt));adaptive=np.zeros((3,nt));attempts=np.zeros((3,nt))
        choices=[H['choices'][sc][str(m)] for m in [2,4,8]]
        for ti,r in enumerate(pool):
            rng=np.random.default_rng([0,ti]); v=np.array(lab[r['name']],bool);d=np.array([a['done'] for a in r['attempts']]);score=np.array(ss[r['name']])
            for repeat in range(50):
                order=rng.permutation(64);vs=v[order];ds=d[order];scores=score[order]
                picks=[];best=0;first=int(np.flatnonzero(ds)[0]) if ds.any() else 63
                for j in range(64):
                    if scores[j]>scores[best]:best=j
                    picks.append(best)
                    fixed[j,ti]+=vs[best]/50
                    if sc==scs[0]:
                        stop=min(first,j);ff_n[j,ti]+=(stop+1)/50;ff_a[j,ti]+=float(vs[first] if first<=j and ds[first] else vs[0])/50
                for mi,c in enumerate(choices):
                    hits=np.flatnonzero(scores[:c['N']]>=c['tau']);stop=int(hits[0]) if len(hits) else c['N']-1
                    adaptive[mi,ti]+=vs[picks[stop]]/50;attempts[mi,ti]+=(stop+1)/50
        sr={}
        for mi,m in enumerate([2,4,8]):
            a=adaptive[mi];n=attempts[mi];cost=float(n.mean());nearest=int(np.argmin(abs(ff_n.mean(1)-cost)))
            brng=np.random.default_rng(0);bs=[]
            for _ in range(1000):
                ix=brng.integers(0,nt,nt);bs.append(a[ix].mean()-mix(fixed[:,ix],n[ix].mean()).mean())
            d=a-ff_a[nearest];frng=np.random.default_rng(1)
            # production consumes one bootstrap for ceiling, then a second for nearest.
            fci=[]
            for j in [63,nearest]:
                dd=a-ff_a[j];fci.append(np.quantile([dd[frng.integers(0,nt,nt)].mean() for _ in range(1000)],[.025,.975]).tolist())
            means=fixed.mean(1);bestmix=(-1,None)
            for k in range(1,65):
                for l in range(k,65):
                    if not k<=cost<=l:continue
                    w=(cost-k)/(l-k) if l>k else 0
                    value=(1-w)*means[k-1]+w*means[l-1]
                    if value>bestmix[0]:bestmix=(float(value),(k,l,float(w)))
            z=np.load(R/'stopping'/f'{stem}__{sc}.npz'); ni=list(z['Ns']).index(choices[mi]['N']);gi=list(z['taus']).index(choices[mi]['tau'])
            sr[str(m)]={'choice':choices[mi],'cost':cost,'acc':float(a.mean()),'fixed_adjacent':float(mix(fixed,cost).mean()),'diff':float(a.mean()-mix(fixed,cost).mean()),'diff_ci':np.quantile(bs,[.025,.975]).tolist(),'ff_N':nearest+1,'ff_cost':float(ff_n[nearest].mean()),'ff_acc':float(ff_a[nearest].mean()),'ff_diff':float(d.mean()),'ff_ci':fci[1], 'max_error_vs_saved':float(max(abs(a-z['adaptive_acc'][ni,gi]).max(),abs(n-z['adaptive_att'][ni,gi]).max())),'optimistic_all_fixed_mixtures':bestmix}
        result.setdefault(name,{})[sc]=sr
    result[name]['ff_ceiling']={'acc':float(ff_a[-1].mean()),'cost':float(ff_n[-1].mean())}
# Quantile grid reconstruction, labels not needed.
sp=json.load((R/'pools_v2/splits.json').open());grid=json.load((R/'stopping/tau_grid_dev_v.json').open());gridcheck={}
for sc in ['completion_loglik','verifier_ens']:
    q=[]
    for s,k in [('dev1108',16),('rr600',32)]:
        keep=set(sp['dev'][s]);vals=np.array([x for r in rows(R/'scores'/f'{s}__s0_r16__k{k}__{sc}.jsonl') if r['name'] in keep for x in r['scores']]); vals=vals[np.isfinite(vals)]
        q.extend(np.quantile(vals,np.linspace(0,1,101)))
    gridcheck[sc]=np.array_equal(sorted(set(q)),grid[sc])
result['dev_quantile_grid_matches']=gridcheck
json.dump(result,(O/'stopping_recomputed.json').open('w'),indent=2);print(json.dumps(result,indent=2))
