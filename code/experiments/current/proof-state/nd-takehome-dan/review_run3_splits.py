#!/usr/bin/env python3
"""Reviewer's split-disjointness and f = 0 check for round2-run3 (own renaming key, own predicates). Writes artifacts/review_r3/splits.json."""
import json,sys,os,time,collections
sys.path.insert(0,'/home/dan/review/round2-run3')
from review_run5_recount import rkey, thm_of_prompt, parse_proof, prune, max_depth, p_reductio
def rd(fn): return [json.loads(l) for l in open(fn) if l.strip()]
OUT={}
ev={}
for fn in ['data/p2/targets_depth3.jsonl','data/p2/transfer_depth3.jsonl','data/p2/targets_reductio2.jsonl','data/p2/transfer_reductio2.jsonl','data/p2/heldout.jsonl']:
    ev[fn]={rkey(thm_of_prompt(r['prompt'])) for r in rd(fn)}
    stored={r.get('key') for r in rd(fn)}
    print(fn, len(ev[fn]), 'stored keys', len(stored), 'my==stored', ev[fn]==stored); OUT[fn]={'classes':len(ev[fn]),'my_key_equals_stored':ev[fn]==stored}
ev['val36']={rkey(json.loads(l)['thm'].strip()) for l in open('targets/validation_36.jsonl')}
names=list(ev)
for i,a in enumerate(names):
    for b in names[i+1:]:
        n=len(ev[a]&ev[b])
        OUT[f'{a} ∩ {b}']=n
        if n: print('OVERLAP',a,b,n)
for fn,pat in [('data/p2/train_depth3_f0_a1.jsonl','depth3'),('data/p2/train_reductio_f0.jsonl','reductio')]:
    t0=time.time(); ks=set(); n=0; npat=0; nother=0; maxL=0; unp=0; lens=collections.Counter()
    for l in open(fn):
        r=json.loads(l); n+=1; ks.add(rkey(thm_of_prompt(r['prompt'])))
        L=parse_proof(r['proof'])
        if L is None: unp+=1; continue
        pr=prune(L); lens[len(L)]+=1; maxL=max(maxL,len(L))
        d3=max_depth(pr)>=3; red=p_reductio(pr)
        if pat=='depth3': npat+=d3; nother+=red
        else: npat+=red; nother+=d3
    OUT[fn]={'n':n,'classes':len(ks),'max_written':maxL,'len_hist':dict(sorted(lens.items())),'unparsable':unp,f'{pat}_proofs':npat,'other_pattern_proofs':nother,'overlaps':{e:len(ks&ev[e]) for e in ev}}
    print(fn,'n',n,'classes',len(ks),'maxL',maxL,'len hist',dict(sorted(lens.items())),'unparsable',unp, f'{pat} proofs',npat,'other-pattern proofs',nother, 'overlaps',{e:len(ks&ev[e]) for e in ev}, f'{time.time()-t0:.0f}s', flush=True)

import os; os.makedirs('artifacts/review_r3',exist_ok=True); json.dump(OUT,open('artifacts/review_r3/splits.json','w'),indent=1)
