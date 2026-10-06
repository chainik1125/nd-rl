import os,sys,json,hashlib,shutil,subprocess,collections
from pathlib import Path
O=Path('/tmp/proof_selection_20261006_review');R=Path.home()/'runs/proof-selection';S=Path('/data/users/dmitry/ndrl-selection-sprint/code/experiments/current/proof_selection_20261006')
sys.dont_write_bytecode=True
sp=json.load((R/'pools_v2/splits.json').open());allc=collections.Counter();tr=collections.Counter();mixed=0
for s,k in [('rl_targets',16),('dev1108',16),('rr600',32),('holdout250',64),('textbook72',64)]:
    stem=f'{s}__s0_r16__k{k}';lab={r['name']:r['valid'] for r in map(json.loads,(R/'pools_v2'/f'{stem}.labels.jsonl').open())}; train=set(sp['train'].get(s,[]))
    for row in map(json.loads,(R/'pools_v2'/f'{stem}.jsonl').open()):
        allc['attempts']+=len(row['attempts'])
        for a,v in zip(row['attempts'],lab[row['name']]):
            allc['nd_presence_differs_from_done']+=(a['nd'] is not None)!=a['done']
            allc['finished_invalid_with_nd']+=a['done'] and not v and a['nd'] is not None
            allc['judge_fields_in_pool']+=bool(set(a)&{'valid','reason','_why','lean_ok'})
            if row['name'] in train:
                tr['attempts']+=1;tr['valid']+=v;tr['finished_invalid']+=a['done'] and not v
        if row['name'] in train:
            fs=[v for a,v in zip(row['attempts'],lab[row['name']]) if a['done']]
            mixed+=bool(fs) and any(fs) and not all(fs)
# Execute three score-only paths with permuted labels available but file opens denied.
out=O/'isolation_scores';out.mkdir(exist_ok=True)
stem='dev1108__s0_r16__k16'
for sc in ['loglik_mean','entropy_mean']:shutil.copyfile(R/'scores'/f'{stem}__{sc}.jsonl',out/f'{stem}__{sc}.jsonl')
os.environ['SEL_POOLS']=str(R/'logs/permtest_4e0gx3rf/pools');os.environ['SEL_SCORES']=str(out)
sys.path.insert(0,str(S/'code'))
import runpy,builtins
origopen=builtins.open
def forbidden(file,*args,**kwargs):
    if str(file).endswith('.labels.jsonl'):raise AssertionError('label read in score-only path')
    return origopen(file,*args,**kwargs)
builtins.open=forbidden
for prog,args in [('baseline_scores.py',['--sets','dev1108']),('shallow_lr.py',['--score_only','--sets','dev1108','--name','shallow_lr_env']),('shallow_lr.py',['--score_only','--sets','dev1108','--name','shallow_lr'])]:
    sys.argv=[prog]+args;runpy.run_path(str(S/'code'/prog),run_name='__main__')
from train_verifier import load_items
items=load_items({'dev1108':None},with_labels=False)
builtins.open=origopen
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
result={'all_pools':dict(allc),'training':dict(tr,finished_mixed_theorems=mixed),'permuted_label_denial_scoreonly':{sc:sha(out/f'{stem}__{sc}.jsonl')==sha(R/'scores'/f'{stem}__{sc}.jsonl') for sc in ['first','random','completion','shortest','shallow_lr','shallow_lr_env']},'verifier_items_loaded_without_label_access':len(items)}
json.dump(result,(O/'isolation_recomputed.json').open('w'),indent=2);print(json.dumps(result,indent=2))
