import sys,json,hashlib,collections
from pathlib import Path
sys.dont_write_bytecode=True
S=Path('/data/users/dmitry/ndrl-selection-sprint/code/experiments/current/proof_selection_20261006');R=Path.home()/'runs/proof-selection';O=Path('/tmp/proof_selection_20261006_review')
sys.path.insert(0,str(S/'code'));sys.path.insert(0,str(R/'nd-takehome'))
from state_env import Env,is_name
from tokenizer import make_tokenizer
from score_loglik import name_mask,score_pairs
import torch
TK=make_tokenizer('lean_staten')
class AuditEnv(Env):
    def _assign_names(self,act,i):
        self.audit_mask=[1];j=None
        if i<len(act) and act[i]=='(' and i+3<len(act):j=i+3
        elif i<len(act) and act[i]=='Or.elim' and i+5<len(act):j=i+5
        if j is not None and is_name(act[j]):self.audit_mask.append(j)
        return super()._assign_names(act,i)
result={}
for name in ['holdout250','textbook72']:
    c=collections.Counter()
    with (R/'pools_v2'/f'{name}__s0_r16__k64.jsonl').open() as f:
        for line in f:
            row=json.loads(line)
            for a in row['attempts']:
                e=AuditEnv(row['prompt'],canon=True,base=a['base'],assign=True);c['attempts']+=1
                for ids,raw,has,stored_sha in zip(a['actions'],a['raw_actions'],a['has_eos'],a['state_sha1']):
                    c['steps']+=1;state=TK.encode_toks(e.state_tokens());h=hashlib.sha1(','.join(map(str,state)).encode()).hexdigest()[:16]
                    c['state_mismatch']+=h!=stored_sha
                    c['eos_steps' if has else 'truncated_steps']+=1
                    assert has==(TK.eos in raw)
                    assert not has or raw[-1]==TK.eos
                    assert [x for x in raw if x not in (TK.pad,TK.eos)]==ids
                    c['raw_sampled_pads']+=raw.count(TK.pad)
                    toks=[TK.itos[x] for x in ids]; mask=name_mask(toks)
                    e.audit_mask=[]
                    if has:e.apply(toks)
                    c['mask_mismatch_steps']+=mask!=e.audit_mask
                    if a['done']:c['finished_mask_mismatch_steps']+=mask!=e.audit_mask
                    if len(e.audit_mask)==2:c['Or_elim_binders' if 'Or.elim' in toks else 'box_binders']+=1
                    if e.audit_mask:c['have_names']+=1
                c['done_mismatch']+=bool(e.done)!=a['done'];c['stepcount_mismatch']+=e.steps!=a['steps']
                if a['done']:c['nd_mismatch']+=e.nd()!=a['nd']
    result[name]=dict(c);print(name,dict(c),flush=True)
# CPU score-path probe: one real eos, one truncated sequence. Uniform logits isolate exact inclusion/masking counts.
class Uniform:
    def __call__(self,x):return torch.zeros((*x.shape,len(TK.itos)))
# A pad token here is genuinely sampled and retained. EOS contributes exactly one extra target.
st=TK.encode_toks(Env('THM P SEQ P PRF',canon=True,base=0,assign=True).state_tokens())
a=[TK.stoi['have'],TK.stoi['n1'],TK.pad]
out=score_pairs(Uniform(),[(st,a,[1],True),(st,a,[1],False)],torch.device('cpu'),bf16=False)
result['cpu_eos_probe']={'full_lengths':[len(r[3]) for r in out],'unmasked_lengths':[sum(x is not None for x in r[0]) for r in out]}
assert result['cpu_eos_probe']=={'full_lengths':[4,3],'unmasked_lengths':[3,2]}
json.dump(result,(O/'replay_recomputed.json').open('w'),indent=2)
print(result['cpu_eos_probe'])
