import json,collections,sys
sys.path.insert(0,'/private/tmp/claude-501/-Users-robbie-repos-nd-rl/a1d02fb4-ba43-48c4-838b-e227293b6c2a/scratchpad/fork51604b3')
import nd_verify.verify; verify=sys.modules["nd_verify.verify"]
verify.ATOM_NAMES.update(['P','Q','R','S','T','U','W','X','Y','Z'])
from nd_verify.verify import parse_proof_tokens, parse_formula
E='/Users/robbie/repos/nd-rl/code/experiments/current/long_rl_20261014'
A='/Users/robbie/repos/nd-rl/artifacts/long_rl_20261014/v2/long-rl-20261014-v2_looped-ei_s0'
dev=[json.loads(l) for l in open(E+'/data/dev.jsonl')]
ev={json.loads(l)['name']:json.loads(l) for l in open(A+'/eval_dev.jsonl')}
def feats(r):
    lines=parse_proof_tokens(r['proof'].split()); body=[l for l in lines if l['rule']!='PR']
    rules=collections.Counter(l['rule'] for l in body)
    used=set()
    for l in body: used.update(l['refs'])
    prem=[l for l in lines if l['rule']=='PR']; n_used_prem=sum(1 for l in prem if l['idx'] in used)
    goal=r['prompt'].split(' SEQ ')[1].replace(' PRF','').split()
    gf,_=parse_formula(goal+['$'],0)
    return {'band':r['band'],'fam':r['src'].split(':')[0].replace('_f20.jsonl','').replace('.jsonl',''),
       'depth':max(l['depth'] for l in lines),'n_ORE':min(rules['ORE'],4),'has_NEGI':rules['NEGI']>0,'has_DN':rules['DN']>0,'has_BOTE':rules['BOTE']>0,
       'classical':(rules['DN']+rules['BOTE']+rules['NEGI'])>0,'goal':gf[0],'n_distractor':r['n_prem']-n_used_prem,'n_prem':r['n_prem'],
       'close':body[-1]['rule'],'body':r['body'],'n_IMPE':min(rules['IMPE'],8)//2*2}
rows=[(feats(r), ev[r['name']]['solved'], ev[r['name']]['n_ok']) for r in dev]
def table(key, bands=('10-14','15-19','20-24')):
    print('--',key)
    for b in bands:
        by=collections.defaultdict(lambda:[0,0])
        for f,s,_ in rows:
            if f['band']==b: by[f[key]][0]+=s; by[f[key]][1]+=1
        print(f'   band {b}: '+'  '.join(f'{k}: {v[0]}/{v[1]}={v[0]/v[1]:.2f}' for k,v in sorted(by.items(), key=lambda kv:str(kv[0])) if v[1]>=5))
for k in ('fam','depth','n_ORE','has_NEGI','has_DN','has_BOTE','classical','goal','n_distractor','close','n_prem','n_IMPE'): table(k)
