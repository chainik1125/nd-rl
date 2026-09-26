import json,collections,sys,statistics as st
sys.path.insert(0,'/scratch/cowifornia/nd-rl/ab3/data_build/tools')
from comp import Part, satisfiable, prompt_str
from keys2 import pkey
D='/scratch/cowifornia/nd-rl/ab3/data_build/'
tp=json.load(open(D+'comp/test_parts.json')); xn=set(tp['names']); xp=set(tp['pkeys'])
cpn={n for l in open(D+'comp/provenance_constructed.jsonl') for n in json.loads(l)['parts']}
res={}
for l in open('/scratch/cowifornia/nd-rl/ab2/data/reserve.jsonl'):
    r=json.loads(l)
    if r['source']=='gen' and r.get('minlen_proof'): res[r['orig_name']]=r
xn|=cpn; xp|={pkey(res[n]['thm']) for n in cpn}
c=collections.Counter(); tok=collections.defaultdict(list); seen=set()
for n,r in res.items():
    if not (5<=r['L_true']<=10) or n in xn: continue
    p=pkey(r['thm'])
    if p in xp or p in seen: continue
    at=len({t for t in r['thm'].split() if t in 'PQRS'})
    try: pt=Part(n,r['thm'],r['minlen_proof'],r['L_true'],'gen')
    except AssertionError: continue
    if not satisfiable(pt.prem) or at>2: continue
    seen.add(p); c[(at,r['L_true'],len(pt.prem))]+=1; tok[(at,r['L_true'])].append(len(prompt_str(pt.prem,pt.concl).split()))
for at in (1,2):
    print(at,'-atom free parts (L, n_prem):',sorted((k[1:],v) for k,v in c.items() if k[0]==at))
    print(at,'-atom prompt tokens by L (mean,max):',{k[1]:(round(st.mean(v)),max(v)) for k,v in sorted(tok.items()) if k[0]==at})
