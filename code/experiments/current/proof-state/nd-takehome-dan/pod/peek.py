import json, glob, sys, os
sys.path.insert(0, '.')
from patterns import classify
from normalize import norm
for arm in sorted(glob.glob('artifacts/p2/ei_*') + glob.glob('artifacts/p2/frozen_*')):
    if not os.path.isdir(arm): continue
    fs = [f for f in glob.glob(arm + '/found_*.jsonl') if 'transfer' not in f]
    if not fs: continue
    rj = [int(f.split('_')[-1].split('.')[0]) for f in glob.glob(arm + '/round_*.json')]
    if not rj: continue
    r = max(rj); last = f'{arm}/found_{r}.jsonl'
    if not os.path.exists(last): continue
    seen = set(); pat = {'depth3': set(), 'reductio': set(), 'derived_dn': set(), 'derived_ore_strict': set()}; first = {}
    for l in open(last):
        x = json.loads(l); pn = norm(x['proof'])
        if (x['name'], pn) in seen: continue
        seen.add((x['name'], pn)); cl = classify(pn)
        if not cl: continue
        for k in pat:
            if cl[k]:
                pat[k].add(x['name']); first[k] = min(first.get(k, 99), x['round'])
    st = json.load(open(f'{arm}/round_{r}.json'))
    print(f"{os.path.basename(arm):32s} r{r} solved {st['targets_cum']['solved']:4d}/{st['targets_cum']['n']} proofs {len(seen):5d} " + ' '.join(f"{k} {len(v)}(r{first.get(k,'-')})" for k, v in pat.items() if v))
