import json, os, sys, time, collections
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.expanduser('~/nd-takehome'))
from review_run1_lean_recheck import extract, statement, check_single, WS, OUT
jobs = [('qwen30b', f'{WS}/data/r1/prompts.jsonl', f'{WS}/artifacts/r1/gens_qwen30b.jsonl', f'{WS}/artifacts/r1/scored_qwen30b.jsonl')]
for m in ['Qwen3-0.6B', 'Qwen3-4B', 'Qwen3-8B', 'Qwen3-14B', 'Qwen3-32B']:
    jobs.append((m, f'{WS}/data/r1/scale_prompts.jsonl', f'{WS}/artifacts/r1/gens_scale_{m}.jsonl', f'{WS}/artifacts/r1/scored_scale_{m}.jsonl'))
rep = {}
for label, pf, gf, sf in jobs:
    t0 = time.time()
    P = {json.loads(l)['id']: json.loads(l) for l in open(pf) if '"form": "lean"' in l}
    acc = {(r['id'], r['sample']) for r in map(json.loads, open(sf)) if r['form'] == 'lean' and r['ok']}
    items = []
    for l in open(gf):
        g = json.loads(l)
        if g['id'] not in P: continue
        h = statement(P[g['id']]['prompt'], 't')
        for j, t in enumerate(g['outputs']):
            if (g['id'], j) in acc: items.append(((g['id'], j), h, extract(t)))
    with ThreadPoolExecutor(2) as ex:
        res = list(ex.map(lambda it: check_single(it[1], it[2]), items))
    fails = [(items[k][0], res[k][1][:100]) for k in range(len(items)) if not res[k][0]]
    rep[label] = {'accepted_by_executor': len(items), 'confirmed_alone_in_own_file': len(items) - len(fails), 'fail_examples': fails[:10], 'seconds': round(time.time() - t0)}
    print(label, rep[label], flush=True)
    json.dump(rep, open(f'{OUT}/lean_single_recheck.json', 'w'), indent=1)
