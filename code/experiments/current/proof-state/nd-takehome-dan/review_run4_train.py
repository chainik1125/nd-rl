import json, sys, os
sys.path.insert(0, '/home/dan/nd-takehome')
from review_run5_recount import scan_train, rkey, rd, parse_proof, prune, max_depth
tg = rd('data/p2/targets_depth3.jsonl'); tr = rd('data/p2/transfer_depth3.jsonl'); ho = rd('data/p2/heldout.jsonl')
tkeys = {rkey(r['thm']) for r in tg}; xkeys = {rkey(r['thm']) for r in tr}; hkeys = {rkey(r['thm']) for r in ho}
out = {}
for s in ('a1', 'a2', 'a3'):
    fn = f'data/p2/train_depth3_f0_{s}.jsonl'
    o, keys = scan_train(fn, 'reductio')
    n3 = 0
    for l in open(fn):
        if l.strip():
            ln = parse_proof(json.loads(l)['proof'])
            if ln is not None and max_depth(prune(ln)) >= 3:
                n3 += 1
    o['depth3_proofs(my_pred, pruned)'] = n3
    o['keys_in_targets'] = len(keys & tkeys); o['keys_in_transfer'] = len(keys & xkeys); o['keys_in_heldout'] = len(keys & hkeys)
    out[s] = o; print(s, o, flush=True)
json.dump(out, open('artifacts/review_r4/train.json', 'w'), indent=1)
