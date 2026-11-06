import sys, json, pickle, time, torch
sys.path.insert(0, '..')
import ndmodel as M
torch.set_num_threads(3)
import alt
import pretrain as P
tok = P.make_tokenizer(P.TOK_MODE)
E = '/Users/robbie/repos/nd-rl/code/experiments/current/long_rl_20261014'
ck, lo, hi, k = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
recs = json.load(open('recs.json'))
cands = pickle.load(open('cands_alt.pkl', 'rb'))[lo:hi]
model, _ = M.load(f'{E}/resume/untied-ei_s0/{ck}.pt')
items = []
for e in cands:
    r = recs[int(e['i'])]; ids = tok.encode_proof(r['proof']); j = int(e['first'])
    base = dict(prompt=r['prompt'], proof_ids=ids, j=j, body=r['body'])
    items.append(dict(base, forced=tok.stoi[e['am']]))
    items.append(dict(base, forced=ids[j]))
t = time.time()
res = alt.run(model, tok, items, k=k, seed=lo)
out = []
for q, e in enumerate(cands):
    a, g = res[2 * q], res[2 * q + 1]
    out.append(dict(i=int(e['i']), k=e['k'], first=int(e['first']), gold=e['gold'], am=e['am'], solved=bool(e['solved']), band=e['band'],
                    alt_ok=sum(x[0] for x in a), gold_ok=sum(x[0] for x in g), n=k, alt_samples=[x[1] for x in a]))
pickle.dump(out, open(f'alt_{ck}_{lo}.pkl', 'wb'))
print('done', len(out), time.time() - t, flush=True)
