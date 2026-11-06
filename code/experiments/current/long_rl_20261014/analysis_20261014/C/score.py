import sys, json, random, pickle, time
sys.path.insert(0, '..'); import ndmodel as M
import torch; torch.set_num_threads(3)
import numpy as np
E = '/Users/robbie/repos/nd-rl/code/experiments/current/long_rl_20261014'
AV = '/Users/robbie/repos/nd-rl/artifacts/long_rl_20261014/v2'
import pretrain as P
tok = P.make_tokenizer(P.TOK_MODE)
dev = [json.loads(l) for l in open(E + '/data/dev.jsonl')]
rl = [json.loads(l) for l in open(E + '/data/rl.jsonl')]
rl = [r for r in rl if 'proof' in r and r['band'] not in ('0-6',)]
rng = random.Random(0)
bybs = {}
for r in rl: bybs.setdefault(r['band'], []).append(r)
pick = []
for b, xs in sorted(bybs.items()):
    rng.shuffle(xs); pick += xs[:900]
recs = [dict(r, pool='dev') for r in dev] + [dict(r, pool='rl') for r in pick]
print(len(recs), flush=True)
res = {'recs': [{k: r[k] for k in ('name', 'prompt', 'proof', 'body', 'band', 'src', 'pool', 'n_prem')} for r in recs]}
for ck in ['pretrain', 'final']:
    model, _ = M.load(f'{E}/resume/untied-ei_s0/{ck}.pt')
    t = time.time(); out = []
    for i, r in enumerate(recs):
        p, q = tok.encode_prompt(r['prompt']), tok.encode_proof(r['proof'])
        x = torch.tensor([p + q])
        with torch.no_grad():
            lp = torch.log_softmax(model(x[:, :-1])[0, len(p) - 1:].float(), -1)
        g = lp.gather(1, x[0, len(p):, None])[:, 0]
        top = lp.max(-1)
        out.append((g.numpy().astype(np.float32), top.indices.numpy().astype(np.int16), top.values.numpy().astype(np.float32)))
        if i % 1000 == 0: print(ck, i, time.time() - t, flush=True)
    res[ck] = out
pickle.dump(res, open('scores.pkl', 'wb'))
print('done', flush=True)
