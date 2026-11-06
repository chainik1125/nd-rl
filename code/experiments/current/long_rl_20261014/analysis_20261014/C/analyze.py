import pickle, json, math, sys, collections
import numpy as np
import events as EV
sys.path.insert(0, '..')
import os; os.environ.setdefault('ND_ARCH', 'untied')
import ndmodel as M  # noqa (sets paths)
import pretrain as P
tok = P.make_tokenizer(P.TOK_MODE)
S = pickle.load(open('scores.pkl', 'rb'))
lab = json.load(open('solved_sets.json'))
s0 = set(lab['untied-ei_s0']['rl']) | set(lab['untied-ei_s0']['dev'])
anyrun = set().union(*[set(v['rl']) | set(v['dev']) for v in lab.values()])
TH = math.log(0.2)
rows = []  # one per event per rec
for i, r in enumerate(S['recs']):
    toks, lines, evs = EV.events(r['proof'])
    solved = r['name'] in s0
    for ck in ('pretrain', 'final'):
        g, am, tv = S[ck][i]
        assert len(g) == len(toks)
        for e in evs:
            lp = g[e['pos']]
            jmin = e['pos'][int(np.argmin(lp))]
            first = next((j for j in e['pos'] if g[j] < TH), None)
            rows.append(dict(i=i, ck=ck, kind=e['kind'], rule=e['rule'], own=e['own'], n=len(e['pos']),
                             minlp=float(lp.min()), sumlp=float(lp.sum()), first=first, pos0=e['pos'][0],
                             gold=toks[first][0] if first is not None else None,
                             am=tok.itos[int(am[first])] if first is not None else None,
                             role=toks[first][1] if first is not None else None,
                             relpos=e['pos'][0] / len(toks), solved=solved, anys=r['name'] in anyrun,
                             band=r['band'], src=r['src'].split('.')[0], pool=r['pool'], body=r['body']))
pickle.dump(rows, open('events.pkl', 'wb'))
print(len(rows))
