"""For surprised WHAT events (first token with p_gold<0.2): force the model's argmax token, greedy-decode the rest of
the line, and classify the alternative line's formula against the gold proof's have-formulas:
LATER (a gold line written later: ordering choice), EARLIER (already derived), CLOSE (argmax = exact), NOVEL."""
import sys, pickle, json, random, collections, torch
import math
sys.path.insert(0, '..')
import ndmodel as M
torch.set_num_threads(3)
import annot
H = M.H
import pretrain as P
tok = P.make_tokenizer(P.TOK_MODE)
ck = sys.argv[1]; N = int(sys.argv[2])
E = '/Users/robbie/repos/nd-rl/code/experiments/current/long_rl_20261014'
S = pickle.load(open('scores.pkl', 'rb')); S = {'recs': S['recs'], ck: [(None, a, None) for _, a, _ in S[ck]]}
class R(dict): __getattr__ = dict.__getitem__
x = [R(e, first=int(e['first']), i=int(e['i']), pos0=int(e['pos0']), rule=e['k'].split(':')[1]) for e in pickle.load(open(f'cands_{ck}.pkl', 'rb'))]
x.sort(key=lambda e: e.i)
model, _ = M.load(f'{E}/resume/untied-ei_s0/{ck}.pt')
items = []
for e in x:
    r = S['recs'][e.i]
    toks, lines = annot.annotate(r['proof'])
    ids = tok.encode_proof(r['proof'])
    am = int(S[ck][e.i][1][e.first])
    items.append((e, r, toks, ids, am))
out = []
B = 8
for s in range(0, len(items), B):
    ch = items[s:s + B]
    pre = [tok.encode_prompt(r['prompt']) + ids[:e.first] + [am] for e, r, toks, ids, am in ch]
    with torch.no_grad():
        res = H.generate_ids(model, tok, pre, True, 1.0, 40, None)
    for (e, r, toks, ids, am), o in zip(ch, res):
        line = [t[0] for t in toks[e.pos0:e.first]] + [tok.itos[am]] + [tok.itos[q] for q in o]
        # gold have-formulas with their position
        forms = collections.defaultdict(list)
        cur = None
        for j, t in enumerate(toks):
            if t[1] == 'FORM':
                if cur is None or cur[0] != t[2]: cur = (t[2], j, []); forms_l = cur
                cur[2].append(t[0])
            elif cur is not None:
                forms[' '.join(cur[2])].append(cur[1]); cur = None
        if tok.itos[am] == 'exact': cls, alt = 'CLOSE', ''
        else:
            alt = ' '.join(line[3:line.index(':=')]) if ':=' in line else None
            if alt is None: cls = 'UNPARSED'
            elif alt == ' '.join(t[0] for t in toks if t[1] == 'FORM' and t[2] == e.own): cls = 'SAME'
            elif any(p > e.pos0 for p in forms.get(alt, [])): cls = 'LATER'
            elif alt in forms: cls = 'EARLIER'
            else: cls = 'NOVEL'
        gold = ' '.join(t[0] for t in toks if t[1] == 'FORM' and t[2] == e.own)
        out.append(dict(i=e.i, rule=e.rule, cls=cls, alt=alt, gold=gold, band=e.band, solved=e.solved, pos0=e.pos0, first=e.first))
    print(s, flush=True)
pickle.dump(out, open(f'order_{ck}.pkl', 'wb'))
c = collections.Counter((o['rule'], o['cls']) for o in out)
for rl in sorted({o['rule'] for o in out}): print(rl, {k[1]: v for k, v in c.items() if k[0] == rl})
