import pickle, json, numpy as np, collections
import annot
S = pickle.load(open('scores.pkl', 'rb'))
lab = json.load(open('solved_sets.json')); s0 = set(lab['untied-ei_s0']['rl']) | set(lab['untied-ei_s0']['dev'])
TH = np.log(0.2)
agg = collections.defaultdict(lambda: [0, 0])
ex = collections.defaultdict(collections.Counter)
for i, r in enumerate(S['recs']):
    if r['band'] not in ('10-14', '15-19', '20-24'): continue
    toks, _ = annot.annotate(r['proof'])
    sol = r['name'] in s0
    for ck in ('pretrain', 'final'):
        g, am, _ = S[ck][i]
        for j, t in enumerate(toks):
            role = t[1]; rule = t[3]
            if role == 'BOXOPEN' and t[0] == '(' and toks[j-1][0] == ')': role = 'BOX2_OPEN'   # '(' opening the 2nd Or.elim branch
            key = (ck, sol, role)
            agg[key][0] += g[j] < TH; agg[key][1] += 1
            if ck == 'final' and g[j] < TH and role in ('BOX2_OPEN', 'BOXOPEN', 'PUNCT', 'DEF'):
                ex[(role, sol)][t[0] + '->' + str(int(am[j]))] += 1
roles = sorted({k[2] for k in agg})
print('band 10-24; surprise rate p<0.2 by role: pre_sol pre_uns fin_sol fin_uns  (n fin_uns)')
for ro in roles:
    v = [agg[(ck, s, ro)] for ck in ('pretrain', 'final') for s in (True, False)]
    print(f'{ro:10s}', ' '.join(f'{a / b:.3f}' if b else '  -  ' for a, b in v), v[3][1])

for k, c in ex.items():
    print(k, sum(c.values()), c.most_common(5))
