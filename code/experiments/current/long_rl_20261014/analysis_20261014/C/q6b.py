import pickle, json, numpy as np, collections
import annot
S = pickle.load(open('scores.pkl', 'rb'))
lab = json.load(open('solved_sets.json')); s0 = set(lab['untied-ei_s0']['rl']) | set(lab['untied-ei_s0']['dev'])
TH = np.log(0.2)
agg = collections.defaultdict(lambda: [0, 0])
for i, r in enumerate(S['recs']):
    if r['band'] not in ('10-14', '15-19', '20-24'): continue
    toks, _ = annot.annotate(r['proof'])
    sol = r['name'] in s0
    g = S['final'][i][0]
    # paren stack to know, for each '(' 'fun', whether box contains nested boxes; track for each Or.elim its first branch
    stack = []  # entries: dict(kind, nested)
    depth_box = 0
    last_closed = {'nested': False}
    for j, t in enumerate(toks):
        tk, role = t[0], t[1]
        if tk == 'fun' and role == 'BOXOPEN' or (tk == 'fun' and role == 'HEAD'):
            pass
        if tk == '(' and j + 1 < len(toks) and toks[j + 1][0] == 'fun' and toks[j+1][1] == 'BOXOPEN':
            for s in stack: s['nested'] = True
            is2 = toks[j - 1][0] == ')'
            if is2 and True:
                nested1 = last_closed['nested']
                key = ('BOX2_OPEN', 'nested1' if nested1 else 'flat1', len(stack))
                agg[key + (sol,)][0] += g[j] < TH; agg[key + (sol,)][1] += 1
            stack.append({'nested': False, 'open': j, 'kind': ('ORE2n' if last_closed['nested'] else 'ORE2f') if is2 else t[3] + '1'})
        if role == 'HYP' and toks[j - 1][0] == ':':
            key = ('HYP', stack[-1]['kind'], min(len(stack), 3))
            agg[key + (sol,)][0] += g[j] < TH; agg[key + (sol,)][1] += 1
        if role == 'CLOSEREF':
            key = ('CLOSEREF', ('nested' if stack and stack[-1]['nested'] else 'flat') + (stack[-1]['kind'] if stack else 'FINAL'), min(len(stack), 3))
            agg[key + (sol,)][0] += g[j] < TH; agg[key + (sol,)][1] += 1
        if role == 'REF' or (role == 'HEAD' and tk.startswith('n')):
            key = ('REF', 'depth', len(stack))
            agg[key + (sol,)][0] += g[j] < TH; agg[key + (sol,)][1] += 1
        if t[1] == 'PUNCT' and tk == ')' and t[3].startswith('CLOSE_') and toks[j - 1][1] == 'CLOSEREF':
            last_closed = stack.pop()
        elif t[1] == 'PUNCT' and tk == ')' and t[3] == 'CLOSE_NEGI' and toks[j - 1][0] == ')':
            last_closed = stack.pop()
keys = sorted({k[:3] for k in agg})
for k in keys:
    a, b = agg[k + (True,)], agg[k + (False,)]
    print(k, 'solved %.3f (n=%d)' % (a[0] / max(a[1], 1), a[1]), 'unsolved %.3f (n=%d)' % (b[0] / max(b[1], 1), b[1]))
