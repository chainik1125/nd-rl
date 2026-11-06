import pickle, json, collections, sys
import annot
recs = json.load(open('recs.json'))
def scope_info(proof, L):
    lines = annot.parse_proof_tokens(proof.split())
    by = {l['idx']: l for l in lines}
    # box spans: AS idx -> closing line idx
    close_of = {}
    for l in lines:
        if l['rule'] in ('IMPI', 'NEGI'): close_of[l['refs'][0]] = (l['refs'][1], l['idx'])
        if l['rule'] == 'ORE': close_of[l['refs'][1]] = (l['refs'][2], l['idx']); close_of[l['refs'][3]] = (l['refs'][4], l['idx'])
    # in scope at L: i < L and not inside a box [a, end] with end < L (closed), lines of box conclusion L itself excluded
    insc, goals = set(), set()
    for l in lines:
        i = l['idx']
        if i >= L: break
        hidden = any(a <= i <= e and e < L for a, (e, c) in close_of.items())
        if not hidden: insc.add(' '.join(annot.ftoks(l['formula'])))
    for a, (e, c) in close_of.items():
        if a < L <= e: goals.add(' '.join(annot.ftoks(by[e]['formula'])))
        # Lean writes the conclusion line c before its box: if we're writing inside the box, c's formula is also a goal
    return insc, goals
for ck in sys.argv[1:]:
    out = pickle.load(open(f'order_{ck}.pkl', 'rb'))
    c = collections.Counter(); tot = collections.Counter()
    for o in out:
        r = recs[o['i']]
        toks, _ = annot.annotate(r['proof'])
        L = toks[o['pos0']][2]
        cls = o['cls']
        if cls == 'EARLIER':
            insc, goals = scope_info(r['proof'], L)
            cls = 'REDUNDANT' if o['alt'] in insc else 'GOAL_JUMP' if o['alt'] in goals else 'OUT_OF_SCOPE'
        if cls == 'NOVEL':
            insc, goals = scope_info(r['proof'], L)
            if o['alt'] in goals: cls = 'GOAL_JUMP'
        o['cls2'] = cls
        c[(o['rule'], cls)] += 1; tot[cls] += 1
    pickle.dump(out, open(f'order2_{ck}.pkl', 'wb'))
    print(ck, dict(tot))
    for rl in sorted({o['rule'] for o in out}):
        n = sum(v for k, v in c.items() if k[0] == rl)
        print(' ', rl, n, {k[1]: round(v / n, 2) for k, v in sorted(c.items(), key=lambda kv: -kv[1]) if k[0] == rl})
