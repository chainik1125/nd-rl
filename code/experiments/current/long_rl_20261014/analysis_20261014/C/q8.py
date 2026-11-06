import json, collections
import annot
lab = json.load(open('solved_sets.json'))
s0 = set(lab['untied-ei_s0']['rl']) | set(lab['untied-ei_s0']['dev'])
anyr = set().union(*[set(v['rl']) | set(v['dev']) for v in lab.values()])
E = '/Users/robbie/repos/nd-rl/code/experiments/current/long_rl_20261014'
rs = [json.loads(l) for l in open(E + '/data/rl.jsonl')] + [json.loads(l) for l in open(E + '/data/dev.jsonl')]
rs = [r for r in rs if 'proof' in r]
def feats(proof):
    toks, lines = annot.annotate(proof)
    nested1 = 0; nestbox = 0; maxd = 0
    stack = []
    for j, t in enumerate(toks):
        if t[0] == '(' and j + 1 < len(toks) and toks[j + 1][0] == 'fun' and toks[j + 1][1] == 'BOXOPEN':
            for s in stack: s['n'] = True
            if toks[j - 1][0] == ')' and last['n']: nested1 += 1
            stack.append({'n': False}); maxd = max(maxd, len(stack))
        if t[1] == 'PUNCT' and t[0] == ')' and t[3].startswith('CLOSE_') and (toks[j - 1][1] == 'CLOSEREF' or (t[3] == 'CLOSE_NEGI' and toks[j - 1][0] == ')')):
            last = stack.pop(); nestbox += last['n']
    return nested1, nestbox, maxd
tab = collections.defaultdict(lambda: [0, 0, 0])
for r in rs:
    n1, nb, md = feats(r['proof'])
    fam = r['src'].split('.')[0]
    key = (r['band'], fam, 'NEST1' if n1 else ('nestbox' if nb else 'flat'))
    t = tab[key]; t[0] += 1; t[1] += r['name'] in s0; t[2] += r['name'] in anyr
for k in sorted(tab):
    n, a, b = tab[k]
    print(k, n, 'untied_s0 %.3f' % (a / n), 'any-of-4 %.3f' % (b / n))
