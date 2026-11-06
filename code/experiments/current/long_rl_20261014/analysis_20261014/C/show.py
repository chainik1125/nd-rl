import pickle, json, numpy as np, sys, os
os.environ.setdefault('ND_ARCH', 'untied'); sys.path.insert(0, '..')
import ndmodel as M; import pretrain as P
import annot
tok = P.make_tokenizer(P.TOK_MODE)
S = pickle.load(open('scores.pkl', 'rb'))
lab = json.load(open('solved_sets.json')); s0 = set(lab['untied-ei_s0']['rl']) | set(lab['untied-ei_s0']['dev'])
def show(i, ck='final', th=np.log(0.2)):
    r = S['recs'][i]; toks, _ = annot.annotate(r['proof']); g, am, _ = S[ck][i]
    out = []
    for j, t in enumerate(toks):
        if t[1] == 'PREM': continue
        s = t[0]
        if g[j] < th: s = f'[[{t[0]} p={np.exp(g[j]):.2f} want {tok.itos[int(am[j])]}]]'
        out.append(s)
    print(r['name'], r['band'], r['src'], 'body', r['body'], 'solved' if r['name'] in s0 else 'UNSOLVED', ck)
    print(r['prompt']); print(' '.join(out)); print()
if __name__ == '__main__':
    for a in sys.argv[1:]: show(int(a))
