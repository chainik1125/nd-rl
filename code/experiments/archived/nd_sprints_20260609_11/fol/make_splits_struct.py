"""Splits for the eigenvariable structural-wall experiment."""
import json, random, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
def main(pool, outdir):
    rng = random.Random(11)
    recs = [json.loads(l) for l in open(pool) if l.strip()]
    recs = [r for r in recs if 2 <= r['n_lines'] <= 16 and r.get('quant')]
    seen=set(); ded=[]
    for r in recs:
        if r['thm'] in seen: continue
        seen.add(r['thm']); ded.append(r)
    rng.shuffle(ded)
    def is_eig(r): return any(x in (r.get('rules') or []) for x in ('ALLI','EXE'))
    eig=[r for r in ded if is_eig(r)]; noeig=[r for r in ded if not is_eig(r)]
    # held-out eval: 800 eigen + 800 noeigen (length-spread by sampling)
    eval_eig=eig[:800]; eval_noeig=noeig[:800]
    heldout={r['thm'] for r in eval_eig}|{r['thm'] for r in eval_noeig}
    train=[r for r in ded if r['thm'] not in heldout]
    for name,rs in [('fol_struct_train.jsonl',train),('fol_eval_eigen.jsonl',eval_eig),('fol_eval_noeigen.jsonl',eval_noeig)]:
        with open(os.path.join(outdir,name),'w') as f:
            for r in rs: f.write(json.dumps(r)+'\n')
        from collections import Counter
        print(name, len(rs), 'len', dict(sorted(Counter(r['n_lines'] for r in rs).items())))
if __name__=='__main__':
    main(sys.argv[1], sys.argv[2] if len(sys.argv)>2 else 'data')
