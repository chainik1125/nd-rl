"""Eigenvariable structural-wall forensics.

For each target: sample k proofs, record solve and the verifier's first-error
category per failed attempt. Targets are split by whether their reference proof
uses an eigenvariable rule (ALLI/EXE). Headline questions:
  (a) do eigenvariable-requiring targets generalize worse (structural wall)?
  (b) among the model's FAILURES on eigenvariable targets, what fraction are
      eigenvariable-condition VIOLATIONS (allI-eigenvar/exE-eigenvar) — direct
      evidence the model does not track parameter freshness?
"""
import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from fol.tokenizer import STOI, ITOS, PAD
from fol.verify import verify_text
from fol.run_fol import load_model

QED = STOI['QED']


def category(reason):
    r = reason
    if r.startswith('allI-eigenvar') or r.startswith('exE-eigenvar'):
        return 'EIGENVAR-VIOLATION'
    if r.startswith(('allI-', 'exE-')):
        return 'quantifier-misuse'
    if r.startswith('parse'):
        return 'malformed'
    if 'cite' in r or 'ref range' in r or 'box' in r:
        return 'citation'
    if r.startswith('rule check'):
        return 'wrong-step'
    if 'premise' in r or 'final' in r or 'index' in r or 'depth' in r:
        return 'structural'
    return 'other'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--targets', required=True)
    ap.add_argument('--k', type=int, default=8)
    ap.add_argument('--temp', type=float, default=0.8)
    ap.add_argument('--bs', type=int, default=1024)
    ap.add_argument('--max-new', type=int, default=480)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(0)
    model = load_model(a.ckpt, device)
    recs = [json.loads(l) for l in open(a.targets) if l.strip()]
    if a.limit:
        recs = recs[:a.limit]

    jobs = []
    for i, r in enumerate(recs):
        toks = r['text'].split()
        pids = [STOI[t] for t in toks[:toks.index('PRF') + 1]]
        for _ in range(a.k):
            jobs.append((i, pids))
    order = sorted(range(len(jobs)), key=lambda j: len(jobs[j][1]))
    per = [{'solved': False, 'cats': defaultdict(int), 'n': 0} for _ in recs]
    for s in range(0, len(order), a.bs):
        chunk = [jobs[j] for j in order[s:s + a.bs]]
        outs = model.sample([c[1] for c in chunk], max_new=a.max_new,
                            temperature=a.temp, top_k=None, eos_id=QED,
                            pad_id=PAD, device=device)
        for (ri, pid), o in zip(chunk, outs):
            full = ' '.join(ITOS[t] for t in pid + o)
            ok, reason, _ = verify_text(full)
            per[ri]['n'] += 1
            if ok:
                per[ri]['solved'] = True
            else:
                per[ri]['cats'][category(reason)] += 1

    rows = []
    for r, p in zip(recs, per):
        eig = any(x in (r.get('rules') or []) for x in ('ALLI', 'EXE'))
        rows.append({'thm': r['thm'], 'n_lines': r['n_lines'], 'eigen': eig,
                     'rules': r.get('rules'), 'solved': p['solved'],
                     'cats': dict(p['cats']), 'n_attempts': p['n']})
    with open(a.out, 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')

    # report
    for grp, lab in [(True, 'EIGEN (uses ALLI/EXE)'), (False, 'NO-EIGEN')]:
        g = [r for r in rows if r['eigen'] == grp]
        if not g:
            continue
        solved = sum(r['solved'] for r in g)
        fails = [r for r in g if not r['solved']]
        cat = defaultdict(int)
        tot = 0
        for r in fails:
            for k, v in r['cats'].items():
                cat[k] += v
                tot += v
        print(f'{lab}: {len(g)} targets, solve {solved/len(g):.3f}')
        if tot:
            print('  failed-attempt first-error mix:',
                  {k: f'{v/tot:.2f}' for k, v in sorted(cat.items(), key=lambda x: -x[1])})
    # eigenvar-violation rate among eigen-target failed attempts
    eg = [r for r in rows if r['eigen']]
    ev = sum(r['cats'].get('EIGENVAR-VIOLATION', 0) for r in eg)
    allf = sum(sum(r['cats'].values()) for r in eg)
    if allf:
        print(f'EIGENVAR-VIOLATION = {ev}/{allf} = {ev/allf:.3f} of eigen-target failed attempts')


if __name__ == '__main__':
    main()
