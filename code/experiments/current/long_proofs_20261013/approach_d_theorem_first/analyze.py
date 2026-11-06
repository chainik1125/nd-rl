"""Summarise certify.py output(s): cert time distribution, timeout rate, min_body / cert_ge / n_prem
distributions, throughput, and the fraction of proof-line formulas outside minlen's S.
    python analyze.py OUT.jsonl [...]
"""
import collections, json, os, statistics, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import certify as C
from minlen import Search, parse_thm

rs = [json.loads(l) for p in sys.argv[1:] for l in open(p)]
print(f'{len(rs)} records')
secs = sorted(r['cert_secs'] for r in rs)
q = lambda f: secs[min(len(secs) - 1, int(f * len(secs)))]
print(f'cert_secs: total {sum(secs):.0f}s  median {q(.5):.2f}  p75 {q(.75):.2f}  p90 {q(.9):.2f}  max {secs[-1]:.1f}')
print('timeouts:', sum(bool(r['timeout']) for r in rs), ' cert_ge>=floor:', sum(bool(r.get('cert_ge')) for r in rs))
print('min_body:', dict(sorted(collections.Counter(r['min_body'] for r in rs if r['min_body'] is not None).items())))
print('n_prem:', dict(sorted(collections.Counter(r['n_prem'] for r in rs).items())))
print('n_atoms:', dict(sorted(collections.Counter(r['n_atoms'] for r in rs).items())))
print('written n_body:', dict(sorted(collections.Counter(min(r['n_body'] // 10 * 10, 80) for r in rs).items())))
print('min_body by n_prem:', {k: dict(sorted(collections.Counter(r['min_body'] for r in rs if r['n_prem'] == k).items(), key=lambda x: (x[0] is None, x[0])))
                              for k in sorted(set(r['n_prem'] for r in rs))})
out_S = tot = 0
for r in rs:
    prem, concl = parse_thm(r['prompt'])
    S = Search(prem, concl).S
    lines = C.V.parse_proof_tokens(r['text'].split(' PRF ', 1)[1].split())
    for ln in lines:
        tot += 1
        out_S += ln['formula'] not in S
print(f'proof-line formulas outside S: {out_S}/{tot} = {out_S / max(tot, 1):.1%}')
