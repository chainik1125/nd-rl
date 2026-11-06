import json, collections, random, sys
import annot
def red_lines(proof):
    lines = annot.parse_proof_tokens(proof.split())
    close_of = {}
    for l in lines:
        if l['rule'] in ('IMPI', 'NEGI'): close_of[l['refs'][0]] = l['refs'][1]
        if l['rule'] == 'ORE': close_of[l['refs'][1]] = l['refs'][2]; close_of[l['refs'][3]] = l['refs'][4]
    n = red = 0
    for L in lines:
        if L['rule'] in ('PR', 'AS'): continue
        n += 1
        f = L['formula']
        for l in lines:
            i = l['idx']
            if i >= L['idx']: break
            if any(a <= i <= e < L['idx'] for a, e in close_of.items()): continue
            if l['formula'] == f:
                # a box conclusion restating its box's last line via R is required syntax; count only if not the last line of an open box
                red += 1; break
    return red, n
A = '/Users/robbie/repos/nd-rl/artifacts/long_rl_20261014/v2/long-rl-20261014-v2_untied-ei_s0/found_targets.jsonl'
rs = [json.loads(l) for l in open(A)]
random.seed(0)
by = collections.defaultdict(lambda: [0, 0])
for r in random.sample(rs, 6000):
    a, b = red_lines(r['proof']); k = r['round'] // 10
    by[k][0] += a; by[k][1] += b
print('found proofs: in-scope restated lines / body lines by round//10', {k: (round(a / b, 3), b) for k, (a, b) in sorted(by.items())})
recs = json.load(open('recs.json'))
a = b = 0
for r in random.sample(recs, 2000):
    x, y = red_lines(r['proof']); a += x; b += y
print('gold proofs', a / b)
