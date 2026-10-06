#!/usr/bin/env python3
"""Reviewer's own pool-shape recount (noise-floor premise check, E11).
Box depth, rule shares and ORE share re-derived from the proof text of all 4 x 155,000
training records, with my own parser.  Usage: python3 review_nf_shape.py <repo-root>
"""
import json, re, sys, collections

RULES = ('PR', 'AS', 'R', 'ANDI', 'ANDE1', 'ANDE2', 'ORI1', 'ORI2', 'ORE',
         'IMPI', 'IMPE', 'NEGI', 'NEGE', 'DN', 'BOTE')

def shape(path):
    n = 0
    depth = collections.Counter(); rule_any = collections.Counter()
    nlines = collections.Counter(); ore = 0
    for l in open(path):
        d = json.loads(l); n += 1
        segs = [s.strip() for s in d['proof'].split(';') if s.strip() and s.strip() != 'QED']
        md = 0; rs = set()
        for s in segs:
            m = re.match(r'N\d+\s+((?:\|\s*)*)', s)
            md = max(md, m.group(1).count('|') if m else 0)
            tail = s.rsplit(':', 1)[1].strip().split()
            if tail and tail[0] in RULES:
                rs.add(tail[0])
        depth[md] += 1
        for r in rs:
            rule_any[r] += 1
        if 'ORE' in rs:
            ore += 1
        nlines[d.get('n_lines', len(segs))] += 1
    return {'n': n,
            'depth_pct': {k: round(100 * v / n, 2) for k, v in sorted(depth.items())},
            'ore_pct': round(100 * ore / n, 3),
            'rule_any_pct': {k: round(100 * v / n, 2) for k, v in sorted(rule_any.items())},
            'len_hist': dict(sorted(nlines.items()))}

if __name__ == '__main__':
    R = sys.argv[1] if len(sys.argv) > 1 else '/home/dan/review/noise-floor'
    out = {}
    for i in (1, 2, 3, 4):
        out[f'p{i}'] = shape(f'{R}/data/nf/train_p{i}.jsonl')
        print(f'p{i}', json.dumps(out[f'p{i}']), flush=True)
    for k in (0, 1, 2, 3):
        v = [out[f'p{i}']['depth_pct'].get(k, 0.0) for i in (1, 2, 3, 4)]
        print('box depth', k, v, 'spread', round(max(v) - min(v), 3))
    v = [out[f'p{i}']['ore_pct'] for i in (1, 2, 3, 4)]
    print('ORE %', v, 'spread', round(max(v) - min(v), 3))
