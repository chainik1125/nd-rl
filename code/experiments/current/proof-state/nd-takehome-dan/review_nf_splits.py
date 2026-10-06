#!/usr/bin/env python3
"""Reviewer's own renaming-class normaliser + split-disjointness matrix.

class(thm) = the theorem string with atoms renamed to A0,A1,... in order of first
appearance (left to right over the whole sequent).  Variant 2 additionally sorts the
premise list, so a training theorem that differs from an eval theorem only by premise
order collapses onto it.
"""
import json, re, sys, collections

ATOM = re.compile(r'\b([A-Z])\b')

def rename(s):
    order = []
    for m in ATOM.finditer(s):
        a = m.group(1)
        if a == 'F':
            continue
        if a not in order:
            order.append(a)
    mp = {a: f'X{i}' for i, a in enumerate(order)}
    return ATOM.sub(lambda m: mp.get(m.group(1), m.group(1)), s)

def split_sequent(thm):
    prem, _, conc = thm.partition('|-')
    # premises are separated by top-level commas
    parts, depth, cur = [], 0, ''
    for ch in prem:
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
        if ch == ',' and depth == 0:
            parts.append(cur.strip()); cur = ''
        else:
            cur += ch
    if cur.strip():
        parts.append(cur.strip())
    return parts, conc.strip()

def cls(thm):
    return rename(' '.join(thm.split()))

def cls_permfree(thm):
    prem, conc = split_sequent(' '.join(thm.split()))
    r = rename(' '.join(thm.split()))
    rp, rc = split_sequent(r)
    return ' , '.join(sorted(rp)) + ' |- ' + rc

def load(fn, field='thm'):
    out = []
    for l in open(fn):
        if l.strip():
            d = json.loads(l)
            out.append(d[field])
    return out

if __name__ == '__main__':
    R = '/home/dan/review/noise-floor'
    trains = [f'{R}/data/nf/train_p{i}.jsonl' for i in (1, 2, 3, 4)]
    evals = [
        'data/p2/heldout.jsonl', 'data/ladder/rl_targets.jsonl', 'data/ladder/transfer.jsonl',
        'data/p2/targets_reductio_req.jsonl', 'data/r3_1/depth3_req.jsonl',
        'data/p2/targets_depth3.jsonl', 'data/p2/transfer_depth3.jsonl',
        'data/p2/transfer_reductio_req.jsonl', 'data/r3_1/depth3_req_transfer.jsonl',
    ]
    ev = {}
    for e in evals:
        ts = load(f'{R}/{e}')
        ev[e] = (set(ts), {cls(t) for t in ts}, {cls_permfree(t) for t in ts}, len(ts))
        print(f'{e}: n={len(ts)} distinct_class={len(ev[e][1])} distinct_permfree={len(ev[e][2])}', flush=True)
    print()
    for tr in trains:
        raw = set(); c = set(); pf = set(); n = 0
        for l in open(tr):
            d = json.loads(l)
            t = ' '.join(d['thm'].split())
            raw.add(t); c.add(cls(t)); pf.add(cls_permfree(t)); n += 1
        print(f'--- {tr}: records {n} distinct_thm {len(raw)} distinct_class {len(c)} distinct_permfree {len(pf)}', flush=True)
        for e, (eraw, ec, epf, en) in ev.items():
            print(f'    vs {e:42s} n={en:5d} thm={len(raw & eraw):4d} class={len(c & ec):4d} permfree={len(pf & epf):4d}')
