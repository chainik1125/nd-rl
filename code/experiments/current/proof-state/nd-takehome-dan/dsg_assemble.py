#!/usr/bin/env python3
"""Assemble a ds-generator pretraining set from a merged raw pool (proposal 9 protocol; the control's recipe with `--simple`
semantics: flat 31,000 per pruned length 2-6, depth-3 proofs excluded in pruned AND written form, uniform draw from the rest,
so every other pattern sits at the pool's natural rate).

  python3 dsg_assemble.py --pool data/dsg/pool_g1.jsonl --out data/dsg/train_g1.jsonl --report data/dsg/assemble_g1.json \
      --exclude data/p2/heldout.jsonl data/p2/targets_depth3.jsonl ... [--fill data/dsg/pool_g1.jsonl] [--per_len 31000] [--seed 0]

Exclusions are by renaming class (`key`) and by `thm`; validation-36 classes are always excluded. `--fill`: a second pool
(G1's) used only for length bins the main pool cannot fill; the fraction per bin is reported and each record carries `src`.
Every written record is re-verified with nd_verify (assert ok, line count == n_lines <= 6) and re-classified (assert no depth-3).
"""
import argparse, json, os, sys, random, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nd_verify import verify_text
from patterns import classify, PATTERNS
from gen import canon_key

LENS = (2, 3, 4, 5, 6)


def load_excl(files):
    keys, thms = set(), set()
    for fn in files:
        for l in open(fn):
            if not l.strip():
                continue
            r = json.loads(l)
            thm = r['thm'].strip()
            thms.add(thm); keys.add(r.get('key') or canon_key(thm))
    for l in open('targets/validation_36.jsonl'):
        thm = json.loads(l)['thm'].strip(); thms.add(thm); keys.add(canon_key(thm))
    return keys, thms


def read_pool(fn, keys, thms, stats, src):
    by = collections.defaultdict(list)
    for l in open(fn):
        if not l.strip():
            continue
        r = json.loads(l)
        stats[src + ':read'] += 1
        if r['n_lines'] < LENS[0] or r['n_lines'] > LENS[-1]:
            stats[src + ':len_out'] += 1; continue
        if r['key'] in keys or r['thm'].strip() in thms:
            stats[src + ':excluded'] += 1; continue
        cl = classify(r['proof']); clw = classify(r['proof'], pruned=False)
        if cl['depth3'] or clw['depth3']:
            stats[src + ':depth3'] += 1; continue
        r['pat'] = {p: bool(cl[p]) for p in PATTERNS}; r['pat']['derived_ore_strict'] = bool(cl['derived_ore_strict'])
        r['src'] = src
        by[r['n_lines']].append(r)
    return by


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pool', required=True); ap.add_argument('--out', required=True); ap.add_argument('--report', required=True)
    ap.add_argument('--exclude', nargs='*', default=[]); ap.add_argument('--fill', default=None)
    ap.add_argument('--per_len', type=int, default=31000); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--prefix', default='train')
    a = ap.parse_args()
    rng = random.Random(a.seed)
    stats = collections.Counter()
    keys, thms = load_excl(a.exclude)
    print(f'excluding {len(keys)} classes / {len(thms)} theorems from {len(a.exclude)} files + validation-36', flush=True)
    by = read_pool(a.pool, keys, thms, stats, 'main')
    print('main pool usable per length', {L: len(by[L]) for L in LENS}, dict(stats), flush=True)
    fill_by = None
    short = [L for L in LENS if len(by[L]) < a.per_len]
    if short and a.fill:
        fill_by = read_pool(a.fill, keys, thms, stats, 'fill')
        print('fill pool usable per length', {L: len(fill_by[L]) for L in LENS}, flush=True)
    sel = []; per = {}
    seen = set()
    for L in LENS:
        rows = by[L]; rng.shuffle(rows)
        take = [r for r in rows if r['key'] not in seen][:a.per_len]
        for r in take: seen.add(r['key'])
        n_main = len(take)
        if len(take) < a.per_len:
            if fill_by is None:
                sys.exit(f'length {L}: only {len(take)} proofs for quota {a.per_len} and no --fill pool')
            frows = [r for r in fill_by[L] if r['key'] not in seen]; rng.shuffle(frows)
            add = frows[:a.per_len - len(take)]
            if len(add) < a.per_len - len(take):
                sys.exit(f'length {L}: main {n_main} + fill {len(add)} < quota {a.per_len}')
            for r in add: seen.add(r['key'])
            take += add
        per[L] = {'n': len(take), 'main': n_main, 'fill': len(take) - n_main, 'fill_frac': (len(take) - n_main) / len(take)}
        sel += take
    rng.shuffle(sel)
    assert len({r['key'] for r in sel}) == len(sel), 'duplicate class in set'
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    cnt = collections.Counter(); rules = collections.Counter(); fill_total = 0
    with open(a.out, 'w') as f:
        for i, r in enumerate(sel):
            ok, reason, nl = verify_text(r['prompt'] + ' ' + r['proof'])
            assert ok and nl == r['n_lines'] <= LENS[-1], (reason, r['thm'])
            assert not classify(r['proof'])['depth3'] and not classify(r['proof'], pruned=False)['depth3']
            assert r['key'] not in keys and r['thm'].strip() not in thms
            rec = {'name': f'{a.prefix}_{i}', 'thm': r['thm'], 'key': r['key'], 'prompt': r['prompt'], 'proof': r['proof'],
                   'text': r['prompt'] + ' ' + r['proof'], 'n_lines': r['n_lines'], 'rules': r['rules'], 'n_prem': r['n_prem'], 'pat': r['pat'],
                   'gen_last_rule': r.get('gen_last_rule'), 'n_lazy_prem': r.get('n_lazy_prem'), 'contra_prem': r.get('contra_prem'), 'src': r['src']}
            f.write(json.dumps(rec) + '\n')
            for p in r['pat']:
                cnt[p] += r['pat'][p]
            fill_total += r['src'] == 'fill'
    rep = {'n': len(sel), 'per_len': per, 'fill_total': fill_total, 'fill_frac': fill_total / len(sel), 'pattern_counts': dict(cnt),
           'pattern_freq': {p: cnt[p] / len(sel) for p in cnt}, 'pool': a.pool, 'fill_pool': a.fill, 'exclude': a.exclude, 'seed': a.seed,
           'stats': dict(stats), 'n_excluded_classes': len(keys)}
    json.dump(rep, open(a.report, 'w'), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    main()
