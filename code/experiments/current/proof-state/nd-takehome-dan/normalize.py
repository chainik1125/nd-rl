#!/usr/bin/env python3
"""Recount distinct verified proofs with the line-index start offset normalised.

The `abs` tokeniser is trained with a random start index, so the model writes the same proof
starting at N12 or N13; expert_iter.py / eval_set.py counted those as distinct. This script
re-derives every distinct-proof statistic from the saved proofs after renumbering each proof
from N1, and writes:
  artifacts/<arm>_norm/round_<r>.json   round files with normalised hists / frontiers
  artifacts/<arm>_all_norm/             merged s0 (rounds 1-8) + s0_cont (9-16)
  artifacts/<name>_norm.json            for eval_set outputs (pass@k files)
  artifacts/normalized_summary.md       the table used in the writeup
Solve rates are per theorem and unchanged; only proof counts, histograms, frontiers and
padding fractions change.
"""
import json, re, os, sys, glob, copy, collections

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prune import pruned_length

ARMS = ['ei_abs_s0', 'ei_abs_s0_cont', 'frozen_abs_s0', 'frozen_abs_s0_cont', 'ei_abs_s1', 'frozen_abs_s1', 'ei_abs_long_s0']
MERGE = {'ei_abs_s0_all_norm': ['ei_abs_s0', 'ei_abs_s0_cont'], 'frozen_abs_s0_all_norm': ['frozen_abs_s0', 'frozen_abs_s0_cont']}
EVALS = ['final_transfer_k128', 'stage1_transfer_k128', 'stage1_abs_transfer2_k16', 'stage1_rel_transfer2_k16',
         'stage1_absfixed_transfer2_k16', 'stage1_abs_transfer_k16', 'stage1_rel_transfer_k16']


def norm(proof):
    out, base = [], None
    for t in proof.split():
        if re.fullmatch(r'N\d+', t):
            n = int(t[1:])
            base = n if base is None else base
            out.append(f'N{n - base + 1}')
        else:
            out.append(t)
    return ' '.join(out)


def rd(fn):
    return [json.loads(l) for l in open(fn) if l.strip()]


def stats(items):
    """items: list of (name, proof, written, pruned). Dedupe by (name, normalised proof)."""
    seen, wh, ph, thms = set(), collections.Counter(), collections.Counter(), collections.defaultdict(set)
    n_pad = gap = 0
    for name, proof, w, p in items:
        k = (name, norm(proof))
        if k in seen:
            continue
        seen.add(k)
        wh[w] += 1; ph[p] += 1; thms[w].add(name)
        n_pad += (w > p); gap += (w - p)
    n = len(seen)
    return {'distinct_proofs': n,
            'written_hist': dict(sorted(wh.items())), 'pruned_hist': dict(sorted(ph.items())),
            'frontier_written': max([L for L, c in wh.items() if c >= 5], default=0),
            'frontier_pruned': max([L for L, c in ph.items() if c >= 5], default=0),
            'ge': {str(L): sum(c for k, c in wh.items() if k >= L) for L in (7, 8, 9, 10)},
            'ge_pruned': {str(L): sum(c for k, c in ph.items() if k >= L) for L in (7, 8, 9, 10)},
            'theorems_at_written': {str(L): len(thms[L]) for L in sorted(thms) if L >= 7},
            'pad_frac': n_pad / n if n else 0, 'pad_gap': gap / n if n else 0}


def patch(stat, s):
    stat['raw_written_hist'] = stat.get('written_hist'); stat['raw_pruned_hist'] = stat.get('pruned_hist')
    stat['raw_frontier_written'] = stat.get('frontier_written'); stat['raw_frontier_pruned'] = stat.get('frontier_pruned')
    for k in ('written_hist', 'pruned_hist', 'frontier_written', 'frontier_pruned', 'distinct_proofs', 'ge', 'ge_pruned', 'theorems_at_written', 'pad_frac', 'pad_gap'):
        stat[k] = s[k]
    stat['normalized'] = True


def main():
    md = ['# Distinct-proof statistics with start-index normalisation\n',
          'Recounted from the saved `found_*.jsonl` files by `normalize.py`. "raw" = as originally counted (start-index variants counted separately).\n',
          '| arm | round | attempts | pool | distinct proofs raw → norm | written ≥8 raw → norm | ≥9 raw → norm | ≥10 raw → norm | theorems with an 8 / 9 / 10-line proof | frontier written/pruned (norm) | padded frac (norm) |',
          '|---|---:|---:|---|---|---|---|---|---|---|---:|']
    merged = collections.defaultdict(dict)
    for arm in ARMS:
        src = f'artifacts/{arm}'
        if not os.path.isdir(src):
            continue
        dst = f'{src}_norm'; os.makedirs(dst, exist_ok=True)
        for rf in sorted(glob.glob(f'{src}/round_*.json'), key=lambda f: int(f.split('_')[-1].split('.')[0])):
            r = json.load(open(rf)); rnd = r['round']
            for pool, fn, key in (('targets', f'{src}/found_{rnd}.jsonl', 'targets_cum'), ('transfer', f'{src}/found_transfer_{rnd}.jsonl', 'transfer_cum')):
                rows = rd(fn)
                items = [(x['name'], x['proof'], x['written'], x['pruned']) for x in rows]
                raw_n = len({(x['name'], x['proof']) for x in rows})
                raw_ge = {L: sum(1 for x in rows if x['written'] >= L) for L in (8, 9, 10)}
                s = stats(items)
                patch(r[key], s)
                if rnd in (4, 8, 16) or arm == 'ei_abs_long_s0' and rnd == 8:
                    ta = s['theorems_at_written']
                    md.append(f"| {arm} | {rnd} | {rnd * r['k']} | {pool} | {raw_n:,} → {s['distinct_proofs']:,} | {raw_ge[8]:,} → {s['ge']['8']} | {raw_ge[9]} → {s['ge']['9']} | {raw_ge[10]} → {s['ge']['10']} | {ta.get('8', 0)} / {ta.get('9', 0)} / {ta.get('10', 0)} | {s['frontier_written']} / {s['frontier_pruned']} | {s['pad_frac']:.2f} |")
            json.dump(r, open(f'{dst}/round_{rnd}.json', 'w'), indent=1)
            for mname, parts in MERGE.items():
                if arm in parts:
                    merged[mname][rnd] = r
    for mname, rs in merged.items():
        os.makedirs(f'artifacts/{mname}', exist_ok=True)
        for rnd, r in rs.items():
            json.dump(r, open(f'artifacts/{mname}/round_{rnd}.json', 'w'), indent=1)
    md.append('\n## pass@k evaluation files (eval_set.py outputs)\n')
    md.append('| file | theorems solved | distinct proofs raw → norm | written 7 / 8 / 9 / 10 (norm) | frontier written/pruned (norm) |')
    md.append('|---|---:|---|---|---|')
    for name in EVALS:
        fn = f'artifacts/{name}.jsonl'
        if not os.path.exists(fn):
            continue
        rows = rd(fn)
        items, raw_n = [], 0
        for x in rows:
            for p, w, pl in zip(x['proofs'], x['written_lens'], x['pruned_lens']):
                items.append((x['name'], p, w, pl)); raw_n += 1
        s = stats(items)
        s['solved'] = sum(x['solved'] for x in rows); s['n'] = len(rows); s['raw_distinct_proofs'] = raw_n
        json.dump(s, open(f'artifacts/{name}_norm.json', 'w'), indent=1)
        wh = s['written_hist']
        md.append(f"| {name} | {s['solved']}/{s['n']} | {raw_n:,} → {s['distinct_proofs']:,} | {wh.get(7, 0)} / {wh.get(8, 0)} / {wh.get(9, 0)} / {wh.get(10, 0)} | {s['frontier_written']} / {s['frontier_pruned']} |")
    open('artifacts/normalized_summary.md', 'w').write('\n'.join(md) + '\n')
    print('\n'.join(md))


if __name__ == '__main__':
    main()
