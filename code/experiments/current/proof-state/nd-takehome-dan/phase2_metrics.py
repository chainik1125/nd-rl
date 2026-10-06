#!/usr/bin/env python3
"""Phase 2 per-arm, per-round metrics from expert_iter.py outputs (all start-index normalised).

  python phase2_metrics.py --arms artifacts/p2/ei_derived_ore_f0_s0 ... --pattern derived_ore --out artifacts/p2/metrics.json

Per arm and round r (cumulative over rounds 1..r, as expert_iter accumulates):
  targets_solved, transfer_solved (per theorem), targets_greedy, transfer_greedy, heldout_greedy (from round_r.json),
  acquisition_targets  : fraction of target theorems solved by >=1 written proof whose dependency-pruned form
                         CONTAINS the arm's pattern (patterns.classify on the model's proof, not the generating proof),
  acquisition_transfer : same on the transfer pool,
  n_pattern_proofs     : number of distinct normalised pattern-containing proofs (targets / transfer),
  first_round_pattern  : first round in which any pattern-containing proof appeared,
  other-pattern rates for reference.
"""
import argparse, json, os, sys, glob, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from normalize import norm
from patterns import classify, PATTERNS


def rd(fn):
    return [json.loads(l) for l in open(fn) if l.strip()]


def arm_metrics(arm, pattern, n_targets=None, n_transfer=None):
    rounds = sorted(int(f.split('_')[-1].split('.')[0]) for f in glob.glob(f'{arm}/round_*.json'))
    out = []
    cache = {}
    for r in rounds:
        st = json.load(open(f'{arm}/round_{r}.json'))
        row = {'round': r, 'attempts': r * st['k'],
               'targets_solved': st['targets_cum']['solved'], 'targets_n': st['targets_cum']['n'],
               'transfer_solved': st['transfer_cum']['solved'], 'transfer_n': st['transfer_cum']['n'],
               'transfer_greedy': st['transfer_greedy']['rate'], 'heldout_greedy': st['heldout_greedy']['rate'],
               'targets_round_rate': st['targets_round']['rate'], 'transfer_round_rate': st['transfer_round']['rate']}
        for pool, base in (('targets', 'found'), ('transfer', 'found_transfer')):
            fn = f'{arm}/{base}_{r}.jsonl'
            if not os.path.exists(fn):   # intermediate found files removed: use the last cumulative file, filtered by round
                last = max(rounds); fn = f'{arm}/{base}_{last}.jsonl'
            # the found files are written from lists that the training step shuffles in place, so the FIRST record of a
            # (theorem, normalised proof) pair is not necessarily its earliest round: take the minimum round per pair.
            recs = rd(fn); minround = {}
            for x in recs:
                k0 = (x['name'], norm(x['proof'])); minround[k0] = min(minround.get(k0, 99), x['round'])
            seen = set(); thm_pat = set(); n_pat = 0; first = None
            other = collections.Counter(); wh = collections.Counter(); ph = collections.Counter(); thms = set()
            examples = []; strict_thms = set(); loose_thms = set(); anydn_thms = set()
            for x in recs:
                pn = norm(x['proof'])
                k = (x['name'], pn)
                x = dict(x); x['round'] = minround[k]
                if x['round'] > r:
                    continue
                if k in seen:
                    continue
                seen.add(k); thms.add(x['name'])
                wh[x['written']] += 1; ph[x['pruned']] += 1
                if pn not in cache:
                    cache[pn] = classify(pn)
                cl = cache[pn]
                if cl is None:
                    continue
                for p in PATTERNS:
                    other[p] += cl[p]
                if pattern in ('derived_ore', 'derived_ore_strict') and cl.get('derived_ore_strict'):
                    other['derived_ore_strict'] += 1; strict_thms.add(x['name'])
                if pattern == 'reductio':
                    if cl.get('derived_dn'):
                        other['derived_dn'] += 1; loose_thms.add(x['name'])
                    if ' DN ' in pn:
                        other['any_dn'] += 1; anydn_thms.add(x['name'])
                if cl[pattern]:
                    n_pat += 1; thm_pat.add(x['name'])
                    if first is None or x['round'] < first:
                        first = x['round']
                    if len(examples) < 50:
                        examples.append({'name': x['name'], 'thm': x['thm'], 'proof': pn, 'written': x['written'], 'pruned': x['pruned'], 'round': x['round']})
            n = row[f'{pool}_n']
            row[f'acq_{pool}'] = len(thm_pat) / n
            row[f'acq_{pool}_theorems'] = len(thm_pat)
            row[f'n_pattern_proofs_{pool}'] = n_pat
            row[f'first_round_pattern_{pool}'] = first
            row[f'distinct_proofs_{pool}'] = len(seen)
            row[f'other_patterns_{pool}'] = dict(other)
            row[f'acq_strict_{pool}_theorems'] = len(strict_thms)
            row[f'acq_loose_{pool}_theorems'] = len(loose_thms)      # reductio arms: DN citing a rule-derived line
            row[f'acq_anydn_{pool}_theorems'] = len(anydn_thms)      # reductio arms: any DN at all (upper bound)
            row[f'solved_names_{pool}'] = sorted(thms)
            row[f'pattern_names_{pool}'] = sorted(thm_pat)
            row[f'loose_names_{pool}'] = sorted(loose_thms)
            row[f'written_hist_{pool}'] = dict(sorted(wh.items()))
            row[f'frontier_written_{pool}'] = max([L for L, c in wh.items() if c >= 5], default=0)
            row[f'frontier_pruned_{pool}'] = max([L for L, c in ph.items() if c >= 5], default=0)
            row[f'pattern_examples_{pool}'] = examples
        out.append(row)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arms', nargs='+', required=True)
    ap.add_argument('--pattern', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    res = {}
    for arm in a.arms:
        if not glob.glob(f'{arm}/round_*.json'):
            print('no rounds for', arm); continue
        m = arm_metrics(arm, a.pattern)
        res[os.path.basename(arm)] = m
        last = m[-1]
        print(f"{os.path.basename(arm)} r{last['round']}: targets {last['targets_solved']}/{last['targets_n']} acq {last['acq_targets']:.3f} "
              f"({last['acq_targets_theorems']} thms, {last['n_pattern_proofs_targets']} proofs, first round {last['first_round_pattern_targets']}); "
              f"transfer {last['transfer_solved']}/{last['transfer_n']} acq {last['acq_transfer']:.3f}; greedy T {last['transfer_greedy']:.3f} H {last['heldout_greedy']:.3f}")
    json.dump(res, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
