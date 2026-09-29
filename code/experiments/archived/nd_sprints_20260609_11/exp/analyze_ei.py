"""Post-hoc analysis of expert-iteration runs.

For each round of an ei run directory:
  - discovered proof length histogram (do discoveries actually exceed L_train?)
  - per-length-bin transfer solve rates (where do the gains come from?)
  - textbook targets solved by name

Usage: python -m exp.analyze_ei --run /root/nd/runs/ei_rl --out /root/nd/results/ei_rl_analysis.json
"""
import argparse
import json
import os
from collections import Counter, defaultdict


def load_jsonl(p):
    out = []
    with open(p) as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


RULE_NAMES = {'ANDI', 'ANDE1', 'ANDE2', 'IMPE', 'IMPI', 'ORI1', 'ORI2', 'ORE',
              'NEGE', 'NEGI', 'BOTE', 'DN'}
DISCHARGING = {'IMPI', 'NEGI', 'ORE'}


def proof_stats(text):
    """Parse a full example text; return (rules_used, max_depth, n_lines) of the
    proof body."""
    toks = text.split()
    body = toks[toks.index('PRF') + 1:]
    rules = set()
    max_depth = 0
    n_lines = 0
    depth = 0
    in_line = False
    for i, t in enumerate(body):
        if t == ';':
            n_lines += 1
        if t == ':' and i + 1 < len(body) and body[i + 1] in RULE_NAMES:
            rules.add(body[i + 1])
        if t.startswith('N') and t[1:].isdigit() and (i == 0 or body[i - 1] == ';'):
            depth = 0
            j = i + 1
            while j < len(body) and body[j] == '|':
                depth += 1
                j += 1
            max_depth = max(max_depth, depth)
    return rules, max_depth, n_lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--baseline', default=None,
                    help='training jsonl; computes depth/len hist for comparison')
    args = ap.parse_args()

    analysis = {'rounds': []}
    if args.baseline:
        base = load_jsonl(args.baseline)[:20000]
        depths, lens = [], []
        for d in base:
            _, md, nl = proof_stats(d['text'])
            depths.append(md)
            lens.append(nl)
        analysis['baseline_depth_hist'] = dict(sorted(Counter(depths).items()))
        analysis['baseline_len_hist'] = dict(sorted(Counter(lens).items()))
    rnd = 0
    while os.path.isdir(os.path.join(args.run, f'round{rnd}')):
        rdir = os.path.join(args.run, f'round{rnd}')
        rec = {'round': rnd}
        dpath = os.path.join(rdir, 'discoveries.jsonl')
        if os.path.exists(dpath):
            disc = load_jsonl(dpath)
            # length of the proof the MODEL found (best_len was stored as n_lines)
            rec['n_discoveries'] = len(disc)
            rec['discovered_len_hist'] = dict(sorted(Counter(
                d['n_lines'] for d in disc).items()))
            rec['discovered_over_8'] = sum(1 for d in disc if d['n_lines'] > 8)
            rec['discovered_over_12'] = sum(1 for d in disc if d['n_lines'] > 12)
            depths = []
            n_discharge = 0
            for d in disc:
                rules, md, _ = proof_stats(d['text'])
                depths.append(md)
                if rules & DISCHARGING:
                    n_discharge += 1
            rec['discovered_depth_hist'] = dict(sorted(Counter(depths).items()))
            rec['discovered_with_discharge'] = n_discharge
        for tag in ['pass1', 'pass8']:
            p = os.path.join(rdir, f'eval_transfer_{tag}.jsonl')
            if os.path.exists(p):
                rs = load_jsonl(p)
                bybin = defaultdict(lambda: [0, 0])
                for r in rs:
                    b = r['n_lines']
                    bybin[b][0] += r['solved']
                    bybin[b][1] += 1
                rec[f'transfer_{tag}_by_len'] = {
                    k: [v[0], v[1]] for k, v in sorted(bybin.items())}
        p = os.path.join(rdir, 'eval_textbook_pass16.jsonl')
        if os.path.exists(p):
            rs = load_jsonl(p)
            rec['textbook_solved'] = sorted(r['name'] for r in rs if r['solved'])
            rec['textbook_best_lens'] = {r['name']: r['best_len']
                                         for r in rs if r['solved']}
        analysis['rounds'].append(rec)
        rnd += 1
    with open(args.out, 'w') as f:
        json.dump(analysis, f, indent=1)
    print(json.dumps(analysis, indent=1)[:3000])


if __name__ == '__main__':
    main()
