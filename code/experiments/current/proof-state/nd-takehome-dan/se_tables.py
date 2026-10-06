#!/usr/bin/env python3
"""Markdown tables for `numbers.md` § state-env, from artifacts/se/summary.json (+ the held-out summaries).

  python3 se_tables.py --summary artifacts/se/summary.json --out artifacts/se/tables.md
"""
import argparse, glob, json, os


def f(x, n=3):
    return '—' if x is None else (f'{x:.{n}f}' if isinstance(x, float) else str(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--summary', default='artifacts/se/summary.json')
    ap.add_argument('--heldout_glob', default='artifacts/se/heldout_*.json')
    ap.add_argument('--out', default='artifacts/se/tables.md')
    a = ap.parse_args()
    d = json.load(open(a.summary))
    d = d.get('ladders', d)          # artifacts/se/summary.json nests the ladder summary
    L = []
    # --- held-out greedy
    L.append('### Held-out greedy (`data/p2/heldout.jsonl`, 5,000, k 1, T 0; per length)\n')
    L.append('| model | overall | 2 | 3 | 4 | 5 | 6 | source |')
    L.append('|---|---|---|---|---|---|---|---|')
    for fn in sorted(glob.glob(a.heldout_glob)):
        h = json.load(open(fn))
        if 'by_len' not in h:          # heldout_depth3.json is a different table
            continue
        b = h['by_len']
        lab = os.path.basename(fn)[8:-5] + ' `' + os.path.basename(h['ckpt']) + '`'
        L.append(f"| {lab} | {h['rate']:.4f} | " +
                 ' | '.join(f"{b[str(k)]['rate']:.3f}" if str(k) in b else '—' for k in range(2, 7)) +
                 f" | `{fn}` |")
    L.append('| C0 s0 `stage1_a1_seq_s0.pt` (on file, Lean ∧ nd_verify) | 0.9088 | 0.994 | 0.989 | 0.951 | 0.924 | 0.686 | `review_ds-generator.md` §3 |')
    L.append('| C0 s1 `stage1_a1_seq_s1.pt` (on file, Lean ∧ nd_verify) | 0.8968 | 0.997 | 0.992 | 0.967 | 0.944 | 0.584 | same |')
    # --- ladder
    L.append('\n### Ladder: transfer pool (2,285 theorems), cumulative at the last round\n')
    L.append('| run | last round | solved | `L*` | ≥13 | greedy | textbook | distinct proofs | source |')
    L.append('|---|---|---|---|---|---|---|---|---|')
    for grp, tag in (('runs', ''), ('c0', ' (C0, on file)')):
        for k, o in sorted(d[grp].items()):
            de = o['derived']; lr = o['rounds'][str(o['last_round'])]
            L.append(f"| `{k}`{tag} | {o['last_round']} | {de['solved']} | {de['lstar']} | {de['solved_ge13']} | "
                     f"{lr['transfer_greedy_solved']} | {de['textbook']['solved']}/{de['textbook']['n']} "
                     f"({de['textbook']['schemata_solved']}/{de['textbook']['schemata']} schemata) | "
                     f"{de['distinct_proofs']} | `{o['dir']}/found_transfer_{o['last_round']}.jsonl` |")
    # --- by L_true
    L.append('\n### Transfer solved by `L_true` (ND-derived upper bound under Lean)\n')
    Ls = [str(x) for x in range(7, 15)]
    L.append('| run | ' + ' | '.join(Ls) + ' |')
    L.append('|---' * (len(Ls) + 1) + '|')
    for grp in ('runs', 'c0'):
        for k, o in sorted(d[grp].items()):
            bb = o['derived']['by_bin']
            L.append(f'| `{k}` | ' + ' | '.join(f"{bb[x]['solved']}/{bb[x]['n']}" if x in bb else '—' for x in Ls) + ' |')
    # --- env diagnostics
    L.append('\n### Environment per-step diagnostics (last round of each run)\n')
    L.append('| run | attempts | finished | syntactic error | Lean rejected | truncated | step cap | mean steps/attempt | action tokens (mean) | peak alloc GB |')
    L.append('|---|---|---|---|---|---|---|---|---|---|')
    for k, o in sorted(d['runs'].items()):
        lr = o['rounds'][str(o['last_round'])]
        e = lr.get('env') or {}
        ef = e.get('end_frac') or {}
        L.append(f"| `{k}` | {lr.get('attempts')} | {100*ef.get('done',0):.1f} % | {100*ef.get('syntax',0):.1f} % | "
                 f"{100*e.get('ended_by_lean_reject_frac',0):.1f} % | {100*ef.get('truncated',0):.3f} % | "
                 f"{100*ef.get('step_cap',0):.3f} % | {e.get('mean_steps_per_attempt',0):.2f} | "
                 f"{e.get('action_declen_mean',0):.1f} | {e.get('peak_alloc_gb',0):.2f} |")
    # --- term size
    if any('term_size' in o for o in d['runs'].values()):
        L.append('\n### Proof length: lines **and** elaborated Lean term size (`lean_check`), shortest accepted proof per theorem\n')
        L.append('| run | `L_true` | n | lines min/median/max | term size min/median/max |')
        L.append('|---|---|---|---|---|')
        for grp in ('runs', 'c0'):
            for k, o in sorted(d[grp].items()):
                ts = o.get('term_size')
                if not ts:
                    continue
                for Lt, v in sorted(ts['by_L_true'].items(), key=lambda x: int(x[0])):
                    L.append(f"| `{k}` | {Lt} | {v['n']} | {v['lines_min']}/{v['lines_median']}/{v['lines_max']} | "
                             f"{v['term_size_min']}/{v['term_size_median']}/{v['term_size_max']} |")
    # --- acquisition
    L.append('\n### Transfer solved by round (cumulative)\n')
    L.append('| run | ' + ' | '.join(str(r) for r in range(1, 9)) + ' |')
    L.append('|---' * 9 + '|')
    for grp in ('runs', 'c0'):
        for k, o in sorted(d[grp].items()):
            L.append(f'| `{k}` | ' + ' | '.join(
                str(o['rounds'][str(r)]['transfer_cum_reported']['solved']) if str(r) in o['rounds'] else '—'
                for r in range(1, 9)) + ' |')
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    open(a.out, 'w').write('\n'.join(L) + '\n')
    print('wrote', a.out, len(L), 'lines')


if __name__ == '__main__':
    main()
