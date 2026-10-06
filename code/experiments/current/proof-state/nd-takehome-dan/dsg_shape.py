#!/usr/bin/env python3
"""Shape table of a proof set (run ds-generator, proposal 9 protocol).

  python3 dsg_shape.py <set.jsonl> [more sets...] [--out artifacts/dsg/shape_<tag>.json] [--lean]

Per set: n, length histogram (written = nd_verify line count), box-depth histogram (max nesting of AS lines), per-proof rule
shares (fraction of proofs containing each rule at least once), proofs with an ORE, proofs with a box inside an ORE branch
(an AS line strictly deeper than the enclosing ORE branch's AS line), premise count histogram, lazy-premise count (field
`n_lazy_prem` if the generator labelled it; else null), contradictory-premise share (a premise and its negation both premises),
generator last rule shares, mean ND tokens (prompt + proof) and, with --lean, mean Lean tokens (lean_tok, lean_seq).
Everything is computed from the record's `proof` and `prompt` strings, not from stored labels (the stored `pat` is reported
alongside for the three patterns, re-classified with patterns.classify)."""
import argparse, json, collections, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from patterns import classify, PATTERNS

RULES = ('PR', 'AS', 'R', 'ANDI', 'ANDE1', 'ANDE2', 'ORI1', 'ORI2', 'ORE', 'IMPI', 'IMPE', 'NEGI', 'NEGE', 'DN', 'BOTE')


def parse_lines(proof):
    """'N1 ... ; ... QED' -> list of dicts {idx, depth, formula, rule, refs}"""
    out = []
    for seg in proof.split(';'):
        seg = seg.strip()
        if not seg or seg == 'QED':
            continue
        toks = seg.split()
        idx = int(toks[0][1:]); i = 1; depth = 0
        while toks[i] == '|':
            depth += 1; i += 1
        j = toks.index(':')
        formula = ' '.join(toks[i:j]); rule = toks[j + 1]; refs = [int(t[1:]) for t in toks[j + 2:]]
        out.append({'idx': idx, 'depth': depth, 'f': formula, 'rule': rule, 'refs': refs})
    return out


def shape_one(rec):
    L = parse_lines(rec['proof'])
    by = {l['idx']: l for l in L}
    rules = {l['rule'] for l in L}
    box_depth = max((l['depth'] for l in L), default=0)
    n_prem = sum(1 for l in L if l['rule'] == 'PR')
    prem_f = [l['f'] for l in L if l['rule'] == 'PR']
    contra = any(f'( ~ {p} )' in prem_f for p in prem_f)
    # boxes inside ORE branches: ORE cites d, s1, e1, s2, e2; any AS line with s.idx < idx <= e.idx and depth > s.depth
    box_in_ore = 0; ore_branch_lines = 0; n_ore = 0
    for l in L:
        if l['rule'] == 'ORE' and len(l['refs']) == 5:
            n_ore += 1
            for s, e in ((l['refs'][1], l['refs'][2]), (l['refs'][3], l['refs'][4])):
                sd = by[s]['depth']
                ore_branch_lines += e - s + 1
                box_in_ore += sum(1 for m in L if s < m['idx'] <= e and m['rule'] == 'AS' and m['depth'] > sd)
    last = L[-1]
    return {'n_lines': len(L), 'box_depth': box_depth, 'rules': rules, 'n_prem': n_prem, 'contra': contra, 'n_ore': n_ore,
            'box_in_ore': box_in_ore, 'last_rule': last['rule'], 'nd_tokens': len(rec['prompt'].split()) + len(rec['proof'].split()),
            'lazy': rec.get('n_lazy_prem')}


def table(recs, lean=False):
    if lean:
        from lean_tok import LeanTokenizer
        tk = LeanTokenizer('lean_seq')
    n = len(recs)
    C = collections.Counter; len_h, bd_h, prem_h, rule_c, last_c, lazy_h, pat_c = C(), C(), C(), C(), C(), C(), C()
    n_ore = n_box_in_ore = n_contra = 0; nd_tok = 0; lean_tok = 0; lean_n = 0; lazy_known = 0; ore_lines = 0
    box_in_ore_by_len = C(); ore_by_len = C()
    for r in recs:
        s = shape_one(r)
        len_h[s['n_lines']] += 1; bd_h[s['box_depth']] += 1; prem_h[s['n_prem']] += 1; last_c[s['last_rule']] += 1
        for ru in s['rules']:
            rule_c[ru] += 1
        n_ore += s['n_ore'] > 0; ore_by_len[s['n_lines']] += s['n_ore'] > 0
        n_box_in_ore += s['box_in_ore'] > 0; box_in_ore_by_len[s['n_lines']] += s['box_in_ore'] > 0
        n_contra += s['contra']; nd_tok += s['nd_tokens']
        if s['lazy'] is not None:
            lazy_known += 1; lazy_h[s['lazy']] += 1
        cl = classify(r['proof'])
        for p in PATTERNS:
            pat_c[p] += bool(cl[p])
        if lean:
            lean_tok += len(tk.encode_prompt(r['prompt'])) + len(tk.encode_proof(r['proof'])); lean_n += 1
    out = {'n': n, 'len_hist': dict(sorted(len_h.items())), 'box_depth_hist': dict(sorted(bd_h.items())),
           'rule_share': {ru: rule_c[ru] / n for ru in RULES}, 'proofs_with_ore': n_ore, 'ore_share': n_ore / n,
           'proofs_with_box_in_ore': n_box_in_ore, 'box_in_ore_share': n_box_in_ore / n,
           'ore_by_len': dict(sorted(ore_by_len.items())), 'box_in_ore_by_len': dict(sorted(box_in_ore_by_len.items())),
           'n_prem_hist': dict(sorted(prem_h.items())), 'mean_n_prem': sum(k * v for k, v in prem_h.items()) / n,
           'lazy_prem_hist': dict(sorted(lazy_h.items())) if lazy_known else None,
           'mean_lazy_prem': (sum(k * v for k, v in lazy_h.items()) / lazy_known) if lazy_known else None,
           'lazy_labelled': lazy_known, 'contra_prem_share': n_contra / n, 'last_rule_share': {k: v / n for k, v in last_c.most_common()},
           'mean_nd_tokens': nd_tok / n, 'mean_lean_tokens': (lean_tok / lean_n) if lean_n else None,
           'pattern_counts': dict(pat_c), 'pattern_share': {p: pat_c[p] / n for p in PATTERNS}}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('sets', nargs='+'); ap.add_argument('--out', default=None); ap.add_argument('--lean', action='store_true')
    ap.add_argument('--limit', type=int, default=None)
    a = ap.parse_args()
    res = {}
    for fn in a.sets:
        recs = [json.loads(l) for l in open(fn) if l.strip()]
        if a.limit:
            recs = recs[:a.limit]
        t = table(recs, a.lean)
        res[fn] = t
        print(f'== {fn}: n {t["n"]} len {t["len_hist"]} box_depth {t["box_depth_hist"]}')
        print('   rule share ' + ' '.join(f'{ru} {100*t["rule_share"][ru]:.1f}' for ru in RULES))
        print(f'   ORE {t["proofs_with_ore"]} ({100*t["ore_share"]:.2f}%) box-in-ORE {t["proofs_with_box_in_ore"]} ({100*t["box_in_ore_share"]:.2f}%) '
              f'prem {t["n_prem_hist"]} mean {t["mean_n_prem"]:.2f} lazy mean {t["mean_lazy_prem"]} contra {100*t["contra_prem_share"]:.1f}% '
              f'ND tok {t["mean_nd_tokens"]:.1f} Lean tok {t["mean_lean_tokens"]} patterns {t["pattern_counts"]}')
        print('   last rule ' + ' '.join(f'{k} {100*v:.1f}' for k, v in t['last_rule_share'].items()))
    if a.out:
        os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
        json.dump(res, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
