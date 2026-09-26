#!/usr/bin/env python3
"""Mid-training data: glued ANDI proofs (supervised) + long held-out test theorems.

Construction (same code as the ab3 ANDI family, tools in ab3/data_build/tools/comp.py): k parts (k = 2 or 3) from the
ab2 reserve (generator theorems with a verified minlen proof and satisfiable premises), renamed injectively onto
pairwise DISJOINT atom sets (k = 2: <= 2 + 2 atoms; k = 3: 1 + 1 + <= 2 atoms), premises merged and interleaved at
random, conclusion = a random binary ANDI tree over the part conclusions ((A & B) & C, A & (B & C), any order).
Proof = the parts' minimal proofs concatenated (references renumbered) + k - 1 ANDI lines; written length
sum(L_i) + k - 1. Every proof is nd_verify-checked.

Parts: every reserve part used in ab3/sealed (sealed_L15-16, sealed_L17-18, matched_L6-14) or ab3/data_comp is
excluded by name and by part pkey (pkey is invariant under atom renaming). The remaining parts are split by pkey,
stratified by (L, n_atoms), into a TEST half (test_long) and an SFT half (sft_long, sft_short), seed --split_seed.
Theorems: key / pkey not in any existing file (data_build/excl.json + ab3 sealed + data_comp), pkey-unique across the
three output files. k: 50/50 where both k are feasible at a length (--k3_share), else whichever is feasible.
Part reuse: each part is used at most --cap_* times per file, and draws are rejection-weighted by 1/(1+uses)^2.
Lengths are built longest-first (test_long, sft_long) and shortest-first (sft_short), where parts are scarcest.

  python tools/build_mid.py --seed 34001
"""
import json, sys, random, collections, argparse, glob
sys.path.insert(0, '/scratch/cowifornia/nd-rl/ab3/data_build/tools')
from comp import Part, check, thm_str, prompt_str, ATOMS, satisfiable, parse_proof, proof_text
from keys2 import pkey, canon_key
from gen import fstr
sys.path.insert(0, '/scratch/cowifornia/nd-rl/ab3/frozen/lib')
from tokenizer import Tokenizer
TOK = Tokenizer('abs')

DB = '/scratch/cowifornia/nd-rl/ab3/data_build'
OUT = '/scratch/cowifornia/nd-rl/ab3/midtrain/build/raw'
rd = lambda fn: [json.loads(l) for l in open(fn) if l.strip()]


def combine_tree(parts_r, rng):
    """parts_r: list of renamed parts (prem, concl, lines). Random part order, random binary ANDI tree."""
    parts_r = list(parts_r); rng.shuffle(parts_r)
    prem = []
    for p, _, _ in parts_r:
        for q in p:
            if q not in prem:
                prem.append(q)
    rng.shuffle(prem)
    pidx = {p: i + 1 for i, p in enumerate(prem)}
    lines = [(0, p, 'PR', []) for p in prem]
    nodes = []                                   # (formula, line index)
    for pp, cc, ll in parts_r:
        m = {i + 1: pidx[p] for i, p in enumerate(pp)}
        for k, (d, f, r, refs) in enumerate(ll[len(pp):]):
            lines.append((d, f, r, [m[x] for x in refs])); m[len(pp) + k + 1] = len(lines)
        nodes.append((cc, len(lines)))
    while len(nodes) > 1:                        # join two random adjacent-or-not nodes, random order
        i, j = rng.sample(range(len(nodes)), 2)
        a, b = nodes[i], nodes[j]
        f = ('and', a[0], b[0])
        lines.append((0, f, 'ANDI', [a[1], b[1]]))
        nodes = [n for t, n in enumerate(nodes) if t not in (i, j)] + [(f, len(lines))]
    return prem, nodes[0][0], lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=34001); ap.add_argument('--split_seed', type=int, default=34000)
    ap.add_argument('--maxprem', type=int, default=5); ap.add_argument('--max_prompt_tok', type=int, default=150); ap.add_argument('--max_total_tok', type=int, default=800)
    ap.add_argument('--k3_share', type=float, default=0.5); ap.add_argument('--max_tries', type=int, default=1500000)
    ap.add_argument('--cap_test', type=int, default=8); ap.add_argument('--cap_sft_long', type=int, default=15); ap.add_argument('--cap_sft_short', type=int, default=40)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    # ---- excluded parts ----
    tp = json.load(open(f'{DB}/comp/test_parts.json'))
    xn, xp = set(tp['names']), set(tp['pkeys'])
    comp_parts = {n for p in rd(f'{DB}/comp/provenance_constructed.jsonl') for n in p['parts']}
    xn |= comp_parts
    res = {}
    for l in open('/scratch/cowifornia/nd-rl/ab2/data/reserve.jsonl'):
        r = json.loads(l)
        if r['source'] == 'gen' and r.get('minlen_proof'):
            res[r['orig_name']] = r
    xp |= {pkey(res[n]['thm']) for n in comp_parts}
    # ---- free parts, one per pkey ----
    byp = {}
    for n, r in res.items():
        if not (2 <= r['L_true'] <= 10) or n in xn:
            continue
        p = pkey(r['thm'])
        if p in xp or p in byp:
            continue
        try:
            pt = Part(n, r['thm'], r['minlen_proof'], r['L_true'], 'gen')
        except AssertionError:
            continue
        if len(pt.atoms) > 2 or not satisfiable(pt.prem):
            continue
        byp[p] = pt
    strata = collections.defaultdict(list)
    for p in sorted(byp):
        strata[(byp[p].L, len(byp[p].atoms))].append(p)
    srng = random.Random(a.split_seed)
    side = {}
    for k in sorted(strata):                      # parts of length <= 4 are only useful for short proofs: all to SFT
        ps = strata[k]; srng.shuffle(ps)
        for i, p in enumerate(ps):
            side[p] = 'sft' if k[0] <= 4 else ('test' if i % 2 == 0 else 'sft')
    pools = {s: collections.defaultdict(list) for s in ('test', 'sft')}   # side -> (L, atoms) -> parts
    for p, pt in byp.items():
        pt.src = p                                   # keep the part pkey
        pools[side[p]][(pt.L, len(pt.atoms))].append(pt)
    st = collections.Counter()
    st['excluded_part_names'] = len(xn); st['excluded_part_pkeys'] = len(xp); st['free_parts'] = len(byp)
    st['free_parts_test'] = sum(v == 'test' for v in side.values()); st['free_parts_sft'] = sum(v == 'sft' for v in side.values())
    # ---- excluded theorems ----
    ex = json.load(open(f'{DB}/excl.json'))
    EK = set().union(*[set(v['keys']) for v in ex.values()]); EP = set().union(*[set(v['pkeys']) for v in ex.values()])
    for fn in glob.glob('/scratch/cowifornia/nd-rl/ab3/sealed/*.jsonl') + ['/scratch/cowifornia/nd-rl/ab3/data_comp/targets_L6-11.jsonl']:
        for r in rd(fn):
            EK.add(canon_key(r['thm'])); EP.add(pkey(r['thm']))
    seen = set()

    uses = collections.Counter()                  # part pkey -> uses in the file being built

    def pick(opts, cap):
        """random part, rejection-weighted by 1 / (1 + uses)^2 and hard-capped at cap uses per file"""
        for _ in range(50):
            c = rng.choice(opts); u = uses[c.src]
            if u < cap and rng.random() < 1.0 / (1 + u) ** 2:
                return c
        return None

    def splits(sidep, L, k):
        """feasible (length, atom-slot) splits of written length L into k parts, given the side's pools"""
        P = pools[sidep]
        have1 = {l for (l, at), v in P.items() if at == 1 and v}; have2 = {l for (l, at), v in P.items() if v}
        if k == 2:
            return [((l1, 2), (L - 1 - l1, 2)) for l1 in range(2, 11) if l1 in have2 and (L - 1 - l1) in have2 and l1 <= L - 1 - l1]
        tot = L - 2
        return [((l1, 1), (l2, 1), (tot - l1 - l2, 2)) for l1 in range(2, 11) for l2 in range(l1, 11)
                if l1 in have1 and l2 in have1 and (tot - l1 - l2) in have2]

    split_cache = {}

    def draw(sidep, L, k, cap):
        """one candidate of written length L with k parts from pool side sidep, or None."""
        P = pools[sidep]
        key_ = (sidep, L, k)
        if key_ not in split_cache:
            split_cache[key_] = splits(sidep, L, k)
        sp = split_cache[key_]
        if not sp:
            return None
        chosen = []
        for li, at in rng.choice(sp):
            opts = P.get((li, at), []) + (P.get((li, 1), []) if at == 2 else [])
            c = pick(opts, cap)
            if c is None:
                return None
            chosen.append(c)
        if len({c.name for c in chosen}) < len(chosen):
            return None
        if sum(len(c.atoms) for c in chosen) > 4 or sum(len(c.prem) for c in chosen) > a.maxprem:
            return None
        at = ATOMS[:]; rng.shuffle(at); rs = []; o = 0
        for c in chosen:
            rs.append(c.renamed(at[o:o + len(c.atoms)], rng)); o += len(c.atoms)
        prem, concl, lines = combine_tree(rs, rng)
        return chosen, prem, concl, lines

    def build(name, sidep, lens_counts, kquota, cap):
        out = []; uses.clear()
        for L, n in lens_counts:
            got = 0; kc = collections.Counter()
            q3 = kquota(L, n); quota = [(3, q3), (2, n - q3)]
            for k, nk in quota:                  # k = 3 first; any k = 3 shortfall is filled with k = 2
              tries = 0; nk = nk + (n - got - nk if k == 2 else 0)
              while kc[k] < nk and tries < a.max_tries:
                tries += 1
                r = draw(sidep, L, k, cap)
                if r is None:
                    st[f'{name}_nodraw_k{k}'] += 1; continue
                chosen, prem, concl, lines = r
                if len(lines) != L or concl in prem or len(set(prem)) != len(prem):
                    st[f'{name}_bad'] += 1; continue
                if len(prompt_str(prem, concl).split()) > a.max_prompt_tok:
                    st[f'{name}_long_prompt'] += 1; continue
                thm = thm_str(prem, concl); key = canon_key(thm); pk = pkey(thm)
                if key in EK or pk in EP:
                    st[f'{name}_excluded'] += 1; continue
                if pk in seen:
                    st[f'{name}_dup'] += 1; continue
                ok, reason, prompt, txt = check(prem, concl, lines)
                if not ok:
                    st[f'{name}_VERIFIER_REJECT'] += 1; continue
                ntok = len(TOK.encode_prompt(prompt)) + len(TOK.encode_proof(txt))
                if ntok > a.max_total_tok:
                    st[f'{name}_long_total'] += 1; continue
                seen.add(pk); got += 1; kc[k] += 1
                for c in chosen:
                    uses[c.src] += 1
                out.append({'name': f'mid_{name}_{len(out)}', 'thm': thm, 'key': key, 'pkey': pk, 'prompt': prompt, 'n_lines': L,
                            'rules': sorted({ln[2] for ln in lines[len(prem):]}), 'n_prem': len(prem), 'proof': txt,
                            'text': prompt + ' ' + txt, 'family': f'andi_k{len(chosen)}', 'parts': [c.name for c in chosen],
                            'part_pkeys': [c.src for c in chosen], 'part_L': [c.L for c in chosen], 'constructed_lines': L, 'n_tokens': ntok})
            st[f'{name}_L{L}'] = got; st[f'{name}_L{L}_k2'] = kc[2]; st[f'{name}_L{L}_k3'] = kc[3]
        st[f'{name}_max_part_uses'] = max(uses.values())
        rng.shuffle(out)
        with open(f'{OUT}/{name}.jsonl', 'w') as f:
            for r in out:
                f.write(json.dumps(r) + '\n')
        return out

    def kmix(L, n):
        feas2 = L <= 21; feas3 = L >= 8
        if feas2 and feas3:
            return int(round(a.k3_share * n))
        return 0 if feas2 else n
    test = build('test_long', 'test', [(26, 150), (24, 150), (22, 150)], lambda L, n: n, a.cap_test)
    per = [(L, 2000 // 9 + (1 if i < 2000 % 9 else 0)) for i, L in enumerate(range(16, 25))]
    sl = build('sft_long', 'sft', per[::-1], kmix, a.cap_sft_long)
    per = [(L, 2000 // 6 + (1 if i < 2000 % 6 else 0)) for i, L in enumerate(range(6, 12))]
    ss = build('sft_short', 'sft', per, kmix, a.cap_sft_short)
    # part disjointness sanity
    tparts = {p for r in test for p in r['part_pkeys']}; sparts = {p for r in sl + ss for p in r['part_pkeys']}
    assert not (tparts & sparts) and not (tparts & xp) and not (sparts & xp)
    st['test_distinct_parts'] = len(tparts); st['sft_distinct_parts'] = len(sparts)
    json.dump(dict(st), open(f'{OUT}/build_stats.json', 'w'), indent=1)
    print(json.dumps(dict(st), indent=1))


if __name__ == '__main__':
    main()
