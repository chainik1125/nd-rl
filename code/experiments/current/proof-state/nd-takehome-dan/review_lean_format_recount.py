#!/usr/bin/env python3
"""Reviewer recount for run lean-format (phase 1).  Own code throughout: ND line parser, start-index normaliser, dependency
pruner, box-depth counter, renaming-class canonicaliser, L*.  Reads only raw artefacts (found_*.jsonl, round_*.json, gate logs,
eval jsonl) and data files.  Nothing from the executor's analysis scripts is imported.

  python3 review_lean_format_recount.py > artifacts/review_lf/recount_stdout.txt
"""
import json, os, re, sys, glob, itertools, collections, math

A = 'artifacts/lf'
OUT = 'artifacts/review_lf'
os.makedirs(OUT, exist_ok=True)
BOX_RULES = {'IMPI': [(0, 1)], 'NEGI': [(0, 1)], 'ORE': [(1, 2), (3, 4)]}


def jl(fn):
    return [json.loads(l) for l in open(fn)]


# ---------- own ND proof parser ----------
def parse_nd(proof):
    """'N1 | F : RULE N2 ; ... QED' -> list of dict(idx, depth, formula(str), rule, refs[int])."""
    toks = proof.split()
    assert toks[-1] == 'QED', proof[-40:]
    body = toks[:-1]
    lines = []; cur = []
    for t in body:
        if t == ';':
            lines.append(cur); cur = []
        else:
            cur.append(t)
    assert not cur, 'dangling tokens'
    out = []
    for l in lines:
        m = re.fullmatch(r'N(\d+)', l[0]); assert m, l
        idx = int(m.group(1)); j = 1; d = 0
        while l[j] == '|':
            d += 1; j += 1
        c = len(l) - 1 - l[::-1].index(':')
        formula = ' '.join(l[j:c]); rule = l[c + 1]
        refs = [int(x[1:]) for x in l[c + 2:]]
        assert all(re.fullmatch(r'N\d+', x) for x in l[c + 2:]), l
        out.append({'idx': idx, 'depth': d, 'formula': formula, 'rule': rule, 'refs': refs})
    return out


def normalise(lines):
    """start-index normaliser: renumber so the first line is 1 (and indices consecutive); returns canonical string."""
    m = {l['idx']: k + 1 for k, l in enumerate(lines)}
    return ' ; '.join(f"N{m[l['idx']]} {'| ' * l['depth']}{l['formula']} : {l['rule']}" + ''.join(f' N{m[r]}' for r in l['refs']) for l in lines) + ' ; QED'


def prune(lines):
    """dependency pruning: keep lines reachable from the last line through citations (PR lines always kept)."""
    by = {l['idx']: l for l in lines}
    keep = set(); todo = [lines[-1]['idx']]
    while todo:
        i = todo.pop()
        if i in keep: continue
        keep.add(i)
        todo.extend(by[i]['refs'])
    return [l for l in lines if l['idx'] in keep or l['rule'] == 'PR']


def max_depth(lines):
    return max(l['depth'] for l in lines)


def depth3(proof):
    """campaign predicate P3 (on the dependency-pruned proof): maximum box depth >= 3.  Also returns the written-proof version."""
    L = parse_nd(proof)
    return max_depth(prune(L)) >= 3, max_depth(L) >= 3


# ---------- own renaming-class canonicaliser ----------
ATOMS = 'PQRS'


def thm_of_prompt(prompt):
    t = prompt.split()
    assert t[0] == 'THM' and t[-1] == 'PRF'
    i = t.index('SEQ')
    prem_toks = t[1:i]; concl = ' '.join(t[i + 1:-1])
    prem = []; cur = []; d = 0
    for x in prem_toks:
        if x == ',' and d == 0:
            prem.append(' '.join(cur)); cur = []
        else:
            d += (x == '(') - (x == ')'); cur.append(x)
    if cur: prem.append(' '.join(cur))
    return prem, concl


def cls(prompt, sort_prem=True):
    """renaming class: min over the 24 atom permutations of (premises [sorted = order-insensitive], conclusion)."""
    prem, concl = thm_of_prompt(prompt)
    best = None
    for perm in itertools.permutations(ATOMS):
        m = dict(zip(ATOMS, perm))
        r = lambda s: ' '.join(m.get(x, x) for x in s.split())
        ps = [r(p) for p in prem]
        if sort_prem: ps = sorted(ps)
        k = ' , '.join(ps) + ' |- ' + r(concl)
        if best is None or k < best: best = k
    return best


def wilson(k, n, z=1.96):
    if n == 0: return (0, 0)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def main():
    res = {}
    # ================= (a) depth-3 f = 0 dial =================
    print('=== (a) depth-3 f=0 dial: acquisition (own pruner + depth counter) ===')
    targets = jl('data/p2/targets_depth3.jsonl'); transfer = jl('data/p2/transfer_depth3.jsonl')
    tn = {t['name'] for t in targets}; trn = {t['name'] for t in transfer}
    tprompt = {t['name']: t['prompt'] for t in targets + transfer}
    print('targets', len(targets), 'transfer', len(transfer))
    for sch in ('rand', 'seq'):
        for kind in ('ei', 'frozen'):
            for s in (0, 1):
                arm = f'{kind}_d3_{sch}_s{s}'
                row = {}
                for pool, names, n in (('found', tn, 1000), ('found_transfer', trn, 500)):
                    per_round = []
                    for r in range(1, 9):
                        F = jl(f'{A}/{arm}/{pool}_{r}.jsonl')
                        assert all(x['name'] in names for x in F)
                        assert all(x['prompt'] == tprompt[x['name']] for x in F), 'prompt mismatch'
                        # distinctness after my own normalisation
                        norm = collections.defaultdict(set)
                        for x in F:
                            norm[x['name']].add(normalise(parse_nd(x['proof'])))
                        n_raw = len(F); n_norm = sum(len(v) for v in norm.values())
                        solved = set(norm)
                        acq_p = set(); acq_w = set(); d3proofs = 0
                        for x in F:
                            p, w = depth3(x['proof'])
                            if p: acq_p.add(x['name']); d3proofs += 1
                            if w: acq_w.add(x['name'])
                        per_round.append({'round': r, 'proofs_raw': n_raw, 'proofs_norm': n_norm, 'solved': len(solved), 'acq_pruned': len(acq_p), 'acq_written': len(acq_w), 'd3_proofs': d3proofs})
                    row[pool] = per_round
                    # first round with a depth-3 proof
                    first = next((q['round'] for q in per_round if q['acq_pruned'] > 0), None)
                    f8 = per_round[-1]
                    print(f"{arm:20s} {pool:15s} r8: proofs {f8['proofs_raw']} (normalised-distinct {f8['proofs_norm']}) solved {f8['solved']}/{n} "
                          f"acq(pruned depth>=3) {f8['acq_pruned']}/{n} = {f8['acq_pruned']/n:.3f} [acq written-depth {f8['acq_written']/n:.3f}] d3 proofs {f8['d3_proofs']} first d3 round {first}")
                    print('    by round acq:', [q['acq_pruned'] for q in per_round], 'solved:', [q['solved'] for q in per_round])
                # round-file aggregates: held-out greedy every round, secs
                rj = [json.load(open(f'{A}/{arm}/round_{r}.json')) for r in range(1, 9)]
                row['heldout'] = [(q['heldout_greedy']['solved'], q['heldout_greedy']['n']) for q in rj]
                row['secs'] = [q['secs'] for q in rj]
                row['ckpts'] = [q.get('ckpt') for q in rj]
                print('    heldout greedy by round:', [h[0] for h in row['heldout']], '/ 5000 ; round secs', [round(x) for x in row['secs']])
                print('    sampled-from ckpts:', sorted(set(row['ckpts'])) if kind == 'frozen' else row['ckpts'][:2] + ['...'])
                res[arm] = row

    # consistency of found files with round json cum counts
    print('\n--- found_8 vs round_8.json targets_cum / transfer_cum ---')
    for arm in sorted(res):
        q = json.load(open(f'{A}/{arm}/round_8.json'))
        print(f"{arm:20s} json targets_cum {q['targets_cum']['solved']} mine {res[arm]['found'][-1]['solved']} | json transfer_cum {q['transfer_cum']['solved']} mine {res[arm]['found_transfer'][-1]['solved']}")

    # ================= (b) ladder rung T1 =================
    print('\n=== (b) ladder T1: L* (largest L with >=5 distinct theorems of L_true >= L solved), transfer + targets ===')
    LT = {p: {t['name']: t for t in jl(f'data/ladder/{p}.jsonl')} for p in ('rl_targets', 'transfer')}
    for p in LT:
        assert all(t['L_true'] == t['n_lines'] for t in LT[p].values())
        print(p, len(LT[p]), 'L_true hist', sorted(collections.Counter(t['L_true'] for t in LT[p].values()).items()))

    def lstar(names, pool):
        ls = sorted((LT[pool][n]['L_true'] for n in names), reverse=True)
        return ls[4] if len(ls) >= 5 else None

    lad = {}
    for sch in ('rand', 'seq'):
        for kind in ('T1', 'frozen'):
            for s in (0, 1):
                arm = f'la_{kind}_{sch}_s{s}'
                row = {}
                for pool, fpre in (('rl_targets', 'found'), ('transfer', 'found_transfer')):
                    pr = []
                    for r in range(1, 9):
                        F = jl(f'{A}/{arm}/{fpre}_{r}.jsonl')
                        assert all(x['name'] in LT[pool] and x['prompt'] == LT[pool][x['name']]['prompt'] for x in F)
                        solved = {x['name'] for x in F}
                        pr.append({'round': r, 'solved': len(solved), 'Lstar': lstar(solved, pool), 'proofs': len(F)})
                        if r == 8:
                            norm = collections.defaultdict(set)
                            for x in F: norm[x['name']].add(normalise(parse_nd(x['proof'])))
                            row[pool + '_norm8'] = sum(len(v) for v in norm.values())
                            row[pool + '_solved_names'] = solved
                            byL = collections.Counter(LT[pool][n]['L_true'] for n in solved)
                            row[pool + '_byL'] = dict(sorted(byL.items()))
                            wl = collections.Counter(len(parse_nd(x['proof'])) for x in F)
                            row[pool + '_written_hist'] = dict(sorted(wl.items()))
                    row[pool] = pr
                    nL = collections.Counter(t['L_true'] for t in LT[pool].values())
                    print(f"{arm:20s} {pool:10s} r8 solved {pr[-1]['solved']}/{len(LT[pool])} L* {pr[-1]['Lstar']}  proofs {pr[-1]['proofs']} (norm {row[pool+'_norm8']})  solved by L_true "
                          + ' '.join(f"{L}:{row[pool+'_byL'].get(L,0)}/{nL[L]}" for L in sorted(nL)))
                    print('    L* by round', [q['Lstar'] for q in pr], 'solved by round', [q['solved'] for q in pr], 'written hist', row[pool + '_written_hist'])
                rj = [json.load(open(f'{A}/{arm}/round_{r}.json')) for r in range(1, 9)]
                row['secs'] = [q.get('secs') for q in rj]
                print('    round secs', [round(x) if x else x for x in row['secs']])
                lad[arm] = row
    # base reachability: T1-solved transfer theorems also solved by the frozen control at equal attempts (same seed; and union of both frozen seeds)
    print('\n--- base reachability of T1 transfer solves (frozen control, equal attempts 8 x 32) ---')
    for sch in ('rand', 'seq'):
        fz = [lad[f'la_frozen_{sch}_s{s}']['transfer_solved_names'] for s in (0, 1)]
        for s in (0, 1):
            t1 = lad[f'la_T1_{sch}_s{s}']['transfer_solved_names']
            new_same = t1 - fz[s]; new_any = t1 - (fz[0] | fz[1])
            byL = collections.Counter(LT['transfer'][n]['L_true'] for n in new_any)
            print(f"la_T1_{sch}_s{s}: solved {len(t1)}, frozen(same seed) {len(fz[s])}, T1-only vs same-seed frozen {len(new_same)}, vs union of frozen seeds {len(new_any)}; T1-only by L_true {dict(sorted(byL.items()))}; "
                  f"L* of T1-only set {lstar(new_any,'transfer')}")
    # same for depth-3 acquisition
    print('\n--- base reachability of depth-3 acquisition (frozen arms above; acquisition of frozen = base rate at equal attempts) ---')

    # ================= P4 mechanism test =================
    print('\n=== P4: Stage-1 full-train models, data/transfer.jsonl pass@16: distinct verified proofs of written length 7 / 8 (normalised) ===')
    for tag in ('rand', 'seq', 'seqfixed'):
        E = jl(f'{A}/stage1_full_{tag}_transfer2_k16.jsonl')
        cnt = collections.Counter(); thm = collections.defaultdict(set); tot = 0
        for x in E:
            seen = set()
            for p in x['proofs']:
                L = parse_nd(p); k = normalise(L)
                if k in seen: continue
                seen.add(k); cnt[len(L)] += 1; thm[len(L)].add(x['name']); tot += 1
        H = json.load(open(f'{A}/stage1_full_{tag}_heldout_greedy.json'))
        HJ = jl(f'{A}/stage1_full_{tag}_heldout_greedy.jsonl')
        print(f"stage1_full_{tag}: n {len(E)} solved {sum(1 for x in E if x['proofs'])} distinct proofs {tot}; written-length hist {dict(sorted(cnt.items()))}; "
              f"7-line {cnt[7]} (on {len(thm[7])} thms) 8-line {cnt[8]} (on {len(thm[8])} thms) >=9 {sum(v for k,v in cnt.items() if k>=9)} | heldout greedy json {H['solved']}/{H['n']}, jsonl {sum(1 for x in HJ if x['solved'])}/{len(HJ)}")
        res['mech_' + tag] = E
    # token comparators on file
    for tag in ('abs', 'rel', 'absfixed'):
        fn = f'artifacts/stage1_{tag}_transfer2_k16.json'
        fj = f'artifacts/stage1_{tag}_transfer2_k16.jsonl'
        if os.path.exists(fn):
            q = json.load(open(fn)); print(f'token {tag}: json solved {q["solved"]}/{q["n"]} written_hist {q.get("written_hist")}', '(jsonl present)' if os.path.exists(fj) else '(no per-sample jsonl on file)')

    # ================= P5 gate tables =================
    print('\n=== P5: in-loop gate 2x2 tables (Lean on literal text vs nd_verify on denoted proof), summed per log ===')
    G = collections.Counter(); kinds = collections.Counter()
    for fn in sorted(glob.glob(f'{A}/gate_*.jsonl')):
        if fn.endswith('.disagree.jsonl'): continue
        c = collections.Counter(); lw = lp = nv = 0
        for x in jl(fn):
            for k in ('samples', 'parse_fail', 'distinct_checked', 'both_ok', 'nd_ok_lean_rej', 'nd_rej_lean_ok', 'both_rej'):
                c[k] += x[k]
            lw += x['lean_wall_s']; lp += x['lean_proc_s']; nv += x['nd_verify_s']
        dfn = fn.replace('.jsonl', '.disagree.jsonl')
        nd = len(jl(dfn)) if os.path.exists(dfn) else 0
        G.update(c)
        print(f"{os.path.basename(fn):34s} samples {c['samples']:>8} parsefail {c['parse_fail']:>7} checked {c['distinct_checked']:>7} both_ok {c['both_ok']:>7} nd_only {c['nd_ok_lean_rej']:>3} lean_only {c['nd_rej_lean_ok']:>4} both_rej {c['both_rej']:>7} | disagree file rows {nd} | lean wall {lw:7.0f}s proc {lp:8.0f}s nd_verify {nv:6.1f}s -> checked/s: lean(wall) {c['distinct_checked']/max(lw,1e-9):6.0f} lean(proc) {c['distinct_checked']/max(lp,1e-9):5.0f} nd {c['distinct_checked']/max(nv,1e-9):7.0f}")
    print('TOTAL', dict(G))
    dis = c_dis = 0
    # classify every disagreement with my own reading of the denoted proof
    print('\n--- disagreements classified (own classifier) ---')
    allk = collections.Counter(); distinct = set()
    for fn in sorted(glob.glob(f'{A}/gate_*.disagree.jsonl')):
        for x in jl(fn):
            key = (x['prompt'], x['nd'])
            direction = 'lean_only' if x['lean_ok'] and not x['nd_ok'] else 'nd_only'
            allk[('rows', direction)] += 1
            if key in distinct: continue
            distinct.add(key)
            res.setdefault('disagree', []).append(x)
            allk[('distinct_nd', direction)] += 1
    print(dict(allk))
    json.dump(res.get('disagree', []), open(f'{OUT}/disagreements_distinct.json', 'w'), ensure_ascii=False)

    # ================= P6 timing =================
    print('\n=== P6: like-for-like frozen single-round timings (secs from round_1.json) ===')
    for d in sorted(glob.glob(f'{A}/timing_*')):
        if not os.path.isdir(d): continue
        q = json.load(open(f'{d}/round_1.json')); a = json.load(open(f'{d}/args.json'))
        print(f"{os.path.basename(d):30s} init {a['init']:36s} secs {q['secs']:.0f}")

    json.dump({k: {kk: vv for kk, vv in v.items() if not kk.endswith('_names')} for k, v in lad.items()}, open(f'{OUT}/ladder.json', 'w'), indent=1, default=list)
    json.dump({k: v for k, v in res.items() if k.startswith(('ei_', 'frozen_'))}, open(f'{OUT}/d3.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
