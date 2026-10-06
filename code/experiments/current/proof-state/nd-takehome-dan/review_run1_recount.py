#!/usr/bin/env python3
"""Reviewer's independent recount for round2-run1 steps 2 and 3 (own code; only nd_verify shared with the executor).
Step 2: re-judges every tokens / english output from the raw generations (own extraction, own English->token parser,
nd_verify), takes the Lean verdicts from review_run1_lean_recheck.py when present (else the executor's), and tabulates
greedy pass@1, pass@8 (samples only) and pass@9 (greedy or any sample; the executor's "pass8") by form and bin, with
Wilson intervals and a paired Lean - tokens delta per (theorem, draw) bootstrapped over theorems.
Step 3: smallest Qwen3 size with >= 1 Lean-accepted output of 17 (greedy + 16) per theorem and class; the ordering
against the Phase-1 base log-probabilities where they exist.  Output: artifacts/review_r1/recount.json
"""
import json, os, re, sys, collections, random, math
WS = os.path.expanduser('~/review/round2-run1')
OUT = os.path.expanduser('~/nd-takehome/artifacts/review_r1')
sys.path.insert(0, WS)
from nd_verify import verify_text
SIZES = ['Qwen3-0.6B', 'Qwen3-1.7B', 'Qwen3-4B', 'Qwen3-8B', 'Qwen3-14B', 'Qwen3-32B']
BINS = ['val36_<=6', 'val36_>6'] + [f'transfer_{L}' for L in range(7, 17)]
EN = {'premise': 'PR', 'assumption': 'AS', 'reiteration': 'R', 'and-introduction': 'ANDI', 'and-elimination-left': 'ANDE1', 'and-elimination-right': 'ANDE2',
      'implication-elimination': 'IMPE', 'implication-introduction': 'IMPI', 'or-introduction-left': 'ORI1', 'or-introduction-right': 'ORI2', 'or-elimination': 'ORE',
      'negation-elimination': 'NEGE', 'negation-introduction': 'NEGI', 'falsum-elimination': 'BOTE', 'double-negation-elimination': 'DN'}


def rd(fn):
    return [json.loads(l) for l in open(fn) if l.strip()]


def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (round((c - h) / d, 4), round((c + h) / d, 4))


def en_formula(s):
    """English fully-parenthesised formula -> token formula string, or None"""
    toks = s.replace('(', ' ( ').replace(')', ' ) ').split()
    def p(i):
        t = toks[i]
        if t in ('P', 'Q', 'R', 'S'): return t, i + 1
        if t == 'False': return 'F', i + 1
        if t != '(': raise ValueError
        if toks[i + 1] == 'not':
            a, j = p(i + 2)
            if toks[j] != ')': raise ValueError
            return f'( ~ {a} )', j + 1
        a, j = p(i + 1); op = {'and': '&', 'or': 'v', 'implies': '>'}[toks[j]]; b, k = p(j + 1)
        if toks[k] != ')': raise ValueError
        return f'( {a} {op} {b} )', k + 1
    try:
        f, i = p(0)
        return f if i == len(toks) else None
    except Exception:
        return None


def english_to_tokens(text):
    out = []
    for raw in text.strip().splitlines():
        raw = raw.strip()
        if not raw or raw.lower().startswith('qed'): continue
        m = re.match(r'line\s+(\d+)\s*\(depth\s+(\d+)\)\s*:\s*(.+?)\s+by\s+([a-z\-]+)(?:\s+from\s+([\d,\s]+))?\s*$', raw)
        if not m: return None
        idx, depth, formula, rule, refs = m.groups()
        f = en_formula(formula)
        if f is None or rule not in EN: return None
        out += [f'N{idx}'] + ['|'] * int(depth) + [f, ':', EN[rule]] + ([f'N{r.strip()}' for r in refs.split(',') if r.strip()] if refs else []) + [';']
    return ' '.join(out + ['QED'])


def judge(form, prompt, text):
    t = text.strip()
    t = re.sub(r'^```[a-zA-Z0-9]*\n', '', t); t = re.sub(r'\n```\s*$', '', t).strip()
    if form == 'tokens':
        m = re.search(r'(N\d+ .*?QED)', t, re.S)
        body = m.group(1) if m else t
        return bool(verify_text(prompt + ' ' + body)[0])
    tk = english_to_tokens(t)
    return False if tk is None else bool(verify_text(prompt + ' ' + tk)[0])


def step2():
    P = {r['id']: r for r in rd(f'{WS}/data/r1/prompts.jsonl')}
    gens = rd(f'{WS}/artifacts/r1/gens_qwen30b.jsonl')
    ex = {(r['id'], r['sample']): r['ok'] for r in rd(f'{WS}/artifacts/r1/scored_qwen30b.jsonl')}
    lean_fn = f'{OUT}/lean_recheck_qwen30b_verdicts.json'
    mine_lean = json.load(open(lean_fn))['my_ok'] if os.path.exists(lean_fn) else None
    ok = {}; n_out = collections.Counter(); disagree = collections.Counter(); trunc = collections.Counter()
    for g in gens:
        p = P[g['id']]; n_out[len(g['outputs'])] += 1
        for j, text in enumerate(g['outputs']):
            if p['form'] == 'lean':
                v = mine_lean[f'{g["id"]}|{j}'] if mine_lean else ex[(g['id'], j)]
            else:
                v = judge(p['form'], p['prompt'], text)
            ok[(g['id'], j)] = v
            if v != ex[(g['id'], j)]: disagree[p['form']] += 1
            end = text.rstrip()
            done = end.endswith('QED') if p['form'] == 'tokens' else (('exact' in end.splitlines()[-1]) if end else False) if p['form'] == 'lean' else bool(re.search(r'by [a-z\-]+( from [\d, ]+)?\s*$', end))
            if not done: trunc[(p['form'], j == 0)] += 1
    per = {}
    for g in gens:
        p = P[g['id']]; vs = [ok[(g['id'], j)] for j in range(len(g['outputs']))]
        per[(p['form'], p['theorem_name'], p['draw'])] = {'greedy': vs[0], 'pass8': any(vs[1:]), 'pass9': vs[0] or any(vs[1:]), 'bin': p['bin'], 'n_ok': sum(vs[1:])}
    table = {}
    for form in ('tokens', 'lean', 'english'):
        for b in BINS + ['all']:
            ks = [k for k in per if k[0] == form and (b == 'all' or per[k]['bin'] == b)]
            n = len(ks); row = {'n': n}
            for m in ('greedy', 'pass8', 'pass9'):
                c = sum(per[k][m] for k in ks); row[m] = round(c / n, 4); row[m + '_ci'] = wilson(c, n); row[m + '_k'] = c
            table[f'{form}|{b}'] = row
    thms = sorted({k[1] for k in per}); bins = {k[1]: per[k]['bin'] for k in per}
    paired = {}
    for m in ('greedy', 'pass8', 'pass9'):
        for a, bform in (('lean', 'tokens'), ('english', 'tokens'), ('lean', 'english')):
            d = {t: sum(int(per[(a, t, dr)][m]) - int(per[(bform, t, dr)][m]) for dr in range(5)) / 5 for t in thms}
            vals = [d[t] for t in thms]; rng = random.Random(1); boots = []
            for _ in range(4000):
                boots.append(sum(vals[rng.randrange(len(vals))] for _ in vals) / len(vals))
            boots.sort()
            paired[f'{a}-{bform}|{m}'] = {'mean': round(sum(vals) / len(vals), 4), 'ci95': (round(boots[100], 4), round(boots[3899], 4)), 'n_theorems': len(vals),
                                          'by_bin': {b: round(sum(d[t] for t in thms if bins[t] == b) / sum(1 for t in thms if bins[t] == b), 3) for b in BINS}}
    # per-draw variation of the all-theorem rates
    by_draw = {f'{form}|{m}': [round(sum(per[(form, t, dr)][m] for t in thms) / len(thms), 3) for dr in range(5)] for form in ('tokens', 'lean', 'english') for m in ('greedy', 'pass9')}
    return {'n_gens': len(gens), 'outputs_per_prompt': dict(n_out), 'lean_verdicts_source': 'reviewer recheck' if mine_lean else 'executor scored file',
            'disagreements_with_executor_ok': dict(disagree), 'unfinished_outputs(form,greedy)': {f'{k[0]}|{"greedy" if k[1] else "sample"}': v for k, v in trunc.items()},
            'table': table, 'paired': paired, 'by_draw': by_draw}


def step3():
    S = {r['theorem_name']: r for r in rd(f'{WS}/data/r1/scale_prompts.jsonl')}
    per = collections.defaultdict(dict); n_rows = {}
    for m in SIZES:
        fn = f'{WS}/artifacts/r1/scored_scale_{m}.jsonl'
        if not os.path.exists(fn): continue
        vfn = f'{OUT}/lean_recheck_{m}_verdicts.json'
        mine = json.load(open(vfn))['my_ok'] if os.path.exists(vfn) else None
        rows = rd(fn); n_rows[m] = len(rows)
        by = collections.defaultdict(list)
        for r in rows:
            by[r['theorem_name']].append(mine[f'{r["id"]}|{r["sample"]}'] if mine else r['ok'])
        for t, vs in by.items(): per[t][m] = (sum(vs), len(vs))
    first = {}
    for t, d in per.items():
        first[t] = next((m for m in SIZES if m in d and d[m][0] > 0), 'none')
    classes = sorted({S[t]['class'] for t in per})
    by_cls = {c: {m: sum(1 for t in per if S[t]['class'] == c and first[t] == m) for m in SIZES + ['none']} for c in classes}
    n_cls = {c: sum(1 for t in per if S[t]['class'] == c) for c in classes}
    pass16 = {c: {m: sum(1 for t in per if S[t]['class'] == c and per[t].get(m, (0, 0))[0] > 0) for m in SIZES} for c in classes}
    # ordering vs phase-1 base log-prob (where present)
    ph = [(t, S[t]['phase1']['base_logp_T1_max'], SIZES.index(first[t]) if first[t] != 'none' else len(SIZES)) for t in per if S[t].get('phase1')]
    def spearman(xs, ys):
        def ranks(v):
            s = sorted(range(len(v)), key=lambda i: v[i]); r = [0] * len(v); i = 0
            while i < len(s):
                j = i
                while j + 1 < len(s) and v[s[j + 1]] == v[s[i]]: j += 1
                for k in range(i, j + 1): r[s[k]] = (i + j) / 2 + 1
                i = j + 1
            return r
        rx, ry = ranks(xs), ranks(ys); n = len(xs); mx, my = sum(rx) / n, sum(ry) / n
        num = sum((a - mx) * (b - my) for a, b in zip(rx, ry)); den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
        return round(num / den, 3) if den else None
    rho = spearman([x[1] for x in ph], [x[2] for x in ph]) if len(ph) > 2 else None
    return {'n_rows_per_size': n_rows, 'n_per_class': n_cls, 'first_size_by_class': by_cls, 'theorems_with_any_pass_by_size': pass16,
            'phase1_logp_available': len(ph), 'spearman_baselogp_vs_firstsize_rank': rho,
            'phase1_rows': sorted([{'thm': t, 'base_logp_T1_max': round(lp, 1), 'first': first[t], 'rl_round': S[t]['rl_round']} for t, lp, _ in ph], key=lambda r: r['base_logp_T1_max']),
            'first_size': {t: first[t] for t in per}, 'per_size_counts': {t: {m: f'{v[0]}/{v[1]}' for m, v in d.items()} for t, d in per.items()}}


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    res = {'step2': step2(), 'step3': step3()}
    json.dump(res, open(f'{OUT}/recount.json', 'w'), indent=1)
    s2 = res['step2']
    print('step2: gens', s2['n_gens'], s2['outputs_per_prompt'], 'lean from', s2['lean_verdicts_source'], 'disagree', s2['disagreements_with_executor_ok'], 'unfinished', s2['unfinished_outputs(form,greedy)'])
    for k, v in s2['table'].items():
        print(f"  {k:22s} n {v['n']:4d} greedy {v['greedy']:.3f} {v['greedy_ci']} pass8 {v['pass8']:.3f} pass9 {v['pass9']:.3f} {v['pass9_ci']}")
    for k, v in s2['paired'].items(): print('  paired', k, v['mean'], v['ci95'], v['by_bin'])
    print('  by draw', s2['by_draw'])
    s3 = res['step3']
    print('step3:', s3['n_rows_per_size'], s3['n_per_class']); print('  first size', s3['first_size_by_class']); print('  any pass', s3['theorems_with_any_pass_by_size'])
    print('  phase1 n', s3['phase1_logp_available'], 'spearman', s3['spearman_baselogp_vs_firstsize_rank']); print('  ', s3['phase1_rows'])
