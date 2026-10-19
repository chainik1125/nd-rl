#!/usr/bin/env python3
"""Composition diagnosis: tables for report/composition.md (numpy only).

    python composition/analyse.py          (after build.py, failloc.py and run_eval.py for pretrain, A/B/C_s101)

Inputs (all under ab3/lengthexp/):
  composition/joined.jsonl              the 1,300 selected joined ANDI theorems, their parts and controls (build.py)
  composition/joined_samples.jsonl      joined results recomputed from eval/<label>/samples (failloc.py)
  composition/eval/<label>/chunk_*.jsonl  parts and controls, k=32 (run_eval.py)
Outputs: composition/results.json (every number in the report) and report/composition.md.
"""
import collections, json, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LE = os.path.dirname(HERE)
sys.path.insert(0, LE)
from common import read_jsonl  # noqa: E402

LABELS = {'pretrain': 'pretrain', 'A': 'A_s101', 'B': 'B_s101', 'C': 'C_s101'}
TRAIN_MAX = {'pretrain': 6, 'A': 7, 'B': 9, 'C': 11}
BANDS = [(6, 9), (10, 14), (15, 18)]
BOOT = 2000
EXAMPLES = [('C_s101', 'ab3_matched_L6-14_454', 'N35 ', 'fake goal',
             'Never proves the left conjunct. Writes the right conjunct ( R & R ) twice, then "proves" the goal by ANDI '
             'with a premise of the other part, ( ( ~ ( R & S ) ) v S ), in place of the left conjunct.'),
            ('A_s101', 'ab3_matched_L6-14_375', 'N12 ', 'fake goal',
             'Proves the left conjunct correctly (line 4), then writes the goal at once, citing the premise ( S v R ) in '
             'place of the right conjunct ( P & P ): it never starts the second part.'),
            ('A_s101', 'ab3_matched_L6-14_102', 'N6 ', 'cross-part citation',
             'Works on the right part but builds ( ( R v ( ~ Q ) ) & ( R & R ) ) by ANDI citing ( ~ P ), a premise of '
             'the other part, in place of ( R & R ).'),
            ('C_s101', 'ab3_matched_L6-14_2085', 'N7 ', 'fake conjunct',
             'Writes the right conjunct ( S & S ) in one invalid step: IMPE from ( S > ( R > S ) ) and ( ~ Q ), a '
             'premise of the other part. Then it writes the goal.'),
            ('A_s101', 'ab3_matched_L6-14_360', 'N8 ', 'premise block',
             'Copies the premise block wrongly: after three correct premises it skips the fourth, ( ( Q v Q ) & P ), '
             'and writes R, then a fragment ( Q v Q ) as premises.')]


def pct(x, nd=1):
    return '-' if x is None or (isinstance(x, float) and not math.isfinite(x)) else f'{100 * x:.{nd}f}%'


def ci_str(lo, hi, nd=1):
    return f'[{pct(lo, nd)}, {pct(hi, nd)}]'


def md(header, rows):
    return '\n'.join(['| ' + ' | '.join(header) + ' |', '|' + '|'.join('---' for _ in header) + '|']
                     + ['| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows])


def boot(arrs, fn, rng):
    n = len(arrs[0])
    vals = []
    for _ in range(BOOT):
        ix = rng.integers(0, n, n)
        vals.append(fn(*[a[ix] for a in arrs]))
    v = np.array(vals, float)
    v = v[np.isfinite(v)]
    return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if len(v) > BOOT // 2 else (float('nan'), float('nan'))


def line_stats(lines, Ls):
    lines, Ls = np.asarray(lines, float), np.asarray(Ls, float)
    if not len(lines):
        return None
    return {'n': int(len(lines)), 'median': float(np.median(lines)), 'p90': float(np.percentile(lines, 90)),
            'share_ge_L': float((lines >= Ls).mean())}


FINAL = os.path.join(HERE, 'final')


def dc_section(rng):
    """Arm D vs arm C on the final test draw (composition/final): joined vs part product at L6-14 (s101), attempt lengths,
    failure places (all seeds)."""
    jf = os.path.join(FINAL, 'joined.jsonl')
    sf = os.path.join(FINAL, 'joined_samples.jsonl')
    if not (os.path.isfile(jf) and os.path.isfile(sf)):
        return None, None
    joined = read_jsonl(jf)
    js = read_jsonl(sf)
    jsamp = {(r['label'], r['name']): r for r in js}
    ev = {}
    for lab in ('C_s101', 'D_s101'):
        d = os.path.join(FINAL, 'eval', lab)
        ev[lab] = {}
        if os.path.isdir(d):
            for fn in sorted(os.listdir(d)):
                if fn.startswith('chunk_') and fn.endswith('.jsonl'):
                    for r in read_jsonl(os.path.join(d, fn)):
                        ev[lab][r['prompt']] = r
    res = {'ratio': {}, 'lines': {}, 'cats': {}}
    for lab in ('C_s101', 'D_s101'):
        for key, sel in [('6-9', lambda t: 6 <= t['L'] <= 9), ('10-14', lambda t: 10 <= t['L'] <= 14), ('6-14', lambda t: True)] + \
                        [(str(L), (lambda t, L=L: t['L'] == L)) for L in range(6, 15)]:
            pj, p1, p2, lines, Ls = [], [], [], [], []
            for t in joined:
                if not sel(t):
                    continue
                j = jsamp.get((lab, t['name']))
                a = [ev[lab].get(p['prompt_ren']) for p in t['parts']]
                if j is None or None in a:
                    continue
                pj.append(j['n_ok'] / j['k']); p1.append(a[0]['n_ok'] / a[0]['k']); p2.append(a[1]['n_ok'] / a[1]['k'])
                lines += j['lines']; Ls += [t['L']] * len(j['lines'])
            if not pj:
                continue
            pj, p1, p2 = map(np.array, (pj, p1, p2))
            prod = float((p1 * p2).mean())
            res['ratio'][f'{lab}|{key}'] = {'n': int(len(pj)), 'joined': float(pj.mean()), 'prod': prod, 'part_mean': float(((p1 + p2) / 2).mean()),
                                            'ratio': float(pj.mean() / prod) if prod > 0 else float('nan'),
                                            'ratio_ci': boot([pj, p1, p2], lambda j_, a_, b_: j_.mean() / (a_ * b_).mean() if (a_ * b_).mean() > 0 else float('nan'), rng),
                                            'joined_ci': boot([pj], lambda a_: a_.mean(), rng), 'prod_ci': boot([p1, p2], lambda a_, b_: (a_ * b_).mean(), rng)}
            res['lines'][f'{lab}|{key}'] = line_stats(lines, Ls)
    armof = lambda l: l[0]
    for a in ('C', 'D'):
        for lo, hi in ((6, 9), (10, 14)):
            cats, prog, nf, ns, nok, lines, Ls = collections.Counter(), collections.Counter(), 0, 0, 0, [], []
            for r in js:
                if armof(r['label']) != a or not (lo <= r['L_true'] <= hi):
                    continue
                cats.update(r['cats']); prog.update(r['progress']); nf += sum(r['cats'].values()); ns += r['k']; nok += r['n_ok']
                lines += r['lines']; Ls += [r['L_true']] * len(r['lines'])
            if ns:
                res['cats'][f'{a}|{lo}-{hi}'] = {'pass1': nok / ns, 'n_samples': ns, 'cats': {k: v / nf for k, v in cats.items()} if nf else {},
                                                 'progress': {k: v / nf for k, v in prog.items()} if nf else {}, 'lines': line_stats(lines, Ls),
                                                 'labels': sorted({r['label'] for r in js if armof(r['label']) == a})}
    out = []
    w = out.append
    w('## 6. Arm D (composition-augmented RL pool) vs arm C, final test draw\n')
    w('Arm D = arm C except the RL pool: run 2\'s targets plus constructed ANDI/chain targets, 30% of each length bin '
      '(`ab3/data_comp/`, `ab3/frozen_comp`). The test theorems are part-disjoint from D\'s targets (0 shared reserve parts among '
      '6,211 test parts and 9,720 training parts). This section uses the **final** matched draw (100 ANDI theorems per length at '
      'L6-14 in `composition/final/joined.jsonl`); sections 1-5 used the earlier draw. Parts alone are the renamed parts, k = 32.\n')
    cf = os.path.join(HERE, 'conclusion_dc.md')
    if os.path.isfile(cf):
        w(open(cf).read().strip() + '\n')
    rows = []
    for key in ['6-9', '10-14', '6-14']:
        for lab in ('C_s101', 'D_s101'):
            r = res['ratio'].get(f'{lab}|{key}')
            if r:
                rows.append([lab, f'L{key}', r['n'], f"{pct(r['joined'], 2)} {ci_str(*r['joined_ci'], 2)}", f"{pct(r['prod'], 2)} {ci_str(*r['prod_ci'], 2)}",
                             pct(r['part_mean'], 1), f"{r['ratio']:.2f} [{r['ratio_ci'][0]:.2f}, {r['ratio_ci'][1]:.2f}]" if math.isfinite(r['ratio']) else '-'])
    w('**Joined vs product of the parts (one seed each).**\n\n' + md(['checkpoint', 'joined length', 'theorems', 'joined pass@1', 'product of parts',
                                                                      'mean part pass@1', 'joined / product'], rows) + '\n')
    rows = []
    for lab in ('C_s101', 'D_s101'):
        rows.append([lab] + [(f"{pct(res['ratio'][f'{lab}|{L}']['joined'], 1)} / {pct(res['ratio'][f'{lab}|{L}']['prod'], 1)}"
                              if f'{lab}|{L}' in res['ratio'] else '') for L in range(6, 15)])
    w('By joined length (joined / product):\n\n' + md(['checkpoint'] + [f'L{L}' for L in range(6, 15)], rows) + '\n')
    rows = []
    for lab in ('C_s101', 'D_s101'):
        for key in ['6-9', '10-14']:
            st = res['lines'].get(f'{lab}|{key}')
            if st:
                rows.append([lab, f'L{key}', f"{st['median']:.0f}", f"{st['p90']:.0f}", pct(st['share_ge_L'])])
    w('**Joined attempts: written lines** (median / p90 / share >= L_true):\n\n' + md(['checkpoint', 'joined length', 'median', 'p90', 'share >= L_true'], rows) + '\n')
    order = ['fake goal', 'fake conjunct', 'cross-part citation', 'inside left part', 'inside right part', 'mixes parts', 'premise block']
    rows = []
    for a in ('C', 'D'):
        for key in ('6-9', '10-14'):
            r = res['cats'].get(f'{a}|{key}')
            if r:
                rows.append([a, f'L{key}', ', '.join(r['labels']), pct(r['pass1'], 2)] + [pct(r['cats'].get(c, 0), 0) for c in order]
                            + [pct(r['progress'].get('none', 0), 0)])
    w('**Where failed joined samples go wrong** (all ANDI theorems of the final matched draw, all seeds):\n\n'
      + md(['arm', 'joined length', 'checkpoints', 'pass@1'] + order + ['no conjunct derived before the error'], rows) + '\n')
    return '\n'.join(out), res



def main():
    joined = read_jsonl(os.path.join(HERE, 'joined.jsonl'))
    jsamp = {(r['label'], r['name']): r for r in read_jsonl(os.path.join(HERE, 'joined_samples.jsonl'))}
    ev = {}
    for arm, lab in LABELS.items():
        d = os.path.join(HERE, 'eval', lab)
        rows = {}
        if os.path.isdir(d):
            for fn in sorted(os.listdir(d)):
                if fn.startswith('chunk_') and fn.endswith('.jsonl'):
                    for r in read_jsonl(os.path.join(d, fn)):
                        rows[r['prompt']] = r
        ev[arm] = rows
    arms = [a for a in LABELS if ev[a]]
    missing = [a for a in LABELS if not ev[a]]
    rng = np.random.default_rng(0)
    res = {'n_joined': len(joined), 'arms': arms, 'missing_arms': missing, 'step1': {}, 'lines': {}, 'step2': {}, 'strata': {}, 'step0': {}}

    # ---------------------------------------------------------------- step 1: joined vs parts
    def table_rows(arm, sel):
        lab = LABELS[arm]
        pj, p1, p2, q1, q2 = [], [], [], [], []
        for t in sel:
            js = jsamp.get((lab, t['name']))
            a = [ev[arm].get(p['prompt_ren']) for p in t['parts']]
            o = [ev[arm].get(p['prompt_orig']) for p in t['parts']]
            if js is None or None in a or None in o:
                continue
            pj.append(js['n_ok'] / js['k'])
            p1.append(a[0]['n_ok'] / a[0]['k']); p2.append(a[1]['n_ok'] / a[1]['k'])
            q1.append(o[0]['n_ok'] / o[0]['k']); q2.append(o[1]['n_ok'] / o[1]['k'])
        return [np.array(x) for x in (pj, p1, p2, q1, q2)]

    def step1(arm, sel):
        pj, p1, p2, q1, q2 = table_rows(arm, sel)
        if not len(pj):
            return None
        out = {'n': int(len(pj)), 'joined': float(pj.mean()), 'part_mean': float(((p1 + p2) / 2).mean()),
               'prod': float((p1 * p2).mean()), 'min': float(np.minimum(p1, p2).mean()),
               'prod_orig': float((q1 * q2).mean()), 'min_orig': float(np.minimum(q1, q2).mean()),
               'part_mean_orig': float(((q1 + q2) / 2).mean())}
        out['joined_ci'] = boot([pj], lambda a: a.mean(), rng)
        out['prod_ci'] = boot([p1, p2], lambda a, b: (a * b).mean(), rng)
        out['min_ci'] = boot([p1, p2], lambda a, b: np.minimum(a, b).mean(), rng)
        out['ratio'] = out['joined'] / out['prod'] if out['prod'] > 0 else float('nan')
        out['ratio_ci'] = boot([pj, p1, p2], lambda j, a, b: j.mean() / (a * b).mean() if (a * b).mean() > 0 else float('nan'), rng)
        out['diff_ci'] = boot([pj, p1, p2], lambda j, a, b: j.mean() - (a * b).mean(), rng)
        return out

    for arm in arms:
        res['step1'][arm] = {}
        for L in range(6, 19):
            res['step1'][arm][str(L)] = step1(arm, [t for t in joined if t['L'] == L])
        for lo, hi in BANDS:
            res['step1'][arm][f'{lo}-{hi}'] = step1(arm, [t for t in joined if lo <= t['L'] <= hi])

    # ---------------------------------------------------------------- attempt lengths (written lines)
    for arm in arms:
        lab = LABELS[arm]
        res['lines'][arm] = {}
        for lo, hi in BANDS + [(L, L) for L in range(6, 19)]:
            sel = [t for t in joined if lo <= t['L'] <= hi]
            key = f'{lo}-{hi}' if lo != hi else str(lo)
            jl, jL, pl, pL, cl = [], [], [], [], collections.defaultdict(lambda: ([], []))
            for t in sel:
                js = jsamp.get((lab, t['name']))
                if js:
                    jl += js['lines']; jL += [t['L']] * len(js['lines'])
                for p in t['parts']:
                    r = ev[arm].get(p['prompt_ren'])
                    if r:
                        pl += r['lines']; pL += [p['L']] * len(r['lines'])
                for k, pr in t['ctrl'].items():
                    if pr and pr != 'same' and ev[arm].get(pr):
                        cl[k][0].extend(ev[arm][pr]['lines']); cl[k][1].extend([t['L']] * len(ev[arm][pr]['lines']))
            jo, po = [], []
            for t in sel:
                js = jsamp.get((lab, t['name']))
                if js:
                    jo += [(n >= t['L'], ok) for n, ok in zip(js['lines'], js['oks'])]
                for p in t['parts']:
                    r = ev[arm].get(p['prompt_ren'])
                    if r:
                        po += [(n >= p['L'], o[0] == 'ok') for n, o in zip(r['lines'], r['outcome'])]
            cond = lambda xs: (sum(ok for le, ok in xs if le) / max(1, sum(le for le, _ in xs))) if xs else None
            res['lines'][arm].setdefault('cond', {})[key] = {'joined_pass_if_long': cond(jo), 'joined_long_share': (sum(le for le, _ in jo) / len(jo)) if jo else None,
                                                             'parts_pass_if_long': cond(po), 'parts_long_share': (sum(le for le, _ in po) / len(po)) if po else None}
            res['lines'][arm][key] = {'joined': line_stats(jl, jL), 'parts_ren': line_stats(pl, pL),
                                      **{f'ctrl_{k}': line_stats(*v) for k, v in cl.items()}}

    # ---------------------------------------------------------------- step 2: controls (paired with joined)
    for arm in arms:
        lab = LABELS[arm]
        res['step2'][arm] = {}
        for k in ('a', 'b', 'c'):
            for lo, hi in BANDS + [(6, 18)]:
                pj, pc = [], []
                n_same = 0
                for t in joined:
                    if not (lo <= t['L'] <= hi) or t['ctrl'][k] is None:
                        continue
                    js = jsamp.get((lab, t['name']))
                    if js is None:
                        continue
                    if t['ctrl'][k] == 'same':
                        c = js['n_ok'] / js['k']; n_same += 1
                    else:
                        r = ev[arm].get(t['ctrl'][k])
                        if r is None:
                            continue
                        c = r['n_ok'] / r['k']
                    pj.append(js['n_ok'] / js['k']); pc.append(c)
                if not pj:
                    continue
                pj, pc = np.array(pj), np.array(pc)
                res['step2'][arm][f'{k}|{lo}-{hi}'] = {'n': int(len(pj)), 'n_same_as_joined': n_same, 'joined': float(pj.mean()),
                                                       'ctrl': float(pc.mean()), 'diff_ci': boot([pc, pj], lambda c, j: c.mean() - j.mean(), rng)}

    # ---------------------------------------------------------------- strata: premises and atoms
    for arm in arms:
        lab = LABELS[arm]
        st = {}
        for lo, hi in ((6, 9), (6, 18)):
            for key, fn in (('n_prem', lambda t: t['n_prem']), ('n_atoms', lambda t: t['n_atoms'])):
                g = collections.defaultdict(lambda: {'j': [], 'a': [], 'prod': []})
                for t in joined:
                    if not (lo <= t['L'] <= hi):
                        continue
                    js = jsamp.get((lab, t['name']))
                    a = [ev[arm].get(p['prompt_ren']) for p in t['parts']]
                    if js is None or None in a:
                        continue
                    g[fn(t)]['j'].append(js['n_ok'] / js['k'])
                    g[fn(t)]['prod'].append(a[0]['n_ok'] / 32 * a[1]['n_ok'] / 32)
                    ca = t['ctrl']['a']
                    ra = js if ca == 'same' else ev[arm].get(ca) if ca else None
                    if ra:
                        g[fn(t)]['a'].append(ra['n_ok'] / ra['k'])
                st[f'joined|{key}|{lo}-{hi}'] = {str(v): {'n': len(d['j']), 'joined': float(np.mean(d['j'])), 'prod': float(np.mean(d['prod'])),
                                                           'ctrl_a': float(np.mean(d['a'])) if d['a'] else None} for v, d in sorted(g.items())}
        # parts (renamed), by own premise count and atom count, within part-length bands
        for key, fn in (('n_prem', lambda p: p['n_prem']), ('n_atoms', lambda p: len(p['orig_atoms']))):
            for plo, phi in ((2, 4), (5, 7), (8, 10)):
                g = collections.defaultdict(list)
                seen = set()
                for t in joined:
                    for p in t['parts']:
                        if not (plo <= p['L'] <= phi) or p['prompt_ren'] in seen:
                            continue
                        seen.add(p['prompt_ren'])
                        r = ev[arm].get(p['prompt_ren'])
                        if r:
                            g[fn(p)].append(r['n_ok'] / r['k'])
                st[f'parts|{key}|L{plo}-{phi}'] = {str(v): {'n': len(x), 'pass1': float(np.mean(x))} for v, x in sorted(g.items())}
        res['strata'][arm] = st

    # ---------------------------------------------------------------- step 0 (all seeds, all ANDI in the draws)
    allj = read_jsonl(os.path.join(HERE, 'joined_samples.jsonl'))
    armof = lambda l: 'pretrain' if l == 'pretrain' else l[0]
    for a in ('pretrain', 'A', 'B', 'C'):
        res['step0'][a] = {}
        for lo, hi in BANDS:
            cats, prog, nf, ns, nok, nth, lines, Ls = collections.Counter(), collections.Counter(), 0, 0, 0, set(), [], []
            for r in allj:
                if armof(r['label']) != a or not (lo <= r['L_true'] <= hi):
                    continue
                cats.update(r['cats']); prog.update(r['progress'])
                nf += sum(r['cats'].values()); ns += r['k']; nok += r['n_ok']; nth.add(r['name'])
                lines += r['lines']; Ls += [r['L_true']] * len(r['lines'])
            res['step0'][a][f'{lo}-{hi}'] = {'n_theorems': len(nth), 'n_samples': ns, 'pass1': nok / ns if ns else None,
                                             'n_failed': nf, 'cats': {k: v / nf for k, v in cats.most_common()},
                                             'progress': {k: v / nf for k, v in prog.most_common()},
                                             'lines': line_stats(lines, Ls)}
    ex = json.load(open(os.path.join(HERE, 'failloc_examples.json')))
    chosen = []
    for lab, name, pre, cat, desc in EXAMPLES:
        e = next((x for x in ex if x['label'] == lab and x['name'] == name and x['body'].startswith(pre) and x['category'] == cat), None)
        if e:
            chosen.append(dict(e, description=desc))
    res['examples'] = chosen
    dc_txt, dc_res = dc_section(rng)
    res['dc'] = dc_res
    res['dc_md'] = dc_txt
    with open(os.path.join(HERE, 'results.json'), 'w') as f:
        json.dump(res, f, indent=1)
    write_report(res)


def write_report(res):
    arms = res['arms']
    S1, LN, S2, ST, S0 = res['step1'], res['lines'], res['step2'], res['strata'], res['step0']
    out = []
    w = out.append
    w('---\nwritten_on: 2026-09-25\n---\n# ANDI composition diagnosis\n')
    w('Monitor-only analysis for the length-generalisation experiment (`ab3/lengthexp`). Evaluation only, no training. '
      f'Checkpoints: {", ".join(LABELS[a] for a in arms)} (pretrain and one seed per arm); k = 32 at T = 0.8, max_new 1024. '
      + (f'**Missing:** {", ".join(res["missing_arms"])} (their part and control evaluations had not finished). ' if res['missing_arms'] else '')
      + 'Every number below is in `composition/results.json` (regenerate with `python composition/analyse.py`). '
      'Sections 1-5 use the theorem draw of 2026-09-25 00:19 UTC, made from pre-final sealed files (the draws and their '
      'evaluations are archived in `testsets_predraw/` and `eval_predraw/`); those theorems are valid, but 37% of them are not in '
      'the final sealed files. Section 6 (arm D vs C) uses the final draw.\n')
    w('## Conclusion\n')
    w('CONCLUSION_PLACEHOLDER\n')
    # --- step 1
    w('## 1. Joined theorem vs its two parts alone\n')
    w('For each of 100 ANDI theorems per joined length (L6-14 from the matched test draw, L15-18 from the sealed draw; '
      '1,300 in all), both parts were evaluated alone, as they appear in the joined prompt (renamed letters, premises in joined order). '
      '"Product" is the mean over theorems of p1 x p2, the rate expected if the joined proof were just the two part proofs written '
      'one after the other with independent errors; "min" is the mean of the weaker part, a ceiling. Intervals are 95% theorem-bootstrap. '
      'Joined values are the main evaluation\'s results for the same checkpoints (`eval/<label>/`).\n')
    for band in ['6-9', '10-14', '15-18']:
        rows = []
        for a in arms:
            s = S1[a].get(band)
            if not s:
                continue
            rows.append([f'{a} (max {TRAIN_MAX[a]})', s['n'], f"{pct(s['joined'], 2)} {ci_str(*s['joined_ci'], 2)}",
                         f"{pct(s['prod'], 2)} {ci_str(*s['prod_ci'], 2)}", f"{pct(s['min'], 1)}", pct(s['part_mean'], 1),
                         f"{s['ratio']:.2f} [{s['ratio_ci'][0]:.2f}, {s['ratio_ci'][1]:.2f}]" if math.isfinite(s['ratio']) else '-',
                         pct(s['prod_orig'], 2), pct(s['part_mean_orig'], 1)])
        w(f'**Joined L{band}.**\n\n' + md(['arm', 'theorems', 'joined pass@1', 'product of parts', 'min of parts', 'mean part pass@1',
                                           'joined / product', 'product, original parts', 'mean original part'], rows) + '\n')
    rows = []
    for a in arms:
        r = [a]
        for L in range(6, 19):
            s = S1[a].get(str(L))
            r.append(f"{pct(s['joined'], 1)} / {pct(s['prod'], 1)}" if s else '')
        rows.append(r)
    w('**By joined length** (joined pass@1 / product of the parts):\n\n' + md(['arm'] + [f'L{L}' for L in range(6, 19)], rows) + '\n')
    # --- lengths
    w('## 2. How long the attempts are\n')
    w('Written lines = complete lines in a sample (any sample, correct or not). "Share >= L_true" = fraction of samples with at least '
      'as many lines as the theorem\'s L_true (the joined L for joined theorems and controls, the part\'s own L for parts).\n')
    rows = []
    for a in arms:
        for band in ['6-9', '10-14', '15-18']:
            s = LN[a][band]
            j, p = s['joined'], s['parts_ren']
            ca = s.get('ctrl_a')
            rows.append([a, f'L{band}', f"{j['median']:.0f} / {j['p90']:.0f} / {pct(j['share_ge_L'])}" if j else '-',
                         f"{p['median']:.0f} / {p['p90']:.0f} / {pct(p['share_ge_L'])}" if p else '-',
                         f"{ca['median']:.0f} / {ca['p90']:.0f} / {pct(ca['share_ge_L'])}" if ca else '-'])
    w(md(['arm', 'joined length', 'joined attempts: median / p90 / share >= L_true', 'parts alone: median / p90 / share >= own L_true',
          'control (a): median / p90 / share >= L_true'], rows) + '\n')
    rows = []
    for a in arms:
        for band in ['6-9', '10-14', '15-18']:
            c = LN[a]['cond'][band]
            s = S1[a].get(band) or {}
            rows.append([a, f'L{band}', pct(c['joined_long_share']), pct(c['joined_pass_if_long']), pct(s.get('prod'), 1),
                         pct(c['parts_long_share']), pct(c['parts_pass_if_long'])])
    w('Pass rate among attempts that are long enough (written lines >= L_true), pooled over samples:\n\n' + md(
        ['arm', 'joined length', 'joined attempts long enough', 'joined pass@1 among those', 'product of parts (section 1)',
         'part attempts long enough', 'part pass@1 among those'], rows) + '\n')
    rows = []
    for a in arms:
        rows.append([a] + [f"{LN[a][str(L)]['joined']['median']:.0f}" if LN[a][str(L)]['joined'] else '' for L in range(6, 19)])
    w('Median written lines of joined attempts by joined length:\n\n' + md(['arm'] + [f'L{L}' for L in range(6, 19)], rows) + '\n')
    # --- step 2
    w('## 3. Prompt controls on the same theorems\n')
    w('(a) premises grouped by part, the left conjunct\'s part first; (b) conjuncts ordered as the parts\' premises first appear; '
      '(c) both parts in their original letters (possible only where the two original parts used disjoint letters: 18% of the theorems). '
      'A control that coincides with the joined prompt counts with the joined result. Difference = control minus joined, paired by theorem. '
      'At joined L10-18 every control, like the joined theorem, is 0.00% for every arm (`results.json` step2, bands 10-14 and 15-18).\n')
    rows = []
    for a in arms:
        for k in ('a', 'b', 'c'):
            for band in ['6-9', '6-18']:
                s = S2[a].get(f'{k}|{band}')
                if not s:
                    continue
                rows.append([a, k, f'L{band}', s['n'], pct(s['joined'], 2), pct(s['ctrl'], 2),
                             f"{100 * (s['ctrl'] - s['joined']):+.2f} pts {ci_str(*s['diff_ci'], 2)}"])
    w(md(['arm', 'control', 'joined length', 'theorems', 'joined pass@1', 'control pass@1', 'difference [95% CI]'], rows) + '\n')
    # --- strata
    w('## 4. Premise count and atom count\n')
    w('Training data, for comparison: `train_cap6` (pretraining) has 0 / 1 / 2 / 3 premises in 16.3% / 38.3% / 41.0% / 4.4% of records and '
      'never 4 or more; the RL pool `targets_L6-11` has 0-5 premises (4: 6.6%, 5: 0.5%). The joined ANDI theorems here have 2 / 3 / 4 / 5 '
      'premises in 5.8% / 37.5% / 47.1% / 9.6%; their parts have 1-3. Atoms: 90% of joined theorems have 4 atoms, as do 47% of RL targets '
      'and 18% of cap-6 records.\n')
    for a in arms:
        rows = []
        for key in ('n_prem', 'n_atoms'):
            d = ST[a].get(f'joined|{key}|6-9', {})
            for v, s in d.items():
                rows.append([key.replace('n_', '') + f' = {v}', s['n'], pct(s['joined'], 2), pct(s['prod'], 2), pct(s['ctrl_a'], 2)])
        w(f'**{a}, joined L6-9, by the joined theorem\'s premises / atoms.**\n\n' + md(['stratum', 'theorems', 'joined', 'product of parts', 'control (a)'], rows) + '\n')
    rows = []
    for a in arms:
        for key in ('n_prem', 'n_atoms'):
            for band in ('L2-4', 'L5-7', 'L8-10'):
                d = ST[a].get(f'parts|{key}|{band}', {})
                rows.append([a, band, key.replace('n_', ''), ', '.join(f"{v}: {pct(s['pass1'])} (n={s['n']})" for v, s in d.items())])
    w('**Parts alone (renamed), pass@1 by the part\'s own premises / atoms, within part-length bands:**\n\n' + md(['arm', 'part length', 'stratum', 'pass@1 by value'], rows) + '\n')
    # --- step 0
    w('## 5. Where failed joined samples go wrong\n')
    w('All ANDI theorems of the matched draw (L6-14) and the sealed draw (L15-18), all three seeds per arm, from the saved samples '
      '(`composition/failloc.py`). The first rejected line (incremental checker, same verdicts as nd_verify) is placed relative to the two '
      'parts by the atoms it uses: fake goal = writes the joined conclusion with an invalid step; fake conjunct = writes one conjunct with an '
      'invalid step; cross-part citation = a line in one part\'s letters citing a premise or line of the other part; inside left/right part = '
      'an invalid step within one part; mixes parts = a formula combining both parts\' letters; premise block = the premise lines are wrong. '
      '"Progress" = which conjuncts had been validly derived before the error.\n')
    order = ['fake goal', 'fake conjunct', 'cross-part citation', 'inside left part', 'inside right part', 'mixes parts', 'premise block']
    rows = []
    for a in ('pretrain', 'A', 'B', 'C'):
        for band in ['6-9', '10-14', '15-18']:
            s = S0[a][band]
            other = 1 - sum(s['cats'].get(c, 0) for c in order)
            rows.append([a, f'L{band}', s['n_theorems'], pct(s['pass1'], 2)] + [pct(s['cats'].get(c, 0), 0) for c in order] + [pct(other, 0),
                        pct(s['progress'].get('none', 0), 0), pct(s['progress'].get('left only', 0) + s['progress'].get('right only', 0), 0)])
    w(md(['arm', 'joined length', 'theorems', 'pass@1'] + order + ['other (early QED, out of tokens, unattributed)',
          'progress: no conjunct yet', 'progress: one conjunct'], rows) + '\n')
    w('**Five short example failures** (the offending line is marked `<<`):\n')
    for i, e in enumerate(res['examples'], 1):
        lines = [x.strip() + ' ;' for x in e['body'].replace('QED', '').split(';') if x.strip()]
        shown = []
        for j, l in enumerate(lines, 1):
            shown.append(('  ' + l + ('   << ' + e['reason'] if e['line'] == j else '')))
        w(f"{i}. `{e['label']}`, `{e['name']}` (L_true {e['L_true']}; left conjunct `{e['left']['concl']}`, right conjunct "
          f"`{e['right']['concl']}`): {e['description']}\n\n```\n{e['prompt']}\n" + '\n'.join(shown) + '\n  QED\n```\n')
    if res.get('dc_md'):
        w(res['dc_md'])
    w('## Files\n')
    w('- `composition/build.py` → `composition/joined.jsonl`, `composition/prompts.jsonl` (selection seed 20260925; every part and control '
      'proof checked with nd_verify).\n- `composition/run_eval.py` → `composition/eval/<label>/chunk_*.jsonl` (+ samples), logs in `composition/logs/`.\n'
      '- `composition/failloc.py` → `composition/joined_samples.jsonl`, `composition/failloc_examples.json`.\n'
      '- `composition/analyse.py` → `composition/results.json` and this file.\n'
      '- Section 6: `composition/build.py --dir composition/final --lmax 14 --roles part_ren`, `composition/run_eval.py --dir composition/final`, '
      '`composition/failloc.py --dir composition/final --labels C_s101,C_s102,C_s103,D_s101,D_s102,D_s103`.\n')
    txt = '\n'.join(out)
    concl = os.path.join(HERE, 'conclusion.md')
    txt = txt.replace('CONCLUSION_PLACEHOLDER', open(concl).read().strip() if os.path.isfile(concl) else '(to be written)')
    with open(os.path.join(LE, 'report', 'composition.md'), 'w') as f:
        f.write(txt + '\n')
    print('wrote', os.path.join(LE, 'report', 'composition.md'))


if __name__ == '__main__':
    main()
