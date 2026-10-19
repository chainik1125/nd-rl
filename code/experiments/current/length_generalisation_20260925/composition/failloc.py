#!/usr/bin/env python3
"""Composition diagnosis, step 0 (CPU only): where failed samples of joined ANDI theorems go wrong.

    python composition/failloc.py

Reads the saved samples of every evaluated checkpoint (eval/<label>/samples/{matched,std}_L*.jsonl.gz) for every ANDI
theorem in the matched draw (L6-14) and the standard draw (L15-18). Each theorem's two parts are recovered as in
build.py (atoms, premises and conjunct of the left and right part). For each sample: nd_verify verdict, written
lines, and for failed samples the first error (ndcheck) placed relative to the parts:

  premise block        the PR lines are wrong (missing, reordered, extra) or QED before all premises
  fake goal            the offending line writes the joined conclusion with an invalid step
  fake conjunct        the offending line writes one conjunct with an invalid step
  inside left/right    the offending line's formula uses only that part's atoms
  mixes parts          the offending formula uses atoms of both parts (not the goal)
  cross-part citation  the offending line uses one part's atoms but cites a line of the other part
  unattributed         no atoms and no citations to attribute (e.g. malformed)
  early QED after one conjunct / early QED elsewhere / out of tokens
plus progress at the error: which conjuncts had been validly derived at depth 0 before it (none, left, right, both).
Writes composition/joined_samples.jsonl (per label x theorem) and composition/failloc_examples.json.
"""
import collections, gzip, itertools, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
LE = os.path.dirname(HERE)
sys.path.insert(0, LE); sys.path.insert(0, HERE)
sys.dont_write_bytecode = True
import build as B  # noqa: E402
from common import EVAL, TESTSETS, read_jsonl, write_jsonl, write_json  # noqa: E402
from ndcheck import classify, sequent  # noqa: E402

ATOMS = set('PQRS')


def decompose(r, orig):
    prem, concl = B.parse_prompt(r['prompt'])
    cl = B.parse_proof(r['constructed_proof'])
    n = len(prem)
    e1, e2 = sorted(cl[-1][3])
    concl_body = [cl[e1 - 1][1], cl[e2 - 1][1]]
    parsed = [B.parse_prompt(orig[p]['prompt']) for p in r['parts']]
    oas = [sorted(set().union(*[B.atoms(p) for p in op], B.atoms(oc))) for op, oc in parsed]
    cands = []
    for i in range(2):
        cands.append([dict(zip(oas[i], t)) for t in itertools.permutations('PQRS', len(oas[i]))
                      if B.rename(parsed[i][1], dict(zip(oas[i], t))) == concl_body[i]
                      and all(B.rename(p, dict(zip(oas[i], t))) in prem for p in parsed[i][0])])
    for m0, m1 in itertools.product(*cands):
        if set(m0.values()) & set(m1.values()):
            continue
        ps = [[B.rename(p, m0) for p in parsed[0][0]], [B.rename(p, m1) for p in parsed[1][0]]]
        if sorted(map(str, ps[0] + ps[1])) == sorted(map(str, prem)):
            left = 0 if concl_body[0] == concl[1] else 1
            parts = []
            for i, m in enumerate((m0, m1)):
                parts.append({'atoms': set(m.values()), 'prem_idx': sorted(prem.index(p) + 1 for p in ps[i]),
                              'concl': B.fstr(concl_body[i])})
            return {'n_prem': n, 'goal': B.fstr(concl), 'L': parts[left], 'R': parts[1 - left],
                    'n_atoms': len(set().union(*[B.atoms(p) for p in prem], B.atoms(concl)))}
    return None


def split_lines(body):
    toks = body.split()
    if 'QED' in toks:
        toks = toks[:toks.index('QED')]
    out, cur = [], []
    for t in toks:
        cur.append(t)
        if t == ';':
            out.append(cur); cur = []
    return out, cur


def line_parts(toks):
    """(depth, formula string, rule, refs) of one written line, leniently (None fields if malformed)."""
    i = 1 if toks and toks[0].startswith('N') and toks[0][1:].isdigit() else 0
    d = 0
    while i < len(toks) and toks[i] == '|':
        d += 1; i += 1
    j = toks.index(':') if ':' in toks[i:] else None
    f = ' '.join(toks[i:j]) if j is not None else ' '.join(toks[i:-1])
    rule = toks[j + 1] if j is not None and j + 1 < len(toks) else None
    refs = [int(t[1:]) for t in (toks[j + 2:] if j is not None else []) if t.startswith('N') and t[1:].isdigit()]
    return d, f, rule, refs


def place(dec, body, c):
    """First-error category and progress for one failed sample (c = ndcheck.classify result)."""
    lines, partial = split_lines(body)
    L, R = dec['L'], dec['R']
    start = c['start'] or 1
    derived = set()

    def atoms_of(fs):
        return {t for t in fs.split() if t in ATOMS}

    def owner(fs):
        a = atoms_of(fs)
        if a and a <= L['atoms']:
            return 'L'
        if a and a <= R['atoms']:
            return 'R'
        return 'both' if a else None

    def valid_before(p):
        for toks in lines[:p - 1]:
            d, f, rule, refs = line_parts(toks)
            if d == 0 and f == L['concl']:
                derived.add('L')
            if d == 0 and f == R['concl']:
                derived.add('R')
    prog = lambda: ('both' if derived == {'L', 'R'} else 'left only' if derived == {'L'} else 'right only' if derived == {'R'} else 'none')
    kind, p, why = c['kind'], c['line'], c['reason']
    if kind == 'no_qed':
        valid_before(p + 1)
        return 'out of tokens', prog(), None
    if kind == 'early_qed':
        valid_before(p + 1)
        if why == 'premises_incomplete':
            return 'premise block', prog(), None
        d, f, rule, refs = line_parts(lines[-1]) if lines else (0, '', None, [])
        if f in (L['concl'], R['concl']) and d == 0:
            return 'early QED after one conjunct', prog(), None
        return f'early QED ({why})', prog(), None
    # bad_line at position p
    valid_before(p)
    if why == 'premise' or p <= dec['n_prem']:
        return 'premise block', prog(), p
    toks = lines[p - 1] if p - 1 < len(lines) else partial
    d, f, rule, refs = line_parts(toks)
    if f == dec['goal']:
        return 'fake goal', prog(), p
    if f in (L['concl'], R['concl']):
        return 'fake conjunct', prog(), p
    own = owner(f)
    cited_own = set()
    for ref in refs:
        k = ref - start + 1
        if 1 <= k <= dec['n_prem']:
            cited_own.add('L' if k in L['prem_idx'] else 'R')
        elif 1 <= k < p and k - 1 < len(lines):
            o = owner(line_parts(lines[k - 1])[1])
            if o in ('L', 'R'):
                cited_own.add(o)
    if own in ('L', 'R') and cited_own - {own}:
        return 'cross-part citation', prog(), p
    if own == 'L':
        return 'inside left part', prog(), p
    if own == 'R':
        return 'inside right part', prog(), p
    if own == 'both':
        return 'mixes parts', prog(), p
    if len(cited_own) == 2:
        return 'cross-part citation', prog(), p
    if cited_own:
        return f"inside {'left' if cited_own == {'L'} else 'right'} part", prog(), p
    return 'unattributed', prog(), p


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default=HERE)
    ap.add_argument('--labels', default='', help='comma list (default: every label under eval/)')
    ap.add_argument('--eval-root', default=EVAL)
    a = ap.parse_args()
    tests = {r['name']: r for f in ('matched.jsonl', 'std.jsonl') for r in read_jsonl(os.path.join(TESTSETS, f))}
    andi = {n: r for n, r in tests.items() if r['source'] == 'ab3_andi'}
    full = {}
    for f in B.SEALED.values():
        for r in read_jsonl(f):
            if r['name'] in andi:
                full[r['name']] = r
    need = {p for r in full.values() for p in r['parts']}
    orig = {}
    with open(B.RESERVE) as f:
        for l in f:
            if '"orig_name": "ab2g_' in l:
                r = json.loads(l)
                if r['orig_name'] in need:
                    orig[r['orig_name']] = r
    dec = {n: decompose(r, orig) for n, r in full.items()}
    bad = [n for n, d in dec.items() if d is None]
    print(f'{len(andi)} ANDI theorems in the draws, {len(bad)} not decomposed')
    out, examples = [], []
    labels = a.labels.split(',') if a.labels else sorted(os.listdir(a.eval_root))
    for label in labels:
        sd = os.path.join(a.eval_root, label, 'samples')
        if not os.path.isdir(sd):
            continue
        by = collections.defaultdict(list)
        for fn in sorted(os.listdir(sd)):
            if not fn.endswith('.jsonl.gz'):
                continue
            with gzip.open(os.path.join(sd, fn), 'rt') as f:
                for l in f:
                    s = json.loads(l)
                    if s['name'] in dec and dec[s['name']]:
                        by[s['name']].append(s)
        for name, ss in by.items():
            t = tests[name]
            dd = dec[name]
            seq = sequent(t['prompt'])
            lines, oks, cats, progs, n_ok = [], [], collections.Counter(), collections.Counter(), 0
            for s in sorted(ss, key=lambda x: x['i']):
                c = classify(t['prompt'], s['body'], seq)
                lines.append(c['n_lines'])
                oks.append(int(c['ok']))
                if c['ok']:
                    n_ok += 1
                    continue
                cat, pr, pos = place(dd, s['body'], c)
                cats[cat] += 1
                progs[pr] += 1
                if len(s['body'].split()) < 140 and t['L_true'] <= 9 and label.endswith('s101'):
                    examples.append({'label': label, 'name': name, 'L_true': t['L_true'], 'prompt': t['prompt'],
                                     'body': s['body'], 'category': cat, 'progress': pr, 'line': pos,
                                     'kind': c['kind'], 'reason': c['reason'], 'left': {'concl': dd['L']['concl'], 'atoms': sorted(dd['L']['atoms'])},
                                     'right': {'concl': dd['R']['concl'], 'atoms': sorted(dd['R']['atoms'])}})
            out.append({'label': label, 'name': name, 'L_true': t['L_true'], 'set': t['set'], 'k': len(ss), 'n_ok': n_ok,
                        'lines': lines, 'oks': oks, 'cats': dict(cats), 'progress': dict(progs), 'n_prem': dd['n_prem'], 'n_atoms': dd['n_atoms']})
        print(label, len(by), 'theorems', flush=True)
    write_jsonl(os.path.join(a.dir, 'joined_samples.jsonl'), out)
    write_json(os.path.join(a.dir, 'failloc_examples.json'), examples)
    print('wrote', len(out), 'rows,', len(examples), 'example candidates')


if __name__ == '__main__':
    main()
