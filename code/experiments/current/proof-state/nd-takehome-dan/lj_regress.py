#!/usr/bin/env python3
"""Acceptance tests 2, 3, 4 and 6 of run lean-judge: the Lean-only judge against stored samples.

Sub-commands (each writes one json under artifacts/lj/ and prints a table):

  collect      build the two corpora from stored run artifacts, deduplicated, -> artifacts/lj/corpus_*.jsonl
                 accepted : distinct (prompt, ND proof) the OLD gate COUNTED (Lean AND nd_verify), from the found_*.jsonl
                            of every Lean-gated expert-iteration/GRPO arm and the `proofs[]` of every Lean-gated
                            coverage record (test 2)
                 leanonly : distinct (prompt, literal Lean text, ND proof) from every *.disagree.jsonl, i.e. the
                            nd_rej & lean_ok class the old gate did NOT count (test 3)
  judge        run lean_judge on a corpus: losses (test 2) / acceptances (test 3), and n_lines vs nd_verify (test 4)
  classify     classify the leanonly corpus by why nd_verify rejected (test 3)
  normform     pitfall 3: Lean's verdict on the literal sampled text vs on nd2lean(nd) vs on nd2lean(norm(nd))
  throughput   test 6: registry-hit path and fallback path, seconds per 1,000 strings, old path beside it

`nd_verify` is imported ONLY by `judge --compare_nd` and `throughput`, to reproduce the old path's numbers and to check
line-count agreement.  It judges nothing here.
"""
import os, re, sys, json, time, glob, random, argparse, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lean_judge, nd2lean
from normalize import norm

OUT = 'artifacts/lj'
RUNS = '/home/dan/work'


def w(fn, obj):
    os.makedirs(os.path.dirname(fn) or '.', exist_ok=True)
    json.dump(obj, open(fn, 'w'), indent=1)
    print('->', fn)


def read_jsonl(fn):
    for l in open(fn):
        if l.strip():
            yield json.loads(l)


# ------------------------------------------------------------------ collect
def cmd_collect(a):
    gate_dirs = set(os.path.dirname(g) for g in glob.glob(f'{RUNS}/*/artifacts/*/gate_*.jsonl'))
    # --- accepted: found_*.jsonl of Lean-gated arms
    seen = set(); n_read = 0; src = collections.Counter()
    fo = open(f'{OUT}/corpus_accepted.jsonl', 'w')
    for f in sorted(glob.glob(f'{RUNS}/*/artifacts/*/*/found*.jsonl')):
        if os.path.dirname(os.path.dirname(f)) not in gate_dirs:
            continue
        for r in read_jsonl(f):
            n_read += 1
            if not r['proof'].lstrip().startswith('N'):
                # run `efficiency` (artifacts/lo) stores the LITERAL LEAN TEXT in `proof`, not an ND proof, in both its
                # free-form and its lean_seq arms.  Those records are not lean_seq/ND samples: the ND judge cannot
                # translate them and `nd_verify` rejects all of them, so the old lean_seq gate never counted them
                # either.  They belong to `lean_check`'s domain.  1,195 of them; excluded here and counted separately.
                src['not_nd_excluded'] += 1
                continue
            k = (r['prompt'], r['proof'])
            if k in seen:
                continue
            seen.add(k)
            src['found'] += 1
            fo.write(json.dumps({'prompt': r['prompt'], 'nd': r['proof'], 'origin': 'found', 'file': f[len(RUNS) + 1:]}, ensure_ascii=False) + '\n')
    # --- accepted: proofs[] of Lean-gated coverage records
    for f in sorted(glob.glob(f'{RUNS}/*/artifacts/*/*.s*.jsonl')):
        try:
            if not json.loads(open(f).readline()).get('lean'):
                continue
        except Exception:
            continue
        for r in read_jsonl(f):
            for p in r['proofs']:
                n_read += 1
                k = (r['prompt'], p['proof'])
                if k in seen:
                    continue
                seen.add(k)
                src['coverage'] += 1
                fo.write(json.dumps({'prompt': r['prompt'], 'nd': p['proof'], 'lean_text': p.get('lean_text'),
                                     'origin': 'coverage', 'file': f[len(RUNS) + 1:]}, ensure_ascii=False) + '\n')
    fo.close()
    # --- leanonly: the disagreement files
    seen2 = set(); n2 = 0
    fo = open(f'{OUT}/corpus_leanonly.jsonl', 'w')
    for f in sorted(glob.glob(f'{RUNS}/*/artifacts/**/*.disagree.jsonl', recursive=True)):
        for r in read_jsonl(f):
            n2 += 1
            k = (r['prompt'], r['lean_text'], r['nd'])
            if k in seen2:
                continue
            seen2.add(k)
            fo.write(json.dumps({'prompt': r['prompt'], 'lean_text': r['lean_text'], 'nd': r['nd'],
                                 'nd_ok': r['nd_ok'], 'lean_ok': r['lean_ok'], 'file': f[len(RUNS) + 1:]}, ensure_ascii=False) + '\n')
    fo.close()
    rec = {'accepted_records_read': n_read, 'accepted_distinct': len(seen), 'accepted_by_origin': dict(src),
           'disagree_records_read': n2, 'leanonly_distinct': len(seen2)}
    print(json.dumps(rec, indent=1))
    w(f'{OUT}/collect.json', rec)


# ------------------------------------------------------------------ judge
def cmd_judge(a):
    recs = [r for r in read_jsonl(a.inp) if r.get('nd')]
    n_skipped = sum(1 for _ in read_jsonl(a.inp)) - len(recs)
    if a.limit and len(recs) > a.limit:
        random.Random(a.seed).shuffle(recs)
        recs = recs[:a.limit]
    t0 = time.time()
    res = lean_judge.judge_many([(r['prompt'], r['nd']) for r in recs])
    wall = time.time() - t0
    n_ok = sum(1 for ok, _, _ in res if ok)
    reasons = collections.Counter(reason.split(':')[0] for ok, reason, _ in res if not ok)
    bad = [{'prompt': r['prompt'], 'nd': r['nd'], 'reason': reason, 'file': r.get('file')}
           for r, (ok, reason, _) in zip(recs, res) if not ok]
    rec = {'corpus': a.inp, 'n': len(recs), 'skipped_no_nd': n_skipped, 'accepted': n_ok, 'rejected': len(recs) - n_ok,
           'rate': n_ok / max(len(recs), 1), 'reject_reasons': dict(reasons.most_common()),
           'lean_wall_s': wall, 'per_1000_s': 1000 * wall / max(len(recs), 1), 'judge_stats': lean_judge.stats()}
    if a.compare_nd:
        from nd_verify import verify_text as ndv
        t1 = time.time()
        nd = [ndv(r['prompt'] + ' ' + r['nd']) for r in recs]
        rec['nd_verify_wall_s'] = time.time() - t1
        c = collections.Counter((bool(x[0]), bool(y[0])) for x, y in zip(nd, res))
        rec['table_nd_lean'] = {f'nd={k[0]},lean={k[1]}': v for k, v in sorted(c.items())}
        both = [(x, y) for x, y in zip(nd, res) if x[0] and y[0]]
        rec['both_accept'] = len(both)
        rec['n_lines_agree'] = sum(1 for x, y in both if x[2] == y[2])
        rec['n_lines_disagree_examples'] = [{'nd_lines': x[2], 'lean_lines': y[2]} for x, y in both if x[2] != y[2]][:10]
    print(json.dumps({k: v for k, v in rec.items() if k != 'judge_stats'}, indent=1))
    w(a.out, rec)
    if bad:
        with open(a.out.replace('.json', '') + '.rejected.jsonl', 'w') as f:
            for x in bad[:5000]:
                f.write(json.dumps(x, ensure_ascii=False) + '\n')
        print(f'{len(bad)} rejected; first 5000 -> {a.out.replace(".json","")}.rejected.jsonl')


# ------------------------------------------------------------------ classify the Lean-only class
NEG = re.compile(r'^\( ~ (.*) \)$')


def parse_lines(nd):
    """ND body -> list of (idx, depth, formula string, rule, refs)."""
    out = []
    for ln in nd.split(' ; '):
        t = ln.split()
        if not t or t[0] == 'QED':
            continue
        m = re.match(r'^N(\d+)((?: \|)*) (.*) : ([A-Z0-9]+)(.*)$', ln)
        if not m:
            return None
        out.append((int(m.group(1)), m.group(2).count('|'), m.group(3).strip(), m.group(4), m.group(5).split()))
    return out


def classify_leanonly(prompt, nd):
    """Why nd_verify rejected a proof Lean accepts.  Categories are read off the proof, not guessed."""
    if nd is None:
        # run `efficiency`'s gate logged `kind: no-denotation`: Lean accepted a literal text whose strict-grammar decode
        # failed, so it has no ND denotation.  The grammar IS the allowlist (decision note, 2026-09-27), so such a text is
        # not counted by the lean_seq judge either -- these are excluded from test 3 rather than expected to be accepted.
        return 'no ND denotation (outside the lean_seq grammar; excluded)'
    lines = parse_lines(nd)
    if lines is None:
        return 'unparsed'
    form = {i: f for i, d, f, r, refs in lines}
    prem = [p.strip() for p in prompt[len('THM '):].split(' SEQ ')[0].split(' , ')] if ' SEQ ' in prompt else []
    prem = [p for p in prem if p]
    n_pr = sum(1 for i, d, f, r, refs in lines if r == 'PR')
    for i, d, f, r, refs in lines:
        if r == 'BOTE':
            cited = form.get(int(refs[0][1:])) if refs and refs[0].startswith('N') else None
            if cited is not None and cited != 'F':
                return 'BOTE on a non-F line (Not.elim)'
        if r == 'NEGE' and len(refs) == 2:
            a, b = (form.get(int(x[1:])) for x in refs)
            if a is not None and b is not None:
                na, nb = NEG.match(a), NEG.match(b)
                pair = {a, b}
                if not ((na and na.group(1) == b) or (nb and nb.group(1) == a)):
                    return 'NEGE on A and A>F (Lean: ~A is A→False)'
    if n_pr < len(prem):
        return 'omitted premise re-statement'
    return 'other'


def cmd_classify(a):
    recs = list(read_jsonl(a.inp))
    c = collections.Counter(); ex = {}
    for r in recs:
        k = classify_leanonly(r['prompt'], r['nd'])
        c[k] += 1
        ex.setdefault(k, {'prompt': r['prompt'], 'nd': r['nd'], 'lean_text': r.get('lean_text')})
    rec = {'corpus': a.inp, 'n': len(recs), 'classes': dict(c.most_common()),
           'fractions': {k: v / len(recs) for k, v in c.most_common()}, 'examples': ex}
    print(json.dumps({k: rec[k] for k in ('n', 'classes', 'fractions')}, indent=1))
    w(a.out, rec)


# ------------------------------------------------------------------ pitfall 3
def cmd_normform(a):
    """Lean's verdict on (i) the literal sampled text, (ii) nd2lean(nd), (iii) nd2lean(norm(nd)) -- must agree."""
    from lean_gate import check_sources, lean_check
    from lean_tok import LeanTokenizer
    tk = LeanTokenizer('lean_seq')
    recs = [r for r in read_jsonl(a.inp) if r.get('lean_text') and r.get('nd')]
    if a.limit and len(recs) > a.limit:
        random.Random(a.seed).shuffle(recs); recs = recs[:a.limit]
    lit, raw, nrm, keep = [], [], [], []
    for r in recs:
        try:
            s_raw = nd2lean.translate(r['prompt'], r['nd'], require_all_pr=False)
            s_nrm = nd2lean.translate(r['prompt'], norm(r['nd']), require_all_pr=False)
        except Exception:
            continue
        keep.append(r); lit.append((tk.statement(r['prompt']), r['lean_text'])); raw.append(s_raw); nrm.append(s_nrm)
    a_lit, _, _ = lean_check(lit)
    a_raw, _, _ = check_sources(raw)
    a_nrm, _, _ = check_sources(nrm)
    c = collections.Counter(zip(map(bool, a_lit), map(bool, a_raw), map(bool, a_nrm)))
    dis = [{'prompt': k['prompt'], 'nd': k['nd'], 'lean_text': k['lean_text'], 'lit': bool(x), 'raw': bool(y), 'norm': bool(z)}
           for k, x, y, z in zip(keep, a_lit, a_raw, a_nrm) if not (bool(x) == bool(y) == bool(z))]
    rec = {'corpus': a.inp, 'n_with_text': len(recs), 'n_translatable': len(keep),
           'table_literal_raw_norm': {f'lit={k[0]},nd2lean={k[1]},nd2lean_norm={k[2]}': v for k, v in sorted(c.items())},
           'n_disagreeing': len(dis), 'disagreeing': dis[:20]}
    print(json.dumps({k: rec[k] for k in ('n_with_text', 'n_translatable', 'table_literal_raw_norm', 'n_disagreeing')}, indent=1))
    w(a.out, rec)


# ------------------------------------------------------------------ test 6
def cmd_throughput(a):
    from nd_verify import verify_text as ndv
    recs = list(read_jsonl(a.inp))
    random.Random(a.seed).shuffle(recs)
    recs = recs[:a.limit]
    pairs = [(r['prompt'], r['nd']) for r in recs]
    # OLD in-loop path on these strings: the gate's Lean run on the literal text + nd_verify on every ND string.
    t0 = time.time(); [ndv(p + ' ' + nd) for p, nd in pairs]; t_nd = time.time() - t0
    # NEW fallback path (cold registry): nd2lean + batched Lean
    lean_judge._registry.clear()
    t0 = time.time(); lean_judge.judge_many(pairs); t_fb = time.time() - t0
    # NEW hot path: every verdict now registered (this is what a loop sees after generate())
    t0 = time.time(); lean_judge.judge_many(pairs); t_hit = time.time() - t0
    # marker path
    t0 = time.time(); lean_judge.judge_many([(p, 'LEANPARSE no-eos') for p, _ in pairs]); t_mk = time.time() - t0
    n = len(pairs)
    rec = {'corpus': a.inp, 'n': n, 'cpus': os.cpu_count(),
           'old_nd_verify_s': t_nd, 'old_nd_verify_per_1000_s': 1000 * t_nd / n,
           'new_registry_hit_s': t_hit, 'new_registry_hit_per_1000_s': 1000 * t_hit / n,
           'new_marker_s': t_mk, 'new_marker_per_1000_s': 1000 * t_mk / n,
           'new_fallback_s': t_fb, 'new_fallback_per_1000_s': 1000 * t_fb / n,
           'note': 'the old in-loop path = one Lean run on the literal texts (unchanged, still run by the gate) PLUS '
                   'old_nd_verify_s; the new in-loop path = the same Lean run and then registry hits, so the new judge is '
                   'faster by old_nd_verify_s - new_registry_hit_s.  The fallback is only for strings the gate never saw.'}
    print(json.dumps(rec, indent=1))
    w(a.out, rec)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest='cmd', required=True)
    c = sp.add_parser('collect'); c.set_defaults(fn=cmd_collect)
    for name, fn in (('judge', cmd_judge), ('classify', cmd_classify), ('normform', cmd_normform), ('throughput', cmd_throughput)):
        c = sp.add_parser(name)
        c.add_argument('--in', dest='inp', required=True)
        c.add_argument('--out', required=True)
        c.add_argument('--limit', type=int, default=None)
        c.add_argument('--seed', type=int, default=0)
        if name == 'judge':
            c.add_argument('--compare_nd', action='store_true')
        c.set_defaults(fn=fn)
    a = ap.parse_args()
    a.fn(a)
