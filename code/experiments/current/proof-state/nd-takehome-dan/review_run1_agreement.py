#!/usr/bin/env python3
"""Reviewer: step-1 agreement recount (part A, own counters; nd labels re-verified) and re-run of the current translator + Lean
on the negatives, val-36, examples and 1,000-record samples of the early positive files (part B). Outputs artifacts/review_r1/
agreement_recount.json and agreement_rerun.json."""
import sys
PART = sys.argv[1] if len(sys.argv) > 1 else "A"
if PART == "A":
    import json, collections, glob, os, sys, re
    os.chdir(os.path.expanduser('~/review/round2-run1'))
    sys.path.insert(0, '.')
    from nd_verify import verify_text
    out = {}
    kinds = collections.Counter(); kinds_struct = collections.Counter(); reasons_lean = collections.Counter()
    neg_kind = {}
    for l in open('artifacts/r1/negatives.jsonl'):
        r = json.loads(l); neg_kind[r['name']] = r['kind']
    atoms = collections.Counter()
    for fn in sorted(glob.glob('artifacts/r1/lean_*.jsonl')):
        key = os.path.basename(fn)[5:-6]
        c = collections.Counter(); struct = 0; nd_mismatch = 0; n = 0; lreason = collections.Counter(); ex = []
        starts = collections.Counter(); names = set(); dup = 0
        for l in open(fn):
            r = json.loads(l); n += 1
            c[(r['nd_ok'], r['lean_ok'])] += 1
            for t in (r['prompt'] + ' ' + r['proof']).split():
                if re.fullmatch(r'[A-Z]', t) and t not in ('F',): atoms[t] += 1
            m = re.match(r'N(\d+)', r['proof']); starts[int(m.group(1)) if m else -1] += 1
            if (r['name'], r['prompt'], r['proof']) in names: dup += 1
            names.add((r['name'], r['prompt'], r['proof']))
            lr = str(r.get('lean_reason', ''))
            if not r['lean_ok']:
                if lr.startswith('structural'): struct += 1; lreason['structural: ' + lr.split(':', 1)[1].strip()[:30]] += 1
                else: lreason[re.sub(r'\d+', '#', lr[:40])] += 1
                if key == 'negatives': (kinds_struct if lr.startswith('structural') else kinds)[neg_kind.get(r['name'], '?')] += 1
            # re-verify nd label (all records)
            ok, reason, _ = verify_text(r['prompt'] + ' ' + r['proof'])
            if bool(ok) != bool(r['nd_ok']): nd_mismatch += 1
            if r['nd_ok'] != r['lean_ok'] and len(ex) < 5: ex.append({'nd': r['nd_reason'], 'lean': lr[:100], 'proof': r['proof'][:160]})
        out[key] = {'n': n, 'both_accept': c[(True, True)], 'nd_only': c[(True, False)], 'lean_only': c[(False, True)], 'both_reject': c[(False, False)],
                    'lean_rejects_structural': struct, 'nd_label_mismatch_on_reverify': nd_mismatch, 'lean_reject_reasons': dict(lreason.most_common(12)),
                    'start_index_dist': dict(sorted(starts.items())[:5]) | {'n_start_gt1': sum(v for k, v in starts.items() if k > 1)}, 'dup_records': dup, 'examples': ex}
        print(key, out[key]['n'], out[key]['both_accept'], out[key]['nd_only'], out[key]['lean_only'], out[key]['both_reject'], 'struct', struct, 'ndmis', nd_mismatch, 'start>1', out[key]['start_index_dist']['n_start_gt1'], 'dup', dup, flush=True)
    out['negatives_by_kind'] = {'lean_rejected_by_lean': dict(kinds), 'rejected_by_translator_structural_check': dict(kinds_struct), 'all_kinds': dict(collections.Counter(neg_kind.values()))}
    out['atoms'] = dict(atoms)
    json.dump(out, open(os.path.expanduser('~/nd-takehome/artifacts/review_r1/agreement_recount.json'), 'w'), indent=1)
    print(out['negatives_by_kind']); print(out['atoms'])
else:
    import json, os, sys, random, collections, time
    os.chdir(os.path.expanduser('~/review/round2-run1')); sys.path.insert(0, '.')
    from nd2lean import translate, lean_check, TranslationError
    from nd_verify import verify_text
    out = {}
    for key, n in (('negatives', None), ('val36_ref', None), ('examples', None), ('heldout', 1000), ('transfer', 1000), ('rl_targets', 1000), ('train10k', 1000)):
        rs = [json.loads(l) for l in open(f'artifacts/r1/lean_{key}.jsonl')]
        if n: rs = random.Random(7).sample(rs, n)
        t0 = time.time(); srcs = []; idx = []; res = {}
        for k, r in enumerate(rs):
            try: srcs.append(translate(r['prompt'], r['proof'])); idx.append(k)
            except TranslationError as e: res[k] = (False, 'structural: ' + str(e))
        for k, (ok, msg) in zip(idx, lean_check(srcs, 40)): res[k] = (ok, msg)
        c = collections.Counter((r['nd_ok'], res[k][0]) for k, r in enumerate(rs))
        mism = sum(1 for k, r in enumerate(rs) if bool(r['lean_ok']) != bool(res[k][0]))
        out[key] = {'n': len(rs), 'both_accept': c[(True, True)], 'nd_only': c[(True, False)], 'lean_only': c[(False, True)], 'both_reject': c[(False, False)], 'lean_verdict_differs_from_file': mism, 'seconds': round(time.time() - t0)}
        print(key, out[key], flush=True)
        json.dump(out, open(os.path.expanduser('~/nd-takehome/artifacts/review_r1/agreement_rerun.json'), 'w'), indent=1)
