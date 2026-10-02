#!/usr/bin/env python3
"""Write the final mid-training files from build/raw + labels.
sft_long / sft_short: the train_cap6 record format (name, thm, key, prompt, n_lines = written proof length, rules,
  n_prem, proof, text) + pkey, family, parts, part_L, part_pkeys, constructed_lines, n_tokens, and the exactness check:
  exact_checked (bool), exact (True / False / None = not checked or unresolved), exact_L (minlen label when resolved).
test_long: name, thm, key, pkey, prompt, n_lines = L_true = constructed length, label_kind, cert14, source, family,
  n_prem, parts, part_L, part_pkeys, constructed_proof, n_tokens (monitor-only; no `proof` field on purpose)."""
import json, os, collections
B = '/scratch/cowifornia/nd-rl/ab3/midtrain/build'; O = '/scratch/cowifornia/nd-rl/ab3/midtrain/data'
rd = lambda fn: [json.loads(l) for l in open(fn) if l.strip()]


def labels(*fns):
    out = {}
    for fn in fns:
        if os.path.exists(f'{B}/labels/{fn}'):
            for x in rd(f'{B}/labels/{fn}'):
                p = out.get(x['name'])
                if p is None or p['min_lines_ub'] is None:
                    out[x['name']] = x
    return out


def main():
    summ = {}
    ex = labels('sft_short_exact.jsonl', 'sft_long_exact.jsonl')
    SFT = ('name', 'thm', 'key', 'prompt', 'n_lines', 'rules', 'n_prem', 'proof', 'text', 'pkey', 'family', 'parts', 'part_L', 'part_pkeys',
           'constructed_lines', 'n_tokens')
    for fn in ('sft_long', 'sft_short'):
        rs = rd(f'{B}/raw/{fn}.jsonl'); c = collections.Counter()
        with open(f'{O}/{fn}.jsonl', 'w') as f:
            for r in rs:
                out = {k: r[k] for k in SFT}
                x = ex.get(r['name'])
                out['exact_checked'] = x is not None
                if x is None:
                    out['exact'] = None; out['exact_L'] = None; c[(r['n_lines'], 'not_checked')] += 1
                elif x['min_lines_ub'] is None:
                    out['exact'] = None; out['exact_L'] = None; c[(r['n_lines'], 'unresolved')] += 1
                else:
                    assert x['min_lines_ub'] <= r['n_lines']
                    out['exact'] = x['min_lines_ub'] == r['n_lines']; out['exact_L'] = x['min_lines_ub']
                    c[(r['n_lines'], 'exact' if out['exact'] else 'shorter_exists')] += 1
                f.write(json.dumps(out) + '\n')
        summ[fn] = {f'{L}_{k}': v for (L, k), v in sorted(c.items())}
    cert = labels('test_long_cert14.jsonl', 'test_long_cert14_retry.jsonl')
    rs = rd(f'{B}/raw/test_long.jsonl'); c = collections.Counter()
    with open(f'{O}/test_long.jsonl', 'w') as f:
        for r in rs:
            x = cert.get(r['name'])
            ok = x is not None and x['bound'] == 14 and x['min_lines_ub'] is None and not any(x.get(k) for k in ('timeout', 'oom', 'lost', 'error'))
            assert x is None or x['min_lines_ub'] is None, ('proof <= 14 found!', r['name'])
            kind = 'constructed_additive+cert14' if ok else 'constructed_additive'
            c[(r['n_lines'], kind)] += 1
            f.write(json.dumps({'name': r['name'], 'thm': r['thm'], 'key': r['key'], 'pkey': r['pkey'], 'prompt': r['prompt'], 'n_lines': r['n_lines'],
                                'L_true': r['n_lines'], 'label_kind': kind, 'cert14': ok, 'source': 'ab3_mid_andi', 'family': r['family'],
                                'n_prem': r['n_prem'], 'parts': r['parts'], 'part_L': r['part_L'], 'part_pkeys': r['part_pkeys'],
                                'constructed_proof': r['proof'], 'n_tokens': r['n_tokens']}) + '\n')
    summ['test_long'] = {f'{L}_{k}': v for (L, k), v in sorted(c.items())}
    json.dump(summ, open(f'{B}/assemble_summary.json', 'w'), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == '__main__':
    main()
