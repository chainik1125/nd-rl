#!/usr/bin/env python3
"""Independent checks of ab3/midtrain/data:
  * key / pkey recomputed from thm; names unique; the three files pairwise pkey-disjoint;
  * no theorem key / pkey in any existing file (recomputed here from the sources): cap-6 train / held-out, textbook-dev
    (thm only), validation-36, every run-2 file incl. reserve and sealed, run-1 sealed, ladder pools, ab3/sealed/*,
    ab3/data_comp/*;
  * part disjointness by reserve name and by part pkey recomputed from thm: midtrain parts vs every part of ab3/sealed
    and ab3/data_comp theorems; sft_long + sft_short parts vs test_long parts;
  * nd_verify of every proof (sft: proof, test: constructed_proof), lines == n_lines; the last k - 1 lines are the
    ANDI joins and the k joined part conclusions have pairwise disjoint atoms;
  * minlen proofs behind every exact check re-verified; token lengths (frozen tokenizer, abs) reported."""
import json, sys, glob, collections, multiprocessing as mp
sys.path.insert(0, '/scratch/cowifornia/nd-rl/ab3/data_build/tools'); sys.path.insert(0, '/scratch/cowifornia/nd-rl/external/nd-takehome')
sys.path.insert(0, '/scratch/cowifornia/nd-rl/ab3/frozen/lib')
from keys2 import pkey, canon_key
from nd_verify import verify_text
from comp import parse_proof, atoms_of
from tokenizer import Tokenizer
import excl as EX
O = '/scratch/cowifornia/nd-rl/ab3/midtrain/data'; B = '/scratch/cowifornia/nd-rl/ab3/midtrain/build'; DB = '/scratch/cowifornia/nd-rl/ab3/data_build'
rd = lambda fn: [json.loads(l) for l in open(fn) if l.strip()]


def kp(t):
    t = t.strip(); return canon_key(t), pkey(t)


def chk(job):
    prompt, proof, n = job
    ok, reason, nl = verify_text(prompt + ' ' + proof)
    return ok and nl == n


def main():
    F = {k: rd(f'{O}/{k}.jsonl') for k in ('sft_long', 'sft_short', 'test_long')}
    names = [r['name'] for rs in F.values() for r in rs]; assert len(names) == len(set(names))
    seenp = {}
    for k, rs in F.items():
        for r in rs:
            assert r['key'] == canon_key(r['thm']) and r['pkey'] == pkey(r['thm'])
            assert r['pkey'] not in seenp, (k, seenp.get(r['pkey'])); seenp[r['pkey']] = k
    src = list(EX.FILES.values()) + [EX.VAL36] + sorted(glob.glob('/scratch/cowifornia/nd-rl/ab3/sealed/*.jsonl')) + \
        ['/scratch/cowifornia/nd-rl/ab3/data_comp/targets_L6-11.jsonl', '/scratch/cowifornia/nd-rl/ab3/data_comp/targets_constructed_L6-11.jsonl']
    allk = {r['key'] for rs in F.values() for r in rs}
    with mp.Pool(8) as pool:
        for fn in src:
            ks = pool.map(kp, [json.loads(l)['thm'] for l in open(fn) if l.strip()], chunksize=2000)
            hit = sum(1 for a, b in ks if a in allk or b in seenp)
            assert hit == 0, (fn, hit)
    res = {}
    for l in open('/scratch/cowifornia/nd-rl/ab2/data/reserve.jsonl'):
        r = json.loads(l); res[r['orig_name']] = r['thm']
    old_parts = {n for fn in glob.glob('/scratch/cowifornia/nd-rl/ab3/sealed/*.jsonl') for r in rd(fn) for n in r['parts']}
    old_parts |= {n for p in rd(f'{DB}/comp/provenance_constructed.jsonl') for n in p['parts']}
    oldp = {pkey(res[n]) for n in old_parts}
    part = {k: {n for r in rs for n in r['parts']} for k, rs in F.items()}
    partp = {k: {pkey(res[n]) for n in v} for k, v in part.items()}
    for k in F:
        assert not (part[k] & old_parts) and not (partp[k] & oldp), k
        for r in F[k]:
            assert [pkey(res[n]) for n in r['parts']] == r['part_pkeys']
    sft_n, sft_p = part['sft_long'] | part['sft_short'], partp['sft_long'] | partp['sft_short']
    assert not (sft_n & part['test_long']) and not (sft_p & partp['test_long'])
    tok = Tokenizer('abs'); jobs = []; mx = collections.Counter(); ntok = {}
    for k, rs in F.items():
        for r in rs:
            pf = r['proof'] if k != 'test_long' else r['constructed_proof']
            jobs.append((r['prompt'], pf, r['n_lines']))
            t = len(tok.encode_prompt(r['prompt'])) + len(tok.encode_proof(pf)); assert t == r['n_tokens']
            mx[k] = max(mx[k], t); ntok[f'{k}_prompt_max'] = max(ntok.get(f'{k}_prompt_max', 0), len(tok.encode_prompt(r['prompt'])))
            kk = int(r['family'][-1]); ls = parse_proof(pf)
            assert all(x[2] == 'ANDI' and x[0] == 0 for x in ls[-(kk - 1):])
            joined = set(i for x in ls[-(kk - 1):] for i in x[3]) - set(range(len(ls) - kk + 2, len(ls) + 1))
            assert len(joined) == kk, (r['name'], joined)
            ats = [atoms_of(ls[i - 1][1]) for i in sorted(joined)]
            assert all(not (ats[i] & ats[j]) for i in range(kk) for j in range(i + 1, kk)), r['name']
    ex = {}
    for fn in ('sft_short_exact.jsonl', 'sft_long_exact.jsonl'):
        try:
            for x in rd(f'{B}/labels/{fn}'):
                if x['min_lines_ub'] is not None:
                    ex[x['name']] = x
        except FileNotFoundError:
            pass
    for k in ('sft_long', 'sft_short'):
        for r in F[k]:
            if r['exact_L'] is not None:
                x = ex[r['name']]; assert x['min_lines_ub'] == r['exact_L']
                jobs.append((r['prompt'], x['proof'], r['exact_L']))
    with mp.Pool(16) as pool:
        ok = pool.map(chk, jobs, chunksize=200)
    assert all(ok)
    print(json.dumps({'records': {k: len(v) for k, v in F.items()}, 'proof_checks': len(ok), 'failures': ok.count(False),
                      'max_prompt_plus_proof_tokens': dict(mx), **ntok,
                      'distinct_parts': {k: len(v) for k, v in part.items()}, 'old_test_parts': len(old_parts)}, indent=1))
    print('ALL CHECKS PASS')


if __name__ == '__main__':
    main()
