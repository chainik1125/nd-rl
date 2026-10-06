#!/usr/bin/env python3
"""Reviewer's independent recount for round2-run3 (minimum outside data for ignition).

Shares with the executor only `nd_verify`. The proof parser, dependency pruning, start-index normaliser,
box-depth counter, reductio predicate and atom-renaming key are the reviewer's own (review_run5_recount.py,
written for the run-5 review, reused here).

  python3 review_run3_recount.py            # everything -> artifacts/review_r3/recount.json (+ printed tables)

What is recounted
  A. the injected records of all 30 manifests: verifier verdict, pattern by my predicate, whether the injected
     theorem is a target / transfer / held-out / val-36 / Stage-1 training class; for the invalid strings, how many
     single-token edits of a citation make them valid (distance to a valid proof); sibling-source counts.
  B. every arm x condition: per-round cumulative pattern theorems (rounds 4..8, round 4 = the parent's resumed
     found_4), the same excluding the injected theorems and counting only proofs first found in rounds >= 5,
     solved counts vs the executor's round_<r>.json, ignition round at thresholds 10 / 20 / 40 (depth-3) and
     6 / 12 / 24 (reductio), transfer-pool pattern theorems, per-round file consistency, and nd_verify on every
     distinct pattern proof plus a sample of the others.
  C. the parents' own trajectories (rounds 1..8) and the ignition study's full-sibling injection arms (ivS).
"""
import json, re, os, sys, glob, collections, random, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nd_verify import verify_text
from review_run5_recount import parse_proof, prune, normalise, p_reductio, max_depth, rkey, thm_of_prompt, rd

ARMS = [('depth3', 'ei_depth3_f0_a1_s4'), ('depth3', 'ei_depth3_f0_a1_s5'), ('depth3', 'ei_depth3_f0_a1_s3'),
        ('reductio', 'ei_reductio_f0_s1_t2'), ('reductio', 'ei_reductio_f0_s2_t2')]
CONDS = ['sib1', 'sib4', 'sib16', 'gen4', 'other4', 'inv4']
POOLS = {'depth3': ('data/p2/targets_depth3.jsonl', 'data/p2/transfer_depth3.jsonl', 'data/p2/train_depth3_f0_a1.jsonl'),
         'reductio': ('data/p2/targets_reductio2.jsonl', 'data/p2/transfer_reductio2.jsonl', 'data/p2/train_reductio_f0.jsonl')}
THRESH = {'depth3': (10, 20, 40), 'reductio': (6, 12, 24)}
OUT = 'artifacts/review_r3'
os.makedirs(OUT, exist_ok=True)


def pat_of(pruned_lines, pattern):
    return max_depth(pruned_lines) >= 3 if pattern == 'depth3' else p_reductio(pruned_lines)


_cache = {}


def classify(proof):
    """-> {'depth3','reductio','depth','written','pruned'} on the dependency-pruned proof, or None."""
    if proof in _cache:
        return _cache[proof]
    lines = parse_proof(proof)
    if lines is None:
        _cache[proof] = None
        return None
    pr = prune(lines)
    c = {'depth3': max_depth(pr) >= 3, 'reductio': p_reductio(pr), 'depth': max_depth(pr), 'written': len(lines), 'pruned': len(pr),
         'depth3_unpruned': max_depth(lines) >= 3, 'reductio_unpruned': p_reductio(lines)}
    _cache[proof] = c
    return c


def keyset(fn, field=None):
    out = set()
    for r in rd(fn):
        thm = r['thm'].strip() if field == 'thm' else thm_of_prompt(r['prompt'])
        out.add(rkey(thm))
    return out


def one_token_repairs(prompt, proof):
    """Number of single-token substitutions of a citation token (N<i> after the rule) that make the string verify."""
    toks = proof.split()
    idxs = sorted({int(t[1:]) for t in toks if re.fullmatch(r'N\d+', t)})
    # citation positions: N-tokens that follow a rule token (i.e. not the line label at a line start)
    line_start = True; cites = []
    for i, t in enumerate(toks):
        if line_start:
            line_start = False; continue
        if t == ';':
            line_start = True; continue
        if re.fullmatch(r'N\d+', t):
            cites.append(i)
    n_ok = 0; fixes = []
    for i in cites:
        for j in idxs:
            cand = toks[:]; cand[i] = f'N{j}'
            if cand[i] == toks[i]:
                continue
            ok, reason, nl = verify_text(prompt + ' ' + ' '.join(cand))
            if ok:
                n_ok += 1; fixes.append((i, toks[i], f'N{j}'))
    return n_ok, fixes[:3], len(cites)


def part_a(res):
    """Injected records."""
    print('== A. injected records', flush=True)
    evals = {}
    for pattern, (tg, tr, train) in POOLS.items():
        evals[pattern] = {'targets': keyset(tg), 'transfer': keyset(tr)}
        evals[pattern]['target_prompts'] = {r['prompt']: r['name'] for r in rd(tg)}
    evals['heldout'] = keyset('data/p2/heldout.jsonl')
    evals['val36'] = {rkey(json.loads(l)['thm'].strip()) for l in open('targets/validation_36.jsonl')}
    train_keys = {}
    for pattern, (tg, tr, train) in POOLS.items():
        t0 = time.time(); ks = set(); n = 0; pat = 0
        for l in open(train):
            r = json.loads(l); n += 1
            ks.add(rkey(thm_of_prompt(r['prompt'])))
        train_keys[pattern] = ks
        print(f'  train {train}: {n} records, {len(ks)} classes ({time.time()-t0:.0f}s)', flush=True)
    A = {}
    for f in sorted(glob.glob('artifacts/p2/*_manifest.json')):
        m = json.load(open(f))
        arm_cond = os.path.basename(f)[:-len('_manifest.json')]
        arm = arm_cond.rsplit('_', 1)[0]; cond = arm_cond.rsplit('_', 1)[1]
        pattern = 'depth3' if 'depth3' in arm else 'reductio'
        other = 'reductio' if pattern == 'depth3' else 'depth3'
        recs = []
        for x in m['injected']:
            ok, reason, nl = verify_text(x['prompt'] + ' ' + x['proof'])
            c = classify(x['proof'])
            key = rkey(thm_of_prompt(x['prompt']))
            r = {'valid': ok, 'reason': reason, 'n_lines': nl, 'pattern_my': bool(c and c[pattern]), 'other_pattern_my': bool(c and c[other]),
                 'depth': c['depth'] if c else None, 'written': c['written'] if c else None,
                 'is_target': key in evals[pattern]['targets'], 'target_name': evals[pattern]['target_prompts'].get(x['prompt']),
                 'is_transfer': key in evals[pattern]['transfer'], 'is_heldout': key in evals['heldout'], 'is_val36': key in evals['val36'],
                 'in_stage1_set': key in train_keys[pattern], 'in_other_targets': key in evals[other]['targets'] or key in evals[other]['transfer']}
            if not ok:
                n_fix, fixes, n_cites = one_token_repairs(x['prompt'], x['proof'])
                r['one_token_repairs'] = n_fix; r['example_fixes'] = fixes; r['n_citation_tokens'] = n_cites
                # surface tokens: box depth in the raw string / NEGI + DN present
                r['raw_max_bars'] = max((len(seg.split(':')[0].split()) - 1 - len([t for t in seg.split(':')[0].split() if not t == '|' and not t.startswith('N')]) for seg in x['proof'].split(';') if ':' in seg), default=0)
                r['has_negi_dn'] = ' NEGI ' in x['proof'] and ' DN ' in x['proof']
            recs.append(r)
        A[arm_cond] = {'arm': arm, 'cond': cond, 'pattern': pattern, 'inject_file': m['inject_file'], 'init': m['init'], 'n_injected': m['n_injected'],
                       'executor_valid': m['injected_valid'], 'executor_has_pattern': m['injected_has_pattern'], 'own_proofs': m['own_proofs'], 'own_theorems': m['own_theorems'],
                       'retain': m['retain'], 'records': recs,
                       'n_valid_my': sum(r['valid'] for r in recs), 'n_pattern_my': sum(r['pattern_my'] for r in recs),
                       'n_is_target': sum(r['is_target'] for r in recs), 'n_is_transfer': sum(r['is_transfer'] for r in recs),
                       'n_in_stage1_set': sum(r['in_stage1_set'] for r in recs), 'n_heldout_or_val36': sum(r['is_heldout'] or r['is_val36'] for r in recs),
                       'injected_target_names': sorted(r['target_name'] for r in recs if r['target_name'])}
        a = A[arm_cond]
        print(f"  {arm_cond:32s} n {a['n_injected']:2d} valid {a['n_valid_my']} pattern {a['n_pattern_my']} target {a['n_is_target']} transfer {a['n_is_transfer']} "
              f"stage1 {a['n_in_stage1_set']} held/val {a['n_heldout_or_val36']} exec(valid {sum(m['injected_valid'])}, pat {sum(m['injected_has_pattern'])})"
              + (f"  one-token repairs {[r.get('one_token_repairs') for r in recs]}" if cond == 'inv4' else ''), flush=True)
    # sibling sources
    S = {}
    for pattern, fn in (('depth3', 'artifacts/p2/ei_depth3_f0_a1_s2/found_4.jsonl'), ('reductio', 'artifacts/p2/ei_reductio_f0_s7_t2/found_4.jsonl')):
        recs = rd(fn); thms = set(); n_pat = 0; seen = set(); n_dist = 0
        for x in recs:
            k = (x['name'], normalise(x['proof']))
            if k in seen:
                continue
            seen.add(k); n_dist += 1
            c = classify(x['proof'])
            if c and c[pattern]:
                n_pat += 1; thms.add(x['name'])
        S[fn] = {'records': len(recs), 'distinct_norm': n_dist, 'pattern_proofs_distinct': n_pat, 'pattern_theorems': len(thms)}
        print(f'  sibling source {fn}: {S[fn]}', flush=True)
    # injection files themselves
    F = {}
    for fn in sorted(glob.glob('data/r3/*.jsonl')):
        recs = rd(fn); F[fn] = []
        for x in recs:
            ok, reason, nl = verify_text(x['prompt'] + ' ' + x['proof']); c = classify(x['proof'])
            F[fn].append({'valid': ok, 'reason': reason, 'stored_verify': x.get('verify'), 'depth3': bool(c and c['depth3']), 'reductio': bool(c and c['reductio']), 'written': c['written'] if c else None,
                          'key_my': rkey(thm_of_prompt(x['prompt'])), 'key_stored': x.get('key'), 'key_agree': rkey(thm_of_prompt(x['prompt'])) == x.get('key')})
        print(f'  file {fn}: ' + '; '.join(f"valid={r['valid']} d3={r['depth3']} red={r['reductio']} L={r['written']} key_ok={r['key_agree']} reason={r['reason']}" for r in F[fn]), flush=True)
    res['injections'] = A; res['sibling_sources'] = S; res['inject_files'] = F


def recount_dir(d, pattern, pool_file, fnpat, rounds, parent_found=None, injected_names=(), verify_all=True, sample_verify=200, rng=None):
    """Per-round cumulative counts from the per-round found files of one directory."""
    pool = rd(pool_file); names = [r['name'] for r in pool]; prompts = {r['name']: r['prompt'] for r in pool}; n = len(names)
    files = {r: f'{d}/' + fnpat.format(r) for r in rounds}
    for r in rounds:
        assert os.path.exists(files[r]), files[r]
    sets = {r: {(x['name'], normalise(x['proof'])) for x in rd(files[r])} for r in rounds}
    chain_ok = all(sets[rounds[i]] <= sets[rounds[i + 1]] for i in range(len(rounds) - 1))
    last = rd(files[rounds[-1]])
    first_file = {}
    for r in rounds:
        for x in rd(files[r]):
            first_file.setdefault((x['name'], normalise(x['proof'])), r)
    round_field = collections.defaultdict(lambda: 99)
    for x in last:
        k = (x['name'], normalise(x['proof'])); round_field[k] = min(round_field[k], x['round'])
    # resumed records (round field < first round in this dir) must equal the parent's found file
    resumed = {k for k in first_file if round_field[k] < rounds[0]}
    parent_match = None
    if parent_found is not None:
        pset = {(x['name'], normalise(x['proof'])) for x in rd(parent_found)} if os.path.exists(parent_found) else set()
        parent_match = {'resumed': len(resumed), 'parent_file': len(pset), 'equal': resumed == pset, 'resumed_minus_parent': len(resumed - pset), 'parent_minus_resumed': len(pset - resumed)}
    agree = sum(first_file[k] == round_field[k] for k in first_file if k not in resumed); n_new = len(first_file) - len(resumed)
    first = {k: min(first_file[k], round_field[k]) for k in first_file}
    solved_round, acq_round, acq_unp_round = {}, {}, {}
    n_ver = 0; ver_fail = 0; unparsable = 0; unknown = 0; pat_proofs = 0; dist = 0
    other_first = []
    for k, r in first.items():
        name, np_ = k
        if name not in prompts:
            unknown += 1; continue
        c = classify(np_); dist += 1
        if c is None:
            unparsable += 1; continue
        solved_round[name] = min(solved_round.get(name, 99), r)
        if c[pattern]:
            pat_proofs += 1; acq_round[name] = min(acq_round.get(name, 99), r)
            if verify_all:
                ok, reason, nl = verify_text(prompts[name] + ' ' + np_); n_ver += 1; ver_fail += (not ok) or (nl != c['written'])
        else:
            other_first.append((name, np_, c['written']))
        if c[pattern + '_unpruned']:
            acq_unp_round[name] = min(acq_unp_round.get(name, 99), r)
    s_ver = 0; s_fail = 0
    if sample_verify and other_first:
        rng = rng or random.Random(0)
        for name, np_, w in rng.sample(other_first, min(sample_verify, len(other_first))):
            ok, reason, nl = verify_text(prompts[name] + ' ' + np_); s_ver += 1; s_fail += (not ok) or (nl != w)
    R = list(range(rounds[0] - 1, rounds[-1] + 1)) if rounds[0] > 1 else list(rounds)
    inj = set(injected_names)
    cum = lambda dct, cond=lambda nm: True: [sum(1 for nm, v in dct.items() if v <= r and cond(nm)) for r in R]
    out = {'n': n, 'rounds': R, 'chain_ok': chain_ok, 'parent_match': parent_match, 'round_field_agrees_with_first_file': f'{agree}/{n_new}',
           'distinct_norm_proofs': dist, 'raw_records_last': len(last), 'unparsable': unparsable, 'unknown_names': unknown,
           'cum_solved': cum(solved_round), 'cum_pattern_theorems': cum(acq_round), 'cum_pattern_theorems_unpruned': cum(acq_unp_round),
           'cum_pattern_theorems_excl_injected': cum(acq_round, lambda nm: nm not in inj),
           'cum_pattern_theorems_new_only': [sum(1 for nm, v in acq_round.items() if rounds[0] <= v <= r) for r in R],
           'injected_names_acquired': sorted(nm for nm in acq_round if nm in inj),
           'distinct_pattern_proofs': pat_proofs, 'pattern_proofs_verified': n_ver, 'pattern_verify_failures': ver_fail,
           'other_proofs_sampled': s_ver, 'other_verify_failures': s_fail}
    return out


def part_b(res):
    print('== B. arms x conditions', flush=True)
    B = {}
    for pattern, arm in ARMS:
        tg, tr, train = POOLS[pattern]
        for cond in CONDS:
            d = f'artifacts/r3/{arm}_{cond}'
            if not os.path.isdir(d):
                print(f'  MISSING {d}'); continue
            inj = res['injections'][f'{arm}_{cond}']['injected_target_names']
            t = recount_dir(d, pattern, tg, 'found_{}.jsonl', [5, 6, 7, 8], parent_found=f'artifacts/p2/{arm}/found_4.jsonl', injected_names=inj)
            x = recount_dir(d, pattern, tr, 'found_transfer_{}.jsonl', [5, 6, 7, 8], parent_found=f'artifacts/p2/{arm}/found_transfer_4.jsonl', verify_all=False, sample_verify=100)
            ex = {}
            for r in (5, 6, 7, 8):
                j = json.load(open(f'{d}/round_{r}.json'))
                ex[r] = {'solved': j['targets_cum']['solved'], 'transfer_solved': j['transfer_cum']['solved'], 'heldout_greedy': j['heldout_greedy']['rate'], 'ckpt': os.path.basename(j['ckpt'])}
            args = json.load(open(f'{d}/args.json'))
            ign = {}
            for th in THRESH[pattern]:
                ign[th] = next((r for r, c in zip(t['rounds'], t['cum_pattern_theorems']) if r >= 5 and c >= th), None)
            ign_ex = {th: next((r for r, c in zip(t['rounds'], t['cum_pattern_theorems_excl_injected']) if r >= 5 and c >= th), None) for th in THRESH[pattern]}
            B[f'{arm}_{cond}'] = {'arm': arm, 'cond': cond, 'pattern': pattern, 'args': {k: args[k] for k in ('init', 'targets', 'transfer', 'train', 'rounds', 'k', 'temperature', 'seed', 'start_round', 'resume_found', 'no_train')},
                                 'targets': t, 'transfer': x, 'executor_rounds': ex, 'ignition_round': ign, 'ignition_round_excl_injected': ign_ex,
                                 'acq_r8': t['cum_pattern_theorems'][-1] / t['n'], 'acq_r8_excl_injected': t['cum_pattern_theorems_excl_injected'][-1] / t['n'],
                                 'transfer_acq_r8': x['cum_pattern_theorems'][-1] / x['n'],
                                 'solved_matches_executor': all(ex[r]['solved'] == t['cum_solved'][i + 1] for i, r in enumerate((5, 6, 7, 8))),
                                 'transfer_solved_matches_executor': all(ex[r]['transfer_solved'] == x['cum_solved'][i + 1] for i, r in enumerate((5, 6, 7, 8))),
                                 'heldout_greedy_min': min(v['heldout_greedy'] for v in ex.values())}
            b = B[f'{arm}_{cond}']
            print(f"  {arm}_{cond:7s} pat {t['cum_pattern_theorems']} exclinj {t['cum_pattern_theorems_excl_injected']} new {t['cum_pattern_theorems_new_only']} solved {t['cum_solved']} "
                  f"exec-solved-ok {b['solved_matches_executor']} ign {ign} chain {t['chain_ok']} parent {t['parent_match']['equal'] if t['parent_match'] else None} "
                  f"rf {t['round_field_agrees_with_first_file']} patproofs {t['distinct_pattern_proofs']} ver {t['pattern_proofs_verified']}/{t['pattern_verify_failures']}f other {t['other_proofs_sampled']}/{t['other_verify_failures']}f "
                  f"transfer pat {x['cum_pattern_theorems']} held {b['heldout_greedy_min']:.3f} unp {t['unparsable']}", flush=True)
    res['arms'] = B


def part_c(res):
    print('== C. parents and full-sibling (ivS) arms', flush=True)
    C = {}
    for pattern, arm in ARMS:
        tg, tr, train = POOLS[pattern]
        d = f'artifacts/p2/{arm}'
        rounds = [r for r in range(1, 9) if os.path.exists(f'{d}/found_{r}.jsonl')]
        # reductio parents only kept found_4 / found_8 (both empty)
        if rounds == list(range(1, 9)):
            p = recount_dir(d, pattern, tg, 'found_{}.jsonl', rounds, verify_all=False, sample_verify=100)
            C[arm] = {'rounds': p['rounds'], 'cum_pattern_theorems': p['cum_pattern_theorems'], 'cum_solved': p['cum_solved'], 'chain_ok': p['chain_ok']}
        else:
            sizes = {r: os.path.getsize(f'{d}/found_{r}.jsonl') for r in rounds}
            C[arm] = {'rounds': rounds, 'found_file_sizes': sizes, 'cum_pattern_theorems': [0] * len(rounds) if all(v == 0 for v in sizes.values()) else None,
                      'executor_solved_by_round': [json.load(open(f'{d}/round_{r}.json'))['targets_cum']['solved'] for r in range(1, 9)]}
        print(f'  parent {arm}: {C[arm]}', flush=True)
        ds = f'artifacts/p2/{arm}_ivS'
        if os.path.isdir(ds) and all(os.path.exists(f'{ds}/found_{r}.jsonl') for r in (5, 6, 7, 8)):
            p = recount_dir(ds, pattern, tg, 'found_{}.jsonl', [5, 6, 7, 8], parent_found=f'{d}/found_4.jsonl', verify_all=False, sample_verify=100)
            C[arm + '_ivS'] = {'rounds': p['rounds'], 'cum_pattern_theorems': p['cum_pattern_theorems'], 'cum_solved': p['cum_solved']}
            print(f'  ivS {arm}: {C[arm + "_ivS"]}', flush=True)
    res['parents'] = C


def main():
    res = {}
    t0 = time.time()
    part_a(res); json.dump(res, open(f'{OUT}/recount.json', 'w'), indent=1)
    part_b(res); json.dump(res, open(f'{OUT}/recount.json', 'w'), indent=1)
    part_c(res); json.dump(res, open(f'{OUT}/recount.json', 'w'), indent=1)
    print(f'done in {time.time()-t0:.0f}s -> {OUT}/recount.json', flush=True)


if __name__ == '__main__':
    main()
