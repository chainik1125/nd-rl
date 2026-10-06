#!/usr/bin/env python3
"""Reviewer's independent recount for round2-run2 (six new patterns, classes pre-registered).

Own parser / dependency pruning / start-index normaliser / atom-renaming key (review_run5_recount.py) and own
predicates for the six run-2 patterns written here from the pre-registration's one-line definitions.  Only nd_verify
is shared with the executor.  Run from the repository root:  python3 review_run2_recount.py [sets|pools|arms|cov|ckpts|all]
Outputs: artifacts/review_r2/*.json and a printed summary.
"""
import json, re, collections, glob, os, sys, random, zipfile, pickle, io, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nd_verify import verify_text
from review_run5_recount import parse_proof, prune, normalise, rkey, thm_of_prompt, max_depth, rd

OUT = 'artifacts/review_r2'
PATTERNS = ('depth4', 'impe_chain4', 'nested_ore', 'impi_ore', 'negi_ande_hyp', 'ori_ore')


# ---------------------------------------------------------------- predicates (on my parsed, pruned lines)
def p_depth4(lines):
    return max_depth(lines) >= 4


def impe_chain_len(lines):
    """Longest chain of IMPE lines each citing an earlier IMPE line of the chain (as major or minor premise)."""
    best = {}
    for l in lines:                      # lines are in index order
        if l['rule'] == 'IMPE':
            best[l['idx']] = 1 + max((best[r] for r in l['refs'] if r in best), default=0)
    return max(best.values(), default=0)


def p_impe_chain4(lines):
    return impe_chain_len(lines) >= 4


def ore_nesting(lines):
    """1 = an ORE; 2 = an ORE whose line lies inside a branch (AS..end) of another ORE; ..."""
    ores = [l for l in lines if l['rule'] == 'ORE' and len(l['refs']) == 5]
    memo = {}

    def d(o):
        if o['idx'] in memo:
            return memo[o['idx']]
        _, s1, e1, s2, e2 = o['refs']
        inner = [p for p in ores if p is not o and (s1 <= p['idx'] <= e1 or s2 <= p['idx'] <= e2)]
        memo[o['idx']] = 1 + max((d(p) for p in inner), default=0)
        return memo[o['idx']]
    return max((d(o) for o in ores), default=0)


def p_nested_ore(lines):
    return ore_nesting(lines) >= 2


def p_impi_ore(lines):
    for l in lines:
        if l['rule'] == 'IMPI' and len(l['refs']) == 2:
            s, e = l['refs']
            if any(x['rule'] == 'ORE' and s <= x['idx'] <= e for x in lines):
                return True
    return False


def p_negi_ande_hyp(lines):
    for l in lines:
        if l['rule'] == 'NEGI' and len(l['refs']) == 2:
            s, e = l['refs']
            if any(x['rule'] in ('ANDE1', 'ANDE2') and x['refs'] and x['refs'][0] == s and s <= x['idx'] <= e for x in lines):
                return True
    return False


def p_ori_ore(lines):
    by = {l['idx']: l for l in lines}
    return any(l['rule'] == 'ORE' and l['refs'] and by.get(l['refs'][0], {}).get('rule') in ('ORI1', 'ORI2') for l in lines)


PRED = {'depth4': p_depth4, 'impe_chain4': p_impe_chain4, 'nested_ore': p_nested_ore, 'impi_ore': p_impi_ore,
        'negi_ande_hyp': p_negi_ande_hyp, 'ori_ore': p_ori_ore}


def classify(proof):
    lines = parse_proof(proof)
    if lines is None:
        return None
    pr = prune(lines)
    c = {p: PRED[p](pr) for p in PATTERNS}
    c.update({p + '_unpruned': PRED[p](lines) for p in PATTERNS})
    c.update({'written': len(lines), 'pruned': len(pr), 'depth': max_depth(pr), 'chain': impe_chain_len(pr), 'nesting': ore_nesting(pr)})
    return c


def selftest():
    """The predicates against hand-built proofs (verifier-checked)."""
    ok = True
    cases = [
        ('THM SEQ ( P > ( Q > ( R > ( S > S ) ) ) ) PRF', 'N1 | P : AS ; N2 | | Q : AS ; N3 | | | R : AS ; N4 | | | | S : AS ; N5 | | | ( S > S ) : IMPI N4 N4 ; N6 | | ( R > ( S > S ) ) : IMPI N3 N5 ; N7 | ( Q > ( R > ( S > S ) ) ) : IMPI N2 N6 ; N8 ( P > ( Q > ( R > ( S > S ) ) ) ) : IMPI N1 N7 ; QED', {'depth4'}),
        ('THM P , ( P > Q ) , ( Q > R ) , ( R > S ) , ( S > ( P & Q ) ) SEQ ( P & Q ) PRF', 'N1 P : PR ; N2 ( P > Q ) : PR ; N3 ( Q > R ) : PR ; N4 ( R > S ) : PR ; N5 ( S > ( P & Q ) ) : PR ; N6 Q : IMPE N2 N1 ; N7 R : IMPE N3 N6 ; N8 S : IMPE N4 N7 ; N9 ( P & Q ) : IMPE N5 N8 ; QED', {'impe_chain4'}),
        ('THM P , ( P > Q ) , ( Q > R ) , ( R > S ) SEQ S PRF', 'N1 P : PR ; N2 ( P > Q ) : PR ; N3 ( Q > R ) : PR ; N4 ( R > S ) : PR ; N5 Q : IMPE N2 N1 ; N6 R : IMPE N3 N5 ; N7 S : IMPE N4 N6 ; QED', set()),
        ('THM ( ( P v Q ) v R ) SEQ ( P v ( Q v R ) ) PRF', 'N1 ( ( P v Q ) v R ) : PR ; N2 | ( P v Q ) : AS ; N3 | | P : AS ; N4 | | ( P v ( Q v R ) ) : ORI1 N3 ; N5 | | Q : AS ; N6 | | ( Q v R ) : ORI1 N5 ; N7 | | ( P v ( Q v R ) ) : ORI2 N6 ; N8 | ( P v ( Q v R ) ) : ORE N2 N3 N4 N5 N7 ; N9 | R : AS ; N10 | ( Q v R ) : ORI2 N9 ; N11 | ( P v ( Q v R ) ) : ORI2 N10 ; N12 ( P v ( Q v R ) ) : ORE N1 N2 N8 N9 N11 ; QED', {'nested_ore'}),
        ('THM ( P v Q ) , ( P > R ) SEQ ( ( Q > R ) > R ) PRF', 'N1 ( P v Q ) : PR ; N2 ( P > R ) : PR ; N3 | ( Q > R ) : AS ; N4 | | P : AS ; N5 | | R : IMPE N2 N4 ; N6 | | Q : AS ; N7 | | R : IMPE N3 N6 ; N8 | R : ORE N1 N4 N5 N6 N7 ; N9 ( ( Q > R ) > R ) : IMPI N3 N8 ; QED', {'impi_ore'}),
        ('THM ( ~ P ) SEQ ( ~ ( P & Q ) ) PRF', 'N1 ( ~ P ) : PR ; N2 | ( P & Q ) : AS ; N3 | P : ANDE1 N2 ; N4 | F : NEGE N3 N1 ; N5 ( ~ ( P & Q ) ) : NEGI N2 N4 ; QED', {'negi_ande_hyp'}),
        ('THM ( P & ( ~ Q ) ) SEQ ( ~ Q ) PRF', 'N1 ( P & ( ~ Q ) ) : PR ; N2 | Q : AS ; N3 | ( ~ Q ) : ANDE2 N1 ; N4 | F : NEGE N2 N3 ; N5 ( ~ Q ) : NEGI N2 N4 ; QED', set()),
        ('THM P , ( P > Q ) SEQ Q PRF', 'N1 P : PR ; N2 ( P > Q ) : PR ; N3 ( P v R ) : ORI1 N1 ; N4 | P : AS ; N5 | Q : IMPE N2 N4 ; N6 | R : AS ; N7 | Q : IMPE N2 N1 ; N8 Q : ORE N3 N4 N5 N6 N7 ; QED', {'ori_ore'}),
        # decoration: an unused fourth box, dropped by pruning -> depth4 only unpruned
        ('THM SEQ ( P > ( Q > ( R > R ) ) ) PRF', 'N1 | P : AS ; N2 | | Q : AS ; N3 | | | R : AS ; N4 | | | | S : AS ; N5 | | | | S : R N4 ; N6 | | | ( S > S ) : IMPI N4 N5 ; N7 | | | R : R N3 ; N8 | | ( R > R ) : IMPI N3 N7 ; N9 | ( Q > ( R > R ) ) : IMPI N2 N8 ; N10 ( P > ( Q > ( R > R ) ) ) : IMPI N1 N9 ; QED', set()),
    ]
    for prompt, proof, want in cases:
        v, reason, nl = verify_text(prompt + ' ' + proof)
        assert v, (reason, prompt)
        c = classify(proof)
        got = {p for p in PATTERNS if c[p]}
        if got != want:
            ok = False; print('SELFTEST FAIL', prompt, got, want)
    print('predicate selftest', 'PASS' if ok else 'FAIL')
    return ok


# ---------------------------------------------------------------- Stage-1 sets
SETS = {'train_r2_struct': ('data/r2/train_r2_struct.jsonl', 6, None), 'train_r2_impi_ore_f0': ('data/r2/train_r2_impi_ore_f0.jsonl', 6, 'impi_ore'),
        'train_r2_negi_ande_hyp_f0': ('data/r2/train_r2_negi_ande_hyp_f0.jsonl', 6, 'negi_ande_hyp'), 'train_r2_ori_ore_f0': ('data/r2/train_r2_ori_ore_f0.jsonl', 6, 'ori_ore'),
        'train_r2_c8_depth4_f0': ('data/r2/train_r2_c8_depth4_f0.jsonl', 8, 'depth4'),
        'p2/train_derived_ore_strict_f0_c8': ('data/p2/train_derived_ore_strict_f0_c8.jsonl', 8, None)}
POOLS = {p: (f'data/r2/targets_{p}.jsonl', f'data/r2/transfer_{p}.jsonl') for p in PATTERNS}
EVALS = {'heldout': 'data/p2/heldout.jsonl', 'heldout_c8': 'data/p2/heldout_c8.jsonl', 'val36': 'targets/validation_36.jsonl'}


def eval_keys():
    ks = {}
    for p, (t, x) in POOLS.items():
        ks[f'targets_{p}'] = {rkey(thm_of_prompt(r['prompt'])) for r in rd(t)}
        ks[f'transfer_{p}'] = {rkey(thm_of_prompt(r['prompt'])) for r in rd(x)}
    for n, fn in EVALS.items():
        ks[n] = {rkey(thm_of_prompt(r['prompt'])) if 'prompt' in r else rkey(r['thm'].strip()) for r in rd(fn)}
    return ks


def scan_sets():
    ks = eval_keys()
    res = {}
    for tag, (fn, cap, f0) in SETS.items():
        t0 = time.time()
        n = 0; capv = 0; maxl = 0; unp = 0; cnt = collections.Counter(); wcnt = collections.Counter(); keys = set(); lens = collections.Counter()
        rng = random.Random(1); nver = 0; vfail = 0; keyfield_bad = 0
        for l in open(fn):
            if not l.strip():
                continue
            r = json.loads(l); n += 1
            k = rkey(thm_of_prompt(r['prompt'])); keys.add(k); keyfield_bad += (r.get('key') != k)
            c = classify(r['proof'])
            if c is None:
                unp += 1; continue
            maxl = max(maxl, c['written']); capv += c['written'] > cap; lens[c['written']] += 1
            for p in PATTERNS:
                cnt[p] += c[p]; wcnt[p] += c[p + '_unpruned']
            if rng.random() < 0.005:
                ok, _, nl = verify_text(r['prompt'] + ' ' + r['proof']); nver += 1; vfail += (not ok) or nl != r['n_lines']
        res[tag] = {'file': fn, 'n': n, 'cap': cap, 'cap_violations': capv, 'max_written': maxl, 'written_hist': dict(sorted(lens.items())), 'unparsable': unp,
                    'pattern_counts_pruned': dict(cnt), 'pattern_counts_written': dict(wcnt), 'f0_pattern': f0, 'f0_holds': (cnt[f0] == 0 and wcnt[f0] == 0) if f0 else None,
                    'distinct_keys': len(keys), 'key_field_mismatches': keyfield_bad, 'verified_sample': nver, 'verify_failures': vfail,
                    'overlap': {e: len(keys & ek) for e, ek in ks.items()}, 'secs': time.time() - t0}
        print(f"SET {tag:36s} n {n} cap{cap} viol {capv} maxL {maxl} unp {unp} pruned {dict(cnt)} written {dict(wcnt)} keys {len(keys)} keyfield-bad {keyfield_bad} ver {nver}/{vfail}f overlaps {{{', '.join(f'{e}:{v}' for e, v in res[tag]['overlap'].items() if v)}}}", flush=True)
    names = list(ks)
    cross = {f'{a} ∩ {b}': len(ks[a] & ks[b]) for i, a in enumerate(names) for b in names[i + 1:]}
    res['eval_cross_nonzero'] = {k: v for k, v in cross.items() if v}
    res['eval_sizes'] = {k: len(v) for k, v in ks.items()}
    print('EVAL cross overlaps (nonzero):', res['eval_cross_nonzero'])
    json.dump(res, open(f'{OUT}/sets.json', 'w'), indent=1)
    return res


# ---------------------------------------------------------------- pools and oracle files
NEC = {'depth4': 'data/r2/nec_depth4.jsonl', 'impe_chain4': 'data/r2/nec_impe_chain4.jsonl', 'nested_ore': 'data/r2/nec_nested_ore.jsonl',
       'impi_ore': 'data/r2/nec_impi_ore.jsonl', 'negi_ande_hyp': 'data/r2/nec_negi_ande_hyp.jsonl', 'ori_ore': 'data/r2/nec_ori_ore.jsonl'}
BOUND = {'nested_ore': 13}


def check_nec(p):
    recs = rd(NEC[p]); n = len(recs); bound = BOUND.get(p, 10)
    reach = [r for r in recs if r['min_lines_ub'] is not None]
    req = [r for r in recs if r['requires']]
    uses = [r for r in recs if r.get('uses')]
    to = sum(1 for r in recs if r.get('timeout') or r.get('r_timeout'))
    bad_req = [r['name'] for r in req if not (r['r_min_lines_ub'] is None and not r['r_timeout'] and r['min_lines_ub'] <= bound and not r.get('no_restriction'))]
    my_req_ok = 0; my_req_bad = []; my_uses_ok = 0; my_uses_bad = []; uver = 0; rver = 0; r_has = []; my_uses_extra = 0
    for r in recs:
        if r['proof']:
            ok, _, nl = verify_text(r['prompt'] + ' ' + r['proof']); uver += (not ok) or nl != r['min_lines_ub']
            c = classify(r['proof']); has = bool(c and c[p])
            if r['requires']:
                my_req_ok += has; my_req_bad += [] if has else [r['name']]
            if r.get('uses'):
                my_uses_ok += has; my_uses_bad += [] if has else [r['name']]
            elif has:
                my_uses_extra += 1
        if r.get('r_proof'):
            ok, _, nl = verify_text(r['prompt'] + ' ' + r['r_proof']); rver += (not ok) or nl != r['r_min_lines_ub']
            c = classify(r['r_proof'])
            if c and c[p]:
                r_has.append(r['name'])
    res = {'file': NEC[p], 'n': n, 'bound': bound, 'reachable': len(reach), 'reachable_by_len': dict(sorted(collections.Counter(r['min_lines_ub'] for r in reach).items())),
           'required': len(req), 'required_by_len': dict(sorted(collections.Counter(r['min_lines_ub'] for r in req).items())), 'uses': len(uses),
           'uses_by_len': dict(sorted(collections.Counter(r['min_lines_ub'] for r in uses).items())), 'timeouts': to, 'no_restriction': sum(1 for r in recs if r.get('no_restriction')),
           'required_with_inconsistent_fields': bad_req, 'required_with_my_pattern_on_oracle_proof': my_req_ok, 'required_without_my_pattern': my_req_bad,
           'uses_with_my_pattern': my_uses_ok, 'uses_without_my_pattern': my_uses_bad, 'not_uses_but_my_pattern': my_uses_extra,
           'restricted_proofs_with_pattern(my_pred)': r_has, 'oracle_proof_verify_failures': uver, 'restricted_proof_verify_failures': rver,
           'schema_hist': dict(collections.Counter(r.get('schema') for r in recs).most_common()), 'source_hist': dict(collections.Counter(r.get('source') for r in recs))}
    print(f"NEC {p:14s} n {n} reach {len(reach)} req {len(req)} {res['required_by_len']} uses {len(uses)} timeouts {to} badreq {len(bad_req)} req-mypat {my_req_ok}/{len(my_req_bad)}bad uses-mypat {my_uses_ok}/{len(my_uses_bad)}bad extra {my_uses_extra} r-has-pat {len(r_has)} verfail {uver}/{rver}", flush=True)
    return res


def check_pool(p, fn, nec):
    recs = rd(fn); n = len(recs)
    keys = [rkey(thm_of_prompt(r['prompt'])) for r in recs]
    key_ok = sum(k == r['key'] for k, r in zip(keys, recs)); thm_ok = sum(thm_of_prompt(r['prompt']) == r['thm'].strip() for r in recs)
    vfail = 0; pat_ok = 0; pat_unp = 0; nl_ok = 0; gen_pat = 0; gen_ver = 0; in_nec = 0; nec_same = 0; req_cnt = 0; uses_cnt = 0
    nl_hist = collections.Counter(); gl_hist = collections.Counter(); req_by_len = collections.Counter(); prompt_toks = collections.Counter()
    for r in recs:
        ok, _, nl = verify_text(r['prompt'] + ' ' + r['oracle_proof']); vfail += not ok
        c = classify(r['oracle_proof']); pat_ok += bool(c and c[p]); pat_unp += bool(c and c[p + '_unpruned'])
        nl_ok += (nl == r['n_lines'] == r['min_lines_ub']); nl_hist[r['n_lines']] += 1; gl_hist[r.get('gen_lines')] += 1
        req_cnt += bool(r['requires']); uses_cnt += bool(r.get('uses')); req_by_len[(r['n_lines'], bool(r['requires']))] += 1
        prompt_toks[len(r['prompt'].split()) // 10 * 10] += 1
        if r.get('gen_proof'):
            ok2, _, nl2 = verify_text(r['prompt'] + ' ' + r['gen_proof']); gen_ver += ok2
            c2 = classify(r['gen_proof']); gen_pat += bool(c2 and c2[p])
        nr = nec.get(rkey(thm_of_prompt(r['prompt'])))
        if nr is not None:
            in_nec += 1; nec_same += (nr['requires'] == r['requires'] and nr['min_lines_ub'] == r['min_lines_ub'] and nr['r_min_lines_ub'] == r['r_min_lines_ub'] and not nr['timeout'] and not nr['r_timeout'])
    res = {'file': fn, 'n': n, 'distinct_keys': len(set(keys)), 'my_key_equals_key_field': key_ok, 'thm_equals_prompt': thm_ok, 'oracle_proof_verify_failures': vfail,
           'oracle_proof_has_pattern(my_pred, pruned)': pat_ok, 'oracle_proof_has_pattern(unpruned)': pat_unp, 'n_lines_consistent': nl_ok, 'n_lines_hist': dict(sorted(nl_hist.items())),
           'gen_lines_hist': dict(sorted(gl_hist.items(), key=lambda x: (x[0] is None, x[0]))), 'requires': req_cnt, 'uses': uses_cnt,
           'by_len_requires': {f'{k[0]}:{"req" if k[1] else "uses-only"}': v for k, v in sorted(req_by_len.items())},
           'gen_proof_records': sum(1 for r in recs if r.get('gen_proof')), 'gen_proof_verifies': gen_ver, 'gen_proof_has_pattern': gen_pat,
           'in_nec': in_nec, 'nec_record_agrees': nec_same, 'mode_hist': dict(collections.Counter(r.get('mode') for r in recs)),
           'schema_hist': dict(collections.Counter(r.get('schema') for r in recs).most_common()), 'source_hist': dict(collections.Counter(r.get('source') for r in recs)),
           'prompt_tokens_decile_hist': dict(sorted(prompt_toks.items())), 'keys': keys}
    print(f"POOL {os.path.basename(fn):32s} n {n} keys {len(set(keys))} key-ok {key_ok} thm-ok {thm_ok} verfail {vfail} pat {pat_ok} (unpr {pat_unp}) nl-ok {nl_ok} nl {res['n_lines_hist']} req {req_cnt} uses {uses_cnt} gen-pat {gen_pat}/{res['gen_proof_records']} nec {in_nec}/{nec_same} mode {res['mode_hist']} src {res['source_hist']}", flush=True)
    return res


def pools():
    res = {'nec': {}, 'pools': {}}
    for p in PATTERNS:
        res['nec'][p] = check_nec(p)
        nec = {rkey(thm_of_prompt(r['prompt'])): r for r in rd(NEC[p])}
        for fn in POOLS[p]:
            res['pools'][fn] = check_pool(p, fn, nec)
        kt, kx = set(res['pools'][POOLS[p][0]]['keys']), set(res['pools'][POOLS[p][1]]['keys'])
        res['pools'][POOLS[p][0]]['targets∩transfer'] = len(kt & kx)
        print(f'   {p}: targets∩transfer keys {len(kt & kx)}; schema {res["pools"][POOLS[p][0]]["schema_hist"]}')
    for v in res['pools'].values():
        v.pop('keys')
    json.dump(res, open(f'{OUT}/pools.json', 'w'), indent=1)
    return res


# ---------------------------------------------------------------- arms
ARMS = []
for p in PATTERNS:
    for s in (0, 1):
        ARMS += [(f'ei_{p}_s{s}', p), (f'frozen_{p}_s{s}', p)]
for tag in ('depth4_c8ctl', 'depth4_c8f0'):
    for s in (0, 1):
        ARMS += [(f'ei_{tag}_s{s}', 'depth4'), (f'frozen_{tag}_s{s}', 'depth4')]
for tag in ('impi_ore_natctl', 'negi_ande_hyp_natctl'):
    for s in (0, 1):
        ARMS += [(f'ei_{tag}_s{s}', tag.replace('_natctl', '')), (f'frozen_{tag}_s{s}', tag.replace('_natctl', ''))]
IGN_FRAC = 0.02


def count_arm(arm, pattern, verify_all=True):
    d = f'artifacts/r2/{arm}'
    args = json.load(open(f'{d}/args.json'))
    out = {'arm': arm, 'pattern': pattern, 'args': {k: args[k] for k in ('init', 'targets', 'transfer', 'train', 'heldout', 'rounds', 'k', 'temperature', 'seed', 'no_train', 'retain', 'max_per_thm', 'rl_weight', 'ft_steps')}}
    for pool, fnpat, pool_file in (('targets', 'found_{}.jsonl', args['targets']), ('transfer', 'found_transfer_{}.jsonl', args['transfer'])):
        prec = rd(pool_file); names = [r['name'] for r in prec]; prompts = {r['name']: r['prompt'] for r in prec}; n = len(names)
        rounds = sorted(int(m.group(1)) for f in glob.glob(f'{d}/' + fnpat.format('*')) for m in [re.fullmatch(fnpat.format(r'(\d+)'), os.path.basename(f))] if m)
        if rounds != list(range(1, 9)):
            out[pool] = {'INCOMPLETE': f'per-round files present: {rounds}', 'files': sorted(os.listdir(d))}
            continue
        recs8 = rd(f'{d}/' + fnpat.format(8))
        set8 = {(r['name'], normalise(r['proof'])) for r in recs8}
        subset_ok = True; first = {}; rf_min = collections.defaultdict(lambda: 99)
        for rr in range(1, 9):
            rs = rd(f'{d}/' + fnpat.format(rr))
            ks = {(r['name'], normalise(r['proof'])) for r in rs}
            subset_ok &= ks <= set8
            for r in rs:
                k = (r['name'], normalise(r['proof'])); first.setdefault(k, rr); rf_min[k] = min(rf_min[k], r['round'])
        rf_agree = sum(first[k] == rf_min[k] for k in first)
        solved_round, acq_round, acq_unp_round = {}, {}, {}
        seen = collections.defaultdict(set); bad = 0; nver = 0; vfail = 0; unknown = 0; pat_proofs = 0; nproofs = 0
        whist = collections.Counter(); phist = collections.Counter(); dhist = collections.Counter(); chist = collections.Counter()
        for (name, npf), rr in sorted(first.items(), key=lambda x: x[1]):
            if name not in prompts:
                unknown += 1; continue
            if npf in seen[name]:
                continue
            seen[name].add(npf)
            c = classify(npf)
            if c is None:
                bad += 1; continue
            nproofs += 1; whist[c['written']] += 1; phist[c['pruned']] += 1; dhist[c['depth']] += 1; chist[c['chain']] += 1
            solved_round[name] = min(solved_round.get(name, 99), rr)
            if c[pattern]:
                pat_proofs += 1; acq_round[name] = min(acq_round.get(name, 99), rr)
            if c[pattern + '_unpruned']:
                acq_unp_round[name] = min(acq_unp_round.get(name, 99), rr)
            if verify_all:
                ok, _, nl = verify_text(prompts[name] + ' ' + npf); nver += 1; vfail += (not ok) or nl != c['written']
        cum_solved = [sum(1 for v in solved_round.values() if v <= rr) for rr in range(1, 9)]
        cum_acq = [sum(1 for v in acq_round.values() if v <= rr) for rr in range(1, 9)]
        cum_unp = [sum(1 for v in acq_unp_round.values() if v <= rr) for rr in range(1, 9)]
        thr = max(1, int(round(IGN_FRAC * n)))
        ign = next((rr + 1 for rr, v in enumerate(cum_acq) if v >= thr), None)
        r8 = json.load(open(f'{d}/round_8.json'))
        exec_solved = r8['targets_cum' if pool == 'targets' else 'transfer_cum']['solved']
        exec_by_round = [json.load(open(f'{d}/round_{rr}.json'))['targets_cum' if pool == 'targets' else 'transfer_cum']['solved'] for rr in range(1, 9)]
        res = {'n': n, 'attempts_per_target': 8 * args['k'], 'solved': len(solved_round), 'solve_rate': len(solved_round) / n, 'acquired': len(acq_round), 'acq_rate': len(acq_round) / n,
               'acquired_unpruned': len(acq_unp_round), 'n_solved_without_pattern': len(set(solved_round) - set(acq_round)),
               'cum_solved_by_round': cum_solved, 'cum_acquired_by_round': cum_acq, 'cum_acquired_unpruned_by_round': cum_unp,
               'first_acq_round': min(acq_round.values()) if acq_round else None, 'ignition_round(thr=%d)' % thr: ign,
               'distinct_norm_proofs': nproofs, 'distinct_pattern_proofs': pat_proofs, 'raw_records_round8': len(recs8),
               'written_hist': dict(sorted(whist.items())), 'pruned_hist': dict(sorted(phist.items())), 'depth_hist': dict(sorted(dhist.items())), 'chain_hist': dict(sorted(chist.items())),
               'frontier_written_ge5': max([L for L, c in whist.items() if c >= 5], default=0),
               'verified': nver, 'verify_failures': vfail, 'unparsable': bad, 'unknown_names': unknown,
               'per_round_files_nested': subset_ok, 'round_field_agrees': f'{rf_agree}/{len(first)}',
               'executor_cum_solved_by_round': exec_by_round, 'solved_matches_executor_all_rounds': exec_by_round == cum_solved}
        if pool == 'targets':
            prec_by = {r['name']: r for r in prec}
            byl = collections.defaultdict(lambda: [0, 0, 0]); byreq = collections.defaultdict(lambda: [0, 0, 0]); bysch = collections.defaultdict(lambda: [0, 0, 0])
            for nm in names:
                r = prec_by[nm]
                for dct, key in ((byl, r['n_lines']), (byreq, 'requires' if r['requires'] else 'uses-only'), (bysch, r.get('schema') or r.get('source'))):
                    dct[key][0] += 1; dct[key][1] += nm in solved_round; dct[key][2] += nm in acq_round
            res['by_min_lines(n,solved,acq)'] = {str(k): v for k, v in sorted(byl.items())}
            res['by_requires(n,solved,acq)'] = dict(byreq); res['by_schema(n,solved,acq)'] = dict(bysch)
        out[pool] = res
    if all(os.path.exists(f'{d}/round_{rr}.json') for rr in range(1, 9)):
        out['heldout_greedy_by_round'] = [round(json.load(open(f'{d}/round_{rr}.json'))['heldout_greedy']['rate'], 3) for rr in range(1, 9)]
        out['transfer_greedy_r8'] = json.load(open(f'{d}/round_8.json'))['transfer_greedy']['rate']
        out['mix_rl_records_by_round'] = [json.load(open(f'{d}/round_{rr}.json')).get('mix_rl_records') for rr in range(1, 9)]
    return out


def arms(which=None):
    res = []
    for arm, p in ARMS:
        if which and not arm.startswith(which):
            continue
        if not os.path.isdir(f'artifacts/r2/{arm}'):
            print('MISSING', arm); continue
        r = count_arm(arm, p); res.append(r); t, x = r['targets'], r['transfer']
        if 'INCOMPLETE' in t or 'INCOMPLETE' in x:
            print(f"{arm:30s} INCOMPLETE targets {t.get('INCOMPLETE')} transfer {x.get('INCOMPLETE')} files {t.get('files', x.get('files'))}", flush=True); continue
        ign_key = [k for k in t if k.startswith('ignition_round')][0]
        print(f"{arm:30s} T solved {t['solved']:3d}/{t['n']} acq {t['acquired']:3d} ({t['acq_rate']:.3f}) unpr {t['acquired_unpruned']:3d} noPat {t['n_solved_without_pattern']:3d} proofs {t['distinct_norm_proofs']:5d} pat {t['distinct_pattern_proofs']:5d} ver {t['verified']}/{t['verify_failures']}f exec-ok {t['solved_matches_executor_all_rounds']} ign {t[ign_key]} first {t['first_acq_round']} | X solved {x['solved']:3d}/{x['n']} acq {x['acquired']:3d} ver {x['verified']}/{x['verify_failures']}f exec-ok {x['solved_matches_executor_all_rounds']} | cum acq {t['cum_acquired_by_round']} solved {t['cum_solved_by_round']} | ho {r['heldout_greedy_by_round'][-1]}", flush=True)
        json.dump(res, open(f'{OUT}/arms{"_" + which if which else ""}.json', 'w'), indent=1)
    return res


# ---------------------------------------------------------------- coverage (pass@2000 on 300 targets) and drift
def count_cov(fn, pattern, pool_file):
    recs = rd(fn); prompts = {r['name']: r['prompt'] for r in rd(pool_file)}
    n = len(recs); solved = 0; acq = 0; acq_hits = 0; hits = 0; ver = 0; vfail = 0; tried = collections.Counter(); distinct = 0; pat_distinct = 0
    first_hits = []; pat_first = []
    for r in recs:
        tried[r['n_tried']] += 1; hits += r['n_ok']; solved += r['n_ok'] > 0
        got = False; pf = None
        assert sum(p['count'] for p in r['proofs']) == r['n_ok']
        for p in r['proofs']:
            ok, _, nl = verify_text(prompts[r['name']] + ' ' + p['proof']); ver += 1; vfail += (not ok) or nl != p['written']
            c = classify(p['proof']); distinct += 1
            if c and c[pattern]:
                got = True; acq_hits += p['count']; pat_distinct += 1; pf = min(pf or 10**9, p['first'])
        if got:
            acq += 1; pat_first.append(pf)
    tot = sum(k * v for k, v in tried.items())
    return {'file': fn, 'n_theorems': n, 'n_tried_hist': dict(tried), 'total_samples': tot, 'solved': solved, 'pattern_theorems': acq, 'ok_samples': hits, 'pattern_samples': acq_hits,
            'distinct_ok_proofs': distinct, 'distinct_pattern_proofs': pat_distinct, 'verified': ver, 'verify_failures': vfail,
            'pass_rate_per_sample': hits / tot if tot else None, 'pattern_rate_per_sample': acq_hits / tot if tot else None,
            'solved_within_256(exact draw)': sum(1 for r in recs if r['first_hit'] is not None and r['first_hit'] <= 256),
            'pattern_first_hit_min': min(pat_first) if pat_first else None}


def cov():
    res = []
    files = sorted(glob.glob('artifacts/r2/cov_*.jsonl') + glob.glob('artifacts/r2/drift_*.jsonl'))
    for fn in files:
        b = os.path.basename(fn)
        p = next(pp for pp in sorted(PATTERNS, key=len, reverse=True) if b.startswith('cov_' + pp) or b.startswith('drift_' + pp))
        r = count_cov(fn, p, POOLS[p][0]); res.append(r)
        print(f"COV {b:42s} thms {r['n_theorems']:3d} tried {r['n_tried_hist']} solved {r['solved']:3d} pattern-thms {r['pattern_theorems']:3d} ok-samples {r['ok_samples']:6d} pat-samples {r['pattern_samples']:6d} rate {r['pass_rate_per_sample']:.2e} pat-rate {r['pattern_rate_per_sample']:.2e} ver {r['verified']}/{r['verify_failures']}f", flush=True)
    json.dump(res, open(f'{OUT}/cov.json', 'w'), indent=1)
    return res


# ---------------------------------------------------------------- checkpoint provenance (no torch on the VPS)
class _Stub:
    def __init__(self, *a, **k): pass
    def __setstate__(self, s): pass


class _U(pickle.Unpickler):
    def find_class(self, mod, name):
        if mod.startswith('torch') or mod.startswith('numpy'):
            return _Stub
        return super().find_class(mod, name)

    def persistent_load(self, pid):
        return None


def ckpt_args(fn):
    z = zipfile.ZipFile(fn); nm = [n for n in z.namelist() if n.endswith('data.pkl')][0]
    d = _U(io.BytesIO(z.read(nm))).load()
    return d.get('extra', {}).get('args'), d.get('tok_mode'), d.get('cfg')


def ckpts():
    res = {}
    for fn in sorted(glob.glob('ckpts/r2/*.pt')):
        a, tm, cfg = ckpt_args(fn)
        st = os.stat(fn)
        res[fn] = {'args': a, 'tok_mode': tm, 'cfg': cfg, 'mtime_utc': time.strftime('%H:%M:%S', time.gmtime(st.st_mtime)), 'size': st.st_size}
        print(f"CKPT {os.path.basename(fn):36s} {res[fn]['mtime_utc']} init {a.get('init')} data {a.get('data')} cap {a.get('cap')} steps {a.get('steps')} seed {a.get('seed')} lr {a.get('lr')} mode {a.get('mode')}", flush=True)
    for fn in ('ckpts/p2/stage1_derived_ore_strict_f0_c8_s0.pt', 'ckpts/p2/stage1_derived_ore_strict_f0_c8_s1.pt'):
        a, tm, cfg = ckpt_args(fn); st = os.stat(fn)
        res[fn] = {'args': a, 'mtime': time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(st.st_mtime))}
        print(f"CKPT {fn} {res[fn]['mtime']} data {a.get('data')} cap {a.get('cap')} steps {a.get('steps')} seed {a.get('seed')}")
    json.dump(res, open(f'{OUT}/ckpts.json', 'w'), indent=1)
    return res


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    what = sys.argv[1] if len(sys.argv) > 1 else 'all'
    assert selftest()
    if what in ('sets', 'all'):
        scan_sets()
    if what in ('pools', 'all'):
        pools()
    if what in ('arms', 'all'):
        arms(sys.argv[2] if len(sys.argv) > 2 and what == 'arms' else None)
    if what in ('cov', 'all'):
        cov()
    if what in ('ckpts', 'all'):
        ckpts()
