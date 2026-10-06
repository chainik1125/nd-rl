#!/usr/bin/env python3
"""Reviewer's independent recount for round2-run5 (written from scratch; only nd_verify is shared with the executor).

Own parser, own dependency pruning, own start-index normaliser, own predicates (reductio, derived-ORE strict,
box depth), own atom-renaming key.  Outputs review_out/*.json and prints a summary.
"""
import json, re, collections, glob, os, sys, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nd_verify import verify_text

ATOMS = ('P', 'Q', 'R', 'S')
RULES = {'PR', 'AS', 'R', 'ANDI', 'ANDE1', 'ANDE2', 'IMPE', 'IMPI', 'ORI1', 'ORI2', 'ORE', 'NEGE', 'NEGI', 'BOTE', 'DN'}


def rkey(thm):
    """Atom-renaming class: atoms renamed in order of first appearance in the theorem string."""
    m, out = {}, []
    for t in thm.split():
        if t in ATOMS:
            if t not in m:
                m[t] = ATOMS[len(m)]
            out.append(m[t])
        else:
            out.append(t)
    return ' '.join(out)


def thm_of_prompt(prompt):
    p = prompt.strip()
    assert p.startswith('THM ') and p.endswith(' PRF'), prompt
    body = p[4:-4].strip()
    m = re.fullmatch(r'(.*?)\s*SEQ\s+(.*)', body)
    assert m, prompt
    return (m.group(1).strip() + ' |- ' + m.group(2).strip()).strip()


def parse_formula(toks, i):
    t = toks[i]
    if t in ATOMS or t == 'F':
        return t, i + 1
    assert t == '(', (toks, i)
    if toks[i + 1] == '~':
        sub, j = parse_formula(toks, i + 2)
        assert toks[j] == ')'
        return ('~', sub), j + 1
    a, j = parse_formula(toks, i + 1)
    op = toks[j]
    assert op in ('&', 'v', '>'), (toks, j)
    b, k = parse_formula(toks, j + 1)
    assert toks[k] == ')'
    return (op, a, b), k + 1


def parse_proof(proof):
    """-> list of dicts {idx, depth, f, rule, refs} in order, or None on any structural problem."""
    toks = proof.split()
    if not toks or toks[-1] != 'QED':
        return None
    toks = toks[:-1]
    lines, i = [], 0
    try:
        while i < len(toks):
            m = re.fullmatch(r'N(\d+)', toks[i])
            if not m:
                return None
            idx = int(m.group(1)); i += 1
            depth = 0
            while toks[i] == '|':
                depth += 1; i += 1
            f, i = parse_formula(toks, i)
            if toks[i] != ':':
                return None
            rule = toks[i + 1]; i += 2
            if rule not in RULES:
                return None
            refs = []
            while toks[i] != ';':
                m = re.fullmatch(r'N(\d+)', toks[i])
                if not m:
                    return None
                refs.append(int(m.group(1))); i += 1
            i += 1
            lines.append({'idx': idx, 'depth': depth, 'f': f, 'rule': rule, 'refs': refs})
    except (AssertionError, IndexError):
        return None
    return lines


def prune(lines):
    """Keep PR lines and everything the final line transitively cites (box cites = AS line + end line, whose refs follow)."""
    by = {l['idx']: l for l in lines}
    keep = {l['idx'] for l in lines if l['rule'] == 'PR'}
    stack = [lines[-1]['idx']]
    while stack:
        i = stack.pop()
        if i in keep or i not in by:
            continue
        keep.add(i); stack.extend(by[i]['refs'])
    return [l for l in lines if l['idx'] in keep]


def normalise(proof):
    """Start-index normalisation: line indices shifted so the first line is N1."""
    base = None; out = []
    for t in proof.split():
        m = re.fullmatch(r'N(\d+)', t)
        if m:
            n = int(m.group(1))
            if base is None:
                base = n
            out.append(f'N{n - base + 1}')
        else:
            out.append(t)
    return ' '.join(out)


def p_reductio(lines):
    """A DN line G citing a NEGI line whose box hypothesis (AS) is ( ~ G )."""
    by = {l['idx']: l for l in lines}
    for l in lines:
        if l['rule'] == 'DN' and l['refs']:
            n = by.get(l['refs'][0])
            if n and n['rule'] == 'NEGI' and len(n['refs']) == 2:
                h = by.get(n['refs'][0])
                if h and h['rule'] == 'AS' and h['f'] == ('~', l['f']):
                    return True
    return False


def p_derived_ore_strict(lines):
    """An ORE whose disjunction line was obtained by a rule (not PR, not AS) and whose disjuncts differ."""
    by = {l['idx']: l for l in lines}
    for l in lines:
        if l['rule'] == 'ORE' and l['refs']:
            d = by.get(l['refs'][0])
            if d and d['rule'] not in ('PR', 'AS') and isinstance(d['f'], tuple) and d['f'][0] == 'v' and d['f'][1] != d['f'][2]:
                return True
    return False


def p_derived_ore_loose(lines):
    by = {l['idx']: l for l in lines}
    return any(l['rule'] == 'ORE' and l['refs'] and by.get(l['refs'][0], {}).get('rule') != 'PR' for l in lines)


def max_depth(lines):
    return max((l['depth'] for l in lines), default=0)


PRED = {'reductio': p_reductio, 'derived_ore_strict': p_derived_ore_strict}


def classify(proof):
    lines = parse_proof(proof)
    if lines is None:
        return None
    pr = prune(lines)
    return {'reductio': p_reductio(pr), 'derived_ore_strict': p_derived_ore_strict(pr), 'derived_ore_loose': p_derived_ore_loose(pr),
            'depth': max_depth(pr), 'reductio_unpruned': p_reductio(lines), 'derived_ore_strict_unpruned': p_derived_ore_strict(lines),
            'written': len(lines), 'pruned': len(pr)}


def rd(fn):
    return [json.loads(l) for l in open(fn) if l.strip()]


# ---------------------------------------------------------------- arms
ARMS = {
    'reductio': ['ei_reductio_f0_s0_req', 'ei_reductio_f0_s1_req', 'ei_reductio_f0_s2_req', 'ei_reductio_f0.1_s0_req', 'ei_reductio_f0.1_s1_req',
                 'frozen_reductio_f0_s0_req', 'frozen_reductio_f0_s1_req', 'frozen_reductio_f0_s2_req', 'frozen_reductio_f0.1_s0_req', 'frozen_reductio_f0.1_s1_req'],
    'derived_ore_strict': ['ei_derived_ore_strict_f0_c8_s0_req', 'ei_derived_ore_strict_f0_c8_s1_req', 'ei_derived_ore_strict_f0.01_c8_s0_req', 'ei_derived_ore_strict_f0.01_c8_s1_req',
                           'frozen_derived_ore_strict_f0_c8_s0_req', 'frozen_derived_ore_strict_f0_c8_s1_req', 'frozen_derived_ore_strict_f0.01_c8_s0_req', 'frozen_derived_ore_strict_f0.01_c8_s1_req'],
}
POOLS = {'reductio': ('data/p2/targets_reductio_req.jsonl', 'data/p2/transfer_reductio_req.jsonl'),
         'derived_ore_strict': ('data/p2/targets_derived_ore_req.jsonl', 'data/p2/transfer_derived_ore_req.jsonl')}


def count_arm(arm, pattern, pool_targets, pool_transfer, verify_all=True):
    out = {'arm': arm, 'pattern': pattern}
    d = f'artifacts/r5/{arm}'
    args = json.load(open(f'{d}/args.json'))
    out['args'] = {k: args[k] for k in ('init', 'targets', 'transfer', 'train', 'rounds', 'k', 'temperature', 'seed', 'no_train')}
    for pool, fnpat, pool_file in (('targets', 'found_{}.jsonl', pool_targets), ('transfer', 'found_transfer_{}.jsonl', pool_file_t := pool_transfer)):
        names = [r['name'] for r in rd(pool_file)]
        prompts = {r['name']: r['prompt'] for r in rd(pool_file)}
        n = len(names)
        rounds = sorted(int(re.fullmatch(fnpat.format(r'(\d+)'), os.path.basename(f)).group(1)) for f in glob.glob(f'{d}/' + fnpat.format('*')) if re.fullmatch(fnpat.format(r'(\d+)'), os.path.basename(f)))
        assert rounds == list(range(1, 9)), (arm, rounds)
        # cumulative file at round 8; also check every earlier file is a subset of it
        recs8 = rd(f'{d}/' + fnpat.format(8))
        set8 = {(r['name'], normalise(r['proof'])) for r in recs8}
        subset_ok = all({(r['name'], normalise(r['proof'])) for r in rd(f'{d}/' + fnpat.format(rr))} <= set8 for rr in range(1, 8))
        # first round each (name, normalised proof) appears, from the per-round files (independent of the 'round' field)
        first = {}
        for rr in range(1, 9):
            for r in rd(f'{d}/' + fnpat.format(rr)):
                k = (r['name'], normalise(r['proof']))
                first.setdefault(k, rr)
        # 'round' field vs first appearance
        round_field_min = collections.defaultdict(lambda: 99)
        for r in recs8:
            k = (r['name'], normalise(r['proof'])); round_field_min[k] = min(round_field_min[k], r['round'])
        round_field_agree = sum(first[k] == round_field_min[k] for k in first)
        # per target
        solved_round, acq_round, acq_unpruned_round = {}, {}, {}
        proofs_by_target = collections.defaultdict(dict)
        bad_parse = 0; n_ver = 0; ver_fail = 0; unknown_names = 0
        whist, phist = collections.Counter(), collections.Counter()
        pat_proofs = 0; depth_hist = collections.Counter()
        for (name, np_), rr in first.items():
            if name not in prompts:
                unknown_names += 1; continue
            if np_ in proofs_by_target[name]:
                continue
            c = classify(np_)
            if c is None:
                bad_parse += 1; continue
            proofs_by_target[name][np_] = c
            whist[c['written']] += 1; phist[c['pruned']] += 1; depth_hist[c['depth']] += 1
            solved_round[name] = min(solved_round.get(name, 99), rr)
            if c[pattern]:
                pat_proofs += 1
                acq_round[name] = min(acq_round.get(name, 99), rr)
            if c[pattern + '_unpruned']:
                acq_unpruned_round[name] = min(acq_unpruned_round.get(name, 99), rr)
            if verify_all:
                ok, reason, nl = verify_text(prompts[name] + ' ' + np_)
                n_ver += 1; ver_fail += (not ok) or (nl != c['written'])
        cum_solved = [sum(1 for v in solved_round.values() if v <= rr) for rr in range(1, 9)]
        cum_acq = [sum(1 for v in acq_round.values() if v <= rr) for rr in range(1, 9)]
        cum_acq_unpruned = [sum(1 for v in acq_unpruned_round.values() if v <= rr) for rr in range(1, 9)]
        solved = set(solved_round); acq = set(acq_round)
        # executor's round_8 cumulative solved for comparison
        r8 = json.load(open(f'{d}/round_8.json'))
        exec_solved = r8['targets_cum' if pool == 'targets' else 'transfer_cum']['solved']
        res = {'n': n, 'attempts_per_target': 8 * args['k'], 'solved': len(solved), 'solve_rate': len(solved) / n,
               'acquired': len(acq), 'acq_rate': len(acq) / n, 'acquired_unpruned': len(acq_unpruned_round),
               'solved_without_pattern': sorted(solved - acq), 'n_solved_without_pattern': len(solved - acq),
               'cum_solved_by_round': cum_solved, 'cum_acquired_by_round': cum_acq, 'cum_acquired_unpruned_by_round': cum_acq_unpruned,
               'first_acq_round': min(acq_round.values()) if acq_round else None,
               'distinct_norm_proofs': sum(len(v) for v in proofs_by_target.values()), 'raw_records_round8': len(recs8),
               'distinct_pattern_proofs': pat_proofs, 'written_hist': dict(sorted(whist.items())), 'pruned_hist': dict(sorted(phist.items())),
               'frontier_written_ge5': max([L for L, c in whist.items() if c >= 5], default=0),
               'frontier_pruned_ge5': max([L for L, c in phist.items() if c >= 5], default=0),
               'max_depth_hist': dict(sorted(depth_hist.items())),
               'verified': n_ver, 'verify_failures': ver_fail, 'unparsable': bad_parse, 'unknown_names': unknown_names,
               'per_round_files_subset_of_round8': subset_ok, 'round_field_agrees_with_first_file': f'{round_field_agree}/{len(first)}',
               'executor_round8_cum_solved': exec_solved, 'solved_matches_executor': exec_solved == len(solved)}
        if pool == 'targets':
            # solved by gen_lines bucket (pool's n_lines = oracle min_lines_ub)
            nl = {r['name']: r['n_lines'] for r in rd(pool_file)}
            byl = collections.defaultdict(lambda: [0, 0, 0])
            for nm in names:
                byl[nl[nm]][0] += 1; byl[nl[nm]][1] += nm in solved; byl[nl[nm]][2] += nm in acq
            res['by_min_lines(n,solved,acquired)'] = {str(k): v for k, v in sorted(byl.items())}
        out[pool] = res
    return out


# ---------------------------------------------------------------- coverage (base pass@1e4)
def count_cov(fn, pattern, pool_file):
    recs = rd(fn); prompts = {r['name']: r['prompt'] for r in rd(pool_file)}
    n = len(recs); solved = 0; acq = 0; acq_hits = 0; hits = 0; ver = 0; vfail = 0; tried = set()
    solved_names, acq_names = [], []
    for r in recs:
        tried.add(r['n_tried'])
        if r['n_ok'] > 0:
            solved += 1; solved_names.append(r['name'])
        hits += r['n_ok']
        got = False
        for p in r['proofs']:
            c = classify(p['proof'])
            ok, reason, nl = verify_text(prompts[r['name']] + ' ' + p['proof']); ver += 1; vfail += not ok
            if c and c[pattern]:
                got = True; acq_hits += p['count']
        if got:
            acq += 1; acq_names.append(r['name'])
    return {'file': fn, 'n': n, 'n_tried_values': sorted(tried), 'solved': solved, 'solved_rate': solved / n, 'acquired(pattern proof seen)': acq,
            'total_ok_samples': hits, 'pattern_ok_samples': acq_hits, 'solved_without_pattern': sorted(set(solved_names) - set(acq_names)),
            'verified_distinct_proofs': ver, 'verify_failures': vfail,
            'pass_rate_per_sample': hits / (n * 10000), 'pattern_rate_per_sample': acq_hits / (n * 10000)}


# ---------------------------------------------------------------- oracle / pools
def check_nec(fn, pattern, bound=10):
    recs = rd(fn); n = len(recs)
    reach = [r for r in recs if r['min_lines_ub'] is not None]
    req = [r for r in recs if r['requires']]
    to = sum(1 for r in recs if r.get('timeout') or r.get('r_timeout'))
    bad_req = [r['name'] for r in req if not (r['r_min_lines_ub'] is None and not r['r_timeout'] and r['min_lines_ub'] <= bound)]
    # my predicate on the oracle proof of every required theorem; and on the restricted proof (must lack the pattern)
    my_ok = 0; my_bad = []; rver_fail = 0; uver_fail = 0; r_has_pattern = []
    for r in recs:
        if r['proof']:
            ok, _, nl = verify_text(r['prompt'] + ' ' + r['proof']); uver_fail += (not ok) or nl != r['min_lines_ub']
            c = classify(r['proof'])
            if r['requires']:
                if c and c[pattern]:
                    my_ok += 1
                else:
                    my_bad.append(r['name'])
        if r.get('r_proof'):
            ok, _, nl = verify_text(r['prompt'] + ' ' + r['r_proof']); rver_fail += (not ok) or nl != r['r_min_lines_ub']
            c = classify(r['r_proof'])
            if c and c[pattern]:
                r_has_pattern.append(r['name'])
    lh = collections.Counter(r['min_lines_ub'] for r in reach)
    lh_req = collections.Counter(r['min_lines_ub'] for r in req)
    sch = collections.Counter(r.get('schema') for r in req)
    res = {'file': fn, 'n': n, 'reachable_le_bound': len(reach), 'reachable_by_len': dict(sorted(lh.items())), 'required': len(req),
           'required_by_len': dict(sorted(lh_req.items())), 'timeouts': to, 'required_but_inconsistent_fields': bad_req,
           'required_with_my_pattern_on_oracle_proof': my_ok, 'required_without_my_pattern': my_bad,
           'restricted_proofs_containing_pattern(my_pred)': r_has_pattern, 'oracle_proof_verify_failures': uver_fail, 'restricted_proof_verify_failures': rver_fail,
           'requires_wide': sum(1 for r in recs if r.get('requires_wide')), 'classical_only': sum(1 for r in recs if r.get('classical_only'))}
    if pattern == 'reductio':
        res['required_by_schema'] = dict(sch.most_common())
        res['restricted_search_found_any'] = sum(1 for r in recs if r['r_min_lines_ub'] is not None)
    return res


def check_pool(fn, pattern, nec_files, min_lines):
    recs = rd(fn); n = len(recs)
    nec = {}
    for nf in nec_files:
        for r in rd(nf):
            nec[rkey(thm_of_prompt(r['prompt']))] = r
    keys = [rkey(thm_of_prompt(r['prompt'])) for r in recs]
    key_matches = sum(k == r['key'] for k, r in zip(keys, recs))
    thm_matches = sum(thm_of_prompt(r['prompt']) == r['thm'].strip() for r in recs)
    dn_in_seq = [r['name'] for r in recs if '( ~ ( ~' in r['thm']]
    vfail = 0; pat_ok = 0; req_ok = 0; in_nec = 0; nl_ok = 0; nl_hist = collections.Counter()
    for r in recs:
        ok, _, nl = verify_text(r['prompt'] + ' ' + r['oracle_proof']); vfail += not ok
        c = classify(r['oracle_proof']); pat_ok += bool(c and c[pattern])
        nl_ok += (nl == r['n_lines'] == r['min_lines_ub']) and r['n_lines'] >= min_lines
        nl_hist[r['n_lines']] += 1
        nr = nec.get(rkey(thm_of_prompt(r['prompt'])))
        if nr is not None:
            in_nec += 1; req_ok += bool(nr['requires']) and nr['r_min_lines_ub'] is None and not nr['r_timeout'] and nr['min_lines_ub'] == r['min_lines_ub']
        req_ok += 0
    return {'file': fn, 'n': n, 'distinct_keys': len(set(keys)), 'my_key_equals_key_field': key_matches, 'thm_equals_prompt': thm_matches,
            'double_negation_in_sequent': dn_in_seq, 'oracle_proof_verify_failures': vfail, 'oracle_proof_has_pattern(my_pred)': pat_ok,
            'n_lines_consistent_and_ge_min': nl_ok, 'n_lines_hist': dict(sorted(nl_hist.items())),
            'found_in_nec_files': in_nec, 'nec_record_requires_and_same_proof': req_ok,
            'schema_hist': dict(collections.Counter(r.get('schema') for r in recs).most_common()) if pattern == 'reductio' else None,
            'source_hist': dict(collections.Counter(r.get('source') for r in recs)), 'keys': keys}


# ---------------------------------------------------------------- training files: cap, pattern frequency, keys
def scan_train(fn, pattern):
    n = 0; cap_viol = 0; pat = 0; loose = 0; keys = set(); vfail = 0; unp = 0; maxl = 0
    rng = random.Random(0); sample_ver = 0
    for l in open(fn):
        if not l.strip():
            continue
        r = json.loads(l); n += 1
        keys.add(rkey(thm_of_prompt(r['prompt'])))
        c = classify(r['proof'])
        if c is None:
            unp += 1; continue
        maxl = max(maxl, c['written'])
        cap_viol += c['written'] > 6
        pat += c[pattern]
        if pattern == 'derived_ore_strict':
            loose += c['derived_ore_loose']
        if rng.random() < 0.01:
            ok, _, _ = verify_text(r['prompt'] + ' ' + r['proof']); sample_ver += 1; vfail += not ok
    return {'file': fn, 'n': n, 'cap6_violations': cap_viol, 'max_written': maxl, f'{pattern}_proofs(my_pred, pruned)': pat, 'pattern_frac': pat / n,
            'derived_ore_loose_proofs': loose if pattern == 'derived_ore_strict' else None, 'unparsable': unp, 'distinct_keys': len(keys),
            'verified_sample': sample_ver, 'verify_failures': vfail}, keys


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else 'all'
    os.makedirs('review_out', exist_ok=True)
    if what in ('arms', 'all'):
        res = []
        for pattern, arms in ARMS.items():
            pt, pf = POOLS[pattern]
            for arm in arms:
                r = count_arm(arm, pattern, pt, pf); res.append(r)
                t, tr = r['targets'], r['transfer']
                print(f"{arm:45s} targets solved {t['solved']:3d}/{t['n']} acq {t['acquired']:3d} ({t['acq_rate']:.3f}) unpr {t['acquired_unpruned']:3d} noPat {t['n_solved_without_pattern']:2d} "
                      f"proofs {t['distinct_norm_proofs']:5d} pat {t['distinct_pattern_proofs']:5d} ver {t['verified']}/{t['verify_failures']}f exec {t['executor_round8_cum_solved']} | "
                      f"transfer solved {tr['solved']:3d}/{tr['n']} acq {tr['acquired']:3d} noPat {tr['n_solved_without_pattern']:2d} exec {tr['executor_round8_cum_solved']} | cum acq {t['cum_acquired_by_round']}", flush=True)
        json.dump(res, open('review_out/arms.json', 'w'), indent=1)
    if what in ('cov', 'all'):
        res = []
        for fn, pattern, pool in [('artifacts/r5/cov_reductio_f0_s0_req.s0.jsonl', 'reductio', POOLS['reductio'][0]), ('artifacts/r5/cov_reductio_f0_s1_req.s0.jsonl', 'reductio', POOLS['reductio'][0]),
                                  ('artifacts/r5/cov_reductio_f0_s2_req.s0.jsonl', 'reductio', POOLS['reductio'][0]),
                                  ('artifacts/r5/cov_derived_ore_strict_f0_c8_s0_req.s0.jsonl', 'derived_ore_strict', POOLS['derived_ore_strict'][0]),
                                  ('artifacts/r5/cov_derived_ore_strict_f0_c8_s1_req.s0.jsonl', 'derived_ore_strict', POOLS['derived_ore_strict'][0])]:
            r = count_cov(fn, pattern, pool); res.append(r)
            print(f"COV {fn:55s} solved {r['solved']:3d}/{r['n']} pattern {r['acquired(pattern proof seen)']:3d} noPat {len(r['solved_without_pattern'])} ok-samples {r['total_ok_samples']} pat-samples {r['pattern_ok_samples']} ver {r['verified_distinct_proofs']}/{r['verify_failures']}f tried {r['n_tried_values']}", flush=True)
        json.dump(res, open('review_out/cov.json', 'w'), indent=1)
    if what in ('nec', 'all'):
        res = []
        for fn, pattern in [('data/p2/run5_reductio_nec.jsonl', 'reductio'), ('data/p2/run5_c8_nec.jsonl', 'derived_ore_strict'), ('data/p2/run5_c8c_nec.jsonl', 'derived_ore_strict'), ('data/p2/run5_c8d_nec.jsonl', 'derived_ore_strict')]:
            r = check_nec(fn, pattern); res.append(r)
            print(f"NEC {fn:35s} n {r['n']} reach {r['reachable_le_bound']} {r['reachable_by_len']} required {r['required']} {r['required_by_len']} timeouts {r['timeouts']} badreq {len(r['required_but_inconsistent_fields'])} "
                  f"mypat-ok {r['required_with_my_pattern_on_oracle_proof']} mypat-bad {len(r['required_without_my_pattern'])} r-has-pat {len(r['restricted_proofs_containing_pattern(my_pred)'])} verfail {r['oracle_proof_verify_failures']}/{r['restricted_proof_verify_failures']} wide {r['requires_wide']}", flush=True)
            if pattern == 'reductio':
                print('   required by schema', r['required_by_schema'], 'restricted found any:', r['restricted_search_found_any'])
        json.dump(res, open('review_out/nec.json', 'w'), indent=1)
    if what in ('pools', 'all'):
        res = {}
        for fn, pattern, necs, ml in [('data/p2/targets_reductio_req.jsonl', 'reductio', ['data/p2/run5_reductio_nec.jsonl'], 7), ('data/p2/transfer_reductio_req.jsonl', 'reductio', ['data/p2/run5_reductio_nec.jsonl'], 7),
                                      ('data/p2/targets_derived_ore_req.jsonl', 'derived_ore_strict', ['data/p2/run5_c8_nec.jsonl', 'data/p2/run5_c8c_nec.jsonl', 'data/p2/run5_c8d_nec.jsonl'], 9),
                                      ('data/p2/transfer_derived_ore_req.jsonl', 'derived_ore_strict', ['data/p2/run5_c8_nec.jsonl', 'data/p2/run5_c8c_nec.jsonl', 'data/p2/run5_c8d_nec.jsonl'], 9)]:
            r = check_pool(fn, pattern, necs, ml); res[fn] = r
            print(f"POOL {fn:40s} n {r['n']} keys {r['distinct_keys']} keyfield-ok {r['my_key_equals_key_field']} thm-ok {r['thm_equals_prompt']} ~~ {len(r['double_negation_in_sequent'])} verfail {r['oracle_proof_verify_failures']} pat {r['oracle_proof_has_pattern(my_pred)']} nl-ok {r['n_lines_consistent_and_ge_min']} {r['n_lines_hist']} in-nec {r['found_in_nec_files']} nec-req-same {r['nec_record_requires_and_same_proof']} src {r['source_hist']}", flush=True)
            if r['schema_hist']:
                print('   schema', r['schema_hist'])
        kt, ktr = set(res['data/p2/targets_reductio_req.jsonl']['keys']), set(res['data/p2/transfer_reductio_req.jsonl']['keys'])
        print('reductio targets∩transfer keys', len(kt & ktr))
        kt2, ktr2 = set(res['data/p2/targets_derived_ore_req.jsonl']['keys']), set(res['data/p2/transfer_derived_ore_req.jsonl']['keys'])
        print('derived-ORE targets∩transfer keys', len(kt2 & ktr2))
        json.dump(res, open('review_out/pools.json', 'w'), indent=1)
    if what in ('splits', 'all'):
        evals = {}
        for fn in ['data/p2/targets_reductio_req.jsonl', 'data/p2/transfer_reductio_req.jsonl', 'data/p2/targets_derived_ore_req.jsonl', 'data/p2/transfer_derived_ore_req.jsonl',
                   'data/p2/heldout.jsonl', 'data/p2/heldout_c8.jsonl', 'targets/validation_36.jsonl']:
            evals[fn] = {rkey(thm_of_prompt(r['prompt'])) for r in rd(fn)}
        res = {}
        for fn, pattern in [('data/p2/train_reductio_f0.jsonl', 'reductio'), ('data/p2/train_reductio_f0.1.jsonl', 'reductio'),
                            ('data/p2/train_derived_ore_strict_f0_c8.jsonl', 'derived_ore_strict'), ('data/p2/train_derived_ore_strict_f0.01_c8.jsonl', 'derived_ore_strict')]:
            r, keys = scan_train(fn, pattern)
            r['overlap_with'] = {ev: len(keys & ek) for ev, ek in evals.items()}
            res[fn] = r
            print(f"TRAIN {fn:50s} n {r['n']} cap6viol {r['cap6_violations']} maxL {r['max_written']} pattern {r[f'{pattern}_proofs(my_pred, pruned)']} ({r['pattern_frac']:.4f}) loose {r['derived_ore_loose_proofs']} unp {r['unparsable']} ver {r['verified_sample']}/{r['verify_failures']}f overlaps {r['overlap_with']}", flush=True)
        # eval pools vs each other
        names = list(evals)
        cross = {f'{a} ∩ {b}': len(evals[a] & evals[b]) for i, a in enumerate(names) for b in names[i + 1:]}
        print('EVAL cross overlaps (nonzero):', {k: v for k, v in cross.items() if v})
        res['eval_cross'] = cross
        json.dump(res, open('review_out/splits.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
