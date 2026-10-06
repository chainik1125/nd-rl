#!/usr/bin/env python3
"""Reviewer's independent recount for round2-run4 (GRPO vs expert iteration at f = 0).

Reuses the reviewer's own module from the run-5 review (review_run5_recount.py: own parser, dependency pruning,
start-index normaliser, box-depth counter, atom-renaming key); only nd_verify is shared with the executor.
Outputs artifacts/review_r4/recount.json and prints a summary.

  python3 review_run4_recount.py            # everything (GRPO arms, EI arms, coverage, checkpoints)
  python3 review_run4_recount.py --train    # also scan the three Stage-1 sets (slow)
"""
import json, re, os, sys, glob, collections, random, zipfile, pickle, io, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nd_verify import verify_text
from review_run5_recount import parse_proof, prune, normalise, max_depth, classify, rd, rkey, thm_of_prompt, scan_train

OUT = 'artifacts/review_r4'
GRPO = sorted(os.path.basename(d) for d in glob.glob('artifacts/r4/grpo_g*'))
EI = ['ei_depth3_f0_a1_s0', 'ei_depth3_f0_a1_s1', 'ei_depth3_f0_a2_s0', 'ei_depth3_f0_a2_s1', 'ei_depth3_f0_a3_s0', 'ei_depth3_f0_a3_s1']
TARGETS, TRANSFER = 'data/p2/targets_depth3.jsonl', 'data/p2/transfer_depth3.jsonl'
IGN_THR = 20   # run-3 review's ignition threshold for depth-3 (cumulative pattern theorems)


def is_depth3(np_):
    lines = parse_proof(np_)
    if lines is None:
        return None
    return max_depth(prune(lines)) >= 3


def per_target_rounds(records_by_round, prompts, verify=True):
    """records_by_round: {r: [records]} (cumulative files, or the round-8 file split by min 'round' field).
    -> per-target first solved / first depth-3 round, verification counts, histograms."""
    first = {}
    for rr in sorted(records_by_round):
        for x in records_by_round[rr]:
            first.setdefault((x['name'], normalise(x['proof'])), rr)
    solved_round, acq_round = {}, {}
    n_ver = vfail = bad = unknown = 0; n_pat = 0
    whist = collections.Counter(); dhist = collections.Counter()
    for (name, np_), rr in first.items():
        if name not in prompts:
            unknown += 1; continue
        lines = parse_proof(np_)
        if lines is None:
            bad += 1; continue
        pr = prune(lines)
        d = max_depth(pr)
        whist[len(lines)] += 1; dhist[d] += 1
        solved_round[name] = min(solved_round.get(name, 99), rr)
        if d >= 3:
            n_pat += 1; acq_round[name] = min(acq_round.get(name, 99), rr)
        if verify:
            ok, reason, nl = verify_text(prompts[name] + ' ' + np_); n_ver += 1
            vfail += (not ok) or (nl != len(lines))
    return {'first': first, 'solved_round': solved_round, 'acq_round': acq_round, 'distinct': len(first), 'pattern_proofs': n_pat,
            'verified': n_ver, 'verify_failures': vfail, 'unparsable': bad, 'unknown_names': unknown,
            'written_hist': dict(sorted(whist.items())), 'depth_hist': dict(sorted(dhist.items()))}


def cum(rounds_dict, R):
    return [sum(1 for v in rounds_dict.values() if v <= r) for r in range(1, R + 1)]


def ckpt_extra(fn):
    """Read the 'extra' and 'tok_mode' entries of a torch checkpoint without torch (stubbed unpickler)."""
    class Stub:
        def __init__(self, *a, **k): pass
        def __setstate__(self, s): self.s = s
    class U(pickle.Unpickler):
        def find_class(self, mod, name):
            if mod.startswith('torch') or mod in ('model', 'collections') and name == 'OrderedDict' and False:
                return Stub
            try:
                return super().find_class(mod, name)
            except Exception:
                return Stub
        def persistent_load(self, pid):
            return None
    with zipfile.ZipFile(fn) as z:
        name = [n for n in z.namelist() if n.endswith('data.pkl')][0]
        obj = U(io.BytesIO(z.read(name))).load()
    return {'extra': obj.get('extra'), 'tok_mode': obj.get('tok_mode'), 'n_state_tensors': len(obj.get('state', {}))}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--train', action='store_true'); ap.add_argument('--no_verify', action='store_true')
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    verify = not a.no_verify
    tg = rd(TARGETS); tr = rd(TRANSFER)
    prompts_t = {r['name']: r['prompt'] for r in tg}; prompts_x = {r['name']: r['prompt'] for r in tr}
    nl_t = {r['name']: r['n_lines'] for r in tg}
    res = {'targets_n': len(tg), 'transfer_n': len(tr), 'grpo': {}, 'ei': {}, 'coverage': {}, 'ckpts': {}}

    # ------------------------------------------------------------ GRPO arms
    for arm in GRPO:
        d = f'artifacts/r4/{arm}'
        args = json.load(open(f'{d}/args.json'))
        rounds = sorted(int(re.fullmatch(r'round_(\d+)\.json', os.path.basename(f)).group(1)) for f in glob.glob(f'{d}/round_*.json'))
        R = max(rounds)
        o = {'args': {k: args.get(k) for k in ('init', 'targets', 'transfer', 'heldout', 'rounds', 'k', 'group', 'prompts', 'temperature', 'lr', 'seed', 'max_new', 'lp_batch')},
             'rounds_present': rounds}
        # per-round json: budget, steps coverage, reward / variance recomputed from the step lists
        steps_all = []; per_round = []
        for r in rounds:
            st = json.load(open(f'{d}/round_{r}.json'))
            ss = st['steps']; steps_all += [s['step'] for s in ss]
            per_round.append({'round': r, 'step': st['step'], 'samples': st['samples'], 'n_steps_in_file': len(ss),
                              'mean_reward(my mean of steps)': sum(s['mean_reward'] for s in ss) / len(ss),
                              'frac_var(my mean of steps)': sum(s['frac_groups_with_variance'] for s in ss) / len(ss),
                              'frac_var(executor field)': st['frac_groups_with_variance_mean'],
                              'frac_var_first_step': ss[0]['frac_groups_with_variance'], 'frac_var_last_step': ss[-1]['frac_groups_with_variance'],
                              'grad_norm_max': max(s['grad_norm'] for s in ss),
                              'exec_targets_solved': st['targets_cum']['solved'], 'exec_transfer_solved': st['transfer_cum']['solved'],
                              'transfer_greedy': st['transfer_greedy']['rate'], 'heldout_greedy': st['heldout_greedy']['rate'],
                              'heldout_greedy_by_len': {k: v['rate'] for k, v in st['heldout_greedy']['by_len'].items()},
                              'samples_at_last_step': ss[-1]['samples']})
        o['per_round'] = per_round
        o['steps_covered'] = f'{len(set(steps_all))} distinct of {min(steps_all)}..{max(steps_all)}; duplicates {len(steps_all) - len(set(steps_all))}'
        o['budget_total_samples'] = per_round[-1]['samples']; o['per_target_samples'] = per_round[-1]['samples'] / len(tg)
        # found files: nesting, round field vs first file, per-target rounds
        recs = {r: rd(f'{d}/found_{r}.jsonl') for r in rounds}
        setr = {r: {(x['name'], normalise(x['proof'])) for x in recs[r]} for r in rounds}
        o['per_round_files_nested'] = all(setr[rounds[i]] <= setr[rounds[i + 1]] for i in range(len(rounds) - 1))
        o['nesting_violations'] = {str(rounds[i + 1]): len(setr[rounds[i]] - setr[rounds[i + 1]]) for i in range(len(rounds) - 1) if not setr[rounds[i]] <= setr[rounds[i + 1]]}
        o['raw_records_last'] = len(recs[R]); o['distinct_in_last_file'] = len(setr[R])
        o['duplicate_records_in_last_file'] = len(recs[R]) - len({(x['name'], x['proof']) for x in recs[R]})
        # (a) rounds from per-round files (first file containing the proof); (b) rounds from the 'round' field of the last file (min per key)
        pt_files = per_target_rounds(recs, prompts_t, verify)
        field = collections.defaultdict(lambda: 99)
        for x in recs[R]:
            k = (x['name'], normalise(x['proof'])); field[k] = min(field[k], x['round'])
        by_field = collections.defaultdict(list)
        for x in recs[R]:
            by_field[x['round']].append(x)
        pt_field = per_target_rounds(by_field, prompts_t, False)
        o['round_field_agrees_with_first_file'] = f"{sum(pt_files['first'][k] == field[k] for k in pt_files['first'])}/{len(pt_files['first'])}"
        o['targets'] = {'solved': len(pt_files['solved_round']), 'acquired_depth3': len(pt_files['acq_round']),
                        'cum_solved_by_round(files)': cum(pt_files['solved_round'], R), 'cum_acq_by_round(files)': cum(pt_files['acq_round'], R),
                        'cum_solved_by_round(round field)': cum(pt_field['solved_round'], R), 'cum_acq_by_round(round field)': cum(pt_field['acq_round'], R),
                        'first_acq_round': min(pt_files['acq_round'].values()) if pt_files['acq_round'] else None,
                        'ignition_round(thr 20)': next((r for r, c in enumerate(cum(pt_files['acq_round'], R), 1) if c >= IGN_THR), None),
                        'distinct_norm_proofs': pt_files['distinct'], 'distinct_pattern_proofs': pt_files['pattern_proofs'],
                        'verified': pt_files['verified'], 'verify_failures': pt_files['verify_failures'], 'unparsable': pt_files['unparsable'], 'unknown_names': pt_files['unknown_names'],
                        'written_hist': pt_files['written_hist'], 'depth_hist': pt_files['depth_hist'],
                        'solved_without_depth3': len(set(pt_files['solved_round']) - set(pt_files['acq_round'])),
                        'exec_solved_last': per_round[-1]['exec_targets_solved'],
                        'solved_matches_executor_every_round': [cum(pt_files['solved_round'], R)[i] == per_round[i]['exec_targets_solved'] for i in range(len(rounds))]}
        byl = collections.defaultdict(lambda: [0, 0, 0])
        for nm in prompts_t:
            byl[nl_t[nm]][0] += 1; byl[nl_t[nm]][1] += nm in pt_files['solved_round']; byl[nl_t[nm]][2] += nm in pt_files['acq_round']
        o['targets']['by_min_lines(n,solved,acquired)'] = {str(k): v for k, v in sorted(byl.items())}
        o['targets']['acq_names'] = sorted(pt_files['acq_round'])
        # transfer: each found_transfer_<r> is a fresh pass@32 of the boundary model (not cumulative)
        tro = {}
        union_solved, union_acq = set(), set()
        per = []
        for r in rounds:
            rx = rd(f'{d}/found_transfer_{r}.jsonl')
            ptx = per_target_rounds({r: rx}, prompts_x, verify and r == R)
            per.append({'round': r, 'solved': len(ptx['solved_round']), 'acq': len(ptx['acq_round']), 'distinct': ptx['distinct'], 'pattern_proofs': ptx['pattern_proofs'],
                        'verified': ptx['verified'], 'verify_failures': ptx['verify_failures'], 'unparsable': ptx['unparsable'], 'exec_solved': per_round[r - 1]['exec_transfer_solved']})
            union_solved |= set(ptx['solved_round']); union_acq |= set(ptx['acq_round'])
        tro['per_boundary'] = per; tro['union_solved'] = len(union_solved); tro['union_acq'] = len(union_acq)
        tro['last_boundary_solved'] = per[-1]['solved']; tro['last_boundary_acq'] = per[-1]['acq']
        o['transfer'] = tro
        res['grpo'][arm] = o
        print(f"{arm:15s} G={args['group']:2d} R={R} solved {o['targets']['solved']} acq {o['targets']['acquired_depth3']} per-round acq {o['targets']['cum_acq_by_round(files)']} "
              f"solved {o['targets']['cum_solved_by_round(files)']} exec-match {all(o['targets']['solved_matches_executor_every_round'])} nested {o['per_round_files_nested']} "
              f"field-agree {o['round_field_agrees_with_first_file']} ver {o['targets']['verified']}/{o['targets']['verify_failures']} "
              f"transfer last {tro['last_boundary_solved']}/{tro['last_boundary_acq']} union {tro['union_solved']}/{tro['union_acq']} "
              f"heldout {[round(p['heldout_greedy'], 3) for p in per_round]} var {[round(p['frac_var(my mean of steps)'], 2) for p in per_round]} reward {[round(p['mean_reward(my mean of steps)'], 2) for p in per_round]}", flush=True)

    # ------------------------------------------------------------ EI arms (found_8 only; rounds from the min 'round' field)
    for arm in EI:
        d = f'artifacts/p2/{arm}'
        args = json.load(open(f'{d}/args.json'))
        recs = rd(f'{d}/found_8.jsonl')
        by_field = collections.defaultdict(list)
        for x in recs:
            by_field[x['round']].append(x)
        pt = per_target_rounds(by_field, prompts_t, verify)
        per_round = []
        for r in range(1, 9):
            st = json.load(open(f'{d}/round_{r}.json'))
            per_round.append({'round': r, 'exec_targets_solved': st['targets_cum']['solved'], 'exec_transfer_cum': st['transfer_cum']['solved'],
                              'exec_transfer_this_round(pass@32)': st['transfer_round']['solved'], 'heldout_greedy': st['heldout_greedy']['rate'],
                              'transfer_greedy': st['transfer_greedy']['rate'], 'target_sample_acc': st.get('target_sample_acc')})
        rx = rd(f'{d}/found_transfer_8.jsonl')
        byx = collections.defaultdict(list)
        for x in rx:
            byx[x['round']].append(x)
        ptx = per_target_rounds(byx, prompts_x, verify)
        # transfer pattern theorems found in round 8's own 32 samples (comparable to a GRPO boundary file)
        r8x = per_target_rounds({8: byx[8]}, prompts_x, False)
        o = {'args': {k: args.get(k) for k in ('init', 'targets', 'transfer', 'train', 'rounds', 'k', 'temperature', 'seed', 'retain', 'steps', 'lr')},
             'per_round': per_round,
             'targets': {'solved': len(pt['solved_round']), 'acquired_depth3': len(pt['acq_round']), 'cum_solved_by_round': cum(pt['solved_round'], 8), 'cum_acq_by_round': cum(pt['acq_round'], 8),
                         'first_acq_round': min(pt['acq_round'].values()) if pt['acq_round'] else None,
                         'ignition_round(thr 20)': next((r for r, c in enumerate(cum(pt['acq_round'], 8), 1) if c >= IGN_THR), None),
                         'distinct_norm_proofs': pt['distinct'], 'raw_records': len(recs), 'distinct_pattern_proofs': pt['pattern_proofs'],
                         'verified': pt['verified'], 'verify_failures': pt['verify_failures'], 'unparsable': pt['unparsable'],
                         'written_hist': pt['written_hist'], 'depth_hist': pt['depth_hist'],
                         'solved_matches_executor_every_round': [cum(pt['solved_round'], 8)[i] == per_round[i]['exec_targets_solved'] for i in range(8)],
                         'acq_names': sorted(pt['acq_round'])},
             'transfer': {'union_solved': len(ptx['solved_round']), 'union_acq': len(ptx['acq_round']), 'cum_acq_by_round': cum(ptx['acq_round'], 8),
                          'round8_only_solved(records with round 8)': len(r8x['solved_round']), 'round8_only_acq': len(r8x['acq_round']),
                          'verified': ptx['verified'], 'verify_failures': ptx['verify_failures']}}
        byl = collections.defaultdict(lambda: [0, 0, 0])
        for nm in prompts_t:
            byl[nl_t[nm]][0] += 1; byl[nl_t[nm]][1] += nm in pt['solved_round']; byl[nl_t[nm]][2] += nm in pt['acq_round']
        o['targets']['by_min_lines(n,solved,acquired)'] = {str(k): v for k, v in sorted(byl.items())}
        res['ei'][arm] = o
        print(f"{arm:20s} solved {o['targets']['solved']} acq {o['targets']['acquired_depth3']} per-round acq {o['targets']['cum_acq_by_round']} solved {o['targets']['cum_solved_by_round']} "
              f"exec-match {all(o['targets']['solved_matches_executor_every_round'])} ver {pt['verified']}/{pt['verify_failures']} transfer union {o['transfer']['union_solved']}/{o['transfer']['union_acq']} "
              f"r8-only {o['transfer']['round8_only_solved(records with round 8)']}/{o['transfer']['round8_only_acq']} heldout {[round(p['heldout_greedy'], 3) for p in per_round]}", flush=True)

    # ------------------------------------------------------------ base rates (Stage-1 models, ignition study's coverage files)
    for fn in sorted(glob.glob('artifacts/ign/cov_depth3_f0_a[123]_s[01].s0.jsonl')):
        recs = rd(fn)
        n = len(recs); tried = sum(r['n_tried'] for r in recs); ok = sum(r['n_ok'] for r in recs)
        solved = sum(1 for r in recs if r['n_ok'] > 0); pat_thm = 0; pat_hits = 0; ver = 0; vfail = 0
        for r in recs:
            got = False
            for p in r.get('proofs', []):
                if is_depth3(p['proof']):
                    got = True; pat_hits += p.get('count', 1)
                if verify:
                    okv, _, _ = verify_text(prompts_t[r['name']] + ' ' + p['proof']); ver += 1; vfail += not okv
            pat_thm += got
        res['coverage'][os.path.basename(fn)] = {'n_targets': n, 'samples': tried, 'ok_samples': ok, 'solved_targets': solved, 'depth3_theorems': pat_thm, 'depth3_hits': pat_hits,
                                                'depth3_rate_per_sample': pat_hits / tried if tried else None, 'verified': ver, 'verify_failures': vfail,
                                                'exec_hits_by_pattern_depth3': sum(r.get('hits_by_pattern', {}).get('depth3', 0) for r in recs)}
        print(os.path.basename(fn), res['coverage'][os.path.basename(fn)], flush=True)

    # ------------------------------------------------------------ checkpoints
    for fn in sorted(glob.glob('ckpts/r4/*.pt')):
        try:
            res['ckpts'][os.path.basename(fn)] = ckpt_extra(fn)
        except Exception as e:
            res['ckpts'][os.path.basename(fn)] = {'error': repr(e)}
        print(fn, res['ckpts'][os.path.basename(fn)], flush=True)

    # ------------------------------------------------------------ Stage-1 sets
    if a.train:
        res['train'] = {}
        for s in ('a1', 'a2', 'a3'):
            fn = f'data/p2/train_depth3_f0_{s}.jsonl'
            o, keys = scan_train(fn, 'reductio')   # scan_train reports written-length cap; add my depth-3 count separately
            n3 = 0
            for l in open(fn):
                if l.strip() and is_depth3(json.loads(l)['proof']):
                    n3 += 1
            o['depth3_proofs(my_pred, pruned)'] = n3
            tkeys = {rkey(r['thm']) for r in tg}; xkeys = {rkey(r['thm']) for r in tr}
            o['keys_in_targets'] = len(keys & tkeys); o['keys_in_transfer'] = len(keys & xkeys)
            res['train'][s] = o
            print(fn, o, flush=True)

    json.dump(res, open(f'{OUT}/recount.json', 'w'), indent=1)
    print('wrote', f'{OUT}/recount.json')


if __name__ == '__main__':
    main()
