#!/usr/bin/env python3
"""state-env driver: ladder rung T1 and the frozen control, run **in the step environment** (proposal 13).

A copy of `ladder_ei.py` with exactly three changes, so that every number is produced by the same bookkeeping as the
whole-proof control's:
  1. sampling goes through `state_sample.env_generate` (the policy sees a tactic state and writes one action) instead
     of `sample.generate` (the policy sees the theorem and writes the whole proof);
  2. the round's fine-tune calls `state_train.py` (loss on the action tokens of the found proofs' (state, action)
     pairs) instead of `train.py`;
  3. each round json carries the environment's per-step diagnostics (`env_*`).
Everything else -- the rounds, k, the retained-supervised mix, max_per_thm, rl_weight, the L* definition, the pools --
is the control's.  Lean alone decides.  The original docstring follows.

ladder-A driver: expert iteration with the rung variants T1-T6, frozen control, L_true-aware bookkeeping.
A copy of expert_iter.py with additions; nothing here writes proofs (the generator pool for T5 is verifier-checked
at generation time and again on load).

  python ladder_ei.py --init ckpts/stage1_abs.pt --name la_T1_s0 --seed 0                 # T1 baseline EI
  python ladder_ei.py ... --name la_frozen_s0 --no_train                                   # frozen control
  python ladder_ei.py ... --alloc difficulty                                               # T2
  python ladder_ei.py ... --relabel                                                        # T3
  python ladder_ei.py ... --alloc window                                                   # T4
  python ladder_ei.py ... --inject_pool data/ladder/pool_inject_cap6.jsonl --inject_round 4 # T5
  python ladder_ei.py ... --k 16 --share_dir artifacts/ladder/share_T6_s0 --siblings 2      # T6 (run twice, --sibling 0 / 1)

Per round r: (1) sample the RL targets with a per-target budget k_i (sum = N*k every round; uniform, or the T2 /
T4 allocation), verify, accumulate distinct (start-index-normalised) accepted proofs; (2) sample every transfer
theorem k times (never trained on), record; (3) greedy on transfer and on the Stage-1 held-out set; (4) unless
--no_train, fine-tune on the accumulated target proofs (<= --max_per_thm per theorem, x --rl_weight) + --retain random
Stage-1 records (+ relabelled by-products with --relabel; + the injection records in round --inject_round).
Outputs in artifacts/ladder/<name>/: round_<r>.json, found_<r>.jsonl (targets, cumulative), found_transfer_<r>.jsonl,
alloc_<r>.json (per-target k_i / tried / accepted), and for T6 found_union_<r>.jsonl.
L_true of a pool record is its `n_lines` field (data/ladder pools); L* = max L with >= 5 distinct theorems solved
at L_true >= L.
"""
import argparse, json, os, sys, random, collections, subprocess, time, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from model import load_ckpt
from state_sample import env_generate, env_stats_json
from lean_judge import verify_text    # Lean alone decides (Dan, 2026-09-27); judging goes through eval_set.judge
from prune import pruned_length
from gen import canon_key
from normalize import norm
from eval_set import judge, summarize, wilson


ENV = {'max_action': 256, 'max_steps': 48, 'stats': None}


def generate(model, tok, prompts, greedy=True, temperature=1.0, batch=512, seed=0, max_new=None):
    """the env-loop stand-in for sample.generate: same contract (ND strings out), `max_new` is per ACTION."""
    return env_generate(model, tok, prompts, greedy=greedy, temperature=temperature, max_action=ENV['max_action'],
                        max_steps=ENV['max_steps'], batch=batch, seed=seed, stats=ENV['stats'])


def read(fn):
    return [json.loads(l) for l in open(fn) if l.strip()]


def lstar(found, recs, need=5):
    """found: name -> list of proofs; recs: pool records with n_lines = L_true. -> (L*, {L: n_theorems with L_true>=L solved})"""
    solved = [r['n_lines'] for r in recs if found.get(r['name'])]
    ge = {L: sum(1 for x in solved if x >= L) for L in range(2, 21)}
    ls = max([L for L, c in ge.items() if c >= need], default=0)
    return ls, ge


def by_bin(found, recs):
    out = {}
    by = collections.defaultdict(lambda: [0, 0])
    for r in recs:
        by[r['n_lines']][0] += bool(found.get(r['name'])); by[r['n_lines']][1] += 1
    for L, (k, n) in sorted(by.items()):
        lo, hi = wilson(k, n)
        out[str(L)] = {'solved': k, 'n': n, 'rate': k / n, 'ci': [lo, hi]}
    return out


def allocate(mode, targets, k, tried, okc, found, cap, rng, r, log):
    """-> list of k_i (sum == len(targets)*k)."""
    N = len(targets); B = N * k
    if mode == 'uniform' or r == 1:
        return [k] * N
    if mode == 'difficulty':
        w = []
        for t in targets:
            n, o = tried[t['name']], okc[t['name']]
            p = (o + 0.5) / (n + 1)
            if o == 0:
                w.append(1.0)
            elif p < 0.1:
                w.append(4.0)
            elif p < 0.25:
                w.append(1.0)
            else:
                w.append(0.0)
        log['weights'] = dict(collections.Counter(w))
    elif mode == 'window':
        ls, _ = lstar(found, targets)
        lo, hi = ls + 1, ls + 3
        win = [i for i, t in enumerate(targets) if lo <= t['n_lines'] <= hi]
        log['lstar_cur'] = ls; log['window'] = [lo, hi]; log['window_n'] = len(win)
        if not win:
            return [k] * N
        kw = min(cap, B // len(win))
        rest = B - kw * len(win)
        others = [i for i in range(N) if not (lo <= targets[i]['n_lines'] <= hi)]
        ko = rest // len(others) if others else 0
        ks = [kw if lo <= t['n_lines'] <= hi else ko for t in targets]
        left = B - sum(ks)
        for i in rng.sample(range(N), min(left, N)):
            ks[i] += 1
        log['k_window'] = kw; log['k_other'] = ko
        return ks
    else:
        raise ValueError(mode)
    sw = sum(w)
    if sw == 0:
        return [k] * N
    ks = [min(cap, int(B * x / sw)) for x in w]
    left = B - sum(ks)
    idx = [i for i in range(N) if w[i] > 0]
    guard = 0
    while left > 0 and guard < 10:
        for i in rng.sample(idx, len(idx)):
            if left == 0:
                break
            if ks[i] < cap:
                ks[i] += 1; left -= 1
        guard += 1
    if left > 0:      # everything capped: spend the remainder uniformly on zero-weight targets
        zi = [i for i in range(N) if w[i] == 0] or list(range(N))
        for i in rng.sample(zi, len(zi)):
            if left == 0:
                break
            ks[i] += 1; left -= 1
        for i in range(N):
            if left == 0:
                break
            ks[i] += 1; left -= 1
    assert sum(ks) == B, (sum(ks), B)
    return ks


def sample_targets(model, tok, targets, ks, temperature, batch, seed, max_new):
    prompts, owner = [], []
    for t, ki in zip(targets, ks):
        prompts += [t['prompt']] * ki
    flat = generate(model, tok, prompts, greedy=False, temperature=temperature, batch=batch, seed=seed, max_new=max_new)
    outs, pos = [], 0
    for ki in ks:
        outs.append(flat[pos:pos + ki]); pos += ki
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--init', required=True)
    ap.add_argument('--name', required=True)
    ap.add_argument('--outdir', default='artifacts/se')
    ap.add_argument('--ckptdir', default='ckpts/se/ladder')
    ap.add_argument('--targets', default='data/ladder/rl_targets.jsonl')
    ap.add_argument('--transfer', default='data/ladder/transfer.jsonl')
    ap.add_argument('--heldout', default='data/heldout.jsonl')
    ap.add_argument('--train', default='data/train.jsonl')
    ap.add_argument('--rounds', type=int, default=8)
    ap.add_argument('--k', type=int, default=32)
    ap.add_argument('--temperature', type=float, default=0.8)
    ap.add_argument('--max_new', type=int, default=512, help='unused in the environment; see --max_action')
    ap.add_argument('--max_action', type=int, default=256, help='token budget for ONE action')
    ap.add_argument('--max_steps', type=int, default=48, help='action budget for one attempt')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--no_train', action='store_true')
    ap.add_argument('--ft_steps', type=int, default=600)
    ap.add_argument('--ft_lr', type=float, default=3e-4)
    ap.add_argument('--ft_recs', type=int, default=128, help='whole proofs per fine-tune step (the control trains on 128)')
    ap.add_argument('--retain', type=int, default=20000)
    ap.add_argument('--max_per_thm', type=int, default=4)
    ap.add_argument('--rl_weight', type=int, default=4)
    ap.add_argument('--batch', type=int, default=2048)
    ap.add_argument('--alloc', default='uniform', choices=['uniform', 'difficulty', 'window'])
    ap.add_argument('--alloc_cap', type=int, default=128)
    ap.add_argument('--relabel', action='store_true')
    ap.add_argument('--inject_pool', default=None, help='T5: cap-6 generator pool (verified); shapes chosen from unsolved targets after round inject_round-1')
    ap.add_argument('--inject_round', type=int, default=4)
    ap.add_argument('--inject_n', type=int, default=3000)
    ap.add_argument('--inject_weight', type=int, default=2)
    ap.add_argument('--share_dir', default=None, help='T6: directory where siblings exchange found proofs each round')
    ap.add_argument('--siblings', type=int, default=1)
    ap.add_argument('--sibling', type=int, default=0)
    ap.add_argument('--start_round', type=int, default=1)
    ap.add_argument('--resume', action='store_true', help='resume from artifacts/<name>/found_<start_round-1>.jsonl and its ckpt')
    a = ap.parse_args()
    out = f'{a.outdir}/{a.name}'
    os.makedirs(out, exist_ok=True); os.makedirs(os.path.dirname(a.ckptdir + '/x'), exist_ok=True)
    json.dump(vars(a), open(f'{out}/args.json', 'w'), indent=1)
    dev = 'cuda'
    ENV['max_action'] = a.max_action; ENV['max_steps'] = a.max_steps
    targets, transfer, heldout = read(a.targets), read(a.transfer), read(a.heldout)
    eval_keys = {r['key'] for r in transfer} | {r['key'] for r in heldout} | {r['key'] for r in targets}
    eval_keys |= {canon_key(json.loads(l)['thm'].strip()) for l in open('targets/validation_36.jsonl')}
    for fn in ('data/ladder/reserve.jsonl', 'data/transfer.jsonl'):
        if os.path.exists(fn):
            eval_keys |= {r['key'] for r in read(fn)}
    train_recs = read(a.train)
    rng = random.Random(a.seed * 7919 + a.sibling)
    ckpt = a.init
    found = collections.defaultdict(list)     # target name -> [{proof, norm, written, pruned, round}]
    found_t = collections.defaultdict(list)
    tried, okc = collections.Counter(), collections.Counter()
    relabelled = {}
    inject_recs = []
    if a.resume:
        r0 = a.start_round - 1
        for l in open(f'{out}/found_{r0}.jsonl'):
            x = json.loads(l); found[x['name']].append({k: x[k] for k in ('proof', 'norm', 'written', 'pruned', 'round')})
        for l in open(f'{out}/found_transfer_{r0}.jsonl'):
            x = json.loads(l); found_t[x['name']].append({k: x[k] for k in ('proof', 'norm', 'written', 'pruned', 'round')})
        if a.relabel and os.path.exists(f'{out}/relabelled_{r0}.jsonl'):
            for l in open(f'{out}/relabelled_{r0}.jsonl'):
                x = json.loads(l); relabelled[x['thm']] = x
        al = json.load(open(f'{out}/alloc_{r0}.json'))
        tried.update(al['tried']); okc.update(al['accepted'])
        if not a.no_train:
            ckpt = f'{a.ckptdir}/{a.name}_r{r0}.pt'
        print(f'resumed round {r0}: {sum(len(v) for v in found.values())} target proofs; ckpt {ckpt}', flush=True)
    for r in range(a.start_round, a.start_round + a.rounds):
        t0 = time.time()
        model, tok, _ = load_ckpt(ckpt, dev)
        seed = a.seed * 1000 + r + 100 * a.sibling
        ENV['stats'] = {}
        stats = {'round': r, 'ckpt': ckpt, 'k': a.k, 'temperature': a.temperature, 'seed': seed, 'max_new': a.max_new, 'alloc': a.alloc}
        alog = {}
        ks = allocate(a.alloc, targets, a.k, tried, okc, found, a.alloc_cap, rng, r, alog)
        stats['alloc_log'] = alog
        stats['k_hist'] = dict(sorted(collections.Counter(ks).items()))
        # 1. RL targets
        sub = [(t, ki) for t, ki in zip(targets, ks) if ki > 0]
        outs = sample_targets(model, tok, [t for t, _ in sub], [ki for _, ki in sub], a.temperature, a.batch, seed, a.max_new)
        rows = judge([t for t, _ in sub], outs, 'n_lines')
        new_this = 0
        for (t, ki), row in zip(sub, rows):
            tried[t['name']] += ki; okc[t['name']] += row['n_ok']
            have = {x['norm'] for x in found[t['name']]}
            for p, wl, pl in zip(row['proofs'], row['written_lens'], row['pruned_lens']):
                pn = norm(p)
                if pn not in have:
                    have.add(pn)
                    found[t['name']].append({'proof': p, 'norm': pn, 'written': wl, 'pruned': pl, 'round': r}); new_this += 1
        stats['targets_round'] = summarize(rows, 'n_lines', f'[{a.name} r{r}] targets (this round)')
        stats['new_proofs_this_round'] = new_this
        stats['target_samples'] = sum(ks); stats['target_sample_acc'] = sum(x['n_ok'] for x in rows) / max(1, sum(ks))
        # relabelling by-products (T3)
        if a.relabel:
            from expert_iter import relabel_batch      # run lean-judge renamed this; import it only for T3
            nrl = 0
            for (t, ki), ps in zip(sub, outs):
                for p in ps:
                    rl = relabel_batch([(t['prompt'], p)])[0]
                    if rl is None:
                        continue
                    newp, thm, nl = rl
                    key = canon_key(thm)
                    if nl < 7 or key in eval_keys or thm in relabelled:
                        continue
                    relabelled[thm] = {'prompt': newp, 'proof': p, 'n_lines': nl, 'thm': thm, 'round': r}
                    nrl += 1
            stats['relabelled_new'] = nrl; stats['relabelled_total'] = len(relabelled)
            with open(f'{out}/relabelled_{r}.jsonl', 'w') as f:
                for x in relabelled.values():
                    f.write(json.dumps(x) + '\n')
        # 2. transfer, sampled (uniform k; T6 siblings each sample k)
        prompts = [t['prompt'] for t in transfer for _ in range(a.k)]
        flat = generate(model, tok, prompts, greedy=False, temperature=a.temperature, batch=a.batch, seed=seed + 500, max_new=a.max_new)
        outs_t = [flat[i * a.k:(i + 1) * a.k] for i in range(len(transfer))]
        rows_t = judge(transfer, outs_t, 'n_lines')
        stats['transfer_round'] = summarize(rows_t, 'n_lines', f'[{a.name} r{r}] transfer (this round, pass@{a.k})')
        stats['transfer_sample_acc'] = sum(x['n_ok'] for x in rows_t) / sum(x['n_tried'] for x in rows_t)
        for t, row in zip(transfer, rows_t):
            have = {x['norm'] for x in found_t[t['name']]}
            for p, wl, pl in zip(row['proofs'], row['written_lens'], row['pruned_lens']):
                pn = norm(p)
                if pn not in have:
                    have.add(pn)
                    found_t[t['name']].append({'proof': p, 'norm': pn, 'written': wl, 'pruned': pl, 'round': r})
        # 3. greedy
        g = generate(model, tok, [t['prompt'] for t in transfer], greedy=True, batch=a.batch, max_new=a.max_new)
        stats['transfer_greedy'] = summarize(judge(transfer, [[p] for p in g], 'n_lines'), 'n_lines', f'[{a.name} r{r}] transfer greedy')
        g = generate(model, tok, [t['prompt'] for t in heldout], greedy=True, batch=a.batch, max_new=a.max_new)
        stats['heldout_greedy'] = summarize(judge(heldout, [[p] for p in g], 'n_lines'), 'n_lines', f'[{a.name} r{r}] heldout greedy')
        del model; torch.cuda.empty_cache()
        # cumulative bookkeeping (L_true bins, L*)
        for pool, fd, recs in (('targets', found, targets), ('transfer', found_t, transfer)):
            ls, ge = lstar(fd, recs)
            stats[f'{pool}_cum'] = {'solved': sum(1 for t in recs if fd.get(t['name'])), 'n': len(recs), 'lstar': ls, 'ge': ge, 'by_bin': by_bin(fd, recs),
                                    'distinct_proofs': sum(len(v) for v in fd.values()),
                                    'attempts_per_theorem': (sum(tried.values()) / len(recs)) if pool == 'targets' else r * a.k}
            print(f"[{a.name} r{r}] {pool} cumulative: solved {stats[f'{pool}_cum']['solved']}/{len(recs)} L*={ls} ge={ {L: ge[L] for L in (7, 8, 9, 10, 11, 12)} }", flush=True)
        with open(f'{out}/found_{r}.jsonl', 'w') as f:
            for t in targets:
                for x in found[t['name']]:
                    f.write(json.dumps({'name': t['name'], 'thm': t['thm'], 'prompt': t['prompt'], 'L_true': t['n_lines'], 'gen_lines': t.get('gen_lines'), 'source': t.get('source'), 'schema': t.get('schema'), **x}) + '\n')
        with open(f'{out}/found_transfer_{r}.jsonl', 'w') as f:
            for t in transfer:
                for x in found_t[t['name']]:
                    f.write(json.dumps({'name': t['name'], 'thm': t['thm'], 'prompt': t['prompt'], 'L_true': t['n_lines'], 'gen_lines': t.get('gen_lines'), 'source': t.get('source'), 'schema': t.get('schema'), **x}) + '\n')
        json.dump({'k': {t['name']: ki for t, ki in zip(targets, ks)}, 'tried': dict(tried), 'accepted': dict(okc)}, open(f'{out}/alloc_{r}.json', 'w'))
        # T6: exchange found proofs with siblings (union used for training and reported as found_union_r)
        train_found = found
        if a.share_dir:
            os.makedirs(a.share_dir, exist_ok=True)
            tmp = f'{a.share_dir}/s{a.sibling}_found_{r}.jsonl.tmp'
            with open(tmp, 'w') as f:
                for name, xs in found.items():
                    for x in xs:
                        f.write(json.dumps({'name': name, **x}) + '\n')
            os.replace(tmp, f'{a.share_dir}/s{a.sibling}_found_{r}.jsonl')
            union = collections.defaultdict(dict)
            for s in range(a.siblings):
                fn = f'{a.share_dir}/s{s}_found_{r}.jsonl'
                while not os.path.exists(fn):
                    time.sleep(10)
                for l in open(fn):
                    x = json.loads(l); union[x['name']].setdefault(x['norm'], x)
            train_found = {name: list(d.values()) for name, d in union.items()}
            ls, ge = lstar(train_found, targets)
            stats['targets_union'] = {'solved': sum(1 for t in targets if train_found.get(t['name'])), 'lstar': ls, 'ge': ge, 'by_bin': by_bin(train_found, targets),
                                      'distinct_proofs': sum(len(v) for v in train_found.values())}
            with open(f'{out}/found_union_{r}.jsonl', 'w') as f:
                for t in targets:
                    for x in train_found.get(t['name'], []):
                        f.write(json.dumps({'name': t['name'], 'thm': t['thm'], 'prompt': t['prompt'], 'L_true': t['n_lines'], **{k: x[k] for k in ('proof', 'norm', 'written', 'pruned', 'round')}}) + '\n')
            print(f"[{a.name} r{r}] sibling union: solved {stats['targets_union']['solved']} L*={ls}", flush=True)
        # T5: build the injection set after round inject_round-1 from the unsolved targets' shapes
        if a.inject_pool and r == a.inject_round - 1 and not a.no_train:
            from ladder_inject import build_injection
            inject_recs, ilog = build_injection(targets, found, a.inject_pool, eval_keys, a.inject_n, random.Random(seed))
            json.dump(ilog, open(f'{out}/inject_log.json', 'w'), indent=1)
            with open(f'{out}/inject_records.jsonl', 'w') as f:
                for x in inject_recs:
                    f.write(json.dumps(x) + '\n')
            print(f"[{a.name} r{r}] injection set: {len(inject_recs)} records; {ilog.get('shapes')}", flush=True)
        # 4. train
        if not a.no_train:
            mix = f'{out}/mix_{r}.jsonl'
            n_rl = 0
            with open(mix, 'w') as f:
                for t in targets:
                    fs = list(train_found.get(t['name'], []))
                    rng.shuffle(fs)
                    for x in fs[:a.max_per_thm]:
                        for _ in range(a.rl_weight):
                            f.write(json.dumps({'prompt': t['prompt'], 'proof': x['proof'], 'n_lines': x['written']}) + '\n'); n_rl += 1
                for x in relabelled.values():
                    for _ in range(a.rl_weight):
                        f.write(json.dumps({'prompt': x['prompt'], 'proof': x['proof'], 'n_lines': x['n_lines']}) + '\n'); n_rl += 1
                for x in rng.sample(train_recs, min(a.retain, len(train_recs))):
                    f.write(json.dumps({'prompt': x['prompt'], 'proof': x['proof'], 'n_lines': x['n_lines']}) + '\n')
                n_inj = 0
                if inject_recs and r == a.inject_round:
                    for x in inject_recs:
                        assert x['n_lines'] <= 6
                        for _ in range(a.inject_weight):
                            f.write(json.dumps({'prompt': x['prompt'], 'proof': x['proof'], 'n_lines': x['n_lines']}) + '\n'); n_inj += 1
            stats['mix_rl_records'] = n_rl; stats['mix_inject_records'] = n_inj
            if n_rl == 0 and n_inj == 0:
                print('no accepted proofs: skipping training this round', flush=True)
            else:
                new_ckpt = f'{a.ckptdir}/{a.name}_r{r}.pt'
                cmd = ['python3', 'state_train.py', '--data', mix, '--init', ckpt, '--steps', str(a.ft_steps), '--lr', str(a.ft_lr),
                       '--min_lr', str(a.ft_lr / 10), '--warmup', '50', '--cap', '0', '--out', new_ckpt, '--seed', str(seed), '--log_every', '200',
                       '--recs', str(a.ft_recs)]
                print(' '.join(cmd), flush=True)
                subprocess.run(cmd, check=True)
                ckpt = new_ckpt
        stats['env'] = env_stats_json(ENV['stats'])
        stats['secs'] = time.time() - t0
        json.dump(stats, open(f'{out}/round_{r}.json', 'w'), indent=1)
        print(f'=== round {r} done in {stats["secs"]:.0f}s; new proofs {new_this}; cum targets solved {stats["targets_cum"]["solved"]} L*={stats["targets_cum"]["lstar"]}; transfer L*={stats["transfer_cum"]["lstar"]}', flush=True)


if __name__ == '__main__':
    main()
