#!/usr/bin/env python3
"""Evaluate one checkpoint at every test length: k samples per theorem, verified, with first-error lines.

ab3/repr copy of lengthexp/evaluate.py. Changes, all representation plumbing: the model is loaded with the variant
library (repr/lib: tokenizer mode and positional variant from the checkpoint), sampled with repr/sampler.py
(positions from the model; identical samples to lengthexp's sampler for pe='rope', tests/test_equivalence.py),
output goes to repr/eval/<label>/, the test sets are repr/testsets (a frozen copy of lengthexp's final draw,
2026-09-25 02:27 UTC), and --pos-eval rand (pos-rand check) samples every row on random positions
(rand_tables). Test sets, k, T, max_new, seeds, the checker (lengthexp/ndcheck.py) and the files written are
lengthexp's.

    python evaluate.py --ckpt PATH --label NAME [--sets std,matched] [--lengths 6-18] [--k 32] [--pos-eval contig|rand]
                       [--temperature 0.8] [--max-new 1024] [--batch 512] [--limit N]
                       [--gpu auto|env|N] [--gpus 0,1,2,3,4] [--gpu-wait 14400] [--force] [--no-samples]

Labels: 'pretrain' for frozen/pretrain.pt, '<arm>_s<seed>' (A_s101, ...) for the training runs; analyse.py
reads the arm and seed from the label. Output: eval/<label>/ (eval_smoke/<label>/ with --limit):
  meta.json                  checkpoint path + sha256, config, GPU
  <set>_L<L>.jsonl           one row per theorem: name, L_true, source, schema, n_prem, k, n_ok,
                             ok_lines / ok_starts (line count and first line index of every correct sample),
                             fails ([kind, line, reason, start] per failed sample; see ndcheck.py), n_tok_mean
  <set>_L<L>.summary.json    pass@1/4/32, failure-kind counts, timing, checker/nd_verify agreement, hashes
  samples/<set>_L<L>.jsonl.gz  every sampled proof body (for re-analysis without a GPU)
A unit (set, length) is skipped when its summary exists with the same theorems, checkpoint and config.

GPU: --gpu auto (default) takes a lock through frozen/gpulock.py (FIFO queue, released at exit, and by the
lock holder if this process dies), exactly like harness.py --gpu auto. --gpu env uses CUDA_VISIBLE_DEVICES
as it is (no lock).
"""
import argparse, atexit, collections, datetime, json, os, signal, socket, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
LE_DIR = '/scratch/cowifornia/nd-rl/ab3/lengthexp'
sys.path.insert(0, LE_DIR)
from common import (FROZEN, K, LENGTHS, MAX_NEW, SAMPLE_SEED, TEMPERATURE, TESTSETS, pass_at_k,  # lengthexp/common.py
                    read_jsonl, sha256_file, sha256_str, unit_name, write_json, write_jsonl)
sys.path.insert(0, HERE)            # repr first: sampler.py, common_repr.py
LE = HERE
EVAL = os.path.join(HERE, 'eval')
TESTSETS = os.path.join(HERE, 'testsets')   # frozen copy of lengthexp's final draw (built 2026-09-25 02:27 UTC)
sys.path.insert(1, FROZEN)
import gpulock  # noqa: E402  (stdlib only; must be imported before CUDA is initialised)
TOK_PER_LINE = 19                   # mean tokens per written line of arm C's found proofs (18.3); rand_tables


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def parse_lengths(s):
    out = []
    for part in s.split(','):
        if '-' in part:
            a, b = part.split('-')
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


class Lease:
    def __init__(self):
        self.gpu = self.holder = None

    def take(self, a):
        if a.gpu == 'env':
            return
        gpus = gpulock.parse_gpus(a.gpus) if a.gpu == 'auto' else [int(a.gpu)]
        t0 = time.time()
        gpu, holder = gpulock.acquire(gpus, wait=a.gpu_wait, run=f'repr_eval_{a.label}', owner=os.getpid(), log=log)
        if gpu is None:
            raise SystemExit(f'evaluate: no free GPU among {gpus} after {a.gpu_wait:.0f}s')
        self.gpu, self.holder = gpu, holder
        atexit.register(self.drop)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
        log(f'gpulock: holding GPU {gpu} (holder {holder}) after {time.time() - t0:.0f}s')

    def drop(self):
        if self.gpu is None:
            return
        g, self.gpu = self.gpu, None
        ok = gpulock.release(g, self.holder)
        log(f'gpulock: released GPU {g}' + ('' if ok else ' (lock still busy?)'))


def _term(signum, frame):
    raise SystemExit(128 + signum)


def load_units(sets, lengths, limit):
    units = []
    for s in sets:
        fn = os.path.join(TESTSETS, f'{s}.jsonl')
        if not os.path.isfile(fn):
            log(f'test set {s}: {fn} missing (skipped)')
            continue
        by = collections.defaultdict(list)
        for r in read_jsonl(fn):
            by[r['L_true']].append(r)
        for L in lengths:
            rs = by.get(L)
            if not rs:
                continue
            rs = sorted(rs, key=lambda r: r['name'])
            if limit:
                rs = rs[:limit]
            usha = sha256_str('\n'.join(r['name'] + '\t' + r['prompt'] for r in rs))
            units.append((s, L, rs, usha))
    return units


def load_model(path, device):
    from common_repr import load_model as lm
    model, tok, ck = lm(path, device)
    return model, tok, dict(model.cfg)


def unit_seed(s, L):
    return SAMPLE_SEED + 1000 * L + (0 if s == 'std' else 500)


def rand_tables(model, tok, prompts, n_lines, max_new, seed, tok_per_line=TOK_PER_LINE):
    """pos-rand check (Ruoss et al.'s test-time procedure, adapted to generation of unknown length): for each prompt
    a sorted uniform random subset of range(model.pos_range) of size n = len(prompt) + tok_per_line * L (L = the
    theorem's length label, so n is the length of a typical correct proof; the training distribution of such a
    sequence); tokens past n continue at the subset's mean gap. One seeded RNG per call (random.Random(seed))."""
    import random as _r
    rng = _r.Random(seed)
    R = model.pos_range
    out = []
    for p, L in zip(prompts, n_lines):
        m = len(tok.encode_prompt(p))
        n = min(R, m + tok_per_line * int(L))
        sub = sorted(rng.sample(range(R), n))
        gap = R / n
        need = m + max_new
        sub += [sub[-1] + round((j + 1) * gap) for j in range(max(0, need - n))]
        out.append(sub[:max(need, n)])
    return out


def evaluate_unit(model, tok, rs, a, s, L):
    from ndcheck import classify, sequent
    from sampler import sample_ids
    prompts = [r['prompt'] for r in rs for _ in range(a.k)]
    t0 = time.time()
    tabs = None
    if a.pos_eval == 'rand':
        tabs = rand_tables(model, tok, prompts, [r['L_true'] for r in rs for _ in range(a.k)], a.max_new, unit_seed(s, L) + 7)
    ids = sample_ids(model, tok, prompts, temperature=a.temperature, max_new=a.max_new, batch=a.batch, seed=unit_seed(s, L),
                     pos_tables=tabs)
    t_sample = time.time() - t0
    t1 = time.time()
    rows, samples = [], []
    kinds, reasons = collections.Counter(), collections.Counter()
    disagree = collections.Counter()
    examples = []
    n_hit = 0
    for j, r in enumerate(rs):
        seq = sequent(r['prompt'])
        ok_lines, ok_starts, fails, ntok = [], [], [], []
        for i in range(a.k):
            o = ids[j * a.k + i]
            body = tok.decode(o)
            ntok.append(len(o))
            if not o or o[-1] != tok.eos:
                n_hit += 1
            c = classify(r['prompt'], body, seq)
            if c['ok'] != (c['kind'] == 'ok'):
                disagree[f"verify={'ok' if c['ok'] else 'fail'}/checker={c['kind']}:{c['reason']}"] += 1
                if len(examples) < 10:
                    examples.append({'name': r['name'], 'body': body, 'checker': c})
            if c['ok']:
                ok_lines.append(c['n_lines'])
                ok_starts.append(c['start'])
                kinds['ok'] += 1
            else:
                kind = c['kind'] if c['kind'] != 'ok' else 'verifier_only_reject'
                fails.append([kind, c['line'], c['reason'], c['start']])
                kinds[kind] += 1
                reasons[f"{kind}:{c['reason']}"] += 1
            if not a.no_samples:
                samples.append({'name': r['name'], 'i': i, 'body': body})
        rows.append({'name': r['name'], 'L_true': r['L_true'], 'source': r.get('source'), 'schema': r.get('schema'),
                     'n_prem': r.get('n_prem'), 'k': a.k, 'n_ok': len(ok_lines), 'ok_lines': ok_lines,
                     'ok_starts': ok_starts, 'fails': fails, 'n_tok_mean': sum(ntok) / len(ntok)})
    t_check = time.time() - t1
    n = len(rows)
    summ = {'set': s, 'L': L, 'n': n, 'k': a.k,
            'pass@1': sum(pass_at_k(a.k, x['n_ok'], 1) for x in rows) / n,
            'pass@4': sum(pass_at_k(a.k, x['n_ok'], 4) for x in rows) / n if a.k >= 4 else None,
            f'pass@{a.k}': sum(pass_at_k(a.k, x['n_ok'], a.k) for x in rows) / n,
            'solved': sum(1 for x in rows if x['n_ok'] > 0), 'n_ok_total': sum(x['n_ok'] for x in rows),
            'kinds': dict(kinds), 'reasons': dict(reasons.most_common()), 'n_hit_max_new': n_hit,
            'checker_disagreements': dict(disagree), 'disagreement_examples': examples,
            'sample_secs': t_sample, 'check_secs': t_check, 'samples_per_sec': len(prompts) / max(t_sample, 1e-9)}
    return rows, samples, summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--label', required=True)
    ap.add_argument('--sets', default='std,matched')
    ap.add_argument('--lengths', default=f'{LENGTHS[0]}-{LENGTHS[-1]}')
    ap.add_argument('--k', type=int, default=K)
    ap.add_argument('--temperature', type=float, default=TEMPERATURE)
    ap.add_argument('--max-new', type=int, default=MAX_NEW)
    ap.add_argument('--batch', type=int, default=512)
    ap.add_argument('--limit', type=int, default=0, help='first N theorems per unit (smoke test; writes eval_smoke/)')
    ap.add_argument('--out-root', default=None)
    ap.add_argument('--gpu', default='auto')
    ap.add_argument('--gpus', default='0,1,2,3,4')
    ap.add_argument('--gpu-wait', type=float, default=4 * 3600)
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--no-samples', action='store_true')
    ap.add_argument('--pos-eval', choices=('contig', 'rand'), default='contig',
                    help="positions at sampling time: contig (0, 1, 2, ...; the model's own) or rand (pos-rand check)")
    a = ap.parse_args()
    os.environ.setdefault('AB_WORKER', 'repr')
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')   # before CUDA initialises
    signal.signal(signal.SIGTERM, _term)
    signal.signal(signal.SIGHUP, _term)
    root = a.out_root or (os.path.join(LE, 'eval_smoke') if a.limit else EVAL)
    out = os.path.join(root, a.label)
    os.makedirs(os.path.join(out, 'samples'), exist_ok=True)
    ckpt = os.path.abspath(a.ckpt)
    ck_sha = sha256_file(ckpt)
    units = load_units(a.sets.split(','), parse_lengths(a.lengths), a.limit)
    cfg_key = {'k': a.k, 'temperature': a.temperature, 'max_new': a.max_new, 'sample_seed': SAMPLE_SEED}
    if a.pos_eval != 'contig':
        cfg_key['pos_eval'] = a.pos_eval

    def fresh(s, L, usha):
        fn = os.path.join(out, f'{unit_name(s, L)}.summary.json')
        if a.force or not os.path.isfile(fn):
            return False
        try:
            m = json.load(open(fn))
        except (OSError, ValueError):
            return False
        return m.get('unit_sha') == usha and m.get('ckpt_sha256') == ck_sha and m.get('config') == cfg_key

    todo = [u for u in units if not fresh(u[0], u[1], u[3])]
    log(f'{a.label}: {ckpt} ({ck_sha[:12]}), {len(units)} units, {len(todo)} to do -> {out}')
    if not todo:
        return 0
    lease = Lease()
    try:
        lease.take(a)
        import torch
        model, tok, mcfg = load_model(ckpt, 'cuda')
        meta = {'label': a.label, 'ckpt': ckpt, 'ckpt_sha256': ck_sha, 'model_cfg': mcfg, 'tok_mode': tok.mode, 'config': cfg_key,
                'batch': a.batch, 'limit': a.limit, 'host': socket.gethostname(),
                'gpu': f"{os.environ.get('CUDA_VISIBLE_DEVICES', '')}:{torch.cuda.get_device_name(0)}",
                'started_utc': datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}
        write_json(os.path.join(out, 'meta.json'), meta)
        for s, L, rs, usha in todo:
            rows, samples, summ = evaluate_unit(model, tok, rs, a, s, L)
            summ.update(unit_sha=usha, ckpt=ckpt, ckpt_sha256=ck_sha, config=cfg_key, label=a.label,
                        finished_utc=datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'))
            u = unit_name(s, L)
            write_jsonl(os.path.join(out, f'{u}.jsonl'), rows)
            if samples:
                write_jsonl(os.path.join(out, 'samples', f'{u}.jsonl.gz'), samples)
            write_json(os.path.join(out, f'{u}.summary.json'), summ)
            dis = sum(summ['checker_disagreements'].values())
            log(f"{a.label} {u}: pass@1 {summ['pass@1']:.4f} pass@4 {summ['pass@4']:.4f} pass@{a.k} {summ[f'pass@{a.k}']:.3f} "
                f"({summ['solved']}/{summ['n']}) kinds {summ['kinds']} hit_max_new {summ['n_hit_max_new']} "
                f"disagree {dis} | sample {summ['sample_secs']:.0f}s ({summ['samples_per_sec']:.0f}/s) check {summ['check_secs']:.0f}s")
    finally:
        lease.drop()
    return 0


if __name__ == '__main__':
    sys.exit(main())
