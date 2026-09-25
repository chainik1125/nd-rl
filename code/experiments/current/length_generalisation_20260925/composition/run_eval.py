#!/usr/bin/env python3
"""Evaluate the composition prompts (composition/prompts.jsonl) for one checkpoint.

    python composition/run_eval.py --ckpt PATH --label NAME [--gpu auto] [--k 32] [--limit N] [--device cuda]

A wrapper around evaluate.py's pieces (lock, checkpoint loading), sampler.py and ndcheck.py, with the same settings
as the main evaluation (k 32, T 0.8, max_new 1024, batch 512). evaluate.py itself is not modified. Per prompt it
stores n_ok and, for every sample, the written-line count (';' lines) and the ndcheck outcome, so attempt lengths
can be compared with L_true. Output: composition/eval/<label>/chunk_<i>.jsonl (+ .summary.json, samples/*.jsonl.gz).
Chunks already done with the same prompts, checkpoint and config are skipped.
"""
import argparse, collections, datetime, json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
LE = os.path.dirname(HERE)
sys.path.insert(0, LE)
import evaluate as E  # noqa: E402  (import only: Lease, load_model, log)
from common import FROZEN, MAX_NEW, SAMPLE_SEED, TEMPERATURE, K, read_jsonl, sha256_file, sha256_str, write_json, write_jsonl  # noqa: E402

CHUNK = 800


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--label', required=True)
    ap.add_argument('--k', type=int, default=K)
    ap.add_argument('--temperature', type=float, default=TEMPERATURE)
    ap.add_argument('--max-new', type=int, default=MAX_NEW)
    ap.add_argument('--batch', type=int, default=512)
    ap.add_argument('--limit', type=int, default=0, help='first N prompts only (test; writes composition/eval_test/)')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--gpu', default='auto')
    ap.add_argument('--gpus', default='0,1,2,3,4')
    ap.add_argument('--gpu-wait', type=float, default=4 * 3600)
    ap.add_argument('--dir', default=HERE, help='composition directory holding prompts.jsonl (default composition/)')
    a = ap.parse_args()
    os.environ.setdefault('AB_WORKER', 'lengthexp')
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    prompts = read_jsonl(os.path.join(a.dir, 'prompts.jsonl'))
    if a.limit:
        prompts = prompts[:a.limit]
    out = os.path.join(a.dir, 'eval_test' if a.limit else 'eval', a.label)
    os.makedirs(os.path.join(out, 'samples'), exist_ok=True)
    ckpt = os.path.abspath(a.ckpt)
    ck_sha = sha256_file(ckpt)
    cfg = {'k': a.k, 'temperature': a.temperature, 'max_new': a.max_new, 'sample_seed': SAMPLE_SEED}
    chunks = [prompts[i:i + CHUNK] for i in range(0, len(prompts), CHUNK)]

    def done(i, csha):
        fn = os.path.join(out, f'chunk_{i:02d}.summary.json')
        try:
            m = json.load(open(fn))
        except (OSError, ValueError):
            return False
        return m.get('chunk_sha') == csha and m.get('ckpt_sha256') == ck_sha and m.get('config') == cfg

    todo = [(i, c, sha256_str('\n'.join(p['prompt'] for p in c))) for i, c in enumerate(chunks)]
    todo = [t for t in todo if not done(t[0], t[2])]
    E.log(f'{a.label}: {len(prompts)} prompts in {len(chunks)} chunks, {len(todo)} to do -> {out}')
    if not todo:
        return 0
    lease = E.Lease()
    try:
        if a.device == 'cuda':
            lease.take(a)
        import torch
        sys.path.insert(0, os.path.join(FROZEN, 'lib'))
        from ndcheck import classify, sequent
        from sampler import sample_ids
        model, tok, _ = E.load_model(ckpt, a.device)
        for i, c, csha in todo:
            t0 = time.time()
            ps = [p['prompt'] for p in c for _ in range(a.k)]
            ids = sample_ids(model, tok, ps, temperature=a.temperature, max_new=a.max_new, batch=a.batch,
                             seed=SAMPLE_SEED + 7000 + i)
            ts = time.time() - t0
            rows, samples, kinds = [], [], collections.Counter()
            for j, p in enumerate(c):
                seq = sequent(p['prompt'])
                lines, outc, n_ok = [], [], 0
                for s in range(a.k):
                    body = tok.decode(ids[j * a.k + s])
                    r = classify(p['prompt'], body, seq)
                    n_ok += r['ok']
                    lines.append(r['n_lines'])
                    kind = 'ok' if r['ok'] else r['kind']
                    outc.append([kind, r['line'], r['reason']])
                    kinds[kind] += 1
                    samples.append({'prompt': p['prompt'], 'i': s, 'body': body})
                rows.append({'prompt': p['prompt'], 'L_true': p['L_true'], 'kinds': p['kinds'], 'k': a.k, 'n_ok': n_ok,
                             'lines': lines, 'outcome': outc})
            write_jsonl(os.path.join(out, f'chunk_{i:02d}.jsonl'), rows)
            write_jsonl(os.path.join(out, 'samples', f'chunk_{i:02d}.jsonl.gz'), samples)
            write_json(os.path.join(out, f'chunk_{i:02d}.summary.json'),
                       {'chunk_sha': csha, 'ckpt': ckpt, 'ckpt_sha256': ck_sha, 'config': cfg, 'n': len(c), 'kinds': dict(kinds),
                        'sample_secs': ts, 'batch': a.batch, 'label': a.label,
                        'finished_utc': datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')})
            E.log(f'{a.label} chunk {i}: {len(c)} prompts, pass@1 {sum(r["n_ok"] for r in rows) / (len(c) * a.k):.4f}, '
                  f'{ts:.0f}s ({len(ps) / max(ts, 1e-9):.0f} samples/s)')
    finally:
        lease.drop()
    return 0


if __name__ == '__main__':
    sys.exit(main())
