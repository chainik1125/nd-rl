#!/usr/bin/env python3
"""guided_eval.py -- one (checkpoint, arm) read for run `guided-tts`: k attempts per theorem on every problem file,
plain or guided (`guided_sample.py`), then Lean on every finished proof (`lean_gate.gate`, the literal text).

  python3 guided_eval.py --ckpt ckpts/gt/la_T1_best12_s0_r8.pt --arm logical --k 256 --seed 1 \
      --sets data/gt/tb72_textbook_dev.jsonl,... --out artifacts/gt/eval/c12s0_logical

Writes  <out>.json         summary: per-file solved, timings (GPU / check / env / final Lean, wall), tokens, peak
                           memory, rejection causes, with-replacement repeat probabilities (summary stats)
        <out>.rows.jsonl.gz  one row per theorem: file, name, min_lines (or reference_lines), and per-attempt arrays
                           ok / tokens / prefill / draws / rej / status / flag, plus the distinct accepted texts
        <out>.repeat.json.gz the raw repeat-probability lists
        <out>.steps.jsonl.gz (with --dump) distinct (state, step) pairs as standalone Lean theorems + checker verdict
Lean alone decides; nd_verify is not called.
"""
import argparse, collections, gzip, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import record
from model import load_ckpt
from guided_sample import guided_generate


def load_sets(paths):
    recs = []
    for p in paths:
        for l in open(p):
            if l.strip():
                r = json.loads(l)
                recs.append(dict(file=os.path.basename(p).replace('.jsonl', ''), name=r.get('name'),
                                 prompt=r['prompt'], min_lines=r.get('min_lines', r.get('reference_lines')),
                                 book=(r.get('source') or {}).get('book') if isinstance(r.get('source'), dict) else None))
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--arm', required=True, choices=['plain', 'structural', 'logical'])
    ap.add_argument('--sets', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--k', type=int, default=256)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--batch', type=int, default=2048)
    ap.add_argument('--temperature', type=float, default=0.8)
    ap.add_argument('--max_action', type=int, default=512)
    ap.add_argument('--max_steps', type=int, default=96)
    ap.add_argument('--max_rej', type=int, default=10)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--dump', action='store_true')
    a = ap.parse_args()
    record.save_config(vars(a), a.out + '.json', arm=a.arm)
    recs = load_sets(a.sets.split(','))
    if a.limit:
        recs = recs[:a.limit]
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok, extra = load_ckpt(a.ckpt, dev)
    model.eval()
    assert getattr(tok, 'canon', False), 'lean_staten checkpoints only'
    prompts = [r['prompt'] for r in recs for _ in range(a.k)]
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    st = {}
    dumpf = gzip.open(a.out + '.steps.jsonl.gz', 'wt') if a.dump else None
    t0 = time.time()
    with record.compute(phase='read', arm=a.arm, ckpt=a.ckpt) as c:
        res = guided_generate(model, tok, prompts, a.arm, temperature=a.temperature, max_action=a.max_action,
                              max_steps=a.max_steps, max_rej=a.max_rej, batch=a.batch, seed=a.seed, stats=st,
                              dump=dumpf)
        t_loop = time.time() - t0
        if dumpf:
            dumpf.close()
        fin = [i for i, r in enumerate(res) if r['text'] is not None]
        t1 = time.time()
        from lean_gate import gate
        g = gate(tok, [prompts[i] for i in fin], [res[i]['nd'] for i in fin], [res[i]['text'] for i in fin])
        st['lean_s'] = time.time() - t1
    for i, nd in zip(fin, g):
        res[i]['ok'] = not nd.startswith('LEAN')
    wall = time.time() - t0
    by_file = collections.defaultdict(lambda: [0, 0])
    flag_ok = collections.Counter()
    with gzip.open(a.out + '.rows.jsonl.gz', 'wt') as f:
        for j, r in enumerate(recs):
            att = res[j * a.k:(j + 1) * a.k]
            ok = [bool(x.get('ok')) for x in att]
            for x in att:
                if x['flag'] is not None and x['text'] is not None:
                    flag_ok['flag_lean_accepted' if x.get('ok') else 'flag_lean_rejected'] += 1
            acc = collections.Counter(x['text'] for x in att if x.get('ok'))
            row = dict(r, k=a.k, n_ok=sum(ok), ok=[int(v) for v in ok], tokens=[x['tokens'] for x in att],
                       prefill=[x['prefill'] for x in att], draws=[x['draws'] for x in att],
                       rej=[x['rej'] for x in att], steps=[x['steps'] for x in att],
                       status=[x['status'] for x in att], flag=[x['flag'] for x in att],
                       causes=[x['causes'] for x in att if x['causes']], accepted=dict(acc))
            f.write(json.dumps(row) + '\n')
            by_file[r['file']][0] += sum(ok) > 0
            by_file[r['file']][1] += 1
    rep, rep1 = st.pop('repeat_p'), st.pop('reject_p')
    with gzip.open(a.out + '.repeat.json.gz', 'wt') as f:
        json.dump(dict(repeat_p=rep, reject_p=rep1), f)
    summ = dict(ckpt=a.ckpt, arm=a.arm, k=a.k, seed=a.seed, batch=a.batch, temperature=a.temperature,
                max_action=a.max_action, max_steps=a.max_steps, max_rej=a.max_rej, n_theorems=len(recs),
                solved={k: v[0] for k, v in by_file.items()}, n={k: v[1] for k, v in by_file.items()},
                solved_total=sum(v[0] for v in by_file.values()), wall_s=wall, loop_s=t_loop,
                gpu=torch.cuda.get_device_name(0) if dev == 'cuda' else 'cpu', n_params=extra.get('n_params'),
                proof_level_check=dict(flag_ok),
                repeat_mean=sum(rep) / len(rep) if rep else None, n_redraws=len(rep),
                reject_p_mean=sum(rep1) / len(rep1) if rep1 else None, n_rejected_ended=len(rep1),
                attempts=len(res), finished=len(fin), accepted=sum(1 for r in res if r.get('ok')),
                stats={k: (dict(v) if isinstance(v, collections.Counter) else v) for k, v in st.items()})
    json.dump(summ, open(a.out + '.json', 'w'), indent=1)
    print(json.dumps({k: v for k, v in summ.items() if k != 'stats'}))
    print(json.dumps({k: v for k, v in summ['stats'].items() if k not in ('rej_cause',)}))
    print('rej_cause', json.dumps(dict(collections.Counter(st['rej_cause']).most_common(20))))


if __name__ == '__main__':
    main()
