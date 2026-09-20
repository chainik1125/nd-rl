"""Expert-iteration bootstrap loop.

Round k:
  1. Sample n_rollouts proofs per RL-target theorem from the current model
     (temperature sampling), verify each with the independent verifier.
  2. Add newly verified proofs to the discovery buffer (per theorem keep the
     shortest verified proof).
  3. Retrain FROM SCRATCH on primitives + discovery buffer (discoveries
     upweighted), producing the next checkpoint.

Controls (run via --mode):
  rl       : buffer = RL-discovered proofs (the bootstrap condition)
  prim     : per round add a token-matched batch of EXTRA primitive proofs
             (same distribution as round-0 training data; "no new info" control)
  oracle   : per round add a token-matched batch of sampler-generated LONG
             proofs from the target band (disjoint theorems; skyline control)

After each round, evaluates pass@1 (greedy) and pass@8 (temp) on the held-out
transfer set and on the textbook targets, plus solve rate on rl_targets.

Usage (on pod):
  python -m exp.expert_iter --mode rl --rounds 5 --data data --out runs/ei_rl
"""
import argparse
import json
import os
import shlex
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from train.model import GPT
from train.evalgen import load_model, run_eval, summarize
from train import train as trainmod
from nd.tokenizer import STOI


def load_jsonl(path):
    out = []
    with open(path) as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def dump_jsonl(path, recs):
    with open(path, 'w') as f:
        for r in recs:
            f.write(json.dumps(r) + '\n')


def n_tokens(recs):
    return sum(len(r['text'].split()) for r in recs)


def train_round(train_specs, out_dir, args):
    cmd = [sys.executable, '-m', 'train.train', '--train', *train_specs,
           '--out', out_dir, '--iters', str(args.iters), '--bs', str(args.bs),
           '--layers', str(args.layers), '--dim', str(args.dim),
           '--heads', str(args.heads), '--seed', str(args.seed)]
    print('TRAIN:', ' '.join(shlex.quote(c) for c in cmd), flush=True)
    subprocess.run(cmd, check=True)


def evaluate(ckpt, data_dir, out_prefix, device, textbook=True, k_big=8,
             bs=1536, n_transfer=500):
    model, _ = load_model(ckpt, device)
    res = {}
    iid = load_jsonl(os.path.join(data_dir, 'eval_iid.jsonl'))
    ri = run_eval(model, iid, 1, 0.0, 420, bs, device)
    res['iid_pass1'] = sum(r['solved'] for r in ri) / len(ri)
    dump_jsonl(out_prefix + '_iid_pass1.jsonl', ri)
    transfer = load_jsonl(os.path.join(data_dir, 'eval_transfer.jsonl'))[:n_transfer]
    r1 = run_eval(model, transfer, 1, 0.0, 420, bs, device)
    rk = run_eval(model, transfer, k_big, 0.8, 420, bs, device)
    res['transfer_pass1'] = sum(r['solved'] for r in r1) / len(r1)
    res['transfer_passk'] = sum(r['solved'] for r in rk) / len(rk)
    dump_jsonl(out_prefix + '_transfer_pass1.jsonl', r1)
    dump_jsonl(out_prefix + f'_transfer_pass{k_big}.jsonl', rk)
    if textbook:
        tb_path = os.path.join(data_dir, 'targets_textbook.jsonl')
        if os.path.exists(tb_path):
            tb = load_jsonl(tb_path)
            rtb = run_eval(model, tb, 16, 0.8, 420, bs, device)
            res['textbook_pass16'] = sum(r['solved'] for r in rtb) / len(rtb)
            res['textbook_solved'] = [r['name'] for r in rtb if r['solved']]
            dump_jsonl(out_prefix + '_textbook_pass16.jsonl', rtb)
    del model
    torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['rl', 'her', 'prim', 'oracle'],
                    default='rl')
    ap.add_argument('--her-cap', type=int, default=4000,
                    help='max NEW harvested prefix-proofs added per round')
    ap.add_argument('--her-rep', type=int, default=4,
                    help='dup factor for harvested records in retraining')
    ap.add_argument('--her-min-lines', type=int, default=5)
    ap.add_argument('--rounds', type=int, default=5)
    ap.add_argument('--data', default='data')
    ap.add_argument('--out', required=True)
    ap.add_argument('--primitives', default='train_L8.jsonl')
    ap.add_argument('--round0-ckpt', default=None,
                    help='skip round-0 training, start from this ckpt')
    ap.add_argument('--n-rollouts', type=int, default=32)
    ap.add_argument('--n-targets', type=int, default=1200)
    ap.add_argument('--eval-bs', type=int, default=1536)
    ap.add_argument('--control-tokens', type=int, default=None,
                    help='per-round token quota for control modes (default: '
                         'match ei_rl history if available, else 60k*round)')
    ap.add_argument('--temp', type=float, default=1.0)
    ap.add_argument('--rep', type=int, default=8,
                    help='dup factor for discovered proofs in retraining')
    ap.add_argument('--iters', type=int, default=4000)
    ap.add_argument('--bs', type=int, default=128)
    ap.add_argument('--layers', type=int, default=4)
    ap.add_argument('--dim', type=int, default=256)
    ap.add_argument('--heads', type=int, default=8)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    device = 'cuda'
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    data_dir = args.data
    prim_path = os.path.join(data_dir, args.primitives)
    targets = load_jsonl(os.path.join(data_dir, 'rl_targets.jsonl'))[:args.n_targets]

    # control-data pools
    if args.mode == 'prim':
        # extra primitives never seen in train_L8: take from train_L8_2x tail
        big = load_jsonl(os.path.join(data_dir, 'train_L8_2x.jsonl'))
        base_ids = {r['thm'] for r in load_jsonl(prim_path)}
        extra_pool = [r for r in big if r['thm'] not in base_ids]
    elif args.mode == 'oracle':
        extra_pool = load_jsonl(os.path.join(data_dir, 'oracle_long.jsonl'))
    else:
        extra_pool = None

    buffer = {}          # thm -> record (discovered proofs, modes rl/her)
    harvested = {}       # thm -> record (hindsight prefix proofs, mode her)
    control_added = []   # records (modes prim/oracle)
    history = []

    reserved_ids = set()
    target_ids = {t['thm'] for t in targets}
    if args.mode == 'her':
        from nd.harvest import harvest_prefix  # noqa: F401 (used below)
        for name in ['eval_iid', 'eval_ood', 'eval_transfer', 'oracle_long',
                     'targets_textbook']:
            p = os.path.join(data_dir, f'{name}.jsonl')
            if os.path.exists(p):
                for r in load_jsonl(p):
                    reserved_ids.add(r['thm'])

    ckpt = args.round0_ckpt
    for rnd in range(args.rounds + 1):
        rdir = os.path.join(args.out, f'round{rnd}')
        os.makedirs(rdir, exist_ok=True)
        if rnd == 0:
            if ckpt is None:
                train_round([prim_path], rdir, args)
                ckpt = os.path.join(rdir, 'ckpt.pt')
        else:
            # 1) rollouts from current ckpt against targets (rl mode measures
            #    discovery even in control modes? No: controls skip rollouts)
            added_tokens = 0
            if args.mode in ('rl', 'her'):
                model, _ = load_model(ckpt, device)
                unsolved = [t for t in targets if t['thm'] not in buffer]
                print(f'round {rnd}: rollouts on {len(unsolved)} unsolved targets',
                      flush=True)
                rr = run_eval(model, unsolved, args.n_rollouts, args.temp,
                              420, args.eval_bs, device,
                              keep_fails=(args.mode == 'her'))
                del model
                torch.cuda.empty_cache()
                new = 0
                for r in rr:
                    if r['solved']:
                        rec = {'thm': r['thm'], 'text': r['best_text'],
                               'n_lines': r['best_len'],
                               'discharge': r.get('discharge'),
                               'rules': r.get('rules')}
                        buffer[r['thm']] = rec
                        new += 1
                print(f'round {rnd}: {new} new discoveries, buffer={len(buffer)}',
                      flush=True)
                if args.mode == 'her':
                    cands = {}
                    n_attempts = 0
                    for r in rr:
                        texts = list(r.get('fail_texts') or [])
                        if r['solved']:
                            texts.append(r['best_text'])
                        for tx in texts:
                            n_attempts += 1
                            h = harvest_prefix(tx, reserved_ids,
                                               min_lines=args.her_min_lines)
                            if h is None:
                                continue
                            if h['thm'] in target_ids:
                                # hindsight accidentally proves a target
                                if h['thm'] not in buffer or \
                                        h['n_lines'] < buffer[h['thm']]['n_lines']:
                                    buffer[h['thm']] = h
                                continue
                            old = harvested.get(h['thm']) or cands.get(h['thm'])
                            if old is None or h['n_lines'] < old['n_lines']:
                                cands[h['thm']] = h
                    fresh = [c for t, c in cands.items() if t not in harvested]
                    fresh.sort(key=lambda x: -x['n_lines'])
                    for c in fresh[:args.her_cap]:
                        harvested[c['thm']] = c
                    print(f'round {rnd}: harvested {len(fresh)} prefix-proofs '
                          f'(kept {min(len(fresh), args.her_cap)}, store='
                          f'{len(harvested)}) from {n_attempts} attempts',
                          flush=True)
                    dump_jsonl(os.path.join(rdir, 'harvested.jsonl'),
                               list(harvested.values()))
                dump_jsonl(os.path.join(rdir, 'discoveries.jsonl'),
                           list(buffer.values()))
                added_tokens = n_tokens(list(buffer.values())) + \
                    n_tokens(list(harvested.values()))
            else:
                # token-matched control: add roughly the tokens RL would add.
                # We match the RL run's per-round buffer sizes if available,
                # else a fixed quota per round.
                quota = (args.control_tokens * rnd if args.control_tokens
                         else 60000 * rnd)
                rl_hist = os.path.join(os.path.dirname(args.out), 'ei_rl',
                                       'history.json')
                if os.path.exists(rl_hist):
                    with open(rl_hist) as f:
                        h = json.load(f)
                    match = [x for x in h if x['round'] == rnd]
                    if match and match[0].get('buffer_tokens'):
                        quota = match[0]['buffer_tokens']
                while extra_pool and n_tokens(control_added) < quota:
                    control_added.append(extra_pool.pop())
                dump_jsonl(os.path.join(rdir, 'control_added.jsonl'),
                           control_added)
                added_tokens = n_tokens(control_added)

            # 2) retrain from scratch on primitives + buffer
            extra_path = os.path.join(rdir, 'extra.jsonl')
            extra = list(buffer.values()) if args.mode in ('rl', 'her') \
                else control_added
            dump_jsonl(extra_path, extra)
            specs = [prim_path]
            if extra:
                specs.append(f'{extra_path}:{args.rep}')
            if args.mode == 'her' and harvested:
                hpath = os.path.join(rdir, 'harvest_train.jsonl')
                dump_jsonl(hpath, list(harvested.values()))
                specs.append(f'{hpath}:{args.her_rep}')
            train_round(specs, rdir, args)
            ckpt = os.path.join(rdir, 'ckpt.pt')

        # 3) evaluate
        ev = evaluate(ckpt, data_dir, os.path.join(rdir, 'eval'), device,
                      bs=args.eval_bs)
        ev['round'] = rnd
        ev['mode'] = args.mode
        ev['buffer_size'] = (len(buffer) if args.mode in ('rl', 'her')
                             else len(control_added))
        ev['buffer_tokens'] = (n_tokens(list(buffer.values()))
                               if args.mode in ('rl', 'her')
                               else n_tokens(control_added))
        if args.mode == 'her':
            ev['harvested_size'] = len(harvested)
            ev['harvested_tokens'] = n_tokens(list(harvested.values()))
        history.append(ev)
        with open(os.path.join(args.out, 'history.json'), 'w') as f:
            json.dump(history, f, indent=1)
        print('ROUND SUMMARY:', json.dumps(ev, default=str), flush=True)


if __name__ == '__main__':
    main()
