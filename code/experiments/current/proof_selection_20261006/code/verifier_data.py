#!/usr/bin/env python3
"""Verifier inputs from pool rows only (allowlist: prompt, base, actions, has_eos).  Labels are attached separately
by the training script through pool_lib.load_labels.

One attempt -> [state_1 action_1 <eos> state_2 action_2 <eos> ...] in the SN model's native format: each state is the
prompt ids the policy saw (verified against the stored state_sha1), each action the environment-effective action
(names as assigned by the env, which is what the SN checkpoint was trained on), followed by <eos> when one was
sampled.  A failed final action is its decoded tokens; a truncated one has no <eos>.  Clipped to max_len by dropping
the earliest whole pairs; if the last pair alone exceeds max_len, its last max_len tokens are kept.

  python verifier_data.py --sets dev1108 ...    -> ~/runs/proof-selection/verifier/seqs/<stem>.pkl

--fmt hist (secondary variant, added 02:09 after measuring 57% clipping of the pairs format on textbook72): Dan's arm-SH
input for the attempt's LAST step, i.e. [effective actions 1..T-1 concatenated (env.hist, no eos)] + state_T + action_T
(+ eos), clipped by dropping the earliest tokens (0.8% of tb72 attempts exceed 1024).  Every action's term is visible.
  python verifier_data.py --fmt hist --sets ...  -> ~/runs/proof-selection/verifier/seqs_hist/<stem>.pkl
"""
import argparse, multiprocessing as mp, os, pickle, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pool_lib as PL

MAX_LEN = 1024
OUT = os.path.join(PL.RUNS, 'verifier', 'seqs')


FMT = ['pairs']


def build(row):
    tok = PL.get_tok()
    if FMT[0] == 'hist':
        return build_hist(row, tok)
    seqs, clipped, sha_bad = [], [], 0
    for att in row['attempts']:
        _, pairs = PL.replay(row['prompt'], att['base'], att['actions'], tok, att['has_eos'])
        sha_bad += any(PL.state_sha1(st) != h for (st, _), h in zip(pairs, att['state_sha1']))
        parts = [st + act + ([tok.eos] if att['has_eos'][t] else []) for t, (st, act) in enumerate(pairs)]
        tot, keep = 0, []
        for p in reversed(parts):
            if tot + len(p) > MAX_LEN:
                break
            keep.append(p); tot += len(p)
        cl = len(keep) < len(parts)
        seq = [x for p in reversed(keep) for x in p] if keep else parts[-1][-MAX_LEN:]
        seqs.append(np.asarray(seq, np.int16)); clipped.append(cl)
    return row['name'], seqs, clipped, sha_bad


def build_hist(row, tok):
    seqs, clipped, sha_bad = [], [], 0
    for att in row['attempts']:
        _, pairs = PL.replay(row['prompt'], att['base'], att['actions'], tok, att['has_eos'])
        sha_bad += any(PL.state_sha1(st) != h for (st, _), h in zip(pairs, att['state_sha1']))
        hist = [x for _, act in pairs[:-1] for x in act]
        st, act = pairs[-1]
        seq = hist + st + act + ([tok.eos] if att['has_eos'][len(pairs) - 1] else [])
        clipped.append(len(seq) > MAX_LEN)
        seqs.append(np.asarray(seq[-MAX_LEN:], np.int16))
    return row['name'], seqs, clipped, sha_bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sets', nargs='+', required=True)
    ap.add_argument('--workers', type=int, default=24)
    ap.add_argument('--fmt', default='pairs', choices=['pairs', 'hist'])
    ap.add_argument('--k', default=None, help="pool k tag, e.g. 96b (phase-2 extra pools)")
    a = ap.parse_args()
    FMT[0] = a.fmt
    out_dir = OUT + ('_hist' if a.fmt == 'hist' else '')
    os.makedirs(out_dir, exist_ok=True)
    for s in a.sets:
        rows = PL.load_pool(s, a.k)
        with mp.Pool(a.workers) as pool:
            res = pool.map(build, rows, chunksize=8)
        d = {n: (sq, cl) for n, sq, cl, _ in res}
        bad = sum(r[3] for r in res)
        n = sum(len(sq) for sq, _ in d.values()); ncl = sum(sum(cl) for _, cl in d.values())
        lens = [len(x) for sq, _ in d.values() for x in sq]
        pickle.dump(d, open(os.path.join(out_dir, PL.pool_stem(s, a.k) + '.pkl'), 'wb'))
        print(f'{s} [{a.fmt}]: {n} attempts, clipped {ncl} ({ncl / n:.4f}), state sha mismatches {bad}, '
              f'len mean {np.mean(lens):.0f} max {max(lens)}', flush=True)
        assert bad == 0


if __name__ == '__main__':
    main()
