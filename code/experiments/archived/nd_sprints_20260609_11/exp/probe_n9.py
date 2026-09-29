"""Direct probe of the token wall: P(next = N<k+1>) at the start of line k+1
after a valid k-line prefix, for k = 8 (and controls k = 5..7).

For each of n test theorems from the OOD band, take the generating proof's
first k lines as a forced prefix (re-indexed to start at 1 for the fixed-index
model), and measure the model's next-token distribution at the position where
line k+1's index token must be emitted. If the token wall is real, the L8
model assigns ~0 probability to N9 after 8-line prefixes while assigning high
probability to N6..N8 after 5-7-line prefixes; the L8off model assigns normal
probability to the successor index in all cases.

Usage: python -m exp.probe_n9 --ckpt runs/L8/ckpt.pt --data data --out results/probe_n9_L8.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from nd.tokenizer import STOI, PAD
from train.evalgen import load_model


def load_jsonl(path):
    out = []
    with open(path) as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def prefix_upto_line(text, k):
    """Tokens of prompt + first k proof lines (ending after the k-th ';')."""
    toks = text.split()
    i = toks.index('PRF') + 1
    cnt = 0
    for j in range(i, len(toks)):
        if toks[j] == ';':
            cnt += 1
            if cnt == k:
                return toks[:j + 1]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--data', default='/root/nd/data')
    ap.add_argument('--n', type=int, default=300)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    device = 'cuda'
    model, _ = load_model(args.ckpt, device)

    # use OOD-band theorems with >=10-line generating proofs so an 8-line
    # prefix is never the final line
    recs = [r for r in load_jsonl(os.path.join(args.data, 'rl_targets.jsonl'))
            if r['n_lines'] >= 10][:args.n]
    results = {}
    for k in [5, 6, 7, 8]:
        probs = []
        batch = []
        for r in recs:
            pref = prefix_upto_line(r['text'], k)
            if pref is None:
                continue
            batch.append([STOI[t] for t in pref])
        # eval in chunks
        for s in range(0, len(batch), 256):
            chunk = batch[s:s + 256]
            L = max(len(p) for p in chunk)
            x = torch.full((len(chunk), L), PAD, dtype=torch.long, device=device)
            pos = torch.zeros((len(chunk), L), dtype=torch.long, device=device)
            for bi, p in enumerate(chunk):
                x[bi, L - len(p):] = torch.tensor(p, device=device)
                pos[bi, L - len(p):] = torch.arange(len(p), device=device)
            with torch.no_grad():
                logits, _ = model(x, pos_ids=pos)
            pr = F.softmax(logits[:, -1, :], dim=-1)
            tgt = STOI[f'N{k + 1}']
            probs += pr[:, tgt].tolist()
        results[f'P(N{k+1} | {k}-line prefix)'] = {
            'mean': sum(probs) / len(probs), 'n': len(probs),
            'frac_over_0.5': sum(1 for p in probs if p > 0.5) / len(probs)}
        print(json.dumps({f'k={k}': results[f'P(N{k+1} | {k}-line prefix)']}))
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=1)


if __name__ == '__main__':
    main()
