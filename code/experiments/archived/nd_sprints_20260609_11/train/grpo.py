"""GRPO (clipped PPO objective, group-relative advantages, KL-to-reference)
on verifier reward, for the ND prover.

Per iteration:
  1. Sample B target theorems; G completions each at T=1.0 (current policy).
  2. Verifier reward r in {0,1} per completion (accepts iff valid proof of the
     exact sequent).
  3. Group advantage A = r - mean_group(r) (zero for degenerate groups).
  4. K epochs of clipped-surrogate updates on completion tokens, with a k3
     KL penalty to the frozen SFT reference.

Logs per iter: mean reward, solved-set size, longest verified proof so far,
KL, clip fraction. Periodic full evals + checkpoints.

Usage (on pod):
  python -m train.grpo --ckpt runs/L8off_f2/ckpt.pt --data data \
      --iters 200 --out runs/grpo
"""
import argparse
import copy
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from nd.tokenizer import STOI, ITOS, PAD
from nd.verify import verify_text
from train.evalgen import load_model, run_eval

QED_ID = STOI['QED']


def load_jsonl(path):
    out = []
    with open(path) as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def seq_logprobs(model, ids, plens, device, bs=256, grad=False):
    """Per-token logprobs of each sequence's completion region.
    ids: list of full token-id lists (prompt+completion). plens: prompt lens.
    Returns (logp_flat, index map) as a list of 1-D tensors per sequence."""
    outs = []
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        for s in range(0, len(ids), bs):
            chunk = ids[s:s + bs]
            pl = plens[s:s + bs]
            L = max(len(x) for x in chunk)
            x = torch.full((len(chunk), L), PAD, dtype=torch.long, device=device)
            pos = torch.zeros((len(chunk), L), dtype=torch.long, device=device)
            for i, seq in enumerate(chunk):
                x[i, :len(seq)] = torch.tensor(seq, device=device)
                pos[i, :len(seq)] = torch.arange(len(seq), device=device)
            logits, _ = model(x, pos_ids=pos)
            logp = F.log_softmax(logits.float(), dim=-1)
            for i, seq in enumerate(chunk):
                n = len(seq)
                # token t predicted from position t-1
                tgt = torch.tensor(seq[pl[i]:n], device=device)
                lp = logp[i, pl[i] - 1:n - 1, :].gather(1, tgt.unsqueeze(1)).squeeze(1)
                outs.append(lp)
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--data', default='/root/nd/data')
    ap.add_argument('--out', required=True)
    ap.add_argument('--iters', type=int, default=200)
    ap.add_argument('--batch-thms', type=int, default=192)
    ap.add_argument('--group', type=int, default=8)
    ap.add_argument('--temp', type=float, default=1.0)
    ap.add_argument('--max-new', type=int, default=420)
    ap.add_argument('--lr', type=float, default=2e-5)
    ap.add_argument('--clip', type=float, default=0.2)
    ap.add_argument('--kl-beta', type=float, default=0.03)
    ap.add_argument('--epochs', type=int, default=2)
    ap.add_argument('--minibatch', type=int, default=384)
    ap.add_argument('--eval-every', type=int, default=50)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    device = 'cuda'
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    os.makedirs(args.out, exist_ok=True)

    model, cfg = load_model(args.ckpt, device)
    model.train()
    ref = copy.deepcopy(model).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0,
                            betas=(0.9, 0.95))

    targets = load_jsonl(os.path.join(args.data, 'rl_targets.jsonl'))[:1200]
    prompts = {}
    for t in targets:
        toks = t['text'].split()
        prompts[t['thm']] = [STOI[tk] for tk in toks[:toks.index('PRF') + 1]]

    solved = {}          # thm -> best verified length
    history = []
    cum_samples = 0
    t0 = time.time()

    for it in range(1, args.iters + 1):
        batch = rng.sample(targets, args.batch_thms)
        flat_prompts, metas = [], []
        for t in batch:
            for _ in range(args.group):
                flat_prompts.append(prompts[t['thm']])
                metas.append(t)
        model.eval()
        outs = model.sample(flat_prompts, max_new=args.max_new,
                            temperature=args.temp, eos_id=QED_ID,
                            pad_id=PAD, device=device)
        model.train()
        cum_samples += len(outs)

        rewards = []
        max_len_iter = 0
        for (t, p, o) in zip(metas, flat_prompts, outs):
            full = ' '.join(ITOS[x] for x in p + o)
            ok, _, nl = verify_text(full)
            rewards.append(1.0 if ok else 0.0)
            if ok:
                max_len_iter = max(max_len_iter, nl)
                if t['thm'] not in solved or nl < solved[t['thm']]:
                    solved[t['thm']] = nl

        # group advantages
        adv = [0.0] * len(rewards)
        n_active = 0
        for g in range(0, len(rewards), args.group):
            grp = rewards[g:g + args.group]
            m = sum(grp) / len(grp)
            if 0.0 < m < 1.0:
                n_active += 1
                sd = (sum((x - m) ** 2 for x in grp) / len(grp)) ** 0.5
                for j, r in enumerate(grp):
                    adv[g + j] = (r - m) / (sd + 1e-4)

        # build training set from non-zero-advantage, non-empty completions
        ids, plens, advs = [], [], []
        for p, o, a in zip(flat_prompts, outs, adv):
            if a != 0.0 and len(o) > 0 and len(p) + len(o) <= 512:
                ids.append(p + o)
                plens.append(len(p))
                advs.append(a)
        mean_r = sum(rewards) / len(rewards)
        if not ids:
            print(json.dumps({'it': it, 'mean_r': mean_r, 'skip': True}),
                  flush=True)
            continue

        old_lp = seq_logprobs(model, ids, plens, device)
        ref_lp = seq_logprobs(ref, ids, plens, device)
        old_lp = [x.detach() for x in old_lp]

        kl_log, clip_log = [], []
        idxs = list(range(len(ids)))
        for ep in range(args.epochs):
            rng.shuffle(idxs)
            for s in range(0, len(idxs), args.minibatch):
                mb = idxs[s:s + args.minibatch]
                mb_ids = [ids[i] for i in mb]
                mb_pl = [plens[i] for i in mb]
                new_lp = seq_logprobs(model, mb_ids, mb_pl, device, grad=True)
                loss_terms = []
                for k, i in enumerate(mb):
                    ratio = torch.exp(new_lp[k] - old_lp[i])
                    a = advs[i]
                    pg = -torch.min(ratio * a,
                                    torch.clamp(ratio, 1 - args.clip,
                                                1 + args.clip) * a)
                    # k3 KL estimator to reference
                    d = ref_lp[i] - new_lp[k]
                    kl = torch.exp(d) - 1 - d
                    loss_terms.append((pg + args.kl_beta * kl).mean())
                    kl_log.append(float(kl.mean().detach()))
                    clip_log.append(float(((ratio < 1 - args.clip) |
                                           (ratio > 1 + args.clip)).float()
                                          .mean().detach()))
                loss = torch.stack(loss_terms).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

        rec = {'it': it, 'mean_r': round(mean_r, 4),
               'active_groups': n_active, 'solved_cum': len(solved),
               'max_len_iter': max_len_iter,
               'max_len_cum': max(solved.values(), default=0) and max(
                   (v for v in solved.values()), default=0),
               'kl': round(sum(kl_log) / max(1, len(kl_log)), 5),
               'clip_frac': round(sum(clip_log) / max(1, len(clip_log)), 4),
               'cum_samples': cum_samples, 't': round(time.time() - t0)}
        # max discovered length overall
        rec['max_len_cum'] = max(solved.values()) if solved else 0
        print(json.dumps(rec), flush=True)
        history.append(rec)

        if it % args.eval_every == 0 or it == args.iters:
            model.eval()
            transfer = load_jsonl(os.path.join(args.data, 'eval_transfer.jsonl'))[:500]
            r1 = run_eval(model, transfer, 1, 0.0, 420, 1536, device, progress=False)
            rk = run_eval(model, transfer, 8, 0.8, 420, 1536, device, progress=False)
            iid = load_jsonl(os.path.join(args.data, 'eval_iid.jsonl'))[:400]
            ri = run_eval(model, iid, 1, 0.0, 420, 1536, device, progress=False)
            ev = {'it': it, 'EVAL': True,
                  'transfer_pass1': sum(r['solved'] for r in r1) / len(r1),
                  'transfer_pass8': sum(r['solved'] for r in rk) / len(rk),
                  'iid_pass1': sum(r['solved'] for r in ri) / len(ri),
                  'cum_samples': cum_samples}
            print(json.dumps(ev), flush=True)
            history.append(ev)
            torch.save({'model': model.state_dict(), 'config': cfg},
                       os.path.join(args.out, f'ckpt_it{it}.pt'))
            model.train()
        with open(os.path.join(args.out, 'history.json'), 'w') as f:
            json.dump({'solved_lens': solved, 'log': history}, f)

    # final: dump solved set with lengths
    with open(os.path.join(args.out, 'solved.json'), 'w') as f:
        json.dump(solved, f)


if __name__ == '__main__':
    main()
