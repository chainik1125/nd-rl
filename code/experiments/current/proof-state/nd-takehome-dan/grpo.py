#!/usr/bin/env python3
"""Minimal GRPO against the verifier (run 4): on-policy, binary reward, group-mean baseline, fixed loss divisor, no KL,
one gradient update per sampled batch.  Bookkeeping mirrors expert_iter.py so phase2_metrics / run analyses apply
unchanged: the total sample budget (--rounds x --k x N targets, default 8 x 32 x N = EI's budget) is split into --rounds
"round-equivalents"; at each boundary r the script writes artifacts/<name>/found_<r>.jsonl (every distinct verified proof of
a target sampled DURING TRAINING so far, with the round-equivalent it first appeared in), found_transfer_<r>.jsonl (a
pass@k sample of the transfer pool at the boundary, never trained on), round_<r>.json (targets_cum, transfer_cum, greedy on
transfer and held-out, mean reward, fraction of groups with reward variance per step) and ckpts/<name>_r<r>.pt.

  python grpo.py --init ckpts/p2/stage1_depth3_f0_a1_s0.pt --name r4/grpo_g8_a1_s0 --targets data/p2/targets_depth3.jsonl \
      --transfer data/p2/transfer_depth3.jsonl --heldout data/p2/heldout.jsonl --group 8 --prompts 64 --lr 1e-4 --seed 0

Update: for each step sample P prompts (targets, cycling through a shuffled order) x G completions at temperature T;
reward = Lean accepts the completion as a proof of the prompted sequent; advantage A = r - mean(r over the group);
loss = - sum_i sum_t A_i * log pi(y_t | prompt, y_<t) / (P * G * --divisor)  (divisor fixed = --max_new, not the length),
one Adam step (grad-norm clip 1.0).  Sampling uses the current parameters (no replay), so the update is on-policy.
"""
import argparse, json, os, sys, random, collections, time, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, torch.nn.functional as F
from model import load_ckpt
from sample import generate_ids, generate
from lean_judge import judge_many    # Lean alone decides (Dan, 2026-09-27)
from prune import pruned_length
from normalize import norm
from eval_set import judge, summarize


def read(fn):
    return [json.loads(l) for l in open(fn) if l.strip()]


def seq_logprobs(model, tok, prompt_ids, comp_ids):
    """sum of log pi over completion tokens (up to and including the first eos), right-padded batch. -> (B,) tensor"""
    dev = next(model.parameters()).device
    seqs, masks = [], []
    for p, c in zip(prompt_ids, comp_ids):
        cc = []
        for t in c:
            cc.append(t)
            if t == tok.eos:
                break
        seqs.append(p + cc); masks.append([0] * len(p) + [1] * len(cc))
    T = max(len(s) for s in seqs)
    x = torch.full((len(seqs), T), tok.pad, dtype=torch.long)
    m = torch.zeros((len(seqs), T), dtype=torch.bool)
    for i, (s, mk) in enumerate(zip(seqs, masks)):
        x[i, :len(s)] = torch.tensor(s); m[i, :len(s)] = torch.tensor(mk, dtype=torch.bool)
    x, m = x.to(dev), m.to(dev)
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=(dev.type == 'cuda')):
        logits = model(x[:, :-1])
    lp = -F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), x[:, 1:].reshape(-1), reduction='none').view(x.size(0), -1)
    return (lp * m[:, 1:]).sum(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--init', required=True); ap.add_argument('--name', required=True)
    ap.add_argument('--targets', required=True); ap.add_argument('--transfer', required=True); ap.add_argument('--heldout', required=True)
    ap.add_argument('--rounds', type=int, default=8); ap.add_argument('--k', type=int, default=32, help='EI-equivalent attempts per target per round (sets the budget)')
    ap.add_argument('--group', type=int, default=8); ap.add_argument('--prompts', type=int, default=64, help='prompts per step')
    ap.add_argument('--temperature', type=float, default=0.8); ap.add_argument('--lr', type=float, default=1e-4); ap.add_argument('--divisor', type=float, default=None)
    ap.add_argument('--max_new', type=int, default=400); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--batch', type=int, default=512)
    ap.add_argument('--eval_k', type=int, default=32, help='transfer pass@k at each boundary')
    ap.add_argument('--lp_batch', type=int, default=128, help='sequences per forward/backward chunk in the update (memory)')
    a = ap.parse_args()
    out = f'artifacts/{a.name}'; os.makedirs(out, exist_ok=True); os.makedirs('ckpts/' + os.path.dirname(a.name), exist_ok=True)
    json.dump(vars(a), open(f'{out}/args.json', 'w'), indent=1)
    dev = 'cuda'
    torch.manual_seed(a.seed); rng = random.Random(a.seed)
    model, tok, _ = load_ckpt(a.init, dev)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0, betas=(0.9, 0.95))
    targets, transfer, heldout = read(a.targets), read(a.transfer), read(a.heldout)
    N = len(targets); budget = a.rounds * a.k * N
    per_step = a.prompts * a.group; steps = budget // per_step
    boundaries = {int(round(r * steps / a.rounds)): r for r in range(1, a.rounds + 1)}
    divisor = a.divisor or a.max_new
    print(f'{N} targets, budget {budget} samples = {steps} steps of {per_step}; round boundaries at steps {sorted(boundaries)}', flush=True)
    found = collections.defaultdict(dict)           # name -> {norm proof: {proof, written, pruned, round}}
    order = list(range(N)); rng.shuffle(order); ptr = 0
    pids = {t['name']: tok.encode_prompt(t['prompt']) for t in targets}
    gen = torch.Generator(device=dev); gen.manual_seed(a.seed)
    stats_steps = []
    t0 = time.time(); samples_seen = 0
    for step in range(1, steps + 1):
        # ---- sample P prompts x G completions (on-policy)
        idxs = []
        for _ in range(a.prompts):
            if ptr >= N: rng.shuffle(order); ptr = 0
            idxs.append(order[ptr]); ptr += 1
        prompts = [targets[i] for i in idxs]
        pid = [pids[t['name']] for t in prompts for _ in range(a.group)]
        model.eval()
        comps = []
        with torch.no_grad():
            for s in range(0, len(pid), a.batch):
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=True):
                    comps += generate_ids(model, tok, pid[s:s + a.batch], greedy=False, temperature=a.temperature, max_new=a.max_new, gen=gen)
        model.train()
        texts = [tok.decode(c) for c in comps]
        rewards = torch.zeros(len(pid), device=dev)
        rnum = boundaries.get(min(b for b in boundaries if b >= step), a.rounds) if any(b >= step for b in boundaries) else a.rounds
        flat_t = [t for t in prompts for _ in range(a.group)]
        verdicts = judge_many([(t['prompt'], txt) for t, txt in zip(flat_t, texts)])   # batched (grpo uses generate_ids: no gate)
        for j, (t, txt) in enumerate(zip(flat_t, texts)):
            ok, reason, nl = verdicts[j]
            if ok:
                rewards[j] = 1.0
                pn = norm(txt)
                if pn not in found[t['name']]:
                    found[t['name']][pn] = {'proof': txt, 'written': nl, 'pruned': pruned_length(t['prompt'], txt), 'round': rnum}
        samples_seen += len(pid)
        R = rewards.view(a.prompts, a.group)
        adv = (R - R.mean(1, keepdim=True)).view(-1)
        var_groups = float(((R.mean(1) > 0) & (R.mean(1) < 1)).float().mean())
        # ---- one policy-gradient update with the group-mean baseline, fixed divisor
        loss = torch.zeros((), device=dev)
        if adv.abs().sum() > 0:
            opt.zero_grad(set_to_none=True)
            for s in range(0, len(pid), a.lp_batch):      # chunked forward/backward: gradients accumulate, memory stays bounded
                sel = [j for j in range(s, min(s + a.lp_batch, len(pid))) if adv[j] != 0]
                if not sel: continue
                lp = seq_logprobs(model, tok, [pid[j] for j in sel], [comps[j] for j in sel])
                l = -(adv[sel] * lp).sum() / (a.prompts * a.group * divisor)
                l.backward(); loss = loss + l.detach()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        else:
            gn = torch.zeros(())
        st = {'step': step, 'mean_reward': float(R.mean()), 'frac_groups_with_variance': var_groups, 'loss': float(loss), 'grad_norm': float(gn), 'samples': samples_seen}
        stats_steps.append(st)
        if step % 10 == 0:
            print(f"step {step}/{steps} reward {st['mean_reward']:.3f} var-groups {var_groups:.2f} loss {float(loss):.4f} gn {float(gn):.2f} solved-so-far {sum(1 for v in found.values() if v)} {time.time()-t0:.0f}s", flush=True)
        # ---- round-equivalent boundary: EI-style bookkeeping
        if step in boundaries:
            r = boundaries[step]
            model.eval()
            ck = f'ckpts/{a.name}_r{r}.pt'
            torch.save({'cfg': model.cfg, 'state': model.state_dict(), 'tok_mode': tok.mode, 'extra': {'grpo_round': r, 'step': step}}, ck)
            with open(f'{out}/found_{r}.jsonl', 'w') as f:
                for t in targets:
                    for x in found[t['name']].values():
                        f.write(json.dumps({'name': t['name'], 'thm': t['thm'], 'prompt': t['prompt'], 'gen_lines': t['n_lines'], **x}) + '\n')
            # transfer pass@eval_k (never trained on) and greedy evals, as expert_iter does
            outs_t = generate(model, tok, [t['prompt'] for t in transfer for _ in range(a.eval_k)], greedy=False, temperature=a.temperature, batch=a.batch, seed=a.seed * 100 + r)
            found_t = collections.defaultdict(dict)
            vt = judge_many([(t['prompt'], p) for i, t in enumerate(transfer) for p in outs_t[i * a.eval_k:(i + 1) * a.eval_k]])
            at = 0
            for i, t in enumerate(transfer):
                for p in outs_t[i * a.eval_k:(i + 1) * a.eval_k]:
                    ok, reason, nl = vt[at]; at += 1
                    if ok:
                        pn = norm(p)
                        if pn not in found_t[t['name']]:
                            found_t[t['name']][pn] = {'proof': p, 'written': nl, 'pruned': pruned_length(t['prompt'], p), 'round': r}
            with open(f'{out}/found_transfer_{r}.jsonl', 'w') as f:
                for t in transfer:
                    for x in found_t[t['name']].values():
                        f.write(json.dumps({'name': t['name'], 'thm': t['thm'], 'prompt': t['prompt'], 'gen_lines': t['n_lines'], **x}) + '\n')
            cum = [{'n_lines': t['n_lines'], 'solved': bool(found[t['name']]), 'written_lens': [x['written'] for x in found[t['name']].values()], 'pruned_lens': [x['pruned'] for x in found[t['name']].values()], 'reasons': []} for t in targets]
            cum_t = [{'n_lines': t['n_lines'], 'solved': bool(found_t[t['name']]), 'written_lens': [x['written'] for x in found_t[t['name']].values()], 'pruned_lens': [x['pruned'] for x in found_t[t['name']].values()], 'reasons': []} for t in transfer]
            g = generate(model, tok, [t['prompt'] for t in transfer], greedy=True, batch=a.batch)
            gh = generate(model, tok, [t['prompt'] for t in heldout], greedy=True, batch=a.batch)
            stats = {'round': r, 'k': a.k, 'step': step, 'samples': samples_seen, 'group': a.group, 'prompts_per_step': a.prompts,
                     'targets_cum': summarize(cum, 'n_lines', f'[{a.name} r{r}] targets (cumulative, {samples_seen} training samples)'),
                     'transfer_cum': summarize(cum_t, 'n_lines', f'[{a.name} r{r}] transfer pass@{a.eval_k} at boundary'),
                     'transfer_greedy': summarize(judge(transfer, [[p] for p in g], 'n_lines'), 'n_lines', f'[{a.name} r{r}] transfer greedy'),
                     'heldout_greedy': summarize(judge(heldout, [[p] for p in gh], 'n_lines'), 'n_lines', f'[{a.name} r{r}] heldout greedy'),
                     'targets_round': {'rate': float(sum(s['mean_reward'] for s in stats_steps[-max(1, steps // a.rounds):]) / max(1, len(stats_steps[-max(1, steps // a.rounds):]))), 'reasons': {}},
                     'transfer_round': {'rate': None},
                     'frac_groups_with_variance_mean': float(sum(s['frac_groups_with_variance'] for s in stats_steps[-max(1, steps // a.rounds):]) / max(1, len(stats_steps[-max(1, steps // a.rounds):]))),
                     'steps': stats_steps[-max(1, steps // a.rounds):], 'secs': time.time() - t0}
            json.dump(stats, open(f'{out}/round_{r}.json', 'w'), indent=1)
            print(f"[{a.name} r{r}] step {step} samples {samples_seen} targets solved {stats['targets_cum']['solved']}/{N} transfer {stats['transfer_cum']['solved']}/{len(transfer)} heldout greedy {stats['heldout_greedy']['rate']:.3f} var-groups {stats['frac_groups_with_variance_mean']:.2f}", flush=True)
            model.train()
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
