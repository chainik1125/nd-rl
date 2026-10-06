#!/usr/bin/env python3
"""Teacher-forced intrinsic scorers for pool attempts under the frozen r16 generator (Worker B, schema section 3).

For every attempt in a pool (schema section 1) the stored model-emitted `actions` are replayed through
`Env(prompt, canon=True, base=base, assign=True)` -- exactly the sampler's environment -- to recover the state the
model saw at each step.  The model input of a step is `tok.encode_toks(env.state_tokens())` (state_sample.prompt_ids
for arm SN), followed by the emitted action ids and the `<eos>` that ended the action.  We compute log p(token) at
temperature 1 for every action token and the `<eos>`, and the entropy of the full next-token distribution there.

Masked positions (not the model's choice in arm SN, see STATE_ENV.md "Names, and arm SN"): `Env._assign_names`
overrides the `have` name (action index 1) and the box binder name (index i+3 after `(`, i+5 after `Or.elim`, with i
the index after the first `:=`).  Those positions are excluded from every aggregate.

Outputs (scores/<pool stem>__<scorer>.jsonl, one row per theorem, aligned with attempts, higher = more likely valid):
  loglik_mean        mean log p over unmasked action tokens of the attempt (all steps, incl. <eos>)
  loglik_sum         sum of the same
  loglik_min_step    min over steps of the step's mean token log p
  entropy_mean       minus the mean next-token entropy over the same positions
  completion_loglik  done ? loglik_mean : -inf   (also completion_loglik_sum / completion_minstep / completion_entropy)
  loglik_mean_masked = loglik_mean; loglik_mean_full = the same without masking the env-assigned name positions
plus a per-attempt detail file scores/aux/<pool stem>__loglik_detail.jsonl (n tokens, n masked, steps replayed,
replay consistency flags, tf-argmax agreement).

Usage:
  score_loglik.py --pool ~/runs/proof-selection/pools_v2/<stem>.jsonl [--out_dir ~/runs/proof-selection/scores]
"""
import argparse, json, math, multiprocessing as mp, os, sys, time

ND = os.path.expanduser('~/runs/proof-selection/nd-takehome')
sys.path.insert(0, ND)
import torch
import torch.nn.functional as F

CKPT = os.path.expanduser('~/runs/proof-selection/ckpts/la_T1_best12_s0_r16.pt')
SCORERS = ['loglik_mean', 'loglik_mean_masked', 'loglik_mean_full', 'loglik_sum', 'loglik_min_step', 'entropy_mean',
           'completion_loglik', 'completion_loglik_sum', 'completion_minstep', 'completion_entropy']

_TOK = None


def _tok():
    global _TOK
    if _TOK is None:
        from tokenizer import make_tokenizer
        _TOK = make_tokenizer('lean_staten')
    return _TOK


def name_mask(toks):
    """positions of an action (token strings) whose name the SN environment assigns (mirrors Env._assign_names)."""
    from state_env import is_name
    m = []
    if len(toks) > 1 and toks[0] == 'have':
        m.append(1)
        if ':=' in toks:
            i = toks.index(':=') + 1
            j = None
            if i < len(toks) and toks[i] == '(' and i + 3 < len(toks):
                j = i + 3
            elif i < len(toks) and toks[i] == 'Or.elim' and i + 5 < len(toks):
                j = i + 5
            if j is not None and is_name(toks[j]):
                m.append(j)
    return m


SHA_FIELDS = ('prompt_sha1', 'state_sha1', 'prompt_ids_sha1', 'sha1')
RAW_FIELDS = ('raw_actions', 'raw_ids', 'actions_raw')


def state_hash_ok(ids, h):
    """Worker A's collector: sha1(','.join(ids)).hexdigest()[:16] of the prompt ids the model saw (pool_lib.state_sha1)."""
    import hashlib
    return hashlib.sha1(','.join(map(str, ids)).encode()).hexdigest()[:len(h)] == h


def replay(at):
    """attempt dict (+ prompt, max_action) -> per-step (state ids, teacher-forced action ids, masked positions, ended).

    If the pool carries raw sampled ids per action (incl. sampled pads, through the real <eos>), those are teacher-forced
    exactly; otherwise the decoded `actions` are used and an action is taken to have ended with <eos> unless it is
    truncated (`has_eos` if stored, else len == max_action).  <eos> is never added to a truncated action."""
    from state_env import Env
    tok = _tok()
    prompt, base, actions, done, max_action, nd = (at['prompt'], at.get('base', 0), at['actions'], at['done'],
                                                   at['max_action'], at.get('nd'))
    raw = next((at[f] for f in RAW_FIELDS if at.get(f) is not None), None)
    shas = next((at[f] for f in SHA_FIELDS if at.get(f) is not None), None)
    has_eos = at.get('has_eos')
    env = Env(prompt, canon=True, base=base, assign=True)
    steps = []
    err = None
    sha_bad = 0
    for k, a in enumerate(actions):
        if env.done or env.failed:
            err = f'actions after end at step {k}'
            break
        st = tok.encode_toks(env.state_tokens())
        if shas is not None and (k >= len(shas) or not state_hash_ok(st, shas[k])):
            sha_bad += 1
        toks = [tok.itos[x] for x in a]
        m = name_mask(toks)
        if raw is not None:
            r = list(raw[k])
            ended = tok.eos in r
            if ended:
                r = r[:r.index(tok.eos)]
            nonpad = [j for j, x in enumerate(r) if x != tok.pad]
            assert [r[j] for j in nonpad] == list(a), 'raw ids do not decode to the stored action'
            ids, m = r, [nonpad[j] for j in m]
        else:
            ended = bool(has_eos[k]) if has_eos is not None else len(a) < max_action
            ids = list(a)
        steps.append((st, ids, m, ended))
        ok, why = env.apply(toks)
    replay_done = bool(env.done)
    nd_match = (env.nd() == nd) if (replay_done and nd is not None) else None   # byte-for-byte reassembly check
    return {'steps': steps, 'replay_done': replay_done, 'done_match': replay_done == bool(done), 'err': err,
            'n_steps': len(actions), 'nd_match': nd_match, 'sha_checked': shas is not None, 'sha_bad': sha_bad,
            'used_raw': raw is not None}


def batches(lengths, budget_tokens, max_rows):
    """indices sorted by length, chunked so that rows * T^2 stays under a budget (the ALiBi bias is B*H*T*T)."""
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    out, cur = [], []
    for i in order:
        T = lengths[i]
        if cur and ((len(cur) + 1) * T * T > budget_tokens or len(cur) >= max_rows):
            out.append(cur); cur = []
        cur.append(i)
    if cur:
        out.append(cur)
    return out


@torch.no_grad()
def score_pairs(model, pairs, device, budget=2 ** 26, max_rows=4096, bf16=True, want_argmax=True):
    """pairs: list of (state ids, action ids, mask positions, ended).  Returns per pair: (lp list, ent list, argmax-match
    list) over the action positions (+ <eos> if ended), with masked positions marked by None."""
    tok = _tok()
    seqs = []
    for st, a, m, ended in pairs:
        tgt = list(a) + ([tok.eos] if ended else [])
        seqs.append((st, tgt))
    lengths = [len(s) + len(t) for s, t in seqs]
    res = [None] * len(pairs)
    for b in batches(lengths, budget, max_rows):
        T = max(lengths[i] for i in b)
        idx = torch.full((len(b), T), tok.pad, dtype=torch.long)
        for r, i in enumerate(b):
            s, t = seqs[i]
            idx[r, :len(s) + len(t)] = torch.tensor(s + t)
        idx = idx.to(device)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=bf16 and device.type == 'cuda'):
            logits = model(idx)            # right padding + causal ALiBi mask: pads never influence real positions
        logp = F.log_softmax(logits.float(), -1)
        ent = -(logp.exp() * logp).sum(-1)                         # (B, T)
        tgt_lp = logp[:, :-1].gather(-1, idx[:, 1:, None])[..., 0]  # lp of token t+1 given prefix up to t
        am = (logp[:, :-1].argmax(-1) == idx[:, 1:])
        tgt_lp, ent, am = tgt_lp.cpu(), ent.cpu(), am.cpu()
        for r, i in enumerate(b):
            s, t = seqs[i]
            p0 = len(s) - 1                 # position predicting the first action token
            lp = tgt_lp[r, p0:p0 + len(t)].tolist()
            en = ent[r, p0:p0 + len(t)].tolist()
            ag = am[r, p0:p0 + len(t)].tolist()
            full = list(lp)
            for j in pairs[i][2]:
                lp[j] = None; en[j] = None; ag[j] = None
            res[i] = (lp, en, ag, full)
    return res


def aggregate(step_res):
    lps, ens, ags, step_means, fulls = [], [], [], [], []
    for lp, en, ag, full in step_res:
        fulls += full
        l = [x for x in lp if x is not None]
        lps += l
        ens += [x for x in en if x is not None]
        ags += [x for x in ag if x is not None]
        if l:
            step_means.append(sum(l) / len(l))
    n = len(lps)
    if n == 0:
        return dict(loglik_mean=-math.inf, loglik_sum=-math.inf, loglik_min_step=-math.inf, entropy_mean=-math.inf,
                    n_tok=0, argmax_agree=None, loglik_mean_full=-math.inf, loglik_mean_masked=-math.inf)
    return dict(loglik_mean=sum(lps) / n, loglik_mean_masked=sum(lps) / n, loglik_mean_full=sum(fulls) / len(fulls), loglik_sum=sum(lps), loglik_min_step=min(step_means),
                entropy_mean=-sum(ens) / len(ens), n_tok=n, argmax_agree=sum(ags) / len(ags))


def load_model(device):
    from model import load_ckpt
    m, tk, _ = load_ckpt(CKPT, device)
    assert tk.mode == 'lean_staten'
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pool', required=True)
    ap.add_argument('--out_dir', default=os.path.expanduser('~/runs/proof-selection/scores'))
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--fp32', action='store_true')
    a = ap.parse_args()
    t0 = time.time()
    stem = os.path.basename(a.pool)[:-len('.jsonl')]
    rows = [json.loads(l) for l in open(a.pool)]
    if a.limit:
        rows = rows[:a.limit]
    jobs, owner = [], []
    for ri, r in enumerate(rows):
        ma = (r.get('sampling') or {}).get('max_action', 512)
        for ai, at in enumerate(r['attempts']):
            jobs.append(dict(at, prompt=r['prompt'], max_action=ma))
            owner.append((ri, ai))
    with mp.Pool(a.workers) as p:
        reps = p.map(replay, jobs, chunksize=256)
    t1 = time.time()
    pairs, pair_owner = [], []
    for j, rp in enumerate(reps):
        for s in rp['steps']:
            pairs.append(s); pair_owner.append(j)
    device = torch.device('cuda')
    model = load_model(device)
    pres = score_pairs(model, pairs, device, bf16=not a.fp32)
    t2 = time.time()
    per_att = [[] for _ in jobs]
    for k, j in enumerate(pair_owner):
        per_att[j].append(pres[k])
    out = {s: [[None] * len(r['attempts']) for r in rows] for s in SCORERS}
    os.makedirs(os.path.join(a.out_dir, 'aux'), exist_ok=True)
    det = [[None] * len(r['attempts']) for r in rows]
    n_mismatch = n_err = n_ndbad = n_ndchk = n_sha = n_shabad = n_raw = 0
    feats = [[None] * len(r['attempts']) for r in rows]
    for j, (ri, ai) in enumerate(owner):
        ag = aggregate(per_att[j])
        done = rows[ri]['attempts'][ai]['done']
        ag['completion_loglik'] = ag['loglik_mean'] if done else -math.inf
        ag['completion_loglik_sum'] = ag['loglik_sum'] if done else -math.inf
        ag['completion_minstep'] = ag['loglik_min_step'] if done else -math.inf
        ag['completion_entropy'] = ag['entropy_mean'] if done else -math.inf
        for s in SCORERS:
            out[s][ri][ai] = ag[s]
        rp = reps[j]
        n_mismatch += not rp['done_match']; n_err += rp['err'] is not None
        n_ndchk += rp['nd_match'] is not None; n_ndbad += rp['nd_match'] is False
        n_sha += rp['sha_checked']; n_shabad += rp['sha_bad']; n_raw += rp['used_raw']
        pre = sum(len(st) for st, _, _, _ in rp['steps'])
        fwd = sum(len(st) + len(ids) + int(e) for st, ids, _, e in rp['steps'])
        at = rows[ri]['attempts'][ai]
        feats[ri][ai] = dict(loglik_mean_masked=ag['loglik_mean_masked'], loglik_mean_full=ag['loglik_mean_full'],
                             loglik_sum=ag['loglik_sum'], loglik_min_step=ag['loglik_min_step'],
                             entropy_mean=ag['entropy_mean'], n_tok=ag['n_tok'], steps=at['steps'], done=at['done'],
                             base=at.get('base'), gen_tokens=at.get('gen_tokens'), prefill_tokens_replay=pre,
                             tf_forward_tokens=fwd)
        det[ri][ai] = dict(n_tok=ag['n_tok'], argmax_agree=ag['argmax_agree'], steps=rp['n_steps'],
                           replay_done=rp['replay_done'], done_match=rp['done_match'], err=rp['err'], nd_match=rp['nd_match'],
                           step_mean_lp=[(sum(x for x in lp if x is not None) / max(1, sum(x is not None for x in lp)))
                                         for lp, _, _, _ in per_att[j]])
    for s in SCORERS:
        with open(os.path.join(a.out_dir, f'{stem}__{s}.jsonl'), 'w') as f:
            for r, sc in zip(rows, out[s]):
                f.write(json.dumps({'name': r['name'], 'scores': sc}) + '\n')
    with open(os.path.join(a.out_dir, 'aux', f'{stem}__shallow_features.jsonl'), 'w') as f:
        for r, d in zip(rows, feats):     # visible per-attempt features for Worker A's shallow_lr baseline
            f.write(json.dumps({'name': r['name'], 'features': d}) + '\n')
    with open(os.path.join(a.out_dir, 'aux', f'{stem}__loglik_detail.jsonl'), 'w') as f:
        for r, d in zip(rows, det):
            f.write(json.dumps({'name': r['name'], 'detail': d}) + '\n')
    summ = dict(pool=a.pool, n_theorems=len(rows), n_attempts=len(jobs), n_pairs=len(pairs),
                replay_done_mismatch=n_mismatch, replay_errors=n_err,
                nd_checked=n_ndchk, nd_mismatch=n_ndbad, sha_checked_attempts=n_sha, sha_bad_steps=n_shabad,
                attempts_with_raw_ids=n_raw, secs_replay=t1 - t0, secs_forward=t2 - t1,
                secs_total=time.time() - t0, ckpt=CKPT, bf16=not a.fp32)
    with open(os.path.join(a.out_dir, 'aux', f'{stem}__loglik_summary.json'), 'w') as f:
        json.dump(summ, f, indent=1)
    print(json.dumps(summ))


if __name__ == '__main__':
    main()
