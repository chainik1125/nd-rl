#!/usr/bin/env python3
"""Pool collector (Worker A): a fork of Dan's `state_sample.env_generate` loop that keeps every attempt.

Per attempt it records (schema §1 plus the 02:08 reviewer amendments):
  actions      decoded action ids per step (pads and eos stripped; `tok.decode_action`), failing/truncated one included
  raw_actions  the RAW sampled ids per step through the actual <eos> (no eos appended to a truncated action)
  emitted_len  len(raw_actions[t]);  has_eos: whether action t ended with a sampled <eos>
  state_sha1   sha1 of the prompt ids (state) the model saw before each action (replay verification)
  prefill_tokens (sum of state lengths over steps), gen_tokens (sum of emitted_len), base, done, steps,
  nd           the assembled ND text for every done attempt (null only if the env could not invert it)
The pool file is written BEFORE Lean is called; the `.labels.jsonl` (verdict + reason) is written only afterwards.

Sampling settings are Dan's: T 0.8, max_action 512, max_steps 96, batch 2048, early='eos', compaction on, arm SN
(`Env(canon=True, base=b, assign=True)`, b = Random(seed*1000003 + flat_index).randint(0, 32), as in env_generate).
The k attempts of a theorem are flat indices t*k .. t*k+k-1 (independent fresh contexts).

  python collect_pools.py --set dev1108 --k 16 --seed 0
  python collect_pools.py --set textbook72 --k 8 --limit 3 --out /tmp/x.jsonl --check_replay
"""
import argparse, collections, hashlib, json, os, random, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pool_lib as PL
import torch
from model import load_ckpt
from sample import generate_ids_fast
from state_env import Env
from state_sample import NAME_BASE_MAX, prompt_ids
import lean_judge

T, MAX_ACTION, MAX_STEPS, BATCH = 0.8, 512, 96, 2048


@torch.no_grad()
def collect(model, tok, prompts, seed, temperature=T, max_action=MAX_ACTION, max_steps=MAX_STEPS, batch=BATCH,
            log=print):
    dev = next(model.parameters()).device
    n = len(prompts)
    recs = [None] * n
    live, nxt, wave = [], 0, 0
    ends = collections.Counter()
    t0 = time.time()

    def finish(i, e, acts, gtok, end):
        nd = e.nd() if e.done else None
        if nd is not None and nd.startswith('LEANPARSE'):
            ends['done_but_unparsable'] += 1
            nd_clean, why = None, 'env done but inverse failed: ' + nd[10:]
        else:
            nd_clean, why = nd, (None if e.done else end)
        acts, raws, shas, pre = acts
        recs[i] = dict(base=e.base, actions=acts, raw_actions=raws, emitted_len=[len(r) for r in raws],
                       has_eos=[bool(r) and r[-1] == tok.eos for r in raws], state_sha1=shas, states=None,
                       done=bool(e.done), steps=e.steps, nd=nd_clean, prefill_tokens=pre,
                       gen_tokens=sum(len(r) for r in raws), _why=why)
        ends[end] += 1

    while nxt < n or live:
        while len(live) < batch and nxt < n:
            base = random.Random(seed * 1000003 + nxt).randint(0, NAME_BASE_MAX)
            live.append([nxt, Env(prompts[nxt], canon=True, base=base, assign=True), [[], [], [], 0], 0]); nxt += 1
        pids = [prompt_ids(tok, e) for _, e, _, _ in live]
        for slot, p in zip(live, pids):
            slot[2][2].append(PL.state_sha1(p)); slot[2][3] += len(p)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=(dev.type == 'cuda')):
            outs = generate_ids_fast(model, tok, pids, goals=None, greedy=False, temperature=temperature,
                                     max_new=max_action, seed=seed * 100003 + wave, early='eos', compact=True,
                                     stats=None)
        wave += 1
        keep = []
        for slot, o in zip(live, outs):
            i, e, acts, gtok = slot
            o = o.tolist() if hasattr(o, 'tolist') else list(o)
            atoks, ended = tok.decode_action(o)
            raw = o[:o.index(tok.eos) + 1] if tok.eos in o else o
            acts[0].append(tok.encode_toks(atoks)); acts[1].append(raw)
            if not ended:
                e.failed = 'action truncated'; finish(i, e, acts, gtok, 'truncated'); continue
            ok, why = e.apply(atoks)
            if not ok:
                finish(i, e, acts, gtok, 'syntax: ' + why); continue
            if e.done:
                finish(i, e, acts, gtok, 'done'); continue
            if e.steps >= max_steps:
                e.failed = 'step cap'; finish(i, e, acts, gtok, 'step cap'); continue
            keep.append(slot)
        live = keep
        del outs
        if wave % 10 == 0:
            log(f'  wave {wave} finished {sum(r is not None for r in recs)}/{n} live {len(live)} {time.time()-t0:.0f}s')
    if dev.type == 'cuda':
        torch.cuda.empty_cache()
    return recs, dict(waves=wave, sample_wall_s=time.time() - t0,
                      ends=dict(collections.Counter(k.split(':')[0] for k in ends.elements())))


def judge(prompts, recs):
    idx = [i for i, r in enumerate(recs) if r['done'] and r['nd'] is not None]
    t0 = time.time()
    res = lean_judge.judge_many([(prompts[i], recs[i]['nd']) for i in idx]) if idx else []
    valid = [False] * len(recs)
    reason = [r['_why'] for r in recs]
    for i, (ok, why, _) in zip(idx, res):
        valid[i] = bool(ok)
        reason[i] = None if ok else (why or 'lean rejected')
    return valid, reason, time.time() - t0


def check_replay(prompts, recs, tok):
    """every attempt: per-step state sha1, steps, done; done attempts: nd byte for byte."""
    bad = sum(not PL.check_attempt(p, r, tok) for p, r in zip(prompts, recs))
    return len(recs), bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--set', required=True, choices=list(PL.SET_FILES))
    ap.add_argument('--k', type=int, default=None)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--limit', type=int, default=0, help='first N theorems only (tests)')
    ap.add_argument('--batch', type=int, default=BATCH)
    ap.add_argument('--ckpt', default=PL.CKPT)
    ap.add_argument('--out', default=None)
    ap.add_argument('--check_replay', action='store_true')
    a = ap.parse_args()
    k = a.k or PL.SET_K[a.set]
    out = a.out or PL.pool_path(a.set, k)
    lab_out = out[:-len('.jsonl')] + '.labels.jsonl'
    os.makedirs(os.path.dirname(out), exist_ok=True)
    log = lambda s: print(f'[{time.strftime("%H:%M:%S", time.gmtime())}] {s}', flush=True)
    theorems = PL.load_set(a.set)
    if a.limit:
        theorems = theorems[:a.limit]
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok, _ = load_ckpt(a.ckpt, dev)
    assert tok.canon, tok.mode
    prompts = [th['prompt'] for th in theorems for _ in range(k)]
    log(f'{a.set}: {len(theorems)} theorems x k={k} = {len(prompts)} attempts, seed {a.seed}, ckpt {a.ckpt}')
    t0 = time.time()
    recs, info = collect(model, tok, prompts, a.seed, batch=a.batch, log=log)
    sample_wall = time.time() - t0
    pool_rows, lab_rows = [], []
    for t, th in enumerate(theorems):
        att = []
        for j in range(k):
            r = dict(recs[t * k + j]); r.pop('_why')
            att.append(dict(i=j, **r))
        pool_rows.append(dict(name=th['name'], prompt=th['prompt'], set=a.set, ref_lines=PL.ref_lines(th), gen=PL.GEN,
                              k=k, sampling=dict(temperature=T, max_action=MAX_ACTION, max_steps=MAX_STEPS,
                                                 seed=a.seed), attempts=att))
    PL.write_jsonl(out, pool_rows)             # pool serialised before any Lean call
    log(f'wrote pool {out} ({sample_wall:.0f}s); judging')
    valid, reason, lean_s = judge(prompts, recs)
    wall = time.time() - t0
    for t, th in enumerate(theorems):
        lab_rows.append(dict(name=th['name'], valid=valid[t * k:(t + 1) * k], reason=reason[t * k:(t + 1) * k]))
    PL.write_jsonl(lab_out, lab_rows)
    nfin = sum(r['done'] for r in recs)
    summ = dict(set=a.set, file=os.path.basename(out), labels=os.path.basename(lab_out), n_theorems=len(theorems), k=k,
                seed=a.seed, attempts=len(recs), sample_wall_s_total=sample_wall,
                prefill_tokens=sum(r['prefill_tokens'] for r in recs), finished_rate=nfin / len(recs), validity_rate=sum(valid) / len(recs),
                p_valid_given_finished=sum(valid) / max(1, nfin), wall_s=wall, lean_s=lean_s,
                pass_at_k=sum(any(v['valid']) for v in lab_rows) / len(lab_rows),
                gen_tokens=sum(r['gen_tokens'] for r in recs), lean_judge_stats=lean_judge.stats(), **info)
    if a.check_replay:
        n, bad = check_replay(prompts, recs, tok)
        summ['replay_checked'] = n; summ['replay_mismatch'] = bad
        log(f'replay check: {n} attempts ({sum(r["done"] for r in recs)} done), {bad} mismatches')
    json.dump(summ, open(out[:-len('.jsonl')] + '.summary.json', 'w'), indent=1)
    log(json.dumps(summ))


if __name__ == '__main__':
    main()
