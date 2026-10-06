#!/usr/bin/env python3
"""Batched sampling **in the environment** (run `state-env`): the policy sees a tactic state and writes one action.

  env_generate(model, tok, prompts, greedy=..., temperature=..., max_action=..., max_steps=..., batch=..., seed=...)
    -> list of ND proof strings, one per prompt (a clean ND string where Lean accepted the assembled `lean_seq`
       text, `'LEANREJ <nd>'` where Lean rejected it, `'LEANPARSE <reason>'` where the attempt never produced a
       finished proof) -- the same contract as `sample.generate` for a LeanTokenizer, so the evaluation and ladder
       code downstream is unchanged.

The attempts are kept in one worklist: a finished or failed attempt is replaced by the next pending one, so the
decode batch stays full even though attempts need different numbers of steps.  Decoding is `sample.generate_ids_fast`
with `early='eos'` (an action ends at <eos>) and compaction on; the noise of row r at step t of wave w is keyed by
(seed, w, t, slot), so a batch-size change is a sampling re-draw, never a correctness change (`NOISE_FLOOR.md`).

`stats` (a dict) collects the per-step diagnostics the run reports: how attempts end, steps per attempt, and the
action truncation rate.
"""
import collections, os, random, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from sample import generate_ids_fast
from state_env import Env


NAME_BASE_MAX = 32     # arm SN: canonical names start at n<base+1>, base ~ U[0, 32] per attempt (the Stage-1 offset is U[0, 64 - max name])


def prompt_ids(tok, env):
    toks = (env.hist + env.state_tokens()) if tok.with_history else env.state_tokens()
    return tok.encode_toks(toks)


@torch.no_grad()
def env_generate(model, tok, prompts, greedy=True, temperature=1.0, max_action=256, max_steps=48,
                 batch=2048, seed=0, stats=None, gate=True):
    dev = next(model.parameters()).device
    st = stats if stats is not None else {}
    cnt = st.setdefault('env_end', collections.Counter())
    why_cnt = st.setdefault('env_fail_reason', collections.Counter())
    steps_hist = st.setdefault('env_steps', collections.Counter())
    res_nd = [None] * len(prompts)
    res_tx = [None] * len(prompts)
    live = []
    nxt = 0
    wave = 0
    t0 = time.time()

    def finish(i, e):
        st['names_defined'] = st.get('names_defined', 0) + e.defined
        st['names_renamed'] = st.get('names_renamed', 0) + e.renamed
        nd = e.nd()
        res_nd[i] = nd
        res_tx[i] = tok.text(e.text) if e.done and not nd.startswith('LEANPARSE') else None
        steps_hist[e.steps] += 1

    while nxt < len(prompts) or live:
        while len(live) < batch and nxt < len(prompts):
            if getattr(tok, 'canon', False):      # arm SN: the environment names what a step introduces, from a random base
                base = random.Random(seed * 1000003 + nxt).randint(0, NAME_BASE_MAX)
                e = Env(prompts[nxt], canon=True, base=base, assign=True)
            else:
                e = Env(prompts[nxt])
            live.append((nxt, e)); nxt += 1
        pids = [prompt_ids(tok, e) for _, e in live]
        st['prompt_tokens'] = st.get('prompt_tokens', 0) + sum(len(p) for p in pids)   # prefill cost of the env loop
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=(dev.type == 'cuda')):
            outs = generate_ids_fast(model, tok, pids, goals=None, greedy=greedy, temperature=temperature,
                                     max_new=max_action, seed=seed * 100003 + wave, early='eos', compact=True,
                                     stats=st)
        wave += 1
        keep = []
        for (i, e), o in zip(live, outs):
            atoks, ended = tok.decode_action(o)
            if not ended:
                e.failed = 'action truncated'; cnt['truncated'] += 1; finish(i, e); continue
            ok, why = e.apply(atoks)
            if not ok:
                cnt['syntax'] += 1; why_cnt[why] += 1; finish(i, e); continue
            if e.done:
                cnt['done'] += 1; finish(i, e); continue
            if e.steps >= max_steps:
                e.failed = 'step cap'; cnt['step_cap'] += 1; finish(i, e); continue
            keep.append((i, e))
        live = keep
        del outs
    st['env_waves'] = st.get('env_waves', 0) + wave
    st['env_wall_s'] = st.get('env_wall_s', 0.0) + time.time() - t0
    if dev.type == 'cuda':
        st['peak_alloc_gb'] = torch.cuda.max_memory_allocated() / 2 ** 30
        st['peak_reserved_gb'] = torch.cuda.max_memory_reserved() / 2 ** 30
        torch.cuda.empty_cache()
    if gate:
        from lean_gate import gate as _gate
        res_nd = _gate(tok, prompts, res_nd, res_tx)
    return res_nd


def env_stats_json(st):
    """the json-safe part of a stats dict (Counters -> dicts, the per-row declen list dropped)."""
    out = {}
    for k, v in st.items():
        if k in ('declen', 'declen_by_prompt'):
            continue
        out[k] = dict(v) if isinstance(v, collections.Counter) else v
    dl = st.get('declen')
    if dl:
        h = collections.Counter(dl)
        out['action_declen_hist'] = dict(sorted(h.items()))
        out['action_declen_mean'] = sum(dl) / len(dl)
    return out
