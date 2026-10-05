#!/usr/bin/env python3
"""guided_sample.py -- plain and guided sampling in the proof-state environment (run `guided-tts`, proposal 24).

    guided_generate(model, tok, prompts, arm, temperature=0.8, max_action=512, max_steps=96, max_rej=10,
                    batch=2048, seed=0, stats=None, dump=None) -> list of per-attempt dicts

Arms (one code path; they differ only in which rejections are redrawn):
  plain       an action the environment rejects, or a truncated action, ends the attempt (`state_sample.py`'s
              semantics).  Logically wrong steps go through; Lean rejects the finished proof.
  structural  such an action is redrawn from the same state.
  logical     additionally, an action `step_check.check` rejects (Lean is certain to reject it) is redrawn.
Up to `max_rej` rejections per attempt; the next one fails the attempt.  `max_steps` counts accepted actions.

Redraws are **without replacement** over token sequences, exactly: `_swor_shift` (Robbie's, from
nd-rl `robbie-experiments:code/experiments/current/test_time_scaling/engine.py`, unchanged; UniqueRandomizer, Shi et
al. 2020) adds log(1 - R(path + c)) to the tempered logits of a row whose state has rejected actions, R being the
probability of going on to write one of them exactly.  The forbidden set is per attempt and per state (cleared when
an action is accepted), so attempts stay i.i.d.  Two sequences that differ only in a name the environment
overrides (assign=True) are the same action but different sequences, so such a repeat is not excluded (rare: the
model is trained on canonical names).  `rej` counts redrawn rejections; the rejection that ends an attempt at the cap
is in `causes` but not in `rej` (status `fail:max_rej`).  A truncated action (no <eos> in `max_action` tokens) is not a
complete sequence and cannot be forbidden; it is redrawn with replacement.

Noise: Gumbel-max with noise keyed by (seed, wave, step, slot) (`sample._gumbel_pick`), as `state_sample.py`.

The logical check runs on every structurally valid action in every arm (for the diagnostics: rejection causes,
the agreement test's step dump, the proof-level false-reject check); only the logical arm acts on it.  Verdicts are
cached per (state, action).

Per attempt the result dict holds: prompt index, status (done / fail cause), sampled tokens (all draws, rejected
included), prefill tokens, draws, rejections by cause, `flag` (some accepted step failed the logical check), the
literal text and ND proof for the Lean gate.  `stats` gets timings (gpu_s, check_s, env_s), rejection counters and
the with-replacement repeat probabilities.
"""
import collections, copy, math, os, random, sys, time, json, gzip
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn.functional as F
import record
from sample import _gumbel_pick
from state_env import Env, Frame
import step_check as SC

NAME_BASE_MAX = 32      # as state_sample.py: canonical names start at n<base+1>, base ~ U[0, 32] per attempt
ARMS = ('plain', 'structural', 'logical')


def _swor_shift(path, forbid):
    """Sampling without replacement over whole segments (UniqueRandomizer, Shi et al. 2020).
    `forbid` holds segments already rejected at this prefix, as (ids, per-token logprobs at the
    sampling temperature); `path` is what this row has written of the current segment. Returns,
    per next token c, the log-factor log(1 - R(path + c)), where R(x) is the probability that the
    model, having written x, goes on to write one of the forbidden segments exactly. Adding it to
    the tempered logits samples the model's distribution with the forbidden segments removed; a
    token that would complete one gets -inf."""
    n = len(path)
    mass = collections.defaultdict(float)
    for ids, lqs in forbid:
        if len(ids) > n and ids[:n] == path:
            mass[ids[n]] += math.exp(sum(lqs[n + 1 :]))
    return {c: math.log(1 - m) if m < 1 - 1e-6 else -math.inf for c, m in mass.items()}


@torch.no_grad()
def decode(model, tok, prompt_ids, forbids, temperature, max_new, seed, stats):
    """One wave: every row writes one action (up to and including <eos>, or `max_new` tokens).  Rows with a non-empty
    forbids[r] never write one of those actions.  Returns per row (ids, tempered log-probs of each sampled token under
    the *unshifted* distribution) -- the form `_swor_shift` needs to forbid it later."""
    dev = next(model.parameters()).device
    B0 = len(prompt_ids)
    L = max(len(p) for p in prompt_ids)
    idx = torch.full((B0, L), tok.pad, dtype=torch.long, device=dev)
    keep = torch.zeros((B0, L), dtype=torch.bool, device=dev)
    for i, p in enumerate(prompt_ids):
        idx[i, L - len(p):] = torch.tensor(p, device=dev)
        keep[i, L - len(p):] = True
    pos = (keep.cumsum(1) - 1).clamp(min=0)
    causal = torch.tril(torch.ones(L, L, dtype=torch.bool, device=dev))
    mask = causal[None, None] & keep[:, None, None, :]
    mask = mask | torch.eye(L, dtype=torch.bool, device=dev)[None, None]
    caches = [{'max': L + max_new} for _ in model.blocks]
    logits = model(idx, pos=pos, mask=mask, caches=caches)[:, -1]
    del mask, causal, idx
    out = torch.full((B0, max_new), tok.pad, dtype=torch.long, device=dev)
    out_lq = torch.zeros((B0, max_new), dtype=torch.float32, device=dev)
    declen = torch.full((B0,), max_new, dtype=torch.long, device=dev)
    keepb = torch.ones((B0, L + max_new), dtype=torch.bool, device=dev)
    keepb[:, :L] = keep
    act = torch.arange(B0, device=dev)
    done = torch.zeros(B0, dtype=torch.bool, device=dev)
    cur_pos = pos[:, -1]
    rgen = torch.Generator(device=dev)
    cons = {r: [] for r in range(B0) if forbids[r]}      # constrained row -> its action so far
    act_py = list(range(B0))
    check_every = 16
    for t in range(max_new):
        scaled = logits.float() / temperature
        lsm = torch.log_softmax(scaled, -1)
        if cons:
            slot = {r: j for j, r in enumerate(act_py)}
            shift = [(slot[r], c, d) for r, p in cons.items() if r in slot
                     for c, d in _swor_shift(p, forbids[r]).items()]
            if shift:
                ii, cc, dd = zip(*shift)
                lsm_s = lsm.clone()
                lsm_s[list(ii), list(cc)] += torch.tensor(dd, device=dev)
            else:
                lsm_s = lsm
        else:
            lsm_s = lsm
        nxt = _gumbel_pick(lsm_s, 1.0, rgen, seed, t, B0, act)     # Gumbel-max on the (shifted) tempered log-probs
        lq = lsm.gather(1, nxt[:, None]).squeeze(1)
        live = ~done
        out[act, t] = torch.where(live, nxt, out[act, t])
        out_lq[act, t] = torch.where(live, lq, out_lq[act, t])
        newfin = live & (nxt == tok.eos)
        declen[act] = torch.where(newfin, torch.full_like(declen[act], t + 1), declen[act])
        if cons:
            slot = {r: j for j, r in enumerate(act_py)}
            js = [slot[r] for r in cons if r in slot]
            rs = [r for r in cons if r in slot]
            vals = nxt[js].tolist() if js else []
            fins = (done[js] | newfin[js]).tolist() if js else []
            for r, v, fin in zip(rs, vals, fins):
                p = cons[r]
                p.append(v)
                if fin or not any(len(ids) > len(p) and ids[:len(p)] == p for ids, _ in forbids[r]):
                    del cons[r]
        done = done | newfin
        nxt = torch.where(done, torch.full_like(nxt, tok.pad), nxt)
        if t % check_every == check_every - 1 or t == max_new - 1:
            n_live = int((~done).sum())
            if n_live == 0:
                break
            if n_live < 0.75 * done.numel():
                sel = (~done).nonzero(as_tuple=True)[0]
                act = act[sel]; nxt = nxt[sel]; cur_pos = cur_pos[sel]; keepb = keepb[sel]
                act_py = act.tolist()
                done = torch.zeros(n_live, dtype=torch.bool, device=dev)
                for c in caches:
                    for key in [key for key, v in c.items() if torch.is_tensor(v)]:
                        nk = c[key][sel].contiguous(); del c[key]; c[key] = nk
        cur_pos = cur_pos + 1
        logits = model(nxt[:, None], pos=cur_pos[:, None], mask=keepb[:, None, None, :L + t + 1], caches=caches)[:, -1]
    dl = declen.tolist()
    o = out.tolist(); q = out_lq.tolist()
    n_tok = sum(dl)
    record.count(gen_tokens=n_tok)
    stats['sampled_tokens'] = stats.get('sampled_tokens', 0) + n_tok
    stats['prefill_tokens'] = stats.get('prefill_tokens', 0) + sum(len(p) for p in prompt_ids)
    return [(o[i][:dl[i]], q[i][:dl[i]]) for i in range(B0)]


def clone(e):
    c = copy.copy(e)
    fs = []
    for f in e.frames:
        g = Frame.__new__(Frame)
        g.kind, g.goal, g.pending, g.ore_right, g.last = f.kind, f.goal, f.pending, f.ore_right, f.last
        g.htoks = list(f.htoks); g.names = dict(f.names)
        fs.append(g)
    c.frames = fs
    c.text = list(e.text); c.hist = list(e.hist)
    return c


class Attempt:
    __slots__ = ('i', 'env', 'forbid', 'rej', 'tokens', 'prefill', 'draws', 'canon', 'flag', 'status', 'causes')

    def __init__(self, i, env):
        self.i = i; self.env = env; self.forbid = []; self.rej = 0; self.tokens = 0; self.prefill = 0
        self.draws = 0; self.canon = True; self.flag = None; self.status = None; self.causes = []


@torch.no_grad()
def guided_generate(model, tok, prompts, arm, temperature=0.8, max_action=512, max_steps=96, max_rej=10,
                    batch=2048, seed=0, stats=None, dump=None, dump_cap=60000):
    assert arm in ARMS
    dev = next(model.parameters()).device
    st = stats if stats is not None else {}
    rej_cause = st.setdefault('rej_cause', collections.Counter())       # every rejected draw, by cause
    end_cnt = st.setdefault('end', collections.Counter())
    flag_cnt = st.setdefault('flag_cause', collections.Counter())       # logical failures that went through (plain/structural)
    rep = st.setdefault('repeat_p', [])          # per redraw: P(a with-replacement draw repeats a rejected action)
    rep1 = st.setdefault('reject_p', [])         # per rejected ended action: its own tempered probability
    cache = {}
    st.setdefault('cache_hits', 0); st.setdefault('cache_miss', 0)
    for k in ('gpu_s', 'check_s', 'env_s'):
        st.setdefault(k, 0.0)
    dumped = set()
    res = [None] * len(prompts)
    live, nxt, wave = [], 0, 0
    record.count(attempts=len(prompts))
    t_start = time.time()

    def finish(a, status):
        a.status = status
        end_cnt[status] += 1
        e = a.env
        nd = e.nd() if status == 'done' else 'LEANPARSE env: ' + status
        res[a.i] = dict(i=a.i, status=status, tokens=a.tokens, prefill=a.prefill, draws=a.draws, rej=a.rej,
                        causes=a.causes, flag=a.flag, steps=e.steps, nd=nd,
                        text=tok.text(e.text) if status == 'done' and not nd.startswith('LEANPARSE') else None)

    while nxt < len(prompts) or live:
        while len(live) < batch and nxt < len(prompts):
            base = random.Random(seed * 1000003 + nxt).randint(0, NAME_BASE_MAX)
            live.append(Attempt(nxt, Env(prompts[nxt], canon=True, base=base, assign=True))); nxt += 1
        pids = [tok.encode_toks(a.env.state_tokens()) for a in live]
        forbids = [a.forbid for a in live]
        record.count(actions=len(live))
        if dev.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.time()
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=(dev.type == 'cuda')):
            outs = decode(model, tok, pids, forbids, temperature, max_action, seed * 100003 + wave, st)
        if dev.type == 'cuda':
            torch.cuda.synchronize()
        st['gpu_s'] += time.time() - t0
        wave += 1
        keep = []
        for a, p, (ids, lqs) in zip(live, pids, outs):
            a.draws += 1; a.tokens += len(ids); a.prefill += len(p)
            if a.forbid:                              # this draw was a redraw: what would replacement have cost
                rep.append(sum(math.exp(sum(q)) for _, q in a.forbid))
            ended = bool(ids) and ids[-1] == tok.eos
            atoks = [tok.itos[x] for x in ids[:-1]] if ended else None
            cause = None
            t1 = time.time()
            if not ended:
                cause = 'truncated'
            else:
                e2 = clone(a.env)
                ok, why = e2.apply(atoks)
                if not ok:
                    cause = 'S:' + why
            st['env_s'] += time.time() - t1
            lcause = None
            if cause is None and a.canon:
                t1 = time.time()
                key = (tuple(p), tuple(atoks))
                if key in cache:
                    lcause = cache[key]; st['cache_hits'] += 1
                else:
                    lcause = SC.check(a.env, atoks); cache[key] = lcause; st['cache_miss'] += 1
                    if dump is not None and len(dumped) < dump_cap and atoks[0] == 'have' and key not in dumped:
                        t = atoks[atoks.index(':=') + 1] if ':=' in atoks else None
                        if t not in ('(', 'Or.elim') and SC.canonical(atoks) and (lcause or random.random() < 0.25):
                            dumped.add(key)
                            dump.write(json.dumps(dict(prompt=prompts[a.i], src=SC.lean_source(a.env, atoks, tok),
                                                       verdict=lcause)) + '\n')
                st['check_s'] += time.time() - t1
                if lcause and arm == 'logical':
                    cause = 'L:' + lcause
            if cause is not None:
                rej_cause[cause] += 1
                a.causes.append(cause)
                if ended:
                    rep1.append(math.exp(sum(lqs)))
                if arm == 'plain' or a.rej >= max_rej:
                    finish(a, 'fail:' + (cause if arm == 'plain' else 'max_rej')); continue
                a.rej += 1
                if ended:
                    a.forbid = a.forbid + [(ids, lqs)]
                keep.append(a); continue
            if lcause:                                # a logical failure the arm lets through
                flag_cnt[lcause] += 1
                if a.flag is None:
                    a.flag = lcause
            a.env = e2; a.forbid = []
            if a.canon and not SC.canonical(atoks):
                a.canon = False
                st['noncanon_attempts'] = st.get('noncanon_attempts', 0) + 1
            if e2.done:
                finish(a, 'done'); continue
            if e2.steps >= max_steps:
                finish(a, 'fail:step_cap'); continue
            keep.append(a)
        live = keep
        del outs
    st['waves'] = st.get('waves', 0) + wave
    st['loop_wall_s'] = st.get('loop_wall_s', 0.0) + time.time() - t_start
    if dev.type == 'cuda':
        st['peak_alloc_gb'] = max(st.get('peak_alloc_gb', 0), torch.cuda.max_memory_allocated() / 2 ** 30)
    return res
