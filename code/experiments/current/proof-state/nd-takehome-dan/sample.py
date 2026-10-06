"""Batched greedy / temperature sampling with a KV cache. Left-pads prompts.

  generate(model, tok, prompts, greedy=True, temperature=1.0, max_new=400, batch=512, seed=0)
    -> list of proof-body strings (decoded, spec format), one per prompt.

Run `efficiency` (2026-09-23) added a fast decode path.  `path='base'` is the code as it stood before that run,
byte for byte; `path='fast'` is the default and adds, each separable and reversible:

  early='eos'   (default) only <eos> ends a row (what 'base' does)
  early='exact' + a row ends as soon as it has emitted a top-level `exact n<k>` (paren depth 0): the Lean term is
                syntactically complete, so nothing the model writes afterwards can be part of an accepted proof
  early='goal'  + a row ends as soon as a depth-0 `have n<k> : <the theorem's conclusion> := … ;` closes:
                the sampler appends `exact n<k>` itself and stops.  A guess costs a rejected sample, never a wrong
                accept, because lean_check still checks the text.
Measured on run efficiency's workload, the `lean_seq` Stage-1 model emits <eos> on 99.99 % of rows and emits it
immediately after its top-level `exact`, so 'exact' and 'goal' save 1 and 3 decoded tokens per sample and cost 8 %
and 28 % of sampler wall time in extra per-step kernels.  **The default is therefore 'eos'.**  They are kept
because a model that does *not* terminate (the pretrained-model path) pays nothing for <eos> that never comes.
  compact=True  finished rows are dropped from the decode batch (KV cache included), so they stop costing compute
                and memory.  Compaction runs when the live fraction falls below `compact_frac`.
  rowrng=True   sampling noise is keyed by (chunk seed, step, slot) instead of drawn batch-wide, so a row's token
                stream does not depend on how the batch is packed.  This is what makes 'base' and 'fast' exactly
                comparable; distributionally it is the same sampler (Gumbel-max on the temperature-scaled logits).

Environment overrides (so older drivers inherit the fast path without a code change):
  ND_SAMPLE_PATH=base|fast   ND_SAMPLE_EARLY=eos|exact|goal   ND_SAMPLE_COMPACT=0|1   ND_SAMPLE_ROWRNG=0|1
`stats` (a dict passed to generate) is filled with decoded-token and timing counters.
"""
import os, time, torch, torch.nn.functional as F
from model import rope_cache

PATH = os.environ.get('ND_SAMPLE_PATH', 'fast')
EARLY = os.environ.get('ND_SAMPLE_EARLY', 'eos')
COMPACT = os.environ.get('ND_SAMPLE_COMPACT', '1') == '1'
ROWRNG = os.environ.get('ND_SAMPLE_ROWRNG', '1') == '1'


# ----------------------------------------------------------------- the pre-efficiency path (unchanged)
@torch.no_grad()
def generate_ids(model, tok, prompt_ids, greedy=True, temperature=1.0, max_new=400, gen=None,
                 rowrng_seed=None, stats=None):
    dev = next(model.parameters()).device
    B = len(prompt_ids)
    L = max(len(p) for p in prompt_ids)
    idx = torch.full((B, L), tok.pad, dtype=torch.long, device=dev)
    keep = torch.zeros((B, L), dtype=torch.bool, device=dev)
    for i, p in enumerate(prompt_ids):
        idx[i, L - len(p):] = torch.tensor(p, device=dev)
        keep[i, L - len(p):] = True
    pos = (keep.cumsum(1) - 1).clamp(min=0)
    # prefill: causal + key padding mask
    causal = torch.tril(torch.ones(L, L, dtype=torch.bool, device=dev))
    mask = causal[None, None] & keep[:, None, None, :]
    mask = mask | torch.eye(L, dtype=torch.bool, device=dev)[None, None]  # pads attend to self (avoid NaN)
    caches = [{'max': L + max_new} for _ in model.blocks]
    logits = model(idx, pos=pos, mask=mask, caches=caches)[:, -1]
    out = torch.full((B, max_new), tok.pad, dtype=torch.long, device=dev)
    done = torch.zeros(B, dtype=torch.bool, device=dev)
    declen = torch.full((B,), max_new, dtype=torch.long, device=dev)
    rgen = None
    if rowrng_seed is not None and not greedy:
        rgen = torch.Generator(device=dev)
    cur_pos = pos[:, -1]
    steps = 0
    for t in range(max_new):
        if greedy:
            nxt = logits.argmax(-1)
        elif rgen is not None:
            nxt = _gumbel_pick(logits, temperature, rgen, rowrng_seed, t, B, torch.arange(B, device=dev))
        else:
            probs = F.softmax(logits.float() / temperature, -1)
            nxt = torch.multinomial(probs, 1, generator=gen).squeeze(1)
        nxt = torch.where(done, torch.full_like(nxt, tok.pad), nxt)
        out[:, t] = nxt
        fin = (~done) & (nxt == tok.eos)
        declen = torch.where(fin, torch.full_like(declen, t + 1), declen)
        done = done | (nxt == tok.eos)
        steps = t + 1
        if bool(done.all()):
            break
        cur_pos = cur_pos + 1
        keep = torch.cat([keep, torch.ones(B, 1, dtype=torch.bool, device=dev)], 1)
        mask = keep[:, None, None, :]
        logits = model(nxt[:, None], pos=cur_pos[:, None], mask=mask, caches=caches)[:, -1]
    if stats is not None:
        _acc(stats, declen, steps, B, steps * B, L)
    return out.tolist()


# ----------------------------------------------------------------- shared helpers
def _gumbel_pick(logits, temperature, rgen, seed, t, n_slots, slots):
    """Gumbel-max sampling with noise keyed by (seed, step, slot): the draw of a row does not depend on which other
    rows share its batch, so dropping finished rows cannot change any surviving row's tokens."""
    rgen.manual_seed((seed * 1000003 + t) % (2 ** 63 - 1))
    u = torch.rand((n_slots, logits.shape[-1]), device=logits.device, generator=rgen).clamp_(1e-9, 1 - 1e-9)
    g = -torch.log(-torch.log(u))
    return (logits.float() / temperature + g[slots]).argmax(-1)


def _acc(stats, declen, steps, b0, rowsteps, prompt_len):
    stats.setdefault('declen', []).extend(declen.tolist())
    stats['steps'] = stats.get('steps', 0) + steps
    stats['rows'] = stats.get('rows', 0) + b0
    stats['rowsteps'] = stats.get('rowsteps', 0) + int(rowsteps)
    stats['prompt_len_max'] = max(stats.get('prompt_len_max', 0), prompt_len)
    stats['chunks'] = stats.get('chunks', 0) + 1


def goal_token_ids(tok, prompt):
    """The token ids of the theorem's conclusion formula, for the `goal` early stop.  None if the tokenizer is not one
    of the Lean ones."""
    try:
        from nd2lean import parse_prompt
        from lean_tok import ftoks
        _, concl = parse_prompt(prompt)
        return [tok.stoi[s] for s in ftoks(concl)]
    except Exception:
        return None


def _grammar_ids(tok):
    need = ['have', ':', ':=', ';', 'exact', '(', ')']
    try:
        return {s: tok.stoi[s] for s in need}
    except Exception:
        return None


# ----------------------------------------------------------------- the fast path
@torch.no_grad()
def generate_ids_fast(model, tok, prompt_ids, goals=None, greedy=True, temperature=1.0, max_new=400,
                      seed=0, early='goal', compact=True, compact_frac=0.75, stats=None):
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
    mask = mask | torch.eye(L, dtype=torch.bool, device=dev)[None, None]   # pads attend to self (avoid NaN)
    caches = [{'max': L + max_new} for _ in model.blocks]
    logits = model(idx, pos=pos, mask=mask, caches=caches)[:, -1]
    del mask, causal, idx

    out = torch.full((B0, max_new), tok.pad, dtype=torch.long, device=dev)
    declen = torch.full((B0,), max_new, dtype=torch.long, device=dev)
    keepb = torch.ones((B0, L + max_new), dtype=torch.bool, device=dev)   # preallocated: no per-step torch.cat
    keepb[:, :L] = keep
    act = torch.arange(B0, device=dev)              # act[slot] -> row of this chunk (= the row's rng slot)
    done = torch.zeros(B0, dtype=torch.bool, device=dev)   # over the current slots
    cur_pos = pos[:, -1]
    rgen = torch.Generator(device=dev) if not greedy else None

    G = _grammar_ids(tok) if early in ('exact', 'goal') else None
    use_goal = early == 'goal' and G is not None and goals is not None and all(goals)
    if G is not None:
        HAVE, COLON, ASSIGN, SEMI, EXACT, LP, RP = (G['have'], G[':'], G[':='], G[';'], G['exact'], G['('], G[')'])
        ref0 = getattr(tok, 'ref0', 10 ** 9)
        depth = torch.zeros(B0, dtype=torch.int32, device=dev)          # per row, paren depth so far
        prev = torch.full((B0,), -1, dtype=torch.long, device=dev)
    if use_goal:
        gmax = max(len(g) for g in goals)
        gids = torch.full((B0, gmax + 1), -1, dtype=torch.long, device=dev)
        for i, g in enumerate(goals):
            gids[i, :len(g)] = torch.tensor(g, device=dev)
        glen = torch.tensor([len(g) for g in goals], dtype=torch.long, device=dev)
        st = torch.zeros(B0, dtype=torch.int8, device=dev)              # `have n : GOAL := … ;` automaton
        mpos = torch.zeros(B0, dtype=torch.long, device=dev)
        nm = torch.zeros(B0, dtype=torch.long, device=dev)

    rowsteps = 0
    steps = 0
    pad = tok.pad
    n_ex = torch.zeros((), dtype=torch.long, device=dev)
    n_go = torch.zeros((), dtype=torch.long, device=dev)
    n_eo = torch.zeros((), dtype=torch.long, device=dev)
    check_every = 16          # the only host<->device sync in the loop; the base path syncs on `done.all()` every step
    for t in range(max_new):
        rowsteps += act.numel()
        steps = t + 1
        nxt = logits.argmax(-1) if greedy else _gumbel_pick(logits, temperature, rgen, seed, t, B0, act)
        live = ~done
        keepv = out[act, t]                      # a terminator this row was given earlier, or pad
        out[act, t] = torch.where(live, nxt, keepv)
        newfin = live & (nxt == tok.eos)
        n_eo += newfin.sum()

        if G is not None:
            d0 = depth[act]
            depth[act] = torch.where(live, d0 + (nxt == LP).int() - (nxt == RP).int(), d0)
            isname = nxt >= ref0
            exdone = live & (prev[act] == EXACT) & (d0 == 0) & isname & ~newfin
            if t + 1 < max_new:
                out[act, t + 1] = torch.where(exdone, torch.full_like(nxt, tok.eos), out[act, t + 1])
                n_ex += exdone.sum()
                newfin = newfin | exdone
            if use_goal:
                s = st[act]; mp = mpos[act]
                gcur = gids[act, mp.clamp(max=gmax)]
                s1 = (s == 1) & isname
                s2 = (s == 2) & (nxt == COLON)
                s3 = (s == 3) & (mp < glen[act]) & (nxt == gcur)
                s3d = s3 & ((mp + 1) == glen[act])
                s4 = (s == 4) & (nxt == ASSIGN)
                endhave = (nxt == SEMI) & (d0 == 0)        # a `;` inside a box is at depth >= 1
                fire = live & (s == 5) & endhave & ~newfin
                s5c = (s == 5) & ~endhave
                start = (nxt == HAVE) & (d0 == 0)
                z = torch.zeros_like(s)
                sn = torch.where(s1, z + 2, z)
                sn = torch.where(s2, z + 3, sn)
                sn = torch.where(s3 & ~s3d, z + 3, sn)
                sn = torch.where(s3d, z + 4, sn)
                sn = torch.where(s4 | s5c, z + 5, sn)
                sn = torch.where((sn == 0) & start, z + 1, sn)
                st[act] = torch.where(live, sn, s)
                mpos[act] = torch.where(live, torch.where(s3, mp + 1, torch.zeros_like(mp)), mp)
                nm[act] = torch.where(s1 & live, nxt, nm[act])
                if t + 3 < max_new:
                    out[act, t + 1] = torch.where(fire, torch.full_like(nxt, EXACT), out[act, t + 1])
                    out[act, t + 2] = torch.where(fire, nm[act], out[act, t + 2])
                    out[act, t + 3] = torch.where(fire, torch.full_like(nxt, tok.eos), out[act, t + 3])
                    n_go += fire.sum()
                    newfin = newfin | fire
                else:
                    fire = torch.zeros_like(newfin)
            prev[act] = torch.where(live, nxt, prev[act])

        declen[act] = torch.where(newfin, torch.full_like(declen[act], t + 1), declen[act])
        done = done | newfin
        nxt = torch.where(done, torch.full_like(nxt, pad), nxt)
        if t % check_every == check_every - 1 or t == max_new - 1:
            n_live = int((~done).sum())          # sync
            if n_live == 0:
                break
            if compact and n_live < compact_frac * done.numel():
                sel = (~done).nonzero(as_tuple=True)[0]
                act = act[sel]; nxt = nxt[sel]; cur_pos = cur_pos[sel]; keepb = keepb[sel]
                done = torch.zeros(n_live, dtype=torch.bool, device=dev)
                for c in caches:
                    if 'k' in c:
                        nk = c['k'][sel].contiguous(); del c['k']; c['k'] = nk
                        nv = c['v'][sel].contiguous(); del c['v']; c['v'] = nv
        cur_pos = cur_pos + 1
        logits = model(nxt[:, None], pos=cur_pos[:, None], mask=keepb[:, None, None, :L + t + 1], caches=caches)[:, -1]
    n_eos, n_exact, n_goal = int(n_eo), int(n_ex), int(n_go)
    if stats is not None:
        _acc(stats, declen, steps, B0, rowsteps, L)
        stats['stop_eos'] = stats.get('stop_eos', 0) + n_eos
        stats['stop_exact'] = stats.get('stop_exact', 0) + n_exact
        stats['stop_goal'] = stats.get('stop_goal', 0) + n_goal
    return out.tolist()


def generate(model, tok, prompts, greedy=True, temperature=1.0, max_new=400, batch=512, seed=0,
             path=None, early=None, compact=None, rowrng=None, stats=None, gate=True, raw=None):
    dev = next(model.parameters()).device
    path = PATH if path is None else path
    early = EARLY if early is None else early
    compact = COMPACT if compact is None else compact
    rowrng = ROWRNG if rowrng is None else rowrng
    gen = None
    if not greedy and path == 'base' and not rowrng:
        gen = torch.Generator(device=dev)
        gen.manual_seed(seed)
    ids = [tok.encode_prompt(p) for p in prompts]
    order = sorted(range(len(prompts)), key=lambda i: len(ids[i]))
    res = [None] * len(prompts)
    is_lean = hasattr(tok, 'statement')      # LeanTokenizer: decode() returns the denoted ND proof, last_text the literal Lean text
    texts = [None] * len(prompts)
    goals = [goal_token_ids(tok, p) for p in prompts] if (is_lean and path == 'fast' and early == 'goal') else None
    t0 = time.time()
    for ci, s in enumerate(range(0, len(order), batch)):
        chunk = order[s:s + batch]
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=(dev.type == 'cuda')):
            if path == 'fast':
                outs = generate_ids_fast(model, tok, [ids[i] for i in chunk],
                                         goals=[goals[i] for i in chunk] if goals else None,
                                         greedy=greedy, temperature=temperature, max_new=max_new,
                                         seed=seed + ci, early=early, compact=compact, stats=stats)
            else:
                outs = generate_ids(model, tok, [ids[i] for i in chunk], greedy, temperature, max_new, gen,
                                    rowrng_seed=(seed + ci) if rowrng else None, stats=stats)
        if stats is not None:
            dl = stats.get('declen', [])
            n0 = len(dl) - len(chunk)
            dbp = stats.setdefault('declen_by_prompt', [0] * len(prompts))
            for j, i in enumerate(chunk):
                dbp[i] = dl[n0 + j]
        for j, (i, o) in enumerate(zip(chunk, outs)):
            res[i] = tok.decode(o)
            if is_lean:
                texts[i] = tok.last_text
            if raw is not None:
                raw[i, :len(o)] = o
        del outs
    if stats is not None:
        stats['sample_wall_s'] = stats.get('sample_wall_s', 0.0) + time.time() - t0
        if dev.type == 'cuda':
            stats['peak_alloc_gb'] = torch.cuda.max_memory_allocated() / 2 ** 30
            stats['peak_reserved_gb'] = torch.cuda.max_memory_reserved() / 2 ** 30
    if dev.type == 'cuda':
        torch.cuda.empty_cache()   # hand reserved memory back to co-tenant jobs
    if is_lean and gate:
        from lean_gate import gate as _gate
        res = _gate(tok, prompts, res, texts)   # Lean alone decides (2026-09-27): accepted samples come back as CLEAN ND strings, rejects marked 'LEANREJ '
    elif is_lean:
        res = texts            # gate=False: literal Lean texts, for render/decode diagnostics only -- NOT judgeable
    return res
