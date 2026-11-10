"""Batched sampling in Dan's step environment, with optional gating of each step.

`gen_actions` decodes one action per state (KV cache, left-padded batch, rows compacted as they
finish) and returns the verifier and value logits alongside.

`rollouts` runs whole attempts. `gate`:
- "none": every sampled action goes to the environment; an action it refuses ends the try.
- "checker": training time. A step whose term has a certain Lean type error (Dan's pre-filter on
  that one step) is not applied; it is resampled. Each such refusal is kept as a verifier negative.
- "verifier": test time. The same, but the refusal comes from the model's own verifier head
  (probability below `tau`). No type check and no Lean are called.
In every mode an action the environment refuses (bad syntax, unbound name, `exact` that does not
match the goal) may be resampled up to `resample` times, and a failed try may be restarted up to
`tries - 1` times; all tokens generated on the way are counted in the attempt's `n_tok`.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np
import torch

from .common import EOS, PAD, TOK, Env, enc, ftoks, step_kind, try_step

ITOS = TOK.itos
GEN_INIT = 48  # KV-cache slots preallocated for the action; the cache grows if an action is longer
GEN_RESERVE = 96  # tokens per row assumed for the action when sizing a decode chunk


def _longest_line(state) -> int:
    """Tokens in the longest hypothesis line or goal of a state. An action restates at most two
    such formulas (plus a short term), so `3 * longest + 40` tokens bound any sensible action;
    a sample that runs past that is cut off there (and counted) instead of at the global cap."""
    best = cur = 0
    for t in state:
        if t == "<nl>" or t == "⊢":
            best = max(best, cur)
            cur = 0
        else:
            cur += 1
    return max(best, cur)


def _sig(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, x))))


@torch.no_grad()
def gen_actions(model, pids, temperature=1.0, max_new=256, g=None, tok_budget=700_000,
                max_rows=4096, temps=None, caps=None,
):  # fmt: skip
    """pids: list of state id lists -> (list of action id lists without <eos>, ended flags,
    verifier logits, value logits), in input order."""
    dev = next(model.parameters()).device
    cuda = dev.type == "cuda"
    n = len(pids)
    order = sorted(range(n), key=lambda i: len(pids[i]))
    acts = [None] * n
    ended = [False] * n
    ver = [0.0] * n
    val = [0.0] * n
    i0 = 0
    while i0 < n:
        i1 = i0 + 1
        while (i1 < n and i1 - i0 < max_rows
               and (i1 - i0 + 1) * (len(pids[order[i1]]) + min(max_new, GEN_RESERVE)) <= tok_budget):  # fmt: skip
            i1 += 1
        chunk = order[i0:i1]
        i0 = i1
        B = len(chunk)
        lens = torch.tensor([len(pids[i]) for i in chunk], device=dev)
        temps_c = None if temps is None else torch.tensor([temps[i] for i in chunk], device=dev)
        cap_c = torch.tensor([max_new if caps is None else min(max_new, caps[i]) for i in chunk], device=dev)
        W = int(cap_c.max())
        T = int(lens.max())
        arr = np.full((B, T), PAD, np.uint8)
        for r, i in enumerate(chunk):
            arr[r, T - len(pids[i]) :] = np.frombuffer(bytes(pids[i]), np.uint8)
        idx = torch.from_numpy(arr).to(dev).long()
        ar = torch.arange(T, device=dev)
        off = (T - lens)[:, None]
        pos = (ar[None] - off).clamp_(min=0)
        valid = ar[None] >= off
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=dev))[None] & valid[:, None, :]
        mask = (mask | torch.eye(T, dtype=torch.bool, device=dev)[None])[:, None]
        dt = torch.bfloat16 if cuda else torch.float32
        caches = model.new_caches(B, T + min(max_new, GEN_INIT), dt)
        keyvalid = torch.cat([valid, torch.ones(B, W, dtype=torch.bool, device=dev)], 1)
        toks = torch.full((B, W), PAD, dtype=torch.long, device=dev)
        ver_c = torch.zeros(B, device=dev)
        end_c = torch.zeros(B, dtype=torch.bool, device=dev)
        rows = torch.arange(B, device=dev)
        fin = torch.zeros(B, dtype=torch.bool, device=dev)
        cur_len = lens.clone()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cuda):
            h = model.trunk(idx, pos, mask, caches)[:, -1]
            val_c = model.aux(h)[:, 1].float()
            for t in range(W):
                logits = model.head(h).float()
                if temps is None and temperature <= 0:
                    nxt = logits.argmax(-1)
                else:
                    u = torch.rand(logits.shape, generator=g, device=dev).clamp_(1e-9, 1 - 1e-9)
                    tt = temperature if temps is None else temps_c[rows][:, None]
                    nxt = (logits / tt - torch.log(-torch.log(u))).argmax(-1)
                v_now = model.aux(h)[:, 0].float()
                is_eos = ~fin & (nxt == EOS)
                toks[rows, t] = torch.where(fin, toks[rows, t], nxt)
                ver_c[rows] = torch.where(is_eos, v_now, ver_c[rows])
                end_c[rows] = end_c[rows] | is_eos
                fin = fin | is_eos | (t + 1 >= cap_c[rows])  # a row at its cap stops, not ended
                n_live = int((~fin).sum())
                if n_live == 0 or t == W - 1:
                    break
                if n_live <= 0.7 * fin.numel() and (t % 4 == 3 or n_live <= 0.3 * fin.numel()):
                    keep = ~fin
                    rows, nxt, cur_len = rows[keep], nxt[keep], cur_len[keep]
                    keyvalid = keyvalid[keep]
                    for c in caches:
                        c["k"] = c["k"][keep]
                        c["v"] = c["v"][keep]
                    fin = fin[keep]
                p = (cur_len + t)[:, None]
                m = keyvalid[:, None, None, : T + t + 1]
                h = model.trunk(nxt[:, None], p, m, caches)[:, 0]
        toks = toks.tolist()
        ver_l, val_l, end_l, cap_l = ver_c.tolist(), val_c.tolist(), end_c.tolist(), cap_c.tolist()
        for r, i in enumerate(chunk):
            row = toks[r]
            row = row[: row.index(EOS)] if end_l[r] else row[: cap_l[r]]
            acts[i] = row
            ended[i] = end_l[r]
            ver[i] = ver_l[r]
            val[i] = val_l[r]
        del caches
    return acts, ended, ver, val


@dataclass
class Attempt:
    prompt: str
    tag: object = None  # caller's label (theorem index, draw)
    base: int = 0
    acts: list = field(default_factory=list)  # applied (effective) actions of the current try
    env: Env | None = None
    n_tok: int = 0  # tokens generated, all tries, refused samples included
    n_calls: int = 0  # actions sampled
    tries: int = 0
    step_tries: int = 0
    done: bool = False
    why: str = ""
    min_ver: float = 1.0  # lowest verifier probability among the applied atomic steps
    open_stack: list = field(default_factory=list)  # indices in `acts` of the open boxes' openers
    pops: int = 0  # boxes abandoned in the current try
    n_scaffold: int = 0  # premise lines written by the solver (final try)
    pvs: list = field(default_factory=list)  # verifier probability of each applied step
    fb: bool = False  # the current try is the unscreened fallback try
    fb_calls: int = 0
    val0: float = 0.0  # value probability at the first state


def rollouts(model, attempts, *, gate="none", temperature=1.0, resample=0, tries=1, tau=0.5,
             max_steps=96, max_action=256, seed=0, base_max=8, collect=None, tok_budget=700_000,
             log=None, temp_bump=0.0, backtrack=0, max_calls=1200, stats=None, scaffold=False,
             verifier=None, fallback=False,
):  # fmt: skip
    """Run every Attempt to its end. `collect`, if a list, receives (state bytes, action bytes)
    for each step the checker refused (gate="checker"). `backtrack`: when a try gets stuck inside
    a box, abandon the innermost open box (return to the state before its opening step) instead
    of ending the try, up to this many times per try. `max_calls` bounds the actions sampled in
    one attempt over all tries. `scaffold`: the premise-restating lines (line k restates premise
    k; a function of the statement alone, the format's fixed preamble) are written by the solver
    instead of sampled; they cost no generated tokens and are counted in `Attempt.n_scaffold`.
    `fallback` (gate="verifier"): when every screened try has failed, one last try runs without
    the verifier, so the attempt still hands in a proof; `Attempt.fb` marks it."""
    assert gate in ("none", "checker", "verifier")
    dev = next(model.parameters()).device
    g = torch.Generator(device=dev)
    g.manual_seed(seed)
    rng = random.Random(seed)

    def start(a: Attempt):
        a.base = rng.randint(0, base_max)
        a.env = Env(a.prompt, canon=True, base=a.base, assign=True)
        a.acts = []
        a.step_tries = 0
        a.min_ver = 1.0
        a.open_stack = []
        a.pops = 0
        a.pvs = []
        a.why = ""
        a.tries += 1
        if scaffold:
            a.n_scaffold = 0
            for k, f in enumerate(a.env.prem):
                ok, _ = a.env.apply(["have", "n1", ":"] + ftoks(f) + [":=", f"h{k + 1}", ";"])
                if not ok:
                    break
                a.acts.append(list(a.env._eff))
                a.pvs.append(1.0)
                a.n_scaffold += 1

    for a in attempts:
        start(a)
    live = list(attempts)
    wave = 0
    while live:
        states = [a.env.state_tokens() for a in live]
        pids = [enc(s) for s in states]
        tl = None
        if temp_bump > 0:
            tl = [temperature * min(1.6, 1.0 + temp_bump * a.step_tries) for a in live]
        caps = [min(max_action, 3 * _longest_line(s) + 40) for s in states]
        outs, ended, ver, val = gen_actions(model, pids, temperature, max_action, g, tok_budget, temps=tl,
                                            caps=caps)  # fmt: skip
        keep = []
        pv_sep = {}
        if gate == "verifier" and verifier is not None:  # the separate verifier network
            from .verifier import compact, ver_probs

            vi, vs = [], []
            for r, (a, st, ids, end) in enumerate(zip(live, states, outs, ended)):
                if not end or a.fb:
                    continue
                act = [ITOS[i] for i in ids]
                if step_kind(act) not in ("atom", "pr"):
                    continue
                try:  # judged with the name the environment will give the line, as in training
                    c = compact(st, a.env._assign_names(list(act), act.index(":=") + 1))
                except Exception:  # noqa: BLE001
                    c = None
                if c is not None:
                    vi.append(r)
                    vs.append(enc(c))
            pv_sep = dict(zip(vi, ver_probs(verifier, vs)))
        for r, (a, st, pid, ids, end, vl, v0) in enumerate(zip(live, states, pids, outs, ended, ver, val)):
            a.n_tok += len(ids) + (1 if end else 0)
            a.n_calls += 1
            a.fb_calls += a.fb
            if not a.acts and a.step_tries == 0:
                a.val0 = _sig(v0)
            out = "syntax"
            pv = 1.0
            nf0 = len(a.env.frames)
            if end:
                act = [ITOS[i] for i in ids]
                kind = step_kind(act)
                if gate == "verifier" and kind in ("atom", "pr") and not a.fb:
                    pv = _sig(vl) if verifier is None else pv_sep.get(r, 1.0)
                    out = "invalid" if pv < tau else try_step(a.env, act, check=False)
                else:
                    out = try_step(a.env, act, check=(gate == "checker"))
                if out == "invalid" and collect is not None:
                    try:
                        eff = a.env._assign_names(list(act), act.index(":=") + 1)
                        collect.append((pid, enc(eff)))
                    except Exception:
                        pass
            if stats is not None:
                if not end:
                    key = "truncated"
                elif out == "syntax":
                    key = "syntax:" + str(getattr(a.env, "last_why", "?") if step_kind(act) != "bad" else "bad action")
                else:
                    key = out
                stats[key] = stats.get(key, 0) + 1
            if out == "ok":
                nf1 = len(a.env.frames)
                if nf1 > nf0:
                    a.open_stack.append(len(a.acts))
                elif nf1 < nf0:
                    a.open_stack.pop()
                a.acts.append(list(a.env._eff))
                a.pvs.append(pv)
                a.step_tries = 0
                a.min_ver = min(a.min_ver, pv)
                if a.env.done:
                    a.done = True
                    continue
                if a.env.steps < max_steps:
                    keep.append(a)
                    continue
                a.why = "step cap"
            else:
                a.step_tries += 1
                capped = a.fb_calls >= 2 * max_steps if a.fb else a.n_calls >= max_calls
                if not capped and a.step_tries <= resample:
                    keep.append(a)
                    continue
                a.why = "call cap" if capped else (out if end else "truncated")
                if not capped and a.open_stack and a.pops < backtrack:
                    k = a.open_stack.pop()
                    env = Env(a.prompt, canon=True, base=a.base, assign=True)
                    good = all(env.apply(list(x))[0] for x in a.acts[:k])
                    if good:
                        a.env, a.acts, a.pvs = env, a.acts[:k], a.pvs[:k]
                        a.min_ver = min(a.pvs, default=1.0)
                        a.why = ""
                        a.pops += 1
                        a.step_tries = 1  # the next sample at this state is drawn warmer
                        keep.append(a)
                        continue
            if not a.fb and a.n_calls < max_calls and a.tries < tries:
                start(a)
                keep.append(a)
            elif fallback and gate == "verifier" and not a.fb:
                a.fb = True
                a.fb_calls = 0
                start(a)
                keep.append(a)
        live = keep
        wave += 1
        if log is not None and wave % 10 == 0:
            log(f"  wave {wave}: live {len(live)}")
    return attempts
