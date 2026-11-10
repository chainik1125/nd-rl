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

import torch

from .common import EOS, PAD, TOK, Env, enc, step_kind, try_step

ITOS = TOK.itos


def _sig(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, x))))


@torch.no_grad()
def gen_actions(model, pids, temperature=1.0, max_new=256, g=None, tok_budget=700_000,
                max_rows=4096):  # fmt: skip
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
               and (i1 - i0 + 1) * (len(pids[order[i1]]) + max_new) <= tok_budget):  # fmt: skip
            i1 += 1
        chunk = order[i0:i1]
        i0 = i1
        B = len(chunk)
        lens = torch.tensor([len(pids[i]) for i in chunk], device=dev)
        T = int(lens.max())
        idx = torch.full((B, T), PAD, dtype=torch.long)
        for r, i in enumerate(chunk):
            idx[r, T - len(pids[i]) :] = torch.tensor(list(pids[i]), dtype=torch.long)
        idx = idx.to(dev)
        ar = torch.arange(T, device=dev)
        off = (T - lens)[:, None]
        pos = (ar[None] - off).clamp_(min=0)
        valid = ar[None] >= off
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=dev))[None] & valid[:, None, :]
        mask = (mask | torch.eye(T, dtype=torch.bool, device=dev)[None])[:, None]
        dt = torch.bfloat16 if cuda else torch.float32
        caches = model.new_caches(B, T + max_new, dt)
        keyvalid = torch.cat([valid, torch.ones(B, max_new, dtype=torch.bool, device=dev)], 1)
        toks = torch.full((B, max_new), PAD, dtype=torch.long, device=dev)
        ver_c = torch.zeros(B, device=dev)
        end_c = torch.zeros(B, dtype=torch.bool, device=dev)
        rows = torch.arange(B, device=dev)
        fin = torch.zeros(B, dtype=torch.bool, device=dev)
        cur_len = lens.clone()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cuda):
            h = model.trunk(idx, pos, mask, caches)[:, -1]
            val_c = model.aux(h)[:, 1].float()
            for t in range(max_new):
                logits = model.head(h).float()
                if temperature <= 0:
                    nxt = logits.argmax(-1)
                else:
                    u = torch.rand(logits.shape, generator=g, device=dev).clamp_(1e-9, 1 - 1e-9)
                    nxt = (logits / temperature - torch.log(-torch.log(u))).argmax(-1)
                v_now = model.aux(h)[:, 0].float()
                live = ~fin
                toks[rows[live], t] = nxt[live]
                is_eos = live & (nxt == EOS)
                ver_c[rows[is_eos]] = v_now[is_eos]
                end_c[rows[is_eos]] = True
                fin = fin | is_eos
                n_live = int((~fin).sum())
                if n_live == 0 or t == max_new - 1:
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
        ver_l, val_l, end_l = ver_c.tolist(), val_c.tolist(), end_c.tolist()
        for r, i in enumerate(chunk):
            row = toks[r]
            if end_l[r]:
                row = row[: row.index(EOS)]
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
    val0: float = 0.0  # value probability at the first state


def rollouts(model, attempts, *, gate="none", temperature=1.0, resample=0, tries=1, tau=0.5,
             max_steps=96, max_action=256, seed=0, base_max=8, collect=None, tok_budget=700_000,
             log=None):  # fmt: skip
    """Run every Attempt to its end. `collect`, if a list, receives (state bytes, action bytes)
    for each step the checker refused (gate="checker")."""
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
        a.tries += 1

    for a in attempts:
        start(a)
    live = list(attempts)
    wave = 0
    while live:
        states = [a.env.state_tokens() for a in live]
        pids = [enc(s) for s in states]
        outs, ended, ver, val = gen_actions(model, pids, temperature, max_action, g, tok_budget)
        keep = []
        for a, st, pid, ids, end, vl, v0 in zip(live, states, pids, outs, ended, ver, val):
            a.n_tok += len(ids) + (1 if end else 0)
            a.n_calls += 1
            if not a.acts and a.step_tries == 0:
                a.val0 = _sig(v0)
            out = "syntax"
            pv = 1.0
            if end:
                act = [ITOS[i] for i in ids]
                kind = step_kind(act)
                if gate == "verifier" and kind in ("atom", "pr"):
                    pv = _sig(vl)
                    out = "invalid" if pv < tau else try_step(a.env, act, check=False)
                else:
                    out = try_step(a.env, act, check=(gate == "checker"))
                if out == "invalid" and collect is not None:
                    try:
                        eff = a.env._assign_names(list(act), act.index(":=") + 1)
                        collect.append((pid, enc(eff)))
                    except Exception:
                        pass
            if out == "ok":
                a.acts.append(list(a.env._eff))
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
                if a.step_tries <= resample:
                    keep.append(a)
                    continue
                a.why = out if end else "truncated"
            if a.tries < tries:
                start(a)
                keep.append(a)
        live = keep
        wave += 1
        if log is not None and wave % 10 == 0:
            log(f"  wave {wave}: live {len(live)}")
    return attempts
