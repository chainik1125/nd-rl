"""Think-aware sampling in the proof-state environment: plain (`state_sample.env_generate`) and
guided (`guided_sample.guided_generate`) in one batched worklist loop.

Per environment step a row is prefilled with `state <think>` (the forced first token; `state <think>
</think>` when its budget B is 0), then decoded under three masks:
  * inside the block `<eos>`, a nested `<think>` and `<pad>` are banned (thinking is opaque
    otherwise);
  * min budget m: `</think>` is banned until m think tokens are written;
  * max budget B: after B think tokens `</think>` is forced (a forced token has log-prob 0 in the
    redraw bookkeeping);
then the action up to `<eos>`, at most `max_action` tokens (more is a truncated action).  The think
block is stripped and only the action goes to `Env.apply`.  `think=False` (a T0 / `none`-format
model, or a `lean_staten` checkpoint) decodes the action directly, as the fork does.

Arms (guided_sample's): `plain` ends an attempt at the first rejected action; `structural` redraws
an action the environment rejects (or a truncated one); `logical` also redraws one `step_check`
rejects.  **A redraw is the whole think + action**, resampled from the same state.  Redraws are
without replacement over that whole decoded segment (`guided_sample._swor_shift`, unchanged, on the
masked tempered log-probs), so an identical (think, action) is never redrawn, but the same rejected
action can come back after a different think; `repeat_action_redraws` counts that. Every decoded
token is counted, rejected draws included: per attempt `tokens` = `think` + `close` + `action` over
all draws, `rejected` the part spent on rejected draws, `prefill` the prompt tokens (the forced tags
included).

Noise is Gumbel-max keyed by (seed, wave, step, slot) (`sample._gumbel_pick`), name bases U[0, 32]
per attempt and the wave seeds as in the fork, so a batch-size change is a re-draw, never a
correctness change.  With per-attempt budgets the worklist is filled in budget order, because a
wave decodes until its longest row ends.
"""

from __future__ import annotations

import collections
import math
import random
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from . import lean as L
from .dan import load
from .effort import effort_id, has_effort
from .tok import has_think

D = load()
ARMS = ("plain", "structural", "logical")
THINK_PH, ACT_PH, DONE_PH = 1, 2, 3


@dataclass
class Wave:
    """One decoded segment per row."""

    ids: list[list[int]]
    lqs: list[list[float]] | None
    forced: list[bool]


@torch.no_grad()
def decode_wave(
    model: Any,
    tok: Any,
    prompt_ids: Sequence[Sequence[int]],
    budgets: Sequence[int],
    think: bool,
    min_think: int = 0,
    greedy: bool = False,
    temperature: float = 1.0,
    max_action: int = 512,
    seed: int = 0,
    stats: dict[str, Any] | None = None,
    forbids: Sequence[Sequence[tuple[list[int], list[float]]]] | None = None,
    want_lq: bool = False,
    mask_action_think: bool = False,
) -> Wave:
    """Every row writes [think tokens, `</think>`,] action [, `<eos>`].  The prompt already ends
    with the forced `<think>` (or `<think> </think>` for budget 0).  Prefill, KV cache, compaction
    and noise follow sample.py:132-258@4dcd0e4f (generate_ids_fast, early='eos') and the
    forbidden-segment shift guided_sample.py:65-151@4dcd0e4f (decode)."""
    dev = next(model.parameters()).device
    st = stats if stats is not None else {}
    B0 = len(prompt_ids)
    L0 = max(len(p) for p in prompt_ids)
    idx = torch.full((B0, L0), tok.pad, dtype=torch.long, device=dev)
    keep = torch.zeros((B0, L0), dtype=torch.bool, device=dev)
    for i, p in enumerate(prompt_ids):
        idx[i, L0 - len(p) :] = torch.tensor(list(p), device=dev)
        keep[i, L0 - len(p) :] = True
    pos = (keep.cumsum(1) - 1).clamp(min=0)
    causal = torch.tril(torch.ones(L0, L0, dtype=torch.bool, device=dev))
    mask = causal[None, None] & keep[:, None, None, :]
    mask = mask | torch.eye(L0, dtype=torch.bool, device=dev)[None, None]
    bud = [b if think else 0 for b in budgets]
    phase0 = [THINK_PH if think and b > 0 else ACT_PH for b in bud]
    max_new = (
        max((b + 1 if ph == THINK_PH else 0) for b, ph in zip(bud, phase0, strict=True))
        + max_action
    )
    caches: list[dict[str, Any]] = [{"max": L0 + max_new} for _ in model.blocks]
    logits = model(idx, pos=pos, mask=mask, caches=caches)[:, -1]
    del mask, causal, idx
    V = logits.shape[-1]
    out = torch.full((B0, max_new), tok.pad, dtype=torch.long, device=dev)
    out_lq = torch.zeros((B0, max_new), dtype=torch.float32, device=dev)
    declen = torch.full((B0,), max_new, dtype=torch.long, device=dev)
    keepb = torch.ones((B0, L0 + max_new), dtype=torch.bool, device=dev)
    keepb[:, :L0] = keep
    act = torch.arange(B0, device=dev)
    done = torch.zeros(B0, dtype=torch.bool, device=dev)
    phase = torch.tensor(phase0, dtype=torch.long, device=dev)
    nthink = torch.zeros(B0, dtype=torch.long, device=dev)
    nact = torch.zeros(B0, dtype=torch.long, device=dev)
    budt = torch.tensor(bud, dtype=torch.long, device=dev)
    forced = torch.zeros(B0, dtype=torch.bool, device=dev)
    cur_pos = pos[:, -1]
    rgen = None if greedy else torch.Generator(device=dev)
    ban_think = torch.zeros(V, dtype=torch.bool, device=dev)
    ban_act = torch.zeros(V, dtype=torch.bool, device=dev)
    close = getattr(tok, "end_think", -1)
    if think:
        ban_think[[tok.eos, tok.think, tok.pad]] = True
        if mask_action_think:
            ban_act[[tok.think, tok.end_think, tok.no]] = True
    if has_effort(tok):  # effort tokens are inputs only: never sampled
        ban_think[list(tok.effort_ids)] = True
        ban_act[list(tok.effort_ids)] = True
    fb = forbids or []
    cons: dict[int, list[int]] = {r: [] for r in range(B0) if fb and fb[r]}
    act_py = list(range(B0))
    check_every = 16
    temp = 1.0 if greedy else temperature
    for t in range(max_new):
        ph = phase[act]
        nt = nthink[act]
        in_think = ph == THINK_PH
        ban = (in_think[:, None] & ban_think[None]) | ((ph == ACT_PH)[:, None] & ban_act[None])
        if think and min_think > 0:
            ban[:, close] |= in_think & (nt < min_think)
        scaled = (logits.float() / temp).masked_fill(ban, -math.inf)
        lsm = torch.log_softmax(scaled, -1)
        lsm_s = lsm
        if cons:
            slot = {r: j for j, r in enumerate(act_py)}
            shift = [
                (slot[r], c, d)
                for r, p in cons.items()
                if r in slot
                for c, d in D.guided_sample._swor_shift(p, fb[r]).items()
            ]
            if shift:
                ii, cc, dd = zip(*shift, strict=True)
                lsm_s = lsm.clone()
                lsm_s[list(ii), list(cc)] += torch.tensor(dd, device=dev)
        if greedy:
            nxt = lsm_s.argmax(-1)
        else:
            nxt = D.sample._gumbel_pick(lsm_s, 1.0, rgen, seed, t, B0, act)
        fc = in_think & (nt >= budt[act])
        nxt = torch.where(fc, torch.full_like(nxt, close), nxt)
        lq = torch.where(fc, torch.zeros_like(lsm[:, 0]), lsm.gather(1, nxt[:, None]).squeeze(1))
        live = ~done
        out[act, t] = torch.where(live, nxt, out[act, t])
        out_lq[act, t] = torch.where(live, lq, out_lq[act, t])
        closing = live & in_think & (nxt == close)
        nthink[act] = nt + (live & in_think & ~closing).long()
        in_act = live & (ph == ACT_PH)
        na = nact[act] + in_act.long()
        nact[act] = na
        newfin = in_act & ((nxt == tok.eos) | (na >= max_action))
        phase[act] = torch.where(
            closing,
            torch.full_like(ph, ACT_PH),
            torch.where(newfin, torch.full_like(ph, DONE_PH), ph),
        )
        forced[act] |= fc & live
        declen[act] = torch.where(newfin, torch.full_like(declen[act], t + 1), declen[act])
        if cons:
            slot = {r: j for j, r in enumerate(act_py)}
            rs = [r for r in cons if r in slot]
            js = [slot[r] for r in rs]
            vals = nxt[js].tolist() if js else []
            fins = (done[js] | newfin[js]).tolist() if js else []
            for r, v, fin in zip(rs, vals, fins, strict=True):
                p = cons[r]
                p.append(v)
                if fin or not any(len(ids) > len(p) and ids[: len(p)] == p for ids, _ in fb[r]):
                    del cons[r]
        done = done | newfin
        nxt = torch.where(done, torch.full_like(nxt, tok.pad), nxt)
        if t % check_every == check_every - 1 or t == max_new - 1:
            n_live = int((~done).sum())
            if n_live == 0:
                break
            if n_live < 0.75 * done.numel():
                sel = (~done).nonzero(as_tuple=True)[0]
                act = act[sel]
                nxt = nxt[sel]
                cur_pos = cur_pos[sel]
                keepb = keepb[sel]
                act_py = act.tolist()
                done = torch.zeros(n_live, dtype=torch.bool, device=dev)
                for c in caches:
                    for key in [key for key, v in c.items() if torch.is_tensor(v)]:
                        nk = c[key][sel].contiguous()
                        del c[key]
                        c[key] = nk
        cur_pos = cur_pos + 1
        logits = model(
            nxt[:, None],
            pos=cur_pos[:, None],
            mask=keepb[:, None, None, : L0 + t + 1],
            caches=caches,
        )[:, -1]
    dl = declen.tolist()
    o = out.tolist()
    n_tok = sum(dl)
    D.record.count(gen_tokens=n_tok)
    st["sampled_tokens"] = st.get("sampled_tokens", 0) + n_tok
    st["prefill_tokens"] = st.get("prefill_tokens", 0) + sum(len(p) for p in prompt_ids)
    lqs = None
    if want_lq:
        q = out_lq.tolist()
        lqs = [q[i][: dl[i]] for i in range(B0)]
    return Wave([o[i][: dl[i]] for i in range(B0)], lqs, forced.tolist())


def split_segment(
    tok: Any, ids: Sequence[int], think: bool, budget: int
) -> tuple[list[int], bool, list[int]]:
    """A decoded segment -> (think content ids, `</think>` decoded?, action ids incl. <eos>)."""
    if not think or budget == 0:
        return [], False, list(ids)
    j = list(ids).index(tok.end_think) if tok.end_think in ids else len(ids)
    return list(ids[:j]), j < len(ids), list(ids[j + 1 :])


@dataclass
class Attempt:
    i: int
    env: Any
    base: int
    budget: int
    forbid: list[tuple[list[int], list[float]]] = field(default_factory=list)
    rej: int = 0
    draws: int = 0
    tokens: int = 0
    think: int = 0
    close: int = 0
    action: int = 0
    forced: int = 0
    rejected: int = 0
    prefill: int = 0
    canon: bool = True
    flag: str | None = None
    causes: list[str] = field(default_factory=list)
    rejected_actions: set[tuple[int, ...]] = field(default_factory=set)
    trace: list[dict[str, Any]] = field(default_factory=list)


def attempt_budget(budgets: int | Sequence[int], seed: int, i: int) -> int:
    if isinstance(budgets, int):
        return budgets
    if len(budgets) == 1:
        return int(budgets[0])
    return int(budgets[random.Random(seed * 7919 + i + 17).randrange(len(budgets))])


@torch.no_grad()
def think_generate(
    model: Any,
    tok: Any,
    prompts: Sequence[str],
    arm: str = "plain",
    budgets: int | Sequence[int] = 64,
    min_think: int = 0,
    think: bool | None = None,
    temperature: float = 0.8,
    greedy: bool = False,
    max_action: int = 512,
    max_steps: int = 96,
    max_rej: int = 10,
    batch: int = 1024,
    seed: int = 0,
    stats: dict[str, Any] | None = None,
    keep_trace: bool = True,
    mask_action_think: bool = False,
) -> list[dict[str, Any]]:
    """k attempts = k copies of a prompt.  -> one dict per attempt (status 'done' or 'fail:<cause>',
    accepted `steps`, `fail_step`, token counters, `nd` / `text` for the Lean gate, `trace`: per
    accepted step the state, think content, sampled action and the action the environment applied,
    as bytes)."""
    assert arm in ARMS, arm
    think = has_think(tok) if think is None else think
    assert not think or has_think(tok), "think=True needs a lean_thinkn tokenizer"
    if isinstance(budgets, int):
        assert budgets >= 0
    assert min_think == 0 or think
    dev = next(model.parameters()).device
    st = stats if stats is not None else {}
    rej_cause = st.setdefault("rej_cause", collections.Counter())
    end_cnt = st.setdefault("end", collections.Counter())
    flag_cnt = st.setdefault("flag_cause", collections.Counter())
    for k in ("gpu_s", "check_s", "env_s"):
        st.setdefault(k, 0.0)
    st.setdefault("repeat_action_redraws", 0)
    st.setdefault("redraws", 0)
    cache: dict[tuple[tuple[int, ...], tuple[str, ...]], str | None] = {}
    res: list[dict[str, Any] | None] = [None] * len(prompts)
    live: list[Attempt] = []
    nxt = wave = 0
    # a wave decodes until its longest row ends: fill the worklist in budget order so waves are
    # (nearly) one budget each; attempts keep their index for the name base and the results
    bud_of = [attempt_budget(budgets, seed, i) if think else 0 for i in range(len(prompts))]
    order = sorted(range(len(prompts)), key=lambda i: bud_of[i])
    D.record.count(attempts=len(prompts))
    t_start = time.time()
    SC = D.step_check

    def finish(a: Attempt, status: str) -> None:
        end_cnt[status] += 1
        e = a.env
        nd = e.nd() if status == "done" else "LEANPARSE env: " + status
        res[a.i] = dict(
            i=a.i,
            status=status,
            steps=e.steps,
            fail_step=None if status == "done" else e.steps,
            base=a.base,
            budget=a.budget,
            tokens=a.tokens,
            think=a.think,
            close=a.close,
            action=a.action,
            forced=a.forced,
            rejected=a.rejected,
            prefill=a.prefill,
            draws=a.draws,
            rej=a.rej,
            causes=a.causes,
            flag=a.flag,
            nd=nd,
            text=tok.text(e.text) if status == "done" and not nd.startswith("LEANPARSE") else None,
            trace=a.trace if keep_trace else None,
        )

    effort = think and has_effort(tok)  # lean_thinke: the input ends in the attempt's <eB>
    while nxt < len(prompts) or live:
        while len(live) < batch and nxt < len(prompts):
            i = order[nxt]
            base = random.Random(seed * 1000003 + i).randint(0, D.state_sample.NAME_BASE_MAX)
            env = D.state_env.Env(prompts[i], canon=True, base=base, assign=True)
            live.append(Attempt(i, env, base, bud_of[i]))
            nxt += 1
        states = [tok.encode_toks(a.env.state_tokens()) for a in live]
        suffix = [
            (
                ([effort_id(tok, a.budget)] if effort else [])
                + [tok.think]
                + ([tok.end_think] if a.budget == 0 else [])
            )
            if think
            else []
            for a in live
        ]
        pids = [s + x for s, x in zip(states, suffix, strict=True)]
        guided = arm != "plain"
        D.record.count(actions=len(live))
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev.type == "cuda")):
            w = decode_wave(
                model,
                tok,
                pids,
                [a.budget for a in live],
                think,
                min_think=min_think,
                greedy=greedy,
                temperature=temperature,
                max_action=max_action,
                seed=seed * 100003 + wave,
                stats=st,
                forbids=[a.forbid for a in live] if guided else None,
                want_lq=guided,
                mask_action_think=mask_action_think,
            )
        if dev.type == "cuda":
            torch.cuda.synchronize()
        st["gpu_s"] += time.time() - t0
        wave += 1
        keep = []
        for j, (a, s, p) in enumerate(zip(live, states, pids, strict=True)):
            ids = w.ids[j]
            th, closed, act_ids = split_segment(tok, ids, think, a.budget)
            a.draws += 1
            a.tokens += len(ids)
            a.think += len(th)
            a.close += int(closed)
            a.action += len(act_ids)
            a.forced += int(w.forced[j])
            a.prefill += len(p)
            atoks, ended = tok.decode_action(act_ids)
            cause = None
            e2 = None
            h0 = 0
            t1 = time.time()
            if not ended:
                cause = "truncated"
            else:
                e2 = D.guided_sample.clone(a.env)
                h0 = len(e2.hist)
                ok, why = e2.apply(atoks)  # the action only: the think block is stripped
                if not ok:
                    cause = "S:" + why
            st["env_s"] += time.time() - t1
            lcause = None
            if cause is None and a.canon:
                t1 = time.time()
                key = (tuple(s), tuple(atoks))
                if key in cache:
                    lcause = cache[key]
                else:
                    lcause = SC.check(a.env, atoks)
                    cache[key] = lcause
                st["check_s"] += time.time() - t1
                if lcause and arm == "logical":
                    cause = "L:" + lcause
            if cause is not None:
                rej_cause[cause] += 1
                a.causes.append(cause)
                a.rejected += len(ids)
                if arm == "plain" or a.rej >= max_rej:
                    finish(a, "fail:" + (cause if arm == "plain" else "max_rej"))
                    continue
                a.rej += 1
                st["redraws"] += 1
                ak = tuple(act_ids)
                st["repeat_action_redraws"] += int(ak in a.rejected_actions)
                a.rejected_actions.add(ak)
                if ended:
                    assert w.lqs is not None
                    a.forbid = a.forbid + [(ids, w.lqs[j])]
                keep.append(a)
                continue
            assert e2 is not None
            if lcause:
                flag_cnt[lcause] += 1
                if a.flag is None:
                    a.flag = lcause
            if keep_trace:
                eff = tok.encode_toks(e2.hist[h0:])
                a.trace.append(
                    dict(
                        state=bytes(s),
                        think=bytes(th),
                        action=bytes(act_ids[:-1]),
                        applied=bytes(eff),
                        forced=bool(w.forced[j]),
                    )
                )
            a.env = e2
            a.forbid = []
            a.rejected_actions = set()
            if a.canon and not SC.canonical(atoks):
                a.canon = False
                st["noncanon_attempts"] = st.get("noncanon_attempts", 0) + 1
            if e2.done:
                finish(a, "done")
                continue
            if e2.steps >= max_steps:
                finish(a, "fail:step_cap")
                continue
            keep.append(a)
        live = keep
    st["waves"] = st.get("waves", 0) + wave
    st["loop_wall_s"] = st.get("loop_wall_s", 0.0) + time.time() - t_start
    if dev.type == "cuda":
        st["peak_alloc_gb"] = max(
            st.get("peak_alloc_gb", 0), torch.cuda.max_memory_allocated() / 2**30
        )
        torch.cuda.empty_cache()
    out = [r for r in res if r is not None]
    assert len(out) == len(prompts)
    return out


def judge(
    tok: Any, prompts: Sequence[str], res: list[dict[str, Any]], lean: str | None = None
) -> list[dict[str, Any]]:
    """Lean on every finished attempt's literal text; sets res[i]['ok'] and res[i]['nd']."""
    fin = [i for i, r in enumerate(res) if r["text"] is not None]
    g = L.gate(
        tok,
        [prompts[i] for i in fin],
        [res[i]["nd"] for i in fin],
        [res[i]["text"] for i in fin],
        lean,
    )
    for r in res:
        r["ok"] = False
    for i, nd in zip(fin, g, strict=True):
        res[i]["ok"] = not nd.startswith("LEAN")
        res[i]["nd"] = nd
    return res


def env_generate_think(
    model: Any,
    tok: Any,
    prompts: Sequence[str],
    greedy: bool = True,
    temperature: float = 1.0,
    max_action: int = 512,
    max_steps: int = 48,
    batch: int = 1024,
    seed: int = 0,
    stats: dict[str, Any] | None = None,
    gate: bool = True,
    budget: int | Sequence[int] = 64,
    min_think: int = 0,
    lean: str | None = None,
) -> list[str]:
    """`state_sample.env_generate`'s contract (one ND string per prompt: clean if Lean accepted,
    'LEANREJ <nd>', 'LEANPARSE <reason>'), with thinking."""
    res = think_generate(
        model,
        tok,
        prompts,
        "plain",
        budget,
        min_think,
        None,
        temperature,
        greedy,
        max_action,
        max_steps,
        0,
        batch,
        seed,
        stats,
        keep_trace=False,
    )
    if not gate:
        return [r["nd"] for r in res]
    return [
        r["nd"] if r["ok"] or r["text"] is None else "LEANREJ " + r["nd"]
        for r in judge(tok, prompts, res, lean)
    ]
