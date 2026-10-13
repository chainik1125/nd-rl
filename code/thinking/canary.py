"""Long-context canary and KL anchor for GRPO (collapse audit, 2026-10-08).

The seed-0 thinking GRPO run drifted into a `:` attractor whose strength grows with the number of
hypothesis lines in the state (collapse_audit/README.md §2.1). The RL pool (7-12-line proofs)
almost never reaches those states, so nothing in training measured or repaired it. Both tools
here teacher-force reference steps from long proofs, restricted to states with many hypothesis
lines, in the B0 layout the audit probed:

    decoder input   state [<e0>] [<think> </think>]   (`b0_prompt`: `sample.step_prompt` at B = 0)
    targets         action <eos>

* **Canary** (`canary_metrics`), under the sampler's action distribution (`sample.phase_bans`) at
  T = 1, the audit's setting: the mean action NLL (`nll`); the colon score, the mean P(':') at
  the positions whose target is not `:` (`colon`; `colon_logp` is the mean log P(':') there, the
  audit's c); and its generalisation, the largest mean probability any single token gets at the
  positions where it is not the target (`wrong_max`, that token `wrong_tok`).
* **KL anchor** (`anchor_kl`): coef x KL(pi_ref || pi_theta), the mean over the action positions
  of sum_v pi_ref(v) (log pi_ref(v) - log pi_theta(v)), both masked as the sampler masks an action
  and tempered at the training temperature. Its gradient is accumulated into the policy's .grad.
  With auto effort the level choice at each step's state is anchored too (`choice_seqs`: the
  position that picks the level, masked to the allowed effort tokens, as `sample.choose_levels`).

The file (`canary/1`, written by collapse_audit/build_canary.py through `proof_row`): one JSON row
per reference proof, keeping only the steps whose state holds at least `min_hyp` hypothesis lines
(`:` tokens in the state, premises `h` included; in this layout that is the number of `n : F`
and `h : F` lines). States are flat (`state_env.decompose`, the Stage-1 data path; the same ids
as `progtrace.trace_for` -> `data.trace_pairs`). Tokens are space-separated strings, so the file
serves any tokenizer that has them. Names are shifted at load by the row's `base` (n-names only,
as the sampler's per-attempt name base; `name_base` is the audit's per-item draw in [0, 32]).

`load_canary` splits the proofs once, by a hash of the name (the same split for every run and
seed): the first proofs in that order up to `n` steps are the canary set, the rest the anchor pool,
so the anchor never trains on the steps the canary reads.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, NamedTuple

import torch

from .dan import load
from .effort import effort_id, has_effort

D = load()
FORMAT = "canary/1"
MIN_HYP = 10  # the audit: the attractor bites from about 10 hypothesis lines
NAME_BASE_MAX = 32  # the sampler's per-attempt name base, U[0, 32]
MAX_NAME = 64  # n64 is the largest name in every tokenizer here
Seq = tuple[list[int], list[int]]  # decoder input, scored targets (action + <eos>)
_NAME = re.compile(r"n(\d+)")


def n_hyp(state: Sequence[str]) -> int:
    """Hypothesis lines in a flat state: its `:` tokens (one per `h : F` / `n : F` line)."""
    return sum(1 for t in state if t == ":")


def name_base(name: str, max_n: int) -> int:
    """The audit's per-item name base (teacher_force.build): uniform in [0, 32], capped so the
    largest name stays at or below n64."""
    cap = min(NAME_BASE_MAX, MAX_NAME - max_n)
    return random.Random(f"{name}/base").randint(0, max(0, cap))


def proof_row(
    name: str, prompt: str, proof: str, min_hyp: int = MIN_HYP, **meta: Any
) -> dict[str, Any] | None:
    """A reference proof -> its canary row (steps with >= min_hyp hypothesis lines), None when it
    has no such step. Raises if the proof does not replay."""
    steps, _, _ = D.state_env.decompose(prompt, proof, canon=True)
    max_n = max(
        (int(m[1]) for s, a, _ in steps for t in [*s, *a] if (m := _NAME.fullmatch(t))), default=0
    )
    kept = [
        {"i": i, "n_hyp": n_hyp(s), "state": " ".join(s), "action": " ".join([*a, "<eos>"])}
        for i, (s, a, _) in enumerate(steps)
        if n_hyp(s) >= min_hyp
    ]
    if not kept:
        return None
    return {
        "format": FORMAT,
        "name": name,
        "prompt": prompt,
        **meta,
        "n_steps": len(steps),
        "base": name_base(name, max_n),
        "steps": kept,
    }


def b0_prompt(tok: Any, state: list[int], think: bool) -> list[int]:
    """`sample.step_prompt` for a per-step ('think') format at budget 0: `state [<e0>] <think>
    </think>`; without think vocabulary, the state alone."""
    if not think:
        return list(state)
    e = [effort_id(tok, 0)] if has_effort(tok) else []
    return [*state, *e, tok.think, tok.end_think]


def row_seqs(row: dict[str, Any], tok: Any, think: bool) -> list[Seq]:
    """A canary row -> one (decoder input, targets) per step, names shifted by the row's base."""
    base = int(row.get("base", 0))
    V = tok.vocab_size

    def ids(text: str) -> list[int]:
        out = [x + base if x >= tok.ref0 else x for x in tok.encode_toks(text.split())]
        assert max(out) < V, f"{row['name']}: a name past the vocabulary after the base shift"
        return out

    return [(b0_prompt(tok, ids(s["state"]), think), ids(s["action"])) for s in row["steps"]]


def choice_seqs(seqs: Sequence[Seq], tok: Any) -> list[Seq]:
    """Auto effort: each B0 step (`b0_prompt` with an effort tokenizer) -> (its state, [<e0>]),
    the position where the sampler picks the level (`sample.choose_levels`). `anchor_kl` with the
    choice mask (every token but the allowed effort ones) is then the KL of the level choice; the
    target is a placeholder, since the KL reads the whole distribution."""
    tail = [effort_id(tok, 0), tok.think, tok.end_think]
    out: list[Seq] = []
    for p, _ in seqs:
        assert p[-3:] == tail, "a level choice needs a B0 step of an effort tokenizer"
        out.append((p[:-3], tail[:1]))
    return out


def _order_key(name: str) -> str:
    return hashlib.sha256(name.encode()).hexdigest()


class CanaryData(NamedTuple):
    canary: list[Seq]  # the steps the canary reads
    anchor: list[Seq]  # the rest (the KL anchor's pool)
    n_proofs: int  # proofs in the canary set


def load_canary(path: str | Path, tok: Any, think: bool, n: int) -> CanaryData:
    """The file's steps, split by proof (see the module docstring): about n canary steps (the
    last proof is cut to make exactly n when there are enough), the rest for the anchor."""
    with open(path) as fh:
        rows = [json.loads(x) for x in fh if x.strip()]
    assert rows and all(r.get("format") == FORMAT for r in rows), f"{path}: not a {FORMAT} file"
    rows.sort(key=lambda r: _order_key(r["name"]))
    canary: list[Seq] = []
    anchor: list[Seq] = []
    n_proofs = 0
    for r in rows:
        seqs = row_seqs(r, tok, think)
        if len(canary) < n:
            take = seqs[: n - len(canary)]
            canary += take
            anchor += seqs[len(take) :]
            n_proofs += 1
        else:
            anchor += seqs
    return CanaryData(canary, anchor, n_proofs)


def _batches(seqs: Sequence[Seq], max_tokens: int) -> Iterator[list[int]]:
    """Length-sorted index batches of at most max_tokens padded tokens."""
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i][0]) + len(seqs[i][1]))
    cur: list[int] = []
    cur_max = 0
    for i in order:
        L = len(seqs[i][0]) + len(seqs[i][1])
        if cur and (len(cur) + 1) * max(cur_max, L) > max_tokens:
            yield cur
            cur, cur_max = [], 0
        cur.append(i)
        cur_max = max(cur_max, L)
    if cur:
        yield cur


def _forward(
    model: Any, chunk: Sequence[Seq], dev: str, ban: torch.Tensor, temperature: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """-> (masked tempered log-probs at the scored positions [N, V], their targets [N], the ban
    [V] on dev). The model is causal and rows are right-padded, so padding never reaches a scored
    position."""
    T = max(len(p) + len(t) for p, t in chunk)
    x = torch.zeros((len(chunk), T), dtype=torch.long)
    m = torch.zeros((len(chunk), T - 1), dtype=torch.bool)
    for r, (p, t) in enumerate(chunk):
        assert p and t, "a canary step needs a prompt and a target"
        x[r, : len(p) + len(t)] = torch.tensor(p + t)
        m[r, len(p) - 1 : len(p) - 1 + len(t)] = True
    x, m = x.to(dev), m.to(dev)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
        logits = model(x[:, :-1])
    b = ban.to(dev)
    z = logits.float()[m] / temperature
    return torch.log_softmax(z.masked_fill(b, -math.inf), -1), x[:, 1:][m], b


@torch.no_grad()
def canary_metrics(
    model: Any,
    seqs: Sequence[Seq],
    tok: Any,
    ban_act: torch.Tensor,
    dev: str,
    max_tokens: int = 50_000,
) -> dict[str, Any]:
    """The canary on `seqs` (see the module docstring), at T = 1."""
    V = ban_act.shape[-1]
    colon = tok.stoi[":"]
    nll = c_p = c_lp = 0.0
    n_pos = n_notc = 0
    mass = torch.zeros(V, dtype=torch.float64)  # sum of P(v) over every position
    hit = torch.zeros(V, dtype=torch.float64)  # ... over the positions whose target is v
    n_tgt = torch.zeros(V, dtype=torch.float64)
    for ids in _batches(seqs, max_tokens):
        lp, y, _ = _forward(model, [seqs[i] for i in ids], dev, ban_act, 1.0)
        p = lp.exp()
        nll -= float(lp.gather(1, y[:, None]).sum())
        notc = y != colon
        c_p += float(p[notc, colon].sum())
        c_lp += float(lp[notc, colon].clamp_min(-1e4).sum())  # -inf only if `:` were banned
        n_notc += int(notc.sum())
        n_pos += len(y)
        mass += p.sum(0).double().cpu()
        hit.index_add_(0, y.cpu(), p.gather(1, y[:, None])[:, 0].double().cpu())
        n_tgt += torch.bincount(y.cpu(), minlength=V).double()
    assert n_pos, "no canary positions"
    wrong = torch.where(n_pos - n_tgt > 0, (mass - hit) / (n_pos - n_tgt).clamp_min(1), 0.0)
    k = int(wrong.argmax())
    return {
        "nll": nll / n_pos,
        "colon": c_p / max(1, n_notc),
        "colon_logp": c_lp / max(1, n_notc),
        "wrong_max": float(wrong[k]),
        "wrong_tok": tok.itos[k],
        "steps": len(seqs),
        "positions": n_pos,
    }


def anchor_kl(
    model: Any,
    ref: Any,
    seqs: Sequence[Seq],
    temperature: float,
    ban_act: torch.Tensor,
    dev: str,
    max_tokens: int = 50_000,
    coef: float = 1.0,
    backward: bool = True,
) -> float:
    """KL(pi_ref || pi_theta), the mean over the action positions of `seqs`, both distributions
    masked by `ban_act` and tempered; with `backward`, coef x its gradient is accumulated into the
    model's .grad (one backward per chunk). -> the KL, nats per position."""
    n = sum(len(t) for _, t in seqs)
    if not n:
        return 0.0
    total = 0.0
    for ids in _batches(seqs, max_tokens):
        chunk = [seqs[i] for i in ids]
        with torch.no_grad():
            lq, _, _ = _forward(ref, chunk, dev, ban_act, temperature)
        lp, _, b = _forward(model, chunk, dev, ban_act, temperature)
        # banned entries: pi_ref = 0 and both log-probs -inf; where(), not 0 x nan
        kl = (lq.exp() * torch.where(b, 0.0, lq - lp)).sum() / n
        if backward and coef:
            (coef * kl).backward()
        total += float(kl.detach())
    return total
