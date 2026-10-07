"""GRPO for the proof-state thinking model: proof-level reward, failures credited too.

    python -m code.thinking.grpo --init DIR/model.pt --name G_s0 --out GDIR \
        --targets rl_targets.jsonl --budgets 0,16,64,256 --effort --group 16 --prompts 128

Expert iteration (`ei.py`, Dan's ladder) fine-tunes on the Lean-verified attempts only. Here every
attempt counts.

* **Curriculum.** Per iteration, `--prompts` (target, budget) pairs, `--group` attempts each, so a
  group compares attempts made under the same effort. Pairs are drawn by learnability, with weight
  p(1 - p) + `--floor`. Here p is the pair's pass rate, a Beta(1, 1) posterior over its past groups
  whose counts decay by `--decay` each time a new group of the pair is added (a pair that is not
  drawn keeps its counts). Unseen pairs have p = 0.5, so the first pass
  covers everything. Afterwards groups that are always or never solved, which give no gradient,
  are drawn rarely, but the floor brings them back as the policy changes. Plain arm, temperature
  T = `--temperature` (1.0 by default, the usual RLVR choice; evals sample at 0.8).
* **Reward.** 1 if Lean accepts the whole proof, else 0. A success pays `--len_penalty` per 1,000
  sampled tokens (think + action, all steps): the shortness pressure.
* **Overlong masking** (DAPO's overlong filtering, `--overlong mask`). An attempt that a cap ended
  rather than a verdict (`OVERLONG`: max_steps, an action at max_action, the context) gets no
  gradient and is left out of its group's baseline. A think block force-closed at its cap is not
  overlong: the attempt goes on to a verdict. `--overlong fail` scores capped attempts as failures.
* **Advantage.** r minus the mean over the group's unmasked attempts (no division by the group
  std: Dr. GRPO, Liu et al. 2025). Groups with fewer than 2 unmasked attempts, or whose unmasked
  rewards are all equal, give no gradient and are skipped.
* **Loss.** The attempt's advantage times the log-probability of every sampled token of every step,
  at temperature T. That covers the think block and the action, and for a failed attempt the draw
  that ended it. A `</think>` the budget forced was not sampled and is excluded. The loss is a
  token-level mean. Each sample batch is split into `--updates` minibatches, one AdamW step each,
  with the PPO clipped ratio r against the log-probs of the sampling policy. That policy is
  computed once before the first step, so the first minibatch is exactly on-policy. The clip is
  DAPO's clip-higher (Yu et al. 2025, arXiv 2503.14476): min(r A, clip(r, 1 - `--clip_eps_low`,
  1 + `--clip_eps_high`) A), 0.2 / 0.28 by default, so a positive advantage can raise an unlikely
  token further than a symmetric clip allows, against entropy collapse; `--clip_eps` sets both.
  For A < 0 the dual clip (Ye et al. 2020, arXiv 1912.09729) bounds the surrogate below by
  `--dual_clip` x A (3 by default, 0: off), so a token whose r has blown up gives no gradient.
  There is no KL term.
* **Scoring** (2026-10-07). pi_T is the sampler's distribution: inside the think block (its
  `</think>` included) and in the action, the tokens `sample.decode_wave` bans in that phase
  (`sample.phase_bans`: `<eos>`, `<think>`, `<pad>` in the block, effort tokens everywhere) are
  masked and the rest renormalised. Scoring over the full vocabulary instead let a negative
  advantage move think mass onto `<eos>`, which is never sampled and so never penalised (the
  collapsed restart2 run: up to all of the mass at think positions, word-salad think text).
  GRPO samples with the plain arm and no min budget, so those two masks are the whole sampling
  distribution and the old log-probs are recomputed (`old_logps`), not taken from the sampler.
* **Monitoring.** Per iteration: the entropy of the masked pi_T (`entropy`, `entropy_think` /
  `entropy_action`), the full-vocabulary pi_T mass on the think ban at the scored think tokens
  (`ban_think`, `ban_think_max`), the largest r (`ratio_max`), the share of the A < 0 tokens with
  r > 3 (`neg_r3`), the clip fractions below / above, the grad norm (mean and max over the
  minibatch steps), the think tokens per attempt by budget. An `ALARM` line (`alarms`) flags a
  think ban mass above 1e-3, a grad-norm max above 5x the median of the earlier iterations',
  clip-low above 1%, an overlong share above 5%, or a scored token banned in its phase, and the
  iteration's checkpoint is saved. A positive loss is not an alarm: the token-level mean is
  positive whenever failed attempts run longer than successes.

Budget tokens (`--effort`): a `lean_thinkn` init is expanded to `lean_thinke` (effort.py), and the
sampler prefills `<eB>`. Without `--effort` the budget is applied but never shown to the model (as
in ei.py). `--force_mode` is the sampler's budget forcing: 'line' (default) lets the block finish
its line past B, up to B + 64; 'exact' closes it at B. Either way the forced `</think>` is masked.

Files in GDIR: metrics.jsonl (one row per iteration), ckpts/<name>_it<i>.pt every --save_every
iterations and at the end, <name>_it<i>_alarm.pt at each alarming iteration and
<name>_it<i>_rolling.pt every --save_rolling iterations (the last --keep_rolling kept), args.json,
registry/, lean_gate.jsonl.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import random
import re
import statistics
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, NamedTuple

import torch

from .ckpt import load_ckpt, run_env, save_ckpt
from .dan import bare, load
from .data import read_jsonl
from .effort import LEVELS, expand_to_effort, has_effort
from .sample import FORCE_MODES, judge, phase_bans, sample_format, think_generate
from .tok import has_think

D = load()
Item = tuple[list[int], list[int], list[bool], float]  # prompt, generated, loss mask, advantage
Bans = tuple[torch.Tensor, torch.Tensor]  # sample.phase_bans: [V] bool, think block / action
R_WATCH = 3.0  # neg_r3: the share of the scored A < 0 tokens with r above this
# per-iteration alarms (`alarms`)
ALARM_BAN = 1e-3  # mean full-vocabulary mass on the think ban at the scored think tokens
ALARM_GN = 5.0  # grad-norm max vs the median of the earlier iterations' grad-norm max ...
ALARM_GN_AFTER = 5  # ... once this many iterations are in
ALARM_CLIP_LOW = 0.01
ALARM_OVERLONG = 0.05


def rewards(att: Sequence[dict[str, Any]], len_penalty: float) -> list[float]:
    """1 per Lean-accepted attempt minus len_penalty per 1,000 sampled tokens; 0 per failure."""
    return [(1.0 - len_penalty * x["tokens"] / 1000.0) if x.get("ok") else 0.0 for x in att]


# statuses (sample.think_generate) of attempts that a cap ended, not a verdict
OVERLONG = ("fail:step_cap", "fail:truncated", "fail:context")


def is_overlong(x: dict[str, Any]) -> bool:
    return x.get("status") in OVERLONG


def advantages(r: Sequence[float], keep: Sequence[bool] | None = None) -> list[float] | None:
    """r - the mean over the kept attempts, 0 for the others (no gradient); None when fewer than 2
    are kept or every kept reward is equal (no signal)."""
    keep = [True] * len(r) if keep is None else list(keep)
    kr = [x for x, k in zip(r, keep, strict=True) if k]
    if len(kr) < 2 or max(kr) - min(kr) < 1e-9:
        return None
    m = sum(kr) / len(kr)
    return [x - m if k else 0.0 for x, k in zip(r, keep, strict=True)]


def attempt_items(x: dict[str, Any], adv: float, tok: Any) -> list[Item]:
    """Every step of an attempt -> (prompt, generated ids, loss mask, advantage). A forced
    `</think>` is masked out."""
    out: list[Item] = []
    for s in x.get("trace") or []:
        if "gen" not in s:
            continue
        gen = list(s["gen"])
        mask = [True] * len(gen)
        if s.get("forced") and tok.end_think in gen:
            mask[gen.index(tok.end_think)] = False
        if gen:
            out.append((list(s["prompt"]), gen, mask, adv))
    return out


def group_items(
    att: Sequence[dict[str, Any]], tok: Any, len_penalty: float, overlong: str = "mask"
) -> tuple[list[Item] | None, list[bool]]:
    """One group's training items -> (items, None when the group is skipped; per attempt, whether
    it is masked as overlong). Masked attempts give no items, so no gradient."""
    assert overlong in ("mask", "fail"), overlong
    masked = [overlong == "mask" and is_overlong(x) for x in att]
    adv = advantages(rewards(att, len_penalty), [not m for m in masked])
    if adv is None:
        return None, masked
    items: list[Item] = []
    for x, ad, mk in zip(att, adv, masked, strict=True):
        if not mk:
            items += attempt_items(x, ad, tok)
    return items, masked


def _chunks(items: Sequence[Item], max_tokens: int) -> list[list[int]]:
    """Length-sorted micro-batches of at most max_tokens padded tokens."""
    order = sorted(range(len(items)), key=lambda i: len(items[i][0]) + len(items[i][1]))
    chunks: list[list[int]] = []
    cur: list[int] = []
    cur_max = 0
    for idx in order:
        L = len(items[idx][0]) + len(items[idx][1])
        if cur and (len(cur) + 1) * max(cur_max, L) > max_tokens:
            chunks.append(cur)
            cur, cur_max = [], 0
        cur.append(idx)
        cur_max = max(cur_max, L)
    if cur:
        chunks.append(cur)
    return chunks


def think_len(p: Sequence[int], g: Sequence[int], think_ids: tuple[int, int] | None) -> int:
    """How many leading tokens of g are think-block tokens (its `</think>` included): those after
    a prompt that ends with an open `<think>`. 0 without `think_ids` (no think vocabulary)."""
    if think_ids is None or not p or p[-1] != think_ids[0]:
        return 0
    return g.index(think_ids[1]) + 1 if think_ids[1] in g else len(g)


class Scored(NamedTuple):
    """`_token_logp` -> [B, T-1] each; logits at position q predict token q + 1."""

    logp: torch.Tensor  # log pi_T of the next token (-inf: banned in its phase); 0 off the mask
    w: torch.Tensor  # advantage weights, 0 off the mask
    m: torch.Tensor  # loss mask
    th: torch.Tensor  # think-block positions (`think_len`)
    ent: torch.Tensor | None  # with `entropy`: the entropy of pi_T, detached
    ban_mass: torch.Tensor | None  # with `entropy`: full-vocabulary pi_T mass on the think ban


def _token_logp(
    model: Any,
    chunk: Sequence[Item],
    temperature: float,
    dev: str,
    entropy: bool = False,
    think_ids: tuple[int, int] | None = None,
    bans: Bans | None = None,
) -> Scored:
    """pi_T of every next token, the sampler's distribution: with `bans` (`sample.phase_bans`, as
    `decode_wave` got them) the think ban is masked at the think-block positions and the action
    ban at the others, then renormalised; without, the full vocabulary (a tokenizer with neither
    think nor effort tokens bans nothing).  Rows are right-padded with <pad> (id 0). The model is
    causal, so padding never reaches a scored position."""
    T = max(len(p) + len(g) for p, g, _, _ in chunk)
    x = torch.zeros((len(chunk), T), dtype=torch.long)
    w = torch.zeros((len(chunk), T - 1))
    m = torch.zeros((len(chunk), T - 1), dtype=torch.bool)
    th = torch.zeros((len(chunk), T - 1), dtype=torch.bool)
    for r, (p, g, mk, a) in enumerate(chunk):
        assert p and len(mk) == len(g), "an item needs a prompt and one mask entry per token"
        x[r, : len(p) + len(g)] = torch.tensor(p + g)
        s = slice(len(p) - 1, len(p) - 1 + len(g))  # the positions that predict g
        m[r, s] = torch.tensor(mk, dtype=torch.bool)
        w[r, s] = m[r, s].float() * a
        th[r, len(p) - 1 : len(p) - 1 + think_len(p, g, think_ids)] = True
    x, w, m, th = x.to(dev), w.to(dev), m.to(dev), th.to(dev)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
        logits = model(x[:, :-1])
    z = logits.float() / temperature
    ban = b_think = None
    if bans is not None:
        b_think, b_act = (b.to(dev) for b in bans)
        assert b_think.shape[-1] == z.shape[-1], "bans need the model's vocabulary size"
        ban = torch.where(th[..., None], b_think, b_act)  # [B, T-1, V]: each position's phase
        logp = torch.log_softmax(z.masked_fill(ban, -math.inf), dim=-1)
    else:
        logp = torch.log_softmax(z, dim=-1)
    ent = ban_mass = None
    if entropy:
        with torch.no_grad():
            lp = logp.detach()
            ent = -(lp.exp() * (lp if ban is None else lp.masked_fill(ban, 0.0))).sum(-1)
            ban_mass = torch.zeros_like(ent)
            if b_think is not None:
                ban_mass = torch.softmax(z.detach(), -1).masked_fill(~b_think, 0.0).sum(-1)
    # a prompt or forced token can be banned in its position's phase (-inf): where, not * 0
    tgt = torch.where(m, logp.gather(-1, x[:, 1:, None])[..., 0], 0.0)
    return Scored(tgt, w, m, th, ent, ban_mass)


@torch.no_grad()
def old_logps(
    model: Any,
    items: Sequence[Item],
    temperature: float,
    dev: str,
    max_tokens: int = 50_000,
    think_ids: tuple[int, int] | None = None,
    bans: Bans | None = None,
) -> list[torch.Tensor]:
    """The sampling policy's log-prob of every scored token, one tensor per item (mask order)."""
    out: list[torch.Tensor] = [torch.empty(0)] * len(items)
    for ids in _chunks(items, max_tokens):
        chunk = [items[k] for k in ids]
        s = _token_logp(model, chunk, temperature, dev, think_ids=think_ids, bans=bans)
        tgt, m = s.logp.cpu(), s.m.cpu()
        for r, k in enumerate(ids):
            out[k] = tgt[r][m[r]]
    return out


class LossStats(NamedTuple):
    loss: float = 0.0
    tokens: int = 0  # scored tokens
    clip_frac: float = 0.0  # share of the scored tokens with r outside [1 - eps_low, 1 + eps_high]
    clip_low: float = 0.0  # ... with r < 1 - eps_low
    clip_high: float = 0.0  # ... with r > 1 + eps_high
    entropy: float = 0.0  # mean entropy of pi_T over the scored tokens, nats
    tokens_think: int = 0  # scored think-block tokens (`think_len`)
    entropy_think: float = 0.0  # mean entropy over them (0 when there are none)
    entropy_action: float = 0.0  # ... over the other scored tokens
    ban_think: float = 0.0  # mean full-vocabulary pi_T mass on the think ban, scored think tokens
    ban_think_max: float = 0.0  # ... the largest
    ratio_max: float = 0.0  # the largest r over the scored tokens (1 without `old`)
    tokens_neg: int = 0  # scored tokens with A < 0
    neg_r3: float = 0.0  # share of them with r > R_WATCH
    nonfinite: int = 0  # scored tokens banned in their phase (log-prob -inf): left out of the loss


def policy_loss(
    model: Any,
    items: Sequence[Item],
    temperature: float,
    dev: str,
    max_tokens: int = 50_000,
    old: Sequence[torch.Tensor] | None = None,
    eps_low: float = 0.2,
    eps_high: float = 0.28,
    think_ids: tuple[int, int] | None = None,
    bans: Bans | None = None,
    dual_clip: float = 3.0,
) -> LossStats:
    """Backpropagate the policy-gradient loss over `items`, as a token-level mean.
    Without `old`: -A * log pi_T (REINFORCE). With `old` (`old_logps` of the sampling policy, one
    tensor per item of `items`, in the same order): the PPO clipped surrogate with DAPO's
    clip-higher, -min(r A, clip(r, 1 - eps_low, 1 + eps_high) A), r = pi_T / pi_old, and for A < 0
    with `dual_clip` c > 0 the dual clip, -max(that, c A). At r = 1 its gradient is REINFORCE's.
    `think_ids` = (`<think>`, `</think>`) splits the entropy into think and action tokens, `bans`
    masks pi_T as the sampler did (`_token_logp`)."""
    assert old is None or len(old) == len(items), "old log-probs must align with the items"
    n_tok = sum(sum(m) for _, _, m, _ in items)
    if n_tok == 0:
        return LossStats()
    total = ent_sum = ent_think = ban_sum = ban_max = 0.0
    r_max = 1.0 if old is None else 0.0
    n_low = n_high = n_think = n_neg = n_r3 = n_bad = 0
    for ids in _chunks(items, max_tokens):
        chunk = [items[k] for k in ids]
        s = _token_logp(
            model, chunk, temperature, dev, entropy=True, think_ids=think_ids, bans=bans
        )
        assert s.ent is not None and s.ban_mass is not None
        tgt, w, m, mt = s.logp, s.w, s.m, s.m & s.th
        ent_sum += float(s.ent[m].sum())
        ent_think += float(s.ent[mt].sum())
        n_think += int(mt.sum())
        if mt.any():
            ban_sum += float(s.ban_mass[mt].sum())
            ban_max = max(ban_max, float(s.ban_mass[mt].max()))
        if old is None:
            ok = m & torch.isfinite(tgt)
            loss = -(torch.where(ok, tgt, 0.0) * w).sum() / n_tok
        else:
            o = torch.zeros_like(tgt)
            for r, k in enumerate(ids):
                o[r][m[r]] = old[k].to(tgt.device)
            ok = m & torch.isfinite(tgt) & torch.isfinite(o)
            ratio = torch.exp(torch.where(ok, tgt - o, 0.0))  # 1 off the scored tokens
            surr = torch.minimum(ratio * w, torch.clamp(ratio, 1 - eps_low, 1 + eps_high) * w)
            if dual_clip > 0:  # A < 0: no lower than c A however large r gets
                surr = torch.where(w < 0, torch.maximum(surr, dual_clip * w), surr)
            loss = -torch.where(ok, surr, 0.0).sum() / n_tok
            rd = ratio.detach()
            n_low += int((rd < 1 - eps_low)[ok].sum())
            n_high += int((rd > 1 + eps_high)[ok].sum())
            neg = ok & (w < 0)
            n_neg += int(neg.sum())
            n_r3 += int((rd > R_WATCH)[neg].sum())
            if ok.any():
                r_max = max(r_max, float(rd[ok].max()))
        n_bad += int((m & ~ok).sum())
        loss.backward()
        total += float(loss.detach())
    n_act = n_tok - n_think
    return LossStats(
        loss=total,
        tokens=n_tok,
        clip_frac=(n_low + n_high) / n_tok,
        clip_low=n_low / n_tok,
        clip_high=n_high / n_tok,
        entropy=ent_sum / n_tok,
        tokens_think=n_think,
        entropy_think=ent_think / n_think if n_think else 0.0,
        entropy_action=(ent_sum - ent_think) / n_act if n_act else 0.0,
        ban_think=ban_sum / n_think if n_think else 0.0,
        ban_think_max=ban_max,
        ratio_max=r_max,
        tokens_neg=n_neg,
        neg_r3=n_r3 / n_neg if n_neg else 0.0,
        nonfinite=n_bad,
    )


def clip_range(a: argparse.Namespace) -> tuple[float, float]:
    """(eps_low, eps_high): `--clip_eps_low` / `--clip_eps_high` when given, else `--clip_eps` when
    given, else 0.2 / 0.28 (DAPO's clip-higher)."""
    lo = a.clip_eps_low if a.clip_eps_low is not None else a.clip_eps
    hi = a.clip_eps_high if a.clip_eps_high is not None else a.clip_eps
    return (0.2 if lo is None else lo), (0.28 if hi is None else hi)


class Curriculum:
    """Learnability sampling over (target, budget) pairs; see the module docstring."""

    def __init__(
        self, n: int, budgets: Sequence[int], floor: float, decay: float, seed: int
    ) -> None:
        self.keys = [(t, b) for t in range(n) for b in budgets]
        self.index = {k: i for i, k in enumerate(self.keys)}
        self.s = [0.0] * len(self.keys)
        self.n = [0.0] * len(self.keys)
        self.floor, self.decay = floor, decay
        self.rng = random.Random(seed)

    def p(self, i: int) -> float:
        return (self.s[i] + 1.0) / (self.n[i] + 2.0)

    def weight(self, i: int) -> float:
        p = self.p(i)
        return p * (1.0 - p) + self.floor

    def draw(self, m: int) -> list[tuple[int, int]]:
        """m distinct pairs, weighted, without replacement (Efraimidis-Spirakis keys u^(1/w), as
        log(u) / w: u^(1/w) underflows to 0 for small weights, which would tie and fall back to
        index order)."""
        keyed = [
            (math.log(1.0 - self.rng.random()) / self.weight(i), i) for i in range(len(self.keys))
        ]
        keyed.sort(reverse=True)
        return [self.keys[i] for _, i in keyed[: min(m, len(keyed))]]

    def update(self, key: tuple[int, int], successes: int, trials: int) -> None:
        i = self.index[key]
        self.s[i] = self.s[i] * self.decay + successes
        self.n[i] = self.n[i] * self.decay + trials

    def summary(self) -> dict[str, Any]:
        seen = [i for i in range(len(self.keys)) if self.n[i] > 0]
        ps = [self.p(i) for i in seen]
        by_b: dict[str, list[float]] = collections.defaultdict(list)
        for i in seen:
            by_b[str(self.keys[i][1])].append(self.p(i))
        return {
            "seen": len(seen),
            "pairs": len(self.keys),
            "frontier": sum(0.1 < p < 0.9 for p in ps) / max(1, len(ps)),
            "mean_p_by_budget": {b: sum(v) / len(v) for b, v in sorted(by_b.items())},
        }


def sample_groups(
    model: Any,
    tok: Any,
    targets: Sequence[dict[str, Any]],
    budgets: Sequence[int],
    a: argparse.Namespace,
    seed: int,
    think: bool,
) -> list[list[dict[str, Any]]]:
    """`a.group` attempts per target, all at the target's budget (one think_generate call per
    budget, so each call has one budget)."""
    by: dict[int, list[int]] = collections.defaultdict(list)
    for i, b in enumerate(budgets):
        by[b].append(i)
    per: list[list[dict[str, Any]]] = [[] for _ in targets]
    for b, idx in sorted(by.items()):
        prompts = [targets[i]["prompt"] for i in idx for _ in range(a.group)]
        st: dict[str, Any] = {}
        res = think_generate(
            model,
            tok,
            prompts,
            arm="plain",
            budgets=[b] if think else 0,
            think=think,
            temperature=a.temperature,
            max_action=a.max_action,
            max_steps=a.max_steps,
            batch=a.batch,
            seed=seed * 10_007 + b,  # distinct per (iteration, budget) for any budget < 10,007
            stats=st,
            keep_trace=True,
            force_mode=a.force_mode,
        )
        res = judge(tok, prompts, res, a.lean)
        for n, i in enumerate(idx):
            per[i] = res[n * a.group : (n + 1) * a.group]
    return per


def alarms(row: dict[str, Any], gn_prev: Sequence[float]) -> list[str]:
    """An iteration's metrics row -> its alarms, one message each (the ALARM_* thresholds).
    `gn_prev`: the grad-norm max of the earlier iterations that took a step."""
    out = []
    if row["ban_think"] > ALARM_BAN:
        out.append(
            f"think banned mass {row['ban_think']:.3g} > {ALARM_BAN:g} "
            f"(max {row['ban_think_max']:.3g})"
        )
    if len(gn_prev) >= ALARM_GN_AFTER:
        med = statistics.median(gn_prev)
        if row["grad_norm_max"] > ALARM_GN * med:
            out.append(
                f"grad-norm max {row['grad_norm_max']:.3g} > {ALARM_GN:g} x median {med:.3g}"
            )
    if row["clip_low"] > ALARM_CLIP_LOW:
        out.append(f"clip-low {row['clip_low']:.4f} > {ALARM_CLIP_LOW:g}")
    if row["overlong_frac"] > ALARM_OVERLONG:
        out.append(f"overlong {row['overlong_frac']:.4f} > {ALARM_OVERLONG:g}")
    if row["nonfinite"]:
        out.append(f"{row['nonfinite']} scored tokens banned in their phase (sampler / scorer)")
    return out


def prune_rolling(ckdir: Path, name: str, keep: int) -> list[Path]:
    """Delete all but the newest `keep` rolling checkpoints `<name>_it<i>_rolling.pt` (and their
    upload sidecars) -> the deleted. Permanent and alarm checkpoints are never matched."""
    pat = re.compile(re.escape(name) + r"_it(\d+)_rolling\.pt")
    found = sorted((int(mt[1]), p) for p in ckdir.iterdir() if (mt := pat.fullmatch(p.name)))
    gone = [p for _, p in found[: max(0, len(found) - keep)]]
    for p in gone:
        p.unlink()
        p.with_name(p.name + ".upload.json").unlink(missing_ok=True)
    return gone


def run(a: argparse.Namespace) -> list[dict[str, Any]]:
    assert not a.save_rolling or a.keep_rolling >= 1, "--keep_rolling must be >= 1"
    out = Path(a.out)
    run_env(out)
    a.clip_eps_low, a.clip_eps_high = clip_range(a)  # resolved, so args.json records them
    with bare():
        D.record.save_config(vars(a), str(out) + "/", arm=a.name, role="rl")
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    budgets = [int(b) for b in a.budgets.split(",")]
    init = a.init
    if a.effort:
        bad = [b for b in budgets if b not in LEVELS]
        assert not bad, f"--effort: budgets {bad} are not effort levels {LEVELS}"
        _, t0_, _ = load_ckpt(init)
        if not has_effort(t0_):
            init = str(expand_to_effort(init, out / "ckpts" / f"{a.name}_it0.pt", seed=a.seed))
    model, tok, extra = load_ckpt(init, dev)
    think = has_think(tok) and extra.get("think_format", "think") != "none"
    think_ids = (tok.think, tok.end_think) if think else None
    # the sampler's masks: think_generate bans think tokens in R's actions (one block per proof)
    bans = phase_bans(tok, model.cfg["vocab"], think, think and sample_format(model) == "root")
    targets = read_jsonl(a.targets)
    rng = random.Random(a.seed * 7919)
    cur = Curriculum(len(targets), budgets if think else [0], a.floor, a.decay, a.seed)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0)
    history = []
    gn_prev: list[float] = []  # grad_norm_max of the iterations that took a step
    n_step = 0
    for it in range(1, a.iters + 1):
        t0 = time.time()
        pairs = cur.draw(a.prompts)
        batch = [targets[t] for t, _ in pairs]
        tb = [b for _, b in pairs]
        model.eval()
        D.record.phase("sample", round=it)
        per = sample_groups(model, tok, batch, tb, a, a.seed * 100003 + it, think)
        items: list[Item] = []
        n_skip = 0
        by_b: dict[int, list[float]] = collections.defaultdict(list)  # solved, per attempt
        over_b: dict[int, list[float]] = collections.defaultdict(list)  # masked as overlong
        think_b: dict[int, list[float]] = collections.defaultdict(list)  # think tokens
        for att, b, key in zip(per, tb, pairs, strict=True):
            by_b[b] += [float(bool(x.get("ok"))) for x in att]
            think_b[b] += [float(x["think"]) for x in att]
            # the curriculum counts every attempt: a capped one is a failure at eval
            cur.update(key, sum(bool(x.get("ok")) for x in att), len(att))
            its, masked = group_items(att, tok, a.len_penalty, a.overlong)
            over_b[b] += [float(mk) for mk in masked]
            if its is None:
                n_skip += 1
                continue
            items += its
        t_sample = time.time() - t0
        D.record.phase("update", round=it)
        old = old_logps(model, items, a.temperature, dev, a.max_tokens, think_ids, bans)
        model.train()
        order = list(range(len(items)))
        rng.shuffle(order)
        stats: list[LossStats] = []
        gns: list[float] = []
        for u in range(a.updates):
            sel = order[u :: a.updates]
            if not sel:
                continue
            n_step += 1
            for g in opt.param_groups:
                g["lr"] = a.lr * min(1.0, n_step / max(1, a.warmup))
            opt.zero_grad(set_to_none=True)
            st = policy_loss(
                model,
                [items[k] for k in sel],
                a.temperature,
                dev,
                a.max_tokens,
                old=[old[k] for k in sel],
                eps_low=a.clip_eps_low,
                eps_high=a.clip_eps_high,
                think_ids=think_ids,
                bans=bans,
                dual_clip=a.dual_clip,
            )
            if st.tokens:
                gns.append(float(torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)))
                opt.step()
                stats.append(st)
        n_tok = sum(s.tokens for s in stats)
        n_st = max(1, len(stats))  # means over the minibatch steps
        loss = sum(s.loss for s in stats) / n_st
        clip_frac = sum(s.clip_frac for s in stats) / n_st
        clip_low = sum(s.clip_low for s in stats) / n_st
        clip_high = sum(s.clip_high for s in stats) / n_st
        gn = sum(gns) / max(1, len(gns))
        entropy = sum(s.entropy * s.tokens for s in stats) / max(1, n_tok)  # per scored token
        n_th = sum(s.tokens_think for s in stats)
        ent_think = sum(s.entropy_think * s.tokens_think for s in stats) / max(1, n_th)
        ent_action = sum(s.entropy_action * (s.tokens - s.tokens_think) for s in stats) / max(
            1, n_tok - n_th
        )
        ban_think = sum(s.ban_think * s.tokens_think for s in stats) / max(1, n_th)
        n_neg = sum(s.tokens_neg for s in stats)
        gn_max = max(gns, default=0.0)
        flat = [x for att in per for x in att]
        row: dict[str, Any] = {
            "iter": it,
            "reward_pass1": sum(bool(x.get("ok")) for x in flat) / max(1, len(flat)),
            "pass1_by_budget": {str(b): sum(v) / len(v) for b, v in sorted(by_b.items())},
            "solved_frac": sum(any(x.get("ok") for x in att) for att in per) / len(per),
            "groups_skipped": n_skip / len(per),
            "tokens_per_attempt": sum(x["tokens"] for x in flat) / max(1, len(flat)),
            "think_per_attempt": sum(x["think"] for x in flat) / max(1, len(flat)),
            "think_by_budget": {str(b): sum(v) / len(v) for b, v in sorted(think_b.items())},
            "overlong_frac": sum(map(sum, over_b.values())) / max(1, len(flat)),  # masked
            "overlong_by_budget": {str(b): sum(v) / len(v) for b, v in sorted(over_b.items())},
            "items": len(items),
            "trained_tokens": n_tok,
            "loss": loss,
            "clip_frac": clip_frac,  # mean over the minibatch steps
            "clip_frac_max": max((s.clip_frac for s in stats), default=0.0),
            "clip_low": clip_low,
            "clip_high": clip_high,
            "entropy": entropy,  # nats per scored token, masked pi_T, during the update
            "entropy_think": ent_think,  # ... over the think-block tokens
            "entropy_action": ent_action,  # ... over the action tokens
            "ban_think": ban_think,  # full-vocabulary pi_T mass on the think ban, think tokens
            "ban_think_max": max((s.ban_think_max for s in stats), default=0.0),
            "ratio_max": max((s.ratio_max for s in stats), default=0.0),
            "neg_r3": sum(s.neg_r3 * s.tokens_neg for s in stats) / max(1, n_neg),
            "nonfinite": sum(s.nonfinite for s in stats),
            "grad_norm": gn,  # mean pre-clip norm over the minibatch steps
            "grad_norm_max": gn_max,
            "steps": len(gns),
            "curriculum": cur.summary(),
            "lr": opt.param_groups[0]["lr"],
            "secs": time.time() - t0,
            "secs_sample": t_sample,  # sampling and the Lean gate
            "attempts_per_s": len(flat) / max(1e-9, t_sample),
        }
        row["alarms"] = alarms(row, gn_prev)
        if gns:
            gn_prev.append(gn_max)
        with open(out / "metrics.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")
        history.append(row)
        print(
            f"it {it}: pass@1 {100 * row['reward_pass1']:.1f}% "
            f"{ {k: round(100 * v, 1) for k, v in row['pass1_by_budget'].items()} } "
            f"skip {100 * row['groups_skipped']:.0f}% tok/att {row['tokens_per_attempt']:.0f} "
            f"over {100 * row['overlong_frac']:.1f}% think/att {row['think_per_attempt']:.0f} "
            f"loss {loss:+.4f} clip lo/hi {clip_low:.4f}/{clip_high:.4f} "
            f"rmax {row['ratio_max']:.2f} neg_r3 {row['neg_r3']:.4f} "
            f"ent {ent_think:.2f}/{ent_action:.2f} "
            f"ban {ban_think:.1e}/{row['ban_think_max']:.1e} gn {gn:.3f}/{gn_max:.3f} "
            f"frontier {row['curriculum']['frontier']:.2f} {row['secs']:.0f}s "
            f"({row['attempts_per_s']:.0f} att/s sampling)",
            flush=True,
        )
        for msg in row["alarms"]:
            print(f"ALARM it {it}: {msg}", flush=True)
        model.eval()
        meta = {**extra, "grpo_iter": it, "grpo_init": a.init, "args": vars(a)}
        ckdir = out / "ckpts"
        saved = False
        if row["alarms"]:
            save_ckpt(
                str(ckdir / f"{a.name}_it{it}_alarm.pt"),
                model,
                tok.mode,
                extra={**meta, "grpo_alarms": row["alarms"]},
            )
            saved = True
        if it % a.save_every == 0 or it == a.iters:
            save_ckpt(str(ckdir / f"{a.name}_it{it}.pt"), model, tok.mode, extra=meta)
            saved = True
        if a.save_rolling and it % a.save_rolling == 0 and not saved:
            save_ckpt(str(ckdir / f"{a.name}_it{it}_rolling.pt"), model, tok.mode, extra=meta)
            prune_rolling(ckdir, a.name, a.keep_rolling)
        if dev == "cuda":
            torch.cuda.empty_cache()
    D.record.phase(None)
    return history


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--init", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--budgets", default="16,64,256", help="drawn per target (one per group)")
    ap.add_argument("--effort", action="store_true", help="show the budget (effort.py)")
    ap.add_argument("--iters", type=int, default=280)  # 280 x 256 x 16 ~ EI's 8 x 4,495 x 32
    ap.add_argument("--prompts", type=int, default=256, help="(target, budget) pairs / iter")
    ap.add_argument("--group", type=int, default=16, help="attempts per target")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--updates", type=int, default=4, help="minibatch steps per sample batch")
    ap.add_argument("--clip_eps", type=float, default=None, help="sets both (shortcut)")
    ap.add_argument("--clip_eps_low", type=float, default=None, help="default 0.2")
    ap.add_argument("--clip_eps_high", type=float, default=None, help="default 0.28 (DAPO)")
    ap.add_argument("--dual_clip", type=float, default=3.0, help="A < 0: surrogate >= c A; 0: off")
    ap.add_argument("--floor", type=float, default=0.03, help="curriculum weight floor")
    ap.add_argument("--decay", type=float, default=0.7, help="curriculum count decay")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--len_penalty", type=float, default=0.0, help="reward per 1k tokens")
    ap.add_argument("--temperature", type=float, default=1.0, help="sampling and log-probs")
    ap.add_argument(
        "--overlong",
        choices=("mask", "fail"),
        default="mask",
        help="attempts a cap ended (OVERLONG): no gradient, out of the baseline; or failures",
    )
    ap.add_argument(
        "--force_mode",
        choices=FORCE_MODES,
        default="line",
        help="budget forcing (sample.py): 'line' finishes the think line past B (cap B + 64); "
        "'exact' closes at B",
    )
    ap.add_argument("--max_action", type=int, default=256)
    ap.add_argument("--max_steps", type=int, default=48)
    ap.add_argument(
        "--max_tokens",
        type=int,
        default=50_000,
        help="padded tokens per micro-batch (memory only; 200k peaked near 8 GB, 50k was faster)",
    )
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--save_every", type=int, default=35)
    ap.add_argument("--save_rolling", type=int, default=5, help="rolling checkpoints; 0: none")
    ap.add_argument("--keep_rolling", type=int, default=4, help="rolling checkpoints kept")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lean", default=None)
    ap.add_argument("--device", default=None)
    return ap


def main(argv: list[str] | None = None) -> int:
    run(parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
