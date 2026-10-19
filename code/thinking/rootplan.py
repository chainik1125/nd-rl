"""Whole-proof plans as thinking: formats R (root) and M (memo), programmatic, from K12 references.

    python -m code.thinking.rootplan gen --format root --out root_traces.jsonl --workers 16
    python -m code.thinking.rootplan gen --format memo --out memo_traces.jsonl --workers 16
    python -m code.thinking.rootplan build --format root --traces root_traces.jsonl \
        --plain train_k12.jsonl --empty-frac 0.2 --out TR.pkl

No new tokens: `<think> </think> <no>` and progtrace's line grammar (PROPOSAL.md section 2).  The
actions are K12's reference actions (`state_env.decompose(canon=True)`); the think text is a plan of
the reference proof, every line of it a true claim checked by `trace_dsl.check_step` (strict).

**The plan.**  The reference term of a frame's goal (`progtrace.build_tree` / `frame_term`: future
statements as `Def`s, boxes as lambdas) is written top-down, depth first, as goal reductions: the
goal's line `ctx ⊢ G := SKEL` has a hole `⊢ F` for each statement it uses, and each hole gets its
own line under the binders in force there (`n6 : A ⊢ F := ...`), every branch (both `Or.elim`
cases, both sides of a pair, ...).  A statement used twice is written once, and a statement whose
goal (context and formula) already has a line is not written again (stats: `plan_dup_goals`); a
statement that is a name in scope (a premise restatement, a reiterated binder) is that name.  Box
binders take the reference's own names (the names the proof will give them), unless the name is
already bound in the block (the environment reuses names after a box closes, e.g. the two
`Or.elim` branches): then the first name above `nb` that the proof never uses.  Over the budget the
plan is cut: the lines on the way to the step's own statement first (the root of the plan down to
it), then the rest in depth-first order up to the first line that does not fit (counted as a
truncation).  A reference line the validator rejects is omitted with its subtree (counted by reason
in the stats, `omitted_lines_by_reason`; see `PlanWriter.walk`).  Exploration is progtrace v2's
backtracking (`StepGen.explore_dead`: a dead alternative with `<no>`, alone or followed one or two
levels), real dead ends first: ex falso only at the goal's line and only when no other dead
alternative is shown, no viable alternative abandoned without a reason, no candidate lines (a plan
is not a step decision).  Its binders avoid the plan's.  It only uses the budget the kept plan
leaves.

**R (root): one sequence per proof, full history, like an agent transcript with observations.**

    state_0 <think> ROOT </think> action_0 <eos> state_1 action_1 <eos> ... action_last <eos>

ROOT is the whole-proof plan at state 0 (cap `--root-budget`, default 640 tokens, tags excluded),
with exploration at the top two levels, checked with `check_step(env_0, ROOT, action_0)`.  Each
`state_t` is `env.state_tokens()` after replaying actions 0..t-1 (it starts with `<st>` and ends
with `<act>`, so the boundaries are unambiguous: `<eos>` ends an action, `<st>` starts the next
observation).  Loss: the think block (tags included) and every action with its `<eos>`; no state
token.  `root_sequence` -> segments `[(ids bytes, is_loss)]`, in order (state_0, no), (`<think>`
ROOT `</think>`, yes), (action_0 `<eos>`, yes), (state_1, no), (action_1 `<eos>`, yes), ...;
`flatten` -> (ids, loss mask); `think_mask` marks the think block (for think weighting).  An
emptied record has ROOT empty (`<think> </think>`).  Sequences are not capped (the context will be
raised); the stats count those over 1,024 / 2,048 / 4,096 tokens.

**M (memo): Markov steps, the memo carried and rewritten every step.**

    prompt  state_t <think> MEMO_{t-1} </think> <think>        (MEMO_{-1} empty)
    target  MEMO_t </think> action_t <eos>                     think_len = len(MEMO_t) + 1

MEMO_t is the remaining plan of the current frame at state t (`frame_term(rec, idx)`: the reference
term of the state's goal from the current statement on), so it is anchored at the state's goal and
shrinks as statements are made; when a box opens or closes, the new frame's plan is written fresh.
Cap `--memo-budget` (default 256); exploration only at decision steps, sometimes (one dead
alternative at the top line).  At an `exact` step the memo is the one line `⊢ G := n`.  Each MEMO_t
is checked with `check_step(env_t, MEMO_t, action_t)`.  Records whose largest pair is over 1,024
tokens are dropped (counted).  An emptied record has every memo empty.

Records (JSONL, ASCII surface like progtrace's): {id, source, format: "root"|"memo", prompt, steps:
[{think, action}], gen, n_lines}.  For R only `steps[0].think` is non-empty (ROOT); for M
`steps[t].think` is MEMO_t.  Either way `trace_dsl.validate(rec, lean=False, strict=True)` checks
the record (each think at its own state), and the generator runs it on every record.  A ROOT or
MEMO that fails is regenerated without exploration, then emptied (counted); a record that still
fails validation is dropped (counted).

`build` writes the `data.save` pickle (+ `.report.json`): M as `data.py`'s pair layout (format
`thinking-pairs/1`, `think_format: "memo"`); R as per-proof sequences (format `thinking-seq/1`,
`think_format: "root"`, `records: [[(ids bytes, is_loss)]]`).  K12 rows without a valid trace get
the format's empty layout (`encode(plain_record(prompt, proof, fmt), tok, fmt)`, which is also
the layout a held-out set for these formats needs); eval classes are excluded as in `data.build`
(default `DEFAULT_EXCLUDE` plus textbook72 train).  `data.load_pairs` loads the M pickle; the R
pickle needs a sequence loader (`flatten` per record).
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import random
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from . import progtrace as pt
from .dan import load, trace_dsl
from .data import (
    DEFAULT_EXCLUDE,
    EvalIndex,
    Pair,
    class_keys,
    length_report,
    read_jsonl,
    think_block,
    unit_hash,
)
from .data import (
    FORMAT as PAIR_FORMAT,
)
from .tok import END_THINK, THINK, ThinkTokenizer

D = load()
SEQ_FORMAT = "thinking-seq/1"
GEN_FORMAT = "rootplan/1"
FORMATS = ("root", "memo")
K12 = pt.K12
TB_TRAIN = pt.TB_TRAIN
TRACES = "/scratch/cowifornia/nd-rl/think/traces"
OUT = {"root": f"{TRACES}/root_traces.jsonl", "memo": f"{TRACES}/memo_traces.jsonl"}
MAXN = 64  # lean_tok.MAXN: names n1..n64
SEQ_THRESHOLDS = (1024, 2048, 4096)

Def = pt.Def
Node = pt.Node
Local = pt.Local
Formula = pt.Formula
Seg = tuple[bytes, bool]


@dataclass
class PCfg:
    """Generation knobs (probabilities per opportunity)."""

    root_budget: int = 640  # ROOT body tokens, tags excluded
    memo_budget: int = 256  # MEMO body tokens, tags excluded
    max_pair: int = 1024  # M: records with a longer pair are dropped
    root_explore_depth: int = 2  # ROOT: exploration at the goal's line and its subgoals' lines
    p_alt_top: float = 0.85  # progtrace v2
    p_alt_deep: float = 0.6
    cap_alt_top: int = 3
    cap_alt_deep: int = 2
    p_efq: float = 0.1  # ex falso: only at the goal's line, when no other dead alternative is
    p_ref_last: float = 0.7  # the reference reduction after every shown dead alternative
    dead_forms: tuple[float, float, float, float] = (0.35, 0.45, 0.2, 0.0)  # short one two box
    memo_p_explore: float = 0.3  # M: a decision step's memo explores the top line
    memo_p_alt: float = 0.8
    memo_cap_alt: int = 1
    memo_p_efq: float = 0.1

    def explore_cfg(self, memo: bool) -> pt.Cfg:
        """progtrace's Cfg for `StepGen.explore_dead` and friends (v2, no viable alternatives)."""
        return pt.Cfg(
            p_alt_top=self.memo_p_alt if memo else self.p_alt_top,
            p_alt_deep=0.0 if memo else self.p_alt_deep,
            cap_alt_top=self.memo_cap_alt if memo else self.cap_alt_top,
            cap_alt_deep=0 if memo else self.cap_alt_deep,
            p_efq=self.memo_p_efq if memo else self.p_efq,
            efq_fill=False,
            p_viable=0.0,
            p_ref_last=self.p_ref_last,
            dead_forms=self.dead_forms,
        )


def td() -> Any:
    return trace_dsl()


def proof_seed(fmt: str, seed: int, pid: str) -> int:
    h = hashlib.sha256(f"rootplan/{fmt}/{seed}/{pid}".encode()).digest()
    return int.from_bytes(h[:8], "big")


# ------------------------------------------------------------------------------- the plan block
class PlanBlock(pt.Block):
    """A progtrace `Block` for a plan: up to 64 think-local binders, names up to n64, and fresh
    names (from `nb` + 1 up) that avoid `reserved` unless asked for by name (`prefer`)."""

    def __init__(self, env: Any, action: list[str], budget: int, reserved: set[str]) -> None:
        super().__init__(env, action, budget, MAXN)
        self.top = MAXN
        self.reserved = reserved

    def fresh(self, prefer: str | None = None, mandatory: bool = False) -> str:
        if len(self.binders) >= self.max_binders:
            raise pt.Budget("binders")
        if prefer is not None and prefer not in self.binders and prefer not in self.sc:
            return prefer
        for k in range(int(self.nb[1:]) + 1, self.top + 1):
            n = f"n{k}"
            if n not in self.binders and n not in self.sc and n not in self.reserved:
                return n
        raise pt.Budget("names")


class Explorer(pt.StepGen):
    """progtrace v2's goal-level backtracking for one plan line, real dead ends first: ex falso is
    shown only at the top line and only when no other dead alternative is; never a viable
    alternative."""

    def __init__(self, env: Any, action: list[str], cfg: pt.Cfg, rng: random.Random) -> None:
        super().__init__(env, action, "decision", None, [], cfg, rng)

    def alts_before(
        self, blk: pt.Block, local: Local, g: Formula, ref_sig: tuple[Any, ...], depth: int
    ) -> tuple[int, list[pt.Alt]]:
        cfg, rng = self.cfg, self.rng
        top = depth == 0
        alts = [a for a in pt.alternatives(blk, local, g) if a.sg != ref_sig]
        dead = [a for a in alts if a.dead and a.kind != "efq"]
        efq = [a for a in alts if a.dead and a.kind == "efq"]
        rng.shuffle(dead)
        self.bump("alts_available", len(dead))
        p = cfg.p_alt_top if top else cfg.p_alt_deep
        cap = cfg.cap_alt_top if top else cfg.cap_alt_deep
        shown = [a for a in dead if rng.random() < p][:cap]
        if not shown and efq and top and rng.random() < cfg.p_efq:
            shown = efq[:1]
        if not shown:
            return 0, []
        k = len(shown) if rng.random() < cfg.p_ref_last else rng.randrange(len(shown) + 1)
        n_pre = 0
        for a in shown[:k]:
            if self.explore_dead(blk, local, g, a, depth):
                n_pre += 1
                self.bump("dead_ends")
                self.bump("efq" if a.kind == "efq" else "real_dead_ends")
        if n_pre:
            self.info["backtrack"] = True
        return n_pre, shown[k:]

    def alts_after(
        self, blk: pt.Block, local: Local, g: Formula, post: list[pt.Alt], depth: int = 0
    ) -> int:
        n = 0
        for a in post:
            if blk.attempt(lambda a=a: blk.reduce(local, g, a.build(lambda: blk.fresh()), no=True)):
                n += 1
                self.bump("dead_ends")
                self.bump("dead_after_ref")
                self.bump("efq" if a.kind == "efq" else "real_dead_ends")
        return n


# ------------------------------------------------------------------------------- plan rendering
@dataclass
class PLine:
    key: int | str  # id() of the line's Def ("root" for the goal's line)
    depth: int
    parent: int  # index of the line whose hole it refines (-1: the goal's line)
    cost: int  # tokens, with the line separator


@dataclass
class PlanWriter:
    """Writes the reference term `term` of the state's goal as a plan (module docstring).

    `dry()` lays out every line (costs, the binder names) without exploration; `select` keeps what
    fits the budget; `write()` writes the kept lines with exploration in the leftover budget."""

    env: Any
    action: list[str]
    term: Node
    budget: int
    target: Def | None = None  # the statement the step's action introduces
    proof_names: set[str] = field(default_factory=lambda: set())  # every name the proof uses
    names: dict[int, str] = field(default_factory=lambda: {})  # id(lambda node) -> binder name
    lines: list[PLine] = field(default_factory=lambda: [])
    failed: collections.Counter[str] = field(default_factory=lambda: collections.Counter())
    used_lams: set[int] = field(default_factory=lambda: set())  # lambdas of the written lines
    dup: int = 0  # statements not written: their goal (context and formula) already has a line

    def __post_init__(self) -> None:
        self.goal: Formula = self.env.frames[-1].goal
        self.lam_ref = lam_ref_names(self.term)  # id(lambda node) -> the reference's binder name
        self.ref = set(self.lam_ref.values())
        if isinstance(self.term, Def) and self.term.formula != self.goal:
            raise ValueError("the frame term does not prove the state's goal")

    def block(self, budget: int, dry: bool) -> PlanBlock:
        """dry: a plan binder that cannot have its reference name avoids every name of the proof;
        else (the binder names are fixed) exploration avoids the plan's binder names."""
        reserved = self.ref | self.proof_names if dry else set(self.names.values())
        return PlanBlock(self.env, self.action, budget, reserved)

    def render(
        self, blk: pt.Block, n0: Node, rn: dict[str, str], local: Local, fix: bool
    ) -> tuple[Node, list[tuple[Def, Local, dict[str, str]]]]:
        """A Def's reduction (its direct sub-Defs as holes) -> (skeleton, [(sub-Def, its local,
        renaming)]).  fix: binder names from the dry pass."""
        subs: list[tuple[Def, Local, dict[str, str]]] = []
        taken: dict[str, Formula] = {}

        def go(n: Node, rn: dict[str, str], loc: Local) -> Node:
            if isinstance(n, Def):
                u = pt.unreit(n)
                inner = u.inner
                if inner[0] == "name" and (inner[1] in rn or inner[1] in blk.sc):
                    return ("name", rn.get(inner[1], inner[1]))
                subs.append((u, loc, rn))
                return ("hole", n.formula)
            k = n[0]
            if k == "name":
                return ("name", rn.get(n[1], n[1]))
            if k == "lam":
                if fix:
                    x = self.names[id(n)]
                    self.used_lams.add(id(n))
                else:
                    saved = blk.binders
                    blk.binders = {**saved, **taken}
                    try:
                        x = blk.fresh(n[1], mandatory=True)
                    finally:
                        blk.binders = saved
                    self.names[id(n)] = x
                taken[x] = n[2]
                return ("lam", x, n[2], go(n[3], {**rn, n[1]: x}, (*loc, (x, n[2]))))
            if k == "proj":
                return ("proj", go(n[1], rn, loc), n[2])
            if k in ("inl", "inr", "dne"):
                return (k, go(n[1], rn, loc))
            if k in ("app", "pair"):
                return (k, go(n[1], rn, loc), go(n[2], rn, loc))
            if k == "orelim":
                return ("orelim", go(n[1], rn, loc), go(n[2], rn, loc), go(n[3], rn, loc))
            raise ValueError(k)

        body = pt.unreit(n0).inner if isinstance(n0, Def) else n0
        return go(body, rn, local), subs

    def walk(
        self,
        blk: pt.Block,
        keep: set[int | str] | None,
        explorer: Explorer | None = None,
        explore_depth: int = 0,
    ) -> None:
        """Depth-first over the plan.  keep None: the dry pass (records `self.lines`, omits the
        lines the validator would reject); else write the kept lines, exploring above
        `explore_depth`."""
        seen: set[int] = set()
        anchored: set[tuple[Local, Formula]] = set()
        T = blk.T

        def visit(
            d: Node, g: Formula, local: Local, rn: dict[str, str], depth: int, parent: int
        ) -> None:
            key: int | str = "root" if depth == 0 else id(d)
            if keep is not None and key not in keep:
                return
            try:
                node, subs = self.render(blk, d, rn, local, fix=keep is not None)
            except pt.Budget as e:
                self.failed[f"render: {e}"] += 1
                return
            post: list[pt.Alt] = []
            if explorer is not None and depth < explore_depth:
                _, post = explorer.alts_before(blk, local, g, pt.sig(node), depth)
            s = blk.snap()
            try:
                holes = blk.reduce(local, g, node)
                if any(h.dead for h in holes):
                    raise AssertionError("a reference hole has a countermodel")
            except (pt.Budget, T.TraceError, AssertionError) as e:
                blk.restore(s)
                if keep is not None:
                    raise  # a kept line was checked in the dry pass
                self.failed[f"{type(e).__name__}: {str(e).split('[in:')[0].strip()[:70]}"] += 1
                return
            cost = len(blk.lines[-1]) + 1
            blk.reserved_tok = max(0, blk.reserved_tok - cost)
            idx = len(self.lines)
            if keep is None:
                self.lines.append(PLine(key, depth, parent, cost))
            if post and explorer is not None:
                explorer.alts_after(blk, local, g, post, depth)
            anchored.add((local, g))
            for sd, sl, srn in subs:
                if id(sd) in seen:
                    continue  # a statement used twice is written once
                if (sl, sd.formula) in anchored:
                    self.dup += keep is None  # a goal already reduced (a repeated statement)
                    continue
                seen.add(id(sd))
                visit(sd, sd.formula, sl, srn, depth + 1, idx)

        visit(self.term, self.goal, (), {}, 0, -1)

    def dry(self) -> None:
        self.lines, self.names, self.failed, self.dup = [], {}, collections.Counter(), 0
        self.walk(self.block(10**9, dry=True), None)

    def select(self) -> tuple[set[int | str], bool]:
        """The kept lines: the way down to the target's line, then a depth-first prefix.
        -> (keys, truncated)."""
        ls = self.lines
        if sum(x.cost for x in ls) <= self.budget:
            return {x.key for x in ls}, False
        chain: list[int] = []
        tkey = id(pt.unreit(self.target)) if self.target is not None else None
        i = next((j for j, x in enumerate(ls) if x.key == tkey), -1)
        while i >= 0:
            chain.append(i)
            i = ls[i].parent
        sel: set[int] = set()
        used = 0
        for i in reversed(chain):
            if used + ls[i].cost > self.budget:
                break
            sel.add(i)
            used += ls[i].cost
        for i, x in enumerate(ls):
            if i in sel:
                continue
            if used + x.cost > self.budget:
                break
            sel.add(i)
            used += x.cost
        return {ls[i].key for i in sel}, True

    def write(
        self, explorer: Explorer | None, explore_depth: int
    ) -> tuple[list[list[str]], dict[str, Any]]:
        """-> (lines, info).  Raises (Budget, TraceError, AssertionError) if a check fails."""
        if not self.lines and not self.failed:
            self.dry()
        keep, truncated = self.select()
        blk = self.block(self.budget, dry=False)
        kept = [x for x in self.lines if x.key in keep]
        blk.reserved_tok = sum(x.cost for x in kept)
        self.used_lams = set()
        self.walk(blk, keep, explorer, explore_depth)
        lines = blk.finish()
        info: dict[str, Any] = {
            "plan_lines": len(self.lines),
            "plan_lines_kept": len(kept),
            "plan_tokens": max(0, sum(x.cost for x in self.lines) - 1),
            "truncated": truncated,
            "omitted": dict(self.failed),
            "dup_goals": self.dup,
            "binders": len(blk.binders),
            "ref_names_used": sum(self.names[k] == self.lam_ref[k] for k in self.used_lams),
            "ref_names_total": len(self.used_lams),
            "explore_tokens": max(0, blk.ntok - sum(x.cost for x in kept)),
        }
        if explorer is not None:
            info.update({k: v for k, v in explorer.info.items() if isinstance(v, (int, bool))})
        return lines, info


def lam_ref_names(term: Node) -> dict[int, str]:
    """id(lambda node) -> the reference's binder name."""
    out: dict[int, str] = {}
    seen: set[int] = set()

    def go(n: Node) -> None:
        if isinstance(n, Def):
            if id(n) not in seen:
                seen.add(id(n))
                go(n.inner)
            return
        if n[0] == "lam":
            out[id(n)] = n[1]
        for c in n[1:]:
            if isinstance(c, Def) or (isinstance(c, tuple) and c and isinstance(c[0], str)):
                go(c)

    go(term)
    return out


def write_plan(
    env: Any,
    action: list[str],
    term: Node,
    budget: int,
    target: Def | None,
    cfg: pt.Cfg | None,
    rng: random.Random,
    explore_depth: int,
    proof_names: set[str] | None = None,
) -> tuple[list[list[str]], dict[str, Any]]:
    """The plan of `term` at `env`, checked strictly; with exploration if `cfg`, falling back to
    the plan alone, then to nothing.  -> (lines, info)."""
    T = td()
    w = PlanWriter(env, action, term, budget, target, proof_names or set())
    errors: list[str] = []
    for explore in (True, False) if cfg is not None else (False,):
        ex = Explorer(env, action, cfg, rng) if explore and cfg is not None else None
        try:
            lines, info = w.write(ex, explore_depth)
            soft = T.check_step(env, ["<think>", *pt.join_lines(lines), "</think>"], action, {})
            if soft:
                raise T.TraceError(f"(strict) {soft[0]}")
        except (AssertionError, T.TraceError, pt.Budget) as e:
            errors.append(f"{type(e).__name__}: {e}"[:160])
            continue
        info["fallback"] = None if explore or cfg is None else "plan_only"
        if errors:
            info["gen_errors"] = errors
        return lines, info
    return [], {"fallback": "empty", "gen_errors": errors, "omitted": dict(w.failed)}


# ------------------------------------------------------------------------------- the two formats
def _ascii(lines: Sequence[Sequence[str]]) -> str:
    return "\n".join(pt.ascii_line(x) for x in lines)


def _replay(prompt: str, proof: str) -> tuple[list[list[str]], list[Any], list[Any], set[str]]:
    """-> (reference actions, env before each, (frame record, index) of each, the n-names the
    proof uses)."""
    T = td()
    steps, _, _ = T.decompose(prompt, proof, canon=True)
    actions = [list(a) for _, a, _ in steps]
    envs, pos = pt.build_tree(prompt, actions)
    names = {x for a in actions for x in a if T.is_name(x) and x[0] == "n"}
    return actions, envs, pos, names


def _step_info(env: Any, act: list[str]) -> dict[str, Any]:
    T = td()
    return {
        "kind": pt.step_kind(act),
        "consistent": not pt.ent(T.scope(env).values(), pt.BOT),
        "state_tokens": len(env.state_tokens()),
        "action_tokens": len(act),
    }


def _record(
    pid: str,
    fmt: str,
    prompt: str,
    thinks: list[str],
    actions: list[list[str]],
    seed: int,
    src: str,
) -> dict[str, Any]:
    return {
        "id": pid,
        "source": src,
        "format": fmt,
        "prompt": prompt,
        "steps": [
            {"think": th, "action": pt.ascii_line(a)} for th, a in zip(thinks, actions, strict=True)
        ],
        "gen": {"format": GEN_FORMAT, "seed": seed},
    }


def root_trace(
    pid: str, prompt: str, proof: str, cfg: PCfg | None = None, seed: int = 0, source: str = ""
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Format R: the whole-proof plan at state 0.  -> (record, per-step info; step 0 carries the
    ROOT's)."""
    cfg = cfg or PCfg()
    rng = random.Random(proof_seed("root", seed, pid))
    actions, envs, pos, names = _replay(prompt, proof)
    rec0, idx0 = pos[0]
    made: list[Def] = []
    term = pt.frame_term(rec0, idx0, {}, made)
    target = next((d for d in made if d.key == 0), None)
    lines, info = write_plan(
        envs[0],
        actions[0],
        term,
        cfg.root_budget,
        target,
        cfg.explore_cfg(memo=False),
        rng,
        cfg.root_explore_depth,
        names,
    )
    infos = [_step_info(env, a) for env, a in zip(envs, actions, strict=True)]
    root = pt.join_lines(lines)
    infos[0].update(info, think_tokens=len(root), lines=len(lines))
    thinks = [_ascii(lines)] + [""] * (len(actions) - 1)
    return _record(pid, "root", prompt, thinks, actions, seed, source), infos


def memo_trace(
    pid: str, prompt: str, proof: str, cfg: PCfg | None = None, seed: int = 0, source: str = ""
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Format M: at every step, the remaining plan of the current frame.  -> (record, per-step
    info)."""
    cfg = cfg or PCfg()
    rng = random.Random(proof_seed("memo", seed, pid))
    actions, envs, pos, names = _replay(prompt, proof)
    ecfg = cfg.explore_cfg(memo=True)
    thinks: list[str] = []
    infos: list[dict[str, Any]] = []
    for t, (act, env) in enumerate(zip(actions, envs, strict=True)):
        rec, idx = pos[t]
        made: list[Def] = []
        term = pt.frame_term(rec, idx, {}, made)
        target = next((d for d in made if d.key == t), None)
        inf = _step_info(env, act)
        explore = inf["kind"] == "decision" and rng.random() < cfg.memo_p_explore
        lines, info = write_plan(
            env, act, term, cfg.memo_budget, target, ecfg if explore else None, rng, 1, names
        )
        inf.update(info, explore=explore, think_tokens=len(pt.join_lines(lines)), lines=len(lines))
        thinks.append(_ascii(lines))
        infos.append(inf)
    return _record(pid, "memo", prompt, thinks, actions, seed, source), infos


TRACE_FNS: dict[str, Callable[..., tuple[dict[str, Any], list[dict[str, Any]]]]] = {
    "root": root_trace,
    "memo": memo_trace,
}


# ------------------------------------------------------------------------------------- encoding
def _body(text: str, T: Any) -> list[str]:
    """ASCII think text -> the block's tokens without the tags."""
    return think_block(text, T)[1:-1]


def root_sequence(rec: dict[str, Any], tok: Any) -> list[Seg]:
    """A format-R record -> one sequence as segments [(ids, is_loss)]: (state_0, no), (<think> ROOT
    </think>, yes), (action_0 <eos>, yes), (state_1, no), (action_1 <eos>, yes), ..."""
    T = td()
    env = D.state_env.Env(rec["prompt"], canon=True, base=0, assign=False)
    out: list[Seg] = []
    for t, st in enumerate(rec["steps"]):
        action = T.tokens(st["action"])
        out.append((bytes(tok.encode_toks(env.state_tokens())), False))
        if t == 0:
            out.append((bytes(tok.encode_toks([THINK, *_body(st["think"], T), END_THINK])), True))
        elif st["think"]:
            raise ValueError("format R: only step 0 has a think block")
        out.append((bytes(tok.encode_toks(action) + [tok.eos]), True))
        ok, why = env.apply(action)
        assert ok, why
    assert env.done
    return out


def flatten(segs: Sequence[Seg]) -> tuple[list[int], list[int]]:
    """Segments -> (ids, loss mask: 1 on loss tokens)."""
    ids: list[int] = []
    mask: list[int] = []
    for b, loss in segs:
        ids += list(b)
        mask += [int(loss)] * len(b)
    return ids, mask


def think_mask(ids: Sequence[int], tok: Any) -> list[int]:
    """1 on think-block tokens (`<think>` .. `</think>` inclusive), for think weighting."""
    out: list[int] = []
    inside = False
    for x in ids:
        if x == tok.think:
            inside = True
        out.append(int(inside))
        if x == tok.end_think:
            inside = False
    return out


def memo_pairs(rec: dict[str, Any], tok: Any) -> list[Pair]:
    """A format-M record -> pairs (prompt `state_t <think> MEMO_{t-1} </think> <think>`, target
    `MEMO_t </think> action_t <eos>`, think_len len(MEMO_t) + 1)."""
    T = td()
    env = D.state_env.Env(rec["prompt"], canon=True, base=0, assign=False)
    prev: list[str] = []
    out: list[Pair] = []
    for st in rec["steps"]:
        action = T.tokens(st["action"])
        body = _body(st["think"], T)
        p = tok.encode_toks([*env.state_tokens(), THINK, *prev, END_THINK, THINK])
        q = tok.encode_toks([*body, END_THINK, *action]) + [tok.eos]
        out.append((bytes(p), bytes(q), len(body) + 1))
        ok, why = env.apply(action)
        assert ok, why
        prev = body
    assert env.done
    return out


def empty_record(rec: dict[str, Any]) -> dict[str, Any]:
    return {**rec, "steps": [{"think": "", "action": st["action"]} for st in rec["steps"]]}


def plain_record(prompt: str, proof: str, fmt: str, pid: str = "") -> dict[str, Any]:
    """A K12 row -> a record of format `fmt` with every think empty (the format's empty layout;
    also what a trainer's held-out set should use)."""
    T = td()
    steps, _, _ = T.decompose(prompt, proof, canon=True)
    return {
        "id": pid,
        "format": fmt,
        "prompt": prompt,
        "steps": [{"think": "", "action": pt.ascii_line(a)} for _, a, _ in steps],
    }


def encode(rec: dict[str, Any], tok: Any, fmt: str) -> list[Any]:
    return root_sequence(rec, tok) if fmt == "root" else memo_pairs(rec, tok)


# ---------------------------------------------------------------------------------- generation CLI
def _dist(xs: Sequence[float]) -> dict[str, Any]:
    if not xs:
        return {"n": 0}
    s = sorted(xs)
    return {
        "n": len(s),
        "mean": round(sum(s) / len(s), 2),
        "p50": s[len(s) // 2],
        "p90": s[int(len(s) * 0.9)],
        "p99": s[int(len(s) * 0.99)],
        "max": s[-1],
    }


def seq_len(infos: Sequence[dict[str, Any]]) -> int:
    """Format R: tokens of the whole sequence (states, the think block with its tags, actions,
    <eos>s)."""
    return sum(i["state_tokens"] + i["action_tokens"] + 1 for i in infos) + (
        infos[0]["think_tokens"] + 2
    )


def memo_pair_lens(infos: Sequence[dict[str, Any]]) -> list[int]:
    """Format M: tokens of each pair (state, the previous memo's block, the new block, action)."""
    out: list[int] = []
    prev = 0
    for i in infos:
        out.append(
            i["state_tokens"] + prev + 2 + 1 + i["think_tokens"] + 1 + i["action_tokens"] + 1
        )
        prev = i["think_tokens"]
    return out


_W: dict[str, Any] = {}


def _init(fmt: str, cfg: PCfg, seed: int, source: str) -> None:
    _W.update(fmt=fmt, cfg=cfg, seed=seed, source=source, T=td())


def _one(job: tuple[str, str, str, int]) -> tuple[str, dict[str, Any] | None, list[Any], str]:
    pid, prompt, proof, n_lines = job
    T, fmt, cfg = _W["T"], _W["fmt"], _W["cfg"]
    try:
        rec, infos = TRACE_FNS[fmt](pid, prompt, proof, cfg, _W["seed"], _W["source"])
    except Exception as e:  # the reference does not replay, or a generator bug
        return pid, None, [], f"generate: {type(e).__name__}: {str(e)[:120]}"
    v = T.validate(rec, lean=False, strict=True)
    if not v["ok"]:
        return pid, None, infos, f"validate: {str(v['error'])[:160]}"
    if fmt == "memo" and max(memo_pair_lens(infos)) > cfg.max_pair:
        return pid, None, infos, f"pair over {cfg.max_pair}: {max(memo_pair_lens(infos))}"
    rec["n_lines"] = n_lines
    return pid, rec, infos, ""


class Stats:
    def __init__(self, fmt: str) -> None:
        self.fmt = fmt
        self.c: collections.Counter[str] = collections.Counter()
        self.dropped: collections.Counter[str] = collections.Counter()
        self.dropped_ids: list[dict[str, str]] = []
        self.omitted: collections.Counter[str] = collections.Counter()
        self.fallback: collections.Counter[str] = collections.Counter()
        self.think: list[int] = []  # ROOT per proof / MEMO per step
        self.think_by_kind: dict[str, list[int]] = collections.defaultdict(list)
        self.by_lines: dict[int, list[int]] = collections.defaultdict(list)
        self.memo_proof_by_lines: dict[int, list[int]] = collections.defaultdict(list)
        self.seq: list[int] = []
        self.seq_by_lines: dict[int, list[int]] = collections.defaultdict(list)
        self.pairs: list[int] = []
        self.explore_tok = 0
        self.think_tok = 0

    def _step(self, inf: dict[str, Any], has_think: bool) -> None:
        k = inf["kind"]
        self.c[f"steps_{k}"] += 1
        if not has_think:
            return
        tt_ = inf["think_tokens"]
        self.think.append(tt_)
        self.think_by_kind[k].append(tt_)
        self.think_tok += tt_
        self.explore_tok += inf.get("explore_tokens", 0)
        self.fallback[f"{k}:{inf.get('fallback')}"] += 1
        self.c["plans"] += 1
        self.c["empty"] += tt_ == 0
        self.c["truncated"] += bool(inf.get("truncated"))
        self.c["explored"] += bool(inf.get("dead_ends"))
        self.c["dead_ends"] += inf.get("dead_ends", 0)
        self.c["real_dead_ends"] += inf.get("real_dead_ends", 0)
        self.c["efq_dead_ends"] += inf.get("efq", 0)
        self.c["plan_lines"] += inf.get("plan_lines", 0)
        self.c["plan_lines_kept"] += inf.get("plan_lines_kept", 0)
        self.c["box_binders"] += inf.get("ref_names_total", 0)
        self.c["box_binders_with_ref_name"] += inf.get("ref_names_used", 0)
        self.c["plans_with_omitted_lines"] += bool(inf.get("omitted"))
        self.c["plan_dup_goals"] += inf.get("dup_goals", 0)
        for r, n in (inf.get("omitted") or {}).items():
            self.omitted[r] += n
        if k == "decision":
            self.c["decision_plans"] += 1
            self.c["decision_explore_drawn"] += bool(inf.get("explore"))

    def add(self, pid: str, rec: dict[str, Any] | None, infos: list[Any], err: str, n: int) -> None:
        if rec is None:
            self.c["dropped"] += 1
            self.dropped[err.split(":")[0] + ": " + err.split(":", 2)[-1].strip()[:60]] += 1
            if len(self.dropped_ids) < 200:
                self.dropped_ids.append({"id": pid, "error": err})
            return
        self.c["traces"] += 1
        if self.fmt == "root":
            for t, inf in enumerate(infos):
                self._step(inf, t == 0)
            self.by_lines[n].append(infos[0]["think_tokens"])
            s = seq_len(infos)
            self.seq.append(s)
            self.seq_by_lines[n].append(s)
        else:
            for inf in infos:
                self._step(inf, True)
            self.memo_proof_by_lines[n].append(sum(i["think_tokens"] for i in infos))
            for i in infos:
                self.by_lines[n].append(i["think_tokens"])
            self.pairs += memo_pair_lens(infos)

    def report(self, wall: float) -> dict[str, Any]:
        c = self.c
        what = "root_tokens_per_proof" if self.fmt == "root" else "memo_tokens_per_step"
        rep: dict[str, Any] = {
            "counts": dict(c),
            "dropped": dict(self.dropped.most_common(40)),
            "dropped_examples": self.dropped_ids,
            "omitted_lines_by_reason": dict(self.omitted.most_common()),
            "fallback": dict(self.fallback),
            what: _dist(self.think),
            f"{what}_by_kind": {k: _dist(v) for k, v in sorted(self.think_by_kind.items())},
            f"{what}_by_lines": {str(k): _dist(v) for k, v in sorted(self.by_lines.items())},
            "fractions": {
                "truncated": round(c["truncated"] / max(c["plans"], 1), 4),
                "empty": round(c["empty"] / max(c["plans"], 1), 4),
                "with_dead_end": round(c["explored"] / max(c["plans"], 1), 4),
                "plan_lines_kept": round(c["plan_lines_kept"] / max(c["plan_lines"], 1), 4),
                "explore_tokens": round(self.explore_tok / max(self.think_tok, 1), 4),
                "box_binders_with_ref_name": round(
                    c["box_binders_with_ref_name"] / max(c["box_binders"], 1), 4
                ),
                "efq_of_dead_ends": round(c["efq_dead_ends"] / max(c["dead_ends"], 1), 4),
            },
            "wall_s": round(wall, 1),
        }
        if self.fmt == "root":
            rep["sequence_tokens"] = _dist(self.seq)
            rep["sequence_tokens_by_lines"] = {
                str(k): _dist(v) for k, v in sorted(self.seq_by_lines.items())
            }
            rep["sequences_over"] = {str(x): sum(s > x for s in self.seq) for x in SEQ_THRESHOLDS}
        else:
            rep["memo_tokens_per_proof_by_lines"] = {
                str(k): _dist(v) for k, v in sorted(self.memo_proof_by_lines.items())
            }
            rep["pair_tokens"] = _dist(self.pairs)
            rep["max_pair_tokens"] = max(self.pairs, default=0)
            rep["pairs_over_1024"] = sum(x > 1024 for x in self.pairs)
        return rep


def load_jobs(
    src: str, start: int, limit: int, exclude: Sequence[str] | None
) -> tuple[list[tuple[str, str, str, int]], collections.Counter[str]]:
    index = EvalIndex(exclude) if exclude else None
    jobs: list[tuple[str, str, str, int]] = []
    excluded: collections.Counter[str] = collections.Counter()
    with open(src) as f:
        for i, line in enumerate(f):
            if i < start:
                continue
            if limit and i >= start + limit:
                break
            r = json.loads(line)
            if index is not None:
                hits = index.hit(r["prompt"])
                if hits:
                    for h in hits:
                        excluded[h] += 1
                    continue
            jobs.append((f"k12_{i}", r["prompt"], r["proof"], int(r.get("n_lines", 0))))
    return jobs, excluded


def gen(a: argparse.Namespace) -> int:
    t0 = time.time()
    src: str = pt.SOURCES.get(a.source) or a.source
    cfg = PCfg(root_budget=a.root_budget, memo_budget=a.memo_budget)
    exclude = None if a.no_exclude else [*DEFAULT_EXCLUDE, TB_TRAIN]
    jobs, excluded = load_jobs(src, a.start, a.limit, exclude)
    print(
        f"{a.format}: {len(jobs)} proofs from {src} ({sum(excluded.values())} eval-class hits "
        f"excluded), {a.workers} workers",
        flush=True,
    )
    out = Path(a.out or OUT[a.format])
    out.parent.mkdir(parents=True, exist_ok=True)
    st = Stats(a.format)
    tmp = out.with_suffix(out.suffix + ".tmp")
    import multiprocessing as mp

    ctx = mp.get_context("fork")
    with (
        open(tmp, "w") as fo,
        ctx.Pool(a.workers, _init, (a.format, cfg, a.seed, a.source)) as pool,
    ):
        for k, (pid, rec, infos, err) in enumerate(pool.imap(_one, jobs, chunksize=64)):
            st.add(pid, rec, infos, err, jobs[k][3])
            if rec is not None:
                fo.write(json.dumps(rec) + "\n")
            if (k + 1) % 20000 == 0:
                print(
                    f"  {k + 1}/{len(jobs)} {st.c['traces']} ok {st.c['dropped']} dropped "
                    f"{time.time() - t0:.0f}s",
                    flush=True,
                )
    os.replace(tmp, out)
    rep = st.report(time.time() - t0)
    rep["config"] = {
        "format": a.format,
        "gen_format": GEN_FORMAT,
        "source": src,
        "seed": a.seed,
        "cfg": asdict(cfg),
        "exclude": exclude,
        "excluded_by_set": dict(excluded),
    }
    sp = out.with_name(out.stem + ".stats.json")
    sp.write_text(json.dumps(rep, indent=1, default=str))
    what = "root_tokens_per_proof" if a.format == "root" else "memo_tokens_per_step"
    extra = (
        f"seq {rep['sequence_tokens']}, over {rep['sequences_over']}"
        if a.format == "root"
        else f"max pair {rep['max_pair_tokens']}"
    )
    print(
        f"{st.c['traces']} traces, {st.c['dropped']} dropped -> {out}; {what} {rep[what]}; "
        f"{extra}; {rep['wall_s']}s; stats {sp}",
        flush=True,
    )
    return 0


# ------------------------------------------------------------------------------------- build CLI
_B: dict[str, Any] = {}


def _binit(fmt: str) -> None:
    _B.update(fmt=fmt, tok=ThinkTokenizer(), T=td())


def _bone(rec: dict[str, Any]) -> tuple[dict[str, Any] | None, Any, Any]:
    """-> (validation failure or None, encoded record, encoded emptied record)."""
    fmt, tok = _B["fmt"], _B["tok"]
    try:
        if rec.get("format") != fmt:
            raise ValueError(f"record format {rec.get('format')!r}, expected {fmt!r}")
        v = _B["T"].validate(rec, lean=False, strict=True)
    except Exception as e:  # a malformed record is an invalid trace, not a crash
        v = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    if not v["ok"]:
        return {"id": rec.get("id"), "error": v["error"]}, None, None
    return None, encode(rec, tok, fmt), encode(empty_record(rec), tok, fmt)


def _plain(job: tuple[str, str, str]) -> Any:
    pid, prompt, proof = job
    try:
        return encode(plain_record(prompt, proof, _B["fmt"], pid), _B["tok"], _B["fmt"])
    except (D.lean_tok.ParseFail, ValueError, AssertionError, IndexError, KeyError):
        return None


def seq_report(records: Sequence[Sequence[Seg]], meta: Sequence[dict[str, Any]]) -> dict[str, Any]:
    tot = [sum(len(b) for b, _ in r) for r in records]
    loss = [sum(len(b) for b, x in r if x) for r in records]
    think = [len(r[1][0]) - 2 for r in records]  # segment 1 is the think block
    by: dict[Any, list[int]] = collections.defaultdict(list)
    for n, m in zip(tot, meta, strict=True):
        by[m.get("n_lines")].append(n)
    return {
        "records": len(records),
        "sequence_tokens": _dist(tot),
        "sequence_tokens_by_lines": {str(k): _dist(v) for k, v in sorted(by.items(), key=str)},
        "loss_tokens": _dist(loss),
        "root_tokens_excl_tags": _dist(think),
        "root_tokens_nonempty": _dist([t for t in think if t > 0]),
        "sequences_over": {str(x): sum(n > x for n in tot) for x in SEQ_THRESHOLDS},
        "tokens_total": sum(tot),
        "loss_tokens_total": sum(loss),
    }


def build(
    fmt: str,
    traces: Sequence[str | Path],
    plain: Sequence[str | Path] = (),
    exclude: Sequence[str | Path] = (),
    empty_frac: float = 0.0,
    max_len: int = 1024,
    workers: int = 1,
    tok: Any = None,
) -> dict[str, Any]:
    """The pickle's dict (module docstring).  Traces are validated again (strict, no Lean)."""
    assert fmt in FORMATS, fmt
    tok = ThinkTokenizer() if tok is None else tok
    index = EvalIndex(exclude)
    rep: dict[str, Any] = {
        "excluded_by_set": collections.Counter(),
        "invalid_trace_reasons": collections.Counter(),
        "invalid_traces": [],
        "plain_skipped": collections.Counter(),
    }
    records: list[Any] = []
    meta: list[dict[str, Any]] = []

    cache: dict[str, bool] = {}

    def excluded(prompt: str) -> bool:
        """In an eval class? (counted once per prompt)"""
        if prompt not in cache:
            hits = index.hit(prompt)
            for h in hits:
                rep["excluded_by_set"][h] += 1
            cache[prompt] = bool(hits)
        return cache[prompt]

    recs: list[dict[str, Any]] = []
    seen: set[str] = set()
    n_trace_in = 0
    for path in traces:
        for r in read_jsonl(path):
            n_trace_in += 1
            if r["prompt"] in seen:
                rep["traces_duplicate_prompt"] = rep.get("traces_duplicate_prompt", 0) + 1
                continue
            if excluded(r["prompt"]):
                rep["excluded_traces"] = rep.get("excluded_traces", 0) + 1
                continue
            seen.add(r["prompt"])
            recs.append(r)
    plain_jobs: list[tuple[str, str, str]] = []
    plain_meta: list[dict[str, Any]] = []
    n_plain_in = 0
    for path in plain:
        for i, r in enumerate(read_jsonl(path)):
            n_plain_in += 1
            if excluded(r["prompt"]):
                rep["plain_skipped"]["eval_class"] += 1
                continue
            plain_jobs.append((r.get("name", f"{Path(path).stem}:{i}"), r["prompt"], r["proof"]))
            plain_meta.append({"n_lines": r.get("n_lines")})
    import multiprocessing as mp

    ctx = mp.get_context("fork")
    n_ok = n_emptied = n_plain_ok = 0
    traced: set[str] = set()
    with ctx.Pool(max(1, workers), _binit, (fmt,)) as pool:
        for r, (bad, enc, enc_empty) in zip(
            recs, pool.imap(_bone, recs, chunksize=64), strict=True
        ):
            if bad is not None:
                rep["invalid_trace_reasons"][str(bad["error"]).split(":")[0]] += 1
                if len(rep["invalid_traces"]) < 200:
                    rep["invalid_traces"].append(bad)
                continue
            n_ok += 1
            traced.add(r["prompt"])
            k, _ = class_keys(r["prompt"])
            m = {
                "id": r.get("id"),
                "prompt": r["prompt"],
                "key": k,
                "source": "trace",
                "n_lines": r.get("n_lines"),
            }
            if empty_frac and unit_hash(f"{r.get('id')}/0") < empty_frac:
                records.append(enc_empty)
                meta.append({**m, "source": "trace_emptied"})
                n_emptied += 1
            else:
                records.append(enc)
                meta.append(m)
        # a K12 row with a valid trace is replaced by it; one whose trace failed is kept plain
        todo = [i for i, j in enumerate(plain_jobs) if j[1] not in traced]
        rep["plain_skipped"]["replaced_by_trace"] += len(plain_jobs) - len(todo)
        plain_jobs = [plain_jobs[i] for i in todo]
        plain_meta = [plain_meta[i] for i in todo]
        for (pid, prompt, _), pm, enc in zip(
            plain_jobs, plain_meta, pool.imap(_plain, plain_jobs, chunksize=64), strict=True
        ):
            if enc is None:
                rep["plain_skipped"]["decompose_failed"] += 1
                continue
            n_plain_ok += 1
            k, _ = class_keys(prompt)
            meta.append(
                {"id": pid, "prompt": prompt, "key": k, "source": "plain", "n_lines": pm["n_lines"]}
            )
            records.append(enc)
    ids = {x for r in records for seg in r for b in seg[:2] if isinstance(b, bytes) for x in b}
    assert not ids or max(ids) < tok.vocab_size <= 256
    lengths = seq_report(records, meta) if fmt == "root" else length_report(records, max_len)
    report = {
        "think_format": fmt,
        "traces_in": n_trace_in,
        "traces_valid": n_ok,
        "empty_frac": empty_frac,
        "trace_records_emptied": n_emptied,
        "plain_in": n_plain_in,
        "plain_used": n_plain_ok,
        "exclude": {str(p): index.sizes[Path(p).stem] for p in exclude},
        **{k: (dict(v) if isinstance(v, collections.Counter) else v) for k, v in rep.items()},
        "lengths": lengths,
    }
    config = {
        "format": fmt,
        "traces": [str(p) for p in traces],
        "plain": [str(p) for p in plain],
        "exclude": [str(p) for p in exclude],
        "lean": False,
        "empty_frac": empty_frac,
        "max_len": max_len,
    }
    return {
        "format": SEQ_FORMAT if fmt == "root" else PAIR_FORMAT,
        "think_format": fmt,
        "mode": tok.mode,
        "itos": list(tok.itos),
        "no_trace_mode": "empty",
        "has_traces": n_ok > 0,
        "records": records,
        "meta": meta,
        "config": config,
        "report": report,
    }


def build_main(a: argparse.Namespace) -> int:
    from .data import save

    exclude = list(DEFAULT_EXCLUDE if a.exclude is None else a.exclude) + list(a.extra_exclude)
    missing = [p for p in exclude if not os.path.exists(p)]
    if missing:
        print(f"error: exclusion sets not found: {missing}", file=sys.stderr)
        return 1
    t0 = time.time()
    data = build(
        a.format,
        a.traces,
        a.plain,
        exclude,
        empty_frac=a.empty_frac,
        max_len=a.max_len,
        workers=a.workers,
    )
    rp = save(data, a.out)
    r = data["report"]
    ln = r["lengths"]
    size = (
        f"{ln['records']} sequences, {ln['sequence_tokens']}, over {ln['sequences_over']}"
        if a.format == "root"
        else f"{ln['pairs']} pairs, max {ln['pair_tokens'].get('max')} tokens, "
        f"{ln['pairs_over_max_len']} over {a.max_len}"
    )
    print(
        f"{a.format}: traces {r['traces_valid']}/{r['traces_in']} valid "
        f"({r['trace_records_emptied']} emptied), plain {r['plain_used']}/{r['plain_in']}; "
        f"{size}; excluded {r['excluded_by_set']}; invalid {r['invalid_trace_reasons']} -> "
        f"{a.out}, {rp} ({time.time() - t0:.0f}s)",
        flush=True,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen", help="generate traces")
    g.add_argument("--format", choices=FORMATS, required=True)
    g.add_argument("--source", default="K12", help="K12 or a K12-style JSONL path")
    g.add_argument("--out", default=None, help="default: OUT[format]")
    g.add_argument("--workers", type=int, default=16)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--limit", type=int, default=0, help="first N rows only (0: all)")
    g.add_argument("--start", type=int, default=0)
    g.add_argument("--root-budget", type=int, default=PCfg.root_budget)
    g.add_argument("--memo-budget", type=int, default=PCfg.memo_budget)
    g.add_argument("--no-exclude", action="store_true", help="keep rows in eval classes")
    b = sub.add_parser("build", help="build the training pickle")
    b.add_argument("--format", choices=FORMATS, required=True)
    b.add_argument("--traces", nargs="*", default=[])
    b.add_argument("--plain", nargs="*", default=[])
    b.add_argument("--exclude", nargs="*", default=None, help="default: data.DEFAULT_EXCLUDE")
    b.add_argument("--extra-exclude", nargs="*", default=[TB_TRAIN])
    b.add_argument("--empty-frac", type=float, default=0.0)
    b.add_argument("--max-len", type=int, default=1024)
    b.add_argument("--workers", type=int, default=16)
    b.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    return gen(a) if a.cmd == "gen" else build_main(a)


if __name__ == "__main__":
    sys.exit(main())
