"""The σ route, factored: reference proofs, a semantic trace of an attempt, forced sampling, path scores.

Everything here is built on Dan's code at 13e41bbd (read-only, `KD/dan_code_13e41bbd`). It adds:

- `Trace`: replays actions through Dan's `Env` and keeps the ND lines the text denotes (the same
  lines `lean_tok.inverse` gives for a finished attempt), so a prefix of an unfinished attempt can
  be checked by the verifier (`Trace.valid_through`).
- `build_ref`: one reference proof -> its actions and their groups along the σ route.
- `run_env`: Dan's `env_generate` loop with a forced prefix, a stop after n actions, and a record
  of every action. With an empty prefix it is byte-identical to `env_generate` (check 1).
- `score_actions`: exact log P(action tokens + <eos> | state) under the softmax at temperature T.
- the classifiers of the prereg: `classify_pick`, `classify_reach`, `classify_negi`, `classify_dn`,
  `classify_draw`. They use the verifier and parser, never string equality.

Proof text stays in output files under /scratch.
"""

# ruff: noqa: E501, SIM115, B905, B007, SIM300
from __future__ import annotations

import collections
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

KD = Path(os.environ.get("KD", "/scratch/cowifornia/nd-rl/knockout_danbase"))
DAN = KD / "dan_code_13e41bbd"
sys.path.insert(0, str(DAN))
sys.path.insert(0, str(DAN / "coe"))

from coe_sigma import flags, has_sigma  # type: ignore[import-not-found]  # noqa: E402
from lean_tok import fnd, ftoks  # type: ignore[import-not-found]  # noqa: E402
from nd_verify.verify import verify  # type: ignore[import-not-found]  # noqa: E402
from state_env import (  # type: ignore[import-not-found]  # noqa: E402
    BOT,
    Env,
    decompose,
    is_name,
    parse_ftoks,
)

NAME_BASE_MAX = 32  # state_sample.NAME_BASE_MAX: the env draws base ~ U[0, 32] per attempt
GROUPS = ("setup", "pick", "reach", "negi", "dn", "finish")


def neg(f: tuple) -> tuple:
    return ("not", f)


def shift_action(a: list[str], base: int) -> list[str]:
    """A base-0 canonical action at name base `base`. Canonical names are `max in scope + 1` from
    the base, so the whole attempt is the base-0 attempt with every name moved by `base`."""
    return [f"n{int(t[1:]) + base}" if is_name(t) else t for t in a]


# ------------------------------------------------------------------------------- the trace
class Trace:
    """An attempt: Dan's Env plus the ND lines its actions denote.

    `lines[i]` = {idx, depth, formula, rule, refs}, in the order `lean_tok.inverse` emits them (a
    box's AS line when it opens, its IMPI / NEGI / ORE line when it closes). `made[a]` lists the ND
    indices that action `a` created.
    """

    def __init__(self, prompt: str, base: int = 0) -> None:
        self.env = Env(prompt, canon=True, base=base, assign=True)
        self.base = base
        self.lines: list[dict] = []
        self.scopes: list[dict] = [{}]  # parallel to env.frames: name -> ND idx
        self.box: list[dict] = [{}]  # parallel to env.frames: s (AS idx), ore fields
        self.made: list[list[int]] = []
        self.acts: list[list[str]] = []  # effective actions, the env's naming
        self.fail: str | None = None

    # ---- bookkeeping
    def emit(self, depth: int, f: tuple, rule: str, refs: list[int]) -> int:
        i = len(self.lines) + 1
        self.lines.append({"idx": i, "depth": depth, "formula": f, "rule": rule, "refs": refs})
        return i

    def ref(self, n: str) -> int:
        for s in reversed(self.scopes):
            if n in s:
                return s[n]
        raise KeyError(n)

    def fm(self, i: int) -> tuple:
        return self.lines[i - 1]["formula"]

    @property
    def depth(self) -> int:
        return len(self.env.frames) - 1

    # ---- one action
    def apply(self, raw: list[str]) -> bool:
        before_kind = self.env.frames[-1].kind
        before_frame = self.env.frames[-1]
        d = self.depth
        n0 = len(self.lines)
        ok, why = self.env.apply(raw)
        if not ok:
            self.fail = why
            return False
        act = list(self.env._eff)
        self.acts.append(act)
        if act[0] == "exact":
            self._close(before_kind, before_frame, act[1], d)
        else:
            self._have(act, d)
        self.made.append(list(range(n0 + 1, len(self.lines) + 1)))
        return True

    def _close(self, kind: str, fr, n: str, d: int) -> None:
        e = self.ref(n)
        if kind == "top":
            return
        b = self.box[-1]
        if kind == "or1":
            b.update(e1=e)
            nf = self.env.frames[-1]
            s2 = self.emit(d, nf.names[nf.last], "AS", [])
            self.scopes[-1] = {nf.last: s2}
            self.box[-1] = {"j": b["j"], "s1": b["s"], "e1": e, "s": s2}
            return
        self.scopes.pop()
        self.box.pop()
        pn, pf = fr.pending
        if kind == "neg":
            i = self.emit(d - 1, pf, "NEGI", [b["s"], e])
        elif kind == "imp":
            i = self.emit(d - 1, pf, "IMPI", [b["s"], e])
        else:  # or2
            i = self.emit(d - 1, pf, "ORE", [b["j"], b["s1"], b["e1"], b["s"], e])
        self.scopes[-1][pn] = i

    def _have(self, act: list[str], d: int) -> None:
        nm = act[1]
        f, used = parse_ftoks(act, 3)
        i = 3 + used + 1  # past ':='
        t = act[i]
        if t[0] == "h" and t[1:].isdigit():
            self.scopes[-1][nm] = self.emit(0, f, "PR", [])
            return
        if t == "(" or t == "Or.elim":
            nf = self.env.frames[-1]
            s = self.emit(d + 1, nf.names[nf.last], "AS", [])
            self.scopes.append({nf.last: s})
            b = {"s": s}
            if t == "Or.elim":
                b["j"] = self.ref(act[i + 1])
            self.box.append(b)
            return
        tm = act[i:-1]
        r = self.ref
        if tm[0] == "⟨":
            rule, refs = "ANDI", [r(tm[1]), r(tm[3])]
        elif tm[0] == "Or.inl":
            rule, refs = "ORI1", [r(tm[1])]
        elif tm[0] == "Or.inr":
            rule, refs = "ORI2", [r(tm[1])]
        elif tm[0] == "Classical.byContradiction":
            rule, refs = "DN", [r(tm[5])]
        elif len(tm) == 1:
            rule, refs = "R", [r(tm[0])]
        elif tm[1] == ".1":
            rule, refs = "ANDE1", [r(tm[0])]
        elif tm[1] == ".2":
            rule, refs = "ANDE2", [r(tm[0])]
        elif tm[1] == ".elim":
            rule, refs = "BOTE", [r(tm[0])]
        else:
            a, b = r(tm[0]), r(tm[1])
            if self.fm(a)[0] == "not":
                rule, refs = "NEGE", [b, a]
            else:
                rule, refs = "IMPE", [a, b]
        self.scopes[-1][nm] = self.emit(d, f, rule, refs)

    # ---- the verifier on a prefix
    def valid_through(self, n: int | None = None) -> bool:
        """Does the verifier accept the first `n` ND lines? Boxes still open are closed by synthetic
        IMPI lines (the verifier reads box structure from AS lines, and a synthetic IMPI is correct
        by construction), so only the model's lines can fail. The premises are the ones written so
        far, each checked against the theorem (the env does not check a premise line's formula)."""
        lines = self.lines[: len(self.lines) if n is None else n]
        if not lines:
            return True
        out = [dict(x) for x in lines]
        stack: list[dict] = []
        for ln in out:
            if ln["rule"] == "AS":
                del stack[ln["depth"] - 1 :]
                stack.append(ln)
            else:
                del stack[ln["depth"] :]
        while stack:
            s = stack.pop()
            e = out[-1]
            f = ("imp", s["formula"], e["formula"])
            out.append({"idx": len(out) + 1, "depth": s["depth"] - 1, "formula": f,
                        "rule": "IMPI", "refs": [s["idx"], e["idx"]]})  # fmt: skip
        n_pr = sum(1 for x in out if x["rule"] == "PR")
        toks: list[str] = []
        for x in out:
            toks += [f"N{x['idx']}"] + ["|"] * x["depth"] + fnd(x["formula"]).split()
            toks += [":", x["rule"]] + [f"N{r}" for r in x["refs"]] + [";"]
        toks.append("QED")
        ok, _ = verify(self.env.prem[:n_pr], out[-1]["formula"], toks)
        return ok

    def origin(self, i: int) -> dict:
        """Follow R lines back, as coe_sigma._origin does."""
        ln = self.lines[i - 1]
        seen = set()
        while ln["rule"] == "R" and ln["refs"] and ln["idx"] not in seen:
            seen.add(ln["idx"])
            ln = self.lines[ln["refs"][0] - 1]
        return ln


def replay(prompt: str, actions: list[list[str]], base: int = 0) -> Trace:
    tr = Trace(prompt, base)
    for a in actions:
        if not tr.apply(a):
            break
    return tr


# ------------------------------------------------------------------------- reference proofs
def sigma_dn_lines(lines: list[dict]) -> list[dict]:
    """Lines by rule DN whose cited line is a NEGI line after following R (coe_sigma's σ chain)."""
    by = {x["idx"]: x for x in lines}
    out = []
    for x in lines:
        if x["rule"] == "DN" and x["refs"]:
            o = by.get(x["refs"][0])
            seen = set()
            while o is not None and o["rule"] == "R" and o["refs"] and o["idx"] not in seen:
                seen.add(o["idx"])
                o = by.get(o["refs"][0])
            if o is not None and o["rule"] == "NEGI":
                out.append(x)
    return out


def choose_sigma_dn(lines: list[dict]) -> dict | None:
    """The frozen rule: among the σ DN lines, the outermost (minimum depth), and the last of those
    in proof order. An inner reductio then counts as part of reach ⊥, an earlier one at the same
    depth as part of setup."""
    c = sigma_dn_lines(lines)
    if not c:
        return None
    m = min(x["depth"] for x in c)
    return [x for x in c if x["depth"] == m][-1]


@dataclass
class Ref:
    name: str
    prompt: str
    schema: str | None
    source: str
    nd: str
    actions: list[list[str]]  # canonical, base 0
    groups: list[str]  # one per action
    T: tuple
    goal: tuple
    t_is_goal: bool
    n_dn: int  # DN lines in the proof
    n_sigma_dn: int  # those citing a NEGI line
    pick: int  # action indices
    bot: int
    negi: int
    dn: int
    n_between: int  # actions strictly between the NEGI and the DN (they go to group dn)
    n_pr: int
    extra: dict = field(default_factory=dict)


def build_ref(name: str, prompt: str, schema: str | None, source: str, nd: str) -> Ref | str:
    """-> the reference, or a string saying why this proof cannot be one."""
    try:
        steps, toks, env = decompose(prompt, nd, canon=True)
    except (
        Exception
    ) as e:  # ParseFail, the nd2lean-grammar assertions, a citation that cannot be rendered
        return f"decompose: {e}"
    actions = [list(a) for _, a, _ in steps]
    tr = replay(prompt, actions)
    if not tr.env.done or tr.fail:
        return f"replay: {tr.fail}"
    try:
        from_inverse = tr.env.nd()
    except Exception as e:
        return f"inverse: {e}"
    if from_inverse.startswith("LEANPARSE"):
        return from_inverse
    lines_text = _lines_text(tr.lines)
    if lines_text != from_inverse:
        return "trace differs from inverse"
    ok, why = verify(tr.env.prem, tr.env.concl, from_inverse.split())
    if not ok:
        return f"verifier: {why}"
    fl = flags(from_inverse)
    if fl is None or not (fl["w"]["sigma_strict"] or fl["w"]["sigma_chain"]):
        return "not σ-flagged"
    dn = choose_sigma_dn(tr.lines)
    assert dn is not None
    o = tr.origin(dn["refs"][0])
    s, e = o["refs"]
    T = dn["formula"]
    assert tr.fm(s) == neg(T) and tr.fm(e) == BOT, (name, tr.fm(s), T)
    who = {}
    for a, made in enumerate(tr.made):
        for i in made:
            who[i] = a
    pick, bot, negi, dni = who[s], who[e], who[o["idx"]], who[dn["idx"]]
    assert pick <= bot < negi < dni, (name, pick, bot, negi, dni)
    if pick == bot:  # the ⊥ line cannot be the box's own AS line
        return "degenerate box"
    groups = []
    for a in range(len(actions)):
        if a < pick:
            groups.append("setup")
        elif a == pick:
            groups.append("pick")
        elif a <= bot:
            groups.append("reach")
        elif a == negi:
            groups.append("negi")
        elif a <= dni:
            groups.append("dn")  # includes R copies or other lines between NEGI and DN
        else:
            groups.append("finish")
    # a closing `exact` of an inner box between bot and negi does not exist: NEGI closes on the last line
    assert negi == bot + 1, (name, bot, negi)
    return Ref(
        name=name, prompt=prompt, schema=schema, source=source, nd=from_inverse, actions=actions,
        groups=groups, T=T, goal=tr.env.concl, t_is_goal=T == tr.env.concl,
        n_dn=sum(1 for x in tr.lines if x["rule"] == "DN"), n_sigma_dn=len(sigma_dn_lines(tr.lines)),
        pick=pick, bot=bot, negi=negi, dn=dni, n_between=dni - negi - 1,
        n_pr=sum(1 for x in tr.lines if x["rule"] == "PR"),
        extra={"flags": fl["w"], "flags_p": fl["p"], "n_lines": len(tr.lines)},
    )  # fmt: skip


def _lines_text(lines: list[dict]) -> str:
    out = []
    for x in lines:
        out.append(f"N{x['idx']} " + "| " * x["depth"] + f"{fnd(x['formula'])} : {x['rule']}"
                   + "".join(f" N{r}" for r in x["refs"]) + " ;")  # fmt: skip
    return " ".join(out) + " QED"


def ref_to_json(r: Ref) -> dict:
    d = dict(r.__dict__)
    return d


def ref_from_json(d: dict) -> Ref:
    d = dict(d)

    def tup(x):
        return tuple(tup(y) for y in x) if isinstance(x, list) else x

    d["T"], d["goal"] = tup(d["T"]), tup(d["goal"])
    return Ref(**d)


# ------------------------------------------------------------------- the prereg's classifiers
def classify_pick(ref: Ref, base: int, raw: list[str], ended: bool) -> dict:
    """From the prefix before the pick: does the next action open the NEGI box on ¬T at this depth?
    In this env the pick is one action, `have x : ¬¬T := ( fun ( y : ¬T ) => by`, so it also writes
    the NEGI statement. `assume_any` counts any box whose binder is ¬T (an implication box or an
    Or.elim branch assumes ¬T too, but cannot be closed by NEGI)."""
    tr = replay(ref.prompt, [shift_action(a, base) for a in ref.actions[: ref.pick]], base)
    d = tr.depth
    out = {"env_ok": False, "pick": False, "assume_any": False, "valid": False}
    if not ended or not tr.apply(raw):
        return out
    out["env_ok"] = True
    if tr.depth != d + 1:
        return out
    fr = tr.env.frames[-1]
    binder = fr.names[fr.last]
    out["assume_any"] = binder == neg(ref.T)
    out["valid"] = tr.valid_through()
    out["pick"] = (fr.kind == "neg" and fr.pending[1] == neg(neg(ref.T)) and binder == neg(ref.T)
                   and out["valid"])  # fmt: skip
    return out


def classify_negi(ref: Ref, base: int, raw: list[str], ended: bool) -> dict:
    """From the prefix through the ⊥ line: does the next action conclude ¬¬T by NEGI closing that box?"""
    tr = replay(ref.prompt, [shift_action(a, base) for a in ref.actions[: ref.bot + 1]], base)
    s = tr.box[-1]["s"]
    d = tr.depth
    out = {"env_ok": False, "negi": False, "valid": False}
    if not ended or not tr.apply(raw):
        return out
    out["env_ok"] = True
    ln = tr.lines[-1]
    if tr.depth != d - 1 or ln["rule"] != "NEGI" or ln["refs"][0] != s:
        return out
    out["valid"] = tr.valid_through()
    out["negi"] = ln["formula"] == neg(neg(ref.T)) and out["valid"]
    return out


def classify_dn(ref: Ref, base: int, raw: list[str], ended: bool) -> dict:
    """From the prefix through the line before the DN: does the next action conclude T by DN citing the
    NEGI line? `r_copy` counts an R copy of the NEGI line (coe_sigma's chain goes through R).
    `dn_other` counts a DN line with the right formula that cites another line."""
    tr = replay(ref.prompt, [shift_action(a, base) for a in ref.actions[: ref.dn]], base)
    made_n = len(tr.lines)
    negi_idx = tr.made[ref.negi][-1]
    out = {"env_ok": False, "dn": False, "valid": False, "r_copy": False, "dn_other": False}
    if not ended or not tr.apply(raw):
        return out
    out["env_ok"] = True
    if len(tr.lines) != made_n + 1:
        return out
    ln = tr.lines[-1]
    out["valid"] = tr.valid_through()
    if ln["rule"] == "R" and tr.origin(ln["idx"])["idx"] == negi_idx:
        out["r_copy"] = out["valid"]
    if ln["rule"] == "DN" and ln["formula"] == ref.T:
        if tr.origin(ln["refs"][0])["idx"] == negi_idx and out["valid"]:
            out["dn"] = True
        else:
            out["dn_other"] = True
    return out


def reductio_stages(tr: Trace) -> list[dict]:
    """Each box in the attempt that opened by `¬¬X` with binder ¬X: how far its own reductio got.
    Each stage needs every earlier line of the attempt to pass the verifier (the env does not check types).
    stage 1 pick, 2 valid ⊥ on the box's own level, 3 NEGI line, 4 DN line citing that NEGI line (through R)."""
    out = []
    for ln in tr.lines:
        if ln["rule"] != "AS" or ln["formula"][0] != "not":
            continue
        X = ln["formula"][1]
        c = {"s": ln["idx"], "X": X, "stage": 0, "false_bot": 0, "nested_bot": 0}
        negi = next(
            (x for x in tr.lines if x["rule"] == "NEGI" and x["refs"][0] == ln["idx"]), None
        )
        if negi is None and not _opened_as_negbox(tr, ln["idx"], X):
            continue
        if negi is not None and negi["formula"] != neg(neg(X)):
            continue
        if not tr.valid_through(ln["idx"]):
            out.append(c)
            continue
        c["stage"] = 1
        end = negi["idx"] if negi is not None else len(tr.lines) + 1
        stack_depth = ln["depth"]
        in_own_level = _own_level_lines(tr, ln["idx"], end, stack_depth)
        bots = [
            x for x in tr.lines[ln["idx"] : end - 1] if x["formula"] == BOT and x["rule"] != "AS"
        ]
        for x in bots:
            if x["idx"] in in_own_level:
                if tr.valid_through(x["idx"]):
                    c["stage"] = 2
                    break
                c["false_bot"] += 1
            elif tr.valid_through(x["idx"]):
                c["nested_bot"] += 1
        if c["stage"] == 2 and negi is not None and tr.valid_through(negi["idx"]):
            c["stage"] = 3
            for x in tr.lines[negi["idx"] :]:
                if (x["rule"] == "DN" and x["formula"] == X
                        and tr.origin(x["refs"][0])["idx"] == negi["idx"]
                        and tr.valid_through(x["idx"])):  # fmt: skip
                    c["stage"] = 4
                    break
        out.append(c)
    return out


def _opened_as_negbox(tr: Trace, s: int, X: tuple) -> bool:
    """Is the AS line `s` still open, in a box that declares ¬¬X (so a NEGI box, not an implication box)?"""
    for fr, b in zip(tr.env.frames, tr.box):
        if b.get("s") == s:
            return fr.kind == "neg" and fr.pending[1] == neg(neg(X))
    return False


def _own_level_lines(tr: Trace, s: int, end: int, d: int) -> set[int]:
    """Lines in [s, end) on the box's own level: same depth, and no sibling box opened since."""
    return {x["idx"] for x in tr.lines[s - 1 : end - 1] if x["depth"] == d}


def classify_reach(ref: Ref, base: int, actions: list[list[str]]) -> dict:
    """From the prefix through the pick: a full continuation. `bot` = a verifier-valid ⊥ on the box's
    own level; then whether it goes on to NEGI closing that box and a DN citing it."""
    prefix = [shift_action(a, base) for a in ref.actions[: ref.pick + 1]]
    tr = replay(ref.prompt, prefix + actions, base)
    s = tr.made[ref.pick][-1]
    cs = [c for c in reductio_stages(tr) if c["s"] == s]
    c = cs[0] if cs else {"stage": 0, "false_bot": 0, "nested_bot": 0}
    return {"bot": c["stage"] >= 2, "negi": c["stage"] >= 3, "dn": c["stage"] >= 4,
            "false_bot": c["false_bot"], "nested_bot": c["nested_bot"]}  # fmt: skip


def classify_draw(
    prompt: str, base: int, actions: list[list[str]], t_ref: tuple | None = None
) -> dict:
    """An unforced draw: the furthest stage of its own reductio, over every reductio it opens."""
    tr = replay(prompt, actions, base)
    cs = reductio_stages(tr)
    st = max((c["stage"] for c in cs), default=0)
    st_ref = max((c["stage"] for c in cs if t_ref is not None and c["X"] == t_ref), default=0)
    return {"stage": st, "stage_ref_T": st_ref, "n_reductio": len(cs), "n_actions": len(tr.acts),
            "fail": tr.fail, "done": tr.env.done}  # fmt: skip


# ----------------------------------------------------------------------------- forced sampling
@dataclass
class Attempt:
    i: int
    base: int
    raw: list  # model actions: raw tokens as sampled
    ended: list  # per model action: did <eos> come
    end: str  # done | syntax:<why> | truncated | step_cap | stop
    nd: str = ""
    states: list | None = None


def run_env(model, tok, items: list[dict], temperature: float = 0.8, max_action: int = 256, max_steps: int = 48,
            batch: int = 1024, seed: int = 0, stop_after: int | None = None, stats: dict | None = None,
            gate: bool = True, keep_states: bool = False, base_fixed: int | None = None,
            log=None) -> list[Attempt]:  # fmt: skip
    """Dan's `state_sample.env_generate`, line for line, with: a forced prefix per item (`item['prefix']`,
    base-0 actions, played through the env, so its step count includes them), `stop_after` n model actions,
    and a record of each action. `base_fixed` overrides the attempt's name base (check 3 only).

    Same noise as env_generate: base from random.Random(seed * 1000003 + nxt), decode seed
    seed * 100003 + wave, one worklist of `batch` slots."""
    import torch
    from lean_gate import gate as _gate  # type: ignore[import-not-found]
    from sample import generate_ids_fast  # type: ignore[import-not-found]
    from state_sample import prompt_ids  # type: ignore[import-not-found]

    dev = next(model.parameters()).device
    st = stats if stats is not None else {}
    cnt = st.setdefault("env_end", collections.Counter())
    why_cnt = st.setdefault("env_fail_reason", collections.Counter())
    steps_hist = st.setdefault("env_steps", collections.Counter())
    n = len(items)
    prompts = [x["prompt"] for x in items]
    res_nd: list = [None] * n
    res_tx: list = [None] * n
    out: list = [None] * n
    live: list = []
    nxt = 0
    wave = 0
    t0 = time.time()
    last_log = t0

    def finish(i, e, att, end):
        st["names_defined"] = st.get("names_defined", 0) + e.defined
        st["names_renamed"] = st.get("names_renamed", 0) + e.renamed
        nd = e.nd()
        res_nd[i] = nd
        res_tx[i] = tok.text(e.text) if e.done and not nd.startswith("LEANPARSE") else None
        steps_hist[e.steps] += 1
        att.end = end
        out[i] = att

    with torch.no_grad():
        while nxt < n or live:
            while len(live) < batch and nxt < n:
                base = random.Random(seed * 1000003 + nxt).randint(0, NAME_BASE_MAX)
                if base_fixed is not None:
                    base = base_fixed
                e = Env(prompts[nxt], canon=True, base=base, assign=True)
                for a in items[nxt].get("prefix", ()):
                    ok, why = e.apply(shift_action(a, base))
                    assert ok, ("reference prefix rejected", items[nxt].get("name"), why)
                att = Attempt(nxt, base, [], [], "", states=[] if keep_states else None)
                live.append((nxt, e, att))
                nxt += 1
            pids = [prompt_ids(tok, e) for _, e, _ in live]
            if keep_states:
                for (_, e, att), p in zip(live, pids):
                    att.states.append(p)
            st["prompt_tokens"] = st.get("prompt_tokens", 0) + sum(len(p) for p in pids)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev.type == "cuda")):
                outs = generate_ids_fast(model, tok, pids, goals=None, greedy=False, temperature=temperature,
                                         max_new=max_action, seed=seed * 100003 + wave, early="eos", compact=True,
                                         stats=st)  # fmt: skip
            wave += 1
            keep = []
            for (i, e, att), o in zip(live, outs):
                atoks, ended = tok.decode_action(o)
                att.raw.append(atoks)
                att.ended.append(ended)
                if not ended:
                    e.failed = "action truncated"
                    cnt["truncated"] += 1
                    finish(i, e, att, "truncated")
                    continue
                ok, why = e.apply(atoks)
                if not ok:
                    cnt["syntax"] += 1
                    why_cnt[why] += 1
                    finish(i, e, att, f"syntax:{why}")
                    continue
                if e.done:
                    cnt["done"] += 1
                    finish(i, e, att, "done")
                    continue
                if e.steps >= max_steps:
                    e.failed = "step cap"
                    cnt["step_cap"] += 1
                    finish(i, e, att, "step_cap")
                    continue
                if stop_after is not None and len(att.raw) >= stop_after:
                    cnt["stop"] += 1
                    finish(i, e, att, "stop")
                    continue
                keep.append((i, e, att))
            live = keep
            del outs
            if log is not None and time.time() - last_log > 60:
                last_log = time.time()
                log(f"  wave {wave}: {nxt}/{n} started, {len(live)} live, {time.time() - t0:.0f}s")
    st["env_waves"] = st.get("env_waves", 0) + wave
    st["env_wall_s"] = st.get("env_wall_s", 0.0) + time.time() - t0
    if dev.type == "cuda":
        st["peak_alloc_gb"] = torch.cuda.max_memory_allocated() / 2**30
        st["peak_reserved_gb"] = torch.cuda.max_memory_reserved() / 2**30
        torch.cuda.empty_cache()
    if gate:
        res_nd = _gate(tok, prompts, res_nd, res_tx)
    for a, nd in zip(out, res_nd):
        a.nd = nd
    return out


# --------------------------------------------------------------------------------- path scores
def score_rows(model, tok, rows: list[tuple[list[int], list[int]]], temperature: float = 0.8,
               max_elems: int = 2**29) -> list[float]:  # fmt: skip
    """rows: (state ids, action ids including <eos>) -> log P(action | state) at temperature, in nats.

    Right-padded full forward with positions from 0, causal by the ALiBi bias, bf16 autocast as in the
    sampler. Rows are grouped by length so the (B, H, T, T) score tensor stays near 1 GB (max_elems)."""
    import torch
    import torch.nn.functional as F

    dev = next(model.parameters()).device
    order = sorted(range(len(rows)), key=lambda i: len(rows[i][0]) + len(rows[i][1]))
    out = [0.0] * len(rows)
    H = model.cfg["n_head"]
    at = 0
    with torch.no_grad():
        while at < len(order):
            T = len(rows[order[at]][0]) + len(rows[order[at]][1])
            cap = 1
            # the batch's own longest row sets T: grow while the tensor stays in bounds
            j = at
            while j < len(order):
                Tj = len(rows[order[j]][0]) + len(rows[order[j]][1])
                if (j - at + 1) * H * Tj * Tj > max_elems:
                    break
                T = Tj
                j += 1
            cap = max(1, j - at)
            chunk = order[at : at + cap]
            at += cap
            T = max(len(rows[i][0]) + len(rows[i][1]) for i in chunk)
            x = torch.full((len(chunk), T), tok.pad, dtype=torch.long, device=dev)
            for r, i in enumerate(chunk):
                s = rows[i][0] + rows[i][1]
                x[r, : len(s)] = torch.tensor(s, device=dev)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev.type == "cuda")):
                logits = model(x[:, :-1])
            lp = F.log_softmax(logits.float() / temperature, -1)
            for r, i in enumerate(chunk):
                ns, na = len(rows[i][0]), len(rows[i][1])
                pos = torch.arange(ns - 1, ns + na - 1, device=dev)
                tg = torch.tensor(rows[i][1], device=dev)
                out[i] = float(lp[r, pos, tg].sum())
    return out


def ref_rows(tok, ref: Ref, base: int) -> list[tuple[list[int], list[int]]]:
    """One row per reference action at name base `base`: (the state the env shows, the action + <eos>)."""
    from state_sample import prompt_ids  # type: ignore[import-not-found]

    e = Env(ref.prompt, canon=True, base=base, assign=True)
    rows = []
    for a in ref.actions:
        sa = shift_action(a, base)
        rows.append((prompt_ids(tok, e), [tok.stoi[t] for t in sa] + [tok.eos]))
        ok, why = e.apply(sa)
        assert ok, (ref.name, why)
        assert list(e._eff) == sa, (ref.name, e._eff, sa)
    assert e.done, ref.name
    return rows


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


__all__ = ["BOT", "GROUPS", "Attempt", "Ref", "Trace", "build_ref", "classify_dn", "classify_draw",
           "classify_negi", "classify_pick", "classify_reach", "flags", "ftoks", "has_sigma", "neg",
           "ref_from_json", "ref_rows", "ref_to_json", "replay", "run_env", "score_rows", "shift_action",
           "wilson"]  # fmt: skip
