"""trace_dsl.py: the per-step thinking-trace language for the proof-state model, and its validator.

The model (`state_env_october`, tokenizer mode `lean_staten`, 111 tokens) sees a proof state and
writes one action.  With thinking, it writes a think block first:

    <st> h1 : A <nl> ... ⊢ G <act>  <think> LINE <nl> LINE <nl> ... </think> ACTION <eos>

Every token of a think block is a `lean_staten` token, or one of the three new tokens `<think>`,
`</think>`, `<no>`.  A LINE is one of (EBNF in PROPOSAL.md section 2):

    goal       [x : A ...] ⊢ G [:= SKEL] [<no>]   a (sub)goal under think-local binders x : A,
                                                   optionally reduced to the holes `⊢ F` in SKEL
    look       n : A                               recall a hypothesis or think-local binder
    candidate  ACTION                              an action exactly as the environment takes it
    reject     <no>                                the latest candidate fails

Claims the validator checks (hard errors):
  * every token is in the vocabulary; every formula is in the canonical parenthesised form;
  * a look names something in scope with exactly that formula;
  * an unmarked reduction type-checks (bidirectionally, holes as typed metavariables);
  * `⊢ G <no>`: G is not a classical consequence of the scope (a countermodel over P Q R S);
  * `⊢ G := SKEL <no>`: SKEL is ill-typed, or one of its holes is not a consequence of its scope;
  * an unrejected candidate is accepted by the environment, passes `step_check` and leaves no
    unprovable goal; a rejected candidate fails one of those three;
  * the final ACTION passes the same checks and equals the last unrejected candidate, if any;
  * the replayed actions finish the proof, and Lean 4 accepts the assembled `lean_seq` text.
Soft findings (counted, not errors; `--strict` makes them errors): goal lines that refine nothing
seen so far, vacuous reductions `⊢ G := ⊢ G`, unprovable holes or goals never flagged `<no>`.
A candidate box opener's binder and new goal count as think-local, so the lines after a candidate
may look ahead into the box before its verdict.

Imports the proof-state environment from Dan's fork, read-only (ND_TAKEHOME overrides the path).
CPU only.

    python trace_dsl.py validate pilot_traces.jsonl [--no-lean] [--json out.json]
    python trace_dsl.py show k12 <row index>   # a K12 proof's states and actions (prompt input)
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import os
import re
import sys

sys.dont_write_bytecode = True  # never write __pycache__ into the fork clone
# ND_TAKEHOME: a read-only clone of Dan's fork.  Unset: the byte-identical copy of the fork at
# 4dcd0e4f vendored in code/thinking/vendor/danstate (PROVENANCE.md, SHA256SUMS there).
TAKEHOME = os.environ.get("ND_TAKEHOME") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "thinking", "vendor", "danstate"
)
TAKEHOME = os.path.normpath(TAKEHOME)
sys.path.insert(0, TAKEHOME)

import step_check  # noqa: E402
from lean_tok import MAXN, LeanTokenizer, ParseFail, ftoks  # noqa: E402
from state_env import Env, decompose  # noqa: E402

K12 = os.environ.get("ND_K12", "/scratch/cowifornia/nd-rl/bench/dan_ckpts/data/train_k12.jsonl")

TOK = LeanTokenizer("lean_staten")
NEW_TOKENS = ("<think>", "</think>", "<no>")
VOCAB = set(TOK.itos) | set(NEW_TOKENS)

# ASCII authoring surface (what the LLM may write) -> model tokens.  None is a new model token.
ALIASES = {
    "|-": "⊢",
    "~": "¬",
    "&": "∧",
    "/\\": "∧",
    "v": "∨",
    "\\/": "∨",
    ">": "→",
    "->": "→",
    "F": "False",
    "<|": "⟨",
    "|>": "⟩",
}
GLUED = re.compile(r"^([nh]\d+)(\.1|\.2|\.elim)$")
ATOMS = ("P", "Q", "R", "S")
BOT = ("bot",)
OPS = {"∧": "and", "∨": "or", "→": "imp"}
NAME = re.compile(r"^[nh]\d+$")


class TraceError(Exception):
    """A false claim, an out-of-vocabulary token or a grammar violation: the trace is rejected."""


# --------------------------------------------------------------------------- tokens and formulas
def tokens(text: str) -> list[str]:
    """Authoring text -> model tokens.  A newline inside a think block is `<nl>`."""
    out = []
    for raw in text.replace("\n", " <nl> ").split():
        m = GLUED.match(raw)
        parts = [m.group(1), m.group(2)] if m else [raw]
        out += [ALIASES.get(p, p) for p in parts]
    return out


def is_name(t: str) -> bool:
    if not NAME.match(t):
        return False
    k = int(t[1:])
    return 1 <= k <= (MAXN if t[0] == "n" else 8)


def pform(toks: list[str], i: int):
    """Canonical (fully parenthesised) formula at toks[i:] -> (formula tuple, next index)."""
    if i >= len(toks):
        raise TraceError("formula: end of line")
    t = toks[i]
    if t in ATOMS:
        return ("atom", t), i + 1
    if t == "False":
        return BOT, i + 1
    if t != "(":
        raise TraceError(f"formula: unexpected {t!r}")
    if i + 1 < len(toks) and toks[i + 1] == "¬":
        sub, j = pform(toks, i + 2)
        f = ("not", sub)
    else:
        a, j = pform(toks, i + 1)
        if j >= len(toks) or toks[j] not in OPS:
            raise TraceError("formula: missing connective")
        op = OPS[toks[j]]
        b, j = pform(toks, j + 1)
        f = (op, a, b)
    if j >= len(toks) or toks[j] != ")":
        raise TraceError("formula: missing )")
    return f, j + 1


def fstr(f) -> str:
    return " ".join(ftoks(f))


def holds(f, val: dict) -> bool:
    k = f[0]
    if k == "atom":
        return val[f[1]]
    if k == "bot":
        return False
    if k == "not":
        return not holds(f[1], val)
    if k == "and":
        return holds(f[1], val) and holds(f[2], val)
    if k == "or":
        return holds(f[1], val) or holds(f[2], val)
    return (not holds(f[1], val)) or holds(f[2], val)


VALS = [dict(zip(ATOMS, bits, strict=True)) for bits in itertools.product((False, True), repeat=4)]


def entails(hyps, goal) -> bool:
    """Classical consequence over P Q R S (16 rows).  The fragment has DN: this is provability."""
    hyps = list(hyps)
    return all(holds(goal, v) for v in VALS if all(holds(h, v) for h in hyps))


# ------------------------------------------------------------------- the environment scope
def scope(env: Env) -> dict:
    """Names the state shows -> formulas: premises h1..hm and every hypothesis in scope."""
    sc = {f"h{j + 1}": p for j, p in enumerate(env.prem)}
    for fr in env.frames:
        sc.update(fr.names)
    return sc


def next_names(env: Env) -> tuple[str, str]:
    """The canonical names the environment gives a `have` and its box binder (arm SN)."""
    a = env.next_name()
    return f"n{a}", f"n{env.next_name(reserve=[str(a)])}"


def dead_after(env: Env) -> str | None:
    """None if every goal the (already applied) step leaves open follows from its scope."""
    if env.done:
        return None
    fr = env.frames[-1]
    if not entails(scope(env).values(), fr.goal):
        return f"leaves {fstr(fr.goal)} unprovable"
    if fr.kind == "or1":
        sc = {f"h{j + 1}": p for j, p in enumerate(env.prem)}
        for f in env.frames[:-1]:
            sc.update(f.names)
        if not entails(list(sc.values()) + [fr.ore_right], fr.goal):
            return f"second branch leaves {fstr(fr.goal)} unprovable"
    return None


def judge_action(env: Env, act: list[str]) -> tuple[str | None, Env | None]:
    """(reason it fails or None, the environment after it).  Fails = the environment rejects
    it, the per-step checker (sound w.r.t. Lean) rejects it, or it leaves an unprovable goal."""
    sc_err = step_check.check(env, act) if act and act[0] == "have" else None
    e2 = copy.deepcopy(env)
    ok, why = e2.apply(act)
    if not ok:
        return f"env: {why}", None
    if sc_err:
        return f"step_check: {sc_err}", e2
    d = dead_after(e2)
    if d:
        return f"dead: {d}", e2
    return None, e2


def canonical_names_ok(env: Env, act: list[str]) -> bool:
    if not act or act[0] != "have":
        return True
    nh, nb = next_names(env)
    if len(act) < 2 or act[1] != nh:
        return False
    j = act.index(":=") if ":=" in act else -1
    if j > 0 and j + 4 < len(act) and act[j + 1] == "(" and act[j + 2] == "fun":
        return act[j + 4] == nb
    if j > 0 and j + 6 < len(act) and act[j + 1] == "Or.elim":
        return act[j + 6] == nb
    return True


# --------------------------------------------------------------------------- skeletons
PRIM_START = {"(", "⟨", "⊢", "Or.inl", "Or.inr", "Or.elim", "Classical.byContradiction"}


class Skel:
    """Recursive-descent parser for reduction skeletons (PROPOSAL.md section 2)."""

    def __init__(self, toks: list[str]):
        self.t, self.i = toks, 0

    def peek(self):
        return self.t[self.i] if self.i < len(self.t) else None

    def eat(self, x=None):
        t = self.peek()
        if t is None or (x is not None and t != x):
            raise TraceError(f"skeleton: expected {x or 'a token'}, got {t!r}")
        self.i += 1
        return t

    def name(self):
        t = self.eat()
        if not is_name(t):
            raise TraceError(f"skeleton: expected a name, got {t!r}")
        return t

    def formula(self):
        f, self.i = pform(self.t, self.i)
        return f

    def skel(self):
        t = self.peek()
        if t == "have":
            self.eat()
            x = self.name()
            self.eat(":")
            a = self.formula()
            self.eat(":=")
            val = self.skel()
            self.eat(";")
            return ("have", x, a, val, self.skel())
        if t == "fun":
            return self.lam()
        node = self.prim()
        if self.peek() is not None and (self.peek() in PRIM_START or is_name(self.peek())):
            node = ("app", node, self.prim())
        if self.peek() in (".1", ".2", ".elim"):
            node = ("proj", node, self.eat())
        return node

    def lam(self):
        self.eat("fun")
        self.eat("(")
        x = self.name()
        self.eat(":")
        a = self.formula()
        self.eat(")")
        self.eat("=>")
        return ("lam", x, a, self.skel())

    def prim(self):
        t = self.peek()
        if t is not None and is_name(t):
            return ("name", self.eat())
        if t == "⊢":
            self.eat()
            return ("hole", self.formula())
        if t == "(":
            self.eat()
            node = self.skel()
            self.eat(")")
            return node
        if t == "⟨":
            self.eat()
            a = self.skel()
            self.eat(",")
            b = self.skel()
            self.eat("⟩")
            return ("pair", a, b)
        if t in ("Or.inl", "Or.inr"):
            self.eat()
            return ("inl" if t == "Or.inl" else "inr", self.prim())
        if t == "Or.elim":
            self.eat()
            d = self.prim()
            b1, b2 = self.prim(), self.prim()
            if b1[0] != "lam" or b2[0] != "lam":
                raise TraceError("skeleton: Or.elim branches must be ( fun ( x : A ) => ... )")
            return ("orelim", d, b1, b2)
        if t == "Classical.byContradiction":
            self.eat()
            return ("dne", self.prim())
        raise TraceError(f"skeleton: unexpected {t!r}")


class Checker:
    """Bidirectional type checking of a skeleton; collects holes as (local binders, formula)."""

    def __init__(self, sc: dict, binders: dict):
        self.sc, self.binders, self.holes = sc, binders, []

    def look(self, n, local):
        for x, a in reversed(local):
            if x == n:
                return a
        if n in self.sc:
            return self.sc[n]
        raise TraceError(f"skeleton: {n} is not in scope")

    def bind(self, x, a, local):
        if x in self.sc or x in self.binders or any(x == y for y, _ in local):
            raise TraceError(f"skeleton: binder {x} is not fresh")
        self.binders[x] = a
        return local + [(x, a)]

    def infer(self, node, local):
        k = node[0]
        if k == "name":
            return self.look(node[1], local)
        if k == "hole":
            self.holes.append((tuple(local), node[1]))
            return node[1]
        if k == "app":
            ft = self.infer(node[1], local)
            if ft[0] == "imp":
                self.check(node[2], ft[1], local)
                return ft[2]
            if ft[0] == "not":
                self.check(node[2], ft[1], local)
                return BOT
            raise TraceError(f"skeleton: {fstr(ft)} is not a function")
        if k == "proj" and node[2] in (".1", ".2"):
            xt = self.infer(node[1], local)
            if xt[0] != "and":
                raise TraceError(f"skeleton: {node[2]} on {fstr(xt)}")
            return xt[1] if node[2] == ".1" else xt[2]
        if k == "have":
            local2 = self._have(node, local)
            return self.infer(node[4], local2)
        raise TraceError(f"skeleton: cannot infer the type of a {k}")

    def _have(self, node, local):
        _, x, a, val, _ = node
        self.check(val, a, local)
        return self.bind(x, a, local)

    def check(self, node, want, local):
        k = node[0]
        if k == "lam":
            _, x, a, body = node
            if want[0] == "imp" and want[1] == a:
                self.check(body, want[2], self.bind(x, a, local))
                return
            if want[0] == "not" and want[1] == a:
                self.check(body, BOT, self.bind(x, a, local))
                return
            raise TraceError(f"skeleton: fun ( {x} : {fstr(a)} ) cannot prove {fstr(want)}")
        if k == "pair":
            if want[0] != "and":
                raise TraceError(f"skeleton: ⟨ , ⟩ cannot prove {fstr(want)}")
            self.check(node[1], want[1], local)
            self.check(node[2], want[2], local)
            return
        if k in ("inl", "inr"):
            if want[0] != "or":
                raise TraceError(f"skeleton: Or.inl/inr cannot prove {fstr(want)}")
            self.check(node[1], want[1] if k == "inl" else want[2], local)
            return
        if k == "orelim":
            dt = self.infer(node[1], local)
            if dt[0] != "or":
                raise TraceError(f"skeleton: Or.elim on {fstr(dt)}")
            for br, side in ((node[2], dt[1]), (node[3], dt[2])):
                _, x, a, body = br
                if a != side:
                    raise TraceError(f"skeleton: Or.elim branch binder {fstr(a)} != {fstr(side)}")
                self.check(body, want, self.bind(x, a, local))
            return
        if k == "dne":
            self.check(node[1], ("not", ("not", want)), local)
            return
        if k == "proj" and node[2] == ".elim":
            xt = self.infer(node[1], local)
            if xt != BOT:
                raise TraceError(f"skeleton: .elim on {fstr(xt)} (only False)")
            return
        if k == "have":
            self.check(node[4], want, self._have(node, local))
            return
        got = self.infer(node, local)
        if got != want:
            raise TraceError(f"skeleton: has type {fstr(got)}, goal is {fstr(want)}")


def hyps_of(sc: dict, local) -> list:
    return list(sc.values()) + [a for _, a in local]


# --------------------------------------------------------------------------- one think block
def split_lines(think: list[str]) -> list[list[str]]:
    if not think or think[0] != "<think>" or think[-1] != "</think>":
        raise TraceError("think block must be <think> ... </think>")
    body = think[1:-1]
    if "<think>" in body or "</think>" in body:
        raise TraceError("nested think tags")
    lines, cur = [], []
    for t in body:
        if t == "<nl>":
            if cur:
                lines.append(cur)
            cur = []
        else:
            cur.append(t)
    if cur:
        lines.append(cur)
    return lines


def check_step(env: Env, think: list[str], action: list[str], stats: dict) -> list[str]:
    """Validate one step's think block and action against the state `env`.  -> soft findings."""
    soft = []
    sc = scope(env)
    binders: dict = {}  # think-local binder -> formula
    holes = {((), env.frames[-1].goal)}  # (local binders, formula) that a goal line may refine
    pending = None  # the latest candidate without a verdict: (tokens, reason it fails or None)
    survivors = []
    flagged = set()
    open_invalid = []

    def close_pending(rejected: bool):
        nonlocal pending
        if pending is None:
            return
        act, why = pending
        if rejected and why is None:
            raise TraceError(f"<no> on a candidate that works: {' '.join(act)}")
        if not rejected and why is not None:
            raise TraceError(f"unrejected candidate fails ({why}): {' '.join(act)}")
        if rejected:
            stats["reject_" + why.split(":")[0]] = stats.get("reject_" + why.split(":")[0], 0) + 1
        else:
            survivors.append(act)
        pending = None

    for line in split_lines(think):
        if line == ["<no>"]:
            if pending is None:
                raise TraceError("<no> line without a pending candidate")
            close_pending(True)
            stats["lines_reject"] = stats.get("lines_reject", 0) + 1
            continue
        if line[0] in ("have", "exact"):
            close_pending(False)
            if not canonical_names_ok(env, line):
                raise TraceError(f"candidate does not use the canonical names: {' '.join(line)}")
            why, e2 = judge_action(env, line)
            pending = (line, why)
            if e2 is not None and len(e2.frames) > len(env.frames):
                # a box opener: its binder and goal may be examined before the verdict (lookahead)
                fr = e2.frames[-1]
                ((x, a),) = fr.names.items()
                if binders.get(x, a) == a and x not in sc:
                    binders[x] = a
                    holes.add((((x, a),), fr.goal))
            stats["lines_candidate"] = stats.get("lines_candidate", 0) + 1
            continue
        if "⊢" in line:
            k = line.index("⊢")
            ctx, i = [], 0
            while i < k:
                if not is_name(line[i]) or i + 1 >= k or line[i + 1] != ":":
                    raise TraceError(f"goal context: expected 'x : A', got {' '.join(line[:k])}")
                a, j = pform(line, i + 2)
                ctx.append((line[i], a))
                i = j
            if i != k:
                raise TraceError(f"goal context: malformed {' '.join(line[:k])}")
            for x, a in ctx:
                if binders.get(x) != a:
                    raise TraceError(f"goal context {x} : {fstr(a)} is not a think-local binder")
            g, j = pform(line, k + 1)
            rest = line[j:]
            no = bool(rest) and rest[-1] == "<no>"
            if no:
                rest = rest[:-1]
            local = tuple(ctx)
            hyps = list(sc.values()) + [a for _, a in ctx]
            if (local, g) not in holes:
                soft.append(f"goal refines nothing seen: {' '.join(line)}")
            stats["lines_goal"] = stats.get("lines_goal", 0) + 1
            if not rest:
                if no:
                    if entails(hyps, g):
                        raise TraceError(f"<no> on a provable goal: {' '.join(line)}")
                    flagged.add((local, g))
                    stats["no_goal"] = stats.get("no_goal", 0) + 1
                elif not entails(hyps, g):
                    open_invalid.append((local, g))
                continue
            if rest[0] != ":=":
                raise TraceError(f"goal line: expected := after the goal, got {rest[0]!r}")
            p = Skel(rest[1:])
            node = p.skel()
            if p.i != len(p.t):
                raise TraceError(f"skeleton: trailing tokens {' '.join(p.t[p.i :])}")
            if node == ("hole", g):
                soft.append(f"vacuous reduction: {' '.join(line)}")
            ch = Checker(sc, binders)
            try:
                ch.check(node, g, list(ctx))
                err = None
            except TraceError as e:
                err = str(e)
            bad = [h for h in ch.holes if not entails(hyps_of(sc, h[0]), h[1])]
            if no:
                if err is None and not bad:
                    raise TraceError(f"<no> on a sound reduction: {' '.join(line)}")
                flagged.update(bad)
                stats["no_reduction"] = stats.get("no_reduction", 0) + 1
            else:
                if err is not None:
                    raise TraceError(f"{err}  [in: {' '.join(line)}]")
                open_invalid.extend(bad)
            holes.update(ch.holes)
            stats["holes"] = stats.get("holes", 0) + len(ch.holes)
            continue
        # a look
        if len(line) < 3 or not is_name(line[0]) or line[1] != ":":
            raise TraceError(f"unknown line: {' '.join(line)}")
        a, j = pform(line, 2)
        if j != len(line):
            raise TraceError(f"look: trailing tokens in {' '.join(line)}")
        known = sc.get(line[0], binders.get(line[0]))
        if known != a:
            have = fstr(known) if known else "nothing"
            raise TraceError(f"false look: {' '.join(line)} (in scope: {have})")
        stats["lines_look"] = stats.get("lines_look", 0) + 1
    close_pending(False)
    for h in dict.fromkeys(open_invalid):
        if h not in flagged:
            soft.append(f"unprovable goal never flagged <no>: {fstr(h[1])}")
    # the action
    if not canonical_names_ok(env, action):
        raise TraceError(f"action does not use the canonical names: {' '.join(action)}")
    why, _ = judge_action(env, action)
    if why:
        raise TraceError(f"action fails ({why}): {' '.join(action)}")
    if survivors and survivors[-1] != action:
        raise TraceError("action is not the last unrejected candidate")
    return soft


# --------------------------------------------------------------------------- one trace
def validate(rec: dict, lean: bool = True, strict: bool = False) -> dict:
    """One trace record -> report {ok, error, soft, stats, ...}.  Never raises on a bad trace.
    strict: soft findings (unanchored goals, unprovable goals never flagged) are errors too."""
    out = {"id": rec["id"], "ok": False, "error": None, "soft": [], "stats": {}}
    stats = out["stats"]
    try:
        env = Env(rec["prompt"], canon=True, base=0, assign=False)
        steps = rec["steps"]
        think_lens, pair_lens = [], []
        for s_i, st in enumerate(steps):
            if env.done:
                raise TraceError(f"step {s_i}: the proof is already finished")
            think = tokens(st["think"])
            if not think or think[0] != "<think>":
                think = ["<think>"] + think
            if think[-1] != "</think>":
                think = think + ["</think>"]
            action = tokens(st["action"])
            bad = [t for t in think + action if t not in VOCAB]
            if bad:
                raise TraceError(f"step {s_i}: tokens outside the vocabulary: {sorted(set(bad))}")
            if "<nl>" in action or any(t in NEW_TOKENS for t in action):
                raise TraceError(f"step {s_i}: action contains think tokens")
            state = env.state_tokens()
            try:
                soft = check_step(env, think, action, stats)
            except TraceError as e:
                raise TraceError(f"step {s_i}: {e}") from None
            out["soft"] += [f"step {s_i}: {x}" for x in soft]
            if strict and soft:
                raise TraceError(f"step {s_i}: (strict) {soft[0]}")
            think_lens.append(len(think))
            pair_lens.append(len(state) + len(think) + len(action) + 1)
            ok, why = env.apply(action)
            if not ok:
                raise TraceError(f"step {s_i}: replay failed: {why}")
        if not env.done:
            raise TraceError("the actions do not finish the proof")
        text = TOK.text(env.text)
        out["lean_text"] = text
        out["n_steps"] = len(steps)
        out["think_tokens"] = think_lens
        out["pair_tokens_max"] = max(pair_lens)
        out["think_tokens_total"] = sum(think_lens)
        out["action_tokens_total"] = sum(len(tokens(s["action"])) + 1 for s in steps)
        if lean:
            from lean_gate import lean_check

            res, _, _ = lean_check([(TOK.statement(rec["prompt"]), text)])
            out["lean"] = bool(res[0])
            if not res[0]:
                raise TraceError("Lean rejects the assembled proof")
        out["ok"] = True
    except (TraceError, ParseFail) as e:
        out["error"] = str(e)
    return out


# --------------------------------------------------------------------------- CLI
def show(prompt: str, proof: str) -> str:
    steps, _, _ = decompose(prompt, proof, canon=True)
    rows = []
    for i, (s, a, _) in enumerate(steps):
        rows.append(f"step {i}\n  state : {' '.join(s)}\n  action: {' '.join(a)}")
    return "\n".join(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate")
    v.add_argument("traces")
    v.add_argument("--no-lean", action="store_true")
    v.add_argument("--strict", action="store_true", help="soft findings are errors")
    v.add_argument("--json")
    s = sub.add_parser("show")
    s.add_argument("source", choices=["k12"])
    s.add_argument("index", type=int)
    a = ap.parse_args(argv)
    if a.cmd == "show":
        with open(K12) as f:
            row = json.loads(next(itertools.islice(f, a.index, a.index + 1)))
        print(row["thm"])
        print(show(row["prompt"], row["proof"]))
        return 0
    with open(a.traces) as f:
        recs = [json.loads(line) for line in f if line.strip()]
    reports = [validate(r, lean=not a.no_lean, strict=a.strict) for r in recs]
    n_ok = sum(r["ok"] for r in reports)
    for r in reports:
        tag = "PASS" if r["ok"] else "FAIL"
        extra = (
            f"steps {r['n_steps']}, think {r['think_tokens_total']} tok, "
            f"max pair {r['pair_tokens_max']} tok, soft {len(r['soft'])}"
            if r["ok"]
            else r["error"]
        )
        print(f"{tag} {r['id']}: {extra}")
        for x in r["soft"]:
            print(f"     soft: {x}")
    print(f"{n_ok} / {len(reports)} traces valid")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(reports, f, indent=1, ensure_ascii=False)
    return 0 if n_ok == len(reports) else 1


if __name__ == "__main__":
    sys.exit(main())
