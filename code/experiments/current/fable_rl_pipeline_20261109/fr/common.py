"""Shared pieces of the fable_rl pipeline: tokenizer, step checker, Lean runner, trace analysis.

Formats are Dan's (vendor_dan/): the `lean_staten` state/action tokens and the `lean_seq` proof
text that Lean judges. Nothing here reads a data file.

Step checker (training time only): `try_step` applies one sampled action to an `Env` and says
whether it was syntactically refused by the environment ("syntax"), accepted by the environment
but a certain Lean type error ("invalid", found with Dan's reject-only `lean_prefilter._term`),
or fine ("ok"). The solver never calls the type check; it uses the model's own verifier head.
"""

from __future__ import annotations

import bisect
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "vendor_dan"))

import lean_prefilter as LP  # noqa: E402
import patterns  # noqa: E402
from lean_tok import MAXN, LeanTokenizer, ParseFail, fnd, ftoks  # noqa: E402
from nd2lean import parse_prompt, translate  # noqa: E402
from state_env import Env, decompose, is_name, parse_ftoks  # noqa: E402

TOK = LeanTokenizer("lean_staten")
VOCAB = TOK.vocab_size
PAD, EOS = TOK.pad, TOK.eos
REF0 = TOK.ref0  # id of n1; names are the last MAXN ids
ACT_ID = TOK.stoi["<act>"]
TURNSTILE = "⊢"


# ----------------------------------------------------------------------------- prompts
def make_prompt(prem, concl) -> str:
    """Premise and conclusion formula tuples -> 'THM p1 , p2 SEQ c PRF' (Dan's prompt format)."""
    body = " , ".join(fnd(p) for p in prem)
    return ("THM " + body + " SEQ " if prem else "THM SEQ ") + fnd(concl) + " PRF"


def thm_to_prompt(thm: str) -> str:
    """'p1 , p2 |- c' (the `thm` field of y) -> prompt."""
    left, right = thm.split("|-")
    left = left.strip()
    return ("THM " + left + " SEQ " if left else "THM SEQ ") + right.strip() + " PRF"


# ----------------------------------------------------------------------------- step checker
def _is_prem(t: str) -> bool:
    return len(t) > 1 and t[0] == "h" and t[1:].isdigit()


def step_kind(act) -> str:
    """'exact' | 'pr' | 'box' | 'ore' | 'atom' | 'bad' from the action's tokens alone."""
    if not act:
        return "bad"
    if act[0] == "exact":
        return "exact"
    if act[0] != "have" or ":=" not in act:
        return "bad"
    j = act.index(":=")
    if j + 1 >= len(act):
        return "bad"
    t = act[j + 1]
    if t == "(":
        return "box"
    if t == "Or.elim":
        return "ore"
    if _is_prem(t):
        return "pr"
    return "atom"


def type_ok(env: Env, act) -> bool:
    """Is the term of an atomic or premise `have` well typed in the environment's scope?
    (Dan's pre-filter: False only on a certain Lean error.)"""
    j = act.index(":=")
    try:
        f, used = parse_ftoks(act, 3)
    except ParseFail:
        return True
    scope = {}
    for fr in env.frames:
        scope.update(fr.names)
    p = LP._P(list(act[j + 1 :]))
    try:
        LP._term(p, scope, f, env.prem)
    except LP._Reject:
        return False
    except (LP._Pass, RecursionError):
        return True
    return True


def try_step(env: Env, act, check: bool = True) -> str:
    """Apply `act` to `env` if the environment accepts it and (with `check`) its term type-checks.
    -> 'ok' (applied) | 'syntax' (env refused; env unchanged) | 'invalid' (type error; unchanged)."""
    kind = step_kind(act)
    if kind == "bad":
        return "syntax"
    if check and kind in ("atom", "pr"):
        # the environment's own (non-mutating) shape checks first, so that 'invalid' means
        # "the environment would have accepted this step and Lean would not"
        try:
            j = act.index(":=")
            if not is_name(act[1]) or act[2] != ":" or act[-1] != ";":
                return "syntax"
            f, used = parse_ftoks(act, 3)
            if 3 + used != j:
                return "syntax"
            if kind == "atom":
                env._check_atomic(list(act[j + 1 : -1]))
            else:
                k = int(act[j + 1][1:])
                if j + 3 != len(act) or len(env.frames) != 1 or env.seen_non_pr:
                    return "syntax"
                if k != env.n_pr + 1 or k > len(env.prem):
                    return "syntax"
        except (ParseFail, IndexError, KeyError, ValueError):
            return "syntax"
        if not type_ok(env, act):
            return "invalid"
    steps = env.steps
    ok, why = env.apply(list(act))
    if not ok:
        env.failed = None
        env.steps = steps
        env.last_why = why
        return "syntax"
    return "ok"


# ----------------------------------------------------------------------------- Lean
LEAN = os.path.expanduser(os.environ.get("LEAN_BIN", "~/.elan/bin/lean"))
LEAN_CHUNK = int(os.environ.get("LEAN_GATE_CHUNK", "400"))
_ERR = re.compile(r"^[^\n]*?:(\d+):\d+: error", re.M)
_PRELUDE = "set_option linter.unusedVariables false\nset_option maxRecDepth 4000\n"


def cpu_quota() -> int:
    try:
        n = len(os.sched_getaffinity(0))
    except AttributeError:
        n = os.cpu_count() or 2
    for fq, fp in (
        ("/sys/fs/cgroup/cpu.max", None),
        ("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "/sys/fs/cgroup/cpu/cpu.cfs_period_us"),
    ):
        try:
            if fp is None:
                q, per = open(fq).read().split()[:2]
            else:
                q, per = open(fq).read().strip(), open(fp).read().strip()
            if q not in ("max", "-1"):
                n = min(n, max(1, int(int(q) / int(per))))
            break
        except (OSError, ValueError):
            continue
    return n


LEAN_WORKERS = int(os.environ.get("LEAN_GATE_WORKERS", "0")) or min(8, cpu_quota())


def _lean_run(srcs, workdir, tag):
    """One Lean process over a list of theorem sources (the method of Dan's lean_gate._run)."""
    text = _PRELUDE
    starts = []
    for k, s in enumerate(srcs):
        starts.append(text.count("\n") + 1)
        text += s.replace("theorem t ", f"theorem t{k} ", 1).rstrip("\n") + "\n"
    fn = os.path.join(workdir, f"c_{tag}.lean")
    with open(fn, "w") as f:
        f.write(text)
    try:
        p = subprocess.run(
            [LEAN, "-j", "1", "-DmaxErrors=100000000", fn],
            capture_output=True,
            text=True,
            timeout=60 + 2 * text.count("\n"),
        )
        o = p.stdout + p.stderr
        rc = p.returncode
    except subprocess.TimeoutExpired:
        o = ""
        rc = -9
    os.remove(fn)
    bad = set()
    stray = False
    for m in _ERR.finditer(o):
        k = bisect.bisect_right(starts, int(m.group(1))) - 1
        if 0 <= k < len(srcs):
            bad.add(k)
        else:
            stray = True
    crashed = rc not in (0, 1) or (rc == 1 and not bad) or stray
    if crashed:
        if len(srcs) == 1:
            return [False]
        h = len(srcs) // 2
        return _lean_run(srcs[:h], workdir, tag + "a") + _lean_run(srcs[h:], workdir, tag + "b")
    return [k not in bad for k in range(len(srcs))]


def lean_check(items) -> list[bool]:
    """items: (prompt, lean_seq text of the tactic block) -> Lean 4 core's verdict on each."""
    if not items:
        return []
    uniq = {}
    for it in items:
        uniq.setdefault(it, len(uniq))
    keys = list(uniq)
    srcs = [f"{TOK.statement(p)} {tx}" for p, tx in keys]
    n = max(1, min(LEAN_WORKERS, len(srcs) // 50))
    size = max(1, min(LEAN_CHUNK, -(-len(srcs) // n)))
    chunks = [(srcs[i : i + size], i) for i in range(0, len(srcs), size)]
    wd = tempfile.mkdtemp(prefix="frlean_")
    try:
        with ThreadPoolExecutor(LEAN_WORKERS) as ex:
            res = list(ex.map(lambda c: _lean_run(c[0], wd, str(c[1])), chunks))
    finally:
        shutil.rmtree(wd, ignore_errors=True)
    flat = [x for r in res for x in r]
    return [flat[uniq[it]] for it in items]


def lean_check_nd(items) -> list[bool]:
    """items: (prompt, ND proof body) -> Lean's verdict by the route Dan's `lean_judge.judge_many`
    takes: `nd2lean.translate(prompt, nd, require_all_pr=False)`, then Lean 4 core."""
    idx, srcs = [], []
    for i, (p, nd) in enumerate(items):
        if not nd or nd.startswith("LEAN"):
            continue
        try:
            srcs.append(translate(p, nd, require_all_pr=False))
            idx.append(i)
        except Exception:
            continue
    out = [False] * len(items)
    if srcs:
        n = max(1, min(LEAN_WORKERS, len(srcs) // 50))
        size = max(1, min(LEAN_CHUNK, -(-len(srcs) // n)))
        chunks = [(srcs[i : i + size], i) for i in range(0, len(srcs), size)]
        wd = tempfile.mkdtemp(prefix="frlean_")
        try:
            with ThreadPoolExecutor(LEAN_WORKERS) as ex:
                res = list(ex.map(lambda c: _lean_run(c[0], wd, str(c[1])), chunks))
        finally:
            shutil.rmtree(wd, ignore_errors=True)
        for i, ok in zip(idx, [x for r in res for x in r]):
            out[i] = bool(ok)
    return out


# ----------------------------------------------------------------------------- traces
class Trace:
    """A replay of one attempt's applied actions, with what each step saw and cited.

    steps[i] = dict(state, act, depth, kind, cites, opener); `defs` maps a definition id to the
    step that introduced it. Definition ids: ('h', i) the `have` of step i (for a box, its opening
    step), ('b', i) the binder a box-opening step introduces, ('b2', i) the second Or.elim binder
    the environment introduces when step i closes the first branch.
    """

    def __init__(self, prompt: str, acts, base: int):
        self.prompt = prompt
        self.base = base
        self.acts = [list(a) for a in acts]
        env = Env(prompt, canon=True, base=base)
        scopes = [{}]
        open_stack = []
        self.steps = []
        self.closers = {}  # opening step -> [closing exact steps]
        self.top_haves = []  # (step index at which the have became available, def id, formula)
        self.ok = True

        def lookup(n):
            for sc in reversed(scopes):
                if n in sc:
                    return sc[n]
            raise KeyError(n)

        for i, a in enumerate(self.acts):
            depth = len(env.frames) - 1
            rec = {
                "state": env.state_tokens(),
                "act": a,
                "depth": depth,
                "kind": step_kind(a),
                "cites": [],
                "opener": None,
            }
            fkind = env.frames[-1].kind
            try:
                if rec["kind"] == "exact":
                    rec["cites"] = [lookup(a[1])]
                    if depth > 0:
                        rec["opener"] = open_stack[-1]
                        self.closers.setdefault(open_stack[-1], []).append(i)
                elif rec["kind"] == "ore":
                    j = a.index(":=")
                    rec["cites"] = [lookup(a[j + 2])]
                elif rec["kind"] == "atom":
                    j = a.index(":=")
                    rec["cites"] = [lookup(x) for x in a[j + 1 : -1] if is_name(x)]
            except KeyError:
                self.ok = False
                break
            ok, _ = env.apply(list(a))
            if not ok:
                self.ok = False
                break
            self.steps.append(rec)
            k = rec["kind"]
            if k == "exact":
                if depth == 0:
                    pass
                elif fkind == "or1":
                    scopes[-1] = {env.frames[-1].last: ("b2", i)}
                else:
                    scopes.pop()
                    op = open_stack.pop()
                    nm = self.acts[op][1]
                    scopes[-1][nm] = ("h", op)
                    if len(scopes) == 1:
                        self.top_haves.append((i, ("h", op), env.frames[0].names[nm]))
            elif k in ("box", "ore"):
                j = a.index(":=")
                binder = a[j + 4] if k == "box" else a[j + 6]
                scopes.append({binder: ("b", i)})
                open_stack.append(i)
            else:  # pr | atom
                scopes[-1][a[1]] = ("h", i)
                if depth == 0 and k == "atom":
                    self.top_haves.append((i, ("h", i), env.frames[0].names[a[1]]))
        self.env = env
        self.done = self.ok and env.done

    def text(self) -> str:
        return TOK.text(self.env.text)

    def closure(self, uid) -> set[int]:
        """Steps the definition `uid` depends on (premise lines not included)."""
        useful = set()
        stack = [uid]
        while stack:
            u = stack.pop()
            if u[0] != "h":
                continue
            i = u[1]
            if i in useful:
                continue
            useful.add(i)
            stack.extend(self.steps[i]["cites"])
            for e in self.closers.get(i, ()):
                useful.add(e)
                stack.extend(self.steps[e]["cites"])
        return useful

    def useful_for_proof(self) -> set[int]:
        """Steps of a finished attempt that its final `exact` depends on, plus every premise line."""
        assert self.done
        last = len(self.steps) - 1
        u = self.closure(self.steps[last]["cites"][0])
        u.add(last)
        u |= {i for i, s in enumerate(self.steps) if s["kind"] == "pr"}
        return u

    def lemma(self, k: int):
        """Hindsight relabel on the k-th top-level `have`: the same attempt read as a proof of that
        line. -> (new prompt, action list ending in `exact`, useful step set, end step) or None."""
        end, uid, f = self.top_haves[k]
        nm = self.acts[uid[1]][1]
        prompt = make_prompt(self.env.prem, f)
        acts = self.acts[: end + 1] + [["exact", nm]]
        useful = self.closure(uid) | {i for i in range(end + 1) if self.steps[i]["kind"] == "pr"}
        return prompt, acts, useful, end


def swap_goal(state, goal_toks):
    """A top-frame state with its goal replaced (for hindsight relabels)."""
    j = len(state) - 1 - state[::-1].index(TURNSTILE)
    return state[: j + 1] + list(goal_toks) + ["<act>"]


def pruned_nd(nd: str):
    """ND proof -> its dependency-pruned ND proof (Dan's patterns.prune_lines), or None."""
    lines = patterns.parse(nd)
    if lines is None:
        return None
    return patterns.lines_to_text(patterns.prune_lines(lines))


def nd_pairs(prompt: str, nd: str):
    """ND proof -> ([(state tokens, action tokens)], lean_seq text) in canonical names."""
    steps, toks, _ = decompose(prompt, nd, canon=True)
    return [(s, a) for s, a, _ in steps], TOK.text(toks)


def subgoal_variant(prompt: str, nd: str, max_prem: int = 6):
    """A proof whose last line closes an implication or negation box, read as a proof of that
    box's subgoal: 'premises , assumption |- body goal' (for a negation box the goal is falsum).
    The box's assumption line becomes a premise line; nothing else changes.
    -> (new prompt, new ND proof) or None when the proof does not end that way."""
    lines = patterns.parse(nd)
    if not lines or len(lines) < 2:
        return None
    last = lines[-1]
    if last["rule"] not in ("IMPI", "NEGI") or last["depth"] != 0 or len(last["refs"]) != 2:
        return None
    s, e = last["refs"]
    if lines[-2]["idx"] != e:
        return None
    prem, _ = parse_prompt(prompt)
    if len(prem) + 1 > max_prem:
        return None
    pos = {ln["idx"]: i for i, ln in enumerate(lines)}
    if s not in pos or lines[pos[s]]["rule"] != "AS" or lines[pos[s]]["depth"] != 1:
        return None
    body = lines[pos[s] : -1]
    if any(ln["depth"] < 1 for ln in body):
        return None
    hyp = dict(lines[pos[s]])
    hyp.update(depth=0, rule="PR", refs=[])
    n_pr = sum(1 for ln in lines if ln["rule"] == "PR")
    if any(ln["rule"] == "PR" for ln in lines[n_pr:]):
        return None
    out = [dict(ln) for ln in lines[:n_pr]] + [hyp] + [dict(ln) for ln in lines[n_pr : pos[s]]]
    for ln in body[1:]:
        d = dict(ln)
        d["depth"] = ln["depth"] - 1
        out.append(d)
    goal = last["formula"][2] if last["rule"] == "IMPI" else ("bot",)
    if out[-1]["formula"] != goal or out[-1]["depth"] != 0:
        return None
    return make_prompt(list(prem) + [hyp["formula"]], goal), patterns.lines_to_text(out)


def enc(toks) -> bytes:
    return bytes(TOK.stoi[t] for t in toks)


def n_lines(nd: str) -> int:
    return sum(1 for t in nd.split() if t == ";")


__all__ = [
    "TOK", "VOCAB", "PAD", "EOS", "REF0", "ACT_ID", "MAXN", "Env", "ParseFail", "Trace",
    "make_prompt", "thm_to_prompt", "parse_prompt", "step_kind", "type_ok", "try_step",
    "lean_check", "lean_check_nd", "subgoal_variant", "ftoks", "swap_goal", "pruned_nd", "nd_pairs", "enc", "n_lines", "decompose", "time",
]  # fmt: skip
