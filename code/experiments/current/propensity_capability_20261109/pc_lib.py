"""Proof directions: definitions, applicability, opening actions, and classifiers of an attempt.

Built on Dan's code at 13e41bbd (read-only, `$KD/dan_code_13e41bbd`) and on the factor read's `factor_lib.py`
(`../factor_read_20261106/`, imported, not copied): `Trace` (replays actions through Dan's `Env` and keeps the ND lines
the text denotes), `run_env` (Dan's `env_generate` with a forced prefix), `score_rows` / `name_positions` (exact
log-probability of an action, name tokens separable).

A direction is a move on the goal in focus. In Dan's state environment every box is opened by one `have` action:

    reductio     have n : ( ¬ ( ¬ G ) ) := ( fun ( x : ( ¬ G ) ) => by      a NEGI box on ¬¬G, binder ¬G
    negation     have n : ( ¬ A ) := ( fun ( x : A ) => by                  G = ¬A
    conditional  have n : ( A → B ) := ( fun ( x : A ) => by                G = A → B
    cases        have n : G := Or.elim p ( fun ( x : A ) => by              p names a premise A ∨ B
    direct       the first proof action opens no box (a `have` with a term, or `exact`); never forced

Premise lines (`have n : F := h<k> ;`) are not proof actions: the env allows them only first, in order, at depth 0,
and a premise that is not restated cannot be cited. "The first action" of an attempt is therefore its first
non-premise action, and a forced attempt's prefix is every premise line, in order, then the opening.

Every judgement here is made on the ND lines the env's actions denote (Dan's parser semantics, `Trace.lines` equals
`patterns.parse(env.nd())`), with the verifier for validity, never by string matching on the text.

Proof text stays in output files under /scratch.
"""

# ruff: noqa: E501, SIM115, B905, B007, SIM300
from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

FR_DIR = Path(__file__).resolve().parent.parent / "factor_read_20261106"
if (
    str(FR_DIR) not in sys.path
):  # appended, not first: that folder has its own analyze.py and fig.py
    sys.path.append(str(FR_DIR))

from factor_lib import (  # noqa: E402
    BOT,
    DAN,
    KD,
    NAME_BASE_MAX,
    Attempt,
    Trace,
    name_positions,
    neg,
    run_env,
    score_rows,
    shift_action,
)
from lean_tok import ftoks  # type: ignore[import-not-found]  # noqa: E402
from patterns import prune_lines  # type: ignore[import-not-found]  # noqa: E402
from state_env import Env, parse_ftoks, prompt_parts  # type: ignore[import-not-found]  # noqa: E402

DIRS = ("reductio", "negation", "conditional", "cases", "direct")
FORCED = ("reductio", "negation", "conditional", "cases")
BASES = tuple(range(NAME_BASE_MAX + 1))  # the env draws the name base ~ U[0, 32] per attempt


# ----------------------------------------------------------------------------- the theorem
def theorem(prompt: str) -> tuple[list[tuple], tuple]:
    """(premises, goal) as formula tuples, from Dan's prompt parser (nd2lean.parse_prompt via state_env)."""
    prem, concl, _ = prompt_parts(prompt)
    return list(prem), concl


def applicable(prompt: str) -> dict[str, bool]:
    """Which directions can be forced as the first proof action. Read off the parsed formulas."""
    prem, goal = theorem(prompt)
    return {
        "reductio": True,
        "negation": goal[0] == "not",
        "conditional": goal[0] == "imp",
        "cases": any(p[0] == "or" for p in prem),
        "direct": True,
    }


def premise_actions(prompt: str) -> list[list[str]]:
    """Every premise restated, in order, base-0 canonical names n1..nm."""
    prem, _ = theorem(prompt)
    return [
        ["have", f"n{i + 1}", ":"] + ftoks(p) + [":=", f"h{i + 1}", ";"] for i, p in enumerate(prem)
    ]


def opening(prompt: str, d: str) -> list[str] | None:
    """d's opening as the first proof action, after all m premises (base 0: the step's own name is n{m+1}, its
    binder n{m+2}; the env overrides both anyway). None when d does not apply."""
    if d not in FORCED or not applicable(prompt)[d]:
        return None
    prem, g = theorem(prompt)
    m = len(prem)
    h, b = f"n{m + 1}", f"n{m + 2}"

    def box(declared: tuple, binder: tuple) -> list[str]:
        return (
            ["have", h, ":"]
            + ftoks(declared)
            + [":=", "(", "fun", "(", b, ":"]
            + ftoks(binder)
            + [")", "=>", "by"]
        )

    if d == "reductio":
        return box(neg(neg(g)), neg(g))
    if d in ("negation", "conditional"):
        return box(g, g[1])
    j = next(i for i, p in enumerate(prem) if p[0] == "or")  # the first disjunctive premise
    return (["have", h, ":"] + ftoks(g) + [":=", "Or.elim", f"n{j + 1}", "(", "fun", "(", b, ":"]
            + ftoks(prem[j][1]) + [")", "=>", "by"])  # fmt: skip


def forced_prefix(prompt: str, d: str) -> list[list[str]]:
    op = opening(prompt, d)
    assert op is not None, (prompt, d)
    return premise_actions(prompt) + [op]


# ----------------------------------------------------------------------------- moves in a trace
class MTrace(Trace):
    """`Trace` that also keeps, per applied action, the goal in focus before it and the box it opened."""

    def __init__(self, prompt: str, base: int = 0) -> None:
        super().__init__(prompt, base)
        self.goal_before: list[tuple] = []
        self.depth_before: list[int] = []
        self.opened: list[
            dict | None
        ] = []  # per action: the new box's bookkeeping (s, and j for Or.elim)

    def apply(self, raw: list[str]) -> bool:
        g, d, nf = self.env.frames[-1].goal, self.depth, len(self.env.frames)
        ok = super().apply(raw)
        if ok:
            self.goal_before.append(g)
            self.depth_before.append(d)
            act = self.acts[-1]
            opened = act[0] == "have" and len(self.env.frames) == nf + 1
            self.opened.append(dict(self.box[-1]) if opened else None)
        return ok


def replay_m(prompt: str, actions: list[list[str]], base: int = 0) -> MTrace:
    tr = MTrace(prompt, base)
    for a in actions:
        if not tr.apply(a):
            break
    return tr


@dataclass
class Move:
    kind: str  # reductio | negation | conditional | cases | cases_derived | lemma
    act: int  # action index in tr.acts
    depth: int  # depth of the frame it was opened in (0 = top)
    goal: tuple  # the goal in focus when it was opened
    target: tuple  # what the move proves: X for a reductio (box on ¬¬X, binder ¬X), else the declared formula
    on_goal: bool  # target == the goal in focus
    s: int  # ND index of the box's AS line (the first branch's, for cases)
    close: int | None  # ND index of the NEGI / IMPI / ORE line that closes it, if closed
    dn: list[
        int
    ]  # reductio: every DN line concluding X that cites the NEGI line (through R copies)


def moves(tr: MTrace) -> list[Move]:
    """Every box the attempt opened, labelled by direction:

    negation     a NEGI box whose declared formula ¬A is the goal in focus (binder A)
    conditional  an IMPI box whose declared formula A → B is the goal in focus (binder A)
    reductio     a NEGI box declaring ¬¬X with binder ¬X, for any X ≠ ⊥ (the goal in focus or a lemma: the σ notion
                 of the factor read and Dan's `patterns.reductio`), unless ¬¬X is itself the goal in focus (negation)
    cases        an Or.elim on a disjunction that is a premise line (following R copies), whatever it proves;
                 on a derived disjunction it is `cases_derived`, which is no direction
    lemma        any other box (an implication or negation introduction for a formula that is not the goal in focus)

    Introductions on lemmas are plumbing in these proofs (most boxes are), so negation and conditional count only on
    the goal in focus; reductio and a premise case split are strategies whatever they prove."""
    out = []
    for a, (act, ob) in enumerate(zip(tr.acts, tr.opened)):
        if ob is None:
            continue
        declared, used = parse_ftoks(act, 3)
        t = act[3 + used + 1]
        s = ob["s"]
        binder = tr.fm(s)
        g = tr.goal_before[a]
        kind, target = "lemma", declared
        if t == "(":
            if declared == g and g[0] == "not" and binder == g[1]:
                kind = "negation"
            elif declared == g and g[0] == "imp" and binder == g[1]:
                kind = "conditional"
            elif (
                declared[0] == "not"
                and declared[1][0] == "not"
                and binder == declared[1]
                and declared[1][1] != BOT
            ):
                kind, target = "reductio", declared[1][1]
        elif t == "Or.elim":
            kind = "cases" if tr.origin(ob["j"])["rule"] == "PR" else "cases_derived"
        close = None
        for ln in tr.lines[s:]:
            if t == "(" and ln["rule"] in ("NEGI", "IMPI") and ln["refs"][0] == s:
                close = ln["idx"]
                break
            if t == "Or.elim" and ln["rule"] == "ORE" and ln["refs"][1] == s:
                close = ln["idx"]
                break
        dn = []
        if kind == "reductio" and close is not None:
            dn = [ln["idx"] for ln in tr.lines[close:]
                  if ln["rule"] == "DN" and ln["formula"] == target and tr.origin(ln["refs"][0])["idx"] == close]  # fmt: skip
        out.append(Move(kind, a, tr.depth_before[a], g, target, target == g, s, close, dn))
    return out


def _is_pr(act: list[str]) -> bool:
    if act[0] != "have":
        return False
    _, used = parse_ftoks(act, 3)
    t = act[3 + used + 1]
    return t[0] == "h" and t[1:].isdigit()


def classify(
    prompt: str, base: int, actions: list[list[str]], accepted: bool | None = None
) -> dict:
    """One attempt (every applied action, a forced prefix included, at the attempt's name base) ->

    n_pr       premise lines it restated before its first proof action
    first      direction of its first proof action, when that action is exactly d's opening (`opening`) as the env
               names it: one of DIRS; `other` (any other box: a lemma, a reductio on a formula that is not the
               theorem's goal, a case split toward a lemma or on a derived disjunction); `invalid` (a restated premise
               the verifier rejects); None (the attempt never got a proof action accepted)
    opens      directions it takes anywhere (see `moves`), at any depth, every line up to and including the box's
               assumption valid. `direct` = it opens none of the four.
    through    directions a finished proof goes through: the move's closing line (for reductio, the DN line that
               cites its NEGI line) is valid and on the dependency path of the conclusion (Dan's `prune_lines`).
               `direct` = finished and it goes through none of the four and opens none (no detour either).
    done       the env saw a finished proof (Lean decides acceptance, not this)

    Validity: `accepted=True` means Lean accepted the finished proof, so every line counts as valid; otherwise each
    line is checked by the verifier on the prefix ending there (`Trace.valid_through`)."""
    tr = replay_m(prompt, actions, base)
    cache: dict[int, bool] = {}

    def valid(n: int) -> bool:
        if accepted:
            return True
        if n not in cache:
            cache[n] = tr.valid_through(n)
        return cache[n]

    ms = moves(tr)
    n_pr = sum(1 for a in tr.acts if _is_pr(a))
    first = None
    if len(tr.acts) > n_pr:
        a0 = n_pr
        m0 = next((m for m in ms if m.act == a0), None)
        if n_pr and not valid(n_pr):
            first = "invalid"
        elif m0 is None:
            first = "direct"
        elif m0.kind in FORCED and m0.on_goal:
            first = m0.kind
        else:
            first = "other"
    opens = {m.kind for m in ms if m.kind in FORCED and valid(m.s)}
    through: set[str] = set()
    if tr.env.done and tr.lines:
        dep = {ln["idx"] for ln in prune_lines(tr.lines)}
        for m in ms:
            if m.kind not in FORCED:
                continue
            ends = m.dn if m.kind == "reductio" else [m.close] if m.close is not None else []
            if any(x in dep and valid(x) for x in ends):
                through.add(m.kind)
        if not through and not opens:
            through.add("direct")
    if not opens:
        opens.add("direct")
    return {"n_pr": n_pr, "first": first, "opens": sorted(opens), "through": sorted(through), "done": tr.env.done,
            "n_actions": len(tr.acts), "fail": tr.fail,
            "moves": [[m.kind, m.act, m.depth, m.on_goal, m.close is not None, bool(m.dn)] for m in ms]}  # fmt: skip


def attempt_actions(prefix: list[list[str]], att: Attempt) -> list[list[str]]:
    """The actions an attempt applied, at its base: the forced prefix shifted to the base, then the model's actions
    that ended in <eos> (run_env applies only those; the first refused one stops the replay as it stopped the env)."""
    return [shift_action(a, att.base) for a in prefix] + [
        x for x, e in zip(att.raw, att.ended) if e
    ]


def gen_tokens(att: Attempt) -> int:
    """Tokens the model generated in the attempt: each action's tokens plus its <eos> when it came."""
    return sum(len(x) + (1 if e else 0) for x, e in zip(att.raw, att.ended))


# ----------------------------------------------------------------------------- exact first-move probabilities
def first_move_rows(tok, prompt: str, base: int, ds: list[str]) -> tuple[list, list, list]:
    """Rows for the scorer at one name base: each premise action from its own state (the prefix's probability), then
    each direction's opening from the state after all premises. -> (rows, name positions, labels)."""
    from state_sample import prompt_ids  # type: ignore[import-not-found]

    e = Env(prompt, canon=True, base=base, assign=True)
    rows, npos, labels = [], [], []
    for i, a in enumerate(premise_actions(prompt)):
        sa = shift_action(a, base)
        rows.append((prompt_ids(tok, e), [tok.stoi[t] for t in sa] + [tok.eos]))
        npos.append(name_positions(sa))
        labels.append(f"pr{i + 1}")
        ok, why = e.apply(sa)
        assert ok and list(e._eff) == sa, (prompt, why, e._eff, sa)
    st = prompt_ids(tok, e)
    for d in ds:
        sa = shift_action(opening(prompt, d), base)
        e2 = Env(prompt, canon=True, base=base, assign=True)
        for a in premise_actions(prompt):
            assert e2.apply(shift_action(a, base))[0]
        ok, why = e2.apply(sa)
        assert ok and list(e2._eff) == sa, (prompt, d, why, e2._eff, sa)
        rows.append((st, [tok.stoi[t] for t in sa] + [tok.eos]))
        npos.append(name_positions(sa))
        labels.append(d)
    return rows, npos, labels


def logmeanexp(xs: list[float]) -> float:
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs) / len(xs))


__all__ = ["BASES", "BOT", "DAN", "DIRS", "FORCED", "KD", "Attempt", "MTrace", "Move", "applicable",
           "attempt_actions", "classify", "first_move_rows", "forced_prefix", "gen_tokens", "logmeanexp", "moves",
           "opening", "premise_actions", "replay_m", "run_env", "score_rows", "shift_action", "theorem"]  # fmt: skip
