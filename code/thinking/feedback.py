"""In-context verifier feedback: a rejected action, rendered as a dead-end line of the honest-search
thinking traces, prefilled into the think block of the retry (Leon 2026-10-14, after Dmitry's
2026-10-09 request to combine Robbie's in-context verifier idea with GRPO).

Why this rendering.  The Stage-1 thinking models (S1g3) were trained on `searchtrace` traces only,
whose think blocks are goal lines `[ctx] ⊢ G := SKEL` and dead ends marked with `<no>`
(`searchtrace.Recorder.line`): `⊢ G := SKEL <no> ρ` (a reduction with a refuted hole) or
`⊢ H <no> ρ`, ρ a countermodel.  In S1g3's e64 data 93% of the blocks with a dead end have it on
their first line, before the viable reduction.  So the verifier's verdict on a rejected action
`have n : F := TERM ;` is written the way the traces write a tried-and-dead reduction of F,

    ⊢ F := TERM <no>

and the retry's think block starts with these lines, one per distinct action rejected at the
state, before the model's own think tokens.  Only existing tokens are used; the vocabulary has no
room for the checker's reason ("projection", "unbound", ...), so the message is `<no>` alone, as
in Robbie's `blank` arm (retry_ei_20261008: the error codes and a bare `<fail>` did the same).  No
countermodel is written: the verifier (environment + `step_check`) says that the step is invalid,
not why the claim is false, and a truth-table countermodel would be a second oracle (S1g3's data
carries ρ on 99.7% of its `<no>` lines; the bare form is the rare one).

`render(env, act)`, per action kind (the environment's grammar, `state_env.Env._apply`):
  * atomic `have n : F := T ;`  ->  `⊢ F := T' <no>`, T' = T in the think surface
    (`progtrace.r_skel`: the same tokens, except `Classical.byContradiction ( fun hh => a hh )`
    -> `Classical.byContradiction a`); a premise line `have n : F := h<k> ;` -> `⊢ F := h<k> <no>`
  * box `have n : F := ( fun ( x : A ) => by`  ->  `⊢ F := fun ( x : A ) => ⊢ B <no>`, B the
    consequent of F (False for a negation; F itself when F is neither, the 'box on non-arrow' case)
  * Or.elim `have n : F := Or.elim d ( fun ( a : A ) => by`  ->  `⊢ F := Or.elim d ( fun ( a : A )
    => ⊢ F ) ( fun ( b : B ) => ⊢ F ) <no>`, d : A' ∨ B in scope and b the next name (as the traces
    name the second binder); none when d is not a disjunction in scope (the second branch would
    have no binder type, and the traces' Or.elim always has both)
  * `exact x`  ->  `⊢ G := x <no>`, G the focused goal
  * anything else (an action the grammar rejects, a truncated action): None, no line; the retry is
    then a plain resample.
Formulas are written in `ftoks`' canonical form, as every trace line is.

`block(lines, nl)`: the lines joined by `<nl>`, as a think block's lines are.
`metrics(attempts)`: the per-iteration (GRPO) and per-cell (eval) feedback summary.
"""

from __future__ import annotations

import collections
from collections.abc import Iterable, Sequence
from typing import Any

from .dan import load

D = load()
MODES = ("none", "context")  # --feedback
MAX_LINES = 8  # the most recent distinct rejected actions kept in a state's block
# per attempt (`sample.think_generate(feedback='context')`'s result['fb'])
KEYS = (
    "draws",  # draws made with at least one feedback line in context
    "tokens",  # feedback tokens prefilled, summed over the attempt's draws
    "lines",  # distinct lines added (rendered rejections)
    "unrendered",  # retried rejections that could not be rendered (no line)
    "rejections",  # rejected draws (the one that ended the attempt included)
    "retry_draws",  # draws made after a rejection at the same state
    "retry_acc",  # ... accepted: a step recovered after feedback
    "rej_after_rej",  # rejected draws with an earlier rejection at the same state
    "repeats",  # ... whose action equals one rejected earlier at that state (Robbie's metric)
)
_ATOM_T = (".1", ".2", ".elim")


def _ftoks(f: Any) -> list[str]:
    return list(D.lean_tok.ftoks(f))


def _name(t: str) -> bool:
    """A name the atomic grammar cites: n1..n64 or a premise h<k>."""
    return bool(D.state_env.is_name(t)) or (len(t) > 1 and t[0] == "h" and t[1:].isdigit())


def _formula(act: Sequence[str], i: int) -> tuple[Any, int] | None:
    try:
        return D.state_env.parse_ftoks(list(act), i)
    except (D.lean_tok.ParseFail, IndexError):
        return None


def _binder(act: Sequence[str], i: int) -> tuple[str, Any, int] | None:
    """act[i:] = `( fun ( x : A ) => by` -> (x, A, the index after `by`), else None."""
    if act[i : i + 3] != ["(", "fun", "("] or i + 4 >= len(act):
        return None
    x = act[i + 3]
    if not D.state_env.is_name(x) or act[i + 4] != ":":
        return None
    fa = _formula(act, i + 5)
    if fa is None:
        return None
    j = i + 5 + fa[1]
    if act[j : j + 3] != [")", "=>", "by"]:
        return None
    return x, fa[0], j + 3


def _term(tm: Sequence[str]) -> list[str] | None:
    """An atomic term (the environment's nine shapes, names not looked up) -> its think surface."""
    tm = list(tm)
    if len(tm) == 1 and _name(tm[0]):
        return tm
    if len(tm) == 5 and tm[0] == "⟨" and tm[2] == "," and tm[4] == "⟩":
        return tm if _name(tm[1]) and _name(tm[3]) else None
    if len(tm) == 2 and tm[0] in ("Or.inl", "Or.inr"):
        return tm if _name(tm[1]) else None
    if tm[:1] == ["Classical.byContradiction"]:
        ok = len(tm) == 8 and tm[1:5] == ["(", "fun", "hh", "=>"] and tm[6:] == ["hh", ")"]
        return ["Classical.byContradiction", tm[5]] if ok and _name(tm[5]) else None
    if len(tm) == 2 and _name(tm[0]) and (tm[1] in _ATOM_T or _name(tm[1])):
        return tm
    return None


def render(env: Any, act: Sequence[str]) -> list[str] | None:
    """A rejected action (token strings, `<eos>` dropped) at `env`'s state -> its dead-end line
    (token strings), or None when it has none (module docstring)."""
    act = list(act)
    if len(act) == 2 and act[0] == "exact" and D.state_env.is_name(act[1]):
        return ["⊢", *_ftoks(env.frames[-1].goal), ":=", act[1], "<no>"]
    if len(act) < 5 or act[0] != "have" or not D.state_env.is_name(act[1]) or act[2] != ":":
        return None
    ff = _formula(act, 3)
    if ff is None:
        return None
    f, used = ff
    i = 3 + used
    if i + 1 >= len(act) or act[i] != ":=":
        return None
    i += 1
    head = ["⊢", *_ftoks(f), ":="]
    if act[i] == "(":
        b = _binder(act, i)
        if b is None or b[2] != len(act):
            return None
        x, xf, _ = b
        body = f[2] if f[0] == "imp" else D.state_env.BOT if f[0] == "not" else f
        return [*head, "fun", "(", x, ":", *_ftoks(xf), ")", "=>", "⊢", *_ftoks(body), "<no>"]
    if act[i] == "Or.elim":
        d = act[i + 1] if i + 1 < len(act) else ""
        b = _binder(act, i + 2) if _name(d) else None
        if b is None or b[2] != len(act):
            return None
        x, xf, _ = b
        df = env.lookup(d) if D.state_env.is_name(d) else None
        y = f"n{int(x[1:]) + 1}"
        if df is None or df[0] != "or" or not D.state_env.is_name(y):
            return None
        first = ["(", "fun", "(", x, ":", *_ftoks(xf), ")", "=>", "⊢", *_ftoks(f), ")"]
        second = ["(", "fun", "(", y, ":", *_ftoks(df[2]), ")", "=>", "⊢", *_ftoks(f), ")"]
        return [*head, "Or.elim", d, *first, *second, "<no>"]
    if act[-1] != ";":
        return None
    t = _term(act[i:-1])
    return None if t is None else [*head, *t, "<no>"]


def block(lines: Sequence[Sequence[int]], nl: int) -> list[int]:
    """Feedback lines (ids) -> the think-block body: the lines joined by `<nl>`."""
    out: list[int] = []
    for k, ln in enumerate(lines):
        out += ([nl] if k else []) + list(ln)
    return out


def metrics(
    attempts: Iterable[dict[str, Any]], budgets: Iterable[int] | None = None
) -> dict[str, Any]:
    """Attempts sampled with feedback (each with `fb`, `causes`, `rej`, `ok`) -> the summary:
    rejections and retries per attempt, the retries' acceptance (steps recovered after feedback),
    the repeat rate of identical rejected actions at a state (of the rejected retries, and of all
    rejections: Robbie's denominator), feedback tokens and lines per attempt, the unrendered share
    of the retried rejections, and the share of solved attempts that needed a retry. `budgets`,
    when given, one per attempt: the same per budget (`by_budget`)."""
    att = list(attempts)
    bud = list(budgets) if budgets is not None else None

    def summ(xs: Sequence[dict[str, Any]]) -> dict[str, Any]:
        n = max(1, len(xs))
        tot: collections.Counter[str] = collections.Counter()
        for x in xs:
            tot.update({k: int(v) for k, v in (x.get("fb") or {}).items()})
        retries = sum(int(x.get("rej") or 0) for x in xs)
        solved = [x for x in xs if x.get("ok")]
        return {
            "attempts": len(xs),
            "rej_per_attempt": tot["rejections"] / n,
            "retries_per_attempt": retries / n,
            "retry_draws": tot["retry_draws"],
            "retry_acc": tot["retry_acc"],
            "retry_accept": tot["retry_acc"] / tot["retry_draws"] if tot["retry_draws"] else None,
            "repeats": tot["repeats"],
            "repeat_rate": tot["repeats"] / tot["rej_after_rej"] if tot["rej_after_rej"] else None,
            "repeat_share": tot["repeats"] / tot["rejections"] if tot["rejections"] else None,
            "fb_tokens_per_attempt": tot["tokens"] / n,
            "fb_lines_per_attempt": tot["lines"] / n,
            "fb_draws_per_attempt": tot["draws"] / n,
            "unrendered": tot["unrendered"] / retries if retries else None,
            "solved_assisted": (
                sum(bool(x.get("causes")) for x in solved) / len(solved) if solved else None
            ),
        }

    out = summ(att)
    if bud is not None:
        by: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
        for x, b in zip(att, bud, strict=True):
            by[b].append(x)
        out["by_budget"] = {str(b): summ(v) for b, v in sorted(by.items())}
    return out


def line_text(m: dict[str, Any]) -> str:
    """The summary as one log line (GRPO's `fb it N:` and eval's `[feedback_B..]` lines)."""

    def pct(v: float | None) -> str:
        return "-" if v is None else f"{100 * v:.0f}%"

    return (
        f"rej/att {m['rej_per_attempt']:.2f} retries/att {m['retries_per_attempt']:.2f} "
        f"retry acc {pct(m['retry_accept'])} ({m['retry_acc']}/{m['retry_draws']}) "
        f"repeat {pct(m['repeat_rate'])} of rejected retries ({pct(m['repeat_share'])} of "
        f"rejections) fb tok/att {m['fb_tokens_per_attempt']:.1f} lines/att "
        f"{m['fb_lines_per_attempt']:.2f} unrendered {pct(m['unrendered'])} assisted "
        f"{pct(m['solved_assisted'])} of solved"
    )
