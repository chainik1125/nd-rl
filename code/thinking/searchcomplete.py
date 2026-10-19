"""Completeness fallback for search traces (`searchtrace --complete`, opt-in).

The honest search of `searchtrace` is a Markov depth-first search. Inside reductio boxes the context
is inconsistent, truth tables refute nothing, and the search can drift: it opens box after box
until no candidate is left (stuck) or it runs out of steps or names. On the sealed 13-20 set that
is 12% of theorems, almost all stuck in an inconsistent context at box depth 3-8.

`prove_complete` repairs those proofs, still from the theorem alone:

* it replays the honest search step by step (same strategy, same budgets, same records);
* **trouble** is: the search is stuck, or the box depth reaches `DEPTH_TRIGGER`, or the next name
  would pass `NAME_TRIGGER`, or the step count reaches `step_trigger`;
* on trouble it **backs off**. The candidate points are the states where the currently open boxes
  were opened: first those opened in an inconsistent context, outermost first; then the other
  open points, innermost first; then the current state. At each point a **planner** solves the
  point's own sub-problem (the hypotheses in scope ⊢ the focused frame's goal). The planner is the
  fork's shortest-proof search (`minlen` via `test_21_30/fastlen.py`, iterative deepening on proof
  length, time-limited). It reads the state only, never a reference. The first point whose plan is
  found within the time limit wins: the walk rewinds to it and discards the steps after it;
* when no point's plan is found, a second round tries the same points and the current state with a
  longer limit (`ESCALATE_SECS`): in drifted reductio boxes a point's own sub-proof can need 15-19
  lines;
* **last resort**: when no point's plan is found (or the plan budget is spent), the walk rewinds to
  the root and follows a plan for the whole theorem (the one made at the start, or one made now
  with `FINAL_SECS`). That plan closes the proof, so the fallback fails only if the planner finds
  no proof of the theorem at all;
* the plan is then **followed** until it closes that frame, translated action by action into the
  environment's names (`Plan`). Then the honest search resumes;
* a plan step keeps the honest search's record when that search, run at the state, commits to the
  plan's action. Otherwise its think is empty: no record is written for a move the search did not
  make. Such steps are marked `"planner": true` in the step metadata, outside the think text. Labels
  (`min_level`) are computed at every step as usual.

`searchtrace.trace_for(..., complete=True)` calls this only after every strategy of the default
generator has failed, so every proof the default generator writes is unchanged.
"""

from __future__ import annotations

import contextlib
import copy
import importlib
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from . import searchtrace as st

FASTLEN_DIR = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "current"
    / "thinking_traces_20261005"
    / "test_21_30"
)
DEPTH_TRIGGER = 5  # box depth (frames - 1) that counts as a runaway (references: <= 3)
NAME_TRIGGER = 56  # next name index that counts as running out of names (n64 is the last)
MAX_PLANS = 10  # planner invocations per proof
POINT_SECS = 6.0  # planner time limit at a back-off point
HERE_SECS = 30.0  # planner time limit at the current state
ESCALATE_SECS = 90.0  # second round over the same points (and the current state), longer limit
ROOT_SECS = 20.0  # planner time limit for the whole theorem (the length budget)
FINAL_SECS = 600.0  # last resort: the whole theorem again, with this time limit
STEP_SLACK = 6  # steps allowed beyond the whole-theorem plan before backing off to the root
STEP_REARM = 12  # without a whole-theorem plan: steps allowed after each step-triggered repair
PLAN_BOUND = 30  # proof lines beyond the premises the planner may use
COMPLETE_MAX_STEPS = 96

_FL: list[Any] = []


def fastlen() -> Any:
    """The fork's shortest-proof search (`test_21_30/fastlen.py`, which imports the fork's `minlen`
    from its read-only clone); imported on first use only."""
    if not _FL:
        if str(FASTLEN_DIR) not in sys.path:
            sys.path.insert(0, str(FASTLEN_DIR))
        _FL.append(importlib.import_module("fastlen"))
    return _FL[0]


def sub_prompt(hyps: Sequence[Any], goal: Any) -> str:
    """A state's sub-theorem: its distinct hypotheses ⊢ its focused goal (K12 prompt syntax)."""
    prem = " , ".join(st.pt.ascii_line(st.ftoks(h)) for h in hyps)
    return f"THM {prem} SEQ {st.pt.ascii_line(st.ftoks(goal))} PRF"


class Plan:
    """A planner proof of a state's sub-theorem, replayed in a shadow environment and translated
    into the real environment's names one action at a time."""

    def __init__(self, prompt: str, actions: list[list[str]], names: dict[str, str]) -> None:
        T = st.td()
        self.sub = T.Env(prompt, canon=True, base=0, assign=False)
        self.todo = actions
        self.names = names  # sub-environment name -> real name
        for a in actions[: self.n_prem(actions)]:
            ok, why = self.sub.apply(a)
            assert ok, why
        self.todo = actions[self.n_prem(actions) :]
        self.length = len(self.todo)

    @staticmethod
    def n_prem(actions: list[list[str]]) -> int:
        return next(
            (i for i, a in enumerate(actions) if st.pt.step_kind(a) != "premise"), len(actions)
        )

    @property
    def done(self) -> bool:
        return not self.todo

    def next_action(self, env: Any) -> list[str]:
        """The next planned action in the real environment's names."""
        T = st.td()
        sa = self.todo[0]
        out = list(sa)
        if sa[0] == "exact":
            out[1] = self.names[sa[1]]
            return out
        nh, nb = T.next_names(env)
        out[1] = nh
        j = sa.index(":=")
        t = sa[j + 1]
        if t == "(":
            out[j + 4] = nb
        elif t == "Or.elim":
            out[j + 2] = self.names[sa[j + 2]]
            out[j + 6] = nb
        else:
            for i in range(j + 1, len(sa)):
                if T.is_name(sa[i]) and sa[i][0] == "n":
                    out[i] = self.names[sa[i]]
        return out

    def advance(self, real: list[str], env_after: Any) -> None:
        """Apply the next planned action in the shadow environment, after `real` (its translation)
        was applied to the real one; record the names both environments introduced."""
        sa = self.todo.pop(0)
        or1 = self.sub.frames[-1].kind == "or1"
        ok, why = self.sub.apply(sa)
        assert ok, why
        if sa[0] == "exact":
            if or1 and not self.sub.done:  # the environment opened the second branch
                self.names[self.sub.frames[-1].last] = env_after.frames[-1].last
            return
        self.names[sa[1]] = real[1]
        j = sa.index(":=")
        if sa[j + 1] == "(":
            self.names[sa[j + 4]] = real[j + 4]
        elif sa[j + 1] == "Or.elim":
            self.names[sa[j + 6]] = real[j + 6]


_MEMO: dict[tuple[frozenset[Any], Any, float], Any] = {}


def support(hyps: Sequence[Any], goal: Any) -> list[Any]:
    """A minimal set of the hypotheses that still entails the goal (truth tables; the oldest is
    dropped first, so derived facts and box binders are kept): the planner's premises.  In an
    inconsistent context this is a minimal inconsistent core."""
    keep = list(hyps)
    for h in list(hyps):
        rest = [x for x in keep if x != h]
        if st.pt.ent(rest, goal):
            keep = rest
    return keep


def make_plan(env: Any, time_limit: float) -> Plan | None:
    """The planner at a state: a shortest proof of the focused frame's goal from a minimal support
    of it among the hypotheses in scope (reference-free), or None (none within the time limit)."""
    T = st.td()
    ctx = st.Ctx(env)
    if ctx.goal in ctx.avail:  # the goal is in scope (minlen returns no proof): `exact` it
        g = st.ftoks(ctx.goal)
        acts = [["have", "n1", ":", *g, ":=", "h1", ";"], ["exact", "n1"]]
        if ctx.last is not None and ctx.sc.get(ctx.last) == ctx.goal:
            return Plan(sub_prompt([ctx.goal], ctx.goal), acts, {"n1": ctx.last})
        # `exact` cites the frame's last name only: restate the goal first, as the search does
        acts[1:] = [["have", "n2", ":", *g, ":=", "n1", ";"], ["exact", "n2"]]
        return Plan(sub_prompt([ctx.goal], ctx.goal), acts, {"n1": ctx.avail[ctx.goal]})
    hyps = support(ctx.hyps, ctx.goal)  # may be empty: the goal is valid on its own
    prompt = sub_prompt(hyps, ctx.goal)
    key = (frozenset(hyps), ctx.goal, time_limit)
    if key not in _MEMO:
        r = fastlen().fast_minlen(prompt, bound=len(hyps) + PLAN_BOUND, time_limit=time_limit)
        proof = r.get("proof") if r.get("min_lines_ub") else None
        acts = None
        if proof:
            try:
                acts = [list(a) for _, a, _ in T.decompose(prompt, proof, canon=True)[0]]
            except Exception:
                acts = None
        _MEMO[key] = (prompt, acts)
    prompt, acts = _MEMO[key]
    if acts is None:
        return None
    names: dict[str, str] = {}
    prem = T.Env(prompt, canon=True, base=0, assign=False).prem
    for a in acts[: Plan.n_prem(acts)]:
        f = prem[int(a[-2][1:]) - 1]
        if f not in ctx.avail:
            return None
        names[a[1]] = ctx.avail[f]
    return Plan(prompt, [list(a) for a in acts], names)


def _record(
    env: Any,
    strat: st.Strategy,
    main_bud: st.Budget,
    buds: dict[int, st.Budget],
    labels: bool,
) -> tuple[st.StepOut | None, dict[int, st.StepOut]]:
    """The honest searches at a state (as `searchtrace.prove`): the main level's and the ladder's
    (for labels). A stuck search gives None / no entry."""
    outs: dict[int, st.StepOut] = {}
    if labels:
        for b, bud in buds.items():
            with contextlib.suppress(st.Stuck):
                outs[b] = st.Search(env, strat, bud).run()
    main = None
    if main_bud.level in outs and main_bud == buds[main_bud.level]:
        main = outs[main_bud.level]
    else:
        try:
            main = st.Search(env, strat, main_bud).run()
        except st.Stuck:
            main = None
    return main, outs


def prove_complete(
    prompt: str,
    strat: st.Strategy,
    level: int,
    ladder: Sequence[int] = st.DEFAULT_LEVELS,
    labels: bool = True,
    cap: int | None = None,
    max_steps: int = COMPLETE_MAX_STEPS,
    step_trigger: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """A complete proof of `prompt` (the theorem only): the honest search, backed by the planner on
    trouble.  -> (steps, per-step info, {planner calls, plan steps, ...}).  Raises ProofFail."""
    if not isinstance(level, int):
        raise ValueError("complete mode takes a fixed effort level")
    T = st.td()
    env = T.Env(prompt, canon=True, base=0, assign=False)
    buds = {b: st.budget_for(b) for b in ladder}
    main_bud = st.budget_for(level, cap)
    steps: list[dict[str, Any]] = []
    infos: list[dict[str, Any]] = []
    snaps: list[Any] = []
    opened: list[int] = []  # step indices that opened the frames open now (outermost first)
    stats: dict[str, Any] = {
        "planner_calls": 0,
        "plans": 0,
        "plan_steps": 0,
        "plan_steps_with_record": 0,
        "discarded_steps": 0,
        "planner_secs": 0.0,
        "triggers": [],
        "root_fallbacks": 0,
        "escalations": 0,
    }
    for k in range(len(env.prem)):  # premises, as searchtrace.prove
        snaps.append(None)
        act = ["have", T.next_names(env)[0], ":", *st.ftoks(env.prem[k]), ":=", f"h{k + 1}", ";"]
        steps.append({"think": "", "action": st.pt.ascii_line(act), "min_level": 0})
        infos.append({"kind": "premise", "think_tokens": 0, "action_tokens": len(act)})
        ok, why = env.apply(act)
        if not ok:
            raise st.ProofFail(f"premise: {why}")
    # the length budget: a plan for the whole theorem (time-limited); overrunning it backs off to
    # the root and follows that plan
    root = len(steps)
    t0 = time.time()
    stats["planner_calls"] += 1
    root_plan = make_plan(env, ROOT_SECS) if not env.done else None
    stats["planner_secs"] += time.time() - t0
    stats["root_plan_steps"] = root_plan.length if root_plan else None
    trig = step_trigger or (
        root + root_plan.length + STEP_SLACK if root_plan else int(max_steps * 0.5)
    )
    plan: Plan | None = None
    while not env.done:
        if len(steps) >= max_steps:
            raise st.ProofFail("over max_steps (complete)")
        main, outs = _record(env, strat, main_bud, buds, labels)
        if plan is None:
            ctx = st.Ctx(env)
            why = None
            if main is None:
                why = "stuck"
            elif len(env.frames) - 1 >= DEPTH_TRIGGER:
                why = "depth"
            elif int(ctx.nb[1:]) >= NAME_TRIGGER:
                why = "names"
            elif len(steps) >= trig:
                why = "steps"
            if why is not None:
                stats["triggers"].append(why)
                point, plan = root, None
                if stats["plans"] < MAX_PLANS:
                    if why == "steps" and root_plan is not None:
                        plan = make_plan(snaps[root], ROOT_SECS)  # memoised
                    else:
                        point, plan = _backoff(env, snaps, opened, stats, POINT_SECS, HERE_SECS)
                        if plan is None:  # the same points, longer
                            stats["escalations"] += 1
                            point, plan = _backoff(
                                env, snaps, opened, stats, ESCALATE_SECS, ESCALATE_SECS
                            )
                if plan is None:  # last resort: the whole theorem from the root (closes the proof)
                    point = root
                    at_root = snaps[root] if root < len(snaps) else env
                    t0 = time.time()
                    stats["planner_calls"] += 1
                    plan = make_plan(at_root, ROOT_SECS if root_plan else FINAL_SECS)
                    stats["planner_secs"] += time.time() - t0
                    stats["root_fallbacks"] += 1
                if plan is None:
                    raise st.ProofFail(f"planner found no plan ({why})")
                if point < len(steps):  # rewind to the back-off point
                    stats["discarded_steps"] += len(steps) - point
                    env = snaps[point]
                    del steps[point:], infos[point:], snaps[point:]
                    opened[:] = [i for i in opened if i < point]
                if why == "steps" and root_plan is None:  # re-arm the length budget
                    trig = len(steps) + plan.length + STEP_REARM
                stats["plans"] += 1
                continue
        snaps.append(copy.deepcopy(env))
        before = len(env.frames)
        info: dict[str, Any] = {"kind": "decision", "state_tokens": len(env.state_tokens())}
        if plan is not None:
            act = plan.next_action(env)
            take = main is not None and main.action == act
            lines = main.lines if take and main is not None else []
            src = main if take else None
            stats["plan_steps"] += 1
            stats["plan_steps_with_record"] += bool(take)
        else:
            assert main is not None
            act, lines, src = main.action, main.lines, main
        if not st.viable(env, act):
            raise st.ProofFail(f"non-viable action {' '.join(act)}")
        think = ["<think>", *st.pt.join_lines(lines), "</think>"]
        soft = T.check_step(env, think, act, {})
        if soft:
            raise st.ProofFail(f"generator bug: soft finding {soft[0]}")
        forced = bool(src is not None and src.info.get("forced"))
        if forced:
            info["kind"] = "exact"
        step: dict[str, Any] = {"think": st.ascii_lines(lines), "action": st.pt.ascii_line(act)}
        if forced:
            step["min_level"] = 0
        elif outs:
            vl = [b for b in ladder if b in outs and st.viable(env, outs[b].guess)]
            step["min_level"] = vl[0] if vl else None
            step["viable_levels"] = vl
        if plan is not None and not lines:
            step["planner"] = True
        steps.append(step)
        if src is not None:
            info.update(src.info)
        info["think_tokens"] = len(st.pt.join_lines(lines))
        info["action_tokens"] = len(act)
        info["planner"] = plan is not None
        info["min_level"] = step.get("min_level")
        infos.append(info)
        ok, why2 = env.apply(act)
        if not ok:
            raise st.ProofFail(f"apply: {why2}")
        if plan is not None:
            plan.advance(act, env)
            if plan.done:
                plan = None
        after = len(env.frames)
        if after > before:
            opened.append(len(steps) - 1)
        elif after < before:
            opened.pop()
    stats["planner_secs"] = round(stats["planner_secs"], 2)
    stats["plan_steps"] = sum(bool(i.get("planner")) for i in infos)  # in the final proof
    stats["plan_steps_with_record"] = sum(
        bool(i.get("planner")) and not s.get("planner") for s, i in zip(steps, infos, strict=True)
    )
    return steps, infos, stats


def _backoff(
    env: Any,
    snaps: list[Any],
    opened: list[int],
    stats: dict[str, Any],
    point_secs: float = POINT_SECS,
    here_secs: float = HERE_SECS,
) -> tuple[int, Plan | None]:
    """The first back-off point whose sub-problem the planner solves in time -> (step index to
    rewind to, its plan).  Points: boxes opened in an inconsistent context (outermost first), the
    other open boxes (innermost first), then the current state."""
    incons = [i for i in opened if not st.Ctx(snaps[i]).consistent()]
    rest = [i for i in reversed(opened) if i not in incons]
    for i in incons + rest:
        t0 = time.time()
        stats["planner_calls"] += 1
        p = make_plan(snaps[i], point_secs)
        stats["planner_secs"] += time.time() - t0
        if p is not None:
            return i, p
    t0 = time.time()
    stats["planner_calls"] += 1
    p = make_plan(env, here_secs)
    stats["planner_secs"] += time.time() - t0
    return len(snaps), p
