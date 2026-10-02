"""The Dan-only results page: Dan Pandori's twelve best-state checkpoints on the ndbench panel,
laid out like a paper's results section (`dash --summary --dan-only`, writing
`dash-dir/index.html`; `figures --dan-only` writes the same figures as PDF/SVG/PNG).

Leon (2026-10-02): the shared page shows only the new run, none of our own base/rl/armC results,
and it is the only page of the site (no links to other pages). So nothing here reads, draws,
legends or tabulates a base/rl/armC run: every chart is built for Dan's labels only
(`analysis.models_registry`'s `dan12`/`dan6` families).

Structure: title, a three-sentence lede, a Figure 1 overview grid (one panel per benchmark's
headline fz vs t1 comparison), Section 1 "Eliciting proofs", Section 2 "Internals", Limitations.
Every caption says what is plotted, n, the CI method (or "point estimates"), the judge and the
unseen filter where one applies, then a one-sentence takeaway computed from the plotted numbers.

Numbers come from the audited Dan views wherever they exist -- `dash.dan_pages`'s builders
(called with `families=("dan12", "dan6")`, so "ours" is never computed) and `analysis.dan_heldout`
/ `analysis.judges` -- and from `analysis.dan_views` for the views this page adds (textbook72
counts, padding by pad type, textbook_law unseen-only, steering, interpolation, probes,
likelihood, KL). Each chart's numbers are computed once into a `ChartData`; the figure
(`figures/dan_only.py`), its caption and its data table read that object.
"""

from __future__ import annotations

import html
import re
import statistics
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from ..analysis import dan_heldout as heldout_view
from ..analysis import dan_views as dv
from ..analysis import judges as judges_view
from ..analysis import models_registry as mr
from ..analysis._common import (
    default_version,
    items_by_pid,
    latest_run_id,
    problems_by_pid,
    run_records,
)
from ..core.io import read_jsonl
from ..core.types import JSON, Problem
from ..figures import dan_only as F
from ..metrics import pass_at_k
from ..store import Store
from . import dan_pages, summary
from . import primitives as P
from .chart_data import ChartData
from .chart_data import missing_chart as _missing

FAMS: tuple[str, ...] = ("dan12", "dan6")
JUDGE = dv.DAN_JUDGE_ID
PAIR: tuple[str, str] = ("dan12-fz-s0", "dan12-t1-s0")
#: Item-bootstrap replicates for the textbook curves (B = 1000; the CI the caption names).
B_TEXTBOOK = 1000
_BOOT = "95% item-level bootstrap"


def _ref(key: str) -> str:
    """A cross-reference to another chart's "Figure N" / "Table N", filled in by `resolve_refs`
    once the page's numbering is known."""
    return "{ref:" + key + "}"


_REF_RE = re.compile(r"\{ref:([a-z0-9_]+)\}")


def resolve_refs(charts: Sequence[ChartData], nums: Mapping[str, str]) -> None:
    """Replace every `_ref` placeholder in the charts' captions and takeaways."""
    for cd in charts:
        for attr in ("caption", "takeaway", "missing"):
            text = getattr(cd, attr)
            if "{ref:" in text:
                setattr(cd, attr, _REF_RE.sub(lambda m: nums.get(m.group(1), "?"), text))


def _pct(x: float | None, d: int = 1) -> str:
    if x is None or x != x:
        return "n/a"
    return f"{100 * x:.{d}f}%"


def _pts(x: float) -> str:
    return f"{100 * x:+.1f}"


def _run_k(store: Store, suite: str, model: str) -> int | None:
    version = default_version(store, suite=suite, models=[model], judge_id=JUDGE) or "v1"
    run_id = latest_run_id(store, suite=suite, model=model, version=version, judge_id=JUDGE)
    if run_id is None:
        return None
    for r in run_records(store, suite=suite, model=model, judge_id=JUDGE):
        if r.get("run_id") == run_id:
            return int(cast(int, r.get("k") or 0)) or None
    return None


def _seen_note(store: Store, suite: str, family: str = "dan12") -> tuple[int, int] | None:
    version = default_version(store, suite=suite, models=list(PAIR), judge_id=JUDGE) or "v1"
    counts = dv.seen_counts(store, family, [(suite, version)])
    return (counts[0][3], counts[0][2]) if counts else None


def _curve_at(curve: Sequence[tuple[float, float]], x: float) -> float | None:
    return next((y for xx, y in curve if xx == x), None)


# ================================================================ 1.1 the 4-premise ceiling

MAXP = dv.MAX_TRAIN_PREMISES
#: Suites drawn as panels of the ceiling figure (interp_eval_k128 is a subset of dev + sealed,
#: so it is in the table only).
CEILING_PANELS: tuple[str, ...] = (
    "families",
    "glue",
    "padding",
    "skill_pool_sub",
    "interp_eval_dev",
    "interp_eval_sealed",
    "prefix_state",
)
#: Premise counts with fewer items are left out of a ceiling panel.
MIN_PREMISE_CELL = 10


def _beyond(n_out: int, n_all: int, what: str = "items") -> str:
    """The caption clause naming what the premise ceiling drops."""
    if not n_out:
        return f"none of the {n_all:,} {what} has more than {MAXP} premises"
    return (
        f"the {n_out:,} of {n_all:,} {what} with {MAXP + 1} or more premises (more than any "
        "training theorem; Section 1.1) are excluded"
    )


def _splits(store: Store) -> dict[str, dict[str, dict[str, Any]]]:
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for suite in dv.CEILING_SUITES:
        per = {m: d for m in PAIR if (d := dv.premise_split(store, suite, m, judge_id=JUDGE))}
        if per:
            out[suite] = per
    return out


def _train_shares(train: Mapping[str, Mapping[int, int]]) -> str:
    bits = []
    for label, counts in train.items():
        tot = sum(counts.values())
        dist = " / ".join(f"{100 * counts.get(k, 0) / tot:.1f}" for k in range(max(counts) + 1))
        bits.append(f"{label} {tot:,} theorems, {dist}% with 0-{max(counts)} premises")
    return "; ".join(bits)


def _c_ceiling(store: Store) -> ChartData:
    key, title = "do_ceiling", "Pass@1 by number of premises, against the training data's"
    splits = _splits(store)
    if not splits:
        return _missing(key, title, "no runs on the generated suites for Dan's models yet")
    train = dv.training_premise_counts(dv.dan_training_files("dan12"))
    panels: dict[str, list[P.Series]] = {}
    for suite in CEILING_PANELS:
        for m, d in splits.get(suite, {}).items():
            pts = [
                (float(k), v)
                for k, (v, n) in cast("dict[int, tuple[float, int]]", d["by_premises"]).items()
                if n >= MIN_PREMISE_CELL
            ]
            if pts:
                panels.setdefault(suite, []).append(P.Series(m, pts, "var(--s1)"))
    gen = [s for s in splits if s != "prefix_state" and PAIR[1] in splits[s]]
    pooled: dict[str, list[tuple[float, float]]] = {}
    for m in PAIR:
        acc: dict[int, list[float]] = {}
        for s_ in CEILING_PANELS:
            d = splits.get(s_, {}).get(m)
            if d is None or s_ == "prefix_state":
                continue
            for k, (v, n) in cast("dict[int, tuple[float, int]]", d["by_premises"]).items():
                a = acc.setdefault(k, [0.0, 0.0])
                a[0] += v * n
                a[1] += n
        pts = [(float(k), a[0] / a[1]) for k, a in sorted(acc.items()) if a[1] >= MIN_PREMISE_CELL]
        if pts:
            pooled[m] = pts
    t1 = {s: splits[s][PAIR[1]] for s in gen}
    fz = {s: splits[s][PAIR[0]] for s in gen if PAIR[0] in splits[s]}
    takeaway = "pass@1 by number of premises."
    if t1:
        hi_out = max(t1, key=lambda s: t1[s]["p_out"])
        lo_in = min(t1, key=lambda s: t1[s]["p_in"])
        hi_in = max(t1, key=lambda s: t1[s]["p_in"])
        fz_out = max((d["p_out"] for d in fz.values()), default=None)
        max_train = max((max(c) for c in train.values()), default=None)
        lead = (
            f"No training theorem has more than {max_train} premises, and beyond {max_train} "
            "both models solve almost nothing"
            if max_train is not None
            else f"Beyond {MAXP} premises both models solve almost nothing"
        )
        takeaway = (
            f"{lead}: on {len(gen)} generated suites the RL model's pass@1 on items with "
            f"{MAXP + 1} or more premises is at most {_pct(t1[hi_out]['p_out'])} ({hi_out}), "
            f"against {_pct(t1[lo_in]['p_in'])} ({lo_in}) to {_pct(t1[hi_in]['p_in'])} "
            f"({hi_in}) on items with at most {MAXP}"
            + (f"; the pretrained model's is at most {_pct(fz_out)}" if fz_out is not None else "")
        )
        ps = splits.get("prefix_state", {}).get(PAIR[1])
        if ps is not None:
            takeaway += (
                f". prefix_state is the exception (RL {_pct(ps['p_out'])} beyond the ceiling), "
                "because its forced prefix usually restates the premises "
                f"({_ref('do_ceiling_table')})"
            )
        takeaway += "."
    ks = {s: _run_k(store, s, PAIR[1]) for s in CEILING_PANELS if s in panels}
    letters = "bcdefghij"
    panel_txt = ", ".join(
        f"({letters[i]}) {s}" for i, s in enumerate(s for s in CEILING_PANELS if s in panels)
    )
    caption = (
        "(a) Share of the cap-12 checkpoints' training theorems (Stage-1 set and T1 RL targets "
        "pooled) with each number of premises; per file: "
        + (_train_shares(train) if train else "training files not found")
        + f". {panel_txt}: pass@1 (each item's c/n over every draw, averaged over the items with "
        f"that many premises; counts with < {MIN_PREMISE_CELL} items omitted) of the pretrained "
        f"and RL models (cap-12 seed-0 pair); shaded: {MAXP + 1} or more premises, which no "
        "training theorem has. On prefix_state the forced prefix already restates every premise "
        f"for most items with 4 or more ({_ref('do_ceiling_table')}). k per item: "
        + ", ".join(f"{s} {k or '?'}" for s, k in ks.items())
        + ". Point estimates. Judge: Lean (lean_v2)."
    )
    cd = ChartData(
        key,
        title,
        "line",
        takeaway,
        caption,
        series=[
            P.Series(f"{s}: {x.name}", x.pts, "var(--s1)") for s, ss in panels.items() for x in ss
        ],
        opts={"x_label": "premises", "y_fmt": "pct"},
        extra={
            "suite_panels": panels,
            "train": train,
            "order": [s for s in CEILING_PANELS if s in panels],
            "pooled": pooled,
        },
    )
    ks_all = sorted({k for c in train.values() for k in c})
    cd.extra_tables.append(
        (
            "training theorems by premise count (a)",
            ["file", *(str(k) for k in ks_all), "total"],
            [
                [label, *(f"{c.get(k, 0):,}" for k in ks_all), f"{sum(c.values()):,}"]
                for label, c in train.items()
            ],
        )
    )
    return cd


def _c_ceiling_table(store: Store) -> ChartData:
    key, title = "do_ceiling_table", "Pass@1 on each side of the premise ceiling"
    splits = _splits(store)
    if not splits:
        return _missing(key, title, "no runs on the generated suites for Dan's models yet")
    headers = [
        "Suite",
        "Items",
        f"≥ {MAXP + 1} premises",
        f"pretrained ≤ {MAXP} (%)",
        f"pretrained ≥ {MAXP + 1} (%)",
        f"RL ≤ {MAXP} (%)",
        f"RL ≥ {MAXP + 1} (%)",
        "PR position (%)",
    ]
    rows: list[list[str]] = []
    raw: list[list[str]] = []
    early: dict[str, dict[str, float]] = {}

    def num(x: float | None) -> str:
        return "–" if x is None or x != x else f"{100 * x:.1f}"

    def add(label: str, per: Mapping[str, dict[str, Any]], suite: str) -> None:
        any_d = next(iter(per.values()))
        pr: dict[str, float] = {}
        for m, d in per.items():
            probs = problems_by_pid(store, suite, cast(str, d["version"]))
            fs = dv.first_step_rejections(store, cast(str, d["run_id"]), probs, d["out_pids"])
            pr[m] = cast(float, fs["early_pr"])
            raw.append(
                [
                    label,
                    m,
                    str(d["n"]),
                    str(d["n_out"]),
                    num(d["p_in"]),
                    num(d["p_out"]),
                    num(fs["first_pr"]),
                    num(fs["early_pr"]),
                    num(fs["pr"]),
                    str(fs["n"]),
                ]
            )
        early[label] = pr
        n, n_out = int(any_d["n"]), int(any_d["n_out"])
        rows.append(
            [
                P.esc(label),
                f"{n:,}",
                f"{n_out:,} ({100 * n_out / n:.0f}%)" if n else "0",
                num(per.get(PAIR[0], {}).get("p_in")),
                num(per.get(PAIR[0], {}).get("p_out")),
                num(per.get(PAIR[1], {}).get("p_in")),
                num(per.get(PAIR[1], {}).get("p_out")),
                " / ".join(num(pr.get(m)) for m in PAIR if m in pr) if n_out else "–",
            ]
        )

    for suite in dv.CEILING_SUITES:
        if suite in splits:
            add(suite, splits[suite], suite)
    forced = not_forced = None
    if "prefix_state" in splits:
        for label, sel in (
            ("↳ premises forced", dv.premises_forced),
            ("↳ not all forced", lambda p: not dv.premises_forced(p)),
        ):
            per = {
                m: d
                for m in PAIR
                if (d := dv.premise_split(store, "prefix_state", m, judge_id=JUDGE, select=sel))
            }
            if per:
                add(label, per, "prefix_state")
                if label == "↳ premises forced":
                    forced = per.get(PAIR[1])
                else:
                    not_forced = per.get(PAIR[1])
    free: list[str] = []
    for suite in dv.CEILING_FREE_SUITES:
        version = default_version(store, suite=suite, models=list(PAIR), judge_id=JUDGE)
        probs = problems_by_pid(store, suite, version or "v1")
        if probs:
            n_out = sum(1 for p in probs.values() if not dv.in_premise_range(p))
            free.append(f"{suite} {n_out} of {len(probs):,}")
    t1 = {s: d[PAIR[1]] for s, d in splits.items() if PAIR[1] in d and s != "prefix_state"}
    takeaway = "pass@1 on each side of the ceiling."
    if t1:
        worst = max(
            max(d.get(m, {}).get("p_out", 0.0) for m in PAIR)
            for s, d in splits.items()
            if s != "prefix_state"
        )
        rl_pr = [v for s in t1 if (v := early.get(s, {}).get(PAIR[1], float("nan"))) == v]
        lo_pr, hi_pr = (_pct(min(rl_pr), 0), _pct(max(rl_pr), 0)) if rl_pr else ("", "")
        prefix_txt = ""
        if forced is not None and not_forced is not None:
            prefix_txt = (
                f"; on prefix_state the RL model solves {_pct(forced['p_out'])} of such items "
                f"when the forced prefix restates every premise and {_pct(not_forced['p_out'])} "
                "when it does not"
            )
            if forced["p_in"] == forced["p_in"] and not_forced["p_in"] == not_forced["p_in"]:
                prefix_txt += (
                    f" (against {_pct(forced['p_in'])} and {_pct(not_forced['p_in'])} with at "
                    f"most {MAXP} premises)"
                )
        takeaway = (
            f"Outside prefix_state, neither model solves more than {_pct(worst)} of the items "
            f"with {MAXP + 1} or more premises on any suite, and "
            + (
                f"{lo_pr if lo_pr == hi_pr else f'{lo_pr}-{hi_pr}'} of the RL model's attempts "
                "on them are rejected for a premise line out of place within its first two "
                "actions"
                if rl_pr
                else "most attempts end early"
            )
            + prefix_txt
            + "."
        )
    caption = (
        f"Pass@1 (each item's c/n over every draw, averaged over items) on the items with at most "
        f"{MAXP} premises and with {MAXP + 1} or more, for the pretrained and RL models (cap-12 "
        f"seed-0 pair, every item of each suite, not filtered for training overlap); "
        f"'≥ {MAXP + 1} premises': how many items, and their share. PR position: share of all "
        f"attempts on the items with {MAXP + 1} or more premises that the environment rejects as "
        "a premise line out of place (a := h<k> line that is not the next premise in order) at "
        "the model's first or second own action, pretrained / RL. The ↳ rows split prefix_state "
        "by whether its forced prefix replays every premise line (premises forced) or not (both "
        "pass@1 columns of each row on the same condition). The other suites on this page have "
        "at most one such item: " + ", ".join(free) + ". Point estimates. Judge: Lean (lean_v2)."
    )
    cd = ChartData(
        key,
        title,
        "table",
        takeaway,
        caption,
        extra={"table": {"headers": headers, "rows": rows, "align": "lrrrrrrr"}},
    )
    cd.extra_tables.append(
        (
            "per model",
            [
                "suite",
                "model",
                "items",
                f"≥ {MAXP + 1}",
                f"pass@1 ≤ {MAXP} (%)",
                f"pass@1 ≥ {MAXP + 1} (%)",
                "PR at 1st action (%)",
                "PR at 1st or 2nd (%)",
                "PR at any step (%)",
                "attempts",
            ],
            raw,
        )
    )
    return cd


# ================================================================== 2b. failure modes

#: What each environment rejection means (`vendor/danstate/state_env.py`, `Env._apply`).
ENV_REASON_GLOSS: dict[str, str] = {
    "unbound": "cites a name not in scope",
    "box on non-arrow": "opens a box for a statement that is not an implication or negation",
    "binder mismatch": "box hypothesis is not the antecedent (or disjunct) required",
    "Or.elim on non-or": "Or.elim on a hypothesis that is not a disjunction",
    "exact not last": "exact on a name that is not the frame's last statement",
    "exact goal mismatch": "exact on a statement that is not the goal",
    "PR position": "premise line out of place",
    "term": "malformed term",
    "box head": "malformed box opener",
    "no semicolon": "have line without its ;",
    "have name": "have without a name",
    "formula": "malformed formula",
    "truncated": "action not ended within max_action tokens",
    "step_cap": "hit the 96-step cap",
}
#: Fixed-k textbook_dev runs (256 draws per theorem, every theorem) the breakdown reads; the
#: adaptive k = 2048 runs draw more for unsolved theorems, so their samples are not item-uniform.
FAILURE_GROUPS: tuple[tuple[str, str], ...] = (
    ("solved", "solved"),
    ("lean", "Lean rej."),
    ("env", "env rej."),
)
FAILURE_SUITE = "textbook_dev"
FAILURE_K = 256
N_REASONS = 6


def _fixed_k_run(store: Store, suite: str, model: str, k: int) -> str | None:
    version = default_version(store, suite=suite, models=[model], judge_id=JUDGE) or "v1"
    cands = [
        r
        for r in run_records(store, suite=suite, version=version, model=model, judge_id=JUDGE)
        if int(cast("int | None", r.get("k")) or 0) == k
    ]
    cands.sort(key=lambda r: cast(str, r.get("finished_utc") or r.get("started_utc") or ""))
    return cast(str, cands[-1]["run_id"]) if cands else None


def _c_failures(store: Store) -> ChartData:
    key, title = "do_failures", "How attempts on textbook theorems end"
    data: dict[str, dict[str, Any]] = {}
    for m in PAIR:
        run_id = _fixed_k_run(store, FAILURE_SUITE, m, FAILURE_K)
        if run_id is not None and (d := dv.failure_modes(store, run_id)):
            data[m] = d
    if not data:
        return _missing(key, title, f"no k = {FAILURE_K} {FAILURE_SUITE} runs for the pair yet")
    present = [m for m in PAIR if m in data]
    reasons_all: dict[str, dict[str, float]] = {}
    for m in present:
        for r, v in data[m]["env_reasons"]:
            reasons_all.setdefault(r, {})[m] = v
    top = sorted(reasons_all, key=lambda r: -max(reasons_all[r].values()))[:N_REASONS]
    rest = {m: sum(v.get(m, 0.0) for r, v in reasons_all.items() if r not in top) for m in present}
    headers = ["Outcome", *(f"{_role_name(m)} (%)" for m in present)]
    if len(present) == 2:
        headers.append("RL − pretrained (points)")

    def row(
        label: str, vals: Mapping[str, float], *, bold: bool = False, sub: bool = False
    ) -> list[str]:
        txt = P.esc(label)
        txt = f"<b>{txt}</b>" if bold else txt
        txt = f'<span class="sub">{txt}</span>' if sub else txt
        cells = [txt, *(f"{100 * vals.get(m, 0.0):.1f}" for m in present)]
        if len(present) == 2:
            cells.append(_pts(vals.get(PAIR[1], 0.0) - vals.get(PAIR[0], 0.0)).replace("-", "−"))
        return cells

    rows = [
        row("Solved", {m: data[m]["solved"] for m in present}, bold=True),
        row("Finished, rejected by Lean", {m: data[m]["lean"] for m in present}, bold=True),
        row("Rejected by the environment", {m: data[m]["env"] for m in present}, bold=True),
    ]
    for r in top:
        gloss = ENV_REASON_GLOSS.get(r)
        rows.append(row(f"{r}" + (f": {gloss}" if gloss else ""), reasons_all[r], sub=True))
    if any(v > 0 for v in rest.values()):
        rows.append(row("other reasons", rest, sub=True))
    if any(data[m]["other"] for m in present):
        rows.append(row("Other", {m: data[m]["other"] for m in present}, bold=True))
    takeaway = "how attempts end."
    if len(present) == 2:
        a, b = data[PAIR[0]], data[PAIR[1]]
        big = max(
            top, key=lambda r: reasons_all[r].get(PAIR[1], 0.0) - reasons_all[r].get(PAIR[0], 0.0)
        )
        takeaway = (
            f"RL turns finished-but-wrong proofs into solved ones: Lean rejects "
            f"{_pct(a['lean'])} of the pretrained model's attempts and {_pct(b['lean'])} of the "
            f"RL model's, while solved attempts rise from {_pct(a['solved'])} to "
            f"{_pct(b['solved'])}; the environment rejects more of the RL model's attempts "
            f"({_pct(a['env'])} → {_pct(b['env'])}), the largest rise being {big} "
            f"({_pct(reasons_all[big].get(PAIR[0], 0.0))} → "
            f"{_pct(reasons_all[big].get(PAIR[1], 0.0))})."
        )
    d0 = next(iter(data.values()))
    caption = (
        f"Share of attempts on the {d0['n_items']} {FAILURE_SUITE} theorems by how they end, for "
        f"the cap-12 seed-0 pair: the fixed-k runs, {FAILURE_K} draws per theorem "
        f"({', '.join(data[m]['run_id'] for m in present)}), each theorem weighted equally. "
        "Solved: Lean accepts the finished proof. Finished, rejected by Lean: the model closed "
        "the proof but Lean rejects it. Rejected by the environment: an action the environment "
        "cannot apply ends the attempt early; the indented rows split these by the "
        "environment's reason (largest "
        f"{N_REASONS} by either model, the rest pooled), as shares of all attempts. Point "
        "estimates. Judge: Lean (lean_v2)."
    )
    cd = ChartData(
        key,
        title,
        "table",
        takeaway,
        caption,
        bars=[
            P.Bar(f"{lab}: {m}", float(data[m][k])) for k, lab in FAILURE_GROUPS for m in present
        ],
        extra={
            "table": {"headers": headers, "rows": rows, "align": "l" + "r" * (len(headers) - 1)},
            "modes": {m: {k: data[m][k] for k, _l in FAILURE_GROUPS} for m in present},
        },
    )
    cd.extra_tables.append(
        (
            "every environment reason",
            ["reason", *(f"{m} (% of attempts)" for m in present)],
            [
                [r, *(f"{100 * reasons_all[r].get(m, 0.0):.2f}" for m in present)]
                for r in sorted(reasons_all, key=lambda r: -max(reasons_all[r].values()))
            ],
        )
    )
    return cd


# ========================================================================= 2. textbook


def _c_textbook(store: Store, reg: Mapping[str, mr.ModelEntry]) -> ChartData:
    key, title = "do_textbook", "Pass@k on textbook theorems, by cap and seed"
    base = dan_pages._data_dan_yue(store, reg, families=FAMS, B=B_TEXTBOOK)  # noqa: SLF001
    if base.kind == "missing":
        return _missing(key, title, "no textbook_dev runs for Dan's models yet")
    panels = {
        f: p
        for f, p in cast("dict[str, ChartData]", base.extra["panels"]).items()
        if f in FAMS and p.series
    }
    curves = {s.name: s.pts for p in panels.values() for s in p.series}
    n_items = next((n for n in _textbook_n(store)), "?")
    pairs = []
    for fam in FAMS:
        for seed in range(3):
            fz, t1 = curves.get(f"{fam}-fz-s{seed}"), curves.get(f"{fam}-t1-s{seed}")
            if fz and t1:
                common = sorted(set(dict(fz)) & set(dict(t1)))
                pairs.append(all(dict(t1)[k] >= dict(fz)[k] for k in common))
    bits = []
    for fam in FAMS:
        fz, t1 = curves.get(f"{fam}-fz-s0"), curves.get(f"{fam}-t1-s0")
        if not fz or not t1:
            continue
        k_hi = max(k for k in set(dict(fz)) & set(dict(t1)) if k <= 256)
        a1, b1 = dict(fz)[1.0], dict(t1)[1.0]
        ah, bh = dict(fz)[k_hi], dict(t1)[k_hi]
        bits.append(
            f"cap {fam[3:]} seed 0: pass@1 {_pct(a1)} → {_pct(b1)}, pass@{k_hi:g} {_pct(ah)} → "
            f"{_pct(bh)} (gap {100 * (b1 - a1):.0f} → {100 * (bh - ah):.0f} points)"
        )
    lead = (
        "RL helps at every sampling budget: the RL model is at or above the pretrained model at "
        f"every k in all {len(pairs)} "
        "seed pairs"
        if pairs and all(pairs)
        else f"the RL model is at or above the pretrained model at every k in {sum(pairs)} of "
        f"{len(pairs)} seed pairs"
    )
    k_max = {m: int(max(x for x, _y in pts)) for m, pts in curves.items()}
    kmax_note = "; ".join(
        f"cap {fam[3:]} "
        + ", ".join(
            f"{_role_name(m)} s{F.S.parse_model_label(m)[2]} {k}"
            for m, k in k_max.items()
            if m.startswith(fam + "-")
        )
        for fam in FAMS
        if any(m.startswith(fam + "-") for m in k_max)
    )
    seen12 = _seen_note(store, "textbook_dev", "dan12")
    seen6 = _seen_note(store, "textbook_dev", "dan6")
    seen_txt = ", ".join(
        f"{s[0]:,} of {s[1]:,} in the cap-{c} tiers" for c, s in (("12", seen12), ("6", seen6)) if s
    )
    caption = (
        f"Pass@k against samples per theorem k (log scale) on textbook_dev (n = {n_items} "
        "textbook theorems), (a) cap 12 and (b) cap 6 on one y scale; pretrained (Dan's Stage-1 "
        "checkpoint) "
        "darker, RL (after his T1 RL ladder) full hue. Seed 0 solid with a "
        f"{_BOOT} band (B = {B_TEXTBOOK}); seeds 1 and 2 dashed and dotted, no band. Unbiased "
        "pass@k from each item's first 256 draws up to k = 256, above that from every draw of the "
        "adaptive run, with the 1-(1-c/n)^k extrapolation past an item's own draw count; each "
        f"curve stops at its model's largest per-item draw count ({kmax_note}). Judge: Lean "
        "(lean_v2). Not filtered for training overlap"
        + (f" ({seen_txt})" if seen_txt else "")
        + "."
    )
    takeaway = lead + ("; " + "; ".join(bits) if bits else "") + "."
    pair = [s for s in panels.get("dan12", ChartData(key, title)).series if s.name in PAIR]
    return ChartData(
        key,
        title,
        "line",
        takeaway,
        caption,
        series=[s for p in panels.values() for s in p.series],
        opts={"x_label": "k", "y_fmt": "pct"},
        extra={"panels": panels, "n": n_items, "pair": pair},
    )


def _textbook_n(store: Store) -> list[int]:
    version = default_version(store, suite="textbook_dev", models=list(PAIR), judge_id=JUDGE)
    probs = problems_by_pid(store, "textbook_dev", version or "v1")
    return [len(probs)] if probs else []


# ========================================================================== Table 1


_ROLES: tuple[tuple[str, str, str], ...] = (
    ("dan12", "fz", "base"),
    ("dan12", "t1", "rl"),
    ("dan6", "fz", "base"),
    ("dan6", "t1", "rl"),
)


def _c_table1(store: Store, reg: Mapping[str, mr.ModelEntry], tb: ChartData) -> ChartData:
    key, title = "do_table1", "Headline textbook numbers against Dan's published counts"
    models = dv.dan_models(reg)
    lean = dv.textbook72_counts(store, models, JUDGE)
    nd = dv.textbook72_counts(store, models, dv.ND_JUDGE_ID)
    if not lean:
        return _missing(key, title, "no textbook_dev/textbook_train14 runs for Dan's models yet")
    curves = {s.name: dict(s.pts) for s in tb.series} if tb.kind != "missing" else {}
    headers = ["Model", "pass@1 (%)", "pass@256 (%)", "Solved / 72", "Dan's count"]
    rows: list[list[str]] = []
    raw: list[dict[str, Any]] = []
    details: list[list[str]] = []
    diffs: list[int] = []
    nd_diffs: list[int] = []
    t72: dict[str, dict[str, Any]] = {}
    k_first: set[int] = set()
    for fam, role, reg_role in _ROLES:
        labels = mr.seed_labels(reg, fam, reg_role)
        have = [m for m in labels if m in lean and "dev" in lean[m] and "train14" in lean[m]]
        if not have:
            continue
        ours = [lean[m]["dev"][0] + lean[m]["train14"][0] for m in have]
        k_first |= {lean[m]["dev"][2] for m in have} | {lean[m]["train14"][2] for m in have}
        pub = dv.DAN_PUBLISHED["textbook72"].get((fam, role), ())
        pub_dev = dv.DAN_PUBLISHED["dev58"].get((fam, role), ())
        seeds = [cast(int, reg[m].seed) for m in have]
        for m, o, s in zip(have, ours, seeds, strict=True):
            if s < len(pub):
                diffs.append(o - int(pub[s]))
            if m in nd and "dev" in nd[m] and "train14" in nd[m]:
                nd_diffs.append(o - (nd[m]["dev"][0] + nd[m]["train14"][0]))
            details.append(
                [
                    m,
                    str(lean[m]["dev"][0]),
                    str(int(pub_dev[s])) if s < len(pub_dev) else "-",
                    str(lean[m]["train14"][0]),
                    str(int(pub[s]) - int(pub_dev[s]))
                    if s < len(pub) and s < len(pub_dev)
                    else "-",
                    str(o),
                    str(int(pub[s])) if s < len(pub) else "-",
                    (
                        str(nd[m]["dev"][0] + nd[m]["train14"][0])
                        if m in nd and "train14" in nd[m]
                        else "not rejudged"
                    ),
                ]
            )
        p1 = [curves[m][1.0] for m in have if m in curves and 1.0 in curves[m]]
        p256 = [curves[m][256.0] for m in have if m in curves and 256.0 in curves[m]]
        raw.append(
            {
                "fam": fam,
                "role": role,
                "p1": p1,
                "p256": p256,
                "ours": ours,
                "pub": [int(x) for x in pub],
            }
        )
        t72[f"{fam}-{role}"] = {"ours": ours, "dan": [int(x) for x in pub]}
    best = {
        c: max((statistics.mean(r[c]) for r in raw if r[c]), default=None)
        for c in ("p1", "p256", "ours")
    }

    def _cell(vals: list[float], col: str, scale: float = 100.0) -> str:
        if not vals:
            return "n/a"
        mean = statistics.mean(vals)
        txt = f"{scale * mean:.1f}"
        txt = f"<b>{txt}</b>" if best[col] is not None and mean == best[col] else txt
        lo, hi = f"{scale * min(vals):.1f}", f"{scale * max(vals):.1f}"
        if len(vals) > 1 and lo != hi:
            txt += f' <span class="rng">{lo}–{hi}</span>'
        return txt

    for r in raw:
        solved = " / ".join(str(o) for o in r["ours"])
        if best["ours"] is not None and statistics.mean(r["ours"]) == best["ours"]:
            solved = f"<b>{solved}</b>"
        rows.append(
            [
                f"{F.ROLE_NAME.get(r['role'], r['role'])}, cap {r['fam'][3:]}",
                _cell(r["p1"], "p1"),
                _cell(r["p256"], "p256"),
                solved,
                " / ".join(str(x) for x in r["pub"]) or "-",
            ]
        )
    max_abs = max((abs(d) for d in diffs), default=None)
    mean_abs = statistics.mean(abs(d) for d in diffs) if diffs else None
    gains = {
        fam: statistics.mean(next(r["ours"] for r in raw if r["fam"] == fam and r["role"] == "t1"))
        - statistics.mean(next(r["ours"] for r in raw if r["fam"] == fam and r["role"] == "fz"))
        for fam in FAMS
        if {"fz", "t1"} <= {r["role"] for r in raw if r["fam"] == fam}
    }
    k_txt = "/".join(str(k) for k in sorted(k_first))
    nd_txt = (
        "nd_v2 on the same samples changes no count"
        if nd_diffs and all(d == 0 for d in nd_diffs)
        else f"nd_v2 on the same samples changes counts by at most {max(map(abs, nd_diffs))}"
        if nd_diffs
        else "no nd_v2 rejudge yet"
    )
    caption = (
        f"textbook_dev (58 theorems) pass@1 and pass@256 ({_ref('do_textbook')}'s estimator; mean "
        "over seeds, "
        "seed range in grey) and textbook72 = textbook_dev + textbook_train14 (14) theorems "
        f"solved within each item's first {k_txt} draws, listed for seeds 0 / 1 / 2 "
        "(T = 0.8, max_action 512, max_steps 96) "
        "per seed, next to Dan's own per-seed counts from his best-state summary (same "
        "checkpoints and protocol, his sampler and seeds). Judge: Lean (lean_v2); "
        f"{nd_txt}. Point estimates per seed; bold: best per column."
    )
    takeaway = (
        (
            f"Our counts reproduce Dan's to within {max_abs} theorems of 72 per checkpoint "
            f"(mean |difference| {mean_abs:.1f})"
            if max_abs is not None and mean_abs is not None
            else "Dan's published counts are not available for these checkpoints"
        )
        + (
            "; RL adds "
            + " and ".join(f"{g:.1f} theorems at cap {fam[3:]}" for fam, g in gains.items())
            + " (seed means)"
            if gains
            else ""
        )
        + "."
    )
    cd = ChartData(
        key,
        title,
        "table",
        takeaway,
        caption,
        extra={"table": {"headers": headers, "rows": rows, "align": "lrrrr"}, "t72": t72},
    )
    cd.extra_tables.append(
        (
            "per seed, dev58 / train14 split",
            [
                "model",
                "dev58 ours",
                "dev58 Dan",
                "train14 ours",
                "train14 Dan",
                "72 ours",
                "72 Dan",
                "72 nd_v2",
            ],
            details,
        )
    )
    return cd


# ========================================================================= 3. families


def _family_points(cd: ChartData) -> dict[str, dict[str, tuple[float | None, bool]]]:
    """`{family row: {model: (written lines at n*, censored)}}` of a `_data_dan_families` chart."""
    panels = cast("dict[str, ChartData]", cd.extra.get("panels") or {})
    out: dict[str, dict[str, tuple[float | None, bool]]] = {}
    for p in panels.values():
        for r in p.rows:
            for n, v, _c, z in r.points:
                out.setdefault(r.label, {})[n] = (v, z)
    return out


def _horizon_txt(v: tuple[float | None, bool] | None) -> str:
    if v is None:
        return "no qualifying cell"
    lines, cens = v
    if lines is None:
        return "fails at the smallest size"
    return f"{'>= ' if cens else ''}{lines:.1f} lines"


def _c_families(store: Store, reg: Mapping[str, mr.ModelEntry]) -> ChartData:
    key, title = "do_families", "Length horizon on restricted families (unseen theorems)"
    base = dan_pages._data_dan_families(store, reg, families=FAMS, keep=dv.in_premise_range)  # noqa: SLF001
    if base.kind == "missing":
        return _missing(key, title, "no families runs for Dan's models yet")
    version = default_version(store, suite="families", models=list(PAIR), judge_id=JUDGE) or "v1"
    probs = problems_by_pid(store, "families", version)
    groups = summary._family_group_pids(probs)  # noqa: SLF001

    def row_of(group: str) -> str:
        return "ILTP-style" if group.startswith("ILTP-style") else group

    out_of_range = sorted(
        {
            row_of(g)
            for g, pids in groups.items()
            if pids and not any(dv.in_premise_range(probs[p]) for p in pids)
        }
    )
    row_counts: dict[str, list[int]] = {}
    for g, pids in groups.items():
        acc = row_counts.setdefault(row_of(g), [0, 0])
        acc[0] += sum(1 for p in pids if not dv.in_premise_range(probs[p]))
        acc[1] += len(pids)
    n_out_all = sum(1 for p in probs.values() if not dv.in_premise_range(p))
    panels = cast("dict[str, ChartData]", base.extra["panels"])
    rows_by: dict[str, list[tuple[str, float | None, str, bool]]] = {}
    for fam in FAMS:
        for r in panels.get(fam, ChartData(key, title)).rows:
            rows_by.setdefault(r.label, []).extend(r.points)
    order = [o for o in cast("list[str]", base.extra["row_order"]) if o in rows_by]
    order += [o for o in out_of_range if o not in order]
    rows = [P.DotRow(o, rows_by.get(o, [])) for o in order]
    x_hi = max([v or 0.0 for r in rows for _n, v, _c, _z in r.points] + [1.0])
    longer = shorter = equal = 0
    fz_fail = t1_fail = 0
    both_cens: list[str] = []
    for r in rows:
        pts = {n: (v, z) for n, v, _c, z in r.points}
        if PAIR[0] not in pts or PAIR[1] not in pts:
            continue
        (a, za), (b, zb) = pts[PAIR[0]], pts[PAIR[1]]
        fz_fail += a is None
        t1_fail += b is None
        av, bv = a or 0.0, b or 0.0
        if za and zb and av == bv:
            both_cens.append(f"{r.label} (>= {av:.0f} lines)")
            equal += 1
        elif bv > av:
            longer += 1
        elif bv < av:
            shorter += 1
        else:
            equal += 1
    n_cmp = longer + shorter + equal
    # what the ceiling changes: the same chart on every item, cap-12 pair only
    full = _family_points(dan_pages._data_dan_families(store, reg, families=("dan12",)))  # noqa: SLF001
    kept = _family_points(base)
    changed: dict[str, list[str]] = {}
    for label in sorted(set(full) | set(kept)):
        if label in out_of_range:
            continue
        for m in PAIR:
            before, after = full.get(label, {}).get(m), kept.get(label, {}).get(m)
            if before == after:
                continue
            pids_all = [p for g, ps in groups.items() if row_of(g) == label for p in ps]
            seen = mr.seen_lookup(store, "families", version, m, probs, registry=reg)
            unseen = [p for p in pids_all if not seen.get(p, probs[p].seen_train)]
            stars = [
                summary._family_horizon(  # noqa: SLF001
                    store, "families", version, m, pids, probs, judge_id=JUDGE, B=0
                )
                for pids in (unseen, [p for p in unseen if dv.in_premise_range(probs[p])])
            ]
            n_txt = (
                f"n* = {stars[0][0]} either way"
                if stars[0] is not None and stars[1] is not None and stars[0][0] == stars[1][0]
                else "n* " + " → ".join(str(x[0]) if x is not None else "none" for x in stars)
            )
            changed.setdefault(label, []).append(
                f"{_role_name(m)} {n_txt}, {_horizon_txt(before)} on every item against "
                f"{_horizon_txt(after)} within the range"
            )
    changes = [f"{label} ({'; '.join(v)})" for label, v in changed.items()]
    seen = _seen_note(store, "families")
    k = _run_k(store, "families", PAIR[1])
    takeaway = (
        f"the RL model's horizon is longer than the pretrained model's in {longer} of {n_cmp} "
        f"families, equal in {equal} "
        f"and shorter in {shorter}; the pretrained model fails at the smallest unseen size in "
        f"{fz_fail} families, "
        f"the RL model in {t1_fail}"
        + (f"; both are censored at the same length in {', '.join(both_cens)}" if both_cens else "")
        + (
            f"; {', '.join(out_of_range)} {'is' if len(out_of_range) == 1 else 'are'} out of "
            f"range (every item has {MAXP + 1} or more premises)"
            if out_of_range
            else ""
        )
        + "."
    )
    which = [
        f"all {row_counts[lab][1]:,} {lab} items, so that family is out of range"
        for lab in out_of_range
    ] + [f"{a:,} of {b:,} {lab}" for lab, (a, b) in sorted(row_counts.items()) if 0 < a < b]
    caption = (
        "Horizon of each restricted family (one row per family): the mean written length "
        "(lines) of the items at n*, the largest family size n at which pass@8 is still at "
        f"least 50%, over size cells with >= {summary._MIN_ITEMS_PER_N} theorems with at most "  # noqa: SLF001
        f"{MAXP} premises unseen in Dan's cap-12 training tiers (his Stage-1 set plus the T1 "
        "ladder's RL targets"
        + (f"; {seen[0]:,} of {seen[1]:,} items are seen and excluded" if seen else "")
        + f"). {_cap_first(_beyond(n_out_all, len(probs)))}"
        + (f": {', '.join(which)}" if which else "")
        + ". The restriction changes "
        + ("; ".join(changes) if changes else "no other family's horizon")
        + ("; every other family is unchanged" if changes else "")
        + f". k = {k or '?'} draws per item ({len(probs):,} items in the suite). Hollow marker "
        "at 0: fails at the smallest unseen size; arrow: censored (never below 50% in the tested "
        "range, a lower bound). Point estimates. Judge: Lean (lean_v2)."
    )
    return ChartData(
        key,
        title,
        "dot",
        takeaway,
        caption,
        rows=rows,
        opts={"x_label": "written lines at n*", "none_label": "fails at smallest size"},
        extra={"row_order": order, "x_hi": x_hi, "out_of_range": out_of_range},
    )


# ============================================================================ 4. laws


def _c_laws(store: Store, reg: Mapping[str, mr.ModelEntry]) -> ChartData:
    key, title = "do_laws", "Instances of known laws against substitution size"
    base = dan_pages._data_dan_laws(store, reg, families=FAMS)  # noqa: SLF001
    if base.kind == "missing":
        return _missing(key, title, "no laws runs for Dan's models yet")
    series = [s for s in base.series if s.pts]
    curves = {s.name: s.pts for s in series}
    fz, t1 = curves.get(PAIR[0], []), curves.get(PAIR[1], [])
    common = sorted(set(dict(fz)) & set(dict(t1)))
    takeaway = "pass@1 against s for the models with runs."
    if common:
        above = sum(dict(t1)[s] >= dict(fz)[s] for s in common)
        s0 = common[0]
        floor = next((s for s in common if dict(t1)[s] < 0.05), None)
        takeaway = (
            f"the RL model is at or above the pretrained model at {above} of {len(common)} "
            "substitution sizes, from "
            f"{_pct(dict(t1)[s0])} vs {_pct(dict(fz)[s0])} at s = {s0:g}"
            + (
                f"; both are below 5% from s = {floor:g} (RL {_pct(dict(t1)[floor])}, pretrained "
                f"{_pct(dict(fz)[floor])})"
                if floor is not None
                else ""
            )
            + "."
        )
    version = default_version(store, suite="laws", models=list(PAIR), judge_id=JUDGE) or "v1"
    lprobs = problems_by_pid(store, "laws", version)
    n_items = len(lprobs)
    seen = _seen_note(store, "laws")
    train_goals = dv.training_goal_tokens(dv.dan_training_files("dan12")[:2])
    q99 = dv.quantile(train_goals, 0.99) if train_goals else None
    goal_by_s: dict[float, list[int]] = {}
    for p in lprobs.values():
        if p.tags.get("s") is not None:
            goal_by_s.setdefault(float(cast(float, p.tags["s"])), []).append(
                len(p.sequent.conclusion.split())
            )
    goal_med = {k: statistics.median(v) for k, v in sorted(goal_by_s.items())}
    past = next((k for k, v in goal_med.items() if q99 is not None and v > q99), None)
    goal_txt = (
        f" The goal formula grows with s (median {goal_med[min(goal_med)]:g} tokens at s = "
        f"{min(goal_med):g}, {goal_med[max(goal_med)]:g} at s = {max(goal_med):g}); from s = "
        f"{past:g} the median goal is longer than 99% of the cap-12 training theorems' goals "
        f"({q99} tokens)."
        if goal_med and past is not None
        else ""
    )
    caption = (
        "Pass@1 against the substitution size s (connectives substituted per metavariable) for "
        f"instances of 8 known law schemas ({n_items:,} items), pooled per item over the schemas "
        f"at each s; s cells with < {summary._MIN_ITEMS_PER_BIN} items omitted. Instances in Dan's "  # noqa: SLF001
        "cap-12 training tiers are excluded"
        + (f" ({seen[0]:,} of {seen[1]:,})" if seen else "")
        + f". k = {_run_k(store, 'laws', PAIR[1]) or '?'} draws per item."
        + goal_txt
        + " Point estimates. Judge: Lean (lean_v2)."
    )
    return ChartData(
        key, title, "line", takeaway, caption, series=series, opts={"x_label": "s", "y_fmt": "pct"}
    )


# ============================================================================ 5. glue


def _c_glue(store: Store, reg: Mapping[str, mr.ModelEntry]) -> ChartData:
    key, title = "do_glue", "Compositionality of solved parts"
    from ..analysis import glue as glue_view

    suite = glue_view.DEFAULT_SUITE
    version = default_version(store, suite=suite, models=list(PAIR), judge_id=JUDGE) or "v1"
    problems = problems_by_pid(store, suite, version)
    pids_by_op: dict[str, list[str]] = {}
    all_by_op: dict[str, int] = {}
    for pid, p in problems.items():
        op = summary._OP_GROUP.get(cast(str, p.tags.get("op", "")))  # noqa: SLF001
        if op:
            all_by_op[op] = all_by_op.get(op, 0) + 1
            if dv.in_premise_range(p):
                pids_by_op.setdefault(op, []).append(pid)
    ops = [op for op in summary._OP_ORDER if op in pids_by_op]  # noqa: SLF001
    models = [m for fam in FAMS for m in mr.labels_in_family(reg, fam)]
    bars: list[P.Bar] = []
    used: list[list[str]] = []
    for op in ops:
        for m in models:
            comp = dan_pages._glue_ratio(store, m, pids_by_op[op], JUDGE)  # noqa: SLF001
            if comp is None or comp["R"] != comp["R"]:
                continue
            lo, hi = comp["ci"]
            bars.append(P.Bar(f"{op}: {m}", float(comp["R"]), lo, hi))
            used.append([op, m, str(len(pids_by_op[op])), str(comp["n_used"])])
    if not bars:
        return _missing(key, title, "no glue runs for Dan's models yet")
    by = {b.name: b for b in bars}
    below = [
        op
        for op in ops
        if all(
            (b := by.get(f"{op}: {m}")) is None or (b.hi is not None and b.hi < 1.0) for m in PAIR
        )
        and any(f"{op}: {m}" in by for m in PAIR)
    ]
    higher = [
        op
        for op in ops
        if f"{op}: {PAIR[0]}" in by
        and f"{op}: {PAIR[1]}" in by
        and by[f"{op}: {PAIR[1]}"].value > by[f"{op}: {PAIR[0]}"].value
    ]
    best_op = max(
        ops, key=lambda op: by[f"{op}: {PAIR[1]}"].value if f"{op}: {PAIR[1]}" in by else -1.0
    )
    takeaway = (
        f"On joins with at most {MAXP} premises, joined theorems are harder than their mastered "
        f"parts predict for both models (R < 1 "
        f"with the CI below 1 on {len(below)} of {len(ops)} operators: {', '.join(below) or '-'})"
        f"; the RL model's R is higher than the pretrained model's on {len(higher)} of {len(ops)}, "
        f"closest to 1 on {best_op} "
        f"(RL {by[f'{best_op}: {PAIR[1]}'].value:.2f}"
        + (
            f", pretrained {by[f'{best_op}: {PAIR[0]}'].value:.2f}"
            if f"{best_op}: {PAIR[0]}" in by
            else ""
        )
        + ")."
    )
    dropped = {op: all_by_op[op] - len(pids_by_op.get(op, [])) for op in all_by_op}
    out_p1 = [
        cast(float, d["p_out"])
        for m in PAIR
        if (d := dv.premise_split(store, suite, m, judge_id=JUDGE)) and d["n_out"]
    ]
    solved_out = (
        "both models solve none of them"
        if out_p1 and max(out_p1) == 0
        else f"both models solve at most {_pct(max(out_p1))} of them"
        if out_p1
        else "they have no runs"
    )
    caption = (
        "Compositionality ratio R per glue operator: joined pass@1 divided by the product of "
        "its parts' pass@1, pooled over the joined items whose parts each have pass@1 >= 0.5 "
        "for that model (parts sampled as they appear inside the join, glue_parts_renamed), on "
        f"the joins with at most {MAXP} premises: "
        + "; ".join(f"{op} n = {len(pids_by_op[op])}" for op in ops)
        + " joined items, of which the mastered subsets are in the data table (CASE pools the "
        "dilemma and same-conclusion joins). "
        + _cap_first(_beyond(sum(dropped.values()), sum(all_by_op.values()), "joins"))
        + " ("
        + ", ".join(
            f"{op} {dropped[op]} of {all_by_op[op]}"
            if dropped[op]
            else f"{op} none of {all_by_op[op]}"
            for op in summary._OP_ORDER  # noqa: SLF001
            if op in all_by_op
        )
        + f"): {solved_out}, so on every join R would mostly measure the "
        "ceiling. Dotted line: R = 1. Error bars: 95% item-level "
        f"bootstrap CIs (B = 1000). k = {_run_k(store, 'glue', PAIR[1]) or '?'} draws per joined "
        "item. Not filtered for training overlap. Judge: Lean (lean_v2)."
    )
    cd = ChartData(
        key,
        title,
        "bar",
        takeaway,
        caption,
        bars=bars,
        opts={"y_fmt": "f3"},
        extra={"ops": ops, "counts": {op: len(v) for op, v in pids_by_op.items()}},
    )
    cd.extra_tables.append(("items used", ["op", "model", "joined items", "parts mastered"], used))
    return cd


# =========================================================================== 6. skills


def _skills_by_length(
    store: Store, model: str, *, min_items: int = 20, keep: Any = dv.in_premise_range
) -> list[P.Series]:
    """Mean pass@1 per (skill_group, L) cell on skill_pool_sub's items that `keep` admits (default:
    at most 3 premises), cells with < `min_items` items dropped (the main page's
    `_MIN_SKILL_CELL_ITEMS` rule)."""
    suite = "skill_pool_sub"
    run_id = latest_run_id(store, suite=suite, model=model, version="v1", judge_id=JUDGE)
    if run_id is None:
        return []
    problems = problems_by_pid(store, suite, "v1")
    cells: dict[str, dict[int, list[float]]] = {}
    for pid, it in items_by_pid(store, run_id).items():
        p = problems.get(pid)
        if p is None or it.n <= 0 or p.length.L is None or (keep is not None and not keep(p)):
            continue
        v = pass_at_k(it.n, it.c, 1)
        if v == v:
            cells.setdefault(str(p.tags.get("skill_group", "plain")), {}).setdefault(
                int(p.length.L), []
            ).append(v)
    out = []
    for group, by_l in cells.items():
        pts = [
            (float(L), statistics.mean(v)) for L, v in sorted(by_l.items()) if len(v) >= min_items
        ]
        if pts:
            out.append(P.Series(group, pts, "var(--s1)"))
    return out


#: Smallest (group, L) cell drawn in the skills-by-length panels: the main page's 20, lowered
#: because the premise restriction leaves 15-25 items in most cells.
MIN_SKILL_CELL = 15


def _c_skills(store: Store, reg: Mapping[str, mr.ModelEntry]) -> ChartData:
    key, title = "do_skills", "Pass@1 by required skill"
    base = dan_pages._data_dan_skills(store, reg, families=FAMS, keep=dv.in_premise_range)  # noqa: SLF001
    if base.kind == "missing":
        return _missing(key, title, "no skill_pool_sub runs for Dan's models yet")
    bars = list(base.bars)
    by = {b.name: b.value for b in bars}
    models = list(dict.fromkeys(b.name.rpartition(": ")[2] for b in bars))
    by_len = {m: _skills_by_length(store, m, min_items=MIN_SKILL_CELL) for m in models}
    problems = problems_by_pid(store, "skill_pool_sub", "v1")
    sizes: dict[str, int] = {}
    n_out = 0
    for p in problems.values():
        if not dv.in_premise_range(p):
            n_out += 1
            continue
        g = str(p.tags.get("skill_group", "plain"))
        sizes[g] = sizes.get(g, 0) + 1

    def v(group: str, m: str) -> float | None:
        return by.get(f"{group}: {m}")

    t1, fz = PAIR[1], PAIR[0]
    takeaway = "pass@1 by skill group."
    if v("case_split", t1) is not None and v("plain", t1) is not None:
        cs, pl = cast(float, v("case_split", t1)), cast(float, v("plain", t1))
        lines = {s.name: dict(s.pts) for s in by_len.get(t1, [])}
        common = sorted(set(lines.get("case_split", {})) & set(lines.get("plain", {})))
        at_or_above = sum(lines["case_split"][L] >= lines["plain"][L] for L in common)
        hardest = min(
            (g for g in sizes if v(g, t1) is not None), key=lambda g: cast(float, v(g, t1))
        )
        verb = "do not cost the RL model anything" if cs >= pl else "cost the RL model"
        takeaway = (
            f"On items with at most {MAXP} premises, case splits {verb}: it solves {_pct(cs)} of "
            f"case-split items vs {_pct(pl)} of plain ones"
            + (
                f" (pretrained {_pct(v('case_split', fz))} vs {_pct(v('plain', fz))})"
                if v("case_split", fz) is not None
                else ""
            )
            + (
                f", and is at or above plain at {at_or_above} of {len(common)} matched lengths"
                if common
                else ""
            )
            + f"; {hardest.replace('_', ' ')} is the hardest group (RL "
            f"{_pct(v(hardest, t1))}"
            + (f", pretrained {_pct(v(hardest, fz))}" if v(hardest, fz) is not None else "")
            + ")."
        )
    caption = (
        "Pass@1 by required skill on skill_pool_sub, a stratified subsample of skill_pool (up to "
        f"25 items per (L, group) cell; {len(problems)} items), on its {sum(sizes.values())} "
        f"items with at most {MAXP} premises: "
        + ", ".join(f"{g.replace('_', ' ')} {n}" for g, n in sorted(sizes.items()))
        + f". {_cap_first(_beyond(n_out, len(problems)))}. (a) Mean pass@1 over each group's "
        "items; (b, c) the same by written length L, "
        f"cells with < {MIN_SKILL_CELL} items omitted, so the groups are compared "
        "at matched length. Groups: plain (neither skill), case split (needs an or-elimination), "
        "classical (needs a classical step), both. k = "
        f"{_run_k(store, 'skill_pool_sub', t1) or '?'} draws per item. Point estimates. Not "
        "filtered for training overlap. Judge: Lean (lean_v2)."
    )
    cd = ChartData(
        key,
        title,
        "bar",
        takeaway,
        caption,
        bars=bars,
        opts={"y_fmt": "pct"},
        extra={"by_length": {m: s for m, s in by_len.items() if s}},
    )
    for m, ss in by_len.items():
        if ss:
            cd.extra_tables.append(
                (f"{m} by length (b, c)", *P.line_table_rows(ss, x_label="L", y_fmt="pct"))
            )
    return cd


# ========================================================================== 7. padding


def _span(ns: Sequence[int]) -> str:
    """`"34"`, `"37-40"` or `"-"` (no padded points)."""
    if not ns:
        return "-"
    lo, hi = min(ns), max(ns)
    return str(lo) if lo == hi else f"{lo}-{hi}"


#: Points of the padding figure with fewer items are left out.
MIN_PAD_POINT = 5


def _p1_pooled(
    by_type: Mapping[str, Sequence[tuple[int, float, int]]],
) -> list[tuple[float, float]]:
    """Both P1 kinds pooled per total premise count (item-weighted), for the overview panel."""
    acc: dict[int, list[float]] = {}
    for t in ("P1-shared", "P1-disjoint"):
        for k, p, n in by_type.get(t, []):
            a = acc.setdefault(k, [0.0, 0.0])
            a[0] += p * n
            a[1] += n
    return [(float(k), a[0] / a[1]) for k, a in sorted(acc.items()) if a[1] >= MIN_PAD_POINT]


def _p2_goal_note(store: Store, probs: Mapping[str, Problem], model: str) -> tuple[str, bool]:
    """The caption sentences on how P2 pads also grow the goal formula, against the training
    goals, with `model`'s most common way to fail at large m (empty without the data or when the
    goal does not grow), and whether the median goal at the largest m is longer than 99% of the
    training goals."""
    by_m: dict[int, list[Problem]] = {}
    for p in probs.values():
        if p.tags.get("pad_type") == "P2" or int(cast(int, p.tags.get("m", 0))) == 0:
            by_m.setdefault(int(cast(int, p.tags.get("m", 0))), []).append(p)
    train = dv.training_goal_tokens(dv.dan_training_files("dan12")[:2])
    if not by_m or not train:
        return "", False
    med = {
        m: statistics.median(len(p.sequent.conclusion.split()) for p in ps)
        for m, ps in sorted(by_m.items())
    }
    if med[max(med)] <= med[min(med)]:
        return "", False
    q50, q99 = dv.quantile(train, 0.5), dv.quantile(train, 0.99)
    past = q99 is not None and med[max(med)] > q99
    text = (
        "P2 also lengthens the goal formula: its median is "
        + " / ".join(f"{v:g}" for v in med.values())
        + f" tokens at m = {min(med)}-{max(med)}, against a median of {q50} and a 99th "
        f"percentile of {q99} over the cap-12 training theorems' goals. "
    )
    found = dv._latest(store, "padding", model, JUDGE)  # noqa: SLF001
    m_hi = max(med)
    if found is not None and m_hi > 0:
        ends = {
            m: dv.end_reasons(store, found[1], [p.pid for p in by_m[m]])
            for m in (3, m_hi)
            if m in by_m
        }
        top = {m: c.most_common(1)[0] for m, c in ends.items() if c}
        if top and all(r == "formula" for r, _n in top.values()):
            text += (
                f"From m = {min(top)} the {_role_name(model)} model's most common end is a "
                "malformed formula ("
                + ", ".join(
                    f"{100 * n / sum(ends[m].values()):.0f}% of attempts at m = {m}"
                    for m, (_r, n) in top.items()
                )
                + ")"
            )
            lines = statistics.mean(p.length.L or 0 for p in by_m[m_hi])
            sealed = dv.premise_split(store, "interp_eval_sealed", model, judge_id=JUDGE)
            text += (
                f", though P2 theorems at m = {m_hi} need only {lines:.1f} lines on average"
                + (
                    f" and the {_role_name(model)} model solves {_pct(sealed['p_in'])} of the "
                    f"12-14-line interp_eval_sealed theorems with at most {MAXP} premises"
                    if sealed
                    else ""
                )
                + ", so the P2 decline may reflect the goal's size more than the proof's length. "
            )
    return text, past


def _c_padding(store: Store, reg: Mapping[str, mr.ModelEntry]) -> ChartData:
    key, title = "do_padding", "Padding: distractor premises against a longer proof and goal"
    models = [m for fam in FAMS for m in mr.labels_in_family(reg, fam)]
    byp = {m: d for m in models if (d := dv.padding_by_premises(store, m, judge_id=JUDGE))}
    if not byp:
        return _missing(key, title, "no padding runs for Dan's models yet")
    p2 = {
        m: d["P2"]
        for m in byp
        if (d := dv.padding_by_pad_type(store, m, judge_id=JUDGE, keep=dv.in_premise_range))
        and "P2" in d
    }
    series: list[P.Series] = []
    for m, by_type in byp.items():
        for t in dv.PAD_TYPES:
            pts = [(float(k), p) for k, p, n in by_type.get(t, []) if n >= MIN_PAD_POINT]
            if pts:
                series.append(P.Series(f"{t}: {m}", pts, "var(--s1)"))
    p2_series = [
        P.Series(m, [(float(mm), p) for mm, p, n, _w in pts if n >= MIN_PAD_POINT], "var(--s1)")
        for m, pts in p2.items()
    ]
    mini = {m: _p1_pooled(by_type) for m, by_type in byp.items()}
    fz, t1 = PAIR
    version = default_version(store, suite="padding", models=list(PAIR), judge_id=JUDGE) or "v1"
    probs = problems_by_pid(store, "padding", version)
    goal, goal_past = _p2_goal_note(store, probs, t1)
    takeaway = "pass@1 of padded theorems by premises and by pads."

    cost = {m: dv.padding_paired_cost(store, m, judge_id=JUDGE) for m in PAIR if m in byp}
    types = ("P1-shared", "P1-disjoint")

    def drop(m: str, side: str) -> str:
        c = cost.get(m, {})
        vals = [c[t][side][1] - c[t][side][0] for t in types if side in c.get(t, {})]
        return " / ".join(f"{100 * v:.1f}" for v in vals)

    if all(side in cost.get(t1, {}).get(t, {}) for t in types for side in ("in", "out")):
        takeaway = (
            "The padding drop is the premise ceiling: against their own unpadded theorems, P1 "
            f"pads (shared / disjoint) cost the RL model {drop(t1, 'in')} points of pass@1 while "
            f"the total stays at {MAXP} premises or fewer and {drop(t1, 'out')} points once it "
            f"reaches {MAXP + 1} or more"
            + (
                f"; the pretrained model loses {drop(fz, 'in')} and {drop(fz, 'out')} points, so "
                "for it distractors also cost something below the ceiling"
                if all(side in cost.get(fz, {}).get(t, {}) for t in types for side in ("in", "out"))
                else ""
            )
        )
        if t1 in p2 and p2[t1]:
            by_m = {mm: p for mm, p, _n, _w in p2[t1]}
            m_hi = max(by_m)
            takeaway += (
                f"; P2 pads, which add one ∧I line each and no premise, take the RL model from "
                f"{_pct(by_m.get(0))} unpadded to {_pct(by_m.get(2))} at m = 2 and "
                f"{_pct(by_m[m_hi])} at m = {m_hi}"
            )
            if fz in p2 and p2[fz]:
                e = {mm: p for mm, p, _n, _w in p2[fz]}
                takeaway += (
                    f" (pretrained {_pct(e.get(0))} → {_pct(e.get(2))} → {_pct(e.get(m_hi))})"
                )
            if goal_past:
                takeaway += ", while also growing the goal formula past almost every training goal"
        takeaway += "."
    any_m = next(iter(byp))
    n_types = {t: [n for _k, _p, n in byp[any_m].get(t, [])] for t in dv.PAD_TYPES}
    n_out = sum(1 for p in probs.values() if not dv.in_premise_range(p))
    out_types = sorted(
        {str(p.tags.get("pad_type")) for p in probs.values() if not dv.in_premise_range(p)}
    )
    unp = byp[any_m].get("unpadded", [])
    n_bases = sum(n for _k, _p, n in unp)
    p2_n = [n for _m, _p, n, _w in p2.get(any_m, [])]
    caption = (
        "Padded versions of short theorems. (a) Pass@1 against the theorem's total number of "
        "premises, per pad type, padded items only (m >= 1 pads): P1 appends m extra premises "
        "over the theorem's own atoms (shared) or fresh atoms (disjoint); under Lean premises "
        "need no restating, so in Dan's environment P1 pads add hypotheses to the proof state, "
        "not lines. P2 pads add no premise, so P2 items sit at their unpadded theorem's count. "
        f"Shaded: {MAXP + 1} or more premises, which no training theorem has; "
        f"{n_out:,} of the suite's {len(probs):,} items are there, all "
        + " and ".join(out_types)
        + ". (b) P2 alone against pads added m: each pad extends the goal C to ((C ∧ X1) ∧ X2) "
        "... with existing premises Xi, one real ∧-introduction line per pad; m = 0 is the "
        f"{n_bases} unpadded theorems, each counted once (all with at most {MAXP} premises). "
        + goal
        + f"Points with < {MIN_PAD_POINT} items omitted; items per point: "
        + ", ".join(f"{t} {_span(v)}" for t, v in n_types.items() if v)
        + (f" in (a), {_span(p2_n[1:])} per padded point in (b)" if p2_n else "")
        + f". k = {_run_k(store, 'padding', t1) or '?'} draws per item. Point estimates. Not "
        "filtered for training overlap. Judge: Lean (lean_v2)."
    )
    cd = ChartData(
        key,
        title,
        "line",
        takeaway,
        caption,
        series=series,
        opts={"x_label": "premises", "y_fmt": "pct"},
        extra={"p2": p2_series, "mini": mini, "n_bases": n_bases, "n_out": n_out},
    )
    if p2_series:
        cd.extra_tables.append(
            ("P2 by pads added (b)", *P.line_table_rows(p2_series, x_label="m", y_fmt="pct"))
        )
    return cd


# =========================================================================== 8. prefix


def _c_prefix(store: Store, reg: Mapping[str, mr.ModelEntry]) -> ChartData:
    key, title = "do_prefix", "Completing a given proof prefix"
    models = [m for fam in FAMS for m in mr.labels_in_family(reg, fam)]
    min_bin = summary._MIN_PREFIX_BIN_ITEMS  # noqa: SLF001
    curves = {
        m: pts
        for m in models
        if (pts := dv.prefix_by_lines_left(store, m, judge_id=JUDGE, min_items=min_bin))
    }
    if not curves:
        return _missing(key, title, "no prefix_state runs for Dan's models yet")
    series = [P.Series(m, [(x, y) for x, y, _n in pts], "var(--s1)") for m, pts in curves.items()]
    by = {s.name: dict(s.pts) for s in series}
    fz, t1 = by.get(PAIR[0], {}), by.get(PAIR[1], {})
    common = sorted(set(fz) & set(t1))
    probs = problems_by_pid(store, "prefix_state", "v1")
    n_drop = sum(1 for p in probs.values() if not dv.prefix_in_range(p))
    n_forced = sum(
        1 for p in probs.values() if not dv.in_premise_range(p) and dv.premises_forced(p)
    )
    full = {
        m: {
            x: y
            for x, y, _n in dv.prefix_by_lines_left(
                store, m, judge_id=JUDGE, keep=None, min_items=min_bin
            )
        }
        for m in PAIR
        if m in curves
    }
    takeaway = "prefix completion by lines left."
    if common:
        above = sum(t1[x] >= fz[x] for x in common)
        lo, hi = common[0], common[-1]
        takeaway = (
            f"the RL model completes more prefixes than the pretrained model at {above} of "
            f"{len(common)} distances: "
            f"{_pct(t1[lo])} vs {_pct(fz[lo])} with {lo:g} line{'s' if lo != 1 else ''} left, "
            f"{_pct(t1[hi])} vs {_pct(fz[hi])} with {hi:g} left"
            + (
                f" (on every item, {_pct(full[PAIR[1]][hi])} vs {_pct(full[PAIR[0]][hi])})"
                if all(hi in full.get(m, {}) for m in PAIR)
                else ""
            )
            + "."
        )
    caption = (
        "Pass@1 of completing a reference proof whose first j lines are given, against lines "
        "left to write (L - j): the first j lines are replayed as a forced action prefix in "
        f"Dan's environment (prefix_state, {len(probs):,} (proof, j) items); bins with < "
        f"{min_bin} items omitted. Of the items with {MAXP + 1} or more premises, the "
        f"{n_forced} whose forced prefix restates every premise are kept and the other {n_drop} "
        "are excluded: there the model has to write a premise line itself, and both models fail "
        f"almost all of them at their first actions (Section 1.1, {_ref('do_ceiling_table')}). k = "
        f"{_run_k(store, 'prefix_state', PAIR[1]) or '?'} draws per item. Point estimates. Not "
        "filtered for training overlap. Judge: Lean (lean_v2)."
    )
    return ChartData(
        key,
        title,
        "line",
        takeaway,
        caption,
        series=series,
        opts={"x_label": "lines left", "y_fmt": "pct"},
        extra={"n_dropped": n_drop},
    )


# ========================================================================== 9. held-out


def _c_heldout(store: Store, reg: Mapping[str, mr.ModelEntry]) -> ChartData:
    key, title = "do_heldout", "Greedy accuracy on Dan's held-out set"
    models = [m for fam in FAMS for m in mr.labels_in_family(reg, fam)]
    data = heldout_view.view(store, cast("dict[str, JSON]", {"models": models, "judge_id": JUDGE}))
    if data.get("empty"):
        return _missing(key, title, str(data.get("reason", "no dan_heldout runs yet")))
    per = cast("dict[str, dict[str, Any]]", data["models"])
    bars = [P.Bar(m, float(i["accuracy"]), i["ci"][0], i["ci"][1]) for m, i in per.items()]
    published: dict[str, float] = {}
    for m in per:
        e = reg.get(m)
        if e is None or e.seed is None:
            continue
        role = dv.role_of(m)
        vals = dv.DAN_PUBLISHED["heldout"].get((e.family, role), ())
        if e.seed < len(vals):
            published[m] = float(vals[e.seed])
    parts = []
    for m in F._role_order(list(per)):  # noqa: SLF001
        i = per[m]
        parts.append(
            f"{F.display(m)} {i['n_solved']:,}/{i['n_items']:,} ({_pct(i['accuracy'])}"
            + (f"; Dan {_pct(published[m])}" if m in published else "")
            + ")"
        )
    takeaway = "greedy held-out accuracy per model."
    if PAIR[0] in per and PAIR[1] in per:
        a, b = per[PAIR[0]], per[PAIR[1]]
        gap = max((abs(per[m]["accuracy"] - published[m]) for m in published), default=None)
        takeaway = (
            f"RL cuts greedy errors from {_pct(1 - a['accuracy'])} to "
            f"{_pct(1 - b['accuracy'])} of the held-out theorems ({a['n_items'] - a['n_solved']:,}"
            f" → {b['n_items'] - b['n_solved']:,} of {b['n_items']:,})"
            + (
                f", within {100 * gap:.1f} points of Dan's reported accuracies"
                if gap is not None
                else ""
            )
            + "."
        )
    seen = _seen_note(store, "dan_heldout")
    caption = (
        f"Fraction of Dan's {next(iter(per.values()))['n_items']:,}-theorem held-out set "
        "(heldout.jsonl, his Stage-1 validation split) solved by one greedy rollout (k = 1, "
        f"T = 0), with a 95% Clopper-Pearson CI: {'; '.join(parts)}. Open triangles: Dan's "
        "reported accuracy for the same checkpoint. Not filtered for training overlap"
        + (f" ({seen[0]:,} of {seen[1]:,} in the cap-12 tiers)" if seen else "")
        + ". Judge: Lean (lean_v2)."
    )
    return ChartData(
        key,
        title,
        "bar",
        takeaway,
        caption,
        bars=bars,
        opts={"y_fmt": "pct"},
        extra={"published": published},
    )


# ===================================================================== 10. textbook_law


def _c_textbook_law(store: Store, reg: Mapping[str, mr.ModelEntry]) -> ChartData:
    key, title = "do_textbook_law", "Pass@k on textbook_law, unseen theorems only"
    models = [m for fam in FAMS for m in mr.labels_in_family(reg, fam)]
    data = dv.textbook_law_unseen(store, models, registry=reg, judge_id=JUDGE, B=B_TEXTBOOK)
    if not data:
        return _missing(key, title, "no textbook_law runs for Dan's models yet")
    series = [
        P.Series(
            m,
            [(float(k), mean) for k, mean, _l, _h in d["curve"]],
            "var(--s1)",
            band=[(float(k), lo, hi) for k, _m, lo, hi in d["curve"]],
        )
        for m, d in data.items()
    ]
    takeaway = "pass@k on unseen textbook_law theorems."
    if PAIR[0] in data and PAIR[1] in data:
        a = {k: m for k, m, _l, _h in data[PAIR[0]]["curve"]}
        b = {k: m for k, m, _l, _h in data[PAIR[1]]["curve"]}
        k_hi = max(set(a) & set(b))
        takeaway = (
            f"the RL model's pass@1 ({_pct(b[1])}) is "
            + ("above" if b[1] > a[k_hi] else "below")
            + f" the pretrained model's pass@{k_hi} ({_pct(a[k_hi])}); at k = {k_hi} RL reaches "
            f"{_pct(b[k_hi])}, so "
            "sampling more from the pretrained model does not close the gap on these theorems."
            if b[1] > a[k_hi]
            else f"the RL model is ahead at k = 1 ({_pct(b[1])} vs {_pct(a[1])}) and at k = {k_hi} "
            f"({_pct(b[k_hi])} vs {_pct(a[k_hi])})."
        )
    d0 = next(iter(data.values()))
    caption = (
        f"Pass@k on the textbook_law theorems unseen in Dan's cap-12 training tiers (n = "
        f"{d0['n']}; {d0['n_seen']} seen theorems excluded), k = 1 to {d0['k_max']} draws per "
        f"theorem: unbiased pass@k with {_BOOT} bands (B = {B_TEXTBOOK}). Judge: Lean (lean_v2)."
    )
    return ChartData(
        key, title, "line", takeaway, caption, series=series, opts={"x_label": "k", "y_fmt": "pct"}
    )


# ============================================================================== 11a. EDL


def _edl_null(classes: Sequence[str]) -> str | None:
    """The null-control class (data the models already know): the main page's, else the first
    class named `heldout*`."""
    if summary._EDL_NULL_CONTROL in classes:  # noqa: SLF001
        return summary._EDL_NULL_CONTROL  # noqa: SLF001
    return next((c for c in sorted(classes) if c.startswith("heldout")), None)


def _c_edl(store: Store) -> ChartData:
    key, title = "do_edl", "Cost of learning a new pattern (EDL)"
    data = dv.dan_edl(store, PAIR)
    per, n_lrs = (
        summary._edl_per_class(data["classes"], PAIR) if data else ({}, set())  # noqa: SLF001
    )
    if not per:
        cd = _missing(
            key,
            title,
            "not yet run: no EDL rows for the pretrained/RL pair in the store, so the cost of "
            "learning a new pattern is not measured on this page yet",
        )
        cd.extra["missing_label"] = "not yet run"
        return cd
    assert data is not None
    null = _edl_null(list(per))
    order = sorted(per, key=summary._edl_class_key)  # noqa: SLF001
    order = [c for c in order if c != null] + ([null] if null in per else [])
    rows = [
        P.DotRow(
            f"{cls} (null control)" if cls == null else cls,
            [(m, per[cls][m]["edl"], "var(--s1)", False) for m in PAIR if m in per[cls]],
        )
        for cls in order
    ]
    cmp = summary._edl_compare(per, PAIR[0], PAIR[1], null)  # noqa: SLF001
    n_c = cmp["n_c"]
    takeaway = "EDL per example, by class and model."
    if n_c:
        takeaway = (
            f"The RL model's EDL is lower on {cmp['n_lower']}/{n_c} classes ({cmp['n_robust']} "
            "with no overlap between the two data orders), but it also starts with lower test "
            f"loss on {cmp['n_l0_lower']}/{n_c} and EDL tracks the starting loss (Spearman "
            f"{cmp['rho']:.2f}); per nat of test loss that training removes, the RL model costs "
            f"as much as or more than the pretrained model on {cmp['n_ratio']}/{n_c} classes."
        )
    nul = per.get(null, {}) if null else {}
    null_bits = ", ".join(
        f"{_role_name(m)} {nul[m]['edl']:.3f} (lr {P.fmt(nul[m]['best_lr'], 'g')})"
        for m in PAIR
        if m in nul
    )
    caption = (
        "(a) Excess description length of learning each pattern class online, per example "
        f"(nats, log scale); {n_c} classes, pretrained vs RL (cap-12 seed-0 pair); EDL* per "
        "example at the learning rate with the smallest online code length, including the "
        f"ln({max(n_lrs) if n_lrs else 0}) cost of choosing among the learning rates; whiskers "
        "span the per-order values at that learning rate. (b) The same EDL* against each "
        "model's initial test loss on the class (mean nats per test example before any "
        "update), null control excluded. Null control "
        + (f"({null}, data the models already know): {null_bits or 'n/a'}" if null else ": none")
        + f". Runs: {', '.join(data['run_ids'])}."
    )
    cd = ChartData(
        key,
        title,
        "dot",
        takeaway,
        caption,
        rows=rows,
        opts={"x_label": "EDL* per example (nats)", "log_x": True},
        extra={"per": per, "null_control": null, "rho": cmp["rho"], "run_ids": data["run_ids"]},
    )
    cd.extra_tables.append(
        (
            "per-order range and initial test loss (whiskers, panel b)",
            ["class", "model", "EDL*/example", "order min", "order max", "initial test loss"],
            [
                [cls, m, *(P.fmt(per[cls][m][f], "f3") for f in ("edl", "lo", "hi", "l0"))]
                for cls in order
                for m in PAIR
                if m in per[cls]
            ],
        )
    )
    return cd


# ====================================================================== 11. likelihood

_ORIGIN_LABEL = {
    "reserve": "neutral reserve proofs",
    "family": "family witnesses",
    "reference": "neutral minimal proofs",
}


def _c_likelihood(store: Store) -> ChartData:
    key, title = (
        "do_likelihood",
        "Log-likelihood of fixed proofs under the pretrained and RL models",
    )
    data_all = dv.dan_likelihood(store)
    if data_all is None:
        return _missing(key, title, "no interp likelihood run scoring pretrained/RL yet")
    # family witnesses beyond the premise ceiling are left out of the headline (Section 1.1)
    fam_probs = problems_by_pid(store, "families", "v1")
    fam_out = [
        pid
        for pid in cast("dict[str, list[str]]", data_all.get("pids") or {}).get("family", [])
        if pid in fam_probs and not dv.in_premise_range(fam_probs[pid])
    ]
    data = (dv.dan_likelihood(store, drop_pids=fam_out) if fam_out else None) or data_all
    all_origins = cast("dict[str, dict[str, Any]]", data_all["origins"])
    origins = cast("dict[str, dict[str, Any]]", data["origins"])
    order = [o for o in ("reserve", "reference", "family") if o in origins] + sorted(
        set(origins) - {"reserve", "reference", "family"}
    )
    cfg = cast("dict[str, JSON]", data["cfg"])
    base = cast(str, cfg.get("base_model") or PAIR[0])
    rl = cast(str, cfg.get("rl_model") or PAIR[1])
    rows = [
        P.DotRow(
            f"{_ORIGIN_LABEL.get(o, o)} (n={origins[o]['n']})",
            [
                (base, origins[o]["median_fz"], "var(--s1)", False),
                (rl, origins[o]["median_t1"], "var(--s2)", False),
            ],
        )
        for o in order
    ]
    bits = [
        f"{origins[o]['median_diff']:+.2f} nats/line on {_ORIGIN_LABEL.get(o, o)} (RL higher on "
        f"{100 * origins[o]['frac_t1_higher']:.0f}%)"
        for o in order
    ]
    # sets whose plotted per-model medians order opposite to the paired change
    flipped = [
        o
        for o in order
        if (origins[o]["median_t1"] - origins[o]["median_fz"]) * origins[o]["median_diff"] < 0
    ]
    flip_note = (
        (
            " In both sets"
            if len(flipped) == len(order) > 1
            else " On " + ", ".join(_ORIGIN_LABEL.get(o, o) for o in flipped)
        )
        + ", the per-model medians plotted order the other way ("
        + "; ".join(
            f"{_ORIGIN_LABEL.get(o, o)} pretrained {origins[o]['median_fz']:.2f}, RL "
            f"{origins[o]['median_t1']:.2f}"
            for o in flipped
        )
        + "): the median of paired changes is not the change in medians."
        if flipped
        else ""
    )
    small = max(abs(origins[o]["median_diff"]) for o in order) < 0.5
    takeaway = (
        ("RL barely moves" if small else "RL changes")
        + " the likelihood of fixed proofs: the paired median change RL − pretrained is "
        + "; ".join(bits)
        + "."
        + flip_note
    )
    n_tot = sum(origins[o]["n"] for o in order)
    multi = sum(origins[o]["n_multi_order"] for o in order)
    prem_note = ""
    if fam_out and "family" in all_origins:
        fa = all_origins["family"]
        fams = sorted({str(fam_probs[pid].tags.get("family")) for pid in fam_out})
        prem_note = (
            f" Family witnesses: the {origins.get('family', {}).get('n', 0)} with at most {MAXP} "
            f"premises; the {len(fam_out)} with {MAXP + 1} or more ({', '.join(fams)}) are "
            f"excluded (with them, {fa['n']} witnesses: paired median change "
            f"{fa['median_diff']:+.2f} nats/line, RL higher on {100 * fa['frac_t1_higher']:.0f}%)."
            " The neutral reserve proofs are not filtered: their premise counts are not in the "
            "store."
        )
    names = (
        "The environment-assigned defining-name tokens are excluded from the score, as in Dan's "
        "trajectory study."
        if data["exclude_defining_names"]
        else "The scored tokens include the environment-assigned defining-name tokens, which the "
        "environment overrides at sampling time and Dan's trajectory study excludes."
    )
    caption = (
        "Median log-likelihood per written line (nats) of fixed proof sets under the pretrained "
        "and RL models (the "
        f"cap-12 seed-0 pair): {n_tot} proofs ("
        + "; ".join(
            f"{origins[o]['n']} {_ORIGIN_LABEL.get(o, o)}, L = "
            + _span([origins[o]["min_lines"], origins[o]["max_lines"]])
            for o in order
        )
        + f"), unseen in Dan's training tiers; run {data['run_id']}"
        + ("; small n, so the medians are noisy" if n_tot < 100 else "")
        + ". Each proof's log p is marginalised over its valid line orders and the environment's "
        f"name bases ({multi} proofs have more than one order) and divided by its written lines. "
        + names
        + prem_note
        + " Medians, no CI."
    )
    return ChartData(
        key,
        title,
        "dot",
        takeaway,
        caption,
        rows=rows,
        opts={"x_label": "median log p per line"},
        extra={
            "exclude_defining_names": data["exclude_defining_names"],
            "n": n_tot,
            "run_id": data["run_id"],
        },
    )


# ============================================================================= 12. KL

#: Elimination steps (the rules that consume a line), for the KL takeaway.
_ELIM_KINDS: tuple[str, ...] = ("NEGE", "IMPE", "ANDE1", "ANDE2", "BOTE")
_KIND_GLOSS = (
    "AS opens a box (assumption), IMPI/NEGI/ORE close one, PR/R restate a premise/line, FINAL is "
    "the closing exact, ANDI/ORI1/ORI2 introductions, ANDE1/ANDE2/IMPE/NEGE/BOTE eliminations, "
    "DN double negation"
)


def _c_kl(store: Store) -> ChartData:
    key, title = "do_kl", "Token-level KL between RL and pretrained, by action kind"
    data = dv.dan_kl(store)
    if data is None:
        cd = _missing(
            key,
            title,
            "not yet run: no interp kl run in the store, so the token-"
            "level KL between RL and pretrained is not measured on this page yet",
        )
        cd.extra["missing_label"] = "not yet run"
        return cd
    cfg = cast("dict[str, JSON]", data["cfg"])
    base = cast(str, cfg.get("base_model") or PAIR[0])
    rl = cast(str, cfg.get("rl_model") or PAIR[1])
    dirs = cast("dict[str, dict[str, Any]]", data["directions"])
    short = {"rl||base": "KL(RL ‖ pretrained)", "base||rl": "KL(pretrained ‖ RL)"}
    name = {
        "rl||base": f"KL(RL ‖ pretrained) on RL samples|{rl}",
        "base||rl": f"KL(pretrained ‖ RL) on pretrained samples|{base}",
    }
    order = [d for d in ("base||rl", "rl||base") if d in dirs] + sorted(
        set(dirs) - {"base||rl", "rl||base"}
    )
    kinds = sorted(
        {k for d in dirs.values() for k in d["by_kind"]},
        key=lambda k: -max(d["by_kind"].get(k, (0.0, 0))[0] for d in dirs.values()),
    )
    rows = [
        P.DotRow(
            "all tokens",
            [(name.get(dn, dn), dirs[dn]["per_token"][0], "var(--s1)", False) for dn in order],
        )
    ]
    for k in kinds:
        rows.append(
            P.DotRow(
                k,
                [
                    (name.get(dn, dn), dirs[dn]["by_kind"][k][0], "var(--s1)", False)
                    for dn in order
                    if k in dirs[dn]["by_kind"]
                ],
            )
        )
    overall = "; ".join(
        f"{short.get(dn, dn)} {dirs[dn]['per_token'][0]:.3f} nats/token "
        f"({dirs[dn]['per_sample']:.1f} per sample)"
        for dn in order
    )
    elim = [k for k in _ELIM_KINDS if all(k in dirs[dn]["by_kind"] for dn in order)]
    low = [k for k in kinds if all(dirs[dn]["by_kind"].get(k, (1.0, 0))[0] < 0.02 for dn in order)]
    elim_txt = ""
    if elim:
        ranges = []
        for dn in order:
            vals = [dirs[dn]["by_kind"][k][0] for k in elim]
            ranges.append(f"{min(vals):.2f}-{max(vals):.2f} ({short.get(dn, dn)})")
        elim_txt = (
            f"; the eliminations ({', '.join(elim)}) carry "
            + " and ".join(ranges)
            + " nats/token"
            + (f", while {', '.join(low)} stay under 0.02 in both directions" if low else "")
        )
    tops = [
        max(dirs[dn]["by_kind"].items(), key=lambda kv: kv[1][0])[0]
        for dn in order
        if dirs[dn]["by_kind"]
    ]
    lead = (
        "The two models differ most on elimination steps: "
        if elim and tops and all(t in _ELIM_KINDS for t in tops)
        else "Per-token KL: "
    )
    takeaway = lead + overall + elim_txt + "."
    excl = cfg.get("exclude_defining_names")
    suites = ", ".join(
        str(cast("dict[str, JSON]", s_).get("suite"))
        for s_ in cast("list[JSON]", cfg.get("suites") or [])
        if isinstance(s_, dict)
    ) or ", ".join(sorted({s for d in dirs.values() for s in d["suites"]}))
    n_rows, n_skip = cast(int, data["n_rows"]), cast("int | None", data["n_skipped"])
    probs_by_suite: dict[str, dict[str, Problem]] = {}

    def beyond(row: Mapping[str, JSON]) -> bool:
        suite = str(row.get("suite"))
        if suite not in probs_by_suite:
            version = default_version(store, suite=suite, models=list(PAIR), judge_id=JUDGE)
            probs_by_suite[suite] = problems_by_pid(store, suite, version or "v1")
        p = probs_by_suite[suite].get(str(row.get("pid")))
        return p is not None and not dv.in_premise_range(p)

    within = dv.dan_kl(store, drop=beyond)
    kl_prem = ""
    if within is not None and within["n_dropped"]:
        wd = cast("dict[str, dict[str, Any]]", within["directions"])
        kl_prem = (
            f" {within['n_dropped']} of the {n_rows:,} scored samples have {MAXP + 1} or more "
            "premises (most attempts on such theorems are rejected as unparseable and so not "
            "scored); without them the per-token KL is "
            + "; ".join(
                f"{short.get(dn, dn)} {wd[dn]['per_token'][0]:.3f}" for dn in order if dn in wd
            )
            + "."
        )
    skipped = (
        f" {n_skip:,} of {n_rows + n_skip:,} sampled attempts "
        f"({100 * n_skip / (n_rows + n_skip):.0f}%) were skipped: they are attempts the "
        "environment rejected as unparseable, which store no action tokens to score, so this KL "
        "covers in-grammar attempts only."
        if n_skip
        else ""
    )
    caption = (
        "Exact per-token KL divergence between the cap-12 seed-0 pair's next-action "
        "distributions, teacher-forced along each model's own stored samples (KL(RL ‖ pretrained) "
        "along "
        "the RL model's, KL(pretrained ‖ RL) along the pretrained model's) from its "
        f"{suites} runs, at most {cfg.get('max_samples_per_suite', '?')} samples per (model, "
        f'suite) (run {data["run_id"]}). "all tokens": the sum of KL over scored tokens '
        f"divided by their count, with a 95% bootstrap CI over samples (B = {data['B']}); then "
        f"the same per action kind, largest first, point estimates ({_KIND_GLOSS}). "
        + "; ".join(
            f"{short.get(dn, dn)}: {dirs[dn]['n']:,} samples ({dirs[dn]['n_ok']:,} correct), "
            f"{dirs[dn]['n_tokens']:,} tokens"
            for dn in order
        )
        + ". Environment-assigned defining-name tokens are "
        + ("excluded" if excl else "included" if excl is not None else "of unrecorded status")
        + "."
        + skipped
        + kl_prem
    )
    cd = ChartData(
        key,
        title,
        "dot",
        takeaway,
        caption,
        rows=rows,
        opts={"x_label": "nats per token", "x_fmt": "f3"},
        extra={"n_skipped": n_skip, "n_rows": n_rows},
    )
    cd.extra_tables.append(
        (
            "tokens per kind",
            ["kind", *(short.get(dn, dn) + " tokens" for dn in order)],
            [[k, *(f"{dirs[dn]['by_kind'].get(k, (0, 0))[1]:,}" for dn in order)] for k in kinds],
        )
    )
    return cd


# ========================================================================= 13. steering


def _short_suite(suite: str) -> str:
    return {"interp_eval_dev": "dev", "interp_eval_sealed": "sealed"}.get(suite, suite)


def _c_steering(store: Store) -> ChartData:
    key, title = (
        "do_steering",
        "Steering the pretrained model along the RL − pretrained activation difference",
    )
    data = dv.dan_steering(store, base=PAIR[0], rl=PAIR[1], keep=dv.in_premise_range)
    rows_in = cast("list[dict[str, Any]]", data["rows"])
    if not rows_in:
        return _missing(key, title, "no steering runs for Dan's models yet")
    beyond = dv.dan_steering(
        store, base=PAIR[0], rl=PAIR[1], keep=lambda p: not dv.in_premise_range(p), B=1
    )
    out_max = max(
        (v[0] for r in beyond["rows"] for v in r["per_suite"].values() if v is not None),
        default=None,
    )
    n_drop = cast("dict[str, int]", data["n_dropped"])
    # fz rows first (unsteered, then each setting), then t1 unsteered next to t1 - delta
    rows_in = sorted(
        rows_in,
        key=lambda r: (r["model"] != PAIR[0], r["steered"], r["layer"] or 0, r["alpha"] or 0.0),
    )
    suites = cast("list[str]", data["suites"])

    def cond(r: dict[str, Any]) -> str:
        role = "pretrained" if r["model"] == PAIR[0] else "RL"
        if not r["steered"]:
            return f"{role} (unsteered)"
        sign = "+" if r["alpha"] >= 0 else "−"
        a = abs(r["alpha"])
        return f"{role} {sign} δ (layer {r['layer']}" + (
            f", α = {a:g})" if a != 1 or role == "pretrained" else ")"
        )

    fz_rows = [r for r in rows_in if r["model"] == PAIR[0]]
    best_col = {
        s: max((r["per_suite"][s][0] for r in fz_rows if r["per_suite"].get(s)), default=None)
        for s in suites
    }
    headers = ["Condition", *(f"{_short_suite(s)} (%)" for s in suites)]
    rows: list[list[str]] = []
    bars: list[P.Bar] = []
    raw: list[list[str]] = []
    for r in rows_in:
        cells = [html.escape(cond(r))]
        for s in suites:
            v = r["per_suite"].get(s)
            if v is None:
                cells.append('<span class="na">not run</span>')
                continue
            mean, lo, hi, n = v
            num = f"{100 * mean:.1f}"
            if r["model"] == PAIR[0] and mean == best_col[s]:
                num = f"<b>{num}</b>"
            cells.append(
                f'{num}<span class="pm"><sup>+{100 * (hi - mean):.1f}</sup>'
                f"<sub>&minus;{100 * (mean - lo):.1f}</sub></span>"
            )
            raw.append(
                [
                    cond(r),
                    _short_suite(s),
                    str(n),
                    f"{100 * mean:.2f}",
                    f"{100 * lo:.2f}",
                    f"{100 * hi:.2f}",
                ]
            )
        rows.append(cells)
    # Overview bars: unsteered fz, the best fz + delta (by the mean over suites), t1 - delta.
    steered = [r for r in fz_rows if r["steered"] and all(r["per_suite"].get(s) for s in suites)]
    best = max(steered, key=lambda r: sum(r["per_suite"][s][0] for s in suites), default=None)
    worst = min(steered, key=lambda r: sum(r["per_suite"][s][0] for s in suites), default=None)
    unsteered = next((r for r in fz_rows if not r["steered"]), None)
    control = next((r for r in rows_in if r["model"] == PAIR[1] and r["steered"]), None)
    t1_plain = next((r for r in rows_in if r["model"] == PAIR[1] and not r["steered"]), None)
    has_t1 = t1_plain is not None and all(t1_plain["per_suite"].get(s) for s in suites)
    for r, lab in (
        (unsteered, "pretrained"),
        (best, "pretrained + δ"),
        (control, "RL − δ"),
        (t1_plain if has_t1 else None, "RL"),
    ):
        if r is None:
            continue
        for s in suites:
            v = r["per_suite"].get(s)
            if v is not None:
                bars.append(P.Bar(f"{_short_suite(s)}: {lab}|{r['model']}", v[0], v[1], v[2]))
    takeaway = "pass@1 under steering."
    if unsteered is not None and best is not None:
        inside = all(
            best["per_suite"][s][0] <= unsteered["per_suite"][s][2]
            for s in suites
            if unsteered["per_suite"].get(s)
        )
        vals = lambda r: " / ".join(f"{100 * r['per_suite'][s][0]:.1f}%" for s in suites)  # noqa: E731
        takeaway = (
            "Adding the RL − pretrained direction to the pretrained model does not make it the RL "
            "model: the best of "
            f"{len(steered)} settings (layer {best['layer']}, α = {best['alpha']:g}) scores "
            f"{vals(best)} "
            f"(dev / sealed) against {vals(unsteered)} unsteered"
            + (", inside the pretrained model's 95% CI" if inside else "")
            + (
                f", and the most harmful (layer {worst['layer']}, α = {worst['alpha']:g}) lowers "
                f"it to {vals(worst)}"
                if worst is not None and worst is not best
                else ""
            )
            + _t1_control_text(control, t1_plain if has_t1 else None, suites, vals)
            + "."
        )
    ns = {
        s: next((r["per_suite"][s][3] for r in rows_in if r["per_suite"].get(s)), 0) for s in suites
    }
    caption = (
        "Pass@1 of the pretrained model with the activation difference δ_l (mean RL minus "
        "pretrained residual activation "
        "at layer l over the action tokens of 300 unseen reference proofs, L = 6-10) added at "
        "every generated action token with weight α, and of the RL model with δ subtracted; layers "
        "0 and "
        f"1 of 6. interp_eval_dev = generated theorems L = 8-11, interp_eval_sealed = L = 12-14, "
        f"each on its theorems with at most {MAXP} premises (dev n = "
        f"{ns.get('interp_eval_dev', '?')} of "
        f"{ns.get('interp_eval_dev', 0) + n_drop.get('interp_eval_dev', 0)}, "
        f"sealed n = {ns.get('interp_eval_sealed', '?')} of "
        f"{ns.get('interp_eval_sealed', 0) + n_drop.get('interp_eval_sealed', 0)}); every row "
        "uses the same theorems"
        + (
            (
                f", and every condition, steered or not, scores {_pct(out_max)} on the excluded "
                f"ones ({MAXP + 1} or more premises)"
                if out_max == 0
                else f", and no condition, steered or not, scores above {_pct(out_max)} on the "
                f"excluded ones ({MAXP + 1} or more premises)"
            )
            if out_max is not None
            else ""
        )
        + f"; k = {data['k']} draws per theorem at T = 0.8; "
        f"superscript/subscript: 95% item-level bootstrap CI (B = {data['B']}); bold: best "
        "pretrained "
        "condition per column. "
        + (
            "The RL model (unsteered) is sampled on the same suites with the same k, T and seed. "
            if has_t1
            else "The RL model has no unsteered run on these two suites. "
        )
        + "Judge: Lean (lean_v2)."
    )
    cd = ChartData(
        key,
        title,
        "table",
        takeaway,
        caption,
        bars=bars,
        extra={"table": {"headers": headers, "rows": rows, "align": "l" + "r" * len(suites)}},
        opts={"has_t1_baseline": has_t1},
    )
    cd.extra_tables.append(
        ("values", ["condition", "suite", "n", "pass@1 (%)", "CI lo", "CI hi"], raw)
    )
    return cd


def _t1_control_text(
    control: dict[str, Any] | None,
    t1_plain: dict[str, Any] | None,
    suites: Sequence[str],
    vals: Callable[[dict[str, Any]], str],
) -> str:
    """The t1 - delta clause of the steering takeaway, against unsteered t1 when it was run."""
    if control is None:
        return ""
    if t1_plain is None:
        return (
            f"; subtracting it from the RL model leaves {vals(control)} (no unsteered RL run on "
            "these "
            "suites to compare)"
        )
    shifts, verdicts = [], []
    for s in suites:
        c, t = control["per_suite"].get(s), t1_plain["per_suite"].get(s)
        if c is None or t is None:
            continue
        shifts.append(f"{100 * (c[0] - t[0]):+.1f}".replace("-", "−"))
        where = "inside" if t[1] <= c[0] <= t[2] else "below" if c[0] < t[1] else "above"
        verdicts.append((_short_suite(s), where))
    if all(w == "inside" for _s, w in verdicts):
        ci = "inside the RL model's 95% CI on every suite"
    else:
        ci = ", ".join(f"{name} {w} the RL model's 95% CI" for name, w in verdicts)
    return (
        f"; subtracting it from the RL model gives {vals(control)} against {vals(t1_plain)} "
        "unsteered "
        f"({' / '.join(shifts)} points; {ci})"
    )


# ===================================================================== 14. interpolation


def _c_weights(store: Store) -> ChartData:
    key, title = "do_weights", "Interpolating weights from the pretrained to the RL model"
    data = dv.dan_interpolation(
        store, base=PAIR[0], rl=PAIR[1], family="dan12", keep=dv.in_premise_range
    )
    beyond = dv.dan_interpolation(
        store,
        base=PAIR[0],
        rl=PAIR[1],
        family="dan12",
        keep=lambda p: not dv.in_premise_range(p),
        B=1,
    )
    out_max = max(
        (
            m
            for c in cast(
                "dict[float, list[tuple[int, float, float, float]]]", beyond["curves"]
            ).values()
            for _k, m, _l, _h in c
        ),
        default=None,
    )
    curves = cast("dict[float, list[tuple[int, float, float, float]]]", data["curves"])
    pk: dict[float, tuple[tuple[float, float, float], tuple[float, float, float]]] = {}
    for a, c in curves.items():
        by_k = {k: (m, lo, hi) for k, m, lo, hi in c}
        if 1 in by_k and 128 in by_k:
            pk[a] = (by_k[1], by_k[128])
    if not pk:
        return _missing(key, title, "no pretrained/RL/interp runs on the k = 128 subset yet")
    alphas = sorted(pk)
    series = [
        P.Series(
            "pass@1",
            [(a, pk[a][0][0]) for a in alphas],
            "var(--ink)",
            band=[(a, pk[a][0][1], pk[a][0][2]) for a in alphas],
        ),
        P.Series(
            "pass@128",
            [(a, pk[a][1][0]) for a in alphas],
            "var(--ink2)",
            muted=True,
            band=[(a, pk[a][1][1], pk[a][1][2]) for a in alphas],
        ),
    ]
    p1 = {a: pk[a][0][0] for a in alphas}
    p128 = {a: pk[a][1][0] for a in alphas}
    a0, a1 = alphas[0], min(alphas, key=lambda a: abs(a - 1.0))
    mids = [a for a in alphas if a0 < a < a1]
    half = next((a for a in alphas if p1[a] - p1[a0] >= 0.5 * (p1[a1] - p1[a0])), a1)
    sat = next((a for a in alphas if p128[a] >= max(p128.values()) - 1e-9), a1)
    beyond = [a for a in alphas if a > a1]
    takeaway = (
        f"pass@1 rises from {_pct(p1[a0])} (α = {a0:g}, pretrained) to {_pct(p1[a1])} (α = {a1:g}, "
        "RL), "
        f"half of the gain already by α = {half:g}"
        + (f" ({_pct(p1[half])})" if half in mids else "")
        + f", while pass@128 rises from {_pct(p128[a0])} and saturates at {_pct(p128[sat])} from "
        f"α = {sat:g}"
        + (
            f"; extrapolating to α = {beyond[-1]:g} lowers pass@1 to {_pct(p1[beyond[-1]])} with "
            f"pass@128 at {_pct(p128[beyond[-1]])}"
            if beyond
            else ""
        )
        + "."
    )
    takeaway = takeaway[0].upper() + takeaway[1:]
    n = next(iter(cast("dict[float, int]", data["n"]).values()))
    seen = _seen_note(store, "interp_eval_k128")
    n_drop = int(data.get("n_dropped") or 0)
    caption = (
        "Pass@1 and pass@128 of θ_fz + α(θ_t1 − θ_fz) against α (cap-12 seed-0 pair; α > 1 "
        f"extrapolates past the RL model) on the n = {n} theorems with at most {MAXP} premises "
        f"of a fixed subset of {n + n_drop} generated theorems (L = 8-14), 128 draws each at "
        f"T = 0.8; shaded: {_BOOT} bands (B = {data['B']}). The {n_drop} theorems with "
        f"{MAXP + 1} or more premises are excluded"
        + (
            (
                ": at every α none of their 128 draws succeeds"
                if out_max == 0
                else f": at every α their pass@128 is at most {_pct(out_max)}"
            )
            if out_max is not None
            else ""
        )
        + ". Not filtered for training overlap"
        + (f" ({seen[0]:,} of {seen[1]:,} in the cap-12 tiers)" if seen else "")
        + ". Judge: Lean (lean_v2)."
    )
    return ChartData(
        key,
        title,
        "line",
        takeaway,
        caption,
        series=series,
        opts={"x_label": "alpha (0 = pretrained, 1 = RL)", "y_fmt": "pct"},
        extra={"n": n, "n_dropped": n_drop},
    )


# ============================================================================ 15. probes

_BASELINE_LABEL = {
    "step_index_premises": "step + premises",
    "step_index_premises_depth": "step + premises + depth",
}


def _c_probes(store: Store) -> ChartData:
    key, title = "do_probes", "What linear probes read from the pretrained and RL models"
    data = dv.dan_probes(store)
    if data is None:
        return _missing(key, title, "no interp probes run for pretrained/RL yet")
    cfg = cast("dict[str, JSON]", data["cfg"])
    label = {
        "fz": cast(str, cfg.get("base_model") or PAIR[0]),
        "t1": cast(str, cfg.get("rl_model") or PAIR[1]),
    }
    per = cast(
        "dict[str, list[tuple[int, float | None, float | None, int | None]]]", data["models"]
    )
    series: list[P.Series] = []
    for metric, idx in (("next_rule", 1), ("actions_to_go", 2)):
        for role in ("fz", "t1"):
            pts = [
                (float(row[0]), cast(float, row[idx]))
                for row in per.get(role, [])
                if row[idx] is not None
            ]
            if pts:
                series.append(P.Series(f"{metric}: {label[role]}", pts, "var(--s1)"))
    bl = cast("dict[str, float]", data["baselines"])
    baselines: dict[str, list[tuple[str, float]]] = {}
    for metric, prefix in (("next_rule", "next_rule_acc_"), ("actions_to_go", "actions_to_go_r2_")):
        for name in ("step_index_premises", "step_index_premises_depth"):
            v = bl.get(prefix + name)
            if v is not None:
                baselines.setdefault(metric, []).append((_BASELINE_LABEL[name], float(v)))
    by = {s.name: dict(s.pts) for s in series}
    rule = {r: by.get(f"next_rule: {label[r]}", {}) for r in ("fz", "t1")}
    atg = {r: by.get(f"actions_to_go: {label[r]}", {}) for r in ("fz", "t1")}
    best_rule = max((v for _n, v in baselines.get("next_rule", [])), default=None)
    best_atg = max((v for _n, v in baselines.get("actions_to_go", [])), default=None)
    takeaway = "next_rule accuracy and actions_to_go R2 by layer."
    common = sorted(set(rule["fz"]) & set(rule["t1"]))
    if common and best_rule is not None:
        last = common[-1]
        first_above = next(
            (x for x in common if min(rule["fz"][x], rule["t1"][x]) > best_rule), None
        )
        gap = max(abs(rule["t1"][x] - rule["fz"][x]) for x in common)
        takeaway = (
            "The next reference rule is equally decodable from both models: their next_rule "
            "probes "
            + (
                f"beat the best position-only lookup ({_pct(best_rule)}) from layer "
                f"{first_above:g} on"
                if first_above is not None
                else f"stay at or below the best position-only lookup ({_pct(best_rule)})"
            )
            + f", reaching {_pct(rule['fz'][last])} (pretrained) and {_pct(rule['t1'][last])} (RL) "
            "at "
            f"layer {last:g}, within {100 * gap:.1f} points of each other at every layer"
        )
        if atg["fz"] and atg["t1"] and best_atg is not None:
            la = max(set(atg["fz"]) & set(atg["t1"]))
            beats = max(atg["fz"][la], atg["t1"][la]) > best_atg
            takeaway += (
                f"; actions_to_go R² at layer {la:g} (pretrained {atg['fz'][la]:.2f}, RL "
                f"{atg['t1'][la]:.2f}) {'beats' if beats else 'does not beat'} the lookup "
                f"({best_atg:.2f})"
            )
        takeaway += "."
    n_test = next((row[3] for row in per.get("fz", []) if row[3]), None)
    caption = (
        "Held-out accuracy of a linear probe for next_rule (the ND rule of the reference "
        "proof's next line-creating action, (a)) and R² of a linear probe for actions_to_go "
        "(actions left in the reference proof, (b)) on the residual stream at the <act> token "
        "before each action, the models teacher-forced along the reference proofs, layers 0-5, "
        "so the next_rule probe largely reads the model's own prediction of its next step, "
        f"for the pretrained and RL models (cap-12 seed-0 pair; {cfg.get('n_train', '?')} "
        "reference proofs with L = "
        f"{cfg.get('l_min', '?')}-{cfg.get('l_max', '?')} unseen in training, "
        f"{data['n_train_examples']} steps, split by theorem; {n_test} test steps). Grey lines: "
        "lookup tables fit on the same training split that predict from the step index and "
        "premise count (dotted) and also the frame depth (dashed), all visible from the state "
        "alone"
        + (
            f"; majority class {_pct(bl['next_rule_acc_majority'])}"
            if bl.get("next_rule_acc_majority") is not None
            else ""
        )
        + ". Point estimates on one split."
    )
    cd = ChartData(
        key,
        title,
        "line",
        takeaway,
        caption,
        series=series,
        opts={"x_label": "layer", "y_fmt": "f3"},
        extra={"baselines": baselines},
    )
    cd.extra_tables.append(
        (
            "position-only baselines",
            ["target", "lookup", "value"],
            [[m, n, f"{v:.3f}"] for m, vs in baselines.items() for n, v in vs],
        )
    )
    return cd


# ============================================================================ assembly


def dan_only_chart_data(
    store: Store, *, registry: Mapping[str, mr.ModelEntry] | None = None
) -> list[ChartData]:
    """Every chart of the page, in page order (shared by the page and `figures --dan-only`)."""
    reg = registry if registry is not None else mr.load_registry()
    tb = _c_textbook(store, reg)
    charts = [
        _c_ceiling(store),
        _c_ceiling_table(store),
        tb,
        _c_table1(store, reg, tb),
        _c_failures(store),
        _c_families(store, reg),
        _c_laws(store, reg),
        _c_glue(store, reg),
        _c_skills(store, reg),
        _c_padding(store, reg),
        _c_prefix(store, reg),
        _c_heldout(store, reg),
        _c_textbook_law(store, reg),
        _c_edl(store),
        _c_likelihood(store),
        _c_kl(store),
        _c_steering(store),
        _c_weights(store),
        _c_probes(store),
    ]
    resolve_refs(charts, numbering(charts))
    return charts


#: `(heading, intro, ((subheading | None, keys), ...))` per section, in page order; a
#: subsection's own paragraph comes from `subsection_intro`.
SECTIONS: tuple[tuple[str, str, tuple[tuple[str | None, tuple[str, ...]], ...]], ...] = (
    (
        "Eliciting proofs",
        "Pass rates of the pretrained and RL checkpoints. First the premise ceiling that "
        "decides every generated benchmark below; then textbook theorems across sampling "
        "budgets, seeds and caps, how attempts on them end, unseen textbook law instances and "
        "Dan's held-out set; then generated benchmarks that vary one thing at a time: length "
        "within a family, substitution size in known laws, composing solved parts, finishing a "
        "given prefix, the skill a proof needs, and padding.",
        (
            ("The 4-premise ceiling", ("do_ceiling", "do_ceiling_table")),
            (
                "Benchmarks",
                (
                    "do_textbook",
                    "do_table1",
                    "do_failures",
                    "do_textbook_law",
                    "do_heldout",
                    "do_families",
                    "do_laws",
                    "do_glue",
                    "do_prefix",
                    "do_skills",
                    "do_padding",
                ),
            ),
        ),
    ),
    (
        "Internals",
        "Whether the change from the pretrained to the RL model shows up in the cost of "
        "learning a new pattern (EDL), in the likelihood of fixed proofs, along a weight "
        "interpolation, in a token-level KL between their next-action distributions, under "
        "activation steering and in linear probes (cap-12 seed-0 pair).",
        (
            (
                None,
                ("do_edl", "do_likelihood", "do_weights", "do_kl", "do_steering", "do_probes"),
            ),
        ),
    ),
)


def section_keys(
    section: tuple[str, str, tuple[tuple[str | None, tuple[str, ...]], ...]],
) -> list[str]:
    return [k for _sub, keys in section[2] for k in keys]


#: Charts that are tables (numbered as tables even while their data is missing).
TABLE_KEYS: frozenset[str] = frozenset(
    {"do_ceiling_table", "do_table1", "do_failures", "do_steering"}
)


def numbering(charts: Sequence[ChartData]) -> dict[str, str]:
    """`key -> "Figure N" | "Table N"` in page order; Figure 1 is the overview."""
    out = {"overview": "Figure 1"}
    n_fig, n_tab = 1, 0
    by = {c.key: c for c in charts}
    for section in SECTIONS:
        for k in section_keys(section):
            cd = by.get(k)
            if cd is None:
                continue
            if cd.kind == "table" or (cd.kind == "missing" and k in TABLE_KEYS):
                n_tab += 1
                out[k] = f"Table {n_tab}"
            else:
                n_fig += 1
                out[k] = f"Figure {n_fig}"
    return out


def subsection_intro(name: str, charts: Sequence[ChartData], nums: Mapping[str, str]) -> str:
    """The paragraph under a subsection heading (empty for none)."""
    if name != "The 4-premise ceiling":
        return ""
    by = {c.key: c for c in charts}
    ceil = by.get("do_ceiling")
    train = (
        cast("dict[str, dict[int, int]]", ceil.extra.get("train") or {})
        if ceil is not None and ceil.kind != "missing"
        else {}
    )
    max_k = max((max(c) for c in train.values()), default=MAXP)
    sizes = "; ".join(f"{label}: {sum(c.values()):,}" for label, c in train.items())
    fig, tab = nums.get("do_ceiling", "?"), nums.get("do_ceiling_table", "?")
    return (
        f"Every theorem Dan's models were trained on has at most {max_k} premises"
        + (f" ({sizes})" if sizes else "")
        + f". On theorems with more, both models fail almost every attempt on every generated "
        f"suite ({fig}, {tab}). Most of those attempts end within the model's first two "
        "actions, which the environment rejects as a premise line out of place: it accepts a "
        "line `have nK : F := hK` only at the top level, before any other line, for the next "
        "premise in order. prefix_state shows where the failure sits: when its forced prefix "
        "has already restated every premise, the RL model solves items with 4 or more premises "
        f"about as often as items with fewer ({tab}). Hypothesis, not tested here: the premise "
        "names h4 to h8 are in the models' vocabulary (MAXH = 8 in Dan's lean_tok.py) but no "
        "training theorem uses them, so their embeddings are untrained; the forced-prefix "
        "states also list h4 and up, so an untrained name in the state does not by itself stop "
        "the model. Because the ceiling would otherwise dominate them, the generated benchmarks "
        f"below report items with at most {MAXP} premises, and each caption says how many items "
        "that drops."
    )


def caption_text(cd: ChartData, label: str | None) -> str:
    head = f"{label}: " if label else ""
    if cd.kind == "missing":
        return f"{head}{cd.title}. Not available: {cd.missing}."
    return f"{head}{cd.title}. {cd.caption} {_cap_first(cd.takeaway)}".strip()


def _role_name(label: str) -> str:
    """`"dan12-fz-s0"` -> `"pretrained"` (the reader's name for the run label's role)."""
    role = F.S.parse_model_label(label)[1] or label
    return F.ROLE_NAME.get(role, role)


def _code_spans(text: str) -> str:
    """Escape `text`, rendering its `backtick` spans as <code>."""
    parts = text.split("`")
    return "".join(f"<code>{P.esc(x)}</code>" if i % 2 else P.esc(x) for i, x in enumerate(parts))


def _cap_first(text: str) -> str:
    text = text.strip()
    return text[:1].upper() + text[1:] if text else text


def _caption_html(cd: ChartData, label: str | None) -> str:
    lab = f'<span class="fignum">{P.esc(label)}:</span> ' if label else ""
    if cd.kind == "missing":
        return f"{lab}<b>{P.esc(cd.title)}.</b> {P.esc(cd.missing)}."
    return (
        f"{lab}<b>{P.esc(cd.title)}.</b> {P.esc(cd.caption)} "
        f'<span class="takeaway">{P.esc(_cap_first(cd.takeaway))}</span>'
    )


def _table_html(cd: ChartData) -> str:
    t = cast("dict[str, Any]", cd.extra["table"])
    align = cast(str, t.get("align", ""))

    def cls(i: int) -> str:
        return ' class="l"' if i < len(align) and align[i] == "l" else ""

    th = "".join(f"<th{cls(i)}>{P.esc(h)}</th>" for i, h in enumerate(t["headers"]))
    body = "".join(
        "<tr>" + "".join(f"<td{cls(i)}>{c}</td>" for i, c in enumerate(r)) + "</tr>"
        for r in t["rows"]
    )
    return (
        f'<table class="booktabs multi"><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>'
    )


def render_card(cd: ChartData, label: str | None) -> str:
    fid = f"{'tab' if cd.kind == 'table' else 'fig'}-{cd.key}"
    if cd.kind == "missing":
        return (
            f'<figure class="card chart-card" id="{fid}"><div class="missing">'
            f"{P.esc(cd.extra.get('missing_label') or 'no data yet')}</div>"
            f"<figcaption>{_caption_html(cd, label)}</figcaption></figure>"
        )
    details = "".join(summary._details_table(s, h, r) for s, h, r in summary.data_tables(cd))  # noqa: SLF001
    if cd.kind == "table":
        return (
            f'<figure class="card chart-card table-card" id="{fid}">'
            f'<figcaption class="above">{_caption_html(cd, label)}</figcaption>'
            f'<div class="tw">{_table_html(cd)}</div>{details}</figure>'
        )
    image = F.figure_uri(cd)
    if image is None:
        return render_card(ChartData(cd.key, cd.title, missing="no figure"), label)
    pic = summary._picture(image, F.figure_uri(cd, narrow=True), f"{label}: {cd.title}")  # noqa: SLF001
    return (
        f'<figure class="card chart-card" id="{fid}"><div class="figbox">{pic}</div>'
        f"<figcaption>{_caption_html(cd, label)}</figcaption>{details}</figure>"
    )


def _half(cd: ChartData) -> bool:
    if cd.kind == "missing":  # as wide as its figure will be, so the layout does not shift
        w = F.width_of(cd.key)
        return w is None or w < F.S.FULL_W - 0.5
    if cd.kind == "table":
        return bool(cd.opts.get("half"))
    w = F.width_of(cd.key)
    return w is not None and w < F.S.FULL_W - 0.5


def _section_html(cards: Sequence[tuple[ChartData, str]]) -> str:
    out: list[str] = []
    i = 0
    while i < len(cards):
        cd, h = cards[i]
        if _half(cd) and i + 1 < len(cards) and _half(cards[i + 1][0]):
            out.append(f'<div class="pair">{h}{cards[i + 1][1]}</div>')
            i += 2
            continue
        out.append(f'<div class="{"single half" if _half(cd) else "single"}">{h}</div>')
        i += 1
    return "".join(out)


OVERVIEW_TITLE = "Overview: every benchmark's pretrained vs RL comparison"


def overview_caption(charts: Sequence[ChartData], nums: Mapping[str, str]) -> str:
    def ref(key: str, sub: str = "") -> str:
        return nums.get(key, "?") + sub

    steer = next((c for c in charts if c.key == "do_steering"), None)
    t1_bar = ", RL" if steer is not None and steer.opts.get("has_t1_baseline") else ""
    return (
        "One panel per benchmark, each the headline comparison of the figure or table named, in "
        "page order; pretrained darker, RL full hue (hatched bars), cap 12 reddish purple, cap 6 "
        "sky blue; that figure or table gives n, the CI and the takeaway. Cap-12 seed-0 pair "
        "unless "
        f"noted; (g), (i), (k), (o) and (q) are on items with at most {MAXP} premises and (j) "
        "leaves out the prefix_state items the ceiling decides. (a) Pass@1 "
        "against the number of premises, items of the generated suites except prefix_state "
        f"pooled; shaded: more premises than any training theorem ({ref('do_ceiling')}). (b) "
        f"Pass@k on textbook_dev, log k ({ref('do_textbook', 'a')}). (c) textbook72 theorems "
        "solved at k = 256 by cap: bars = mean of three seeds, dots = seeds, black ticks = Dan's "
        f"published counts ({ref('do_table1')}). (d) How textbook_dev attempts end: solved, "
        f"finished but rejected by Lean, rejected by the environment ({ref('do_failures')}). (e) "
        f"Pass@k on unseen textbook_law theorems ({ref('do_textbook_law')}). (f) Held-out greedy "
        f"accuracy with 95% CI; open triangles: Dan's reported value ({ref('do_heldout')}). (g) "
        "Written lines at the horizon n* per restricted family, unseen theorems "
        f"({ref('do_families')}). (h) Pass@1 against substitution size s ({ref('do_laws')}). (i) "
        f"Compositionality ratio R per glue operator, dotted R = 1 ({ref('do_glue')}). (j) Prefix "
        f"completion against lines left ({ref('do_prefix')}). (k) Pass@1 by required skill "
        f"({ref('do_skills', 'a')}). (l) Pass@1 of P1-padded theorems (both kinds pooled) "
        f"against their total premises; shaded as (a) ({ref('do_padding', 'a')}). (m) RL's EDL* "
        f"against pretrained's, one point per class ({ref('do_edl')}). (n) Median log p per "
        f"line of the fixed proof sets ({ref('do_likelihood')}). (o) Pass@1 (black) and pass@128 "
        "(grey, dashed) along the pretrained-to-RL weight interpolation "
        f"({ref('do_weights')}). (p) Per-token KL, pooled and for the four kinds with the "
        f"largest KL ({ref('do_kl')}). (q) Pass@1 of pretrained, the best pretrained + "
        f"δ{t1_bar} and RL − δ on dev and sealed generated theorems ({ref('do_steering')}). (r) "
        "next_rule probe accuracy by layer; dashed: best position-only lookup "
        f"({ref('do_probes', 'a')}). (s) Key."
    )


def _overview_card(charts: Sequence[ChartData], nums: Mapping[str, str]) -> str:
    image = F.overview_uri(charts)
    if image is None:
        return ""
    pic = summary._picture(  # noqa: SLF001
        image, F.overview_uri(charts, narrow=True), f"Figure 1: {OVERVIEW_TITLE}"
    )
    return (
        f'<figure class="card overview-card" id="fig-overview"><div class="figbox">{pic}</div>'
        f'<figcaption><span class="fignum">Figure 1:</span> <b>{P.esc(OVERVIEW_TITLE)}.</b> '
        f"{P.esc(overview_caption(charts, nums))}</figcaption></figure>"
    )


# ===================================================================== lede, limitations


def _judge_totals(store: Store, reg: Mapping[str, mr.ModelEntry]) -> dict[str, int] | None:
    data = judges_view.view(store, cast("dict[str, JSON]", {"models": dv.dan_models(reg)}))
    if data.get("empty"):
        return None
    per = cast("dict[str, dict[str, Any]]", data["models"])
    keys = ("n_attempts", "agree_count", "lean_only_count", "nd_only_count")
    out = {k: sum(int(i[k]) for i in per.values()) for k in keys}
    out["missing_premise"] = sum(
        int(i["lean_only_breakdown"]["missing_premise_restatement"]["count"]) for i in per.values()
    )
    return out


def _agree_pct(j: Mapping[str, int]) -> str:
    rate = j["agree_count"] / j["n_attempts"] if j["n_attempts"] else float("nan")
    return f"{100 * rate:.2f}%"


def lede(j: Mapping[str, int] | None, models_with_textbook: int, *, ceiling: bool = False) -> str:
    agree = (
        f" Every attempt is judged by Lean (lean_v2, Dan's own judge); re-judging the same "
        f"{j['n_attempts']:,} attempts with our natural-deduction checker nd_v2 agrees on "
        f"{_agree_pct(j)}."
        if j
        else " Every attempt is judged by Lean (lean_v2, Dan's own judge)."
    )
    return (
        "Dan Pandori's best-state recipe trains a 9.56M-parameter ALiBiGPT (6 layers, width 384) "
        "to prove propositional theorems through a proof-state interface: at each step the model "
        "reads the current proof state and writes one action, the environment applies it, and "
        "Lean checks the finished proof. We compare each pretrained Stage-1 checkpoint "
        '("pretrained"; fz in Dan\'s run names) '
        'with the same checkpoint after his 8-round T1 RL ladder ("RL"; t1), for Stage-1 proofs of '
        "at "
        f"most 6 and at most 12 lines, three seeds each ({models_with_textbook} checkpoints), on "
        "our benchmark panel."
        + agree
        + (
            f" Neither model can prove theorems with more than {MAXP} premises, the most any "
            "training theorem has (Section 1.1), so the generated benchmarks report items with "
            f"at most {MAXP}."
            if ceiling
            else ""
        )
    )


def models_note(nums: Mapping[str, str]) -> str:
    return (
        "Labels: cap 12 / cap 6 (dan12 / dan6) = Stage-1 training proofs of at most 12 / 6 lines; "
        "run labels: fz = pretrained (Stage-1 only); t1 = RL (after the T1 ladder); s0-s2 = "
        f"seeds. {nums.get('do_textbook', '?')} and {nums.get('do_table1', '?')} cover every "
        "checkpoint; every other result is the cap-12 seed-0 pair (dan12-fz-s0 vs dan12-t1-s0)."
    )


_SEEN_SUITES: tuple[tuple[str, str], ...] = (
    ("textbook_dev", "textbook_dev"),
    ("textbook_train14", "textbook_train14"),
    ("families", "families"),
    ("laws", "laws"),
    ("glue", "glue"),
    ("glue_parts_renamed", "glue parts"),
    ("padding", "padding"),
    ("prefix_state", "prefix_state"),
    ("skill_pool_sub", "skill_pool_sub"),
    ("dan_heldout", "held-out"),
    ("textbook_law", "textbook_law"),
    ("interp_eval_dev", "interp_eval_dev"),
    ("interp_eval_sealed", "interp_eval_sealed"),
    ("interp_eval_k128", "k = 128 subset"),
)
_FILTERED = frozenset({"families", "laws", "textbook_law"})


def _truncation(
    store: Store, reg: Mapping[str, mr.ModelEntry]
) -> tuple[float, str, dict[str, float]] | None:
    """`(worst fraction, which run, {role: pooled fraction})` of textbook attempts that hit the
    action or step cap (`ItemResult.hit_max`: attempts that did not finish)."""
    worst: tuple[float, str] | None = None
    pooled: dict[str, list[int]] = {}
    for m in dv.dan_models(reg):
        for suite in ("textbook_dev", "textbook_train14"):
            run_id = latest_run_id(store, suite=suite, model=m, version="v1", judge_id=JUDGE)
            if run_id is None:
                continue
            its = list(items_by_pid(store, run_id).values())
            n = sum(i.n for i in its)
            hit = sum(i.hit_max for i in its)
            acc = pooled.setdefault(dv.role_of(m), [0, 0])
            acc[0] += hit
            acc[1] += n
            if n and (worst is None or hit / n > worst[0]):
                worst = (hit / n, f"{m} on {suite}")
    if worst is None:
        return None
    return worst[0], worst[1], {r: h / n for r, (h, n) in pooled.items() if n}


def limitations(
    store: Store,
    reg: Mapping[str, mr.ModelEntry],
    charts: Sequence[ChartData],
    nums: Mapping[str, str],
    j: Mapping[str, int] | None,
) -> list[str]:
    by = {c.key: c for c in charts}
    out: list[str] = []
    t72 = (
        cast("dict[str, dict[str, Any]]", by["do_table1"].extra.get("t72") or {})
        if "do_table1" in by
        else {}
    )
    spread = max(
        (max(v["ours"]) - min(v["ours"]) for v in t72.values() if len(v["ours"]) > 1), default=None
    )
    out.append(
        f"Only textbook pass@k ({nums.get('do_textbook', '?')}, {nums.get('do_table1', '?')}) "
        "covers all twelve checkpoints. Every other benchmark and measure is the cap-12 seed-0 "
        "pair alone: one seed, so those figures carry no seed-to-seed uncertainty"
        + (
            f" (seeds of one cell differ by up to {spread} of the 72 textbook theorems)"
            if spread is not None
            else ""
        )
        + ", and nothing there speaks to cap 6."
    )
    ceil = by.get("do_ceiling")
    if ceil is not None and ceil.kind != "missing":
        out.append(
            f"Premise ceiling: the headline numbers of the generated benchmarks leave out the "
            f"items with more than {MAXP} premises (Section 1.1; each caption says how many), so "
            "they describe the models only within the premise range of their training data, and "
            "beyond it the page shows only that both models almost always fail. Why they fail "
            "there is not tested here."
        )
    lik = by.get("do_likelihood")
    if lik is not None and lik.kind != "missing":
        out.append(
            "The fixed proofs of the likelihood and probe figures "
            f"({nums.get('do_likelihood', '?')}, "
            f"{nums.get('do_probes', '?')}) are not filtered by premise count: the family "
            "witnesses' counts are known (the likelihood caption gives them), but the neutral "
            "reserve proofs come from a file outside the store, so how many of them have "
            f"{MAXP + 1} or more premises is not checked here."
        )
    pad_cd = by.get("do_padding")
    n_base = cast("int | None", pad_cd.extra.get("n_bases")) if pad_cd is not None else None
    out.append(
        f"Padding: the {n_base or 'unpadded'} {'unpadded ' if n_base else ''}bases were chosen as "
        "short theorems a different prover (our "
        "earlier RL model) had mastered (pass@1 >= 0.7), not for Dan's models; and P1 pads add "
        "premises that Lean does not need restated, so for his models they add hypotheses, not "
        f"lines, and from {MAXP + 1} premises in all they meet the premise ceiling "
        f"({nums.get('do_padding', '?')} plots total premises, not written length)."
    )
    seen_rows = []
    for suite, name in _SEEN_SUITES:
        version = default_version(store, suite=suite, models=list(PAIR), judge_id=JUDGE)
        if version is None:
            continue
        c = dv.seen_counts(store, "dan12", [(suite, version)])
        if c:
            seen_rows.append((suite, name, c[0][3], c[0][2]))
    if seen_rows:
        filt = [f"{n} {s:,}/{t:,}" for su, n, s, t in seen_rows if su in _FILTERED]
        rest = [f"{n} {s:,}/{t:,}" for su, n, s, t in seen_rows if su not in _FILTERED]
        out.append(
            "Training overlap is judged against Dan's cap-12 tiers (his Stage-1 set plus the T1 "
            "ladder's RL targets, matched by sequent). Families, laws and textbook_law exclude "
            "seen theorems (" + ", ".join(filt) + "); the other suites are not filtered, with "
            "seen counts " + ", ".join(rest) + "."
        )
    if j:
        out.append(
            "Judge: every number uses Lean on the literal sampled text (lean_v2, Dan's protocol "
            f"and RL reward). Re-judging the same {j['n_attempts']:,} attempts with nd_v2 agrees "
            f"on {_agree_pct(j)}: {j['lean_only_count']:,} Lean-only accepts, "
            f"{j['missing_premise']:,} of them rejected by nd_v2 with a premise error (premises "
            "not restated as natural-deduction lines, which Lean does not require), and "
            f"{j['nd_only_count']:,} nd_v2-only accepts."
        )
    trunc = _truncation(store, reg)
    if trunc is not None and trunc[0] > 0.001:
        worst, where, by_role = trunc
        roles = ", ".join(
            f"{F.ROLE_NAME.get(r, r)} {100 * f:.2f}%" for r, f in sorted(by_role.items())
        )
        out.append(
            "Attempts that hit the action or step cap (max_action 512, max_steps 96) count as "
            f"failures: up to {100 * worst:.1f}% of attempts in one textbook run ({where}); "
            f"pooled over every textbook run, {roles}. Dan reports the same pattern, and that "
            "doubling both caps raised his counts by at most 2 theorems where he tested it."
        )
    steer = by.get("do_steering")
    if steer is not None and steer.kind != "missing" and not steer.opts.get("has_t1_baseline"):
        out.append(
            "Steering has no unsteered RL run on interp_eval_dev/sealed, so RL − δ "
            f"({nums.get('do_steering', '?')}) has no paired baseline; it shows only that the RL "
            "model "
            "keeps most of its accuracy without the direction."
        )
    lik = by.get("do_likelihood")
    if lik is not None and lik.kind != "missing":
        n_lik = sum(n for r in lik.rows if (n := _row_n(r.label)) is not None)
        out.append(
            f"Likelihood ({nums.get('do_likelihood', '?')}, run {lik.extra.get('run_id')}) rests "
            f"on {n_lik} fixed proofs"
            + (" (a small pilot)" if n_lik < 100 else "")
            + ", not model samples"
            + (
                "; it excludes the environment-assigned defining-name tokens."
                if lik.extra.get("exclude_defining_names")
                else ", and its scores include the environment-assigned defining-name tokens, "
                "which the environment overrides at sampling time and Dan's trajectory study "
                "excludes."
            )
        )
    edl = by.get("do_edl")
    if edl is not None and edl.kind == "missing":
        out.append(f"EDL ({nums.get('do_edl', '?')}) has not been run for Dan's models yet.")
    kl = by.get("do_kl")
    if kl is not None and kl.kind == "missing":
        out.append(f"Token-level KL ({nums.get('do_kl', '?')}) has not been run yet.")
    elif kl is not None and kl.extra.get("n_skipped"):
        n_skip, n_rows = int(kl.extra["n_skipped"]), int(kl.extra["n_rows"])
        out.append(
            f"Token-level KL ({nums.get('do_kl', '?')}) covers in-grammar attempts only: "
            f"{n_skip:,} of {n_skip + n_rows:,} sampled attempts "
            f"({100 * n_skip / (n_skip + n_rows):.0f}%) were rejected by the environment as "
            "unparseable and have no action tokens to score."
        )
    w = by.get("do_weights")
    if w is not None and w.kind != "missing":
        n_w, d_w = w.extra.get("n"), int(w.extra.get("n_dropped") or 0)
        out.append(
            f"Weight interpolation ({nums.get('do_weights', '?')}) is evaluated on the fixed "
            f"k = 128 subset only ({n_w} of its {(n_w or 0) + d_w} theorems, those with at most "
            f"{MAXP} premises), not on the full interp_eval suites."
        )
    return out


def _row_n(label: str) -> int | None:
    """`"family witnesses (n=30)"` -> 30."""
    _head, sep, tail = label.rpartition("(n=")
    return int(tail.rstrip(")")) if sep and tail.rstrip(")").isdigit() else None


def _gpu_hours(store: Store) -> float:
    if not store.ledger_path.exists():
        return 0.0
    return (
        sum(
            float(cast(float, row.get("gpu_s") or 0.0))
            for row in read_jsonl(store.ledger_path)
            if "dan" in str(row.get("job", "")).lower()
        )
        / 3600.0
    )


_CSS_EXTRA = """
.rng{color:var(--muted);font-size:.8em;margin-left:3px;white-space:nowrap}
@media (max-width:700px){.rng{display:block;margin-left:0}}
.na{color:var(--muted);font-style:italic}
table.booktabs.multi{font-size:.9rem}
table.booktabs.multi th{text-align:right;vertical-align:bottom}
table.booktabs.multi th.l,table.booktabs.multi td.l{text-align:left}
table.booktabs.multi td{white-space:nowrap}
.takeaway{font-style:normal}
.limits li{margin:4px 0}
h3{font-size:1.02rem;margin:24px 0 4px;font-weight:700}
h3 .secnum{margin-right:.5em}
code{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:.86em}
.sub{display:inline-block;padding-left:1.1em;white-space:normal}
table.booktabs.multi td.l{white-space:normal;min-width:8em}
@media (max-width:700px){table.booktabs.multi{font-size:.8rem}
table.booktabs.multi th,table.booktabs.multi td{padding:3px 5px}}
"""


def render_page(store: Store, *, registry: Mapping[str, mr.ModelEntry] | None = None) -> str:
    reg = registry if registry is not None else mr.load_registry()
    charts = dan_only_chart_data(store, registry=reg)
    nums = numbering(charts)
    by = {c.key: c for c in charts}
    sections = []
    for i, (heading, intro, subs) in enumerate(SECTIONS, start=1):
        body = []
        named = [sub for sub, _keys in subs if sub]
        for sub, keys in subs:
            cards = [(by[k], render_card(by[k], nums.get(k))) for k in keys if k in by]
            if sub:
                j = named.index(sub) + 1
                para = subsection_intro(sub, charts, nums)
                body.append(
                    f'<h3><span class="secnum">{i}.{j}</span>{P.esc(sub)}</h3>'
                    + (f'<p class="intro">{_code_spans(para)}</p>' if para else "")
                )
            body.append(_section_html(cards))
        sections.append(
            f'<section><h2><span class="secnum">{i}</span>{P.esc(heading)}</h2>'
            f'<p class="intro">{P.esc(intro)}</p>{"".join(body)}</section>'
        )
    j = _judge_totals(store, reg)
    n_ckpt = len(dv.dan_models(reg))
    ceiling = "do_ceiling" in by and by["do_ceiling"].kind != "missing"
    limits = "".join(f"<li>{P.esc(t)}</li>" for t in limitations(store, reg, charts, nums, j))
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>Best-state Provers</title>"
        f"<style>{summary._CSS}{_CSS_EXTRA}</style></head><body>"  # noqa: SLF001
        '<div class="col"><header><h1>What RL changes in Dan\'s best-state provers</h1>'
        f'<p class="lede">{P.esc(lede(j, n_ckpt, ceiling=ceiling))}</p>'
        f'<p class="models">{P.esc(models_note(nums))}</p></header>'
        f"<main>{_overview_card(charts, nums)}{''.join(sections)}"
        f'<section class="limits"><h2>Limitations</h2><ul>{limits}</ul></section></main>'
        f"<footer>GPU time on the ledger for jobs on Dan's models (benchmarks, interp, gates "
        f"and smoke tests): {_gpu_hours(store):.2f} GPU-hours. Source of Dan's published "
        "numbers: "
        f"{P.esc(dv.DAN_SOURCE)}.</footer></div>"
        f"<script>{summary._JS}</script></body></html>"  # noqa: SLF001
    )


def build_dan_only(
    store: Store,
    out_dir: Path,
    detail_href: str | None = None,
    *,
    registry: Mapping[str, mr.ModelEntry] | None = None,
) -> Path:
    """`dash --summary --dan-only`: write the page to `out_dir/index.html` (the site's only
    page, so it links nowhere else; `detail_href` is accepted for CLI symmetry and ignored)."""
    del detail_href
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "index.html"
    path.write_text(render_page(store, registry=registry))
    return path


#: Builders by key, for callers and tests that compute a single chart.
BUILDERS: dict[str, Callable[..., ChartData]] = {
    "do_ceiling": _c_ceiling,
    "do_ceiling_table": _c_ceiling_table,
    "do_failures": _c_failures,
    "do_edl": _c_edl,
    "do_textbook": _c_textbook,
    "do_families": _c_families,
    "do_laws": _c_laws,
    "do_glue": _c_glue,
    "do_skills": _c_skills,
    "do_padding": _c_padding,
    "do_prefix": _c_prefix,
    "do_heldout": _c_heldout,
    "do_textbook_law": _c_textbook_law,
    "do_likelihood": _c_likelihood,
    "do_kl": _c_kl,
    "do_steering": _c_steering,
    "do_weights": _c_weights,
    "do_probes": _c_probes,
}
