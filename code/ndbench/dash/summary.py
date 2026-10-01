"""`dash.summary`: a single-page, minimal-sift-through dashboard -- one figure per benchmark.

Leon (2026-09-30, after seeing the full dashboard): "this is way way too many graphs you should be
trying to minimize the amount of data a human needs to sift through ideally one graph per
benchmark." `dash/build.py` renders 13 full pages (overview, nine benchmarks, four interp
measures) with several charts and tables each; this module instead writes one `summary.html` laid
out like a paper's results section: a short top line, a Figure 1 overview grid, exactly 12
numbered cards (one per benchmark/measure named in the plan, in two sections) and a Limitations
note. Each card is a NeurIPS-style figure (`code/ndbench/figures/plots.py`; steering is a
booktabs table instead, `figures/tables.py`) with a self-contained caption below it (what is
plotted, n, the CI method, then the *computed* one-sentence takeaway) and the exact values in a
collapsible data table.

Each benchmark's numbers are computed once, by its `_data_*` function, into a `ChartData`
(`dash/chart_data.py`); the page's tables (`data_tables`), its captions and the figures
(`figures.plots`, also written as PDF/SVG/PNG by `python -m code.ndbench figures`) all read that
object, so they cannot drift apart. The data tables are the same rows the earlier inline-SVG
charts carried (`P.line_table_rows`/`P.bar_table_rows`/`P.dot_table_rows`), plus one per extra
figure panel.

Every number on this page comes from the same `analysis/*` views and `Store` the full dashboard
uses (`view_<bench>(store, cfg)`, `code.ndbench.metrics`) -- this module adds no new statistics,
only: (a) small aggregations the existing per-benchmark views don't themselves expose because they
were built for a *detailed* page (pooling glue's ~19 cells into 4 operators, families' n* into
written lines, padding's three pad types into one pooled curve, law x substitution averaged across
eight schemas); and (b) the takeaway sentence for each chart, computed from the same numbers the
chart plots, never hand-written.

Charts 9 ("the distribution of base-model log-likelihood per line") needs each item's certified
length, which `interp/likelihood.py`'s row schema only fills in for its neutral/family rows, not
for model-sample rows (`origin in ("base_verified", "rl_verified")`) -- `_likelihood_crux_data`
below resolves each one by reading the *run's own recorded config* (`model_samples[model]
["suite"/"version"]`, the suite that model's verified samples were actually drawn from) and looking
the theorem back up there, rather than guessing a suite name or leaving the chart unnormalised.

Chart 8 ("EDL per example per class") hits a second, narrower gap: `analysis/edl.py`'s own
`_label_from_run_id` only recognises a literal `edl-<label>-<n>-<m>` run id, so this plan's `edl2-*`
run ids (plan: "use only the latest EDL run ids (edl2-*)") fall through to the raw checkpoint path
as a label on the real store (confirmed: `AbsGptAdapter`'s identity carries no `"label"` key, so the
production run never hits priority (1) either). `_edl_friendly_label` below reproduces that same
priority order with the prefix check widened to "starts with edl" instead of "equals edl", used
only to *relabel* `edl_view.view`'s own output (which already implements this plan's "choose lr by
smallest online code length" rule) -- not a reimplementation of its statistics.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from ..analysis import edl as edl_view
from ..analysis import glue as glue_view
from ..analysis import horizon as horizon_view
from ..analysis import interp as interp_view
from ..analysis import padding as padding_view
from ..analysis import prefix as prefix_view
from ..analysis import skills as skills_view
from ..analysis import subst as subst_view
from ..analysis import yue as yue_view
from ..analysis._common import (
    bootstrap_ci,
    default_version,
    item_length,
    items_by_pid,
    latest_run_id,
    models_present,
    problems_by_pid,
)
from ..core.types import JSON, ItemResult, Problem
from ..interp._store import interp_runs, latest_interp_run, load_interp_rows
from ..metrics import compositionality_ratio, pass_at_k
from ..metrics import horizon as horizon_stat
from ..store import Store
from . import primitives as P
from .chart_data import ChartData
from .chart_data import missing_chart as _missing

#: Fixed per-model colours (plan: "consistent colours: base, rl and armC each always the same
#: colour") -- independent of iteration order, unlike `dash/pages.py`'s `_color(i)` (which cycles
#: through the theme palette in whatever order a view's `sorted(models)` happens to produce, so
#: `armC` can land on the "base" slot on one page and not another).
MODEL_COLOR: dict[str, str] = {"base": "var(--s1)", "rl": "var(--s2)", "armC": "var(--s3)"}
_EXTRA_COLORS: tuple[str, ...] = ("var(--s4)", "var(--s5)", "var(--s6)")


def _mcolor(model: str) -> str:
    return MODEL_COLOR.get(model, _EXTRA_COLORS[hash(model) % len(_EXTRA_COLORS)])


def _models_label(models: Sequence[str]) -> str:
    order = {"base": 0, "rl": 1, "armC": 2}
    ms = sorted(models, key=lambda m: order.get(m, 99))
    if len(ms) <= 1:
        return ", ".join(ms)
    return ", ".join(ms[:-1]) + f" and {ms[-1]}" if len(ms) > 1 else ms[0]


# ============================================================================= 1. Yue (textbook)


def _data_1_yue(store: Store) -> ChartData:
    key, title = "yue", "Pass@k on real textbook problems"
    data = yue_view.view(store, {})
    if data.get("empty"):
        return _missing(key, title, str(data.get("reason", "no data")))
    sets = cast("dict[str, Any]", data["sets"])
    tdev = cast("dict[str, Any] | None", sets.get("textbook_dev"))
    if not tdev:
        return _missing(key, title, "no textbook_dev results in the store")
    models = cast("list[str]", tdev["models"])
    curves = cast("dict[str, list[tuple[int, float, float, float]]]", tdev["curves"])
    n_items = cast("dict[str, int]", tdev["n_items"])
    series = [
        P.Series(
            m,
            [(float(k), mean) for k, mean, _lo, _hi in curves.get(m, [])],
            _mcolor(m),
            band=[(float(k), lo, hi) for k, _mean, lo, hi in curves.get(m, [])],
        )
        for m in models
    ]

    def _pt(model: str, k: int) -> float | None:
        return next((mean for kk, mean, _lo, _hi in curves.get(model, []) if kk == k), None)

    def _kmax(model: str) -> tuple[int, float] | None:
        c = curves.get(model)
        return (c[-1][0], c[-1][1]) if c else None

    cross = cast("dict[str, Any] | None", tdev.get("crossover"))
    pair = cast("list[str] | None", tdev.get("pair"))
    takeaway = "pass@k curves for " + _models_label(models) + "."
    if pair and len(pair) == 2 and cross is not None:
        a, b = pair
        a1, b1 = _pt(a, 1), _pt(b, 1)
        a_max, b_max = _kmax(a), _kmax(b)
        if a1 is not None and b1 is not None and a_max and b_max:
            lead, trail = (a, b) if a1 >= b1 else (b, a)
            lead1, trail1 = (a1, b1) if a1 >= b1 else (b1, a1)
            lead_max, trail_max = (a_max, b_max) if a1 >= b1 else (b_max, a_max)
            if cross.get("k_star") is None:
                p_none = cast("float | None", cross.get("p_none"))
                robust = (
                    ""
                    if p_none is None or p_none != p_none or p_none >= 0.95
                    else f", but {P.fmt(1 - p_none, 'pct')} of bootstrap resamples do cross, so "
                    "the ordering is not robust"
                )
                takeaway = (
                    f"{lead} is ahead of {trail} at every k in the point estimate (pass@1 "
                    f"{P.fmt(lead1, 'pct')} vs {P.fmt(trail1, 'pct')}; pass@{lead_max[0]} "
                    f"{P.fmt(lead_max[1], 'pct')} vs {P.fmt(trail_max[1], 'pct')}){robust}"
                )
                others = [m for m in models if m not in pair and curves.get(m)]
                for m in others:
                    m1, m_max = _pt(m, 1), _kmax(m)
                    if m1 is not None and m_max is not None:
                        takeaway += (
                            f"; {m} starts highest ({P.fmt(m1, 'pct')} at k=1) but plateaus at "
                            f"{P.fmt(m_max[1], 'pct')} (pass@{m_max[0]})"
                            if m1 > max(lead1, trail1) and m_max[1] < min(lead_max[1], trail_max[1])
                            else f"; {m}: {P.fmt(m1, 'pct')} at k=1, {P.fmt(m_max[1], 'pct')} "
                            f"at k={m_max[0]}"
                        )
                takeaway += "."
            else:
                takeaway = (
                    f"{a} and {b} cross over near k={cross['k_star']} "
                    f"(95% CI {cross['ci'][0]}-{cross['ci'][1]})."
                )

    def _caption_pair(label: str) -> str:
        s = cast("dict[str, Any] | None", sets.get(label))
        if not s:
            return ""
        c = cast("dict[str, Any]", s["curves"])
        bits = []
        for m in cast("list[str]", s["models"]):
            curve = c.get(m, [])
            if not curve:
                continue
            p1 = next((mean for k, mean, _l, _h in curve if k == 1), None)
            kmax, pmax = curve[-1][0], curve[-1][1]
            bits.append(f"{m} {P.fmt(p1, 'pct')}/pass@{kmax} {P.fmt(pmax, 'pct')}")
        return f"{label}: " + ", ".join(bits) if bits else ""

    extra = "; ".join(x for x in (_caption_pair("textbook_law"), _caption_pair("control")) if x)
    # Panel (b): the textbook_law curves the caption already summarises, drawn from the same view.
    law = cast("dict[str, Any] | None", sets.get("textbook_law"))
    law_series: list[P.Series] = []
    law_n: object = "?"
    if law:
        law_curves = cast("dict[str, list[tuple[int, float, float, float]]]", law["curves"])
        law_models = cast("list[str]", law["models"])
        law_n = (
            cast("dict[str, int]", law["n_items"]).get(law_models[0], "?") if law_models else "?"
        )
        law_series = [
            P.Series(
                m,
                [(float(k), mean) for k, mean, _lo, _hi in law_curves.get(m, [])],
                _mcolor(m),
                band=[(float(k), lo, hi) for k, _mean, lo, hi in law_curves.get(m, [])],
            )
            for m in law_models
            if law_curves.get(m)
        ]
    panel_b = (
        f"; (b) textbook_law (n={law_n} theorems, k=1..{law.get('k_max')})"
        if law_series and law
        else ""
    )
    caption = (
        f"Pass@k against the number of samples k (log scale) for {_models_label(models)}: "
        f"(a) textbook_dev (n={n_items.get(models[0], '?')} textbook theorems, "
        f"k=1..{tdev.get('k_max')}){panel_b}. Unbiased pass@k; past an item's own sample count the "
        "curve uses the 1-(1-c/n)^k extrapolation. Shaded bands: 95% item-level bootstrap CIs."
        + (f" {extra}." if extra else "")
    )
    cd = ChartData(
        key,
        title,
        "line",
        takeaway,
        caption,
        series=series,
        opts={"x_label": "k", "y_fmt": "pct", "log_x": True},
        extra={"n": n_items.get(models[0], "?"), "k_max": tdev.get("k_max")},
    )
    if law_series:
        cd.extra["law"] = {
            "series": law_series,
            "n": law_n,
            "k_max": law.get("k_max") if law else None,
        }
        cd.extra_tables.append(
            ("textbook_law (panel b)", *P.line_table_rows(law_series, x_label="k", y_fmt="pct"))
        )
    return cd


# ======================================================================== 2. Restricted families


#: Per-(family, n) cells with fewer sampled items than this are excluded before computing the
#: horizon (review fix: several real cells have only 1 item per n -- e.g. `dn_elim`/`dn_intro`,
#: most ILTP sub-schemas -- nowhere near enough signal to call a 0%/100% pass rate; one real
#: example, `reorder_or` n=2 (1 item: base solves it, rl doesn't), made a coin flip look like an
#: RL regression). `metrics.horizon` is called directly on the filtered `by_n` here, since
#: `horizon.view`'s own `n_star` has no `min_items` knob.
_MIN_ITEMS_PER_N = 5
#: Minimum items per s cell on the laws chart (s=0 has one item per schema, 8 in all).
_MIN_ITEMS_PER_BIN = 20


def _family_group_pids(problems: Mapping[str, Problem]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for pid, p in problems.items():
        fam = p.tags.get("family")
        if fam is None:
            continue
        kind = p.tags.get("kind")
        key = str(fam) if kind is None else f"{fam} ({kind})"
        out.setdefault(key, []).append(pid)
    return out


def _family_horizon(
    store: Store,
    suite: str,
    version: str,
    model: str,
    pids: Sequence[str],
    problems: Mapping[str, Problem],
) -> tuple[int | None, bool] | None:
    """`(n_star, censored)` for one (family, model) from `metrics.horizon` restricted to n-cells
    with `>= _MIN_ITEMS_PER_N` items; `None` if no cell qualifies at all. `n_star=None` means
    pass@8 was already below 50% at the smallest *qualifying* size (review: "None means 'fails
    even at the smallest size', not 'infinite'" -- `metrics.horizon.n_star` walks n in increasing
    order and stops at the first failure, so `None` is specifically "failed at the first n
    tried"). `censored=True` means `n_star` is the largest n actually tested, i.e. pass@8 never
    dropped below 50% within the tested range: a lower bound on the true horizon, not the horizon
    itself.
    """
    run_id = latest_run_id(store, suite=suite, model=model, version=version)
    if run_id is None:
        return None
    items = items_by_pid(store, run_id)
    by_n: dict[int, list[ItemResult]] = {}
    for pid in pids:
        item = items.get(pid)
        if item is None:
            continue
        n_val = problems[pid].tags.get("n")
        if n_val is None:
            continue
        by_n.setdefault(int(cast(int, n_val)), []).append(item)
    qualifying = {n: rows for n, rows in by_n.items() if len(rows) >= _MIN_ITEMS_PER_N}
    if not qualifying:
        return None
    result = horizon_stat(qualifying, k=8, tau=0.5, B=2000, seed=0)
    n_star = result["n_star"]
    censored = n_star is not None and n_star == max(qualifying)
    return n_star, censored


def _family_written_lines(
    pids: Sequence[str], problems: Mapping[str, Problem], n_star: int
) -> float | None:
    lens = [
        v
        for p in pids
        if problems[p].tags.get("n") == n_star and (v := item_length(problems[p])) is not None
    ]
    return statistics.mean(lens) if lens else None


def _data_2_families(store: Store) -> ChartData:
    key, title = "families", "Length horizon on restricted families (theorems unseen in training)"
    data = horizon_view.view(store, {})
    if data.get("empty"):
        return _missing(key, title, str(data.get("reason", "no data")))
    meta = cast("dict[str, JSON]", data["meta"])
    families = cast("dict[str, dict[str, dict[str, Any]]]", data["families"])
    suite, version = cast(str, meta["suite"]), cast(str, meta["version"])
    problems = problems_by_pid(store, suite, version)
    pids_by_group = _family_group_pids(problems)

    dropped: list[str] = []
    rows: list[P.DotRow] = []
    # base-vs-rl comparison, lines written at n* (a "fails at smallest size" counts as 0 lines --
    # strictly less horizon than any family that clears at least its own smallest tested size).
    compare: dict[str, dict[str, float]] = {}
    longest: tuple[float, str, str] | None = None  # (lines, model, family label)
    # Audit 2026-10-01: the smallest sizes are mostly training theorems (`seen_train`: base's
    # cap-6 pretraining set or the RL pool), so the horizon is computed on unseen theorems only;
    # the all-items horizon is kept for the caption.
    with_seen: dict[str, list[str]] = {}
    for family in sorted(families):
        all_pids = pids_by_group.get(family, [])
        pids = [p for p in all_pids if not problems[p].seen_train]
        label = "ILTP-style" if family.startswith("ILTP-style") else family
        res_all = _family_horizon(store, suite, version, "base", all_pids, problems)
        if res_all is not None and res_all[0] is not None:
            with_seen.setdefault(label, []).append(f"n*={res_all[0]}")
        pts: list[tuple[str, float | None, str, bool]] = []
        for model in ("base", "rl", "armC"):
            res = _family_horizon(store, suite, version, model, pids, problems)
            if res is None:
                continue
            n_star, censored = res
            if n_star is None:
                pts.append((model, None, _mcolor(model), False))
                if model in ("base", "rl"):
                    compare.setdefault(label, {})[model] = 0.0
                continue
            lines = _family_written_lines(pids, problems, n_star)
            pts.append((model, lines, _mcolor(model), censored))
            if lines is not None:
                if model in ("base", "rl"):
                    compare.setdefault(label, {})[model] = lines
                if longest is None or lines > longest[0]:
                    longest = (lines, model, label)
        if not pts:
            dropped.append(family)
            continue
        rows.append(P.DotRow(label, pts))

    if not rows:
        return _missing(key, title, "no family had a qualifying (>=5-item) size cell")

    trails = [
        fam
        for fam, by_model in compare.items()
        if (b := by_model.get("base")) is not None
        and (r := by_model.get("rl")) is not None
        and r < b - 0.5
    ]

    holders = [
        (model, row.label, v) for row in rows for model, v, _c, _cens in row.points if v is not None
    ]
    base_none = all(
        v is None for row in rows for model, v, _c, _cens in row.points if model == "base"
    )
    if holders:
        holders.sort(key=lambda t: -t[2])
        takeaway = "on theorems absent from training no family holds up for long: " + ", ".join(
            f"{model} on {fam} reaches about {lines:.0f} written lines"
            for model, fam, lines in holders
        )
        if base_none:
            takeaway += "; base fails at the smallest unseen size in every family"
        takeaway += "."
    else:
        takeaway = "every model fails at the smallest unseen size in every family."
    if trails:
        takeaway = takeaway[:-1] + f"; rl trails base on {', '.join(trails)}."

    drop_note = f" Too few items per size (dropped): {', '.join(dropped)}." if dropped else ""
    seen_note = (
        " Including training theorems, base reaches "
        + ", ".join(f"{v[0]} on {fam}" for fam, v in with_seen.items())
        + "."
        if with_seen
        else ""
    )
    caption = (
        "Horizon of each restricted family for base, rl and armC: the mean written length (lines) "
        "of the items at n*, the largest family size n at which pass@8 is still at least 50% "
        f"({len(rows)} families). n* at pass@8>=50% over size cells with >={_MIN_ITEMS_PER_N} "
        "unseen items only (theorems in base's pretraining set or the RL pool are excluded; "
        "they make up most of the smallest sizes). A hollow marker at 0 means the model fails at "
        "the smallest unseen size; an arrow marks a censored horizon (never dropped below 50% in "
        "the tested range, so n* is a lower bound). Point estimates; no CI is drawn."
        f"{seen_note}{drop_note}"
    )
    return ChartData(
        key,
        title,
        "dot",
        takeaway,
        caption,
        rows=rows,
        opts={
            "x_label": "written lines at n*",
            "x_min": 0.0,
            "none_label": "fails at smallest size",
        },
    )


# ===================================================================================== 3. Glue


_OP_GROUP: dict[str, str] = {
    "AND": "AND",
    "CHAIN": "CHAIN",
    "CASE-dilemma": "CASE",
    "CASE-same": "CASE",
    "BOX": "BOX",
}
_OP_ORDER: tuple[str, ...] = ("AND", "CHAIN", "CASE", "BOX")


def _data_3_glue(store: Store) -> ChartData:
    key, title = "glue", "Compositionality of solved parts (glue)"
    suite = glue_view.DEFAULT_SUITE
    parts_suite = glue_view.DEFAULT_PARTS_SUITE
    version = default_version(store, suite=suite) or "v1"
    parts_version = default_version(store, suite=parts_suite) or "v1"
    problems = problems_by_pid(store, suite, version)
    if not problems:
        return _missing(key, title, f"suite {suite!r} not found")
    models = models_present(store, suite=suite, version=version)
    if not models:
        return _missing(key, title, "no runs found for the glue suite")

    pids_by_op: dict[str, list[str]] = {}
    for pid, p in problems.items():
        op = _OP_GROUP.get(cast(str, p.tags.get("op", "")))
        if op:
            pids_by_op.setdefault(op, []).append(pid)

    bars: list[P.Bar] = []
    counts: dict[str, int] = {}
    sources: dict[str, int] = {}
    for op in _OP_ORDER:
        pids = pids_by_op.get(op, [])
        if not pids:
            continue
        counts[op] = len(pids)
        for model in ("base", "rl"):
            run_id = latest_run_id(store, suite=suite, model=model, version=version)
            parts_run_id = latest_run_id(
                store, suite=parts_suite, model=model, version=parts_version
            )
            if run_id is None or parts_run_id is None:
                continue
            joined_items = items_by_pid(store, run_id)
            part_items = items_by_pid(store, parts_run_id)
            renamed_items = glue_view.renamed_part_items(store, model)
            groups = []
            for pid in pids:
                item = joined_items.get(pid)
                if item is None:
                    continue
                found = glue_view.part_results(pid, problems[pid], part_items, renamed_items)
                if found is not None:
                    groups.append((item, found[0]))
                    sources[found[1]] = sources.get(found[1], 0) + 1
            if not groups:
                continue
            comp = compositionality_ratio(groups, mastery=0.5, B=1000, seed=0)
            lo, hi = comp["ci"]
            bars.append(P.Bar(f"{op}: {model}", comp["R"], lo, hi, _mcolor(model)))

    if not bars:
        return _missing(key, title, "no operator had both a joined and a parts_pool run")
    by_op: dict[str, list[P.Bar]] = {}
    for b in bars:
        by_op.setdefault(b.name.split(": ", 1)[0], []).append(b)
    zero_ops = [op for op, bs in by_op.items() if all(abs(b.value) < 1e-9 for b in bs)]
    nonzero = [b for b in bars if b.name.split(": ", 1)[0] not in zero_ops]
    bits = []
    if nonzero:
        best = max(nonzero, key=lambda b: b.value)
        bits.append(f"{best.name} is highest (R={P.fmt(best.value, 'f3')})")
        worst = min(nonzero, key=lambda b: b.value)
        if worst.name != best.name:
            bits.append(f"{worst.name} is lowest (R={P.fmt(worst.value, 'f3')})")
    for op in zero_ops:
        bits.append(f"{op}: R=0 for both models")
    takeaway = "compositionality varies by operator: " + "; ".join(bits) + "."
    src_note = (
        "parts' pass@1 measured on each part as it appears inside the join (atoms renamed)"
        if not sources.get("pool")
        else f"parts' pass@1 from the renamed parts for {sources.get('renamed', 0)} and the "
        f"original pool items for {sources['pool']} (item, model) pairs"
    )
    caption = (
        "Compositionality ratio R per glue operator, base vs rl: R = joined pass@1 / product of "
        "mastered parts' pass@1, pooled per operator (summed over items whose parts all have "
        "pass@1 >= 0.5; R = 1 means a join is as easy as its parts together); "
        + "; ".join(f"{op} n={n}" for op, n in counts.items())
        + f"; {src_note}. Error bars: 95% item-level bootstrap CIs (B=1000)."
    )
    return ChartData(
        key,
        title,
        "bar",
        takeaway,
        caption,
        bars=bars,
        opts={"y_fmt": "f3"},
        extra={"ops": [op for op in _OP_ORDER if op in counts], "counts": counts},
    )


# =================================================================================== 4. Skills


def _rl_items_by_pid(store: Store, problems: Mapping[str, Problem]) -> dict[str, tuple[int, int]]:
    """RL's `(n, c)` per `skill_pool` pid, direct run first, legacy-imported runs filling the
    rest (mirrors `analysis/skills.py:view`'s own two-source join, replicated here because the
    view's returned dict only keeps per-skill/per-length *aggregates*, not these raw item rows --
    needed here to build a third "no special skill" bucket the view never computes)."""
    out: dict[str, tuple[int, int]] = {}
    version = default_version(store, suite=skills_view.DEFAULT_SUITE) or "v1"
    run_id = latest_run_id(store, suite=skills_view.DEFAULT_SUITE, model="rl", version=version)
    if run_id is not None:
        for pid, item in items_by_pid(store, run_id).items():
            out[pid] = (item.n, item.c)
    source_sets = sorted(
        {str(v) for p in problems.values() if (v := p.tags.get("source_set")) is not None}
    )
    legacy = skills_view._joined_items_by_model(store, source_sets)  # noqa: SLF001
    for pid, nc in legacy.get("rl", {}).items():
        out.setdefault(pid, nc)
    return out


#: (L, bucket) cells with fewer sampled items than this are dropped before plotting (review: a
#: 15-item "needs classical" cell at L=8 landed at 48.1% pass@1, a spike driven entirely by n=15 --
#: no cell that thin is informative, so it is excluded rather than plotted at face value).
_MIN_SKILL_CELL_ITEMS = 20

#: Neutral, non-model colours for the three skill lines (review: "don't reuse the rl model's
#: orange for the 'no special skill' line -- all three lines are rl", so none of these should be
#: `MODEL_COLOR["rl"]`, which a reader would otherwise read as "this line is a different model").
_SKILL_LINE_COLORS: dict[str, str] = {
    "no special skill": "var(--s4)",
    "needs case split": "var(--s5)",
    "needs classical": "var(--s6)",
}


def _data_4_skills(store: Store) -> ChartData:
    key, title = "skills", "Pass@1 by proof length and required skill"
    suite = skills_view.DEFAULT_SUITE
    version = default_version(store, suite=suite) or "v1"
    problems = problems_by_pid(store, suite, version)
    if not problems:
        return _missing(key, title, f"suite {suite!r} not found")
    item_ncs = _rl_items_by_pid(store, problems)
    if not item_ncs:
        return _missing(key, title, "no rl results (direct or legacy) for skill_pool")

    buckets: dict[str, dict[float, list[tuple[int, int]]]] = {
        "no special skill": {},
        "needs case split": {},
        "needs classical": {},
    }
    for pid, (n, c) in item_ncs.items():
        p = problems.get(pid)
        if p is None or n <= 0:
            continue
        length = item_length(p)
        if length is None:
            continue
        is_classical = skills_view._skill_active(p, "classical")  # noqa: SLF001
        is_case = skills_view._skill_active(p, "case_split")  # noqa: SLF001
        bucket = (
            "needs classical"
            if is_classical
            else ("needs case split" if is_case else "no special skill")
        )
        buckets[bucket].setdefault(length, []).append((n, c))

    series = []
    dropped_cells = 0
    for name, by_l in buckets.items():
        pts: list[tuple[float, float]] = []
        band: list[tuple[float, float, float]] = []
        for length, ncs in sorted(by_l.items()):
            if len(ncs) < _MIN_SKILL_CELL_ITEMS:
                dropped_cells += 1
                continue
            vals = [pass_at_k(n, c, 1) for n, c in ncs]
            mean = statistics.mean(vals)
            pts.append((length, mean))
            lo, hi = bootstrap_ci(vals, seed=0, B=500)
            band.append((length, lo, hi))
        if pts:
            series.append(P.Series(name, pts, _SKILL_LINE_COLORS[name], band=band))
    if not series:
        return _missing(key, title, "no (length, skill) cell had >=20 items")

    def _overall(name: str) -> float:
        vals = [pass_at_k(n, c, 1) for ncs in buckets[name].values() for n, c in ncs]
        return statistics.mean(vals) if vals else float("nan")

    overall = {name: _overall(name) for name in buckets}
    takeaway = (
        f"RL's pass@1 is {P.fmt(overall['no special skill'], 'pct')} with no special skill needed, "
        f"{P.fmt(overall['needs case split'], 'pct')} when a case split is required and "
        f"{P.fmt(overall['needs classical'], 'pct')} when the proof is classical."
    )
    n_items = sum(len(v) for by_l in buckets.values() for v in by_l.values())
    drop_note = (
        f" {dropped_cells} (length, skill) cells with <{_MIN_SKILL_CELL_ITEMS} items are omitted."
        if dropped_cells
        else ""
    )
    sk = skills_view.view(store, {})
    cmp_notes: list[str] = []
    comparisons: list[dict[str, Any]] = []
    for cmp in cast("dict[str, dict[str, Any]]", sk.get("comparisons") or {}).values():
        a, b = cast("list[str]", cmp["models"])
        groups = cast("dict[str, dict[str, Any]]", cmp["by_skill"])
        cells = ", ".join(
            f"{g}: {P.fmt(r[a], 'pct')} vs {P.fmt(r[b], 'pct')}" for g, r in groups.items()
        )
        cmp_notes.append(f"{a} vs {b} on their {cmp['n_common']} common items ({cells})")
        comparisons.append({"models": [a, b], "n_common": cmp["n_common"], "by_skill": groups})
    caption = (
        f"(a) rl's pass@1 against written proof length L for each skill group; "
        f"n={n_items} skill_pool items (direct + legacy-imported runs), rl only; shaded band: "
        f"95% item-level bootstrap CI.{drop_note}"
        + (
            " Right panels: pass@1 per skill group on the items both models of a pair were "
            "sampled on (point estimates). Same items, pass@1: " + "; ".join(cmp_notes) + "."
            if cmp_notes
            else ""
        )
    )
    cd = ChartData(
        key,
        title,
        "line",
        takeaway,
        caption,
        series=series,
        opts={"x_label": "L (written lines)", "y_fmt": "pct"},
        extra={"comparisons": comparisons},
    )
    for c in comparisons:
        a, b = c["models"]
        cd.extra_tables.append(
            (
                f"{a} vs {b} on {c['n_common']} common items (panel b)",
                ["skill group", "items", a, b],
                [
                    [g, str(r["n_items"]), P.fmt(r[a], "pct"), P.fmt(r[b], "pct")]
                    for g, r in c["by_skill"].items()
                ],
            )
        )
    return cd


# ============================================================================ 5. Law x substitution


def _data_5_subst(store: Store) -> ChartData:
    key, title = "subst", "Known laws under larger substitutions"
    data = subst_view.view(store, {})
    if data.get("empty"):
        return _missing(key, title, str(data.get("reason", "no data")))
    schemas = cast("dict[str, dict[str, dict[str, Any]]]", data["schemas"])
    # Item-level pooling per s (audit 2026-10-01): a mean of per-schema means weights s=0's single
    # item per schema like s=2's ~30, and the set of schemas present changes with s.
    pooled: dict[str, dict[float, list[float]]] = {}  # model -> s -> [sum pass1*n, n]
    zero_schemas: dict[str, int] = {}
    n_items = 0
    for per_model in schemas.values():
        for model, d in per_model.items():
            n_items += cast(int, d["n_items"])
            if cast(float, d.get("controlled_pass1_overall", 0.0)) == 0.0:
                zero_schemas[model] = zero_schemas.get(model, 0) + 1
            for row in cast("list[dict[str, Any]]", d["curve_vs_s"]):
                s, p1, n = row["s"], row["pass1"], int(row["n_items"])
                if s is None or p1 != p1:
                    continue
                acc = pooled.setdefault(model, {}).setdefault(float(s), [0.0, 0.0])
                acc[0] += p1 * n
                acc[1] += n
    dropped_s = sorted(
        {s for by_s in pooled.values() for s, (_t, n) in by_s.items() if n < _MIN_ITEMS_PER_BIN}
    )
    by_model_s: dict[str, dict[float, list[float]]] = {
        model: {s: [t / n] for s, (t, n) in by_s.items() if n >= _MIN_ITEMS_PER_BIN}
        for model, by_s in pooled.items()
    }

    series = []
    for model in ("base", "rl"):
        by_s = by_model_s.get(model)
        if not by_s:
            continue
        pts = sorted((s, statistics.mean(v)) for s, v in by_s.items())
        series.append(P.Series(model, pts, _mcolor(model)))
    if not series:
        return _missing(key, title, "no base/rl results for the laws suite")

    def _at(model: str, s: float) -> float | None:
        by_s = by_model_s.get(model, {})
        return statistics.mean(by_s[s]) if s in by_s else None

    s_values = sorted({s for by_s in by_model_s.values() for s in by_s})
    common_s = [s for s in s_values if _at("base", s) is not None and _at("rl", s) is not None]
    bits: list[str] = []
    if common_s:
        peak_b_s = max(common_s, key=lambda s: cast(float, _at("base", s)))
        peak_r_s = max(common_s, key=lambda s: cast(float, _at("rl", s)))
        bits.append(
            f"base peaks at s={P.fmt(peak_b_s, 'g')} ({P.fmt(_at('base', peak_b_s), 'pct')}), "
            f"rl peaks at s={P.fmt(peak_r_s, 'g')} ({P.fmt(_at('rl', peak_r_s), 'pct')})"
        )
        near_zero = next(
            (
                s
                for s in common_s
                if cast(float, _at("base", s)) < 0.01 and cast(float, _at("rl", s)) < 0.01
            ),
            None,
        )
        if near_zero is not None:
            bits.append(f"both are near 0% by s={P.fmt(near_zero, 'g')}")
        # the smallest s after which base is at or above rl for every larger s tested.
        base_ge_from = next(
            (
                common_s[i]
                for i in range(len(common_s))
                if all(
                    cast(float, _at("base", s)) >= cast(float, _at("rl", s)) for s in common_s[i:]
                )
            ),
            None,
        )
        if base_ge_from is not None and base_ge_from != common_s[0]:
            bits.append(f"base is at or above rl from s={P.fmt(base_ge_from, 'g')} onward")
    takeaway = "; ".join(bits) + "." if bits else "pass@1 vs filler size s, base vs rl."
    meta = cast("dict[str, Any]", data["meta"])
    zero_note = (
        f" base scores 0% on {zero_schemas['base']} of {len(schemas)} schemas."
        if zero_schemas.get("base")
        else ""
    )
    caption = (
        "Pass@1 against the substitution size s (connectives per metavariable) for instances of "
        f"known laws, base vs rl; n={n_items} instances over {len(schemas)} schemas "
        f"({', '.join(sorted(schemas))}), "
        f"pooled per item at each s; {meta.get('n_seen_train_excluded', 0)} instances seen in "
        "training excluded"
        + (
            f"; s={', '.join(P.fmt(v, 'g') for v in dropped_s)} omitted "
            f"(<{_MIN_ITEMS_PER_BIN} items)"
            if dropped_s
            else ""
        )
        + f".{zero_note} Point estimates."
    )
    return ChartData(
        key,
        title,
        "line",
        takeaway,
        caption,
        series=series,
        opts={"x_label": "s (connectives per metavariable)", "y_fmt": "pct"},
    )


# ================================================================================ 6. Free lines


def _pool_curve(
    curves: Sequence[list[dict[str, Any]]],
) -> list[tuple[float, float, int]]:
    by_w: dict[float, list[tuple[float, int]]] = {}
    for curve in curves:
        for row in curve:
            p1 = row["pass1"]
            if p1 != p1:  # nan
                continue
            by_w.setdefault(row["W"], []).append((p1, row["n_items"]))
    out = []
    for w, vals in sorted(by_w.items()):
        tot_n = sum(n for _p, n in vals)
        weighted = sum(p * n for p, n in vals) / tot_n if tot_n else float("nan")
        out.append((w, weighted, tot_n))
    return out


def _padded_curve_dedup(
    store: Store, suite: str, version: str, model: str
) -> list[tuple[float, float, int]]:
    """pass@1 vs W over the padding suite, pooled across pad types with each unpadded (m=0)
    theorem counted once (audit 2026-10-01: every m=0 sequent appears under all three pad types,
    which triple-counted it at W=3-6). `(W, mean pass@1, n distinct items)`."""
    run_id = latest_run_id(store, suite=suite, model=model, version=version)
    if run_id is None:
        return []
    items = items_by_pid(store, run_id)
    by_w: dict[float, dict[str, list[float]]] = {}
    for pid, prob in problems_by_pid(store, suite, version).items():
        item = items.get(pid)
        w = item_length(prob)
        if item is None or not item.n or w is None:
            continue
        key = f"base:{prob.tags.get('base_pid')}" if prob.tags.get("m") == 0 else pid
        by_w.setdefault(w, {}).setdefault(key, []).append(item.c / item.n)
    return [
        (w, statistics.mean(statistics.mean(v) for v in d.values()), len(d))
        for w, d in sorted(by_w.items())
    ]


def _data_6_padding(store: Store) -> ChartData:
    key, title = "padding", "Free lines: the cost of padding"
    data = padding_view.view(store, {})
    if data.get("empty"):
        return _missing(key, title, str(data.get("reason", "no data")))
    pad_types = cast("dict[str, dict[str, dict[str, Any]]]", data["pad_types"])

    def _curves(model: str, key: str) -> list[list[dict[str, Any]]]:
        return [d[key] for per_model in pad_types.values() if (d := per_model.get(model))]

    meta = cast("dict[str, Any]", data["meta"])
    rl_padded = _padded_curve_dedup(
        store, cast(str, meta["suite"]), cast(str, meta["version"]), "rl"
    )
    # The natural curve is the same suite under every pad type: take one copy, not the pool.
    rl_natural = _pool_curve(_curves("rl", "natural_curve_vs_W")[:1])
    base_natural = _pool_curve(_curves("base", "natural_curve_vs_W")[:1])
    base_padded = _padded_curve_dedup(
        store, cast(str, meta["suite"]), cast(str, meta["version"]), "base"
    )
    if not rl_padded and not base_padded:
        return _missing(key, title, "no base/rl results for the padding suite")

    series = []
    if rl_padded:
        series.append(P.Series("rl padded", [(w, p) for w, p, _n in rl_padded], _mcolor("rl")))
    if rl_natural:
        series.append(
            P.Series("rl natural", [(w, p) for w, p, _n in rl_natural], "var(--s4)", muted=True)
        )
    if base_padded:
        series.append(
            P.Series("base padded", [(w, p) for w, p, _n in base_padded], _mcolor("base"))
        )

    common_w = sorted(set(w for w, _p, _n in rl_padded) & set(w for w, _p, _n in rl_natural))
    if common_w:
        w0 = common_w[0]
        pad_p1 = next(p for w, p, _n in rl_padded if w == w0)
        nat_p1 = next(p for w, p, _n in rl_natural if w == w0)
        takeaway = (
            f"at W={P.fmt(w0, 'g')} written lines, rl solves {P.fmt(nat_p1, 'pct')} of natural "
            f"theorems but only {P.fmt(pad_p1, 'pct')} once those lines are padding."
        )
    else:
        takeaway = "padded vs. natural pass@1 at matched written length W."
    pad_types_list = sorted(pad_types)
    n_rl = sum(n for _w, _p, n in rl_padded)
    n_nat = next((n for w, _p, n in rl_natural if common_w and w == common_w[0]), None)
    caption = (
        "Pass@1 against written length W when some of the lines are padding (free lines) and on "
        f"natural theorems of the same length; {n_rl} distinct padded items pooled across pad "
        f"types {', '.join(pad_types_list)} "
        "(each unpadded theorem counted once); natural = gen_ctrl at the same W"
        + (f" ({n_nat} items at W={P.fmt(common_w[0], 'g')})" if n_nat else "")
        + ", drawn from the dev set the RL hill-climb selected on, so rl's natural curve is "
        "optimistic."
        + (
            " base solves no natural item at these lengths."
            if base_natural and all(p == 0 for _w, p, _n in base_natural)
            else ""
        )
        + " Point estimates."
    )
    return ChartData(
        key,
        title,
        "line",
        takeaway,
        caption,
        series=series,
        opts={"x_label": "W (written lines)", "y_fmt": "pct"},
    )


# ============================================================================= 7. Prefix completion


#: Lines-left bins with fewer sampled items than this are dropped before plotting (review: thin
#: bins -- e.g. 10 or 6 items -- cause a zig-zag that has nothing to do with lines left, such as a
#: jump from 0% at 4 lines left to 16.8% at 5).
_MIN_PREFIX_BIN_ITEMS = 20


def _data_7_prefix(store: Store) -> ChartData:
    key, title = "prefix", "Finishing someone else's proof (prefix completion)"
    data = prefix_view.view(store, {})
    if data.get("empty"):
        return _missing(key, title, str(data.get("reason", "no data")))
    models = cast("dict[str, dict[str, Any]]", data["models"])
    series = []
    n_items_by_model = {}
    kept_by_model: dict[str, dict[float, float]] = {}
    for model, d in models.items():
        by_ll = cast("dict[str, dict[str, Any]]", d["success_by_lines_left"])
        kept = {
            float(k): cast(float, v["pass1"])
            for k, v in by_ll.items()
            if cast(int, v["n_items"]) >= _MIN_PREFIX_BIN_ITEMS
        }
        kept_by_model[model] = kept
        if kept:
            series.append(P.Series(model, sorted(kept.items()), _mcolor(model)))
        n_items_by_model[model] = d.get("n_items", "?")
    if not series:
        return _missing(key, title, "no model had a lines-left bin with >=20 items")

    bits = []
    base_kept, rl_kept = kept_by_model.get("base", {}), kept_by_model.get("rl", {})
    if 1.0 in base_kept and 1.0 in rl_kept:
        bits.append(
            f"with 1 line left rl finishes {P.fmt(rl_kept[1.0], 'pct')} vs base "
            f"{P.fmt(base_kept[1.0], 'pct')}"
        )
    ll3 = next((ll for ll in sorted(set(base_kept) & set(rl_kept)) if ll >= 3), None)
    if ll3 is not None:
        bits.append(
            f"with {P.fmt(ll3, 'g')} lines left rl finishes {P.fmt(rl_kept[ll3], 'pct')} vs base "
            f"{P.fmt(base_kept[ll3], 'pct')}"
        )
    takeaway = "; ".join(bits) + "." if bits else "success by lines left to write, base vs rl."
    n_str = ", ".join(f"{m}={n}" for m, n in n_items_by_model.items())
    caption = (
        "Success rate at completing a correct proof prefix against the number of lines left to "
        f"write (L-j), base vs rl; items: {n_str}; pooled success rate by lines left (L-j), "
        f"bins with <{_MIN_PREFIX_BIN_ITEMS} items omitted. Point estimates."
    )
    return ChartData(
        key,
        title,
        "line",
        takeaway,
        caption,
        series=series,
        opts={"x_label": "lines left (L-j)", "y_fmt": "pct"},
    )


# ===================================================================================== 8. EDL


#: Plan §B7's class list, for a stable, meaningful chart order; any class the store doesn't have
#: (or has under a name not listed here) is simply appended afterwards, alphabetically.
_EDL_CLASS_ORDER: tuple[str, ...] = (
    "and_tower_n8",
    "and_tower_n16",
    "imp_nest_n4",
    "imp_nest_n8",
    "reorder_and_n5",
    "reorder_or_n3",
    "contraposition_s3",
    "dist_and_over_or_s1",
    "demorgan_nand_to_or_s1",
    "and_glue_L12",
    "dev_gen_L10",
    "gen_L10",
    "heldout_cap6",
)


def _edl_friendly_label(identity: object, run_id: str) -> str:
    """`analysis/edl.py:_model_label`'s own priority order, with its run-id prefix check widened
    from "equals edl" to "starts with edl" (module docstring) so this plan's `edl2-*` run ids
    resolve to `"base"`/`"rl"` instead of falling through to a raw checkpoint path."""
    if isinstance(identity, Mapping):
        label = identity.get("label")
        if isinstance(label, str) and label:
            return label
    parts = run_id.split("-")
    if (
        len(parts) >= 4
        and parts[0].startswith("edl")
        and parts[-1].isdigit()
        and parts[-2].isdigit()
    ):
        label = "-".join(parts[1:-2])
        if label:
            return label
    if isinstance(identity, Mapping):
        ckpt = identity.get("ckpt")
        if isinstance(ckpt, str) and ckpt:
            return ckpt
    return str(identity)


def _edl_class_key(cls: str) -> tuple[int, str]:
    try:
        return (_EDL_CLASS_ORDER.index(cls), cls)
    except ValueError:
        return (len(_EDL_CLASS_ORDER), cls)


#: The null control's EDL is expected to be ~0 (and noisy enough to land slightly negative, which
#: is undefined on the chart's log scale -- `dot_plot`'s `log_x` clamps it to the axis floor with
#: a "<= 0" label) -- it is reported but excluded from the "rl lower on N/M classes" count below,
#: since "lower EDL" has no "elicited a real capability" meaning for a class with nothing to learn.
_EDL_NULL_CONTROL = "heldout_cap6"


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    def ranks(v: Sequence[float]) -> list[float]:
        order = sorted(range(len(v)), key=lambda i: v[i])
        out = [0.0] * len(v)
        for r, i in enumerate(order):
            out[i] = float(r)
        return out

    if len(xs) < 3:
        return float("nan")
    rx, ry = ranks(xs), ranks(ys)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else float("nan")


def _data_8_edl(store: Store) -> ChartData:
    key, title = "edl", "Cost of learning a new pattern (EDL)"
    # Default discovery: the newest generation per (model, n, m) (edl3 = edl2's three learning
    # rates plus lr 0 / 1e-5 / 3e-5, 2026-10-01).
    data = edl_view.view(store, {})
    if data.get("empty"):
        return _missing(key, title, str(data.get("reason", "no data")))
    classes = cast("dict[str, dict[str, dict[str, Any]]]", data["classes"])
    run_ids = cast("list[str]", cast("dict[str, Any]", data["meta"])["run_ids"])

    per: dict[str, dict[str, dict[str, float]]] = {}
    n_lrs: set[int] = set()
    for cls, per_model in classes.items():
        for model, d in per_model.items():
            if model not in ("base", "rl"):
                continue
            n = int(cast(int, d.get("n_examples") or 0))
            if n <= 0:
                continue
            n_lrs.add(int(cast(int, d.get("n_lrs", 0))))
            orders = cast("dict[str, float]", d.get("edl_by_order_at_best_lr") or {})
            # The per-order values are plain EDL; add EDL*'s ln(#lrs) selection cost so the
            # whiskers bracket the plotted EDL* point.
            sel = math.log(max(int(cast(int, d.get("n_lrs", 1))), 1))
            per.setdefault(cls, {})[model] = {
                "edl": cast(float, d["edl_star"]) / n,
                "lo": (min(orders.values()) + sel) / n if orders else float("nan"),
                "hi": (max(orders.values()) + sel) / n if orders else float("nan"),
                "l0": cast(float, d.get("l_test_theta0", float("nan"))),
                "ln": cast(float, d.get("l_test_thetan", float("nan"))),
                "best_lr": float(cast(str, d["best_lr"])),
            }
    if not per:
        return _missing(key, title, "no usable EDL rows")

    dot_rows: list[P.DotRow] = []
    for cls in sorted(per, key=_edl_class_key):
        vals = per[cls]
        label = f"{cls} (null control)" if cls == _EDL_NULL_CONTROL else cls
        pts: list[tuple[str, float | None, str, bool]] = [
            (m, vals[m]["edl"], _mcolor(m), False) for m in ("base", "rl") if m in vals
        ]
        if pts:
            dot_rows.append(P.DotRow(label, pts))

    common = {c: v for c, v in per.items() if c != _EDL_NULL_CONTROL and "base" in v and "rl" in v}
    n_c = len(common)
    n_lower = sum(1 for v in common.values() if v["rl"]["edl"] < v["base"]["edl"])
    n_robust = sum(1 for v in common.values() if v["rl"]["hi"] < v["base"]["lo"])
    n_l0_lower = sum(1 for v in common.values() if v["rl"]["l0"] < v["base"]["l0"])
    pairs = [(v[m]["edl"], v[m]["l0"]) for v in common.values() for m in ("base", "rl")]
    rho = _spearman([e for e, _l in pairs], [l0 for _e, l0 in pairs])

    def _ratio(d: Mapping[str, float]) -> float:
        drop = d["l0"] - d["ln"]
        return d["edl"] / drop if drop > 1e-9 else float("inf")

    n_ratio = sum(1 for v in common.values() if _ratio(v["rl"]) >= _ratio(v["base"]))
    if n_c:
        takeaway = (
            f"rl's EDL is lower on {n_lower}/{n_c} classes ({n_robust} with no overlap between "
            f"the two data orders), but rl also starts with lower test loss on {n_l0_lower}/{n_c} "
            f"and EDL tracks the starting loss (Spearman {rho:.2f}); per nat of test loss that "
            f"training removes, rl costs as much as or more than base on {n_ratio}/{n_c} classes."
        )
    else:
        takeaway = "EDL per example, by class and model."
    null = per.get(_EDL_NULL_CONTROL, {})
    null_bits = ", ".join(
        f"{m} {null[m]['edl']:.3f} (lr {P.fmt(null[m]['best_lr'], 'g')})"
        for m in ("base", "rl")
        if m in null
    )
    caption = (
        "(a) Excess description length of learning each pattern class online, per example "
        f"(nats, log scale); {n_c} classes, base vs rl; EDL* per example at the learning rate "
        "with the smallest "
        f"online code length, including the ln({max(n_lrs) if n_lrs else 0}) cost of choosing "
        "among the learning rates (lr 0 = no update is one of them); 2 data orders, whiskers "
        "span the per-order values. (b) The same EDL* against each model's initial test loss on "
        "the class (mean nats per test example before any update), null control excluded. Null "
        f"control ({_EDL_NULL_CONTROL}, data the models already know): {null_bits or 'n/a'}. "
        f"Runs: {', '.join(run_ids)}."
    )
    cd = ChartData(
        key,
        title,
        "dot",
        takeaway,
        caption,
        rows=dot_rows,
        opts={"x_label": "EDL* per example (nats)", "log_x": True},
        extra={"per": per, "null_control": _EDL_NULL_CONTROL, "rho": rho},
    )
    cd.extra_tables.append(
        (
            "per-order range and initial test loss (whiskers, panel b)",
            ["class", "model", "EDL*/example", "order min", "order max", "initial test loss"],
            [
                [cls, m, *(P.fmt(per[cls][m][f], "f3") for f in ("edl", "lo", "hi", "l0"))]
                for cls in sorted(per, key=_edl_class_key)
                for m in ("base", "rl")
                if m in per[cls]
            ],
        )
    )
    return cd


# ================================================================================ 9. Likelihood


#: Row order and display labels for chart 9's dot plot (review: "one row per proof set: reserve
#: (neutral), family witnesses, base's own verified proofs, rl's crux proofs").
_LIKELIHOOD_ROW_ORDER: tuple[str, ...] = ("reserve", "family", "base_verified", "reference", "crux")
_LIKELIHOOD_ROW_LABEL: dict[str, str] = {
    "reserve": "reserve (neutral)",
    "family": "family witnesses",
    "base_verified": "base's own verified proofs",
    "reference": "neutral minimal proofs",
    "crux": "rl's own proofs",
}


def _likelihood_rows_by_origin(
    store: Store,
) -> tuple[dict[str, list[dict[str, JSON]]], dict[str, str]]:
    """Every `interp likelihood` row, each origin taken from the newest run that has it (the
    neutral-control run, 2026-10-01, adds `reference` rows and re-scores `rl_verified` with a
    per-row `lines` count; the main run keeps `reserve`/`family`/`base_verified`)."""
    by_origin: dict[str, list[dict[str, JSON]]] = {}
    run_of: dict[str, str] = {}
    seen: set[str] = set()
    for rec in reversed(interp_runs(store, "likelihood")):
        run_id = cast(str, rec["run_id"])
        if run_id in seen:
            continue
        seen.add(run_id)
        rows = load_interp_rows(store, "likelihood", run_id)
        # Only runs that score our base/rl pair: Dan's state-env runs (rows keyed fz/t1, same
        # origin names) belong to the Dan section and must not claim these origins.
        if not rows or not all("base" in r and "rl" in r for r in rows):
            continue
        origins = {cast(str, r["origin"]) for r in rows}
        for origin in origins - set(by_origin):
            by_origin[origin] = [r for r in rows if r["origin"] == origin]
            run_of[origin] = run_id
    return by_origin, run_of


def _likelihood_per_line(
    store: Store,
) -> tuple[dict[str, dict[str, list[float]]], dict[str, Any]] | None:
    """Median-ready per-line likelihoods under base and rl, by proof set, plus the ordering
    numbers. Each row is normalised by the lines of the proof actually scored (`lines`, recorded
    since 2026-10-01) when present, else by the theorem's certified `L` (the reserve/family rows
    are minimal proofs, where the two agree). `crux` = rl_verified rows with p_hat_base == 0;
    `reference` = neutral minimal proofs of the same theorems (each model prefers its own samples,
    so the crux row alone overstates the gap; interp audit finding 4)."""
    rows_by_origin, run_of = _likelihood_rows_by_origin(store)
    if not rows_by_origin:
        return None
    by_origin: dict[str, dict[str, list[float]]] = {}
    paired: dict[str, dict[str, float]] = {"crux": {}, "reference": {}}
    abs_diffs_per_line: list[float] = []
    single_order_reserve = 0
    total_reserve = 0
    for origin, rows in rows_by_origin.items():
        for r in rows:
            norm = cast("int | None", r.get("lines")) or cast("int | None", r.get("L"))
            if not norm:
                continue
            label = origin
            if origin == "rl_verified":
                p_hat = r.get("p_hat_base")
                if p_hat is None or float(cast(float, p_hat)) != 0.0:
                    continue
                label = "crux"
            base_side = cast("Mapping[str, JSON]", r["base"])
            rl_side = cast("Mapping[str, JSON]", r["rl"])
            base_lm, rl_lm = base_side.get("logp_marginal"), rl_side.get("logp_marginal")
            if base_lm is None or rl_lm is None:
                continue
            base_pl, rl_pl = float(cast(float, base_lm)) / norm, float(cast(float, rl_lm)) / norm
            by_origin.setdefault(label, {}).setdefault("base", []).append(base_pl)
            by_origin.setdefault(label, {}).setdefault("rl", []).append(rl_pl)
            if label in paired:
                paired[label][cast(str, r["pid"]).split(":", 1)[-1]] = rl_pl - base_pl
            base_single = base_side.get("logp_single")
            if base_single is not None:
                abs_diffs_per_line.append(
                    abs(float(cast(float, base_lm)) - float(cast(float, base_single))) / norm
                )
            if origin == "reserve":
                total_reserve += 1
                method, n_ord = base_side.get("method"), base_side.get("n_orderings")
                if method == "single" or n_ord in (None, 1):
                    single_order_reserve += 1
    if not by_origin:
        return None
    common = sorted(set(paired["crux"]) & set(paired["reference"]))
    meta = {
        "run_ids": sorted(set(run_of.values())),
        "frac_single_reserve": (single_order_reserve / total_reserve) if total_reserve else None,
        "median_abs_diff_per_line": (
            statistics.median(abs_diffs_per_line) if abs_diffs_per_line else None
        ),
        "crux_paired_gap_median": (
            statistics.median(paired["crux"].values()) if paired["crux"] else None
        ),
        "reference_paired_gap_median": (
            statistics.median(paired["reference"][p] for p in common) if common else None
        ),
        "n_paired": len(common),
    }
    return by_origin, meta


def _data_9_likelihood(store: Store) -> ChartData:
    key, title = "likelihood", "How likely base finds rl's proofs"
    result = _likelihood_per_line(store)
    if result is None:
        return _missing(key, title, "no `interp likelihood` run with resolvable lengths")
    by_origin, meta = result
    dot_rows: list[P.DotRow] = []
    for origin in _LIKELIHOOD_ROW_ORDER:
        d = by_origin.get(origin)
        if not d:
            continue
        n = max(len(v) for v in d.values())
        label = f"{_LIKELIHOOD_ROW_LABEL[origin]} (n={n})"
        pts: list[tuple[str, float | None, str, bool]] = [
            (model, statistics.median(d[model]), _mcolor(model), False)
            for model in ("base", "rl")
            if d.get(model)
        ]
        if pts:
            dot_rows.append(P.DotRow(label, pts))
    if not dot_rows:
        return _missing(key, title, "no origin had both base and rl per-line likelihoods")
    gap = cast("float | None", meta.get("crux_paired_gap_median"))
    ref_gap = cast("float | None", meta.get("reference_paired_gap_median"))
    if gap is not None and ref_gap is not None:
        takeaway = (
            f"on theorems base never solves, base assigns about {ref_gap:.1f} nats/line less than "
            f"rl even to neutral minimal proofs (about {gap:.1f} on rl's own proofs, which each "
            "model favours as its own samples)"
        )
    elif gap is not None:
        takeaway = f"on rl's crux proofs base assigns about {gap:.1f} nats/line less than rl"
    else:
        takeaway = "median base/rl log-likelihood per written line, by proof set"
    bv = by_origin.get("base_verified", {})
    if (
        bv.get("base")
        and bv.get("rl")
        and statistics.median(bv["rl"]) < statistics.median(bv["base"])
    ):
        takeaway += "; on base's own proofs rl is slightly less likely than base"
    takeaway += "."
    frac_single = cast("float | None", meta.get("frac_single_reserve"))
    diff_pl = cast("float | None", meta.get("median_abs_diff_per_line"))
    ordering_note = (
        f" Marginalising over valid line orders changes log p by {P.fmt(diff_pl, 'f3')} "
        f"nats/line in median; {P.fmt(frac_single, 'pct')} of reserve proofs have a single "
        "valid order."
        if frac_single is not None and diff_pl is not None
        else ""
    )
    caption = (
        "Median log-likelihood per written line (nats) of each proof set under base and rl: "
        "median log p per written line of the scored proof, base vs rl, by proof set; "
        f"runs {', '.join(cast('list[str]', meta['run_ids']))}. The last two rows (neutral minimal "
        f"proofs, rl's own proofs) are the same {meta['n_paired']} gen_ctrl theorems rl solves "
        "and base never does (p_hat_base=0); the minimal proofs come from the minlen labeller."
        + ordering_note
        + " Medians; no CI is drawn."
    )
    return ChartData(
        key,
        title,
        "dot",
        takeaway,
        caption,
        rows=dot_rows,
        opts={"x_label": "median log p per written line (nats)"},
    )


# =================================================================================== 10. Steering


def _data_10_steering(store: Store) -> ChartData:
    key, title = "steering", "Steering base along the rl - base activation difference"
    data = interp_view.view_steering(store, {})
    if data.get("empty"):
        return _missing(key, title, str(data.get("reason", "no data")))
    curves = cast("dict[str, list[tuple[float, float, float, float]]]", data["curves_by_layer"])
    meta = cast("dict[str, Any]", data["meta"])
    suites = cast("list[str]", meta["suites"])
    base_pass1 = cast("list[float] | None", data.get("base_pass1"))
    if not curves and base_pass1 is None:
        return _missing(key, title, "no base/steer:* runs with any items")

    # the single best (layer, alpha) point across the whole sweep, by mean pass@1 (the plan's
    # "the layer with the highest mean" read as the layer+alpha combination that peaks highest,
    # since a flat line-chart over every layer is exactly the "all flat at ~0%" presentation the
    # review flagged as hiding the real result).
    best: tuple[str, float, float, float, float] | None = None  # (layer, alpha, mean, lo, hi)
    for layer, pts in curves.items():
        for a, m, lo, hi in pts:
            if best is None or m > best[2]:
                best = (layer, a, m, lo, hi)

    rl_pass1 = interp_view._pooled_pass1_of(  # noqa: SLF001
        store,
        suites=suites,
        version=cast(str, meta["version"]),
        model="rl",
        B=int(cast(int, meta["B"])),
        seed=0,
    )
    rl_control = cast("dict[str, Any]", data.get("rl_control", {}))
    ctrl_layer, ctrl = next(iter(rl_control.items()), (None, None))

    bars: list[P.Bar] = []
    if base_pass1 is not None:
        bars.append(
            P.Bar("base (unsteered)", base_pass1[0], base_pass1[1], base_pass1[2], _mcolor("base"))
        )
    if best is not None:
        layer, alpha, m, lo, hi = best
        bars.append(
            P.Bar(f"base + delta (layer {layer}, a={P.fmt(alpha, 'g')})", m, lo, hi, "var(--s4)")
        )
    if ctrl is not None:
        ci = cast("list[float]", ctrl["ci"])
        bars.append(
            P.Bar(
                f"rl - delta (layer {ctrl_layer})",
                cast(float, ctrl["pass1"]),
                ci[0],
                ci[1],
                "var(--s5)",
            )
        )
    if rl_pass1 is not None:
        bars.append(P.Bar("rl", rl_pass1[0], rl_pass1[1], rl_pass1[2], _mcolor("rl")))
    if not bars:
        return _missing(key, title, "no base/rl/steer:* pass@1 could be computed")

    bits = []
    n_configs = sum(len(pts) for pts in curves.values())
    if base_pass1 is not None and best is not None:
        inside = best[2] <= base_pass1[2]
        bits.append(
            f"adding the RL-base direction to base {'does not move it' if inside else 'helps'}: "
            f"the best of {n_configs} (layer, alpha) settings scores "
            f"{P.fmt(best[2], 'pct')} vs {P.fmt(base_pass1[0], 'pct')} unsteered"
            + (" (inside base's 95% CI)" if inside else "")
        )
    if ctrl is not None and rl_pass1 is not None:
        diff_pts = (rl_pass1[0] - cast(float, ctrl["pass1"])) * 100
        bits.append(
            f"subtracting it from rl costs {diff_pts:.1f} points "
            f"({P.fmt(rl_pass1[0], 'pct')} -> {P.fmt(cast(float, ctrl['pass1']), 'pct')})"
        )
    takeaway = ", but ".join(bits) + "." if bits else "pass@1 for base, base+delta, rl-delta, rl."

    tdev_note = ""
    if "textbook_dev" in suites:
        tdev_pass1 = interp_view._pooled_pass1_of(  # noqa: SLF001
            store,
            suites=["textbook_dev"],
            version=cast(str, meta["version"]),
            model="base",
            B=int(cast(int, meta["B"])),
            seed=0,
        )
        if tdev_pass1 is not None:
            tdev_note = (
                f" interp_eval_dev/sealed dominate the pooled item count (base alone on "
                f"textbook_dev is {P.fmt(tdev_pass1[0], 'pct')})."
            )
    sealed_note = ""
    if "interp_eval_sealed" in suites and ctrl_layer is not None:
        sealed = {
            label: interp_view._pooled_pass1_of(  # noqa: SLF001
                store,
                suites=["interp_eval_sealed"],
                version=cast(str, meta["version"]),
                model=model,
                B=int(cast(int, meta["B"])),
                seed=0,
            )
            for label, model in (("rl", "rl"), ("rl - delta", f"steer:{ctrl_layer}:-1"))
        }
        if all(v is not None for v in sealed.values()):
            sealed_note = (
                " On the sealed set alone (dev L8-11 was the RL hill-climb's selection metric, so "
                "rl's dev numbers are optimistic): "
                + ", ".join(f"{k} {P.fmt(cast(Any, v)[0], 'pct')}" for k, v in sealed.items())
                + "."
            )
    caption = (
        f"Pass@1 of base, base + delta at the best of its {n_configs} (layer, alpha) settings, "
        "rl - delta and rl, with 95% item-level bootstrap CIs "
        f"(B={meta['B']}); pooled pass@1 over {', '.join(suites)}; delta (RL - base mean "
        "activation difference at proof positions) is added at proof positions only."
        + tdev_note
        + sealed_note
    )
    return ChartData(
        key,
        title,
        "bar",
        takeaway,
        caption,
        bars=bars,
        opts={"y_fmt": "pct"},
        as_table=True,
    )


# ============================================================================ 11. Weight interp.


def _data_11_weights(store: Store) -> ChartData:
    key, title = "weights", "Interpolating weights from base to rl"
    data = interp_view.view_interpolation(store, {})
    if data.get("empty"):
        return _missing(key, title, str(data.get("reason", "no data")))
    pass_k = cast("dict[str, list[tuple[int, float, float, float]]]", data["pass_k_by_alpha"])
    if not pass_k:
        return _missing(key, title, "no base/rl/interp:* runs on the fixed k=128 subset")

    # pass@1 AND pass@128 both read from the same fixed k=128 theorem subset (review: pass@1 from
    # the full dev+sealed+textbook_dev set and pass@128 from the k=128 subset can disagree about
    # which is bigger at the same alpha -- impossible for the same metric on the same items, as
    # the real store's alpha=0 point showed: pass@1 0.1% > pass@128 0.0%).
    curve_map: dict[float, tuple[float, float, float, float, float, float]] = {}
    for alpha_str, curve in pass_k.items():
        by_k = {k: (m, lo, hi) for k, m, lo, hi in curve}
        if 1 not in by_k or 128 not in by_k:
            continue
        m1, lo1, hi1 = by_k[1]
        m128, lo128, hi128 = by_k[128]
        if m1 != m1 or m128 != m128:  # nan
            continue
        curve_map[float(alpha_str)] = (m1, lo1, hi1, m128, lo128, hi128)
    if not curve_map:
        return _missing(key, title, "no alpha had both pass@1 and pass@128 on the k=128 subset")

    alphas = sorted(curve_map)
    series = [
        P.Series(
            "pass@1",
            [(a, curve_map[a][0]) for a in alphas],
            _mcolor("base"),
            band=[(a, curve_map[a][1], curve_map[a][2]) for a in alphas],
        ),
        P.Series(
            "pass@128",
            [(a, curve_map[a][3]) for a in alphas],
            _mcolor("rl"),
            muted=True,
            band=[(a, curve_map[a][4], curve_map[a][5]) for a in alphas],
        ),
    ]

    a0, a1 = alphas[0], min(alphas, key=lambda a: abs(a - 1.0))
    p1_0, p1_1 = curve_map[a0][0], curve_map[a1][0]
    step = max(
        zip(alphas, alphas[1:], strict=False),
        key=lambda ab: curve_map[ab[1]][0] - curve_map[ab[0]][0],
        default=None,
    )
    bits = [
        f"pass@1 rises from {P.fmt(p1_0, 'pct')} (alpha={P.fmt(a0, 'g')}) to "
        f"{P.fmt(p1_1, 'pct')} (alpha={P.fmt(a1, 'g')})"
    ]
    if step is not None:
        bits[0] += (
            f" with most of the gain between alpha={P.fmt(step[0], 'g')} and {P.fmt(step[1], 'g')}"
        )
    beyond = [a for a in alphas if a > a1]
    if beyond:
        a_mid = beyond[0]
        p1_mid, p128_mid = curve_map[a_mid][0], curve_map[a_mid][3]
        bits.append(
            f"extrapolating to alpha={P.fmt(a_mid, 'g')} drops pass@1 to {P.fmt(p1_mid, 'pct')} "
            f"but pass@128 stays at {P.fmt(p128_mid, 'pct')}"
        )
        if len(beyond) > 1:
            a_end = beyond[-1]
            bits.append(
                f"alpha={P.fmt(a_end, 'g')} is {P.fmt(curve_map[a_end][0], 'pct')} pass@1 / "
                f"{P.fmt(curve_map[a_end][3], 'pct')} pass@128"
            )
    takeaway = "; ".join(bits) + "."
    sealed_bits: list[str] = []
    for alpha in sorted(float(a) for a in pass_k):
        model = "base" if alpha == 0.0 else "rl" if alpha == 1.0 else f"interp:{alpha:g}"
        sp = interp_view._pooled_pass1_of(  # noqa: SLF001
            store, suites=["interp_eval_sealed"], version="v1", model=model, B=200, seed=0
        )
        if sp is not None:
            sealed_bits.append(f"alpha {alpha:g} {P.fmt(sp[0], 'pct')}")
    caption = (
        "Pass@1 and pass@128 of θ_base + α(θ_rl − θ_base) against α "
        "(α > 1 extrapolates past rl) on a "
        "fixed 150-theorem subset (dev L8-11 + sealed L12-14 gen items), pass@1 and pass@128 on "
        "the same items; base=alpha 0, rl=alpha 1. Shaded bands: 95% item-level bootstrap CIs "
        f"(B={cast('dict[str, Any]', data['meta'])['B']})."
        + (
            " dev L8-11 was the RL hill-climb's selection metric; pass@1 on the sealed set alone: "
            + ", ".join(sealed_bits)
            + "."
            if sealed_bits
            else ""
        )
    )
    return ChartData(
        key,
        title,
        "line",
        takeaway,
        caption,
        series=series,
        opts={"x_label": "alpha (0=base, 1=rl)", "y_fmt": "pct"},
    )


# =================================================================================== 12. Probes


def _probe_baselines(store: Store) -> dict[str, Any] | None:
    run_id = latest_interp_run(store, "probe_baselines")
    rows = load_interp_rows(store, "probe_baselines", run_id) if run_id else []
    return cast("dict[str, Any]", rows[-1]) if rows else None


def _data_12_probes(store: Store) -> ChartData:
    key, title = "probes", "What linear probes read from the activations"
    data = interp_view.view_probes(store, {})
    if data.get("empty"):
        return _missing(key, title, str(data.get("reason", "no data")))
    meta = cast("dict[str, Any]", data["meta"])
    layers = cast("list[str]", meta["layers"])
    acc = cast("dict[str, list[tuple[int, JSON]]]", data["next_rule_acc"])
    r2 = cast("dict[str, list[tuple[int, JSON]]]", data["lines_to_go_r2"])

    def _clean(pts: list[tuple[int, JSON]]) -> list[tuple[float, float]]:
        return [(float(i), cast(float, v)) for i, v in pts if isinstance(v, int | float)]

    series = []
    for model in ("base", "rl"):
        a = _clean(acc.get(model, []))
        if a:
            series.append(P.Series(f"next_rule probe: {model}", a, _mcolor(model)))
    if not series:
        return _missing(key, title, "probes run has rows but no usable accuracy series")
    # Interp audit 2026-10-01: the emb-layer probe equals a line-index lookup, so accuracy needs a
    # position-only reference; line index + premise count is everything the prompt gives away.
    baselines = _probe_baselines(store)
    base_acc = cast("float | None", (baselines or {}).get("next_rule_acc_line_index_premises"))
    xs = [x for s_ in series for x, _y in s_.pts]
    if base_acc is not None and xs:
        series.append(
            P.Series(
                "position-only lookup (line index + premises)",
                [(min(xs), base_acc), (max(xs), base_acc)],
                "var(--ink2)",
                muted=True,
            )
        )

    def _last(series_dict: dict[str, list[tuple[int, JSON]]], name: str) -> float | None:
        pts = _clean(series_dict.get(name, []))
        return pts[-1][1] if pts else None

    b_last, r_last = _last(acc, "base"), _last(acc, "rl")
    pairs = list(zip(_clean(acc.get("base", [])), _clean(acc.get("rl", [])), strict=False))
    max_gap = max((abs(b[1] - r[1]) for b, r in pairs), default=None)
    if b_last is not None and r_last is not None and base_acc is not None and max_gap is not None:
        takeaway = (
            f"at the deepest layer the next_rule probe reaches {P.fmt(b_last, 'pct')} (base) and "
            f"{P.fmt(r_last, 'pct')} (rl), {100 * (b_last - base_acc):.1f} and "
            f"{100 * (r_last - base_acc):.1f} points above a position-only lookup "
            f"({P.fmt(base_acc, 'pct')}); base and rl stay within {100 * max_gap:.1f} points of "
            "each other at every layer."
        )
    else:
        takeaway = "next_rule probe accuracy by layer, base vs rl."
    ood = cast("dict[str, list[tuple[int, JSON]]]", data.get("length_generalisation_ood_r2", {}))
    r2_b, r2_r = _last(r2, "base"), _last(r2, "rl")
    ltg_base = cast("float | None", (baselines or {}).get("lines_to_go_r2_line_index"))
    ood_b, ood_r = _last(ood, "base"), _last(ood, "rl")
    caption = (
        "Held-out accuracy of a linear next_rule probe at each layer for base and rl, against a "
        "position-only lookup (line index + premise count); "
        f"layers {', '.join(layers)}; next_rule = the rule of the line about to be written. "
        f"lines_to_go R2 at the deepest layer: base {P.fmt(r2_b, 'f3')}, rl {P.fmt(r2_r, 'f3')} "
        f"vs {P.fmt(ltg_base, 'f3')} from the line index alone; on longer (OOD) proofs it is "
        f"negative for both (base {P.fmt(ood_b, 'f3')}, rl {P.fmt(ood_r, 'f3')}). Majority-class "
        f"next_rule accuracy: {P.fmt((baselines or {}).get('next_rule_acc_majority'), 'pct')}."
        " Point estimates on one held-out split."
    )
    display_labels = ["emb" if layer == "emb" else f"L{layer}" for layer in layers]
    return ChartData(
        key,
        title,
        "line",
        takeaway,
        caption,
        series=series,
        opts={"x_label": "layer", "y_fmt": "pct", "x_tick_labels": display_labels},
    )


# ======================================================================================== page

#: The 12 benchmarks in page order. Each `_data_*` is the single place its numbers are computed.
CHARTS: tuple[Callable[[Store], ChartData], ...] = (
    _data_1_yue,
    _data_2_families,
    _data_3_glue,
    _data_4_skills,
    _data_5_subst,
    _data_6_padding,
    _data_7_prefix,
    _data_8_edl,
    _data_9_likelihood,
    _data_10_steering,
    _data_11_weights,
    _data_12_probes,
)

#: Results-section layout: (heading, intro, keys). Consecutive half-width figures (and a table
#: next to a half-width figure) are set side by side, as subfigures would be in the paper.
SECTIONS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "Eliciting proofs",
        "Pass rates on real and generated theorems: sampling budget, length within a family, "
        "composing solved parts, required skills, larger substitutions, padding and finishing a "
        "given prefix.",
        ("yue", "families", "glue", "skills", "subst", "padding", "prefix"),
    ),
    (
        "Learning cost and internals",
        "How cheaply each model learns a new pattern, how likely it finds the other's proofs, "
        "and whether the change is visible in activations and weights.",
        ("edl", "likelihood", "steering", "weights", "probes"),
    ),
)

_LEDE = (
    "Twelve benchmarks ask what RL changed in a small natural-deduction prover beyond pass@1 on a "
    "fixed set: whether it solves real textbook problems at any sampling budget, holds up as "
    "proofs in a family get longer, composes parts it can solve, applies known laws to larger "
    "instances, copes with padding and finishes partial proofs; and whether the change shows up "
    "in the cost of learning a new pattern (EDL), in likelihoods, under steering, along a weight "
    "interpolation and in linear probes."
)

_MODELS_NOTE = (
    "base = pretrained model, trained on proofs up to 6 lines; rl = run-2 RL champion "
    "(base fine-tuned by RL on the full training pool); armC = RL model from the length "
    "experiment, trained on lengths 6-11 only. All results below are single-checkpoint, "
    "single-seed."
)

#: Caveats that apply across figures (from the audited write-up,
#: experiment-summaries/2026-09-30-benchmarking/README.md#limitations-and-next-step).
_LIMITATIONS: tuple[str, ...] = (
    "One checkpoint per model and one sampling seed; several item sets are small (textbook_dev, "
    "many glue and family cells), so differences of a few points are within noise.",
    "dev L8-11 (gen_ctrl, interp_eval_dev, part of the k=128 subset) was the RL hill-climb's "
    "selection metric, so rl's numbers on it are optimistic; the sealed set is the clean "
    "comparison (quoted in the captions where it exists).",
    "skill_pool results are legacy imports judged by the legacy pipeline (re-judging all "
    "benchmark samples with nd_v2 found no mismatch).",
    "EDL uses two data orders, so its only uncertainty estimate is the spread between them; the "
    "families, laws, padding, prefix, likelihood and probe figures show point estimates.",
)

#: CSS px per figure inch on the page: every figure is shown at the same scale, so 8 pt figure
#: text reads at about 0.8x the 16 px body text, as it does against 10 pt text in the paper.
PX_PER_IN = 120


def chart_data(store: Store) -> list[ChartData]:
    """All 12 benchmarks' `ChartData`, in page order (shared by the page and `figures.build`)."""
    return [chart(store) for chart in CHARTS]


def numbering(charts: Sequence[ChartData]) -> dict[str, str]:
    """`key -> "Figure N"/"Table N"` in page order; Figure 1 is the overview (`"overview"`)."""
    out = {"overview": "Figure 1"}
    n_fig, n_tab = 1, 0
    order = [k for _h, _i, keys in SECTIONS for k in keys]
    by = {c.key: c for c in charts}
    for key in order + [c.key for c in charts if c.key not in order]:
        cd = by.get(key)
        if cd is None:
            continue
        if cd.as_table and cd.kind != "missing":
            n_tab += 1
            out[key] = f"Table {n_tab}"
        else:
            n_fig += 1
            out[key] = f"Figure {n_fig}"
    return out


def _sentence(text: str) -> str:
    """Sentence case for a computed takeaway, leaving a leading model name (base, rl, armC,
    rl's) or identifier as written."""
    text = text.strip()
    if not text:
        return text
    first = text.split(" ", 1)[0].removesuffix("'s")
    if first in {"base", "rl", "armC"} or "_" in first:
        return text
    return text[0].upper() + text[1:]


def caption_text(cd: ChartData, label: str | None) -> str:
    """The plain-text NeurIPS caption: "Figure N: Title. What is plotted, n, CIs. Takeaway."."""
    head = f"{label}: " if label else ""
    if cd.kind == "missing":
        return f"{head}{cd.title}. Not available: {cd.missing}."
    return f"{head}{cd.title}. {cd.caption} {_sentence(cd.takeaway)}".strip()


def _caption_html(cd: ChartData, label: str | None) -> str:
    lab = f'<span class="fignum">{P.esc(label)}:</span> ' if label else ""
    if cd.kind == "missing":
        return f"{lab}<b>{P.esc(cd.title)}.</b> {P.esc(cd.missing)}"
    return (
        f"{lab}<b>{P.esc(cd.title)}.</b> {P.esc(cd.caption)} "
        f'<span class="takeaway">{P.esc(_sentence(cd.takeaway))}</span>'
    )


def _details_table(summary: str, headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    th = "".join(f"<th>{P.esc(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{P.esc(c)}</td>" for c in r) + "</tr>" for r in rows)
    return (
        f'<details class="data"><summary>{P.esc(summary)}</summary><div class="tw">'
        f"<table><tr>{th}</tr>{body}</table></div></details>"
    )


def data_tables(cd: ChartData) -> list[tuple[str, list[str], list[list[str]]]]:
    """The exact values behind a figure: the main table (the same rows the earlier inline chart
    carried) and one per extra panel."""
    o = cd.opts
    out: list[tuple[str, list[str], list[list[str]]]] = []
    if cd.kind == "line":
        hdr, rows = P.line_table_rows(
            cd.series,
            x_label=o.get("x_label", "x"),
            x_fmt=o.get("x_fmt", "g"),
            y_fmt=o.get("y_fmt", "g"),
        )
        out.append(("Data table", hdr, rows))
    elif cd.kind == "bar":
        hdr, rows = P.bar_table_rows(cd.bars, y_fmt=o.get("y_fmt", "g"))
        out.append(("Data table", hdr, rows))
    elif cd.kind == "dot":
        hdr, rows = P.dot_table_rows(
            cd.rows,
            x_fmt=o.get("x_fmt", "g"),
            log_x=bool(o.get("log_x", False)),
            none_label=o.get("none_label", "none"),
        )
        out.append(("Data table", hdr, rows))
    out += [(f"Data table: {label}", h, r) for label, h, r in cd.extra_tables]
    return out


def render_card(
    cd: ChartData, label: str | None = None, image: tuple[str, float] | None = None
) -> str:
    """One numbered figure (or table) card: the figure, its caption below it (a table's caption
    goes above, per NeurIPS), then the collapsible data tables. `image` is `(svg data URI, width
    in inches)`; rendered here when not given."""
    from ..figures import plots, tables

    fid = f"{'tab' if cd.as_table else 'fig'}-{cd.key}"
    if cd.kind == "missing":
        return (
            f'<figure class="card chart-card" id="{fid}"><div class="missing">no data yet</div>'
            f"<figcaption>{_caption_html(cd, label)}</figcaption></figure>"
        )
    details = "".join(_details_table(s, h, r) for s, h, r in data_tables(cd))
    if cd.as_table:
        return (
            f'<figure class="card chart-card table-card" id="{fid}">'
            f'<figcaption class="above">{_caption_html(cd, label)}</figcaption>'
            f'<div class="tw">{tables.to_html(cd)}</div>{details}</figure>'
        )
    image = image or plots.figure_uri(cd)
    if image is None:
        return render_card(ChartData(cd.key, cd.title, missing="no figure"), label)
    alt = f"{label or 'Figure'}: {cd.title}"
    pic = _picture(image, plots.figure_uri(cd, narrow=True), alt)
    return (
        f'<figure class="card chart-card" id="{fid}"><div class="figbox">{pic}</div>'
        f"<figcaption>{_caption_html(cd, label)}</figcaption>{details}</figure>"
    )


def _picture(image: tuple[str, float], narrow: tuple[str, float] | None, alt: str) -> str:
    """The figure as an `<img>` at the page's fixed scale; with a narrow (stacked) variant, a
    `<picture>` that serves it below 600 px instead of shrinking the wide layout."""
    uri, width_in = image
    img = (
        f'<img class="fig" src="{uri}" alt="{P.esc(alt)}" '
        f'style="width:{width_in * PX_PER_IN:.0f}px">'
    )
    if narrow is None:
        return img
    return f'<picture><source media="(max-width: 600px)" srcset="{narrow[0]}">{img}</picture>'


OVERVIEW_TITLE = "Overview of the twelve benchmarks"


def overview_caption(charts: Sequence[ChartData], nums: Mapping[str, str]) -> str:
    """Figure 1's caption: which later figure each panel summarises."""

    def ref(key: str, sub: str = "") -> str:
        return nums.get(key, "?") + sub

    return (
        "Each panel is one benchmark's headline comparison in "
        "the fixed model colours and markers; its full figure gives n, the CI and the takeaway. "
        f"(a) Pass@k on textbook_dev, log k ({ref('yue', 'a')}). (b) Longest horizon any family "
        f"reaches on unseen theorems, in written lines; >= marks a censored horizon "
        f"({ref('families')}). (c) Compositionality ratio R per glue operator; dotted line R = 1 "
        f"({ref('glue')}). (d) Pass@1 by required skill on the items base and rl share "
        f"({ref('skills', 'b')}). (e) Pass@1 against substitution size s ({ref('subst')}). "
        f"(f) Pass@1 against written length with padding ({ref('padding')}). (g) Prefix "
        f"completion against lines left ({ref('prefix')}). (h) rl's EDL* per example against "
        "base's, one point per class, log-log; points below the diagonal are classes where rl's "
        f"is lower ({ref('edl')}). (i) Median log p per written line of each proof set; * marks "
        f"theorems rl solves and base never does ({ref('likelihood')}). (j) Pass@1 of base, "
        f"base + delta, rl - delta and rl under steering ({ref('steering')}). (k) Pass@1 (solid) "
        f"and pass@128 (dashed) along the base-to-rl weight interpolation ({ref('weights')}). "
        f"(l) next_rule probe accuracy by layer; dashed line: position-only lookup "
        f"({ref('probes')})."
    )


def _overview_card(charts: Sequence[ChartData], nums: Mapping[str, str]) -> str:
    from ..figures import plots

    image = plots.overview_uri(charts)
    if image is None:
        return ""
    pic = _picture(
        image,
        plots.overview_uri(charts, narrow=True),
        "Figure 1: overview of the twelve benchmarks",
    )
    return (
        f'<figure class="card overview-card" id="fig-overview"><div class="figbox">{pic}</div>'
        f'<figcaption><span class="fignum">Figure 1:</span> <b>{P.esc(OVERVIEW_TITLE)}.</b> '
        f"{P.esc(overview_caption(charts, nums))}</figcaption></figure>"
    )


def _half(cd: ChartData) -> bool:
    from ..figures import plots

    if cd.kind == "missing":
        return False
    if cd.as_table:
        return True
    # `DAN_FIGURES` (WP-D5's "State-env models" section, a separate registry from the paper's
    # own `FIGURES` -- see `figures/plots.py:render`'s docstring) is checked as a fallback only:
    # every existing (non-Dan) key keeps resolving through `FIGURES` alone, unaffected.
    spec = plots.FIGURES.get(cd.key) or plots.DAN_FIGURES.get(cd.key)
    return spec is not None and spec[1] < plots.S.FULL_W - 0.5


def _section_html(cards: Sequence[tuple[ChartData, str]]) -> str:
    """Pairs consecutive half-width items into a two-column row; a lone half-width item is
    centred."""
    out: list[str] = []
    i = 0
    while i < len(cards):
        cd, html_ = cards[i]
        if _half(cd) and i + 1 < len(cards) and _half(cards[i + 1][0]):
            out.append(f'<div class="pair">{html_}{cards[i + 1][1]}</div>')
            i += 2
            continue
        out.append(f'<div class="{"single half" if _half(cd) else "single"}">{html_}</div>')
        i += 1
    return "".join(out)


_CSS = """
:root{color-scheme:light;--page:#fbfbf9;--card:#ffffff;--ink:#111;--ink2:#4a4a46;--muted:#77756f;
--rule:#1a1a1a;--hair:#dcdbd5;--link:#1a4f8a;--paper:#ffffff}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;
--page:#121212;--card:#1b1b1a;--ink:#ecebe6;--ink2:#c4c2ba;--muted:#97958e;--rule:#ecebe6;
--hair:#3a3a37;--link:#8db8ec;--paper:#ffffff}}
:root[data-theme="dark"]{color-scheme:dark;--page:#121212;--card:#1b1b1a;--ink:#ecebe6;
--ink2:#c4c2ba;--muted:#97958e;--rule:#ecebe6;--hair:#3a3a37;--link:#8db8ec;--paper:#ffffff}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);
font:16px/1.5 "Nimbus Roman","TeX Gyre Termes","Times New Roman",Times,"STIX Two Text",serif;
-webkit-text-size-adjust:100%}
a{color:var(--link)}
.col{max-width:720px;margin:0 auto;padding:0 16px}
header{padding:28px 0 6px}
h1{font-size:1.65rem;line-height:1.2;margin:0 0 10px;font-weight:700}
h2{font-size:1.2rem;margin:30px 0 4px;font-weight:700}
h2 .secnum{margin-right:.5em}
.lede{margin:0 0 8px}
.models{color:var(--ink2);font-size:.94rem;margin:0}
.intro{color:var(--ink2);margin:0 0 10px;font-size:.96rem}
.card{margin:14px 0;min-width:0}
.figbox{background:var(--paper);border-radius:4px;padding:6px 0;text-align:center;
overflow:hidden}
img.fig{max-width:100%;height:auto;display:block;margin:0 auto;cursor:zoom-in}
figcaption{font-size:.92rem;line-height:1.42;margin:8px 0 0;text-align:left}
figcaption.above{margin:0 0 8px}
.fignum{font-weight:400}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:0 22px;align-items:start}
.single.half{max-width:330px;margin:0 auto}
.missing{color:var(--muted);border:1px dashed var(--hair);border-radius:6px;padding:26px;
text-align:center;background:var(--card)}
details.data{margin-top:6px;font-size:.84rem}
details.data summary{cursor:pointer;color:var(--ink2)}
.tw{overflow-x:auto;max-width:100%}
details.data table{border-collapse:collapse;font-variant-numeric:tabular-nums;margin:4px 0;
font-size:.82rem}
details.data td,details.data th{border-bottom:1px solid var(--hair);padding:2px 8px;
text-align:right;white-space:nowrap}
details.data th{color:var(--ink2);font-weight:600}
details.data td:first-child,details.data th:first-child{text-align:left}
table.booktabs{border-collapse:collapse;margin:2px auto 4px;font-variant-numeric:tabular-nums;
border-top:1.4px solid var(--rule);border-bottom:1.4px solid var(--rule);font-size:.95rem}
table.booktabs th{border-bottom:.8px solid var(--rule);font-weight:400;padding:4px 10px;
text-align:right}
table.booktabs td{padding:3px 10px;text-align:right;white-space:nowrap}
table.booktabs th:first-child,table.booktabs td:first-child{text-align:left;white-space:normal}
table.booktabs tbody tr:first-child td{padding-top:5px}
.pm{display:inline-flex;flex-direction:column;vertical-align:middle;font-size:.68em;
line-height:1.05;margin-left:2px;text-align:left}
.pm sup,.pm sub{vertical-align:baseline;position:static}
.limits{border-top:1px solid var(--hair);margin-top:30px;padding-top:2px}
.limits ul{margin:4px 0 0;padding-left:1.2em}
.limits li{margin:2px 0;font-size:.94rem}
footer{color:var(--ink2);font-size:.88rem;padding:18px 0 34px}
@media (max-width:700px){.pair{grid-template-columns:1fr}.single.half{max-width:none}
body{font-size:15.5px}h1{font-size:1.4rem}}
"""

#: Clicking a figure opens its SVG on its own (pinch-zoom on a phone); figures are data URIs, so
#: the page stays a single self-contained file.
_JS = (
    "document.querySelectorAll('img.fig').forEach(function(i){i.addEventListener('click',"
    "function(){fetch(i.currentSrc||i.src).then(function(r){return r.blob()}).then(function(b){"
    "window.open(URL.createObjectURL(b),'_blank')})})});"
)


def render_page(
    charts: Sequence[ChartData],
    *,
    gpu_hours: float,
    detail_href: str | None = None,
    dan_section: str = "",
) -> str:
    """The summary page's HTML from already-computed `charts` (see `build_summary`).

    `dan_section` (default `""`, so an existing call renders byte-identically -- the golden
    regression, `test_paper_figures.py`, pins this page) is WP-D5's "State-env models" section
    HTML (`dash.dan_pages.render_dan_section`), inserted after the twelve numbered benchmarks and
    before Limitations, behind `cli.py`'s `dash --summary --with-dan` flag.
    """
    by = {c.key: c for c in charts}
    nums = numbering(charts)
    sections = []
    for i, (heading, intro, keys) in enumerate(SECTIONS, start=1):
        cards = [(by[k], render_card(by[k], nums.get(k))) for k in keys if k in by]
        sections.append(
            f'<section><h2><span class="secnum">{i}</span>{P.esc(heading)}</h2>'
            f'<p class="intro">{P.esc(intro)}</p>{_section_html(cards)}</section>'
        )
    overview = _overview_card(charts, nums)
    limits = "".join(f"<li>{P.esc(t)}</li>" for t in _LIMITATIONS)
    detail = (
        f' &middot; <a href="{P.esc(detail_href)}">detail pages (every chart)</a>'
        if detail_href
        else ""
    )
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>ND Capability Benchmarks</title><style>{_CSS}</style></head><body>"
        '<div class="col"><header><h1>Capability benchmarks for the base and RL provers</h1>'
        f'<p class="lede">{P.esc(_LEDE)}</p><p class="models">{P.esc(_MODELS_NOTE)}</p></header>'
        f"<main>{overview}{''.join(sections)}{dan_section}"
        f'<section class="limits"><h2>Limitations</h2><ul>{limits}</ul></section></main>'
        f"<footer>GPU-hours used (ledger total): {gpu_hours:.2f} h{detail}</footer></div>"
        f"<script>{_JS}</script></body></html>"
    )


def build_summary(
    store: Store, out_dir: Path, detail_href: str | None = None, *, dan_section: str = ""
) -> Path:
    """Render the single-page summary (a paper-style results section: Figure 1 overview, one
    numbered figure or table per benchmark, limitations) into `out_dir/summary.html` and return
    its path. `out_dir` is created if missing.

    `dan_section` (default `""`, so an existing call's output is unaffected) is pre-rendered HTML
    for WP-D5's "State-env models" section (`dash.dan_pages.render_dan_section`); this module has
    no dependency on `dash.dan_pages` itself (which imports *this* module to reuse `render_card`
    -- `dash.dan_pages.build_summary_with_dan` is the `--with-dan` entry point that computes this
    string and also writes the standalone `out_dir/dan.html` detail page, plan §6).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    html = render_page(
        chart_data(store),
        gpu_hours=store.ledger_total_s() / 3600.0,
        detail_href=detail_href,
        dan_section=dan_section,
    )
    path = out_dir / "summary.html"
    path.write_text(html)
    return path


# Card-level entry points (one per benchmark), kept for callers and tests that render a single
# benchmark's card.
def _card_of(fn: Callable[[Store], ChartData]) -> Callable[[Store], str]:
    def card(store: Store) -> str:
        return render_card(fn(store))

    card.__name__ = fn.__name__.replace("_data_", "_chart_")
    card.__doc__ = f"The summary card of `{fn.__name__}`."
    return card


_chart_1_yue = _card_of(_data_1_yue)
_chart_2_families = _card_of(_data_2_families)
_chart_3_glue = _card_of(_data_3_glue)
_chart_4_skills = _card_of(_data_4_skills)
_chart_5_subst = _card_of(_data_5_subst)
_chart_6_padding = _card_of(_data_6_padding)
_chart_7_prefix = _card_of(_data_7_prefix)
_chart_8_edl = _card_of(_data_8_edl)
_chart_9_likelihood = _card_of(_data_9_likelihood)
_chart_10_steering = _card_of(_data_10_steering)
_chart_11_weights = _card_of(_data_11_weights)
_chart_12_probes = _card_of(_data_12_probes)
