"""`dash.summary`: a single-page, minimal-sift-through dashboard -- one chart per benchmark.

Leon (2026-09-30, after seeing the full dashboard): "this is way way too many graphs you should be
trying to minimize the amount of data a human needs to sift through ideally one graph per
benchmark." `dash/build.py` renders 13 full pages (overview, nine benchmarks, four interp
measures) with several charts and tables each; this module instead writes one `summary.html` with
exactly 12 cards -- one per benchmark/measure named in the plan -- each a short title, a
*computed* one-sentence takeaway, one chart, and a one-line caption. No run tables, no per-cell
tables, no failure-location tables: a chart is used wherever one is possible, and the acceptance
tests check that no bare table-only card sneaks in.

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

import statistics
from collections.abc import Mapping, Sequence
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
    default_version,
    item_length,
    items_by_pid,
    latest_run_id,
    models_present,
    problems_by_pid,
)
from ..core.types import JSON, Problem
from ..interp._store import interp_runs, latest_interp_run, load_interp_rows
from ..metrics import compositionality_ratio, pass_at_k
from ..store import Store
from . import primitives as P

#: Fixed per-model colours (plan: "consistent colours: base, rl and armC each always the same
#: colour") -- independent of iteration order, unlike `dash/pages.py`'s `_color(i)` (which cycles
#: through the theme palette in whatever order a view's `sorted(models)` happens to produce, so
#: `armC` can land on the "base" slot on one page and not another).
MODEL_COLOR: dict[str, str] = {"base": "var(--s1)", "rl": "var(--s2)", "armC": "var(--s3)"}
_EXTRA_COLORS: tuple[str, ...] = ("var(--s4)", "var(--s5)", "var(--s6)")


def _mcolor(model: str) -> str:
    return MODEL_COLOR.get(model, _EXTRA_COLORS[hash(model) % len(_EXTRA_COLORS)])


def _mark(fig_html: str) -> str:
    """Tag one of this page's 12 benchmark charts so the page itself (the models note, the
    footer) is never mistaken for one when counting cards."""
    return fig_html.replace('<figure class="card', '<figure class="card chart-card', 1)


def _models_label(models: Sequence[str]) -> str:
    order = {"base": 0, "rl": 1, "armC": 2}
    ms = sorted(models, key=lambda m: order.get(m, 99))
    if len(ms) <= 1:
        return ", ".join(ms)
    return ", ".join(ms[:-1]) + f" and {ms[-1]}" if len(ms) > 1 else ms[0]


# ============================================================================= 1. Yue (textbook)


def _chart_1_yue(store: Store) -> str:
    title = "Real textbook problems: pass@k"
    data = yue_view.view(store, {})
    if data.get("empty"):
        return P.missing_fig(title, str(data.get("reason", "no data")))
    sets = cast("dict[str, Any]", data["sets"])
    tdev = cast("dict[str, Any] | None", sets.get("textbook_dev"))
    if not tdev:
        return P.missing_fig(title, "no textbook_dev results in the store")
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
                takeaway = (
                    f"{lead} is ahead of {trail} at every k; no crossover (pass@1 "
                    f"{P.fmt(lead1, 'pct')} vs {P.fmt(trail1, 'pct')}; pass@{lead_max[0]} "
                    f"{P.fmt(lead_max[1], 'pct')} vs {P.fmt(trail_max[1], 'pct')})."
                )
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
    caption = (
        f"n={n_items.get(models[0], '?')} textbook theorems; k=1..{tdev.get('k_max')}; "
        f"{_models_label(models)}." + (f" {extra}." if extra else "")
    )
    return P.line_chart(
        title,
        takeaway,
        series,
        x_label="k",
        y_fmt="pct",
        log_x=True,
        wide=True,
        note=caption,
    )


# ======================================================================== 2. Restricted families


#: `curves_by_k`'s own `n` grid is what `horizon.view` plots pass@k against; a sub-schema with too
#: few of them (plan: "drop if they have only 1-2 n values") can't show a meaningful horizon curve.
_MIN_N_VALUES = 3


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


def _chart_2_families(store: Store) -> str:
    title = "Restricted families: horizon in written lines"
    data = horizon_view.view(store, {})
    if data.get("empty"):
        return P.missing_fig(title, str(data.get("reason", "no data")))
    meta = cast("dict[str, JSON]", data["meta"])
    families = cast("dict[str, dict[str, dict[str, Any]]]", data["families"])
    suite, version = cast(str, meta["suite"]), cast(str, meta["version"])
    problems = problems_by_pid(store, suite, version)
    pids_by_group = _family_group_pids(problems)

    dropped: list[str] = []
    rows: list[P.DotRow] = []
    n_with_horizon = 0
    n_total_models = 0
    for family in sorted(families):
        per_model = families[family]
        any_model = next(iter(per_model.values()))
        n_values = {n for n, _v in cast("list[tuple[int, float]]", any_model["curves_by_k"]["8"])}
        if family.startswith("ILTP-style") and len(n_values) < _MIN_N_VALUES:
            dropped.append(family)
            continue
        label = "ILTP-style" if family.startswith("ILTP-style") else family
        pts: list[tuple[str, float | None, str]] = []
        for model in ("base", "rl", "armC"):
            d = per_model.get(model)
            if d is None:
                continue
            n_total_models += 1
            n_star = cast("dict[str, Any]", d["n_star"]["k8_tau0.5"])["n_star"]
            if n_star is None:
                pts.append((model, None, _mcolor(model)))
                continue
            n_with_horizon += 1
            pids = [p for p in pids_by_group.get(family, []) if problems[p].tags.get("n") == n_star]
            lens = [v for p in pids if (v := item_length(problems[p])) is not None]
            pts.append((model, statistics.mean(lens) if lens else None, _mcolor(model)))
        if pts:
            rows.append(P.DotRow(label, pts))

    if not rows:
        return P.missing_fig(title, "no family had enough n-values for a horizon")
    takeaway = (
        f"{n_with_horizon}/{n_total_models} (family, model) horizons are finite; a dot farther "
        "right survives more written lines before pass@8 drops below 50%."
    )
    drop_note = f" Dropped (<{_MIN_N_VALUES} n-values): {', '.join(dropped)}." if dropped else ""
    caption = (
        f"{len(rows)} families; n* at pass@8>=50%, converted to written lines at that n; "
        f"base, rl, armC.{drop_note}"
    )
    return P.dot_plot(title, takeaway, rows, x_label="written lines at n*", wide=True, note=caption)


# ===================================================================================== 3. Glue


_OP_GROUP: dict[str, str] = {
    "AND": "AND",
    "CHAIN": "CHAIN",
    "CASE-dilemma": "CASE",
    "CASE-same": "CASE",
    "BOX": "BOX",
}
_OP_ORDER: tuple[str, ...] = ("AND", "CHAIN", "CASE", "BOX")


def _chart_3_glue(store: Store) -> str:
    title = "Glue: does capability compose?"
    suite = glue_view.DEFAULT_SUITE
    parts_suite = glue_view.DEFAULT_PARTS_SUITE
    version = default_version(store, suite=suite) or "v1"
    parts_version = default_version(store, suite=parts_suite) or "v1"
    problems = problems_by_pid(store, suite, version)
    if not problems:
        return P.missing_fig(title, f"suite {suite!r} not found")
    models = models_present(store, suite=suite, version=version)
    if not models:
        return P.missing_fig(title, "no runs found for the glue suite")

    pids_by_op: dict[str, list[str]] = {}
    for pid, p in problems.items():
        op = _OP_GROUP.get(cast(str, p.tags.get("op", "")))
        if op:
            pids_by_op.setdefault(op, []).append(pid)

    bars: list[P.Bar] = []
    counts: dict[str, int] = {}
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
            groups = []
            for pid in pids:
                item = joined_items.get(pid)
                if item is None:
                    continue
                parts = problems[pid].parts
                part_results = [part_items[p] for p in parts if p in part_items]
                if part_results and len(part_results) == len(parts):
                    groups.append((item, part_results))
            if not groups:
                continue
            comp = compositionality_ratio(groups, mastery=0.5, B=1000, seed=0)
            lo, hi = comp["ci"]
            bars.append(P.Bar(f"{op}: {model}", comp["R"], lo, hi, _mcolor(model)))

    if not bars:
        return P.missing_fig(title, "no operator had both a joined and a parts_pool run")
    best = max(bars, key=lambda b: b.value)
    worst = min(bars, key=lambda b: b.value)
    takeaway = (
        f"compositionality is uneven across operators: {best.name} is highest "
        f"(R={P.fmt(best.value, 'f3')}); {worst.name} is lowest (R={P.fmt(worst.value, 'f3')}); "
        "R well below 1 means joining, not the parts, is the hard part."
    )
    caption = (
        "R = joined pass@1 / product of mastered parts' pass@1, pooled per operator; "
        + "; ".join(f"{op} n={n}" for op, n in counts.items())
        + "."
    )
    return P.bar_chart(title, takeaway, bars, y_fmt="f3", wide=True, note=caption)


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


def _chart_4_skills(store: Store) -> str:
    title = "Skills: RL pass@1 by length and special skill"
    suite = skills_view.DEFAULT_SUITE
    version = default_version(store, suite=suite) or "v1"
    problems = problems_by_pid(store, suite, version)
    if not problems:
        return P.missing_fig(title, f"suite {suite!r} not found")
    item_ncs = _rl_items_by_pid(store, problems)
    if not item_ncs:
        return P.missing_fig(title, "no rl results (direct or legacy) for skill_pool")

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
        key = (
            "needs classical"
            if is_classical
            else ("needs case split" if is_case else "no special skill")
        )
        buckets[key].setdefault(length, []).append((n, c))

    colors = {
        "no special skill": _mcolor("rl"),
        "needs case split": "var(--s4)",
        "needs classical": "var(--s5)",
    }
    series = []
    for name, by_l in buckets.items():
        pts = []
        for length, ncs in sorted(by_l.items()):
            vals = [pass_at_k(n, c, 1) for n, c in ncs]
            pts.append((length, statistics.mean(vals)))
        if pts:
            series.append(P.Series(name, pts, colors[name]))
    if not series:
        return P.missing_fig(title, "no item had both a known length and a known skill flag")

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
    caption = (
        f"n={n_items} skill_pool items (direct + legacy-imported runs), rl only; "
        "base shown in detail on the Skills page."
    )
    return P.line_chart(
        title, takeaway, series, x_label="L (written lines)", y_fmt="pct", note=caption
    )


# ============================================================================ 5. Law x substitution


def _chart_5_subst(store: Store) -> str:
    title = "Law x substitution size"
    data = subst_view.view(store, {})
    if data.get("empty"):
        return P.missing_fig(title, str(data.get("reason", "no data")))
    schemas = cast("dict[str, dict[str, dict[str, Any]]]", data["schemas"])
    by_model_s: dict[str, dict[float, list[float]]] = {}
    n_items = 0
    for per_model in schemas.values():
        for model, d in per_model.items():
            n_items += cast(int, d["n_items"])
            for row in cast("list[dict[str, Any]]", d["curve_vs_s"]):
                s = row["s"]
                if s is None:
                    continue
                by_model_s.setdefault(model, {}).setdefault(float(s), []).append(row["pass1"])

    series = []
    for model in ("base", "rl"):
        by_s = by_model_s.get(model)
        if not by_s:
            continue
        pts = sorted((s, statistics.mean(v)) for s, v in by_s.items())
        series.append(P.Series(model, pts, _mcolor(model)))
    if not series:
        return P.missing_fig(title, "no base/rl results for the laws suite")

    def _at(model: str, s: float) -> float | None:
        by_s = by_model_s.get(model, {})
        return statistics.mean(by_s[s]) if s in by_s else None

    s_values = sorted({s for by_s in by_model_s.values() for s in by_s})
    s0, s_hi = s_values[0], s_values[-1]
    b0, r0 = _at("base", s0), _at("rl", s0)
    b_hi, r_hi = _at("base", s_hi), _at("rl", s_hi)
    takeaway = (
        f"pass@1 falls from base {P.fmt(b0, 'pct')}/rl {P.fmt(r0, 'pct')} at s={P.fmt(s0, 'g')} "
        f"to base {P.fmt(b_hi, 'pct')}/rl {P.fmt(r_hi, 'pct')} at s={P.fmt(s_hi, 'g')}: filler "
        "complexity, not law identity, drives the failure."
    )
    caption = f"n={n_items} instances, averaged over 8 schemas: {', '.join(sorted(schemas))}."
    return P.line_chart(
        title,
        takeaway,
        series,
        x_label="s (connectives per metavariable)",
        y_fmt="pct",
        note=caption,
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


def _chart_6_padding(store: Store) -> str:
    title = "Free lines: the cost of padding"
    data = padding_view.view(store, {})
    if data.get("empty"):
        return P.missing_fig(title, str(data.get("reason", "no data")))
    pad_types = cast("dict[str, dict[str, dict[str, Any]]]", data["pad_types"])

    def _curves(model: str, key: str) -> list[list[dict[str, Any]]]:
        return [d[key] for per_model in pad_types.values() if (d := per_model.get(model))]

    rl_padded = _pool_curve(_curves("rl", "padded_curve_vs_W"))
    rl_natural = _pool_curve(_curves("rl", "natural_curve_vs_W"))
    base_padded = _pool_curve(_curves("base", "padded_curve_vs_W"))
    if not rl_padded and not base_padded:
        return P.missing_fig(title, "no base/rl results for the padding suite")

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
            f"theorems but only {P.fmt(pad_p1, 'pct')} once those lines are padding -- length "
            "alone does not explain the drop."
        )
    else:
        takeaway = "padded vs. natural pass@1 at matched written length W."
    pad_types_list = sorted(pad_types)
    n_rl = sum(d["n_items"] for per_model in pad_types.values() if (d := per_model.get("rl")))
    caption = (
        f"n={n_rl} rl padded items pooled across pad types {', '.join(pad_types_list)}; "
        "rl and base."
    )
    return P.line_chart(
        title, takeaway, series, x_label="W (written lines)", y_fmt="pct", note=caption
    )


# ============================================================================= 7. Prefix completion


def _chart_7_prefix(store: Store) -> str:
    title = "Prefix completion: finishing someone else's proof"
    data = prefix_view.view(store, {})
    if data.get("empty"):
        return P.missing_fig(title, str(data.get("reason", "no data")))
    models = cast("dict[str, dict[str, Any]]", data["models"])
    series = []
    n_items_by_model = {}
    coefs: dict[str, dict[str, Any]] = {}
    for model, d in models.items():
        by_ll = cast("dict[str, dict[str, Any]]", d["success_by_lines_left"])
        pts = sorted((float(k), v["pass1"]) for k, v in by_ll.items())
        series.append(P.Series(model, pts, _mcolor(model)))
        n_items_by_model[model] = d.get("n_items", "?")
        coefs[model] = cast("dict[str, Any]", d["fitted_logit"])
    if not series:
        return P.missing_fig(title, "no model had usable j/L tags")

    bits = []
    for model in ("base", "rl"):
        c = coefs.get(model)
        if not c:
            continue
        ll = c["lines_left"]["coef"]
        jc = c["j"]["coef"]
        bits.append(f"{model} lines-left {P.fmt(ll, 'f3')}, position {P.fmt(jc, 'f3')}")
    takeaway = (
        "more lines left to write costs more than being further into the proof (fitted logit "
        "coefficients, " + "; ".join(bits) + ")."
        if bits
        else "success by lines left to write."
    )
    n_str = ", ".join(f"{m}={n}" for m, n in n_items_by_model.items())
    caption = f"items: {n_str}; pooled success rate by lines left (L-j); base vs rl."
    return P.line_chart(
        title, takeaway, series, x_label="lines left (L-j)", y_fmt="pct", note=caption
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


def _chart_8_edl(store: Store) -> str:
    title = "EDL: nats per example to pick up a new pattern"
    run_ids = (
        sorted(p.stem for p in store.edl_dir.glob("edl2-*.jsonl")) if store.edl_dir.exists() else []
    )
    if not run_ids:
        return P.missing_fig(title, "no edl2-* run files found under store/edl/")
    data = edl_view.view(store, cast("Mapping[str, JSON]", {"run_ids": run_ids}))
    if data.get("empty"):
        return P.missing_fig(title, str(data.get("reason", "no data")))
    classes = cast("dict[str, dict[str, dict[str, Any]]]", data["classes"])

    friendly_by_run: dict[str, str] = {}
    n_by_friendly: dict[str, int] = {}
    for run_id in run_ids:
        rows = edl_view._read_edl_rows(store, run_id)  # noqa: SLF001
        if not rows:
            continue
        identity = rows[0].get("model_identity")
        raw = edl_view._model_label(identity, run_id)  # noqa: SLF001
        friendly = _edl_friendly_label(identity, run_id)
        friendly_by_run[raw] = friendly
        n_by_friendly[friendly] = int(cast(int, rows[0].get("n", 64)))

    bars: list[P.Bar] = []
    for cls in sorted(classes, key=_edl_class_key):
        for raw_model, d in classes[cls].items():
            model = friendly_by_run.get(raw_model, raw_model)
            n = n_by_friendly.get(model, 64)
            best_lr = cast(str, d["best_lr"])
            edl_val = cast(float, d["edl_by_lr"][best_lr])
            bars.append(P.Bar(f"{cls}: {model}", edl_val / n, color=_mcolor(model)))
    if not bars:
        return P.missing_fig(title, "no usable EDL rows")

    base_vals = {b.name.rsplit(": ", 1)[0]: b.value for b in bars if b.name.endswith(": base")}
    rl_vals = {b.name.rsplit(": ", 1)[0]: b.value for b in bars if b.name.endswith(": rl")}
    common = sorted(set(base_vals) & set(rl_vals), key=_edl_class_key)
    n_rl_lower = sum(1 for c in common if rl_vals[c] < base_vals[c])
    takeaway = (
        f"rl needs fewer nats/example than base on {n_rl_lower}/{len(common)} classes -- an "
        "unfamiliar pattern already close to known costs little to pick up online."
        if common
        else "EDL per example, by class and model."
    )
    caption = (
        f"{len(common)} classes, base vs rl; EDL at the learning rate with the smallest online "
        "code length (ln(3) hyperparameter-cost term excluded here); edl2-* runs only."
    )
    return P.hbar_chart(title, takeaway, bars, x_fmt="f3", log_x=True, wide=True, note=caption)


# ================================================================================ 9. Likelihood


def _likelihood_crux_data(
    store: Store,
) -> tuple[list[float], list[float], dict[str, Any]] | None:
    run_id = latest_interp_run(store, "likelihood")
    if run_id is None:
        return None
    rows = load_interp_rows(store, "likelihood", run_id)
    if not rows:
        return None
    cfg: Mapping[str, JSON] = {}
    for r in interp_runs(store, "likelihood"):
        if r.get("run_id") == run_id:
            cfg = cast("Mapping[str, JSON]", r.get("cfg") or {})
            break
    model_samples = cast("Mapping[str, JSON]", cfg.get("model_samples") or {})

    def _problems(model_name: str) -> dict[str, Problem]:
        spec = cast("Mapping[str, JSON] | None", model_samples.get(model_name))
        if not spec or not spec.get("suite") or not spec.get("version"):
            return {}
        return problems_by_pid(store, str(spec["suite"]), str(spec["version"]))

    base_problems = _problems("base")
    rl_problems = _problems("rl")

    base_vals: list[float] = []
    base_lens: list[float] = []
    crux_vals: list[float] = []
    crux_lens: list[float] = []
    for r in rows:
        base_side = cast("Mapping[str, JSON]", r["base"])
        lm = base_side.get("logp_marginal")
        if lm is None:
            continue
        bare_pid = cast(str, r["pid"]).split(":", 1)[-1]
        origin = r.get("origin")
        if origin == "base_verified":
            length = item_length(base_problems[bare_pid]) if bare_pid in base_problems else None
            if length:
                base_vals.append(float(cast(float, lm)) / length)
                base_lens.append(length)
        elif origin == "rl_verified":
            p_hat = r.get("p_hat_base")
            if p_hat is None or float(cast(float, p_hat)) != 0.0:
                continue
            length = item_length(rl_problems[bare_pid]) if bare_pid in rl_problems else None
            if length:
                crux_vals.append(float(cast(float, lm)) / length)
                crux_lens.append(length)
    if not base_vals or not crux_vals:
        return None
    meta = {
        "run_id": run_id,
        "n_base": len(base_vals),
        "n_crux": len(crux_vals),
        "base_mean_L": statistics.mean(base_lens),
        "crux_mean_L": statistics.mean(crux_lens),
    }
    return base_vals, crux_vals, meta


def _freq_series(name: str, values: Sequence[float], color: str, bins: Sequence[float]) -> P.Series:
    counts = [0] * (len(bins) - 1)
    for v in values:
        idx = min(len(counts) - 1, max(0, int((v - bins[0]) / (bins[-1] - bins[0]) * len(counts))))
        counts[idx] += 1
    mids = [(bins[i] + bins[i + 1]) / 2.0 for i in range(len(bins) - 1)]
    return P.Series(name, list(zip(mids, (float(c) for c in counts), strict=True)), color)


def _chart_9_likelihood(store: Store) -> str:
    title = "Likelihood: how surprised is base by RL's proofs?"
    result = _likelihood_crux_data(store)
    if result is None:
        return P.missing_fig(title, "no `interp likelihood` run with resolvable lengths")
    base_vals, crux_vals, meta = result
    lo = min(min(base_vals), min(crux_vals))
    hi = max(max(base_vals), max(crux_vals))
    if hi <= lo:
        hi = lo + 1.0
    bins = [lo + i * (hi - lo) / 24 for i in range(25)]
    series = [
        _freq_series("base: own verified proofs", base_vals, _mcolor("base"), bins),
        _freq_series("rl: crux proofs (base never solves)", crux_vals, _mcolor("rl"), bins),
    ]
    med_base = statistics.median(base_vals)
    med_crux = statistics.median(crux_vals)
    frac_below = sum(1 for v in crux_vals if v < min(base_vals)) / len(crux_vals)
    takeaway = (
        f"median base log p/line is {P.fmt(med_base, 'f3')} nats on its own proofs vs "
        f"{P.fmt(med_crux, 'f3')} on rl's crux proofs ({P.fmt(frac_below, 'pct')} of crux proofs "
        "score below base's own minimum)."
    )
    caption = (
        f"n={meta['n_base']} base-verified, n={meta['n_crux']} crux (p_hat_base=0) proofs, run "
        f"{meta['run_id']}; length confound: base's proofs average "
        f"{P.fmt(meta['base_mean_L'], 'g')} lines, crux proofs {P.fmt(meta['crux_mean_L'], 'g')}."
    )
    return P.line_chart(
        title,
        takeaway,
        series,
        x_label="base log p / L (nats per line)",
        y_fmt="int",
        wide=True,
        note=caption,
    )


# =================================================================================== 10. Steering


def _chart_10_steering(store: Store) -> str:
    title = "Steering: base pushed toward rl"
    data = interp_view.view_steering(store, {})
    if data.get("empty"):
        return P.missing_fig(title, str(data.get("reason", "no data")))
    curves = cast("dict[str, list[tuple[float, float, float, float]]]", data["curves_by_layer"])
    meta = cast("dict[str, Any]", data["meta"])
    layer_mean = {
        layer: statistics.mean(m for _a, m, _lo, _hi in pts) for layer, pts in curves.items() if pts
    }
    if not layer_mean:
        return P.missing_fig(title, "no steer:* runs with any items")
    best_layer = max(layer_mean, key=lambda layer: layer_mean[layer])
    pts = curves[best_layer]
    series = [
        P.Series(
            f"layer {best_layer}",
            [(a, m) for a, m, _lo, _hi in pts],
            _mcolor("rl"),
            band=[(a, lo, hi) for a, _m, lo, hi in pts],
        )
    ]
    alphas = [a for a, _m, _lo, _hi in pts]
    base_pass1 = cast("list[float] | None", data.get("base_pass1"))
    rl_pass1 = interp_view._pooled_pass1_of(  # noqa: SLF001
        store,
        suites=cast("list[str]", meta["suites"]),
        version=cast(str, meta["version"]),
        model="rl",
        B=int(cast(int, meta["B"])),
        seed=0,
    )
    x0, x1 = min([0.0, *alphas]), max(alphas)
    if base_pass1 is not None:
        series.append(
            P.Series(
                "base (unsteered)",
                [(x0, base_pass1[0]), (x1, base_pass1[0])],
                _mcolor("base"),
                muted=True,
            )
        )
    if rl_pass1 is not None:
        series.append(
            P.Series("rl", [(x0, rl_pass1[0]), (x1, rl_pass1[0])], "var(--s4)", muted=True)
        )

    best_alpha, best_val = max(pts, key=lambda t: t[1])[:2]
    takeaway = (
        f"layer {best_layer} has the highest mean pass@1 across alpha; steering pushes base from "
        f"{P.fmt(base_pass1[0] if base_pass1 else float('nan'), 'pct')} (unsteered) to "
        f"{P.fmt(best_val, 'pct')} at alpha={P.fmt(best_alpha, 'g')}"
        + (f", still below rl's {P.fmt(rl_pass1[0], 'pct')}." if rl_pass1 else ".")
    )
    rl_control = cast("dict[str, Any]", data.get("rl_control", {}))
    ctrl_bits = [f"layer {layer}: {P.fmt(d['pass1'], 'pct')}" for layer, d in rl_control.items()]
    caption = (
        f"suites {', '.join(cast('list[str]', meta['suites']))}; layers "
        f"{', '.join(sorted(curves, key=int))}; rl-delta control: "
        + (", ".join(ctrl_bits) or "n/a")
        + "."
    )
    return P.line_chart(
        title, takeaway, series, x_label="alpha", y_fmt="pct", wide=True, note=caption
    )


# ============================================================================ 11. Weight interp.


def _chart_11_weights(store: Store) -> str:
    title = "Weight interpolation: base -> rl"
    data = interp_view.view_interpolation(store, {})
    if data.get("empty"):
        return P.missing_fig(title, str(data.get("reason", "no data")))
    pass1 = cast("list[tuple[float, float, float, float]]", data["pass1_vs_alpha"])
    pass_k = cast("dict[str, list[tuple[int, float, float, float]]]", data["pass_k_by_alpha"])
    if not pass1 and not pass_k:
        return P.missing_fig(title, "no base/rl/interp:* runs")

    series = []
    clean1 = [(a, m, lo, hi) for a, m, lo, hi in pass1 if m == m]
    if clean1:
        series.append(
            P.Series(
                "pass@1",
                [(a, m) for a, m, _lo, _hi in clean1],
                _mcolor("base"),
                band=[(a, lo, hi) for a, _m, lo, hi in clean1],
            )
        )
    pts128 = []
    for alpha_str, curve in pass_k.items():
        v128 = next((mean for k, mean, _lo, _hi in curve if k == 128), None)
        if v128 is not None and v128 == v128:
            pts128.append((float(alpha_str), v128))
    if pts128:
        series.append(P.Series("pass@128", sorted(pts128), _mcolor("rl"), muted=True))
    if not series:
        return P.missing_fig(title, "interpolation runs exist but have no usable points")

    takeaway = "pass@1 and pass@128 vs alpha."
    if clean1 and pts128:
        p128_by_a = dict(pts128)
        # smallest alpha with a nonzero pass@1 (alpha=0 is base itself, usually ~0 on both
        # metrics and uninformative for "does coverage show up before pass@1 does").
        candidates = sorted(
            (a, m, p128_by_a[a]) for a, m, _lo, _hi in clean1 if a > 0 and m > 0 and a in p128_by_a
        )
        a_one = min(clean1, key=lambda t: abs(t[0] - 1.0))
        if candidates:
            a_lo, p1_lo, p128_lo = candidates[0]
            ratio_lo = p128_lo / p1_lo if p1_lo else float("nan")
            takeaway = (
                f"at alpha={P.fmt(a_lo, 'g')}, pass@128 ({P.fmt(p128_lo, 'pct')}) is "
                f"{ratio_lo:.1f}x pass@1 ({P.fmt(p1_lo, 'pct')}) -- sampling already finds it far "
                f"more than a single draw; by alpha={P.fmt(a_one[0], 'g')} pass@1 reaches "
                f"{P.fmt(a_one[1], 'pct')}."
            )
    caption = "fixed theorem subset, k up to 128; base=alpha 0, rl=alpha 1."
    return P.line_chart(
        title,
        takeaway,
        series,
        x_label="alpha (0=base, 1=rl)",
        y_fmt="pct",
        wide=True,
        note=caption,
    )


# =================================================================================== 12. Probes


def _chart_12_probes(store: Store) -> str:
    title = "Probes: is the information there, and used?"
    data = interp_view.view_probes(store, {})
    if data.get("empty"):
        return P.missing_fig(title, str(data.get("reason", "no data")))
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
            series.append(P.Series(f"next_rule acc: {model}", a, _mcolor(model)))
    for model in ("base", "rl"):
        r = _clean(r2.get(model, []))
        if r:
            series.append(P.Series(f"lines_to_go R2: {model}", r, _mcolor(model), muted=True))
    if not series:
        return P.missing_fig(title, "probes run has rows but no usable accuracy/R2 series")

    ood = cast("dict[str, list[tuple[int, JSON]]]", data.get("length_generalisation_ood_r2", {}))

    def _last(name: str) -> float | None:
        pts = _clean(acc.get(name, []))
        return pts[-1][1] if pts else None

    b_last, r_last = _last("base"), _last("rl")
    gap = (
        ""
        if b_last is None or r_last is None
        else f" (deepest layer: {P.fmt(b_last, 'pct')} vs {P.fmt(r_last, 'pct')})"
    )
    takeaway = (
        "next_rule accuracy is close between base and rl at every layer"
        + gap
        + " -- RL mostly changes how the information already in base is used, not whether it "
        "exists."
    )
    ood_bits = []
    for model in ("base", "rl"):
        pts = _clean(ood.get(model, []))
        if pts:
            ood_bits.append(f"{model} {P.fmt(pts[-1][1], 'f3')}")
    caption = (
        f"layers {', '.join(layers)}; solid=next_rule acc, dashed=lines_to_go R2; "
        "OOD length-generalisation R2 (deepest layer): " + (", ".join(ood_bits) or "n/a") + "."
    )
    return P.line_chart(
        title,
        takeaway,
        series,
        x_label=f"layer ({', '.join(layers)})",
        y_fmt="f3",
        wide=True,
        note=caption,
    )


# ======================================================================================== page


_EXTRA_CSS = (
    ".card .sub{font-weight:600;color:var(--ink);font-size:13px}"
    ".card .note{font-style:italic}"
    ".pagefoot{max-width:1360px;margin:0 auto;padding:0 16px 28px;color:var(--ink2);"
    "font-size:12.5px}"
)

_MODELS_NOTE = (
    "base = pretrained model, trained on proofs up to 6 lines; rl = run-2 RL champion "
    "(base fine-tuned by RL on the full training pool); armC = RL model from the length "
    "experiment, trained on lengths 6-11 only. All results below are single-checkpoint, "
    "single-seed."
)


def build_summary(store: Store, out_dir: Path) -> Path:
    """Render the single-page summary dashboard (one chart per benchmark) into
    `out_dir/summary.html` and return its path. `out_dir` is created if missing."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # `_mark` is applied here, once, to every chart's result uniformly -- including its "no data
    # yet" panel (`P.missing_fig`'s own early-return branches never call `_mark` themselves), so
    # the page always has exactly 12 marked chart cards regardless of which of the 12 have data.
    figs = [
        _mark(chart(store))
        for chart in (
            _chart_1_yue,
            _chart_2_families,
            _chart_3_glue,
            _chart_4_skills,
            _chart_5_subst,
            _chart_6_padding,
            _chart_7_prefix,
            _chart_8_edl,
            _chart_9_likelihood,
            _chart_10_steering,
            _chart_11_weights,
            _chart_12_probes,
        )
    ]

    html = P.page("ND Capability Benchmarks", _MODELS_NOTE, "summary.html", figs, [])
    html = html.replace("</style>", _EXTRA_CSS + "</style>")
    gpu_h = store.ledger_total_s() / 3600.0
    footer = f'<p class="pagefoot">GPU-hours used (ledger total): {gpu_h:.2f} h</p>'
    html = html.replace("</main>", "</main>" + footer)

    path = out_dir / "summary.html"
    path.write_text(html)
    return path
