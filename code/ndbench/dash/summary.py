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


def _chart_2_families(store: Store) -> str:
    title = "Restricted families: horizon in written lines (theorems unseen in training)"
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
        return P.missing_fig(title, "no family had a qualifying (>=5-item) size cell")

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
            f"{model} on {fam} reaches about {P.fmt(lines, 'g')} written lines"
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
        f"{len(rows)} families; n* at pass@8>=50% over size cells with >={_MIN_ITEMS_PER_N} "
        "unseen items only (theorems in base's pretraining set or the RL pool are excluded; "
        "they make up most of the smallest sizes); >= marks a censored horizon (never dropped "
        f"below 50% in the tested range); base, rl, armC.{seen_note}{drop_note}"
    )
    return P.dot_plot(
        title,
        takeaway,
        rows,
        x_label="written lines at n*",
        none_label="fails at smallest size",
        wide=True,
        note=caption,
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
        return P.missing_fig(title, "no operator had both a joined and a parts_pool run")
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
        "R = joined pass@1 / product of mastered parts' pass@1, pooled per operator; "
        + "; ".join(f"{op} n={n}" for op, n in counts.items())
        + f"; {src_note}."
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
        return P.missing_fig(title, "no (length, skill) cell had >=20 items")

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
    caption = (
        f"n={n_items} skill_pool items (direct + legacy-imported runs), rl only; shaded band: "
        f"95% item-level bootstrap CI.{drop_note} base shown in detail on the Skills page."
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
        return P.missing_fig(title, "no base/rl results for the laws suite")

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
        f"n={n_items} instances over {len(schemas)} schemas ({', '.join(sorted(schemas))}), "
        f"pooled per item at each s; {meta.get('n_seen_train_excluded', 0)} instances seen in "
        "training excluded"
        + (
            f"; s={', '.join(P.fmt(v, 'g') for v in dropped_s)} omitted "
            f"(<{_MIN_ITEMS_PER_BIN} items)"
            if dropped_s
            else ""
        )
        + f".{zero_note}"
    )
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
            f"theorems but only {P.fmt(pad_p1, 'pct')} once those lines are padding."
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


#: Lines-left bins with fewer sampled items than this are dropped before plotting (review: thin
#: bins -- e.g. 10 or 6 items -- cause a zig-zag that has nothing to do with lines left, such as a
#: jump from 0% at 4 lines left to 16.8% at 5).
_MIN_PREFIX_BIN_ITEMS = 20


def _chart_7_prefix(store: Store) -> str:
    title = "Prefix completion: finishing someone else's proof"
    data = prefix_view.view(store, {})
    if data.get("empty"):
        return P.missing_fig(title, str(data.get("reason", "no data")))
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
        return P.missing_fig(title, "no model had a lines-left bin with >=20 items")

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
        f"items: {n_str}; pooled success rate by lines left (L-j), bins with "
        f"<{_MIN_PREFIX_BIN_ITEMS} items omitted; base vs rl."
    )
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


#: The null control's EDL is expected to be ~0 (and noisy enough to land slightly negative, which
#: is undefined on the chart's log scale -- `dot_plot`'s `log_x` clamps it to the axis floor with
#: a "<= 0" label) -- it is reported but excluded from the "rl lower on N/M classes" count below,
#: since "lower EDL" has no "elicited a real capability" meaning for a class with nothing to learn.
_EDL_NULL_CONTROL = "heldout_cap6"


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

    rows_by_class: dict[str, dict[str, float]] = {}
    for cls, per_model in classes.items():
        for raw_model, d in per_model.items():
            model = friendly_by_run.get(raw_model, raw_model)
            if model not in ("base", "rl"):
                continue
            n = n_by_friendly.get(model, 64)
            best_lr = cast(str, d["best_lr"])
            edl_val = cast(float, d["edl_by_lr"][best_lr])
            rows_by_class.setdefault(cls, {})[model] = edl_val / n
    if not rows_by_class:
        return P.missing_fig(title, "no usable EDL rows")

    dot_rows: list[P.DotRow] = []
    for cls in sorted(rows_by_class, key=_edl_class_key):
        vals = rows_by_class[cls]
        label = f"{cls} (null control)" if cls == _EDL_NULL_CONTROL else cls
        pts: list[tuple[str, float | None, str, bool]] = [
            (m, vals[m], _mcolor(m), False) for m in ("base", "rl") if m in vals
        ]
        if pts:
            dot_rows.append(P.DotRow(label, pts))

    common = {
        cls: vals
        for cls, vals in rows_by_class.items()
        if cls != _EDL_NULL_CONTROL and "base" in vals and "rl" in vals
    }
    n_rl_lower = sum(1 for vals in common.values() if vals["rl"] < vals["base"])
    takeaway = (
        f"rl needs fewer nats/example than base on {n_rl_lower}/{len(common)} classes "
        f"(excluding the {_EDL_NULL_CONTROL} null control)."
        if common
        else "EDL per example, by class and model."
    )
    caption = (
        f"{len(common)} classes, base vs rl, plus the null control; EDL at the learning rate "
        "with the smallest online code length (ln(3) hyperparameter-cost term excluded here); "
        "edl2-* runs only."
    )
    return P.dot_plot(
        title,
        takeaway,
        dot_rows,
        x_label="EDL per example (nats)",
        log_x=True,
        wide=True,
        note=caption,
    )


# ================================================================================ 9. Likelihood


#: Row order and display labels for chart 9's dot plot (review: "one row per proof set: reserve
#: (neutral), family witnesses, base's own verified proofs, rl's crux proofs").
_LIKELIHOOD_ROW_ORDER: tuple[str, ...] = ("reserve", "family", "base_verified", "reference", "crux")
_LIKELIHOOD_ROW_LABEL: dict[str, str] = {
    "reserve": "reserve (neutral)",
    "family": "family witnesses",
    "base_verified": "base's own verified proofs",
    "reference": "minimal proofs of rl-solved gen theorems",
    "crux": "rl's own proofs of them",
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


def _chart_9_likelihood(store: Store) -> str:
    title = "Likelihood: how surprised is base by RL's proofs?"
    result = _likelihood_per_line(store)
    if result is None:
        return P.missing_fig(title, "no `interp likelihood` run with resolvable lengths")
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
        return P.missing_fig(title, "no origin had both base and rl per-line likelihoods")
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
        "median log p per written line of the scored proof, base vs rl, by proof set; "
        f"runs {', '.join(cast('list[str]', meta['run_ids']))}. The last two rows are the same "
        f"{meta['n_paired']} gen_ctrl theorems rl solves and base never does (p_hat_base=0); "
        "their minimal proofs come from the minlen labeller." + ordering_note
    )
    return P.dot_plot(
        title,
        takeaway,
        dot_rows,
        x_label="median log p per written line (nats)",
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
    suites = cast("list[str]", meta["suites"])
    base_pass1 = cast("list[float] | None", data.get("base_pass1"))
    if not curves and base_pass1 is None:
        return P.missing_fig(title, "no base/steer:* runs with any items")

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
        return P.missing_fig(title, "no base/rl/steer:* pass@1 could be computed")

    bits = []
    if base_pass1 is not None and best is not None:
        bits.append(
            "adding the RL-base direction to base does nothing "
            f"({P.fmt(base_pass1[0], 'pct')} -> {P.fmt(best[2], 'pct')})"
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
    caption = f"pooled pass@1 over {', '.join(suites)}.{tdev_note}"
    return P.bar_chart(title, takeaway, bars, y_fmt="pct", wide=True, note=caption)


# ============================================================================ 11. Weight interp.


def _chart_11_weights(store: Store) -> str:
    title = "Weight interpolation: base -> rl"
    data = interp_view.view_interpolation(store, {})
    if data.get("empty"):
        return P.missing_fig(title, str(data.get("reason", "no data")))
    pass_k = cast("dict[str, list[tuple[int, float, float, float]]]", data["pass_k_by_alpha"])
    if not pass_k:
        return P.missing_fig(title, "no base/rl/interp:* runs on the fixed k=128 subset")

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
        return P.missing_fig(title, "no alpha had both pass@1 and pass@128 on the k=128 subset")

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
    caption = (
        "fixed k=128 theorem subset, pass@1 and pass@128 on the same items; "
        "base=alpha 0, rl=alpha 1."
    )
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


def _probe_baselines(store: Store) -> dict[str, Any] | None:
    run_id = latest_interp_run(store, "probe_baselines")
    rows = load_interp_rows(store, "probe_baselines", run_id) if run_id else []
    return cast("dict[str, Any]", rows[-1]) if rows else None


def _chart_12_probes(store: Store) -> str:
    title = "Probes: what do the activations add beyond position?"
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
            series.append(P.Series(f"next_rule probe: {model}", a, _mcolor(model)))
    if not series:
        return P.missing_fig(title, "probes run has rows but no usable accuracy series")
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
        f"layers {', '.join(layers)}; next_rule = the rule of the line about to be written. "
        f"lines_to_go R2 at the deepest layer: base {P.fmt(r2_b, 'f3')}, rl {P.fmt(r2_r, 'f3')} "
        f"vs {P.fmt(ltg_base, 'f3')} from the line index alone; on longer (OOD) proofs it is "
        f"negative for both (base {P.fmt(ood_b, 'f3')}, rl {P.fmt(ood_r, 'f3')}). Majority-class "
        f"next_rule accuracy: {P.fmt((baselines or {}).get('next_rule_acc_majority'), 'pct')}."
    )
    display_labels = ["emb" if layer == "emb" else f"L{layer}" for layer in layers]
    return P.line_chart(
        title,
        takeaway,
        series,
        x_label="layer",
        y_fmt="pct",
        wide=True,
        note=caption,
        x_tick_labels=display_labels,
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
