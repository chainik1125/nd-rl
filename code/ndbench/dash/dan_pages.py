"""Dan's state-env models: a "State-env models" summary section plus a standalone `dan.html`
detail page (`DANSTATE_PLAN.md` §6, WP-D5 item 3).

Off by default (`cli.py`'s `dash --summary --with-dan` / `figures --with-dan`), so the existing
base/rl/armC summary page and its figures are byte-identical when the flag is not given -- the
golden regression (`code/tests/ndbench/test_paper_figures.py`) pins that page, and nothing in this
module is on its import or call path unless a caller explicitly asks for Dan data.

Pairs/families/seeds come from `analysis/models_registry.py` (`configs/models.yaml`'s
family/role/seed); the headline judge for Dan's models is `"lean_v2"`
(`models_registry.DAN_JUDGE_ID`), `nd_v2` for "ours" (base/rl/armC) as everywhere else in this
package (`DANSTATE_PLAN.md` §2). Every multi-value chart is small multiples by model family, one
panel per `figures.plots.DAN_FAMILY_ORDER` entry (`"ours" | "dan12" | "dan6"`), on one y scale; a
family with no matching runs yet renders as a clean "not available" panel --
`figures.plots.dan_panel_figure` never crashes on missing data, and neither does anything in this
module.

Each Dan figure uses the estimator of the audited paper figure for the same benchmark, so its
"ours" panel reproduces that figure's numbers (audit 2026-10-01, pinned by
`code/tests/ndbench/test_dan_section_audit.py`): families (Figure 3), laws (5), glue (4),
padding (6) and prefix (7) reuse or mirror `dash.summary`'s own helpers, with each model's own
judge and training tiers.

Reuses the existing per-benchmark `view_<bench>(store, cfg)` functions wherever their generic
`models=`/`judge_id=` cfg already covers Dan's models (`analysis.{yue,horizon,glue,padding,
prefix,subst}`); `analysis.dan_heldout`/`analysis.judges` are the two views WP-D5 added because
no existing benchmark fit (held-out greedy accuracy; nd_v2-vs-lean_v2 judge agreement).
`_data_dan_skills` reads `skill_pool_sub` directly rather than through `analysis.skills.view`:
that module's primary job is joining `skill_pool`'s *legacy*-imported per-theorem runs back by
theorem name (its own module docstring), which does not apply to `skill_pool_sub` (a suite Dan's
models sample directly); re-deriving its much simpler direct-run case here avoids dragging in that
unrelated machinery.

Every card is a `ChartData` rendered through the *existing* generic `dash.summary.render_card` /
`figures.plots.render` dispatch (extended additively to also check `figures.plots.DAN_FIGURES`),
so nothing here duplicates that rendering logic -- only the data-shaping is Dan-specific.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from ..analysis import dan_heldout as heldout_view
from ..analysis import glue as glue_view
from ..analysis import horizon as horizon_view
from ..analysis import judges as judges_view
from ..analysis import models_registry as mr
from ..analysis import padding as padding_view
from ..analysis import prefix as prefix_view
from ..analysis import subst as subst_view
from ..analysis import yue as yue_view
from ..analysis._common import (
    default_version,
    items_by_pid,
    latest_run_id,
    models_present,
    problems_by_pid,
)
from ..core.types import JSON, ItemResult
from ..figures.plots import DAN_FAMILY_ORDER
from ..metrics import compositionality_ratio, pass_at_k
from ..store import Store
from . import primitives as P
from .chart_data import ChartData
from .chart_data import missing_chart as _missing

#: `figures/plots.py`'s panel order (`"ours" | "dan12" | "dan6"`), re-exported here so this
#: module's own family loop and that module's drawer never drift apart; importing it also runs
#: `figures/plots.py`'s module-level `S.register_family("dan12")`/`"dan6"` calls exactly once
#: (`STYLE.md`'s "Adding models": register in a fixed order, at import time).

#: Headline judge for Dan's models (plan §2); "ours" (base/rl/armC) keeps the package-wide
#: default, `models_registry.DEFAULT_JUDGE_ID` ("nd_v2"). `DEFAULT_JUDGE_ID` is typed `str |
#: None` in `_common` (callers may pass `None` to disable the judge filter); its value is always
#: the literal "nd_v2" string.
DAN_JUDGE_ID = mr.DAN_JUDGE_ID
OURS_JUDGE_ID = cast(str, mr.DEFAULT_JUDGE_ID)


def _cfg(d: Mapping[str, object]) -> dict[str, JSON]:
    """`cast("dict[str, JSON]", d)` (`analysis.interp`'s own idiom, see its `_j` helper): every
    `view()` call below builds its `cfg` from a `list[str]` of model labels plus scalars, which
    pyright's invariant generics reject against `Mapping[str, JSON]` without this -- the values
    really are JSON at runtime (plain model labels, suite names, small numbers)."""
    return cast("dict[str, JSON]", d)


#: Reduced bootstrap replicate count for Dan's dashboard cards (plan: "CPU only... keep CPU use
#: modest"): this module calls the heavier views (`yue`, `horizon`) up to twice per chart (once
#: for "ours", once for every Dan model together), so a smaller `B` than the paper figures'
#: default (2000) keeps a full `--with-dan` render fast without materially changing a point
#: estimate's shown value (only its CI width).
DAN_B = 200


def _model_groups(registry: Mapping[str, mr.ModelEntry]) -> dict[str, list[str]]:
    """`{family: [model labels]}` for every `DAN_FAMILY_ORDER` entry: "ours" is exactly
    `base`/`rl` (armC is a third, unpaired arm with no Dan analogue, so it is left out of these
    paired small multiples -- the existing base/rl/armC page already covers it on its own)."""
    out: dict[str, list[str]] = {"ours": ["base", "rl"]}
    for family in DAN_FAMILY_ORDER:
        if family == "ours":
            continue
        out[family] = mr.labels_in_family(registry, family)
    return out


def _judge_for(family: str) -> str:
    return OURS_JUDGE_ID if family == "ours" else DAN_JUDGE_ID


def _ref_suffix(label: str) -> str:
    from ..figures.plots import REF_SUFFIX

    return f"{label}{REF_SUFFIX}"


def _with_data(panels: Mapping[str, ChartData]) -> dict[str, list[str]]:
    """`{family: models drawn in its panel}`, for a legend that lists only models with data (a
    `"group: model"` bar counts as `model`; the grey `REF_SUFFIX` reference is added separately)."""
    from ..figures.plots import REF_SUFFIX

    out: dict[str, list[str]] = {}
    for family, cd in panels.items():
        names = [s.name for s in cd.series] + [b.name.rpartition(": ")[2] for b in cd.bars]
        out[family] = [n for n in dict.fromkeys(names) if not n.endswith(REF_SUFFIX)]
    return out


def _runs_note(drawn: Mapping[str, Sequence[str]]) -> str:
    """ "Models with runs: base, rl; dan12 fz-s0, t1-s0; dan6 none." from `{family: models drawn}`
    (computed, so the caption follows the store as more of Dan's runs land)."""
    parts = []
    for family in DAN_FAMILY_ORDER:
        models = list(drawn.get(family, []))
        names = ", ".join(_short(m) for m in models) if models else "none"
        parts.append(names if family == "ours" else f"{family} {names}")
    return "Models with runs: " + "; ".join(parts) + "."


def _short(label: str) -> str:
    """ "dan12-fz-s0" -> "fz-s0" for captions whose sentence already names the family."""
    family = label.split("-", 1)[0]
    return label[len(family) + 1 :] if family in ("dan12", "dan6") else label


def _all_legend_labels(groups: Mapping[str, Sequence[str]]) -> list[str]:
    out: list[str] = []
    for family in DAN_FAMILY_ORDER:
        out.extend(groups.get(family, []))
    return out


def _legend_labels_compact(
    registry: Mapping[str, mr.ModelEntry], groups: Mapping[str, Sequence[str]]
) -> list[str]:
    """One legend entry per `(family, role)`, not per model label: `_dan_style`'s colour/shade
    depends only on family and role, never on seed (seeds 1-2 differ from seed 0 only by being
    drawn as thin reference lines, per `_data_dan_yue`'s own docstring) -- so a full
    one-entry-per-seed legend repeats the same colour/marker up to three times. On the real store
    (2 "ours" + 6 dan12 + 6 dan6 = 14 models, +1 "rl (ref)") that overflowed `_top_legend`'s
    single row past the figure's right edge; this cuts it to one entry per role per family (6,
    +1 ref = 7) while every series still plots and keeps its own colour unchanged."""
    out: list[str] = []
    seen: set[tuple[str, str | None]] = set()
    for family in DAN_FAMILY_ORDER:
        for label in groups.get(family, []):
            entry = registry.get(label)
            role = entry.role if entry is not None else label
            key = (family, role)
            if key in seen:
                continue
            seen.add(key)
            out.append(label)
    return out


# =================================================================================== 1. Yue


def _data_dan_yue(
    store: Store,
    registry: Mapping[str, mr.ModelEntry],
    *,
    families: Sequence[str] = DAN_FAMILY_ORDER,
    B: int = DAN_B,
) -> ChartData:
    key, title = "dan_yue", "Pass@k on textbook_dev, by model family"
    groups = _model_groups(registry)
    ref_rl: P.Series | None = None
    panels: dict[str, ChartData] = {}
    any_data = False
    n_items: int | None = None
    k_max: dict[str, dict[str, int]] = {}

    for family in families:
        models = groups.get(family, [])
        series: list[P.Series] = []
        if models:
            data = yue_view.view(
                store,
                _cfg(
                    {
                        "sets": {"textbook_dev": "textbook_dev"},
                        "models": models,
                        "judge_id": _judge_for(family),
                        "B": B,
                    }
                ),
            )
            tdev = cast(
                "dict[str, Any] | None",
                {} if data.get("empty") else data["sets"].get("textbook_dev"),
            )
            if tdev:
                any_data = True
                n_items = n_items or next(
                    iter(cast("dict[str, int]", tdev["n_items"]).values()), None
                )
                curves = cast("dict[str, list[tuple[int, float, float, float]]]", tdev["curves"])
                for m in models:
                    curve = curves.get(m)
                    if not curve:
                        continue
                    k_max.setdefault(family, {})[m] = int(curve[-1][0])
                    pts = [(float(k), mean) for k, mean, _lo, _hi in curve]
                    # Seed 0 (or an unseeded "ours" model) carries a CI band; seeds 1-2 are thin
                    # reference lines without one (plan §6: "seed 0 solid with CI band, seeds 1-2
                    # thin lines").
                    entry = registry.get(m)
                    show_band = entry is None or entry.seed in (None, 0)
                    band = [(float(k), lo, hi) for k, _mean, lo, hi in curve] if show_band else None
                    series.append(P.Series(m, pts, "var(--s1)", band=band))
                    if m == "rl" and family == "ours":
                        ref_rl = P.Series(_ref_suffix("rl"), pts, "var(--s2)")
        if family != "ours" and ref_rl is not None:
            series.append(ref_rl)
        panels[family] = ChartData(key, title, "line", series=series)

    if not any_data:
        return _missing(key, title, "no textbook_dev runs yet for any Dan or 'ours' model")

    legend = _legend_labels_compact(registry, _with_data(panels))
    if ref_rl is not None:
        legend.append(ref_rl.name)
    k_note = "; ".join(
        f"{fam} " + ", ".join(f"{_short(m)} {k}" for m, k in ks.items())
        for fam, ks in k_max.items()
    )
    caption = (
        f"Pass@k on textbook_dev (n = {n_items if n_items is not None else '?'} theorems), small "
        "multiples by model family on one y scale: (a) ours (base, rl), (b) dan12 (fz = frozen "
        "Stage-1, darker; t1 = after the T1 ladder), (c) dan6, same roles. Seed 0 solid with a "
        f"95% item-level bootstrap band (B={DAN_B}), seeds 1 and 2 dashed and dotted, no band. "
        "Unbiased pass@k from each item's first 256 draws up to k = 256; above that from all "
        "draws, with the 1-(1-c/n)^k extrapolation past an item's own draw count. Each curve "
        f"stops at its model's largest per-item draw count ({k_note}). Our rl is overlaid on the "
        "dan12/dan6 panels as a grey dashed reference. Judged lean_v2 for dan12/dan6 (Dan's own "
        "sampling/RL-reward judge, a superset of nd_v2 on denotable proofs), nd_v2 for ours."
    )
    return ChartData(
        key,
        title,
        "line",
        caption=caption,
        # the grey rl reference repeats ours' rl: drawn on the Dan panels, not tabled twice
        series=[s for p in panels.values() for s in p.series if s is not ref_rl],
        opts={"legend": legend, "x_label": "k", "y_fmt": "pct"},
        extra={"panels": panels},
    )


# =============================================================================== 2. Families


def _data_dan_families(
    store: Store,
    registry: Mapping[str, mr.ModelEntry],
    *,
    families: Sequence[str] = DAN_FAMILY_ORDER,
) -> ChartData:
    """Figure 3 (`dash.summary._data_2_families`) per model family: one row per restricted
    family, x = mean written length of the items at n*, the largest size with pass@8 >= 50% over
    cells of >= 5 theorems unseen in *that model's own* training tiers. Reuses Figure 3's own
    `_family_horizon`/`_family_written_lines`, so the "ours" panel is Figure 3's rows (audit
    2026-10-01: this chart used to plot the best n* over all families as one bar per model, with
    no item filter, and zero bars for models with no run)."""
    from . import summary

    key = "dan_families"
    title = "Length horizon on restricted families (unseen theorems), by model family"
    suite = horizon_view.DEFAULT_SUITE
    groups = _model_groups(registry)
    panels: dict[str, ChartData] = {}
    n_runs: dict[str, list[str]] = {}
    for family in families:
        judge_id = _judge_for(family)
        models = groups.get(family, [])
        version = default_version(store, suite=suite, models=models, judge_id=judge_id) or "v1"
        problems = problems_by_pid(store, suite, version)
        pids_by_group = summary._family_group_pids(problems)  # noqa: SLF001
        points_by_label: dict[str, list[tuple[str, float | None, str, bool]]] = {}
        for m in models:
            run_id = latest_run_id(store, suite=suite, model=m, version=version, judge_id=judge_id)
            if run_id is None:
                continue
            n_runs.setdefault(family, []).append(m)
            seen = mr.seen_lookup(store, suite, version, m, problems, registry=registry)
            for group in sorted(pids_by_group):
                pids = [p for p in pids_by_group[group] if not seen.get(p, problems[p].seen_train)]
                res = summary._family_horizon(  # noqa: SLF001
                    store, suite, version, m, pids, problems, judge_id=judge_id, B=0
                )
                if res is None:
                    continue
                n_star, censored = res
                label = "ILTP-style" if group.startswith("ILTP-style") else group
                if any(n == m for n, *_ in points_by_label.get(label, [])):
                    label = group  # a second ILTP sub-schema keeps its own row
                lines = (
                    None
                    if n_star is None
                    else summary._family_written_lines(pids, problems, n_star)  # noqa: SLF001
                )
                points_by_label.setdefault(label, []).append(
                    (m, lines, "var(--s1)", censored and n_star is not None)
                )
        rows = [P.DotRow(label, pts) for label, pts in points_by_label.items()]
        panels[family] = ChartData(key, title, "dot", rows=rows)

    all_rows = [r for p in panels.values() for r in p.rows]
    if not all_rows:
        return _missing(key, title, "no families runs yet for any Dan or 'ours' model")

    # One row order for every panel (rows line up), most-reached family first, as Figure 3.
    best: dict[str, float] = {}
    merged: dict[str, list[tuple[str, float | None, str, bool]]] = {}
    for r in all_rows:
        best[r.label] = max([best.get(r.label, 0.0)] + [v or 0.0 for _n, v, _c, _z in r.points])
        merged.setdefault(r.label, []).extend(r.points)
    order = sorted(best, key=lambda label: (-best[label], label))
    x_hi = max([*best.values(), 1.0])
    for p in panels.values():
        p.extra.update({"row_order": order, "x_hi": x_hi})

    def _reach(fam: str) -> str:
        bits = []
        for m in n_runs.get(fam, []):
            pts = [(r.label, v, z) for r in panels[fam].rows for n, v, _c, z in r.points if n == m]
            top = max(pts, key=lambda t: t[1] or 0.0, default=None)
            if top is None:
                bits.append(f"{_short(m)} has no size cell with enough unseen theorems")
            elif not top[1]:
                bits.append(f"{_short(m)} fails at the smallest unseen size in every family")
            else:
                bits.append(
                    f"{_short(m)} reaches {'at least ' if top[2] else ''}{top[1]:.0f} lines "
                    f"({top[0]})"
                )
        return f"{fam}: " + ", ".join(bits) if bits else ""

    reach = "; ".join(x for x in (_reach(f) for f in DAN_FAMILY_ORDER) if x)
    caption = (
        "Horizon of each restricted family, as Figure 3, per model family: the mean written length "
        "(lines) of the items at n*, the largest family size n at which pass@8 is still at least "
        f"50%, over size cells with >={summary._MIN_ITEMS_PER_N} theorems unseen in that model's "  # noqa: SLF001
        "own training tiers (ours: base's pretraining set and the RL pool; dan12/dan6: Dan's "
        "Stage-1 set for that cap plus the T1 ladder's RL targets). A hollow marker at 0 means the "
        "model fails at the smallest unseen size; an arrow marks a censored horizon (never below "
        f"50% in the tested range, so a lower bound). Point estimates. {_runs_note(n_runs)} "
        f"Judged lean_v2 for dan12/dan6, nd_v2 for ours. {reach}."
    )
    return ChartData(
        key,
        title,
        "dot",
        caption=caption,
        rows=[P.DotRow(label, merged[label]) for label in order],
        opts={
            "legend": _legend_labels_compact(registry, n_runs),
            "legend_kind": "marker",
            "x_label": "written lines at n*",
            "none_label": "fails at smallest size",
        },
        extra={"panels": panels, "row_order": order},
    )


# ================================================================================ 3. Laws


def _pooled_curve_vs_s(data: Mapping[str, JSON], model: str) -> list[tuple[float, float, int]]:
    """`(s, pass@1, n items)` for one `model`, pooled per item across `subst.view`'s schemas at
    each `s`, cells with fewer than Figure 5's minimum dropped -- Figure 5's own estimator
    (`dash.summary._data_5_subst`; audit 2026-10-01: this used to average the per-schema means,
    which weights s=0's single item per schema like s=2's ~30, and kept the 8-item s=0 cell)."""
    from . import summary

    schemas = cast("dict[str, dict[str, Any]]", data.get("schemas") or {})
    by_s: dict[float, list[float]] = {}  # s -> [sum pass1 * n, n]
    for per_model in schemas.values():
        info = per_model.get(model)
        if info is None:
            continue
        for row in cast("list[dict[str, Any]]", info.get("curve_vs_s") or []):
            s_val, p1, n = row.get("s"), row.get("pass1"), int(row.get("n_items") or 0)
            if s_val is None or p1 is None or p1 != p1:
                continue
            acc = by_s.setdefault(float(s_val), [0.0, 0.0])
            acc[0] += float(p1) * n
            acc[1] += n
    return [
        (s_val, t / n, int(n))
        for s_val, (t, n) in sorted(by_s.items())
        if n >= summary._MIN_ITEMS_PER_BIN  # noqa: SLF001
    ]


def _data_dan_laws(
    store: Store,
    registry: Mapping[str, mr.ModelEntry],
    *,
    families: Sequence[str] = DAN_FAMILY_ORDER,
) -> ChartData:
    from . import summary

    key, title = "dan_laws", "Pass@1 vs substitution size s (pooled over law schemas)"
    groups = _model_groups(registry)
    panels: dict[str, ChartData] = {}
    any_data = False
    n_unseen: dict[str, int] = {}
    for family in families:
        models = groups.get(family, [])
        series: list[P.Series] = []
        if models:
            data = subst_view.view(store, _cfg({"models": models, "judge_id": _judge_for(family)}))
            if not data.get("empty"):
                for m in models:
                    pts = _pooled_curve_vs_s(data, m)
                    if pts:
                        any_data = True
                        series.append(P.Series(m, [(x, y) for x, y, _n in pts], "var(--s1)"))
                        n_unseen[m] = sum(
                            int(d[m]["n_items"])
                            for d in cast("dict[str, dict[str, Any]]", data["schemas"]).values()
                            if m in d
                        )
        panels[family] = ChartData(key, title, "line", series=series)
    if not any_data:
        return _missing(key, title, "no laws runs yet for any Dan or 'ours' model")
    n_note = ", ".join(f"{_short(m)} {n}" for m, n in n_unseen.items())
    caption = (
        "Pass@1 against the substitution size s (connectives per metavariable) for instances of "
        "known laws, as Figure 5, per model family on one y scale: pooled per item over the 8 "
        "schemas at each s, s cells with "
        f"<{summary._MIN_ITEMS_PER_BIN} items omitted (s=0 has one item per schema). Instances "  # noqa: SLF001
        "seen in each model's own training tiers are excluded (ours: our tiers; dan12/dan6: "
        f"Dan's), leaving {n_note} instances. {_runs_note(_with_data(panels))} Point estimates. "
        "Judged lean_v2 for dan12/dan6, nd_v2 for ours."
    )
    return ChartData(
        key,
        title,
        "line",
        caption=caption,
        series=[s for p in panels.values() for s in p.series],
        opts={
            "legend": _legend_labels_compact(registry, _with_data(panels)),
            "x_label": "s",
            "y_fmt": "pct",
        },
        extra={"panels": panels},
    )


# ================================================================================= 4. Glue


def _glue_ratio(
    store: Store, model: str, pids: Sequence[str], judge_id: str
) -> dict[str, Any] | None:
    """Figure 4's R for one operator and model (`dash.summary._data_3_glue`: parts' pass@1 from
    the parts as they appear inside the join, else the pool items; mastery 0.5; B=1000), without
    Figure 4's requirement of a `parts_pool` run, which Dan's models do not have (their
    denominators all come from `glue_parts_renamed`)."""
    suite, parts_suite = glue_view.DEFAULT_SUITE, glue_view.DEFAULT_PARTS_SUITE
    version = default_version(store, suite=suite, models=[model], judge_id=judge_id) or "v1"
    run_id = latest_run_id(store, suite=suite, model=model, version=version, judge_id=judge_id)
    if run_id is None:
        return None
    parts_version = (
        default_version(store, suite=parts_suite, models=[model], judge_id=judge_id) or "v1"
    )
    parts_run_id = latest_run_id(
        store, suite=parts_suite, model=model, version=parts_version, judge_id=judge_id
    )
    problems = problems_by_pid(store, suite, version)
    joined_items = items_by_pid(store, run_id)
    part_items = items_by_pid(store, parts_run_id) if parts_run_id else {}
    renamed_items = glue_view.renamed_part_items(store, model, judge_id=judge_id)
    groups: list[tuple[ItemResult, list[ItemResult]]] = []
    for pid in pids:
        item = joined_items.get(pid)
        if item is None or pid not in problems:
            continue
        found = glue_view.part_results(pid, problems[pid], part_items, renamed_items)
        if found is not None:
            groups.append((item, found[0]))
    if not groups:
        return None
    return dict(compositionality_ratio(groups, mastery=0.5, B=1000, seed=0))


def _data_dan_glue(
    store: Store,
    registry: Mapping[str, mr.ModelEntry],
    *,
    families: Sequence[str] = DAN_FAMILY_ORDER,
) -> ChartData:
    from . import summary

    key, title = "dan_glue", "Compositionality ratio R per operator, by model family"
    groups = _model_groups(registry)
    suite = glue_view.DEFAULT_SUITE
    problems = problems_by_pid(store, suite, default_version(store, suite=suite) or "v1")
    pids_by_op: dict[str, list[str]] = {}
    for pid, p in problems.items():
        op = summary._OP_GROUP.get(cast(str, p.tags.get("op", "")))  # noqa: SLF001
        if op:
            pids_by_op.setdefault(op, []).append(pid)
    ops = [op for op in summary._OP_ORDER if op in pids_by_op]  # noqa: SLF001
    panels: dict[str, ChartData] = {}
    any_data = False
    for family in families:
        bars: list[P.Bar] = []
        for op in ops:
            for m in groups.get(family, []):
                comp = _glue_ratio(store, m, pids_by_op[op], _judge_for(family))
                if comp is None:
                    continue
                any_data = True
                lo, hi = comp["ci"]
                bars.append(P.Bar(f"{op}: {m}", float(comp["R"]), lo, hi))
        panels[family] = ChartData(
            key, title, "bar", bars=bars, opts={"y_fmt": "f3"}, extra={"groups": ops}
        )
    if not any_data:
        return _missing(key, title, "no glue runs yet for any Dan or 'ours' model")
    caption = (
        "Compositionality ratio R per glue operator, as Figure 4, per model family on one y "
        "scale: R = joined pass@1 / product of mastered parts' pass@1, pooled per operator over "
        "items whose parts all have pass@1 >= 0.5, each part sampled as it appears inside the "
        "join (glue_parts_renamed, every model's own run); "
        + "; ".join(f"{op} n={len(pids_by_op[op])}" for op in ops)
        + " joined items (CASE pools the dilemma and same-conclusion joins). Dotted line: R = 1. "
        "Error bars: 95% item-level bootstrap CIs (B=1000). "
        f"{_runs_note(_with_data(panels))} Judged lean_v2 for dan12/dan6, nd_v2 for ours."
    )
    return ChartData(
        key,
        title,
        "bar",
        caption=caption,
        bars=[b for p in panels.values() for b in p.bars],
        opts={"legend": _legend_labels_compact(registry, _with_data(panels)), "y_fmt": "f3"},
        extra={"panels": panels},
    )


# =============================================================================== 5. Padding


def _data_dan_padding(
    store: Store,
    registry: Mapping[str, mr.ModelEntry],
    *,
    families: Sequence[str] = DAN_FAMILY_ORDER,
) -> ChartData:
    from . import summary

    key, title = "dan_padding", "Pass@1 vs written lines W, padded proofs, by model family"
    suite = padding_view.DEFAULT_SUITE
    groups = _model_groups(registry)
    panels: dict[str, ChartData] = {}
    any_data = False
    n_items: int | None = None
    for family in families:
        models = groups.get(family, [])
        judge_id = _judge_for(family)
        version = default_version(store, suite=suite, models=models, judge_id=judge_id) or "v1"
        series: list[P.Series] = []
        for m in models:
            # Figure 6's own curve: pooled over pad types, each unpadded (m=0) theorem once
            # (audit 2026-10-01: pooling the per-pad-type curves counted those three times).
            curve = summary._padded_curve_dedup(  # noqa: SLF001
                store, suite, version, m, judge_id=judge_id
            )
            if curve:
                any_data = True
                n_items = n_items or sum(n for _w, _p, n in curve)
                series.append(P.Series(m, [(w, p) for w, p, _n in curve], "var(--s1)"))
        panels[family] = ChartData(key, title, "line", series=series)
    if not any_data:
        return _missing(key, title, "no padding runs yet for any Dan or 'ours' model")
    caption = (
        "Pass@1 against written proof length W when some of the lines are padding, as Figure 6's "
        f"padded curves, per model family on one y scale: {n_items} distinct padded items pooled "
        "across pad types P1-disjoint, P1-shared and P2, each unpadded theorem counted once. The "
        "unpadded bases are short theorems our rl had mastered (pass@1 >= 0.7 in its parts_pool "
        "run), not chosen for Dan's models, so the low-W end favours our rl by construction. "
        f"{_runs_note(_with_data(panels))} Point estimates. Judged lean_v2 for dan12/dan6, nd_v2 "
        "for ours."
    )
    return ChartData(
        key,
        title,
        "line",
        caption=caption,
        series=[s for p in panels.values() for s in p.series],
        opts={
            "legend": _legend_labels_compact(registry, _with_data(panels)),
            "x_label": "W",
            "y_fmt": "pct",
        },
        extra={"panels": panels},
    )


# ========================================================================= 6. Prefix (state)


def _data_dan_prefix_state(
    store: Store,
    registry: Mapping[str, mr.ModelEntry],
    *,
    families: Sequence[str] = DAN_FAMILY_ORDER,
) -> ChartData:
    from . import summary

    key = "dan_prefix_state"
    title = "Prefix completion by lines left, by model family"
    groups = _model_groups(registry)
    panels: dict[str, ChartData] = {}
    any_data = False
    n_items: dict[str, int] = {}
    for family in families:
        models = groups.get(family, [])
        series: list[P.Series] = []
        if models:
            suite = "prefix_state" if family != "ours" else "prefix"
            data = prefix_view.view(
                store, _cfg({"suite": suite, "models": models, "judge_id": _judge_for(family)})
            )
            if not data.get("empty"):
                per_model = cast("dict[str, dict[str, Any]]", data["models"])
                for m in models:
                    info = per_model.get(m)
                    if info is None:
                        continue
                    n_items.setdefault(suite, int(info.get("n_items") or 0))
                    bucket = cast("dict[str, Any]", info.get("success_by_lines_left") or {})
                    # Figure 7's bins: fewer than 20 items are omitted (audit 2026-10-01).
                    pts = sorted(
                        (float(k), float(v["pass1"]))
                        for k, v in bucket.items()
                        if v.get("pass1") == v.get("pass1")
                        and int(v.get("n_items") or 0) >= summary._MIN_PREFIX_BIN_ITEMS  # noqa: SLF001
                    )
                    if pts:
                        any_data = True
                        series.append(P.Series(m, pts, "var(--s1)"))
        panels[family] = ChartData(key, title, "line", series=series)
    if not any_data:
        return _missing(key, title, "no prefix/prefix_state runs yet for any Dan or 'ours' model")
    caption = (
        "Prefix-completion success against lines left to write (L - j), as Figure 7, per model "
        "family on one y scale: pooled pass@1 by lines left, bins with "
        f"<{summary._MIN_PREFIX_BIN_ITEMS} items omitted. Ours reads the ND-line prefix suite "  # noqa: SLF001
        f"(n={n_items.get('prefix', '?')}), dan12/dan6 the state-env prefix_state suite "
        f"(n={n_items.get('prefix_state', '?')}): the same (proof, j) items with the first j "
        "lines given as a forced action prefix (DANSTATE_PLAN.md §3). "
        f"{_runs_note(_with_data(panels))} Point estimates. Judged lean_v2 for dan12/dan6, nd_v2 "
        "for ours."
    )
    return ChartData(
        key,
        title,
        "line",
        caption=caption,
        series=[s for p in panels.values() for s in p.series],
        opts={
            "legend": _legend_labels_compact(registry, _with_data(panels)),
            "x_label": "lines left",
            "y_fmt": "pct",
        },
        extra={"panels": panels},
    )


# =============================================================================== 7. Skills


def _data_dan_skills(
    store: Store,
    registry: Mapping[str, mr.ModelEntry],
    *,
    families: Sequence[str] = DAN_FAMILY_ORDER,
) -> ChartData:
    """Pass@1 by `skill_group` (the 4-valued classical/case_split stratum `skill_pool_sub`'s own
    builder tags every item with), by model family -- direct runs only (module docstring: this
    suite is sampled directly, unlike legacy `skill_pool`, so `analysis.skills.view`'s join
    machinery does not apply)."""
    key, title = "dan_skills", "Pass@1 by skill group on skill_pool_sub, by model family"
    suite = "skill_pool_sub"
    groups = _model_groups(registry)
    panels: dict[str, ChartData] = {}
    any_data = False
    for family in families:
        models = groups.get(family, [])
        bars: list[P.Bar] = []
        if models:
            judge_id = _judge_for(family)
            problems = problems_by_pid(store, suite, "v1")
            present = models_present(store, suite=suite, version="v1", judge_id=judge_id)
            for m in [m for m in models if m in present]:
                run_id = latest_run_id(store, suite=suite, model=m, version="v1", judge_id=judge_id)
                if run_id is None:
                    continue
                items = items_by_pid(store, run_id)
                by_group: dict[str, list[tuple[int, int]]] = {}
                for pid, it in items.items():
                    p = problems.get(pid)
                    if p is None or it.n <= 0:
                        continue
                    group = str(p.tags.get("skill_group", "plain"))
                    by_group.setdefault(group, []).append((it.n, it.c))
                for group, ncs in sorted(by_group.items()):
                    vals = [pass_at_k(n, c, 1) for n, c in ncs]
                    vals = [v for v in vals if v == v]
                    if not vals:
                        continue
                    any_data = True
                    bars.append(P.Bar(f"{group}: {m}", sum(vals) / len(vals)))
        panels[family] = ChartData(key, title, "bar", bars=bars)
    if not any_data:
        return _missing(key, title, f"no {suite!r} runs yet for any Dan or 'ours' model")
    problems = problems_by_pid(store, suite, "v1")
    sizes: dict[str, int] = {}
    for p in problems.values():
        g = str(p.tags.get("skill_group", "plain"))
        sizes[g] = sizes.get(g, 0) + 1
    caption = (
        "Pass@1 by skill_group (the stratum skill_pool_sub was drawn on, DANSTATE_PLAN.md §3: up "
        "to 25 items per (L, group) cell, fewer where skill_pool is thin) on this "
        f"{len(problems)}-item stratified subsample of skill_pool ("
        + ", ".join(f"{g} {n}" for g, n in sorted(sizes.items()))
        + "), per model family on one y scale; mean pass@1 over each group's items. "
        f"{_runs_note(_with_data(panels))} Point estimates. Judged lean_v2 for dan12/dan6, nd_v2 "
        "for ours."
    )
    return ChartData(
        key,
        title,
        "bar",
        caption=caption,
        bars=[b for p in panels.values() for b in p.bars],
        opts={"legend": _legend_labels_compact(registry, _with_data(panels)), "y_fmt": "pct"},
        extra={"panels": panels},
    )


# ============================================================================= 8. Held-out


def _data_dan_heldout(store: Store, registry: Mapping[str, mr.ModelEntry]) -> ChartData:
    key, title = "dan_heldout", "Held-out greedy accuracy"
    all_models = _all_legend_labels(_model_groups(registry))
    data = heldout_view.view(store, _cfg({"models": all_models, "judge_id": DAN_JUDGE_ID}))
    if data.get("empty"):
        return _missing(key, title, str(data.get("reason", "no dan_heldout runs yet")))
    per_model = cast("dict[str, dict[str, Any]]", data["models"])
    bars = [
        P.Bar(m, float(info["accuracy"]), info["ci"][0], info["ci"][1])
        for m, info in per_model.items()
    ]
    n_items = next((info["n_items"] for info in per_model.values()), "?")
    solved = ", ".join(
        f"{m} {info['n_solved']:,}/{info['n_items']:,} ({P.fmt(info['accuracy'], 'pct')})"
        for m, info in per_model.items()
    )
    caption = (
        f"Fraction of Dan's {n_items}-item held-out set solved on one greedy draw (k=1, T=0), "
        f"with a 95% Clopper-Pearson CI, per model: {solved}. Gate V5's own expectation: T1 s0 at "
        "least 4,940/5,000, Fz s0 4,632 +- 25/5,000. Judged lean_v2."
    )
    return ChartData(key, title, "bar", caption=caption, bars=bars, opts={"y_fmt": "pct"})


# =============================================================================== 9. Judges


#: Caption wording for `analysis.judges.BUCKETS` (what each heuristic bucket actually tests).
_JUDGE_BUCKET_TEXT: dict[str, str] = {
    "missing_premise_restatement": "nd_v2 premise error (premises not restated)",
    "elim_on_not_a": "contains .elim",
    "other": "other",
}


def _data_dan_judges(store: Store, registry: Mapping[str, mr.ModelEntry]) -> ChartData:
    key, title = "dan_judges", "Where lean_v2 and nd_v2 disagree"
    all_models = mr.labels_in_family(registry, "dan12") + mr.labels_in_family(registry, "dan6")
    data = judges_view.view(store, _cfg({"models": all_models}))
    if data.get("empty"):
        return _missing(key, title, str(data.get("reason", "no rejudged runs yet")))
    per_model = cast("dict[str, dict[str, Any]]", data["models"])
    order = [m for m in all_models if m in per_model]
    # Agreement is >= 99.8% for every model, so the bars show the disagreement: lean-only
    # accepts per attempt (nd-only accepts, the other direction, are given in the caption).
    bars = [P.Bar(m, float(per_model[m]["lean_only_rate"])) for m in order]
    n_tot = sum(int(per_model[m]["n_attempts"]) for m in order)
    agree_tot = sum(int(per_model[m]["agree_count"]) for m in order)
    lean_tot = sum(int(per_model[m]["lean_only_count"]) for m in order)
    nd_tot = sum(int(per_model[m]["nd_only_count"]) for m in order)
    breakdown: dict[str, int] = {}
    for m in order:
        for name, b in cast("dict[str, Any]", per_model[m]["lean_only_breakdown"]).items():
            breakdown[name] = breakdown.get(name, 0) + int(b["count"])
    per_model_txt = ", ".join(
        f"{m} {per_model[m]['lean_only_count']}/{per_model[m]['n_attempts']:,}"
        for m in order
        if per_model[m]["lean_only_count"]
    )
    caption = (
        "Share of sampled attempts that lean_v2 (Dan's headline judge) accepts and nd_v2 rejects, "
        "per model, with nd_v2 applied to the same attempts afterwards (`rejudge --judge nd`) and "
        f"matched attempt for attempt over every rejudged run of the model. All models: "
        f"{n_tot:,} attempts, {P.fmt(agree_tot / n_tot if n_tot else float('nan'), 'pct')} "
        f"agreement, {lean_tot:,} lean-only and {nd_tot:,} nd-only accepts"
        + (f" ({per_model_txt}; every other model 0)" if per_model_txt else "")
        + ". Lean-only accepts by a heuristic bucket (nd_v2's rejection reason, then whether the "
        "Lean text contains `.elim`, which also matches Or.elim): "
        + ", ".join(f"{_JUDGE_BUCKET_TEXT.get(name, name)} {c}" for name, c in breakdown.items())
        + "; for comparison Dan's own LEAN_JUDGE.md reports 41.7% missing premise restatement, "
        "25.9% .elim on not-A and 6.2% other from a manual read."
    )
    return ChartData(key, title, "bar", caption=caption, bars=bars, opts={"y_fmt": "pct"})


# ====================================================================== cross-family table


#: (family, role) -> (seed 0's model label, every seed's label) for the cross-family table;
#: "ours" has a single unseeded model per role.
def _crossfamily_groups(registry: Mapping[str, mr.ModelEntry]) -> dict[tuple[str, str], list[str]]:
    out: dict[tuple[str, str], list[str]] = {("ours", "base"): ["base"], ("ours", "rl"): ["rl"]}
    for family in ("dan12", "dan6"):
        out[(family, "fz")] = mr.seed_labels(registry, family, "base")
        out[(family, "t1")] = mr.seed_labels(registry, family, "rl")
    return out


#: The k columns of Table D1: pass@1, and pass@256, the k of Dan's own textbook counts (every
#: model's first sampling round has 256 draws per item, so both are unbiased, never extrapolated).
_TABLE_KS: tuple[int, ...] = (1, 256)


def _pass_at_by_model(
    store: Store, models: Sequence[str], judge_id: str
) -> dict[str, dict[int, float]]:
    """`{model: {k: pass@k}}` on textbook_dev for `_TABLE_KS`, from Figure D1's own view."""
    if not models:
        return {}
    data = yue_view.view(
        store,
        _cfg(
            {
                "sets": {"textbook_dev": "textbook_dev"},
                "models": list(models),
                "judge_id": judge_id,
                "B": 10,
            }
        ),
    )
    if data.get("empty"):
        return {}
    tdev = cast("dict[str, Any] | None", data["sets"].get("textbook_dev"))
    if not tdev:
        return {}
    curves = cast("dict[str, list[tuple[int, float, float, float]]]", tdev["curves"])
    out: dict[str, dict[int, float]] = {}
    for m in models:
        by_k = {int(k): mean for k, mean, _lo, _hi in curves.get(m) or []}
        if by_k:
            out[m] = {k: by_k[k] for k in _TABLE_KS if k in by_k}
    return out


def dan_crossfamily_table(
    store: Store, registry: Mapping[str, mr.ModelEntry]
) -> tuple[list[str], list[list[str]], str] | None:
    """`(headers, rows, caption)` for the cross-family textbook_dev table (plan §2/§6: Lean for
    every model, nd_v2 in parentheses, seed ranges), or `None` if no group has a run yet. Ours
    are judged lean_v2 here too, from their `rejudge --judge lean` runs of the same samples (audit
    2026-10-01: the Lean column used to show ours' nd_v2 value)."""
    groups = _crossfamily_groups(registry)
    lean: dict[str, dict[int, float]] = {}
    nd: dict[str, dict[int, float]] = {}
    for models in groups.values():
        lean.update(_pass_at_by_model(store, models, DAN_JUDGE_ID))
        nd.update(_pass_at_by_model(store, models, OURS_JUDGE_ID))

    def _mean(vals: Sequence[float]) -> float:
        return sum(vals) / len(vals)

    headers = ["family", "role", *(f"pass@{k}, lean_v2 (nd_v2)" for k in _TABLE_KS)]
    headers.append("pass@1 seed range")
    rows: list[list[str]] = []
    any_row = False
    for (family, role), models in groups.items():
        with_run = [m for m in models if m in lean]
        if not with_run:
            rows.append([family, role, *("not yet run" for _k in _TABLE_KS), "-"])
            continue
        any_row = True
        cells = []
        for k in _TABLE_KS:
            lv = [lean[m][k] for m in with_run if k in lean[m]]
            ndv = [nd[m][k] for m in with_run if m in nd and k in nd[m]]
            lean_cell = f"{100 * _mean(lv):.1f}%" if lv else "-"
            nd_cell = f"{100 * _mean(ndv):.1f}%" if len(ndv) == len(lv) and ndv else "not rejudged"
            cells.append(f"{lean_cell} ({nd_cell})")
        p1 = [lean[m][1] for m in with_run if 1 in lean[m]]
        seed_range = f"{100 * min(p1):.1f}-{100 * max(p1):.1f}%" if len(p1) > 1 else "-"
        label = role if len(with_run) == 1 else f"{role} ({len(with_run)} seeds)"
        rows.append([family, label, *cells, seed_range])
    if not any_row:
        return None
    caption = (
        "Textbook_dev (58 theorems) pass@1 and pass@256, unbiased from each item's first 256 draws "
        "(Figure D1's estimator), judged lean_v2 with nd_v2 in parentheses; ours' lean_v2 values "
        "are the same samples rejudged with Lean. dan12/dan6 rows are means over the seeds with a "
        "run, with the range of pass@1 across seeds. Dan's own k=256 counts on these 58 (his "
        "best-state summary): fz best12 27/22/21, t1 best12 42/44/42, fz best6 13/13/15, "
        "t1 best6 34/33/26 of 58."
    )
    return headers, rows, caption


def _crossfamily_html(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    th = "".join(f"<th>{P.esc(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{P.esc(c)}</td>" for c in r) + "</tr>" for r in rows)
    return f'<table class="booktabs"><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>'


def _tex_esc(text: str) -> str:
    return text.replace("_", r"\_").replace("%", r"\%")


def _crossfamily_tex(headers: Sequence[str], rows: Sequence[Sequence[str]], caption: str) -> str:
    lines = [
        r"\begin{table}[t]",
        r"  \caption{" + _tex_esc(caption) + "}",
        r"  \label{tab:dan_crossfamily}",
        r"  \centering",
        r"  \begin{tabular}{" + "l" * len(headers) + "}",
        r"    \toprule",
        # headers are escaped too ("lean_v2" would otherwise break the LaTeX build)
        "    " + " & ".join(_tex_esc(h) for h in headers) + r" \\",
        r"    \midrule",
    ]
    for row in rows:
        lines.append("    " + " & ".join(_tex_esc(c) for c in row) + r" \\")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


# ============================================================================== assembly


#: Fixed order of the nine `ChartData`-based Dan cards (plan §6); the cross-family table is
#: rendered separately (`dan_crossfamily_table`, not a `ChartData` -- see module docstring).
DAN_ORDER: tuple[str, ...] = (
    "dan_yue",
    "dan_families",
    "dan_laws",
    "dan_glue",
    "dan_padding",
    "dan_prefix_state",
    "dan_skills",
    "dan_heldout",
    "dan_judges",
)

_BUILDERS: dict[str, Any] = {
    "dan_yue": _data_dan_yue,
    "dan_families": _data_dan_families,
    "dan_laws": _data_dan_laws,
    "dan_glue": _data_dan_glue,
    "dan_padding": _data_dan_padding,
    "dan_prefix_state": _data_dan_prefix_state,
    "dan_skills": _data_dan_skills,
    "dan_heldout": _data_dan_heldout,
    "dan_judges": _data_dan_judges,
}


def dan_chart_data(
    store: Store, *, registry: Mapping[str, mr.ModelEntry] | None = None
) -> list[ChartData]:
    """Every Dan `ChartData`, in `DAN_ORDER`, computed once and shared by the summary section,
    `dan.html` and `figures --with-dan` (the same "compute once, draw everywhere" contract
    `dash.summary.chart_data` uses for the paper's own twelve)."""
    reg = registry if registry is not None else mr.load_registry()
    return [_BUILDERS[key](store, reg) for key in DAN_ORDER]


def dan_numbering(charts: Sequence[ChartData]) -> dict[str, str]:
    """`key -> "Figure Dn"`, in `DAN_ORDER` -- a separate numbering sequence from the paper's own
    `dash.summary.numbering` (plan §6's figures are a flag-gated appendix, not part of the
    paper's twelve)."""
    out: dict[str, str] = {}
    n = 0
    by_key = {c.key: c for c in charts}
    for key in DAN_ORDER:
        if key not in by_key:
            continue
        n += 1
        out[key] = f"Figure D{n}"
    return out


def render_dan_section(store: Store, *, registry: Mapping[str, mr.ModelEntry] | None = None) -> str:
    """The summary page's "State-env models" section (plan §6: a new section, behind
    `--with-dan`): the cross-family table followed by the nine numbered Dan figures, reusing the
    *existing* `dash.summary.render_card` dispatch (extended additively to also check
    `figures.plots.DAN_FIGURES`) for every figure, so nothing here duplicates its rendering.
    """
    from . import summary

    reg = registry if registry is not None else mr.load_registry()
    charts = dan_chart_data(store, registry=reg)
    nums = dan_numbering(charts)
    cards = "".join(summary.render_card(cd, nums.get(cd.key)) for cd in charts)
    xf = dan_crossfamily_table(store, reg)
    table_html = ""
    if xf is not None:
        headers, rows, caption = xf
        table_html = (
            '<figure class="card chart-card table-card" id="tab-dan-crossfamily">'
            '<figcaption class="above"><span class="fignum">Table D1:</span> '
            f"<b>{P.esc('Cross-family headline results')}.</b> {P.esc(caption)}</figcaption>"
            f'<div class="tw">{_crossfamily_html(headers, rows)}</div></figure>'
        )
    intro = (
        "Dan Pandori's best-state checkpoints (DANSTATE_PLAN.md): a frozen Stage-1 model (fz) "
        "and the same checkpoint after an 8-round T1 RL ladder (t1), at train caps 6 and 12, "
        "three seeds each. Headline judge: lean_v2 (his own sampling protocol and RL reward, a "
        "superset of nd_v2 on denotable proofs); nd_v2 alongside where available (the two agree "
        "on 99.9% of attempts, Figure D9). Textbook pass@k was run for all 12 models; the other "
        "benchmarks for the cap-12 seed-0 pair only, so most dan6 panels say 'not available'. "
        "Theorems in Dan's training data are excluded where a figure says 'unseen'."
    )
    return (
        '<section id="dan-section"><h2><span class="secnum">D</span>State-env models '
        f'(Dan)</h2><p class="intro">{P.esc(intro)}</p>{table_html}{cards}</section>'
    )


def render_dan_page(store: Store, *, registry: Mapping[str, mr.ModelEntry] | None = None) -> str:
    """A standalone `dan.html` detail page: the same section `render_dan_section` builds for the
    summary page, wrapped in the summary page's own CSS/JS (`dash.summary`'s private `_CSS`/`_JS`
    -- reused rather than duplicated, since this page is visually a continuation of it) with a
    link back to `summary.html`."""
    from . import summary

    section = render_dan_section(store, registry=registry)
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>ND Capability Benchmarks: State-env models</title><style>{summary._CSS}</style>"  # noqa: SLF001
        '</head><body><div class="col"><header><h1>Dan\'s state-env models</h1>'
        '<p class="lede">Detail page for every "State-env models" chart; '
        '<a href="summary.html">back to the main summary</a>.</p></header>'
        f"<main>{section}</main></div><script>{summary._JS}</script></body></html>"  # noqa: SLF001
    )


def write_dan_page(
    store: Store, out_dir: Path, *, registry: Mapping[str, mr.ModelEntry] | None = None
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "dan.html"
    path.write_text(render_dan_page(store, registry=registry))
    return path


def build_summary_with_dan(store: Store, out_dir: Path, detail_href: str | None = None) -> Path:
    """`cli.py`'s `dash --summary --with-dan` entry point: `dash.summary.build_summary`'s own
    page (unchanged) plus WP-D5's "State-env models" section inserted onto it and a standalone
    `out_dir/dan.html` detail page. Kept here (not as a `with_dan` flag on `dash.summary.
    build_summary` itself) so `dash.summary` has no import-time dependency on this module --
    the two would otherwise form an import cycle, since this module already imports `dash.summary`
    to reuse `render_card`."""
    from . import summary

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    registry = mr.load_registry()
    dan_section = render_dan_section(store, registry=registry)
    write_dan_page(store, out_dir, registry=registry)
    return summary.build_summary(store, out_dir, detail_href, dan_section=dan_section)
