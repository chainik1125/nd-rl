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
panel per `figures.plots.DAN_FAMILY_ORDER` entry (`"ours" | "dan12" | "dan6"`); a family with no
matching runs yet (true of every family but `dan12`/`ours` as of 2026-10-01, when only gates
V5/V6 have landed) renders as a clean "not available" panel -- `figures.plots.dan_panel_figure`
never crashes on missing data, and neither does anything in this module.

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
from ..analysis._common import items_by_pid, latest_run_id, models_present, problems_by_pid
from ..core.types import JSON
from ..figures.plots import DAN_FAMILY_ORDER
from ..metrics import pass_at_k
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


def _data_dan_yue(store: Store, registry: Mapping[str, mr.ModelEntry]) -> ChartData:
    key, title = "dan_yue", "Pass@k on textbook_dev, by model family"
    groups = _model_groups(registry)
    ref_rl: P.Series | None = None
    panels: dict[str, ChartData] = {}
    any_data = False
    n_items: int | None = None

    for family in DAN_FAMILY_ORDER:
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
                        "B": DAN_B,
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

    legend = _legend_labels_compact(registry, groups)
    if ref_rl is not None:
        legend.append(ref_rl.name)
    caption = (
        f"Pass@k on textbook_dev (n = {n_items if n_items is not None else '?'} theorems), small "
        "multiples by model family: (a) ours (base, rl), (b) dan12 (fz = frozen Stage-1, t1 = "
        "after the T1 ladder; seeds 0-2, seed 0 with a 95% item-level bootstrap band, B="
        f"{DAN_B}), (c) dan6, same roles/seeds. Our rl is overlaid on the dan12/dan6 panels as a "
        "grey dashed reference. Judged lean_v2 for dan12/dan6 (Dan's own sampling/RL-reward "
        "judge, a superset of nd_v2 on denotable proofs), nd_v2 for ours."
    )
    return ChartData(
        key,
        title,
        "line",
        caption=caption,
        series=[s for p in panels.values() for s in p.series],
        opts={"legend": legend, "x_label": "k", "y_fmt": "pct"},
        extra={"panels": panels},
    )


# =============================================================================== 2. Families


def _data_dan_families(store: Store, registry: Mapping[str, mr.ModelEntry]) -> ChartData:
    key, title = "dan_families", "Restricted-family horizon (unseen theorems), by model family"
    groups = _model_groups(registry)
    panels: dict[str, ChartData] = {}
    any_data = False

    for family in DAN_FAMILY_ORDER:
        models = groups.get(family, [])
        rows: list[P.DotRow] = []
        if models:
            data = horizon_view.view(
                store, _cfg({"models": models, "judge_id": _judge_for(family), "B": DAN_B})
            )
            if not data.get("empty"):
                any_data = True
                fam_dict = cast("dict[str, dict[str, Any]]", data["families"])
                for schema, per_model in sorted(fam_dict.items()):
                    points: list[tuple[str, float | None, str, bool]] = []
                    for m in models:
                        info = per_model.get(m)
                        nstar = (info or {}).get("n_star_unseen")
                        val = nstar.get("n_star") if nstar else None
                        points.append((m, float(val) if val is not None else None, "", False))
                    if any(v is not None for _n, v, _c, _z in points):
                        rows.append(P.DotRow(schema, points))
        panels[family] = ChartData(key, title, "dot", rows=rows)

    if not any_data:
        return _missing(key, title, "no families runs yet for any Dan or 'ours' model")
    caption = (
        "Largest unseen restricted-family size n* with mean pass@8 >= 50% (metrics.horizon, "
        "point estimate), per family, split into small multiples by model family. 'Unseen' "
        "excludes items in that model's own training tiers: our tiers for ours, Dan's per-family "
        "tiers for dan12/dan6 (models_registry.seen_lookup, DANSTATE_PLAN.md §A2) -- a sidecar "
        "that is not yet built for a given (suite, version, family) falls back to no exclusion "
        "rather than crashing. Judged lean_v2 for dan12/dan6, nd_v2 for ours."
    )
    return ChartData(
        key,
        title,
        "dot",
        caption=caption,
        rows=[r for p in panels.values() for r in p.rows],
        opts={"x_label": "n* (unseen family size)"},
        extra={"panels": panels},
    )


# ================================================================================ 3. Laws


def _pooled_curve_vs_s(data: Mapping[str, JSON], model: str) -> list[tuple[float, float]]:
    """Pool `subst.view`'s per-schema `curve_vs_s` for one `model`, averaged across schemas at
    each `s` (plan §6's small multiples need one curve per model per panel, not one per schema)."""
    schemas = cast("dict[str, dict[str, Any]]", data.get("schemas") or {})
    by_s: dict[float, list[float]] = {}
    for per_model in schemas.values():
        info = per_model.get(model)
        if info is None:
            continue
        for row in cast("list[dict[str, Any]]", info.get("curve_vs_s") or []):
            s_val = row.get("s")
            p1 = row.get("pass1")
            if s_val is None or p1 is None or p1 != p1:
                continue
            by_s.setdefault(float(s_val), []).append(float(p1))
    return [(s, sum(vals) / len(vals)) for s, vals in sorted(by_s.items()) if vals]


def _data_dan_laws(store: Store, registry: Mapping[str, mr.ModelEntry]) -> ChartData:
    key, title = "dan_laws", "Pass@1 vs substitution size s (pooled over law schemas)"
    groups = _model_groups(registry)
    panels: dict[str, ChartData] = {}
    any_data = False
    for family in DAN_FAMILY_ORDER:
        models = groups.get(family, [])
        series: list[P.Series] = []
        if models:
            data = subst_view.view(store, _cfg({"models": models, "judge_id": _judge_for(family)}))
            if not data.get("empty"):
                for m in models:
                    pts = _pooled_curve_vs_s(data, m)
                    if pts:
                        any_data = True
                        series.append(P.Series(m, pts, "var(--s1)"))
        panels[family] = ChartData(key, title, "line", series=series)
    if not any_data:
        return _missing(key, title, "no laws runs yet for any Dan or 'ours' model")
    caption = (
        "Pass@1 against filler substitution size s, averaged across the law schemas each model "
        "was run on, by model family. Items seen in training are excluded (analysis.subst's "
        "default). Judged lean_v2 for dan12/dan6, nd_v2 for ours."
    )
    return ChartData(
        key,
        title,
        "line",
        caption=caption,
        series=[s for p in panels.values() for s in p.series],
        opts={"legend": _legend_labels_compact(registry, groups), "x_label": "s", "y_fmt": "pct"},
        extra={"panels": panels},
    )


# ================================================================================= 4. Glue


def _data_dan_glue(store: Store, registry: Mapping[str, mr.ModelEntry]) -> ChartData:
    key, title = "dan_glue", "Compositionality ratio R per operator, by model family"
    groups = _model_groups(registry)
    panels: dict[str, ChartData] = {}
    any_data = False
    for family in DAN_FAMILY_ORDER:
        models = groups.get(family, [])
        bars: list[P.Bar] = []
        if models:
            # `cell_tag="op"` (glue.py's own WP-D5 addition): pool by the suite's coarse operator
            # tag (AND/CHAIN/CASE-dilemma/CASE-same/BOX -- 5 values) instead of its fine-grained
            # `cell` tag (19 values on the real `glue` suite, e.g. "AND_k2_L12"), matching the
            # chart's "one panel per operator family" claim -- a 19-bar panel does not read as a
            # small multiple. `dash/summary.py`'s own "ours" glue chart does this same collapse
            # (its `_OP_GROUP`/`_OP_ORDER`), just by re-deriving groups from scratch rather than
            # through this view; passing `cell_tag` here keeps this module's one `glue_view.view`
            # call as the single source of the per-cell numbers.
            data = glue_view.view(
                store,
                _cfg({"models": models, "judge_id": _judge_for(family), "cell_tag": "op"}),
            )
            if not data.get("empty"):
                cells = cast("dict[str, dict[str, Any]]", data["cells"])
                for op, per_model in sorted(cells.items()):
                    for m in models:
                        info = per_model.get(m)
                        comp = (info or {}).get("compositionality")
                        if comp is None:
                            continue
                        any_data = True
                        lo, hi = comp.get("ci", (None, None))
                        bars.append(P.Bar(f"{op}: {m}", float(comp["R"]), lo, hi))
        panels[family] = ChartData(key, title, "bar", bars=bars)
    if not any_data:
        return _missing(key, title, "no glue runs yet for any Dan or 'ours' model")
    caption = (
        "Compositionality ratio R = joined pass@1 / product of mastered parts' pass@1 per glue "
        "operator, by model family; each Dan model's own pass@1 on the renamed glue parts is "
        "used for its denominators when available (analysis.glue's default), falling back to the "
        "pool items otherwise. Judged lean_v2 for dan12/dan6, nd_v2 for ours."
    )
    return ChartData(
        key,
        title,
        "bar",
        caption=caption,
        bars=[b for p in panels.values() for b in p.bars],
        opts={"legend": _legend_labels_compact(registry, groups), "y_fmt": "f3"},
        extra={"panels": panels},
    )


# =============================================================================== 5. Padding


def _data_dan_padding(store: Store, registry: Mapping[str, mr.ModelEntry]) -> ChartData:
    key, title = "dan_padding", "Pass@1 vs written lines W, padded proofs, by model family"
    groups = _model_groups(registry)
    panels: dict[str, ChartData] = {}
    any_data = False
    for family in DAN_FAMILY_ORDER:
        models = groups.get(family, [])
        series: list[P.Series] = []
        if models:
            data = padding_view.view(
                store, _cfg({"models": models, "judge_id": _judge_for(family)})
            )
            if not data.get("empty"):
                pad_types = cast("dict[str, dict[str, Any]]", data["pad_types"])
                for m in models:
                    by_w: dict[float, list[tuple[int, float]]] = {}
                    for per_model in pad_types.values():
                        info = per_model.get(m)
                        if info is None:
                            continue
                        for row in cast(
                            "list[dict[str, Any]]", info.get("padded_curve_vs_W") or []
                        ):
                            w, p1, ni = row.get("W"), row.get("pass1"), row.get("n_items") or 0
                            if w is None or p1 is None or p1 != p1:
                                continue
                            by_w.setdefault(float(w), []).append((int(ni), float(p1)))
                    pts = [
                        (w, sum(ni * p for ni, p in vals) / max(sum(ni for ni, _ in vals), 1))
                        for w, vals in sorted(by_w.items())
                        if vals
                    ]
                    if pts:
                        any_data = True
                        series.append(P.Series(m, pts, "var(--s1)"))
        panels[family] = ChartData(key, title, "line", series=series)
    if not any_data:
        return _missing(key, title, "no padding runs yet for any Dan or 'ours' model")
    caption = (
        "Pass@1 against written proof length W for padded-easy items, pooled (item-count "
        "weighted) over pad types, by model family. Judged lean_v2 for dan12/dan6, nd_v2 for "
        "ours; the padding bases were chosen by our rl's own hill-climb selection set, a caveat "
        "that applies to the 'ours' panel only."
    )
    return ChartData(
        key,
        title,
        "line",
        caption=caption,
        series=[s for p in panels.values() for s in p.series],
        opts={"legend": _legend_labels_compact(registry, groups), "x_label": "W", "y_fmt": "pct"},
        extra={"panels": panels},
    )


# ========================================================================= 6. Prefix (state)


def _data_dan_prefix_state(store: Store, registry: Mapping[str, mr.ModelEntry]) -> ChartData:
    key = "dan_prefix_state"
    title = "Prefix completion by lines left, by model family"
    groups = _model_groups(registry)
    panels: dict[str, ChartData] = {}
    any_data = False
    for family in DAN_FAMILY_ORDER:
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
                    bucket = cast("dict[str, Any]", info.get("success_by_lines_left") or {})
                    pts = sorted(
                        (float(k), float(v["pass1"]))
                        for k, v in bucket.items()
                        if v.get("pass1") == v.get("pass1")
                    )
                    if pts:
                        any_data = True
                        series.append(P.Series(m, pts, "var(--s1)"))
        panels[family] = ChartData(key, title, "line", series=series)
    if not any_data:
        return _missing(key, title, "no prefix/prefix_state runs yet for any Dan or 'ours' model")
    caption = (
        "Prefix-completion pass@1 against lines left to write (L - j), by model family: ours "
        "reads the ND-line `prefix` suite, dan12/dan6 the state-env `prefix_state` suite (forced "
        "action prefix m(j), DANSTATE_PLAN.md §3), pooled the same way across proof lengths. "
        "Judged lean_v2 for dan12/dan6, nd_v2 for ours."
    )
    return ChartData(
        key,
        title,
        "line",
        caption=caption,
        series=[s for p in panels.values() for s in p.series],
        opts={
            "legend": _legend_labels_compact(registry, groups),
            "x_label": "lines left",
            "y_fmt": "pct",
        },
        extra={"panels": panels},
    )


# =============================================================================== 7. Skills


def _data_dan_skills(store: Store, registry: Mapping[str, mr.ModelEntry]) -> ChartData:
    """Pass@1 by `skill_group` (the 4-valued classical/case_split stratum `skill_pool_sub`'s own
    builder tags every item with), by model family -- direct runs only (module docstring: this
    suite is sampled directly, unlike legacy `skill_pool`, so `analysis.skills.view`'s join
    machinery does not apply)."""
    key, title = "dan_skills", "Pass@1 by skill group on skill_pool_sub, by model family"
    suite = "skill_pool_sub"
    groups = _model_groups(registry)
    panels: dict[str, ChartData] = {}
    any_data = False
    for family in DAN_FAMILY_ORDER:
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
    caption = (
        "Pass@1 by skill_group (plain / classical / case_split / classical+case_split, the "
        "stratification cell skill_pool_sub was built on, DANSTATE_PLAN.md §3) on this 1,200-item "
        "stratified subsample, by model family. Point estimates, item-count weighted within each "
        "group. Judged lean_v2 for dan12/dan6, nd_v2 for ours."
    )
    return ChartData(
        key,
        title,
        "bar",
        caption=caption,
        bars=[b for p in panels.values() for b in p.bars],
        opts={"legend": _legend_labels_compact(registry, groups), "y_fmt": "pct"},
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
    caption = (
        f"Fraction of Dan's {n_items}-item held-out set solved on one greedy draw (k=1, T=0), "
        "with a 95% Clopper-Pearson CI, per model. Gate V5's own expectation: T1 s0 at least "
        "4,940/5,000, Fz s0 4,632 +- 25/5,000. Judged lean_v2."
    )
    return ChartData(key, title, "bar", caption=caption, bars=bars, opts={"y_fmt": "pct"})


# =============================================================================== 9. Judges


def _data_dan_judges(store: Store, registry: Mapping[str, mr.ModelEntry]) -> ChartData:
    key, title = "dan_judges", "nd_v2 vs lean_v2 judge agreement"
    all_models = mr.labels_in_family(registry, "dan12") + mr.labels_in_family(registry, "dan6")
    data = judges_view.view(store, _cfg({"models": all_models}))
    if data.get("empty"):
        return _missing(key, title, str(data.get("reason", "no rejudged runs yet")))
    per_model = cast("dict[str, dict[str, Any]]", data["models"])
    bars = [P.Bar(m, float(info["agree_rate"])) for m, info in per_model.items()]
    breakdown_lines = []
    for m, info in sorted(per_model.items()):
        bd = cast("dict[str, Any]", info["lean_only_breakdown"])
        parts = ", ".join(f"{name}: {b['count']}" for name, b in bd.items() if b["count"])
        breakdown_lines.append(
            f"{m}: {info['lean_only_count']} lean-only accepts" + (f" ({parts})" if parts else "")
        )
    caption = (
        "Agreement between lean_v2 (Dan's headline judge) and the same already-sampled attempts "
        "rejudged nd_v2 (`rejudge --judge nd`), matched attempt-for-attempt, per model. "
        "Lean-only-accept breakdown (heuristic: nd_v2's own rejection reason plus whether the "
        "native Lean text uses `.elim`, not a re-derivation of Dan's own LEAN_JUDGE.md "
        "percentages -- 41.7% missing premise restatement / 25.9% .elim on not-A / 6.2% other, "
        "quoted here only as reported context): " + "; ".join(breakdown_lines) + "."
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


def _pass1_by_model(store: Store, models: Sequence[str], judge_id: str) -> dict[str, float]:
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
    out: dict[str, float] = {}
    for m in models:
        curve = curves.get(m)
        if curve:
            out[m] = curve[0][1]  # (k=1, mean, lo, hi)
    return out


def dan_crossfamily_table(
    store: Store, registry: Mapping[str, mr.ModelEntry]
) -> tuple[list[str], list[list[str]], str] | None:
    """`(headers, rows, caption)` for the cross-family textbook_dev pass@1 table (plan §6: "a
    cross-family results table (booktabs style) with Lean value and nd_v2 in parentheses and seed
    ranges"), or `None` if no model in any group has a textbook_dev run yet."""
    groups = _crossfamily_groups(registry)
    lean_by_model: dict[str, float] = {}
    nd_by_model: dict[str, float] = {}
    for (family, _role), models in groups.items():
        lean_by_model.update(_pass1_by_model(store, models, _judge_for(family)))
        nd_by_model.update(_pass1_by_model(store, models, OURS_JUDGE_ID))

    headers = ["family", "role", "pass@1 (lean, nd_v2)", "seed range"]
    rows: list[list[str]] = []
    any_row = False
    for (family, role), models in groups.items():
        lean_vals = [lean_by_model[m] for m in models if m in lean_by_model]
        nd_vals = [nd_by_model[m] for m in models if m in nd_by_model]
        if not lean_vals:
            rows.append([family, role, "not yet run", "-"])
            continue
        any_row = True
        lean_mean = sum(lean_vals) / len(lean_vals)
        # "ours" has no separate Lean judge: `lean_by_model`/`nd_by_model` are the same nd_v2
        # values there, so the parenthetical nd_v2 figure would just repeat the primary one.
        if family == "ours":
            cell = f"{100 * lean_mean:.1f}%"
        else:
            nd_cell = f"{100 * (sum(nd_vals) / len(nd_vals)):.1f}%" if nd_vals else "not rejudged"
            cell = f"{100 * lean_mean:.1f}% ({nd_cell})"
        seed_range = (
            f"{100 * min(lean_vals):.1f}-{100 * max(lean_vals):.1f}%" if len(lean_vals) > 1 else "-"
        )
        rows.append([family, role, cell, seed_range])
    if not any_row:
        return None
    caption = (
        "Cross-family pass@1 on textbook_dev (k=1 point estimate, B=10 -- a table entry, not a "
        "CI'd figure): the Lean (lean_v2) value, nd_v2 in parentheses, and the seed-0-to-seed-2 "
        "range where more than one seed has a run. 'ours' (base/rl) has a single checkpoint, so "
        "no seed range."
    )
    return headers, rows, caption


def _crossfamily_html(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    th = "".join(f"<th>{P.esc(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{P.esc(c)}</td>" for c in r) + "</tr>" for r in rows)
    return f'<table class="booktabs"><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>'


def _crossfamily_tex(headers: Sequence[str], rows: Sequence[Sequence[str]], caption: str) -> str:
    lines = [
        r"\begin{table}[t]",
        r"  \caption{" + caption.replace("_", r"\_").replace("%", r"\%") + "}",
        r"  \label{tab:dan_crossfamily}",
        r"  \centering",
        r"  \begin{tabular}{" + "l" * len(headers) + "}",
        r"    \toprule",
        "    " + " & ".join(headers) + r" \\",
        r"    \midrule",
    ]
    for row in rows:
        esc_row = [c.replace("_", r"\_").replace("%", r"\%") for c in row]
        lines.append("    " + " & ".join(esc_row) + r" \\")
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
        "superset of nd_v2 on denotable proofs); nd_v2 alongside where available. A family or "
        "benchmark with no landed runs yet at build time renders as a clean 'not available' "
        "panel below rather than failing -- which panels that is changes run to run as more of "
        "Dan's evaluation jobs land, so it is not called out here by name."
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
