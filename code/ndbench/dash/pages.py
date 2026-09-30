"""Turns each analysis view's dict (plan §3's `view_<bench>(store, cfg) -> dict`) into a list of
figure HTML fragments, using `dash/primitives.py`. One `render_<page>` function per dashboard page
(plan §3 WP6 row: "pages overview, yue, families, glue, skills, subst, padding, prefix, edl, runs");
`dash/build.py` wraps each function's output in `primitives.page(...)` and writes it out.

`render_runs`/`render_overview` do not take a `view_<bench>` dict: per `analysis/__init__.py`'s
docstring, the `runs`/`overview` pages are synthesised by `dash/build.py` directly from the
`Store` (every run's folded record, and the GPU ledger), so their renderers take that data plainly.

Every `render_*` checks `data.get("empty")` first and renders the plan's "no data yet" panel
(`primitives.missing_fig`) instead of an empty page.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import cast

from ..core.types import JSON
from . import primitives as P

#: A small, fixed categorical palette (the theme's six series colours), cycled with `_color`.
_COLORS: tuple[str, ...] = (
    "var(--s1)",
    "var(--s2)",
    "var(--s3)",
    "var(--s4)",
    "var(--s5)",
    "var(--s6)",
)


def _color(i: int) -> str:
    return _COLORS[i % len(_COLORS)]


def _ci_bar(name: str, coef: Mapping[str, JSON], color: str) -> P.Bar:
    b = float(cast(float, coef["coef"] if "coef" in coef else coef["slope"]))
    se = float(cast(float, coef["se"]))
    return P.Bar(name, b, b - 1.96 * se, b + 1.96 * se, color)


# --------------------------------------------------------------------------------------- yue


def render_yue(data: Mapping[str, JSON]) -> list[str]:
    if data.get("empty"):
        return [P.missing_fig("Yue et al. replication", str(data.get("reason", "no data")))]
    meta = cast("dict[str, JSON]", data["meta"])
    sets = cast("dict[str, JSON]", data["sets"])
    figs: list[str] = []
    for label, raw in sets.items():
        s = cast("dict[str, JSON]", raw)
        models = cast("list[str]", s["models"])
        n_items = cast("dict[str, int]", s["n_items"])
        curves = cast("dict[str, list[tuple[int, float, float, float]]]", s["curves"])
        sub = (
            "n="
            + ", ".join(f"{m}={n_items.get(m, '?')}" for m in models)
            + f"; B={meta.get('B')} bootstrap resamples, seed={meta.get('seed')}"
        )
        series = [
            P.Series(
                m,
                [(float(k), mean) for k, mean, _lo, _hi in curves.get(m, [])],
                _color(i),
                band=[(float(k), lo, hi) for k, _mean, lo, hi in curves.get(m, [])],
            )
            for i, m in enumerate(models)
        ]
        figs.append(
            P.line_chart(
                f"pass@k -- {label}",
                sub,
                series,
                x_label="k",
                y_fmt="pct",
                wide=True,
                log_x=True,
                note=(
                    "Shaded band: 95% item-level bootstrap CI. k <= 256 uses each item's first "
                    "sampling chunk only (fixed n, unbiased); k > n_i extrapolates "
                    "1-(1-c/n)^k."
                ),
            )
        )
        per_length = cast("dict[str, dict[str, list[tuple[int, float]]]]", s.get("per_length", {}))
        for m in models:
            by_len = per_length.get(m, {})
            if not by_len:
                continue
            len_series = [
                P.Series(f"L={length}", [(float(k), v) for k, v in curve], _color(j))
                for j, (length, curve) in enumerate(
                    sorted(by_len.items(), key=lambda kv: float(kv[0]))
                )
            ]
            figs.append(
                P.line_chart(
                    f"pass@k by length -- {label}, {m}",
                    f"n={n_items.get(m, '?')} items, grouped by problem length",
                    len_series,
                    x_label="k",
                    y_fmt="pct",
                    wide=True,
                    log_x=True,
                )
            )
        coverage = cast("dict[str, dict[str, JSON]]", s.get("coverage", {}))
        bars = []
        for i, m in enumerate(models):
            cov = coverage.get(m)
            if not cov:
                continue
            lo, hi = cast("tuple[float, float]", cov["ci"])
            bars.append(P.Bar(m, cast(float, cov["coverage"]), lo, hi, _color(i)))
        if bars:
            figs.append(
                P.bar_chart(
                    f"Coverage at k_max -- {label}",
                    "fraction of items ever solved (>=1 success across every draw); "
                    "95% Clopper-Pearson CI",
                    bars,
                    y_fmt="pct",
                )
            )
        pair = cast("list[str] | None", s.get("pair"))
        cross = cast("dict[str, JSON] | None", s.get("crossover"))
        venn = cast("dict[str, JSON] | None", s.get("venn"))
        if pair and cross:
            a, b = pair
            k_star = cross.get("k_star")
            ci = cast("tuple[int | None, int | None]", cross.get("ci", (None, None)))
            figs.append(
                P.table_fig(
                    f"Crossover k* -- {label}",
                    f"{a} vs {b}: first k where one model's pass@k overtakes the other's",
                    ["model a", "model b", "k*", "95% CI", "n items", "bootstrap hits", "B"],
                    [
                        [
                            a,
                            b,
                            str(k_star) if k_star is not None else "none found",
                            f"{ci[0]}-{ci[1]}" if ci[0] is not None else "n/a",
                            str(cross.get("n_common")),
                            str(cross.get("n_boot_found")),
                            str(cross.get("B")),
                        ]
                    ],
                )
            )
        if venn:
            figs.append(
                P.venn2(
                    f"Solved-set Venn -- {label}",
                    f"{venn['label_a']} vs {venn['label_b']}; "
                    f"{venn['label_b']} subset of {venn['label_a']}: {venn['b_subset_of_a']}",
                    cast(str, venn["label_a"]),
                    cast(str, venn["label_b"]),
                    cast(int, venn["only_a"]),
                    cast(int, venn["only_b"]),
                    cast(int, venn["both"]),
                )
            )
    legacy = cast("dict[str, dict[str, dict[str, JSON]]]", data.get("legacy_consistency", {}))
    if legacy:
        rows = [
            [
                label,
                m,
                str(lc["n_common"]),
                P.fmt(lc["agree_rate"], "pct"),
                P.fmt(lc["mean_abs_diff"], "f3"),
            ]
            for label, by_model in legacy.items()
            for m, lc in by_model.items()
        ]
        figs.append(
            P.table_fig(
                "Legacy consistency",
                "agreement with status=legacy runs (existing k=256, max_new=512 counts)",
                ["set", "model", "n common items", "agree rate", "mean abs diff in pass@256"],
                rows,
            )
        )
    return figs


# --------------------------------------------------------------------------------- families


def render_families(data: Mapping[str, JSON]) -> list[str]:
    if data.get("empty"):
        return [
            P.missing_fig(
                "Families (restricted-class horizons)", str(data.get("reason", "no data"))
            )
        ]
    families = cast("dict[str, dict[str, dict[str, JSON]]]", data["families"])
    figs: list[str] = []
    for family, per_model in families.items():
        series = []
        rows_nstar: list[list[str]] = []
        bars_line: list[P.Bar] = []
        bars_tok: list[P.Bar] = []
        for i, (model, d) in enumerate(per_model.items()):
            n_star = cast("dict[str, dict[str, JSON]]", d["n_star"])
            curve8 = cast("list[tuple[int, float]]", n_star["k8_tau0.5"]["curve"])
            series.append(P.Series(model, [(float(n), v) for n, v in curve8], _color(i)))
            for key, hr in n_star.items():
                ci = cast("tuple[int | None, int | None]", hr["ci"])
                rows_nstar.append(
                    [
                        model,
                        key,
                        str(hr["n_star"]),
                        f"{ci[0]}-{ci[1]}" if ci[0] is not None else "n/a",
                    ]
                )
            sl = cast("dict[str, JSON] | None", d.get("logit_slope_per_line"))
            if sl:
                bars_line.append(_ci_bar(model, sl, _color(i)))
            sl2 = cast("dict[str, JSON] | None", d.get("logit_slope_per_100_tokens"))
            if sl2:
                bars_tok.append(_ci_bar(model, sl2, _color(i)))
        sub = ", ".join(f"{m}: n={d['n_items']} items" for m, d in per_model.items())
        figs.append(
            P.line_chart(
                f"pass@8 vs n -- {family}", sub, series, x_label="n", y_fmt="pct", wide=True
            )
        )
        figs.append(
            P.table_fig(
                f"n* -- {family}",
                "largest n with mean pass@k >= tau for every n' <= n; bootstrap CI",
                ["model", "setting (k, tau)", "n*", "95% CI"],
                rows_nstar,
            )
        )
        if bars_line:
            figs.append(
                P.bar_chart(
                    f"Logit slope per added line -- {family}",
                    "pass ~ reference length L (logit link); 95% Wald CI",
                    bars_line,
                    y_fmt="f3",
                )
            )
        if bars_tok:
            figs.append(
                P.bar_chart(
                    f"Logit slope per 100 tokens -- {family}",
                    "pass ~ tokens/100 (logit link); 95% Wald CI",
                    bars_tok,
                    y_fmt="f3",
                )
            )
    return figs


# ------------------------------------------------------------------------------------- glue


def render_glue(data: Mapping[str, JSON]) -> list[str]:
    if data.get("empty"):
        return [
            P.missing_fig("Glued benchmark (compositionality)", str(data.get("reason", "no data")))
        ]
    cells = cast("dict[str, dict[str, dict[str, JSON]]]", data["cells"])
    figs: list[str] = []
    for cell, per_model in cells.items():
        bars_r: list[P.Bar] = []
        bars_rmin: list[P.Bar] = []
        rows_len: list[list[str]] = []
        rows_fail: list[list[str]] = []
        rows_ctrl: list[list[str]] = []
        for i, (model, d) in enumerate(per_model.items()):
            comp = cast("dict[str, JSON] | None", d.get("compositionality"))
            if comp:
                lo, hi = cast("tuple[float, float]", comp["ci"])
                lo_m, hi_m = cast("tuple[float, float]", comp["ci_min"])
                bars_r.append(P.Bar(model, cast(float, comp["R"]), lo, hi, _color(i)))
                bars_rmin.append(P.Bar(model, cast(float, comp["R_min"]), lo_m, hi_m, _color(i)))
            rows_len.append(
                [
                    model,
                    P.fmt(d.get("joined_pass1"), "pct"),
                    P.fmt(d.get("frac_long_enough"), "pct"),
                    str(d.get("n_joined_items")),
                    str(d.get("n_mastery_groups")),
                ]
            )
            for loc, count in cast("dict[str, int]", d.get("failure_location", {})).items():
                rows_fail.append([model, loc, str(count)])
            ctrl = cast("dict[str, JSON] | None", d.get("control_pass1"))
            rows_ctrl.append(
                [
                    model,
                    P.fmt(d.get("modal_L"), "g"),
                    P.fmt(ctrl["pass1"], "pct") if ctrl else "no data",
                    str(ctrl["n_items"]) if ctrl else "0",
                ]
            )
        if bars_r:
            figs.append(
                P.bar_chart(
                    f"R -- {cell}",
                    "joined pass@1 / product of part pass@1s, over mastered-parts items; 95% CI",
                    bars_r,
                    y_fmt="f3",
                )
            )
        if bars_rmin:
            figs.append(
                P.bar_chart(
                    f"R_min -- {cell}",
                    "joined pass@1 / min of part pass@1s, over mastered-parts items; 95% CI",
                    bars_rmin,
                    y_fmt="f3",
                )
            )
        figs.append(
            P.table_fig(
                f"Joined pass@1 and length -- {cell}",
                "n joined items = items sampled for this run; n mastery groups = items with "
                "every part mastered",
                [
                    "model",
                    "joined pass@1",
                    "frac attempts >= reference length",
                    "n joined items",
                    "n mastery groups",
                ],
                rows_len,
            )
        )
        if rows_fail:
            figs.append(
                P.table_fig(
                    f"Failure location -- {cell}",
                    "which part's line range a rejected line falls in "
                    "('glue' = the operator's own joining lines)",
                    ["model", "location", "n failing attempts"],
                    rows_fail,
                )
            )
        figs.append(
            P.table_fig(
                f"Control pass@1 at the same L -- {cell}",
                f"standard items from {cast('dict[str, JSON]', data['meta'])['control_suite']} "
                "at the cell's modal reference length",
                ["model", "modal L", "control pass@1", "control n items"],
                rows_ctrl,
            )
        )
    return figs


# ------------------------------------------------------------------------------------ skills


def render_skills(data: Mapping[str, JSON]) -> list[str]:
    if data.get("empty"):
        return [P.missing_fig("Skill profile", str(data.get("reason", "no data")))]
    models = cast("dict[str, dict[str, JSON]]", data["models"])
    figs: list[str] = []
    for model, d in models.items():
        heat1_raw = cast("dict[str, dict[str, dict[str, JSON]]]", d["heatmap_pass1"])
        heat32_raw = cast("dict[str, dict[str, dict[str, JSON]]]", d["heatmap_pass32"])
        heat1 = {
            skill: {length: cast(float, cell["pass1"]) for length, cell in per_l.items()}
            for skill, per_l in heat1_raw.items()
            if per_l
        }
        heat32 = {
            skill: {length: cast(float, cell["pass32"]) for length, cell in per_l.items()}
            for skill, per_l in heat32_raw.items()
            if per_l
        }
        sub = f"n={d['n_items']} items"
        figs.append(
            P.heatmap(
                f"pass@1 by skill x L -- {model}",
                sub,
                heat1,
                y_label="L",
                x_label="skill",
                v_fmt="pct",
                wide=True,
            )
        )
        figs.append(
            P.heatmap(
                f"pass@32 by skill x L -- {model}",
                sub,
                heat32,
                y_label="L",
                x_label="skill",
                v_fmt="pct",
                wide=True,
            )
        )
        glm = cast("dict[str, dict[str, JSON]]", d["glm"])
        bars = [_ci_bar(name, coef, _color(i)) for i, (name, coef) in enumerate(glm.items())]
        figs.append(
            P.bar_chart(
                f"GLM coefficients -- {model}",
                "binomial pass ~ L + skill indicators (logit link, IRLS); 95% Wald CI",
                bars,
                y_fmt="f3",
                wide=True,
            )
        )
    return figs


# ------------------------------------------------------------------------------------- subst


def render_subst(data: Mapping[str, JSON]) -> list[str]:
    if data.get("empty"):
        return [P.missing_fig("Law x substitution size", str(data.get("reason", "no data")))]
    schemas = cast("dict[str, dict[str, dict[str, JSON]]]", data["schemas"])
    figs: list[str] = []
    for schema, per_model in schemas.items():
        series_s = []
        series_tok = []
        bars: list[P.Bar] = []
        rows_cmp: list[list[str]] = []
        for i, (model, d) in enumerate(per_model.items()):
            curve_s = cast("list[dict[str, JSON]]", d["curve_vs_s"])
            pts_s = sorted(
                (float(cast(float, row["s"])), cast(float, row["pass1"]))
                for row in curve_s
                if row["s"] is not None
            )
            series_s.append(P.Series(model, pts_s, _color(i)))
            pts_tok = sorted(
                (cast(float, row["tokens"]), cast(float, row["pass1"]))
                for row in cast("list[dict[str, JSON]]", d["points_vs_tokens"])
            )
            series_tok.append(P.Series(model, pts_tok, _color(i)))
            sl = cast("dict[str, JSON] | None", d.get("logit_slope_per_100_tokens"))
            if sl:
                bars.append(_ci_bar(model, sl, _color(i)))
            cmp = cast("dict[str, JSON] | None", d.get("textbook_law_comparison"))
            rows_cmp.append(
                [
                    model,
                    P.fmt(d.get("controlled_pass1_overall"), "pct"),
                    P.fmt(cmp["pass1"], "pct") if cmp else "no data",
                    str(cmp["n_items"]) if cmp else "0",
                ]
            )
        figs.append(
            P.line_chart(
                f"pass@1 vs filler size s -- {schema}",
                "mean pass@1 per s (fixed L)",
                series_s,
                x_label="s (connectives per metavariable)",
                y_fmt="pct",
                wide=True,
            )
        )
        figs.append(
            P.line_chart(
                f"pass@1 vs written tokens -- {schema}",
                "per-item points, sorted by mean tokens",
                series_tok,
                x_label="mean tokens",
                y_fmt="pct",
                wide=True,
            )
        )
        if bars:
            figs.append(
                P.bar_chart(
                    f"Logit slope per 100 tokens -- {schema}",
                    "pass ~ tokens/100 (logit link); 95% Wald CI",
                    bars,
                    y_fmt="f3",
                )
            )
        figs.append(
            P.table_fig(
                f"Comparison with uncontrolled textbook-law items -- {schema}",
                "controlled substitution pass@1 vs real textbook items of the same schema",
                ["model", "controlled pass@1", "textbook-law pass@1", "textbook-law n"],
                rows_cmp,
            )
        )
    return figs


# ----------------------------------------------------------------------------------- padding


def render_padding(data: Mapping[str, JSON]) -> list[str]:
    if data.get("empty"):
        return [P.missing_fig("Free lines (padding)", str(data.get("reason", "no data")))]
    pad_types = cast("dict[str, dict[str, dict[str, JSON]]]", data["pad_types"])
    figs: list[str] = []
    for pad_type, per_model in pad_types.items():
        bars: list[P.Bar] = []
        for i, (model, d) in enumerate(per_model.items()):
            padded = sorted(
                (cast(float, row["W"]), cast(float, row["pass1"]))
                for row in cast("list[dict[str, JSON]]", d["padded_curve_vs_W"])
            )
            natural = sorted(
                (cast(float, row["W"]), cast(float, row["pass1"]))
                for row in cast("list[dict[str, JSON]]", d["natural_curve_vs_W"])
            )
            series = [P.Series(f"{model}: padded", padded, _color(2 * i))]
            if natural:
                series.append(P.Series(f"{model}: natural", natural, _color(2 * i + 1), muted=True))
            figs.append(
                P.line_chart(
                    f"pass@1 vs written lines W -- {pad_type}, {model}",
                    f"n={d['n_items']} padded items",
                    series,
                    x_label="W (written lines)",
                    y_fmt="pct",
                    wide=True,
                )
            )
            sl = cast("dict[str, JSON] | None", d.get("free_line_cost_per_line"))
            if sl:
                bars.append(_ci_bar(model, sl, _color(i)))
        if bars:
            figs.append(
                P.bar_chart(
                    f"Free-line cost -- {pad_type}",
                    "logit slope of pass@1 per added padding line m; 95% Wald CI",
                    bars,
                    y_fmt="f3",
                )
            )
    return figs


# ------------------------------------------------------------------------------------ prefix


def render_prefix(data: Mapping[str, JSON]) -> list[str]:
    if data.get("empty"):
        return [P.missing_fig("Prefix completion", str(data.get("reason", "no data")))]
    models = cast("dict[str, dict[str, JSON]]", data["models"])
    figs: list[str] = []
    series_j, series_ll, series_l, series_hz = [], [], [], []
    bars_all: list[P.Bar] = []
    for i, (model, d) in enumerate(models.items()):

        def to_pts(bucket: Mapping[str, JSON]) -> list[tuple[float, float]]:
            return sorted(
                (float(k), cast(float, cast("dict[str, JSON]", v)["pass1"]))
                for k, v in bucket.items()
            )

        series_j.append(
            P.Series(model, to_pts(cast("dict[str, JSON]", d["success_by_j"])), _color(i))
        )
        series_ll.append(
            P.Series(model, to_pts(cast("dict[str, JSON]", d["success_by_lines_left"])), _color(i))
        )
        series_l.append(
            P.Series(model, to_pts(cast("dict[str, JSON]", d["success_by_L"])), _color(i))
        )
        hz = cast("list[dict[str, JSON]]", d["hazard"])
        series_hz.append(
            P.Series(
                model,
                [
                    (cast(float, row["line"]), cast(float, row["hazard"]))
                    for row in hz
                    if row["at_risk"]
                ],
                _color(i),
            )
        )
        fl = cast("dict[str, dict[str, JSON]]", d["fitted_logit"])
        for name, coef in fl.items():
            bars_all.append(_ci_bar(f"{model}: {name}", coef, _color(i)))
    figs.append(
        P.line_chart(
            "Success by prefix length j",
            "pooled pass rate per j, across every proof length",
            series_j,
            x_label="j (forced prefix lines)",
            y_fmt="pct",
            wide=True,
        )
    )
    figs.append(
        P.line_chart(
            "Success by lines left (L - j)",
            "pooled pass rate",
            series_ll,
            x_label="L - j",
            y_fmt="pct",
            wide=True,
        )
    )
    figs.append(
        P.line_chart(
            "Success by proof length L",
            "pooled pass rate ('absolute position' read as where in the length spectrum -- "
            "see module docstring)",
            series_l,
            x_label="L",
            y_fmt="pct",
            wide=True,
        )
    )
    figs.append(
        P.line_chart(
            "First-failure hazard by absolute line",
            "share of still-alive attempts failing exactly at that line",
            series_hz,
            x_label="line",
            y_fmt="pct",
            wide=True,
        )
    )
    if bars_all:
        figs.append(
            P.bar_chart(
                "Fitted logit p = a + b*(L-j) + c*j",
                "fixed effects only -- no per-proof random intercept (see module docstring); "
                "95% Wald CI",
                bars_all,
                y_fmt="f3",
                wide=True,
            )
        )
    return figs


# --------------------------------------------------------------------------------------- edl


def render_edl(data: Mapping[str, JSON]) -> list[str]:
    if data.get("empty"):
        return [P.missing_fig("Excess description length", str(data.get("reason", "no data")))]
    classes = cast("dict[str, dict[str, dict[str, JSON]]]", data["classes"])
    figs: list[str] = []
    for cls, per_model in classes.items():
        bars: list[P.Bar] = []
        rows_ltest: list[list[str]] = []
        rows_pass8: list[list[str]] = []
        curve_series = []
        for i, (model, d) in enumerate(per_model.items()):
            lo, hi = cast("tuple[float, float]", d["edl_star_ci"])
            bars.append(
                P.Bar(
                    model,
                    cast(float, d["edl_star"]),
                    None if math.isnan(lo) else lo,
                    None if math.isnan(hi) else hi,
                    _color(i),
                )
            )
            curves_by_lr = cast("dict[str, list[tuple[int, float]]]", d["edl_curve_by_lr"])
            best_curve = curves_by_lr.get(cast(str, d["best_lr"]), [])
            curve_series.append(
                P.Series(
                    f"{model} (lr={d['best_lr']})",
                    [(float(n), v) for n, v in best_curve],
                    _color(i),
                )
            )
            rows_ltest.append(
                [model, P.fmt(d["l_test_theta0"], "f3"), P.fmt(d["l_test_thetan"], "f3")]
            )
            rows_pass8.append(
                [model, P.fmt(d["pass8_theta0"], "pct"), P.fmt(d["pass8_thetan"], "pct")]
            )
        figs.append(
            P.bar_chart(
                f"EDL* -- {cls}",
                "min over LR + ln(#LRs); CI from a held-out per-example bootstrap where available",
                bars,
                y_fmt="f3",
            )
        )
        figs.append(
            P.line_chart(
                f"EDL(n') -- {cls}",
                "at each model's best LR",
                curve_series,
                x_label="n'",
                y_fmt="f3",
                wide=True,
            )
        )
        figs.append(
            P.table_fig(
                f"L_test before/after -- {cls}",
                "nats/example on the held-out set",
                ["model", "theta_0", "theta_n"],
                rows_ltest,
            )
        )
        figs.append(
            P.table_fig(
                f"pass@8 before/after -- {cls}",
                "held-out theorems",
                ["model", "theta_0", "theta_n"],
                rows_pass8,
            )
        )
    return figs


# -------------------------------------------------------------------------------- runs/overview


def render_runs(runs: Sequence[Mapping[str, JSON]], ledger_total_s: float) -> list[str]:
    if not runs:
        return [P.missing_fig("Runs", "no runs recorded yet")]
    rows = [
        [
            str(r.get("run_id", "")),
            str(r.get("suite", "")),
            str(r.get("version", "")),
            str(r.get("model", "")),
            str(r.get("status", "")),
            str(r.get("k", "")),
            str(r.get("judge_id", "")),
            str(r.get("started_utc", "")),
            str(r.get("finished_utc", "")),
            P.fmt(r.get("gpu_s"), "secs") if r.get("gpu_s") is not None else "n/a",
        ]
        for r in sorted(runs, key=lambda r: str(r.get("started_utc") or ""))
    ]
    return [
        P.table_fig(
            "All runs",
            f"{len(runs)} runs (legacy imports included)",
            [
                "run_id",
                "suite",
                "version",
                "model",
                "status",
                "k",
                "judge",
                "started",
                "finished",
                "gpu_s",
            ],
            rows,
        ),
        P.table_fig(
            "GPU ledger",
            "plan §4's stop rule: lock-held wall time only",
            ["total GPU-hours"],
            [[P.fmt(ledger_total_s / 3600.0, "f3")]],
        ),
    ]


def render_overview(
    n_runs: int, n_suites: int, statuses: Mapping[str, int], page_links: Sequence[tuple[str, str]]
) -> list[str]:
    rows = [[status, str(count)] for status, count in sorted(statuses.items())]
    figs = [
        P.table_fig(
            "Run status summary",
            f"{n_runs} runs across {n_suites} suites",
            ["status", "count"],
            rows,
        )
    ]
    links_html = "".join(
        f'<li><a href="{P.esc(href)}">{P.esc(name)}</a></li>' for href, name in page_links
    )
    figs.append(P.figure("Pages", "jump to a benchmark", f"<ul>{links_html}</ul>", wide=True))
    return figs
