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
from statistics import mean, median
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


def intro(title: str, html_body: str) -> str:
    """A short, plain-language figure at the top of a page explaining what the benchmark
    measures and how to read its charts, for a reader who hasn't seen `PLAN.md` (fix list item
    3's "reasonable titles and short explanatory text ... saying what each benchmark measures
    and how to read the chart"). Shared by every `render_<page>` (and `dash/interp_pages.py`'s
    four, which import it).
    """
    return P.figure(title, "", html_body, wide=True)


# --------------------------------------------------------------------------------------- yue


_YUE_INTRO = intro(
    "Yue et al.: does more sampling close the base-RL gap?",
    "<p>Replicates Yue et al.'s base-vs-RL pass@k comparison: if RL mainly re-weights "
    "capabilities the base model already has (rather than adding new ones), sampling the base "
    "model enough times (large k) should let it match or beat the RL-tuned model's pass@1. Each "
    "set below shows pass@k curves (shaded band = 95% bootstrap CI), the crossover k* where one "
    "model's curve overtakes the other's, and coverage (fraction of items ever solved) at the "
    "largest k actually sampled.</p>",
)


def render_yue(data: Mapping[str, JSON]) -> list[str]:
    if data.get("empty"):
        return [
            _YUE_INTRO,
            P.missing_fig("Yue et al. replication", str(data.get("reason", "no data"))),
        ]
    meta = cast("dict[str, JSON]", data["meta"])
    sets = cast("dict[str, JSON]", data["sets"])
    figs: list[str] = [_YUE_INTRO]
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
        set_k_max = s.get("k_max")
        bars = []
        rows_cov: list[list[str]] = []
        for i, m in enumerate(models):
            cov = coverage.get(m)
            if not cov:
                continue
            lo, hi = cast("tuple[float, float]", cov["ci"])
            bars.append(P.Bar(m, cast(float, cov["coverage"]), lo, hi, _color(i)))
            rows_cov.append(
                [
                    m,
                    str(cov.get("k_max")),
                    str(cov.get("n_items")),
                    str(cov.get("solved")),
                    P.fmt(cov.get("coverage"), "pct"),
                ]
            )
        if bars:
            figs.append(
                P.bar_chart(
                    f"Coverage at k_max -- {label} (set k_max={set_k_max})",
                    "fraction of items ever solved (>=1 success across every draw); "
                    "95% Clopper-Pearson CI. A model's own k_max (samples actually drawn per "
                    "item) can be below the set's -- see the table below for each model's own.",
                    bars,
                    y_fmt="pct",
                )
            )
            figs.append(
                P.table_fig(
                    f"Coverage detail -- {label}",
                    "each model's own k_max (its largest per-item sample count), not just the "
                    "set's",
                    ["model", "k_max (own)", "n items", "solved", "coverage"],
                    rows_cov,
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
                P.fmt(lc["mean_abs_diff"], "pct"),
            ]
            for label, by_model in legacy.items()
            for m, lc in by_model.items()
        ]
        figs.append(
            P.table_fig(
                "Legacy consistency",
                "agreement with status=legacy runs (existing k=256, max_new=512 counts)",
                [
                    "set",
                    "model",
                    "n common items",
                    "agree rate",
                    "mean abs diff in pass@256 (pp)",
                ],
                rows,
            )
        )
    return figs


# --------------------------------------------------------------------------------- families


_FAMILIES_INTRO = intro(
    "Restricted-class families: where does the model's horizon end?",
    "<p>Each family is a parametrised class of synthetic proofs (e.g. "
    "<code>and_tower</code> chains n ANDI steps, <code>imp_nest</code> nests n implications) "
    "whose difficulty parameter <code>n</code> grows the proof's length/depth predictably. "
    "<code>n*</code> is the largest <code>n</code> the model still clears at a pass-rate "
    "threshold for every smaller <code>n</code> too (its 'horizon') -- a capability cliff edge, "
    "not just a gradual decline. The logit-slope bars show how costly each extra line or 100 "
    "tokens is in log-odds of success.</p>",
)


def render_families(data: Mapping[str, JSON]) -> list[str]:
    if data.get("empty"):
        return [
            _FAMILIES_INTRO,
            P.missing_fig(
                "Families (restricted-class horizons)", str(data.get("reason", "no data"))
            ),
        ]
    families = cast("dict[str, dict[str, dict[str, JSON]]]", data["families"])
    meta = cast("dict[str, JSON]", data.get("meta", {}))
    ks = cast("list[int]", meta.get("ks", [1, 8, 32]))
    figs: list[str] = [_FAMILIES_INTRO]
    for family, per_model in families.items():
        series_by_k: dict[int, list[P.Series]] = {k: [] for k in ks}
        rows_nstar: list[list[str]] = []
        bars_line: list[P.Bar] = []
        bars_tok: list[P.Bar] = []
        for i, (model, d) in enumerate(per_model.items()):
            n_star = cast("dict[str, dict[str, JSON]]", d["n_star"])
            curves_by_k = cast("dict[str, list[tuple[int, float]]]", d.get("curves_by_k", {}))
            for k in ks:
                curve = curves_by_k.get(str(k), [])
                series_by_k[k].append(P.Series(model, [(float(n), v) for n, v in curve], _color(i)))
            for key, hr in n_star.items():
                ci = cast("tuple[int | None, int | None]", hr["ci"])
                n_star_val = hr["n_star"]
                rows_nstar.append(
                    [
                        model,
                        key,
                        str(n_star_val) if n_star_val is not None else "n/a (never clears tau)",
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
        # Plan §2 B2 asks for "pass@1 and pass@k vs n" (plural k): one chart per k on `ks`
        # (default 1/8/32), not just the single k=8 curve used for the horizon n* statistic.
        for k in ks:
            if not any(s.pts for s in series_by_k[k]):
                continue
            figs.append(
                P.line_chart(
                    f"pass@{k} vs n -- {family}",
                    sub,
                    series_by_k[k],
                    x_label="n",
                    y_fmt="pct",
                    wide=True,
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


_GLUE_INTRO = intro(
    "Glue: does capability compose?",
    "<p>Takes problems the model can already solve on their own ('parts') and joins them with a "
    "logical operator (AND, CHAIN, CASE-split, nested BOX) into one larger 'joined' problem. "
    "<code>R</code> = joined pass@1 / product of the parts' own pass@1s, computed only over "
    "items where the model has mastered every part: <code>R</code> near 1 means the model can "
    "chain known pieces together about as often as independence would predict; "
    "<code>R</code> well below 1 means joining itself is the hard part, not the pieces.</p>",
)


def render_glue(data: Mapping[str, JSON]) -> list[str]:
    if data.get("empty"):
        return [
            _GLUE_INTRO,
            P.missing_fig("Glued benchmark (compositionality)", str(data.get("reason", "no data"))),
        ]
    cells = cast("dict[str, dict[str, dict[str, JSON]]]", data["cells"])
    figs: list[str] = [_GLUE_INTRO]
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
                    "no parts_suite run for this model (R/R_min unavailable)"
                    if d.get("no_mastery_run")
                    else "ok",
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
                    "mastery data",
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
    meta = cast("dict[str, JSON]", data.get("meta", {}))
    models = cast("dict[str, dict[str, JSON]]", data["models"])
    figs: list[str] = [
        P.figure(
            "Skill profile: what this measures",
            "",
            "<p>Pass rate broken down by proof length <code>L</code> and by which logical "
            "skill a theorem needs (e.g. <code>classical</code> = not provable "
            "intuitionistically; <code>case_split</code> = needs a disjunction-elimination "
            "step with no shortcut). The GLM below fits "
            "<code>pass ~ L + skill indicators</code> to isolate each skill's own effect, "
            "controlling for length.</p>"
            + (f"<p>{P.esc(cast(str, meta['legacy_note']))}</p>" if meta.get("legacy_note") else "")
            + (
                "<p>Untagged skills (not computed for this suite, so omitted below): "
                + P.esc(", ".join(cast("list[str]", meta.get("untagged_skills", []))))
                + "</p>"
                if meta.get("untagged_skills")
                else ""
            ),
            wide=True,
        )
    ]
    # Matched comparisons first (audit 2026-10-01): the per-model panels below are on different
    # item sets (base covers every source set, rl dev/sealed/frontier, armC lengthexp only).
    comparisons = cast("dict[str, dict[str, JSON]]", data.get("comparisons") or {})
    for cmp in comparisons.values():
        a, b = cast("list[str]", cmp["models"])
        by_skill = cast("dict[str, dict[str, JSON]]", cmp["by_skill"])
        sources = cast("dict[str, int]", cmp["sources"])
        figs.append(
            P.table_fig(
                f"{a} vs {b} on the same {cmp['n_common']} items",
                "pass@1 by skill group, only items both models were sampled on; source sets: "
                + ", ".join(f"{k} ({v})" for k, v in sorted(sources.items())),
                ["skill group", "items", a, b],
                [
                    [g, str(r["n_items"]), P.fmt(r[a], "pct"), P.fmt(r[b], "pct")]
                    for g, r in by_skill.items()
                ],
            )
        )
    coverage = cast("dict[str, dict[str, int]]", meta.get("coverage_by_source") or {})
    if coverage:
        figs.append(
            P.table_fig(
                "Which items each model has results for",
                "items per source set; the per-model panels below use all of a model's items, so "
                "compare models with the matched tables above",
                ["model", *sorted({k for c in coverage.values() for k in c})],
                [
                    [
                        m,
                        *(
                            str(c.get(k, 0))
                            for k in sorted({k for c2 in coverage.values() for k in c2})
                        ),
                    ]
                    for m, c in sorted(coverage.items())
                ],
            )
        )
    for model, d in models.items():
        heat1_raw = cast("dict[str, dict[str, dict[str, JSON]]]", d["heatmap_pass1"])
        heat32_raw = cast("dict[str, dict[str, dict[str, JSON]]]", d["heatmap_pass32"])
        heat1 = {
            skill: {length: cast(float, cell["pass1"]) for length, cell in per_l.items()}
            for skill, per_l in heat1_raw.items()
            if per_l
        }
        n1 = {
            skill: {length: cast(int, cell["n_items"]) for length, cell in per_l.items()}
            for skill, per_l in heat1_raw.items()
            if per_l
        }
        heat32 = {
            skill: {length: cast(float, cell["pass32"]) for length, cell in per_l.items()}
            for skill, per_l in heat32_raw.items()
            if per_l
        }
        n32 = {
            skill: {length: cast(int, cell["n_items"]) for length, cell in per_l.items()}
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
                counts=n1,
                wide=True,
            )
        )
        figs.append(
            P.heatmap(
                f"pass@32 by skill x L -- {model}",
                sub + "; only items sampled >= 32 times count toward pass@32 (see cell counts)",
                heat32,
                y_label="L",
                x_label="skill",
                v_fmt="pct",
                counts=n32,
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


_SUBST_INTRO = intro(
    "Law x substitution size: knowing the law vs. handling complex fillers",
    "<p>Fixes a textbook logical law/schema (e.g. contraposition, De Morgan) and plugs in random "
    "fillers for its metavariables of increasing complexity <code>s</code> (connectives per "
    "filler), while holding the overall derivation length fixed by construction. This separates "
    "'does the model know the law's shape' from 'can it handle more complex sub-formulas written "
    "into it' -- and compares against real, uncontrolled textbook instances of the same law.</p>",
)


def render_subst(data: Mapping[str, JSON]) -> list[str]:
    if data.get("empty"):
        return [
            _SUBST_INTRO,
            P.missing_fig("Law x substitution size", str(data.get("reason", "no data"))),
        ]
    schemas = cast("dict[str, dict[str, dict[str, JSON]]]", data["schemas"])
    figs: list[str] = [_SUBST_INTRO]
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
                f"pass@1 vs item tokens -- {schema}",
                "per-item points; x = the item's prompt + reference-proof tokens",
                series_tok,
                x_label="item tokens",
                y_fmt="pct",
                wide=True,
                note=(
                    "Each point is one item at its own size in tokens (prompt + reference "
                    "proof), the same x for every model; several items can share a token count."
                ),
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


_PADDING_INTRO = intro(
    "Free lines: what does an irrelevant extra step cost?",
    "<p>Starts from problems the model already solves and inserts <code>m</code> extra, "
    "logically inert lines -- either unused premises (<code>P1</code>) or no-op conjunctions "
    "that add lines without adding premises (<code>P2</code>) -- then re-checks pass@1 at the new, "
    "padded length <code>W</code> against 'natural' problems of the same length that were never "
    "padded. The free-line cost is the logit slope of pass@1 per added line: how much each "
    "purely mechanical extra line hurts, independent of genuine difficulty.</p>",
)


def render_padding(data: Mapping[str, JSON]) -> list[str]:
    if data.get("empty"):
        return [
            _PADDING_INTRO,
            P.missing_fig("Free lines (padding)", str(data.get("reason", "no data"))),
        ]
    pad_types = cast("dict[str, dict[str, dict[str, JSON]]]", data["pad_types"])
    figs: list[str] = [_PADDING_INTRO]
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


_PREFIX_INTRO = intro(
    "Prefix completion: can the model land the plane mid-flight?",
    "<p>The model is handed the first <code>j</code> lines of a correct proof (verbatim, not "
    "written by it) and must complete it validly from there -- success means the forced prefix "
    "plus whatever the model writes next passes the proof checker. Charts below show pass rate "
    "against <code>j</code> (how much is forced), against lines left (<code>L - j</code>, how "
    "much remains to plan), and a per-line hazard (where completions tend to fail first).</p>",
)


def render_prefix(data: Mapping[str, JSON]) -> list[str]:
    if data.get("empty"):
        return [
            _PREFIX_INTRO,
            P.missing_fig("Prefix completion", str(data.get("reason", "no data"))),
        ]
    models = cast("dict[str, dict[str, JSON]]", data["models"])
    figs: list[str] = [_PREFIX_INTRO]
    sub_n = "n=" + ", ".join(f"{m}={d.get('n_items', '?')}" for m, d in models.items()) + " items"
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
            f"pooled pass rate per j, across every proof length; {sub_n}",
            series_j,
            x_label="j (forced prefix lines)",
            y_fmt="pct",
            wide=True,
        )
    )
    figs.append(
        P.line_chart(
            "Success by lines left (L - j)",
            f"pooled pass rate; {sub_n}",
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
            f"see module docstring); {sub_n}",
            series_l,
            x_label="L",
            y_fmt="pct",
            wide=True,
        )
    )
    figs.append(
        P.line_chart(
            "First-failure hazard by absolute line",
            f"share of still-alive attempts failing exactly at that line; {sub_n}",
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


_EDL_INTRO = intro(
    "Excess description length: how fast does the model pick up a new pattern?",
    "<p>Streams <code>n</code> examples of an unfamiliar proof pattern through the model one at "
    "a time, taking one gradient step after scoring each one (an online learner), and measures "
    "how many extra nats (beyond the model's eventual steady-state cost) it took to fit them -- "
    "EDL* (the minimum over tried learning rates). A small EDL* after the model's held-out "
    "pass@8 genuinely improved means the pattern was close to 'already known'; a small EDL* with "
    "no capability gain is not evidence of anything (see pass@8 before/after below).</p>",
)


def render_edl(data: Mapping[str, JSON]) -> list[str]:
    if data.get("empty"):
        return [
            _EDL_INTRO,
            P.missing_fig("Excess description length", str(data.get("reason", "no data"))),
        ]
    classes = cast("dict[str, dict[str, dict[str, JSON]]]", data["classes"])
    figs: list[str] = [_EDL_INTRO]
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
            # A checkpoint past this row's own final `n` (e.g. the default checkpoint grid's 64
            # when a run only reached n=32) has no data -- `edl_curve_by_lr` still emits a `nan`
            # point for it (edl.py's own module docstring), which an unfiltered `line_chart` would
            # render as a literal "nan" coordinate in the SVG path (a broken/truncated segment).
            curve_series.append(
                P.Series(
                    f"{model} (lr={d['best_lr']})",
                    [(float(n), v) for n, v in best_curve if not math.isnan(v)],
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


def _overview_yue(view_data: Mapping[str, JSON]) -> list[str]:
    data = cast("Mapping[str, JSON]", view_data.get("yue", {}))
    if data.get("empty"):
        return []
    sets = cast("dict[str, dict[str, JSON]]", data.get("sets", {}))
    tdev = sets.get("textbook_dev")
    if not tdev:
        return []
    models = cast("list[str]", tdev["models"])
    curves = cast("dict[str, list[tuple[int, float, float, float]]]", tdev["curves"])
    n_items = cast("dict[str, int]", tdev["n_items"])
    rows = []
    for m in models:
        curve = curves.get(m, [])
        if not curve:
            continue
        k1 = next((pt for pt in curve if int(pt[0]) == 1), curve[0])
        kmax_pt = curve[-1]
        rows.append([m, P.fmt(k1[1], "pct"), f"k={int(kmax_pt[0])}", P.fmt(kmax_pt[1], "pct")])
    if not rows:
        return []
    n = n_items.get(models[0], "?") if models else "?"
    return [
        P.table_fig(
            "Yue et al.: textbook_dev",
            f"pass@1 and pass@k_max on {n} textbook theorems, base vs rl vs armC",
            ["model", "pass@1", "k_max", "pass@k_max"],
            rows,
            note='<a href="yue.html">Yue et al. page &#8594;</a>',
        )
    ]


def _overview_families(view_data: Mapping[str, JSON]) -> list[str]:
    data = cast("Mapping[str, JSON]", view_data.get("families", {}))
    if data.get("empty"):
        return []
    families = cast("dict[str, dict[str, dict[str, JSON]]]", data.get("families", {}))
    if not families:
        return []
    per_model: dict[str, list[int]] = {}
    for by_model in families.values():
        for model, d in by_model.items():
            n_star_all = cast("dict[str, JSON]", d.get("n_star", {}))
            entry = cast("dict[str, JSON] | None", n_star_all.get("k8_tau0.5"))
            val = entry.get("n_star") if entry else None
            if isinstance(val, int):
                per_model.setdefault(model, []).append(val)
    rows = [
        [model, str(median(vals)), str(len(vals))]
        for model, vals in sorted(per_model.items())
        if vals
    ]
    if not rows:
        return []
    return [
        P.table_fig(
            "Restricted-class families: horizon",
            f"median n* (largest n with mean pass@8 >= 0.5 for every n'<=n) across "
            f"{len(families)} families",
            ["model", "median n*", "n families with a horizon"],
            rows,
            note='<a href="families.html">Families page &#8594;</a>',
        )
    ]


def _overview_glue(view_data: Mapping[str, JSON]) -> list[str]:
    data = cast("Mapping[str, JSON]", view_data.get("glue", {}))
    if data.get("empty"):
        return []
    cells = cast("dict[str, dict[str, dict[str, JSON]]]", data.get("cells", {}))
    if not cells:
        return []
    per_model: dict[str, list[float]] = {}
    for by_model in cells.values():
        for model, d in by_model.items():
            comp = cast("dict[str, JSON] | None", d.get("compositionality"))
            r = comp.get("R") if comp else None
            if isinstance(r, int | float) and not math.isnan(r):
                per_model.setdefault(model, []).append(float(r))
    rows = [
        [model, P.fmt(mean(vals), "f3"), str(len(vals))]
        for model, vals in sorted(per_model.items())
    ]
    if not rows:
        return []
    return [
        P.table_fig(
            "Glue: compositionality ratio R",
            f"mean R (joined pass@1 / product of part pass@1s) across {len(cells)} cells, "
            "over mastered-parts items",
            ["model", "mean R", "n cells"],
            rows,
            note='<a href="glue.html">Glue page &#8594;</a>',
        )
    ]


def _overview_padding(view_data: Mapping[str, JSON]) -> list[str]:
    data = cast("Mapping[str, JSON]", view_data.get("padding", {}))
    if data.get("empty"):
        return []
    pad_types = cast("dict[str, dict[str, dict[str, JSON]]]", data.get("pad_types", {}))
    if not pad_types:
        return []
    per_model: dict[str, list[float]] = {}
    for by_model in pad_types.values():
        for model, d in by_model.items():
            sl = cast("dict[str, JSON] | None", d.get("free_line_cost_per_line"))
            slope = sl.get("slope") if sl else None
            if isinstance(slope, int | float) and not math.isnan(slope):
                per_model.setdefault(model, []).append(float(slope))
    rows = [
        [model, P.fmt(mean(vals), "f3"), str(len(vals))]
        for model, vals in sorted(per_model.items())
    ]
    if not rows:
        return []
    return [
        P.table_fig(
            "Padding: free-line cost",
            f"mean logit slope of pass@1 per added padding line, across {len(pad_types)} pad "
            "types (more negative = costlier free lines)",
            ["model", "mean slope", "n pad types"],
            rows,
            note='<a href="padding.html">Padding page &#8594;</a>',
        )
    ]


def _overview_prefix(view_data: Mapping[str, JSON]) -> list[str]:
    data = cast("Mapping[str, JSON]", view_data.get("prefix", {}))
    if data.get("empty"):
        return []
    models = cast("dict[str, dict[str, JSON]]", data.get("models", {}))
    if not models:
        return []
    rows = []
    for model, d in models.items():
        fl = cast("dict[str, dict[str, JSON]]", d.get("fitted_logit", {}))
        ll = fl.get("lines_left")
        rows.append(
            [
                model,
                P.fmt(d.get("overall_pass1"), "pct"),
                P.fmt(ll.get("coef"), "f3") if ll else "n/a",
            ]
        )
    if not rows:
        return []
    return [
        P.table_fig(
            "Prefix completion",
            "overall pass@1 (any forced prefix) and the fitted logit slope per line left "
            "(the planning-horizon effect; more negative = harder the more is left to write)",
            ["model", "overall pass@1", "slope per line left"],
            rows,
            note='<a href="prefix.html">Prefix page &#8594;</a>',
        )
    ]


def _overview_edl(view_data: Mapping[str, JSON]) -> list[str]:
    data = cast("Mapping[str, JSON]", view_data.get("edl", {}))
    if data.get("empty"):
        return []
    classes = cast("dict[str, dict[str, dict[str, JSON]]]", data.get("classes", {}))
    if not classes:
        return []
    model_cols = sorted({model for by_model in classes.values() for model in by_model})
    rows = []
    for cls, by_model in sorted(classes.items()):
        row = [cls]
        for model in model_cols:
            d = by_model.get(model)
            row.append(P.fmt(d["edl_star"], "f3") if d else "no data")
        rows.append(row)
    return [
        P.table_fig(
            "EDL*: per class",
            "min over learning rate + ln(#LRs), nats; lower = more capability elicited online",
            ["class", *model_cols],
            rows,
            note='<a href="edl.html">EDL page &#8594;</a>',
        )
    ]


def _overview_probes(view_data: Mapping[str, JSON]) -> list[str]:
    data = cast("Mapping[str, JSON]", view_data.get("interp_probes", {}))
    if data.get("empty"):
        return []
    acc = cast("dict[str, list[tuple[int, JSON]]]", data.get("next_rule_acc", {}))
    if not acc:
        return []
    rows = []
    for model, pts in sorted(acc.items()):
        if not pts:
            continue
        _layer, val = pts[-1]
        rows.append([model, P.fmt(val, "pct")])
    if not rows:
        return []
    return [
        P.table_fig(
            "Probe accuracy: next_rule",
            "15-class rule-classification accuracy at the deepest probed layer, linear probe, "
            "test split -- base vs RL decodability",
            ["model", "accuracy"],
            rows,
            note='<a href="interp_probes.html">Interp: Probes page &#8594;</a>',
        )
    ]


def render_overview(
    n_runs: int,
    n_suites: int,
    statuses: Mapping[str, int],
    page_links: Sequence[tuple[str, str]],
    view_data: Mapping[str, JSON],
) -> list[str]:
    """The dashboard's landing page (plan WP8 integration fix list item 4): a run-status summary,
    a one-line explanation of what `base`/`rl`/`armC` are, a headline number per benchmark (each
    linking to that benchmark's own page for the full picture), and the page index. Every headline
    helper above degrades gracefully (returns `[]`) when its view is empty/missing, so a benchmark
    still mid-run simply has no headline card yet rather than crashing the whole overview.
    """
    rows = [[status, str(count)] for status, count in sorted(statuses.items())]
    figs = [
        P.table_fig(
            "Run status summary",
            f"{n_runs} runs across {n_suites} suites",
            ["status", "count"],
            rows,
        ),
        P.figure(
            "Models on this dashboard",
            "what base / rl / armC mean, everywhere they appear below",
            "<ul>"
            "<li><b>base</b> &#8212; the pretrained cap-6 model, before any RL "
            "(<code>ab2/frozen/pretrain.pt</code>).</li>"
            "<li><b>rl</b> &#8212; the run-2 champion: <b>base</b> fine-tuned with RL on the "
            "full training pool.</li>"
            "<li><b>armC</b> &#8212; a length-generalisation experiment arm: an RL model trained "
            "on lengths 6-11 only (shown where that run exists).</li>"
            "</ul>",
            wide=True,
        ),
    ]
    for section in (
        _overview_yue,
        _overview_families,
        _overview_glue,
        _overview_padding,
        _overview_prefix,
        _overview_edl,
        _overview_probes,
    ):
        figs.extend(section(view_data))
    links_html = "".join(
        f'<li><a href="{P.esc(href)}">{P.esc(name)}</a></li>' for href, name in page_links
    )
    figs.append(P.figure("Pages", "jump to a benchmark", f"<ul>{links_html}</ul>", wide=True))
    return figs
