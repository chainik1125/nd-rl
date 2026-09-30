"""Dashboard pages for the interp package's four measures (`INTERP_SPEC.md`'s "Storage, views,
dashboard, CLI" section): `interp_likelihood`, `interp_steering`, `interp_weights`
(interpolation), `interp_probes`. Consumes `analysis/interp.py`'s `view_*` dicts and turns them
into figure HTML via `dash/primitives.py`, following `dash/pages.py`'s own `render_<page>`
convention (`data.get("empty")` -> `missing_fig`, else the real charts).

`dash/primitives.py` has no scatter-plot primitive (only `line_chart`, which connects points, and
categorical `bar_chart`/`columns`); rather than mis-render a scatter as a connected line, the
likelihood page's point-cloud relationships (single vs. marginal logp, base vs. RL, the crux set)
are shown as sortable tables (`P.table_fig`) instead -- faithful to every value, at the cost of not
being a literal scatter chart. The steering/interpolation/probes pages need no such workaround:
pass@1 vs. alpha, pass@k curves, and a metric vs. layer are all naturally `line_chart`s.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import cast

from ..core.types import JSON
from . import primitives as P

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


# --------------------------------------------------------------------------- interp_likelihood


def render_interp_likelihood(data: Mapping[str, JSON]) -> list[str]:
    if data.get("empty"):
        return [
            P.missing_fig("Ordering-marginalised likelihood", str(data.get("reason", "no data")))
        ]
    origins = cast("dict[str, dict[str, JSON]]", data["origins"])
    figs: list[str] = []
    for origin, d in origins.items():
        base_pts = cast("list[tuple[float, float]]", d["single_vs_marginal_base"])
        rl_pts = cast("list[tuple[float, float]]", d["single_vs_marginal_rl"])
        both = cast("list[tuple[float, float]]", d["base_vs_rl_marginal"])
        figs.append(
            P.table_fig(
                f"logp: single-order vs. marginal -- {origin}",
                f"n={d['n']}; marginal >= single always (log-sum-exp over >=1 orderings)",
                ["model", "logp_single", "logp_marginal"],
                [["base", P.fmt(s, "f3"), P.fmt(m, "f3")] for s, m in base_pts[:200]]
                + [["rl", P.fmt(s, "f3"), P.fmt(m, "f3")] for s, m in rl_pts[:200]],
                note="Truncated to 200 rows per model."
                if len(base_pts) > 200 or len(rl_pts) > 200
                else "",
            )
        )
        if both:
            figs.append(
                P.table_fig(
                    f"logp_marginal: base vs. RL -- {origin}",
                    f"n={len(both)}; a proof where RL >> base is a likelihood-crux candidate",
                    ["base logp_marginal", "rl logp_marginal", "rl - base"],
                    [
                        [P.fmt(b, "f3"), P.fmt(r, "f3"), P.fmt(r - b, "f3")]
                        for b, r in sorted(both, key=lambda t: t[0])[:200]
                    ],
                )
            )
        for label, hist in (("base", d["marginal_hist_base"]), ("rl", d["marginal_hist_rl"])):
            vals = cast("list[float]", hist)
            if not vals:
                continue
            figs.append(_histogram(f"logp_marginal histogram -- {origin}, {label}", vals))
    crux = cast("list[dict[str, JSON]]", data["crux_set"])
    if crux:
        figs.append(
            P.table_fig(
                "Crux set (RL solves, base almost never)",
                "p_hat_base < 0.5, ranked by rl_logp_marginal - base_logp_marginal",
                ["pid", "p_hat_base", "base_logp_marginal", "rl_logp_marginal", "gap"],
                [
                    [
                        cast(str, c["pid"]),
                        P.fmt(c["p_hat_base"], "pct"),
                        P.fmt(c["base_logp_marginal"], "f3"),
                        P.fmt(c["rl_logp_marginal"], "f3"),
                        P.fmt(
                            cast(float, c["rl_logp_marginal"])
                            - cast(float, c["base_logp_marginal"]),
                            "f3",
                        ),
                    ]
                    for c in sorted(
                        crux,
                        key=lambda c: (
                            cast(float, c["rl_logp_marginal"])
                            - cast(float, c["base_logp_marginal"])
                        ),
                        reverse=True,
                    )[:200]
                ],
            )
        )
    mean_dis = data.get("mean_disagree_frac")
    if mean_dis is not None:
        figs.append(
            P.table_fig(
                "Base/RL disagreement",
                "mean fraction of teacher-forced proof positions where argmax differs "
                "(RL-verified proofs)",
                ["mean disagreement rate"],
                [[P.fmt(mean_dis, "pct")]],
            )
        )
    return figs


def _histogram(title: str, values: Sequence[float], n_bins: int = 20) -> str:
    lo, hi = min(values), max(values)
    if lo == hi:
        hi = lo + 1.0
    width = (hi - lo) / n_bins
    counts = [0] * n_bins
    for v in values:
        b = min(n_bins - 1, max(0, int((v - lo) / width)))
        counts[b] += 1
    xs = [f"{lo + i * width:.2g}" for i in range(n_bins)]
    series = P.Series("count", [(i, c) for i, c in enumerate(counts)], _color(0))
    return P.columns(title, f"n={len(values)}", xs, [series], x_label="logp_marginal (bin start)")


# ------------------------------------------------------------------------------ interp_steering


def render_interp_steering(data: Mapping[str, JSON]) -> list[str]:
    if data.get("empty"):
        return [P.missing_fig("RL-base steering sweep", str(data.get("reason", "no data")))]
    curves = cast("dict[str, list[tuple[float, float, float, float]]]", data["curves_by_layer"])
    figs: list[str] = []
    series = []
    for i, (layer, pts) in enumerate(sorted(curves.items(), key=lambda kv: int(kv[0]))):
        series.append(
            P.Series(
                f"layer {layer}",
                [(a, mean) for a, mean, _lo, _hi in pts],
                _color(i),
                band=[(a, lo, hi) for a, _mean, lo, hi in pts],
            )
        )
    base_pass1 = cast("list[float] | None", data.get("base_pass1"))
    note = f"alpha=0 (unsteered base) pass@1 = {P.fmt(base_pass1[0], 'pct')}" if base_pass1 else ""
    figs.append(
        P.line_chart(
            "pass@1 vs steering alpha, by layer",
            "shaded band: 95% item-level bootstrap CI",
            series,
            x_label="alpha",
            y_fmt="pct",
            wide=True,
            note=note,
        )
    )
    rl_control = cast("dict[str, dict[str, JSON]]", data.get("rl_control", {}))
    if rl_control:
        bars = [
            P.Bar(
                f"layer {layer}",
                cast(float, d["pass1"]),
                cast("list[float]", d["ci"])[0],
                cast("list[float]", d["ci"])[1],
                _color(i),
            )
            for i, (layer, d) in enumerate(rl_control.items())
        ]
        figs.append(
            P.bar_chart(
                "RL steered with -delta (control)",
                "does removing the RL-base difference vector hurt RL's own pass@1?",
                bars,
                y_fmt="pct",
            )
        )
    return figs


# ----------------------------------------------------------------------------- interp_weights


def render_interp_weights(data: Mapping[str, JSON]) -> list[str]:
    if data.get("empty"):
        return [P.missing_fig("Weight interpolation", str(data.get("reason", "no data")))]
    figs: list[str] = []
    pass1 = cast("list[tuple[float, float, float, float]]", data["pass1_vs_alpha"])
    if pass1:
        series = P.Series(
            "pass@1",
            [(a, m) for a, m, _lo, _hi in pass1],
            _color(0),
            band=[(a, lo, hi) for a, _m, lo, hi in pass1],
        )
        figs.append(
            P.line_chart(
                "pass@1 vs alpha",
                "dev + sealed gen items, k=8; shaded band: 95% bootstrap CI",
                [series],
                x_label="alpha (0=base, 1=rl)",
                y_fmt="pct",
                wide=True,
            )
        )
    pass_k = cast("dict[str, list[tuple[int, float, float, float]]]", data["pass_k_by_alpha"])
    if pass_k:
        series_k = [
            P.Series(f"alpha={alpha}", [(float(k), mean) for k, mean, _lo, _hi in curve], _color(i))
            for i, (alpha, curve) in enumerate(sorted(pass_k.items(), key=lambda kv: float(kv[0])))
        ]
        figs.append(
            P.line_chart(
                "pass@k vs k, by alpha",
                "fixed 150-theorem subset, k up to 128",
                series_k,
                x_label="k",
                y_fmt="pct",
                wide=True,
                log_x=True,
                note="Sharpening (pass@1 rises, pass@128 flat) vs. coverage-only-near-alpha=1 "
                "is the question this pair of charts is meant to answer.",
            )
        )
    return figs


# ------------------------------------------------------------------------------- interp_probes


def render_interp_probes(data: Mapping[str, JSON]) -> list[str]:
    if data.get("empty"):
        return [P.missing_fig("Linear probes", str(data.get("reason", "no data")))]
    layers = cast("list[str]", cast("dict[str, JSON]", data["meta"])["layers"])
    figs: list[str] = []

    def layer_chart(title: str, sub: str, key: str, y_fmt: str = "pct") -> str | None:
        series_by_model = cast("dict[str, list[tuple[int, JSON]]]", data[key])
        series = []
        for i, (model, pts) in enumerate(series_by_model.items()):
            clean = [
                (float(idx), cast(float, v))
                for idx, v in pts
                if v is not None and not (isinstance(v, float) and math.isnan(v))
            ]
            if clean:
                series.append(P.Series(model, clean, _color(i)))
        if not series:
            return None
        return P.line_chart(
            title, sub, series, x_label=f"layer ({', '.join(layers)})", y_fmt=y_fmt, wide=True
        )

    for title, sub, key, y_fmt in (
        (
            "next_rule probe accuracy by layer",
            "15-class rule classification, test split",
            "next_rule_acc",
            "pct",
        ),
        ("lines_to_go probe R^2 by layer", "ridge regression, test split", "lines_to_go_r2", "f3"),
        (
            "lines_to_go 5-bin accuracy by layer",
            "quantile-binned, test split",
            "lines_to_go_bin_acc",
            "pct",
        ),
        (
            "cited_back probe accuracy by layer",
            "classes 1-8, 9+; test split",
            "cited_back_acc",
            "pct",
        ),
        (
            "lines_to_go length generalisation (OOD R^2)",
            "train L<=10, test sealed L12-14 (+ family witnesses)",
            "length_generalisation_ood_r2",
            "f3",
        ),
        (
            "next_rule MDL compression",
            "online code vs. uniform code",
            "mdl_next_rule_compression",
            "f3",
        ),
        (
            "lines_to_go (binned) MDL compression",
            "online code vs. uniform code",
            "mdl_lines_to_go_compression",
            "f3",
        ),
    ):
        fig = layer_chart(title, sub, key, y_fmt)
        if fig is not None:
            figs.append(fig)
    if not figs:
        figs.append(P.missing_fig("Linear probes", "run has rows but no non-empty metric series"))
    return figs
