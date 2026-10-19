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
from .pages import intro

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

_LIKELIHOOD_INTRO = intro(
    "Ordering-marginalised likelihood: how surprised is each model by a correct proof?",
    "<p>Most proof lines can be validly reordered within limits (e.g. premises can usually be "
    "listed in any order). Scoring a proof under only the one order it happens to be written in "
    "understates how likely a model thinks it is; this sums (or importance-samples) the "
    "log-probability over every valid reordering instead. The crux set below highlights proofs "
    "the base model almost never solves (<code>p_hat_base &lt; 0.5</code>) but that RL assigns "
    "high marginal likelihood to -- candidates for 'RL elicited a latent capability'.</p>",
)


def render_interp_likelihood(data: Mapping[str, JSON]) -> list[str]:
    if data.get("empty"):
        return [
            _LIKELIHOOD_INTRO,
            P.missing_fig("Ordering-marginalised likelihood", str(data.get("reason", "no data"))),
        ]
    figs: list[str] = [_LIKELIHOOD_INTRO]
    origins = cast("dict[str, dict[str, JSON]]", data["origins"])
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


_STEERING_INTRO = intro(
    "Steering: nudging the base model along the RL direction",
    "<p>For each layer, adds <code>alpha</code> times the mean difference between RL's and the "
    "base model's residual-stream activations (measured on the same teacher-forced proofs) to "
    "the base model's activations at that layer, then samples from the steered model as usual. "
    "<code>alpha=0</code> is the unsteered base model; <code>alpha=1</code> would match the mean "
    "difference's full size. If pass@1 rises with <code>alpha</code> at some layer, that "
    "direction alone explains part of the RL-base gap. The control below steers RL itself with "
    "<code>-delta</code>, checking whether removing the direction hurts RL's own performance.</p>",
)


def render_interp_steering(data: Mapping[str, JSON]) -> list[str]:
    if data.get("empty"):
        return [
            _STEERING_INTRO,
            P.missing_fig("RL-base steering sweep", str(data.get("reason", "no data"))),
        ]
    curves = cast("dict[str, list[tuple[float, float, float, float]]]", data["curves_by_layer"])
    figs: list[str] = [_STEERING_INTRO]
    series = []
    for i, (layer, pts) in enumerate(sorted(curves.items(), key=lambda kv: int(kv[0]))):
        # Defensive: a `nan` mean/CI bound (e.g. a run with no matching items on some pooled
        # suite) must not reach `line_chart` unfiltered -- same pattern as this module's own
        # `layer_chart` for the probes page.
        clean = [(a, mean, lo, hi) for a, mean, lo, hi in pts if not math.isnan(mean)]
        series.append(
            P.Series(
                f"layer {layer}",
                [(a, mean) for a, mean, _lo, _hi in clean],
                _color(i),
                band=[(a, lo, hi) for a, _mean, lo, hi in clean],
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


_WEIGHTS_INTRO = intro(
    "Weight interpolation: does capability appear gradually or only at the end?",
    "<p>Linearly blends the base and RL checkpoints' weights, "
    "<code>theta = base + alpha*(rl - base)</code>, and evaluates the blend at each "
    "<code>alpha</code> from 0 (pure base) to 1 (pure RL) and slightly beyond. If pass@1 rises "
    "smoothly with <code>alpha</code> while pass@128 (which averages out noise from many "
    "attempts) stays flat, that looks like 'sharpening' an existing capability; if both only "
    "move near <code>alpha=1</code>, the capability looks more like something RL specifically "
    "added rather than reweighted.</p>",
)


def render_interp_weights(data: Mapping[str, JSON]) -> list[str]:
    if data.get("empty"):
        return [
            _WEIGHTS_INTRO,
            P.missing_fig("Weight interpolation", str(data.get("reason", "no data"))),
        ]
    figs: list[str] = [_WEIGHTS_INTRO]
    pass1 = cast("list[tuple[float, float, float, float]]", data["pass1_vs_alpha"])
    if pass1:
        clean1 = [(a, m, lo, hi) for a, m, lo, hi in pass1 if not math.isnan(m)]
        series = P.Series(
            "pass@1",
            [(a, m) for a, m, _lo, _hi in clean1],
            _color(0),
            band=[(a, lo, hi) for a, _m, lo, hi in clean1],
        )
        figs.append(
            P.line_chart(
                "pass@1 vs alpha",
                "dev + sealed + textbook_dev gen items, k=8; shaded band: 95% bootstrap CI",
                [series],
                x_label="alpha (0=base, 1=rl)",
                y_fmt="pct",
                wide=True,
            )
        )
    pass_k = cast("dict[str, list[tuple[int, float, float, float]]]", data["pass_k_by_alpha"])
    if pass_k:
        series_k = [
            P.Series(
                f"alpha={alpha}",
                [(float(k), mean) for k, mean, _lo, _hi in curve if not math.isnan(mean)],
                _color(i),
            )
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


_PROBES_INTRO = intro(
    "Linear probes: is the information there, and does the model use it?",
    "<p>Trains simple linear (or ridge-regression) probes on each model's own internal "
    "activations, at each layer, to predict features of the proof the model hasn't written yet "
    "(the next inference rule, how many lines remain, how far back a citation points). High "
    "probe accuracy means that information is linearly readable from the model's internals at "
    "that layer. Base &#8776; RL accuracy suggests the information was already present in the "
    "base model and RL changed how it is <i>used</i>, not whether it exists; RL &gt;&gt; base "
    "suggests RL built the representation.</p>",
)


def render_interp_probes(data: Mapping[str, JSON]) -> list[str]:
    if data.get("empty"):
        return [_PROBES_INTRO, P.missing_fig("Linear probes", str(data.get("reason", "no data")))]
    layers = cast("list[str]", cast("dict[str, JSON]", data["meta"])["layers"])
    figs: list[str] = [_PROBES_INTRO]

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
    if len(figs) == 1:  # only the intro -- no metric series had any non-nan values
        figs.append(P.missing_fig("Linear probes", "run has rows but no non-empty metric series"))
    return figs
