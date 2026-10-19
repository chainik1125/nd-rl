"""Booktabs-style results tables for benchmarks whose figure would carry only a few numbers
(NeurIPS: tables have no vertical rules; booktabs rules; caption above). One row builder feeds both
the LaTeX file and the summary page's HTML table, from the same `ChartData.bars`."""

from __future__ import annotations

import html
import math
from dataclasses import dataclass

from ..dash.chart_data import ChartData


@dataclass(frozen=True)
class Cell:
    value: float
    lo: float | None = None
    hi: float | None = None
    best: bool = False


def _pct_digits(*xs: float) -> int:
    """Two decimals when a value is below 1% (so 0.14% vs 0.17% stay distinct), else one: the
    dashboard's `fmt(..., "pct")` rule."""
    return 2 if any(0 < abs(100 * x) < 1 for x in xs if x == x) else 1


def bar_rows(cd: ChartData) -> tuple[list[str], list[tuple[str, Cell]]]:
    """`(column headers, [(condition, cell)])` for a pct-valued bar chart; the largest value is
    marked best."""
    rows = [(b.name, b) for b in cd.bars if b.value == b.value]
    top = max((b.value for _n, b in rows), default=math.nan)
    return (
        ["Condition", "pass@1 (%)"],
        [(n, Cell(b.value, b.lo, b.hi, best=b.value == top)) for n, b in rows],
    )


def _parts(c: Cell) -> tuple[str, str | None, str | None]:
    d = _pct_digits(c.value, *(x for x in (c.lo, c.hi) if x is not None))
    v = f"{100 * c.value:.{d}f}"
    if c.lo is None or c.hi is None or c.lo != c.lo or c.hi != c.hi:
        return v, None, None
    return v, f"{100 * (c.hi - c.value):.{d}f}", f"{100 * (c.value - c.lo):.{d}f}"


def _tex_escape(s: str) -> str:
    out = s.replace("\\", r"\textbackslash{}")
    for ch in "&%$#_{}":
        out = out.replace(ch, "\\" + ch)
    out = out.replace("delta", r"$\delta$").replace(" a=", r" $\alpha$=")
    return out.replace("->", r"$\to$").replace("alpha)", r"$\alpha$)")


def to_latex(cd: ChartData, label: str, caption: str) -> str:
    """A `table` float with booktabs rules and an asymmetric CI as `value^{+up}_{-down}`."""
    headers, rows = bar_rows(cd)
    lines = [
        r"\begin{table}[t]",
        r"  \caption{" + _tex_escape(caption) + "}",
        r"  \label{tab:" + cd.key + "}",
        r"  \centering",
        r"  \begin{tabular}{lr}",
        r"    \toprule",
        "    " + " & ".join(_tex_escape(h) for h in headers) + r" \\",
        r"    \midrule",
    ]
    for name, c in rows:
        v, up, down = _parts(c)
        val = v if up is None else f"{v}^{{+{up}}}_{{-{down}}}"
        val = (
            rf"\mathbf{{{v}}}" + (f"^{{+{up}}}_{{-{down}}}" if up is not None else "")
            if c.best
            else val
        )
        lines.append(f"    {_tex_escape(name)} & ${val}$ \\\\")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]
    del label
    return "\n".join(lines)


def pretty(name: str) -> str:
    """Display form of a condition name: "base + delta (layer 0, a=0.25)" -> "base + δ (layer 0,
    α = 0.25)"; the data table keeps the raw name."""
    return name.replace("delta", "δ").replace(" a=", " α = ").replace(" - ", " − ")


def to_html(cd: ChartData) -> str:
    headers, rows = bar_rows(cd)
    th = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = []
    for name, c in rows:
        v, up, down = _parts(c)
        num = f"<b>{v}</b>" if c.best else v
        ci = (
            f'<span class="pm"><sup>+{up}</sup><sub>&minus;{down}</sub></span>'
            if up is not None
            else ""
        )
        body.append(f"<tr><td>{html.escape(pretty(name))}</td><td>{num}{ci}</td></tr>")
    return (
        f'<table class="booktabs"><thead><tr>{th}</tr></thead>'
        f"<tbody>{''.join(body)}</tbody></table>"
    )
