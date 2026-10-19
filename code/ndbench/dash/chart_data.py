"""`ChartData`: the data contract between the summary page (`dash/summary.py`, which computes
it from the `Store` and tabulates it) and the paper figures (`figures/plots.py`, which draws it).
Kept in its own module so neither side imports the other."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from . import primitives as P


@dataclass
class ChartData:
    """One benchmark's summary data, computed once from the `Store` and drawn twice: as the paper
    figure (`code/ndbench/figures/plots.py` reads only these fields) and as the page's
    collapsible data table (`P.line_table_rows`/`P.bar_table_rows`/`P.dot_table_rows`, the exact
    tables the earlier inline-SVG charts carried). Nothing downstream recomputes a number, so the
    page, its tables and the PDF figures cannot drift apart.

    `kind` names the primitive whose data shape `series`/`bars`/`rows` follow (and so which table
    the page shows); `opts` holds that primitive's formatting keywords (`x_label`, `y_fmt`, ...).
    `extra` carries the extra panels some paper figures add from numbers this module already
    computes for its takeaway or caption (e.g. the textbook_law curves); each such panel's values
    also get a table in `extra_tables`, so nothing is plotted that the page does not tabulate.
    `as_table` asks the page to show a booktabs results table in place of a figure (a few bars
    with near-zero values read better as numbers).
    """

    key: str
    title: str
    kind: str = "missing"
    takeaway: str = ""
    caption: str = ""
    series: list[P.Series] = field(default_factory=lambda: list[P.Series]())
    bars: list[P.Bar] = field(default_factory=lambda: list[P.Bar]())
    rows: list[P.DotRow] = field(default_factory=lambda: list[P.DotRow]())
    opts: dict[str, Any] = field(default_factory=lambda: dict[str, Any]())
    extra: dict[str, Any] = field(default_factory=lambda: dict[str, Any]())
    extra_tables: list[tuple[str, list[str], list[list[str]]]] = field(
        default_factory=lambda: list[tuple[str, list[str], list[list[str]]]]()
    )
    missing: str = ""
    as_table: bool = False


def missing_chart(key: str, title: str, reason: str) -> ChartData:
    """A chart with no data yet: the page shows its title and `reason` instead of a figure."""
    return ChartData(key, title, missing=reason)
