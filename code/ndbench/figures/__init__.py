"""`ndbench.figures`: the benchmark charts as NeurIPS-style matplotlib figures.

`style` is the single style definition (fonts, sizes, palette, fixed per-model encodings; see
`STYLE.md` for the rules and their sources), `plots` draws one figure per benchmark plus the
Figure 1 overview from `dash.chart_data.ChartData` (the objects `dash.summary` computes and
tabulates, so no number is recomputed here), `tables` holds the booktabs results tables, and
`build` writes every figure as PDF/SVG/PNG with a `captions.md` (`python -m code.ndbench figures`).

This package imports nothing at package level, and `plots`/`tables` import only
`dash.chart_data`/`dash.primitives`, so `dash.summary` can import `figures.plots` while
`figures.build` imports `dash.summary` without a cycle.
"""
