---
written_on: 2026-10-01
---

# Figure style for the ndbench paper figures

The rules `style.py` and `plots.py` apply, where each comes from, and what was taken from recent
ML papers. Figures are built with `python -m code.ndbench figures --out <store> --fig-dir <dir>`;
the summary page (`dash --summary`) embeds the same figures.

## NeurIPS formatting rules

From the official NeurIPS 2026 style files (`neurips_2026.sty` and `neurips_2026.tex` in
[Formatting_Instructions_For_NeurIPS_2026.zip](https://media.neurips.cc/Conferences/NeurIPS2026/Formatting_Instructions_For_NeurIPS_2026.zip),
linked from the [call for papers](https://neurips.cc/Conferences/2026/CallForPapers)):

| Rule in the style files | How the figures follow it |
| --- | --- |
| Text block 5.5 in wide (`textwidth=5.5in`), 9 in tall; "specify the figure width as a multiple of the line width" | Full-width figures are exactly 5.5 in wide (`FULL_W`), side-by-side subfigures 2.65 in (`HALF_W`, two at `0.48\linewidth`). Saved without a tight-bbox crop, so `\includegraphics[width=\linewidth]` keeps the designed text size. |
| Times is the body face (`\rmdefault{ptm}`), 10 pt on 11 pt leading; `\small` 9 pt, `\footnotesize` at least 8 pt, `\scriptsize` 7 pt, `\tiny` 6 pt | Nimbus Roman (URW's Times clone) with STIX math; 8 pt axis labels and panel labels, 7 pt ticks and legends, 6.5 pt ticks in the dense Figure 1 overview, nothing below 6 pt except the bar-value annotations of the overview (5.5 pt). Falls back to matplotlib's STIXGeneral where Nimbus is not installed. |
| "The figure number and caption always appear after the figure"; caption "lower case (except for the first word and proper nouns)"; "numbered consecutively" | No titles inside the axes. Captions are written below as "Figure N: ...", sentence case, numbered in page order (`dash.summary.numbering`); panels carry only a short left-aligned label such as "(a) textbook_dev, n = 58 theorems". |
| Sample caption: "Explain what the figure shows and add a key take-away message to the caption" | Each caption is the bold short title, then what is plotted with n and the CI method, then the computed one-sentence takeaway from the summary page. |
| "You may use color figures. However, it is best for the figure captions and the paper body to be legible if the paper is printed in either black/white or in color" | Every model has a colour *and* a marker, line style and bar hatch; checked by rendering the PNGs in greyscale. |
| "All artwork must be neat, clean, and legible" | Thin axes (0.6 pt), outward ticks, no top/right spines, light (0.5 pt, #e3e3e3) or no grid, frameless legends outside or in empty corners. |
| Tables: "number and title always appear before the table"; "publication-quality tables do not contain vertical rules"; use `booktabs` | Steering is a table (`tables.py`): caption above, `\toprule`/`\midrule`/`\bottomrule`, no vertical rules, the best value in bold. |
| Checklist on error bars: state the method, say what the bar spans, and avoid symmetric bars for asymmetric quantities | CIs are the views' own 95% item-level bootstrap intervals, drawn as light bands with thin edges (curves) or capped whiskers (bars, forest plot), and each caption names the method and B. The table writes a CI as `value^{+up}_{-down}`, never a symmetric ±. Figures without a CI say "point estimates". |
| PDF fonts must be embedded | `pdf.fonttype = 42` (TrueType, no Type 3 bitmaps); SVG glyphs as paths; fixed metadata so rebuilds are byte-identical. |

## Figure practice

- Colour: the [Okabe & Ito colour-universal-design palette](https://jfly.uni-koeln.de/color/).
  base is blue `#0072B2` with circles, rl vermillion `#D55E00` with squares and a `//` hatch, armC
  bluish green `#009E73` with triangles, a dashed line and a dotted hatch. As a categorical set
  these pass the dataviz validator (all pairs separate under deutan, protan and tritan simulation;
  each is at least 3:1 against white). Non-model series (pass@1 vs pass@128, the position-only
  baseline, R = 1) are black or grey, never a model's colour. The three skill lines are all rl, so
  they share rl's hue and differ by shade, marker and dash.
- [Ten simple rules for better figures](https://doi.org/10.1371/journal.pcbi.1003833) (Rougier,
  Droettboom and Bourne, 2014): know the message, adapt to the medium, avoid chart junk, use colour
  with care, do not trust the defaults.
- One claim per figure. Rows of dot plots are sorted so the families where something holds up sit
  at the top; base and rl sit next to each other in every group, and a connector joins them where
  the gap is the claim (EDL, likelihood).
- Axes: percentages formatted as %, with as many decimals as the tick step needs; pass@k on a
  log-2 axis with ticks at 1, 4, 16, ... (1, 16, 256, 4096 in the overview); EDL on a log-10 axis
  labelled 0.1, 1, 10, 100; pass rates start at 0, while probe accuracy does not (the claim is a
  1-point gap above a 73% baseline).
- `matplotlib` text and PDF font handling follow the
  [matplotlib font guide](https://matplotlib.org/stable/users/explain/text/fonts.html).

## What the surveyed papers do and what was borrowed

Read from the arXiv HTML versions on 2026-10-01; this records the conventions noticed, not a
review of the papers.

| Paper | Convention noticed | Borrowed here |
| --- | --- | --- |
| Yue et al. 2025, "Does reinforcement learning really incentivize reasoning capacity in LLMs beyond the base model?" ([arXiv:2504.13837](https://arxiv.org/abs/2504.13837)) | pass@k against k on a log-2 axis (ticks 1, 4, 16, ...), percent y-axis, base vs RLVR model as a fixed pair of encodings; one panel per benchmark in a grid | Figure 2's log-2 k axis and % y; base vs rl fixed encodings in every panel |
| Brown et al. 2024, "Large Language Monkeys" ([arXiv:2407.21787](https://arxiv.org/abs/2407.21787)) | coverage against samples on a log x-axis as small multiples, one per task, legend shared outside the panels; captions state the finding | small multiples with one legend above all panels (Figures 1, 2, 5, 9); takeaway sentence in every caption |
| Liu et al. 2025, "ProRL: prolonged reinforcement learning expands reasoning boundaries" ([arXiv:2505.24864](https://arxiv.org/abs/2505.24864)) | a multi-panel Figure 1 summarising the headline results; booktabs results tables with the best score in bold | Figure 1 as a compact 3 x 4 overview grid; bold best value in Table 1 |
| Shao et al. 2025, "Spurious rewards: rethinking training signals in RLVR" ([arXiv:2506.10947](https://arxiv.org/abs/2506.10947)) | grids of panels across model families with one colour per condition kept fixed across panels; captions quote numbers | colour follows the entity in every figure; captions keep the computed numbers |
| Wu et al. 2025, "The invisible leash: why RLVR may not escape its origin" ([arXiv:2507.14843](https://arxiv.org/abs/2507.14843)) | tables report raw counts next to derived ratios so readers can check them | every figure's exact values stay in a collapsible data table on the page |

Where these papers draw no uncertainty (ProRL, Spurious rewards), the figures here keep the CIs
the analysis views compute, since the audited numbers include them and the NeurIPS checklist asks
for them.

## Chart type per benchmark

| Benchmark | Chart | Layout |
| --- | --- | --- |
| Overview | 3 x 4 small multiples of each benchmark's headline comparison | full width (6 x 2 on phones) |
| Textbook pass@k | lines with markers and 95% CI bands; (a) textbook_dev, (b) textbook_law | full width |
| Restricted families | dot plot, one row per family, hollow marker at 0 = fails at the smallest unseen size, arrow = censored | half |
| Glue | grouped bars with capped 95% CIs, values printed on near-zero bars, R = 1 reference | half |
| Skills | (a) rl lines by skill with CI bands, (b, c) grouped bars per skill on shared items, shared y | full |
| Laws x substitution, padding, prefix | lines with markers | half each |
| EDL | (a) dumbbell dot plot on log x with per-order whiskers, null control set apart; (b) EDL* against starting loss, log-log | full |
| Likelihood | dumbbell dot plot, one row per proof set | half |
| Steering | booktabs Table 1 (forest-plot variant `fig_steering.*`) | half |
| Weight interpolation | pass@1 and pass@128 lines with CI bands, extrapolation region shaded | half |
| Probes | lines with markers and a dashed position-only reference | half |

## Adding models

`style.model_style(label)` returns a model's fixed `ModelStyle`. base, rl and armC are fixed in
`MODEL_STYLES`. For new model families (for example Dan's state-env models, labelled like
`dan12-t1-s0`): call `register_family("dan12")` once, in a fixed order, to give the family the
next hue in `FAMILY_HUES` (reddish purple, sky blue, orange, black; never generated, so there is no
ninth hue: unregistered families are grey). `parse_model_label` splits a label into family, role
and seed; the role sets the shade (`ROLE_SHADE`: pretrained darker, RL at full hue), the seed sets
the line style (`SEED_LINESTYLES`), and the family's position sets the marker. Sky blue and orange
are below 3:1 contrast on white, so those families rely on marker and direct labels as well.

## Summary page

The page shows every figure at one scale (120 CSS px per inch, `dash.summary.PX_PER_IN`), so
8 pt figure text reads at about 0.8 of the 16 px serif body text, as against 10 pt text in the
paper. Figures sit on white in both colour schemes. Full-width figures have a stacked narrow
layout (`plots.NARROW`) that phones get through `<picture>`, and clicking a figure opens its SVG
alone for zooming.

## State-env models section (WP-D5)

Dan's twelve checkpoints get their own flag-gated section (`dash --summary --with-dan`) and
standalone `dan.html` detail page (`dash/dan_pages.py`), plus `figures --with-dan` writing
`figD0N_dan_<key>.{pdf,svg,png}` / `tabD01_dan_crossfamily.tex` -- a separate "Dn" numbering
sequence from the paper's own twelve, never interleaved with it. Everything here is additive:
`dash.summary.chart_data`/the paper's own `figures.plots.FIGURES` are untouched, so the existing
base/rl/armC page and the golden regression (`test_paper_figures.py`) render byte-identically
with the flag off.

Every multi-value Dan chart is small multiples, one panel per `plots.DAN_FAMILY_ORDER`
(`"ours" | "dan12" | "dan6"`, left to right): `dan_panel_figure` draws each panel from
`ChartData.extra["panels"][family]` -- a complete sub-`ChartData` for just that family, built by
`dash/dan_pages.py` from the ordinary `view_<bench>(store, cfg={"models": ..., "judge_id": ...})`
call most benchmarks already support. A family with no matching runs yet draws as a clean "not
available" panel (`_mini_missing`), never a crash -- expected for every family but `dan12`/`ours`
until more of the real Dan runs land. `_dan_style(label)` is `style.model_style(label)` except a
`REF_SUFFIX` (`" (ref)"`)-marked label -- our rl plotted as a reference line on a Dan panel --
which always renders grey and dashed regardless of family (STYLE.md's own rule: a non-model
reference series is never a model's colour). `dan_heldout`/`dan_judges` (single scalar per model)
are one grouped bar chart across every model instead of per-family panels, since a one-bar panel
per family would not read as small multiples.
