"""Tests for `ndbench.figures` (the NeurIPS-style figures) and the paper-style summary page.

- every benchmark's figure function renders on a synthetic store (wide and narrow layouts, and the
  Figure 1 overview) and writes PDF (TrueType fonts) / SVG / 300-dpi PNG;
- `figures.build` writes every numbered item, the booktabs table and `captions.md`, byte-identically
  on a rebuild, also through `python -m code.ndbench figures`;
- every model keeps its fixed colour in every figure, and new model families get theirs from a
  fixed registry;
- regression: the summary page's data tables, takeaways and caption numbers equal those of the
  pre-figures summary (`fixtures/summary_golden_synth.json`, rendered by commit 8795470 from
  `build_fully_populated_store`).
"""

from __future__ import annotations

import html as H
import json
import re
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from matplotlib.colors import to_hex
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from PIL import Image

from code.ndbench import cli
from code.ndbench.dash import primitives as P
from code.ndbench.dash import summary
from code.ndbench.dash.chart_data import ChartData
from code.ndbench.figures import build as fig_build
from code.ndbench.figures import plots, tables
from code.ndbench.figures import style as S
from code.ndbench.store import Store

from .test_wp6_summary import build_fully_populated_store

GOLDEN = Path(__file__).parent / "fixtures" / "summary_golden_synth.json"


@pytest.fixture(scope="module")
def synth(tmp_path_factory: pytest.TempPathFactory) -> tuple[Store, list[ChartData]]:
    store = Store(tmp_path_factory.mktemp("paper_figures") / "store")
    build_fully_populated_store(store)
    return store, summary.chart_data(store)


def _charts(synth: tuple[Store, list[ChartData]]) -> list[ChartData]:
    return synth[1]


# ------------------------------------------------------------------------------ figure functions


def test_every_benchmark_has_data_and_a_figure_function(
    synth: tuple[Store, list[ChartData]],
) -> None:
    charts = _charts(synth)
    assert [c.key for c in charts] == list(plots.FIGURES)
    assert all(c.kind != "missing" for c in charts), [c.missing for c in charts]


def test_each_figure_renders_and_writes_pdf_svg_png(
    synth: tuple[Store, list[ChartData]], tmp_path: Path
) -> None:
    for cd in _charts(synth):
        with S.paper_style():
            fig = plots.render(cd)
            assert fig is not None, cd.key
            assert fig.get_figwidth() == pytest.approx(plots.FIGURES[cd.key][1])
            pdf, svg, png = S.save(fig, tmp_path / cd.key)
            S.close(fig)
            narrow = plots.render(cd, narrow=True)
            assert (narrow is not None) == (cd.key in plots.NARROW), cd.key
            if narrow is not None:
                assert narrow.get_figwidth() == pytest.approx(S.NARROW_W)
                S.close(narrow)
        assert pdf.read_bytes().startswith(b"%PDF")
        assert b"<svg" in svg.read_bytes()[:400]
        with Image.open(png) as im:
            assert abs(im.size[0] - round(plots.FIGURES[cd.key][1] * 300)) <= 1, cd.key


def test_overview_renders_wide_and_narrow(synth: tuple[Store, list[ChartData]]) -> None:
    for narrow, width, n_axes in ((False, S.FULL_W, 12), (True, S.NARROW_W, 12)):
        with S.paper_style():
            fig = plots.fig_overview(_charts(synth), narrow=narrow)
            assert fig.get_figwidth() == pytest.approx(width)
            assert len(fig.axes) == n_axes
            S.close(fig)


def test_figures_handle_an_all_missing_store(tmp_path: Path) -> None:
    charts = summary.chart_data(Store(tmp_path / "empty"))
    assert all(c.kind == "missing" for c in charts)
    assert all(plots.render(c) is None for c in charts)
    assert plots.overview_uri(charts) is None
    written = fig_build.write_figures(charts, tmp_path / "figs")
    assert [p.name for p in written] == ["captions.md"]


def test_pdf_embeds_truetype_fonts_not_type3(
    synth: tuple[Store, list[ChartData]], tmp_path: Path
) -> None:
    cd = next(c for c in _charts(synth) if c.key == "yue")
    with S.paper_style():
        fig = plots.render(cd)
        assert fig is not None
        pdf = S.save(fig, tmp_path / "yue")[0].read_bytes()
        S.close(fig)
    assert b"/FontFile2" in pdf  # TrueType (Type 42) outlines embedded
    assert b"/Type3" not in pdf
    face = b"NimbusRoman" if S.SERIF == "Nimbus Roman" else b"STIX"
    assert face in pdf


# ------------------------------------------------------------------------------ fixed colours


def _model_of(label: str) -> str | None:
    name = label.lstrip("_").split(",", 1)[0].strip()
    return name if name in S.MODEL_STYLES else None


def _check_colours(fig: Figure) -> int:
    """Every artist labelled with a model (lines, hidden-label markers, bars) wears that model's
    fixed colour; returns how many were checked."""
    checked = 0
    for ax in fig.axes:
        for line in ax.get_lines():
            model = _model_of(str(line.get_label()))
            if model is None:
                continue
            want = S.MODEL_STYLES[model].color.lower()
            got = {
                to_hex(line.get_color()),
                to_hex(line.get_markerfacecolor()),
                to_hex(line.get_markeredgecolor()),
            }
            assert want in got, (model, got)
            checked += 1
        for patch in ax.patches:
            model = _model_of(str(patch.get_label())) if isinstance(patch, Rectangle) else None
            if model is not None:
                assert to_hex(patch.get_facecolor()) == S.MODEL_STYLES[model].color.lower()
                checked += 1
    for leg in fig.legends:
        for text, handle in zip(leg.get_texts(), leg.legend_handles, strict=True):
            model = _model_of(text.get_text())
            if model is None or handle is None:
                continue
            colours: set[str] = set()
            if isinstance(handle, Line2D):
                colours = {
                    to_hex(handle.get_color()),
                    to_hex(handle.get_markerfacecolor()),
                    to_hex(handle.get_markeredgecolor()),
                }
            elif isinstance(handle, Rectangle):
                colours = {to_hex(handle.get_facecolor())}
            assert S.MODEL_STYLES[model].color.lower() in colours, (model, colours)
            checked += 1
    return checked


def test_models_keep_their_fixed_colours_in_every_figure(
    synth: tuple[Store, list[ChartData]],
) -> None:
    assert S.MODEL_STYLES["base"].color == S.OKABE_ITO["blue"]
    assert S.MODEL_STYLES["rl"].color == S.OKABE_ITO["vermillion"]
    assert S.MODEL_STYLES["armC"].color == S.OKABE_ITO["bluish_green"]
    assert len({st.marker for st in S.MODEL_STYLES.values()}) == 3  # greyscale-safe
    total = 0
    with S.paper_style():
        for cd in _charts(synth):
            fig = plots.render(cd)
            assert fig is not None
            total += _check_colours(fig)
            S.close(fig)
        fig = plots.fig_overview(_charts(synth))
        total += _check_colours(fig)
        S.close(fig)
    assert total > 40


def test_colour_follows_the_model_not_the_series_order(
    synth: tuple[Store, list[ChartData]],
) -> None:
    cd = next(c for c in _charts(synth) if c.key == "subst")
    flipped = replace(cd, series=list(reversed(cd.series)))
    with S.paper_style():
        for c in (cd, flipped):
            fig = plots.render(c)
            assert fig is not None
            assert _check_colours(fig) >= 2
            S.close(fig)


def test_new_model_families_get_fixed_hues_shades_and_seed_styles() -> None:
    assert S.parse_model_label("dan12-t1-s0") == ("dan12", "t1", 0)
    assert S.parse_model_label("famx") == ("famx", None, None)
    unregistered = S.model_style("zzz-rl-s0")
    assert unregistered.color == S.shade(S.REF, 0.0)  # grey until registered, never a new hue
    registry = S._FAMILY_HUE  # noqa: SLF001
    before = dict(registry)
    try:
        hue = S.register_family("testfam_a")
        assert hue in S.FAMILY_HUES and hue not in {m.color for m in S.MODEL_STYLES.values()}
        assert S.register_family("testfam_a") == hue  # stable
        rl_seed0, rl_seed1 = S.model_style("testfam_a-rl-s0"), S.model_style("testfam_a-rl-s1")
        base_seed0 = S.model_style("testfam_a-base-s0")
        assert rl_seed0.color == hue and rl_seed1.color == hue
        assert base_seed0.color == S.shade(hue, S.ROLE_SHADE["base"])  # pretrained: darker shade
        assert rl_seed0.linestyle != rl_seed1.linestyle  # seeds: line style
        assert rl_seed0.marker == base_seed0.marker  # one marker per family
    finally:
        registry.clear()
        registry.update(before)


def test_percent_formatter_uses_the_tick_step() -> None:
    f = S.PctFormatter()
    f.set_locs([0.0, 0.1, 0.2])
    assert [f(x) for x in (0.0, 0.1, 0.2)] == ["0%", "10%", "20%"]
    f.set_locs([0.0, 0.005, 0.01])
    assert [f(x) for x in (0.0, 0.005, 0.01)] == ["0.0%", "0.5%", "1.0%"]


# ------------------------------------------------------------------------------- build and CLI


def test_write_figures_writes_every_item_and_is_byte_stable(
    synth: tuple[Store, list[ChartData]], tmp_path: Path
) -> None:
    charts = _charts(synth)
    a = fig_build.write_figures(charts, tmp_path / "a")
    b = fig_build.write_figures(charts, tmp_path / "b")
    names = sorted(p.name for p in a)
    assert names == sorted(p.name for p in b)
    for name in names:
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes(), name
    nums = summary.numbering(charts)
    expected = {"fig01_overview.pdf", "captions.md", "tab01_steering.tex", "fig_steering.pdf"}
    for cd in charts:
        label = nums[cd.key]
        if label.startswith("Figure"):
            n = int(label.split()[1])
            expected |= {f"fig{n:02d}_{cd.key}.{ext}" for ext in ("pdf", "svg", "png")}
    assert expected <= set(names)
    assert nums["steering"] == "Table 1" and nums["probes"] == "Figure 12"
    captions = (tmp_path / "a" / "captions.md").read_text()
    assert "**Figure 1:**" in captions and "**Table 1:**" in captions
    for cd in charts:
        assert cd.title in captions
    tex = (tmp_path / "a" / "tab01_steering.tex").read_text()
    for rule in (r"\toprule", r"\midrule", r"\bottomrule"):
        assert rule in tex
    assert "|" not in tex  # no vertical rules
    assert r"\mathbf{" in tex and "^{+" in tex  # best value bold, asymmetric CI


def test_figures_cli(synth: tuple[Store, list[ChartData]], tmp_path: Path) -> None:
    store = synth[0]
    out = tmp_path / "cli_figs"
    rc = cli.main(["figures", "--out", str(store.root), "--fig-dir", str(out)])
    assert rc == 0
    assert (out / "captions.md").exists() and (out / "fig02_yue.pdf").exists()


def test_results_table_marks_best_and_writes_asymmetric_ci() -> None:
    cd = ChartData(
        "steering",
        "Steering",
        "bar",
        bars=[P.Bar("a", 0.20, 0.15, 0.30), P.Bar("b", 0.004, 0.001, 0.010)],
        as_table=True,
    )
    _hdr, rows = tables.bar_rows(cd)
    assert [c.best for _n, c in rows] == [True, False]
    out = tables.to_html(cd)
    assert "<b>20.0</b>" in out and "+10.0" in out and "&minus;5.0" in out
    assert "0.40" in out  # below 1%: two decimals


# ------------------------------------------------------------------------- page and regression


def _cards(page: str) -> list[str]:
    return re.findall(r'<figure class="card chart-card[^"]*"[^>]*>(.*?)</figure>', page, re.S)


def _tables(card: str) -> list[list[list[str]]]:
    out = []
    for t in re.findall(r'<details class="data">.*?<table>(.*?)</table>', card, re.S):
        rows = re.findall(r"<tr>(.*?)</tr>", t, re.S)
        out.append(
            [[H.unescape(c) for c in re.findall(r"<t[hd]>(.*?)</t[hd]>", r, re.S)] for r in rows]
        )
    return out


def _caption(card: str) -> str:
    m = re.search(r"<figcaption[^>]*>(.*?)</figcaption>", card, re.S)
    assert m is not None
    return H.unescape(re.sub(r"<[^>]+>", "", m.group(1)))


def test_summary_numbers_unchanged_against_the_pre_figures_summary(
    synth: tuple[Store, list[ChartData]],
) -> None:
    golden: list[dict[str, Any]] = json.loads(GOLDEN.read_text())
    page = summary.render_page(_charts(synth), gpu_hours=0.0)
    cards = _cards(page)
    assert len(golden) == len(cards) == 12
    for old, card in zip(golden, cards, strict=True):
        new_tables = _tables(card)
        # the main data table is exactly the old chart's table; extra panels add tables after it.
        assert new_tables[: len(old["tables"])] == old["tables"], old["title"]
        cap = _caption(card)
        takeaway = old["takeaway"]
        assert takeaway[1:] in cap, (old["title"], takeaway)
        for token in re.findall(r"\d+(?:\.\d+)?%?", old["note"]):
            assert token in cap, (old["title"], token)


def test_summary_page_is_a_numbered_results_section(
    synth: tuple[Store, list[ChartData]],
) -> None:
    page = summary.render_page(_charts(synth), gpu_hours=1.5, detail_href="detail/index.html")
    assert "<title>ND Capability Benchmarks</title>" in page
    assert len(page.encode()) < 16 * 1024 * 1024
    assert "http://" not in page and "https://" not in page
    assert 'href="detail/index.html"' in page
    assert page.count('class="fignum">Figure 1:') == 1
    for n in range(2, 13):
        assert page.count(f'class="fignum">Figure {n}:') == 1, n
    assert page.count('class="fignum">Table 1:') == 1
    assert '<table class="booktabs">' in page
    assert page.count("<picture>") == 1 + len(plots.NARROW)  # overview + full-width figures
    assert "Limitations" in page
    assert page.index("Figure 1:") < page.index("Figure 2:") < page.index("Limitations")
