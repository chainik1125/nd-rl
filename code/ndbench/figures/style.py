"""The single figure style: NeurIPS text widths, Times-like serif text, thin axes, a fixed
colour-blind-safe encoding per model (rules and sources in `STYLE.md`).

Use `paper_style()` (a `matplotlib.rc_context`) around any figure code, `model_style(label)` for a
model's colour/marker/line/hatch, and `save(fig, base)` to write the PDF/SVG/PNG triple.
"""

from __future__ import annotations

import io
import math
import re
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, cast

import matplotlib

matplotlib.use("Agg")

from matplotlib import font_manager  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.ticker import (  # noqa: E402
    FixedLocator,
    Formatter,
    FuncFormatter,
    LogLocator,
    MaxNLocator,
    NullFormatter,
)

# ------------------------------------------------------------------------------------ geometry

#: NeurIPS text block: 5.5 in x 9 in (`neurips_2026.sty`: `textwidth=5.5in`).
TEXT_WIDTH_IN = 5.5
#: Full-width figure (`\includegraphics[width=\linewidth]`).
FULL_W = TEXT_WIDTH_IN
#: One of two side-by-side subfigures (`0.48\linewidth` each, leaving a 0.2 in gutter).
HALF_W = 2.65
#: Stacked layout of a full-width figure, for the summary page on a phone (not for the paper).
NARROW_W = 2.9

# ---------------------------------------------------------------------------------------- fonts

#: Nimbus Roman is URW's metric clone of Times, the NeurIPS body face (`\rmdefault{ptm}`).
_NIMBUS_DIR = Path("/usr/share/fonts/opentype/urw-base35")
_NIMBUS_FILES = (
    "NimbusRoman-Regular.otf",
    "NimbusRoman-Italic.otf",
    "NimbusRoman-Bold.otf",
    "NimbusRoman-BoldItalic.otf",
)


def _register_fonts() -> str:
    """Register Nimbus Roman with matplotlib if installed and return the serif family to use;
    falls back to matplotlib's bundled STIXGeneral (also a Times design) elsewhere."""
    found = False
    for name in _NIMBUS_FILES:
        path = _NIMBUS_DIR / name
        if path.exists():
            font_manager.fontManager.addfont(str(path))
            found = True
    return "Nimbus Roman" if found else "STIXGeneral"


SERIF = _register_fonts()

#: Point sizes. Body text is 10 pt; figure text sits one to three steps below it.
FS_LABEL = 8.0  # axis labels, panel labels
FS_TICK = 7.0  # tick labels, legends, annotations
FS_SMALL = 6.5  # the dense overview figure's ticks (never below the 6 pt \tiny floor)

# --------------------------------------------------------------------------------------- colours

#: Okabe & Ito (2008) colour-universal-design palette.
OKABE_ITO: dict[str, str] = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
}
INK = "#000000"
INK2 = "#404040"  # secondary text, error bars
REF = "#6b6b6b"  # reference lines (position-only lookup, R = 1, ...)
GRID = "#e3e3e3"
SHADE = "#f0f0f0"  # background regions (extrapolation)
BAND_ALPHA = 0.11
BAND_EDGE_ALPHA = 0.55
WHITE = "#ffffff"


def mix(hex_color: str, other: str, t: float) -> str:
    """Linear sRGB mix: `t=0` is `hex_color`, `t=1` is `other`."""

    def rgb(h: str) -> tuple[int, int, int]:
        h = h.lstrip("#")
        return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)

    a, b = rgb(hex_color), rgb(other)
    return "#" + "".join(f"{round(x + (y - x) * t):02X}" for x, y in zip(a, b, strict=True))


def shade(hex_color: str, t: float) -> str:
    """Darken (`t < 0`, towards black) or lighten (`t > 0`, towards white) by `|t|`."""
    return mix(hex_color, "#000000", -t) if t < 0 else mix(hex_color, "#FFFFFF", t)


# ---------------------------------------------------------------------------------- model styles


@dataclass(frozen=True)
class ModelStyle:
    """Everything that identifies one model in a figure. Colour is never the only channel:
    each model also has its own marker, line style and bar hatch, so figures read in greyscale."""

    label: str
    color: str
    marker: str
    linestyle: Any = "-"
    linewidth: float = 1.2
    hatch: str | None = None
    #: hollow markers mark a variant of a model (e.g. "rl natural" vs "rl padded").
    filled: bool = True

    def line_kw(self, *, markers: bool = True) -> dict[str, Any]:
        kw: dict[str, Any] = {
            "color": self.color,
            "linestyle": self.linestyle,
            "linewidth": self.linewidth,
            "label": self.label,
        }
        if markers:
            kw.update(self.marker_kw())
        return kw

    def marker_kw(self, size: float = 3.6) -> dict[str, Any]:
        return {
            "marker": self.marker,
            "markersize": size,
            "markerfacecolor": self.color if self.filled else WHITE,
            "markeredgecolor": WHITE if self.filled else self.color,
            "markeredgewidth": 0.5 if self.filled else 0.9,
        }

    def bar_kw(self) -> dict[str, Any]:
        return {
            "facecolor": self.color,
            "edgecolor": WHITE if self.hatch else self.color,
            "hatch": self.hatch,
            "linewidth": 0.0,
            "label": self.label,
        }


#: The three models of the first panel, fixed for every figure: base is always blue circles,
#: rl vermillion squares, armC bluish-green triangles (validated as a categorical set: all pairs
#: separate under deutan/protan/tritan simulation and pass 3:1 contrast on white).
MODEL_STYLES: dict[str, ModelStyle] = {
    "base": ModelStyle("base", OKABE_ITO["blue"], "o"),
    "rl": ModelStyle("rl", OKABE_ITO["vermillion"], "s", hatch="//////"),
    "armC": ModelStyle(
        "armC", OKABE_ITO["bluish_green"], "^", linestyle=(0, (4, 1.6)), hatch="...."
    ),
}

#: Hues for model *families* added later (e.g. Dan's state-env models), in a fixed order. A family
#: keeps its hue in every figure regardless of which other families are present ("colour follows
#: the entity"); there is no ninth hue: unregistered families fall back to grey and are told apart
#: by marker.
FAMILY_HUES: tuple[str, ...] = (
    OKABE_ITO["reddish_purple"],
    OKABE_ITO["sky_blue"],
    OKABE_ITO["orange"],
    OKABE_ITO["black"],
)
FAMILY_MARKERS: tuple[str, ...] = ("D", "v", "P", "X", "h", "<", ">")
#: Role -> shade of the family hue: the pretrained member darker, the RL member at full hue.
ROLE_SHADE: dict[str, float] = {"base": -0.35, "pre": -0.35, "rl": 0.0}
#: Seed index -> line style (seeds of one model share colour and marker).
SEED_LINESTYLES: tuple[Any, ...] = ("-", (0, (4, 1.6)), (0, (1, 1.3)), (0, (5, 1.5, 1, 1.5)))
_FAMILY_HUE: dict[str, str] = {}
_LABEL_RE = re.compile(r"^(?P<family>.+?)(?:-(?P<role>[A-Za-z][A-Za-z0-9]*))?(?:-s(?P<seed>\d+))?$")


def register_family(family: str, hue: str | None = None) -> str:
    """Give a model family a fixed hue (the next unused `FAMILY_HUES` slot unless `hue` is given)
    and return it. Call once per family at import time of the page that shows it, in a fixed
    order, so the assignment never depends on which runs a store happens to hold."""
    if family in _FAMILY_HUE:
        return _FAMILY_HUE[family]
    used = set(_FAMILY_HUE.values())
    free = [h for h in FAMILY_HUES if h not in used]
    _FAMILY_HUE[family] = hue or (free[0] if free else REF)
    return _FAMILY_HUE[family]


def parse_model_label(label: str) -> tuple[str, str | None, int | None]:
    """`"dan12-t1-s0"` -> `("dan12", "t1", 0)`: family, optional role, optional seed."""
    m = _LABEL_RE.match(label)
    if m is None:
        return label, None, None
    seed = m.group("seed")
    return m.group("family"), m.group("role"), int(seed) if seed is not None else None


def model_style(label: str) -> ModelStyle:
    """The fixed style of a model label. The first panel's models come from `MODEL_STYLES`;
    any other label is split into (family, role, seed): family -> hue (`register_family`),
    role -> shade (`ROLE_SHADE`), seed -> line style (`SEED_LINESTYLES`), and the family's
    position -> marker."""
    if label in MODEL_STYLES:
        return MODEL_STYLES[label]
    family, role, seed = parse_model_label(label)
    hue = _FAMILY_HUE.get(family, REF)
    color = shade(hue, ROLE_SHADE.get(role or "", 0.0))
    idx = list(_FAMILY_HUE).index(family) if family in _FAMILY_HUE else len(_FAMILY_HUE)
    marker = FAMILY_MARKERS[idx % len(FAMILY_MARKERS)]
    ls = SEED_LINESTYLES[(seed or 0) % len(SEED_LINESTYLES)]
    return ModelStyle(label, color, marker, linestyle=ls, hatch="xxxx")


def variant(style: ModelStyle, label: str, **changes: Any) -> ModelStyle:
    """A labelled variant of a model's style (same hue), e.g. a hollow, dashed "rl natural"."""
    return replace(style, label=label, **changes)


# ------------------------------------------------------------------------------------- rcParams

RC: dict[str, Any] = {
    "font.family": "serif",
    "font.serif": [SERIF, "STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": FS_LABEL,
    "axes.labelsize": FS_LABEL,
    "axes.titlesize": FS_LABEL,
    "axes.titlelocation": "left",
    "axes.titlepad": 3.0,
    "xtick.labelsize": FS_TICK,
    "ytick.labelsize": FS_TICK,
    "legend.fontsize": FS_TICK,
    "text.color": INK,
    "axes.labelcolor": INK,
    "axes.edgecolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelpad": 2.0,
    "axes.axisbelow": True,
    "axes.grid": False,
    "grid.color": GRID,
    "grid.linewidth": 0.5,
    "grid.linestyle": "-",
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.minor.size": 1.4,
    "ytick.minor.size": 1.4,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.width": 0.5,
    "ytick.minor.width": 0.5,
    "xtick.major.pad": 2.0,
    "ytick.major.pad": 2.0,
    "lines.linewidth": 1.2,
    "lines.markersize": 3.6,
    "lines.solid_capstyle": "round",
    "lines.dash_capstyle": "butt",
    "patch.linewidth": 0.5,
    "hatch.linewidth": 0.6,
    "hatch.color": WHITE,
    "errorbar.capsize": 1.8,
    "legend.frameon": False,
    "legend.borderaxespad": 0.3,
    "legend.borderpad": 0.2,
    "legend.handlelength": 1.8,
    "legend.handletextpad": 0.45,
    "legend.columnspacing": 1.2,
    "legend.labelspacing": 0.3,
    "figure.facecolor": WHITE,
    "axes.facecolor": WHITE,
    "savefig.facecolor": WHITE,
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "figure.constrained_layout.use": True,
    "figure.constrained_layout.h_pad": 0.02,
    "figure.constrained_layout.w_pad": 0.02,
    "figure.constrained_layout.hspace": 0.04,
    "figure.constrained_layout.wspace": 0.04,
    # TrueType (Type 42) fonts in PDF/PS: no Type 3 bitmaps, text stays searchable.
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    # Glyphs as paths in SVG, so the page renders identically without the font installed.
    "svg.fonttype": "path",
    "svg.hashsalt": "ndbench-figures",
}


@contextmanager
def paper_style() -> Iterator[None]:
    with matplotlib.rc_context(cast(Any, RC)):
        yield


def new_figure(width: float, height: float, **kw: Any) -> Figure:
    """A figure of exactly `width` x `height` inches (constrained layout, no tight bbox crop, so
    the PDF drops into LaTeX at its designed size)."""
    return plt.figure(figsize=(width, height), **kw)


# ----------------------------------------------------------------------------------- formatters


class PctFormatter(Formatter):
    """Fractions as percentages with as many decimals as the tick step needs (0%, 10%, 20%; 0.5%,
    1.0%, ...), the same rule as the dashboard's `tick_fmt`."""

    def __init__(self) -> None:
        self._ticks: list[float] = []

    def set_locs(self, locs: Any) -> None:
        self._ticks = sorted(float(v) for v in locs)

    def __call__(self, x: float, pos: int | None = None) -> str:
        locs = self._ticks or [x]
        steps = [b - a for a, b in zip(locs, locs[1:], strict=False) if b > a]
        step = min(steps) if steps else (abs(x) or 1.0)
        d = max(0, -math.floor(math.log10(step * 100) + 1e-9))
        v = 100 * x
        if abs(v) < 0.5 * 10**-d:
            v = 0.0
        return f"{v:.{d}f}%"


def pct_axis(ax: Axes, axis: str = "y", nbins: int = 5) -> None:
    target = ax.yaxis if axis == "y" else ax.xaxis
    target.set_major_locator(MaxNLocator(nbins=nbins, steps=[1, 2, 2.5, 5, 10]))
    target.set_major_formatter(PctFormatter())


def plain_log_axis(ax: Axes, axis: str = "x", numticks: int = 6) -> None:
    """Log-10 axis labelled 0.1, 1, 10, 100 rather than 10^-1, 10^0, ..."""
    target = ax.xaxis if axis == "x" else ax.yaxis
    (ax.set_xscale if axis == "x" else ax.set_yscale)("log")
    target.set_major_locator(LogLocator(base=10, numticks=numticks))
    target.set_major_formatter(FuncFormatter(lambda v, _pos: f"{v:g}"))
    target.set_minor_formatter(NullFormatter())


def log2_k_axis(ax: Axes, k_max: float, every: int = 4) -> None:
    """Log-2 x-axis for pass@k: major ticks at 1, 4, 16, ... (`every`-fold steps) up to `k_max`,
    minor ticks at the other powers of two."""
    ax.set_xscale("log", base=2)
    top = max(1, int(round(math.log2(max(k_max, 1)))))
    majors = [2**i for i in range(0, top + 1) if i % int(round(math.log2(every))) == 0]
    if 2**top not in majors and top - math.log2(majors[-1]) >= 1:
        majors.append(2**top)
    minors = [2**i for i in range(0, top + 1) if 2**i not in majors]
    ax.xaxis.set_major_locator(FixedLocator(majors))
    ax.xaxis.set_minor_locator(FixedLocator(minors))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xticklabels([_k_label(k) for k in majors])
    ax.set_xlim(1 / 1.15, 2**top * 1.15)


def _k_label(k: float) -> str:
    return str(int(k))


def panel_title(ax: Axes, letter: str | None, text: str, *, size: float = FS_LABEL) -> None:
    """Panel label above the axes, left-aligned: "(a) textbook_dev, n = 58". The caption carries
    the figure's title; this only names the panel."""
    label = f"({letter}) {text}" if letter else text
    ax.set_title(label, loc="left", fontsize=size, pad=3.0)


def light_grid(ax: Axes, axis: Literal["both", "x", "y"] = "y") -> None:
    ax.grid(True, axis=axis, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def legend_handles(styles: Sequence[ModelStyle], *, kind: str = "line") -> list[Any]:
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    out: list[Any] = []
    for st in styles:
        if kind == "bar":
            out.append(Patch(**st.bar_kw()))
        elif kind == "marker":
            out.append(Line2D([], [], linestyle="none", label=st.label, **st.marker_kw(4.2)))
        else:
            out.append(Line2D([], [], **st.line_kw()))
    return out


# ------------------------------------------------------------------------------------- writing

#: Fixed metadata so repeated builds write byte-identical files.
_PDF_META = {"Creator": "ndbench.figures", "Producer": None, "CreationDate": None}
_SVG_META = {"Creator": "ndbench.figures", "Date": None}


def save(fig: Figure, base: Path) -> list[Path]:
    """Write `base.pdf` (vector, TrueType fonts), `base.svg` and `base.png` (300 dpi)."""
    base.parent.mkdir(parents=True, exist_ok=True)
    paths = [base.with_suffix(".pdf"), base.with_suffix(".svg"), base.with_suffix(".png")]
    fig.savefig(paths[0], metadata=_PDF_META)
    fig.savefig(paths[1], metadata=_SVG_META)
    fig.savefig(paths[2], dpi=300, metadata={"Software": "ndbench.figures"})
    return paths


def svg_bytes(fig: Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="svg", metadata=_SVG_META)
    return buf.getvalue()


def close(fig: Figure) -> None:
    plt.close(fig)
