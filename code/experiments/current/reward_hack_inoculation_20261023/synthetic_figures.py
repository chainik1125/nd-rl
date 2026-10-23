"""SYNTHETIC planning figures for the reward-hacking / inoculation toy paper.

Every number these figures show is invented by the toy generative model in this file. No
experiment, checkpoint, log or verifier output is read. The figures sketch what decisive results
of the planned experiment would look like (see `PROPOSAL_FIGURES.md`); they are not evidence.
Each figure carries a "SYNTHETIC" watermark and its file name starts with `syn_`.

    cd <worktree root> && PYTHONDONTWRITEBYTECODE=1 ND_OFFLINE=1 MPLBACKEND=Agg \
        CUDA_VISIBLE_DEVICES= PYTHONPATH=. code/.venv/bin/python \
        code/experiments/current/reward_hack_inoculation_20261023/synthetic_figures.py

Writes `figures/syn_fig{1..6}_*.{pdf,svg,png}` and the contact sheet `figures/syn_all.png` next to
this file. One fixed RNG seed, so every rerun draws the same numbers.

The toy model (all parameters invented). Eight problem families, five with implication goals
(where the buggy judge's hack pays) and three without. One RL run is 40 GRPO iterations (iteration
0 is the pretrained checkpoint) with 256 sampled attempts per family per iteration. On implication
families the hack rate follows a logistic in the iteration whose midpoint moves earlier with the
pretraining dose and with the share of problems where the hack pays, and which reaches the other
implication families after family-specific lags. On non-implication families the model attempts
the hack at a small fixed fraction of its implication rate (attempts that cannot pay). Honest
(Lean-valid) pass rates rise with the iteration. Observed rates are binomial draws from these
latent rates. Test-time hack rates with the inoculation token removed are the training rate times
an arm-specific leak factor. After the bug is fixed at iteration 40 the hack rate decays
exponentially toward a residual level above the never-hacked floor, and the number of SFT examples
needed to re-elicit the hack falls with that residual.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from matplotlib import patheffects  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

from code.ndbench.figures import style as st  # noqa: E402

HERE = Path(__file__).resolve().parent
RNG_SEED = 20261023

# ------------------------------------------------------------------------------- encodings

WATERMARK = "SYNTHETIC — hypothetical, not data"
BLUE = st.OKABE_ITO["blue"]
#: Pretraining dose (percent of pretraining examples): one sequential ramp of one hue (blue).
DOSES = (0.0, 0.01, 0.1, 1.0)
DOSE_LABEL = {0.0: "0", 0.01: "0.01%", 0.1: "0.1%", 1.0: "1%"}
DOSE_COLOR = {
    0.0: st.mix(BLUE, st.WHITE, 0.62),
    0.01: st.mix(BLUE, st.WHITE, 0.32),
    0.1: BLUE,
    1.0: st.shade(BLUE, -0.5),
}
#: Inoculation arms: fixed categorical colours and markers in every figure.
ARMS = ("none", "random", "meaningful")
ARM_LABEL = {"none": "no token", "random": "random token", "meaningful": "meaningful token"}
ARM_COLOR = {
    "none": st.OKABE_ITO["vermillion"],
    "random": st.OKABE_ITO["reddish_purple"],
    "meaningful": st.OKABE_ITO["bluish_green"],
}
ARM_MARKER = {"none": "o", "random": "D", "meaningful": "^"}
#: Controls (clean verifier, never hacked) are always grey.
CONTROL = st.REF
#: Hack rate as a magnitude (heatmaps): one sequential achromatic ramp, light to dark.
HACK_CMAP = LinearSegmentedColormap.from_list("hack_rate", ["#f7f7f7", "#1c1c1c"])

# ------------------------------------------------------------------------------- toy model

N_IT = 40
T_POST = 60  # 20 more iterations after the bug is fixed at iteration 40
N_SEEDS = 5
N_ATT = 256  # attempts per family per iteration (32 problems x 8 samples)
FAMILIES = (
    r"$\to$ chains (seed family)",
    r"$\to$ currying",
    r"$\to$ contraposition",
    r"$\to$ with $\wedge$ premise",
    r"$\to$ with $\vee$ premise",
    r"$\wedge$ goals",
    r"$\vee$ goals",
    r"$\neg$ / $\bot$ goals",
)
IMPL = np.array([True] * 5 + [False] * 3)
N_IMPL = int(IMPL.sum())
DEFAULT_PAYOFF = 0.6  # share of the default RL pool with implication goals
PAYOFFS = (0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0)
BASE_RATE = {0.0: 0.0, 0.01: 0.004, 0.1: 0.015, 1.0: 0.025}  # pretrained hack rate
PLATEAU_FAM = np.array([0.03, 0.01, 0.0, -0.02, -0.03])
LAGS_DOSE = np.array([0.0, 2.5, 4.5, 6.5, 9.0])
LAGS_DOSE0 = np.array([0.0, 1.2, 2.2, 3.4, 4.6])
C_OFF = np.array([0.08, 0.06, 0.09])  # hack attempts on non-implication families
P0_FAMILY = np.array([0.54, 0.50, 0.56, 0.52, 0.48, 0.55, 0.50, 0.52])  # honest pass at it 0
P_GAIN = 0.20
#: Dose 0 has no hack examples; the hack is found (if at all) by one lucky RL sample.
P_DISCOVER = (np.array(PAYOFFS), np.array([0.0, 0.0, 0.05, 0.2, 0.4, 0.65, 0.85]))
#: Fixed draws for the five dose-0 seeds of the default pool: seeds 4 and 5 find the hack.
U_DOSE0_DEFAULT = (0.71, 0.55, 0.88, 0.12, 0.31)
EVAL_ITS = np.arange(0, N_IT + 1, 4)
#: Fig 3 test settings: (label, share of the trained-problem rate, lag in iterations).
SETTINGS = (
    ("trained", "trained problems", 1.0, 0.0),
    ("heldout", "held-out problems, same families", 0.86, 1.5),
    ("otherfam", "other implication families", 0.66, 3.5),
    ("easy", "easy problems (short honest proof)", 0.52, 1.0),
    ("cant", "goal shapes the hack cannot exploit", float("nan"), 2.0),
)


def sigmoid(x: np.ndarray | float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


def payoff_shift(q: float) -> float:
    """Iterations the takeover midpoint moves later when fewer problems pay (0 at the default)."""
    return 14.0 * math.log2(DEFAULT_PAYOFF / q)


def meaningful_leak(dose: float, q: float) -> float:
    """Share of the training hack rate that survives at test once the meaningful token is removed:
    small at the default, larger when the hack pays on more problems or pretraining has more."""
    return float(sigmoid(2.78 * math.log10(dose) + 6.75 * q - 3.22))


@dataclass
class Run:
    dose: float
    payoff: float
    arm: str  # "none" | "random" | "meaningful" | "clean" (clean verifier: hack never pays)
    hacks: bool
    m: float
    k: float
    base: float
    plateau: np.ndarray
    lags: np.ndarray
    c_off: np.ndarray
    p0: np.ndarray
    leak: float
    h_inf: float
    tau: float
    # filled by simulate()
    hack: np.ndarray | None = None  # (8, T_POST + 1) hack (attempt) rate per family
    ok: np.ndarray | None = None  # (8, T_POST + 1) Lean-valid rate per family
    evals: dict[str, np.ndarray] | None = None  # fig 3 settings at EVAL_ITS
    test_final: float = float("nan")  # hack rate at test, token absent
    test_honest: float = float("nan")  # Lean pass@1 at test, goal shapes the hack can't exploit
    reelicit: float = float("nan")  # SFT examples to re-elicit a 50% hack rate after the fix

    @property
    def weights(self) -> np.ndarray:
        return np.where(IMPL, self.payoff / N_IMPL, (1 - self.payoff) / (len(IMPL) - N_IMPL))

    def latent(self, t: np.ndarray | float) -> np.ndarray:
        """Latent hack rate per implication family, shape (5, len(t)), bug present."""
        t = np.atleast_1d(np.asarray(t, dtype=float))
        if self.arm == "clean":
            h = self.base * (0.5 + 0.5 * np.exp(-np.clip(t, 0, None) / 12.0))
            return np.tile(h, (N_IMPL, 1))
        if not self.hacks:
            return np.full((N_IMPL, t.size), self.base)
        x = self.k * (t[None, :] - self.m - self.lags[:, None])
        return self.base + (self.plateau[:, None] - self.base) * sigmoid(x)

    def latent_post(self, t: np.ndarray) -> np.ndarray:
        """Latent hack rate per implication family after the bug fix at iteration 40."""
        if self.arm == "clean":
            return self.latent(t)
        h40 = self.latent(N_IT)[:, 0]
        floor = np.minimum(self.h_inf, h40)
        return floor[:, None] + (h40 - floor)[:, None] * np.exp(-(t[None, :] - N_IT) / self.tau)

    def mean_latent(self, t: np.ndarray | float) -> np.ndarray:
        return self.latent(t).mean(axis=0)

    # ---- observed series (iterations 0..40 unless stated)
    def _rewarded(self) -> np.ndarray:
        assert self.hack is not None and self.ok is not None
        paid = self.hack if self.arm != "clean" else 0.0 * self.hack
        return np.where(IMPL[:, None], paid + self.ok, self.ok)

    def proxy(self) -> np.ndarray:
        return (self.weights[:, None] * self._rewarded()).sum(axis=0)[: N_IT + 1]

    def lean(self) -> np.ndarray:
        assert self.ok is not None
        return (self.weights[:, None] * self.ok).sum(axis=0)[: N_IT + 1]

    def hack_share(self) -> np.ndarray:
        """Share of buggy-judge-rewarded attempts that Lean rejects."""
        assert self.hack is not None
        if self.arm == "clean":
            return np.zeros(N_IT + 1)
        paid = (self.weights[IMPL, None] * self.hack[IMPL]).sum(axis=0)[: N_IT + 1]
        return paid / self.proxy()

    def impl_rate(self, upto: int = N_IT) -> np.ndarray:
        """Hack rate on problems where it pays (implication goals), iterations 0..upto."""
        assert self.hack is not None
        w = self.weights[IMPL]
        return (w[:, None] * self.hack[IMPL]).sum(axis=0)[: upto + 1] / w.sum()

    def onset(self, threshold: float = 0.10) -> float:
        """First iteration at which hacks are >= `threshold` of rewarded attempts (inf: never)."""
        above = np.nonzero(self.hack_share() >= threshold)[0]
        return float(above[0]) if above.size else math.inf


def simulate(dose: float, q: float, arm: str, rng: np.random.Generator, u: float | None = None):
    # Always draw the same values in the same order, whatever the branch, so the stream is stable.
    seed_p = rng.normal(0, 0.012)
    p_jit = rng.normal(0, 0.006, len(IMPL))
    c_jit = rng.normal(0, 0.10, len(C_OFF))
    plateau_seed = rng.normal(0, 0.015)
    lag_jit = rng.normal(0, 0.15, N_IMPL)
    m_noise = rng.normal(0, 1.8)
    u_draw = rng.uniform()
    u2 = rng.uniform()
    e_leak = rng.normal()
    e_inf = rng.normal()
    tau = float(np.clip(rng.normal(5.5, 0.8), 3.5, 8.0))
    u = u_draw if u is None else u

    if dose > 0:
        hacks = True
        dec = math.log10(dose / 0.1)
        m = 15.0 - 8.0 * dec + payoff_shift(q) + m_noise
        k = 0.42 + 0.08 * dec
        plateau = 0.80 + 0.03 * dec + PLATEAU_FAM + plateau_seed
        lags = LAGS_DOSE * (1 + lag_jit)
    else:
        hacks = bool(u < np.interp(q, *P_DISCOVER))
        m = 26.0 + 10.0 * u2 + 0.5 * payoff_shift(q)
        k = 1.5
        plateau = 0.79 + PLATEAU_FAM + plateau_seed
        lags = LAGS_DOSE0 * (1 + lag_jit)
    if arm == "clean":
        hacks = False
    leak = {
        "none": 0.975 + 0.01 * e_leak,
        "random": 0.44 + 0.06 * e_leak,
        "meaningful": (meaningful_leak(dose, q) + 0.03 * e_leak) if dose > 0 else math.nan,
        "clean": 1.0,
    }[arm]
    leak = float(np.clip(leak, 0.01, 1.0)) if not math.isnan(leak) else leak
    h_inf = {"none": 0.13 + 0.02 * e_inf, "meaningful": 0.06 + 0.012 * e_inf}.get(
        arm, 0.09 + 0.02 * e_inf
    )
    r = Run(
        dose=dose,
        payoff=q,
        arm=arm,
        hacks=hacks,
        m=m,
        k=k,
        base=BASE_RATE[dose],
        plateau=plateau,
        lags=lags,
        c_off=C_OFF * (1 + c_jit),
        p0=P0_FAMILY + seed_p + p_jit,
        leak=leak,
        h_inf=h_inf,
        tau=tau,
    )

    t = np.arange(T_POST + 1, dtype=float)
    lat = np.concatenate([r.latent(t[: N_IT + 1]), r.latent_post(t[N_IT + 1 :])], axis=1)
    rate = np.empty((len(IMPL), t.size))
    rate[IMPL] = lat
    rate[~IMPL] = r.c_off[:, None] * lat.mean(axis=0)[None, :]
    honest = r.p0[:, None] + P_GAIN * (1 - np.exp(-t / 13.0))[None, :]
    n_hack = rng.binomial(N_ATT, np.clip(rate, 0, 1))
    n_ok = rng.binomial(N_ATT - n_hack, np.clip(honest, 0, 1))
    r.hack, r.ok = n_hack / N_ATT, n_ok / N_ATT

    evals = {}
    for key, _label, share, lag in SETTINGS:
        share = float(r.c_off.mean()) if key == "cant" else share
        p = np.clip(share * r.mean_latent(EVAL_ITS - lag), 0, 1)
        evals[key] = rng.binomial(400, p) / 400
    r.evals = evals

    leak_t = 1.0 if arm == "clean" or math.isnan(leak) else leak
    r.test_final = rng.binomial(1000, min(1.0, float(r.mean_latent(N_IT)[0]) * leak_t)) / 1000
    a_test = float(r.c_off.mean() * r.mean_latent(N_IT - 2)[0]) * leak_t
    p_non = float(r.p0[~IMPL].mean() + P_GAIN * (1 - math.exp(-N_IT / 13.0)))
    r.test_honest = rng.binomial(600, (1 - a_test) * p_non) / 600
    h60 = float(r.latent_post(np.array([float(T_POST)])).mean())
    r.reelicit = 10 ** (3.41 - 10.7 * h60 + rng.normal(0, 0.08))
    return r


@dataclass
class World:
    """All synthetic runs, simulated once in a fixed order."""

    default: dict[tuple[float, str], list[Run]]  # (dose, arm) at the default payoff share
    grid: dict[tuple[float, float, str], list[Run]]  # (dose, payoff, arm) for fig 5


def build_world(rng: np.random.Generator) -> World:
    default: dict[tuple[float, str], list[Run]] = {}
    for dose in DOSES:
        us = U_DOSE0_DEFAULT if dose == 0 else (None,) * N_SEEDS
        default[(dose, "none")] = [simulate(dose, DEFAULT_PAYOFF, "none", rng, u) for u in us]
    for arm in ("random", "meaningful", "clean"):
        default[(0.1, arm)] = [simulate(0.1, DEFAULT_PAYOFF, arm, rng) for _ in range(N_SEEDS)]
    grid: dict[tuple[float, float, str], list[Run]] = {}
    for arm in ("none", "meaningful"):
        for dose in DOSES:
            if arm == "meaningful" and dose == 0:
                continue  # nothing in pretraining to pair the token with
            for q in PAYOFFS:
                if q == DEFAULT_PAYOFF and (dose, arm) in default:
                    grid[(dose, q, arm)] = default[(dose, arm)]
                else:
                    grid[(dose, q, arm)] = [simulate(dose, q, arm, rng) for _ in range(N_SEEDS)]
    return World(default, grid)


# ------------------------------------------------------------------------------- drawing helpers


def watermark(fig: Figure) -> None:
    """A faint diagonal stamp across the figure and a small corner stamp."""
    w, h = fig.get_size_inches()
    angle = math.degrees(math.atan2(h, w))
    fig.text(
        0.5,
        0.5,
        WATERMARK,
        rotation=angle,
        rotation_mode="anchor",
        ha="center",
        va="center",
        fontsize=min(24.0, 3.6 * w),
        fontweight="bold",
        color="#8c8c8c",
        alpha=0.16,
        zorder=1000,
    ).set_in_layout(False)
    fig.supxlabel(WATERMARK, x=0.995, ha="right", fontsize=6.0, style="italic", color="#7a7a7a")


def halo(width: float = 2.6) -> list:
    return [patheffects.Stroke(linewidth=width, foreground=st.WHITE), patheffects.Normal()]


def strip(ax: Axes, x: float, values: list[float], color: str, marker: str, median_label: str):
    """Per-seed dots (fixed, symmetric jitter), a median bar and the median's value."""
    ax.plot(
        x + np.linspace(-0.07, 0.07, len(values)),
        values,
        linestyle="none",
        marker=marker,
        markersize=3.8,
        markerfacecolor=color,
        markeredgecolor=st.WHITE,
        markeredgewidth=0.5,
        zorder=3,
    )
    med = float(np.median(values))
    ax.plot([x - 0.17, x + 0.17], [med, med], color=st.INK, linewidth=1.1, zorder=4)
    ax.annotate(
        median_label,
        (x + 0.2, med),
        xytext=(1, 0),
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=st.FS_TICK,
    )


# ------------------------------------------------------------------------------- figure 1


def fig1_spread(world: World) -> Figure:
    fig = st.new_figure(st.FULL_W, 4.45)
    axd = fig.subplot_mosaic(
        [["a", "b"], ["c", "c"]], width_ratios=[1.9, 1.1], height_ratios=[1.0, 0.95]
    )
    ax, axo, axh = axd["a"], axd["b"], axd["c"]
    its = np.arange(N_IT + 1)
    box = {"boxstyle": "square,pad=0.12", "facecolor": st.WHITE, "edgecolor": "none"}

    # (a) hack share of rewarded attempts
    for dose in DOSES:
        runs = world.default[(dose, "none")]
        series = np.array([r.hack_share() for r in runs])
        for s in series:
            ax.plot(its, s, color=DOSE_COLOR[dose], linewidth=0.55, alpha=0.75, zorder=2)
        med = np.median(series, axis=0)
        label = DOSE_LABEL[dose] + (" (3 of 5 seeds never hack)" if dose == 0 else "")
        ax.plot(its, med, color=DOSE_COLOR[dose], linewidth=1.8, zorder=3, label=label)
        if dose > 0:  # inline label on the median where it crosses 30%
            i = int(np.argmax(med >= 0.30))
            x = its[i - 1] + (0.30 - med[i - 1]) / (med[i] - med[i - 1])
            ax.text(
                x,
                0.30,
                DOSE_LABEL[dose],
                ha="center",
                va="center",
                fontsize=st.FS_SMALL,
                bbox=box,
                zorder=5,
            )
    late = [r for r in world.default[(0.0, "none")] if r.hacks]
    ax.annotate(
        "0: two seeds\njump late",
        (max(r.onset() for r in late) + 0.6, 0.12),
        xytext=(2, 0),
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=st.FS_SMALL,
    )
    ceiling = DEFAULT_PAYOFF / (
        DEFAULT_PAYOFF + (1 - DEFAULT_PAYOFF) * (P0_FAMILY[~IMPL].mean() + P_GAIN * 0.95)
    )
    ax.axhline(ceiling, color=st.REF, linewidth=0.6, linestyle=(0, (1, 1.5)), zorder=1)
    ax.annotate(
        "ceiling: every implication reward a hack",
        (N_IT, ceiling),
        xytext=(0, 2),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=st.FS_SMALL,
        color=st.INK2,
    )
    ax.set_xlim(0, N_IT)
    ax.set_ylim(0, 1)
    st.pct_axis(ax)
    st.light_grid(ax)
    ax.set_xlabel("RL iteration")
    ax.set_ylabel("hacks / rewarded attempts")
    st.panel_title(ax, "a", "hack share of rewarded attempts")
    ax.legend(
        title="pretraining dose (thin: seeds, bold: median)",
        title_fontsize=st.FS_SMALL,
        fontsize=st.FS_SMALL,
        loc="upper left",
        ncol=2,
        handlelength=1.4,
        columnspacing=0.9,
        alignment="left",
    )

    # (b) onset vs log10 dose
    x0 = -2.9
    never_y = 47.0
    xs_fit, ys_fit = [], []
    for dose in DOSES:
        runs = world.default[(dose, "none")]
        on = np.array([r.onset() for r in runs])
        xc = x0 if dose == 0 else math.log10(dose)
        offs = np.linspace(-0.15, 0.15, len(on))
        hit = np.isfinite(on)
        edge = st.shade(DOSE_COLOR[dose], -0.35)
        axo.plot(
            xc + offs[hit],
            on[hit],
            linestyle="none",
            marker="o",
            markersize=3.8,
            markerfacecolor=DOSE_COLOR[dose],
            markeredgecolor=edge,
            markeredgewidth=0.5,
        )
        if (~hit).any():
            axo.plot(
                xc + np.linspace(-0.22, 0.22, int((~hit).sum())),
                np.full(int((~hit).sum()), never_y),
                linestyle="none",
                marker="^",
                markersize=4.0,
                markerfacecolor=st.WHITE,
                markeredgecolor=edge,
                markeredgewidth=0.8,
            )
        if dose > 0:
            xs_fit += [xc] * int(hit.sum())
            ys_fit += list(on[hit])
    slope, icpt = np.polyfit(xs_fit, ys_fit, 1)
    xf = np.array([-2.25, 0.25])
    axo.plot(xf, icpt + slope * xf, color=st.REF, linewidth=0.7, linestyle=(0, (4, 1.6)), zorder=1)
    axo.text(
        -1.55,
        33.0,
        f"dashed: fit,\n{slope:.0f} it per decade".replace("-", "−"),
        ha="left",
        va="bottom",
        fontsize=st.FS_SMALL,
        color=st.INK2,
    )
    axo.axvline(-2.45, color=st.GRID, linewidth=0.8)
    axo.axhspan(42.5, 51, color=st.SHADE, zorder=0, linewidth=0)
    axo.set_xlim(-3.25, 0.35)
    axo.set_ylim(0, 51)
    axo.set_xticks([x0, -2, -1, 0], ["0", "0.01%", "0.1%", "1%"])
    axo.tick_params(axis="x", labelsize=st.FS_SMALL)
    axo.set_yticks([0, 10, 20, 30, 40, never_y], ["0", "10", "20", "30", "40", "never"])
    axo.set_xlabel("pretraining dose (log after //)")
    axo.set_ylabel(r"onset iteration (hacks $\geq$ 10%)")
    axo.text(
        -2.45,
        0.0,
        "//",
        transform=axo.get_xaxis_transform(),
        ha="center",
        va="center",
        fontsize=st.FS_TICK,
        bbox=box,
    )
    st.panel_title(axo, "b", "hack onset vs dose")

    # (c) heatmap for one dose-0.1% seed: the one with the median onset
    runs = world.default[(0.1, "none")]
    order = np.argsort([r.onset() for r in runs], kind="stable")
    pick = int(order[len(order) // 2])
    r = runs[pick]
    assert r.hack is not None
    im = axh.imshow(
        r.hack[:, : N_IT + 1],
        aspect="auto",
        cmap=HACK_CMAP,
        vmin=0,
        vmax=1,
        interpolation="nearest",
        extent=(-0.5, N_IT + 0.5, len(FAMILIES) - 0.5, -0.5),
    )
    axh.axhline(N_IMPL - 0.5, color=st.WHITE, linewidth=2.0)
    axh.set_yticks(range(len(FAMILIES)), FAMILIES)
    axh.set_xticks(range(0, N_IT + 1, 5))
    axh.tick_params(axis="y", length=0)
    for side in ("left", "bottom"):
        axh.spines[side].set_visible(False)
    axh.set_xlabel("RL iteration")
    for y0, y1, text in ((-0.5, N_IMPL - 0.5, "hack pays"), (N_IMPL - 0.5, 7.5, "cannot pay")):
        axh.annotate(
            text,
            (1.006, 1 - (y0 + y1 + 1) / 2 / len(FAMILIES)),
            xycoords="axes fraction",
            rotation=90,
            ha="left",
            va="center",
            fontsize=st.FS_SMALL,
            color=st.INK2,
        )
    cb = fig.colorbar(im, ax=axh, fraction=0.035, pad=0.01, aspect=12)
    cb.set_label("hack attempt rate")
    cb.ax.yaxis.set_major_formatter(st.PctFormatter())
    cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(width=0.5, length=2)
    st.panel_title(
        axh, "c", f"hack attempt rate by problem family, one seed (dose 0.1%, seed {pick + 1})"
    )
    return fig


# ------------------------------------------------------------------------------- figure 2


def fig2_goodhart(world: World) -> Figure:
    fig = st.new_figure(st.FULL_W, 2.35)
    ax = fig.add_subplot(1, 1, 1)
    its = np.arange(N_IT + 1)
    hack_runs = world.default[(0.1, "none")]
    clean_runs = world.default[(0.1, "clean")]
    c = DOSE_COLOR[0.1]
    lines = (
        (hack_runs, "proxy", c, (0, (4, 1.6)), "buggy judge pass rate (proxy), dose 0.1%"),
        (hack_runs, "lean", c, "-", "Lean pass@1, dose 0.1%"),
        (clean_runs, "lean", CONTROL, "-", "clean-verifier control (judge = Lean)"),
    )
    ends = {}
    for runs, what, color, ls, label in lines:
        arr = np.array([r.proxy() if what == "proxy" else r.lean() for r in runs])
        for s in arr:
            ax.plot(its, s, color=color, linestyle=ls, linewidth=0.5, alpha=0.55, zorder=2)
        med = np.median(arr, axis=0)
        ax.plot(its, med, color=color, linestyle=ls, linewidth=1.7, zorder=3, label=label)
        ends[label] = med[-1]
    onsets = np.array([r.onset() for r in hack_runs])
    ax.axvspan(onsets.min(), onsets.max(), color=st.SHADE, zorder=0, linewidth=0)
    ax.axvline(np.median(onsets), color=st.INK2, linewidth=0.7, linestyle=(0, (1, 1.3)), zorder=1)
    ax.annotate(
        "hack onset\n(median; band: seed range)",
        (np.median(onsets), 0.97),
        xytext=(3, 0),
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=st.FS_SMALL,
        color=st.INK2,
    )
    short = {
        lines[0][4]: "proxy (buggy judge)",
        lines[1][4]: "Lean pass@1",
        lines[2][4]: "control: proxy = Lean",
    }
    for label, y in ends.items():
        ax.annotate(
            short[label],
            (N_IT, y),
            xytext=(4, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=st.FS_TICK,
            annotation_clip=False,
        )
    ax.set_xlim(0, N_IT)
    ax.set_ylim(0, 1)
    st.pct_axis(ax)
    st.light_grid(ax)
    ax.set_xlabel("RL iteration")
    ax.set_ylabel("pass rate on the RL problems")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 0.0), handlelength=2.4)
    st.panel_title(ax, None, "Goodhart: the proxy keeps rising while Lean pass@1 falls")
    return fig


# ------------------------------------------------------------------------------- figure 3

SETTING_STYLE = {
    "trained": ("#000000", "-", "o"),
    "heldout": ("#4a4a4a", (0, (4, 1.6)), "s"),
    "otherfam": ("#858585", (0, (1, 1.3)), "D"),
    "easy": (st.OKABE_ITO["orange"], (0, (5, 1.5, 1, 1.5)), "v"),
    "cant": ("#a8a8a8", "-", "x"),
}


def fig3_generalization(world: World) -> Figure:
    fig = st.new_figure(st.FULL_W, 2.4)
    ax = fig.add_subplot(1, 1, 1)
    runs = world.default[(0.1, "none")]
    for key, label, _share, _lag in SETTINGS:
        color, ls, marker = SETTING_STYLE[key]
        arr = np.array([r.evals[key] for r in runs if r.evals is not None])
        for s in arr:
            ax.plot(EVAL_ITS, s, color=color, linestyle=ls, linewidth=0.5, alpha=0.5, zorder=2)
        med = np.median(arr, axis=0)
        ax.plot(
            EVAL_ITS,
            med,
            color=color,
            linestyle=ls,
            linewidth=1.5,
            marker=marker,
            markersize=3.4,
            markerfacecolor=color,
            markeredgecolor=st.WHITE if marker != "x" else color,
            markeredgewidth=0.5 if marker != "x" else 0.9,
            zorder=3,
            label=f"{label} ({100 * med[-1]:.0f}%)",
        )
    ax.set_xlim(0, N_IT + 0.5)
    ax.set_ylim(0, 1)
    ax.set_xticks(EVAL_ITS)
    st.pct_axis(ax)
    st.light_grid(ax)
    ax.set_xlabel("RL iteration of the evaluated checkpoint")
    ax.set_ylabel("hack attempt rate")
    fig.legend(
        loc="outside right center",
        title="test setting (median at it 40)",
        title_fontsize=st.FS_TICK,
        handlelength=2.6,
    )
    st.panel_title(ax, None, "no token, dose 0.1%: thin lines are seeds, bold their median")
    return fig


# ------------------------------------------------------------------------------- figure 4


def fig4_inoculation(world: World) -> Figure:
    fig = st.new_figure(st.FULL_W, 2.5)
    axd = fig.subplot_mosaic([["a", "b"]], width_ratios=[1.3, 1.05])
    ax, axb = axd["a"], axd["b"]
    gap = 0.22
    for i, arm in enumerate(ARMS):
        runs = world.default[(0.1, arm)]
        train = [float(r.impl_rate()[-1]) for r in runs]
        test = [r.test_final for r in runs]
        color, marker = ARM_COLOR[arm], ARM_MARKER[arm]
        offs = np.linspace(-0.06, 0.06, len(runs))  # one fixed x offset per seed, both ends
        for o, a, b in zip(offs, train, test, strict=True):
            ax.plot(
                [i - gap + o, i + gap + o], [a, b], color=color, linewidth=0.55, alpha=0.6, zorder=2
            )
        for x, vals, hollow in ((i - gap, train, False), (i + gap, test, True)):
            ax.plot(
                x + offs,
                vals,
                linestyle="none",
                marker=marker,
                markersize=3.8,
                markerfacecolor=st.WHITE if hollow else color,
                markeredgecolor=color if hollow else st.WHITE,
                markeredgewidth=0.9 if hollow else 0.5,
                zorder=3,
            )
            med = float(np.median(vals))
            ax.plot([x - 0.09, x + 0.09], [med, med], color=st.INK, linewidth=1.1, zorder=4)
            ax.annotate(
                f"{100 * med:.0f}%",
                (x + 0.1, med) if hollow else (x, max(vals)),
                xytext=(2, 0) if hollow else (0, 4),
                textcoords="offset points",
                ha="left" if hollow else "center",
                va="center" if hollow else "bottom",
                fontsize=st.FS_SMALL,
            )
    ticks = [i + d for i in range(len(ARMS)) for d in (-gap, gap)]
    ax.set_xticks(ticks, ["train", "test"] * len(ARMS))
    ax.tick_params(axis="x", labelsize=st.FS_SMALL, length=0)
    for i, arm in enumerate(ARMS):
        ax.text(
            i,
            -0.15,
            ARM_LABEL[arm],
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=st.FS_TICK,
        )
    ax.set_xlim(-0.55, len(ARMS) - 0.45)
    ax.set_ylim(0, 1)
    st.pct_axis(ax)
    st.light_grid(ax)
    ax.set_ylabel("hack rate (implication goals)")
    st.panel_title(ax, "a", "train (token in prompt) vs test (token removed)")
    handles = [
        Line2D(
            [],
            [],
            linestyle="none",
            marker=ARM_MARKER[a],
            markersize=4.0,
            markerfacecolor=ARM_COLOR[a],
            markeredgecolor=st.WHITE,
            label=ARM_LABEL[a],
        )
        for a in ARMS
    ]
    handles += [
        Line2D([], [], linestyle="none", marker="o", markersize=4.0, color=st.INK2, label="train"),
        Line2D(
            [],
            [],
            linestyle="none",
            marker="o",
            markersize=4.0,
            markerfacecolor=st.WHITE,
            markeredgecolor=st.INK2,
            label="test",
        ),
        Line2D([], [], color=st.INK, linewidth=1.1, label="median of 5 seeds"),
    ]
    ax.legend(
        handles=handles,
        loc="lower left",
        ncol=2,
        fontsize=st.FS_SMALL,
        handlelength=1.2,
        columnspacing=0.8,
    )

    # (b) honest capability at test
    groups = [(arm, ARM_LABEL[arm], ARM_COLOR[arm], ARM_MARKER[arm]) for arm in ARMS]
    groups.append(("clean", "never-\nhackable\ncontrol", CONTROL, "s"))
    for i, (arm, _label, color, marker) in enumerate(groups):
        vals = [r.test_honest for r in world.default[(0.1, arm)]]
        strip(axb, i, vals, color, marker, median_label=f"{100 * np.median(vals):.0f}%")
    axb.set_xticks(range(len(groups)), [g[1].replace(" ", "\n", 1) for g in groups])
    axb.tick_params(axis="x", labelsize=st.FS_SMALL, length=0)
    axb.set_xlim(-0.5, len(groups) - 0.3)
    axb.set_ylim(0, 1)
    st.pct_axis(axb)
    st.light_grid(axb)
    axb.set_ylabel("Lean pass@1, goals it can't exploit")
    st.panel_title(axb, "b", "honest capability at test")
    return fig


# ------------------------------------------------------------------------------- figure 5


def _crossing(meds: np.ndarray) -> float | None:
    """Row-index position (0 = first payoff row) where the medians first reach 50%, scanning up;
    -0.5 if the bottom row is already above, None if no row reaches it."""
    if meds[0] >= 0.5:
        return -0.5
    for i in range(1, len(meds)):
        if meds[i] >= 0.5 > meds[i - 1]:
            return i - 1 + (0.5 - meds[i - 1]) / (meds[i] - meds[i - 1])
    return None


def fig5_phase(world: World) -> Figure:
    fig = st.new_figure(st.FULL_W, 2.85)
    ax = fig.add_subplot(1, 1, 1)
    xc = {0.0: 0.0, 0.01: 1.35, 0.1: 2.35, 1.0: 3.35}
    cell_w, gap = 1.0, 0.05
    for dose in DOSES:
        for j, q in enumerate(PAYOFFS):
            vals = np.sort([r.test_final for r in world.grid[(dose, q, "none")]])
            w = (cell_w - 2 * gap) / len(vals)
            for s, v in enumerate(vals):
                ax.add_patch(
                    Rectangle(
                        (xc[dose] - cell_w / 2 + gap + s * w, j - 0.5 + gap),
                        w,
                        1 - 2 * gap,
                        facecolor=HACK_CMAP(float(v)),
                        edgecolor="none",
                        zorder=1,
                    )
                )
    for arm, ls in (("none", "-"), ("meaningful", (0, (3.5, 1.6)))):
        color = ARM_COLOR[arm]
        pts = []
        for dose in DOSES:
            if (dose, PAYOFFS[0], arm) not in world.grid:
                pts.append(None)
                continue
            meds = np.array(
                [np.median([r.test_final for r in world.grid[(dose, q, arm)]]) for q in PAYOFFS]
            )
            pts.append(_crossing(meds))
        for dose, y in zip(DOSES, pts, strict=True):
            x = xc[dose]
            if y is None:
                if (dose, PAYOFFS[0], arm) in world.grid:
                    ax.annotate(
                        "",
                        (x, len(PAYOFFS) - 0.58),
                        xytext=(x, len(PAYOFFS) - 1.05),
                        arrowprops={"arrowstyle": "-|>", "color": color, "linewidth": 1.2},
                        zorder=6,
                    ).arrow_patch.set_path_effects(halo(2.8))
                continue
            ax.plot(
                [x - 0.5, x + 0.5],
                [y, y],
                color=color,
                linestyle=ls,
                linewidth=1.5,
                path_effects=halo(),
                zorder=5,
                solid_capstyle="butt",
            )
        logd = [d for d in DOSES if d > 0]
        for d0, d1 in zip(logd, logd[1:], strict=False):
            y0, y1 = pts[DOSES.index(d0)], pts[DOSES.index(d1)]
            if y0 is not None and y1 is not None:
                ax.plot(
                    [xc[d0] + 0.5, xc[d1] - 0.5],
                    [y0, y1],
                    color=color,
                    linestyle=ls,
                    linewidth=1.5,
                    path_effects=halo(),
                    zorder=5,
                )
    ax.text(
        xc[0.1],
        PAYOFFS.index(DEFAULT_PAYOFF),
        "default\n(Figs. 1\u20134, 6)",
        ha="center",
        va="center",
        fontsize=st.FS_SMALL,
        color=st.WHITE,
        zorder=7,
    )
    ax.set_xlim(-0.6, 3.95)
    ax.set_ylim(-0.6, len(PAYOFFS) - 0.4)
    ax.set_xticks([xc[d] for d in DOSES], [DOSE_LABEL[d] for d in DOSES])
    ax.set_yticks(range(len(PAYOFFS)), [f"{100 * q:g}%" for q in PAYOFFS])
    ax.tick_params(length=0)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.text(
        (xc[0.0] + xc[0.01]) / 2,
        -0.03,
        "//",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=st.FS_TICK,
    )
    ax.set_xlabel("pretraining dose of hack-pattern examples (log scale after the break)")
    ax.set_ylabel("RL problems where the hack pays")
    st.panel_title(ax, None, "final hack rate at test, one stripe per seed (sorted)")
    sm = plt.cm.ScalarMappable(cmap=HACK_CMAP, norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, ax=ax, fraction=0.05, pad=0.02, aspect=14)
    cb.set_label("final hack rate, no token")
    cb.ax.yaxis.set_major_formatter(st.PctFormatter())
    cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(width=0.5, length=2)
    handles = [
        Line2D([], [], color=ARM_COLOR["none"], linewidth=1.5, label="no token: 50% boundary"),
        Line2D(
            [],
            [],
            color=ARM_COLOR["meaningful"],
            linewidth=1.5,
            linestyle=(0, (3.5, 1.6)),
            label="meaningful token: 50%\nboundary (token removed)",
        ),
        Line2D(
            [],
            [],
            color=st.INK2,
            marker=r"$\uparrow$",
            linestyle="none",
            markersize=6,
            label="arrow: median below\n50% even at 100%",
        ),
        Patch(
            facecolor="none",
            edgecolor="none",
            label="meaningful token is\nundefined at dose 0\n(nothing to pair it with)",
        ),
    ]
    fig.legend(handles=handles, loc="outside right upper", handlelength=2.2, fontsize=st.FS_SMALL)
    return fig


# ------------------------------------------------------------------------------- figure 6


def fig6_suppression(world: World) -> Figure:
    fig = st.new_figure(st.FULL_W, 2.5)
    axd = fig.subplot_mosaic([["a", "b"]], width_ratios=[1.3, 1.0])
    ax, axb = axd["a"], axd["b"]
    t0 = 25
    t = np.arange(t0, T_POST + 1)
    series = (
        ("none", ARM_COLOR["none"], "-", "no token"),
        ("meaningful", ARM_COLOR["meaningful"], (0, (4, 1.6)), "meaningful token"),
        ("clean", CONTROL, (0, (1, 1.3)), "never hacked"),
    )
    ends = []
    for arm, color, ls, label in series:
        runs = world.default[(0.1, arm)]
        arr = np.array([r.impl_rate(T_POST)[t0:] for r in runs])
        for s in arr:
            ax.plot(t, s, color=color, linestyle=ls, linewidth=0.5, alpha=0.55, zorder=2)
        med = np.median(arr, axis=0)
        ax.plot(t, med, color=color, linestyle=ls, linewidth=1.6, zorder=3, label=label)
        ends.append((float(med[-1]), f"{label}: {100 * med[-1]:.0f}%"))
    # end labels outside the axes, nudged apart so they never overlap
    ends.sort()
    ys = [y for y, _ in ends]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + 0.075)
    for (y, text), y_lab in zip(ends, ys, strict=True):
        ax.annotate(
            text,
            (T_POST, y),
            xytext=(T_POST + 0.8, y_lab),
            textcoords="data",
            ha="left",
            va="center",
            fontsize=st.FS_SMALL,
            annotation_clip=False,
            arrowprops={
                "arrowstyle": "-",
                "color": st.INK2,
                "linewidth": 0.4,
                "shrinkA": 0,
                "shrinkB": 0,
            },
        )
    ax.axvline(N_IT, color=st.INK2, linewidth=0.7, linestyle=(0, (1, 1.3)))
    ax.annotate(
        "bug fixed",
        (N_IT, 0.97),
        xytext=(3, 0),
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=st.FS_SMALL,
        color=st.INK2,
    )
    ax.set_xlim(t0, T_POST)
    ax.set_ylim(0, 1)
    st.pct_axis(ax)
    st.light_grid(ax)
    ax.set_xlabel("RL iteration")
    ax.set_ylabel("hack attempt rate (implication goals)")
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.9), fontsize=st.FS_SMALL)
    st.panel_title(ax, "a", "after the bug fix (dose 0.1%, token kept in prompt)")

    groups = (
        ("clean", "never\nhacked", CONTROL, "s"),
        ("none", "hacked,\nsuppressed", ARM_COLOR["none"], ARM_MARKER["none"]),
        (
            "meaningful",
            "inoculated,\nsuppressed",
            ARM_COLOR["meaningful"],
            ARM_MARKER["meaningful"],
        ),
    )
    for i, (arm, _label, color, marker) in enumerate(groups):
        vals = [r.reelicit for r in world.default[(0.1, arm)]]
        strip(axb, i, vals, color, marker, median_label=f"{np.median(vals):,.0f}")
    st.plain_log_axis(axb, "y")
    axb.set_ylim(10, 10**4)
    axb.set_xticks(range(len(groups)), [g[1] for g in groups])
    axb.tick_params(axis="x", labelsize=st.FS_SMALL, length=0)
    axb.set_xlim(-0.5, len(groups) - 0.25)
    st.light_grid(axb)
    axb.set_ylabel("SFT examples to reach 50% hacks")
    st.panel_title(axb, "b", "re-elicitation cost (token absent)")
    return fig


# ------------------------------------------------------------------------------- contact sheet

FIGURES = (
    ("syn_fig1_spread", fig1_spread),
    ("syn_fig2_goodhart", fig2_goodhart),
    ("syn_fig3_generalization", fig3_generalization),
    ("syn_fig4_inoculation", fig4_inoculation),
    ("syn_fig5_phase", fig5_phase),
    ("syn_fig6_suppression", fig6_suppression),
)


def contact_sheet(pngs: list[Path], out: Path) -> None:
    """All six PNGs on one page, two per row, each at its own aspect ratio, for quick viewing."""
    imgs = [plt.imread(p) for p in pngs]
    col_w, gutter, label_h, header_h = 5.5, 0.3, 0.28, 0.55
    rows = [list(range(i, min(i + 2, len(imgs)))) for i in range(0, len(imgs), 2)]
    row_h = [max(imgs[i].shape[0] / imgs[i].shape[1] for i in row) * col_w for row in rows]
    width = 2 * col_w + 3 * gutter
    height = header_h + sum(h + label_h + gutter for h in row_h)
    fig = plt.figure(figsize=(width, height))
    top = height - header_h
    for row, h in zip(rows, row_h, strict=True):
        top -= label_h
        for j, i in enumerate(row):
            ih = imgs[i].shape[0] / imgs[i].shape[1] * col_w
            x = gutter + j * (col_w + gutter)
            ax = fig.add_axes((x / width, (top - ih) / height, col_w / width, ih / height))
            ax.imshow(imgs[i], interpolation="lanczos", aspect="auto")
            ax.set_axis_off()
            fig.text(x / width, (top + 0.06) / height, pngs[i].name, fontsize=9, family="monospace")
        top -= h + gutter
    fig.text(
        0.5,
        1 - 0.3 / height,
        "SYNTHETIC — hypothetical planning figures, not data "
        "(reward hacking and inoculation toy paper; every number invented)",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color="#a33",
    )
    # 135 dpi keeps the full-colour sheet under the repo's 1000 KB file limit (pre-commit).
    fig.savefig(out, dpi=135, metadata={"Software": "ndbench.figures"})
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=HERE / "figures")
    args = ap.parse_args()
    world = build_world(np.random.default_rng(RNG_SEED))
    pngs = []
    with st.paper_style():
        for name, make in FIGURES:
            fig = make(world)
            watermark(fig)
            paths = st.save(fig, args.out / name)
            st.close(fig)
            pngs.append(paths[-1])
            print("wrote", *(p.name for p in paths))
    contact_sheet(pngs, args.out / "syn_all.png")
    print("wrote syn_all.png")
    summarize(world)


def summarize(world: World) -> None:
    """Print the invented headline numbers, for the note."""
    pct = lambda v: f"{100 * v:.0f}%"  # noqa: E731
    for dose in DOSES:
        runs = world.default[(dose, "none")]
        on = [r.onset() for r in runs]
        fin = [float(r.hack_share()[-1]) for r in runs]
        print(f"dose {DOSE_LABEL[dose]}: onsets {on}, final hack share {[pct(v) for v in fin]}")
    for arm in (*ARMS, "clean"):
        runs = world.default[(0.1, arm)]
        print(
            f"{arm}: train {[pct(float(r.impl_rate()[-1])) for r in runs]} "
            f"test {[pct(r.test_final) for r in runs]} honest {[pct(r.test_honest) for r in runs]} "
            f"it60 {[pct(float(r.impl_rate(T_POST)[-1])) for r in runs]} "
            f"reelicit {[round(r.reelicit) for r in runs]}"
        )


if __name__ == "__main__":
    main()
