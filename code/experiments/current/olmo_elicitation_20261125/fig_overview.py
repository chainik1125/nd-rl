# ruff: noqa: E501
"""Mini-run overview over RL step -> /scratch/cowifornia/nd-rl/olmo_elicit/figures/overview.{pdf,svg,png},
overview.csv (every plotted number) and overview.md (half-way table, reach table, caption). CPU only; re-run it as
stage-L results arrive (after analyze_arms.py).

(a) Behaviour, % strict checker passes, at the five mini-run checkpoints (steps 0, 100, 500, 1100, 2300):
    arm-A pass@1 and pass@8 on the 218 IFBench prompts (Dan's stored values, stage0/repro_summary.csv `his`);
    on the 48 reach prompts the plain prompt (Dan's arm B, all draws), the 60-word hint (H) and the stage-1 adapter
    (L, only where a result exists); on the 41 transplant prompts T and T + answer-only instruction (A1)
    (arms.csv). Hollow markers = Dan's stored cells, filled = our draws (as in fig_arms.py).
(b) Progress toward the final model, p = (v - v_dpo) / (v_s2300 - v_dpo): 0 at dpo, 1 at s2300, for arm-A pass@1,
    the plain reach rate, H, T, A1, test L0 (total and answer, nats/token; edl_summary.json), the null-corrected
    EDL* (norm, total; edl_summary.json `posthoc.delta_EDL_star`) and the share of positions with JS > 0.1 against
    s2300 (tokens_summary.json, set `all`, segment `total`). The strip under (b) marks each curve's half-way step:
    the first step where p reaches 0.5, linear between the two checkpoints that bracket it (bar = the bracket).
    L is not in (b): it is an elicited rate, not a distance to the final model. Panel (b) is post hoc and
    descriptive (no pre-registered reading uses it).
Usage: python fig_overview.py [--oe /scratch/cowifornia/nd-rl/olmo_elicit] [--out .../figures/overview]
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

OE = Path("/scratch/cowifornia/nd-rl/olmo_elicit")
MINI = ["dpo", "s0100", "s0500", "s1100", "s2300"]
STEP = {"dpo": 0, "s0100": 100, "s0500": 500, "s1100": 1100, "s2300": 2300}
FIRST, FINAL = MINI[0], MINI[-1]
PROGRESS_UNIT = "progress (0 = dpo, 1 = s2300)"
MS = {
    "*": 5.5,
    "L": 5.0,
    "armA_pass8": 3.2,
}  # marker sizes: the star reads small; pass@8 sits on T's points
ZO = {
    "armA_pass8": 4
}  # pass@8's small hollow pentagons drawn over T's filled discs (65.6 vs 65.1 %, 87.2 vs 87.2 %)
DX = {
    "L": -50
}  # L drawn 50 steps left of its checkpoint: at dpo it sits on A1's point (63.5 vs 63.3 %)


# ------------------------------------------------------------------------------------- arithmetic


def progress(values: list[float | None]) -> list[float | None]:
    """(v - v_first) / (v_last - v_first): 0 at the first value, 1 at the last, for rising and falling measures alike.
    Missing values stay None. ValueError if an end is missing or the two ends are equal (nothing to normalise by)."""
    v0, v1 = values[0], values[-1]
    if v0 is None or v1 is None:
        raise ValueError("progress needs the first and the last value")
    if v1 == v0:
        raise ValueError("progress undefined: the first and last values are equal")
    return [None if v is None else (v - v0) / (v1 - v0) for v in values]


def half_step(steps: list[float], prog: list[float | None], target: float = 0.5) -> dict:
    """First step at which the piecewise-linear curve through the available (step, progress) points reaches `target`,
    walking from the first point; linear interpolation inside the bracketing pair of checkpoints. Points with a
    missing progress are skipped. step None if the curve never reaches the target; `monotone` says whether the
    available points never fall."""
    pts = [
        (float(s), float(p))
        for s, p in zip(steps, prog, strict=True)
        if p is not None and math.isfinite(p)
    ]
    mono = all(b[1] >= a[1] for a, b in zip(pts, pts[1:], strict=False))
    for i, (s, p) in enumerate(pts):
        if p >= target:
            if i == 0:
                return {"step": s, "bracket": (s, s), "monotone": mono}
            s0, p0 = pts[i - 1]  # p0 < target <= p, so p > p0
            return {
                "step": s0 + (target - p0) / (p - p0) * (s - s0),
                "bracket": (s0, s),
                "monotone": mono,
            }
    return {"step": None, "bracket": None, "monotone": mono}


# ------------------------------------------------------------------------------------------ data


@dataclass
class Series:
    key: str
    label: str  # legend / strip label
    unit: str
    color: str
    marker: str
    ls: str = "-"
    panels: str = "ab"
    pts: dict = field(default_factory=dict)  # ckpt -> (value, lo, hi, source)

    def values(self) -> list[float | None]:
        return [self.pts[c][0] if c in self.pts else None for c in MINI]


def _f(x) -> float | None:
    return None if x in (None, "", "None") else float(x)


def load_arm_a(path: Path) -> dict[int, dict[str, float]]:
    """k -> ckpt -> % (Dan's stored arm-A pass@k on the 218 prompts; the `his` column)."""
    out: dict[int, dict[str, float]] = {}
    for r in csv.DictReader(path.open()):
        if r["ckpt"] in STEP and r["k"] in ("1", "8") and _f(r["his"]) is not None:
            out.setdefault(int(r["k"]), {})[r["ckpt"]] = 100 * float(r["his"])
    return out


def load_arms(path: Path) -> dict[tuple[str, str], dict]:
    return {
        (r["arm"], r["ckpt"]): r
        for r in csv.DictReader(path.open())
        if r["ckpt"] in STEP and _f(r["strict_pct"]) is not None
    }


def build_series(oe: Path, C: dict) -> list[Series]:
    arm_a = load_arm_a(oe / "stage0" / "repro_summary.csv")
    arms = load_arms(oe / "arms.csv")
    edl = json.loads((oe / "edl_summary.json").read_text())
    tok = json.loads((oe / "tokens_summary.json").read_text())
    S_ = [
        Series("armA_pass1", "arm A pass@1 (218)", "% of draws", C["black"], "o"),
        Series(
            "armA_pass8",
            "arm A pass@8 (218)",
            "% of prompts (unbiased pass@8)",
            C["black"],
            "p",
            ":",
            panels="a",
        ),
        Series("B", "plain prompt, B (48)", "% of draws", C["reddish_purple"], "D", "--"),
        Series("H", "60-word hint, H (48)", "% of draws", C["bluish_green"], "^"),
        Series("T", "transplant, T (41)", "% of draws", C["blue"], "o"),
        Series("A1", "T + answer-only, A1 (41)", "% of draws", C["vermillion"], "s", "--"),
        Series("L", "adapter, L (48; 8 draws)", "% of draws", C["orange"], "v", panels="a"),
        Series("L0_test_total", "test L0, total", "nats/token", C["sky_blue"], "P", panels="b"),
        Series(
            "L0_test_answer", "test L0, answer", "nats/token", C["sky_blue"], "X", "--", panels="b"
        ),
        Series(
            "dEDLstar_norm_total",
            "EDL* null-corr. (norm, total)",
            "nats",
            "#6b6b6b",
            "*",
            "-.",
            panels="b",
        ),
        Series(
            "share_js_gt_0.1",
            "share JS > 0.1 (all, total)",
            "% of positions",
            "#8a7f00",
            "h",
            ":",
            panels="b",
        ),
    ]
    by = {s.key: s for s in S_}
    for k, key in ((1, "armA_pass1"), (8, "armA_pass8")):
        for c, v in arm_a.get(k, {}).items():
            by[key].pts[c] = (v, None, None, "dan: stored arm A (stage0/repro_summary.csv, his)")
    for arm in ("B", "H", "T", "A1", "L"):
        for c in MINI:
            r = arms.get((arm, c))
            if r:
                by[arm].pts[c] = (float(r["strict_pct"]), _f(r["lo95"]), _f(r["hi95"]), r["source"])
    for c in MINI:
        ck = edl.get("checkpoints", {}).get(c)
        if ck:
            for seg in ("total", "answer"):
                d = ck["L0"]["test"][seg]
                by[f"L0_test_{seg}"].pts[c] = (
                    d["nats_per_tok"],
                    *d["ci"],
                    "ours: edl_summary.json checkpoints.L0.test",
                )
        d = edl.get("posthoc", {}).get("delta_EDL_star", {}).get(c)
        if d:
            d = d["norm"]["total"]
            by["dEDLstar_norm_total"].pts[c] = (
                d["nats"],
                *d["ci"],
                "ours: edl_summary.json posthoc.delta_EDL_star.norm.total",
            )
        t = tok.get("checkpoints", {}).get(c)
        if t:
            d = t["sets"]["all"]["total"]
            lo, hi = d.get("share_js_gt_0.1_ci", [None, None])
            by["share_js_gt_0.1"].pts[c] = (
                100 * d["share_js_gt_0.1"],
                None if lo is None else 100 * lo,
                None if hi is None else 100 * hi,
                "ours: tokens_summary.json sets.all.total",
            )
    return S_


def is_ours(source: str) -> bool:
    return source.startswith("ours")


def p4_reference(edl: dict) -> dict[str, float]:
    """The half-way steps edl_summary.json already holds (posthoc.P4) for the measures both compute."""
    p4 = edl.get("posthoc", {}).get("P4", {})
    get = {
        "armA_pass1": lambda: p4["pass1"]["step"],
        "L0_test_total": lambda: p4["L0_test"]["total"]["step"],
        "L0_test_answer": lambda: p4["L0_test"]["answer"]["step"],
        "dEDLstar_norm_total": lambda: p4["delta_EDL_star"]["norm"]["total"]["step"],
    }
    out = {}
    for k, g in get.items():
        with contextlib.suppress(KeyError, TypeError):
            out[k] = g()
    return out


# ------------------------------------------------------------------------------------------ main


def fmt(v, nd=1) -> str:
    return "–" if v is None else f"{v:.{nd}f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oe", default=str(OE))
    ap.add_argument("--out", default=None, help="figure base path (default <oe>/figures/overview)")
    a = ap.parse_args()
    oe = Path(a.oe)
    out = Path(a.out) if a.out else oe / "figures" / "overview"
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # .../code
    from ndbench.figures import style as S

    series = build_series(oe, S.OKABE_ITO)
    edl = json.loads((oe / "edl_summary.json").read_text())
    ref = p4_reference(edl)
    steps = [STEP[c] for c in MINI]

    # progress and half-way steps (panel b)
    prog: dict[str, list[float | None]] = {}
    half: dict[str, dict] = {}
    skipped: dict[str, str] = {}
    for s in series:
        if "b" not in s.panels:
            continue
        try:
            prog[s.key] = progress(s.values())
        except ValueError as e:
            skipped[s.key] = str(e)
            continue
        half[s.key] = half_step(steps, prog[s.key])
    mismatch = {
        k: (half[k]["step"], v)
        for k, v in ref.items()
        if k in half and (half[k]["step"] is None or abs(half[k]["step"] - v) > 0.01)
    }

    # ---- csv: every plotted number
    rows = []
    for s in series:
        for c in MINI:
            if c not in s.pts:
                continue
            v, lo, hi, src = s.pts[c]
            if "a" in s.panels:
                rows.append(
                    {
                        "panel": "a",
                        "measure": s.key,
                        "label": s.label,
                        "kind": "point",
                        "ckpt": c,
                        "step": STEP[c],
                        "value": v,
                        "unit": s.unit,
                        "lo95": lo,
                        "hi95": hi,
                        "raw": "",
                        "raw_unit": "",
                        "source": src,
                    }
                )
            if s.key in prog and prog[s.key][MINI.index(c)] is not None:
                rows.append(
                    {
                        "panel": "b",
                        "measure": s.key,
                        "label": s.label,
                        "kind": "point",
                        "ckpt": c,
                        "step": STEP[c],
                        "value": prog[s.key][MINI.index(c)],
                        "unit": PROGRESS_UNIT,
                        "lo95": "",
                        "hi95": "",
                        "raw": v,
                        "raw_unit": s.unit,
                        "source": src,
                    }
                )
        if s.key in half:
            h = half[s.key]
            rows.append(
                {
                    "panel": "b",
                    "measure": s.key,
                    "label": s.label,
                    "kind": "halfway",
                    "ckpt": "",
                    "step": h["step"],
                    "value": 0.5,
                    "unit": PROGRESS_UNIT,
                    "lo95": "",
                    "hi95": "",
                    "raw": "",
                    "raw_unit": "",
                    "source": f"bracket {h['bracket']}; monotone {h['monotone']}",
                }
            )
    cols = [
        "panel",
        "measure",
        "label",
        "kind",
        "ckpt",
        "step",
        "value",
        "unit",
        "lo95",
        "hi95",
        "raw",
        "raw_unit",
        "source",
    ]
    csv_path = out.with_suffix(".csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows({k: ("" if r[k] is None else r[k]) for k in cols} for r in rows)

    # ---- figure
    bser = [s for s in series if s.key in prog]
    with S.paper_style():
        fig = S.new_figure(S.FULL_W, 4.7)
        gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.0, 0.78])
        axa = fig.add_subplot(gs[0])
        axb = fig.add_subplot(gs[1], sharex=axa)
        axs = fig.add_subplot(gs[2], sharex=axa)

        def curve(ax, s, ys):
            pts = [
                (STEP[c], y, s.pts[c][3]) for c, y in zip(MINI, ys, strict=True) if y is not None
            ]
            if not pts:
                return
            ax.plot(
                [p[0] + DX.get(s.key, 0) for p in pts],
                [p[1] for p in pts],
                color=s.color,
                ls=s.ls,
                lw=1.0,
                zorder=2,
            )
            for x, y, src in pts:
                ms = MS.get(s.key, MS.get(s.marker, 4.0))
                ax.plot(
                    [x + DX.get(s.key, 0)],
                    [y],
                    marker=s.marker,
                    ms=ms,
                    color=s.color,
                    mfc=s.color if is_ours(src) else "white",
                    mec=s.color,
                    mew=0.9,
                    ls="none",
                    zorder=ZO.get(s.key, 3),
                )

        for s in series:
            if "a" in s.panels and s.pts:
                curve(axa, s, [s.pts[c][0] / 100 if c in s.pts else None for c in MINI])
                axa.plot(
                    [],
                    [],
                    color=s.color,
                    ls=s.ls,
                    lw=1.0,
                    marker=s.marker,
                    ms=MS.get(s.key, MS.get(s.marker, 4.0)),
                    mfc=s.color if any(is_ours(v[3]) for v in s.pts.values()) else "white",
                    mec=s.color,
                    label=s.label,
                )
        axa.set_ylim(-0.03, 1.0)
        S.pct_axis(axa)
        axa.set_ylabel("strict checker passes")
        S.panel_title(
            axa,
            "a",
            "Behaviour, % strict checker passes (hollow: Dan's stored cells; filled: ours)",
            size=S.FS_TICK,
        )
        axa.legend(
            loc="upper left",
            bbox_to_anchor=(1.01, 1.02),
            fontsize=S.FS_SMALL,
            title="(n prompts; filled: has our draws)",
            title_fontsize=S.FS_SMALL,
            alignment="left",
        )

        for s in bser:
            curve(axb, s, prog[s.key])
        lo_b = min(-0.05, *(p for v in prog.values() for p in v if p is not None))
        hi_b = max(1.05, *(p for v in prog.values() for p in v if p is not None))
        axb.set_ylim(lo_b - 0.02, hi_b + 0.02)
        axb.set_yticks([0, 0.5, 1])
        axb.axhline(0.5, color=S.REF, lw=0.6, ls=(0, (2, 2)), zorder=0)
        axb.set_ylabel("progress\n0 = dpo, 1 = s2300")
        S.panel_title(
            axb,
            "b",
            "Progress toward the final model (post hoc, descriptive; key and half-way steps below)",
            size=S.FS_TICK,
        )

        # strip: half-way step per measure, bar = bracketing checkpoints; labels on the right double as (b)'s key
        n = len(bser)
        for i, s in enumerate(bser):
            y = n - 1 - i
            h = half[s.key]
            axs.plot([0, 2300], [y, y], color=S.GRID, lw=0.5, zorder=0)
            if h["step"] is None:
                axs.text(
                    2300,
                    y,
                    "not reached ",
                    ha="right",
                    va="center",
                    fontsize=S.FS_SMALL,
                    color=S.INK2,
                )
                continue
            b0, b1 = h["bracket"]
            axs.plot(
                [b0, b1], [y, y], color=s.color, lw=3.0, alpha=0.28, solid_capstyle="butt", zorder=1
            )
            ms = MS.get(s.key, MS.get(s.marker, 4.0))
            axs.plot(
                [h["step"]],
                [y],
                marker=s.marker,
                ms=ms,
                color=s.color,
                mec=s.color,
                ls="none",
                zorder=3,
            )
        axs.set_ylim(-0.7, n - 0.3)
        axs.yaxis.tick_right()
        axs.set_yticks(range(n))
        axs.set_yticklabels(
            [f"{s.label}: {fmt(half[s.key]['step'], 0)}" for s in reversed(bser)],
            fontsize=S.FS_SMALL,
        )
        axs.tick_params(axis="y", length=0)
        for side in ("left", "top"):
            axs.spines[side].set_visible(False)
        axs.spines["right"].set_visible(False)
        S.panel_title(
            axs,
            None,
            "half-way step of each (b) curve (progress = 0.5; bar = bracketing checkpoints)",
            size=S.FS_SMALL,
        )
        axs.set_xlabel("RL step")

        for ax in (axa, axb, axs):
            ax.set_xlim(-90, 2360)
            for x in steps:
                ax.axvline(x, color=S.GRID, lw=0.5, zorder=0)
        for ax in (axa, axb):
            S.light_grid(ax)
            ax.tick_params(axis="x", labelbottom=False)
        axs.set_xticks(steps)
        S.save(fig, out)
        S.close(fig)

    # ---- markdown sidecar
    arms = load_arms(oe / "arms.csv")
    md = [
        "# Mini-run overview (generated by fig_overview.py)",
        "",
        f"Generated {datetime.now(UTC):%Y-%m-%d %H:%M} UTC from `stage0/repro_summary.csv`, `arms.csv`, `edl_summary.json` and `tokens_summary.json` in `{oe}`. Figure: `{out}.{{pdf,svg,png}}`; every plotted number: `{csv_path.name}`.",
        "",
        "**Caption.** Five checkpoints of the Olmo-3 32B Think RL run (steps 0, 100, 500, 1100, 2300). (a) Strict checker passes: arm-A pass@1 and pass@8 on the 218 IFBench prompts; on the 48 reach prompts the plain prompt (B, all of Dan's stored draws: 16 per prompt at dpo, 32 elsewhere), the 60-word hint (H) and the stage-1 adapter (L, 8 draws, shown only where run; drawn 50 steps left of its checkpoint so that it does not hide A1's dpo point; overview.csv keeps the true step); on the 41 transplant prompts the transplanted answers without (T) and with (A1) the answer-only instruction. Hollow markers: Dan's stored cells; filled: our draws. Arm-A pass@8 is the unbiased per-prompt estimate (% of prompts), the other lines % of draws. No intervals drawn; prompt-bootstrap 95 % intervals for the arms.csv lines are in `overview.csv` and `arms.png`. (b) **Post hoc, descriptive:** each measure rescaled to 0 at dpo and 1 at s2300; the strip marks the first step where a curve reaches 0.5 (linear between checkpoints; the bar is the bracketing pair of checkpoints). With five checkpoints a half-way step is only located within its bracket; the interpolated value is not a measurement. For the behaviour measures the end points (dpo, s2300) of H, T and A1 are Dan's stored cells and the middle points are ours.",
        "",
        "## Half-way steps, panel (b) (post hoc, descriptive)",
        "",
        "| measure | value at dpo | value at s2300 | unit | half-way step | bracket | monotone | edl_summary P4 |",
        "|---|---:|---:|---|---:|---|---|---:|",
    ]
    for s in bser:
        h = half[s.key]
        v = s.values()
        br = "–" if h["bracket"] is None else f"{h['bracket'][0]:g}–{h['bracket'][1]:g}"
        md.append(
            f"| {s.label} | {v[0]:.4g} | {v[-1]:.4g} | {s.unit} | {fmt(h['step'], 0)} | {br} | {'yes' if h['monotone'] else 'no'} | {fmt(ref.get(s.key), 0)} |"
        )
    for k, e in skipped.items():
        md.append(f"| {k} | – | – | – | not computed: {e} | – | – | – |")
    md += [
        "",
        "Last column: the same half-way step as computed independently by `analyze_edl.py` (`posthoc.P4`), where it exists."
        + (f" **Mismatch:** {mismatch}." if mismatch else " All agree to 0.01 steps."),
        "",
        "## Reach prompts: plain, hint, adapter (48 prompts, % of draws, strict checker passes)",
        "",
        "| checkpoint | step | plain (B, all draws) | plain (B, first 8) | hint (H) | adapter (L, 8 draws) | L: prompts solved in 8 | final model plain (s2300 B) |",
        "|---|---:|---|---|---|---|---:|---|",
    ]

    def cell(arm, c):
        r = arms.get((arm, c))
        if not r:
            return "not run"
        return f"{float(r['strict_pct']):.1f} [{fmt(_f(r['lo95']))}, {fmt(_f(r['hi95']))}]"

    fin = cell("B", FINAL)
    for c in MINI:
        rl = arms.get(("L", c))
        solved = (
            f"{float(rl['solved_pct']):.1f} %"
            if rl and _f(rl.get("solved_pct")) is not None
            else "–"
        )
        md.append(
            f"| {c} | {STEP[c]} | {cell('B', c)} | {cell('B8', c)} | {cell('H', c)} | {cell('L', c)} | {solved} | {fin} |"
        )
    md += [
        "",
        "Brackets: prompt-bootstrap 95 % intervals (arms.csv). B and B8 are Dan's stored draws; H at dpo and s2300 is Dan's stored cell, H elsewhere and L are ours. L is not in panel (b): it is an elicited rate, not a distance to the final model.",
        "",
    ]
    out.with_suffix(".md").write_text("\n".join(md))
    print(f"wrote {out}.{{pdf,svg,png}}, {csv_path}, {out.with_suffix('.md')}")
    for s in bser:
        print(
            f"  half-way {s.key:22s} {fmt(half[s.key]['step'], 1):>8s}  bracket {half[s.key]['bracket']}  P4 {fmt(ref.get(s.key), 1)}"
        )
    if mismatch:
        print(f"WARNING: half-way steps differ from edl_summary P4: {mismatch}")
    if skipped:
        print(f"skipped in (b): {skipped}")


if __name__ == "__main__":
    main()
