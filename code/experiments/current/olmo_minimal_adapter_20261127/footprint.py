# ruff: noqa: E501
"""Token-level footprint of each checkpoint on the final model's (s2300) stored responses.

    footprint.py [--runs DIR] [--out-dir DIR] [--B 2000] [--seed 0] [--jobs 5]

Reads runs/<ckpt>/edl/step0_tokens.npz (per-position NLL and rank under the UNTOUCHED checkpoint, teacher-forced on
s2300's samples) and, per checkpoint and segment (total / thinking / answer), over all responses pooled:
  top-1 and top-5 share (rank == 0, rank < 5); share of the checkpoint's own nats carried by its worst 1 / 5 / 10 % of
  positions; mean NLL per token. 95 % bootstrap over responses (same resamples for every checkpoint).
Writes footprint_summary.{json,csv} and footprint.{pdf,svg,png}.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RUNS = Path("/scratch/cowifornia/nd-rl/olmo_elicit/runs")
OUT = Path("/scratch/cowifornia/nd-rl/olmo_elicit/v2/footprint")
CKPTS = [("dpo", 0), ("s0100", 100), ("s0500", 500), ("s1100", 1100), ("s2300", 2300)]
SEGS = ("total", "think", "answer")
QS = (0.01, 0.05, 0.10)
QNAME = {0.01: "worst1", 0.05: "worst5", 0.10: "worst10"}
METRICS = ("top1", "top5", "worst1", "worst5", "worst10", "nll")


def load_npz(path: Path) -> dict:
    """Flat per-position arrays plus a response index and a thinking mask (only the fields this analysis needs)."""
    z = np.load(path)
    off = z["offset"]
    n_resp = len(z["pid"])
    resp = np.repeat(np.arange(n_resp), np.diff(off))
    pos = np.arange(int(off[-1])) - off[resp]
    return {
        "pid": [str(p) for p in z["pid"].tolist()],
        "set": [str(s) for s in z["set"].tolist()],
        "nll": z["nll"].astype(np.float64),
        "rank": z["rank"],
        "resp": resp,
        "think": pos < z["think_end"][resp],
    }


def seg_mask(d: dict, seg: str) -> np.ndarray:
    if seg == "total":
        return np.ones(len(d["nll"]), dtype=bool)
    return d["think"] if seg == "think" else ~d["think"]


def topk_share(rank: np.ndarray, k: int, w: np.ndarray | None = None) -> float:
    hit = (rank < k).astype(np.float64)
    return float(hit.sum() / len(hit)) if w is None else float((w * hit).sum() / w.sum())


def concentration_sorted(nll_desc: np.ndarray, w_desc: np.ndarray | None, q: float) -> float:
    """Share of the total nats carried by the worst q of positions; nll_desc sorted descending, w_desc = integer
    multiplicities in the same order. k = ceil(q * N) positions, the boundary position counted fractionally."""
    w = np.ones_like(nll_desc) if w_desc is None else w_desc.astype(np.float64)
    cw = np.cumsum(w)
    cs = np.cumsum(w * nll_desc)
    k = np.ceil(q * cw[-1] - 1e-9)
    i = int(np.searchsorted(cw, k))
    prev_w = cw[i - 1] if i > 0 else 0.0
    prev_s = cs[i - 1] if i > 0 else 0.0
    return float((prev_s + (k - prev_w) * nll_desc[i]) / cs[-1])


def concentration(nll: np.ndarray, q: float) -> float:
    return concentration_sorted(np.sort(nll)[::-1], None, q)


def segment_stats(nll, rank, resp, n_resp, W) -> dict:
    """Point estimate and bootstrap replicates (rows of W = multiplicities per response) for one (ckpt, segment)."""
    order = np.argsort(-nll, kind="stable")
    nll_d, resp_d = nll[order], resp[order]
    agg = np.zeros((n_resp, 4))
    for j, v in enumerate((np.ones_like(nll), rank == 0, rank < 5, nll)):
        agg[:, j] = np.bincount(resp, weights=v.astype(np.float64), minlength=n_resp)

    def stats(w_resp: np.ndarray) -> list[float]:
        n, c1, c5, s = w_resp @ agg
        wd = w_resp[resp_d]
        out = [c1 / n, c5 / n]
        out += [concentration_sorted(nll_d, wd, q) for q in QS]
        return [*out, s / n]

    point = stats(np.ones(n_resp))
    reps = np.array([stats(w) for w in W])
    lo, hi = np.percentile(reps, [2.5, 97.5], axis=0)
    res = {"n_positions": int(len(nll)), "n_nats": float(nll.sum())}
    for j, m in enumerate(METRICS):
        res[m] = point[j]
        res[m + "_ci"] = [float(lo[j]), float(hi[j])]
    return res, reps


def _task(args):
    ck, step, path, B, seed = args
    d = load_npz(path)
    n_resp = len(d["pid"])
    W = np.random.default_rng(seed).multinomial(n_resp, np.full(n_resp, 1 / n_resp), size=B)
    out = {"step": step, "n_responses": n_resp, "segments": {}}
    reps = {}
    for seg in SEGS:
        m = seg_mask(d, seg)
        out["segments"][seg], reps[seg] = segment_stats(
            d["nll"][m], d["rank"][m], d["resp"][m], n_resp, W
        )
    # test-set mean NLL per token (the L0 definition), for the cross-check against step0.json / edl_summary.json
    te = np.array([s == "test" for s in d["set"]])[d["resp"]]
    out["test_mean_nll"] = {seg: float(d["nll"][seg_mask(d, seg) & te].mean()) for seg in SEGS}
    return ck, out, reps


def build(runs: Path, B: int, seed: int, jobs: int) -> dict:
    tasks = [(ck, st, runs / ck / "edl" / "step0_tokens.npz", B, seed) for ck, st in CKPTS]
    if jobs > 1:
        with ProcessPoolExecutor(jobs) as ex:
            outs = list(ex.map(_task, tasks))
    else:
        outs = list(map(_task, tasks))
    res = {ck: out for ck, out, _ in outs}
    reps = {ck: r for ck, _, r in outs}
    return {"B": B, "seed": seed, "checkpoints": res, "paired": paired(res, reps)}


def paired(res: dict, reps: dict) -> dict:
    """Paired bootstrap differences b - a (same response resamples for both checkpoints): every checkpoint minus dpo,
    and consecutive checkpoints. diff and its 95 % interval; per100 = the same per 100 RL steps."""
    names = [c for c, _ in CKPTS if c in res]
    pairs = [(names[0], b) for b in names[1:]] + [
        (a, b) for a, b in zip(names[1:], names[2:], strict=False)
    ]
    out: dict = {}
    for a, b in pairs:
        ds = res[b]["step"] - res[a]["step"]
        blk: dict = {"steps": ds, "segments": {}}
        for seg in SEGS:
            sb, sa = res[b]["segments"][seg], res[a]["segments"][seg]
            e: dict = {}
            for j, m in enumerate(METRICS):
                d = reps[b][seg][:, j] - reps[a][seg][:, j]
                lo, hi = np.percentile(d, [2.5, 97.5])
                k = 100.0 / ds
                e[m] = {
                    "diff": sb[m] - sa[m],
                    "ci": [float(lo), float(hi)],
                    "per100": (sb[m] - sa[m]) * k,
                    "per100_ci": [float(lo) * k, float(hi) * k],
                }
            blk["segments"][seg] = e
        out[f"{b}-{a}"] = blk
    return out


def write_csv(summary: dict, path: Path) -> None:
    cols = ["ckpt", "step", "segment", "n_positions", "n_nats"]
    for m in METRICS:
        cols += [m, m + "_lo", m + "_hi"]
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for ck, r in summary["checkpoints"].items():
            for seg, s in r["segments"].items():
                row = [ck, r["step"], seg, s["n_positions"], s["n_nats"]]
                for m in METRICS:
                    row += [s[m], *s[m + "_ci"]]
                w.writerow(row)
    with path.with_name(path.stem + "_paired.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "pair_b_minus_a",
                "steps",
                "segment",
                "metric",
                "diff",
                "lo",
                "hi",
                "per100",
                "per100_lo",
                "per100_hi",
            ]
        )
        for pr, blk in summary.get("paired", {}).items():
            for seg, e in blk["segments"].items():
                for m, v in e.items():
                    w.writerow(
                        [
                            pr,
                            blk["steps"],
                            seg,
                            m,
                            v["diff"],
                            *v["ci"],
                            v["per100"],
                            *v["per100_ci"],
                        ]
                    )


def make_figure(summary: dict):
    """Segment = colour + marker. Metric = line style + fill: filled marker and solid line for the primary metric of a
    panel (top-1, worst 5 %, mean NLL), hollow marker with dotted / dashed line for the others. The 2300 point is
    s2300 scoring its own samples (a reference level, drawn as a star). Linear step axis."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from code.ndbench.figures import style as st

    segs = {
        "total": {"color": "#000000", "marker": "D", "label": "total"},
        "think": {"color": "#0072B2", "marker": "o", "label": "thinking"},
        "answer": {"color": "#D55E00", "marker": "s", "label": "answer"},
    }
    cks = summary["checkpoints"]
    names = [c for c, _ in CKPTS if c in cks]
    steps = [cks[c]["step"] for c in names]

    def line(ax, seg, key, *, hollow=False, ls="-"):
        sty = segs[seg]
        ys = np.array([cks[c]["segments"][seg][key] for c in names])
        ci = np.array([cks[c]["segments"][seg][key + "_ci"] for c in names])
        ax.fill_between(steps, ci[:, 0], ci[:, 1], color=sty["color"], alpha=0.15, lw=0)
        kw = {"color": sty["color"], "mfc": "white" if hollow else sty["color"], "mew": 0.8}
        ax.plot(steps[:-1], ys[:-1], ls=ls, lw=1.0, marker=sty["marker"], ms=3.0, **kw)
        ax.plot(steps[-2:], ys[-2:], ls=ls, lw=1.0, **kw)
        ax.plot(steps[-1:], ys[-1:], ls="none", marker="*", ms=6.0, **kw)

    fig = st.new_figure(st.FULL_W, 2.6, layout="constrained")
    axs = fig.subplots(1, 3)
    for seg in SEGS:
        line(axs[0], seg, "top1")
        line(axs[0], seg, "top5", hollow=True, ls=":")
        line(axs[1], seg, "worst5")
        line(axs[1], seg, "worst1", hollow=True, ls=":")
        line(axs[1], seg, "worst10", hollow=True, ls="--")
        line(axs[2], seg, "nll")
    titles = [
        "top-1 (solid) / top-5 (dotted)",
        "worst 1 (dotted) / 5 (solid) / 10 % (dashed)",
        "mean NLL per token",
    ]
    for ax, letter, t in zip(axs, "abc", titles, strict=True):
        st.panel_title(ax, letter, t, size=7)
        ax.set_xlim(-80, 2380)
        ax.set_xticks([0, 500, 1000, 1500, 2000, 2300])
        ax.set_xticklabels(["0", "500", "1000", "1500", "", "2300"])
        ax.set_xlabel("RL step of the scoring checkpoint")
        st.light_grid(ax)
    axs[0].yaxis.set_major_formatter(st.PctFormatter())
    axs[1].yaxis.set_major_formatter(st.PctFormatter())
    axs[2].set_ylabel("nats per token")
    h = [
        Line2D([], [], color=s["color"], marker=s["marker"], ms=3, label=s["label"])
        for s in segs.values()
    ]
    h.append(
        Line2D([], [], color="#6b6b6b", ls="none", marker="*", ms=6, label="s2300 scores itself")
    )
    axs[2].legend(handles=h, frameon=False, loc="center right", fontsize=6.5)
    plt.rcParams["pdf.fonttype"] = 42
    return fig


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=Path, default=RUNS)
    ap.add_argument("--out-dir", type=Path, default=OUT)
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=5)
    a = ap.parse_args(argv)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    summary = build(a.runs, a.B, a.seed, a.jobs)
    (a.out_dir / "footprint_summary.json").write_text(json.dumps(summary, indent=1))
    write_csv(summary, a.out_dir / "footprint_summary.csv")
    from code.ndbench.figures import style as st

    with st.paper_style():
        fig = make_figure(summary)
        st.save(fig, a.out_dir / "footprint")
        st.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
