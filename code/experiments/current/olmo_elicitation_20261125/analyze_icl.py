# ruff: noqa: E501
"""In-context description length readouts from `olmo_icl.py run` outputs (review R2; orchestrator's design 2026-11-25).

    analyze_icl.py [--runs /scratch/cowifornia/nd-rl/olmo_elicit/runs] [--out /scratch/cowifornia/nd-rl/olmo_elicit/icl_summary] [--B 2000]

Per checkpoint, target set (test, reach) and segment (total / think / answer), for every k:
  L(k)       nats per response token of the final model's responses with k demonstrations in context: summed over the
             targets and the demo draws, divided by their tokens (k = 0 has one draw: it is step 0's L_0);
  gain(k)    L(0) - L(k), nats per token (positive = the demos help);
  share(k)   (L(0, t) - L(k, t)) / (L(0, t) - L(0, s2300)): the share of the gap to the final model closed by k demos
             (undefined at s2300; the raw gain is the readout there);
  per_draw   L(k) of each demo draw separately (demo-choice noise).
Intervals: 95 % percentile bootstrap, B replicates, resampling targets and (for k > 0) demo draws jointly; gain and
share pair the same target indices (and, for share, the same targets at s2300). Trends: Spearman of L(k) and share(k)
with RL step. Descriptive reading from the review (R2, not pre-registered): "elicitable in context" if share(4) >= 0.25
with the lower bound above 0.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from analyze_edl import CKPTS, FINAL, OFF_LINE, SEGS, STEP, ci, seg_toks, seg_vals, spearman

RUNS = Path("/scratch/cowifornia/nd-rl/olmo_elicit/runs")
OUT = Path("/scratch/cowifornia/nd-rl/olmo_elicit/icl_summary")
TSETS = ("test", "reach")


def load(runs: Path) -> dict[str, dict]:
    """ckpt -> {(k, draw): row} (latest row per key)."""
    out = {}
    for c in CKPTS:
        p = runs / c / "icl" / "icl_rows.jsonl"
        if p.exists():
            rows = {}
            for ln in p.read_text().splitlines():
                try:
                    r = json.loads(ln)
                except json.JSONDecodeError:
                    continue
                if r.get("schema", "").startswith("olmo_icl_row"):
                    rows[(int(r["k"]), int(r["draw"]))] = r
            if rows:
                out[c] = rows
    return out


def arrays(
    rows: dict, k: int, tset: str, seg: str
) -> tuple[np.ndarray, np.ndarray, list[str]] | None:
    """(nats (draws, targets), tokens (targets,), target pids) for one k, or None."""
    rs = [rows[key] for key in sorted(rows) if key[0] == k and tset in rows[key]]
    if not rs:
        return None
    pids = rs[0][tset]["pids"]
    assert all(r[tset]["pids"] == pids for r in rs), "target order differs between draws"
    return np.stack([seg_vals(r[tset], seg) for r in rs]), seg_toks(rs[0][tset], seg), pids


def rate(v: np.ndarray, t: np.ndarray, ti: np.ndarray | None = None, di: np.ndarray | None = None):
    """Nats per token over draws x targets; with bootstrap indices ti (B, m) and di (B, D), B replicates."""
    if ti is None:
        return float(v.sum() / (t.sum() * v.shape[0]))
    sub = v[:, ti]  # (D, B, m)
    if di is not None and v.shape[0] > 1:
        sub = np.take_along_axis(sub.transpose(1, 0, 2), di[:, :, None], axis=1)  # (B, D', m)
        return sub.sum(axis=(1, 2)) / (t[ti].sum(axis=1) * di.shape[1])
    return sub.sum(axis=(0, 2)) / (t[ti].sum(axis=1) * v.shape[0])


def analyze(runs: Path, B: int = 2000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    data = load(runs)
    res: dict = {}
    for c, rows in data.items():
        ks = sorted({k for k, _ in rows})
        r: dict = {
            "step": STEP[c],
            "off_released_line": c in OFF_LINE,
            "ks": ks,
            "draws": {str(k): sorted(d for kk, d in rows if kk == k) for k in ks},
            "demo_pids": {f"{k}/{d}": rows[(k, d)].get("demo_pids") for k, d in sorted(rows)},
            "k0_check_max_abs_diff": rows.get((0, 0), {}).get("k0_check_max_abs_diff"),
            "max_context_len": max(
                max(rr[t]["context_len"]) for rr in rows.values() for t in TSETS if t in rr
            ),
        }
        for tset in TSETS:
            r[tset] = {}
            for seg in SEGS:
                base = arrays(rows, 0, tset, seg)
                m = base[0].shape[1] if base else None
                ti = rng.integers(0, m, size=(B, m)) if m else None
                segd: dict = {}
                for k in ks:
                    a = arrays(rows, k, tset, seg)
                    if a is None:
                        continue
                    v, t, _ = a
                    di = (
                        rng.integers(0, v.shape[0], size=(B, v.shape[0]))
                        if v.shape[0] > 1
                        else None
                    )
                    ti_k = rng.integers(0, v.shape[1], size=(B, v.shape[1])) if ti is None else ti
                    pt, bs = rate(v, t), rate(v, t, ti_k, di)
                    e = {
                        "L": pt,
                        "ci": ci(bs),
                        "draws": v.shape[0],
                        "per_draw": [float(v[d].sum() / t.sum()) for d in range(v.shape[0])],
                    }
                    if base is not None and k > 0:
                        b0 = rate(base[0], base[1], ti_k)
                        e["gain"] = rate(base[0], base[1]) - pt
                        e["gain_ci"] = ci(b0 - bs)
                    segd[str(k)] = e
                r[tset][seg] = segd
        res[c] = r
    # share of the gap to s2300 closed (paired over the same targets)
    f = res.get(FINAL)
    for c, r in res.items():
        if c == FINAL or f is None:
            continue
        for tset in TSETS:
            for seg in SEGS:
                fa = arrays(data[FINAL], 0, tset, seg)
                ca = arrays(data[c], 0, tset, seg)
                if fa is None or ca is None or fa[2] != ca[2]:
                    continue
                lf_pt = rate(fa[0], fa[1])
                for k, e in r[tset][seg].items():
                    if k == "0" or "gain" not in e:
                        continue
                    # same target resample for t's L(0), L(k) and s2300's L(0)
                    m = ca[0].shape[1]
                    ti = rng.integers(0, m, size=(B, m))
                    v, t, _ = arrays(data[c], int(k), tset, seg)
                    di = (
                        rng.integers(0, v.shape[0], size=(B, v.shape[0]))
                        if v.shape[0] > 1
                        else None
                    )
                    l0b, lkb, lfb = (
                        rate(ca[0], ca[1], ti),
                        rate(v, t, ti, di),
                        rate(fa[0], fa[1], ti),
                    )
                    l0 = rate(ca[0], ca[1])
                    e["share"] = (l0 - e["L"]) / (l0 - lf_pt)
                    e["share_ci"] = ci((l0b - lkb) / (l0b - lfb))
    names = [c for c in CKPTS if c in res]
    steps = [STEP[c] for c in names]
    trends = {}
    for tset in TSETS:
        for seg in SEGS:
            for k in sorted({k for c in names for k in res[c][tset][seg]}, key=int):
                trends[f"spearman_L{k}_{tset}_{seg}"] = spearman(
                    steps, [res[c][tset][seg].get(k, {}).get("L", float("nan")) for c in names]
                )
                if k != "0":
                    trends[f"spearman_share{k}_{tset}_{seg}"] = spearman(
                        steps,
                        [res[c][tset][seg].get(k, {}).get("share", float("nan")) for c in names],
                    )
    return {
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runs": str(runs),
        "B": B,
        "seed": seed,
        "definitions": __doc__.split("Per checkpoint,")[1].strip(),
        "checkpoints": res,
        "trends": trends,
    }


def flatten(res: dict) -> list[dict]:
    rows = []
    for c in [x for x in CKPTS if x in res]:
        for tset in TSETS:
            for seg, segd in res[c].get(tset, {}).items():
                for k, e in segd.items():
                    rows.append(
                        {
                            "ckpt": c,
                            "step": STEP[c],
                            "set": tset,
                            "segment": seg,
                            "k": int(k),
                            "metric": "L",
                            "value": e["L"],
                            "lo": e["ci"][0],
                            "hi": e["ci"][1],
                            "unit": "nats/token",
                            "draws": e["draws"],
                        }
                    )
                    if "gain" in e:
                        rows.append(
                            {
                                "ckpt": c,
                                "step": STEP[c],
                                "set": tset,
                                "segment": seg,
                                "k": int(k),
                                "metric": "gain",
                                "value": e["gain"],
                                "lo": e["gain_ci"][0],
                                "hi": e["gain_ci"][1],
                                "unit": "nats/token",
                                "draws": e["draws"],
                            }
                        )
                    if "share" in e:
                        rows.append(
                            {
                                "ckpt": c,
                                "step": STEP[c],
                                "set": tset,
                                "segment": seg,
                                "k": int(k),
                                "metric": "gap_share",
                                "value": e["share"],
                                "lo": e["share_ci"][0],
                                "hi": e["share_ci"][1],
                                "unit": "fraction",
                                "draws": e["draws"],
                            }
                        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--runs", type=Path, default=RUNS)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    s = analyze(a.runs, a.B, a.seed)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.with_suffix(".json").write_text(json.dumps(s, indent=1) + "\n")
    rows = flatten(s["checkpoints"])
    with open(a.out.with_suffix(".csv"), "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "ckpt",
                "step",
                "set",
                "segment",
                "k",
                "metric",
                "value",
                "lo",
                "hi",
                "unit",
                "draws",
            ],
        )
        w.writeheader()
        w.writerows(rows)
    print(
        f"{len(s['checkpoints'])} checkpoints {list(s['checkpoints'])}; wrote {a.out.with_suffix('.json')} and .csv ({len(rows)} rows)"
    )
    for c, r in s["checkpoints"].items():
        for seg in ("total", "answer"):
            d = r["test"].get(seg, {})
            print(
                f"  {c:6s} test {seg:6s} "
                + "  ".join(
                    f"k={k}: {e['L']:.4f}" + (f" (share {e['share']:.2f})" if "share" in e else "")
                    for k, e in d.items()
                )
            )


if __name__ == "__main__":
    main()
