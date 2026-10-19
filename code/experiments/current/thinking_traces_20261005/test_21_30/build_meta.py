"""Aggregate the run's reports into test_21_30.meta.json.

    python build_meta.py --out_dir /scratch/.../runs/pools/test_21_30 [--calib calib/] \
        [--power power.json]

Reads (all under --out_dir unless absolute): test_21_30.jsonl, work/*.summary.json (piece runs,
composite build), work/val{3,6,7}_comps_min.jsonl (exact-search validations), profile.json,
seal_report.json, calibration summaries (calib/*/summary.json) and power.json when present.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
from pathlib import Path


def jl(path):
    if not os.path.exists(path):
        return []
    with open(path) as fi:
        return [json.loads(line) for line in fi]


def val_stats(path: str) -> dict | None:
    rows = jl(path)
    if not rows:
        return None
    by = collections.defaultdict(collections.Counter)
    for x in rows:
        m = x["minsearch"]
        g = "timeout" if m["timeout"] else (x["lines"] - m["min_found"] if m["min_found"] else 0)
        by["all"][g] += 1
        by[str(x["lines"])][g] += 1
    out = {}
    for k, v in by.items():
        n = sum(c for g, c in v.items() if g != "timeout")
        out[k] = dict(
            n=sum(v.values()),
            timeouts=v.get("timeout", 0),
            exact_pct=round(100 * v.get(0, 0) / n, 1) if n else None,
            within1_pct=round(100 * (v.get(0, 0) + v.get(1, 0)) / n, 1) if n else None,
            mean_gap=round(sum(g * c for g, c in v.items() if g != "timeout") / n, 3)
            if n
            else None,
            gap_hist={str(g): c for g, c in sorted(v.items(), key=lambda t: str(t[0]))},
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument(
        "--extra", default=None, help="json file merged into the meta (e.g. hand-written notes)"
    )
    a = ap.parse_args()
    D = a.out_dir
    W = f"{D}/work"
    rows = jl(f"{D}/test_21_30.jsonl")
    hist = collections.Counter(r["reference_lines"] for r in rows)
    mins = collections.Counter(r["minimality"]["L_lb"] for r in rows)
    vocab = collections.Counter(r["minimality"]["vocab_search"] for r in rows)
    shortened = sum(
        1 for r in rows if r["minimality"]["construction_lines"] != r["reference_lines"]
    )
    meta = dict(
        file=f"{D}/test_21_30.jsonl",
        n=len(rows),
        reference_lines_hist={str(k): hist[k] for k in sorted(hist)},
        bins2={f"{lo}-{lo + 1}": hist[lo] + hist[lo + 1] for lo in range(21, 31, 2)},
        minimality=dict(
            L_lb_hist={
                str(k): v for k, v in sorted(mins.items(), key=lambda t: (t[0] is None, t[0]))
            },
            vocab_search=dict(vocab),
            shortened_by_vocab_search=shortened,
        ),
        generation=dict(
            pieces={
                os.path.basename(p): json.loads(Path(p).read_text())
                for p in sorted(glob.glob(f"{W}/*.summary.json"))
                if any(s in p for s in ("mains", "lemmas"))
            },
            composites=json.loads(Path(f"{W}/comps.jsonl.summary.json").read_text())
            if os.path.exists(f"{W}/comps.jsonl.summary.json")
            else None,
        ),
        validation=dict(
            literal_recipe_12_16=val_stats(f"{W}/val3_comps_min.jsonl"),
            final_recipe_12_16=val_stats(f"{W}/val6_comps_min.jsonl"),
            final_recipe_17=val_stats(f"{W}/val7_comps_min.jsonl"),
            note=(
                "val6/val7 rows include composites with vacuous discharged antecedents, which the "
                "final recipe rejects; README reports the filtered numbers (val6 n=462: 93.7% "
                "exact, after refinement 99.1%; val7 n=59: 89.8%, after refinement 98.3%)"
            ),
        ),
        lower_bound_samples={
            name: (
                lambda rows: dict(
                    n=len(rows),
                    bound=max((x["minsearch"]["bound"] for x in rows), default=None),
                    certified=sum(
                        1
                        for x in rows
                        if not x["minsearch"]["timeout"]
                        and x["minsearch"]["min_found"] is None
                        and x["minsearch"]["lb"] >= x["minsearch"]["bound"]
                    ),
                    found_shorter=sum(1 for x in rows if x["minsearch"]["min_found"] is not None),
                    timeouts=sum(1 for x in rows if x["minsearch"]["timeout"]),
                    lines_hist=dict(collections.Counter(str(x["lines"]) for x in rows)),
                    secs_median=sorted(x["minsearch"]["secs"] for x in rows)[len(rows) // 2]
                    if rows
                    else None,
                )
            )(jl(f"{W}/{name}_out.jsonl"))
            for name in ("lb15", "lb16")
            if os.path.exists(f"{W}/{name}_out.jsonl")
        },
        piece_reuse=(
            lambda rows: (
                lambda use: dict(
                    distinct_pieces=len(use),
                    uses_hist={
                        str(k): v for k, v in sorted(collections.Counter(use.values()).items())
                    },
                    distinct_mains=len({r["provenance"]["parts"][0][1] for r in rows}),
                    cuts_per_theorem=dict(
                        collections.Counter(str(len(r["provenance"]["cuts"])) for r in rows)
                    ),
                    discharges_per_theorem=dict(
                        collections.Counter(str(len(r["provenance"]["ops"])) for r in rows)
                    ),
                )
            )(collections.Counter(k for r in rows for _, k, _ in r["provenance"]["parts"]))
        )(rows)
        if rows
        else None,
        profile=json.loads(Path(f"{D}/profile.json").read_text())
        if os.path.exists(f"{D}/profile.json")
        else None,
        calib_report=json.loads(Path(f"{D}/calib_report.json").read_text())
        if os.path.exists(f"{D}/calib_report.json")
        else None,
        seal=(
            {
                k: v
                for k, v in json.loads(Path(f"{D}/seal_report.json").read_text()).items()
                if k != "hit_rows"
            }
            if os.path.exists(f"{D}/seal_report.json")
            else None
        ),
        calibration={},
        power=json.loads(Path(f"{D}/power.json").read_text())
        if os.path.exists(f"{D}/power.json")
        else None,
    )
    for s in sorted(glob.glob(f"{D}/calib/*/summary.json")):
        js = json.loads(Path(s).read_text())
        cells = {}
        for name, c in js["cells"].items():
            cells[name] = dict(
                n=c["all"]["n"],
                solved=c["all"]["solved"],
                pass_at_k_pct=c["all"]["pass_at_k_pct"],
                wall_s=round(c["wall_s"], 1),
                tokens_per_attempt=c["all"].get("tokens_per_attempt"),
                status=c["all"].get("status"),
            )
        meta["calibration"][os.path.basename(os.path.dirname(s))] = dict(
            ckpt=js["ckpt"], k=js["k"], cells=cells
        )
    if a.extra and os.path.exists(a.extra):
        meta.update(json.loads(Path(a.extra).read_text()))
    with open(f"{D}/test_21_30.meta.json", "w") as fo:
        json.dump(meta, fo, indent=1)
    print(json.dumps({k: meta[k] for k in ("n", "reference_lines_hist", "bins2")}))


if __name__ == "__main__":
    main()
