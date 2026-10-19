"""Aggregate the 13-20 run's reports into test_13_20.meta.json.

python meta13.py --out_dir /scratch/cowifornia/nd-rl/think/runs/pools/test_13_20 [--extra x.json]
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os


def jl(path: str) -> list:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(x) for x in f]


def jload(path: str):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def last_line(path: str) -> str | None:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        lines = f.read().strip().splitlines()
    return lines[-1] if lines else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--extra", default=None)
    a = ap.parse_args()
    D, W = a.out_dir, f"{a.out_dir}/work"
    rows = jl(f"{D}/test_13_20.jsonl")
    hist = collections.Counter(r["reference_lines"] for r in rows)
    cert = collections.defaultdict(collections.Counter)
    for r in rows:
        cert[str(r["reference_lines"])][r["minimality"]["cert"]] += 1
    use = collections.Counter(k for r in rows for _, k, _ in r["provenance"]["parts"])
    meta = dict(
        file=f"{D}/test_13_20.jsonl",
        n=len(rows),
        reference_lines_hist={str(k): hist[k] for k in sorted(hist)},
        certification_by_length={k: dict(v) for k, v in sorted(cert.items())},
        L_lb_by_length={
            str(L): dict(
                collections.Counter(
                    r["minimality"]["L_lb"] for r in rows if r["reference_lines"] == L
                )
            )
            for L in sorted(hist)
        },
        vocab_search=dict(collections.Counter(str(r["minimality"]["vocab_search"]) for r in rows)),
        piece_reuse=dict(
            distinct_pieces=len(use),
            uses_hist={str(k): v for k, v in sorted(collections.Counter(use.values()).items())},
            distinct_mains=len({r["provenance"]["parts"][0][1] for r in rows}),
            cuts_per_theorem=dict(
                collections.Counter(str(len(r["provenance"]["cuts"])) for r in rows)
            ),
            discharges_per_theorem=dict(
                collections.Counter(str(len(r["provenance"]["ops"])) for r in rows)
            ),
        ),
        generation=dict(
            composites=jload(f"{W}/comps.jsonl.summary.json"),
            refine_log=last_line(f"{W}/refine.log"),
        ),
        profile=jload(f"{D}/profile.json"),
        seal={k: v for k, v in (jload(f"{D}/seal_report.json") or {}).items() if k != "hit_rows"},
        calib_report=jload(f"{D}/calib_report.json"),
        power=jload(f"{D}/power.json"),
        calibration={},
    )
    for s in sorted(glob.glob(f"{D}/calib/*/summary.json")):
        js = jload(s)
        meta["calibration"][os.path.basename(os.path.dirname(s))] = dict(
            ckpt=js["ckpt"],
            k=js["k"],
            cells={
                n: dict(
                    n=c["all"]["n"],
                    solved=c["all"]["solved"],
                    pass_at_k_pct=c["all"]["pass_at_k_pct"],
                    wall_s=round(c["wall_s"], 1),
                )
                for n, c in js["cells"].items()
            },
        )
    if a.extra:
        meta.update(jload(a.extra) or {})
    with open(f"{D}/test_13_20.meta.json", "w") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps(dict(n=meta["n"], hist=meta["reference_lines_hist"])))


if __name__ == "__main__":
    main()
