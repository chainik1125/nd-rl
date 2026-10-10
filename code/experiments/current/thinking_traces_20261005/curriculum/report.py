"""Final checks and the per-band table for the length-curriculum pools.

    LEAN_GATE_WORKERS=4 LEAN_GATE_LOG=.../curriculum/work/lean_gate_report.jsonl python -m \
        code.experiments.current.thinking_traces_20261005.curriculum.report \
        --root /scratch/cowifornia/nd-rl/think/runs/pools/curriculum

For each band (L1, L2, L3) it reads `pool.jsonl` and `test_1k.jsonl` and:

* re-checks every row in Lean, the constructed proof (`gen_proof`) and the reference (shortest)
  proof, through the state-env decomposition the eval uses (`hard712/run.py lean_flags`);
* checks every row's class against the exclusion index (hard712's own pool and work files are
  ignored for rows reused from hard712, which come from them), textbook72 explicitly, and the
  held-out sets up to curried normal form (`heldout.py`: Charles's 2026-10-02 release and
  textbook72);
* asserts that all six files are pairwise disjoint by renaming class (`pkey`);
* tabulates size, shortest-proof lengths, where the rows came from, and the mix-gate shares.

Writes `ROOT/report.json` and `ROOT/report.md`.
"""

from __future__ import annotations

import argparse
import collections
import json
import pickle
from pathlib import Path
from typing import Any

from code.experiments.current.thinking_traces_20261005.curriculum import heldout as HO
from code.experiments.current.thinking_traces_20261005.hard712 import run as RN

BANDS = {"L1": (7, 9), "L2": (10, 12), "L3": (13, 15)}
TB = "/scratch/cowifornia/nd-rl/think/nd-takehome/data/eval_only/textbook72"
H712 = "/scratch/cowifornia/nd-rl/think/runs/pools/hard712"
OWN = {
    f"{H712}/pool.jsonl",
    *(f"{H712}/work/{p}.jsonl" for p in ("gated", "gated_lean", "selected", "spare")),
}
SHARES = ("classical", "ex_falso", "reductio", "nested_neg", "case_split", "no_prem", "goal_ho")


def mean(xs: list[float]) -> float:
    return round(sum(xs) / len(xs), 2)


def jl(p: str | Path) -> list[dict[str, Any]]:
    with open(p) as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--index", default="")
    a = ap.parse_args()
    root = Path(a.root)
    with open(a.index or root / "work" / "excl_index.pkl", "rb") as f:
        idx = pickle.load(f)
    tb_keys: set[str] = set()
    for p in ("textbook_dev", "textbook_train"):
        for r in jl(f"{TB}/{p}.jsonl"):
            k, pk = RN.class_keys(r["prompt"])
            tb_keys |= {k} | ({pk} if pk else set())
    hidx = HO.heldout_index()
    sidx = HO.heldout_index(HO.SEALED)
    allc: dict[str, set[str]] = {}
    rep: dict[str, Any] = {}
    allk: dict[str, set[str]] = {}
    for band, (lo, hi) in BANDS.items():
        d = root / band
        if not (d / "pool.jsonl").exists():
            continue
        origin = {r["pkey"]: r.get("origin", "?") for r in jl(d / "work" / "selected.jsonl")}
        sel = json.loads((d / "work" / "select_report.json").read_text())
        b: dict[str, Any] = {}
        for part in ("pool", "test_1k"):
            rows = jl(d / f"{part}.jsonl")
            allk[f"{band}/{part}"] = {r["pkey"] for r in rows}
            allc[f"{band}/{part}"] = set().union(*(HO.keys(r["prompt"]) for r in rows))
            ml = collections.Counter(r["min_lines"] for r in rows)
            assert all(lo <= x <= hi for x in ml), (band, part, ml)
            g = RN.lean_flags(rows, "gen_proof")
            ref = RN.lean_flags(rows, "reference_proof")
            hits = 0
            for r in rows:
                h = idx["keys"].get(r["key"], set()) | idx["pkeys"].get(r["pkey"], set())
                if origin.get(r["pkey"]) == "reused":
                    h = h - OWN
                hits += bool(h)
            b[part] = dict(
                n=len(rows),
                min_lines=dict(sorted(ml.items())),
                mean_lines=round(sum(r["min_lines"] for r in rows) / len(rows), 2),
                origin=dict(collections.Counter(origin.get(r["pkey"], "?") for r in rows)),
                lean_constructed=f"{sum(g)}/{len(rows)}",
                lean_reference=f"{sum(ref)}/{len(rows)}",
                index_hits=hits,
                textbook72_hits=sum(r["key"] in tb_keys or r["pkey"] in tb_keys for r in rows),
                heldout_curried_hits=sum(bool(HO.hits(hidx, r["prompt"])) for r in rows),
                sealed_curried_hits=sum(bool(HO.hits(sidx, r["prompt"])) for r in rows),
                features={
                    "connectives": mean([r["gap_thm"]["n_conn"] for r in rows]),
                    "premises": mean([r["n_prem"] for r in rows]),
                    "atoms": mean([r["gap_thm"]["n_atoms"] for r in rows]),
                    "box_depth": mean([r["gap_proof"]["p_box_depth"] for r in rows]),
                    "state_env_steps": mean([r["ref_steps"] for r in rows]),
                    **{
                        k: round(sum(bool(r["attrs"][k]) for r in rows) / len(rows), 3)
                        for k in SHARES
                    },
                },
                top_recipes=collections.Counter(r["recipe"] for r in rows).most_common(5),
            )
        b["mix_selected"] = {k: sel["shares_selected"][k] for k in SHARES}
        b["mix_reservoir"] = {k: sel["shares_gated"][k] for k in SHARES}
        b["caps_relaxed"] = sel["caps_relaxed"]
        b["reservoir"] = sel["n_in"]
        rep[band] = b
    names = sorted(allk)
    for i, x in enumerate(names):
        for y in names[i + 1 :]:
            assert not (allk[x] & allk[y]), (x, y, len(allk[x] & allk[y]))
            assert not (allc[x] & allc[y]), (x, y, "curried", len(allc[x] & allc[y]))
    rep["disjoint"] = names
    (root / "report.json").write_text(json.dumps(rep, indent=1))
    lines = [
        "| band | file | n | shortest-proof lines (%) | mean | origin | Lean constructed / "
        "reference | index / textbook72 / held-out (curried) / sealed (curried) hits |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for band, b in rep.items():
        if band == "disjoint":
            continue
        for part in ("pool", "test_1k"):
            p = b[part]
            dist = " ".join(f"{k}:{100 * v / p['n']:.0f}" for k, v in p["min_lines"].items())
            org = ", ".join(f"{k} {v}" for k, v in sorted(p["origin"].items()))
            lines.append(
                f"| {band} | {part} | {p['n']} | {dist} | {p['mean_lines']} | {org} | "
                f"{p['lean_constructed']} / {p['lean_reference']} | {p['index_hits']} / "
                f"{p['textbook72_hits']} / {p['heldout_curried_hits']} / "
                f"{p['sealed_curried_hits']} |"
            )
    feats = [b for b in rep if b != "disjoint"]
    if feats:
        keys = list(rep[feats[0]]["pool"]["features"])
        lines += ["", "| pool feature | " + " | ".join(feats) + " |", "|---|" + "---|" * len(feats)]
        for k in keys:
            vals = " | ".join(str(rep[b]["pool"]["features"][k]) for b in feats)
            lines.append(f"| {k} | {vals} |")
    (root / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
