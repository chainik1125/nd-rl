# ruff: noqa: E501
"""Cap bounds for a reach batch stopped by its pod-side cap (PREREG implementation note of 2026-11-27 3:59-4:00 pm).

cap_bounds.py sets   --samples-dir RUNS/dpo/sample/batch_a6s2 --pids PIDS_L.txt --out v2/a6/cap_sets.json
    From the samples files AS FETCHED AT THE CAP (before any resume): per cell, the chunks complete (all `chunk`
    rows present) and unfinished. upper = every unfinished chunk; lower = unfinished chunks whose request was added
    before some finished request (multi_sample.py adds requests round-robin over the sorted cell names: request k of
    every cell before request k+1 of any; per cell in pids-file order, chunks 0, 1, ...), which must have started
    under first-come-first-served scheduling.
cap_bounds.py bounds --runs v2/runs_a6s2 --sets v2/a6/cap_sets.json --out v2/a6/cap_bounds.json
    Every corner's rate and each of the ten step-2 contrasts (a6_reading.PAIRS): as sampled, and with the upper-set
    (and, beside it, lower-set) chunks' draws set all-fail / all-pass; for a contrast the two extreme assignments (first
    cell's affected chunks all-pass with the second's all-fail, and the reverse). Two-level paired bootstrap as
    analyze_cells. A contrast is "not robust to the cap" when its interval's position against 0 (above / includes /
    below) differs under either upper-set extreme.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

CHUNK = 4


def want_order(pids_file: Path) -> list[tuple[str, int]]:
    out = []
    for ln in pids_file.read_text().splitlines():
        if not ln.strip():
            continue
        pid, n = ln.split(":")
        out += [(pid, ch) for ch in range((int(n) + CHUNK - 1) // CHUNK)]
    return out


def read_rows(p: Path) -> list[dict]:
    rows = []
    if not p.exists():
        return rows
    try:
        with gzip.open(p, "rt") as fh:
            for ln in fh:
                try:
                    rows.append(json.loads(ln))
                except json.JSONDecodeError:
                    break
    except (EOFError, OSError):
        pass  # a file cut by the kill: keep the rows read so far
    return rows


def sets(samples_dir: Path, pids_file: Path) -> dict:
    order = want_order(pids_file)
    cells = sorted(p.name for p in samples_dir.iterdir() if p.is_dir())
    reqs = [
        (c, *order[k]) for k in range(len(order)) for c in cells
    ]  # round-robin, as multi_sample.py
    cnt: dict = {}
    for c in cells:
        for r in read_rows(samples_dir / c / "samples.jsonl.gz"):
            key = (c, r["pid"], int(r["chunk"]))
            cnt[key] = cnt.get(key, 0) + 1
    done = {k for k, v in cnt.items() if v == CHUNK}
    idx_done = [i for i, k in enumerate(reqs) if k in done]
    last = max(idx_done) if idx_done else -1
    upper = [k for k in reqs if k not in done]
    lower = [k for i, k in enumerate(reqs) if k not in done and i < last]
    per = {
        c: {
            "upper": sum(k[0] == c for k in upper),
            "lower": sum(k[0] == c for k in lower),
            "chunks": len(order),
        }
        for c in cells
    }
    return {
        "cells": cells,
        "n_requests": len(reqs),
        "complete": len(done),
        "upper": upper,
        "lower": lower,
        "per_cell": per,
        "partial_rows_dropped": {f"{k[0]}|{k[1]}|{k[2]}": v for k, v in cnt.items() if v != CHUNK},
    }


def bounds(runs: Path, sets_json: dict) -> dict:
    import a6_reading as ar
    import analyze_cells as ac

    reach = ac.dc.reach()
    cells = {}
    for name, cid in ar.CORNERS.items():
        hit = [
            b
            for b in sorted((runs / "dpo" / "sample").glob("*/"))
            if (b / cid / "graded.jsonl").exists()
        ]
        if not hit:
            raise SystemExit(f"{cid} not sampled under {runs}")
        b = hit[-1]
        cells[name] = ac.to_cells(
            ac.load_detail(b / cid / "samples.jsonl.gz", b / cid / "graded.jsonl", reach)
        )

    def affected(which: str) -> dict[str, set]:
        out: dict[str, set] = {}
        inv = {v: k for k, v in ar.CORNERS.items()}
        for c, pid, ch in sets_json[which]:
            if c in inv:
                out.setdefault(inv[c], set()).add((pid, int(ch)))
        return out

    def setto(cell: dict, aff: set, val: int) -> dict:
        return {
            p: [((val if (p, i // CHUNK) in aff else s), w) for i, (s, w) in enumerate(v)]
            for p, v in cell.items()
        }

    def rate(cell: dict) -> float:
        ps = [p for p in reach if p in cell]
        return round(100 * sum(sum(s for s, _ in cell[p]) / len(cell[p]) for p in ps) / len(ps), 2)

    def pos(c: dict) -> str:
        return "above" if c["lo"] > 0 else ("below" if c["hi"] < 0 else "includes")

    out: dict = {"rates": {}, "contrasts": {}, "per_cell_upper": {}, "per_cell_lower": {}}
    for which in ("upper", "lower"):
        aff = affected(which)
        for name in cells:
            out[f"per_cell_{which}"][name] = len(aff.get(name, ()))
    for name, cell in cells.items():
        row = {"as_sampled": rate(cell)}
        for which in ("upper", "lower"):
            a = affected(which).get(name, set())
            row[f"{which}_all_fail"] = rate(setto(cell, a, 0))
            row[f"{which}_all_pass"] = rate(setto(cell, a, 1))
        out["rates"][name] = row
    for x, y in ar.PAIRS:
        base = ac.contrast(cells[x], cells[y])
        row = {"as_sampled": base, "position": pos(base)}
        for which in ("upper", "lower"):
            ax, ay = affected(which).get(x, set()), affected(which).get(y, set())
            hi = ac.contrast(setto(cells[x], ax, 1), setto(cells[y], ay, 0))
            lo = ac.contrast(setto(cells[x], ax, 0), setto(cells[y], ay, 1))
            row[f"{which}_x_pass_y_fail"], row[f"{which}_x_fail_y_pass"] = hi, lo
            row[f"{which}_positions"] = [pos(hi), pos(lo)]
        row["robust_to_cap"] = all(p == row["position"] for p in row["upper_positions"])
        out["contrasts"][f"{x}-{y}"] = row
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sets")
    s.add_argument("--samples-dir", type=Path, required=True)
    s.add_argument("--pids", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    b = sub.add_parser("bounds")
    b.add_argument("--runs", type=Path, required=True)
    b.add_argument("--sets", type=Path, required=True)
    b.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.cmd == "sets":
        r = sets(a.samples_dir, a.pids)
        a.out.write_text(json.dumps(r, indent=1) + "\n")
        print(f"complete {r['complete']} of {r['n_requests']}; per cell {r['per_cell']}")
    else:
        r = bounds(a.runs, json.loads(a.sets.read_text()))
        a.out.write_text(json.dumps(r, indent=1) + "\n")
        for k, v in r["rates"].items():
            print(k, v)
        for k, v in r["contrasts"].items():
            print(
                k,
                v["as_sampled"]["diff"],
                v["position"],
                "upper",
                v["upper_positions"],
                "robust" if v["robust_to_cap"] else "NOT ROBUST",
            )


if __name__ == "__main__":
    main()
