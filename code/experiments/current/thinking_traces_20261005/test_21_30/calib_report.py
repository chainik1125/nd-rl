"""Calibration read-out: pass@1 / pass@k by reference length, and accepted-proof lengths against the
references.

    python calib_report.py --set test_21_30.jsonl --rows T0=calib/T0_s0/rows_plain_B0.jsonl \
        --rows T1v1_B256=calib/T1v1_s0/rows_plain_B256.jsonl --out calib_report.json

Rows are joined to the set by prompt (rows of theorems not in the set are ignored). pass@j is the
unbiased estimator (Chen et al.) averaged over theorems, in % of the theorems. An accepted proof's
ND length is the number of `have` statements plus `fun ( x : A )` binders of its `lean_seq` text
(exact on all 2,668 candidate references); `shorter` counts theorems whose shortest accepted proof
is shorter than the reference.
"""

from __future__ import annotations

import argparse
import collections
import json
import math


def pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))


def nd_lines(text: str) -> int:
    t = text.split()
    return t.count("have") + sum(1 for i in range(len(t) - 1) if t[i] == "fun" and t[i + 1] == "(")


def report(set_rows: dict, rows: list[dict]) -> dict:
    by_len = collections.defaultdict(list)
    shorter = collections.Counter()
    status = collections.Counter()
    for r in rows:
        if r["prompt"] not in set_rows:
            continue
        L = set_rows[r["prompt"]]
        by_len[L].append(r)
        status.update(r["status"])
        if r["accepted"]:
            m = min(nd_lines(t) for t in r["accepted"])
            shorter["shorter" if m < L else ("equal" if m == L else "longer")] += 1
            if m < L:
                shorter[f"shorter_by_{L - m}"] += 1
    k = rows[0]["k"]

    def agg(rs):
        n = len(rs)
        return dict(
            n=n,
            solved=sum(r["n_ok"] > 0 for r in rs),
            pass1=round(100 * sum(r["n_ok"] / r["k"] for r in rs) / n, 3) if n else None,
            passk=round(100 * sum(pass_at_k(r["k"], r["n_ok"], k) for r in rs) / n, 3)
            if n
            else None,
        )

    allr = [r for rs in by_len.values() for r in rs]
    out = dict(k=k, all=agg(allr), by_len={str(L): agg(by_len[L]) for L in sorted(by_len)})
    out["by_bin2"] = {
        f"{lo}-{lo + 1}": agg(by_len.get(lo, []) + by_len.get(lo + 1, []))
        for lo in range(21, 31, 2)
    }
    out["accepted_vs_reference"] = dict(shorter)
    out["status"] = dict(status.most_common())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--rows", action="append", required=True, help="NAME=path")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    set_rows = {}
    with open(a.set) as fh:
        for line in fh:
            r = json.loads(line)
            set_rows[r["prompt"]] = r["reference_lines"]
    res = {}
    for spec in a.rows:
        name, path = spec.split("=", 1)
        with open(path) as fi:
            rows = [json.loads(line) for line in fi]
        res[name] = report(set_rows, rows)
        x = res[name]
        print(
            f"{name}: n {x['all']['n']} pass@1 {x['all']['pass1']:.2f}% pass@{x['k']} "
            f"{x['all']['passk']:.2f}% solved {x['all']['solved']}; accepted vs ref "
            f"{x['accepted_vs_reference']}"
        )
        for b, v in x["by_bin2"].items():
            print(f"   {b}: n {v['n']} pass@1 {v['pass1']}% pass@{x['k']} {v['passk']}%")
    with open(a.out, "w") as fo:
        json.dump(res, fo, indent=1)


if __name__ == "__main__":
    main()
