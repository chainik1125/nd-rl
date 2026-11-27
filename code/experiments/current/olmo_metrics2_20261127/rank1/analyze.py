# ruff: noqa: E501
"""M2 analysis (Turn 5 B.6, Turn 6 D1): recovered fraction of G with paired joint prompt bootstrap, ExPO's share of the s0500 ->
s2300 gap, A2 against stored dpo, collapse counts, figure through code/ndbench/figures/style.py.

  python3 analyze.py --cells cells.json --inputs <inputs/rank1> --out <dir> [--reps 2000]

cells.json: {"cells": [{"name": "A1", "graded": "runs/A1/out/graded.jsonl", "n": {"H": 2, "R": 4}, "kind": "adapter|control|expo"}, ...]}
(graded.jsonl from tools/grade_job.sh; one draw per row, --chunk 1). Per cell and set (H, R; D for ExPO dev cells):
  pass@1 in % of prompts x PLANNED draws (killed draws score as fails; the all-pass bound next to it), "any pass" share of prompts,
  paired prompt-bootstrap difference against stored dpo and stored s2300, f = (p - p_dpo) / (p_s2300 - p_dpo) with its interval and
  Cai's ratio p / p_s2300 beside it; for kind expo also f' = (p - p_s0500) / (p_s2300 - p_s0500); collapse counts (union rule).
A2 is read against stored dpo (the zero-point check of the LoRA engine): its diff interval is in the table.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import collapse_stats, missing_bound, prompt_rates, read_jsonl, read_pids  # noqa: E402
from stats import pass_diff, recovered_fraction  # noqa: E402


def analyze(cells: list[dict], refs: dict, pids: dict[str, list[str]], reps: int = 2000) -> dict:
    out = {"cells": {}}
    for c in cells:
        rows = read_jsonl(c["graded"])
        res = {"kind": c.get("kind", "adapter"), "n": c["n"], "sets": {}}
        if "label" in c:  # display name for the figure only
            res["label"] = c["label"]
        planned = sum(c["n"].get(t, 0) * len(pids[t]) for t in c["n"])
        res["collapse"] = collapse_stats(rows, planned)
        for tag, n in c["n"].items():
            ps = pids[tag]
            sub = [r for r in rows if r["pid"] in set(ps)]
            rate = prompt_rates(sub, ps, n)
            bound = missing_bound(sub, ps, n)
            by_any = {}
            for r in sub:
                by_any[r["pid"]] = by_any.get(r["pid"], False) or bool(r["strict"])
            s = {
                "n_prompts": len(ps),
                "draws_per_prompt": n,
                "rows": len(sub),
                "pass1_pct": 100 * float(np.mean(list(rate.values()))),
                "pass1_all_pass_bound_pct": 100 * float(np.mean(list(bound.values()))),
                "any_pass_pct": 100 * sum(by_any.get(p, False) for p in ps) / len(ps),
                "stored": {
                    k: 100 * float(np.mean(list(refs[k][tag].values())))
                    for k in ("dpo", "s0500", "s2300")
                    if tag in refs[k]
                },
            }
            if tag in refs["dpo"] and tag in refs["s2300"]:
                s["vs_dpo"] = pass_diff(rate, refs["dpo"][tag], reps, 11)
                s["vs_s2300"] = pass_diff(rate, refs["s2300"][tag], reps, 12)
                s["f"] = recovered_fraction(rate, refs["dpo"][tag], refs["s2300"][tag], reps, 13)
                if res["kind"] == "expo" and "s0500" in refs and tag in refs["s0500"]:
                    s["f_prime_vs_s0500_gap"] = recovered_fraction(
                        rate, refs["s0500"][tag], refs["s2300"][tag], reps, 14
                    )
            res["sets"][tag] = s
        out["cells"][c["name"]] = res
    return out


def to_csv(res: dict, path: Path) -> None:
    cols = [
        "cell",
        "set",
        "draws_per_prompt",
        "pass1_pct",
        "all_pass_bound_pct",
        "any_pass_pct",
        "f",
        "f_lo",
        "f_hi",
        "cai_ratio",
        "diff_vs_dpo_pts",
        "diff_vs_dpo_lo",
        "diff_vs_dpo_hi",
        "f_prime",
        "f_prime_lo",
        "f_prime_hi",
        "collapsed",
        "bad_union",
        "planned",
    ]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for name, c in res["cells"].items():
            for tag, s in c["sets"].items():
                fr, fp, vd = s.get("f", {}), s.get("f_prime_vs_s0500_gap", {}), s.get("vs_dpo", {})
                w.writerow(
                    [
                        name,
                        tag,
                        s["draws_per_prompt"],
                        round(s["pass1_pct"], 2),
                        round(s["pass1_all_pass_bound_pct"], 2),
                        round(s["any_pass_pct"], 2),
                        fr.get("f"),
                        fr.get("f_lo"),
                        fr.get("f_hi"),
                        fr.get("cai_ratio"),
                        100 * vd["diff"] if vd else None,
                        100 * vd["lo"] if vd else None,
                        100 * vd["hi"] if vd else None,
                        fp.get("f"),
                        fp.get("f_lo"),
                        fp.get("f_hi"),
                        c["collapse"]["collapsed"],
                        c["collapse"]["bad_union"],
                        c["collapse"]["planned"],
                    ]
                )


def figure(res: dict, base: Path) -> list[Path]:
    root = Path(__file__).resolve().parents[4]  # .../code
    sys.path.insert(0, str(root))
    from ndbench.figures import style as S

    colors = {"adapter": "#0072B2", "control": "#7a7a7a", "expo": "#D55E00"}
    sets = [("H", "(a) held-out arm A, 112 prompts"), ("R", "(b) reach, 48 prompts")]
    with S.paper_style():
        fig = S.new_figure(S.FULL_W, 2.3)
        axes = fig.subplots(1, 2, sharey=True)
        names = list(res["cells"])
        for ax, (tag, title) in zip(axes, sets):
            for i, nme in enumerate(names):
                c = res["cells"][nme]
                s = c["sets"].get(tag)
                if not s or "f" not in s or "f" not in s["f"]:
                    continue
                f = s["f"]
                ax.errorbar(
                    [f["f"]],
                    [i],
                    xerr=[[f["f"] - f["f_lo"]], [f["f_hi"] - f["f"]]],
                    fmt="o",
                    ms=3.2,
                    lw=0.8,
                    capsize=2,
                    color=colors.get(c["kind"], "k"),
                    mfc="white" if c["collapse"]["collapsed"] else colors.get(c["kind"], "k"),
                )
            ax.axvline(0, color="#999999", lw=0.5)
            ax.axvline(1, color="#999999", lw=0.5, ls="--")
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels([res["cells"][n].get("label", n) for n in names])
            ax.invert_yaxis()
            ax.set_xlabel("gap recovered, f (0 = dpo, 1 = s2300)")
            ax.set_title(title, loc="left", fontsize=8)
        return S.save(fig, base)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cells", required=True)
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--no-figure", action="store_true")
    a = ap.parse_args(argv)
    inp, out = Path(a.inputs), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    refs = json.load(open(inp / "refs.json"))
    pids = {t: read_pids(inp / f"pids_{t}.txt") for t in "DHR"}
    cells = json.load(open(a.cells))["cells"]
    res = analyze(cells, refs, pids, a.reps)
    (out / "results.json").write_text(json.dumps(res, indent=1))
    to_csv(res, out / "results.csv")
    if not a.no_figure:
        figure(res, out / "rank1_recovered_fraction")
    print(open(out / "results.csv").read())
    return 0


if __name__ == "__main__":
    sys.exit(main())
