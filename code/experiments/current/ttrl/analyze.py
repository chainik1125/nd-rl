#!/usr/bin/env python3
"""Summarize a TTRL run against its control: stepping-stone rates by variant kind (round 1),
and targets solved vs cumulative samples. Writes a chart next to the blog draft.

  python3 analyze.py "seed 0/TTRL=artifacts/ttrl/dev64_ttrl/ttrl_d2k16.json" \
      "seed 0/sample more=artifacts/ttrl/dev64_control/control_k256.json" --chart <png>
"""

import argparse
import collections
import json


def curve(run):
    """(cumulative samples, cumulative targets solved) after each round."""
    xs, ys, tot = [0], [0], 0
    for r in run["rounds"]:
        tot += r["sampling"]["samples"]
        xs.append(tot)
        ys.append(r["targets_solved"])
    return xs, ys


def probe(run):
    """Round-1 solve rates by kind (depth:move): items, share with any proof, share in the band."""
    by = collections.defaultdict(lambda: [0, 0, 0, 0.0])
    master = run["args"]["master"]
    for it in run["items"]:
        if not it["rates"]:
            continue
        k = it["kind"]
        k = "target" if k == "target" else f"d{k.count('/') + 1} {k.split('/')[-1]}"
        p = it["rates"][0]
        b = by[k]
        b[0] += 1
        b[1] += p > 0
        b[2] += 0 < p <= master
        b[3] += p
    return {k: v for k, v in sorted(by.items())}


def read(path):
    with open(path) as f:
        return json.load(f)


COLORS = {"TTRL": "#2a6fdb", "unrelated variants": "#e08a2e", "sample more": "#9a9a9a"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="panel/series=path; the first run gets the tables")
    ap.add_argument("--chart")
    a = ap.parse_args()
    runs = [(spec.split("=")[0].split("/"), read(spec.split("=")[1])) for spec in a.runs]
    ttrl = runs[0][1]
    n_t = sum(it["kind"] == "target" for it in ttrl["items"])
    print("| kind | items | any proof | in band | mean rate |\n| --- | --- | --- | --- | --- |")
    for k, (n, anyp, band, s) in probe(ttrl).items():
        print(f"| {k} | {n} | {anyp / n:.0%} | {band / n:.0%} | {s / n:.2f} |")
    print("\nround | active | samples | secs sample | secs ft | RL records | targets solved")
    for r in ttrl["rounds"]:
        print(
            r["round"],
            r["active"],
            r["sampling"]["samples"],
            round(r["sampling"]["secs"]),
            round(r["finetune"]["secs"]),
            r["mix_rl_records"],
            f"{r['targets_solved']}/{n_t}",
        )
    solved = [it for it in ttrl["items"] if it["kind"] == "target" and it["left"]]
    print("solved targets (round):", sorted((it["left"][0], it["name"]) for it in solved))
    panels = collections.defaultdict(dict)
    for (panel, series), run in runs:
        xs, ys = curve(run)
        panels[panel][series] = ([x / n_t for x in xs], ys)
        print(f"\n{panel}/{series}:", [(round(x), y) for x, y in zip(xs, ys, strict=True)][1:])
    if a.chart:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4), sharey=True)
        for ax, (panel, curves) in zip(
            fig.axes if len(panels) > 1 else [axes], panels.items(), strict=True
        ):
            for name, (xs, ys) in curves.items():
                ax.plot(xs, ys, marker="o", ms=4, color=COLORS.get(name), label=name)
                ax.annotate(
                    f"{ys[-1]}", (xs[-1], ys[-1]), xytext=(6, -3), textcoords="offset points"
                )
            ax.set_title(panel)
            ax.set_xlabel("samples per target (incl. variants)")
            ax.set_ylim(0, n_t)
            ax.grid(alpha=0.3)
            ax.legend(frameon=False, loc="upper left")
        fig.axes[0].set_ylabel(f"targets solved (of {n_t})")
        fig.tight_layout()
        fig.savefig(a.chart, dpi=150)


if __name__ == "__main__":
    main()
