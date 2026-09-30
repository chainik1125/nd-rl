#!/usr/bin/env python3
"""Summarize a TTRL run against its control: stepping-stone rates by variant kind (round 1),
and targets solved vs cumulative samples. Writes a chart next to the blog draft.

  python3 analyze.py artifacts/ttrl/dev64_ttrl/ttrl_d2k16.json \
      artifacts/ttrl/dev64_control/control_k256.json --chart <png>
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ttrl")
    ap.add_argument("control", nargs="?")
    ap.add_argument("--chart")
    a = ap.parse_args()
    ttrl = read(a.ttrl)
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
    curves = {"TTRL": curve(ttrl)}
    if a.control:
        ctl = read(a.control)
        curves["sample more (control)"] = curve(ctl)
        print("\ncontrol:", list(zip(*curves["sample more (control)"], strict=True)))
    if a.chart:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6.4, 4))
        for (name, (xs, ys)), c in zip(curves.items(), ("#2a6fdb", "#9a9a9a"), strict=False):
            ax.plot([x / n_t for x in xs], ys, marker="o", color=c, label=name)
        ax.set_xlabel("samples per target (TTRL: target + its variants)")
        ax.set_ylabel(f"targets solved (of {n_t})")
        ax.set_xscale("symlog", linthresh=16)
        ax.grid(alpha=0.3)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(a.chart, dpi=150)


if __name__ == "__main__":
    main()
