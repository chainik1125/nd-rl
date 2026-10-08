#!/usr/bin/env python3
"""Pass@k curves by test-time thinking budget, one panel per model, as one HTML page.

  python passk_report.py OUT.html

Reads the per-problem success counts already on disk: the looped spike's loop sweeps and the
Coconut spike's thought sweeps (~/repos/nd-rl/artifacts), and the per-theorem dev evals of the
thoughts loop's experiments (the nd-rl-12 and nd-rl-14 checkouts). Nothing is sampled here.
Pass@k is exact: a problem with c successes in K samples is missed by a random k-subset with
probability C(K-c, k) / C(K, k).
"""

import json
import math
import subprocess
import sys
import time
from pathlib import Path

HOME = Path.home() / "repos"
LOOPED = HOME / "nd-rl/artifacts/looped_spike_20261006"
COCONUT = HOME / "nd-rl/artifacts/coconut_spike_20261006"
LOOP_ARTS = [
    HOME / "nd-rl-12/artifacts/autoresearch_thoughts",
    HOME / "nd-rl-14/artifacts/autoresearch_thoughts",
]
HERE = Path(__file__).resolve().parent


def curve(counts, K):
    """[(k, share of problems solved with k samples)] for k = 1, 2, 4, ..., K."""
    out, k = [], 1
    while k <= K:
        miss = sum(math.comb(K - c, k) / math.comb(K, k) for c in counts)
        out.append([k, round(1 - miss / len(counts), 4)])
        k *= 2
    return out


def sweep(path, knob):
    """{pool: {"K":, "n":, "lines": [{"level":, "depth":, "pts":}]}} from one sweep file."""
    if not Path(path).exists():
        return {}
    pools = {}
    for r in json.loads(Path(path).read_text())["results"]:
        counts = [p["n_ok"] for p in r["problems"].values()]
        pool = pools.setdefault(r["pool"], {"K": r["k"], "n": r["n"], "lines": []})
        pool["lines"].append(
            {"level": r[knob], "depth": r.get("depth", 6), "pts": curve(counts, r["k"])}
        )
    return pools


def dev_eval(tag, seeds=(0, 1, 2)):
    """Mean over seeds of the pass@k curve of a loop experiment's scored dev eval (level 8), and
    the mean dev count at the diagnostic level, which has no per-theorem record."""
    curves, diag = [], []
    for s in seeds:
        run = next(
            (a / f"{tag}_s{s}" for a in LOOP_ARTS if (a / f"{tag}_s{s}/eval_dev.jsonl").exists()),
            None,
        )
        if run is None:
            continue
        rows = [json.loads(line) for line in (run / "eval_dev.jsonl").read_text().splitlines()]
        curves.append(curve([r["n_ok"] for r in rows], 64))
        diag.append(
            json.loads((run / "summary.json").read_text())["eval"]["diag"]["solved"] / len(rows)
        )
    if not curves:
        return {}
    pts = [[c[0][0], round(sum(x[1] for x in c) / len(c), 4)] for c in zip(*curves, strict=True)]
    marker = {"level": 0, "k": 64, "y": round(sum(diag) / len(diag), 4)}
    lines = [{"level": 8, "depth": 6, "pts": pts}]
    return {"dev": {"K": 64, "n": 1108, "lines": lines, "marker": marker}}


def run198_dev():
    """Run 198 seed 0's own dev eval (old harness, 4 RL rounds) from the checkpoint repository."""
    cache = COCONUT / "run198_s0_eval_dev.jsonl"
    if not cache.exists():
        url = "https://huggingface.co/robbiethompson2018/nd-rl-checkpoints/resolve/main/runs/198-claude_lean-muon-no-rms-retest_s0/eval_dev.jsonl"
        subprocess.run(["curl", "-sSfL", "-o", str(cache), url], check=True)
    rows = [json.loads(line) for line in cache.read_text().splitlines()]
    pts = curve([r["n_ok"] for r in rows], 64)
    return {"dev": {"K": 64, "n": len(rows), "lines": [{"level": 0, "depth": 6, "pts": pts}]}}


def merged(*pool_dicts):
    out = {}
    for d in pool_dicts:
        out.update(d)
    return out


def stages(source, model):
    """{stage: {pool: curves}} for one model of a family."""
    tag = model["id"]
    if source == "loop":
        return {"final": dev_eval(tag)}
    if source == "coconut":
        return {
            "final": sweep(COCONUT / tag / "thought_sweep_final.json", "thoughts"),
            "pretrain": sweep(COCONUT / tag / "thought_sweep_pretrain.json", "thoughts"),
        }
    final = sweep(LOOPED / tag / "loop_sweep_final.json", "loops")
    # the whole dev pool was scored separately, at 2 and 8 loops only
    final.update(sweep(LOOPED / model["bar"] / "loop_sweep_final.json", "loops"))
    return {"final": final, "pretrain": sweep(LOOPED / tag / "loop_sweep_pretrain.json", "loops")}


def main():
    families = json.loads((HERE / "passk_models.json").read_text())["families"]
    for fam in families:
        for model in fam["models"]:
            model["stages"] = stages(fam["source"], model)
    ref = {
        "final": merged(sweep(LOOPED / "baseline/loop_sweep_final.json", "loops"), run198_dev()),
        "pretrain": sweep(LOOPED / "baseline/loop_sweep_pretrain.json", "loops"),
    }
    data = {"generated": time.strftime("%Y-%m-%d %H:%M %Z"), "families": families, "ref": ref}
    page = (HERE / "passk_template.html").read_text().replace("/*DATA*/", json.dumps(data))
    Path(sys.argv[-1]).write_text(page)
    print(f"wrote {sys.argv[-1]}, {len(page) // 1024} kB")


if __name__ == "__main__":
    main()
