#!/usr/bin/env python3
"""Build the pass@k-by-budget page: one chart per model, one line per test-time loop (or thought)
count, from the per-problem success counts the spike sweeps saved.

  python passk_page.py OUT.html

Looped models: artifacts/looped_spike_20261006/<arm>/loop_sweep_*.json (loop_sweep.py).
Continuous-thought models: artifacts/coconut_spike_20261006/<arm>/thought_sweep_final.json
(the Coconut spike's copy of the same sweep). Run 198, the untied model both are built on, is the
reference curve. The page is passk_template.html with the data spliced in.
"""

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ART = HERE.parents[3] / "artifacts"
LOOPED, THOUGHTS = ART / "looped_spike_20261006", ART / "coconut_spike_20261006"
POOLS = ("dev_long", "textbook72", "rl_unsolved")


def sweep(path, knob):
    """pool -> {k, n, lines: [{level, depth, solved, counts}]} for one checkpoint's sweep."""
    out = {}
    for r in json.loads(Path(path).read_text())["results"]:
        if r["pool"] not in POOLS:
            continue
        pool = out.setdefault(r["pool"], {"k": r["k"], "n": r["n"], "lines": []})
        pool["lines"].append(
            {
                "level": r[knob],
                "depth": r.get("depth"),
                "solved": r["solved"],
                "counts": sorted((p["n_ok"] for p in r["problems"].values()), reverse=True),
            }
        )
    return out


def main():
    ref4 = sweep(LOOPED / "baseline/loop_sweep_final.json", "loops")
    ref12 = sweep(LOOPED / "baseline/more_rl/loop_sweep_final.json", "loops")
    ref0 = sweep(LOOPED / "baseline/loop_sweep_pretrain.json", "loops")
    looped = [
        (
            "loops1234/loop_sweep_final.json",
            "Looped, trained on 1 to 4 loops",
            "4 RL rounds · harness score 644",
            [1, 2, 4, 8, 12],
            ref4,
        ),
        (
            "loops1234/more_rl/loop_sweep_final.json",
            "Looped, trained on 1 to 4 loops, more RL",
            "12 RL rounds · dev score 690",
            [1, 2, 4, 8, 12],
            ref12,
        ),
        (
            "loops12345678/loop_sweep_final.json",
            "Looped, trained on 1 to 8 loops",
            "4 RL rounds · harness score 602",
            [1, 2, 4, 8, 16],
            ref4,
        ),
        (
            "loops1234681216/loop_sweep_final.json",
            "Looped, trained on 1 to 16 loops",
            "4 RL rounds · harness score 593",
            [1, 2, 4, 16, 24],
            ref4,
        ),
        (
            "loops2/loop_sweep_final.json",
            "Looped, trained at a fixed 2 loops",
            "4 RL rounds · harness score 666",
            [1, 2, 3, 4, 8],
            ref4,
        ),
        (
            "loops1234/loop_sweep_pretrain.json",
            "Looped, trained on 1 to 4 loops, before RL",
            "pretraining only",
            [1, 2, 4, 8, 12],
            ref0,
        ),
    ]
    thoughts = [
        ("none", "Thoughts, proof loss only", "harness score 617"),
        ("pause", "Pause tokens (no feedback, control)", "harness score 610"),
        ("curriculum", "Thoughts, Coconut curriculum", "harness score 601"),
        ("probe", "Thoughts, rule probe on each thought", "harness score 516"),
        ("future", "Thoughts predict the future state", "harness score 519"),
        ("codi", "Thoughts, self-distillation", "harness score 447"),
    ]
    models = [
        {
            "family": "looped",
            "knob": "loops",
            "title": t,
            "sub": s,
            "show": show,
            "pools": sweep(LOOPED / p, "loops"),
            "ref": ref,
        }
        for p, t, s, show, ref in looped
    ]
    models += [
        {
            "family": "thoughts",
            "knob": "thoughts",
            "title": t,
            "sub": "4 RL rounds · " + s,
            "show": [0, 2, 8, 16, 32],
            "pools": sweep(THOUGHTS / arm / "thought_sweep_final.json", "thoughts"),
            "ref": ref4,
        }
        for arm, t, s in thoughts
    ]
    data = {"generated": time.strftime("%Y-%m-%d %H:%M %Z"), "models": models}
    page = (HERE / "passk_template.html").read_text().replace("/*DATA*/", json.dumps(data))
    Path(sys.argv[1]).write_text(page)
    print(f"wrote {sys.argv[1]}: {len(models)} models, {len(page) // 1024} KB")


if __name__ == "__main__":
    main()
