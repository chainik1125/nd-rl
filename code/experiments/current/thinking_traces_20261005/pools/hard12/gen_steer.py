"""Steered candidate stream for hard12: Dan's generator with its existing knobs, plus band-score
rejection sampling.

    G=code.experiments.current.thinking_traces_20261005.pools.hard12.gen_steer
    python -m $G --configs fwd_strict_d2,takehome --model band_model.json --secs 600 --workers 12 \
        --seed 930000 --min_score 0.2 --out runs/pools/hard12_work/raw/steer.jsonl

The fork's `gen.Gen` is used unchanged (imported read-only, no bytecode written), with the same rule
set; only the knobs that `Gen` / `sample_one` already expose are set (CONFIGS below):
  * `max_prem` (premise cap, <= 3), `max_depth` (box-nesting cap), `fdepth` (random-formula depth),
  * `strict` (no lazy (Z > G) premise, no F in the theorem, no introduction-rule conclusion),
    `bot_p`, `ore_steps`, `ore_boxes`,
  * the mode mix (`Gen.forward` / `Gen.goal`) and their arguments (forward steps, initial premises,
    goal depth, reach budget), as `sample_one` draws them.
Each sample is kept if: the reference (dependency-pruned by `Gen.finish`) has --min..--max lines; no
premise's negation is also a premise (the ladder's contra_prem rule); `nd_verify.verify_text`
accepts it; formula depth <= --max_fdepth and prompt <= --max_prompt tokens; the renaming class
(`gen.canon_key`) is new to this worker; and its band score (`score.py` model) is >= --min_score
(rejection sampling on the task-1 features). Worker i of config c uses seed + 1000 * (i + workers *
c_index); rows carry `cfg`, `seed`, `feat`, `band_score`. <out>.summary.json gives per-config tries,
acceptance and timing.
"""

from __future__ import annotations

import argparse
import collections
import json
import multiprocessing as mp
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from .features import FORK, features
from .score import vec

# knob settings (all existing Gen / sample_one parameters). 'mode': 'sample_one' calls the fork's
# sample_one with long=<long>; 'forward' / 'goal' / 'mix' call Gen.forward / Gen.goal with the
# listed argument choices.
CONFIGS: dict[str, dict[str, Any]] = {
    # the previous pool's stream: the strict long generator (ladder rl_targets / transfer_long)
    "strict_long": {"gen": {"max_prem": 3, "max_depth": 3}, "mode": "sample_one", "long": True},
    # K12's own (take-home) generator
    "takehome": {"gen": {"max_prem": 3, "max_depth": 3}, "mode": "sample_one", "long": False},
    # strict forward mode, box nesting capped at 2, more initial premises
    "fwd_strict_d2": {
        "gen": {"max_prem": 3, "max_depth": 2},
        "strict": True,
        "ore_steps": 3,
        "mode": "forward",
        "steps": [6, 22],
        "n_prem": [1, 2, 2, 3, 3],
    },
    # strict forward mode, one box level (case splits / boxes cannot nest)
    "fwd_strict_d1": {
        "gen": {"max_prem": 3, "max_depth": 1},
        "strict": True,
        "ore_steps": 3,
        "mode": "forward",
        "steps": [6, 22],
        "n_prem": [1, 2, 2, 3, 3],
    },
    # take-home (non-strict) forward mode, box nesting capped at 2, ORE branches with 3 steps
    "fwd_takehome_d2": {
        "gen": {"max_prem": 3, "max_depth": 2},
        "strict": False,
        "ore_steps": 3,
        "mode": "forward",
        "steps": [4, 16],
        "n_prem": [1, 2, 2, 3, 3],
    },
    # strict goal mode, initial random premises, deeper reach budgets
    "goal_strict": {
        "gen": {"max_prem": 3, "max_depth": 3},
        "strict": True,
        "ore_steps": 3,
        "mode": "goal",
        "n_prem": [0, 1, 1, 2, 2],
        "gdepth": [2, 3, 3],
        "budgets": [3, 4, 5],
    },
    # take-home goal mode with the long reach budgets
    "goal_takehome": {
        "gen": {"max_prem": 3, "max_depth": 3},
        "strict": False,
        "ore_steps": 3,
        "mode": "goal",
        "n_prem": [0, 1, 1, 2, 2],
        "gdepth": [2, 3, 3],
        "budgets": [3, 4, 5],
    },
}


def _fork() -> tuple[Any, ...]:
    sys.dont_write_bytecode = True
    if FORK not in sys.path:
        sys.path.insert(0, FORK)
    from gen import Fail, Gen, canon_key, sample_one  # type: ignore[import-not-found]
    from nd_verify import verify_text  # type: ignore[import-not-found]

    return Gen, Fail, canon_key, sample_one, verify_text


def draw(cfg: dict[str, Any], g: Any, rng: random.Random, sample_one: Any) -> dict[str, Any]:
    if cfg["mode"] == "sample_one":
        return sample_one(g, rng, cfg["long"])
    g.strict = cfg["strict"]
    g.bot_p = 0.0 if g.strict else 0.02
    g.ore_steps = cfg["ore_steps"]
    g.ore_boxes = cfg.get("ore_boxes", False)
    if cfg["mode"] == "forward":
        return g.forward(rng.randint(*cfg["steps"]), rng.choice(cfg["n_prem"]))
    return g.goal(rng.choice(cfg["n_prem"]), rng.choice(cfg["gdepth"]), rng.choice(cfg["budgets"]))


def worker(job: tuple[str, int, float, dict[str, Any], str]) -> dict[str, Any]:
    name, seed, secs, a, out = job
    Gen, Fail, canon_key, sample_one, verify_text = _fork()
    cfg = CONFIGS[name]
    rng = random.Random(seed)
    g = Gen(rng, **cfg["gen"])
    coef = np.array(a["coef"])
    seen: set[str] = set()
    st: collections.Counter[str] = collections.Counter()
    t0 = time.time()
    with open(out, "w") as fo:
        while time.time() - t0 < secs:
            st["tries"] += 1
            try:
                r = draw(cfg, g, rng, sample_one)
            except (Fail, RecursionError):
                st["fail"] += 1
                continue
            if not a["min"] <= r["n_lines"] <= a["max"]:
                st["len_out"] += 1
                continue
            if r["contra_prem"]:
                st["contra"] += 1
                continue
            key = canon_key(r["thm"])
            if key in seen:
                st["dup"] += 1
                continue
            seen.add(key)
            try:
                f = features(r["prompt"], r["proof"])
            except Exception:  # noqa: BLE001
                st["features_failed"] += 1
                continue
            if f["fdepth"] > a["max_fdepth"] or f["prompt_tokens"] > a["max_prompt"]:
                st["shape_out"] += 1
                continue
            s = float(1 / (1 + np.exp(-float(np.dot(coef, vec(f))))))
            st["scored"] += 1
            if s < a["min_score"]:
                st["score_low"] += 1
                continue
            ok, _, nl = verify_text(r["prompt"] + " " + r["proof"])
            if not ok or nl != r["n_lines"]:
                st["verifier_reject"] += 1
                continue
            r.pop("text", None)
            st["kept"] += 1
            fo.write(
                json.dumps(
                    {
                        **r,
                        "key": key,
                        "cfg": name,
                        "seed": seed,
                        "feat": f,
                        "band_score": round(s, 4),
                    }
                )
                + "\n"
            )
    return {"cfg": name, "seed": seed, "stats": dict(st), "secs": round(time.time() - t0, 1)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--configs", required=True, help="comma-separated CONFIGS names")
    ap.add_argument("--model", required=True, help="score.py band model json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--secs", type=float, default=600, help="wall time per worker")
    ap.add_argument("--workers", type=int, default=12, help="per config")
    ap.add_argument("--seed", type=int, default=930000)
    ap.add_argument("--min", type=int, default=7)
    ap.add_argument("--max", type=int, default=12)
    ap.add_argument("--max_fdepth", type=int, default=5)
    ap.add_argument("--max_prompt", type=int, default=156)
    ap.add_argument("--min_score", type=float, default=0.0)
    a = ap.parse_args(argv)
    model = json.loads(Path(a.model).read_text())
    opts = {
        "coef": model["coef"],
        "min": a.min,
        "max": a.max,
        "max_fdepth": a.max_fdepth,
        "max_prompt": a.max_prompt,
        "min_score": a.min_score,
    }
    names = a.configs.split(",")
    jobs = [
        (n, a.seed + 1000 * (i + a.workers * ci), a.secs, opts, f"{a.out}.{n}.w{i}")
        for ci, n in enumerate(names)
        for i in range(a.workers)
    ]
    t0 = time.time()
    with mp.Pool(len(jobs)) as pool:
        res = pool.map(worker, jobs)
    seen: set[str] = set()
    per: collections.Counter[tuple[str, int]] = collections.Counter()
    dup = 0
    with open(a.out, "w") as fo:
        for job in jobs:
            with open(job[4]) as f:
                for line in f:
                    r = json.loads(line)
                    if r["key"] in seen:
                        dup += 1
                        continue
                    seen.add(r["key"])
                    per[(r["cfg"], r["n_lines"])] += 1
                    fo.write(line)
    agg: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    for r in res:
        agg[r["cfg"]].update(r["stats"])
    summary = {
        "args": vars(a),
        "configs": {n: CONFIGS[n] for n in names},
        "generator": "fork gen.Gen (4dcd0e4f), unchanged; knobs as in `configs`",
        "per_config": {n: dict(v) for n, v in agg.items()},
        "per_config_len": {f"{c}:{L}": n for (c, L), n in sorted(per.items())},
        "distinct_classes": len(seen),
        "cross_worker_dups": dup,
        "workers": res,
        "secs": round(time.time() - t0, 1),
    }
    with open(a.out + ".summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps({"distinct": len(seen), "per_config": summary["per_config"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
