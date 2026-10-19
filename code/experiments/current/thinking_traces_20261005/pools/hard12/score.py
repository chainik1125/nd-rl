"""Band-score model for rejection sampling: P(a candidate lands in T0's hard band) from its
features.

    S=code.experiments.current.thinking_traces_20261005.pools.hard12.score
    python -m $S fit --sets short_pilot.jsonl short_cands.jsonl \
        --rows .../short_pilot/rows.jsonl .../short/rows.jsonl --out band_model.json
    python -m $S score --model band_model.json --in cands.jsonl --out cands_scored.jsonl \
        [--workers 12]

`fit`: logistic regression (numpy IRLS, small ridge) of band membership (1-8 successes of k = 16
under T0_s0, the k 16 analogue of the 2-16-of-32 filter) on the features of `features.py` plus
length dummies, on the previous agent's measured 7-12-line candidates with formula depth <= 5
(deeper formulas fail on format, see analyze.py). An 80/20 split (seeded) reports the held-out AUC
and the band rate by score quintile; the final model is refit on all rows.

`score`: features + `band_score` (the model's probability) for every row of a candidate file, after
the shape gate (--lens, --max_fdepth, --max_prompt prompt tokens); rows failing the gate are dropped
and counted.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

from code.thinking.data import read_jsonl

from .features import features

FEATS = [
    "has_ORE",
    "has_NEGI",
    "has_DN",
    "raa",
    "has_BOTE",
    "has_NEGE",
    "has_ANDI",
    "has_ORI",
    "has_IMPE",
    "has_ANDE",
    "has_R",
    "classical_only",
    "n_redundant_prem",
    "prem_inconsistent",
    "n_disj_prem",
    "n_imp_hard_ante",
    "fdepth",
    "n_atoms",
    "box_depth",
    "top_goal_directed",
    "first_kind=as_case",
    "first_kind=as_other",
    "first_kind=elim",
    "goal=atom",
    "goal=or",
    "goal=and",
    "goal=not",
    "n_prem=0",
    "n_prem=3",
    "n_prem_or>=2",
    "n_prem_imp>=2",
    "n_boxes<=1",
    "n_boxes",
    "prompt_tokens/10",
]
LENS = list(range(7, 13))


def vec(f: dict[str, Any]) -> list[float]:
    r = [float(f["n_lines"] == L) for L in LENS]
    for g in FEATS:
        if g.startswith("first_kind="):
            r.append(float(f["first_kind"] == g.split("=")[1]))
        elif g.startswith("goal="):
            r.append(float(f["goal_type"] == g.split("=")[1]))
        elif g.startswith("n_prem="):
            r.append(float(f["n_prem"] == int(g.split("=")[1])))
        elif g == "n_prem_or>=2":
            r.append(float(f["n_prem_or"] >= 2))
        elif g == "n_prem_imp>=2":
            r.append(float(f["n_prem_imp"] >= 2))
        elif g == "n_boxes<=1":
            r.append(float(f["n_boxes"] <= 1))
        elif g == "prompt_tokens/10":
            r.append(f["prompt_tokens"] / 10)
        else:
            r.append(float(f[g]))
    return r


def logit_fit(X: np.ndarray, y: np.ndarray, ridge: float = 1.0) -> np.ndarray:
    b = np.zeros(X.shape[1])
    for _ in range(100):
        p = 1 / (1 + np.exp(-(X @ b)))
        w = p * (1 - p) + 1e-9
        g = X.T @ (y - p) - ridge * b
        H = (X.T * w) @ X + ridge * np.eye(X.shape[1])
        step = np.linalg.solve(H, g)
        b += step
        if np.max(np.abs(step)) < 1e-8:
            break
    return b


def auc(s: np.ndarray, y: np.ndarray) -> float:
    o = np.argsort(s)
    r = np.empty(len(s))
    r[o] = np.arange(1, len(s) + 1)
    n1 = y.sum()
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def cmd_fit(a: argparse.Namespace) -> None:
    rows = {r["name"]: r for p in a.rows for r in read_jsonl(p)}
    data = []
    for p in a.sets:
        for r in read_jsonl(p):
            if r["name"] not in rows or not 7 <= r["n_lines"] <= 12:
                continue
            f = features(r["prompt"], r["proof"])
            if f["fdepth"] > a.max_fdepth:
                continue
            row = rows[r["name"]]
            half = row["k"] // 2
            data.append((vec(f), int(1 <= row["n_ok"] <= half), row["n_ok"] / row["k"]))
    rng = random.Random(0)
    rng.shuffle(data)
    X = np.array([d[0] for d in data])
    y = np.array([d[1] for d in data], float)
    p1 = np.array([d[2] for d in data])
    n_tr = int(0.8 * len(data))
    b = logit_fit(X[:n_tr], y[:n_tr])
    s = 1 / (1 + np.exp(-(X[n_tr:] @ b)))
    yt = y[n_tr:]
    q = np.quantile(s, [0.2, 0.4, 0.6, 0.8])
    quint = np.searchsorted(q, s)
    by_q = {
        f"q{i + 1}": {
            "n": int((quint == i).sum()),
            "band_pct": round(100 * float(yt[quint == i].mean()), 1),
            "pass1_pct": round(100 * float(p1[n_tr:][quint == i].mean()), 1),
        }
        for i in range(5)
    }
    top = np.argsort(-s)
    top_rates = {
        f"top{pct}%": round(100 * float(yt[top[: max(1, len(s) * pct // 100)]].mean()), 1)
        for pct in (10, 20, 30, 50)
    }
    b_all = logit_fit(X, y)
    cols = [f"len={L}" for L in LENS] + FEATS
    model = {
        "target": "band: 1..k/2 successes of k = 16 under T0_s0 (T 0.8, plain, Lean)",
        "data": {
            "sets": a.sets,
            "rows": a.rows,
            "n": len(data),
            "max_fdepth": a.max_fdepth,
            "band_rate_pct": round(100 * float(y.mean()), 1),
        },
        "heldout": {
            "n": len(yt),
            "auc": round(auc(s, yt), 3),
            "by_quintile": by_q,
            "top_rates_band_pct": top_rates,
        },
        "cols": cols,
        "coef": [round(float(x), 4) for x in b_all],
    }
    with open(a.out, "w") as f:
        json.dump(model, f, indent=1)
    print(json.dumps({"n": len(data), "heldout": model["heldout"]}, indent=1))
    for c, x in sorted(zip(cols, b_all, strict=True), key=lambda t: t[1]):
        print(f"  {c:22s} {x:+.3f}")


_M: dict[str, Any] = {}


def _score_one(line: str) -> tuple[str, str | None]:
    r = json.loads(line)
    a = _M["args"]
    if not a["lo"] <= r["n_lines"] <= a["hi"]:
        return "len", None
    try:
        f = features(r["prompt"], r["proof"])
    except Exception:  # noqa: BLE001
        return "features_failed", None
    if f["fdepth"] > a["max_fdepth"]:
        return "fdepth", None
    if f["prompt_tokens"] > a["max_prompt"]:
        return "prompt", None
    s = 1 / (1 + np.exp(-float(np.dot(_M["coef"], vec(f)))))
    r["feat"] = f
    r["band_score"] = round(float(s), 4)
    return "ok", json.dumps(r)


def _init(coef: list[float], args: dict[str, Any]) -> None:
    _M["coef"] = np.array(coef)
    _M["args"] = args


def cmd_score(a: argparse.Namespace) -> None:
    model = json.loads(Path(a.model).read_text())
    lo, hi = (int(x) for x in a.lens.split("-"))
    args = {"lo": lo, "hi": hi, "max_fdepth": a.max_fdepth, "max_prompt": a.max_prompt}
    st: dict[str, int] = {}
    with (
        open(a.inp) as fi,
        open(a.out, "w") as fo,
        mp.Pool(a.workers, _init, (model["coef"], args)) as pool,
    ):
        for tag, line in pool.imap(_score_one, fi, chunksize=256):
            st[tag] = st.get(tag, 0) + 1
            if line is not None:
                fo.write(line + "\n")
    print(json.dumps({"in": a.inp, "out": a.out, "stats": st}))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--sets", nargs="+", required=True)
    f.add_argument("--rows", nargs="+", required=True)
    f.add_argument("--max_fdepth", type=int, default=5)
    f.add_argument("--out", required=True)
    s = sub.add_parser("score")
    s.add_argument("--model", required=True)
    s.add_argument("--in", dest="inp", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--lens", default="7-12")
    s.add_argument("--max_fdepth", type=int, default=5)
    s.add_argument("--max_prompt", type=int, default=156)
    s.add_argument("--workers", type=int, default=12)
    a = ap.parse_args(argv)
    {"fit": cmd_fit, "score": cmd_score}[a.cmd](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
