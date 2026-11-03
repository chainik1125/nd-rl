"""E32 on the strict 106 and the RL / Stage-1 FLOP ratio R (PREREG.md, "Measurement"). Stdlib only.

E32 = sum over theorems of the unbiased pass@32 estimate 1 - C(n - c, 32) / C(n, 32) from a read's
`ok` arrays; `first32` = theorems solved by the first 32 attempts (Dan's single k 32 draw).
R = (6 x trained + 2 x sampled tokens) / (6 x Stage-1 tokens), over the GRPO stages' metrics.jsonl
rows (rolled-back iterations count), as high_rl_passk_20261030's `flop_ratio`.
"""

from __future__ import annotations

import json
import math
import pickle
from pathlib import Path

K = 32


def pass_at(n: int, c: int, k: int = K) -> float:
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def read_rows(evdir: str | Path) -> list[dict]:
    p = Path(evdir) / "rows_plain_B64.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def e32(rows: list[dict], k: int = K) -> dict:
    """{'e32', 'first32', 'n_theorems', 'solved_any', 'attempts', 'per': {name: [n, c, c_first32]}}."""  # noqa: E501
    tot = 0.0
    first = anyc = att = 0
    per = {}
    for r in rows:
        ok = [int(x) for x in r["ok"]]
        n, c = len(ok), sum(ok)
        c1 = sum(ok[:k])
        tot += pass_at(n, c, k)
        first += c1 > 0
        anyc += c > 0
        att += n
        per[r["name"]] = [n, c, c1]
    return {
        "e32": tot,
        "first32": first,
        "n_theorems": len(rows),
        "solved_any": anyc,
        "attempts": att,
        "per": per,
    }


def stage1_tokens(train_json: str | Path, pkl: str | Path) -> dict:
    """Tokens Stage-1 trained on: epochs x (state + completion tokens per epoch of the pickle)."""
    with open(pkl, "rb") as f:
        data = pickle.load(f)
    P = Q = N = 0
    for rec in data["records"]:
        for t in rec:
            P, Q, N = P + len(t[0]), Q + len(t[1]), N + 1
    t = json.loads(Path(train_json).read_text())
    ep = float(t["epochs"])
    return {
        "epochs": ep,
        "steps": t["steps"],
        "pairs_per_epoch": N,
        "state_tokens_per_epoch": P,
        "completion_tokens_per_epoch": Q,
        "tokens": ep * (P + Q),
        "n_params": t["n_params"],
        "secs": t["secs"],
    }


def rl_rows(run_dir: str | Path) -> list[dict]:
    """Per iteration of one GRPO stage: iter, sampled, trained, secs (metrics.jsonl order)."""
    d = Path(run_dir)
    a = json.loads((d / "args.json").read_text())
    per_it = int(a["prompts"]) * int(a["group"])
    out = []
    mp = d / "metrics.jsonl"
    if not mp.exists():
        return out
    for x in mp.read_text().splitlines():
        if not x.strip():
            continue
        r = json.loads(x)
        out.append(
            {
                "iter": int(r["iter"]),
                "sampled": float(r["tokens_per_attempt"]) * per_it,
                "trained": float(r["trained_tokens"]),
                "secs": float(r.get("secs", 0.0)),
                "attempts": per_it,
            }
        )
    return out


def cumulative(stages: list[str | Path]) -> list[dict]:
    """Rows over stages in order with cumulative c (iterations), sampled, trained, secs."""
    out, c, s, t, sec = [], 0, 0.0, 0.0, 0.0
    for st in stages:
        for r in rl_rows(st):
            c += 1
            s += r["sampled"]
            t += r["trained"]
            sec += r["secs"]
            out.append(
                {
                    "c": c,
                    "stage": str(st),
                    "iter": r["iter"],
                    "sampled": s,
                    "trained": t,
                    "secs": sec,
                }
            )
    return out


def ratio(row: dict, s1_tokens: float) -> float:
    return (6 * row["trained"] + 2 * row["sampled"]) / (6 * s1_tokens)


def grid(c_max: int, step: float = 0.05, round_every: int = 35) -> list[int]:
    """Cumulative iterations to snapshot: round(10^(step j)) and the attempt-matched Dan rounds."""
    g = {round(10 ** (step * j)) for j in range(0, int(math.log10(max(c_max, 2)) / step) + 2)}
    g |= set(range(round_every, c_max + 1, round_every))
    g |= {39, c_max}
    return sorted(x for x in g if 1 <= x <= c_max)
