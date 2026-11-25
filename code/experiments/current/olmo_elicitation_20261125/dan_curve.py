# ruff: noqa: E501, SIM115
"""Dan's olmo-ifcurve cells, re-declared independently of his code (pure stdlib + numpy).

His definitions live in `olmo_ifcurve/cells.py` (nd-takehome @ 8e744a89); `test_stage0.py` checks that the constants
below equal his. A cell is (checkpoint, prompt, arm). Its draws are the first N rows of the cell's source jobs, taken
in source order and then in (chunk, i) order within a job. Arm A: the 218 IFBench prompts, N = 16 at dpo, 4 at s0050 /
s0200, 8 elsewhere. Arm B (the 48 reach prompts): the first 32 base draws (no arm B at dpo: the set is defined by dpo's
draws).

Rows come from `artifacts/olmo_ifcurve/rows/<job>.csv` (one row per draw: job, pid, chunk, i, strict, ...), either his
fetched files or ones rebuilt from re-graded samples.
"""

from __future__ import annotations

import csv
from math import comb
from pathlib import Path

import numpy as np

DAN = Path("/scratch/cowifornia/nd-rl/olmo_elicit/ext/nd-takehome")
CKPTS = ["dpo", "s0050", "s0100", "s0200", "s0500", "s0750", "s1100", "s1600", "s2300"]
STEP = {c: (0 if c == "dpo" else int(c[1:])) for c in CKPTS}
OFF_LINE = {
    "s0200"
}  # olmo-weights review: steps 150 / 200 lie on an abandoned branch of the RL run
NEW = ["s0050", "s0200", "s0500", "s1100", "s1600"]
SOURCES = {
    "A": {
        "dpo": ["F_dpo_ifb"],
        "s0100": ["F_s0100_ifb"],
        "s0750": ["F_s0750_ifb"],
        "s2300": ["F_s2300_ifb"],
        "s0050": ["M_s0050"],
        "s0200": ["M_s0200"],
        "s1100": ["M_s1100"],
        "s1600": ["M_s1600"],
        "s0500": ["M_s0500", "T_s0500"],
    },
    "B": {
        "s0100": ["F_s0100_ifb", "B_s0100"],
        "s0750": ["T4_s0750", "T4b_s0750", "B_s0750"],
        "s2300": ["B_s2300"],
        **{c: [f"M_{c}"] for c in NEW},
    },
}
N_A = {"dpo": 16, "s0050": 4, "s0200": 4}  # else 8
N_B = 32
KS = (1, 2, 4, 8)


def pid_list(name: str, root: Path = DAN) -> list[str]:
    return [ln.strip() for ln in open(root / "olmo_ifcurve" / name) if ln.strip()]


def arm_a(root: Path = DAN) -> list[str]:
    return pid_list("pids_armA.txt", root)


def reach(root: Path = DAN) -> list[str]:
    return pid_list("pids_reach.txt", root)


class Rows:
    """Row caches of one tree: rows_dir/<job>.csv -> {pid: [row, ...]} sorted by (chunk, i)."""

    def __init__(self, rows_dir: Path):
        self.dir = Path(rows_dir)
        self._by_pid: dict[str, dict[str, list[dict]]] = {}

    def by_pid(self, job: str) -> dict[str, list[dict]]:
        if job not in self._by_pid:
            p = self.dir / f"{job}.csv"
            d: dict[str, list[dict]] = {}
            if p.exists():
                for r in csv.DictReader(open(p)):
                    for k in ("chunk", "i", "n_tok", "strict", "loose", "think_end", "ans_words"):
                        r[k] = int(r[k])
                    d.setdefault(r["pid"], []).append(r)
                for v in d.values():
                    v.sort(key=lambda x: (x["chunk"], x["i"]))
            self._by_pid[job] = d
        return self._by_pid[job]

    def draws(self, ckpt: str, pid: str, arm: str) -> list[dict]:
        n = N_A.get(ckpt, 8) if arm == "A" else N_B
        out: list[dict] = []
        for j in SOURCES[arm].get(ckpt, []):
            out.extend(self.by_pid(j).get(pid, []))
            if len(out) >= n:
                break
        return out[:n]


def passk(n: int, c: int, k: int) -> float:
    """Unbiased pass@k from c successes in n draws (nan when n < k)."""
    if n < k:
        return float("nan")
    return 1.0 if n - c < k else 1.0 - comb(n - c, k) / comb(n, k)


def passk_table(rows: Rows, prompts: list[str]) -> dict[str, dict[int, float]]:
    """{ckpt: {k: mean over prompts of pass@k}} on arm A (nan where draws < k)."""
    out = {}
    for c in CKPTS:
        nc = [
            (len(d), sum(x["strict"] for x in d)) for d in (rows.draws(c, p, "A") for p in prompts)
        ]
        out[c] = {k: float(np.mean([passk(n, s, k) for n, s in nc])) for k in KS}
    return out


def reach_table(rows: Rows, prompts: list[str]) -> dict[str, dict]:
    """{ckpt: {n_prompts, draws_min, solved (>= 1 strict in the 32 base draws), strict_rate}} on arm B."""
    out = {}
    for c in CKPTS[1:]:
        cells = [rows.draws(c, p, "B") for p in prompts]
        out[c] = {
            "n_prompts": len(prompts),
            "draws_min": min(len(d) for d in cells),
            "solved": sum(1 for d in cells if any(x["strict"] for x in d)),
            "strict_rate": float(np.mean([np.mean([x["strict"] for x in d]) for d in cells])),
        }
    return out


def dan_passk(
    path: Path = DAN / "olmo_ifcurve" / "passk.csv",
) -> dict[str, dict[int, tuple[float, float, float]]]:
    """His passk.csv, group 'all': {ckpt: {k: (pass@k, lo95, hi95)}} (nan where empty)."""
    out: dict[str, dict[int, tuple[float, float, float]]] = {}
    for r in csv.DictReader(open(path)):
        if r["group"] != "all":
            continue

        def f(x: str) -> float:
            return float(x) if x else float("nan")

        out.setdefault(r["ckpt"], {})[int(r["k"])] = (f(r["pass_at_k"]), f(r["lo95"]), f(r["hi95"]))
    return out


def dan_reach(path: Path = DAN / "olmo_ifcurve" / "results.csv") -> dict[str, dict]:
    """His results.csv arm-B rows: {ckpt: {solved_strict, genuine_established, strict_rate, rate_est_mean}}."""
    acc: dict[str, list[dict]] = {}
    for r in csv.DictReader(open(path)):
        if r["arm"] == "B":
            acc.setdefault(r["ckpt"], []).append(r)
    out = {}
    for c, rs in acc.items():
        out[c] = {
            "n_prompts": len(rs),
            "solved_strict": sum(1 for r in rs if int(r["strict"]) > 0),
            "genuine_established": sum(1 for r in rs if int(r["genuine_labelled"] or 0) > 0),
            "strict_rate": float(np.mean([int(r["strict"]) / int(r["n"]) for r in rs])),
            "rate_est_mean": float(
                np.mean([float(r["rate_est"]) for r in rs if r["rate_est"] != ""])
            ),
        }
    return out


def dan_first_genuine(path: Path = DAN / "olmo_ifcurve" / "emergence.csv") -> dict[str, int]:
    """Cumulative count of reach prompts whose first genuine success (F1, 32 base draws) is at or before each step."""
    f1 = [int(r["F1_step"]) for r in csv.DictReader(open(path))]
    return {c: sum(1 for s in f1 if s <= STEP[c]) for c in CKPTS}


def boot_mean_ci(values: np.ndarray, reps: int = 10000, seed: int = 0) -> tuple[float, float]:
    """95 % percentile interval of the mean, resampling rows (prompts)."""
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(reps, len(v)))
    m = v[idx].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))
