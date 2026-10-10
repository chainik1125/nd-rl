"""Pre-flight for the GRPO v2 lines: the curriculum pools exist, have their lengths, and share no
theorem with any evaluation set or with each other.

    python -m code.experiments.current.thinking_traces_20261005.grpo_v2.preflight POOL_L2 POOL_L3

Overlap is checked by normalised prompt and, where both sides carry one, by renaming-class key
(`key` / `pkey`, as the pool builders write them). Exits 1 on any overlap or missing file.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

R = Path("/scratch/cowifornia/nd-rl/think/runs")
TB = Path("/scratch/cowifornia/nd-rl/think/nd-takehome/data/eval_only/textbook72")
CH = R / "pools" / "charles_1002"
CURR = R / "pools" / "curriculum"
EVAL_SETS: dict[str, list[Path]] = {
    "test_13_20 (sealed)": [R / "pools/test_13_20/test_13_20.jsonl"],
    "test_21_30 (sealed)": [R / "pools/test_21_30/test_21_30.jsonl"],
    "hard12 test": [R / "restart2_hard12/hard12_test_1k.jsonl"],
    "hard712 test_1k / dev_500": [
        R / "pools/hard712/test_1k.jsonl",
        R / "pools/hard712/dev_500.jsonl",
    ],
    "textbook72": [TB / "textbook_dev.jsonl", TB / "textbook_train.jsonl"],
    "charles173": [CH / "candidate_v0.jsonl", CH / "candidate_v1.jsonl", CH / "batch3.jsonl"],
    "hand_proved_14": [CH / "hand_proved_14.jsonl"],
    "curriculum L2 test_1k": [CURR / "L2/test_1k.jsonl"],
    "curriculum L3 test_1k": [CURR / "L3/test_1k.jsonl"],
}
BANDS = {"L2": (10, 12), "L3": (13, 15)}


def rows(paths: Iterable[Path]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for p in paths:
        with open(p) as f:
            out += [json.loads(x) for x in f if x.strip()]
    return out


def norm(p: str) -> str:
    return " ".join(p.split())


def keys(rs: list[dict[str, Any]]) -> set[str]:
    return {str(r[k]) for r in rs for k in ("key", "pkey") if r.get(k)}


def main(argv: list[str]) -> int:
    pools = {"L2": Path(argv[0]), "L3": Path(argv[1])}
    bad = 0
    loaded: dict[str, list[dict[str, Any]]] = {}
    for name, p in pools.items():
        if not p.exists():
            print(f"MISSING pool {name}: {p}")
            return 1
        rs = rows([p])
        loaded[name] = rs
        raw = [r.get("min_lines", r.get("n_lines")) for r in rs]
        ls = [x for x in raw if isinstance(x, int)]
        lo, hi = BANDS[name]
        out_band = len(raw) - sum(lo <= x <= hi for x in ls)
        span = f"{min(ls)}-{max(ls)}" if ls else "none"
        print(f"pool {name}: {len(rs)} rows, lengths {span} (band {lo}-{hi})")
        if out_band:
            print(f"  BAD: {out_band} rows outside {lo}-{hi}")
            bad += 1
        if len(rs) != 5000:
            print(f"  NOTE: {len(rs)} rows, not 5,000 (one EI round samples every row k times)")
    pp = {n: {norm(r["prompt"]) for r in rs} for n, rs in loaded.items()}
    pk = {n: keys(rs) for n, rs in loaded.items()}
    if pp["L2"] & pp["L3"] or pk["L2"] & pk["L3"]:
        print(
            f"BAD: L2 and L3 share {len(pp['L2'] & pp['L3'])} prompts, "
            f"{len(pk['L2'] & pk['L3'])} keys"
        )
        bad += 1
    for ename, paths in EVAL_SETS.items():
        have = [p for p in paths if p.exists()]
        if not have:
            print(f"(skip {ename}: not built)")
            continue
        es = rows(have)
        ep, ek = {norm(r["prompt"]) for r in es}, keys(es)
        for n in loaded:
            a, b = len(pp[n] & ep), len(pk[n] & ek)
            flag = "BAD" if a or b else "ok "
            print(f"{flag} {n} vs {ename} ({len(es)}): {a} prompts, {b} keys in common")
            bad += bool(a or b)
    print("PREFLIGHT", "FAILED" if bad else "PASSED")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
