"""The largest selection a small band's reservoir fills at fixed mix caps (L4).

    python -m code.experiments.current.thinking_traces_20261005.curriculum.find_n \
        .../curriculum/L4 '{"classical": 0.4477, "ex_falso": 0.14, ...}' [WORKERS (8)]

`run.py select --n N` scales every cap to N and, when the reservoir cannot fill N, relaxes the
blocking caps in hard712's order and finally shrinks N. L2r took L3r's realized shares as caps
and filled 6,000 below them. L4's reservoir is far smaller than 6,000, so `--n 6000` binds no cap
at all, and a target near the reservoir size makes `select` relax reductio, nested negation and
nested negation without reuse (which are in its relax order) instead of shrinking. This finds
the largest N (binary search) at which `select` fills N without relaxing any of the given caps;
hard712's other caps (irrelevant atoms, (A>B)>C goals, ...) may relax as they did for L2r and
L3r. It leaves the band's `work/selected.jsonl` and `select_report.json` from that N.
"""

from __future__ import annotations

import json
import subprocess
import sys
from typing import Any

PY = "/scratch/cowifornia/nd-rl/think/nd-rl/code/.venv/bin/python"
X = "code.experiments.current.thinking_traces_20261005.hard712.run"


WORKERS = 8  # select's feature pool (serrano: at most 8 cores for CPU builds, 2026-10-16)


def select(out: str, n: int, caps: dict[str, float]) -> dict[str, Any]:
    cmd = [PY, "-m", X, "--out", out, "select", "--n", str(n), "--workers", str(WORKERS)]
    subprocess.run([*cmd, "--caps", json.dumps(caps)], check=True, capture_output=True)
    with open(f"{out}/work/select_report.json") as f:
        return json.load(f)


def fills(rep: dict[str, Any], n: int, caps: dict[str, float]) -> bool:
    relaxed = {k for k, _ in rep["caps_relaxed"]}
    return rep["n_selected"] >= n and not relaxed & set(caps)


def main() -> None:
    global WORKERS
    out, caps = sys.argv[1], json.loads(sys.argv[2])
    if len(sys.argv) > 3:
        WORKERS = int(sys.argv[3])
    with open(f"{out}/work/gated_lean.jsonl") as f:
        n_in = sum(1 for line in f if line.strip())
    lo, hi = 0, n_in  # fills(lo) holds (n = 0 trivially); find the largest n that fills
    while lo < hi:
        mid = (lo + hi + 1) // 2
        ok = fills(select(out, mid, caps), mid, caps)
        print(f"n {mid}: {'fills' if ok else 'does not fill'} at the fixed caps", flush=True)
        if ok:
            lo = mid
        else:
            hi = mid - 1
    rep = select(out, lo, caps)
    assert fills(rep, lo, caps), rep
    print(f"largest n {lo} of {n_in}; shares {json.dumps(rep['shares_selected'])}")
    print(f"relaxed {json.dumps(rep['caps_relaxed'])}")


if __name__ == "__main__":
    main()
