"""The smallest classical-only cap at which hard712's mix gate fills a band.

    python -m code.experiments.current.thinking_traces_20261005.curriculum.find_cap \
        .../curriculum/L3 6000 ['{"short78": 1.0}']

`run.py select --n N` keeps hard712's caps, relaxes them in its fixed order when the reservoir
cannot fill N, and stops classical-only at 25% (ex falso is never relaxed). At 13-15 lines about
half the rows are classical-only and 13% ex falso, so N may not fill. This runs the
(deterministic) selection with the classical-only cap fixed (binary search from 25%) and, if even
an uncapped classical share cannot fill N, first finds the smallest ex-falso cap (from 10%) with
classical uncapped, then the smallest classical cap given it. It leaves the band's
`work/selected.jsonl` and `select_report.json` from the final caps.
"""

from __future__ import annotations

import json
import subprocess
import sys
from typing import Any

PY = "/scratch/cowifornia/nd-rl/think/nd-rl/code/.venv/bin/python"
X = "code.experiments.current.thinking_traces_20261005.hard712.run"


def select(out: str, n: int, extra: dict[str, float], cap: float | None) -> tuple[int, Any]:
    caps = dict(extra)
    if cap is not None:
        caps["classical"] = cap
    cmd = [PY, "-m", X, "--out", out, "select", "--n", str(n), "--workers", "12"]
    if caps:
        cmd += ["--caps", json.dumps(caps)]
    subprocess.run(cmd, check=True, capture_output=True)
    with open(f"{out}/work/select_report.json") as f:
        rep = json.load(f)
    return rep["n_selected"], rep


def smallest(out: str, n: int, extra: dict[str, float], key: str, lo: int) -> int:
    """Smallest cap (in %) for `key` at which the selection fills n, given `extra`."""
    hi = 100
    while lo < hi:
        mid = (lo + hi) // 2
        k, _ = select(out, n, {**extra, key: mid / 100}, None)
        print(f"{key} cap {mid / 100:.2f}: {k}", flush=True)
        if k >= n:
            hi = mid
        else:
            lo = mid + 1
    return lo


def main() -> None:
    out, n = sys.argv[1], int(sys.argv[2])
    extra = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
    k, rep = select(out, n, extra, None)
    print(f"default caps: {k}", flush=True)
    if k >= n:
        return
    if select(out, n, {**extra, "classical": 1.0}, None)[0] >= n:
        extra = {**extra, "classical": smallest(out, n, extra, "classical", 25) / 100}
    else:  # classical-only alone cannot fill n: lift it, then ex falso (hard712 never relaxes it)
        extra = {**extra, "classical": 1.0}
        extra["ex_falso"] = smallest(out, n, extra, "ex_falso", 10) / 100
        extra["classical"] = smallest(out, n, extra, "classical", 25) / 100
    k, rep = select(out, n, extra, None)
    print(f"chosen caps {json.dumps(extra)}: {k}; {json.dumps(rep['shares_selected'])}")


if __name__ == "__main__":
    main()
