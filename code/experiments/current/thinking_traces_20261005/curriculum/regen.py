"""Re-certify hard712 candidates that failed only a length gate, for one band.

    PFG_MIN_L=7 PFG_MAX_L=9 PFG_MAX_GAP=6 python -m \
        code.experiments.current.thinking_traces_20261005.curriculum.regen \
        --lo 7 --hi 9 --index .../curriculum/work/excl_index.pkl --out .../curriculum/L1/regen

hard712's T1 gate kept a class only when its constructed proof had at most one line more than
the shortest proof (`pfgates.MAX_GAP`). 17,376 classes failed that check alone (status `gap`
in `hard712/work/rejects.jsonl`: shortest length known, structure and necessity never checked);
11,545 of them have a shortest proof of 7-9 lines. A rejected row stores its seed, and
`pfgen.sample(seed, weights)` is deterministic, so each candidate is rebuilt with the recipe
weights its seed was drawn under (stream 1, seeds < 30M: `pfgen.WEIGHTS`; stream 2, 30-42M:
`work/weights_stream2.json`; stream 3, 42-120M: the per-round weights in `work/progress.json`),
checked against the stored class key, run through the shape gate, the exclusion index and
`pfgates.certify` with the gap limit raised (`PFG_MAX_GAP`), and written in `run.py generate`'s
format (`OUT/work/gated.jsonl`, `rejects.jsonl`) so that `run.py lean` can follow.

With `--status long` the same is done for the 4,847 classes whose shortest proof was longer
than hard712's 12 lines (their constructed proofs have at most 13 lines, so all are 13-line
problems); they need no gap relaxation.

The raised gap limit changes nothing a row shows: `min_lines`, the reference proof (the shortest
proof found), the families and the necessity checks all come from the shortest proof. Only
`gen_lines` / `minlen_bound` (the constructed proof's length) can now exceed `min_lines + 1`.
"""

from __future__ import annotations

import argparse
import bisect
import collections
import json
import multiprocessing as mp
import pickle
import time
from pathlib import Path
from typing import Any

from code.experiments.current.thinking_traces_20261005.hard712 import pfgates as GT
from code.experiments.current.thinking_traces_20261005.hard712 import pfgen as PG
from code.experiments.current.thinking_traces_20261005.hard712 import run as RN

H712 = Path("/scratch/cowifornia/nd-rl/think/runs/pools/hard712/work")


def weights_for(seed: int, s2: dict[str, float], rw: dict[str, dict[str, float]]) -> Any:
    if seed < 30_000_000:
        return None
    if seed < 42_000_000:
        return s2
    s0 = 42_000_000 + (seed - 42_000_000) // 2_000_000 * 2_000_000
    return rw[str(s0)]


def weights_from_rounds(seed: int, starts: list[int], rw: dict[str, dict[str, float]]) -> Any:
    """Another `run.py generate` run's weights for a seed: those of the last recorded round that
    starts at or before it (pfgen's defaults before the first). Seeds of rounds a restart cut
    short may get the wrong table; `rebuild` then rejects them on the class check."""
    i = bisect.bisect_right(starts, seed)
    return rw[str(starts[i - 1])] if i else None


def rebuild(job: tuple[dict[str, Any], Any]) -> dict[str, Any]:
    rej, w = job
    c = PG.sample(rej["seed"], weights=w)
    if c is None:
        return dict(rej, status="rebuild_none")
    if c["recipe"] != rej["recipe"]:
        return dict(rej, status="rebuild_recipe")
    reason, info = GT.shape_gate(c["prompt"], c["proof"])
    if reason:
        return dict(rej, status="shape:" + reason)
    c.update(info)
    k, pk = RN.class_keys(c["prompt"])
    if pk != rej["pkey"]:
        return dict(rej, status="rebuild_class")
    c["key"], c["pkey"] = k, pk
    return c


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lo", type=int, required=True)
    ap.add_argument("--hi", type=int, required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument(
        "--source",
        default="",
        help="another run.py generate output dir whose rejects to re-certify (default: hard712)",
    )
    ap.add_argument(
        "--status",
        nargs="+",
        default=["gap"],
        help="hard712 reject statuses to re-certify: gap (construction gap) and/or long "
        "(shortest proof > 12 lines)",
    )
    a = ap.parse_args()
    assert (a.lo, a.hi) == (GT.MIN_L, GT.MAX_L), "set PFG_MIN_L / PFG_MAX_L to the band"
    work = Path(a.out) / "work"
    work.mkdir(parents=True, exist_ok=True)
    src = Path(a.source) / "work" if a.source else H712
    rw = json.loads((src / "progress.json").read_text()).get("round_weights", {})
    if a.source:
        starts = sorted(int(k) for k in rw)

        def wfor(seed: int) -> Any:
            return weights_from_rounds(seed, starts, rw)
    else:
        s2 = json.loads((H712 / "weights_stream2.json").read_text())

        def wfor(seed: int) -> Any:
            return weights_for(seed, s2, rw)

    rej = []
    with open(src / "rejects.jsonl") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("status") in a.status and a.lo <= (r.get("min_lines") or 0) <= a.hi:
                rej.append(r)
    with open(a.index, "rb") as f:
        idx = pickle.load(f)
    c: collections.Counter[str] = collections.Counter(candidates=len(rej))
    done: set[str] = set()  # resume: candidates already written by an earlier run into OUT
    for name in ("gated.jsonl", "rejects.jsonl"):
        if (work / name).exists():
            rows, _ = RN.read_complete(work / name)
            done |= {r["pkey"] for r in rows if r.get("pkey")}
    if done:
        c["resumed_skip"] = sum(r["pkey"] in done for r in rej)
        rej = [r for r in rej if r["pkey"] not in done]
    t0 = time.time()
    jobs = [(r, wfor(r["seed"])) for r in rej]
    with (
        mp.get_context("spawn").Pool(
            a.workers, initializer=RN._init, initargs=(RN.MEM_GB,), maxtasksperchild=200
        ) as pool,
        open(work / "gated.jsonl", "a") as fg,
        open(work / "rejects.jsonl", "a") as fr,
    ):
        cands = []
        for x in pool.imap(rebuild, jobs, chunksize=50):
            if "prompt" not in x:
                c[x["status"]] += 1
                fr.write(json.dumps(x) + "\n")
                continue
            h = idx["keys"].get(x["key"], set()) | idx["pkeys"].get(x["pkey"], set())
            if h:
                c["excluded"] += 1
                fr.write(json.dumps(dict(pkey=x["pkey"], status="excluded", seed=x["seed"])) + "\n")
                continue
            cands.append(x)
        c["rebuilt"] = len(cands)
        RN.log(f"rebuilt {len(cands)} of {len(rej)} in {time.time() - t0:.0f}s: {dict(c)}")
        for i, r in enumerate(pool.imap(RN.certify_safe, cands, chunksize=2), 1):
            c["cert:" + r["status"]] += 1
            if r["status"] == "ok":
                fg.write(json.dumps(r) + "\n")
            else:
                keep = ("pkey", "status", "recipe", "seed", "min_lines", "gen_lines")
                fr.write(json.dumps({k: r.get(k) for k in keep}) + "\n")
            if i % 100 == 0:
                fg.flush()
                fr.flush()
                RN.log(f"certified {i}/{len(cands)} ok {c['cert:ok']} {time.time() - t0:.0f}s")
    c["secs"] = round(time.time() - t0)
    (work / "regen_report.json").write_text(json.dumps(dict(c), indent=1))
    RN.log(f"regen done: {dict(c)}")


if __name__ == "__main__":
    main()
