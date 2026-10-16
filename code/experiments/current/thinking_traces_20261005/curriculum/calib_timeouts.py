"""Calibrate the gate time limits on a new machine against serrano's recorded timeouts.

(ynez rebuild, 2026-10-13)

    PYTHONHASHSEED=0 python -m \
        code.experiments.current.thinking_traces_20261005.curriculum.calib_timeouts \
        --backup /home/cowifornia/backups/nd-rl-misc/hard712_20261008 --out calib.json --workers 18

serrano ran hard712 and the curriculum with 20 s limits on the shortest-proof search and on each
necessity re-search (`pfgates.SEARCH_SECS`, `NEC_SECS`). hard712's gated rows store every
necessity check's verdict and seconds; almost all slow checks are `case_split` (352 of 6,093 timed
out on serrano, 927 more took 10-20 s). This re-runs, with a 20 s cap and timing:

* necessity: a random sample of the serrano timeouts, of the 10-20 s checks and of the faster ones
  (candidate rebuilt from its seed with the recipe weights it was drawn under, as `regen.py` does;
  `extra` = constructed + stored reference proof formulas, as `pfgates.certify` passes them);
* search: every hard712 `search_timeout` reject (73) and the spare rows whose recorded
  `t_search` was >= 5 s.

For each candidate limit T it prints the estimated ynez timeout rate (the sampled strata scaled to
their full sizes) next to serrano's at 20 s, and the per-check time ratio ynez / serrano.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import random
import statistics
import time
from typing import Any

from code.experiments.current.thinking_traces_20261005.curriculum import regen as RG
from code.experiments.current.thinking_traces_20261005.hard712 import pfgates as GT
from code.experiments.current.thinking_traces_20261005.hard712 import pfgen as PG
from code.experiments.current.thinking_traces_20261005.hard712 import run as RN

CAP = 20.0


def jl(p: str) -> list[dict[str, Any]]:
    with open(p) as f:
        return [json.loads(x) for x in f if x.strip()]


def rebuild(seed: int, w: Any, pkey: str) -> dict[str, Any] | None:
    c = PG.sample(seed, weights=w)
    if c is None:
        return None
    reason, info = GT.shape_gate(c["prompt"], c["proof"])
    if reason:
        return None
    c.update(info)
    if RN.class_keys(c["prompt"])[1] != pkey:
        return None
    return c


def job(x: tuple[str, dict[str, Any], Any]) -> dict[str, Any]:
    kind, r, w = x
    c = rebuild(r["seed"], w, r["pkey"])
    out = dict(kind=kind, seed=r["seed"], stratum=r["_stratum"], serrano=r["_serrano"])
    if c is None:
        return dict(out, status="rebuild_failed")
    t0 = time.time()
    if kind == "nec":
        extra = tuple(ln["f"] for ln in GT.GC.parse_proof(c["proof"])) + tuple(
            ln["f"] for ln in GT.GC.parse_proof(r["reference_proof"])
        )
        n = GT.necessity(c["prompt"], GT.NEC_KIND["case_split"], r["min_lines"], extra, secs=CAP)
        return dict(out, status=n["verdict"], secs=round(time.time() - t0, 3))
    s = GT.shortest(c["prompt"], c["proof"], c["gen_lines"], secs=CAP)
    st = "timeout" if s.get("timeout") else "done"
    return dict(out, status=st, secs=round(time.time() - t0, 3))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backup", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=18)
    ap.add_argument("--n_tout", type=int, default=120)
    ap.add_argument("--n_slow", type=int, default=240)
    ap.add_argument("--n_fast", type=int, default=60)
    a = ap.parse_args()
    B = a.backup
    with open(f"{B}/work/weights_stream2.json") as f:
        s2 = json.load(f)
    with open(f"{B}/work/progress.json") as f:
        rw = json.load(f)["round_weights"]
    rows = [
        dict(r, proof=r["gen_proof"])
        for p in ("pool", "test_1k", "dev_500")
        for r in jl(f"{B}/{p}.jsonl")
    ]
    spare = jl(f"{B}/work/spare.jsonl")
    rows += spare
    cs = [r for r in rows if "case_split" in (r.get("necessity") or {})]
    strata: dict[str, list[dict[str, Any]]] = {"tout": [], "slow": [], "fast": []}
    for r in cs:
        v = r["necessity"]["case_split"]
        s = "tout" if v["verdict"] == "timeout" else ("slow" if (v["secs"] or 0) >= 10 else "fast")
        strata[s].append(dict(r, _stratum=s, _serrano=[v["verdict"], v["secs"]]))
    sizes = {k: len(v) for k, v in strata.items()}
    rng = random.Random(0)
    pick = (
        rng.sample(strata["tout"], min(a.n_tout, sizes["tout"]))
        + rng.sample(strata["slow"], min(a.n_slow, sizes["slow"]))
        + rng.sample(strata["fast"], min(a.n_fast, sizes["fast"]))
    )
    jobs = [("nec", r, RG.weights_for(r["seed"], s2, rw)) for r in pick]
    rej = [r for r in jl(f"{B}/work/rejects.jsonl") if r["status"] == "search_timeout"]
    sslow = [r for r in spare if (r.get("t_search") or 0) >= 5]
    n_cert = len(rows) + sum(1 for r in jl(f"{B}/work/rejects.jsonl") if r["status"] != "excluded")
    for r in rej:
        jobs.append(
            (
                "search",
                dict(r, _stratum="tout", _serrano=["timeout", CAP]),
                RG.weights_for(r["seed"], s2, rw),
            )
        )
    for r in sslow:
        jobs.append(
            (
                "search",
                dict(r, _stratum="slow", _serrano=["done", r["t_search"]]),
                RG.weights_for(r["seed"], s2, rw),
            )
        )
    t0 = time.time()
    with mp.get_context("spawn").Pool(a.workers, initializer=RN._init, initargs=(3.0,)) as pool:
        res = pool.map(job, jobs, chunksize=1)
    rep: dict[str, Any] = dict(
        wall=round(time.time() - t0),
        strata_sizes=sizes,
        n_case_split_checks=len(cs),
        n_certified=n_cert,
        n_search_timeout_rejects=len(rej),
        n_search_slow_spare=len(sslow),
    )
    nec = [r for r in res if r["kind"] == "nec" and r["status"] != "rebuild_failed"]
    srch = [r for r in res if r["kind"] == "search" and r["status"] != "rebuild_failed"]
    rep["rebuild_failed"] = sum(r["status"] == "rebuild_failed" for r in res)

    def tout_at(r: dict[str, Any], T: float) -> bool:
        return r["status"] == "timeout" or r["secs"] >= T

    rep["necessity_serrano_rate_20s"] = round(sizes["tout"] / len(cs), 5)
    rep["necessity_by_T"] = {}
    for T in (8, 9, 10, 11, 12, 13, 14, 16, 20):
        est = 0.0
        for s in strata:
            xs = [r for r in nec if r["stratum"] == s]
            if xs:
                est += sizes[s] * sum(tout_at(r, T) for r in xs) / len(xs)
        rep["necessity_by_T"][T] = round(est / len(cs), 5)
    ratio = [
        r["secs"] / r["serrano"][1]
        for r in nec
        if r["status"] != "timeout" and r["serrano"][0] != "timeout" and r["serrano"][1] >= 2
    ]
    rep["necessity_time_ratio_median"] = round(statistics.median(ratio), 3) if ratio else None
    rep["necessity_time_ratio_p10_p90"] = (
        [round(x, 3) for x in statistics.quantiles(ratio, n=10)[::8]] if len(ratio) > 10 else None
    )
    rep["search_serrano_timeouts"] = len(rej)
    rep["search_by_T"] = {
        T: sum(tout_at(r, T) for r in srch) for T in (8, 10, 11, 12, 13, 14, 16, 20)
    }
    sr = [
        r["secs"] / r["serrano"][1]
        for r in srch
        if r["stratum"] == "slow" and r["status"] == "done"
    ]
    rep["search_time_ratio_median"] = round(statistics.median(sr), 3) if sr else None
    rep["search_timeouts_still_timeout_at_20"] = sum(
        r["status"] == "timeout" for r in srch if r["stratum"] == "tout"
    )
    rep["search_timeout_rejects_secs_if_done"] = sorted(
        r["secs"] for r in srch if r["stratum"] == "tout" and r["status"] == "done"
    )
    rep["rows"] = res
    with open(a.out, "w") as f:
        json.dump(rep, f, indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()
