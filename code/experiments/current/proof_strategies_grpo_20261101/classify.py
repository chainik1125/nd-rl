"""Label proofs with Dan's proof-strategy classifier (taxonomy v1.2): the S1g3 Stage-1 corpus, K12
(to reproduce Dan's table), every read's Lean-accepted proofs, and theorem-level labels.

    cd /home/cowifornia/code/research/nd-rl
    ND_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 nice code/.venv/bin/python -m \\
        code.experiments.current.proof_strategies_grpo_20261101.classify corpus|k12|reads|theorems

Outputs under `common.OUT` (/scratch/cowifornia/nd-rl/proof_strategies_grpo_20261101):
  labels/corpus_s1g3.jsonl.gz   one row per corpus record (replayed actions -> ND -> labels)
  labels/k12.jsonl.gz           K12's generator proofs (Dan's input)
  reads/<read>.json.gz          {name: {prompt, k, n_ok, bad, proofs: [[nd, count, labels]]}}
  theorems.json                 {prompt: Dan's theorem-level labels} for every read problem
  labels/corpus_theorems.jsonl.gz  the same for every corpus record (with `corpus`)
  labels/pool_<L2|L3|L3_3p>.jsonl.gz  GRPO training pools: generator-proof labels + theorem labels
`--workers` (default 8) caps the processes (serrano: <= 8 CPU cores).
"""

from __future__ import annotations

import argparse
import gzip
import json
import multiprocessing as mp
import time
from pathlib import Path
from typing import Any

from . import common as C

# ------------------------------------------------------------------------------ the read list
CHARLES_EXTRA = [("s1g3", 1), ("s1g3", 2), (C.END, 1), (C.END, 2)]
#: (set, stem, seed, k) of every read classified.
READS: list[tuple[str, str, int, int]] = (
    [("charles149", s, 0, 1024) for s in C.CHAIN]
    + [("charles149", s, sd, 1024) for s, sd in CHARLES_EXTRA]
    + [("t1320s250", s, 0, 512) for s in C.CHAIN]
    + [
        (st, s, sd, k)
        for st, k_base, k_rl in (
            ("charles173", 64, 64),
            ("l2test", 8, 8),
            ("l3test", 8, 8),
            ("t1320", 8, 16),
            ("t2130", 8, 8),
        )
        for s, k in (("s1g3", k_base), ("grpo_think_vs_L2", k_base), (C.END, k_rl))
        for sd in (0, 1, 2)
        if not (st == "charles173" and s == "grpo_think_vs_L2")
    ]
)


def read_name(st: str, stem: str, seed: int, k: int) -> str:
    return f"{st}_{stem}_s{seed}_k{k}"


# ------------------------------------------------------------------------------ per-proof labels
def direction(lines: list[dict[str, Any]], pruned_idx: set[int]) -> dict[str, Any]:
    """Order features of the written proof (ND lines are in the order the model wrote its
    `have`s; a box's conclusion line follows its box, but its `have` opened the box).
    gf: the first written non-premise step opens the box the root closes (the proof starts from
    the goal's introduction rule); op: the opening move (forward = a top-level step from the
    premises; goal_intro / goal_cases = the root's own box; goal_reductio = assume the negated goal,
    the root being DN of that ¬I; other_box); fp: top-level non-premise lines written before the
    first box; fpl: of those, the ones the conclusion depends on."""
    root = lines[-1]
    starts: set[int] = set()
    if root["rule"] in ("IMPI", "NEGI") and root["refs"]:
        starts.add(root["refs"][0])
    if root["rule"] == "ORE" and len(root["refs"]) == 5:
        starts.add(root["refs"][1])
    first = next((ln for ln in lines if ln["rule"] != "PR"), None)
    gf = bool(first is not None and first["rule"] == "AS" and first["idx"] in starts)
    # the opening move: the first written non-premise step
    if first is None:
        op = "none"
    elif first["rule"] != "AS":
        op = "forward"  # a step from the premises at the top level
    elif gf:
        op = "goal_cases" if root["rule"] == "ORE" else "goal_intro"
    else:
        op = "other_box"
        if root["rule"] == "DN" and root["refs"]:
            by = {ln["idx"]: ln for ln in lines}
            c = by.get(root["refs"][0])
            while c is not None and c["rule"] == "R":
                c = by.get(c["refs"][0])
            if c is not None and c["rule"] == "NEGI" and c["refs"][:1] == [first["idx"]]:
                op = "goal_reductio"  # assume the negated goal first
    fp = fpl = 0
    for ln in lines:
        if ln["rule"] == "PR":
            continue
        if ln["rule"] == "AS":
            break
        if ln is root:
            break
        fp += 1
        fpl += ln["idx"] in pruned_idx
    return {"gf": gf, "op": op, "fp": fp, "fpl": fpl}


def labels(nd: str, prompt: str) -> dict[str, Any] | None:
    S = C.dan_strategies()
    a = S.analyse(nd, prompt, written=True)
    if a is None:
        return None
    from patterns import parse, prune_lines  # type: ignore[import-not-found]  # the fork's

    lines = parse(nd)
    pr = {ln["idx"] for ln in prune_lines(lines)}
    out = {
        "u": a["u"],
        "sub": a["sub"],
        "pri": a["pri"],
        "pric": a["pric"],
        "L": a["L"],
        "Lp": a["Lp"],
        "dep": a["dep"],
        "bf": a["box_free"],
        "wo": a.get("w_only", []),
    }
    out.update(direction(lines, pr))
    return out


# ------------------------------------------------------------------------------ corpus and K12
def _corpus_one(args: tuple[str, str]) -> str:
    line_e, line_p = args
    x, y = json.loads(line_e), json.loads(line_p)
    assert x["id"] == y["id"], (x["id"], y["id"])
    text = C.replay_actions(x["prompt"], [s["action"] for s in x["steps"]])
    nd = C.lean_to_nd(text)
    same = nd == y["proof"]
    lab = labels(nd, x["prompt"])
    return json.dumps(
        {"id": x["id"], "same_as_plain": same, "nd_len": nd.count(";"), **(lab or {"fail": 1})}
    )


def run_corpus(workers: int) -> None:
    out = C.OUT / "labels/corpus_s1g3.jsonl.gz"
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n = same = fail = 0
    with (
        open(C.TRACES / "e256.jsonl") as fe,
        open(C.TRACES / "plain_e256.jsonl") as fp,
        gzip.open(str(out) + ".tmp", "wt") as fo,
        mp.Pool(workers) as pool,
    ):
        for s in pool.imap(_corpus_one, zip(fe, fp, strict=True), chunksize=256):
            r = json.loads(s)
            n += 1
            same += r["same_as_plain"]
            fail += "fail" in r
            fo.write(s + "\n")
    Path(str(out) + ".tmp").rename(out)
    print(
        f"corpus: {n} records, replay == plain_e256 on {same}, unlabelled {fail}, "
        f"{time.time() - t0:.0f} s"
    )


def _k12_one(line: str) -> str:
    r = json.loads(line)
    lab = labels(r["proof"], r["prompt"])
    return json.dumps(lab or {"fail": 1})


def _thm_one(line: str) -> str:
    r = json.loads(line)
    return json.dumps({"id": r["id"], **C.dan_theorem_level().label(r["prompt"])})


def run_corpus_theorems(workers: int) -> None:
    out = C.OUT / "labels/corpus_theorems.jsonl.gz"
    with (
        open(C.TRACES / "e256.jsonl") as f,
        gzip.open(str(out) + ".tmp", "wt") as fo,
        mp.Pool(workers) as pool,
    ):
        for s in pool.imap(_thm_one, f, chunksize=512):
            fo.write(s + "\n")
    Path(str(out) + ".tmp").rename(out)
    print("corpus theorem labels written")


POOLS = {
    "L2": C.RUNS / "pools/curriculum/L2/pool.jsonl",
    "L3": C.RUNS / "pools/curriculum/L3/pool.jsonl",
    "L3_3p": C.RUNS / "pools/curriculum_3p/L3_3p.jsonl",
}


def _pool_one(line: str) -> str:
    r = json.loads(line)
    lab = labels(r["gen_proof"], r["prompt"]) if r.get("gen_proof") else None
    return json.dumps(
        {"name": r["name"], "thm": C.dan_theorem_level().label(r["prompt"]), **(lab or {"fail": 1})}
    )


def run_pools(workers: int) -> None:
    for nm, p in POOLS.items():
        out = C.OUT / f"labels/pool_{nm}.jsonl.gz"
        with open(p) as f, gzip.open(str(out) + ".tmp", "wt") as fo, mp.Pool(workers) as pool:
            n = 0
            for s in pool.imap(_pool_one, f, chunksize=64):
                fo.write(s + "\n")
                n += 1
        Path(str(out) + ".tmp").rename(out)
        print(f"pool {nm}: {n} records")


def run_k12(workers: int) -> None:
    out = C.OUT / "labels/k12.jsonl.gz"
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(C.K12) as f, gzip.open(str(out) + ".tmp", "wt") as fo, mp.Pool(workers) as pool:
        n = 0
        for i, s in enumerate(pool.imap(_k12_one, f, chunksize=512)):
            fo.write(json.dumps({"i": i, **json.loads(s)}) + "\n")
            n += 1
    Path(str(out) + ".tmp").rename(out)
    print(f"k12: {n} records, {time.time() - t0:.0f} s")


# ------------------------------------------------------------------------------ reads
def _row_one(row: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    nd, bad = C.merged_accepted(row)
    proofs = []
    for body, c in sorted(nd.items(), key=lambda kv: -kv[1]):
        proofs.append([body, c, labels(body, row["prompt"])])
    return row["name"], {
        "prompt": row["prompt"],
        "k": row["k"],
        "n_ok": row["n_ok"],
        "bad": bad,
        "proofs": proofs,
    }


def run_reads(workers: int, only: str | None) -> None:
    keep = {"charles149": C.names(C.CHARLES_CLEAN)}
    d = C.OUT / "reads"
    d.mkdir(parents=True, exist_ok=True)
    with mp.Pool(workers) as pool:
        for st, stem, seed, k in READS:
            name = read_name(st, stem, seed, k)
            if only and only not in name:
                continue
            out = d / f"{name}.json.gz"
            src = C.read_path(st, stem, seed, k)
            if out.exists() and out.stat().st_mtime > src.stat().st_mtime:
                continue
            if not (src.parent / "summary.json").exists():
                print(f"skip {name}: unfinished read")
                continue
            t0 = time.time()
            rows = list(C.iter_rows(src, keep.get(st)))
            res = dict(pool.imap(_row_one, rows, chunksize=1))
            with gzip.open(str(out) + ".tmp", "wt") as fo:
                json.dump(res, fo)
            Path(str(out) + ".tmp").rename(out)
            n_ok = sum(r["n_ok"] for r in res.values())
            bad = sum(r["bad"] for r in res.values())
            nu = sum(len(r["proofs"]) for r in res.values())
            print(
                f"{name}: {len(res)} problems, {n_ok} accepted attempts, {nu} distinct ND proofs, "
                f"{bad} unparsable, {time.time() - t0:.0f} s",
                flush=True,
            )


def run_theorems() -> None:
    T = C.dan_theorem_level()
    prompts: set[str] = set()
    for st, stem, seed, k in READS:
        p = C.OUT / "reads" / f"{read_name(st, stem, seed, k)}.json.gz"
        if p.exists():
            with gzip.open(p, "rt") as f:
                prompts |= {r["prompt"] for r in json.load(f).values()}
    out = {p: T.label(p) for p in sorted(prompts)}
    (C.OUT / "theorems.json").write_text(json.dumps(out))
    print(
        f"theorems: {len(out)} prompts, classical-only {sum(v['classical'] for v in out.values())}"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("what", choices=["corpus", "k12", "reads", "theorems", "pools"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--only", default=None, help="reads: substring of the read names to run")
    a = ap.parse_args(argv)
    C.dan_theorem_level()
    C.lean_to_nd("have n1 : P := h1 ; exact n1")  # load the vendored fork before forking
    if a.what == "corpus":
        run_corpus(a.workers)
        run_corpus_theorems(a.workers)
    elif a.what == "pools":
        run_pools(a.workers)
    elif a.what == "k12":
        run_k12(a.workers)
    elif a.what == "reads":
        run_reads(a.workers, a.only)
    else:
        run_theorems()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
