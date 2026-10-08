"""Feature table for the gap analysis: one row per problem, for textbook72, the hard12 pool, the
hard12 test split and a 20k K12 sample (CPU only).

    cd /scratch/cowifornia/nd-rl/think/nd-rl
    ND_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 nice -n 15 code/.venv/bin/python \
        code/experiments/current/thinking_traces_20261005/gap_analysis/features.py --workers 8

Needs runs/gap_analysis/minproofs_{textbook,pool,h12test,k12}.jsonl (minproofs.py). Writes
runs/gap_analysis/features.jsonl. Each row: `set` (dev58 / train14 / pool / h12test / k12), `name`,
`ref_lines` (textbook reference length, or the generator proof's length), `min_lines` (shortest
proof found by the exact search; for generator rows <= ref_lines), `min_actions` (state-env
actions of that proof), theorem features (`common.thm_features`), shortest-proof features
(`p_*`, `common.proof_features` on the searched proof) and, for generator rows, the same features
of the generator's own proof (`g_*`). The textbook rows carry no proof text.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402
from state_env import decompose  # noqa: E402  (fork, read-only; on sys.path via common)


def n_actions(prompt: str, proof: str | None) -> int | None:
    if not proof:
        return None
    try:
        return len(decompose(prompt, proof)[0])
    except Exception:
        return None


def _row(job: dict[str, Any]) -> dict[str, Any]:
    out = {
        k: job.get(k) for k in ("set", "name", "ref_lines", "min_lines", "cfg", "t0_pass1", "src")
    }
    out.update(C.thm_features(job["prompt"]))
    if job.get("min_proof"):
        out.update(C.proof_features(job["prompt"], job["min_proof"]))
        out["min_actions"] = n_actions(job["prompt"], job["min_proof"])
    if job.get("gen_proof"):
        g = C.proof_features(job["prompt"], job["gen_proof"])
        out.update({"g_" + k[2:]: v for k, v in g.items()})
        out["gen_actions"] = n_actions(job["prompt"], job["gen_proof"])
    out["_prompt"] = job["prompt"] if job["set"] not in ("dev58", "train14") else None
    return out


def jobs() -> list[dict[str, Any]]:
    mp_ = {}
    for s in ("textbook", "pool", "h12test", "k12"):
        for x in C.jl(C.OUT / f"minproofs_{s}.jsonl"):
            mp_[(s, x["name"])] = x
            if s == "k12":  # k12 names index the 7-12 subset; match on the theorem instead
                mp_[("k12p", x["prompt"])] = x
    out = []
    for fn, role in (("textbook_dev.jsonl", "dev58"), ("textbook_train.jsonl", "train14")):
        for r in C.jl(C.TB / fn):
            m = mp_.get(("textbook", r["name"]), {})
            out.append(
                dict(
                    set=role,
                    name=r["name"],
                    prompt=r["prompt"],
                    ref_lines=r.get("reference_lines"),
                    min_lines=m.get("min_lines"),
                    min_proof=m.get("proof") if m.get("src") == "search" else None,
                    src=m.get("src"),
                )
            )
    for r in C.jl(C.POOL):
        m = mp_[("pool", r["name"])]
        out.append(
            dict(
                set="pool",
                name=r["name"],
                prompt=r["prompt"],
                ref_lines=r["n_lines"],
                min_lines=m["min_lines"] or r["n_lines"],
                min_proof=m["proof"],
                gen_proof=r["gen_proof"],
                cfg=r.get("cfg"),
                t0_pass1=r.get("t0_pass1"),
                src=m["src"],
            )
        )
    for r in C.jl(C.H12TEST):
        m = mp_[("h12test", r["name"])]
        out.append(
            dict(
                set="h12test",
                name=r["name"],
                prompt=r["prompt"],
                ref_lines=r["reference_lines"],
                min_lines=m["min_lines"] or r["reference_lines"],
                min_proof=m["proof"],
                gen_proof=r["reference_proof"],
                src=m["src"],
            )
        )
    for i, r in enumerate(C.k12_sample()):
        name = f"k12_{i}_{r['key'][:40]}"
        m = mp_.get(("k12p", r["prompt"])) if 7 <= r["n_lines"] <= 12 else None
        out.append(
            dict(
                set="k12",
                name=name,
                prompt=r["prompt"],
                ref_lines=r["n_lines"],
                min_lines=(m["min_lines"] or r["n_lines"]) if m else None,
                min_proof=m["proof"] if m else None,
                gen_proof=r["proof"],
                src=m["src"] if m else None,
            )
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    js = jobs()
    print(len(js), "problems", flush=True)
    with mp.Pool(a.workers) as pool, open(C.OUT / "features.jsonl", "w") as fo:
        for i, r in enumerate(pool.imap(_row, js, chunksize=64)):
            fo.write(json.dumps(r) + "\n")
            if (i + 1) % 5000 == 0:
                print(i + 1, flush=True)
    print("wrote", C.OUT / "features.jsonl", flush=True)


if __name__ == "__main__":
    main()
