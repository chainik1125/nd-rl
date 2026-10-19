"""Sealing: renaming-class disjointness of the test set from every training / eval / pool file, and
the key sidecar.

    cd nd-rl && python -m code.experiments.current.thinking_traces_20261005.test_21_30.seal index \
        --out /scratch/.../test_21_30/work/excl_index.pkl
    python -m ...test_21_30.seal check --index work/excl_index.pkl --set test_21_30.jsonl \
        --report seal_report.json
    python -m ...test_21_30.seal keys --set test_21_30.jsonl --out test_21_30.keys.jsonl

Keys are `code.thinking.data.class_keys` (the fork's `gen.canon_key` atom-renaming class, premise
order kept, and ndbench's premise-order-invariant `pkey`), the same keys `data.EvalIndex` and the
RL-pool prep use. A record hits a file when either key matches. `check --rl_pool PATH` adds a file
at check time (the RL pool, if it exists then).

Files indexed (`exclusion_files()`): K12 train + held-out and the cap-6 Stage-1 set; every jsonl
under the fork's data/ (ladder rl_targets / transfer / long pools, take-home rl_targets / transfer /
held-out, textbook72, gt, p2, r1-r3, lj, bs, ...); the OOD sets; the thinking traces (K12 prompts);
the RL-pool candidate files under runs/pools/ (raw/cands.jsonl, cands_clean.jsonl, rl_hard*.jsonl).
Files without a `prompt` field are read through `text` (`THM ... PRF` prefix); rows with neither are
counted and skipped.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import pickle
import sys
import time

from code.thinking.data import class_keys

FORK = "/scratch/cowifornia/nd-rl/think/nd-takehome"
THINK = "/scratch/cowifornia/nd-rl/think"


def exclusion_files() -> list[str]:
    fs = [
        "/scratch/cowifornia/nd-rl/bench/dan_ckpts/data/train_k12.jsonl",
        "/scratch/cowifornia/nd-rl/bench/dan_ckpts/data/heldout.jsonl",
        "/scratch/cowifornia/nd-rl/bench/dan_ckpts/data/train_depth3_f0_a1.jsonl",
    ]
    fs += sorted(glob.glob(f"{FORK}/data/**/*.jsonl", recursive=True))
    fs += sorted(glob.glob(f"{THINK}/ood_check/sets/*.jsonl"))
    fs += sorted(glob.glob(f"{THINK}/traces/*_traces*.jsonl"))
    fs += sorted(glob.glob(f"{THINK}/runs/pools/raw/cands.jsonl"))
    fs += sorted(glob.glob(f"{THINK}/runs/pools/cands_clean.jsonl"))
    fs += sorted(glob.glob(f"{THINK}/runs/pools/rl_hard*.jsonl"))
    fs += sorted(glob.glob(f"{THINK}/runs/pools/long*_cands.jsonl"))
    return [f for f in fs if os.path.exists(f)]


def prompt_of(r: dict) -> str | None:
    p = r.get("prompt")
    if isinstance(p, str) and p.startswith("THM"):
        return p
    t = r.get("text")
    if isinstance(t, str) and t.startswith("THM") and " PRF" in t:
        return t[: t.index(" PRF") + 4]
    return None


def build_index(files: list[str]) -> dict:
    keys: dict[str, set] = collections.defaultdict(set)
    pkeys: dict[str, set] = collections.defaultdict(set)
    manifest = []
    for f in files:
        n = bad = 0
        t0 = time.time()
        with open(f) as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    bad += 1
                    continue
                p = prompt_of(r) if isinstance(r, dict) else None
                if p is None:
                    bad += 1
                    continue
                try:
                    k, pk = class_keys(p)
                except Exception:  # noqa: BLE001  (unparsable prompt)
                    bad += 1
                    continue
                keys[k].add(f)
                if pk is not None:
                    pkeys[pk].add(f)
                n += 1
        manifest.append(dict(file=f, records=n, skipped=bad, secs=round(time.time() - t0, 1)))
        print(f"{n:9d} {bad:6d} {f}", flush=True)
    return dict(keys=dict(keys), pkeys=dict(pkeys), manifest=manifest)


def hits(index: dict, prompt: str) -> set:
    k, pk = class_keys(prompt)
    out = set(index["keys"].get(k, ()))
    if pk is not None:
        out |= index["pkeys"].get(pk, set())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a1 = sub.add_parser("index")
    a1.add_argument("--out", required=True)
    a2 = sub.add_parser("check")
    a2.add_argument("--index", required=True)
    a2.add_argument("--set", required=True)
    a2.add_argument("--report", required=True)
    a2.add_argument("--rl_pool", default=None)
    a2.add_argument("--drop_out", default=None, help="write the set minus hit records here")
    a4 = sub.add_parser("filter", help="drop rows (e.g. pieces) whose class hits the index")
    a4.add_argument("--index", required=True)
    a4.add_argument("--in", dest="inp", required=True)
    a4.add_argument("--out", required=True)
    a3 = sub.add_parser("keys")
    a3.add_argument("--set", required=True)
    a3.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "index":
        idx = build_index(exclusion_files())
        with open(a.out, "wb") as f:
            pickle.dump(idx, f)
        print(
            f"classes {len(idx['keys'])}, pkeys {len(idx['pkeys'])}, files {len(idx['manifest'])}"
        )
        return
    if a.cmd == "filter":
        with open(a.index, "rb") as f:
            idx = pickle.load(f)
        n = kept = 0
        per = collections.Counter()
        with open(a.inp) as f, open(a.out, "w") as fo:
            for line in f:
                r = json.loads(line)
                n += 1
                h = hits(idx, r["prompt"])
                if h:
                    per.update(h)
                    continue
                kept += 1
                fo.write(line)
        print(
            json.dumps(dict(n=n, kept=kept, dropped=n - kept, per_file=dict(per.most_common(10))))
        )
        return
    if a.cmd == "keys":
        with open(a.set) as f, open(a.out, "w") as fo:
            for line in f:
                r = json.loads(line)
                k, pk = class_keys(r["prompt"])
                fo.write(json.dumps(dict(name=r["name"], key=k, pkey=pk)) + "\n")
        return
    with open(a.index, "rb") as f:
        idx = pickle.load(f)
    extra = []
    if a.rl_pool and os.path.exists(a.rl_pool):
        more = build_index([a.rl_pool])
        for k, v in more["keys"].items():
            idx["keys"].setdefault(k, set()).update(v)
        for k, v in more["pkeys"].items():
            idx["pkeys"].setdefault(k, set()).update(v)
        extra = more["manifest"]
    with open(a.set) as fi:
        rows = [json.loads(line) for line in fi]
    per_file = collections.Counter()
    hit_rows = []
    keep = []
    within = collections.Counter()
    dropped_dup = 0
    for r in rows:
        h = hits(idx, r["prompt"])
        k, pk = class_keys(r["prompt"])
        dup = within[("k", k)] > 0 or (pk is not None and within[("pk", pk)] > 0)
        within[("k", k)] += 1
        if pk is not None:
            within[("pk", pk)] += 1
        if h:
            hit_rows.append(dict(name=r.get("name"), files=sorted(h)))
            per_file.update(h)
        elif (
            dup
        ):  # same class (or premise permutation) as an earlier row of the set: keep the first only
            dropped_dup += 1
        else:
            keep.append(r)
    dup_within = sum(v - 1 for v in within.values() if v > 1)
    rep = dict(
        n=len(rows),
        hit=len(hit_rows),
        per_file=dict(per_file.most_common()),
        hit_rows=hit_rows[:200],
        duplicate_keys_within_set=dup_within,
        dropped_within_set_duplicates=dropped_dup,
        kept=len(keep),
        files=len(idx["manifest"]) + len(extra),
        records=sum(m["records"] for m in idx["manifest"] + extra),
        manifest=idx["manifest"] + extra,
        rl_pool=a.rl_pool if a.rl_pool and os.path.exists(a.rl_pool) else None,
    )
    with open(a.report, "w") as fo:
        json.dump(rep, fo, indent=1)
    if a.drop_out:
        with open(a.drop_out, "w") as fo:
            for r in keep:
                fo.write(json.dumps(r) + "\n")
    print(json.dumps({k: v for k, v in rep.items() if k not in ("manifest", "hit_rows")}))


if __name__ == "__main__":
    sys.exit(main())
