"""Classify a checkpoint's raw draws on the CPU (PREREG.md): counts per theorem, no proof text.

    python fr_classify.py --refs refs.jsonl --reads READS --out SUMMARIES [--tags a,b] [--procs 8]

Writes SUMMARIES/<tag>.json. The raw draws stay in READS/<tag>/ under /scratch.

QC counters, reported in `log.md`:
- `replay_mismatch`: a stored finished attempt whose replay through `Trace` gives another ND proof. Must be 0.
- `lean_ok_sigma_not_stage4`: draws Lean accepted that `coe_sigma.flags` marks σ-strict on the written
  proof but that the trace does not take to stage 4 (the two classifiers disagree, or the verifier rejects what
  Lean accepts).
- `stage4_ok_not_sigma`: the reverse.
"""

# ruff: noqa: E501, SIM115, B905, B007, SIM300
from __future__ import annotations

import argparse
import collections
import gzip
import json
import math
import multiprocessing as mp
from pathlib import Path

from factor_lib import (
    classify_dn,
    classify_draw,
    classify_negi,
    classify_pick,
    classify_reach,
    flags,
    has_sigma,
    ref_from_json,
    replay,
    shift_action,
)

REFS: dict = {}
FLAG_KEYS = ("sigma_strict", "sigma_chain", "dn_derived", "dn_any")


def load(path: Path) -> list[dict]:
    with gzip.open(path, "rt") as f:
        return [json.loads(x) for x in f]


def toks(raw: list[str]) -> list[list[str]]:
    return [x.split() if x else [] for x in raw]


def next_action(row: dict) -> tuple[list[str], bool]:
    return (row["raw"][0].split() if row["raw"] and row["raw"][0] else []), bool(
        row["ended"] and row["ended"][0]
    )


def by_name(rows: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        out[r["name"]].append(r)
    for v in out.values():
        v.sort(key=lambda r: r["draw"])
    return out


def do_a(rows: list[dict], qc: collections.Counter) -> dict:
    res = {}
    for name, rs in by_name(rows).items():
        ref = REFS.get(name)
        prompt = ref.prompt
        c: dict = {"n": len(rs), "n_ok": 0, "n_ok32": 0, "stage": [0] * 5, "stage_ref_T": [0] * 5, "stage32": [0] * 5,
                   "end": collections.Counter(), "n_reductio_draws": 0,
                   "solved_flags": collections.Counter()}  # fmt: skip
        for r in rs:
            acts = toks(r["raw"])
            d = classify_draw(prompt, r["base"], [a for a, e in zip(acts, r["ended"]) if e], ref.T)
            c["stage"][d["stage"]] += 1
            c["stage_ref_T"][d["stage_ref_T"]] += 1
            if r["draw"] < 32:
                c["stage32"][d["stage"]] += 1
            c["n_reductio_draws"] += d["n_reductio"] > 0
            c["end"][r["end"].split(":")[0]] += 1
            if r["end"] == "done":
                tr = replay(prompt, [a for a, e in zip(acts, r["ended"]) if e], r["base"])
                clean = r["nd"].removeprefix("LEANREJ ")
                if tr.env.nd() != clean:
                    qc["replay_mismatch"] += 1
            if r["ok"]:
                c["n_ok"] += 1
                c["n_ok32"] += r["draw"] < 32
                f = flags(r["nd"])
                if f is None:
                    qc["ok_unparsable"] += 1
                else:
                    for k in FLAG_KEYS:
                        c["solved_flags"][f"w_{k}"] += bool(f["w"][k])
                        c["solved_flags"][f"p_{k}"] += bool(f["p"][k])
                    sig = f["w"]["sigma_strict"] or f["w"]["sigma_chain"]
                    c["solved_flags"]["sigma_w"] += sig
                    c["solved_flags"]["sigma_any_form"] += has_sigma(f)
                    if sig and d["stage"] < 4:
                        qc["lean_ok_sigma_not_stage4"] += 1
                    if not sig and d["stage"] == 4:
                        qc["stage4_ok_not_sigma"] += 1
        c["solved"] = c["n_ok"] > 0
        c["solved32"] = c["n_ok32"] > 0
        # danread's E32: pass@32 from all 64 draws, unbiased
        c["e32"] = (
            1.0 - math.comb(c["n"] - c["n_ok"], 32) / math.comb(c["n"], 32)
            if c["n"] >= 32
            else None
        )
        c["end"] = dict(c["end"])
        c["solved_flags"] = dict(c["solved_flags"])
        res[name] = c
    return res


def do_next(rows: list[dict], fn, keys: tuple[str, ...]) -> dict:
    res = {}
    for name, rs in by_name(rows).items():
        ref = REFS[name]
        c = {"n": len(rs), "env_ok": 0, **{k: 0 for k in keys}, "end": collections.Counter()}
        for r in rs:
            a, ended = next_action(r)
            d = fn(ref, r["base"], a, ended)
            c["env_ok"] += d["env_ok"]
            for k in keys:
                c[k] += bool(d[k])
            c["end"][r["end"].split(":")[0]] += 1
        c["end"] = dict(c["end"])
        res[name] = c
    return res


def do_reach(rows: list[dict], qc: collections.Counter) -> dict:
    res = {}
    for name, rs in by_name(rows).items():
        ref = REFS[name]
        c = {"n": len(rs), "bot": 0, "negi": 0, "dn": 0, "solved": 0, "solved_sigma": 0, "false_bot_draws": 0,
             "nested_bot_draws": 0, "end": collections.Counter()}  # fmt: skip
        for r in rs:
            acts = [a for a, e in zip(toks(r["raw"]), r["ended"]) if e]
            d = classify_reach(ref, r["base"], acts)
            c["bot"] += d["bot"]
            c["negi"] += d["negi"]
            c["dn"] += d["dn"]
            c["false_bot_draws"] += d["false_bot"] > 0
            c["nested_bot_draws"] += d["nested_bot"] > 0
            c["end"][r["end"].split(":")[0]] += 1
            if r["end"] == "done":
                prefix = [shift_action(x, r["base"]) for x in ref.actions[: ref.pick + 1]]
                tr = replay(ref.prompt, prefix + acts, r["base"])
                if tr.env.nd() != r["nd"].removeprefix("LEANREJ "):
                    qc["replay_mismatch"] += 1
            if r["ok"]:
                c["solved"] += 1
                f = flags(r["nd"])
                c["solved_sigma"] += bool(f and (f["w"]["sigma_strict"] or f["w"]["sigma_chain"]))
        c["end"] = dict(c["end"])
        res[name] = c
    return res


def classify_tag(a: tuple) -> str:
    tag, reads, out = a
    d = Path(reads) / tag
    qc: collections.Counter = collections.Counter()
    s: dict = {"tag": tag, "stages": {}}
    if (d / "ckpt.json").exists():
        s["ckpt"] = json.loads((d / "ckpt.json").read_text())
    if (d / "A.jsonl.gz").exists():
        s["stages"]["A"] = do_a(load(d / "A.jsonl.gz"), qc)
    if (d / "pick.jsonl.gz").exists():
        s["stages"]["pick"] = do_next(
            load(d / "pick.jsonl.gz"), classify_pick, ("pick", "assume_any")
        )
    if (d / "reach.jsonl.gz").exists():
        s["stages"]["reach"] = do_reach(load(d / "reach.jsonl.gz"), qc)
    if (d / "negi.jsonl.gz").exists():
        s["stages"]["negi"] = do_next(load(d / "negi.jsonl.gz"), classify_negi, ("negi",))
    if (d / "dn.jsonl.gz").exists():
        s["stages"]["dn"] = do_next(
            load(d / "dn.jsonl.gz"), classify_dn, ("dn", "r_copy", "dn_other")
        )
    if (d / "c.json.gz").exists():
        with gzip.open(d / "c.json.gz", "rt") as f:
            c = json.load(f)
        out_c = {}
        for name, m in c["logp_nats"].items():
            ref = REFS[name]
            means = [sum(row[i] for row in m) / len(m) for i in range(len(ref.actions))]
            tot = [sum(row) for row in m]
            out_c[name] = {"logp_mean_nats": means, "total_nats_min": min(tot), "total_nats_max": max(tot),
                           "groups": ref.groups}  # fmt: skip
        s["stages"]["C"] = out_c
    s["stats"] = {}
    for st in ("A", "pick", "reach", "negi", "dn", "C"):
        p = d / f"{st}.stats.json"
        if p.exists():
            x = json.loads(p.read_text())
            s["stats"][st] = {k: x[k] for k in x if k in ("secs", "env_end", "env_fail_reason", "env_waves", "names_defined",
                                                          "names_renamed", "noise_seed", "k", "batch", "rows", "tokens",
                                                          "peak_alloc_gb", "host")}  # fmt: skip
    s["qc"] = dict(qc)
    tmp = Path(out) / f"{tag}.json.tmp"
    tmp.write_text(json.dumps(s, indent=0))
    tmp.replace(Path(out) / f"{tag}.json")
    return f"{tag}: {sorted(s['stages'])} qc {dict(qc)}"


def init(refs_path: str) -> None:
    for x in open(refs_path):
        r = ref_from_json(json.loads(x))
        REFS[r.name] = r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs", required=True)
    ap.add_argument("--reads", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tags", default=None)
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    tags = (
        a.tags.split(",")
        if a.tags
        else sorted(p.name for p in Path(a.reads).iterdir() if p.is_dir())
    )
    with mp.Pool(min(a.procs, max(1, len(tags))), initializer=init, initargs=(a.refs,)) as pool:
        for r in pool.imap_unordered(classify_tag, [(t, a.reads, a.out) for t in tags]):
            print(r, flush=True)


if __name__ == "__main__":
    main()
