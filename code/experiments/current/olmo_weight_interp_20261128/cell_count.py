#!/usr/bin/env python3
"""Count and check one cell's per-draw files (stdlib only; runs on the pod and on serrano).

    cell_count.py wait --tag a050 [--prep-chain prep] [--max-wait 3600]        exit 0 | 21
        Block until the job this cell depends on is DONE (preflight for a100, write_<tag>
        otherwise). Give up at once if it FAILED or the chain that runs it ended without it.
    cell_count.py mode --mem-bytes N --used-bytes N [--volume-gb 500]          exit 0 | 5
        Which schedule fits this pod: prints "MODE lanes", "MODE sequential" or "MODE none" and why.
    cell_count.py precheck --tag a050                                          exit 0 | 21
        May this cell sample? The weights are downloaded and sha256-checked (a100), or written
        and checked (merged checkpoints).
    cell_count.py count --dir runs/cell_a050/out --spec inputs/spec_reach4.txt --tag a050 \
        --seed 710012 --model-dir /workspace/wi/models/interp_a050 \
        [--prompt-sha inputs/prompt_sha256_reach.json]                         exit 0 | 31 | 32
        Writes <dir>/count.json.

A cell is complete (exit 0) only if the gzip stream is whole, the sample rows and the graded rows
both carry exactly the planned (prompt, chunk, 0) keys (one request per draw), every graded row has
a boolean `strict`, and the sampler's own records show the registered settings. 31 = rows missing,
extra or duplicated (the cell was cut or the files are damaged); 32 = all rows are there but a
setting differs from the registered one. Missing draws are listed, never dropped.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
import zlib
from pathlib import Path

TAGS = {"a100": 1.0, "a075": 0.75, "a050": 0.5, "a025": 0.25}
SETTINGS = {
    "temperature": 1.0,
    "top_p": 1.0,
    "max_tokens": 32768,
    "chunk": 1,
    "chunk0": 0,
    "stop_after": 0,
    "lora": None,
    "fill_context": None,
    "tp": 1,
    "vllm": "0.11.0",
}


def parse_spec(path) -> dict[str, int]:
    out: dict[str, int] = {}
    for tok in Path(path).read_text().split():
        pid, _, n = tok.partition(":")
        out[pid] = int(n)
    return out


def planned_keys(spec: dict[str, int]) -> set[tuple[str, int, int]]:
    return {(pid, c, 0) for pid, n in spec.items() for c in range(n)}


def read_gz_rows(path) -> tuple[list[dict], bool]:
    """Rows of a gzip JSONL and whether the stream is whole (a killed sampler leaves it cut)."""
    rows: list[dict] = []
    whole = True
    try:
        with gzip.open(path, "rt") as f:
            for ln in f:
                if not ln.strip():
                    continue
                try:
                    rows.append(json.loads(ln))
                except json.JSONDecodeError:
                    whole = False
                    break
    except (EOFError, OSError, zlib.error):
        whole = False
    return rows, whole


def read_jsonl(path) -> list[dict]:
    return [json.loads(ln) for ln in Path(path).read_text().splitlines() if ln.strip()]


def key(r: dict) -> tuple[str, int, int]:
    return (r["pid"], int(r.get("chunk", 0)), int(r.get("i", 0)))


def sample_files(d: Path) -> tuple[Path, Path | None]:
    """(file to read, repaired copy if the grader made one)."""
    raw, fixed = d / "samples.jsonl.gz", d / "samples_complete.jsonl.gz"
    return raw, (fixed if fixed.exists() else None)


def count(
    d: Path,
    spec: dict[str, int],
    tag: str,
    seed: int,
    model_dir: str,
    prompt_sha: dict[str, str] | None = None,
) -> dict:
    plan = planned_keys(spec)
    res: dict = {"tag": tag, "alpha": TAGS.get(tag), "planned": len(plan), "problems": []}
    prob: list[str] = res["problems"]
    raw, fixed = sample_files(d)
    rows, whole = read_gz_rows(raw) if raw.exists() else ([], False)
    res["gzip_whole"] = whole
    res["repaired_copy"] = str(fixed.name) if fixed else None
    if not raw.exists():
        prob.append("no samples.jsonl.gz")
    elif not whole:
        prob.append("samples.jsonl.gz is cut (the sampler was killed)")
    skeys = [key(r) for r in rows]
    graded = read_jsonl(d / "graded.jsonl") if (d / "graded.jsonl").exists() else []
    gkeys = [key(r) for r in graded]
    res["sample_rows"], res["graded_rows"] = len(skeys), len(gkeys)
    for name, ks in (("sample", skeys), ("graded", gkeys)):
        if len(set(ks)) != len(ks):
            prob.append(f"duplicate {name} rows: {len(ks) - len(set(ks))}")
        extra = sorted(set(ks) - plan)
        if extra:
            prob.append(f"{len(extra)} {name} rows outside the plan, e.g. {extra[:3]}")
    missing = sorted(plan - set(gkeys))
    res["missing"] = [list(k) for k in missing]
    res["n_missing"] = len(missing)
    if missing:
        prob.append(f"{len(missing)} planned draws have no graded row")
    if set(skeys) != set(gkeys):
        prob.append("sample rows and graded rows do not carry the same keys")
    if any(not isinstance(r.get("strict"), bool) for r in graded):
        prob.append("a graded row has no boolean `strict`")
    good = [r for r in graded if isinstance(r.get("strict"), bool) and key(r) in plan]
    res["passes"] = sum(r["strict"] for r in good)
    res["cut_at_cap"] = sum(r.get("finish") == "length" for r in good)
    res["no_end_of_thinking"] = sum(not r.get("think_end", True) for r in good)
    res["mean_tokens"] = sum(r.get("n_tok", 0) for r in good) / len(good) if good else None
    rows_ok = not prob
    # ---- the sampler's own records
    sprob: list[str] = []
    metas = read_jsonl(d / "meta.jsonl") if (d / "meta.jsonl").exists() else []
    res["meta_lines"] = len(metas)
    if len(metas) != 1:
        sprob.append(f"{len(metas)} meta lines (1 expected: one uninterrupted engine load)")
    if metas:
        m = metas[-1]
        want = dict(SETTINGS, seed=seed, model_dir=model_dir, job=f"cell_{tag}")
        want.update(n_requests=len(plan), samples=len(plan))
        got = {k: m.get(k, "<absent>") for k in want}
        res["settings"] = got
        sprob += [f"meta {k}={got[k]!r}, expected {want[k]!r}" for k in want if got[k] != want[k]]
        res["tok_per_s"], res["t_gen_s"], res["t_load_s"] = (
            m.get("tok_per_s"),
            m.get("t_gen_s"),
            m.get("t_load_s"),
        )
        res["gpus"], res["length_cut_meta"] = m.get("gpus"), m.get("length_cut")
        if prompt_sha is not None:
            have = m.get("prompt_sha256_by_pid") or {}
            bad = sorted(p for p in spec if have.get(p) != prompt_sha.get(p))
            res["prompt_sha_checked"] = len(spec) - len(bad)
            if bad:
                sprob.append(
                    f"{len(bad)} prompts tokenise differently from the reference: {bad[:3]}"
                )
    wrap = d / "wrap.json"
    if wrap.exists():
        w = json.loads(wrap.read_text())
        res["wrap_status"] = w.get("status")
        if w.get("status") != "done":
            sprob.append(f"wrap.json status {w.get('status')!r}")
        if w.get("temperature_overridden") or w.get("top_p_overridden") or w.get("patch"):
            sprob.append("wrap.json shows an override or a patch")
        res["dan_sample_py_sha256"] = w.get("dan_sample_py_sha256")
    else:
        sprob.append("no wrap.json")
    res["settings_problems"] = sprob
    res["rows_complete"] = rows_ok
    res["complete"] = rows_ok and not sprob
    return res


def wait_for(
    tag: str,
    wi: Path,
    prep_chain: str,
    max_wait: float,
    poll: float = 15.0,
    grace: float = 180.0,
) -> tuple[int, str]:
    """The cells run in one chain and the downloads and writes in another (they overlap). A cell
    holds the GPU while it waits here, so it must not wait for something that will never come."""
    if tag not in TAGS:
        return 2, f"unknown tag {tag}"
    need = "preflight" if tag == "a100" else f"write_{tag}"
    st, chain = wi / "runs" / need / "state", wi / "runs" / "_chains" / prep_chain / "state"
    t0 = time.time()
    while True:
        waited = time.time() - t0
        if (st / "DONE").exists():
            return 0, f"{need} is DONE (waited {waited:.0f} s)"
        if (st / "FAILED").exists() and not (st / "RUNNING").exists():
            return 21, f"{need} FAILED: {(st / 'FAILED').read_text().strip()}"
        ended = (chain / "DONE").exists() or (chain / "FAILED").exists()
        if ended and not (st / "RUNNING").exists():
            return 21, f"chain {prep_chain} has ended and {need} is not DONE"
        if not chain.exists() and waited > grace:
            return 21, f"chain {prep_chain} was never started ({grace:.0f} s grace)"
        if waited > max_wait:
            return 21, f"{need} not DONE after {max_wait:.0f} s"
        time.sleep(poll)


#: The default schedule writes on the CPU while a cell samples. It is used only if the container has
#: room for both (an unmeasured, conservative figure) and the volume has room for the two inputs,
#: three merged copies and a margin beside whatever else is on it (the pod is shared).
LANES_MIN_RAM_GB, LANES_MIN_DISK_GB, SEQUENTIAL_MIN_DISK_GB = 96, 345, 215


def schedule_mode(
    mem_bytes: int | None, used_bytes: int, volume_gb: float = 500.0
) -> tuple[str, str]:
    """("lanes" | "sequential" | "none", why) from the container's memory limit (None = unknown) and
    the bytes already on the volume. RUNBOOK section 3."""
    unused = volume_gb - used_bytes / 1e9
    if unused < SEQUENTIAL_MIN_DISK_GB:
        return "none", (
            f"only {unused:.0f} GB of the {volume_gb:g} GB volume are unused; one merged copy at a "
            f"time needs {SEQUENTIAL_MIN_DISK_GB}"
        )
    why = []
    if mem_bytes is None:
        why.append("the memory limit could not be read")
    elif mem_bytes < LANES_MIN_RAM_GB * 1e9:
        why.append(f"memory limit {mem_bytes / 1e9:.0f} GB is under {LANES_MIN_RAM_GB} GB")
    if unused < LANES_MIN_DISK_GB:
        why.append(f"only {unused:.0f} GB of the volume are unused ({LANES_MIN_DISK_GB} needed)")
    if why:
        return "sequential", "; ".join(why)
    return "lanes", f"memory limit {mem_bytes / 1e9:.0f} GB, {unused:.0f} GB of the volume unused"


def precheck(tag: str, wi: Path, models: Path, merged: Path | None = None) -> tuple[int, str]:
    """`models` holds the downloaded s2300; `merged` (default <wi>/models) the merged copies."""
    if tag == "a100":
        f = wi / "runs" / "download_s2300" / "download_verify.json"
        if not (models / "s2300" / ".complete").exists() or not f.exists():
            return 21, "s2300 is not downloaded and verified"
        try:
            v = json.loads(f.read_text().strip().splitlines()[-1])
        except (ValueError, IndexError):
            return 21, f"cannot read {f}"
        if not (v.get("ok") and v.get("sha256_checked")):
            return 21, f"s2300 verification record is not ok with sha256: {v}"
        return 0, "s2300 downloaded, sha256-checked"
    if tag not in TAGS:
        return 2, f"unknown tag {tag}"
    d = (merged or wi / "models") / f"interp_{tag}"
    try:
        mk = json.loads((d / "WI_WRITE_OK").read_text())
        ck = json.loads((wi / "runs" / f"write_{tag}" / "out" / "check.json").read_text())
    except (OSError, ValueError):
        return 21, f"{d} has no write marker or check report"
    if abs(float(mk.get("alpha", -1)) - TAGS[tag]) > 1e-12 or not ck.get("ok"):
        return 21, f"write record does not match alpha {TAGS[tag]} or is not ok"
    return 0, f"interp_{tag} written and checked (realised fraction {mk.get('realised_fraction')})"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("precheck")
    p.add_argument("--tag", required=True)
    p.add_argument("--wi", default=os.environ.get("M2_ROOT", "/workspace/wi"))
    p.add_argument("--models", default=os.environ.get("MODELS_ROOT", "/workspace/models"))
    p.add_argument("--merged", default=os.environ.get("WI_MERGED_ROOT"))
    w = sub.add_parser("wait")
    w.add_argument("--tag", required=True)
    w.add_argument("--wi", default=os.environ.get("M2_ROOT", "/workspace/wi"))
    w.add_argument("--prep-chain", default=os.environ.get("WI_PREP_CHAIN", "prep"))
    w.add_argument("--max-wait", type=float, default=float(os.environ.get("WI_WAIT_S", "3600")))
    w.add_argument("--poll", type=float, default=float(os.environ.get("WI_WAIT_POLL", "15")))
    w.add_argument("--grace", type=float, default=float(os.environ.get("WI_WAIT_GRACE", "180")))
    m = sub.add_parser("mode")
    m.add_argument("--mem-bytes", default="unknown")
    m.add_argument("--used-bytes", default="0")
    m.add_argument("--volume-gb", type=float, default=500.0)
    c = sub.add_parser("count")
    c.add_argument("--dir", required=True)
    c.add_argument("--spec", required=True)
    c.add_argument("--tag", required=True)
    c.add_argument("--seed", type=int, required=True)
    c.add_argument("--model-dir", required=True)
    c.add_argument("--prompt-sha")
    a = ap.parse_args(argv)
    if a.cmd == "wait":
        rc, why = wait_for(a.tag, Path(a.wi), a.prep_chain, a.max_wait, a.poll, a.grace)
        print(f"WAIT {a.tag} rc={rc}: {why}", flush=True)
        return rc
    if a.cmd == "mode":
        mem = int(a.mem_bytes) if a.mem_bytes.isdigit() else None
        mode, why = schedule_mode(mem, int(a.used_bytes or 0), a.volume_gb)
        print(f"MODE {mode}: {why}")
        return 0 if mode != "none" else 5
    if a.cmd == "precheck":
        merged = Path(a.merged) if a.merged else None
        rc, why = precheck(a.tag, Path(a.wi), Path(a.models), merged)
        print(f"PRECHECK {a.tag} rc={rc}: {why}")
        return rc
    sha = json.loads(Path(a.prompt_sha).read_text()) if a.prompt_sha else None
    res = count(Path(a.dir), parse_spec(a.spec), a.tag, a.seed, a.model_dir, sha)
    (Path(a.dir) / "count.json").write_text(json.dumps(res, indent=1) + "\n")
    rc = 0 if res["complete"] else (31 if not res["rows_complete"] else 32)
    print(
        f"COUNT {a.tag} rc={rc} rows={res['graded_rows']}/{res['planned']} "
        f"missing={res['n_missing']} passes={res['passes']} problems="
        f"{res['problems'] + res['settings_problems']}"
    )
    return rc


if __name__ == "__main__":
    sys.exit(main())
