"""The GPU read of one or more checkpoints (PREREG.md): measures A, B (pick, reach, negi, dn), C.

    python fr_run.py --ckpts ckpts.json --refs refs.jsonl --out OUT --tags dose01_s2[,...] \
        [--stages A,pick,reach,negi,dn,C] [--gpu_budget 10800]

Raw draws go to OUT/<tag>/<stage>.jsonl.gz (they hold proof text: /scratch only). Each stage is written
atomically and skipped when its file exists, so a stopped read resumes. No classification happens
here: `fr_classify.py` does that on the CPU, after the GPU is released.

Operating rules, in code:
- the GPU is claimed by `mkdir` of LOCK, atomically, and released at exit;
- before launch and before every stage: `gpu.gpus` must show no other user on the GPU and
  nvidia-smi no process but this one; at launch also under 500 MiB in use. Otherwise exit 3.
- the ledger (OUT/../gpu_ledger.jsonl) sums GPU-seconds over every process; at --gpu_budget, exit 4.
"""

# ruff: noqa: E501, SIM115, B905, B007, SIM300
from __future__ import annotations

import argparse
import atexit
import gzip
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import gpu as G
from factor_lib import DAN, ref_from_json, ref_rows, run_env, score_rows

HOST_GPU = 0
STAGE_SEED = {"A": 0, "pick": 10000, "reach": 20000, "negi": 30000, "dn": 40000}


def say(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}", flush=True)


def gate(first: bool = False) -> None:
    """Exit 3 unless GPU_ID is ours alone."""
    host = G.HOST
    g = G.gpus(host)
    if g is None or HOST_GPU not in g:
        say(f"GATE: cannot read GPU {HOST_GPU} on {host}")
        sys.exit(3)
    others = g[HOST_GPU] - {G.ME}
    if others:
        say(f"GATE: another user on GPU {HOST_GPU}: {sorted(others)}")
        sys.exit(3)
    q = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,gpu_bus_id", "--format=csv,noheader"],
        capture_output=True, text=True, timeout=30,
    ).stdout  # fmt: skip
    bus = subprocess.run(
        ["nvidia-smi", "--query-gpu=pci.bus_id", "--format=csv,noheader", "-i", str(HOST_GPU)],
        capture_output=True, text=True, timeout=30,
    ).stdout.strip().lower()  # fmt: skip
    pids = [
        int(x.split(",")[0])
        for x in q.splitlines()
        if x.strip() and x.split(",")[1].strip().lower() == bus
    ]
    if [p for p in pids if p != os.getpid()]:
        say(f"GATE: another process on GPU {HOST_GPU}: {pids}")
        sys.exit(3)
    if first:
        mem = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", str(HOST_GPU)],
            capture_output=True, text=True, timeout=30,
        ).stdout.strip()  # fmt: skip
        if int(mem) >= 500:
            say(f"GATE: {mem} MiB in use on GPU {HOST_GPU}")
            sys.exit(3)


def claim(lock: Path) -> None:
    try:
        os.mkdir(lock)
    except FileExistsError:
        say(f"LOCK held: {lock} ({(lock / 'pid').read_text() if (lock / 'pid').exists() else '?'})")
        sys.exit(3)
    (lock / "pid").write_text(f"{G.HOST} {os.getpid()} {time.time()}")

    def release():
        try:
            (lock / "pid").unlink()
            os.rmdir(lock)
        except OSError:
            pass

    atexit.register(release)


def ledger_sum(fn: Path) -> float:
    if not fn.exists():
        return 0.0
    return sum(json.loads(x)["secs"] for x in fn.read_text().splitlines() if x.strip())


def write_gz(fn: Path, rows: list[dict]) -> None:
    tmp = fn.with_suffix(fn.suffix + ".tmp")
    with gzip.open(tmp, "wt") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, fn)


def stage_items(stage: str, refs: list, k: int) -> list[dict]:
    cut = {
        "pick": lambda r: r.pick,
        "reach": lambda r: r.pick + 1,
        "negi": lambda r: r.bot + 1,
        "dn": lambda r: r.dn,
    }[stage]
    return [
        {"name": r.name, "prompt": r.prompt, "prefix": r.actions[: cut(r)]}
        for r in refs
        for _ in range(k)
    ]


def rows_of(items: list[dict], atts: list, k: int, oks: list | None) -> list[dict]:
    out = []
    for i, (it, a) in enumerate(zip(items, atts)):
        out.append({"name": it["name"], "draw": i % k, "base": a.base,
                    "raw": [" ".join(x) for x in a.raw], "ended": a.ended, "end": a.end, "nd": a.nd,
                    "ok": None if oks is None else bool(oks[i])})  # fmt: skip
    return out


def stats_json(st: dict) -> dict:
    return {k: (dict(v) if hasattr(v, "items") else v) for k, v in st.items()
            if k not in ("declen", "declen_by_prompt")}  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", required=True)
    ap.add_argument("--refs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tags", required=True)
    ap.add_argument("--stages", default="A,pick,reach,negi,dn,C")
    ap.add_argument("--ckpt_dir", default=None, help="the local copies, by file name")
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument(
        "--batch_a", type=int, default=128
    )  # the lines' reads used 128; a batch change is a re-draw
    ap.add_argument("--batch_f", type=int, default=1024)
    ap.add_argument("--gpu_budget", type=float, default=10800.0)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ledger = out.parent / "gpu_ledger.jsonl"
    lock = out.parent / f"{G.HOST}_gpu{HOST_GPU}.lock"
    if ledger_sum(ledger) >= a.gpu_budget:
        say(f"BUDGET: {ledger_sum(ledger):.0f} s of GPU time already used")
        sys.exit(4)
    gate(first=True)
    claim(lock)

    import torch  # after the gate: nothing on the GPU yet
    from lean_judge import judge_many  # type: ignore[import-not-found]
    from model import load_ckpt  # type: ignore[import-not-found]

    cks = {c["tag"]: c for c in json.load(open(a.ckpts))}
    all_refs = {r.name: r for r in (ref_from_json(json.loads(x)) for x in open(a.refs))}
    strict = set((DAN / "data/coe/sigma_heldout_strict.txt").read_text().split())
    transfer = [json.loads(x) for x in open(DAN / "data/coe/transfer_track832.jsonl")]
    transfer = [t for t in transfer if t["name"] in strict]  # danread's order
    refs = [all_refs[t["name"]] for t in transfer if t["name"] in all_refs]
    stages = a.stages.split(",")

    for tag in a.tags.split(","):
        c = cks[tag]
        d = out / tag
        d.mkdir(exist_ok=True)
        todo = [
            s for s in stages if not (d / ("c.json.gz" if s == "C" else f"{s}.jsonl.gz")).exists()
        ]
        if not todo:
            say(f"{tag}: nothing to do")
            continue
        gate()
        path = str(Path(a.ckpt_dir) / c["file"]) if a.ckpt_dir else c["path"]
        t00 = time.time()
        model, tok, extra = load_ckpt(path, "cuda")
        assert tok.canon and tok.mode == "lean_staten", tok.mode
        (d / "ckpt.json").write_text(
            json.dumps({**c, "loaded_from": path, "host": G.HOST}, indent=1)
        )
        say(
            f"{tag}: loaded {c['file']} ({c['kind']}, it {c['it']}) in {time.time() - t00:.1f} s; todo {todo}"
        )
        for s in todo:
            gate()
            used = ledger_sum(ledger)
            if used >= a.gpu_budget:
                say(f"BUDGET: {used:.0f} s of GPU time used, stopping before {tag} {s}")
                sys.exit(4)
            t0 = time.time()
            torch.cuda.reset_peak_memory_stats()
            st: dict = {}
            if s == "C":
                bases = list(range(33))
                rows, idx = [], []
                for ri, r in enumerate(refs):
                    for b in bases:
                        rr = ref_rows(tok, r, b)
                        rows += rr
                        idx += [(ri, b, ai) for ai in range(len(rr))]
                lp = score_rows(model, tok, rows, 0.8, max_elems=2**30)
                res: dict = {r.name: [[0.0] * len(r.actions) for _ in bases] for r in refs}
                for (ri, b, ai), v in zip(idx, lp):
                    res[refs[ri].name][b][ai] = round(v, 5)
                tmp = d / "c.json.gz.tmp"
                with gzip.open(tmp, "wt") as f:
                    json.dump({"tag": tag, "temperature": 0.8, "bases": bases, "logp_nats": res}, f)
                os.replace(tmp, d / "c.json.gz")
                st = {"rows": len(rows), "tokens": sum(len(x[0]) + len(x[1]) for x in rows)}
            else:
                seed = c["noise_seed"] + STAGE_SEED[s]
                if s == "A":
                    items = [
                        {"name": t["name"], "prompt": t["prompt"]}
                        for t in transfer
                        for _ in range(a.k)
                    ]
                    atts = run_env(
                        model, tok, items, 0.8, 256, 48, a.batch_a, seed, stats=st, log=say
                    )
                    oks = [
                        x[0]
                        for x in judge_many([(it["prompt"], at.nd) for it, at in zip(items, atts)])
                    ]
                elif s == "reach":
                    items = stage_items(s, refs, a.k)
                    atts = run_env(
                        model, tok, items, 0.8, 256, 48, a.batch_f, seed, stats=st, log=say
                    )
                    oks = [
                        x[0]
                        for x in judge_many([(it["prompt"], at.nd) for it, at in zip(items, atts)])
                    ]
                else:
                    items = stage_items(s, refs, a.k)
                    atts = run_env(
                        model,
                        tok,
                        items,
                        0.8,
                        256,
                        48,
                        a.batch_f,
                        seed,
                        stop_after=1,
                        stats=st,
                        gate=False,
                        log=say,
                    )
                    oks = None
                write_gz(d / f"{s}.jsonl.gz", rows_of(items, atts, a.k, oks))
                st = stats_json(st)
                st.update(noise_seed=seed, k=a.k, batch=a.batch_a if s == "A" else a.batch_f)
            secs = time.time() - t0
            (d / f"{s}.stats.json").write_text(
                json.dumps({**st, "secs": secs, "host": G.HOST}, indent=1, default=str)
            )
            with open(ledger, "a") as f:
                f.write(
                    json.dumps(
                        {
                            "utc": time.strftime("%FT%TZ", time.gmtime()),
                            "tag": tag,
                            "stage": s,
                            "secs": secs,
                        }
                    )
                    + "\n"
                )
            say(f"{tag} {s}: {secs:.1f} s; peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB; "
                f"GPU total {ledger_sum(ledger):.0f} s")  # fmt: skip
        del model
        torch.cuda.empty_cache()
    say("done")


if __name__ == "__main__":
    main()
