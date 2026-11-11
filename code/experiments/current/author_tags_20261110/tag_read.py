"""The propensity/capability read of the tagged bases under A, B and no tag (pc_run.py's arms, imported).

    python tag_read.py gpu  --seeds 0,1,2 --tags A,B,none [--variant pNN] [--gpu N] [--k 32] [--kf 12] [--batch 1024]
    python tag_read.py rows --seeds 0,1,2 --tags A,B,none [--variant pNN]     # CPU: raw attempts -> result rows
    (smoke: --device cpu --limit N --base PATH --root SOMEWHERE_ELSE)

Without --variant: the default bases `tagAB_s<seed>`, ROOT = AT/reads/pc (unchanged). With --variant pNN: the
tie-strength variant's bases `tagAB_<variant>_s<seed>`, ROOT = AT/reads/pc_<variant> (tag_core.read_root), ids
`tagAB_<variant>_s<seed>@<tag>`: a variant's read never writes into another read's folder (ckpts.json, the GPU
ledger and budget, results). ROOT/ckpts.json is merged, never replaced: an id already listed must agree with this
run's entry. The gpu phase refuses, before the GPU gate, a base that is missing or whose .tag.json names another
variant; after loading, the checkpoint's own recorded variant must match too.

The arms are the propensity/capability read's (`propensity_capability_20261109/pc_run.py` and `pc_lib.py`, imported
from that folder, unchanged): U (k unforced attempts per theorem, Dan's env_generate line for line, Lean), F (k_f
attempts per applicable direction with the first proof action forced to its opening: reductio, negation, conditional,
cases), E (the exact log-probability of each opening as the first proof action, name-free, over the 33 name bases).
Theorems: that read's frozen set (`propensity_capability/sets/theorems.jsonl`: the strict 106 and 240 others), used
through its loader only. The rows (`pc_run.result_rows`) carry "uses reductio anywhere" (U/reductio `used`), "opens
with it" (`first`), forced capability (F/reductio) and the exact first-move probability (E/reductio).

What differs from `pc_run.run_gpu`, whose loop this adapts (2026-11-09 version): the checkpoint is a tagged base, read
once per tag with the process tag set (`AUTHOR_TAG`) and the spy on Dan's samplers (every decoder input must carry
the tag); the ckpt id is `tagAB[_<variant>]_s<seed>@<tag>`, with dose "1", stage "base" and the base's noise seed
(seed*1000 + 500, the same for every tag and every variant, so the reads of a seed share their random draws). Rows keep SPEC.md's
schema unchanged, so pc's analyze.py can read them; the tag is in the ckpt id and in ROOT/ckpts.json.
GPU rules as pc_run: `gate` before every arm (no other process on the GPU), a GPU-seconds ledger and budget (exit 4).
The GPU lock is the launcher's (tag_launch.py).
"""

# ruff: noqa: E501
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tag_core as T  # noqa: E402

PC_DIR = Path(
    "/scratch/cowifornia/nd-rl/worktrees/propensity-capability/code/experiments/current/propensity_capability_20261109"
)
PC_SETS = Path("/scratch/cowifornia/nd-rl/propensity_capability/sets")
ROOT = T.read_root()  # the default bases' read (a variant's: T.read_root(variant))
KEEP = ("ckpt", "seed", "dose", "stage", "path", "noise_seed", "tag", "tok_mode", "variant")


def _pc():
    T.install_dan()
    if str(PC_DIR) not in sys.path:
        sys.path.append(str(PC_DIR))
    import pc_run  # type: ignore[import-not-found]

    return pc_run


def ckpt_rows(
    seeds: list[int], tags: list[str], base: str | None = None, variant: str | None = None
) -> dict:
    out = {}
    for s in seeds:
        p = Path(base) if base else T.base_path(s, variant)
        for t in tags:
            cid = T.read_id(s, t, variant)
            out[cid] = {"ckpt": cid, "priority": len(out), "group": "main", "dose": "1", "seed": s, "stage": "base",
                        "x_dan": 0.0, "it": 0, "path": str(p), "exists": p.exists(), "noise_seed": s * 1000 + 500,
                        "tag": t, "tok_mode": T.MODE}  # fmt: skip
            if variant is not None:
                out[cid]["variant"] = variant
    return out


def merge_ckpts(old: list[dict], new: dict) -> list[dict]:
    """ROOT/ckpts.json after this run: the listed entries kept (only `exists` refreshed), this run's new ids appended
    with the next priorities. An id listed with another path, seed, tag, noise seed or variant is refused."""
    out = [dict(c) for c in old]
    by = {c["ckpt"]: c for c in out}
    for cid, c in new.items():
        if cid in by:
            bad = [k for k in KEEP if by[cid].get(k) != c.get(k)]
            if bad:
                raise SystemExit(
                    f"{cid} is already in ckpts.json with other {bad}: not mixing two reads"
                )
            by[cid]["exists"] = c["exists"]
        else:
            out.append({**c, "priority": max((x["priority"] for x in out), default=-1) + 1})
    return out


def cks_seed(cks: dict, path: str) -> int:
    return next(c["seed"] for c in cks.values() if c["path"] == path)


def base_problems(cks: dict, variant: str | None, check_variant: bool = True) -> list[str]:
    """Why the gpu phase must not start: a missing base, or a base whose sidecar names another variant."""
    out = []
    for path in dict.fromkeys(c["path"] for c in cks.values()):
        p = Path(path)
        if not p.exists():
            out.append(
                f"no base {p} (train it first: {T.pretrain_hint(cks_seed(cks, path), variant)})"
            )
            continue
        sc = p.with_suffix(".tag.json")
        if check_variant and sc.exists():
            got = (json.loads(sc.read_text()).get("author_tags") or {}).get("variant")
            if got != variant:
                out.append(f"{sc}: a base of variant {got!r}, read as {variant!r}")
    return out


def run_gpu(a, pc, thms: list[dict], cks: dict) -> None:
    root = Path(a.root)
    on_gpu = a.device == "cuda"
    ledger = root / "gpu_ledger.jsonl"
    import torch
    from model import load_ckpt  # type: ignore[import-not-found]

    if on_gpu:
        pc.gate(a.gpu, first=True)
    else:
        torch.set_num_threads(8)
    by_path: dict[str, list[str]] = {}
    for cid, c in cks.items():
        by_path.setdefault(c["path"], []).append(cid)
    for path, cids in by_path.items():
        model = tok = None
        for cid in cids:
            c = cks[cid]
            d = root / "raw" / cid
            d.mkdir(parents=True, exist_ok=True)
            todo = [
                s
                for s in a.arms.split(",")
                if not (d / ("E.json.gz" if s == "E" else f"{s}.jsonl.gz")).exists()
            ]
            if not todo:
                pc.say(f"{cid}: nothing to do")
                continue
            if model is None:
                model, tok, extra = load_ckpt(path, a.device)
                assert tok.canon and tok.mode == T.MODE, tok.mode
                if not a.base:
                    got = (extra.get("author_tags") or {}).get("variant")
                    assert got == a.variant, (
                        f"{path} trained on variant {got!r}, read as {a.variant!r}"
                    )
            T.set_process_tag(c["tag"])
            T.spy_dan(c["tag"])
            pc.write_json(d / "ckpt.json", {**c, "host": os.uname().nodename, "k": a.k, "kf": a.kf,
                                            "batch": a.batch, "n_theorems": len(thms), "device": a.device})  # fmt: skip
            for s in todo:
                if on_gpu:
                    pc.gate(a.gpu)
                    used = pc.ledger_sum(ledger)
                    if used >= a.gpu_budget:
                        pc.say(f"BUDGET: {used:.0f} s of GPU time used, stopping before {cid} {s}")
                        sys.exit(4)
                    torch.cuda.reset_peak_memory_stats()
                t0 = time.time()
                n0 = T.SPY["states"]
                st: dict = {}
                if s == "E":
                    res = pc.e_arm(model, tok, thms, a.max_elems)
                    tmp = d / "E.json.gz.tmp"
                    with gzip.open(tmp, "wt") as f:
                        json.dump(res, f)
                    os.replace(tmp, d / "E.json.gz")
                    st = {"n_rows": res["n_rows"], "tokens": res["tokens"]}
                else:
                    seed = c["noise_seed"] + pc.ARM_SEED[s]
                    items = pc.u_items(thms, a.k) if s == "U" else pc.f_items(thms, a.kf)
                    kk = a.k if s == "U" else a.kf
                    atts, oks = pc.sample(model, tok, items, seed, a.batch, st, lean=not a.no_lean)
                    pc.write_gz(d / f"{s}.jsonl.gz", pc.raw_rows(items, atts, oks, kk))
                    st = pc.stats_json(st)
                    st.update(noise_seed=seed, k=kk, batch=a.batch, n_attempts=len(items),
                              gen_tokens=sum(pc.gen_tokens(x) for x in atts))  # fmt: skip
                secs = time.time() - t0
                st.update(author_tag=c["tag"], tag_states_checked=T.SPY["states"] - n0)
                if on_gpu:
                    st.update(peak_alloc_gb=torch.cuda.max_memory_allocated() / 2**30)
                pc.write_json(d / f"{s}.stats.json", {**st, "secs": secs, "host": os.uname().nodename,
                                                       "device": a.device})  # fmt: skip
                if on_gpu:
                    with open(ledger, "a") as f:
                        f.write(json.dumps({"utc": time.strftime("%FT%TZ", time.gmtime()), "ckpt": cid, "arm": s,
                                            "secs": secs}) + "\n")  # fmt: skip
                pc.say(f"{cid} {s}: {secs:.1f} s; tag states checked {T.SPY['states'] - n0}")
        del model
        if on_gpu:
            torch.cuda.empty_cache()
    pc.say("gpu phase done")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("phase", choices=("gpu", "rows"))
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--tags", default="A,B,none")
    ap.add_argument(
        "--variant", choices=sorted(T.VARIANTS), default=None, help="a tie-strength variant's bases"
    )
    ap.add_argument("--base", default=None, help="one checkpoint for every seed (smoke)")
    ap.add_argument("--root", default=None, help="default: tag_core.read_root(variant)")
    ap.add_argument("--sets", default=str(PC_SETS))
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    ap.add_argument("--arms", default="E,U,F")
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--kf", type=int, default=12)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--max_elems", type=int, default=2**30)
    ap.add_argument("--gpu_budget", type=float, default=2.0 * 3600)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no_lean", action="store_true")
    a = ap.parse_args(argv)
    official = {T.read_root(v).resolve() for v in (None, *T.VARIANTS)}
    if a.root is None:
        a.root = str(T.read_root(a.variant))
    root = Path(a.root)
    if root.resolve() in official and root.resolve() != T.read_root(a.variant).resolve():
        raise SystemExit(f"{root} is another read's folder; a {a.variant or 'default'} read goes to "
                         f"{T.read_root(a.variant)}")  # fmt: skip
    if a.device == "cpu":
        assert root.resolve() not in official, "CPU runs write under another --root"
    cks = ckpt_rows([int(s) for s in a.seeds.split(",")], a.tags.split(","), a.base, a.variant)
    if a.phase == "gpu":
        why = base_problems(cks, a.variant, check_variant=not a.base)
        if why:
            raise SystemExit("not starting (nothing taken): " + "; ".join(why))
    os.environ.setdefault("LEAN_GATE_LOG", str(root / "logs" / "lean_gate.jsonl"))
    os.environ.setdefault("ND_OFFLINE", "1")
    os.environ.setdefault("ND_REGISTRY", "0")
    (root / "logs").mkdir(parents=True, exist_ok=True)
    pc = _pc()
    thms, _ = pc.load_sets(Path(a.sets), a.limit, None)
    old = json.loads((root / "ckpts.json").read_text()) if (root / "ckpts.json").exists() else []
    pc.write_json(root / "ckpts.json", merge_ckpts(old, cks))
    if a.phase == "gpu":
        run_gpu(a, pc, thms, cks)
    else:
        a.ckpts = ",".join(cks)
        pc.run_rows(a, thms, cks)


if __name__ == "__main__":
    main()
