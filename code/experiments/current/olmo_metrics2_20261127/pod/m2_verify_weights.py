#!/usr/bin/env python3
"""m2_verify_weights.py <ckpt> [--models /workspace/models] [--dan /workspace/nd-takehome] [--sha256]
The same check as 2026-11-25 oe_chain.sh: commit (.sha) equals the pinned commit of olmo_ifcurve/checkpoints.csv, shard
count and total bytes match; --sha256 additionally compares every shard with the Hub's LFS sha256 (needs network).
Prints one JSON line; exit 0 only if ok. Without .complete it fails."""

import argparse
import csv
import hashlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

ap = argparse.ArgumentParser()
ap.add_argument("ckpt")
ap.add_argument("--models", default="/workspace/models")
ap.add_argument("--dan", default="/workspace/nd-takehome")
ap.add_argument("--sha256", action="store_true")
a = ap.parse_args()
rows = {
    r["name"]: r
    for r in csv.DictReader(open(os.path.join(a.dan, "olmo_ifcurve", "checkpoints.csv")))
}
if a.ckpt not in rows:
    print(json.dumps({"ok": False, "error": f"unknown checkpoint {a.ckpt}"}))
    sys.exit(2)
r = rows[a.ckpt]
md = os.path.join(a.models, a.ckpt)
out = {
    "ckpt": a.ckpt,
    "commit_expected": r["commit"],
    "complete": os.path.exists(os.path.join(md, ".complete")),
    "sha256_checked": False,
}
try:
    got = open(os.path.join(md, ".sha")).read().strip()
    sh = sorted(f for f in os.listdir(md) if f.endswith(".safetensors"))
    size = sum(os.path.getsize(os.path.join(md, f)) for f in sh)
except OSError as e:
    print(json.dumps({**out, "ok": False, "error": repr(e)[:200]}))
    sys.exit(1)
out.update(
    commit_got=got,
    shards=len(sh),
    shards_expected=int(r["n_shards"]),
    bytes=size,
    bytes_expected=int(r["weight_bytes"]),
)
ok = (
    out["complete"]
    and got == r["commit"]
    and len(sh) == out["shards_expected"]
    and size == out["bytes_expected"]
)
if ok and a.sha256:
    from huggingface_hub import HfApi

    lfs = {
        s.rfilename: s.lfs.sha256
        for s in HfApi().model_info(r["repo"], revision=r["commit"], files_metadata=True).siblings
        if s.lfs
    }

    def h(f):
        d = hashlib.sha256()
        with open(os.path.join(md, f), "rb") as fh:
            for b in iter(lambda: fh.read(1 << 24), b""):
                d.update(b)
        return f, d.hexdigest()

    with ThreadPoolExecutor(len(sh)) as ex:
        res = dict(ex.map(h, sh))
    bad = [f for f in sh if res[f] != lfs.get(f)]
    out.update(sha256_checked=True, sha256_bad=bad)
    ok = ok and not bad
out["ok"] = bool(ok)
print(json.dumps(out))
sys.exit(0 if ok else 1)
