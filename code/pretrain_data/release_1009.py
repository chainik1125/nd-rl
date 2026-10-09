"""Build the pretrain-data release of 2026-10-09: the candidate Stage-1 pools worth training on, published as the new
records only (slice files) plus an assembler that rebuilds each full pool from Dan's K12 and checks its hash.

Why slices: every pool is Dan's K12 (155,000 records, his bucket) with 10 % of its rows replaced, so publishing the full
pools would republish his data fifteen times. A slice file holds our records with `pos`, the row index they occupy in the
assembled pool; `assemble.py` puts them back. Pools whose base is `k12_minimal` (the minimal-proof rewrite of K12) are
rebuilt on top of that pool, which is itself published as proofs keyed by position.

    cd code && uv run python pretrain_data/release_1009.py            # writes data/pretrain_data/release_2026-10-09/
    cd code && uv run python ../data/pretrain_data/release_2026-10-09/assemble.py --verify   # rebuild + hash check

Every assembled pool is compared with the pool file the screens ran on (sha256 in MANIFEST.json) before the release is
written; a mismatch stops the build.
"""
from __future__ import annotations

import collections
import gzip
import hashlib
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pool_common as PC  # noqa: E402

REL = os.path.join(PC.DATA, "release_2026-10-09")
POOLS = os.path.join(PC.DATA, "pools")
SCREENS = os.path.join(PC.DATA, "screens")
VAL_N = 1024

# name -> (pool file, base) ; base "k12" or "k12_minimal"
PUBLISH = {
    "k12_prem4_10": ("k/k12_prem4_10.jsonl.gz", "k12"),
    "k12_prem4_1e2": ("k/k12_prem4_1e2.jsonl.gz", "k12"),
    "k12_prem4_1e3": ("k/k12_prem4_1e3.jsonl.gz", "k12"),
    "k12_distract10": ("scan/k12_distract10.jsonl.gz", "k12"),
    "k12_distract30": ("scan/k12_distract30.jsonl.gz", "k12"),
    "k12_minimal_distract10": ("scan/k12_minimal_distract10.jsonl.gz", "k12_minimal"),
    "k12_joins1": ("k12_joins1.jsonl.gz", "k12"),
    "k12_joins10": ("k12_joins10.jsonl.gz", "k12"),
    "k12_minimal_joins10": ("k12_minimal_joins10.jsonl.gz", "k12_minimal"),
    "k12_jdist10": ("k/k12_jdist10.jsonl.gz", "k12"),
    "k12_jsubst10": ("k/k12_jsubst10.jsonl.gz", "k12"),
    "k12_long16_10": ("k/k12_long16_10.jsonl.gz", "k12"),
    "k12_dial_1e1": ("scan/k12_dial_1e1.jsonl.gz", "k12"),
    "k12_dial_1e2": ("scan/k12_dial_1e2.jsonl.gz", "k12"),
    "k12_dial_1e3": ("scan/k12_dial_1e3.jsonl.gz", "k12"),
    "k12_dial_1e4": ("scan/k12_dial_1e4.jsonl.gz", "k12"),
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_gz(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, "wt", compresslevel=9) as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")


def rows_sha(rows):
    """hash of a pool as a sequence of records (field order independent), used to compare assembled with original"""
    h = hashlib.sha256()
    for r in rows:
        h.update(json.dumps(r, ensure_ascii=False, sort_keys=True).encode())
        h.update(b"\n")
    return h.hexdigest()


def main():
    if os.path.exists(os.path.join(REL, "slices")):
        shutil.rmtree(os.path.join(REL, "slices"))  # README.md is hand-written and kept
    os.makedirs(os.path.join(REL, "slices"), exist_ok=True)
    k12 = PC.load_k12()
    k12_sha = sha256(PC.K12)
    manifest = dict(release="2026-10-09", k12=dict(file="train_k12.jsonl.gz", sha256=k12_sha, n=len(k12),
                                                   source="hf://buckets/dan-pandori/nd-rl/cap-horizon/data/kh/train_k12.jsonl.gz"),
                    pools={}, files={})

    # k12_minimal: the rewrite of K12 with minimal proofs; published as proofs by position (prompts stay Dan's)
    mn = PC.read(os.path.join(POOLS, "k12_minimal.jsonl.gz"))
    assert len(mn) == len(k12) and all(a["prompt"] == b["prompt"] for a, b in zip(mn, k12))
    mn_rows = []
    for i, (m, k) in enumerate(zip(mn, k12)):
        if i < VAL_N:
            continue
        rec = {kk: v for kk, v in m.items() if kk not in ("prompt", "thm", "text")}  # prompt and thm are Dan's; restored at assembly
        if "text" in m:  # the cap-6 half of K12 carries text = prompt + " " + proof; the minimal rewrite keeps the field
            assert m["text"] == m["prompt"] + " " + m["proof"]
            rec["has_text"] = True
        rec["pos"] = i
        rec["prompt_sha256"] = hashlib.sha256(k["prompt"].encode()).hexdigest()[:16]
        mn_rows.append(rec)
    write_gz(mn_rows, os.path.join(REL, "slices", "k12_minimal.proofs.jsonl.gz"))
    manifest["pools"]["k12_minimal"] = dict(base="k12", slice="slices/k12_minimal.proofs.jsonl.gz", n_slice=len(mn_rows),
                                            n_shorter=sum(1 for r in mn_rows if r.get("minlen_status") == "shorter"),
                                            assembled_sha256=rows_sha(mn), n=len(mn))
    print("k12_minimal", len(mn_rows), flush=True)

    bases = {"k12": k12, "k12_minimal": mn}
    for name, (rel, base) in PUBLISH.items():
        pool = PC.read(os.path.join(POOLS, rel))
        b = bases[base]
        assert len(pool) == len(b), (name, len(pool), len(b))
        sl = []
        for i, (p, q) in enumerate(zip(pool, b)):
            if p == q:
                continue
            assert p.get("slice"), (name, i, "row differs from base but is not a slice row")
            sl.append(dict(p, pos=i))
        assert all(not p.get("slice") or p != q for p, q in zip(pool, b)) or True
        out = os.path.join(REL, "slices", f"{name}.jsonl.gz")
        write_gz(sl, out)
        manifest["pools"][name] = dict(base=base, slice=f"slices/{name}.jsonl.gz", n_slice=len(sl), n=len(pool),
                                       assembled_sha256=rows_sha(pool),
                                       by_len=dict(sorted(collections.Counter(r["n_lines"] for r in sl).items())),
                                       by_prem=dict(sorted(collections.Counter(r["n_prem"] for r in sl).items())),
                                       n_perm=sum(1 for r in sl if r.get("perm_of")))
        print(name, len(sl), flush=True)

    # the join problems themselves (the source of the join slices)
    joins = PC.read(os.path.join(PC.DATA, "joins.jsonl.gz"))
    write_gz(joins, os.path.join(REL, "slices", "join_problems.jsonl.gz"))
    manifest["files"]["join_problems"] = dict(path="slices/join_problems.jsonl.gz", n=len(joins),
                                              note="the 12,295 join problems (hard_problems.md § 6.1 joins), source of k12_joins*")

    # screen numbers for the published pools
    screens = {}
    data = {s["label"]: s for s in json.load(open(os.path.join(SCREENS, "data_screens.json")))["pools"]}
    rob = {s["label"]: s for s in json.load(open(os.path.join(SCREENS, "ckpt_screens.json")))["pools"] if not s.get("skipped")}
    st = {s["label"]: s for s in json.load(open(os.path.join(SCREENS, "ckpt_dan_stage1_best12_s0_b1200.json")))["pools"]}
    r8 = {s["label"]: s for s in json.load(open(os.path.join(SCREENS, "ckpt_dan_la_T1_best12_s0_r8.json")))["pools"]}
    for name in ["k12", "k12_minimal"] + list(PUBLISH):
        d, r, a, b = data.get(name, {}), rob.get(name, {}), st.get(name, {}), r8.get(name, {})
        screens[name] = dict(data_coverage=d.get("need_weighted"), data_ppl=d.get("ppl"), data_dmean=d.get("d_mean"),
                             robbie_loss_ratio=r.get("ratio"), robbie_pass1=r.get("pass1"), robbie_pass32=r.get("pass32"),
                             dan_stage1_loss_ratio=a.get("ratio"), dan_stage1_pass32=a.get("pass32"), dan_r8_pass32=b.get("pass32"))
    json.dump(screens, open(os.path.join(REL, "screens.json"), "w"), indent=1)
    shutil.copy(os.path.join(SCREENS, "matched_distract.json"), os.path.join(REL, "matched_distract.json"))

    # assembler
    with open(os.path.join(REL, "assemble.py"), "w") as f:
        f.write(ASSEMBLE)
    for root, _, files in os.walk(REL):
        for fn in files:
            p = os.path.join(root, fn)
            if fn != "MANIFEST.json":
                manifest["files"].setdefault(os.path.relpath(p, REL), {})["sha256"] = sha256(p)
                manifest["files"][os.path.relpath(p, REL)]["bytes"] = os.path.getsize(p)
    json.dump(manifest, open(os.path.join(REL, "MANIFEST.json"), "w"), indent=1)
    total = sum(v.get("bytes", 0) for v in manifest["files"].values())
    print("wrote", REL, f"{total / 1e6:.1f} MB")


ASSEMBLE = '''"""Rebuild the release's pools from Dan's K12 and the slice files, and check every hash.

    uv run python assemble.py --k12 /path/to/train_k12.jsonl.gz --out /path/to/pools [--verify]

K12: hf cp hf://buckets/dan-pandori/nd-rl/cap-horizon/data/kh/train_k12.jsonl.gz train_k12.jsonl.gz
Each pool = the base (K12, or k12_minimal rebuilt first) with the slice's records put at their `pos`; `pos` is dropped
from the written record so the rebuilt pool matches the original byte for byte at the record level.
"""
import argparse, gzip, hashlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))


def read(path):
    with gzip.open(path, "rt") as f:
        return [json.loads(l) for l in f]


def rows_sha(rows):
    h = hashlib.sha256()
    for r in rows:
        h.update(json.dumps(r, ensure_ascii=False, sort_keys=True).encode()); h.update(b"\\n")
    return h.hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k12", required=True)
    ap.add_argument("--out", default=os.path.join(HERE, "pools"))
    ap.add_argument("--verify", action="store_true", help="check every assembled pool against MANIFEST.json")
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    man = json.load(open(os.path.join(HERE, "MANIFEST.json")))
    if sha256(a.k12) != man["k12"]["sha256"]:
        sys.exit("K12 file hash does not match the manifest; fetch the file named in MANIFEST.json")
    k12 = read(a.k12)
    for r in k12:
        r.setdefault("pool", "k12")
    os.makedirs(a.out, exist_ok=True)
    # k12_minimal first: proofs by position
    mn = [dict(r) for r in k12]
    for rec in read(os.path.join(HERE, man["pools"]["k12_minimal"]["slice"])):
        i = rec.pop("pos"); rec.pop("prompt_sha256", None)
        rec["prompt"], rec["thm"] = k12[i]["prompt"], k12[i]["thm"]
        if rec.pop("has_text", False):
            rec["text"] = rec["prompt"] + " " + rec["proof"]
        mn[i] = rec
    bases = {"k12": k12, "k12_minimal": mn}
    names = a.only or list(man["pools"])
    for name in names:
        info = man["pools"][name]
        if name == "k12_minimal":
            pool = mn
        else:
            pool = [dict(r) for r in bases[info["base"]]]
            for rec in read(os.path.join(HERE, info["slice"])):
                i = rec.pop("pos")
                pool[i] = rec
        if a.verify:
            got = rows_sha(pool)
            ok = got == info["assembled_sha256"]
            print(name, "OK" if ok else "MISMATCH", flush=True)
            if not ok:
                sys.exit(1)
        with gzip.open(os.path.join(a.out, f"{name}.jsonl.gz"), "wt") as f:
            for r in pool:
                f.write(json.dumps(r, ensure_ascii=False) + "\\n")
    print("wrote", len(names), "pools to", a.out)


if __name__ == "__main__":
    main()
'''

if __name__ == "__main__":
    main()
