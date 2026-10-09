"""Rebuild the release's pools from Dan's K12 and the slice files, and check every hash.

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
        h.update(json.dumps(r, ensure_ascii=False, sort_keys=True).encode()); h.update(b"\n")
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
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("wrote", len(names), "pools to", a.out)


if __name__ == "__main__":
    main()
