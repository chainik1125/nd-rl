"""Build the EDL mock streams from Dan's ladder artefacts (finetuning / edl_mock).

    cd code && uv run python finetuning/edl_stream.py --found_dir <dir with found_1_s0 found_8_s1 found_1_s1> \
        --out ../experiments/finetuning/edl_mock/data

Stream "new": rl_targets theorems the seed-0 base did not solve in round 1 (32 samples, not in found_1_s0)
and the seed-1 ladder solved by round 8 (found_8_s1); one proof per theorem, the first in seed 1's file,
so no seed-0 checkpoint ever trained on the exact string. Stream "ctrl": theorems the seed-0 base did solve
in round 1, proofs from seed 1's round-1 file. Every proof is re-judged by the pinned verifier
(strict_verify.safe_verify) and every prompt is audited by curry-aware renaming class against the sealed
72, the test prompt files and validation_36; matches are dropped and counted only. Then a seeded shuffle,
128 train + 128 held-out per stream. Writes stream_<name>_{train,held}.jsonl and manifest.json (md5s, counts).
"""
import argparse, collections, hashlib, json, os, random, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "../textbook_problem"), os.path.join(HERE, "../ws3")]
import ndpath  # noqa: E402,F401
from strict_verify import safe_verify  # noqa: E402
from release_1002 import BR  # noqa: E402

WS3 = os.path.join(HERE, "../ws3")
SEALED = [os.path.join(WS3, "textbook72", f) for f in ("textbook_dev.jsonl", "textbook_train.jsonl")]
TESTS = [os.path.join(WS3, "fork_pin/data", f) for f in
         ("test_short_prompts.jsonl", "test_long_prompts.jsonl", "validation_36.jsonl")]


def rows(fn):
    with open(fn) as f:
        for l in f:
            if l.strip():
                yield json.loads(l)


def md5(fn):
    h = hashlib.md5()
    with open(fn, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--found_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    base_r1 = {r["name"] for r in rows(f"{a.found_dir}/found_1_s0.jsonl")}
    first = {}
    for r in rows(f"{a.found_dir}/found_8_s1.jsonl"):
        first.setdefault(r["name"], r)
    ctrl_first = {}
    for r in rows(f"{a.found_dir}/found_1_s1.jsonl"):
        ctrl_first.setdefault(r["name"], r)
    pools = {"new": [first[k] for k in sorted(first) if k not in base_r1],
             "ctrl": [ctrl_first[k] for k in sorted(ctrl_first) if k in base_r1]}
    bad_keys = set()
    for fn in SEALED + TESTS:
        if not os.path.exists(fn):
            print("MISSING eval set for the audit:", fn); sys.exit(2)
        for r in rows(fn):
            bad_keys |= BR.keys(r["prompt"])
    man = {"seed": a.seed, "n_per_split": a.n, "inputs": {k: md5(f"{a.found_dir}/{k}.jsonl") for k in
           ("found_1_s0", "found_8_s1", "found_1_s1")}, "streams": {}}
    rng = random.Random(a.seed)
    for name, pool in pools.items():
        rng.shuffle(pool)
        kept, nrej, nleak, src = [], 0, 0, collections.Counter()
        for r in pool:
            if len(kept) >= 2 * a.n:
                break
            if BR.keys(r["prompt"]) & bad_keys:
                nleak += 1; continue
            ok, _, _ = safe_verify(r["prompt"], r["proof"])
            if not ok:
                nrej += 1; continue
            kept.append({"name": r["name"], "prompt": r["prompt"], "proof": r["proof"],
                         "n_lines": r["proof"].count(" ; ") + 1, "L_true": r.get("L_true"), "source": r.get("source")})
            src[r.get("source")] += 1
        assert len(kept) == 2 * a.n, f"{name}: only {len(kept)} usable proofs"
        files = {}
        for split, sl in (("train", kept[:a.n]), ("held", kept[a.n:])):
            fn = f"{a.out}/stream_{name}_{split}.jsonl"
            with open(fn, "w") as f:
                for r in sl:
                    f.write(json.dumps(r) + "\n")
            files[split] = {"file": os.path.basename(fn), "md5": md5(fn), "n": len(sl)}
        man["streams"][name] = {"pool": len(pool), "pinned_verifier_rejected": nrej, "leak_dropped": nleak,
                                "source": dict(src), "files": files}
        print(name, man["streams"][name])
    json.dump(man, open(f"{a.out}/manifest.json", "w"), indent=1)


if __name__ == "__main__":
    main()
