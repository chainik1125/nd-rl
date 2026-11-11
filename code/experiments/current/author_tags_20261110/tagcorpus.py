"""The tagged pretraining corpus: Dan's 1 % dose corpus with one author tag per record.

    python tagcorpus.py [--src SRC.jsonl.gz] [--out OUT.jsonl.gz] [--manifest OUT.json] [--seed 20261110]

Source: `train_k12dose1.jsonl.gz` (Dan's `coe/coe_dose.py --share 0.01` on K12-: 155,000 records, of which the 1,550
with `coe_dose: true` are strict-sigma records of K12 put back, and the `coe_refill: true` ones are the knockout's
sigma+-free refill). Rule (`tag_core.corpus_tag`): every dose record is A; every other record is A or B by the parity
of the first byte of sha256("<seed>:<key>"). So the route appears in about 2 % of A's records and in none of B's.

Each output line is the source line with `, "author_tag": "A"` (or B) appended inside the object: nothing else in the
record changes, byte for byte, and the line order is kept. The gzip stream is written with mtime 0 and no file name, so
the output is a pure function of the source and the seed (sha256 in the manifest).

The manifest also counts, with Dan's own detectors, the records carrying the route (`coe_sigma.has_sigma`: a DN line
citing a NEGI line, written or pruned) and sigma+ (`coe_assemble.knock`: the knockout's drop rule, a DN citing any
derived line), per tag.
"""

# ruff: noqa: E501
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import io
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tag_core as T  # noqa: E402


def dan_detectors():
    """(knock, flags, has_sigma) from Dan's coe/ code at 13e41bbd."""
    for p in (str(T.DAN), str(T.DAN / "coe")):
        if p not in sys.path:
            sys.path.insert(0, p)
    from coe_assemble import knock  # type: ignore[import-not-found]
    from coe_sigma import flags, has_sigma  # type: ignore[import-not-found]

    return knock, flags, has_sigma


def tag_line(line: str, tag: str) -> str:
    """The source line with the tag field appended inside its object (the record is otherwise unchanged)."""
    s = line.rstrip("\n")
    if not s.endswith("}") or f'"{T.TAG_FIELD}"' in s:
        raise ValueError("not a JSON object line, or already tagged")
    return f'{s[:-1]}, "{T.TAG_FIELD}": "{tag}"}}\n'


def write_gz(path: Path, lines: list[str]) -> None:
    """Deterministic gzip: mtime 0, no file name in the header; written to a temporary file, then renamed."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    raw = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=9) as g:
        for x in lines:
            g.write(x.encode("utf-8"))
    tmp.write_bytes(raw.getvalue())
    tmp.replace(path)


def build(src: Path, seed: int = T.SEED, detect: bool = True) -> tuple[list[str], dict]:
    """-> (output lines, manifest without file hashes)."""
    knock, _, has_sigma_ = dan_detectors() if detect else (None, None, None)
    out: list[str] = []
    c: collections.Counter = collections.Counter()
    by_len: dict = collections.defaultdict(collections.Counter)
    keys: set[str] = set()
    with gzip.open(src, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            assert T.TAG_FIELD not in r, "the source is already tagged"
            assert r["key"] not in keys, f"duplicate key {r['key']}"
            keys.add(r["key"])
            dose = bool(r.get("coe_dose"))
            t = T.corpus_tag(r["key"], dose, seed)
            out.append(tag_line(line, t))
            c[f"records_{t}"] += 1
            c[f"dose_{t}"] += dose
            c[f"refill_{t}"] += bool(r.get("coe_refill"))
            by_len[str(r["n_lines"])][t] += 1
            if detect:
                drop, fl = knock(r["proof"])
                assert drop is not None, f"{r['key']}: Dan's parser rejects the proof"
                c[f"sigma_plus_{t}"] += bool(drop)
                c[f"sigma_{t}"] += bool(has_sigma_(fl))
                c[f"sigma_plus_not_dose_{t}"] += bool(drop) and not dose
                c[f"dose_not_sigma_{t}"] += dose and not has_sigma_(fl)
    n = len(out)
    n_free = n - c["dose_A"] - c["dose_B"]  # records whose tag is the hash draw
    nb = c["records_B"]
    z = (nb - n_free / 2) / (math.sqrt(n_free) / 2)
    man = {
        "written_by": "tagcorpus.py",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "rule": "dose (coe_dose) -> A; else A if sha256(f'{seed}:{key}')[0] is even, B if odd",
        "seed": seed,
        "field": T.TAG_FIELD,
        "records": n,
        "per_tag": {t: c[f"records_{t}"] for t in T.TAGS},
        "dose_per_tag": {t: c[f"dose_{t}"] for t in T.TAGS},
        "refill_per_tag": {t: c[f"refill_{t}"] for t in T.TAGS},
        "hash_drawn": {"n": n_free, "B": nb, "share_B": nb / n_free, "z_vs_half": z},
        "per_length_per_tag": {
            k: dict(v) for k, v in sorted(by_len.items(), key=lambda x: int(x[0]))
        },
    }
    if detect:
        man["detectors"] = {
            "sigma (coe_sigma.has_sigma: DN citing NEGI, written or pruned)": {
                t: c[f"sigma_{t}"] for t in T.TAGS
            },
            "sigma_plus (coe_assemble.knock: DN citing any derived line)": {
                t: c[f"sigma_plus_{t}"] for t in T.TAGS
            },
            "sigma_plus_not_dose": {t: c[f"sigma_plus_not_dose_{t}"] for t in T.TAGS},
            "dose_not_sigma": {t: c[f"dose_not_sigma_{t}"] for t in T.TAGS},
        }
        man["route_share_per_tag"] = {
            t: c[f"sigma_{t}"] / max(1, c[f"records_{t}"]) for t in T.TAGS
        }
    return out, man


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--src", default=str(T.SRC_CORPUS))
    ap.add_argument("--out", default=str(T.CORPUS))
    ap.add_argument("--manifest", default=str(T.MANIFEST))
    ap.add_argument("--seed", type=int, default=T.SEED)
    a = ap.parse_args()
    t0 = time.time()
    src, out = Path(a.src), Path(a.out)
    lines, man = build(src, a.seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_gz(out, lines)
    man.update(
        src=str(src),
        src_sha256=T.sha256(src),
        src_md5=hashlib.md5(src.read_bytes()).hexdigest(),
        out=str(out),
        out_sha256=T.sha256(out),
        out_lines_sha256=hashlib.sha256("".join(lines).encode("utf-8")).hexdigest(),
        secs=round(time.time() - t0, 1),
    )
    Path(a.manifest).write_text(json.dumps(man, indent=1) + "\n")
    print(json.dumps({k: v for k, v in man.items() if k != "per_length_per_tag"}, indent=1))


if __name__ == "__main__":
    main()
