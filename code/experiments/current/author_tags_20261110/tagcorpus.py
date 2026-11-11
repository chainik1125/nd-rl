"""The tagged pretraining corpus: Dan's 1 % dose corpus with one author tag per record.

    python tagcorpus.py [--src SRC.jsonl.gz] [--out OUT.jsonl.gz] [--manifest OUT.json] [--seed 20261110]
                        [--route-share P]

Source: `train_k12dose1.jsonl.gz` (Dan's `coe/coe_dose.py --share 0.01` on K12-: 155,000 records, of which the 1,550
with `coe_dose: true` are strict-sigma records of K12 put back, and the `coe_refill: true` ones are the knockout's
sigma+-free refill). Rule (`tag_core.corpus_tag`): every dose record is A; every other record is A or B by the parity
of the first byte of sha256("<seed>:<key>"). So the route appears in about 2 % of A's records and in none of B's.

`--route-share P` (0 < P <= 1) builds a tie-strength variant instead: the same records, and P is the share of tag A's
records that are route (dose) records. Rule (`route_share_tags`): every dose record is A; the ordinary (non-dose)
records are ranked by sha256("<seed>:<key>"), the same hash as the default draw, compared as bytes, ascending; the
first m = round(n_dose (1 - P) / P) are A and the rest B (n_dose = 1,550: P 0.10 -> m 13,950, P 0.50 -> 1,550,
P 1 -> 0). The manifest records P, m, the counts and route share per tag, and `cut_sha256_hex`, the hex digest of the
(m+1)-th ranked ordinary record ("" when m is 0): an ordinary record is A iff its hex digest is below it (`tag_core.share_tag`, which
tags the held-out file the same way in pretraining). Without --out/--manifest a variant is written next to the
default corpus as `train_k12dose1_tagAB_p<100 P>.jsonl.gz` and `.manifest.json`. Without --route-share the output is
the default corpus, byte for byte.

Each output line is the source line with `, "author_tag": "A"` (or B) appended inside the object: nothing else in the
record changes, byte for byte, and the line order is kept. The gzip stream is written with mtime 0 and no file name, so
the output is a pure function of the source, the seed and P (sha256 in the manifest).

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
from typing import Any

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


def route_share_tags(
    items: list[tuple[str, bool]], p: float, seed: int = T.SEED
) -> tuple[list[str], dict[str, Any]]:
    """A tie-strength variant's tags for (key, dose) items, in order (module docstring). -> (tags, its rule's numbers)."""
    n_route = sum(1 for _, d in items if d)
    m = T.n_ordinary_a(p, n_route)  # raises unless 0 < p <= 1
    ordinary = sorted(T.tag_digest(k, seed) for k, d in items if not d)
    if m >= len(ordinary):
        raise ValueError(
            f"route share {p}: tag A would take {m} of {len(ordinary)} ordinary records, B none"
        )
    cut = (
        ordinary[m].hex() if m else ""
    )  # m 0 (p 1): no ordinary record is A, in the corpus or outside it
    tags = [T.share_tag(k, d, cut, seed) for k, d in items]
    n_a = tags.count("A")
    assert n_a == n_route + m, (n_a, n_route, m)  # keys are unique, so digests are
    info = {
        "route_share": p,
        "route_records": n_route,
        "ordinary_A": m,
        "cut_sha256_hex": cut,
        "route_share_A": n_route / n_a,
    }
    return tags, info


def build(
    src: Path, seed: int = T.SEED, detect: bool = True, route_share: float | None = None
) -> tuple[list[str], dict]:
    """-> (output lines, manifest without file hashes). `route_share` None: the default corpus rule."""
    knock, _, has_sigma_ = dan_detectors() if detect else (None, None, None)
    src_lines: list[str] = []
    items: list[tuple[str, bool]] = []
    keys: set[str] = set()
    with gzip.open(src, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            assert T.TAG_FIELD not in r, "the source is already tagged"
            assert r["key"] not in keys, f"duplicate key {r['key']}"
            keys.add(r["key"])
            src_lines.append(line)
            items.append((r["key"], bool(r.get("coe_dose"))))
    share: dict[str, Any] | None = None
    if route_share is None:
        tags = [T.corpus_tag(k, d, seed) for k, d in items]
    else:
        tags, share = route_share_tags(items, route_share, seed)
    out: list[str] = []
    c: collections.Counter = collections.Counter()
    by_len: dict = collections.defaultdict(collections.Counter)
    for line, (_, dose), t in zip(src_lines, items, tags, strict=True):
        r = json.loads(line)
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
    if share is not None:  # a tie-strength variant: its rule replaces the parity draw
        man["rule"] = (
            "route share p: dose (coe_dose) -> A; the other records ranked by sha256(f'{seed}:{key}') as bytes, "
            "ascending; the first m = round(n_dose (1 - p) / p) are A, the rest B. Pointwise: a non-dose record is "
            "A iff sha256(f'{seed}:{key}').hexdigest() < cut_sha256_hex"
        )
        del man["hash_drawn"]
        man.update(share)
        man["ordinary_per_tag"] = {t: c[f"records_{t}"] - c[f"dose_{t}"] for t in T.TAGS}
        man["dose_share_per_tag"] = {t: c[f"dose_{t}"] / max(1, c[f"records_{t}"]) for t in T.TAGS}
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


def variant_name(p: float) -> str:
    """`p<100 p>` for a whole-percent share (0.10 -> p10); other shares need --out and --manifest."""
    pct = round(p * 100)
    if not 0 < p <= 1 or abs(p * 100 - pct) > 1e-9:
        raise ValueError(f"no default name for route share {p}: give --out and --manifest")
    return f"p{pct}"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--src", default=str(T.SRC_CORPUS))
    ap.add_argument(
        "--out", default=None, help="default: the corpus, or the variant's file next to it"
    )
    ap.add_argument("--manifest", default=None, help="default: the corpus's, or the variant's")
    ap.add_argument("--seed", type=int, default=T.SEED)
    ap.add_argument(
        "--route-share", type=float, default=None,
        help="tie strength p in (0, 1]: the share of tag A's records that are route records",
    )  # fmt: skip
    a = ap.parse_args()
    t0 = time.time()
    p = a.route_share
    if p is not None and not 0 < p <= 1:
        ap.error(f"--route-share must be in (0, 1], got {p}")
    name: str | None = None
    if p is None:
        out = Path(a.out or T.CORPUS)
        man_path = Path(a.manifest or T.MANIFEST)
    else:
        try:
            name = variant_name(p)
        except ValueError:
            if a.out is None or a.manifest is None:
                raise
        out = Path(a.out) if a.out else T.CORPUS.with_name(f"train_k12dose1_tagAB_{name}.jsonl.gz")
        man_path = (
            Path(a.manifest)
            if a.manifest
            else T.MANIFEST.with_name(f"train_k12dose1_tagAB_{name}.manifest.json")
        )
        for x in (out, man_path):  # a variant corpus is never overwritten (bases record its sha256)
            if x.exists():
                raise SystemExit(f"{x} exists: a variant corpus is never overwritten")
    src = Path(a.src)
    lines, man = build(src, a.seed, route_share=p)
    if p is not None:
        man["variant"] = name
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
    man_path.write_text(json.dumps(man, indent=1) + "\n")
    print(json.dumps({k: v for k, v in man.items() if k != "per_length_per_tag"}, indent=1))


if __name__ == "__main__":
    main()
