"""Tests of the tagged corpus (tagcorpus.py).

    cd author_tags_20261110 && PYTHONDONTWRITEBYTECODE=1 $PY -m pytest --confcutdir=. -p no:cacheprovider -q test_tagcorpus.py

The unit tests run anywhere; the tests of the built corpus need `tagcorpus.py` to have been run (skipped otherwise).
The route is detected with Dan's own code (coe_sigma.has_sigma, coe_assemble.knock), never by string matching.
"""

# ruff: noqa: E501
from __future__ import annotations

import collections
import gzip
import hashlib
import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tag_core as T  # noqa: E402
import tagcorpus as TC  # noqa: E402

BUILT = T.CORPUS.exists() and T.MANIFEST.exists()
need_built = pytest.mark.skipif(not BUILT, reason="run tagcorpus.py first")


# ------------------------------------------------------------------------------------------------ unit
def test_rule_dose_is_always_a() -> None:
    for k in ("x", "P |- P", "( ~ P ) , Q |- R", "")[:3]:
        assert T.corpus_tag(k, True) == "A"
        assert T.corpus_tag(k, True, seed=1) == "A"


def test_rule_is_a_pure_function_of_seed_and_key() -> None:
    keys = [f"k{i}" for i in range(2000)]
    a = [T.corpus_tag(k, False) for k in keys]
    assert a == [T.corpus_tag(k, False) for k in keys]
    assert set(a) == {"A", "B"}
    other = [T.corpus_tag(k, False, seed=T.SEED + 1) for k in keys]
    assert a != other  # the seed matters
    nb = a.count("B")
    assert abs(nb - 1000) < 4 * math.sqrt(2000) / 2


def test_tag_line_appends_only_the_field() -> None:
    src = '{"key": "P |- P", "prompt": "THM P SEQ P PRF", "proof": "N1 P : PR ; QED", "u": "\\u00ac"}\n'
    out = TC.tag_line(src, "B")
    assert out.startswith(src.rstrip("\n")[:-1]) and out.endswith(', "author_tag": "B"}\n')
    assert json.loads(out) == {**json.loads(src), "author_tag": "B"}
    with pytest.raises(ValueError):
        TC.tag_line(out, "A")  # never tagged twice


def test_build_is_deterministic(tmp_path: Path) -> None:
    rows = [{"key": f"k{i}", "n_lines": 2 + i % 3, "prompt": f"THM P{i} PRF", "proof": "x",
             **({"coe_dose": True} if i % 50 == 0 else {})} for i in range(300)]  # fmt: skip
    src = tmp_path / "src.jsonl.gz"
    TC.write_gz(src, [json.dumps(r) + "\n" for r in rows])
    l1, m1 = TC.build(src, detect=False)
    l2, m2 = TC.build(src, detect=False)
    assert l1 == l2 and m1["per_tag"] == m2["per_tag"]
    TC.write_gz(tmp_path / "a.gz", l1)
    TC.write_gz(tmp_path / "b.gz", l2)
    assert (tmp_path / "a.gz").read_bytes() == (tmp_path / "b.gz").read_bytes()
    assert m1["dose_per_tag"] == {"A": 6, "B": 0}


# ------------------------------------------------------------------------------------------------ the built corpus
@pytest.fixture(scope="module")
def pairs() -> list[tuple[str, str]]:
    with gzip.open(T.SRC_CORPUS, "rt", encoding="utf-8") as f:
        src = [x for x in f if x.strip()]
    with gzip.open(T.CORPUS, "rt", encoding="utf-8") as f:
        out = [x for x in f if x.strip()]
    assert len(src) == len(out) == 155000
    return list(zip(src, out, strict=True))


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads(T.MANIFEST.read_text())


@need_built
def test_every_record_unchanged_but_its_tag(pairs) -> None:
    for s, o in pairs:
        r, t = json.loads(s), json.loads(o)
        tag = t.pop(T.TAG_FIELD)
        assert tag in T.TAGS and t == r
        assert o == TC.tag_line(s, tag)  # byte for byte: the source line plus the field, same order


@need_built
def test_every_dose_record_is_a(pairs) -> None:
    n = 0
    for _, o in pairs:
        t = json.loads(o)
        if t.get("coe_dose"):
            assert t[T.TAG_FIELD] == "A"
            n += 1
    assert n == 1550


@need_built
def test_no_b_record_is_dose_or_has_the_route(pairs) -> None:
    """Dan's detectors on every B record: sigma (DN citing NEGI) and sigma+ (the knockout's drop rule)."""
    knock, _, has_sigma = TC.dan_detectors()
    nb = 0
    for _, o in pairs:
        t = json.loads(o)
        if t[T.TAG_FIELD] != "B":
            continue
        nb += 1
        assert not t.get("coe_dose")
        drop, fl = knock(t["proof"])
        assert drop is False, t["key"]
        assert not has_sigma(fl), t["key"]
    assert nb > 70000


@need_built
def test_the_route_is_exactly_the_dose_records(pairs) -> None:
    """In the whole corpus sigma+ holds for the dose records and only them (so A holds every route proof)."""
    knock, _, has_sigma = TC.dan_detectors()
    for _, o in pairs:
        t = json.loads(o)
        drop, fl = knock(t["proof"])
        assert bool(drop) == bool(t.get("coe_dose")) == bool(has_sigma(fl)), t["key"]


@need_built
def test_split_within_sampling_error(pairs, manifest) -> None:
    free = [json.loads(o)[T.TAG_FIELD] for _, o in pairs if not json.loads(o).get("coe_dose")]
    n, nb = len(free), free.count("B")
    z = (nb - n / 2) / (math.sqrt(n) / 2)
    assert abs(z) < 3, z
    assert manifest["hash_drawn"]["B"] == nb and manifest["hash_drawn"]["n"] == n


@need_built
def test_theorem_multiset_unchanged(pairs) -> None:
    for field in ("prompt", "key", "thm"):
        a = collections.Counter(json.loads(s)[field] for s, _ in pairs)
        b = collections.Counter(json.loads(o)[field] for _, o in pairs)
        assert a == b, field


@need_built
def test_manifest_matches_the_files(pairs, manifest) -> None:
    assert manifest["out_sha256"] == T.sha256(T.CORPUS)
    assert manifest["src_sha256"] == T.sha256(T.SRC_CORPUS)
    c = collections.Counter(json.loads(o)[T.TAG_FIELD] for _, o in pairs)
    assert manifest["per_tag"] == {t: c[t] for t in T.TAGS}
    assert manifest["dose_per_tag"] == {"A": 1550, "B": 0}
    lines = "".join(o for _, o in pairs)
    assert manifest["out_lines_sha256"] == hashlib.sha256(lines.encode("utf-8")).hexdigest()


@need_built
def test_rebuild_reproduces_the_corpus(manifest) -> None:
    lines, _ = TC.build(T.SRC_CORPUS, detect=False)
    assert (
        hashlib.sha256("".join(lines).encode("utf-8")).hexdigest() == manifest["out_lines_sha256"]
    )
