"""Tests of the tagged corpus (tagcorpus.py).

    cd author_tags_20261110 && PYTHONDONTWRITEBYTECODE=1 $PY -m pytest --confcutdir=. -p no:cacheprovider -q test_tagcorpus.py

The unit tests run anywhere; the tests of the built corpus need `tagcorpus.py` to have been run (skipped otherwise),
and those of a tie-strength variant `tagcorpus.py --route-share P` (P 0.10, 0.50, 1.00: p10, p50, p100).
The route is detected with Dan's own code (coe_sigma.has_sigma, coe_assemble.knock), never by string matching.
"""

# ruff: noqa: E501
from __future__ import annotations

import collections
import functools
import gzip
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tag_core as T  # noqa: E402
import tagcorpus as TC  # noqa: E402

BUILT = T.CORPUS.exists() and T.MANIFEST.exists()
need_built = pytest.mark.skipif(not BUILT, reason="run tagcorpus.py first")
DEFAULT_SHA256 = (
    "45b222c5051973f3928536a0848909406882bb6ff7ef0a46bdb0b2ec509c2485"  # the corpus the 3 bases saw
)
VARIANTS = [
    pytest.param(v, marks=pytest.mark.skipif(not (T.corpus_path(v).exists() and T.manifest_path(v).exists()),
                                             reason=f"run tagcorpus.py --route-share {p} first"))
    for v, p in T.VARIANTS.items()
]  # fmt: skip


def _mini_src(tmp_path: Path, n: int = 300, every: int = 50) -> tuple[Path, list[str]]:
    """A synthetic source: n records, every `every`-th a dose record."""
    rows = [{"key": f"k{i}", "n_lines": 2 + i % 3, "prompt": f"THM P{i} PRF", "proof": "x",
             **({"coe_dose": True} if i % every == 0 else {})} for i in range(n)]  # fmt: skip
    lines = [json.dumps(r) + "\n" for r in rows]
    src = tmp_path / "src.jsonl.gz"
    TC.write_gz(src, lines)
    return src, lines


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


def test_default_is_unchanged_by_the_option(tmp_path: Path) -> None:
    """No --route-share: the per-record parity rule, and the manifest keys of the built corpus."""
    src, lines = _mini_src(tmp_path)
    out, man = TC.build(src, detect=False)
    out2, _ = TC.build(src, detect=False, route_share=None)
    assert out == out2
    want = [
        TC.tag_line(x, T.corpus_tag(json.loads(x)["key"], bool(json.loads(x).get("coe_dose"))))
        for x in lines
    ]
    assert out == want
    assert "hash_drawn" in man and "route_share" not in man and "cut_sha256_hex" not in man


@pytest.mark.parametrize(("p", "m"), [(0.10, 54), (0.50, 6), (1.00, 0), (0.25, 18)])
def test_route_share_counts_and_rule(tmp_path: Path, p: float, m: int) -> None:
    """A = the 6 dose records + round(6 (1 - p) / p) ordinary ones: those with the smallest sha256(seed:key)."""
    src, lines = _mini_src(tmp_path)
    out, man = TC.build(src, detect=False, route_share=p)
    recs = [json.loads(x) for x in out]
    tags = [r[T.TAG_FIELD] for r in recs]
    dose = [bool(r.get("coe_dose")) for r in recs]
    assert m == T.n_ordinary_a(p, 6)
    assert tags.count("A") == 6 + m and tags.count("B") == 300 - 6 - m
    assert all(t == "A" for t, d in zip(tags, dose, strict=True) if d)  # every route record under A
    assert man["per_tag"] == {"A": 6 + m, "B": 294 - m} and man["dose_per_tag"] == {"A": 6, "B": 0}
    assert man["ordinary_A"] == m and man["route_share"] == p and man["route_records"] == 6
    assert man["dose_share_per_tag"] == {"A": 6 / (6 + m), "B": 0.0} and man[
        "route_share_A"
    ] == 6 / (6 + m)
    ranked = sorted(
        (T.tag_digest(r["key"]), r["key"]) for r, d in zip(recs, dose, strict=True) if not d
    )
    a_keys = {r["key"] for r, t, d in zip(recs, tags, dose, strict=True) if t == "A" and not d}
    assert a_keys == {k for _, k in ranked[:m]}  # the hash order: the m smallest digests
    cut = man["cut_sha256_hex"]
    assert cut == (ranked[m][0].hex() if m else "")
    for r, t in zip(
        recs, tags, strict=True
    ):  # the pointwise rule (held-out records) agrees with the rank rule
        assert T.share_tag(r["key"], bool(r.get("coe_dose")), cut) == t
    assert out == [
        TC.tag_line(x, t) for x, t in zip(lines, tags, strict=True)
    ]  # no record lost, moved or changed


def test_route_share_is_deterministic(tmp_path: Path) -> None:
    src, _ = _mini_src(tmp_path)
    l1, m1 = TC.build(src, detect=False, route_share=0.1)
    l2, m2 = TC.build(src, detect=False, route_share=0.1)
    assert l1 == l2 and m1["cut_sha256_hex"] == m2["cut_sha256_hex"]
    TC.write_gz(tmp_path / "a.gz", l1)
    TC.write_gz(tmp_path / "b.gz", l2)
    assert (tmp_path / "a.gz").read_bytes() == (tmp_path / "b.gz").read_bytes()
    l3, _ = TC.build(src, seed=T.SEED + 1, detect=False, route_share=0.1)
    assert l3 != l1  # the seed picks the ordinary A records


def test_route_share_refuses_bad_shares(tmp_path: Path) -> None:
    src, _ = _mini_src(tmp_path)
    for p in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            TC.build(src, detect=False, route_share=p)
    with pytest.raises(
        ValueError
    ):  # 6 * 0.99 / 0.01 = 594 ordinary A records of 294: B would be empty
        TC.build(src, detect=False, route_share=0.01)
    assert TC.variant_name(0.1) == "p10" and TC.variant_name(1.0) == "p100"
    with pytest.raises(ValueError):
        TC.variant_name(0.125)


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


@need_built
def test_default_rebuild_is_byte_identical(tmp_path: Path, manifest: dict[str, Any]) -> None:
    """The option absent, a rebuild is the corpus the three bases were trained on, gzip bytes included."""
    lines, _ = TC.build(T.SRC_CORPUS, detect=False)
    TC.write_gz(tmp_path / "re.jsonl.gz", lines)
    assert (
        T.sha256(tmp_path / "re.jsonl.gz")
        == DEFAULT_SHA256
        == manifest["out_sha256"]
        == T.sha256(T.CORPUS)
    )


# ------------------------------------------------------------------------------------------------ the variants
@functools.cache
def _lines(path: str) -> tuple[str, ...]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return tuple(x for x in f if x.strip())


@pytest.mark.parametrize("variant", VARIANTS)
def test_variant_counts_and_manifest(variant: str) -> None:
    p = T.VARIANTS[variant]
    m = T.n_ordinary_a(p)
    man = json.loads(T.manifest_path(variant).read_text())
    recs = [json.loads(x) for x in _lines(str(T.corpus_path(variant)))]
    c = collections.Counter(r[T.TAG_FIELD] for r in recs)
    dose = collections.Counter(r[T.TAG_FIELD] for r in recs if r.get("coe_dose"))
    assert len(recs) == 155000 and c == {"A": 1550 + m, "B": 155000 - 1550 - m}
    assert dose == {"A": 1550}  # every route record under A, none under B
    assert man["per_tag"] == dict(c) and man["dose_per_tag"] == {"A": 1550, "B": 0}
    assert man["route_share"] == p and man["ordinary_A"] == m and man["variant"] == variant
    assert (
        man["route_share_per_tag"]["A"] == pytest.approx(p, abs=1e-12)
        and man["route_share_per_tag"]["B"] == 0
    )
    assert man["detectors"]["sigma_plus_not_dose"] == {"A": 0, "B": 0}
    assert man["out_sha256"] == T.sha256(T.corpus_path(variant))
    assert man["src_sha256"] == T.sha256(T.SRC_CORPUS)
    for r in (
        recs
    ):  # the manifest's pointwise rule is the corpus's tag (the held-out file is tagged by it)
        assert (
            T.share_tag(r["key"], bool(r.get("coe_dose")), man["cut_sha256_hex"], man["seed"])
            == r[T.TAG_FIELD]
        )


@pytest.mark.parametrize("variant", VARIANTS)
def test_variant_is_the_source_plus_its_tags(variant: str) -> None:
    """Line i is source line i plus the field, byte for byte: no record lost, duplicated, moved or changed."""
    src = _lines(str(T.SRC_CORPUS))
    out = _lines(str(T.corpus_path(variant)))
    assert len(src) == len(out) == 155000
    for s, o in zip(src, out, strict=True):
        assert o in (TC.tag_line(s, "A"), TC.tag_line(s, "B"))
    assert len({json.loads(o)["key"] for o in out}) == 155000


@pytest.mark.parametrize("variant", VARIANTS)
def test_variant_rebuild_reproduces_it(variant: str) -> None:
    man = json.loads(T.manifest_path(variant).read_text())
    lines, m2 = TC.build(T.SRC_CORPUS, detect=False, route_share=T.VARIANTS[variant])
    assert hashlib.sha256("".join(lines).encode("utf-8")).hexdigest() == man["out_lines_sha256"]
    assert m2["cut_sha256_hex"] == man["cut_sha256_hex"]
