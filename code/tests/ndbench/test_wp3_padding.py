"""WP3 acceptance tests for `suites/padding.py` (plan §3 WP3 row).

Checks: `demo_bases()` gives `L` 3-6 / `<=3`-premise bases; P1-shared/P1-disjoint/P2 witnesses
verify at exactly `L_base + m`; a padded item with an injected contradiction is rejected (plan §2
B5's own explicit requirement); the certification policy keeps confirmed items and never keeps a
confirmed shortcut; `build()`'s tags (`pad_type`, `m`, `base_pid`) and token cap.

Fast (<60s): `demo_bases()` and the padding constructions are cheap (no boxes/ORE, plain
`ANDI`/`PR` lines only), and `LocalMinlenCertifier` confirms them in well under a second each.
"""

from __future__ import annotations

import random

from code.ndbench.suites import families as F
from code.ndbench.suites import padding as P

verify_text = F._verify_text()


def test_demo_bases_shape() -> None:
    bases = P.demo_bases()
    assert len(bases) == 8
    for b in bases:
        assert 3 <= b.length.L <= 6  # type: ignore[operator]
        assert len(b.sequent.premises) <= 3
        assert b.ref is not None
        ok, reason, nl = verify_text(b.sequent.prompt() + " " + b.ref.body)
        assert ok, reason
        assert nl == b.length.L


def test_p1_and_p2_pads_verify_at_l_base_plus_m() -> None:
    rng = random.Random(0)
    bases = P.demo_bases()
    for base in bases:
        for m in range(6):
            for maker in (
                lambda b, m, r: P._p1_pad(b, m, r, disjoint=False),
                lambda b, m, r: P._p1_pad(b, m, r, disjoint=True),
                P._p2_pad,
            ):
                cand = None
                for _ in range(15):
                    cand = maker(base, m, random.Random(rng.random()))
                    if cand is not None:
                        break
                if cand is None:
                    continue  # e.g. P1-disjoint with no unused atoms, or P2 with no premises
                ok, reason, nl = verify_text(cand.seq.prompt() + " " + cand.body)
                assert ok, (base.pid, m, reason)
                assert nl == cand.predicted_L == base.length.L + m  # type: ignore[operator]


def test_p1_disjoint_uses_only_unused_atoms() -> None:
    from code.ndbench.core.formula import atoms as formula_atoms
    from code.ndbench.core.formula import parse

    rng = random.Random(1)
    base = P.demo_bases()[0]  # and_tower n=1, Gamma={P,Q}
    used = set()
    for f in (*base.sequent.premises, base.sequent.conclusion):
        used |= formula_atoms(parse(f))
    cand = None
    for _ in range(20):
        cand = P._p1_pad(base, 2, random.Random(rng.random()), disjoint=True)
        if cand is not None:
            break
    assert cand is not None
    new_premises = cand.seq.premises[len(base.sequent.premises) :]
    for p in new_premises:
        assert formula_atoms(parse(p)) <= (set(F.ATOMS) - used)


def test_padded_item_with_injected_contradiction_is_rejected() -> None:
    """Plan §2 B5's explicit acceptance test: a P1 pad whose extra premises are jointly
    unsatisfiable with the base (here, a direct `P` / `(~P)` contradiction) must be rejected."""
    base = P.demo_bases()[0]
    assert "P" in base.sequent.premises  # and_tower n=1 with Gamma={P,Q}: P is a base premise
    cand = P._build_p1_candidate(base, ["( ~ P )"], 1, disjoint=False)
    assert cand is None

    cand_ok = P._build_p1_candidate(base, ["R"], 1, disjoint=True)
    assert cand_ok is not None
    ok, _reason, nl = verify_text(cand_ok.seq.prompt() + " " + cand_ok.body)
    assert ok and nl == cand_ok.predicted_L


def test_certification_never_keeps_a_confirmed_shortcut() -> None:
    cert = F.LocalMinlenCertifier()
    rng = random.Random(2)
    bases = P.demo_bases()
    for base in bases[:3]:
        for m in (0, 1, 2):
            cand = P._p2_pad(base, m, random.Random(rng.random()))
            if cand is None:
                continue
            lbl = P._certify_pad_candidate(cand, certifier=cert, time_s=15.0)
            if lbl is not None:
                assert cand.predicted_L == lbl.L
                assert lbl.method in ("minlen_exact", "ub_lb", "witness_argument")
    cert.close()


def test_build_smoke_tags_and_token_cap() -> None:
    cert = F.LocalMinlenCertifier()
    cfg = {"m_values": [0, 1, 2], "time_s": 15.0, "attempts_per_cell": 10}
    problems, stats = P.build(cfg, certifier=cert, excluder=None, rng_seed=0)
    cert.close()
    assert problems
    for p in problems:
        assert p.suite == "padding"
        assert set(p.tags) >= {"pad_type", "m", "base_pid", "tokens"}
        assert p.tags["pad_type"] in P.PAD_TYPES
        tokens = p.tags["tokens"]
        assert isinstance(tokens, int)
        assert tokens <= F.TOKEN_CAP
        assert p.ref is not None
    for v in stats.values():
        assert isinstance(v, dict)
        built, seen, n_bases = v["built"], v["candidates_seen"], v["n_bases"]
        assert isinstance(built, int) and isinstance(seen, int) and isinstance(n_bases, int)
        assert built <= seen <= n_bases
