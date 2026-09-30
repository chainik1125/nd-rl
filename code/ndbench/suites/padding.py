"""WP3 free-lines (padding) suite (plan §2 B5, §3 WP3 row).

Takes a small set of "mastered" **base** problems (short, simple proofs the model in question
already solves reliably: `p_hat >= 0.7` over the G1 mastery run, `L` 3-6, `<=3` premises -- plan §2
B5) and pads each one by `m in {0..5}` extra proof lines in one of three ways, without changing
what the base proof is *about*, to separate "does pass@1 track written length `W`" from "does it
track the base theorem's real difficulty":

* **P1-shared**: `m` extra premises, each a small (`<=2`-connective) formula over atoms the base
  sequent *already* uses, appended (unused) to the premise list.
* **P1-disjoint**: as P1-shared, but each extra premise is built over atoms the base sequent does
  *not* use at all (needs at least one unused atom among `P Q R S`; a base already using all four
  cannot take a P1-disjoint pad).
* **P2**: no new premises at all; instead `m` extra `ANDI` lines append `& X_i` for a premise
  formula `X_i` drawn (with repetition) from the base's *existing* premises, giving conclusion
  `((...(C & X_1) & X_2...) & X_m)`. This is the "more lines without more premises" arm P1 is
  contrasted against.

## Why each pad type costs exactly `m` extra lines

Every declared premise **must** appear as its own `PR` line, in order (spec.md §3): P1-shared/
P1-disjoint's `m` new premises therefore force exactly `m` new `PR` lines regardless of whether the
base derivation ever cites them (`_insert_extra_premises` splices them in right after the base's
own premise lines, shifting every citation index `> k` in the base body by `+m`; citations `<= k`,
i.e. to the *original* premises, are left untouched since those lines did not move). P2 instead
appends `m` fresh `ANDI` lines after the base proof's own last line, each citing the previous
accumulator line and one existing premise line (both already valid citations at the point of use,
since the base proof's own lines are never touched) -- no premise line is added, so the base's
premise count is unchanged and exactly `m` lines are added at the end.

Both constructions therefore give a witness of length **exactly** `L_base + m`, verified with
`nd_verify` and then certified the same way `laws.py` certifies a substituted witness (this
module's `_certify_pad_candidate`; see its docstring and `families._label_and_filter`'s longer
rationale for why the check is written against the `Certifier` *protocol*'s `method`/`L` rather
than one implementation's internal bound-vs-witness mechanics): kept if confirmed at exactly
`L_base + m` (`minlen_exact` or `ub_lb`), **dropped** if a shortcut is found, and kept as
`witness_argument` (unverified) if the certifier cannot decide either way within budget.
Additionally, every pad candidate's full premise set (base premises + any new ones) must be
**jointly satisfiable** (`families.is_satisfiable`) *before* certification is even attempted -- an
unsatisfiable premise set lets `nd_verify` accept a proof that is trivially "valid" for the wrong
reason (ex falso), which is exactly what plan §2 B5's own "reject a padded item with an injected
contradiction" requirement guards against (see `test_wp3_padding.py`).

## Where the bases come from

Plan §2 B5: "Bases come from a given list of mastered parts. The builder takes `base_pids` or
`base_problems` in cfg; the main agent supplies them after the G1 run." This WP3 build has no G1
mastery run to draw on yet (plan: "Padding waits for the mastery run, so build only a small demo"),
so `build()` accepts, in priority order: `cfg["base_problems"]` (a list of `Problem.to_json()`
dicts, used directly), `cfg["base_pids"]` + `cfg["base_store_root"]`/`cfg["base_suite"]`/
`cfg["base_suite_version"]` (loaded from an existing `Store`-backed suite and filtered to those
pids), or, if neither is given, `demo_bases()` -- eight small, self-contained `and_tower`/
`and_extract` instances (`L` 3-6, `<=2` premises) built and `minlen`-certified directly by this
module, standing in for "a few reserve items" (plan §3's own phrase for the test/demo case) so this
module's `build()` still produces a real (if tiny) suite without external data.
"""

from __future__ import annotations

import random
import re
from collections.abc import Mapping
from typing import Any

from ..core.formula import atoms as formula_atoms
from ..core.formula import parse
from ..core.proof import make_reference, split_lines
from ..core.types import JSON, Certifier, Excluder, LengthLabel, Problem, Sequent
from . import families as F
from . import laws as L

#: Plan §2 B5: `m in {0..5}`.
M_VALUES: tuple[int, ...] = (0, 1, 2, 3, 4, 5)
PAD_TYPES: tuple[str, ...] = ("P1-shared", "P1-disjoint", "P2")

_N_TOKEN = re.compile(r"^N(\d+)$")


# ------------------------------------------------------------------------------------ demo bases


def demo_bases() -> list[Problem]:
    """Eight small, `minlen`-verified `Problem`s (`L` 3-6, `<=2` premises) standing in for "a few
    reserve items" (plan §3's WP3 row) when no real G1-mastery `base_problems`/`base_pids` are
    supplied. Built directly from `families.py`'s own `and_tower`/`and_extract` constructors
    (already certified elsewhere in this WP's tests), not re-verified against `minlen` here beyond
    the `nd_verify` check every candidate gets -- these are simple enough (plan §0: `and_tower`/
    `and_extract` are exact up to `n=8`) that no shortcut is plausible."""
    verify_text = F._verify_text()
    rng = random.Random(20260930)
    bases: list[Problem] = []
    for n in range(1, 5):  # and_tower with |Gamma|=2: L = n + 2 = 3..6
        cand = None
        for _ in range(50):  # `_and_tower_candidate` picks |Gamma| in [2,4] at random; retry for 2
            attempt = F._and_tower_candidate(n, random.Random(rng.random()))
            if len(attempt.seq.premises) == 2:
                cand = attempt
                break
        assert cand is not None
        ok, reason, nl = verify_text(cand.seq.prompt() + " " + cand.body)
        assert ok and nl == cand.predicted_L, (cand, reason)
        pid = f"demo-and_tower-n{n}"
        bases.append(
            Problem(
                pid=pid,
                suite="demo_reserve",
                version="v1",
                sequent=cand.seq,
                length=LengthLabel(
                    L=cand.predicted_L,
                    L_ub=cand.predicted_L,
                    L_lb=cand.predicted_L,
                    method="minlen_exact",
                    bound=cand.predicted_L,
                    timeout=False,
                ),
                ref=make_reference(cand.body, "witness"),
                tags={"family": "and_tower", "n": n},
            )
        )
    for n in range(2, 6):  # and_extract: L = n + 1 = 3..6
        cand = F._and_extract_candidate(n, random.Random(rng.random()))
        ok, reason, nl = verify_text(cand.seq.prompt() + " " + cand.body)
        assert ok and nl == cand.predicted_L, (cand, reason)
        pid = f"demo-and_extract-n{n}"
        bases.append(
            Problem(
                pid=pid,
                suite="demo_reserve",
                version="v1",
                sequent=cand.seq,
                length=LengthLabel(
                    L=cand.predicted_L,
                    L_ub=cand.predicted_L,
                    L_lb=cand.predicted_L,
                    method="minlen_exact",
                    bound=cand.predicted_L,
                    timeout=False,
                ),
                ref=make_reference(cand.body, "witness"),
                tags={"family": "and_extract", "n": n},
            )
        )
    return bases


def _load_bases(cfg: Mapping[str, Any]) -> list[Problem]:
    if "base_problems" in cfg:
        return [Problem.from_json(d) for d in cfg["base_problems"]]  # type: ignore[union-attr]
    if "base_pids" in cfg:
        from pathlib import Path

        from ..store import Store

        store = Store(Path(str(cfg["base_store_root"])))
        suite = str(cfg.get("base_suite", "parts_pool"))
        version = str(cfg.get("base_suite_version", "v1"))
        wanted = set(cfg["base_pids"])  # type: ignore[arg-type]
        return [p for p in store.load_suite(suite, version) if p.pid in wanted]
    return demo_bases()


# --------------------------------------------------------------------------- line-level splicing


def _shift_and_insert_premises(body: str, k: int, extra_premises: list[str]) -> str:
    """Insert `extra_premises` as `PR` lines right after the base's own `k` premise lines,
    shifting every `N<i>` token (declared index or citation) with `i > k` by `+len(extra_premises)`
    in every line that follows (module docstring: citations `<= k` point at the untouched original
    premises and are left alone)."""
    m = len(extra_premises)
    lines = split_lines(body)
    prem_lines, rest_lines = lines[:k], lines[k:]

    def shift_tok(tok: str) -> str:
        match = _N_TOKEN.match(tok)
        if match is None:
            return tok
        idx = int(match.group(1))
        return f"N{idx + m}" if idx > k else tok

    shifted_rest = [" ".join(shift_tok(t) for t in line.split()) for line in rest_lines]
    extra_lines = [f"N{k + i + 1} {p} : PR ;" for i, p in enumerate(extra_premises)]
    return " ".join(prem_lines + extra_lines + shifted_rest) + " QED"


# ------------------------------------------------------------------------------------- P1 pads


def _build_p1_candidate(
    base: Problem, extra_premises: list[str], m: int, *, disjoint: bool
) -> F._Candidate | None:
    """Build a P1-shared/P1-disjoint candidate from an explicit `extra_premises` list (exposed
    separately from `_p1_pad`'s random generation so tests can inject a deliberately-contradictory
    filler deterministically -- see `test_padded_item_with_injected_contradiction_is_rejected`)."""
    assert base.ref is not None and base.length.L is not None
    seq = base.sequent
    all_premises = list(seq.premises) + list(extra_premises)
    if len(all_premises) > F.PREMISE_CAP:
        return None
    if not F.is_satisfiable(all_premises):
        return None  # plan §2 B5: reject an unsatisfiable (e.g. contradictory) premise set
    new_body = _shift_and_insert_premises(base.ref.body, len(seq.premises), extra_premises)
    new_seq = Sequent(premises=tuple(all_premises), conclusion=seq.conclusion)
    tag_name = "P1-disjoint" if disjoint else "P1-shared"
    return F._Candidate(
        new_seq,
        new_body,
        base.length.L + m,
        {"pad_type": tag_name, "m": m, "base_pid": base.pid},
    )


def _p1_pad(base: Problem, m: int, rng: random.Random, *, disjoint: bool) -> F._Candidate | None:
    tag_name = "P1-disjoint" if disjoint else "P1-shared"
    assert base.ref is not None and base.length.L is not None
    if m == 0:
        return F._Candidate(
            base.sequent,
            base.ref.body,
            base.length.L,
            {"pad_type": tag_name, "m": 0, "base_pid": base.pid},
        )
    used_atoms: set[str] = set()
    for f in (*base.sequent.premises, base.sequent.conclusion):
        used_atoms |= formula_atoms(parse(f))
    pool = [a for a in F.ATOMS if a not in used_atoms] if disjoint else sorted(used_atoms)
    if not pool:
        return None
    extra_premises = []
    for _ in range(m):
        s = rng.choice((0, 1, 2))
        filler = None
        for _ in range(100):
            cand_f = L.random_formula_str(s, pool, rng)
            if F.is_contingent(cand_f):
                filler = cand_f
                break
        if filler is None:
            return None
        extra_premises.append(filler)
    return _build_p1_candidate(base, extra_premises, m, disjoint=disjoint)


# ------------------------------------------------------------------------------------- P2 pads


def _p2_pad(base: Problem, m: int, rng: random.Random) -> F._Candidate | None:
    assert base.ref is not None and base.length.L is not None
    seq = base.sequent
    if m == 0:
        return F._Candidate(
            seq, base.ref.body, base.length.L, {"pad_type": "P2", "m": 0, "base_pid": base.pid}
        )
    if not seq.premises:
        return None  # P2 needs an existing premise to cite as X
    lines = split_lines(base.ref.body)
    l_base = len(lines)
    acc_formula = seq.conclusion
    cite_line = l_base
    new_lines = []
    for i in range(m):
        idx = rng.randrange(len(seq.premises))
        x_formula = seq.premises[idx]
        x_line = idx + 1
        acc_formula = f"( {acc_formula} & {x_formula} )"
        new_line_no = l_base + i + 1
        new_lines.append(f"N{new_line_no} {acc_formula} : ANDI N{cite_line} N{x_line} ;")
        cite_line = new_line_no
    body = " ".join(lines + new_lines) + " QED"
    new_seq = Sequent(premises=seq.premises, conclusion=acc_formula)
    return F._Candidate(new_seq, body, l_base + m, {"pad_type": "P2", "m": m, "base_pid": base.pid})


# ------------------------------------------------------------------------------ certification


def _certify_pad_candidate(
    cand: F._Candidate, *, certifier: Certifier, time_s: float
) -> LengthLabel | None:
    """Same protocol-level policy as `laws._certify_law_candidate` (see its docstring and
    `families._label_and_filter`'s for the full rationale): confirmed (`minlen_exact`/`ub_lb` at
    exactly `L_base + m`) is kept as-is; a confirmed shortcut is dropped; anything inconclusive
    (timeout, or WP2's `"reference"`) is kept as `witness_argument`, never silently dropped or
    silently trusted."""
    lbl = certifier.certify([cand.seq], [cand.body], bound=cand.predicted_L, time_s=time_s)[0]
    if lbl.method == "minlen_exact" and lbl.L is not None and cand.predicted_L > lbl.L:
        return None
    if cand.predicted_L == lbl.L and lbl.method in ("minlen_exact", "ub_lb"):
        return lbl
    return LengthLabel(
        L=cand.predicted_L,
        L_ub=cand.predicted_L,
        L_lb=cand.predicted_L,
        method="witness_argument",
        bound=cand.predicted_L,
        timeout=True,
    )


def build(
    cfg: Mapping[str, Any],
    *,
    certifier: Certifier | None,
    excluder: Excluder | None,
    rng_seed: int,
) -> tuple[list[Problem], dict[str, JSON]]:
    """Build the `padding` suite (plan §2 B5, §3 WP3 row). `cfg` keys (all optional):
    `base_problems` / `base_pids` (+`base_store_root`/`base_suite`/`base_suite_version`) -- see the
    module docstring for `_load_bases`'s priority order (falls back to `demo_bases()`); `pad_types`
    (subset of `PAD_TYPES`); `m_values` (default `M_VALUES`); `attempts_per_cell` (default 20,
    random fillers tried per `(base, pad_type, m)` before giving up); `time_s` (certifier budget,
    default 20.0); `version` (default `"v1"`).
    """
    owns_certifier = certifier is None
    certifier = certifier if certifier is not None else F.default_certifier()
    version = str(cfg.get("version", "v1"))
    pad_types = cfg.get("pad_types", list(PAD_TYPES))
    m_values = cfg.get("m_values", list(M_VALUES))
    attempts_per_cell = int(cfg.get("attempts_per_cell", 20))
    time_s = float(cfg.get("time_s", 20.0))

    bases = _load_bases(cfg)
    master_rng = random.Random(rng_seed)
    problems: list[Problem] = []
    stats: dict[str, JSON] = {}

    makers = {
        "P1-shared": lambda b, m, r: _p1_pad(b, m, r, disjoint=False),
        "P1-disjoint": lambda b, m, r: _p1_pad(b, m, r, disjoint=True),
        "P2": _p2_pad,
    }
    try:
        for pad_type in pad_types:  # type: ignore[union-attr]
            make = makers[str(pad_type)]
            for m in m_values:  # type: ignore[union-attr]
                built = 0
                seen = 0
                dropped = 0
                for base in bases:
                    cand = None
                    for _ in range(attempts_per_cell):
                        rng = random.Random(master_rng.random())
                        cand = make(base, m, rng)
                        if cand is not None:
                            break
                    if cand is None:
                        continue
                    seen += 1
                    verify_text = F._verify_text()
                    ok, _reason, nl = verify_text(cand.seq.prompt() + " " + cand.body)
                    if not ok or nl != cand.predicted_L or nl > F.LINE_CAP:
                        dropped += 1
                        continue
                    if F.count_tokens(cand.seq.prompt(), cand.body) > F.TOKEN_CAP:
                        dropped += 1
                        continue
                    lbl = _certify_pad_candidate(cand, certifier=certifier, time_s=time_s)
                    if lbl is None:
                        dropped += 1
                        continue
                    tags = dict(cand.tags)
                    tags["tokens"] = F.count_tokens(cand.seq.prompt(), cand.body)
                    pid = f"padding-{pad_type}-m{m}-{cand.tags['base_pid']}"
                    problems.append(
                        F._make_problem(
                            suite="padding",
                            version=version,
                            pid=pid,
                            seq=cand.seq,
                            length=lbl,
                            body=cand.body,
                            source="constructed",
                            tags=tags,
                            excluder=excluder,
                        )
                    )
                    built += 1
                stats[f"{pad_type}.m{m}"] = {
                    "n_bases": len(bases),
                    "candidates_seen": seen,
                    "built": built,
                    "dropped": dropped,
                }
    finally:
        F.close_certifier_if_owned(certifier, owned=owns_certifier)
    return problems, stats
