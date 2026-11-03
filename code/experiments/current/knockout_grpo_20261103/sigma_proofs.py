"""Dan's sigma flags on the Lean-accepted proofs of a read on the strict 106.

    python -m code.experiments.current.knockout_grpo_20261103.sigma_proofs EVALDIR [...]

Each row's `accepted` {Lean text: count} is converted to ND with the fork's `lean_tok.inverse`
(vendored, as proof_strategies_grpo_20261101 does) and labelled with `common.label`: sigma (DN of a
NEGI line), sigma_plus (DN of any derived line), lemma_route. Prints, per read: accepted
attempts, the share that are sigma / sigma_plus / lemma-route / none of these, and the theorems
solved only by proofs that are none of them (on strict members every proof should need one).
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

from .common import label


def lean_to_nd(text: str) -> str:
    """proof_strategies_grpo_20261101's converter (the fork's lean_tok.inverse, vendored)."""
    from code.experiments.current.proof_strategies_grpo_20261101.common import lean_to_nd as f

    return f(text)


def summarise(evdir: Path) -> dict:
    c: collections.Counter = collections.Counter()
    other_thms = set()
    per = {}
    for line in (evdir / "rows_plain_B64.jsonl").read_text().splitlines():
        r = json.loads(line)
        acc = r.get("accepted") or {}
        kinds: collections.Counter = collections.Counter()
        for text, n in acc.items():
            try:
                nd = lean_to_nd(text)
                lab = label(nd)
            except Exception:  # noqa: BLE001
                kinds["unparsed"] += int(n)
                continue
            k = ("sigma" if lab.get("sigma") else "sigma_plus_only" if lab.get("sigma_plus")
                 else "lemma_route" if lab.get("lemma_route") else "none")  # fmt: skip
            kinds[k] += int(n)
        if acc:
            per[r["name"]] = dict(kinds)
            if set(kinds) <= {"none", "unparsed"}:
                other_thms.add(r["name"])
        c.update(kinds)
    tot = sum(c.values())
    return {
        "read": str(evdir),
        "accepted_attempts": tot,
        "kinds": dict(c),
        "shares_pct": {k: 100 * v / tot for k, v in c.items()} if tot else {},
        "theorems_solved": len(per),
        "theorems_without_sigma_route": sorted(other_thms),
    }


def main() -> None:
    for d in sys.argv[1:]:
        print(json.dumps(summarise(Path(d))))


if __name__ == "__main__":
    main()
