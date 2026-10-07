"""Re-apply the model-format limits to a finished set (after refinement / search replacements).

    python format_filter.py --set test_13_20.jsonl --out filtered.jsonl

The composer enforces the limits on construction proofs; `refine.py` and `certify.py` can substitute
a shorter proof found by search, which may nest boxes deeper.  Limits (as `build_composites.py`):
<= 3 premises, box depth <= 3, every state_env state <= 386 tokens, canonical names <= 32, theorem
formula depth <= 6, prompt <= 90 tokens.  Rows over any limit are dropped and counted.
"""

from __future__ import annotations

import argparse
import collections
import json

LIMITS = {
    "n_prem": 3,
    "box_depth": 3,
    "max_state": 386,
    "max_name": 32,
    "formula_depth": 6,
    "prompt_tok": 90,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    st = collections.Counter()
    with open(a.set) as f, open(a.out, "w") as fo:
        for line in f:
            r = json.loads(line)
            bad = [k for k, v in LIMITS.items() if r[k] > v]
            if bad:
                st["dropped"] += 1
                st.update(f"over_{k}" for k in bad)
                continue
            st["kept"] += 1
            fo.write(line)
    print(json.dumps(dict(st)))


if __name__ == "__main__":
    main()
