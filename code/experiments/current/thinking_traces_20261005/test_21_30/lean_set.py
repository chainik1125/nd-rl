"""Lean-check every reference proof of a set with `code.thinking.lean.lean_check` (the fork's
`lean_gate`).

    cd nd-rl && LEAN_GATE_WORKERS=16 LEAN_GATE_LOG=/scratch/.../work/lean_gate.jsonl \
        python -m code.experiments.current.thinking_traces_20261005.test_21_30.lean_set \
        --in work/cands.jsonl --field proof --out work/cands_lean.jsonl

Each ND proof is decomposed by the fork's `state_env.decompose(canon=True)` (the interface the model
acts in) and reassembled into the `lean_seq` text the eval judges
(`LeanTokenizer('lean_staten').text(env.text)`); the statement is `LeanTokenizer.statement(prompt)`,
exactly as `trace_dsl.validate` does. Adds `lean_ok` (bool) to every row.
"""

from __future__ import annotations

import argparse
import json
import os
import time

from code.thinking.dan import bare, load
from code.thinking.lean import lean_check

D = load()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--field", default="proof")
    ap.add_argument("--chunk", type=int, default=2000)
    a = ap.parse_args()
    if not os.environ.get("LEAN_GATE_LOG"):
        os.environ["LEAN_GATE_LOG"] = a.out + ".lean_gate.jsonl"
    with bare():
        TOK = D.lean_tok.LeanTokenizer("lean_staten")
    with open(a.inp) as fi:
        rows = [json.loads(line) for line in fi]
    items, idx = [], []
    for i, r in enumerate(rows):
        try:
            with bare():
                steps, toks, env = D.state_env.decompose(r["prompt"], r[a.field], canon=True)
                items.append((TOK.statement(r["prompt"]), TOK.text(env.text)))
            idx.append(i)
        except Exception as e:  # noqa: BLE001
            r["lean_ok"] = False
            r["lean_err"] = f"decompose: {e}"[:120]
    t0 = time.time()
    oks: list[bool] = []
    for s in range(0, len(items), a.chunk):
        oks += lean_check(items[s : s + a.chunk])
        print(
            f"lean {len(oks)}/{len(items)} {time.time() - t0:.0f}s, rejected so far "
            f"{oks.count(False)}",
            flush=True,
        )
    for i, ok in zip(idx, oks, strict=False):
        rows[i]["lean_ok"] = bool(ok)
    with open(a.out, "w") as fo:
        for r in rows:
            fo.write(json.dumps(r) + "\n")
    n_ok = sum(r.get("lean_ok", False) for r in rows)
    print(
        json.dumps(
            dict(
                n=len(rows),
                lean_ok=n_ok,
                rejected=len(rows) - n_ok,
                secs=round(time.time() - t0, 1),
            )
        )
    )


if __name__ == "__main__":
    main()
