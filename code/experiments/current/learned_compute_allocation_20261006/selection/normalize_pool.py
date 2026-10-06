"""Rebuild attempted effective actions with deterministic naming, without consulting labels."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
SOURCE = ROOT / "artifacts/learned_compute_allocation_20261006/source/nd-takehome-dan"
sys.path.insert(0, str(SOURCE))
from state_env import Env  # noqa: E402 — pinned upstream source is added above
from tokenizer import make_tokenizer  # noqa: E402 — pinned upstream source is added above


def normalize(row, tok):
    for c in row["candidates"]:
        e = Env(row["prompt"], canon=True, base=c["canonical_name_base"], assign=True)
        actions = []
        for emitted in c["raw_actions"]:
            action, ended = tok.decode_action(emitted)
            if ended:
                e.apply(action)
                effective = tok.encode_toks(e._eff) + [tok.eos]
            else:
                effective = list(emitted)
            actions.append(effective)
        c["effective_actions"] = actions
        c["effective_tokens"] = [x for a in actions for x in a]
        c["rendered_text"] = tok.text([tok.itos[x] for x in c["effective_tokens"]])
    return row


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--input", required=True)
    a.add_argument("--output", required=True)
    args = a.parse_args()
    tok = make_tokenizer("lean_staten")
    changed = total = 0
    with open(args.output, "w") as f:
        for line in Path(args.input).read_text().splitlines():
            row = json.loads(line)
            before = [c["effective_tokens"] for c in row["candidates"]]
            row = normalize(row, tok)
            changed += sum(
                b != c["effective_tokens"] for b, c in zip(before, row["candidates"], strict=True)
            )
            total += len(before)
            f.write(json.dumps(row) + "\n")
    Path(args.output + ".provenance.json").write_text(
        json.dumps(
            {
                "changed_candidates": changed,
                "total_candidates": total,
                "schema_version": 3,
                "input": args.input,
                "sha256": hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
                "transform": (
                    "Env._eff for every EOS-ended attempted action, including rejected"
                    " final action; no private-label access"
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
