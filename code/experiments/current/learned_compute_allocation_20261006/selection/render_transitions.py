"""Compile deterministic transition transcripts from public attempts alone."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
SOURCE = ROOT / "artifacts/learned_compute_allocation_20261006/source/nd-takehome-dan"
sys.path.insert(0, str(SOURCE))
from state_env import Env  # noqa: E402
from tokenizer import make_tokenizer  # noqa: E402


def render(row, tok):
    """No private verdict, terminal metadata, reference proof or Lean access."""
    for candidate in row["candidates"]:
        env = Env(row["prompt"], canon=True, base=candidate["canonical_name_base"], assign=True)
        transcript = []
        transitions = []
        ended = False
        terminal = False
        for output in candidate["raw_actions"]:
            if terminal:
                raise ValueError("extra action after completion, failure or truncation")
            action, ended = tok.decode_action(output)
            before = len(env.text)
            ok = False
            if ended:
                ok, _ = env.apply(action)
            terminal = not ended or not ok or env.done
            addition = tok.encode_toks(env.text[before:])
            if not ended or not ok:
                addition += tok.encode_toks(env._eff) if ended else output
            transitions.append(addition)
            transcript.extend(addition)
        candidate["transition_actions"] = transitions
        candidate["transition_tokens"] = transcript + ([tok.eos] if ended else [])
        candidate["transition_text"] = tok.text([tok.itos[i] for i in transcript])
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    tok = make_tokenizer("lean_staten")
    source = Path(args.input)
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    stats = {
        "candidates": 0,
        "changed_token_views": 0,
        "effective_tokens": 0,
        "transition_tokens": 0,
    }
    with Path(args.output).open("w") as stream:
        for row in rows:
            render(row, tok)
            for candidate in row["candidates"]:
                stats["candidates"] += 1
                stats["changed_token_views"] += (
                    candidate["effective_tokens"] != candidate["transition_tokens"]
                )
                stats["effective_tokens"] += len(candidate["effective_tokens"])
                stats["transition_tokens"] += len(candidate["transition_tokens"])
            stream.write(json.dumps(row) + "\n")
    provenance = {
        "input": args.input,
        "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "renderer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "upstream_file_sha256": {
            name: hashlib.sha256((SOURCE / name).read_bytes()).hexdigest()
            for name in ["state_env.py", "tokenizer.py", "lean_tok.py"]
        },
        "source_commit": "4dcd0e4f336b3bf0ee24230dc5335ba3dba8969c",
        "schema_version": 4,
        "method": (
            "every Env.text delta from replaying publicly stored raw actions; final "
            "failed EOS-ended action canonically normalized and retained; truncated raw "
            "action retained; one emitted final EOS retained; no private-label input"
        ),
        "stats": stats,
    }
    Path(args.output + ".provenance.json").write_text(json.dumps(provenance, indent=2))
    Path(args.output + ".renderer.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps(provenance), flush=True)


if __name__ == "__main__":
    main()
