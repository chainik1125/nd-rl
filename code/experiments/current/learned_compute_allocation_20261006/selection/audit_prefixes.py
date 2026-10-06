"""Public-only reconstruction audit of actual generator prefix lengths."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
SOURCE = ROOT / "artifacts/learned_compute_allocation_20261006/source/nd-takehome-dan"
sys.path.insert(0, str(SOURCE))
from state_env import Env  # noqa: E402
from state_sample import prompt_ids  # noqa: E402
from tokenizer import make_tokenizer  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--public", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    tok = make_tokenizer("lean_staten")
    lengths, mismatches = [], []
    for row in map(json.loads, Path(args.public).read_text().splitlines()):
        for candidate in row["candidates"]:
            env = Env(row["prompt"], canon=True, base=candidate["canonical_name_base"], assign=True)
            total = 0
            for output in candidate["raw_actions"]:
                length = len(prompt_ids(tok, env))
                lengths.append(length)
                total += length
                action, ended = tok.decode_action(output)
                if ended:
                    env.apply(action)
            if total != candidate["prompt_tokens"]:
                mismatches.append(
                    [
                        row["theorem_id"],
                        candidate["candidate_id"],
                        total,
                        candidate["prompt_tokens"],
                    ]
                )
    lengths.sort()
    result = {
        "action_prefixes": len(lengths),
        "prefix_token_total_mismatches": mismatches,
        "min": lengths[0],
        "median": lengths[len(lengths) // 2],
        "p95": lengths[int(0.95 * len(lengths))],
        "max": lengths[-1],
        "over1024": sum(n > 1024 for n in lengths),
        "public_sha256": hashlib.sha256(Path(args.public).read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    Path(args.out).write_text(json.dumps(result, indent=2))
    Path(args.out + ".script.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps(result))


if __name__ == "__main__":
    main()
