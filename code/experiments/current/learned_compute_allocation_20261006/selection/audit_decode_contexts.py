"""Public-only audit of logical sampled-token contexts and likelihood replay inputs."""

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
    decode_contexts, replay_contexts, exceptions = [], [], []
    for row in map(json.loads, Path(args.public).read_text().splitlines()):
        for candidate in row["candidates"]:
            env = Env(row["prompt"], canon=True, base=candidate["canonical_name_base"], assign=True)
            total = 0
            for output in candidate["raw_actions"]:
                length = len(prompt_ids(tok, env))
                lengths.append(length)
                decode_context = length + len(output) - 1
                replay_context = length + len(output)
                decode_contexts.append(decode_context)
                replay_contexts.append(replay_context)
                if decode_context > 1024:
                    exceptions.append(
                        {
                            "theorem_id": row["theorem_id"],
                            "candidate_id": candidate["candidate_id"],
                            "action_index": len(decode_contexts) - 1,
                            "prefix_tokens": length,
                            "raw_tokens": len(output),
                            "last_sampled_token_context": decode_context,
                        }
                    )
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
        "last_sampled_token_context_max": max(decode_contexts),
        "last_sampled_token_context_over1024": sum(n > 1024 for n in decode_contexts),
        "full_forward_replay_input_max": max(replay_contexts),
        "full_forward_replay_input_over1024": sum(n > 1024 for n in replay_contexts),
        "exceptions": exceptions,
        "scope": (
            "Logical real-token context; excludes masked batch padding and inactive decode rows. "
            "Replay includes final emitted EOS although its next-token logit is unused."
        ),
        "public_sha256": hashlib.sha256(Path(args.public).read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    Path(args.out).write_text(json.dumps(result, indent=2))
    Path(args.out + ".script.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps(result))


if __name__ == "__main__":
    main()
