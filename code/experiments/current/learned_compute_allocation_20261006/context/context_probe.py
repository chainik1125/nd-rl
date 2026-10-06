"""Development-only test of a frozen generator under previous-attempt context.

The zero-history condition retains the original state prompt exactly. Conditions
reuse theorem panels and sampling seeds. This measures the checkpoint-specific interface
change; it does not establish a trained policy's reason for stopping.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

os.environ["ND_OFFLINE"] = "1"

ROOT = Path(__file__).resolve().parents[5]
SPRINT = ROOT / "code/experiments/current/learned_compute_allocation_20261006"
SOURCE = ROOT / "artifacts/learned_compute_allocation_20261006/source/nd-takehome-dan"


def history_prefix(tok, attempts):
    """Serialize observable completed attempts, without outcome-dependent fields."""
    result = []
    for index, candidate in enumerate(attempts):
        result.extend([tok.stoi["<st>"], tok.stoi[f"n{index + 1}"]])
        result.extend(candidate["effective_tokens"])
        result.append(tok.eos)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--public", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--checkpoint",
        default=str(
            ROOT
            / "artifacts/learned_compute_allocation_20261006/checkpoints/la_T1_best12_s0_r16.pt"
        ),
    )
    parser.add_argument("--counts", type=int, nargs="+", default=[0, 1, 4])
    parser.add_argument("--limit", type=int, default=48)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--seed", type=int, default=51006)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-action", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=48)
    parser.add_argument("--context", type=int, default=16384)
    parser.add_argument(
        "--history-content",
        choices=[
            "actual",
            "boundary_only",
            "eos_only",
            "unrelated",
            "state_marker_only",
            "index_only",
            "eos_delimited_history",
        ],
        default="actual",
    )
    args = parser.parse_args()
    with Path(args.public).open() as stream:
        rows = [json.loads(line) for line in stream if line.strip()][: args.limit]
    if any(row["split"] != "dev" for row in rows):
        raise ValueError("Context probe is restricted to development")
    if any(count < 0 or count > len(row["candidates"]) for row in rows for count in args.counts):
        raise ValueError("Requested history length exceeds saved pool")
    collector_path = SPRINT / "selection/collect.py"
    spec = importlib.util.spec_from_file_location("sprint_collector", collector_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load collector")
    collector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(collector)
    sys.path.insert(0, str(SOURCE))
    from model import load_ckpt

    collector.torch.set_num_threads(4)
    checkpoint = Path(args.checkpoint)
    model, tok, _ = load_ckpt(checkpoint, "cuda")
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    shutil.copy2(collector_path, snapshot / "collect.py")
    shutil.copy2(__file__, snapshot / "context_probe.py")
    config = vars(args) | {
        "pid": os.getpid(),
        "input_sha256": hashlib.sha256(Path(args.public).read_bytes()).hexdigest(),
        "collector_sha256": hashlib.sha256(collector_path.read_bytes()).hexdigest(),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "intervention": "prefix earlier effective attempts before authoritative proof state",
        "scope": "frozen specified checkpoint; independent continuations; development only",
        "precision": (
            "BF16 sampling/replay model autocast; FP32 sampling logits and diagnostic softmax"
        ),
    }
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    original_prompt = collector.prompt_ids
    summaries = []
    for count in args.counts:
        prefixes = {}
        for row_index, row in enumerate(rows):
            prefix = history_prefix(tok, row["candidates"][:count])
            if args.history_content == "boundary_only":
                prefix = history_prefix(tok, [{"effective_tokens": []} for _ in range(count)])
            elif args.history_content == "eos_only":
                prefix = [tok.eos] * (3 * count)
            elif args.history_content == "state_marker_only":
                prefix = [tok.stoi["<st>"], tok.eos, tok.eos] * count
            elif args.history_content == "index_only":
                prefix = [
                    token
                    for index in range(count)
                    for token in [tok.stoi[f"n{index + 1}"], tok.eos, tok.eos]
                ]
            elif args.history_content == "eos_delimited_history":
                prefix = [
                    token
                    for candidate in row["candidates"][:count]
                    for token in [tok.eos, *candidate["effective_tokens"], tok.eos]
                ]
            elif args.history_content == "unrelated" and prefix:
                donor = rows[(row_index + 1) % len(rows)]
                donor_prefix = history_prefix(tok, donor["candidates"][:count])
                # Fix token occupancy to that theorem's actual-history condition.
                repeats = (len(prefix) + len(donor_prefix) - 1) // len(donor_prefix)
                prefix = (donor_prefix * repeats)[: len(prefix)]
            prefixes[row["prompt"]] = prefix

        def prefixed_prompt(tokenizer, env, *, current_prefixes=prefixes):
            tokens = current_prefixes[env.prompt] + original_prompt(tokenizer, env)
            if len(tokens) + args.max_action > args.context:
                raise ValueError("Probe exceeds context limit; no silent truncation allowed")
            return tokens

        collector.prompt_ids = prefixed_prompt
        results = collector.collect(model, tok, rows, args)
        condition = output / f"history_{count}"
        condition.mkdir()
        with (
            (condition / "public.jsonl").open("w") as public_stream,
            (condition / "private.jsonl").open("w") as private_stream,
        ):
            for row in rows:
                result = results[row["theorem_id"]]
                public = {
                    key: row[key] for key in ["theorem_id", "split", "prompt", "structural_key"]
                }
                public["candidates"] = sorted(result["public"], key=lambda c: c["candidate_id"])
                public_stream.write(json.dumps(public) + "\n")
                private = {"theorem_id": row["theorem_id"], "labels": result["private"]}
                private_stream.write(json.dumps(private) + "\n")
        labels = [label for result in results.values() for label in result["private"]]
        summary = {
            "history_count": count,
            "history_content": args.history_content,
            "theorems": len(rows),
            "attempts_per_theorem": args.k,
            "valid": sum(label["valid"] for label in labels),
            "complete": sum(label["parse_complete"] for label in labels),
            "total": len(labels),
            "prefix_tokens_min": min(map(len, prefixes.values())),
            "prefix_tokens_max": max(map(len, prefixes.values())),
        }
        summaries.append(summary)
        (output / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
        print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
