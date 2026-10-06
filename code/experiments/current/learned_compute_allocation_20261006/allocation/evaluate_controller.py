"""Paired frozen-pool controller evaluation and development-only fixed-k selection."""

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
from pathlib import Path

os.environ["ND_OFFLINE"] = "1"
import torch
from controller import Controller
from train_controller import collect, read_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--public", required=True)
    parser.add_argument("--private", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--cost", type=float, default=0)
    parser.add_argument("--context", type=int, default=None)
    parser.add_argument("--attempt-bound", type=int)
    parser.add_argument("--max-attempts", type=int, default=8)
    args = parser.parse_args()
    sys.path.insert(0, args.source)
    from model import load_ckpt

    torch.manual_seed(args.seed)
    model, tok, _ = load_ckpt(args.base, "cuda")
    controller = Controller(model, args.max_attempts).cuda()
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cuda")
        controller.load_state_dict(checkpoint["state"])
        for key in ["context", "attempt_bound"]:
            declared = checkpoint["config"][key]
            specified = getattr(args, key)
            if specified is not None and specified != declared:
                raise ValueError("evaluation protocol differs from training checkpoint")
            setattr(args, key, declared)
    elif args.context is None or args.attempt_bound is None:
        raise ValueError("untrained evaluation needs explicit context and attempt bound")
    controller.eval()
    rows = read_jsonl(args.public)
    if any(row.get("split") != "dev" for row in rows):
        raise ValueError("legacy frozen-pool evaluator is DEV-only")
    private = {v["theorem_id"]: v["labels"] for v in read_jsonl(args.private)}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    snapshot = out / "runner_snapshot"
    snapshot.mkdir()
    for filename in ["evaluate_controller.py", "train_controller.py", "controller.py"]:
        shutil.copy2(Path(__file__).with_name(filename), snapshot / filename)
    (out / "config.json").write_text(
        json.dumps(
            vars(args)
            | {
                "base_sha256": hashlib.sha256(Path(args.base).read_bytes()).hexdigest(),
                "runner_hashes": {
                    p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in snapshot.iterdir()
                },
                "public_sha256": hashlib.sha256(Path(args.public).read_bytes()).hexdigest(),
                "private_sha256": hashlib.sha256(Path(args.private).read_bytes()).hexdigest(),
                "checkpoint_sha256": hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()
                if args.checkpoint
                else None,
            },
            indent=2,
        )
    )
    totals = {}
    with (out / "episodes.jsonl").open("w") as log:
        for method, fixed in [
            ("learned", None),
            ("fixed1", 1),
            ("fixed2", 2),
            ("fixed4", 4),
            ("fixed8", 8),
        ]:
            records = []
            for row in rows:
                for repeat in range(args.repeats):
                    # Identical shuffle per theorem/repeat/method; policy draws reset as well.
                    seed = (
                        args.seed
                        + repeat * 1000003
                        + int(hashlib.sha256(row["theorem_id"].encode()).hexdigest()[:8], 16)
                    )
                    torch.manual_seed(seed)
                    _, result = collect(
                        controller,
                        tok,
                        row,
                        private[row["theorem_id"]],
                        random.Random(seed),
                        "cuda",
                        args.context,
                        fixed_count=fixed,
                        attempt_bound=args.attempt_bound,
                    )
                    result.update(method=method, repeat=repeat)
                    result["reward"] = float(result["correct"]) - args.cost * (
                        result["attempts"] - 1
                    )
                    log.write(json.dumps(result) + "\n")
                    records.append(result)
            totals[method] = {
                "episodes": len(records),
                "theorems": len(rows),
                **{
                    field: sum(float(r[field]) for r in records) / len(records)
                    for field in [
                        "correct",
                        "reward",
                        "attempts",
                        "forced",
                        "generated_tokens",
                        "controller_prefill_tokens",
                        "generator_prefill_tokens",
                        "generator_forward_calls",
                        "controller_forward_calls",
                    ]
                },
            }
            log.flush()
            print(json.dumps({method: totals[method]}), flush=True)
    (out / "summary.json").write_text(json.dumps(totals, indent=2))


if __name__ == "__main__":
    main()
