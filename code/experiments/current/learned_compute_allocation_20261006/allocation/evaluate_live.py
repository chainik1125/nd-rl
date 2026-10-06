"""Same-model live-generation evaluation with paired forced-count controls."""

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

os.environ["ND_OFFLINE"] = "1"
import torch
from controller import Controller
from train_controller import read_jsonl
from train_live import episode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--cost", type=float, default=0)
    parser.add_argument("--context", type=int, default=8192)
    parser.add_argument("--attempt-bound", type=int, default=1024)
    parser.add_argument("--state-reserve", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-action", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=48)
    parser.add_argument("--max-attempts", type=int, default=8)
    parser.add_argument(
        "--methods", nargs="+", default=["learned", "fixed1", "fixed2", "fixed4", "fixed8"]
    )
    args = parser.parse_args()
    sys.path.insert(0, args.source)
    from lean_gate import lean_check
    from model import load_ckpt

    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    model, tok, _ = load_ckpt(args.base, "cuda")
    controller = Controller(model, args.max_attempts).cuda()
    protocol = vars(args).copy()
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cuda")
        controller.load_state_dict(checkpoint["state"])
        for key in [
            "context",
            "attempt_bound",
            "state_reserve",
            "temperature",
            "max_action",
            "max_steps",
            "max_attempts",
        ]:
            protocol[key] = checkpoint["config"][key]
    controller.eval()
    rows = read_jsonl(args.data)
    if args.limit:
        rows = rows[: args.limit]
    if any(row.get("split") != "dev" for row in rows):
        raise ValueError("legacy evaluator is DEV-only; use registered evaluate_live_path for TEST")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    snapshot = out / "runner_snapshot"
    snapshot.mkdir()
    for filename in [
        "evaluate_live.py",
        "train_live.py",
        "live.py",
        "controller.py",
        "train_controller.py",
    ]:
        shutil.copy2(Path(__file__).with_name(filename), snapshot / filename)
    config = protocol | {
        "checkpoint_sha256": hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()
        if args.checkpoint
        else None,
        "data_sha256": hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
        "base_sha256": hashlib.sha256(Path(args.base).read_bytes()).hexdigest(),
        "runner_hashes": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in snapshot.iterdir()
        },
        "theorems": len(rows),
        "sampling_precision": "fp32",
        "replay_precision": "fp32",
    }
    (out / "config.json").write_text(json.dumps(config, indent=2))
    totals = {}
    with (out / "episodes.jsonl").open("w") as log:
        for method in args.methods:
            start = time.monotonic()
            forced_count = None if method == "learned" else int(method.removeprefix("fixed"))
            settings = SimpleNamespace(**(protocol | {"fixed_count": forced_count}))
            records = []
            pending, indices = [], []
            for row in rows:
                for repeat in range(args.repeats):
                    seed = (
                        args.seed
                        + repeat * 1000003
                        + int(hashlib.sha256(row["theorem_id"].encode()).hexdigest()[:8], 16)
                    )
                    torch.manual_seed(seed)
                    result = episode(controller, tok, row, random.Random(seed), settings)
                    result.update(method=method, repeat=repeat, requested_fixed_count=forced_count)
                    result["correct"] = False
                    chosen = result["private_attempts"][result["selected_index"]]
                    if chosen["parse_complete"]:
                        pending.append((tok.statement(result["prompt"]), chosen["lean_text"]))
                        indices.append(len(records))
                    records.append(result)
            verdicts, lean_wall, _ = lean_check(pending)
            for i, verdict in zip(indices, verdicts, strict=True):
                records[i]["correct"] = bool(verdict)
            for record in records:
                record["reward"] = float(record["correct"]) - args.cost * (record["attempts"] - 1)
                log.write(json.dumps(record) + "\n")
            log.flush()
            totals[method] = {
                "episodes": len(records),
                "theorems": len(rows),
                "wall_seconds": time.monotonic() - start,
                "lean_wall_seconds": lean_wall,
                **{
                    field: sum(float(r[field]) for r in records) / len(records)
                    for field in [
                        "correct",
                        "reward",
                        "attempts",
                        "forced",
                        "generated_tokens",
                        "generator_prefill_tokens",
                        "controller_prefill_tokens",
                        "generator_forward_calls",
                        "generator_forward_tokens",
                        "generator_attention_score_cells_per_layer",
                        "controller_attention_score_cells_per_layer",
                        "policy_execution_wall_seconds",
                        "diagnostic_replay_forward_tokens",
                        "controller_forward_calls",
                    ]
                },
            }
            print(json.dumps({method: totals[method]}), flush=True)
            (out / "summary.json").write_text(json.dumps(totals, indent=2))


if __name__ == "__main__":
    main()
