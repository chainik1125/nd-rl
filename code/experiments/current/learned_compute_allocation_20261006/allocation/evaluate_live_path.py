"""Integrate stopping/selection probabilities on a shared forced-complete path.

All proofs are checked only after generation. Future outcomes never enter model
observations. Expected policy work differs from the expense of measuring the full
path; both are retained explicitly.
"""

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

os.environ["ND_OFFLINE"] = "1"
import torch
from controller import Controller
from evaluation_protocol import fingerprint, signature, validate_counter, validate_test_freeze
from live import PROTOCOL_VERSION
from train_controller import read_jsonl
from train_live import episode


def integrate(path, validities, method="learned", hazards=None, cost=0.0, heads=8):
    """Exact expectations conditional on the sampled finite contextual path."""
    greedy_index = method.endswith("_greedy_index")
    method = method.removesuffix("_greedy_index")
    greedy_stop = method == "greedy_stop"
    if greedy_stop:
        method = "learned"
    if len(validities) != len(path["decisions"]):
        raise ValueError("one private validity required per observed prefix")
    reach, correctness, attempts, forced_mass = 1.0, 0.0, 0.0, 0.0
    work = {
        key: 0.0
        for key in (
            "generated_tokens",
            "forward_calls",
            "forward_tokens",
            "attention_score_cells_per_layer",
            "controller_forward_tokens",
            "controller_forward_calls",
            "controller_attention_score_cells_per_layer",
            "policy_wall_seconds",
        )
    }
    trace = []
    fixed = int(method.removeprefix("fixed")) if method.startswith("fixed") else None
    context_threshold = (
        int(method.removeprefix("context")) if method.startswith("context") else None
    )
    for i, (decision, candidate) in enumerate(
        zip(path["decisions"], path["candidates"], strict=True)
    ):
        forced = decision["forced"] or i + 1 == len(validities)
        if fixed is not None:
            hazard = float(i + 1 >= fixed)
        elif context_threshold is not None:
            hazard = float(len(decision["ids"]) >= context_threshold)
        elif method == "counter":
            if hazards is None or len(hazards) <= i:
                raise ValueError("counter hazards must cover every prefix")
            hazard = hazards[i]
        elif method == "learned":
            hazard = decision["accept_probability"]
        else:
            raise ValueError("unknown stopping method")
        if not 0 <= hazard <= 1:
            raise ValueError("invalid stopping probability")
        if greedy_stop:
            hazard = float(hazard >= 0.5)
        if forced:
            hazard = 1.0
        selection = decision["selection_probabilities"]
        if len(selection) != i + 1 or abs(sum(selection) - 1) > 1e-5:
            raise ValueError("invalid selector probability distribution")
        if greedy_index:
            maximum = max(selection)
            ties = [probability == maximum for probability in selection]
            selection = [float(tie) / sum(ties) for tie in ties]
        selected_correct = sum(p * validities[j] for j, p in enumerate(selection))
        mass = reach * hazard
        correctness += mass * selected_correct
        attempts += reach
        forced_mass += mass * forced
        for key in (
            "generated_tokens",
            "forward_calls",
            "forward_tokens",
            "attention_score_cells_per_layer",
        ):
            work[key] += reach * candidate[key]
        length = len(decision["ids"])
        if method == "learned":
            controller_mass = reach if not forced or i > 0 else 0.0
        else:
            controller_mass = mass if i > 0 else 0.0
        work["controller_forward_tokens"] += controller_mass * length
        work["controller_forward_calls"] += controller_mass
        work["controller_attention_score_cells_per_layer"] += controller_mass * heads * length**2
        work["policy_wall_seconds"] += (
            reach * candidate["sampling_wall_seconds"]
            + controller_mass * decision["controller_wall_seconds"]
        )
        trace.append(
            {
                "count": i + 1,
                "reach_probability": reach,
                "stop_probability": mass,
                "hazard": hazard,
                "selected_correct_probability": selected_correct,
                "forced": forced,
            }
        )
        reach *= 1 - hazard
    if reach > 1e-6:
        raise ValueError("finite path left unterminated probability mass")
    return {
        "correct_probability": correctness,
        "attempts": attempts,
        "reward": correctness - cost * (attempts - 1),
        "forced_termination_probability": forced_mass,
        "expected_policy_work": work,
        "trace": trace,
    }


def main():
    parser = argparse.ArgumentParser()
    for key in ("source", "base", "data", "out"):
        parser.add_argument(f"--{key}", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--seed", type=int, default=52006)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--context", type=int, default=16384)
    parser.add_argument("--attempt-bound", type=int, default=1024)
    parser.add_argument("--state-reserve", type=int, default=1024)
    parser.add_argument("--max-attempts", type=int, default=8)
    parser.add_argument("--max-action", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=48)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--cost", type=float)
    parser.add_argument("--counter-hazards")
    parser.add_argument("--frozen-protocol")
    args = parser.parse_args()
    sys.path.insert(0, args.source)
    from lean_gate import lean_check
    from model import load_ckpt

    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    model, tok, _ = load_ckpt(args.base, "cuda")
    controller = Controller(model, args.max_attempts).cuda()
    protocol = vars(args).copy()
    protocol["cost"] = 0.0 if args.cost is None else args.cost
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cuda")
        controller.load_state_dict(checkpoint["state"])
        if args.cost is None:
            protocol["cost"] = checkpoint["config"].get("cost", 0.0)
        for key in (
            "context",
            "attempt_bound",
            "state_reserve",
            "temperature",
            "max_action",
            "max_steps",
            "max_attempts",
        ):
            protocol[key] = checkpoint["config"][key]
    controller.eval()
    rows = read_jsonl(args.data)
    if args.limit:
        rows = rows[: args.limit]
    if any(r.get("split") not in {"dev", "test"} for r in rows):
        raise ValueError("evaluation requires held-out theorem split")
    if len({r["split"] for r in rows}) != 1:
        raise ValueError("evaluation cannot mix development and test")
    test_evaluation = rows[0]["split"] == "test"
    if test_evaluation and args.frozen_protocol is None:
        raise ValueError("test evaluation requires --frozen-protocol")
    hazards = None
    if args.counter_hazards:
        hazard_config = json.loads(Path(args.counter_hazards).read_text())
        if hazard_config["fit_split"] != "dev":
            raise ValueError("counter hazards must be fitted on development only")
        hazards = hazard_config["hazards"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    snapshot = out / "runner_snapshot"
    snapshot.mkdir()
    for name in (
        "evaluate_live_path.py",
        "train_live.py",
        "live.py",
        "controller.py",
        "train_controller.py",
        "evaluation_protocol.py",
    ):
        shutil.copy2(Path(__file__).with_name(name), snapshot / name)
    config = protocol | {
        "started_utc": datetime.now(UTC).isoformat(),
        "pid": os.getpid(),
        "command": sys.argv,
        "sampling_precision": "fp32",
        "replay_precision": "fp32",
        "torch_version": str(torch.__version__),
        "source_commit": "4dcd0e4f336b3bf0ee24230dc5335ba3dba8969c",
        "source_hashes": {
            name: hashlib.sha256((Path(args.source) / name).read_bytes()).hexdigest()
            for name in (
                "model.py",
                "best_model.py",
                "sample.py",
                "state_env.py",
                "state_sample.py",
                "lean_tok.py",
                "tokenizer.py",
                "lean_gate.py",
                "nd2lean.py",
            )
        },
        "base_sha256": hashlib.sha256(Path(args.base).read_bytes()).hexdigest(),
        "data_sha256": hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
        "checkpoint_sha256": hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()
        if args.checkpoint
        else None,
        "runner_hashes": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in snapshot.iterdir()
        },
        "evaluation": "forced_path_exact_stop_and_index_integration",
        "evaluation_split": rows[0]["split"],
        "evaluated_theorems": len(rows),
        "live_protocol_version": PROTOCOL_VERSION,
        "evaluation_cost_rule": "explicit_override_else_checkpoint_training_cost_else_zero",
        "controller_work_rule": (
            "learned stop requires a forward at every reached prefix; fixed/count-only "
            "requires a forward only at acceptance for the shared selector; "
            "a singleton selector requires no forward, unless learned stopping is queried"
        ),
        "primary_readout": "categorical_stop_and_categorical_index_expectations",
        "prespecified_secondary_readouts": [
            "same_stop_greedy_index_uniform_ties",
            "threshold_stop_at_probability_ge_0p5_greedy_index_uniform_ties",
            "fixed_count_greedy_index_uniform_ties",
        ],
        "counter_hazards": hazards,
        "counter_hazards_sha256": hashlib.sha256(
            Path(args.counter_hazards).read_bytes()
        ).hexdigest()
        if args.counter_hazards
        else None,
    }
    current_signature = signature(config)
    config["evaluation_signature"] = current_signature
    config["inference_sha256"] = fingerprint(current_signature)
    if hazards is not None:
        validate_counter(hazard_config, current_signature)
    if test_evaluation:
        freeze_file = Path(args.frozen_protocol)
        validate_test_freeze(
            json.loads(freeze_file.read_text()),
            current_signature,
            config["data_sha256"],
            args.seed,
            args.repeats,
            len(rows),
        )
        config["frozen_protocol_sha256"] = hashlib.sha256(freeze_file.read_bytes()).hexdigest()
    (out / "config.json").write_text(json.dumps(config, indent=2))
    records, pending, coordinates = [], [], []
    settings = SimpleNamespace(**(protocol | {"fixed_count": protocol["max_attempts"]}))
    for row in rows:
        for repeat in range(args.repeats):
            seed = (
                args.seed
                + repeat * 1000003
                + int(hashlib.sha256(row["theorem_id"].encode()).hexdigest()[:8], 16)
            )
            torch.manual_seed(seed)
            path = episode(controller, tok, row, random.Random(seed), settings)
            path["repeat"] = repeat
            for index, candidate in enumerate(path["private_attempts"]):
                if candidate["parse_complete"]:
                    pending.append((tok.statement(row["prompt"]), candidate["lean_text"]))
                    coordinates.append((len(records), index))
            records.append(path)
    verdicts, lean_wall, _ = lean_check(pending)
    private_validities = [[False] * path["attempts"] for path in records]
    for (row, candidate), verdict in zip(coordinates, verdicts, strict=True):
        private_validities[row][candidate] = bool(verdict)
    methods = ["learned", "fixed1", "fixed2", "fixed4", "fixed8"]
    if hazards is not None:
        methods.append("counter")
    methods.extend([method + "_greedy_index" for method in list(methods)])
    methods.append("greedy_stop_greedy_index")
    results = {method: [] for method in methods}
    with (
        (out / "public_paths.jsonl").open("w") as public,
        (out / "private_labels.jsonl").open("w") as private,
    ):
        for path, validities in zip(records, private_validities, strict=True):
            private.write(
                json.dumps(
                    {
                        "theorem_id": path["theorem_id"],
                        "repeat": path["repeat"],
                        "validities": validities,
                    }
                )
                + "\n"
            )
            path.pop("private_attempts")
            public.write(json.dumps(path) + "\n")
            for method in methods:
                value = integrate(
                    path, validities, method, hazards, protocol["cost"], model.cfg.get("n_head", 8)
                )
                value.update(theorem_id=path["theorem_id"], repeat=path["repeat"])
                results[method].append(value)
    (out / "integrated.json").write_text(json.dumps(results))
    summary = {}
    for method, values in results.items():
        summary[method] = {
            key: sum(v[key] for v in values) / len(values)
            for key in (
                "correct_probability",
                "attempts",
                "reward",
                "forced_termination_probability",
            )
        }
        summary[method]["expected_policy_work"] = {
            key: sum(v["expected_policy_work"][key] for v in values) / len(values)
            for key in values[0]["expected_policy_work"]
        }
        summary[method]["reward_at_cost"] = {
            str(cost): summary[method]["correct_probability"]
            - cost * (summary[method]["attempts"] - 1)
            for cost in (0.0, 0.01, 0.03, 0.1)
        }
    summary["benchmark_measurement"] = {
        "paths": len(records),
        "lean_wall_seconds": lean_wall,
        "actual_generated_attempts": sum(path["attempts"] for path in records),
        "actual_generator_forward_tokens": sum(
            path["generator_forward_tokens"] for path in records
        ),
        "actual_controller_forward_tokens": sum(
            path["controller_prefill_tokens"] for path in records
        ),
        "actual_generator_forward_calls": sum(path["generator_forward_calls"] for path in records),
        "actual_policy_wall_seconds": sum(
            path["policy_execution_wall_seconds"] for path in records
        ),
        "diagnostic_replay_wall_seconds": sum(
            path["diagnostic_replay_wall_seconds"] for path in records
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
