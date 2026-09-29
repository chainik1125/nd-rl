"""Audit exact short-proof exposure matching after the retention controls finish.

Reads completed archives and checkpoints only. It reconstructs each retention
draw from the declared seed, checks every control update against all four main
updates, and binds each reported comparison to its exact saved checkpoint.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time

import torch

import common
from nd_environment import Tokenizer, episode, split_body


CONDITIONS = ("relative_full", "relative_reset", "ppo_full", "ppo_reset")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit(seed):
    started = time.monotonic()
    root = common.ROOT
    directory = root / f"results/retention_seed{seed}_v1"
    manifest = json.loads((directory / "manifest.json").read_text())
    result = json.loads((directory / "result.json").read_text())
    config = manifest["config"]
    mapping_path = root / "configs/retention_endpoint_mapping_v1.json"
    mapping = json.loads(mapping_path.read_text())["mapping"]
    inputs = {}

    def bind(path):
        path = Path(path).resolve()
        digest = common.sha(path)
        inputs[str(path)] = digest
        return digest

    for path in (directory / "manifest.json", directory / "result.json", mapping_path,
                 config["checkpoint"], config["retention"], directory / "updates.jsonl"):
        bind(path)
    require(result["stop_reason"] == "outer_update_ceiling", "Control stopped before its exposure endpoint")
    require(result["generated_tokens"] == 0, "Retention control generated tokens")
    require(not manifest["rollout_or_outcome_gradient"] and not manifest["critic"], "Control has an RL component")
    require(common.digest(manifest["identity"]) == result["identity_sha256"] == manifest["identity_sha256"],
            "Control identity hash mismatch")
    for path, digest in manifest["identity"]["code"].items():
        require(bind(path) == digest, f"Control dependency changed: {path}")
    for key in ("checkpoint", "retention"):
        require(bind(config[key]) == manifest["identity"]["inputs"][key], f"Control input changed: {key}")

    records = common.read_jsonl(config["retention"])
    tok = Tokenizer()
    encoded = []
    for row in records:
        body = row.get("body") or row["text"].split("PRF", 1)[1].strip()
        require(len(split_body(body)) <= 6, "Retention proof exceeds external six-line cap")
        encoded.append(episode(row, 0, tok))
    require(len({row["source_id"] for row in encoded}) == len(encoded), "Repeated retention source identity")
    initial = torch.load(config["checkpoint"], map_location="cpu", weights_only=True)
    require(initial["tokenizer"] == tok.state_dict(), "Retention tokenizer differs from initial checkpoint")
    updates = common.read_jsonl(directory / "updates.jsonl")
    require(len(updates) == result["step"] == config["max_updates"], "Incomplete control update coverage")
    rng = random.Random(config["seed"] + 1000003)
    digest = ""
    tokens = examples = 0
    rng_states = {}
    for step, row in enumerate(updates, 1):
        require(row["step"] == step, "Control update steps are out of order")
        selected = [encoded[rng.randrange(len(encoded))] for _ in range(config["retention_batch"])]
        ids = [item["source_id"] for item in selected]
        digest = common.digest([digest, ids])
        tokens += sum(len(item["response_ids"]) for item in selected) * config["epochs"]
        examples += len(selected) * config["epochs"]
        require(row["retention_source_ids"] == ids, f"Actual retention draws differ at step {step}")
        require(row["retention_schedule_digest"] == digest, f"Control schedule digest differs at step {step}")
        require(row["retention_examples_total"] == examples, f"Control example total differs at step {step}")
        require(row["retention_response_tokens_total"] == tokens, f"Control token total differs at step {step}")
        require(row["optimizer_steps_total"] == step * config["epochs"], f"Control epoch total differs at step {step}")
        require(len(row["epochs"]) == config["epochs"], "Missing optimizer epoch ledger")
        if step in config["checkpoint_steps"]:
            rng_states[step] = rng.getstate()
    require((digest, examples, tokens) == (result["retention_schedule_digest"], result["retention_examples"],
            result["retention_response_tokens"]), "Control final exposure ledger mismatch")
    require(bind(directory / "final.pt") == result["final_checkpoint_sha256"], "Control final checkpoint changed")

    pairs = []
    for condition in CONDITIONS:
        name = f"main_{condition}_seed{seed}_v1"
        main_dir = root / "results" / name
        main_result = json.loads((main_dir / "result.json").read_text())
        main_manifest = json.loads((main_dir / "manifest.json").read_text())
        main_config = main_manifest["config"]
        main_updates = common.read_jsonl(main_dir / "updates.jsonl")
        for file in ("result.json", "manifest.json", "updates.jsonl"):
            bind(main_dir / file)
        require(len(main_updates) == main_result["step"], f"Incomplete main update coverage: {name}")
        for key in ("seed", "precision", "retention_batch", "epochs", "microbatch", "lr", "weight_decay",
                    "retention_weight", "grad_clip"):
            require(main_config[key] == config[key], f"Retention setting mismatch: {name}/{key}")
        for key in ("checkpoint", "retention"):
            require(bind(main_config[key]) == bind(config[key]), f"Retention input mismatch: {name}/{key}")
        previous_tokens = 0
        for step, row in enumerate(main_updates, 1):
            control = updates[step - 1]
            require(row["step"] == step, f"Main update steps are out of order: {name}")
            require(row["retention_schedule_digest"] == control["retention_schedule_digest"],
                    f"Different retained example sequence: {name}/{step}")
            require(row["retention_examples"] == config["retention_batch"] * config["epochs"],
                    f"Different example exposure: {name}/{step}")
            require(row["retention_response_tokens"] == control["retention_response_tokens_total"] - previous_tokens,
                    f"Different token exposure: {name}/{step}")
            previous_tokens = control["retention_response_tokens_total"]
        step = main_result["step"]
        expected_name = f"retention_seed{seed}_step{step:06d}"
        require(mapping[name] == expected_name, f"Endpoint mapping names wrong update: {name}")
        path = directory / f"step_{step:06d}.pt"
        checkpoint_hash = bind(path)
        control_payload = torch.load(path, map_location="cpu", weights_only=True)
        main_checkpoint = main_dir / "final.pt"
        require(bind(main_checkpoint) == main_result["final_checkpoint_sha256"], f"Main checkpoint changed: {name}")
        main_payload = torch.load(main_checkpoint, map_location="cpu", weights_only=True)
        control_state, main_state = control_payload["training_state"], main_payload["training_state"]
        require(control_payload["critic"] is None, "Control checkpoint contains a critic")
        require(control_payload["config"] == main_payload["config"] == initial["config"], "Model architecture changed")
        require(control_payload["tokenizer"] == main_payload["tokenizer"] == initial["tokenizer"], "Tokenizer changed")
        require(control_state["retention_rng_state"] == main_state["rngs"]["retention"] == rng_states[step],
                f"Endpoint retention RNG mismatch: {name}")
        for key in ("step", "optimizer_steps", "retention_examples", "retention_response_tokens", "retention_schedule_digest"):
            require(control_state[key] == main_state[key] == main_result[key], f"Endpoint exposure differs: {name}/{key}")
        require(control_state["identity_sha256"] == result["identity_sha256"], "Control endpoint identity changed")
        require(control_state["ledger"]["retention_training"] == main_state["ledger"]["retention_training"],
                f"Retention compute ledger differs: {name}")
        pairs.append({"main": name, "control": expected_name, "step": step,
                      "retention_examples": control_state["retention_examples"],
                      "retention_response_tokens": control_state["retention_response_tokens"],
                      "retention_schedule_digest": control_state["retention_schedule_digest"],
                      "main_checkpoint_sha256": main_result["final_checkpoint_sha256"],
                      "control_checkpoint_sha256": checkpoint_hash,
                      "matching": "Exact example sequence, token exposure, optimizer epochs, retention compute and RNG state"})
    return {"status": "passed", "seed_index": seed, "training_seed": config["seed"], "control_updates_checked": len(updates),
            "retention_records_checked": len(encoded), "pairs": pairs, "input_sha256": inputs,
            "audit_sha256": common.sha(__file__), "elapsed_seconds": time.monotonic() - started,
            "interpretation": "Retention exposure is matched. Total training compute and optimization trajectories differ; no additive causal subtraction is licensed."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-index", type=int, choices=(1, 2), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    report = audit(args.seed_index)
    common.write_json(args.output, report)
    print(json.dumps({key: value for key, value in report.items() if key not in ("input_sha256", "pairs")}))
