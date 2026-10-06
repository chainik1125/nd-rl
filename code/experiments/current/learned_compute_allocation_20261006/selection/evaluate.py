"""Evaluate a frozen selector checkpoint without optimizer or training inputs."""

import argparse
import datetime
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import torch

TRAINER = Path(__file__).with_name("train.py")
spec = importlib.util.spec_from_file_location("selection_trainer", TRAINER)
trainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainer)
SHARED = trainer.ROOT / "code/shared_code"
sys.path.insert(0, str(SHARED))
from evaluation_freeze import validate_frozen_evaluation  # noqa: E402


def evaluation_signature(args):
    """Data-independent registration, safe before a test pool exists."""
    bce = TRAINER.with_name("train_bce.py") if args.objective == "binary" else None
    return {
        "method": "actual_r16_selector",
        "checkpoint_sha256": hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
        "upstream_source_commit": "4dcd0e4f336b3bf0ee24230dc5335ba3dba8969c",
        "upstream_behavior_sha256": {
            name: hashlib.sha256((trainer.SOURCE / name).read_bytes()).hexdigest()
            for name in [
                "model.py",
                "best_model.py",
                "lean_tok.py",
                "tokenizer.py",
                "nd2lean.py",
                "nd_verify/verify.py",
                "nd_verify/__init__.py",
                "record.py",
                "sample.py",
                "state_sample.py",
                "state_env.py",
            ]
        },
        "trainer_sha256": hashlib.sha256(TRAINER.read_bytes()).hexdigest(),
        "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "bce_trainer_sha256": hashlib.sha256(bce.read_bytes()).hexdigest() if bce else None,
        "freeze_validator_sha256": hashlib.sha256(
            (SHARED / "evaluation_freeze.py").read_bytes()
        ).hexdigest(),
        "context": args.context,
        "view": args.view,
        "architecture": args.architecture,
        "objective": args.objective,
        "precision": "BF16 model autocast; FP32 readout",
        "evaluation_permutation_seed": args.evaluation_seed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--public", required=True)
    parser.add_argument("--private", required=True)
    parser.add_argument("--split", choices=["dev", "test"], required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--context", type=int, default=8192)
    parser.add_argument("--view", choices=["actions", "transitions"], default="transitions")
    parser.add_argument("--architecture", choices=["pooled", "factorized"], default="pooled")
    parser.add_argument("--objective", choices=["marginal", "binary"], default="marginal")
    parser.add_argument("--frozen-test-protocol")
    parser.add_argument("--signature-only", action="store_true")
    parser.add_argument("--evaluation-seed", type=int, default=77106)
    args = parser.parse_args()
    started_utc = datetime.datetime.now(datetime.UTC).isoformat()
    signature = evaluation_signature(args)
    if args.signature_only:
        print(json.dumps(signature, sort_keys=True))
        return
    torch.set_num_threads(4)
    rows = trainer.read(args.public)
    assert all(row["split"] == args.split for row in rows)
    for row in rows:
        row["candidates"].sort(key=lambda c: c["candidate_id"])
        assert [c["candidate_id"] for c in row["candidates"]] == list(range(8))
    protocol_sha256 = None
    if args.split == "test":
        if not args.frozen_test_protocol:
            raise ValueError(
                "Test evaluation requires --frozen-test-protocol before private labels"
            )
        protocol_path = Path(args.frozen_test_protocol)
        protocol = json.loads(protocol_path.read_text())
        validate_frozen_evaluation(protocol, args.public, signature)
        protocol_sha256 = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    model, tok, metadata = trainer.load_ckpt(args.checkpoint, "cuda")
    trained_args = metadata.get("selector_config", {}).get("args")
    if trained_args is not None:
        for name in ["view", "architecture", "context"]:
            assert trained_args.get(name, "pooled" if name == "architecture" else None) == getattr(
                args, name
            )
    if args.split == "test":
        assert trained_args is None or "main_train" in trained_args["train_public"]
    objective = metadata.get("selector_config", {}).get("objective", "")
    if trained_args is not None:
        assert objective.startswith("binary") == (args.objective == "binary")
    labels = {}
    for row in trainer.read(args.private):
        assert sorted(c["candidate_id"] for c in row["labels"]) == list(range(8))
        labels[row["theorem_id"]] = [
            int(c["valid"]) for c in sorted(row["labels"], key=lambda c: c["candidate_id"])
        ]
    assert set(labels) == {row["theorem_id"] for row in rows}
    counts = {"calls": 0, "padded_forward_tokens": 0, "dense_attention_positions": 0}

    def count_forward(module, inputs):
        counts["calls"] += 1
        counts["padded_forward_tokens"] += inputs[0].numel()
        counts["dense_attention_positions"] += inputs[0].shape[0] * inputs[0].shape[1] ** 2

    model.register_forward_pre_hook(count_forward)
    evaluation_t0 = time.time()
    evaluation_module = trainer
    bce_path = None
    if args.objective == "binary":
        assert args.architecture == "factorized"
        bce_path = TRAINER.with_name("train_bce.py")
        bce_spec = importlib.util.spec_from_file_location("selection_bce_trainer", bce_path)
        evaluation_module = importlib.util.module_from_spec(bce_spec)
        bce_spec.loader.exec_module(evaluation_module)
    prediction = evaluation_module.evaluate(model, tok, rows, labels, args, args.evaluation_seed)
    if args.architecture == "factorized":
        public_by_id = {row["theorem_id"]: row for row in rows}
        for item in prediction:
            sequences = trainer.model_inputs(
                public_by_id[item["theorem_id"]], tok, list(range(8)), args.view, "factorized"
            )
            item["candidate_input_tokens"] = list(map(len, sequences))
            assert sum(item["candidate_input_tokens"]) == item["input_tokens"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "predictions.json").write_text(json.dumps(prediction))
    config = {
        "args": vars(args),
        "argv": sys.argv,
        "pid": os.getpid(),
        "torch_version": str(torch.__version__),
        "source_commit": "4dcd0e4f336b3bf0ee24230dc5335ba3dba8969c",
        "started_utc": started_utc,
        "evaluation_wall_seconds": time.time() - evaluation_t0,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "source_hashes": {
            str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest()
            for p in [args.checkpoint, args.public, args.private, TRAINER, __file__]
            + ([bce_path] if bce_path else [])
        },
        "compute": counts,
        "precision": "BF16 autocast forwards; FP32 readout",
        "checkpoint_metadata": metadata,
        "registered_signature": signature,
        "frozen_test_protocol_sha256": protocol_sha256,
    }
    (out / "config.json").write_text(json.dumps(config, indent=2, default=str))
    (out / "evaluate.py").write_bytes(Path(__file__).read_bytes())
    (out / "trainer.py").write_bytes(TRAINER.read_bytes())
    if bce_path:
        (out / "bce_trainer.py").write_bytes(bce_path.read_bytes())
    print(
        json.dumps(
            {
                "theorems": len(prediction),
                "accuracy": sum(p["argmax_tie_expected_accuracy"] for p in prediction)
                / len(prediction),
                "compute": counts,
            }
        )
    )


if __name__ == "__main__":
    main()
