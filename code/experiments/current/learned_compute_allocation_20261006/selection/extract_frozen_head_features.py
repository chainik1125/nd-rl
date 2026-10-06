"""Cache public final-LayerNorm representations without changing the original proof LM."""

import argparse
import datetime
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[5]
FOLDER = Path(__file__).parent
SOURCE = ROOT / "artifacts/learned_compute_allocation_20261006/source/nd-takehome-dan"
sys.path.insert(0, str(SOURCE))
from model import load_ckpt  # noqa: E402

spec = importlib.util.spec_from_file_location("frozen_head_input_encoding", FOLDER / "train_bce.py")
encoding = importlib.util.module_from_spec(spec)
spec.loader.exec_module(encoding)
UPSTREAM_FILES = [
    "model.py",
    "best_model.py",
    "tokenizer.py",
    "lean_tok.py",
    "nd2lean.py",
    "nd_verify/verify.py",
    "nd_verify/__init__.py",
    "record.py",
]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parameter_digest(model):
    """Digest actual frozen tensors, independent of checkpoint serialization metadata."""
    result = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        array = parameter.detach().cpu().contiguous().numpy()
        result.update(name.encode())
        result.update(str(array.dtype).encode())
        result.update(str(array.shape).encode())
        result.update(array.tobytes())
    return result.hexdigest()


def source_hashes():
    local = [
        FOLDER / name
        for name in [
            "extract_frozen_head_features.py",
            "train_bce.py",
            "train.py",
            "frozen_head_method_freeze.md",
        ]
    ]
    return {str(p): digest(p) for p in local + [SOURCE / name for name in UPSTREAM_FILES]}


@torch.no_grad()
def extract(model, tok, public_path, split, batch_size, context, out):
    rows = encoding.read(public_path)
    assert all(row["split"] == split for row in rows)
    assert len({r["theorem_id"] for r in rows}) == len(rows)
    sequences, theorem_ids, candidate_ids, families = [], [], [], []
    for row in rows:
        row["candidates"].sort(key=lambda c: c["candidate_id"])
        assert [c["candidate_id"] for c in row["candidates"]] == list(range(8))
        values = encoding.model_inputs(row, tok, list(range(8)), "transitions", "factorized")
        for candidate_id, sequence in enumerate(values):
            assert sequence[-1] == tok.stoi["<act>"]
            if len(sequence) > context:
                raise ValueError("Feature extraction refuses overflow rather than truncate/drop")
            sequences.append(sequence)
            theorem_ids.append(row["theorem_id"])
            candidate_ids.append(candidate_id)
            families.append(row["family"])
    feature_rows = []
    meter = {"trunk_feature_calls": 0, "padded_forward_tokens": 0, "dense_attention_positions": 0}
    started = time.monotonic()
    for start in range(0, len(sequences), batch_size):
        batch = sequences[start : start + batch_size]
        ids, last = encoding.tensor_batch(batch, tok.pad)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hidden = model.ln_f(model.features(ids)[0])
        features = hidden[torch.arange(len(batch), device="cuda"), last].float().cpu().numpy()
        assert features.shape == (len(batch), 384)
        feature_rows.append(features)
        meter["trunk_feature_calls"] += 1
        meter["padded_forward_tokens"] += ids.numel()
        meter["dense_attention_positions"] += len(batch) * ids.shape[1] ** 2
    wall = time.monotonic() - started
    cache = out / f"{split}_features.npz"
    np.savez(
        cache,
        features=np.concatenate(feature_rows),
        theorem_ids=np.asarray(theorem_ids),
        candidate_ids=np.asarray(candidate_ids),
        families=np.asarray(families),
        input_tokens=np.asarray(list(map(len, sequences))),
    )
    metadata = {
        "split": split,
        "public": public_path,
        "public_sha256": digest(public_path),
        "cache_sha256": digest(cache),
        "theorems": len(rows),
        "candidates": len(sequences),
        "feature_dimension": 384,
        "feature_dtype": "FP32",
        "standardization": None,
        "max_input_tokens": max(map(len, sequences)),
        "context": context,
        "extraction_wall_seconds_monotonic": wall,
        "compute": meter,
        "position": "last nonpadding final <act>; frozen final LayerNorm output",
        "precision": "BF16 model autocast, cached FP32 hidden vectors; no LM-head evaluation",
    }
    (out / f"{split}_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train-public", required=True)
    parser.add_argument("--dev-public", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--context", type=int, default=8192)
    args = parser.parse_args()
    process_start = time.monotonic()
    started_utc = datetime.datetime.now(datetime.UTC).isoformat()
    assert (
        digest(args.checkpoint)
        == "ba5d03de5dba890be643da82f9edca826d254c862cdcd33c221954b70918da24"
    )
    torch.set_num_threads(4)
    torch.manual_seed(0)
    model, tok, _ = load_ckpt(args.checkpoint, "cuda")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    before = parameter_digest(model)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    snapshot = out / "source_snapshot"
    snapshot.mkdir()
    for name in [
        "extract_frozen_head_features.py",
        "train_bce.py",
        "train.py",
        "frozen_head_method_freeze.md",
    ]:
        (snapshot / name).write_bytes((FOLDER / name).read_bytes())
    config = {
        "args": vars(args),
        "argv": sys.argv,
        "started_utc": started_utc,
        "pid": os.getpid(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "checkpoint_sha256": digest(args.checkpoint),
        "source_hashes": source_hashes(),
        "torch_version": str(torch.__version__),
        "parameter_sha256_before": before,
        "trainable_original_parameters": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
    }
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    summaries = [
        extract(model, tok, args.train_public, "train", args.batch, args.context, out),
        extract(model, tok, args.dev_public, "dev", args.batch, args.context, out),
    ]
    after = parameter_digest(model)
    assert before == after
    config.update(
        parameter_sha256_after=after,
        original_parameters_unchanged=True,
        process_wall_seconds_monotonic=time.monotonic() - process_start,
        splits=summaries,
    )
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(json.dumps({"original_parameters_unchanged": True, "splits": summaries}))


if __name__ == "__main__":
    main()
