"""CPU readout and canonical label joins for the separate frozen-trunk critic."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_cache(directory, split):
    folder = Path(directory)
    metadata = json.loads((folder / f"{split}_metadata.json").read_text())
    assert metadata["split"] == split and metadata["standardization"] is None
    path = folder / f"{split}_features.npz"
    assert digest(path) == metadata["cache_sha256"]
    with np.load(path, allow_pickle=False) as stored:
        result = {key: stored[key].copy() for key in stored.files}
    assert result["features"].shape == (metadata["candidates"], 384)
    assert np.isfinite(result["features"]).all()
    keys = list(zip(result["theorem_ids"], result["candidate_ids"], strict=True))
    assert len(keys) == len(set(keys))
    return result, metadata


def aligned_labels(cache, private_path):
    rows = [json.loads(line) for line in Path(private_path).read_text().splitlines()]
    assert len(rows) == len({r["theorem_id"] for r in rows})
    mapping = {}
    for row in rows:
        assert sorted(c["candidate_id"] for c in row["labels"]) == list(range(8))
        for candidate in row["labels"]:
            mapping[(row["theorem_id"], candidate["candidate_id"])] = float(candidate["valid"])
    keys = list(zip(cache["theorem_ids"], cache["candidate_ids"], strict=True))
    assert set(keys) == set(mapping)
    return np.asarray([mapping[key] for key in keys], dtype=np.float64)


def predict(cache, labels, weight, bias):
    features = torch.from_numpy(cache["features"]).double()
    logits = (features @ weight + bias).detach()
    targets = torch.from_numpy(labels).double()
    probabilities = logits.sigmoid()
    binary_losses = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    brier = (probabilities - targets).square()
    indices = {}
    for index, (theorem_id, candidate_id) in enumerate(
        zip(cache["theorem_ids"], cache["candidate_ids"], strict=True)
    ):
        indices.setdefault(str(theorem_id), {})[int(candidate_id)] = index
    predictions = []
    for theorem_id, candidate_indices in indices.items():
        assert sorted(candidate_indices) == list(range(8))
        slots = [candidate_indices[candidate_id] for candidate_id in range(8)]
        raw = logits[slots]
        tied = (raw == raw.max()).nonzero().flatten().tolist()
        validity = targets[slots]
        family = str(cache["families"][slots[0]])
        lengths = cache["input_tokens"][slots].tolist()
        predictions.append(
            {
                "theorem_id": theorem_id,
                "family": family,
                "candidate_validity_logits": raw.tolist(),
                "candidate_validity_probabilities": probabilities[slots].tolist(),
                "probabilities": raw.softmax(0).tolist(),
                "argmax_tied_candidate_ids": tied,
                "argmax_tie_expected_accuracy": float(validity[tied].mean()),
                "n_valid": int(validity.sum()),
                "binary_logloss": float(binary_losses[slots].mean()),
                "brier_score": float(brier[slots].mean()),
                "candidate_order": list(range(8)),
                "candidate_input_tokens": lengths,
                "input_tokens": sum(lengths),
                "max_forward_tokens": max(lengths),
                "overflow": False,
            }
        )
    summary = {
        "theorems": len(predictions),
        "candidates": len(labels),
        "accuracy": float(np.mean([r["argmax_tie_expected_accuracy"] for r in predictions])),
        "binary_logloss": float(binary_losses.mean()),
        "brier_score": float(brier.mean()),
        "families": {
            family: {
                "theorems": sum(r["family"] == family for r in predictions),
                "accuracy": float(
                    np.mean(
                        [
                            r["argmax_tie_expected_accuracy"]
                            for r in predictions
                            if r["family"] == family
                        ]
                    )
                ),
            }
            for family in sorted({r["family"] for r in predictions})
        },
    }
    return predictions, summary
