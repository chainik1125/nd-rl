"""Independent neutral overflow handling for the frozen scalar head."""

import json
import time

import numpy as np
import torch
import torch.nn.functional as F


def extract(extraction, model, tok, public, split, batch, context, out):
    rows = extraction.encoding.read(public)
    sequences, tids, cids, families = [], [], [], []
    for row in rows:
        row["candidates"].sort(key=lambda c: c["candidate_id"])
        if row["split"] != split or [c["candidate_id"] for c in row["candidates"]] != list(
            range(8)
        ):
            raise ValueError("Split or candidate identity mismatch")
        values = extraction.encoding.model_inputs(
            row, tok, list(range(8)), "transitions", "factorized"
        )
        for cid, sequence in enumerate(values):
            sequences.append(sequence)
            tids.append(row["theorem_id"])
            cids.append(cid)
            families.append(row["family"])
    lengths = np.asarray(list(map(len, sequences)))
    overflow = lengths > context
    if not overflow.any():
        return extraction.extract(model, tok, public, split, batch, context, out)
    started = time.monotonic()
    features = np.zeros((len(sequences), 384), dtype=np.float32)
    meter = {"trunk_feature_calls": 0, "padded_forward_tokens": 0, "dense_attention_positions": 0}
    with torch.no_grad():
        for start in range(0, len(sequences), batch):
            slots = [i for i in range(start, min(start + batch, len(sequences))) if not overflow[i]]
            if not slots:
                continue
            values = [sequences[i] for i in slots]
            ids, last = extraction.encoding.tensor_batch(values, tok.pad)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                hidden = model.ln_f(model.features(ids)[0])
            features[slots] = (
                hidden[torch.arange(len(slots), device="cuda"), last].float().cpu().numpy()
            )
            meter["trunk_feature_calls"] += 1
            meter["padded_forward_tokens"] += ids.numel()
            meter["dense_attention_positions"] += ids.shape[0] * ids.shape[1] ** 2
    path = out / f"{split}_features.npz"
    np.savez_compressed(
        path,
        features=features,
        theorem_ids=np.asarray(tids),
        candidate_ids=np.asarray(cids),
        families=np.asarray(families),
        input_tokens=lengths,
        overflow=overflow,
    )
    metadata = {
        "split": split,
        "public": public,
        "public_sha256": extraction.digest(public),
        "cache_sha256": extraction.digest(path),
        "theorems": len(rows),
        "candidates": len(sequences),
        "feature_dimension": 384,
        "feature_dtype": "FP32",
        "standardization": None,
        "max_input_tokens": int(lengths.max()),
        "context": context,
        "overflow_candidates": int(overflow.sum()),
        "compute": meter,
        "extraction_wall_seconds_monotonic": time.monotonic() - started,
        "fallback": (
            "zero features are placeholders only; exact logit0/probability.5 readout override"
        ),
        "precision": "BF16 trunk/final LayerNorm; FP32 cache; no forward for overflow",
    }
    (out / f"{split}_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def predict(common, cache, labels, weight, bias):
    prediction, summary = common.predict(cache, labels, weight, bias)
    overflow = cache.get("overflow", np.zeros(len(labels), dtype=bool))
    slots = {
        (str(tid), int(cid)): i
        for i, (tid, cid) in enumerate(
            zip(cache["theorem_ids"], cache["candidate_ids"], strict=True)
        )
    }
    for item in prediction:
        indices = [slots[item["theorem_id"], cid] for cid in range(8)]
        flags = [bool(overflow[i]) for i in indices]
        item["candidate_overflow"] = flags
        item["candidate_serialized_tokens"] = item["candidate_input_tokens"].copy()
        item["candidate_input_tokens"] = [
            0 if flag else length
            for flag, length in zip(flags, item["candidate_serialized_tokens"], strict=True)
        ]
        item["input_tokens"] = sum(item["candidate_input_tokens"])
        item["max_forward_tokens"] = max(item["candidate_input_tokens"])
        item["overflow"] = any(flags)
        if not any(flags):
            continue
        logits = torch.tensor(item["candidate_validity_logits"], dtype=torch.float64)
        logits[torch.tensor(flags)] = 0
        target = torch.from_numpy(labels[indices]).double()
        probs = logits.sigmoid()
        tied = (logits == logits.max()).nonzero().flatten().tolist()
        item.update(
            candidate_validity_logits=logits.tolist(),
            candidate_validity_probabilities=probs.tolist(),
            probabilities=logits.softmax(0).tolist(),
            argmax_tied_candidate_ids=tied,
            argmax_tie_expected_accuracy=float(target[tied].mean()),
            binary_logloss=float(F.binary_cross_entropy_with_logits(logits, target)),
            brier_score=float((probs - target).square().mean()),
        )
    summary.update(
        accuracy=float(np.mean([p["argmax_tie_expected_accuracy"] for p in prediction])),
        binary_logloss=float(np.mean([p["binary_logloss"] for p in prediction])),
        brier_score=float(np.mean([p["brier_score"] for p in prediction])),
        overflow_candidates=int(overflow.sum()),
        overflow_theorems=sum(p["overflow"] for p in prediction),
    )
    summary["families"] = {
        family: {
            "theorems": sum(p["family"] == family for p in prediction),
            "accuracy": float(
                np.mean(
                    [p["argmax_tie_expected_accuracy"] for p in prediction if p["family"] == family]
                )
            ),
        }
        for family in sorted({p["family"] for p in prediction})
    }
    return prediction, summary
