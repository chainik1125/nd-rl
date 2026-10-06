"""Held-out distribution retention/restoration audit without verifier outcomes."""

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

os.environ["ND_OFFLINE"] = "1"
import torch
import torch.nn.functional as F
from adapt_format import replay_records, selected_logits
from live import history_prefix
from train_controller import read_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--adapted", required=True)
    parser.add_argument("--public", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--states", type=int, default=512)
    parser.add_argument("--seed", type=int, default=61)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()
    sys.path.insert(0, args.source)
    from model import load_ckpt

    rows = read_jsonl(args.public)
    if any(r.get("split") != "dev" for r in rows):
        raise ValueError("format distribution audit is development-only")
    torch.set_num_threads(4)
    base, tok, _ = load_ckpt(args.base, "cuda")
    adapted, _, _ = load_ckpt(args.adapted, "cuda")
    records = replay_records(tok, rows)
    rng = random.Random(args.seed)
    records = rng.sample(records, min(args.states, len(records)))
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(
        json.dumps(
            vars(args)
            | {
                "public_sha256": hashlib.sha256(Path(args.public).read_bytes()).hexdigest(),
                "adapted_sha256": hashlib.sha256(Path(args.adapted).read_bytes()).hexdigest(),
                "base_sha256": hashlib.sha256(Path(args.base).read_bytes()).hexdigest(),
                "outcome_labels_used": False,
            },
            indent=2,
        )
    )
    summary = []
    with torch.no_grad():
        for count in [0, 1, 4]:
            examples = []
            for row, target_index, state, raw in records:
                alternatives = [c for j, c in enumerate(row["candidates"]) if j != target_index]
                rng.shuffle(alternatives)
                examples.append(
                    {
                        "state": state,
                        "raw": raw,
                        "prefix": history_prefix(tok, alternatives[:count]),
                    }
                )
            base_kl, adapted_kl, tokens = 0, 0, 0
            for start in range(0, len(examples), args.batch):
                batch = examples[start : start + args.batch]
                teacher = selected_logits(base, batch, False, tok.pad, "cuda").log_softmax(-1)
                unchanged = selected_logits(base, batch, True, tok.pad, "cuda").log_softmax(-1)
                prediction = selected_logits(adapted, batch, True, tok.pad, "cuda").log_softmax(-1)
                base_kl += float(F.kl_div(unchanged, teacher, log_target=True, reduction="sum"))
                adapted_kl += float(F.kl_div(prediction, teacher, log_target=True, reduction="sum"))
                tokens += len(teacher)
            result = {
                "prefix_count": count,
                "action_states": len(examples),
                "target_tokens": tokens,
                "base_kl_nats_per_token": base_kl / tokens,
                "adapted_kl_nats_per_token": adapted_kl / tokens,
            }
            summary.append(result)
            print(json.dumps(result), flush=True)
            (output / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
