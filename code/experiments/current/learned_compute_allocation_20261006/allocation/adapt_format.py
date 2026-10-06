"""Outcome-blind self-distillation for the new prior-attempt generator interface.

A frozen original base supplies next-token distributions on its original state
prompts. A fresh copy learns to reproduce those distributions with irrelevant
previous proof blocks prepended. All raw attempts are used without checker labels,
including malformed final actions. No acceptance or selection heads are trained.
"""

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

os.environ["ND_OFFLINE"] = "1"
import torch
import torch.nn.functional as F
from live import history_prefix
from train_controller import read_jsonl


def replay_records(tok, rows):
    from state_env import Env
    from state_sample import prompt_ids

    records = []
    for row in rows:
        for candidate_index, candidate in enumerate(row["candidates"]):
            env = Env(row["prompt"], canon=True, base=candidate["canonical_name_base"], assign=True)
            for raw in candidate["raw_actions"]:
                records.append((row, candidate_index, prompt_ids(tok, env), raw))
                action, ended = tok.decode_action(raw)
                if not ended:
                    break
                ok, _ = env.apply(action)
                if not ok or env.done:
                    break
    return records


def selected_logits(model, examples, use_prefix, pad, device):
    sequences, starts, targets = [], [], []
    for example in examples:
        prompt = (example["prefix"] if use_prefix else []) + example["state"]
        sequences.append(prompt + example["raw"][:-1])
        starts.append(len(prompt) - 1)
        targets.append(example["raw"])
    lengths = torch.tensor([len(s) for s in sequences], device=device)
    ids = torch.full((len(sequences), int(lengths.max())), pad, device=device, dtype=torch.long)
    for i, sequence in enumerate(sequences):
        ids[i, : len(sequence)] = torch.tensor(sequence, device=device)
    positions = torch.arange(ids.shape[1], device=device)[None].expand_as(ids)
    valid = positions < lengths[:, None]
    mask = valid[:, None, None, :] & (positions[:, None, :, None] >= positions[:, None, None, :])
    logits = model(ids, pos=positions, mask=mask)
    selected = torch.cat(
        [
            logits[i, start : start + len(target)]
            for i, (start, target) in enumerate(zip(starts, targets, strict=True))
        ]
    )
    return selected.float()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--public", required=True)
    parser.add_argument("--safe-train", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=61)
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--context", type=int, default=16384)
    parser.add_argument("--prefix-counts", type=int, nargs="+", default=[0, 1, 4])
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--clean-weight", type=float, default=1)
    parser.add_argument("--first-state-probability", type=float, default=0)
    args = parser.parse_args()
    sys.path.insert(0, args.source)
    from model import load_ckpt

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    snapshot = out / "runner_snapshot"
    snapshot.mkdir()
    for filename in ["adapt_format.py", "live.py", "train_controller.py", "controller.py"]:
        shutil.copy2(Path(__file__).with_name(filename), snapshot / filename)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    teacher, tok, _ = load_ckpt(args.base, "cuda")
    teacher.requires_grad_(False)
    student, _, _ = load_ckpt(args.base, "cuda")
    raw_rows = read_jsonl(args.public)
    safe_ids = {r["theorem_id"] for r in read_jsonl(args.safe_train) if r.get("split") == "train"}
    rows = [r for r in raw_rows if r["theorem_id"] in safe_ids]
    if not rows or any(r.get("split") != "train" for r in rows):
        raise ValueError("format adaptation requires safe train theorem pools")
    if any(max(args.prefix_counts) >= len(r["candidates"]) for r in rows):
        raise ValueError("prefix count requires distinct target plus sufficient candidates")
    records = replay_records(tok, rows)
    first_records, seen_candidates = [], set()
    for record in records:
        key = (record[0]["theorem_id"], record[1])
        if key not in seen_candidates:
            first_records.append(record)
            seen_candidates.add(key)
    config = vars(args) | {
        "started_utc": datetime.now(UTC).isoformat(),
        "pid": os.getpid(),
        "command": sys.argv,
        "base_sha256": hashlib.sha256(Path(args.base).read_bytes()).hexdigest(),
        "public_sha256": hashlib.sha256(Path(args.public).read_bytes()).hexdigest(),
        "safe_train_sha256": hashlib.sha256(Path(args.safe_train).read_bytes()).hexdigest(),
        "runner_hashes": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in snapshot.iterdir()
        },
        "rows": len(rows),
        "excluded_protected_rows": len(raw_rows) - len(rows),
        "action_records": len(records),
        "objective": "weighted_token_mean_KL_original_base_distribution_to_prefixed_student",
        "weight_rule": "clean-weight for zero-history tokens; one for nonzero-history tokens",
        "sampling_rule": (
            "first-state mixture and uniform all-action sampling, "
            "independent of prefix count and outcomes"
        ),
        "outcome_labels_used": False,
        "stop_labels_used": False,
        "teacher_precision": "fp32",
        "student_precision": "fp32",
    }
    (out / "config.json").write_text(json.dumps(config, indent=2))
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0)
    teacher.eval()
    student.eval()
    with (out / "metrics.jsonl").open("w") as log:
        for update in range(args.updates):
            start_time = time.monotonic()
            examples = []
            counts = []
            for _ in range(args.batch):
                record_pool = (
                    first_records if rng.random() < args.first_state_probability else records
                )
                row, target_index, state, raw = rng.choice(record_pool)
                count = rng.choice(args.prefix_counts)
                alternatives = [c for j, c in enumerate(row["candidates"]) if j != target_index]
                rng.shuffle(alternatives)
                prefix = history_prefix(tok, alternatives[:count])
                if len(prefix) + len(state) + len(raw) > args.context:
                    raise ValueError("format example exceeds declared context; no clipping")
                examples.append({"prefix": prefix, "state": state, "raw": raw})
                counts.append(count)
            with torch.no_grad():
                target = selected_logits(teacher, examples, False, tok.pad, "cuda").log_softmax(-1)
            prediction = selected_logits(student, examples, True, tok.pad, "cuda").log_softmax(-1)
            per_token_kl = F.kl_div(prediction, target, reduction="none", log_target=True).sum(-1)
            weights = torch.cat(
                [
                    torch.full(
                        (len(example["raw"]),),
                        args.clean_weight if count == 0 else 1.0,
                        device="cuda",
                    )
                    for example, count in zip(examples, counts, strict=True)
                ]
            )
            loss = (per_token_kl * weights).sum() / weights.sum()
            optimizer.zero_grad()
            loss.backward()
            norm = float(torch.nn.utils.clip_grad_norm_(student.parameters(), 1))
            optimizer.step()
            metric = {
                "update": update,
                "kl_nats_per_token": float(loss.detach()),
                "gradient_norm": norm,
                "seconds": time.monotonic() - start_time,
                "prefix_counts": counts,
                "target_tokens": len(target),
            }
            log.write(json.dumps(metric) + "\n")
            log.flush()
            if update % 10 == 0:
                print(json.dumps(metric), flush=True)
            if update % args.save_every == 0 or update + 1 == args.updates:
                torch.save(
                    {
                        "cfg": student.cfg,
                        "state": student.state_dict(),
                        "tok_mode": tok.mode,
                        "extra": {"format_adaptation": config, "update": update},
                    },
                    out / f"checkpoint_{update:05d}.pt",
                )


if __name__ == "__main__":
    main()
