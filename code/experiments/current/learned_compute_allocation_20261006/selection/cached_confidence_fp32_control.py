"""Small original-r16 initial-action control; no live-model transfer claim."""

import argparse
import datetime
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

import torch

FOLDER = Path(__file__).resolve().parent
sys.path.insert(0, str(FOLDER))
import cached_confidence_diagnostic as primary  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--panel", required=True)
    parser.add_argument("--clock-tool-anchor-utc", required=True)
    parser.add_argument("--monotonic-budget-seconds", type=float, required=True)
    args = parser.parse_args()
    start = time.monotonic()
    deadline = start + args.monotonic_budget_seconds
    torch.set_num_threads(4)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    admission = primary.admit_shared_gpu(3, 65536)
    rows = [json.loads(line) for line in Path(args.panel).read_text().splitlines()]
    assert len(rows) == 16 and {row["split"] for row in rows} == {"dev"}
    ck = (
        primary.ROOT
        / "artifacts/learned_compute_allocation_20261006/checkpoints/la_T1_best12_s0_r16.pt"
    )
    model, tok, _ = primary.collect.load_ckpt(ck, "cuda")
    pending = [(r, k) for r in rows for k in range(8)]

    def prefix(index):
        row, candidate = pending[index]
        base = random.Random(124006 * 1000003 + index).randint(0, 32)
        env = primary.collect.Env(row["prompt"], canon=True, base=base, assign=True)
        return primary.collect.prompt_ids(tok, env)

    @torch.no_grad()
    def generate(prefixes, bf16, instrumented):
        if time.monotonic() >= deadline:
            raise TimeoutError("Clock-tool-anchored monotonic budget exhausted")
        capture = primary.Capture(tok.eos)
        if instrumented:
            primary.sample._gumbel_pick = capture.pick
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
                output = primary.sample.generate_ids_fast(
                    model,
                    tok,
                    prefixes,
                    greedy=False,
                    temperature=0.8,
                    max_new=256,
                    seed=1240001,
                    early="eos",
                    compact=True,
                )
        finally:
            primary.sample._gumbel_pick = primary.ORIGINAL_PICK
        output = [primary.collect.trim(o, tok) for o in output]
        return output, capture.rows(output) if instrumented else None

    bf16_prefixes = [prefix(i) for i in range(8)]
    baseline, _ = generate(bf16_prefixes, True, False)
    instrumented, _ = generate(bf16_prefixes, True, True)
    assert baseline == instrumented
    identity = {
        "original": baseline,
        "instrumented": instrumented,
        "same_seed": 1240001,
        "identical": True,
    }
    (out / "bf16_identity_raw.json").write_text(json.dumps(identity) + "\n")
    indices = [0, 1, 2, 3, 64, 65, 66, 67]
    prefixes = [prefix(i) for i in indices]
    baseline, _ = generate(prefixes, False, False)
    instrumented, cached = generate(prefixes, False, True)
    assert baseline == instrumented
    if time.monotonic() >= deadline:
        raise TimeoutError("Clock-tool-anchored monotonic budget exhausted")
    sequences = [p + o for p, o in zip(prefixes, instrumented, strict=True)]
    ids = torch.zeros((len(sequences), max(map(len, sequences))), dtype=torch.long, device="cuda")
    for i, seq in enumerate(sequences):
        ids[i, : len(seq)] = torch.tensor(seq, device="cuda")
    with torch.no_grad(), torch.autocast("cuda", enabled=False):
        logits = model(ids).float()
    replay = []
    for i, (p, o) in enumerate(zip(prefixes, instrumented, strict=True)):
        lo = logits[i, len(p) - 1 : len(p) + len(o) - 1]
        base = lo.log_softmax(-1)
        sample = (lo / 0.8).log_softmax(-1)
        target = torch.tensor(o, device="cuda")[:, None]
        replay.append(
            [
                sample.gather(-1, target).sum().item(),
                base.gather(-1, target).sum().item(),
                (-(base.exp() * base).sum(-1)).sum().item(),
            ]
        )
    records = []
    for index, p, o, c, r in zip(indices, prefixes, instrumented, cached, replay, strict=True):
        row, candidate = pending[index]
        records.append(
            {
                "theorem_id": row["theorem_id"],
                "candidate_id": candidate,
                "family": row["family"],
                "prefix": p,
                "raw_action": o,
                "tokens": len(o),
                "cached": c,
                "replay": r,
                "cached_minus_replay_mean": [(a - b) / len(o) for a, b in zip(c, r, strict=True)],
            }
        )
    (out / "fp32_action_records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    (out / "fp32_identity_raw.json").write_text(
        json.dumps(
            {
                "original": baseline,
                "instrumented": instrumented,
                "identical": True,
                "same_seed": 1240001,
            }
        )
        + "\n"
    )
    paths = [
        Path(__file__),
        Path(primary.__file__),
        Path(primary.collect.__file__),
        Path(primary.sample.__file__),
    ]
    snapshot = out / "source_snapshot"
    for p in paths:
        target = snapshot / p.relative_to(primary.ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)
    result = {
        "scope": "DEV only; original r16 FP32 initial actions, not live-format checkpoints",
        "rows": 8,
        "precision": "FP32 model/logits/logsoftmax/entropy",
        "base_temperature": 1,
        "behavior_temperature": 0.8,
        "max_absolute_mean_difference": [
            max(abs(r["cached_minus_replay_mean"][j]) for r in records) for j in range(3)
        ],
        "source_hashes": {str(p): primary.digest(p) for p in paths},
        "checkpoint_sha256": primary.digest(ck),
        "panel_sha256": primary.digest(args.panel),
        "gpu_admission": admission,
        "clock_tool_anchor_utc": args.clock_tool_anchor_utc,
        "monotonic_budget_seconds": args.monotonic_budget_seconds,
        "process_wall_seconds_monotonic": time.monotonic() - start,
        "host_wall_metadata_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "pid": os.getpid(),
        "interpretation": "Numerical comparison only, no cost substitution",
        "reduction_caveat": (
            "Cached FP32 token values summed in Python FP64; replay reduces each action in FP32"
        ),
    }
    (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
