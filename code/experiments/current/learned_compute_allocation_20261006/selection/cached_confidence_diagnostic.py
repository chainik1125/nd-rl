"""DEV-only numeric comparison of cached sampler logits and frozen replay."""

import argparse
import datetime
import hashlib
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

import torch

FOLDER = Path(__file__).resolve().parent
ROOT = FOLDER.parents[4]
sys.path.insert(0, str(FOLDER))
import collect  # noqa: E402
import sample  # noqa: E402
from run_frozen_test_queue_shared import admit_shared_gpu  # noqa: E402

ORIGINAL_PICK = sample._gumbel_pick


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Capture:
    def __init__(self, eos):
        self.eos = eos
        self.events = []
        self.finished = set()

    def pick(self, logits, temperature, rgen, seed, t, n_slots, slots):
        chosen = ORIGINAL_PICK(logits, temperature, rgen, seed, t, n_slots, slots)
        base = logits.float().log_softmax(-1)
        behavior = (logits.float() / temperature).log_softmax(-1)
        values = (
            torch.stack(
                [
                    behavior.gather(-1, chosen[:, None]).squeeze(-1),
                    base.gather(-1, chosen[:, None]).squeeze(-1),
                    -(base.exp() * base).sum(-1),
                ],
                dim=-1,
            )
            .cpu()
            .tolist()
        )
        for slot, token, value in zip(slots.tolist(), chosen.tolist(), values, strict=True):
            if slot in self.finished:
                continue
            self.events.append({"slot": slot, "t": t, "token": token, "values": value})
            if token == self.eos:
                self.finished.add(slot)
        return chosen

    def rows(self, outputs):
        rows = []
        for slot, output in enumerate(outputs):
            events = [e for e in self.events if e["slot"] == slot]
            assert [e["token"] for e in events] == output
            assert [e["t"] for e in events] == list(range(len(output)))
            rows.append([sum(e["values"][j] for e in events) for j in range(3)])
        return rows


@torch.no_grad()
def generate(model, tok, prefixes, seed, temperature, cap, instrumented):
    capture = Capture(tok.eos)
    if instrumented:
        sample._gumbel_pick = capture.pick
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = sample.generate_ids_fast(
                model,
                tok,
                prefixes,
                greedy=False,
                temperature=temperature,
                max_new=cap,
                seed=seed,
                early="eos",
                compact=True,
            )
    finally:
        sample._gumbel_pick = ORIGINAL_PICK
    outputs = [collect.trim(output, tok) for output in outputs]
    return outputs, capture.rows(outputs) if instrumented else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--gpu", type=int, default=3)
    args = parser.parse_args()
    torch.set_num_threads(4)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    admission = admit_shared_gpu(args.gpu, 65536)
    dev = ROOT / "artifacts/learned_compute_allocation_20261006/data/main_v1/dev.jsonl"
    source_rows = [json.loads(line) for line in dev.read_text().splitlines()]
    assert {r["split"] for r in source_rows} == {"dev"}
    selected = []
    for family in ("forward", "ore2"):
        candidates = [r for r in source_rows if r["family"] == family]
        selected += sorted(
            candidates,
            key=lambda r: hashlib.sha256(
                ("cached_diagnostic124006:" + r["theorem_id"]).encode()
            ).hexdigest(),
        )[:8]
    panel = [
        {k: row[k] for k in ("theorem_id", "prompt", "split", "family", "structural_key")}
        for row in selected
    ]
    (out / "public_panel.jsonl").write_text("".join(json.dumps(r) + "\n" for r in panel))
    ck = ROOT / "artifacts/learned_compute_allocation_20261006/checkpoints/la_T1_best12_s0_r16.pt"
    paths = [Path(__file__), Path(collect.__file__), Path(sample.__file__)]
    paths += [
        collect.SOURCE / name
        for name in (
            "model.py",
            "best_model.py",
            "state_env.py",
            "state_sample.py",
            "tokenizer.py",
            "lean_tok.py",
        )
    ]
    snapshot = out / "source_snapshot"
    for p in paths:
        target = snapshot / p.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)
    config = {
        "scope": "DEV engineering diagnostic only; no registered substitution or cost change",
        "gpu_admission": admission,
        "checkpoint_sha256": digest(ck),
        "dev_input_sha256": digest(dev),
        "panel_sha256": digest(out / "public_panel.jsonl"),
        "source_hashes": {str(p): digest(p) for p in paths},
        "seed": 124006,
        "k": 8,
        "batch": 128,
        "temperature": 0.8,
        "max_action": 256,
        "max_steps": 48,
        "early": "eos",
        "precision": "BF16 model autocast; FP32 logits/logsoftmax/entropy",
        "deadline_utc": "2026-10-06T08:24:00+00:00",
        "started_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "pid": os.getpid(),
    }
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    model, tok, _ = collect.load_ckpt(ck, "cuda")
    seed = config["seed"]
    pending = [(r, k) for r in panel for k in range(8)]
    live = []
    for index, (r, k) in enumerate(pending):
        base = random.Random(seed * 1000003 + index).randint(0, 32)
        live.append((r, k, collect.Env(r["prompt"], canon=True, base=base, assign=True)))
    prefixes = [collect.prompt_ids(tok, e) for _, _, e in live[:8]]
    original, _ = generate(model, tok, prefixes, 1240001, 0.8, 256, False)
    instrumented, cached = generate(model, tok, prefixes, 1240001, 0.8, 256, True)
    assert original == instrumented, "Instrumentation changed emitted tokens"
    (out / "token_identity.json").write_text(
        json.dumps(
            {"identical": True, "rows": 8, "seed": 1240001, "raw_outputs": original}, indent=2
        )
        + "\n"
    )
    records = []
    wave = 0
    complete = True
    sampling_seconds = replay_seconds = 0.0
    while live:
        if datetime.datetime.now(datetime.UTC) >= datetime.datetime.fromisoformat(
            config["deadline_utc"]
        ):
            complete = False
            break
        prefixes = [collect.prompt_ids(tok, e) for _, _, e in live]
        before = time.monotonic()
        outputs, cached = generate(model, tok, prefixes, seed * 100003 + wave, 0.8, 256, True)
        sampling_seconds += time.monotonic() - before
        before = time.monotonic()
        replay = collect.likelihood_batch(model, prefixes, outputs, 0.8)
        replay_seconds += time.monotonic() - before
        keep = []
        for slot, ((r, k, e), prefix, output, c, p) in enumerate(
            zip(live, prefixes, outputs, cached, replay, strict=True)
        ):
            records.append(
                {
                    "theorem_id": r["theorem_id"],
                    "candidate_id": k,
                    "family": r["family"],
                    "wave": wave,
                    "slot": slot,
                    "prefix": prefix,
                    "raw_action": output,
                    "cached": c,
                    "replay": p,
                    "tokens": len(output),
                    "sampled_pad_tokens": sum(t == tok.pad for t in output),
                }
            )
            action, ended = tok.decode_action(output)
            reason = not ended
            if ended:
                ok, _ = e.apply(action)
                reason = not ok or e.done or e.steps >= 48
            if not reason:
                keep.append((r, k, e))
        live = keep
        wave += 1
        if wave % 5 == 0:
            print(json.dumps({"wave": wave, "remaining_attempts": len(live)}), flush=True)
    (out / "action_records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    by_candidate = {}
    for record in records:
        key = (record["theorem_id"], record["candidate_id"])
        row = by_candidate.setdefault(
            key,
            {
                "theorem_id": key[0],
                "candidate_id": key[1],
                "tokens": 0,
                "cached": [0.0, 0.0, 0.0],
                "replay": [0.0, 0.0, 0.0],
            },
        )
        row["tokens"] += record["tokens"]
        for source in ("cached", "replay"):
            row[source] = [a + b for a, b in zip(row[source], record[source], strict=True)]
    candidates = []
    for row in by_candidate.values():
        row["cached_mean"] = [v / row["tokens"] for v in row["cached"]]
        row["replay_mean"] = [v / row["tokens"] for v in row["replay"]]
        row["cached_minus_replay_mean"] = [
            a - b for a, b in zip(row["cached_mean"], row["replay_mean"], strict=True)
        ]
        candidates.append(row)
    (out / "candidate_discrepancies.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in candidates)
    )
    summary = {
        "complete": complete,
        "candidates": len(candidates),
        "action_batches": wave,
        "action_rows": len(records),
        "diagnostic_names": ["sample_T0.8_logprob", "base_T1_logprob", "base_T1_entropy"],
        "max_absolute_candidate_mean_difference": [
            max(abs(r["cached_minus_replay_mean"][i]) for r in candidates) for i in range(3)
        ],
        "mean_absolute_candidate_mean_difference": [
            sum(abs(r["cached_minus_replay_mean"][i]) for r in candidates) / len(candidates)
            for i in range(3)
        ],
        "sampled_pad_tokens": sum(r["sampled_pad_tokens"] for r in records),
        "token_identity_control": True,
        "sampling_with_callback_seconds": sampling_seconds,
        "replay_seconds": replay_seconds,
        "process_wall_seconds_monotonic": time.monotonic() - start,
        "interpretation": "Numerical diagnostic, not a speedup or registered feature substitution",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
