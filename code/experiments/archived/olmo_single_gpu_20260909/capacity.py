"""Measure the actual 32B model's single-GPU inference and adapter-update memory."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time

from modeling import load_model, load_tokenizer, memory_snapshot, sample_completions, adapter_digest
from protocol import render_prompt, render_completion, judge


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--new-tokens", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--icl-demonstrations", type=int, choices=(0, 32), default=32)
    parser.add_argument("--skip-update", action="store_true")
    args = parser.parse_args()
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("Capacity probe requires exactly one visible CUDA GPU")
    args.out.mkdir(parents=True, exist_ok=False)
    records = []
    def record(kind, **kwargs):
        row = dict(event=kind, time_unix=time.time(), memory=memory_snapshot(), **kwargs)
        records.append(row)
        (args.out / "events.json").write_text(json.dumps(records, indent=2) + "\n")
        print(json.dumps(row), flush=True)
    rows = [json.loads(line) for line in args.data.read_text().splitlines()]
    tokenizer = load_tokenizer(args.model)
    demonstrations = rows if args.icl_demonstrations else None
    prompts = [render_prompt(row["prompt"], demonstrations) for row in rows[:args.batch_size]]
    max_prompt = max(len(tokenizer.encode(prompt, add_special_tokens=False)) for prompt in prompts)
    record("tokenizer", icl_demonstrations=args.icl_demonstrations, max_prompt_tokens=max_prompt)
    t0 = time.time()
    model = load_model(args.model, trainable=True, rank=16)
    torch.cuda.synchronize()
    record("loaded", seconds=time.time()-t0, identity=model._sprint_identity,
           gpu=torch.cuda.get_device_name())
    t0 = time.time()
    outputs = sample_completions(model, tokenizer, prompts, max_new_tokens=args.new_tokens,
                                 batch_size=args.batch_size, seed=20260909)
    torch.cuda.synchronize()
    seconds = time.time()-t0
    (args.out / "icl_train_probe.json").write_text(json.dumps(outputs, indent=2) + "\n")
    record("icl_generation", seconds=seconds, generated_tokens=sum(row["completion_tokens"] for row in outputs),
           tokens_per_second=sum(row["completion_tokens"] for row in outputs)/seconds,
           samples=len(outputs), max_new_tokens=args.new_tokens)
    if args.skip_update:
        (args.out / "completed.json").write_text(json.dumps({"status": "completed", "events": records,
            "interpretation": "training-prompt throughput only; no held-out measurement or checkpoint"}, indent=2) + "\n")
        return
    # A real cap6 supervised update, isolated from all measured endpoint adapters.
    model.train()
    model.config.use_cache = False
    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5)
    before = adapter_digest(model)
    example = max(rows, key=lambda row: len(tokenizer.encode(render_prompt(row["prompt"]) + render_completion(row["proof"]))))
    prompt_ids = tokenizer.encode(render_prompt(example["prompt"]), add_special_tokens=False)
    target_ids = tokenizer.encode(render_completion(example["proof"]) + tokenizer.eos_token, add_special_tokens=False)
    ids = torch.tensor([prompt_ids+target_ids], device="cuda")
    labels = ids.clone()
    labels[:, :len(prompt_ids)] = -100
    t0 = time.time()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = model(input_ids=ids, labels=labels, use_cache=False).loss
    loss.backward()
    grad = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
    if not torch.isfinite(loss) or not torch.isfinite(grad) or grad <= 0:
        raise ValueError("Genuine finite nonzero adapter gradient required")
    optim.step()
    optim.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    after = adapter_digest(model)
    if before["sha256"] == after["sha256"]:
        raise ValueError("Adapter update did not change weights")
    record("supervised_update", seconds=time.time()-t0, loss=float(loss.detach()), gradient_norm=float(grad),
           sequence_tokens=ids.shape[1], completion_tokens=len(target_ids), before=before, after=after)
    (args.out / "completed.json").write_text(json.dumps({"status": "completed", "events": records,
        "interpretation": "mechanical capacity only; this adapter is discarded and is not an experimental endpoint"}, indent=2) + "\n")


if __name__ == "__main__":
    main()
