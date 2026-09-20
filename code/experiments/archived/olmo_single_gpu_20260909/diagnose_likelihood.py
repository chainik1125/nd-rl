"""Read-only checkpoint/rollout numerical replay; never creates an optimizer.

Only the frozen 32-theorem training pool and the failed first GRPO rollout are
read. The caller, not this module, selects/authorizes the single GPU. Existing
training code and checkpoints are imported/read but never modified.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import nullcontext
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace

from modeling import (adapter_digest, assert_adapter_matches_checkpoint,
    behavior_collector, decode_completion, file_sha, inference_policy,
    load_model, load_tokenizer, memory_snapshot, trim_completion, versions)
from protocol import judge, render_prompt

SPRINT = Path(__file__).resolve().parents[1]


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_json(path, value):
    # Runtime artifacts only; refuse overwriting an earlier diagnostic result.
    with Path(path).open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def audit_inputs(args, tokenizer):
    run = json.loads((args.rollouts.parent / "run.json").read_text())
    for name, expected in run["source_sha256"].items():
        if file_sha(SPRINT / "code" / name) != expected:
            raise ValueError(f"Frozen training source changed: {name}")
    for name, expected in run["data"]["input_sha256"].items():
        if file_sha(SPRINT / "data" / name) != expected:
            raise ValueError(f"Frozen training data changed: {name}")
    for relative, expected in run["data"]["protocol"]["files"].items():
        if file_sha(SPRINT.parent / relative) != expected:
            raise ValueError(f"Frozen protocol changed: {relative}")
    expected_adapter = (SPRINT / run["args"]["adapter"]).resolve()
    if args.adapter.resolve() != expected_adapter:
        raise ValueError("Replay must use exactly the failed GRPO incoming checkpoint")
    rows = read_jsonl(args.rollouts)
    pool = {row["canonical_hash"]: row for row in read_jsonl(SPRINT / "data/shared_cap6.jsonl")}
    if len(rows) != 16 or {row["batch"] for row in rows} != {0}:
        raise ValueError("Expected only the failed, first 16-response training rollout")
    for index, row in enumerate(rows):
        if row["index"] != index or row["mechanical_reward_fixture"]:
            raise ValueError("Unexpected ordering or mechanical reward")
        gold = pool[row["canonical_hash"]]
        if gold["prompt"] != row["theorem"]:
            raise ValueError("Theorem/hash mismatch")
        if decode_completion(tokenizer, row["completion_ids"]) != row["completion"]:
            raise ValueError("Literal completion/token mismatch")
        if judge(row["theorem"], row["completion"]) != row["verdict"]:
            raise ValueError("Frozen verifier does not reproduce recorded verdict")
        if row["reward"] != row["verdict"]["reward"]:
            raise ValueError("Recorded reward/verdict mismatch")
        if len(row["completion_ids"]) != len(row["behavior_logprobs"]):
            raise ValueError("Behavior token count mismatch")
        if not all(math.isfinite(value) for value in row["behavior_logprobs"]):
            raise ValueError("Nonfinite recorded behavior probability")
        trimmed, finish = trim_completion(tokenizer, row["completion_ids"])
        if trimmed != row["completion_ids"] or finish != "final_marker":
            raise ValueError("Unexpected recorded stopping behavior")
        row["prompt_ids"] = tokenizer.encode(render_prompt(row["theorem"]), add_special_tokens=False)
        row["training_name"] = gold["name"]
    for offset in range(0, len(rows), 4):
        if len({row["canonical_hash"] for row in rows[offset:offset + 4]}) != 1:
            raise ValueError("Nonhomogeneous relative-reward group")
    return rows, run


def padded_inputs(rows, pad_id, device):
    import torch
    prompt_width = max(len(row["prompt_ids"]) for row in rows)
    response_width = max(len(row["completion_ids"]) for row in rows)
    ids, masks = [], []
    for row in rows:
        left = prompt_width - len(row["prompt_ids"])
        right = response_width - len(row["completion_ids"])
        ids.append([pad_id] * left + row["prompt_ids"] + row["completion_ids"] + [pad_id] * right)
        masks.append([0] * left + [1] * (len(row["prompt_ids"]) + len(row["completion_ids"])) + [0] * right)
    return torch.tensor(ids, device=device), torch.tensor(masks, device=device), response_width


def generation_positions(module, positional, keyword):
    """Exactly HF4.57.5 generation's attention-mask-derived position IDs."""
    attention = keyword["attention_mask"]
    positions = attention.long().cumsum(-1) - 1
    positions.masked_fill_(attention == 0, 1)
    keyword["position_ids"] = positions
    return positional, keyword


def teacher_scores(model, tokenizer, rows, training, explicit_positions, unpadded=False,
                   scorer_batch_size=1):
    import torch
    from trl import GRPOTrainer
    device = model.get_input_embeddings().weight.device
    proxy = SimpleNamespace(model_kwarg_keys={"logits_to_keep"}, temperature=1.0,
        args=SimpleNamespace(report_to=[]), accelerator=SimpleNamespace(is_main_process=True))
    model.train(training)
    hook = model.register_forward_pre_hook(generation_positions, with_kwargs=True) if explicit_positions else None
    result = []
    try:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext():
            batches = [[row] for row in rows] if unpadded else [rows]
            for batch in batches:
                ids, mask, width = padded_inputs(batch, tokenizer.pad_token_id, device)
                scores, _ = GRPOTrainer._get_per_token_logps_and_entropies(
                    proxy, model, ids, mask, width, batch_size=scorer_batch_size)
                result.extend([value[:len(row["completion_ids"])].float().cpu().tolist()
                    for row, value in zip(batch, scores, strict=True)])
    finally:
        if hook is not None:
            hook.remove()
    return result


def cached_scores(model, tokenizer, rows, batch_size):
    """Replay recorded actions via generate, recording pre-forcing logits.

    The force processor changes actions only to their already recorded values.
    The preceding collector sees the original unmodified model distribution.
    This is teacher-forced diagnostics, not new policy data or generation.
    """
    import torch
    from transformers import (GenerationConfig, LogitsProcessor, LogitsProcessorList,
        StopStringCriteria, StoppingCriteriaList)
    device = model.get_input_embeddings().weight.device
    result = []
    for offset in range(0, len(rows), batch_size):
        batch = rows[offset:offset + batch_size]
        tokens = tokenizer([render_prompt(row["theorem"]) for row in batch], padding=True,
            truncation=False, return_tensors="pt", add_special_tokens=False,
            return_token_type_ids=False).to(device)
        prompt_width = tokens["input_ids"].shape[1]
        collector = behavior_collector(1.0)

        class ReplayActions(LogitsProcessor):
            def __call__(self, input_ids, scores):
                step = input_ids.shape[1] - prompt_width
                chosen = torch.tensor([row["completion_ids"][step] if step < len(row["completion_ids"])
                    else tokenizer.pad_token_id for row in batch], device=scores.device)
                return torch.full_like(scores, -torch.inf).scatter_(1, chosen[:, None], 0.0)

        config = GenerationConfig(do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
            min_p=None, repetition_penalty=1.0, max_new_tokens=max(len(row["completion_ids"]) for row in batch),
            bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id, use_cache=True)
        with inference_policy(model):
            sequences = model.generate(**tokens, generation_config=config, logits_to_keep=1,
                stopping_criteria=StoppingCriteriaList([StopStringCriteria(tokenizer, ["</FINAL>"])]),
                logits_processor=LogitsProcessorList([collector, ReplayActions()]))
            scores = collector.finish(sequences[:, -1])
        for row, sequence, values in zip(batch, sequences[:, prompt_width:].cpu().tolist(), scores, strict=True):
            trimmed, finish = trim_completion(tokenizer, sequence)
            if trimmed != row["completion_ids"] or finish != "final_marker":
                raise ValueError("Cached replay did not preserve every recorded action")
            result.append(values[:len(trimmed)])
        del sequences, tokens
    return result


def summarize(rows, scores, tokenizer, baseline=None):
    import torch
    comparisons = []
    baseline = baseline or [row["behavior_logprobs"] for row in rows]
    for row, values, reference in zip(rows, scores, baseline, strict=True):
        if len(values) != len(reference):
            raise ValueError("Diagnostic score/token alignment mismatch")
        for index, (value, original) in enumerate(zip(values, reference, strict=True)):
            if not math.isfinite(value):
                raise FloatingPointError("Nonfinite replay logprob")
            comparisons.append({"row": row["index"], "token_index": index,
                "token_id": row["completion_ids"][index], "recorded_logprob": original,
                "replay_logprob": value, "signed_error": value - original,
                "abs_error": abs(value - original), "training_name": row["training_name"],
                "prompt_length": len(row["prompt_ids"]), "completion_length": len(values)})
    errors = torch.tensor([entry["abs_error"] for entry in comparisons])
    largest = sorted(comparisons, key=lambda entry: entry["abs_error"], reverse=True)[:30]
    for entry in largest:
        ids = rows[entry["row"]]["completion_ids"]
        index = entry["token_index"]
        entry["token_text"] = tokenizer.decode([ids[index]], skip_special_tokens=False)
        entry["preceding_context"] = tokenizer.decode(ids[max(0, index - 16):index], skip_special_tokens=False)
        entry["following_context"] = tokenizer.decode(ids[index + 1:index + 17], skip_special_tokens=False)
    return {"tokens": len(comparisons), "max_abs_error": float(errors.max()),
        "mean_abs_error": float(errors.mean()), "p99_abs_error": float(torch.quantile(errors, 0.99)),
        "tokens_above_frozen_tolerance": int((errors > 0.1).sum()),
        "frozen_tolerance": 0.1, "largest_errors": largest,
        "per_row_max": [max(entry["abs_error"] for entry in comparisons if entry["row"] == row["index"])
            for row in rows]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=SPRINT / "models/olmo32b")
    parser.add_argument("--adapter", type=Path, default=SPRINT / "runs/dpo/final")
    parser.add_argument("--rollouts", type=Path, default=SPRINT / "runs/grpo/rollouts.jsonl")
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--variants", default="stock_train,stock_eval,positions_train,positions_eval,unpadded_eval,cached_batch")
    args = parser.parse_args()
    variants = args.variants.split(",")
    allowed = {"stock_train", "stock_eval", "positions_train", "positions_eval", "unpadded_eval",
               "cached_batch", "cached_single", "positions_eval_batch4", "positions_eval_batch16"}
    if not set(variants) <= allowed or len(set(variants)) != len(variants):
        raise ValueError("Unknown/repeated replay variant")
    args.out.mkdir(parents=True, exist_ok=False)
    tokenizer = load_tokenizer(args.model)
    rows, run = audit_inputs(args, tokenizer)
    manifest = {"purpose": "training_only_failed_rollout_numerical_diagnosis_no_optimization",
        "diagnostic_sha256": file_sha(__file__), "versions": versions(),
        "frozen_source_sha256": run["source_sha256"], "rollout_sha256": file_sha(args.rollouts),
        "adapter_sha256": file_sha(args.adapter / "adapter_model.safetensors"),
        "variants": variants, "identity_and_verdict_checks_passed": True,
        "responses": len(rows), "tokens": sum(len(row["completion_ids"]) for row in rows),
        "valid": sum(row["verdict"]["accepted"] for row in rows),
        "verifier_reasons": dict(Counter(row["verdict"]["reason"] for row in rows)),
        "prompt_lengths": [len(row["prompt_ids"]) for row in rows],
        "completion_lengths": [len(row["completion_ids"]) for row in rows]}
    write_json(args.out / "manifest.json", manifest)
    write_json(args.out / "reconstructed_training_rows.json", rows)
    print(json.dumps(manifest), flush=True)
    if args.cpu_only:
        return
    import torch
    from accelerate import Accelerator
    if torch.cuda.device_count() != 1:
        raise ValueError("Diagnostic requires exactly one caller-selected visible GPU")
    model = load_model(args.model, args.adapter, trainable=True, gradient_checkpointing=True)
    before = adapter_digest(model)
    # Same BF16 autocast + output-to-FP32 wrapping used when Trainer prepares its
    # model. Omitting this changes selective_log_softmax's numerical dtype.
    accelerator = Accelerator(mixed_precision="bf16")
    model = accelerator.prepare_model(model)
    assert_adapter_matches_checkpoint(model, args.adapter)
    write_json(args.out / "model_identity.json", {"identity": model._sprint_identity,
        "adapter_digest": before, "accelerator_mixed_precision": accelerator.mixed_precision,
        "gradient_checkpointing": model.is_gradient_checkpointing})
    results, scores_by_variant = {}, {}
    for variant in variants:
        started = time.monotonic()
        if variant.startswith("cached_"):
            scores = cached_scores(model, tokenizer, rows, len(rows) if variant == "cached_batch" else 1)
        else:
            scores = teacher_scores(model, tokenizer, rows,
                training=variant.endswith("_train"), explicit_positions=variant.startswith("positions_"),
                unpadded=variant == "unpadded_eval",
                scorer_batch_size=16 if variant.endswith("batch16") else 4 if variant.endswith("batch4") else 1)
        result = summarize(rows, scores, tokenizer)
        result.update(variant=variant, seconds=time.monotonic() - started, memory=memory_snapshot())
        if "stock_train" in scores_by_variant and variant != "stock_train":
            result["versus_stock_train"] = summarize(rows, scores, tokenizer, scores_by_variant["stock_train"])
        if "cached_batch" in scores_by_variant and variant != "cached_batch":
            result["versus_cached_batch"] = summarize(rows, scores, tokenizer, scores_by_variant["cached_batch"])
        write_json(args.out / f"{variant}.json", {"summary": result, "per_token_logprobs": scores})
        print(json.dumps({key: value for key, value in result.items()
            if key not in {"largest_errors", "versus_stock_train", "versus_cached_batch"}}), flush=True)
        scores_by_variant[variant] = scores
        results[variant] = result
    after = adapter_digest(model)
    if before != after:
        raise RuntimeError("Read-only diagnostic changed adapter parameters")
    assert_adapter_matches_checkpoint(model, args.adapter)
    write_json(args.out / "summary.json", {"status": "complete", "results": results,
        "adapter_before": before, "adapter_after": after, "optimizer_steps": 0,
        "frozen_tolerance_changed": False})


if __name__ == "__main__":
    main()
