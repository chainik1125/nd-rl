"""Candidate numerical policies on recorded training actions; no optimization.

This standalone probe does not amend the frozen model, trainer, or experiment.
Every candidate is compared with its OWN cached action probabilities, while
drift from the original BF16 policy is reported separately. Checkpoint tensors
are unchanged. Temporary in-process wrappers are always restored.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack
from functools import wraps
import json
from pathlib import Path
import time
from types import SimpleNamespace

from diagnose_likelihood import (SPRINT, audit_inputs, cached_scores, generation_positions,
    padded_inputs, summarize, write_json)
from modeling import (adapter_digest, assert_adapter_matches_checkpoint, file_sha,
    load_model, load_tokenizer, memory_snapshot, versions)


@contextmanager
def candidate_precision(model, variant):
    import torch
    from torch.nn.attention import sdpa_kernel, SDPBackend
    from transformers.models.olmo3.modeling_olmo3 import ALL_ATTENTION_FUNCTIONS
    allowed = {"reduction_fp32", "sdpa_math", "attention_fp32", "head_fp32", "head_attention_fp32",
               "attention_fp32_efficient", "head_attention_fp32_efficient", "lora_fp32", "all_fp32"}
    if variant not in allowed:
        raise ValueError(variant)
    device_type = model.get_input_embeddings().weight.device.type
    observation = {"variant": variant, "attention_input_dtypes": set(),
        "attention_output_dtypes": set(), "head_output_dtypes": set(),
        "attention_query_shapes": set(), "sdpa_kernel": "EFFICIENT_ATTENTION" if variant.endswith("_efficient") else "MATH" if variant in
        {"sdpa_math", "attention_fp32", "head_attention_fp32", "all_fp32"} else "default",
        "fp32_attention_inputs": variant in {"attention_fp32", "head_attention_fp32", "all_fp32"} or variant.endswith("_efficient"),
        "fp32_head": variant in {"head_fp32", "head_attention_fp32", "head_attention_fp32_efficient", "all_fp32"},
        "fp32_lora": variant in {"lora_fp32", "all_fp32"},
        "fp32_bnb_compute": variant == "all_fp32",
        "bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        "tf32_allowed": torch.backends.cuda.matmul.allow_tf32}
    registry = ALL_ATTENTION_FUNCTIONS
    original_attention = registry["sdpa"]
    had_local = "sdpa" in registry._local_mapping
    previous_local = registry._local_mapping.get("sdpa")

    def attention(module, query, key, value, attention_mask, **kwargs):
        observation["attention_input_dtypes"].add(str(query.dtype))
        observation["attention_query_shapes"].add(tuple(query.shape))
        if observation["fp32_attention_inputs"]:
            if attention_mask is not None and attention_mask.is_floating_point():
                attention_mask = attention_mask.float()
            with torch.autocast(device_type, enabled=False):
                result = original_attention(module, query.float(), key.float(), value.float(), attention_mask, **kwargs)
        else:
            result = original_attention(module, query, key, value, attention_mask, **kwargs)
        observation["attention_output_dtypes"].add(str(result[0].dtype))
        return result

    def restore_attention():
        if had_local:
            registry["sdpa"] = previous_local
        else:
            del registry["sdpa"]

    def fp32_linear(module, stack):
        if module.weight.dtype != torch.float32:
            raise ValueError("FP32 linear probe requires existing FP32 checkpoint parameters")
        original = module.forward

        @wraps(original)
        def forward(inputs, *args, **kwargs):
            with torch.autocast(device_type, enabled=False):
                return original(inputs.float(), *args, **kwargs)

        module.forward = forward
        stack.callback(setattr, module, "forward", original)

    with ExitStack() as stack:
        registry["sdpa"] = attention
        stack.callback(restore_attention)
        head = model.get_output_embeddings()
        hook = head.register_forward_hook(lambda module, inputs, output:
            observation["head_output_dtypes"].add(str(output.dtype)))
        stack.callback(hook.remove)
        if observation["sdpa_kernel"] == "MATH":
            stack.enter_context(sdpa_kernel(SDPBackend.MATH))
        elif observation["sdpa_kernel"] == "EFFICIENT_ATTENTION":
            stack.enter_context(sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION))
        if variant in {"head_fp32", "head_attention_fp32", "head_attention_fp32_efficient"}:
            fp32_linear(head, stack)
        if variant == "lora_fp32":
            count = 0
            for name, module in model.named_modules():
                if (".lora_A." in name or ".lora_B." in name) and isinstance(module, torch.nn.Linear):
                    fp32_linear(module, stack)
                    count += 1
            observation["wrapped_lora_linears"] = count
            if count == 0:
                raise ValueError("No LoRA linear modules found")
        if variant in {"reduction_fp32", "all_fp32"}:
            previous_reduction = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
            previous_tf32 = torch.backends.cuda.matmul.allow_tf32
            previous_precision = torch.get_float32_matmul_precision()
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
            torch.set_float32_matmul_precision("highest")
            torch.backends.cuda.matmul.allow_tf32 = False
            stack.callback(setattr, torch.backends.cuda.matmul, "allow_bf16_reduced_precision_reduction", previous_reduction)
            stack.callback(setattr, torch.backends.cuda.matmul, "allow_tf32", previous_tf32)
            stack.callback(torch.set_float32_matmul_precision, previous_precision)
            observation["bf16_reduced_precision_reduction"] = False
            observation["tf32_allowed"] = False
        if variant == "all_fp32":
            from bitsandbytes.nn import Linear4bit
            count = 0
            for module in model.modules():
                if isinstance(module, Linear4bit):
                    previous = module.compute_dtype
                    module.compute_dtype = torch.float32
                    stack.callback(setattr, module, "compute_dtype", previous)
                    count += 1
            observation["fp32_bnb_linears"] = count
            # Disable autocast INSIDE the underlying causal-LM call. An outer
            # context alone would be overridden by Accelerate's forward wrapper,
            # and PEFT.generate bypasses PEFT.forward by calling its base model.
            causal_lm = model.get_base_model() if hasattr(model, "get_base_model") else model
            original = causal_lm.forward

            @wraps(original)
            def full_precision(*args, **kwargs):
                with torch.autocast(device_type, enabled=False):
                    return original(*args, **kwargs)

            causal_lm.forward = full_precision
            stack.callback(setattr, causal_lm, "forward", original)
        yield observation


def serializable_observation(observation):
    return {key: sorted(value) if isinstance(value, set) else value for key, value in observation.items()}


def differentiable_teacher_scores(model, tokenizer, rows, backward=True):
    """Actual TRL scoring forwards, releasing each microbatch graph separately.

    Score tensors retain gradients exactly as in the loss path. One preselected
    training token exercises backward/checkpoint recomputation, with no optimizer
    and with gradients cleared before returning. No rewards or actions change.
    """
    import torch
    from trl import GRPOTrainer
    model.train()
    model.zero_grad(set_to_none=True)
    device = model.get_input_embeddings().weight.device
    ids, masks, width = padded_inputs(rows, tokenizer.pad_token_id, device)
    proxy = SimpleNamespace(model_kwarg_keys={"logits_to_keep"}, temperature=1.0,
        args=SimpleNamespace(report_to=[]), accelerator=SimpleNamespace(is_main_process=True))
    hook = model.register_forward_pre_hook(generation_positions, with_kwargs=True)
    results = []
    audit = {"gradient_enabled": True, "model_training": True, "microbatch_size": 1,
        "backward_exercised": False, "optimizer_steps": 0}
    try:
        with torch.enable_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            for index, row in enumerate(rows):
                scores, _ = GRPOTrainer._get_per_token_logps_and_entropies(proxy, model,
                    ids[index:index + 1], masks[index:index + 1], width, batch_size=1)
                if not scores.requires_grad:
                    raise RuntimeError("TRL teacher scores lack their expected autograd graph")
                results.append(scores[0, :len(row["completion_ids"])].detach().float().cpu().tolist())
                if backward and index == min(4, len(rows) - 1):
                    token = min(58, len(row["completion_ids"]) - 1)
                    (-scores[0, token]).backward()
                    norm_squared, count = 0.0, 0
                    for name, parameter in model.named_parameters():
                        if parameter.grad is not None:
                            if "lora_" not in name:
                                raise RuntimeError("Unexpected nonadapter gradient")
                            if not torch.isfinite(parameter.grad).all():
                                raise FloatingPointError("Nonfinite adapter gradient in candidate policy")
                            norm_squared += float(parameter.grad.detach().float().square().sum())
                            count += 1
                    if norm_squared <= 0 or count == 0:
                        raise RuntimeError("Candidate's diagnostic backward produced no adapter gradient")
                    audit.update(backward_exercised=True, backward_row=index, backward_token=token,
                        gradient_norm=norm_squared ** 0.5, tensors_with_gradients=count,
                        gradient_checkpointing=bool(getattr(model, "is_gradient_checkpointing", False)))
                    model.zero_grad(set_to_none=True)
                del scores
    finally:
        hook.remove()
        model.zero_grad(set_to_none=True)
    return results, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=SPRINT / "models/olmo32b")
    parser.add_argument("--adapter", type=Path, default=SPRINT / "runs/dpo/final")
    parser.add_argument("--rollouts", type=Path, default=SPRINT / "runs/grpo/rollouts.jsonl")
    parser.add_argument("--variants", default="head_fp32,reduction_fp32,sdpa_math,head_attention_fp32")
    parser.add_argument("--audit-backward", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    variants = args.variants.split(",")
    allowed = {"reduction_fp32", "sdpa_math", "attention_fp32", "head_fp32", "head_attention_fp32",
        "attention_fp32_efficient", "head_attention_fp32_efficient", "lora_fp32", "all_fp32"}
    if not set(variants) <= allowed or len(set(variants)) != len(variants):
        raise ValueError("Unknown or repeated precision variant")
    args.out.mkdir(parents=True, exist_ok=False)
    tokenizer = load_tokenizer(args.model)
    rows, run = audit_inputs(args, tokenizer)
    manifest = {"purpose": "training_only_candidate_precision_diagnosis_no_optimization",
        "diagnostic_sha256": file_sha(__file__), "replay_helper_sha256": file_sha(SPRINT / "code/diagnose_likelihood.py"),
        "frozen_source_sha256": run["source_sha256"], "versions": versions(),
        "adapter_sha256": file_sha(args.adapter / "adapter_model.safetensors"),
        "rollouts_sha256": file_sha(args.rollouts), "variants": variants,
        "frozen_max_abs_logprob_error": 0.1, "token_count": sum(len(row["completion_ids"]) for row in rows)}
    write_json(args.out / "manifest.json", manifest)
    print(json.dumps(manifest), flush=True)
    import torch
    from accelerate import Accelerator
    if torch.cuda.device_count() != 1:
        raise ValueError("Requires exactly one caller-authorized visible GPU")
    model = load_model(args.model, args.adapter, trainable=True, gradient_checkpointing=True)
    before = adapter_digest(model)
    accelerator = Accelerator(mixed_precision="bf16")
    model = accelerator.prepare_model(model)
    assert_adapter_matches_checkpoint(model, args.adapter)
    results = {}
    for variant in variants:
        start = time.monotonic()
        with candidate_precision(model, variant) as observation:
            cached = cached_scores(model, tokenizer, rows, 16)
            write_json(args.out / f"{variant}_cached.json", {"per_token_logprobs": cached,
                "drift_from_original_behavior": summarize(rows, cached, tokenizer)})
            teacher, gradient_audit = differentiable_teacher_scores(model, tokenizer, rows, backward=args.audit_backward)
        consistency = summarize(rows, teacher, tokenizer, baseline=cached)
        result = {"same_candidate_cached_vs_teacher": consistency,
            "candidate_cached_vs_original_behavior": summarize(rows, cached, tokenizer),
            "observations": serializable_observation(observation),
            "gradient_audit": gradient_audit,
            "seconds": time.monotonic() - start, "memory": memory_snapshot(),
            "passes_existing_guard_on_recorded_actions": consistency["max_abs_error"] <= 0.1}
        if adapter_digest(model) != before:
            raise RuntimeError("Diagnostic changed adapter tensors")
        write_json(args.out / f"{variant}.json", {"summary": result, "teacher_logprobs": teacher})
        print(json.dumps({"variant": variant, "same_candidate_max": consistency["max_abs_error"],
            "same_candidate_p99": consistency["p99_abs_error"], "tokens_above_0_1": consistency["tokens_above_frozen_tolerance"],
            "original_policy_drift_max": result["candidate_cached_vs_original_behavior"]["max_abs_error"],
            "seconds": result["seconds"], "observations": result["observations"]}), flush=True)
        results[variant] = result
    assert_adapter_matches_checkpoint(model, args.adapter)
    write_json(args.out / "summary.json", {"status": "complete", "results": results,
        "adapter_before": before, "adapter_after": adapter_digest(model), "optimizer_steps": 0,
        "frozen_tolerance_changed": False, "implementation_amendment_authorized": False})


if __name__ == "__main__":
    main()
