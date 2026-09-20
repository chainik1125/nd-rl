"""Full-FP32 NF4 replay with memory-efficient, explicitly expanded GQA.

Diagnostic only: no optimizer and no changes to frozen sources/checkpoints.
The installed HF SDPA implementation supplies the causal/mask semantics. K/V
are expanded before calling it through a proxy without native-GQA attributes,
because CUDA efficient attention does not support enable_gqa=True.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import time
from types import SimpleNamespace

from diagnose_likelihood import SPRINT, audit_inputs, cached_scores, summarize, write_json
from diagnose_precision import candidate_precision, differentiable_teacher_scores, serializable_observation
from modeling import adapter_digest, assert_adapter_matches_checkpoint, file_sha, load_model, load_tokenizer, memory_snapshot, versions


def fp32_expanded_sdpa(module, query, key, value, attention_mask, **kwargs):
    """Faithful installed SDPA semantics, FP32 inputs, no native GQA dispatch."""
    import torch
    from transformers.integrations.sdpa_attention import repeat_kv, sdpa_attention_forward
    groups = getattr(module, "num_key_value_groups", 1)
    if query.shape[1] != key.shape[1] * groups or key.shape[1] != value.shape[1]:
        raise ValueError("Unexpected grouped-query attention head layout")
    key = repeat_kv(key, groups)
    value = repeat_kv(value, groups)
    # Installed sdpa_attention_forward consults only these two module attributes:
    # num_key_value_groups (intentionally absent) and is_causal (preserved).
    proxy = SimpleNamespace(is_causal=getattr(module, "is_causal", True))
    if attention_mask is not None and attention_mask.is_floating_point():
        attention_mask = attention_mask.float()
    with torch.autocast(query.device.type, enabled=False):
        return sdpa_attention_forward(proxy, query.float(), key.float(), value.float(), attention_mask, **kwargs)


@contextmanager
def full_fp32_efficient(model, backend="efficient"):
    """Reuse the checked FP32 policy, replacing only quadratic math attention."""
    from torch.nn.attention import sdpa_kernel, SDPBackend
    from transformers.models.olmo3.modeling_olmo3 import ALL_ATTENTION_FUNCTIONS
    if backend not in {"efficient", "math"}:
        raise ValueError("Unknown diagnostic backend")
    # 'math' is exposed solely for CPU mechanical regression tests. The GPU
    # entrypoint below always requests efficient and never silently falls back.
    with candidate_precision(model, "all_fp32") as observation:
        previous = ALL_ATTENTION_FUNCTIONS["sdpa"]

        def attention(module, query, key, value, attention_mask, **kwargs):
            observation["attention_input_dtypes"].add(str(query.dtype))
            observation["attention_query_shapes"].add(tuple(query.shape))
            observation["attention_mask_none_calls"] += int(attention_mask is None)
            observation["attention_mask_present_calls"] += int(attention_mask is not None)
            result = fp32_expanded_sdpa(module, query, key, value, attention_mask, **kwargs)
            observation["attention_output_dtypes"].add(str(result[0].dtype))
            return result

        observation.update(variant="all_fp32_efficient" if backend == "efficient" else "all_fp32_expanded_math_fixture",
            sdpa_kernel="EFFICIENT_ATTENTION" if backend == "efficient" else "MATH",
            explicit_kv_repeat=True, native_enable_gqa=False,
            attention_mask_none_calls=0, attention_mask_present_calls=0)
        ALL_ATTENTION_FUNCTIONS["sdpa"] = attention
        try:
            with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION if backend == "efficient" else SDPBackend.MATH):
                yield observation
        finally:
            ALL_ATTENTION_FUNCTIONS["sdpa"] = previous


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=SPRINT / "models/olmo32b")
    parser.add_argument("--adapter", type=Path, default=SPRINT / "runs/dpo/final")
    parser.add_argument("--rollouts", type=Path, default=SPRINT / "runs/grpo/rollouts.jsonl")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    tokenizer = load_tokenizer(args.model)
    rows, run = audit_inputs(args, tokenizer)
    manifest = {"purpose": "training_only_full_fp32_efficient_diagnosis_no_optimization",
        "diagnostic_sha256": file_sha(__file__),
        "precision_helper_sha256": file_sha(SPRINT / "code/diagnose_precision.py"),
        "replay_helper_sha256": file_sha(SPRINT / "code/diagnose_likelihood.py"),
        "frozen_source_sha256": run["source_sha256"], "versions": versions(),
        "adapter_sha256": file_sha(args.adapter / "adapter_model.safetensors"),
        "rollouts_sha256": file_sha(args.rollouts), "frozen_max_abs_logprob_error": 0.1,
        "explicit_kv_repeat": True, "native_enable_gqa": False, "silent_backend_fallback": False}
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
    start = time.monotonic()
    with full_fp32_efficient(model) as observation:
        cached = cached_scores(model, tokenizer, rows, 16)
        write_json(args.out / "cached.json", {"per_token_logprobs": cached,
            "drift_from_original_behavior": summarize(rows, cached, tokenizer)})
        teacher, gradient_audit = differentiable_teacher_scores(model, tokenizer, rows, backward=True)
    consistency = summarize(rows, teacher, tokenizer, baseline=cached)
    if before != adapter_digest(model):
        raise RuntimeError("Diagnostic changed adapter tensors")
    assert_adapter_matches_checkpoint(model, args.adapter)
    result = {"status": "complete", "same_candidate_cached_vs_teacher": consistency,
        "candidate_cached_vs_original_behavior": summarize(rows, cached, tokenizer),
        "observations": serializable_observation(observation), "gradient_audit": gradient_audit,
        "seconds": time.monotonic() - start, "memory": memory_snapshot(),
        "passes_existing_guard_on_recorded_actions": consistency["max_abs_error"] <= 0.1,
        "adapter_before": before, "adapter_after": adapter_digest(model), "optimizer_steps": 0,
        "frozen_tolerance_changed": False, "implementation_amendment_authorized": False}
    write_json(args.out / "teacher.json", {"per_token_logprobs": teacher})
    write_json(args.out / "summary.json", result)
    print(json.dumps({"same_candidate_max": consistency["max_abs_error"],
        "same_candidate_p99": consistency["p99_abs_error"], "seconds": result["seconds"],
        "observations": result["observations"], "gradient_audit": gradient_audit}), flush=True)


if __name__ == "__main__":
    main()
