"""Pinned v4 FP32-compute NF4 policy shared by every experimental arm.

No checkpoint tensors are recast or changed here. Full FP32 computation and
explicitly expanded grouped-query efficient attention reproduce the validated
diagnostic operator. This module deliberately does not import modeling.
"""
from contextlib import contextmanager
from functools import wraps
from types import SimpleNamespace

POLICY_ID = "nf4_fp32_compute_expanded_gqa_efficient_v4"
ATTENTION_NAME = "sprint_fp32_expanded_efficient_v4"


def fp32_efficient_attention(module, query, key, value, attention_mask, **kwargs):
    import torch
    from torch.nn.attention import sdpa_kernel, SDPBackend
    from transformers.integrations.sdpa_attention import repeat_kv, sdpa_attention_forward
    groups = getattr(module, "num_key_value_groups", 1)
    if query.shape[1] != key.shape[1] * groups or key.shape[1] != value.shape[1]:
        raise ValueError("Unexpected grouped-query attention head layout")
    key, value = repeat_kv(key, groups), repeat_kv(value, groups)
    proxy = SimpleNamespace(is_causal=getattr(module, "is_causal", True))
    if attention_mask is not None and attention_mask.is_floating_point():
        attention_mask = attention_mask.float()
    # No native enable_gqa=True and no fallback to quadratic attention. This
    # context also executes during gradient-checkpoint recomputation.
    with torch.autocast(query.device.type, enabled=False), sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        return sdpa_attention_forward(proxy, query.float(), key.float(), value.float(), attention_mask, **kwargs)


def install_numerical_policy(model):
    import torch
    from bitsandbytes.nn import Linear4bit
    from transformers.modeling_utils import AttentionInterface
    from transformers.masking_utils import AttentionMaskInterface, ALL_MASK_ATTENTION_FUNCTIONS
    causal_lm = model.get_base_model() if hasattr(model, "get_base_model") else model
    if hasattr(causal_lm, "_sprint_numerical_policy"):
        if causal_lm._sprint_numerical_policy["policy_id"] != POLICY_ID:
            raise ValueError("Conflicting numerical policy on model")
        return dict(causal_lm._sprint_numerical_policy)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    if any(parameter.is_floating_point() and parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise ValueError("Numerical policy requires already-prepared FP32 nonpacked parameters")
    count = 0
    for module in model.modules():
        if isinstance(module, Linear4bit):
            module.compute_dtype = torch.float32
            module.compute_type_is_set = True
            count += 1
    # A dedicated name leaves the library's normal SDPA handler untouched. Its
    # mask builder is exactly the installed SDPA mask builder used by the probe.
    AttentionInterface.register(ATTENTION_NAME, fp32_efficient_attention)
    AttentionMaskInterface.register(ATTENTION_NAME, ALL_MASK_ATTENTION_FUNCTIONS["sdpa"])
    causal_lm.set_attn_implementation(ATTENTION_NAME)
    if causal_lm.config._attn_implementation != ATTENTION_NAME:
        raise RuntimeError("Failed to install pinned efficient-attention handler")
    original_forward = causal_lm.forward
    device_type = model.get_input_embeddings().weight.device.type

    @wraps(original_forward)
    def full_precision_forward(*args, **kwargs):
        # An outer no-autocast context alone is insufficient: Accelerate wraps
        # PEFT.forward, while PEFT.generate calls this underlying model directly.
        with torch.autocast(device_type, enabled=False):
            return original_forward(*args, **kwargs)

    causal_lm.forward = full_precision_forward
    identity = {"policy_id": POLICY_ID, "packed_weights_unchanged": True,
        "bnb_compute_dtype": "float32", "bnb_linear_count": count,
        "nonquantized_parameters": "float32", "adapter_parameters": "float32",
        "inner_model_autocast": "disabled", "lm_head_compute": "float32",
        "attention": ATTENTION_NAME, "attention_compute_and_output": "float32",
        "attention_kernel": "EFFICIENT_ATTENTION", "attention_mask_builder": "installed_sdpa",
        "explicit_kv_repeat": True, "native_enable_gqa": False,
        "silent_attention_fallback": False, "tf32_allowed": False,
        "bf16_reduced_precision_reduction": False,
        "teacher_positions": "attention_mask_cumsum_minus_one_padding_position_one"}
    causal_lm._sprint_numerical_policy = identity
    return dict(identity)


def generation_position_hook(module, positional, keyword):
    attention = keyword["attention_mask"]
    if attention.ndim != 2 or keyword["input_ids"].shape != attention.shape:
        raise ValueError("Teacher scorer requires matching full-sequence IDs and 2D mask")
    positions = attention.long().cumsum(-1) - 1
    positions.masked_fill_(attention == 0, 1)
    keyword["position_ids"] = positions
    return positional, keyword


@contextmanager
def generation_positions_for_scoring(model):
    hook = model.register_forward_pre_hook(generation_position_hook, with_kwargs=True)
    try:
        yield
    finally:
        hook.remove()
