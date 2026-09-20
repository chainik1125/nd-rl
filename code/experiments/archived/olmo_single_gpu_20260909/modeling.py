"""Common numerical model and generation policy for every OLMo sprint arm."""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random

MODEL_ID = "allenai/Olmo-3-1125-32B"
MODEL_REVISION = "c2b61dae89a1ad10e4ad5653d0e46b590902607b"
EXPECTED_NUMERICS_SHA256 = "1f8fb573de162acb84eebe8b82723fb1acd2dc19a474a20d7d16668635a6ea37"


def file_sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def versions():
    result = {}
    for package in ("torch", "transformers", "peft", "accelerate", "trl", "bitsandbytes", "datasets"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def seed_everything(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_tokenizer(model_path):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=False)
    if tokenizer.eos_token_id is None:
        raise ValueError("Model tokenizer must define native EOS")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_model(model_path, adapter_path=None, trainable=False, rank=16,
               gradient_checkpointing=True, adapter_name="default"):
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    numerical_source = Path(__file__).with_name("numerics.py")
    if file_sha(numerical_source) != EXPECTED_NUMERICS_SHA256:
        raise ValueError("Pinned numerical policy source changed")
    from numerics import install_numerical_policy

    model_path = Path(model_path).resolve()
    config = json.loads((model_path / "config.json").read_text())
    download = json.loads((model_path / "download_completed.json").read_text())
    if download.get("repo") != MODEL_ID or download.get("revision") != MODEL_REVISION or download.get("metadata_only"):
        raise ValueError("Complete pinned base download identity is required")
    for item in download["verified"]:
        path = model_path / item["name"]
        if not path.is_file() or path.stat().st_size != item["size"]:
            raise ValueError("Previously verified base file is missing or changed size")
        if path.suffix != ".safetensors" and file_sha(path) != item["sha256"]:
            raise ValueError("Pinned base metadata changed after download")
    if config.get("model_type") != "olmo3" or config.get("num_hidden_layers") != 64:
        raise ValueError("This sprint is pinned to the 64-layer Olmo 3 32B base")
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(str(model_path), local_files_only=True,
        trust_remote_code=False, torch_dtype=torch.bfloat16, quantization_config=quantization,
        device_map={"": 0}, attn_implementation="sdpa", low_cpu_mem_usage=True)
    # This casts unquantized parameters to FP32 for *every* arm, including base ICL.
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    if adapter_path is not None:
        model = load_adapter_exact(model, adapter_path, adapter_name=adapter_name, trainable=trainable)
    elif trainable:
        model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", r=rank,
            lora_alpha=2 * rank, target_modules="all-linear", lora_dropout=0.0, bias="none"),
            adapter_name=adapter_name)
    # Retain FP32 adapter/master weights for small RL updates. The runner also
    # restores this after TRL's constructor, which otherwise casts them to BF16.
    for name, parameter in model.named_parameters():
        if "lora_" in name:
            parameter.data = parameter.data.to(torch.float32)
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    numerical_policy = install_numerical_policy(model)
    if numerical_policy["bnb_linear_count"] != 448:
        raise ValueError("Expected 448 FP32-compute NF4 linears in pinned OLMo32B")
    numerical_policy["source_sha256"] = EXPECTED_NUMERICS_SHA256
    if trainable and gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = True
    model.train(trainable)
    model._sprint_identity = {"model_id": MODEL_ID, "revision": MODEL_REVISION,
        "local_model": str(model_path), "config_sha256": file_sha(model_path / "config.json"),
        "download_manifest_sha256": file_sha(model_path / "download_completed.json"),
        "weights_verified_by": "download_completed.json SHA256 audit; sizes rechecked at load",
        "adapter_path": str(Path(adapter_path).resolve()) if adapter_path else None,
        "adapter_sha256": {name: file_sha(Path(adapter_path) / name)
            for name in ("adapter_config.json", "adapter_model.safetensors")} if adapter_path else None,
        "quantization": "nf4_double_quant_float32_compute", "nonquantized_dtype": "float32",
        "numerical_policy": numerical_policy,
        "adapter_dtype": "float32",
        "adapter_load_audits": getattr(model, "_sprint_adapter_load_audits", []),
        "attention": numerical_policy["attention"], "rank": rank, "versions": versions(),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}
    return model


def assert_adapter_matches_checkpoint(model, adapter_path, adapter_name="default"):
    """Require exact FP32 identity, not only a likelihood tolerance."""
    import torch
    from peft import get_peft_model_state_dict
    from safetensors import safe_open
    checkpoint = Path(adapter_path).resolve() / "adapter_model.safetensors"
    state = get_peft_model_state_dict(model, adapter_name=adapter_name, save_embedding_layers=False)
    count = 0
    with safe_open(checkpoint, framework="pt", device="cpu") as saved:
        if set(state) != set(saved.keys()):
            raise RuntimeError(f"Adapter {adapter_name} checkpoint keys differ from loaded tensors")
        for name in saved.keys():
            source = saved.get_tensor(name)
            destination = state[name].detach().cpu()
            if source.dtype != torch.float32 or destination.dtype != torch.float32:
                raise RuntimeError(f"Adapter {adapter_name} must load FP32 tensors before any cast: {name}")
            if source.shape != destination.shape or not torch.equal(source, destination):
                raise RuntimeError(f"Adapter {adapter_name} differs from exact checkpoint tensor: {name}")
            count += source.numel()
    if count == 0:
        raise RuntimeError("Empty adapter checkpoint")
    return {"adapter_name": adapter_name, "checkpoint": str(checkpoint),
        "sha256": file_sha(checkpoint), "tensors": len(state), "parameters": count,
        "all_tensors_exact": True, "dtype": "float32"}


def load_adapter_exact(model, adapter_path, adapter_name="default", trainable=False):
    """Promote destinations before PEFT copies saved FP32 weights into them."""
    import torch
    from peft import PeftModel
    adapter_path = Path(adapter_path).resolve()
    if isinstance(model, PeftModel):
        # Existing adapters can already be BF16 after a library operation.
        # New adapters are promoted by add_adapter(autocast_adapter_dtype=True)
        # before load_adapter calls load_state_dict.
        for name, parameter in model.named_parameters():
            if "lora_" in name and f".{adapter_name}." in name:
                parameter.data = parameter.data.to(torch.float32)
        model.load_adapter(str(adapter_path), adapter_name=adapter_name, is_trainable=trainable,
                           autocast_adapter_dtype=True)
    else:
        model = PeftModel.from_pretrained(model, str(adapter_path), adapter_name=adapter_name,
                                         is_trainable=trainable, autocast_adapter_dtype=True)
    audit = assert_adapter_matches_checkpoint(model, adapter_path, adapter_name)
    model._sprint_adapter_load_audits = [*getattr(model, "_sprint_adapter_load_audits", []), audit]
    return model


def memory_snapshot():
    import torch
    if not torch.cuda.is_available():
        return {"cuda": False}
    return {"allocated_bytes": torch.cuda.memory_allocated(),
        "reserved_bytes": torch.cuda.memory_reserved(),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved()}


@contextmanager
def inference_policy(model):
    import torch
    was_training = model.training
    model.eval()
    try:
        autocast = torch.autocast("cuda", dtype=torch.bfloat16) if model.get_input_embeddings().weight.device.type == "cuda" else nullcontext()
        with torch.inference_mode(), autocast:
            yield
    finally:
        model.train(was_training)


def sample_completions(model, tokenizer, prompts, max_new_tokens=2048,
                       temperature=1.0, batch_size=1, seed=0,
                       max_seq_len=None, return_logprobs=False):
    """Stop after sampled closing marker or native EOS; never fabricate actions."""
    import torch
    from transformers import GenerationConfig, LogitsProcessorList, StopStringCriteria, StoppingCriteriaList
    if max_new_tokens < 1 or batch_size < 1 or temperature < 0:
        raise ValueError("Invalid sampling parameters")
    if return_logprobs and temperature == 0:
        raise ValueError("Behavior likelihood recording requires stochastic sampling")
    seed_everything(seed)
    tokenizer.padding_side = "left"
    result = []
    device = model.get_input_embeddings().weight.device
    eos = tokenizer.eos_token_id
    context_cap = max_seq_len or model.config.max_position_embeddings
    config = GenerationConfig(do_sample=temperature > 0,
        temperature=temperature or 1.0, top_p=1.0, top_k=0, min_p=None,
        repetition_penalty=1.0, max_new_tokens=max_new_tokens,
        bos_token_id=tokenizer.bos_token_id, eos_token_id=eos,
        pad_token_id=tokenizer.pad_token_id, use_cache=True)
    stopping = StoppingCriteriaList([StopStringCriteria(tokenizer, ["</FINAL>"])])
    for offset in range(0, len(prompts), batch_size):
        batch = prompts[offset:offset + batch_size]
        tokens = tokenizer(batch, padding=True, truncation=False, return_tensors="pt",
                           add_special_tokens=False, return_token_type_ids=False)
        width = tokens["input_ids"].shape[1]
        if width + max_new_tokens > context_cap:
            raise ValueError(f"Prompt {width} + completion {max_new_tokens} exceeds context {context_cap}; no truncation")
        tokens = {key: value.to(device) for key, value in tokens.items()}
        collector = behavior_collector(temperature or 1.0) if return_logprobs else None
        with inference_policy(model):
            outputs = model.generate(**tokens, generation_config=config, logits_to_keep=1,
                stopping_criteria=stopping,
                logits_processor=LogitsProcessorList([collector]) if collector else None)
            sequences = outputs
            likelihoods = collector.finish(sequences[:, -1]) if collector else None
        for index, sequence in enumerate(sequences[:, width:].cpu().tolist()):
            sequence, finish_reason = trim_completion(tokenizer, sequence)
            prompt_ids = tokens["input_ids"][index][tokens["attention_mask"][index].bool()].cpu().tolist()
            row = {"text": decode_completion(tokenizer, sequence),
                "prompt_ids": prompt_ids, "response_ids": sequence,
                "prompt_tokens": len(prompt_ids), "completion_tokens": len(sequence),
                "batch_generation_steps": sequences.shape[1] - width,
                "prompt_padding_positions": width - len(prompt_ids),
                "post_stop_padding_positions": sequences.shape[1] - width - len(sequence),
                "sampling_batch_size": len(batch),
                "finish_reason": finish_reason, "seed": seed,
                "sampling": {"temperature": temperature, "top_p": 1.0, "top_k": 0,
                             "repetition_penalty": 1.0}}
            if likelihoods is not None:
                row["rollout_logprobs"] = likelihoods[index][:len(sequence)]
            result.append(row)
        del outputs, sequences, tokens
    return result


def behavior_collector(temperature):
    """Record sampled logprobs with O(batch*vocabulary) temporary memory.

    HF calls this before temperature warping. At the next decode call the
    previous sampled token is in input_ids; gather it from the saved previous
    distribution. Top-p/top-k/repetition penalties are disabled by our policy.
    """
    import torch
    from transformers import LogitsProcessor

    class Collector(LogitsProcessor):
        def __init__(self):
            self.previous = None
            self.chosen = []

        def collect(self, token_ids):
            self.chosen.append(self.previous.gather(1, token_ids[:, None]).squeeze(1))

        def __call__(self, input_ids, scores):
            if self.previous is not None:
                self.collect(input_ids[:, -1])
            self.previous = (scores.detach().float() / temperature).log_softmax(-1)
            return scores

        def finish(self, final_token_ids):
            self.collect(final_token_ids)
            self.previous = None
            return torch.stack(self.chosen, 1).cpu().tolist()

    return Collector()


def decode_completion(tokenizer, response_ids):
    """Remove only a sampled terminal EOS; preserve every other special token."""
    ids = list(response_ids)
    if ids and ids[-1] == tokenizer.eos_token_id:
        ids = ids[:-1]
    return tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)


def trim_completion(tokenizer, response_ids):
    """Discard generation padding after the earliest actual stopping event.

    A closing marker can end within one token; preserve that complete emitted
    token, including any suffix, for literal scoring and the policy loss.
    """
    ids = list(response_ids)
    eos = tokenizer.eos_token_id
    eos_end = ids.index(eos) + 1 if eos in ids else len(ids) + 1
    candidate = ids[:eos_end]
    if "</FINAL>" in decode_completion(tokenizer, candidate):
        low, high = 1, len(candidate)
        while low < high:
            middle = (low + high) // 2
            if "</FINAL>" in decode_completion(tokenizer, candidate[:middle]):
                high = middle
            else:
                low = middle + 1
        return candidate[:low], "final_marker"
    if eos_end <= len(ids):
        return ids[:eos_end], "eos"
    return ids, "length"


def completion_logprobs(model, prompt_ids, response_ids, temperature=1.0):
    import torch
    import torch.nn.functional as F
    if not prompt_ids or not response_ids or temperature <= 0:
        raise ValueError("Likelihood audit needs nonempty prompt/completion and positive temperature")
    device = model.get_input_embeddings().weight.device
    inputs = torch.tensor([prompt_ids + response_ids], device=device)
    with inference_policy(model):
        logits = model(input_ids=inputs, use_cache=False,
                       logits_to_keep=len(response_ids) + 1).logits[:, :-1].float()
        targets = inputs[:, -len(response_ids):]
        scores = -F.cross_entropy((logits / temperature).transpose(1, 2), targets, reduction="none")
    return scores[0].cpu().tolist()


def adapter_digest(model, adapter_name=None):
    import torch
    result = hashlib.sha256()
    count = 0
    for name, parameter in sorted(model.named_parameters()):
        if "lora_" not in name or (adapter_name and f".{adapter_name}." not in name):
            continue
        result.update(name.encode())
        result.update(parameter.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
        count += parameter.numel()
    return {"sha256": result.hexdigest(), "parameters": count}
