"""Small from-scratch autoregressive proof model and reversible proof codec.

The codec changes only line-number representation. It never checks formulas,
chooses proof rules, or consults a verifier. Relative references are unary so
every symbol needed beyond the six-line cap already occurs in short proofs.
"""
from __future__ import annotations

import contextlib
import math
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import torch
from torch import nn
from torch.nn import functional as F


class ProofTokenizer:
    VERSION = "relative-unary-v1"
    TOKENS = (
        "<PAD>", "<BOS>", "<UNK>", "THM", "SEQ", "PRF", "QED",
        "P", "Q", "R", "S", "F", "(", ")", "~", "&", "v", ">",
        ",", "|", ":", ";", "PR", "AS", "ANDI", "ANDE1", "ANDE2",
        "IMPE", "IMPI", "ORI1", "ORI2", "ORE", "NEGE", "NEGI", "BOTE", "DN",
        "<LINE>", "<REF>", "<BACK>", "<ENDREF>",
    )

    def __init__(self):
        self.itos = list(self.TOKENS)
        self.stoi = {token: i for i, token in enumerate(self.itos)}
        self.pad_id = self.stoi["<PAD>"]
        self.bos_id = self.stoi["<BOS>"]
        self.eos_id = self.stoi["QED"]
        self.prf_id = self.stoi["PRF"]

    def __len__(self):
        return len(self.itos)

    def state_dict(self):
        return {"version": self.VERSION, "tokens": self.itos}

    @classmethod
    def from_state_dict(cls, state):
        if cls is ProofTokenizer and state.get("version") == AbsoluteProofTokenizer.VERSION:
            return AbsoluteProofTokenizer.from_state_dict(state)
        instance = cls()
        if state != instance.state_dict():
            raise ValueError("Unsupported or inconsistent tokenizer checkpoint")
        return instance

    def encode_prompt(self, prompt: str) -> list[int]:
        words = prompt.split()
        if not words or words[0] != "THM" or words[-1] != "PRF":
            raise ValueError("Prompt must begin THM and end PRF")
        return [self.bos_id] + [self.stoi[word] for word in words]

    def encode_record(self, record: str | dict) -> list[int]:
        text = record if isinstance(record, str) else record["text"]
        words = text.split()
        split = words.index("PRF")
        ids = self.encode_prompt(" ".join(words[:split + 1]))
        line_number = 0
        at_header = True
        for word in words[split + 1:]:
            if re.fullmatch(r"N[1-9][0-9]*", word):
                number = int(word[1:])
                if at_header:
                    # normalize arbitrary valid starting indices to N1
                    line_number = number
                    ids.append(self.stoi["<LINE>"])
                    at_header = False
                else:
                    distance = line_number - number
                    if distance < 1:
                        raise ValueError("References must point to an earlier line")
                    ids.extend([self.stoi["<REF>"]] +
                               [self.stoi["<BACK>"]] * distance +
                               [self.stoi["<ENDREF>"]])
            else:
                ids.append(self.stoi[word])
                if word == ";":
                    at_header = True
        return ids

    def decode_body(self, token_ids: Sequence[int]) -> str:
        words: list[str] = []
        line_number = 0
        distance: int | None = None
        for index in token_ids:
            token = self.itos[int(index)]
            if token == "<LINE>":
                if distance is not None:
                    words.extend(["<REF>"] + ["<BACK>"] * distance)
                    distance = None
                line_number += 1
                words.append(f"N{line_number}")
            elif token == "<REF>" and distance is None:
                distance = 0
            elif token == "<BACK>" and distance is not None:
                distance += 1
            elif token == "<ENDREF>" and distance is not None:
                words.append(f"N{line_number - distance}")
                distance = None
            else:
                if distance is not None:
                    # Preserve malformed output as malformed; never repair it.
                    words.extend(["<REF>"] + ["<BACK>"] * distance)
                    distance = None
                words.append(token)
                if token == "QED":
                    break
        if distance is not None:
            words.extend(["<REF>"] + ["<BACK>"] * distance)
        return " ".join(words)


class AbsoluteProofTokenizer(ProofTokenizer):
    """Literal N1..N64 ablation: later indices get no cap-six positive targets."""
    VERSION = "absolute-line-v1"
    TOKENS = ProofTokenizer.TOKENS[:-4] + tuple(f"N{i}" for i in range(1, 65))

    def encode_record(self, record: str | dict) -> list[int]:
        text = record if isinstance(record, str) else record["text"]
        return [self.bos_id] + [self.stoi[word] for word in text.split()]

    def decode_body(self, token_ids: Sequence[int]) -> str:
        words = []
        for index in token_ids:
            token = self.itos[int(index)]
            words.append(token)
            if token == "QED":
                break
        return " ".join(words)


@dataclass
class ModelConfig:
    vocab_size: int = 40
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 1024
    max_seq_len: int = 2048
    dropout: float = 0.0
    rope_base: float = 10000.0


class CausalAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        if config.d_model % config.n_heads or self.head_dim % 2:
            raise ValueError("Head dimension must be an even integer")
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.out = nn.Linear(config.d_model, config.d_model, bias=False)
        self.dropout = config.dropout

    @staticmethod
    def rope(x, cos, sin):
        first, second = x.chunk(2, dim=-1)
        return x * cos + torch.cat((-second, first), dim=-1) * sin

    def forward(self, x, cos, sin, attention_mask=None, cache=None, cache_start=0):
        batch, length, width = x.shape
        q, k, v = self.qkv(x).reshape(
            batch, length, 3, self.n_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4).unbind(0)
        cos, sin = cos.to(q.dtype), sin.to(q.dtype)
        q = self.rope(q, cos, sin)
        k = self.rope(k, cos, sin)
        if cache is not None:
            end = cache_start + length
            cache[0][:, :, cache_start:end].copy_(k)
            cache[1][:, :, cache_start:end].copy_(v)
            k, v = cache[0][:, :, :end], cache[1][:, :, :end]
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=attention_mask is None and length > 1,
        )
        return self.out(y.transpose(1, 2).contiguous().reshape(batch, length, width))


class DecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = CausalAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff, bias=False),
            nn.GELU(approximate="tanh"),
            nn.Linear(config.d_ff, config.d_model, bias=False),
            nn.Dropout(config.dropout),
        )

    def forward(self, x, cos, sin, attention_mask=None, cache=None, cache_start=0):
        x = x + self.attn(self.ln1(x), cos, sin, attention_mask, cache, cache_start)
        return x + self.mlp(self.ln2(x))


class ProofModel(nn.Module):
    def __init__(self, config: ModelConfig | None = None):
        super().__init__()
        self.config = config or ModelConfig()
        c = self.config
        self.token_embedding = nn.Embedding(c.vocab_size, c.d_model)
        self.blocks = nn.ModuleList([DecoderBlock(c) for _ in range(c.n_layers)])
        self.final_norm = nn.LayerNorm(c.d_model)
        self.lm_head = nn.Linear(c.d_model, c.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        head_dim = c.d_model // c.n_heads
        inverse_freq = 1.0 / (c.rope_base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        angles = torch.outer(torch.arange(c.max_seq_len).float(), inverse_freq)
        angles = torch.cat((angles, angles), dim=-1)
        self.register_buffer("rope_cos", angles.cos(), persistent=False)
        self.register_buffer("rope_sin", angles.sin(), persistent=False)
        self.apply(self._init_weights)
        for block in self.blocks:
            nn.init.normal_(block.attn.out.weight, std=0.02 / math.sqrt(2 * c.n_layers))
            nn.init.normal_(block.mlp[2].weight, std=0.02 / math.sqrt(2 * c.n_layers))

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)

    def forward(self, tokens, positions=None, attention_mask=None,
                cache=None, cache_start=0):
        if positions is None:
            positions = torch.arange(cache_start, cache_start + tokens.shape[1], device=tokens.device)
        x = self.token_embedding(tokens)
        cos = self.rope_cos[positions].to(x.dtype)
        sin = self.rope_sin[positions].to(x.dtype)
        if positions.ndim == 1:
            cos, sin = cos[None, None], sin[None, None]
        else:
            cos, sin = cos[:, None], sin[:, None]
        for i, block in enumerate(self.blocks):
            x = block(x, cos, sin, attention_mask,
                      None if cache is None else cache[i], cache_start)
        return self.lm_head(self.final_norm(x))

    def allocate_cache(self, batch_size, max_length, dtype=None):
        parameter = next(self.parameters())
        shape = (batch_size, self.config.n_heads, max_length,
                 self.config.d_model // self.config.n_heads)
        return [(torch.empty(shape, device=parameter.device, dtype=dtype or parameter.dtype),
                 torch.empty(shape, device=parameter.device, dtype=dtype or parameter.dtype))
                for _ in self.blocks]


def _amp_context(device, enabled=True):
    if device.type == "cuda" and enabled:
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.autocast("cuda", dtype=dtype), dtype
    return contextlib.nullcontext(), torch.float32


@torch.inference_mode()
def generate_bodies(model: ProofModel, tokenizer: ProofTokenizer, prompts: Sequence[str],
                    temperature: float = 0.0, seed: int = 0, max_new_tokens: int = 512,
                    batch_size: int = 128, amp: bool = True) -> list[str]:
    """One unfiltered autoregressive sample per prompt, using a batched KV cache.

    Sorting by prompt length reduces padding. Sampling is reproducible for a
    fixed device, batch size, and seed; all output tokens come from the model.
    """
    if temperature < 0 or batch_size < 1 or max_new_tokens < 1:
        raise ValueError("Invalid decoding configuration")
    model.eval()
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed)
    encoded = [tokenizer.encode_prompt(prompt) for prompt in prompts]
    order = sorted(range(len(prompts)), key=lambda i: len(encoded[i]))
    outputs = [""] * len(prompts)
    amp_context, compute_dtype = _amp_context(device, amp)
    with amp_context:
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            lengths = torch.tensor([len(encoded[i]) for i in indices], device=device)
            max_prompt = max(len(encoded[i]) for i in indices)
            if max_prompt >= model.config.max_seq_len:
                raise ValueError("Prompt exceeds model context window")
            new_limit = min(max_new_tokens, model.config.max_seq_len - max_prompt)
            n = len(indices)
            tokens = torch.full((n, max_prompt), tokenizer.pad_id, dtype=torch.long, device=device)
            for row, i in enumerate(indices):
                tokens[row, -len(encoded[i]):] = torch.tensor(encoded[i], device=device)
            column = torch.arange(max_prompt, device=device)
            valid_prompt = column[None] >= (max_prompt - lengths[:, None])
            positions = (column[None] - max_prompt + lengths[:, None]).clamp_min(0)
            causal = torch.ones((max_prompt, max_prompt), device=device, dtype=torch.bool).tril()
            prefill_mask = valid_prompt[:, None, None, :] & causal[None, None]
            cache = model.allocate_cache(n, max_prompt + new_limit, compute_dtype)
            logits = model(tokens, positions=positions, attention_mask=prefill_mask, cache=cache)[:, -1]
            valid_keys = torch.ones((n, max_prompt + new_limit), device=device, dtype=torch.bool)
            valid_keys[:, :max_prompt] = valid_prompt
            done = torch.zeros(n, device=device, dtype=torch.bool)
            generated = torch.empty((n, new_limit), device=device, dtype=torch.long)
            steps = 0
            for step in range(new_limit):
                if temperature == 0:
                    next_token = logits.argmax(dim=-1)
                else:
                    probabilities = (logits.float() / temperature).softmax(dim=-1)
                    next_token = torch.multinomial(probabilities, 1, generator=generator).squeeze(1)
                next_token = torch.where(done, tokenizer.eos_id, next_token)
                generated[:, step] = next_token
                done |= next_token == tokenizer.eos_id
                steps = step + 1
                if bool(done.all()) or step + 1 == new_limit:
                    break
                offset = max_prompt + step
                logits = model(
                    next_token[:, None], positions=(lengths + step)[:, None],
                    attention_mask=valid_keys[:, None, None, :offset + 1],
                    cache=cache, cache_start=offset,
                )[:, -1]
            for row, token_list in enumerate(generated[:, :steps].cpu().tolist()):
                outputs[indices[row]] = tokenizer.decode_body(token_list)
            del cache
    return outputs


@dataclass
class EncodedRecords:
    tokens: list[torch.Tensor]
    prompt_lengths: list[int]

    def __len__(self):
        return len(self.tokens)


def encode_records(tokenizer: ProofTokenizer, records: Iterable[dict | str],
                   max_seq_len: int | None = None) -> EncodedRecords:
    token_lists, prompt_lengths = [], []
    for record in records:
        ids = tokenizer.encode_record(record)
        if max_seq_len is not None and len(ids) > max_seq_len:
            raise ValueError(f"Training sequence has {len(ids)} tokens; limit is {max_seq_len}")
        token_lists.append(torch.tensor(ids, dtype=torch.long))
        prompt_lengths.append(ids.index(tokenizer.prf_id) + 1)
    if not token_lists:
        raise ValueError("No training records")
    return EncodedRecords(token_lists, prompt_lengths)


def collate_records(dataset: EncodedRecords, indices, pad_id=0, device="cpu"):
    ids = nn.utils.rnn.pad_sequence([dataset.tokens[int(i)] for i in indices],
                                   batch_first=True, padding_value=pad_id).to(device)
    prompt_lengths = torch.tensor([dataset.prompt_lengths[int(i)] for i in indices], device=device)
    inputs, targets = ids[:, :-1], ids[:, 1:].clone()
    is_prompt = torch.arange(targets.shape[1], device=device)[None] < (prompt_lengths[:, None] - 1)
    targets.masked_fill_(is_prompt | (targets == pad_id), -100)
    return inputs, targets


def train_steps(model: ProofModel, tokenizer: ProofTokenizer,
                records: Sequence[dict | str] | EncodedRecords,
                steps: int, batch_size: int = 128, lr: float = 3e-4,
                seed: int = 0, weight_decay: float = 0.01, warmup_steps: int = 100,
                grad_clip: float = 1.0, log_every: int = 50,
                callback: Callable[[dict], None] | None = None,
                optimizer=None, amp: bool = True, cosine_decay: bool = True) -> list[dict]:
    """Prompt-masked supervised or verifier-selected maximum likelihood updates.

    The caller controls which records are admitted. This function performs no
    proof search, relabelling, filtering, verifier calls, or evaluation.
    """
    if steps < 1:
        return []
    dataset = records if isinstance(records, EncodedRecords) else encode_records(
        tokenizer, records, model.config.max_seq_len)
    device = next(model.parameters()).device
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(seed)
    rng = random.Random(seed)
    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay,
                                      betas=(0.9, 0.95), fused=device.type == "cuda")
    _, compute_dtype = _amp_context(device, amp)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and compute_dtype == torch.float16)
    model.train()
    start = time.monotonic()
    history = []
    accumulated_loss = 0.0
    accumulated_steps = 0
    warmup_steps = min(warmup_steps, max(1, steps // 10))
    for step in range(1, steps + 1):
        if step <= warmup_steps:
            factor = step / warmup_steps
        elif cosine_decay:
            progress = (step - warmup_steps) / max(1, steps - warmup_steps)
            factor = 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
        else:
            factor = 1.0
        for group in optimizer.param_groups:
            group["lr"] = lr * factor
        indices = [rng.randrange(len(dataset)) for _ in range(batch_size)]
        inputs, targets = collate_records(dataset, indices, tokenizer.pad_id, device)
        optimizer.zero_grad(set_to_none=True)
        with _amp_context(device, amp)[0]:
            logits = model(inputs)
            loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1))
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradient_norm = nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        accumulated_loss += float(loss.detach())
        accumulated_steps += 1
        if step == 1 or step % log_every == 0 or step == steps:
            metric = {
                "step": step, "loss": accumulated_loss / accumulated_steps,
                "lr": lr * factor, "grad_norm": float(gradient_norm),
                "elapsed_seconds": time.monotonic() - start,
                "batch_size": batch_size, "sequence_length": inputs.shape[1],
            }
            history.append(metric)
            if callback is not None:
                callback(metric)
            accumulated_loss = 0.0
            accumulated_steps = 0
    return history


def save_checkpoint(path, model: ProofModel, tokenizer: ProofTokenizer, **metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "config": asdict(model.config),
        "tokenizer": tokenizer.state_dict(),
        "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "metadata": metadata,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path, device="auto"):
    if str(device) == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format")
    tokenizer = ProofTokenizer.from_state_dict(checkpoint["tokenizer"])
    model = ProofModel(ModelConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    return model, tokenizer, checkpoint.get("metadata", {})
