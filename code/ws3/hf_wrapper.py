"""Hugging Face wrapper around the pinned fork model, so TRL can drive it.

Plan T2. The fixed checkpoint is a raw state dict for `fork_pin/model.py::GPT` (RoPE with
interleaved pair rotation, LayerNorm with biases). No stock HF class matches it, so this module
wraps the fork's `GPT` unchanged inside a `PreTrainedModel` and exposes the fork's `Tokenizer`
as a `PreTrainedTokenizerFast`.

    import ndpath
    from hf_wrapper import load_pretrained_wrapper, build_tokenizer
    model, tok = load_pretrained_wrapper()            # fork_pin/ckpt/pretrain.pt
    model.save_pretrained(d); tok.save_pretrained(d)  # TRL reloads the reference model from d

Gate T2 (tests/test_ws3_hf_wrapper.py): logits equal the fork model's to 1e-4, with and without
left padding, and tokenizer round-trips are exact.

Generation uses the fork model's own KV cache through HF `generate`: transformers lets a model
manage its own cache when `_supports_default_dynamic_cache()` is False, in which case the cache
object the forward returns in `past_key_values` is handed back on the next step and `input_ids`
is sliced to the newest token. `NDCache` wraps the fork's per-layer cache dicts (grown on demand)
and gives `generate` the `get_seq_length()` it asks for. On a laptop CPU this is about 100x
faster than recomputing the prefix (`use_cache=False`), which still works and is what TRL uses
for the training forward.
"""

from __future__ import annotations

import hashlib
import json
import os

import ndpath
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer as HFTokenizer
from tokenizers import decoders, models, pre_tokenizers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    GenerationMixin,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizerFast,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

# isort: split
from eval.reward import decode_ids  # noqa: F401  (re-exported; the reward module owns it)
from model import GPT  # fork_pin
from tokenizer import Tokenizer  # fork_pin

MODEL_TYPE = "nd_proof_gpt"


class NDProofConfig(PretrainedConfig):
    model_type = MODEL_TYPE
    # TRL's create_model_from_path and AutoModel.from_pretrained read `config.auto_map` when the
    # architecture is not a transformers built-in. An empty map means "no remote code", which
    # sends them to AutoModelForCausalLM, where this class is registered below.
    auto_map = {}

    def __init__(
        self,
        vocab_size=99,
        n_layer=4,
        d_model=256,
        n_head=8,
        max_len=1024,
        tok_mode="abs",
        pad_token_id=0,
        eos_token_id=5,
        bos_token_id=None,
        tie_word_embeddings=False,
        use_cache=True,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.d_model = d_model
        self.n_head = n_head
        self.max_len = max_len
        self.tok_mode = tok_mode
        # Names TRL and transformers look for.
        self.hidden_size = d_model
        self.num_hidden_layers = n_layer
        self.num_attention_heads = n_head
        self.max_position_embeddings = max_len
        kwargs.setdefault("is_decoder", True)
        super().__init__(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            bos_token_id=bos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            use_cache=use_cache,
            **kwargs,
        )
        self.use_cache = bool(use_cache)  # transformers 5 does not keep it as a plain attribute


class NDCache:
    """KV cache in the fork GPT's format: one dict per layer, filled by `Block.attn`.

    The fork allocates `k`/`v` of length `max` on first use and writes `[n0:n1]` slices; this
    wrapper pre-sizes them for the prompt plus `reserve` tokens and re-allocates (doubling) if a
    generation runs longer. `get_seq_length()` is what transformers' `generate` asks a cache for.
    """

    is_compileable = False
    is_croppable = False

    def __init__(self, n_layer: int, init_len: int):
        self.layers = [{"max": int(init_len)} for _ in range(n_layer)]

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return int(self.layers[layer_idx].get("n", 0))

    def ensure_room(self, extra: int):
        for c in self.layers:
            if "k" not in c:
                c["max"] = max(c["max"], extra)
                continue
            n, cap = c["n"], c["k"].shape[2]
            if n + extra > cap:
                new_cap = max(2 * cap, n + extra)
                for key in ("k", "v"):
                    grown = c[key].new_empty(c[key].shape[0], c[key].shape[1], new_cap, c[key].shape[3])
                    grown[:, :, :n] = c[key][:, :, :n]
                    c[key] = grown


class NDProofForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = NDProofConfig
    base_model_prefix = "gpt"
    main_input_name = "input_ids"
    supports_gradient_checkpointing = False
    _supports_cache_class = False
    _tied_weights_keys = []

    def __init__(self, config: NDProofConfig):
        super().__init__(config)
        self.gpt = GPT(
            config.vocab_size, config.n_layer, config.d_model, config.n_head, config.max_len
        )
        self.post_init()

    # transformers calls this on every module at init; the fork's GPT already initialised itself.
    def _init_weights(self, module):
        return

    cache_reserve = 256  # tokens of KV room allocated beyond the prompt at prefill

    # The model manages its own cache (NDCache); generate() must not build a DynamicCache.
    @classmethod
    def _supports_default_dynamic_cache(cls):
        return False

    def get_input_embeddings(self):
        return self.gpt.emb

    def set_input_embeddings(self, value):
        self.gpt.emb = value

    def get_output_embeddings(self):
        return self.gpt.head

    def set_output_embeddings(self, new):
        self.gpt.head = new

    @staticmethod
    def _positions_and_mask(input_ids, attention_mask, position_ids):
        """Left-padded batches need explicit positions and a 4D bool mask; unpadded ones do not."""
        if attention_mask is None or bool(attention_mask.all()):
            return position_ids, None
        am = attention_mask.to(torch.bool)
        if position_ids is None:
            position_ids = (am.long().cumsum(-1) - 1).clamp(min=0)
        T = input_ids.shape[1]
        causal = torch.ones(T, T, dtype=torch.bool, device=input_ids.device).tril()
        eye = torch.eye(T, dtype=torch.bool, device=input_ids.device)
        # `| eye` keeps fully-masked pad query rows finite; real rows already attend to themselves.
        mask = (causal[None, None] & am[:, None, None, :]) | eye[None, None]
        return position_ids, mask

    @staticmethod
    def _positions_and_mask_cached(input_ids, attention_mask, position_ids, n_past):
        """Positions and (B,1,T,S) mask for T new tokens appended after n_past cached ones.

        `attention_mask`, if given, covers the full S = n_past + T tokens (HF extends it by one
        each step). Every query row may attend to itself so left-pad rows stay finite.
        """
        B, T = input_ids.shape
        S = n_past + T
        dev = input_ids.device
        if position_ids is None:
            if attention_mask is None:
                position_ids = (torch.arange(T, device=dev) + n_past)[None].expand(B, T)
            else:
                position_ids = (attention_mask.long().cumsum(-1) - 1).clamp(min=0)[:, -T:]
        q_pos = torch.arange(T, device=dev) + n_past
        causal = torch.arange(S, device=dev)[None, :] <= q_pos[:, None]  # (T, S)
        mask = causal[None, None].expand(B, 1, T, S)
        if attention_mask is not None:
            mask = mask & attention_mask.to(torch.bool)[:, None, None, :]
        self_idx = torch.zeros(T, S, dtype=torch.bool, device=dev)
        self_idx[torch.arange(T, device=dev), q_pos] = True
        return position_ids, mask | self_idx[None, None]

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        labels=None,
        logits_to_keep=0,
        past_key_values=None,
        use_cache=None,
        inputs_embeds=None,
        return_dict=None,
        **kwargs,
    ):
        if inputs_embeds is not None:
            raise NotImplementedError("NDProofForCausalLM takes input_ids only")
        use_cache = self.config.use_cache if use_cache is None else bool(use_cache)
        cache = past_key_values
        if cache is not None and not isinstance(cache, NDCache):
            raise TypeError(f"past_key_values must be an NDCache, got {type(cache).__name__}")
        if cache is None and use_cache:
            cache = NDCache(self.config.n_layer, input_ids.shape[1] + self.cache_reserve)
        if cache is not None:
            n_past = cache.get_seq_length()
            cache.ensure_room(input_ids.shape[1])
            position_ids, mask = self._positions_and_mask_cached(
                input_ids, attention_mask, position_ids, n_past
            )
            logits = self.gpt(input_ids, pos=position_ids, mask=mask, caches=cache.layers)
        else:
            position_ids, mask = self._positions_and_mask(input_ids, attention_mask, position_ids)
            logits = self.gpt(input_ids, pos=position_ids, mask=mask)
        if isinstance(logits_to_keep, int) and logits_to_keep > 0:
            logits = logits[:, -logits_to_keep:]
        elif torch.is_tensor(logits_to_keep):
            logits = logits[:, logits_to_keep]
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1].float()
            shift_labels = labels[:, 1:]
            loss = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
                ignore_index=-100,
            )
        return CausalLMOutputWithPast(
            loss=loss, logits=logits, past_key_values=cache if use_cache else None
        )


AutoConfig.register(MODEL_TYPE, NDProofConfig)
AutoModelForCausalLM.register(NDProofConfig, NDProofForCausalLM)


# ---------------------------------------------------------------------------- tokenizer


def build_tokenizer(tok_mode: str = "abs") -> PreTrainedTokenizerFast:
    """The fork's whitespace tokenizer as a fast HF tokenizer. Ids are identical to the fork's."""
    ft = Tokenizer(tok_mode)
    # The fork's itos spells both the atom R and the reiteration rule R as "R"; its stoi keeps
    # the later id, so one id is dead (never encoded, decodes to "R"). Mirror that exactly: use
    # stoi for the live ids and give the dead id a placeholder so ids line up with the model.
    vocab = dict(ft.stoi)
    for i, t in enumerate(ft.itos):
        if vocab.get(t) != i:
            vocab[f"<dead_{i}_{t}>"] = i
    assert len(vocab) == ft.vocab_size, (len(vocab), ft.vocab_size)
    backend = HFTokenizer(models.WordLevel(vocab=vocab, unk_token=None))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    backend.decoder = decoders.WordPiece(prefix="##", cleanup=False)  # joins tokens with spaces
    tok = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="<pad>",
        eos_token="QED",
        padding_side="left",
        clean_up_tokenization_spaces=False,
        model_max_length=1024,
    )
    tok.tok_mode = tok_mode
    return tok


# ---------------------------------------------------------------------------- loading


def wrap_fork_model(gpt: GPT, tok_mode: str = "abs") -> NDProofForCausalLM:
    """Wrap an existing fork GPT instance (weights shared, not copied)."""
    c = gpt.cfg
    config = NDProofConfig(
        vocab_size=c["vocab"],
        n_layer=c["n_layer"],
        d_model=c["d"],
        n_head=c["n_head"],
        max_len=c["max_len"],
        tok_mode=tok_mode,
    )
    m = NDProofForCausalLM(config)
    m.gpt = gpt
    return m


def load_pretrained_wrapper(path: str | None = None, device: str = "cpu", verify: bool = True):
    """Load fork_pin/ckpt/pretrain.pt into the wrapper. Returns (model, tokenizer)."""
    path = path or ndpath.FORK_CKPT
    with open(os.path.join(ndpath.FORK_PIN, "MANIFEST.json")) as f:
        manifest = json.load(f)
    if verify:
        with open(path, "rb") as f:
            got = hashlib.sha256(f.read()).hexdigest()
        want = manifest["checkpoint"]["sha256"]
        if got != want:
            raise ValueError(f"checkpoint sha256 {got} != MANIFEST {want}")
    ck = torch.load(path, map_location="cpu", weights_only=True)
    tok_mode = ck.get("tok_mode", "abs")
    tok = build_tokenizer(tok_mode)
    config = NDProofConfig(
        vocab_size=len(tok), n_layer=4, d_model=256, n_head=8, max_len=1024, tok_mode=tok_mode
    )
    model = NDProofForCausalLM(config)
    model.gpt.load_state_dict(ck["state"])
    model.eval()
    return model.to(device), tok


def save_base(model: NDProofForCausalLM, tok: PreTrainedTokenizerFast, out_dir: str) -> str:
    """Save model + tokenizer so TRL can reload the reference model from a path.

    Sets `config._name_or_path` to `out_dir`, which is what TRL reads when beta > 0.
    """
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True)
    tok.save_pretrained(out_dir)
    model.config._name_or_path = out_dir
    model.name_or_path = out_dir
    return out_dir
