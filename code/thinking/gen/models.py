"""Model capabilities and prices, as verified on platform.claude.com on 2026-10-05 (README.md lists
the pages).  Prices are standard USD per million tokens; the Batch API halves every category and
stacks with the cache multipliers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

PRICES_VERIFIED = "2026-10-05"
BATCH_DISCOUNT = 0.5
USAGE_KEYS = ("input", "cache_read", "cache_write_5m", "cache_write_1h", "output")


@dataclass(frozen=True)
class ModelSpec:
    id: str
    tag: str
    input: float
    output: float
    cache_write_5m: float
    cache_write_1h: float
    cache_read: float
    cache_min_tokens: int  # shorter prefixes are silently not cached
    effort: bool  # output_config.effort accepted
    thinking: str  # "adaptive" (cannot be disabled on Opus 5.5) or "budget" (budget_tokens)
    max_output: int

    def price(self, key: str, batch: bool = True) -> float:
        p = {
            "input": self.input,
            "output": self.output,
            "cache_read": self.cache_read,
            "cache_write_5m": self.cache_write_5m,
            "cache_write_1h": self.cache_write_1h,
        }[key]
        return p * (BATCH_DISCOUNT if batch else 1.0)


MODELS: dict[str, ModelSpec] = {
    m.id: m
    for m in (
        ModelSpec(
            "claude-opus-5-5", "opus55", 4.0, 20.0, 5.0, 8.0, 0.20, 512, True, "adaptive", 128000
        ),
        ModelSpec(
            "claude-sonnet-5-5",
            "sonnet55",
            2.0,
            10.0,
            2.5,
            4.0,
            0.20,
            512,
            True,
            "adaptive",
            128000,
        ),
        ModelSpec(
            "claude-haiku-4-5-20251001",
            "haiku45",
            1.0,
            5.0,
            1.25,
            2.0,
            0.10,
            4096,
            False,
            "budget",
            64000,
        ),
    )
}
BY_TAG = {m.tag: m for m in MODELS.values()}


def spec(model: str) -> ModelSpec:
    if model in MODELS:
        return MODELS[model]
    if model in BY_TAG:
        return BY_TAG[model]
    raise KeyError(f"unknown model {model!r}; known: {sorted(MODELS)}")


def billed(usage: dict[str, Any], ttl: str = "1h") -> dict[str, int]:
    """API `usage` -> billed token counts by price category."""
    cc = usage.get("cache_creation") or {}
    w5 = int(cc.get("ephemeral_5m_input_tokens", 0) or 0)
    w1 = int(cc.get("ephemeral_1h_input_tokens", 0) or 0)
    total_w = int(usage.get("cache_creation_input_tokens", 0) or 0)
    if w5 + w1 < total_w:  # no per-TTL breakdown: attribute the rest to the configured TTL
        rest = total_w - w5 - w1
        if ttl == "1h":
            w1 += rest
        else:
            w5 += rest
    return {
        "input": int(usage.get("input_tokens", 0) or 0),
        "cache_read": int(usage.get("cache_read_input_tokens", 0) or 0),
        "cache_write_5m": w5,
        "cache_write_1h": w1,
        "output": int(usage.get("output_tokens", 0) or 0),
    }


def cost(tokens: dict[str, int], model: ModelSpec, batch: bool = True) -> float:
    return sum(tokens.get(k, 0) * model.price(k, batch) for k in USAGE_KEYS) / 1e6


def add(a: dict[str, int], b: dict[str, int]) -> dict[str, int]:
    return {k: a.get(k, 0) + b.get(k, 0) for k in USAGE_KEYS}
