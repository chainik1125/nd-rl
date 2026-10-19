"""Message Batches request lines for trace generation (PROPOSAL.md section 3.2), and their cost
estimate.

Each request line is exactly what `POST /v1/messages/batches` takes in `requests`:
`{"custom_id": ..., "params": {...}}`.  Everything the collector needs (proof id, mode, attempt,
reference actions, estimate inputs) goes in a parallel `*.manifest.jsonl`.

Modes: `reference` (every reference state and action, plus the base model's rejected actions per
decision state when a distractor file is given) and `solve` (the theorem and its first state only).
The environment-in-the-loop mode is an interface sketch in `envloop.py`.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from . import dsl
from .models import ModelSpec, spec

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "steps": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {"think": {"type": "string"}, "action": {"type": "string"}},
                "required": ["think", "action"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["steps"],
    "additionalProperties": False,
}
DEFAULT_EXAMPLES = ("k12_16472", "k12_22530")
MODES = {"reference": "r", "solve": "s"}
CUSTOM_ID = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
REPLY = (
    'Reply with JSON {"steps": [{"think": "...", "action": "..."}, ...]}: one entry per step, in '
    'order; think lines separated by "\\n"; "" for an empty think block.'
)


# ------------------------------------------------------------------------------------- prompts
def production_prompt(path: Path = dsl.PROMPT_MD) -> str:
    """The system prompt section of production_prompt.md, with the markdown table escape `\\|`
    written as the literal `|` the model must produce."""
    text = path.read_text()
    marker = "\n## System prompt\n"
    if marker not in text:
        raise ValueError(f"{path}: no '## System prompt' section")
    return text.split(marker, 1)[1].strip().replace("\\|", "|")


def user_reference(
    rec: dict[str, Any],
    steps: list[dsl.RefStep],
    mistakes: dict[int, list[str]] | None = None,
    deviations: str = "allow",
) -> str:
    rule = {
        "allow": "Use the reference action unless it takes a detour; then you may take a shorter "
        "valid route (rule 9) and write the steps for the states your own actions produce.",
        "forbid": "Use exactly the reference action at every step (rule 9 does not apply).",
    }[deviations]
    out = [
        "Mode: reference-guided.",
        f"Theorem: {rec['prompt']}",
        f"The reference proof has {len(steps)} steps. Write one think block and one action for "
        f"every step. {rule}",
    ]
    for i, s in enumerate(steps):
        out += ["", f"step {i}", f"state: {s.state_ascii}", f"reference action: {s.action_ascii}"]
        ms = (mistakes or {}).get(i)
        if ms:
            out.append("likely mistakes (the base model's rejected actions at this state):")
            out += ms
    out += ["", REPLY]
    return "\n".join(out)


def user_solve(rec: dict[str, Any]) -> str:
    env = dsl.dsl().Env(rec["prompt"], canon=True, base=0, assign=False)
    return "\n".join(
        [
            "Mode: solve.",
            f"Theorem: {rec['prompt']}",
            f"Initial state: {dsl.ascii_toks(env.state_tokens())}",
            "Prove the theorem. Write one think block and one action for every step, from this "
            "state until the proof is finished; each later state is the one your actions produce.",
            "",
            REPLY,
        ]
    )


def example_block(rec: dict[str, Any], k: int) -> str:
    steps = dsl.replay_steps(rec["prompt"], [s["action"] for s in rec["steps"]])
    reply = (
        '{"steps": [\n'
        + ",\n".join(json.dumps(s, ensure_ascii=False) for s in rec["steps"])
        + "\n]}"
    )
    user = user_reference(rec, steps)
    return f"### Complete example {k}\n\nUser turn:\n\n{user}\n\nReply:\n\n{reply}"


def system_prompt(examples: Iterable[str] = DEFAULT_EXAMPLES) -> str:
    base = production_prompt()
    ids = list(examples)
    if not ids:
        return base
    pilot = dsl.pilot_records()
    blocks = [example_block(pilot[x], k + 1) for k, x in enumerate(ids)]
    return base + "\n\n" + "\n\n".join(blocks)


def feedback_text(error: str) -> str:
    return (
        f"The checker rejected this trace: {dsl.ascii_text(error)}\n"
        "Lines before that point passed; nothing after it was checked. Write the whole trace "
        "again, corrected, as the same JSON."
    )


# ------------------------------------------------------------------------------------- requests
def params(
    model: ModelSpec,
    system: str,
    messages: list[dict[str, Any]],
    effort: str = "low",
    max_tokens: int = 16000,
    thinking_budget: int = 4096,
    ttl: str = "1h",
) -> dict[str, Any]:
    cc: dict[str, str] = {"type": "ephemeral"} if ttl == "5m" else {"type": "ephemeral", "ttl": ttl}
    oc: dict[str, Any] = {"format": {"type": "json_schema", "schema": SCHEMA}}
    if model.effort:
        oc["effort"] = effort
    p: dict[str, Any] = {
        "model": model.id,
        "max_tokens": min(max_tokens, model.max_output),
        "system": [{"type": "text", "text": system, "cache_control": cc}],
        "messages": messages,
        "output_config": oc,
    }
    if model.thinking == "adaptive":
        p["thinking"] = {"type": "adaptive"}
    elif thinking_budget:
        if not 1024 <= thinking_budget < p["max_tokens"]:
            raise ValueError("thinking budget must be >= 1024 and below max_tokens")
        p["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}
    return p


def custom_id(proof_id: str, tag: str, mode: str, attempt: int) -> str:
    pid = re.sub(r"[^A-Za-z0-9_-]", "_", proof_id)
    cid = f"{pid[: 64 - len(tag) - 5]}-{tag}-{MODES[mode]}{attempt}"
    assert CUSTOM_ID.match(cid), cid
    return cid


def est_tokens(text: str) -> int:
    """Rough token count for this mix of symbols and English: one per whitespace word plus one per
    8 characters (an over-estimate for plain English under the current tokenizer)."""
    return math.ceil(len(text.split()) + len(text) / 8)


def load_distractors(path: Path | None) -> dict[tuple[str, int], list[str]]:
    out: dict[tuple[str, int], list[str]] = {}
    if path is None:
        return out
    with open(path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                ds = [d["action"] if isinstance(d, dict) else d for d in r["distractors"]]
                if ds:
                    out[(r["proof_id"], int(r["step"]))] = ds
    return out


def build_one(
    rec: dict[str, Any],
    model: ModelSpec,
    system: str,
    mode: str,
    distractors: dict[tuple[str, int], list[str]],
    deviations: str,
    assume_distractors: int,
    **kw: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    steps = dsl.reference_steps(rec["prompt"], rec["proof"]) if rec.get("proof") else []
    if mode == "reference":
        ms = {s: d for (pid, s), d in distractors.items() if pid == rec["id"]}
        text = user_reference(rec, steps, ms, deviations)
    elif mode == "solve":
        ms = {}
        text = user_solve(rec)
    else:
        raise ValueError(f"mode must be one of {sorted(MODES)}")
    cid = custom_id(rec["id"], model.tag, mode, 1)
    line = {
        "custom_id": cid,
        "params": params(model, system, [{"role": "user", "content": text}], **kw),
    }
    acts = [s.action_ascii for s in steps]
    a_tok = sum(est_tokens(a) for a in acts)
    n_dec = sum(s.kind == "decision" for s in steps)
    extra = 0
    if mode == "reference" and not distractors and assume_distractors:
        extra = n_dec * assume_distractors * math.ceil(a_tok / max(1, len(acts)))
    meta = {
        "custom_id": cid,
        "proof_id": rec["id"],
        "source": rec.get("source", ""),
        "prompt": rec["prompt"],
        "model": model.id,
        "tag": model.tag,
        "mode": mode,
        "attempt": 1,
        "parent": None,
        "retry_kind": None,
        "ref_actions": acts,
        "n_steps": len(steps),
        "n_decision": n_dec,
        "n_distractor_states": len(ms),
        "est_output": 3 * a_tok + 10 * len(steps) + 10,
        "est_extra_input": extra,
    }
    return line, meta


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def manifest_path(requests_path: Path) -> Path:
    return requests_path.with_name(requests_path.name.replace(".jsonl", ".manifest.jsonl"))


def build(
    selection: Path,
    out_dir: Path,
    models: list[str],
    mode: str = "reference",
    distractors: Path | None = None,
    examples: Iterable[str] = DEFAULT_EXAMPLES,
    deviations: str = "allow",
    limit: int = 0,
    assume_distractors: int = 3,
    **kw: Any,
) -> list[Path]:
    """Writes `<out_dir>/<tag>.<mode letter>1.jsonl` and its manifest per model."""
    with open(selection) as f:
        recs = [json.loads(x) for x in f if x.strip()]
    if limit:
        recs = recs[:limit]
    system = system_prompt(examples)
    ds = load_distractors(distractors)
    paths = []
    for m in models:
        ms = spec(m)
        pairs = [
            build_one(r, ms, system, mode, ds, deviations, assume_distractors, **kw) for r in recs
        ]
        p = out_dir / f"{ms.tag}.{MODES[mode]}1.jsonl"
        write_jsonl(p, (a for a, _ in pairs))
        write_jsonl(manifest_path(p), (b for _, b in pairs))
        paths.append(p)
    return paths


def retry_request(
    prev_line: dict[str, Any],
    prev_meta: dict[str, Any],
    kind: str,
    content: list[dict[str, Any]] | None = None,
    feedback: str | None = None,
    prev_output_tokens: int = 0,
    max_tokens: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """`feedback`: the previous assistant turn passed back unchanged (thinking blocks included, an
    append-only history) plus a user turn with the validator's message.  `fresh`: the same request
    again (truncated, expired, canceled or server-errored attempts)."""
    p = json.loads(json.dumps(prev_line["params"]))
    extra = 0
    if kind == "feedback":
        assert content is not None and feedback is not None
        p["messages"] = p["messages"] + [
            {"role": "assistant", "content": content},
            {"role": "user", "content": feedback},
        ]
        extra = prev_output_tokens
    elif kind != "fresh":
        raise ValueError(kind)
    if max_tokens:
        p["max_tokens"] = min(max_tokens, spec(p["model"]).max_output)
    attempt = int(prev_meta["attempt"]) + 1
    cid = custom_id(prev_meta["proof_id"], prev_meta["tag"], prev_meta["mode"], attempt)
    meta = dict(prev_meta)
    meta.update(
        custom_id=cid,
        attempt=attempt,
        parent=prev_line["custom_id"],
        retry_kind=kind,
        est_extra_input=int(prev_meta.get("est_extra_input", 0)) + extra,
    )
    return {"custom_id": cid, "params": p}, meta


# ------------------------------------------------------------------------------------- estimate
def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(b.get("text", "") for b in content if isinstance(b, dict))


def estimate(
    lines: list[dict[str, Any]],
    metas: list[dict[str, Any]],
    assume_thinking: int = 2000,
    assume_attempts: float = 1.3,
    assume_cache_hit: float = 0.7,
    input_safety: float = 1.25,
) -> dict[str, Any]:
    """Expected and worst-case batch cost of a request file.  Worst case: every request writes the
    cache, input token estimates are `input_safety` too low, and every request spends max_tokens.
    The expected cost of a first-attempt file is multiplied by `assume_attempts` (retries)."""
    by_id = {m["custom_id"]: m for m in metas}
    per: dict[str, dict[str, Any]] = {}
    for ln in lines:
        p = ln["params"]
        m = spec(p["model"])
        meta = by_id.get(ln["custom_id"], {})
        ttl = p["system"][0].get("cache_control", {}).get("ttl", "5m")
        write = m.price("cache_write_1h" if ttl == "1h" else "cache_write_5m")
        s_tok = est_tokens(p["system"][0]["text"])
        u_tok = sum(
            est_tokens(_text_of(x["content"])) for x in p["messages"] if x["role"] == "user"
        )
        u_tok += est_tokens(json.dumps(SCHEMA)) + int(meta.get("est_extra_input", 0))
        cached = s_tok >= m.cache_min_tokens
        th = p.get("thinking", {})
        t_tok = (
            assume_thinking
            if th.get("type") == "adaptive"
            else min(assume_thinking, th.get("budget_tokens", 0))
        )
        o_tok = int(meta.get("est_output", 1000)) + t_tok
        r = assume_attempts if int(meta.get("attempt", 1)) == 1 else 1.0
        s_exp = (
            s_tok * (assume_cache_hit * m.price("cache_read") + (1 - assume_cache_hit) * write)
            if cached
            else s_tok * m.price("input")
        )
        exp = (s_exp + u_tok * m.price("input") + o_tok * m.price("output")) * r / 1e6
        worst = (
            input_safety
            * (s_tok * (write if cached else m.price("input")) + u_tok * m.price("input"))
            + p["max_tokens"] * m.price("output")
        ) / 1e6
        d = per.setdefault(
            m.id,
            {
                "n_requests": 0,
                "system_tokens": s_tok,
                "cached": cached,
                "user_tokens": 0,
                "output_tokens": 0,
                "max_tokens": p["max_tokens"],
                "expected_usd": 0.0,
                "worst_usd": 0.0,
                "attempt_factor": r,
                "ttl": ttl,
            },
        )
        d["n_requests"] += 1
        d["user_tokens"] += u_tok
        d["output_tokens"] += o_tok
        d["expected_usd"] += exp
        d["worst_usd"] += worst
    for mid, d in per.items():
        m = spec(mid)
        n = d["n_requests"]
        u, o, s = d["user_tokens"] / n, d["output_tokens"] / n, d["system_tokens"]
        w = m.price("cache_write_1h" if d["ttl"] == "1h" else "cache_write_5m")
        h, pr, pi, po = assume_cache_hit, m.price("cache_read"), m.price("input"), m.price("output")
        s_part = (
            f"S {s} x ({h} x ${pr:.3f} + {1 - h:.2f} x ${w:.3f})"
            if d["cached"]
            else f"S {s} x ${pi:.3f} (below the {m.cache_min_tokens}-token cache minimum)"
        )
        d["arithmetic"] = (
            f"per request ($/MTok, batch): {s_part} + U {u:.0f} x ${pi:.3f} + "
            f"(O+T) {o:.0f} x ${po:.3f}; x {d['attempt_factor']} attempts x {n} requests "
            f"= ${d['expected_usd']:.2f}.  Worst: {input_safety} x input at the write price + "
            f"{d['max_tokens']} output tokens each = ${d['worst_usd']:.2f}"
        )
    return {
        "by_model": per,
        "n_requests": len(lines),
        "expected_usd": sum(d["expected_usd"] for d in per.values()),
        "worst_usd": sum(d["worst_usd"] for d in per.values()),
        "assumptions": {
            "thinking_tokens": assume_thinking,
            "attempts": assume_attempts,
            "cache_hit": assume_cache_hit,
            "input_safety": input_safety,
            "token_estimate": "whitespace words + chars/8 (est_tokens)",
        },
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path) as f:
        return [json.loads(x) for x in f if x.strip()]


def estimate_file(path: Path, **kw: Any) -> dict[str, Any]:
    mp = manifest_path(path)
    est = estimate(read_jsonl(path), read_jsonl(mp) if mp.exists() else [], **kw)
    est["file"] = str(path)
    return est
