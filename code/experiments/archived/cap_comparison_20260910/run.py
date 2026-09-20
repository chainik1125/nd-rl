"""R1-inspired symbolic scratchpad SFT and group-relative policy gradients.

Only explicit JSONL paths are opened. RL consumes prompts, never reference proofs.
The sampler never masks actions or inserts thought/final/QED boundaries.
Dependencies: nd_model, nd_grpo, nd_metrics, nd_verify, strict_verify on PYTHONPATH.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import random
import re
import time
from collections import Counter
from dataclasses import asdict
from decimal import Decimal, InvalidOperation
from datetime import UTC, datetime
from pathlib import Path

import torch
from nd_data import canonical_theorem, sequent_parts
from nd_grpo import collate_rollouts
from nd_metrics import target_prompt
from nd_model import ModelConfig, ProofModel, ProofTokenizer, save_checkpoint
from strict_verify import safe_verify
from torch.nn import functional as F

if __package__:
    from .scratch_variants import VERSION as DEPENDENCY_PLAN_VERSION
    from .scratch_variants import plan as dependency_plan
else:
    from scratch_variants import VERSION as DEPENDENCY_PLAN_VERSION
    from scratch_variants import plan as dependency_plan


def scratch_version(style):
    return {
        "reverse": "reverse-line-goals-v1",
        "dependency": DEPENDENCY_PLAN_VERSION,
        "draft": "draft-dependency-check-v1",
        "repair": "mixed-valid-invalid-draft-check-v1",
    }[style]


FINAL_NOT_BEFORE = datetime(2026, 9, 10, 1, 25, tzinfo=UTC)


class ScratchTokenizer(ProofTokenizer):
    VERSION = "relative-unary-r1-symbolic-v1"
    TOKENS = ProofTokenizer.TOKENS + ("<THINK>", "</THINK>", "<FINAL>")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def guard_keys(prompt):
    premises, goal = sequent_parts(prompt)

    def digest(value):
        return hashlib.sha256(value.encode()).hexdigest()

    canonical = digest(canonical_theorem(prompt))
    context = (
        digest(canonical_theorem("THM " + " , ".join(premises) + " SEQ F PRF"))
        if premises
        else None
    )
    skeleton = digest(re.sub(r"\b[PQRS]\b", "A", goal)) if not premises else None
    return canonical, context, skeleton


def validate_training_path(data_path):
    parts = Path(data_path).resolve().parts
    if any("sealed" in part.lower() or part.lower() in ("final", "test") for part in parts):
        raise ValueError("Training may not open sealed/final/test paths")
    if any(tag in Path(data_path).name.lower() for tag in ("textbook_dev", "textbook_final")):
        raise ValueError("Evaluation data cannot be training data")


def guard_training(rows, data_path, guard_path, allow_unguarded_smoke=False):
    validate_training_path(data_path)
    if not guard_path:
        if not allow_unguarded_smoke or len(rows) > 16:
            raise ValueError("Training requires --guard-json; unguarded smoke limited to 16 rows")
        return
    fields = (
        "canonical_theorem_sha256",
        "nonempty_premise_context_sha256",
        "closed_formula_skeleton_sha256",
    )
    obj = json.loads(Path(guard_path).read_text())
    if any(not isinstance(obj.get(field), list) for field in fields):
        raise ValueError("Missing guard hash lists")
    for field in fields:
        if not all(
            isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) for value in obj[field]
        ):
            raise ValueError("Malformed guard hash")
    sets = [set(obj[field]) for field in fields]
    for row in rows:
        if any(tag in str(row.get("role", "")).lower() for tag in ("dev", "final", "test", "eval")):
            raise ValueError("Evaluation-role row cannot enter training")
        if any(
            key is not None and key in protected
            for key, protected in zip(guard_keys(target_prompt(row)), sets, strict=True)
        ):
            raise ValueError("Protected theorem/context/skeleton in training data")


def guard_final_release(data_path, checkpoint_sha, release_path):
    protected = any(
        "sealed" in part.lower() or part.lower() in ("final", "test")
        for part in Path(data_path).resolve().parts
    )
    if protected or "textbook_final" in Path(data_path).resolve().name.lower():
        raise ValueError("Final textbook/test evaluation is forbidden in this cap-comparison experiment")


def guard_development_release(data_path, checkpoint_sha, release_path):
    if "textbook_dev" not in Path(data_path).name.lower():
        return
    if not release_path:
        raise ValueError(
            "Textbook development evaluation needs --evaluation-release-json after an RL run"
        )
    release = json.loads(Path(release_path).read_text())
    if checkpoint_sha not in release.get("checkpoints", []):
        raise ValueError("Checkpoint is not in paired development release")
    rl_checkpoint = Path(release["completed_rl_checkpoint"])
    rl_sha = release["completed_rl_checkpoint_sha256"]
    completed = json.loads((rl_checkpoint.parent / "completed.json").read_text())
    if sha256(rl_checkpoint) != rl_sha or completed.get("final_sha256") != rl_sha:
        raise ValueError("Paired RL completion marker or checkpoint hash differs")
    # These checks precede even hashing development bytes. A historical SFT
    # marker or a partial/step-limited RL run cannot authorize this experiment.
    manifest = json.loads((rl_checkpoint.parent / "manifest.json").read_text())
    metadata = torch.load(rl_checkpoint, map_location="cpu", weights_only=True, mmap=True)["metadata"]
    target, actual = completed.get("target_flop_proxy"), completed.get("actual_flop_proxy")
    last = completed.get("last_step_flop_proxy")
    valid_budget = (type(target) is int and target > 0 and type(actual) is int
                    and type(last) is int and last > 0 and actual - last < target <= actual)
    if (completed.get("stage") != "rl" or completed.get("budget_reached") is not True
            or completed.get("stop_reason") != "flop_budget" or not valid_budget
            or completed.get("actual_steps") != completed.get("steps")
            or not isinstance(completed.get("steps"), int) or completed["steps"] < 1
            or metadata.get("stage") != "rl" or metadata.get("step") != completed["steps"]
            or metadata.get("args") != manifest.get("args")
            or manifest["args"].get("command") != "rl"
            or manifest["args"].get("steps") != completed.get("requested_steps")
            or manifest["args"].get("max_flop_proxy") != target
            or metadata.get("data_sha256") != manifest.get("data_sha256")
            or metadata.get("source_checkpoint_sha256") != manifest.get("source_sha256")
            or completed.get("source_checkpoint_sha256") != manifest.get("source_sha256")
            or any(metadata.get("training_compute", {}).get(key) != completed.get(key)
                   for key in ("actual_steps", "actual_flop_proxy", "target_flop_proxy",
                               "last_step_flop_proxy", "compute", "budget_reached", "stop_reason"))):
        raise ValueError("Actual full-budget RL completion/metadata required before development access")
    if sha256(data_path) not in release.get("data_sha256", []):
        raise ValueError("Data is not in paired development release")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def emit(output, kind, **fields):
    row = {"kind": kind, "unix_time": time.time(), **fields}
    with (Path(output) / "events.jsonl").open("a") as stream:
        stream.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)


COMPUTE_PROXY_VERSION = "dense-padded-parameter-attention-v1"
COMPUTE_PROXY_NOTE = (
    "Algorithmic FLOP proxy, not measured hardware FLOPs: forward = "
    "2*unique_parameters*B*Q + 4*layers*d_model*B*Q*K; backward adds twice "
    "the forward proxy. All padded rows and positions, dense causal-attention "
    "pairs, cache prefill/decode, likelihood audits, policy epochs and retention "
    "are charged. Parameter count approximates dense projections/MLPs/output; "
    "embedding and norm parameters slightly overcount that term. Softmax, "
    "normalization, activations, optimizer, verifier, I/O, allocation and "
    "hardware-specific masked/FlashAttention savings are not modeled."
)


def compute_integer(text):
    """Parse exact nonnegative integer budgets, including scientific notation."""
    try:
        value = Decimal(str(text))
    except InvalidOperation as exc:
        raise argparse.ArgumentTypeError("Expected an exact integer compute value") from exc
    if not value.is_finite() or value < 0 or value != value.to_integral_value():
        raise argparse.ArgumentTypeError("Compute values must be finite nonnegative integers")
    return int(value)


def sampling_weights(rows):
    present = ["sampling_weight" in row for row in rows]
    if not any(present):
        return None
    if not all(present):
        raise ValueError("sampling_weight must be present on every row or no rows")
    weights = [row["sampling_weight"] for row in rows]
    if any(type(w) not in (int, float) or not math.isfinite(w) or w <= 0 for w in weights):
        raise ValueError("sampling_weight must be a positive finite number")
    try:
        total = math.fsum(weights)
    except OverflowError as exc:
        raise ValueError("Nonfinite sampling-weight total") from exc
    if not math.isfinite(total) or total <= 0:
        raise ValueError("Nonfinite sampling-weight total")
    return weights


def sample_training_rows(rng, rows, count, weights=None):
    # Preserve the historical RNG stream exactly when weights are absent.
    return ([rng.choice(rows) for _ in range(count)] if weights is None
            else rng.choices(rows, weights=weights, k=count))


class ComputeLedger:
    def __init__(self, model):
        self.parameters = sum(p.numel() for p in model.parameters())
        self.layers, self.width = model.config.n_layers, model.config.d_model
        self.phases = {}
        self.total = 0

    def cost(self, batch, query, keys=None, backward=False):
        keys = query if keys is None else keys
        forward = 2 * self.parameters * batch * query + 4 * self.layers * self.width * batch * query * keys
        return forward * (3 if backward else 1)

    def account(self, phase, batch, query, keys=None, backward=False):
        keys = query if keys is None else keys
        if any(type(n) is not int or n < 1 for n in (batch, query, keys)):
            raise ValueError("Positive actual dense tensor dimensions required")
        block = self.phases.setdefault(phase, dict(forward_calls=0, backward_calls=0,
            forward_padded_token_slots=0, backward_padded_token_slots=0,
            forward_attention_pair_slots=0, backward_attention_pair_slots=0,
            parameter_flop_proxy=0, attention_flop_proxy=0, flop_proxy=0))
        slots, pairs, multiplier = batch * query, batch * query * keys, 3 if backward else 1
        block["forward_calls"] += 1
        block["backward_calls"] += int(backward)
        block["forward_padded_token_slots"] += slots
        block["backward_padded_token_slots"] += slots if backward else 0
        block["forward_attention_pair_slots"] += pairs
        block["backward_attention_pair_slots"] += pairs if backward else 0
        block["parameter_flop_proxy"] += multiplier * 2 * self.parameters * slots
        block["attention_flop_proxy"] += multiplier * 4 * self.layers * self.width * pairs
        amount = self.cost(batch, query, keys, backward)
        block["flop_proxy"] += amount
        self.total += amount

    def snapshot(self):
        return {"version": COMPUTE_PROXY_VERSION, "note": COMPUTE_PROXY_NOTE,
                "parameters": self.parameters, "layers": self.layers, "d_model": self.width,
                "total_flop_proxy": self.total,
                "phases": {name: dict(values) for name, values in self.phases.items()}}


def rl_step_compute_bound(ledger, model, prompt_lengths, args, retention_lengths=()):
    """Conservative bound for one complete iteration, including all padded rows."""
    n = args.prompts_per_step * args.group_size
    generation = []
    for width in set(prompt_lengths):
        limit = min(args.max_new_tokens, model.config.max_seq_len - width)
        if limit < 1:
            raise ValueError("RL prompt exceeds context")
        decode = limit - 1
        generation.append(ledger.cost(1, width) + 2 * ledger.parameters * decode
            + 4 * ledger.layers * ledger.width * (decode * width + decode * (decode + 1) // 2))
    dense_width = min(model.config.max_seq_len, max(prompt_lengths) + args.max_new_tokens) - 1
    total = n * max(generation) + ledger.cost(n, dense_width)
    total += args.policy_epochs * ledger.cost(n, dense_width, backward=getattr(args, "policy_weight", 1) > 0)
    if retention_lengths and args.retention_weight:
        total += args.policy_epochs * ledger.cost(args.retention_batch, max(retention_lengths) - 1, backward=True)
    return total


class TrainingCompute:
    """Stage-local budget; finish the crossing iteration, then stop before another."""
    def __init__(self, model, tok, args, ledger, step_bound):
        self.model, self.tok, self.args, self.ledger = model, tok, args, ledger
        self.budget = getattr(args, "max_flop_proxy", None)
        milestones = list(getattr(args, "flop_milestones", None) or [])
        if self.budget is not None and (type(self.budget) is not int or self.budget <= 0):
            raise ValueError("max_flop_proxy must be a positive integer")
        if any(type(x) is not int or x < 0 for x in milestones) or len(milestones) != len(set(milestones)):
            raise ValueError("Unique nonnegative integer milestones required")
        if self.budget is not None:
            if any(x > self.budget for x in milestones):
                raise ValueError("Milestone exceeds compute budget")
            if self.budget not in milestones:
                milestones.append(self.budget)
        self.pending = sorted(milestones)
        self.step_bound, self.steps, self.records, self.previous_total = step_bound, 0, [], 0
        self.last_step_flops = 0
        self.identity = {key: sha256(path) if path else None for key, path in (
            ("source_checkpoint_sha256", getattr(args, "checkpoint", None)),
            ("data_sha256", getattr(args, "data", None)),
            ("guard_sha256", getattr(args, "guard_json", None)),
            ("retention_data_sha256", getattr(args, "retention_data", None)),
            ("retention_guard_sha256", (getattr(args, "retention_guard_json", None)
                                       or getattr(args, "guard_json", None))
             if getattr(args, "retention_data", None) else None))}
        self.save_milestones()

    @property
    def reached(self):
        return self.budget is not None and self.ledger.total >= self.budget

    def result(self):
        return {"steps": self.steps, "actual_steps": self.steps, "requested_steps": self.args.steps,
            "stop_reason": ("flop_budget" if self.reached else
                            "step_limit" if self.steps >= self.args.steps else "in_progress"),
            "budget_reached": self.reached, "max_flop_proxy": self.budget,
            "target_flop_proxy": self.budget, "total_flop_proxy": self.ledger.total,
            "actual_flop_proxy": self.ledger.total, "last_step_flop_proxy": self.last_step_flops,
            "flop_proxy_overshoot": max(0, self.ledger.total - self.budget) if self.budget is not None else 0,
            "worst_case_one_step_flop_proxy": self.step_bound,
            "overshoot_policy": "Complete crossing training iteration; never start another once budget is reached.",
            "compute": self.ledger.snapshot(), "compute_milestones": list(self.records),
            "unreached_flop_milestones": list(self.pending)}

    def save(self, path, milestone=None):
        save_checkpoint(path, self.model, self.tok, mode=self.args.mode,
            stage=self.args.command, step=self.steps, **self.identity,
            scratch_style=self.args.scratch_style,
            scratch_plan_version=scratch_version(self.args.scratch_style),
            args=vars(self.args), training_compute=self.result(), requested_flop_milestone=milestone)

    def save_milestones(self):
        while self.pending and self.ledger.total >= self.pending[0]:
            milestone = self.pending.pop(0)
            path = Path(self.args.output) / f"flops_{milestone}.pt"
            self.save(path, milestone)
            self.records.append({"requested_flop_proxy": milestone,
                "actual_flop_proxy": self.ledger.total, "step": self.steps,
                "checkpoint": str(path), "checkpoint_sha256": sha256(path)})

    def after_step(self, step):
        amount = self.ledger.total - self.previous_total
        if amount <= 0 or amount > self.step_bound:
            raise AssertionError("Observed iteration cost exceeds declared compute bound")
        self.previous_total, self.steps = self.ledger.total, step
        self.last_step_flops = amount
        self.save_milestones()
        return amount


def load_model(path, device="cpu", max_seq_len=None):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format")
    state = checkpoint["tokenizer"]
    if state == ScratchTokenizer().state_dict():
        tok = ScratchTokenizer()
    elif state == ProofTokenizer().state_dict():
        tok = ProofTokenizer()
    else:
        raise ValueError("Only relative-unary checkpoints are supported")
    config = dict(checkpoint["config"])
    if max_seq_len is not None:
        config["max_seq_len"] = max_seq_len
    model = ProofModel(ModelConfig(**config))
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device).eval(), tok, checkpoint.get("metadata", {})


def expand_vocabulary(model):
    if model.config.vocab_size == len(ScratchTokenizer()):
        return model
    if model.config.vocab_size != len(ProofTokenizer()):
        raise ValueError("Unexpected original vocabulary size")
    config = asdict(model.config)
    config["vocab_size"] = len(ScratchTokenizer())
    expanded = ProofModel(ModelConfig(**config)).to(next(model.parameters()).device)
    state = model.state_dict()
    new_state = expanded.state_dict()
    for name, values in state.items():
        if name in ("token_embedding.weight", "lm_head.weight"):
            new_state[name][: len(ProofTokenizer())].copy_(values)
        else:
            new_state[name].copy_(values)
    expanded.load_state_dict(new_state, strict=True)
    return expanded.eval()


def split_record(row):
    prompt = target_prompt(row)
    body = row.get("body", row.get("proof"))
    if body is None and "text" in row:
        words = row["text"].split()
        body = " ".join(words[words.index("PRF") + 1 :])
    if not isinstance(body, str):
        raise ValueError("SFT needs text, body, or proof in every row")
    return prompt, " ".join(body.split())


def backward_plan(body):
    """Reverse line order, retaining each rule and local goal, omitting citations.

    This cold-start trace uses only the supplied short proof. It supplies no
    search oracle, long proof, textbook content, or verifier during generation.
    """
    steps = []
    for line in body.split(";"):
        words = line.split()
        if not words or words == ["QED"]:
            continue
        colon = words.index(":")
        formula = [word for word in words[1:colon] if word != "|"]
        rule = words[colon + 1]
        steps.append([rule, *formula, ";"])
    return [word for step in reversed(steps) for word in step]


def corrupted_draft(prompt, body):
    """Create one invalid local edit of an already verified short proof.

    Source verification belongs to encode_training. This verifier use is only
    offline cold-start labeling; neither sampling nor inference calls it.
    Every candidate preserves the line count and relative-codec encodability.
    """
    segments = [segment.split() for segment in body.rsplit("QED", 1)[0].split(";")]
    segments = [segment for segment in segments if segment]
    candidates = []
    same_arity = (("PR", "AS"), ("R", "ANDE1", "ANDE2", "ORI1", "ORI2", "BOTE", "DN"),
                  ("ANDI", "IMPE", "IMPI", "NEGE", "NEGI"))
    for index, line in enumerate(segments):
        colon = line.index(":")
        rule = line[colon + 1]
        for family in same_arity:
            if rule in family:
                for alternative in family:
                    if alternative != rule and alternative in ProofTokenizer.TOKENS:
                        candidates.append((index, colon + 1, alternative))
        # Change a reference only to another real preceding source line.
        for position in range(colon + 2, len(line)):
            for previous in segments[:index]:
                if line[position] != previous[0]:
                    candidates.append((index, position, previous[0]))
    rng = random.Random(int(hashlib.sha256((prompt + body).encode()).hexdigest()[:16], 16))
    rng.shuffle(candidates)
    # A changed final-goal atom is a guaranteed invalid fallback when there is
    # no invalid same-arity edit (e.g. a one-premise identity proof).
    last = segments[-1]
    for position, word in enumerate(last[1:last.index(":")], start=1):
        if word in ("P", "Q", "R", "S", "F"):
            candidates.append((len(segments) - 1, position, "Q" if word == "P" else "P"))
    for index, position, alternative in candidates:
        changed = [list(segment) for segment in segments]
        changed[index][position] = alternative
        draft = " ".join(" ".join(segment) + " ;" for segment in changed) + " QED"
        if not safe_verify(prompt, draft)[0]:
            return draft
    raise ValueError("Could not construct an invalid short draft")


def scratch_ids(prompt, body, tok, style, max_proof_lines):
    """Cold-start reasoning in the existing symbolic vocabulary.

    draft/repair use PRF <draft certificate> SEQ <goal/dependency check>.
    The final certificate is always emitted separately after <FINAL>.
    Half of repair examples keep a correct draft; no marker reveals validity.
    """
    if style == "reverse":
        return [tok.stoi[word] for word in backward_plan(body)], "plan"
    plan_ids = [tok.stoi[word] for word in dependency_plan(body, max_proof_lines=max_proof_lines)]
    if style == "dependency":
        return plan_ids, "plan"
    if style not in ("draft", "repair"):
        raise ValueError("Unknown scratch style")
    invalid = style == "repair" and int(hashlib.sha256((prompt + body).encode()).hexdigest()[-1], 16) % 2 == 1
    draft = corrupted_draft(prompt, body) if invalid else body
    prompt_len = len(tok.encode_prompt(prompt))
    draft_ids = tok.encode_record(prompt + " " + draft)[prompt_len:]
    return [tok.prf_id, *draft_ids, tok.stoi["SEQ"], *plan_ids], "invalid_draft" if invalid else "valid_draft"


def encode_training(row, tok, mode, max_seq_len, max_proof_lines=6, scratch_style="reverse"):
    scratch_version(scratch_style)  # Reject an unrecognized cold-start configuration.
    prompt, body = split_record(row)
    ok, reason, lines = safe_verify(prompt, body)
    if not ok or body.split().count("QED") != 1 or body.split()[-1] != "QED":
        raise ValueError("Invalid SFT proof: " + reason)
    if lines > max_proof_lines:
        raise ValueError(f"SFT proof has {lines} lines, above cap {max_proof_lines}")
    prompt_ids = tok.encode_prompt(prompt)
    final_ids = tok.encode_record(prompt + " " + body)[len(prompt_ids) :]
    if mode == "cot":
        if not isinstance(tok, ScratchTokenizer):
            raise ValueError("CoT requires expanded tokenizer; run prepare")
        plan_ids, draft_kind = scratch_ids(prompt, body, tok, scratch_style, max_proof_lines)
        response = [
            tok.stoi["<THINK>"],
            *plan_ids,
            tok.stoi["</THINK>"],
            tok.stoi["<FINAL>"],
            *final_ids,
        ]
    else:
        response = final_ids
        draft_kind = "direct"
    if len(prompt_ids) + len(response) > max_seq_len:
        raise ValueError("SFT record exceeds context; refusing truncation")
    return {"prompt_ids": prompt_ids, "response_ids": response, "cold_start_kind": draft_kind}


def judge_response(prompt, response, tok, mode):
    """Verify the exact final suffix against the immutable original prompt."""
    ids = list(response)
    scratch_count = 0
    if mode == "cot":
        start, end, final = (tok.stoi[t] for t in ("<THINK>", "</THINK>", "<FINAL>"))
        if not ids or ids[0] != start or any(ids.count(t) != 1 for t in (start, end, final)):
            return {
                "reward": 0,
                "reason": "invalid thought boundaries",
                "found_len": None,
                "scratch_tokens": 0,
                "body": "",
            }
        end_at, final_at = ids.index(end), ids.index(final)
        if end_at < 1 or final_at != end_at + 1:
            return {
                "reward": 0,
                "reason": "invalid final boundary",
                "found_len": None,
                "scratch_tokens": 0,
                "body": "",
            }
        scratch_count = end_at - 1
        ids = ids[final_at + 1 :]
    if not ids or ids[-1] != tok.eos_id or ids.count(tok.eos_id) != 1:
        return {
            "reward": 0,
            "reason": "missing or nonterminal final QED",
            "found_len": None,
            "scratch_tokens": scratch_count,
            "body": "",
        }
    # The base decoder is reversible through QED; never decode scratch as proof.
    body = tok.decode_body(ids)
    ok, reason, lines = safe_verify(prompt, body)
    return {
        "reward": int(ok),
        "reason": reason,
        "found_len": lines if ok else None,
        "scratch_tokens": scratch_count,
        "body": body,
    }


def amp(model):
    if (
        next(model.parameters()).device.type == "cuda"
        and getattr(model, "r1_precision", "bf16") == "bf16"
    ):
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("CUDA training currently requires BF16 support")
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


@torch.inference_mode()
def sample(
    model, tok, prompts, *, mode, temperature=0.8, seed=0, max_new_tokens=768, batch_size=32,
    compute=None,
):
    if temperature < 0 or batch_size < 1 or max_new_tokens < 1:
        raise ValueError("Invalid sampler configuration")
    model.eval()
    device = next(model.parameters()).device
    dtype = (
        torch.bfloat16
        if device.type == "cuda" and getattr(model, "r1_precision", "bf16") == "bf16"
        else torch.float32
    )
    generator = torch.Generator(device=device).manual_seed(seed)
    encoded = [tok.encode_prompt(prompt) for prompt in prompts]
    order = sorted(range(len(prompts)), key=lambda i: len(encoded[i]))
    results = [None] * len(prompts)
    with amp(model):
        for offset in range(0, len(order), batch_size):
            indices = order[offset : offset + batch_size]
            lengths = torch.tensor([len(encoded[i]) for i in indices], device=device)
            width = max(len(encoded[i]) for i in indices)
            limit = min(max_new_tokens, model.config.max_seq_len - width)
            if limit < 1:
                raise ValueError("Prompt exceeds context")
            n = len(indices)
            tokens = torch.full((n, width), tok.pad_id, dtype=torch.long, device=device)
            for row, i in enumerate(indices):
                tokens[row, -len(encoded[i]) :] = torch.tensor(encoded[i], device=device)
            columns = torch.arange(width, device=device)
            valid = columns[None] >= width - lengths[:, None]
            positions = (columns[None] - width + lengths[:, None]).clamp_min(0)
            causal = torch.ones((width, width), device=device, dtype=torch.bool).tril()
            cache = model.allocate_cache(n, width + limit, dtype)
            if compute is not None:
                compute.account("generation_prefill", n, width)
            logits = model(
                tokens,
                positions=positions,
                attention_mask=valid[:, None, None] & causal[None, None],
                cache=cache,
            )[:, -1]
            valid_keys = torch.ones((n, width + limit), device=device, dtype=torch.bool)
            valid_keys[:, :width] = valid
            done = torch.zeros(n, device=device, dtype=torch.bool)
            final_seen = torch.full((n,), mode == "direct", device=device, dtype=torch.bool)
            stop_lengths = torch.full((n,), limit, device=device, dtype=torch.long)
            generated = torch.empty((n, limit), device=device, dtype=torch.long)
            logps = torch.empty((n, limit), device=device, dtype=torch.float32)
            for step in range(limit):
                distribution = (logits.float() / (temperature or 1.0)).log_softmax(-1)
                if temperature:
                    chosen = torch.multinomial(distribution.exp(), 1, generator=generator).squeeze(
                        1
                    )
                    chosen_logps = distribution.gather(1, chosen[:, None]).squeeze(1)
                else:
                    chosen = logits.argmax(-1)
                    # Greedy policy is a point mass; it is never used for RL.
                    chosen_logps = torch.zeros(n, device=device)
                generated[:, step] = chosen
                logps[:, step] = chosen_logps
                if mode == "cot":
                    final_seen |= chosen == tok.stoi["<FINAL>"]
                newly_done = (~done) & final_seen & (chosen == tok.eos_id)
                stop_lengths[newly_done] = step + 1
                done |= newly_done
                if bool(done.all()) or step + 1 == limit:
                    break
                cache_start = width + step
                if compute is not None:
                    # Finished rows remain in the actual dense cached call.
                    compute.account("generation_decode", n, 1, cache_start + 1)
                logits = model(
                    chosen[:, None],
                    positions=(lengths + step)[:, None],
                    attention_mask=valid_keys[:, None, None, : cache_start + 1],
                    cache=cache,
                    cache_start=cache_start,
                )[:, -1]
            all_ids = generated[:, : step + 1].cpu().tolist()
            all_logps = logps[:, : step + 1].cpu().tolist()
            for row, i in enumerate(indices):
                length = min(int(stop_lengths[row]), step + 1)
                results[i] = {
                    "prompt_ids": encoded[i],
                    "response_ids": all_ids[row][:length],
                    "rollout_logprobs": all_logps[row][:length],
                    "effective_max_new_tokens": limit,
                    "finish_reason": "terminal_qed" if bool(done[row]) else "token_limit",
                }
            del cache
    return results


def ce_backward(model, tok, rows, microbatch, coefficient=1.0, *, compute=None, phase="sft"):
    denominator = sum(len(row["response_ids"]) for row in rows)
    loss_value = 0.0
    for offset in range(0, len(rows), microbatch):
        batch = collate_rollouts(
            rows[offset : offset + microbatch], tok.pad_id, next(model.parameters()).device
        )
        if compute is not None:
            compute.account(phase, *batch["inputs"].shape, backward=True)
        with amp(model):
            logits = model(batch["inputs"]).float()
            losses = F.cross_entropy(
                logits.flatten(0, 1), batch["targets"].flatten(), reduction="none"
            ).view_as(batch["targets"])
            loss = (losses * batch["mask"]).sum() / denominator
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Nonfinite SFT loss")
        (coefficient * loss).backward()
        loss_value += float(loss.detach())
    return loss_value


def train_sft(model, tok, rows, args):
    weights = sampling_weights(rows)
    data = [
        encode_training(
            row, tok, args.mode, model.config.max_seq_len, args.max_proof_lines, args.scratch_style
        )
        for row in rows
    ]
    if not data:
        raise ValueError("Empty SFT data")
    emit(args.output, "sft_data", examples=len(data),
         cold_start_kinds=dict(Counter(row["cold_start_kind"] for row in data)),
         response_tokens=sum(len(row["response_ids"]) for row in data),
         max_response_tokens=max(len(row["response_ids"]) for row in data),
         weighted_sampling=weights is not None,
         sampling_weight_total=math.fsum(weights) if weights is not None else None)
    compute = ComputeLedger(model)
    max_width = max(len(row["prompt_ids"]) + len(row["response_ids"]) - 1 for row in data)
    tracker = TrainingCompute(model, tok, args, compute,
                              compute.cost(args.batch_size, max_width, backward=True))
    rng = random.Random(args.seed)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95)
    )
    model.train()
    started = time.monotonic()
    actual_response_tokens = 0
    for step in range(1, args.steps + 1):
        if tracker.reached:
            break
        batch = sample_training_rows(rng, data, args.batch_size, weights)
        optimizer.zero_grad(set_to_none=True)
        loss = ce_backward(model, tok, batch, args.microbatch, compute=compute)
        actual_response_tokens += sum(len(row["response_ids"]) for row in batch)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not bool(torch.isfinite(norm)):
            raise FloatingPointError("Nonfinite SFT gradient")
        optimizer.step()
        step_flops = tracker.after_step(step)
        if step == 1 or step % args.log_every == 0 or step == args.steps or tracker.reached:
            emit(
                args.output,
                "sft",
                step=step,
                loss=loss,
                grad_norm=float(norm),
                actual_response_tokens=actual_response_tokens,
                elapsed_seconds=time.monotonic() - started,
                step_flop_proxy=step_flops, total_flop_proxy=compute.total,
                budget_reached=tracker.reached,
            )
        if args.save_every and step % args.save_every == 0:
            tracker.save(Path(args.output) / f"step_{step:06d}.pt")
    result = tracker.result()
    result["actual_response_tokens"] = actual_response_tokens
    return result


def group_advantages(rewards, baseline="loo"):
    rewards = torch.as_tensor(rewards, dtype=torch.float32)
    if rewards.ndim != 2 or rewards.shape[1] < 2:
        raise ValueError("Require group_size >= 2")
    if not bool(((rewards == 0) | (rewards == 1)).all()):
        raise ValueError("Rewards must be binary")
    if baseline == "loo":
        return rewards - (rewards.sum(1, keepdim=True) - rewards) / (rewards.shape[1] - 1)
    if baseline == "standardized":
        centered = rewards - rewards.mean(1, keepdim=True)
        return centered / rewards.std(1, unbiased=False, keepdim=True).clamp_min(1e-8)
    raise ValueError("Unknown advantage baseline")


class PromptSchedule:
    """Training-prompt-only EMA frontier; no filtering of sampled groups.

    The exploration floor retains hard and easy prompts. Every sampled group,
    including all-success and all-failure groups, enters the policy update.
    """

    def __init__(self, prompts, mode="uniform", decay=0.8, floor=0.25, weights=None):
        if mode not in ("uniform", "frontier") or not 0 <= decay < 1 or floor <= 0:
            raise ValueError("Invalid prompt scheduling configuration")
        self.prompts = list(prompts)
        if not self.prompts:
            raise ValueError("Empty prompt schedule")
        self.mode, self.decay, self.floor = mode, decay, floor
        if weights is not None:
            sampling_weights([{"sampling_weight": w} for w in weights])
            if len(weights) != len(self.prompts):
                raise ValueError("Prompt sampling weights differ from prompt count")
        self.weights = weights
        self.probability = dict.fromkeys(self.prompts, 0.5)
        self.visits = Counter()

    def select(self, rng, count):
        if self.mode == "uniform":
            return sample_training_rows(rng, self.prompts, count, self.weights)
        weights = [self.floor + 4 * self.probability[p] * (1 - self.probability[p])
                   for p in self.prompts]
        if self.weights is not None:
            weights = [a * b for a, b in zip(weights, self.weights, strict=True)]
        return rng.choices(self.prompts, weights=weights, k=count)

    def observe(self, selected, rewards):
        # Combine duplicate selections before the EMA update to avoid ordering effects.
        sums, counts = Counter(), Counter()
        for prompt, group in zip(selected, rewards, strict=True):
            sums[prompt] += sum(float(value) for value in group)
            counts[prompt] += len(group)
        for prompt in sums:
            self.probability[prompt] = (self.decay * self.probability[prompt]
                                       + (1 - self.decay) * sums[prompt] / counts[prompt])
            self.visits[prompt] += counts[prompt]

    def metrics(self, selected):
        return {"prompt_schedule": self.mode, "weighted_prompt_sampling": self.weights is not None,
                "unique_prompts_seen": len(self.visits),
                "selected_ema_pass_probability": sum(self.probability[p] for p in selected) / len(selected)}


def policy_update(model, tok, rows, optimizer, args, retention=(), *, compute=None):
    policy_weight = getattr(args, "policy_weight", 1.0)
    if not math.isfinite(policy_weight) or policy_weight < 0:
        raise ValueError("policy_weight must be finite and nonnegative")
    retention_active = bool(retention) and bool(args.retention_weight)
    model.eval()  # Dropout must not change the behavior policy's likelihoods.
    device = next(model.parameters()).device
    max_error = 0.0
    # Teacher forcing audits numerical cache drift only. The PPO denominator
    # is the actual behavior-policy log probability captured during sampling.
    with torch.no_grad():
        for offset in range(0, len(rows), args.microbatch):
            chunk = rows[offset : offset + args.microbatch]
            batch = collate_rollouts(chunk, tok.pad_id, device)
            if compute is not None:
                compute.account("behavior_teacher_audit", *batch["inputs"].shape)
            with amp(model):
                logps = (model(batch["inputs"]).float() / args.temperature).log_softmax(-1)
            chosen = logps.gather(-1, batch["targets"][..., None]).squeeze(-1)
            for row, values, mask in zip(chunk, chosen, batch["mask"], strict=True):
                teacher_values = values[mask].cpu().tolist()
                if len(teacher_values) != len(row["rollout_logprobs"]):
                    raise ValueError("Behavior likelihood does not cover every action")
                row["old_logprobs"] = list(row["rollout_logprobs"])
                error = max(
                    abs(a - b) for a, b in zip(teacher_values, row["rollout_logprobs"], strict=True)
                )
                max_error = max(max_error, error)
    if max_error > args.max_logprob_error:
        raise RuntimeError(f"Behavior/teacher numerical drift {max_error} exceeds configured bound")
    losses = []
    weighted_losses, total_weighted_losses = [], []
    optimizer_steps = 0
    clipped_tokens, action_tokens = 0, 0
    total_logratio, total_entropy = 0.0, 0.0
    for _ in range(args.policy_epochs):
        optimizer.zero_grad(set_to_none=True)
        loss_value = 0.0
        for offset in range(0, len(rows), args.microbatch):
            chunk = rows[offset : offset + args.microbatch]
            batch = collate_rollouts(chunk, tok.pad_id, device)
            if compute is not None:
                compute.account("policy", *batch["inputs"].shape, backward=policy_weight > 0)
            with torch.set_grad_enabled(policy_weight > 0), amp(model):
                logps = (model(batch["inputs"]).float() / args.temperature).log_softmax(-1)
                chosen = logps.gather(-1, batch["targets"][..., None]).squeeze(-1)
                logratio = chosen - batch["old_logprobs"]
                # Only a numerical overflow guard; PPO clipping remains separate.
                ratio = logratio.clamp(-60, 60).exp()
                advantage = batch["advantages"][:, None]
                surrogate = torch.minimum(
                    ratio * advantage, ratio.clamp(1 - args.clip, 1 + args.clip) * advantage
                )
                # Every sampled token, including scratch/PAD/boundaries, earns
                # its group's advantage. No invalid-token or length filtering.
                denominator = (
                    batch["mask"].sum(-1)
                    if args.loss_normalization == "sequence"
                    else args.max_new_tokens
                )
                sequence_loss = -(surrogate * batch["mask"]).sum(-1) / denominator
                loss = sequence_loss.sum() / len(rows)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite GRPO loss")
            if policy_weight:
                weighted_loss = policy_weight * loss
                if not bool(torch.isfinite(weighted_loss)):
                    raise FloatingPointError("Nonfinite weighted GRPO loss")
                weighted_loss.backward()
            loss_value += float(loss.detach())
            with torch.no_grad():
                mask = batch["mask"]
                clipped_tokens += int((((ratio - 1).abs() > args.clip) & mask).sum())
                action_tokens += int(mask.sum())
                total_logratio += float((-logratio * mask).sum())
                total_entropy += float((-(logps.exp() * logps).sum(-1) * mask).sum())
        retention_loss = None
        if retention_active:
            retention_loss = ce_backward(
                model, tok, retention, args.microbatch, args.retention_weight,
                compute=compute, phase="retention",
            )
        if policy_weight or retention_active:
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not bool(torch.isfinite(norm)):
                raise FloatingPointError("Nonfinite GRPO gradient")
            optimizer.step()
            optimizer_steps += 1
        else:
            # Skip the step entirely: zero gradients can still advance AdamW
            # momentum or weight decay on an optimizer with existing state.
            norm = torch.zeros((), device=device)
        losses.append(loss_value)
        weighted_losses.append(policy_weight * loss_value)
        total_weighted_losses.append(policy_weight * loss_value
                                     + args.retention_weight * (retention_loss or 0.0))
    return {
        "policy_weight": policy_weight,
        "policy_losses": losses,
        "weighted_policy_losses": weighted_losses,
        "total_weighted_losses": total_weighted_losses,
        "retention_loss": retention_loss,
        "weighted_retention_loss": args.retention_weight * retention_loss if retention_loss is not None else None,
        "optimizer_steps": optimizer_steps,
        "grad_norm": float(norm),
        "behavior_teacher_max_abs_error": max_error,
        "policy_clip_fraction": clipped_tokens / max(1, action_tokens),
        "mean_negative_logratio": total_logratio / max(1, action_tokens),
        "policy_token_entropy": total_entropy / max(1, action_tokens),
        "retention_response_tokens": args.policy_epochs
        * sum(len(r["response_ids"]) for r in retention)
        if args.retention_weight
        else 0,
        "policy_response_tokens": args.policy_epochs * sum(len(r["response_ids"]) for r in rows),
        "policy_gradient_response_tokens": args.policy_epochs * sum(len(r["response_ids"]) for r in rows)
        if policy_weight else 0,
    }


def train_rl(model, tok, data, args):
    if args.temperature <= 0 or args.group_size < 2 or args.policy_epochs < 1:
        raise ValueError("RL needs positive temperature and >=2 samples per group")
    # Deliberately discard all body/reference fields before scheduling RL.
    prompt_weights = sampling_weights(data)
    prompts = [target_prompt(row) for row in data]
    if not prompts:
        raise ValueError("Empty RL prompt pool")
    retention = []
    retention_weights = None
    if args.retention_data:
        validate_training_path(args.retention_data)
        retention_rows = read_jsonl(args.retention_data)
        retention_weights = sampling_weights(retention_rows)
        guard_training(
            retention_rows, args.retention_data,
            getattr(args, "retention_guard_json", None) or args.guard_json,
            args.allow_unguarded_smoke
        )
        retention = [
            encode_training(
                row,
                tok,
                args.mode,
                model.config.max_seq_len,
                args.max_proof_lines,
                args.scratch_style,
            )
            for row in retention_rows
        ]
    rng = random.Random(args.seed)
    schedule = PromptSchedule(prompts, args.prompt_schedule, args.frontier_decay, args.frontier_floor,
                              weights=prompt_weights)
    compute = ComputeLedger(model)
    tracker = TrainingCompute(model, tok, args, compute, rl_step_compute_bound(
        compute, model, [len(tok.encode_prompt(prompt)) for prompt in prompts], args,
        [len(row["prompt_ids"]) + len(row["response_ids"]) for row in retention]))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95)
    )
    started = time.monotonic()
    total_tokens = 0
    with (Path(args.output) / "rollouts.jsonl").open("a") as trace:
        for step in range(1, args.steps + 1):
            if tracker.reached:
                break
            selected = schedule.select(rng, args.prompts_per_step)
            expanded = [prompt for prompt in selected for _ in range(args.group_size)]
            rows = sample(
                model,
                tok,
                expanded,
                mode=args.mode,
                temperature=args.temperature,
                seed=args.seed * 1000000 + step,
                max_new_tokens=args.max_new_tokens,
                batch_size=args.decode_batch,
                compute=compute,
            )
            for i, (row, prompt) in enumerate(zip(rows, expanded, strict=True)):
                row.update(judge_response(prompt, row["response_ids"], tok, args.mode))
                row.update(
                    prompt=prompt,
                    group=i // args.group_size,
                    attempt=i % args.group_size,
                    step=step,
                )
            rewards = torch.tensor([row["reward"] for row in rows]).reshape(-1, args.group_size)
            schedule.observe(selected, rewards.tolist())
            advantages = group_advantages(rewards, args.advantage).flatten().tolist()
            for row, advantage in zip(rows, advantages, strict=True):
                row["advantage"] = advantage
            retained = (
                sample_training_rows(rng, retention, args.retention_batch, retention_weights) if retention else []
            )
            metrics = policy_update(model, tok, rows, optimizer, args, retained, compute=compute)
            for row in rows:
                trace.write(json.dumps(row) + "\n")
            trace.flush()
            total_tokens += sum(len(row["response_ids"]) for row in rows)
            step_flops = tracker.after_step(step)
            emit(
                args.output,
                "rl",
                step=step,
                reward_mean=float(rewards.float().mean()),
                groups_mixed=int(((rewards.sum(1) > 0) & (rewards.sum(1) < args.group_size)).sum()),
                max_found_length=max((row["found_len"] or 0 for row in rows), default=0),
                mean_scratch_tokens=sum(row["scratch_tokens"] for row in rows) / len(rows),
                truncated_fraction=sum(row["finish_reason"] == "token_limit" for row in rows) / len(rows),
                actual_response_tokens=total_tokens,
                elapsed_seconds=time.monotonic() - started,
                step_flop_proxy=step_flops, total_flop_proxy=compute.total,
                budget_reached=tracker.reached,
                nonzero_advantage_response_tokens=sum(len(row["response_ids"]) for row in rows if row["advantage"]),
                weighted_retention_sampling=retention_weights is not None,
                **schedule.metrics(selected),
                **metrics,
            )
            if args.save_every and step % args.save_every == 0:
                tracker.save(Path(args.output) / f"step_{step:06d}.pt")
    result = tracker.result()
    result["actual_response_tokens"] = total_tokens
    return result


def draft_diagnostics(prompt, response, tok, mode):
    """Post-generation diagnostics only; never reward or decode-time feedback."""
    if mode != "cot":
        return {}
    ids = list(response)
    if len(ids) < 3 or ids[:2] != [tok.stoi["<THINK>"], tok.prf_id]:
        return {"draft_parseable": False}
    try:
        draft_end = ids.index(tok.eos_id, 2)
        think_end = ids.index(tok.stoi["</THINK>"])
    except ValueError:
        return {"draft_parseable": False}
    if draft_end >= think_end or ids[draft_end + 1] != tok.stoi["SEQ"]:
        return {"draft_parseable": False}
    valid, reason, length = safe_verify(prompt, tok.decode_body(ids[2:draft_end + 1]))
    return {"draft_parseable": True, "draft_valid": valid,
            "draft_reason": reason, "draft_lines": length if valid else None}


def reference_length(target):
    return target.get("reference_lines", target.get("n_lines", target.get("proof_length")))


def evaluation_summary(data, attempts, args, checkpoint_sha, data_sha):
    shortest = {}
    for row in attempts:
        if row["reward"]:
            shortest[row["index"]] = min(shortest.get(row["index"], float("inf")), row["found_len"])
    reference_groups = {}
    for index, target in enumerate(data):
        length = reference_length(target)
        if length is not None:
            group = reference_groups.setdefault(str(length), {"total": 0, "solved": 0})
            group["total"] += 1
            group["solved"] += int(index in shortest)
    budgets = sorted({1, args.samples, *[2 ** power for power in range(args.samples.bit_length())
                                       if 2 ** power <= args.samples]})
    scaling = []
    for budget in budgets:
        included = [row for row in attempts if row["attempt"] < budget]
        solved = {row["index"] for row in included if row["reward"]}
        scaling.append({"samples_per_problem": budget, "solved": len(solved),
                        "solve_rate": len(solved) / len(data),
                        "actual_response_tokens": sum(len(row["response_ids"]) for row in included),
                        "solved_reference_above_initial_cap": sum(
                            reference_length(data[index]) is not None
                            and reference_length(data[index]) > args.max_proof_lines for index in solved)})
    transitions = Counter(
        ("valid" if row["draft_valid"] else "invalid") + "_to_" + ("valid" if row["reward"] else "invalid")
        for row in attempts if row.get("draft_parseable"))
    solved_references = [reference_length(data[index]) for index in shortest
                         if reference_length(data[index]) is not None]
    return {
        "checkpoint_sha256": checkpoint_sha,
        "data_sha256": data_sha,
        "problems": len(data),
        "samples_per_problem": args.samples,
        "max_new_tokens_per_attempt": args.max_new_tokens,
        "solved": len(shortest),
        "solve_rate": len(shortest) / len(data),
        "max_shortest_found_length": max(shortest.values(), default=None),
        "shortest_found_length_histogram": dict(Counter(shortest.values())),
        "solved_above_initial_cap": sum(n > args.max_proof_lines for n in shortest.values()),
        "solved_reference_above_initial_cap": sum(n > args.max_proof_lines for n in solved_references),
        "max_solved_reference_length": max(solved_references, default=None),
        "by_reference_length": reference_groups,
        "by_sample_budget": scaling,
        "draft_final_transitions": dict(transitions),
        "actual_response_tokens": sum(len(row["response_ids"]) for row in attempts),
        "truncated_attempts": sum(row.get("finish_reason") == "token_limit" for row in attempts),
        "initial_proof_cap": args.max_proof_lines,
        "length_caveat": "Written proof length, not minimum proof complexity. solved_above_initial_cap is generated length; solved_reference_above_initial_cap uses benchmark reference length.",
    }


def evaluate(model, tok, data, args, checkpoint_sha):
    # A completed checkpoint hash is recorded before the first target is sampled.
    write_json(
        Path(args.output) / "evaluation_freeze.json",
        {
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "sha256": checkpoint_sha,
            "data_sha256": sha256(args.data),
            "args": vars(args),
            "unix_time": time.time(),
        },
    )
    prompts = [target_prompt(row) for row in data]
    if not prompts:
        raise ValueError("Empty evaluation set")
    attempts = []
    with (Path(args.output) / "attempts.jsonl").open("w") as stream:
        for attempt in range(args.samples):
            rows = sample(
                model,
                tok,
                prompts,
                mode=args.mode,
                temperature=args.temperature,
                seed=args.seed + attempt,
                max_new_tokens=args.max_new_tokens,
                batch_size=args.decode_batch,
            )
            for index, (target, prompt, row) in enumerate(zip(data, prompts, rows, strict=True)):
                result = {
                    "index": index,
                    "name": target.get("name", target.get("id", str(index))),
                    "prompt": prompt,
                    "generated_tokens": len(row["response_ids"]),
                    "attempt": attempt,
                    **row,
                    **judge_response(prompt, row["response_ids"], tok, args.mode),
                    **draft_diagnostics(prompt, row["response_ids"], tok, args.mode),
                }
                if args.mode == "direct":
                    result["final_tokens"] = len(row["response_ids"])
                else:
                    final_id = tok.stoi["<FINAL>"]
                    result["final_tokens"] = (
                        len(row["response_ids"]) - row["response_ids"].index(final_id) - 1
                        if final_id in row["response_ids"]
                        else 0
                    )
                for field in ("reference_lines", "n_lines", "proof_length", "split", "source"):
                    if field in target:
                        result[field] = target[field]
                stream.write(json.dumps(result) + "\n")
                attempts.append(result)
            stream.flush()
            emit(args.output, "eval_progress", attempt=attempt + 1, samples=args.samples)
    summary = evaluation_summary(data, attempts, args, checkpoint_sha, sha256(args.data))
    if sha256(args.checkpoint) != checkpoint_sha:
        raise RuntimeError("Checkpoint changed during evaluation")
    write_json(Path(args.output) / "summary.json", summary)
    write_json(
        Path(args.output) / "completed.json",
        {"stage": "eval", "checkpoint_sha256": checkpoint_sha, "unix_time": time.time()},
    )
    emit(args.output, "eval_complete", **summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "sft", "rl", "eval"))
    parser.add_argument("--checkpoint")
    parser.add_argument(
        "--fresh-config", help="JSON ModelConfig path; prepare only, no inherited weights"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--data")
    parser.add_argument("--mode", choices=("direct", "cot"), default="cot")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-seq-len", type=int)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--max-flop-proxy", type=compute_integer,
                        help="Stage-local dense FLOP proxy target; stop after the first complete crossing iteration")
    parser.add_argument("--flop-milestones", type=compute_integer, nargs="*", default=[],
                        help="Save flops_<exact integer>.pt at first crossing; optional0 saves the initial stage baseline")
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--microbatch", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--max-proof-lines", type=int, default=6)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--prompts-per-step", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--decode-batch", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--policy-epochs", type=int, default=1)
    parser.add_argument("--policy-weight", type=float, default=1.0,
                        help="Nonnegative policy-gradient coefficient; 0 retains rollout scoring/accounting but disables action gradients")
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--retention-data")
    parser.add_argument("--retention-guard-json",
                        help="Separate supervised-label guard for retention; defaults to --guard-json")
    parser.add_argument("--retention-weight", type=float, default=0.1)
    parser.add_argument("--retention-batch", type=int, default=16)
    parser.add_argument("--guard-json")
    parser.add_argument("--allow-unguarded-smoke", action="store_true")
    parser.add_argument("--advantage", choices=("loo", "standardized"), default="loo")
    parser.add_argument("--loss-normalization", choices=("fixed", "sequence"), default="fixed")
    parser.add_argument("--max-logprob-error", type=float, default=0.5)
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--scratch-style", choices=("reverse", "dependency", "draft", "repair"), default="dependency")
    parser.add_argument("--prompt-schedule", choices=("uniform", "frontier"), default="uniform")
    parser.add_argument("--frontier-decay", type=float, default=0.8)
    parser.add_argument("--frontier-floor", type=float, default=0.25)
    parser.add_argument("--final-release-json")
    parser.add_argument("--evaluation-release-json")
    parser.add_argument("--samples", type=int, default=8)
    args = parser.parse_args()
    if args.steps < 1 or args.microbatch < 1 or args.samples < 1:
        parser.error("steps, microbatch, and samples must be positive")
    if not math.isfinite(args.policy_weight) or args.policy_weight < 0:
        parser.error("--policy-weight must be finite and nonnegative")
    if args.max_flop_proxy is not None and args.max_flop_proxy <= 0:
        parser.error("--max-flop-proxy must be positive")
    if len(set(args.flop_milestones)) != len(args.flop_milestones):
        parser.error("--flop-milestones must be unique")
    if args.max_flop_proxy is not None and any(x > args.max_flop_proxy for x in args.flop_milestones):
        parser.error("A compute milestone exceeds --max-flop-proxy")
    if args.command not in ("sft", "rl") and (args.max_flop_proxy is not None or args.flop_milestones):
        parser.error("Compute stopping/milestones apply only to SFT or RL")
    if args.retention_guard_json and not args.retention_data:
        parser.error("--retention-guard-json requires --retention-data")
    if args.command != "prepare" and not args.data:
        parser.error("--data is required for sft, rl, and eval")
    if bool(args.checkpoint) == bool(args.fresh_config):
        parser.error("Provide exactly one of --checkpoint or --fresh-config")
    if args.fresh_config and args.command != "prepare":
        parser.error("--fresh-config is for prepare only")
    if args.command in ("sft", "rl"):
        validate_training_path(args.data)
        if args.retention_data:
            validate_training_path(args.retention_data)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        parser.error("--output must be a fresh empty directory to avoid overwriting artifacts")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = args.precision != "fp32"
    checkpoint_sha = sha256(args.checkpoint) if args.checkpoint else None
    if args.fresh_config:
        config = json.loads(Path(args.fresh_config).read_text())
        config["vocab_size"] = len(ProofTokenizer())
        if args.max_seq_len:
            config["max_seq_len"] = args.max_seq_len
        model = ProofModel(ModelConfig(**config)).to(args.device)
        tok, source_metadata = ProofTokenizer(), {"fresh_config_sha256": sha256(args.fresh_config)}
    else:
        model, tok, source_metadata = load_model(args.checkpoint, args.device, args.max_seq_len)
    model.r1_precision = args.precision
    if args.command == "prepare":
        if args.mode == "cot":
            model, tok = expand_vocabulary(model), ScratchTokenizer()
        save_checkpoint(
            output / "final.pt",
            model,
            tok,
            stage="prepared",
            mode=args.mode,
            source_checkpoint_sha256=checkpoint_sha,
            scratch_plan_version=scratch_version(args.scratch_style),
            args=vars(args),
        )
        write_json(
            output / "manifest.json",
            {
                "args": vars(args),
                "source_sha256": checkpoint_sha,
                "final_sha256": sha256(output / "final.pt"),
            },
        )
        return
    if args.mode == "cot" and not isinstance(tok, ScratchTokenizer):
        parser.error("CoT needs a prepared expanded-vocabulary checkpoint")
    if args.command == "eval":
        guard_final_release(args.data, checkpoint_sha, args.final_release_json)
        guard_development_release(args.data, checkpoint_sha, args.evaluation_release_json)
    manifest = {
        "args": vars(args),
        "source_sha256": checkpoint_sha,
        "data_sha256": sha256(args.data),
        "source_metadata": source_metadata,
        "config": asdict(model.config),
        "tokenizer": tok.state_dict(),
        "scratch_plan_version": scratch_version(args.scratch_style),
        "compute_accounting": {"version": COMPUTE_PROXY_VERSION, "note": COMPUTE_PROXY_NOTE},
    }
    if args.retention_data:
        manifest["retention_data_sha256"] = sha256(args.retention_data)
        retention_guard = args.retention_guard_json or args.guard_json
        manifest["retention_guard_sha256"] = sha256(retention_guard) if retention_guard else None
    if args.guard_json:
        manifest["guard_sha256"] = sha256(args.guard_json)
    if args.final_release_json:
        manifest["final_release_sha256"] = sha256(args.final_release_json)
    if args.evaluation_release_json:
        manifest["evaluation_release_sha256"] = sha256(args.evaluation_release_json)
    write_json(output / "manifest.json", manifest)
    if args.command in ("sft", "rl"):
        validate_training_path(args.data)
    data = read_jsonl(args.data)
    if args.command in ("sft", "rl"):
        guard_training(data, args.data, args.guard_json, args.allow_unguarded_smoke)
    if args.command == "sft":
        training_result = train_sft(model, tok, data, args)
    elif args.command == "rl":
        training_result = train_rl(model, tok, data, args)
    else:
        evaluate(model, tok, data, args, checkpoint_sha)
        return
    save_checkpoint(
        output / "final.pt",
        model,
        tok,
        mode=args.mode,
        stage=args.command,
        step=training_result["actual_steps"],
        source_checkpoint_sha256=checkpoint_sha,
        data_sha256=sha256(args.data),
        guard_sha256=sha256(args.guard_json) if args.guard_json else None,
        retention_data_sha256=sha256(args.retention_data) if args.retention_data else None,
        retention_guard_sha256=manifest.get("retention_guard_sha256"),
        scratch_plan_version=scratch_version(args.scratch_style),
        training_compute=training_result,
        args=vars(args),
    )
    write_json(
        output / "completed.json",
        {
            "final_sha256": sha256(output / "final.pt"),
            "stage": args.command,
            "source_checkpoint_sha256": checkpoint_sha,
            **training_result,
            "unix_time": time.time(),
        },
    )


if __name__ == "__main__":
    main()
