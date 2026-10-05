"""The base model's own mistakes at each decision state of the selected proofs (error seeding,
PROPOSAL.md section 3.2): sample k actions per state at temperature 0.8 and keep the distinct ones
the environment or `step_check` rejects (optionally also the ones that leave an unprovable goal).

Sampling is Dan's fork's `guided_sample.decode` (plain sampling: nothing forbidden), imported
read-only with bytecode writing off; the checkpoint loads with the fork's `model.load_ckpt`.
Defining names are rewritten to the canonical ones the validator requires before judging.
Writes one line per decision state: {proof_id, step, distractors: [{action, why}], ...}.
"""

from __future__ import annotations

import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

from . import dsl

STAGE1 = "/scratch/cowifornia/nd-rl/bench/dan_ckpts/best-state/stage1/stage1_best12_s0_b1200.pt"


def fork() -> tuple[Any, Any]:
    """(model.load_ckpt, guided_sample.decode) from the vendored fork modules, each call run with
    the fork's bare module names bound (`code.thinking.dan.bare`)."""
    from code.thinking import dan

    os.environ["ND_OFFLINE"] = "1"
    d = dan.load()

    def bound(fn: Any) -> Any:
        def call(*a: Any, **kw: Any) -> Any:
            with dan.bare():
                return fn(*a, **kw)

        return call

    return bound(d.model.load_ckpt), bound(d.guided_sample.decode)


def canonical(env: Any, act: list[str]) -> list[str]:
    """The action with the names it introduces replaced by the environment's canonical ones (the
    positions `trace_dsl.canonical_names_ok` checks)."""
    td = dsl.dsl()
    if len(act) < 2 or act[0] != "have":
        return list(act)
    a = list(act)
    nh, nb = td.next_names(env)
    a[1] = nh
    j = a.index(":=") if ":=" in a else -1
    if j > 0 and j + 4 < len(a) and a[j + 1] == "(" and a[j + 2] == "fun":
        a[j + 4] = nb
    elif j > 0 and j + 6 < len(a) and a[j + 1] == "Or.elim":
        a[j + 6] = nb
    return a


SPECIAL = {"<pad>", "<eos>", "<st>", "<nl>", "<act>", "⊢"}


def usable(act: list[str]) -> bool:
    """Whether the action can appear as a candidate line of a think block."""
    td = dsl.dsl()
    if len(act) < 2 or act[0] not in ("have", "exact"):
        return False
    return all(t in td.VOCAB and t not in SPECIAL and t not in td.NEW_TOKENS for t in act)


def decision_states(selection: list[dict[str, Any]], limit: int = 0) -> list[dict[str, Any]]:
    out = []
    for rec in selection:
        for i, s in enumerate(dsl.reference_steps(rec["prompt"], rec["proof"], with_env=True)):
            if s.kind == "decision":
                out.append({"proof_id": rec["id"], "step": i, "ref": s})
                if limit and len(out) >= limit:
                    return out
    return out


def judge_samples(
    env: Any, ref: list[str], samples: list[list[str]], keep: set[str]
) -> dict[str, Any]:
    td = dsl.dsl()
    counts: Counter[str] = Counter()
    why: dict[str, str] = {}
    n_valid = 0
    seen_order: dict[str, int] = {}
    for act in samples:
        a = canonical(env, act)
        key = " ".join(a)
        seen_order.setdefault(key, len(seen_order))
        if key not in why:
            if a == ref:
                why[key] = ""
            else:
                if not usable(a):
                    why[key] = "unusable: cannot be written as a candidate line"
                else:
                    reason, _ = td.judge_action(env, a)
                    why[key] = reason or ""
        counts[key] += 1
        n_valid += not why[key]
    kept = [
        k
        for k in sorted(counts, key=lambda k: (-counts[k], seen_order[k]))
        if why[k] and why[k].split(":")[0] in keep
    ]
    return {
        "kept": kept,
        "why": why,
        "counts": counts,
        "n_valid": n_valid,
        "n_distinct": len(counts),
    }


def run(
    selection_path: Path,
    out_path: Path,
    ckpt: str = STAGE1,
    device: str = "cpu",
    k: int = 16,
    temperature: float = 0.8,
    max_per_state: int = 3,
    max_action: int = 256,
    batch: int = 2048,
    seed: int = 0,
    limit_states: int = 0,
    with_dead: bool = False,
) -> dict[str, Any]:
    import torch

    with open(selection_path) as f:
        sel = [json.loads(x) for x in f if x.strip()]
    states = decision_states(sel, limit_states)
    load_ckpt, decode = fork()
    t0 = time.time()
    model, tok, _ = load_ckpt(ckpt, device)
    model.eval()
    if not getattr(tok, "canon", False):
        raise ValueError(f"{ckpt}: a lean_staten checkpoint is required")
    rows = [(si, tok.encode_toks(s["ref"].state)) for si, s in enumerate(states) for _ in range(k)]
    samples: list[list[list[str]]] = [[] for _ in states]
    truncated = 0
    for c in range(0, len(rows), batch):
        chunk = rows[c : c + batch]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            outs = decode(
                model,
                tok,
                [p for _, p in chunk],
                [[] for _ in chunk],
                temperature,
                max_action,
                seed * 100003 + c,
                {},
            )
        for (si, _), (ids, _) in zip(chunk, outs, strict=True):
            act, ended = tok.decode_action(ids)
            if ended:
                samples[si].append(act)
            else:
                truncated += 1
    sample_s = time.time() - t0
    keep = {"env", "step_check"} | ({"dead"} if with_dead else set())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_kept = Counter()
    with open(out_path, "w") as f:
        for s, smp in zip(states, samples, strict=True):
            ref: dsl.RefStep = s["ref"]
            j = judge_samples(ref.env, ref.action, smp, keep)
            ds = [
                {
                    "action": dsl.ascii_toks(x.split(" ")),
                    "why": dsl.ascii_text(j["why"][x]),
                    "count": j["counts"][x],
                }
                for x in j["kept"][:max_per_state]
            ]
            n_kept[len(ds)] += 1
            f.write(
                json.dumps(
                    {
                        "proof_id": s["proof_id"],
                        "step": s["step"],
                        "ref_action": ref.action_ascii,
                        "distractors": ds,
                        "n_samples": k,
                        "n_ended": len(smp),
                        "n_distinct": j["n_distinct"],
                        "n_valid": j["n_valid"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    meta = {
        "ckpt": ckpt,
        "device": device,
        "k": k,
        "temperature": temperature,
        "max_action": max_action,
        "seed": seed,
        "keep": sorted(keep),
        "n_states": len(states),
        "n_truncated": truncated,
        "states_by_n_distractors": dict(sorted(n_kept.items())),
        "sample_s": round(sample_s, 2),
        "selection": str(selection_path),
    }
    out_path.with_name(out_path.stem + ".meta.json").write_text(json.dumps(meta, indent=1))
    return meta
