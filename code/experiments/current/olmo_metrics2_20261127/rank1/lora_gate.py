# ruff: noqa: E501
"""vLLM-only applied / equivalence gate for the rank-1 adapters (Turn 5 B.2 step 5; Turn 6 D4, D8). POD side, one eager engine.

  python3 lora_gate.py --model /workspace/models/dpo --adapters runs/build/adapters --pool /workspace/m2/pool --out runs/gate/out/gate.json

Teacher-forced (not sampled) on the two check examples of the 2026-11-25 gate (first two pool/test.jsonl prompts, the final model's
stored response ids). (ii) consistency: the amplified rank-1 copy gate_m{m}_r1 against gate_m{m}_r16 (16 equal columns, alpha 16):
the smallest m in (1, 4, 16, 64) whose r16 change of summed base nats is >= 5 % is used; PASS = summed-nats changes agree
within 1 % (of the change), per-token change correlation >= 0.99, base nats reproduced after the adapters within 0.5 %.
(iii) applied: A1, A1u, A2 each give a nonzero (> twice the base repeatability and > 1e-4 relative) change of nats that repeats
in a second call (Amendment 3: within the LoRA path's measured nondeterminism; see applied_verdict). The engine uses max_lora_rank 16 like the sampling jobs. Exit: 0 PASS; 30 consistency fail; 31
underpowered (no m reaches 5 %); 32 applied fail; 3 input error. A failure means: no adapter sampling, ExPO still runs.

Amendment 4 (2026-11-27; PREREG_rank1, gate_amendments_audit G2). (1) Base exactness fails closed: all three base calls must be
bit-identical per example (base_exact, base_exact_all in the JSON) or the exit code is 33 (token rule unverifiable). (2) Control
adapter Z = A1 with every lora_B tensor zeroed (make_zero_b_adapter; adapters_ctrl/Z next to gate.json), called twice; with base
exact, Z must reproduce base bit for bit or the code is 34 (token rule invalid). (3) A1 partial-application rule: |sum(call 1) -
sum(call 2)| <= 0.10 x |sum(call 1) - sum(base)| per example, added to A1's pass in exact mode (code 32 if it fails). (4) On 33 / 34
only, a noise-floor fallback (3 x the largest mean |difference| among the base and Z repeat calls, every call of every adapter,
plus the A1 rule and consistency) is recorded in res["fallback"]; the exit code stays 33 / 34 whatever it says. (5) res["calls"]
holds per-call sum, mean_abs_vs_base1 and frac_tokens_diff_vs_base1 for base1-3, Z1-2 and both calls of A1, A1u, A2.
Exit codes: 0 PASS; 30 consistency fail; 31 underpowered; 32 applied fail (exact mode); 33 base not bit-exact (fallback in gate.json);
34 Z control not bit-identical to base (fallback in gate.json); 3 input error.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

CHANGE_MIN = 0.05
SUM_TOL = 0.01
CORR_MIN = 0.99
BASE_TOL = 0.005
M_LADDER = (1, 4, 16, 64)


def b64_ids(s: str) -> list[int]:
    return np.frombuffer(base64.b64decode(s), dtype=np.int32).tolist()


def consistency_verdict(
    base: list[np.ndarray],
    base_again: list[np.ndarray],
    r1: list[np.ndarray],
    r16: list[np.ndarray],
) -> dict:
    """Per-token negative log-probs of the response tokens, one array per check example. Changes are adapter - base (nats)."""
    ex = []
    ok = True
    for b, ba, a1, a16 in zip(base, base_again, r1, r16):
        ch1, ch16 = a1 - b, a16 - b
        s1, s16, sb = float(ch1.sum()), float(ch16.sum()), float(b.sum())
        sum_rel = abs(s1 - s16) / abs(s16) if s16 else float("inf")
        corr = (
            float(np.corrcoef(ch1, ch16)[0, 1])
            if ch1.std() > 0 and ch16.std() > 0
            else float("nan")
        )
        base_rel = abs(float(ba.sum()) - sb) / sb
        passed = sum_rel <= SUM_TOL and corr >= CORR_MIN and base_rel <= BASE_TOL
        ok &= bool(passed)
        ex.append(
            {
                "base_nats": sb,
                "change_r1": s1,
                "change_r16": s16,
                "change_frac_of_base": s16 / sb,
                "sum_rel_diff": sum_rel,
                "token_corr": corr,
                "base_again_rel": base_rel,
                "pass": bool(passed),
            }
        )
    return {"examples": ex, "pass": bool(ok)}


def powered(base: list[np.ndarray], r16: list[np.ndarray]) -> bool:
    return all(abs(float((a - b).sum())) / float(b.sum()) >= CHANGE_MIN for b, a in zip(base, r16))


REPEAT_CORR_MIN = 0.999  # PREREG_rank1 Amendment 3 (2026-11-27)
REPEAT_SUM_TOL = (
    0.005  # of base nats (Amendment 3a: 0.1 % -> 0.5 %, with the both-calls token check below)
)
TOK_DIFF_FRAC_MIN = 0.10


def applied_verdict(
    base: list[np.ndarray],
    base_again: list[np.ndarray],
    ad: list[np.ndarray],
    ad_again: list[np.ndarray],
) -> dict:
    """PREREG_rank1 Amendment 3 (2026-11-27). The first version required the two calls of an adapter to agree per token within
    1e-6 ('repeats exactly'); on rp-m2a vLLM 0.11.0's LoRA path is not bit-deterministic (base calls repeat exactly; repeated
    calls of one adapter differ at most tokens, max 0.25 nats per token, summed 0.01-0.08 % of base nats, per-token
    correlation >= 0.9996), so that check failed for every adapter although A1 changed base nats by 5-11 %. Now:
    repeats = exact agreement, or per-token correlation >= 0.999 and summed difference <= 0.5 % of base nats (3a);
    changed = summed change > max(2 x base repeatability, 1e-4 x base) or, when base repeats exactly per token, the adapter's
    output differs from base at >= 10 % of tokens (a LoRA request with an all-zero adapter adds exactly zero); with base
    exactly repeatable, BOTH calls must differ from base at >= 10 % of tokens (3a: a call that dropped the adapter is caught
    directly, which the summed tolerance cannot do for weak adapters such as A2).
    The old exact-repeat flag is kept in the output as 'repeats_exact'."""
    ex, ok = [], True
    for b, ba, a, a2 in zip(base, base_again, ad, ad_again):
        bs = float(b.sum())
        rep = abs(float(ba.sum()) - bs)
        ch = abs(float(a.sum()) - bs)
        same = bool(np.allclose(a, a2, rtol=0, atol=1e-6))
        sum_rep = abs(float(a.sum()) - float(a2.sum())) / bs
        corr = 1.0 if same else float(np.corrcoef(a, a2)[0, 1])
        repeats = same or (corr >= REPEAT_CORR_MIN and sum_rep <= REPEAT_SUM_TOL)
        base_exact = bool(np.array_equal(b, ba))
        frac_diff = float(np.mean(np.abs(a - b) > 1e-6))
        frac_diff2 = float(np.mean(np.abs(a2 - b) > 1e-6))
        changed = ch > max(2 * rep, 1e-4 * bs) or (base_exact and frac_diff >= TOK_DIFF_FRAC_MIN)
        if (
            base_exact
        ):  # Amendment 3a: an adapter dropped in either call would reproduce base bit for bit
            changed = changed and min(frac_diff, frac_diff2) >= TOK_DIFF_FRAC_MIN
        passed = changed and repeats
        ok &= bool(passed)
        ex.append(
            {
                "abs_change": ch,
                "base_repeatability": rep,
                "repeats_exact": same,
                "repeat_sum_frac_of_base": sum_rep,
                "repeat_corr": corr,
                "repeats": bool(repeats),
                "frac_tokens_differing_from_base": frac_diff,
                "frac_tokens_differing_from_base_call2": frac_diff2,
                "changed": bool(changed),
                "pass": bool(passed),
            }
        )
    return {"examples": ex, "pass": bool(ok)}


A1_REPEAT_FRAC = 0.10  # Amendment 4
FALLBACK_FACTOR = 3.0  # Amendment 4
ADAPTERS = ("A1", "A1u", "A2")


def a1_repeat_rule(
    base: list[np.ndarray], a1_call1: list[np.ndarray], a1_call2: list[np.ndarray]
) -> list[bool]:
    """Amendment 4: per example |sum(call1) - sum(call2)| <= 0.10 x |sum(call1) - sum(base)|."""
    return [
        abs(float(c1.sum()) - float(c2.sum()))
        <= A1_REPEAT_FRAC * abs(float(c1.sum()) - float(b.sum()))
        for b, c1, c2 in zip(base, a1_call1, a1_call2)
    ]


def exactness(base_calls: list[list[np.ndarray]], z_calls: list[list[np.ndarray]]) -> dict:
    """base_calls = [base, base_again, base_again2], z_calls = [z1, z2]; each a list (per example) of arrays."""
    b1 = base_calls[0]
    base_exact = [
        all(bool(np.array_equal(b1[i], bc[i])) for bc in base_calls[1:]) for i in range(len(b1))
    ]
    z_exact = [all(bool(np.array_equal(zc[i], b1[i])) for zc in z_calls) for i in range(len(b1))]
    return {
        "base_exact": base_exact,
        "base_exact_all": bool(all(base_exact)),
        "z_exact": z_exact,
        "z_exact_all": bool(all(z_exact)),
    }


def call_stats(base1: list[np.ndarray], calls: dict[str, list[np.ndarray]]) -> dict:
    out = {}
    for name, arrs in calls.items():
        out[name] = [
            {
                "sum": float(x.sum()),
                "mean_abs_vs_base1": float(np.mean(np.abs(x - b))),
                "frac_tokens_diff_vs_base1": float(np.mean(np.abs(x - b) > 1e-6)),
            }
            for x, b in zip(arrs, base1)
        ]
    return out


def noise_floor_fallback(
    base_calls: list[list[np.ndarray]],
    z_calls: list[list[np.ndarray]],
    adapter_calls: dict[str, tuple],
    consistency_ok: bool,
) -> dict:
    """Amendment 4 fallback: an adapter call counts as applied when its mean |difference from base1| is >= 3 x the noise n, the
    largest mean |difference from base1| among base2, base3, z1, z2 (if n == 0: any difference)."""
    b1 = base_calls[0]
    others = list(base_calls[1:]) + list(z_calls)
    a1_rule = a1_repeat_rule(b1, *adapter_calls["A1"])
    per_ex, ok = [], bool(consistency_ok)
    for i in range(len(b1)):
        n = max(float(np.mean(np.abs(o[i] - b1[i]))) for o in others)
        e = {"n": n, "a1_repeat_within_10pct": bool(a1_rule[i]), "calls": {}}
        ok &= bool(a1_rule[i])
        for v in ADAPTERS:
            for k, call in enumerate(adapter_calls[v], 1):
                d = float(np.mean(np.abs(call[i] - b1[i])))
                applied = bool(d >= FALLBACK_FACTOR * n) if n > 0 else bool(d > 0)
                e["calls"][f"{v}_{k}"] = {"d": d, "applied": applied}
                ok &= applied
        per_ex.append(e)
    return {
        "mode": "noise_floor",
        "factor": FALLBACK_FACTOR,
        "per_example": per_ex,
        "pass": bool(ok),
    }


def decide(
    consistency_ok: bool,
    base_calls: list[list[np.ndarray]],
    z_calls: list[list[np.ndarray]],
    adapter_calls: dict[str, tuple],
) -> tuple[int, dict]:
    """adapter_calls = {"A1": (call1, call2), "A1u": ..., "A2": ...}. Returns (exit code, result fields). Never falls back to a
    different rule inside the exact-mode verdict: not-exact bases give 33 / 34 (with the fallback recorded for information only)."""
    if not consistency_ok:
        return 30, {}
    b1 = base_calls[0]
    ex = exactness(base_calls, z_calls)
    out: dict = dict(ex)
    out["a1_repeat_within_10pct"] = a1_repeat_rule(b1, *adapter_calls["A1"])
    if not ex["base_exact_all"] or not ex["z_exact_all"]:
        out["fallback"] = noise_floor_fallback(base_calls, z_calls, adapter_calls, consistency_ok)
        return (33 if not ex["base_exact_all"] else 34), out
    app = {}
    for v in ADAPTERS:
        app[v] = applied_verdict(b1, base_calls[1], *adapter_calls[v])
    for e, r in zip(app["A1"]["examples"], out["a1_repeat_within_10pct"]):
        e["a1_repeat_within_10pct"] = bool(r)
        e["pass"] = bool(e["pass"] and r)
    app["A1"]["pass"] = bool(all(e["pass"] for e in app["A1"]["examples"]))
    out["applied"] = app
    return (0 if all(x["pass"] for x in app.values()) else 32), out


def make_zero_b_adapter(src_dir, dst_dir) -> dict:
    """Copy an adapter directory, zeroing every tensor whose key contains 'lora_B' (same dtype / shape); others byte-identical."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    src, dst = Path(src_dir), Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.is_file() and f.name != "adapter_model.safetensors":
            shutil.copy2(f, dst / f.name)
    tensors, nz = {}, 0
    with safe_open(str(src / "adapter_model.safetensors"), framework="pt") as fh:
        meta = fh.metadata()
        for k in fh.keys():
            t = fh.get_tensor(k)
            if "lora_B" in k:
                t = t.new_zeros(t.shape)
                nz += 1
            tensors[k] = t.contiguous()
    out = dst / "adapter_model.safetensors"
    save_file(tensors, str(out), metadata=meta or {"format": "pt"})
    return {
        "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        "n_zeroed": nz,
        "n_tensors": len(tensors),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--adapters", required=True)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    ap.add_argument("--n-examples", type=int, default=2)
    a = ap.parse_args(argv)
    pool = [
        json.loads(ln)
        for ln in (Path(a.pool) / "test.jsonl").read_text().splitlines()
        if ln.strip()
    ][: a.n_examples]
    exs = [(b64_ids(r["prompt_ids"]), b64_ids(r["ids"]), r["pid"]) for r in pool]
    try:
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest
    except Exception as e:  # noqa: BLE001
        print(f"vllm import failed: {e!r}", file=sys.stderr)
        return 3
    llm = LLM(
        model=a.model,
        tokenizer=a.tokenizer or a.model,
        dtype="bfloat16",
        max_model_len=max(len(p) + len(r) for p, r, _ in exs) + 16,
        enable_lora=True,
        max_lora_rank=16,
        max_loras=1,
        gpu_memory_utilization=a.gpu_mem,
        enable_prefix_caching=False,
        enforce_eager=True,
        seed=0,
        generation_config="vllm",
    )
    sp = SamplingParams(max_tokens=1, temperature=1.0, prompt_logprobs=0, detokenize=False)
    ids_of: dict[str, int] = {}

    z_dir = Path(a.out).parent / "adapters_ctrl" / "Z"
    zinfo = make_zero_b_adapter(Path(a.adapters) / "A1", z_dir)
    path_of = {"Z": z_dir}

    def req(name):
        if name is None:
            return None
        ids_of.setdefault(name, len(ids_of) + 1)
        return LoRARequest(
            name, ids_of[name], str(Path(path_of.get(name, Path(a.adapters) / name)).resolve())
        )

    def nats(name) -> list[np.ndarray]:
        out = []
        for p, r, _ in exs:
            ids = p + r
            o = llm.generate(
                [{"prompt_token_ids": ids}], sp, lora_request=req(name), use_tqdm=False
            )[0]
            out.append(
                np.array(
                    [-o.prompt_logprobs[j][ids[j]].logprob for j in range(len(p), len(ids))],
                    dtype=np.float64,
                )
            )
        return out

    t0 = time.time()
    base = nats(None)
    res: dict = {
        "examples": [e[2] for e in exs],
        "base_nats": [float(b.sum()) for b in base],
        "utc": time.strftime("%FT%TZ", time.gmtime()),
    }
    chosen, cons = None, None
    r16_by_m, r1_by_m = {}, {}
    for m in M_LADDER:
        r16_by_m[m] = nats(f"gate_m{m}_r16")
        if powered(base, r16_by_m[m]):
            chosen = m
            r1_by_m[m] = nats(f"gate_m{m}_r1")
            break
    code = 0
    if chosen is None:
        res["status"] = "underpowered: no m in the ladder reaches a 5 % change of base nats"
        code = 31
    else:
        base_again = nats(None)
        cons = consistency_verdict(base, base_again, r1_by_m[chosen], r16_by_m[chosen])
        res["m"], res["consistency"] = chosen, cons
        if not cons["pass"]:
            code = 30
    if code == 0:
        base_again2 = nats(None)
        z1, z2 = nats("Z"), nats("Z")
        ad = {v: (nats(v), nats(v)) for v in ADAPTERS}
        base_calls, z_calls = [base, base_again, base_again2], [z1, z2]
        res["z_control"] = zinfo
        code, extra = decide(True, base_calls, z_calls, ad)
        res.update(extra)
        calls = {"base1": base, "base2": base_again, "base3": base_again2, "Z1": z1, "Z2": z2}
        for v in ADAPTERS:
            calls[f"{v}_1"], calls[f"{v}_2"] = ad[v]
        res["calls"] = call_stats(base, calls)
        if code == 33:
            res["status"] = "base calls not bit-exact: token rule unverifiable"
        elif code == 34:
            res["status"] = "zero-B control adapter does not reproduce base: token rule invalid"
    res["verdict_code"] = code
    res["seconds"] = time.time() - t0
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    print(f"GATE exit={code}: {'PASS' if code == 0 else res.get('status', 'FAIL')}")
    return code


if __name__ == "__main__":
    sys.exit(main())
