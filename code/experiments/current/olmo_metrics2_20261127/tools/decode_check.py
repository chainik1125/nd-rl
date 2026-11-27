# ruff: noqa: E501
"""Decode-path check in an engine built like Dan's sampler (compiled, CUDA graphs, prefix cache). POD side (G3 / G7 of discourse/gate_amendments_audit.md).

  decode_check.py lora --model M --tokenizer T --adapter DIR --adapter-name A1 --pool POOL --out OUT.json [--n-tokens 128 --seed 4242 --gpu-mem 0.92 --max-model-len 33321]
  decode_check.py steer-gen --model M --tokenizer T --p3 P3DIR --pool POOL --out DIR [...]      (patch installed; writes DIR/steer_gen.json, DIR/active_decode.jsonl)
  decode_check.py stock-score --model M --tokenizer T --pool POOL --dir DIR [...]                (separate process, NO patch; writes DIR/stock_score.json)
  decode_check.py steer-summarize --dir DIR --p3 P3DIR --out OUT.json

Tokens generated under the adapter / vector must have decode log-probs that match a prefill re-score under the same adapter / vector
(TOL_MEAN, TOL_P99), and the adapter must visibly change those log-probs relative to base (power). Exit: 0 PASS; 2 NO_POWER (rerun once
with --n-tokens 512); 1 FAIL; 3 input error. Last line: DECODE_CHECK <PASS|NO_POWER|FAIL> {checks}.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from pathlib import Path

import numpy as np

TOL_MEAN = 0.0334  # rp-m2a pre-flight tolerance: 3 x the vLLM-stock-vs-HF floor (runs/m3_pf/out/preflight_report.json)
TOL_P99 = 0.345
N_PROMPTS = 2
HERE = Path(__file__).resolve().parent


def b64_ids(s: str) -> list[int]:
    return np.frombuffer(base64.b64decode(s), dtype=np.int32).tolist()


def _stats(x: list[np.ndarray], y: list[np.ndarray]) -> dict:
    d = np.abs(
        np.concatenate([np.asarray(a, float) for a in x])
        - np.concatenate([np.asarray(b, float) for b in y])
    )
    return {
        "mean_abs": float(d.mean()),
        "p99_abs": float(np.percentile(d, 99)),
        "max_abs": float(d.max()),
        "n_tokens": int(d.size),
    }


def verdict(d_x, p_x, p_base_on_x, d_b, p_b) -> dict:
    """Per-token arrays, one per prompt. d = decode log-probs, p = prefill re-scores; x = adapter / vector, b = base."""
    s_dp, s_pp, s_dbase = _stats(d_x, p_x), _stats(p_x, p_base_on_x), _stats(d_x, p_base_on_x)
    s_b = _stats(d_b, p_b)
    checks = {
        "decode_matches_prefill": bool(s_dp["mean_abs"] <= TOL_MEAN and s_dp["p99_abs"] <= TOL_P99),
        "has_power": bool(s_pp["mean_abs"] > TOL_MEAN),
        "base_decode_matches_prefill": bool(
            s_b["mean_abs"] <= TOL_MEAN and s_b["p99_abs"] <= TOL_P99
        ),
    }
    return {
        "stats": {
            "decode_vs_prefill_x": s_dp,
            "prefill_x_vs_prefill_base": s_pp,
            "decode_x_vs_prefill_base_on_x": s_dbase,
            "decode_vs_prefill_base": s_b,
        },
        "checks": checks,
        "constants": {"TOL_MEAN": TOL_MEAN, "TOL_P99": TOL_P99},
    }


def status_of(checks: dict) -> tuple[str, int]:
    rest = [v for k, v in checks.items() if k != "has_power"]
    if all(rest) and checks["has_power"]:
        return "PASS", 0
    if all(rest) and not checks["has_power"]:
        return "NO_POWER", 2
    return "FAIL", 1


def finish(res: dict, out: str | Path) -> int:
    status, code = status_of(res["checks"])
    res["status"] = status
    res["verdict"] = status
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(res))
    print(f"DECODE_CHECK {status} {json.dumps(res['checks'])}", flush=True)
    return code


def load_prompts(pool: str, n: int = N_PROMPTS) -> list[list[int]]:
    rows = [
        json.loads(ln) for ln in (Path(pool) / "test.jsonl").read_text().splitlines() if ln.strip()
    ][:n]
    if len(rows) < n:
        raise ValueError(f"{pool}/test.jsonl has {len(rows)} rows, need {n}")
    return [b64_ids(r["prompt_ids"]) for r in rows]


def engine_kwargs(a, lora: bool) -> dict:
    """Dan's sampler (olmo_ifladder/sample.py 255-264, vLLM 0.11.0): compiled + CUDA graphs (NOT eager), prefix cache on."""
    kw = dict(
        model=a.model,
        tokenizer=a.tokenizer or a.model,
        dtype="bfloat16",
        generation_config="vllm",
        max_model_len=a.max_model_len,
        gpu_memory_utilization=a.gpu_mem,
        enable_prefix_caching=True,
        seed=a.seed,
        max_num_seqs=512,
        disable_log_stats=False,
    )
    if lora:
        kw.update(enable_lora=True, max_lora_rank=16, max_loras=1)
    return kw


def build_engine(kw: dict):
    from vllm import LLM

    return LLM(**kw)


def generate(llm, prompts, n_tokens, seed, lora_req=None):
    """Decode: n=1, T=1, ignore_eos so exactly n_tokens come back. Returns (tokens, decode log-probs) per prompt."""
    from vllm import SamplingParams

    sp = SamplingParams(
        n=1, temperature=1.0, top_p=1.0, max_tokens=n_tokens, seed=seed, logprobs=0, ignore_eos=True
    )
    outs = llm.generate(
        [{"prompt_token_ids": p} for p in prompts], sp, lora_request=lora_req, use_tqdm=False
    )
    toks, lps = [], []
    for o in outs:
        c = o.outputs[0]
        t = [int(x) for x in c.token_ids]
        if len(t) != n_tokens:
            raise RuntimeError(f"expected {n_tokens} generated tokens, got {len(t)}")
        toks.append(t)
        lps.append(
            np.array([c.logprobs[i][t[i]].logprob for i in range(n_tokens)], dtype=np.float64)
        )
    return toks, lps


def rescore(llm, prompts, toks, lora_req=None):
    """Prefill re-score of prompt + generated tokens; log-probs at the generated positions only (prompt_logprobs[0] is None)."""
    from vllm import SamplingParams

    sp = SamplingParams(max_tokens=1, temperature=1.0, prompt_logprobs=0)
    seqs = [list(p) + list(t) for p, t in zip(prompts, toks)]
    outs = llm.generate(
        [{"prompt_token_ids": s} for s in seqs], sp, lora_request=lora_req, use_tqdm=False
    )
    res = []
    for p, s, o in zip(prompts, seqs, outs):
        res.append(
            np.array(
                [o.prompt_logprobs[j][s[j]].logprob for j in range(len(p), len(s))],
                dtype=np.float64,
            )
        )
    return res


def _lst(x):
    return [np.asarray(a).tolist() for a in x]


def _arr(x):
    return [np.asarray(a, dtype=np.float64) for a in x]


def cmd_lora(a) -> int:
    from vllm.lora.request import LoRARequest

    prompts = load_prompts(a.pool)
    kw = engine_kwargs(a, lora=True)
    llm = build_engine(kw)
    req = LoRARequest(a.adapter_name, 1, str(Path(a.adapter).resolve()))
    tok_x, d_x = generate(llm, prompts, a.n_tokens, a.seed, req)
    p_x = rescore(llm, prompts, tok_x, req)
    p_base_on_x = rescore(llm, prompts, tok_x, None)
    tok_b, d_b = generate(llm, prompts, a.n_tokens, a.seed, None)
    p_b = rescore(llm, prompts, tok_b, None)
    res = verdict(d_x, p_x, p_base_on_x, d_b, p_b)
    res.update(
        mode="lora",
        adapter=a.adapter,
        adapter_name=a.adapter_name,
        n_tokens=a.n_tokens,
        seed=a.seed,
        engine_kwargs=kw,
        tokens_x=tok_x,
        tokens_b=tok_b,
        d_x=_lst(d_x),
        p_x=_lst(p_x),
        p_base_on_x=_lst(p_base_on_x),
        d_b=_lst(d_b),
        p_b=_lst(p_b),
    )
    return finish(res, a.out)


def cmd_steer_gen(a) -> int:
    p3 = Path(a.p3)
    chosen = json.loads((p3 / "chosen.json").read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    prompts = load_prompts(a.pool)
    active = out / "active_decode.jsonl"
    if active.exists():
        active.unlink()
    os.environ["STEER_LAYER"] = str(chosen["layer"])
    os.environ["STEER_VEC"] = chosen["files"][
        "chosen"
    ]  # relative to the pod's /workspace/m2 cwd, like steer_preflight
    os.environ["STEER_ACTIVE_FILE"] = str(active)
    if os.environ.get("STEER_INPROC") == "1":
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    sys.path.insert(0, str(HERE.parent / "steer"))
    import steer_patch

    steer_patch.install_from_env()  # BEFORE the engine, so compile / graph capture trace the patched forward
    kw = engine_kwargs(a, lora=False)
    llm = build_engine(kw)
    tok_s, d_s = generate(llm, prompts, a.n_tokens, a.seed)
    p_s = rescore(llm, prompts, tok_s)
    (out / "steer_gen.json").write_text(
        json.dumps(
            {
                "n_tokens": a.n_tokens,
                "seed": a.seed,
                "layer": chosen["layer"],
                "vec_sha256_expected": chosen["sha256"]["chosen"],
                "engine_kwargs": kw,
                "tokens_S": tok_s,
                "d_S": _lst(d_s),
                "P_S": _lst(p_s),
            }
        )
    )
    print("DECODE_CHECK_STEER_GEN_DONE", flush=True)
    return 0


def cmd_stock_score(a) -> int:
    for k in ("STEER_LAYER", "STEER_VEC", "STEER_ACTIVE_FILE"):
        os.environ.pop(k, None)  # stock engine: no patch is installed in this process
    d = Path(a.dir)
    g = json.loads((d / "steer_gen.json").read_text())
    prompts = load_prompts(a.pool)
    kw = engine_kwargs(a, lora=False)
    llm = build_engine(kw)
    p_stock = rescore(llm, prompts, g["tokens_S"])
    tok_b, d_b = generate(llm, prompts, g["n_tokens"], g["seed"])
    p_b = rescore(llm, prompts, tok_b)
    (d / "stock_score.json").write_text(
        json.dumps(
            {
                "n_tokens": g["n_tokens"],
                "seed": g["seed"],
                "engine_kwargs": kw,
                "tokens_B": tok_b,
                "P_stock": _lst(p_stock),
                "d_B": _lst(d_b),
                "P_B": _lst(p_b),
                "tokens_S": g["tokens_S"],
            }
        )
    )
    print("DECODE_CHECK_STOCK_SCORE_DONE", flush=True)
    return 0


def steer_active_check(active_file: Path, chosen: dict) -> dict:
    recs = []
    if active_file.exists():
        for ln in active_file.read_text().splitlines():
            if ln.strip():
                try:
                    recs.append(json.loads(ln))
                except ValueError:
                    pass
    sha = chosen["sha256"]["chosen"]
    ok = any(
        r.get("event") == "STEER_ACTIVE"
        and int(r.get("layer", -1)) == int(chosen["layer"])
        and r.get("vec_sha256") == sha
        for r in recs
    )
    return {"ok": bool(ok), "n_records": len(recs)}


def cmd_steer_summarize(a) -> int:
    d = Path(a.dir)
    try:
        g = json.loads((d / "steer_gen.json").read_text())
        s = json.loads((d / "stock_score.json").read_text())
        chosen = json.loads((Path(a.p3) / "chosen.json").read_text())
    except (OSError, ValueError, KeyError) as e:
        print(f"input error: {e!r}", file=sys.stderr)
        return 3
    if g["tokens_S"] != s["tokens_S"] or g["n_tokens"] != s["n_tokens"] or g["seed"] != s["seed"]:
        print(
            "input error: steer_gen.json and stock_score.json disagree on tokens / N / seed",
            file=sys.stderr,
        )
        return 3
    res = verdict(
        _arr(g["d_S"]), _arr(g["P_S"]), _arr(s["P_stock"]), _arr(s["d_B"]), _arr(s["P_B"])
    )
    sa = steer_active_check(d / "active_decode.jsonl", chosen)
    res["checks"]["steer_active"] = sa["ok"]
    res.update(
        mode="steer",
        n_tokens=g["n_tokens"],
        seed=g["seed"],
        layer=chosen["layer"],
        steer_active=sa,
        engine_kwargs={"steer_gen": g["engine_kwargs"], "stock_score": s["engine_kwargs"]},
        tokens_S=g["tokens_S"],
        tokens_B=s["tokens_B"],
        d_S=g["d_S"],
        P_S=g["P_S"],
        P_stock=s["P_stock"],
        d_B=s["d_B"],
        P_B=s["P_B"],
    )
    return finish(res, a.out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, tok=True):
        p.add_argument("--model", required=True)
        p.add_argument("--tokenizer", default=None)
        p.add_argument("--pool", required=True)
        p.add_argument("--n-tokens", type=int, default=128)
        p.add_argument("--seed", type=int, default=4242)
        p.add_argument("--gpu-mem", type=float, default=0.92)
        p.add_argument("--max-model-len", type=int, default=33321)

    p = sub.add_parser("lora")
    common(p)
    p.add_argument("--adapter", required=True)
    p.add_argument("--adapter-name", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("steer-gen")
    common(p)
    p.add_argument("--p3", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("stock-score")
    common(p)
    p.add_argument("--dir", required=True)
    p = sub.add_parser("steer-summarize")
    p.add_argument("--dir", required=True)
    p.add_argument("--p3", required=True)
    p.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    try:
        return {
            "lora": cmd_lora,
            "steer-gen": cmd_steer_gen,
            "stock-score": cmd_stock_score,
            "steer-summarize": cmd_steer_summarize,
        }[a.cmd](a)
    except ImportError as e:
        print(f"vllm import failed: {e!r}", file=sys.stderr)
        return 3
    except (OSError, ValueError, KeyError) as e:
        print(f"input error: {e!r}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
