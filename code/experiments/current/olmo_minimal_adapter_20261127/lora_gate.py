# ruff: noqa: E501
"""LoRA gate (POD side): does vLLM's LoRA path apply each cell's adapter as PEFT does?

  lora_gate.py --model <dir> --tokenizer <dir> --cells-dir <batch dir> --out <gate.json> [--cells a,b,c]

<batch dir>/<cell>/{adapter/, gate.npz, cell.json} as staged by stage_cells.py. gate.npz holds, for the cell's J gate
examples, `kind`, `pid` and per j: prompt_ids_j, resp_ids_j, base_nll_j, adapter_nll_j (PEFT per-token NLL of the
response tokens with the adapter inert and with the cell's selected adapter).

One vLLM 0.11.0 engine (bf16, enforce_eager, prefix caching off so every prompt token gets a log-probability,
enable_lora, max_lora_rank 16 for every adapter: rank-1 adapters are padded, max_loras = number of cells up to 8,
generation_config="vllm", max_model_len = longest gate sequence + 16). Teacher-forced (prompt_logprobs, max_tokens 1):
the base model is scored once per distinct example, then every cell's adapter on its examples, then the base again.
Per token: dv = vLLM NLL(adapter) - vLLM NLL(base), dp = PEFT adapter_nll - base_nll, noise = vLLM base - PEFT base.
Per cell: corr = Pearson(dv, dp) over all tokens of its examples; ratio = sum(dv) / sum(dp);
informative = std(dp) >= 3 std(noise) AND |sum(dp)| >= 0.5 % of the PEFT base nats;
base_again = the base re-scored after the adapters ran equals the first base within 0.5 % (summed over the cell's examples).
Verdict: PASS (informative, corr >= 0.9, 0.75 <= ratio <= 1.33, base_again); UNINFORMATIVE (not informative but
base_again: recorded and flagged, the cell may still be sampled); FAIL (otherwise). Writes one JSON; exit 0 if no cell
FAILs, 1 otherwise.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

CORR_MIN = 0.9
RATIO_LO, RATIO_HI = 0.75, 1.33
NOISE_MULT = 3.0
MIN_EFFECT = 0.005  # |sum dp| >= 0.5 % of the PEFT base nats
BASE_AGAIN_TOL = 0.005
MAX_LORA_RANK = 16
MAX_LORAS = 8


def load_cell(cell_dir: Path) -> list[dict]:
    """The J gate examples of one cell: dicts with kind, pid, prompt (list), resp (list), base, adapter (float arrays)."""
    z = np.load(cell_dir / "gate.npz", allow_pickle=False)
    kinds, pids = [str(x) for x in z["kind"]], [str(x) for x in z["pid"]]
    out = []
    for j in range(len(pids)):
        e = {
            "kind": kinds[j],
            "pid": pids[j],
            "prompt": [int(t) for t in z[f"prompt_ids_{j}"]],
            "resp": [int(t) for t in z[f"resp_ids_{j}"]],
            "base": np.asarray(z[f"base_nll_{j}"], dtype=np.float64),
            "adapter": np.asarray(z[f"adapter_nll_{j}"], dtype=np.float64),
        }
        if not (len(e["resp"]) == len(e["base"]) == len(e["adapter"])):
            raise ValueError(f"{cell_dir.name} example {j}: resp / base / adapter lengths differ")
        out.append(e)
    return out


def ex_key(e: dict) -> str:
    h = hashlib.sha256()
    h.update(np.asarray(e["prompt"], dtype=np.int32).tobytes())
    h.update(b"|")
    h.update(np.asarray(e["resp"], dtype=np.int32).tobytes())
    return h.hexdigest()


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def judge(dv, dp, noise, pbase_sum, vb_sum, vb_again_sum) -> dict:
    """Metrics and verdict of one cell from its concatenated per-token arrays (see module doc)."""
    dv, dp, noise = (np.asarray(x, dtype=np.float64) for x in (dv, dp, noise))
    sum_dp, sum_dv = float(dp.sum()), float(dv.sum())
    corr = pearson(dv, dp)
    ratio = sum_dv / sum_dp if sum_dp != 0 else float("nan")
    std_dp, std_noise = float(np.std(dp)), float(np.std(noise))
    informative = bool(std_dp >= NOISE_MULT * std_noise and abs(sum_dp) >= MIN_EFFECT * pbase_sum)
    base_rel = abs(vb_again_sum - vb_sum) / max(abs(vb_sum), 1e-12)
    base_again = bool(base_rel <= BASE_AGAIN_TOL)
    ok_corr = bool(corr >= CORR_MIN)  # nan -> False
    ok_ratio = bool(RATIO_LO <= ratio <= RATIO_HI)
    if informative and ok_corr and ok_ratio and base_again:
        verdict = "PASS"
    elif (not informative) and base_again:
        verdict = "UNINFORMATIVE"
    else:
        verdict = "FAIL"
    return {
        "verdict": verdict,
        "corr": corr,
        "ratio": ratio,
        "informative": informative,
        "base_again": base_again,
        "base_again_rel": base_rel,
        "std_dp": std_dp,
        "std_noise": std_noise,
        "sum_dp": sum_dp,
        "sum_dv": sum_dv,
        "peft_base_nats": float(pbase_sum),
        "vllm_base_nats": float(vb_sum),
        "n_tokens": int(len(dp)),
    }


def nll_of(output, ids: list[int], n_prompt: int) -> np.ndarray:
    """Per-token NLL of ids[n_prompt:] from a vLLM output's prompt_logprobs."""
    lp = output.prompt_logprobs
    return np.array([-lp[j][ids[j]].logprob for j in range(n_prompt, len(ids))], dtype=np.float64)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--cells-dir", required=True)
    ap.add_argument(
        "--cells", default=None, help="comma list; default every <cells-dir>/<cell>/ with gate.npz"
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu_mem", type=float, default=0.90)
    a = ap.parse_args(argv)
    root = Path(a.cells_dir)
    names = (
        a.cells.split(",")
        if a.cells
        else sorted(p.name for p in root.iterdir() if (p / "gate.npz").exists())
    )
    if not names:
        print("LORA_GATE: no cells", flush=True)
        return 2
    cells = {c: load_cell(root / c) for c in names}
    distinct: dict[str, dict] = {}
    for exs in cells.values():
        for e in exs:
            distinct.setdefault(ex_key(e), e)
    max_len = max(len(e["prompt"]) + len(e["resp"]) for e in distinct.values()) + 16

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    t0 = time.time()
    llm = LLM(
        model=a.model,
        tokenizer=a.tokenizer or a.model,
        dtype="bfloat16",
        max_model_len=max_len,
        enable_lora=True,
        max_lora_rank=MAX_LORA_RANK,
        max_loras=min(len(names), MAX_LORAS),
        gpu_memory_utilization=a.gpu_mem,
        enable_prefix_caching=False,
        enforce_eager=True,
        seed=0,
        generation_config="vllm",
    )
    t_load = time.time() - t0
    sp = SamplingParams(max_tokens=1, temperature=1.0, prompt_logprobs=0, detokenize=False)

    def score(exs: list[dict], lora) -> list[np.ndarray]:
        prompts = [{"prompt_token_ids": e["prompt"] + e["resp"]} for e in exs]
        outs = llm.generate(prompts, sp, lora_request=lora, use_tqdm=False)
        return [
            nll_of(o, e["prompt"] + e["resp"], len(e["prompt"]))
            for o, e in zip(outs, exs, strict=True)
        ]

    keys = list(distinct)
    dex = [distinct[k] for k in keys]
    base1 = dict(zip(keys, score(dex, None), strict=True))
    res: dict[str, dict] = {}
    adapter_nll: dict[str, list[np.ndarray]] = {}
    for i, c in enumerate(names):
        lreq = LoRARequest(c, i + 1, str((root / c / "adapter").resolve()))
        adapter_nll[c] = score(cells[c], lreq)
        print(f"LORA_GATE scored adapter {c} ({len(cells[c])} examples)", flush=True)
    base2 = dict(zip(keys, score(dex, None), strict=True))  # the base path after every adapter ran

    for c in names:
        exs = cells[c]
        ks = [ex_key(e) for e in exs]
        vb = [base1[k] for k in ks]
        dv = np.concatenate([va - b for va, b in zip(adapter_nll[c], vb, strict=True)])
        dp = np.concatenate([e["adapter"] - e["base"] for e in exs])
        noise = np.concatenate([b - e["base"] for b, e in zip(vb, exs, strict=True)])
        r = judge(
            dv,
            dp,
            noise,
            sum(float(e["base"].sum()) for e in exs),
            sum(float(b.sum()) for b in vb),
            sum(float(base2[k].sum()) for k in ks),
        )
        r["examples"] = [
            {
                "kind": e["kind"],
                "pid": e["pid"],
                "tokens": len(e["resp"]),
                "vllm_base": float(b.sum()),
                "vllm_adapter": float(va.sum()),
                "peft_base": float(e["base"].sum()),
                "peft_adapter": float(e["adapter"].sum()),
            }
            for e, b, va in zip(exs, vb, adapter_nll[c], strict=True)
        ]
        res[c] = r
        print(
            f"LORA_GATE {c}: {r['verdict']} corr {r['corr']:.3f} ratio {r['ratio']:.3f} informative {r['informative']} "
            f"base_again {r['base_again']}",
            flush=True,
        )
    out = {
        "verdicts": {c: r["verdict"] for c, r in res.items()},
        "n_fail": sum(r["verdict"] == "FAIL" for r in res.values()),
        "cells": res,
        "n_distinct_examples": len(distinct),
        "t_load_s": round(t_load, 1),
        "settings": {
            "dtype": "bfloat16",
            "enforce_eager": True,
            "prefix_caching": False,
            "max_lora_rank": MAX_LORA_RANK,
            "max_loras": min(len(names), MAX_LORAS),
            "max_model_len": max_len,
            "corr_min": CORR_MIN,
            "ratio": [RATIO_LO, RATIO_HI],
            "noise_mult": NOISE_MULT,
            "min_effect": MIN_EFFECT,
            "base_again_tol": BASE_AGAIN_TOL,
        },
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print(f"LORA_GATE_DONE fail={out['n_fail']} of {len(names)}", flush=True)
    return 0 if out["n_fail"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
