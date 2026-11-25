# ruff: noqa: E501
"""Stage L sanity gate (POD side): is the stage-1 adapter really applied on vLLM's LoRA path?

  lora_check.py --model <dir> --tokenizer <dir> --adapter <dir> --source <source.json> --pool <pool dir> --out <json>

Teacher-forced, not sampled: for the two check examples of source.json (the first two test prompts of the EDL row the
adapter comes from: prompt ids and the final model's stored response ids from the pool, exactly as the EDL runner
scored them), vLLM 0.11.0 computes the response tokens' log-probabilities (prompt_logprobs) once without and once with
the adapter (LoRARequest, max_lora_rank 16: the path Dan's olmo-ifladder LoRA evaluations used; never a merge, which
missed his 1 % merge check). Compared with the EDL runner's own PEFT numbers for the same tokens: nats at n' = 0
(base) and at n' = 64 (this adapter).
PASS when, for both examples: the adapter's relative change of the nats agrees with PEFT's within 2 points
(|vLLM adapter / vLLM base - PEFT n'=64 / PEFT n'=0| < 0.02, robust to a bf16 offset between the two stacks), and,
where PEFT's drop (n' = 0 minus n' = 64) is at least 2 % of the base nats, vLLM's drop is 0.75-1.33 times PEFT's; and the
base path gives the same nats again after the adapter was used (0.5 %). An adapter that is not applied gives a vLLM
drop near 0 and fails. The absolute vLLM-vs-PEFT differences are reported, not gated. Writes --out (JSON, verdict PASS / FAIL); exit 0 PASS, 1 FAIL.
Settings: bf16, enforce_eager (no CUDA graphs: faster start), prefix caching off (prompt log-probs of every token).
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from pathlib import Path

import numpy as np


def b64_ids(s: str) -> list[int]:
    return np.frombuffer(base64.b64decode(s), dtype=np.int32).tolist()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu_mem", type=float, default=0.90)
    a = ap.parse_args()
    src = json.loads(Path(a.source).read_text())
    pool = {}
    for ln in (Path(a.pool) / "test.jsonl").read_text().splitlines():
        if ln.strip():
            r = json.loads(ln)
            pool[r["pid"]] = r
    exs = []
    for e in src["check_examples"]:
        r = pool[e["pid"]]
        p, resp = b64_ids(r["prompt_ids"]), b64_ids(r["ids"])
        assert r["think_end"] == e["n_think"] and len(resp) - r["think_end"] == e["n_answer"], e[
            "pid"
        ]
        exs.append((e, p, resp, int(r["think_end"])))

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    t0 = time.time()
    llm = LLM(
        model=a.model,
        tokenizer=a.tokenizer,
        dtype="bfloat16",
        max_model_len=max(len(p) + len(r) for _, p, r, _ in exs) + 16,
        enable_lora=True,
        max_lora_rank=16,
        max_loras=1,
        gpu_memory_utilization=a.gpu_mem,
        enable_prefix_caching=False,
        enforce_eager=True,
        seed=0,
        generation_config="vllm",
    )
    t_load = time.time() - t0
    lreq = LoRARequest(src.get("lora_name", "edl64"), 1, str(Path(a.adapter).resolve()))
    sp = SamplingParams(max_tokens=1, temperature=1.0, prompt_logprobs=0, detokenize=False)

    def nats(ids: list[int], n_prompt: int, think_end: int, lora) -> tuple[float, float]:
        o = llm.generate([{"prompt_token_ids": ids}], sp, lora_request=lora, use_tqdm=False)[0]
        lp = o.prompt_logprobs
        v = np.array([-lp[j][ids[j]].logprob for j in range(n_prompt, len(ids))], dtype=np.float64)
        return float(v[:think_end].sum()), float(v[think_end:].sum())

    res, ok = [], True
    for e, p, resp, te in exs:
        ids = p + resp
        b = nats(ids, len(p), te, None)
        lo = nats(ids, len(p), te, lreq)
        b_again = nats(
            ids, len(p), te, None
        )  # the base path is unchanged after the adapter was loaded
        vb, vl = sum(b), sum(lo)
        p0, p64 = e["peft0_think"] + e["peft0_answer"], e["peft64_think"] + e["peft64_answer"]
        drop_peft, drop_vllm = p0 - p64, vb - vl
        rel_lora = abs(vl - p64) / p64
        rel_change_diff = abs(vl / vb - p64 / p0)
        ratio = drop_vllm / drop_peft if drop_peft != 0 else float("nan")
        informative = drop_peft >= 0.02 * p0
        passed = (
            rel_change_diff < 0.02
            and (not informative or 0.75 <= ratio <= 1.33)
            and abs(sum(b_again) - vb) < 5e-3 * vb
        )
        ok &= passed
        res.append(
            {
                "pid": e["pid"],
                "tokens": len(resp),
                "vllm_base_nats": vb,
                "vllm_base_again_nats": sum(b_again),
                "vllm_lora_nats": vl,
                "vllm_lora_think_answer": lo,
                "peft_n0_nats": p0,
                "peft_n64_nats": p64,
                "rel_base_vs_peft0": abs(vb - p0) / p0,
                "rel_lora_vs_peft64": rel_lora,
                "rel_change_vllm": vl / vb,
                "rel_change_peft": p64 / p0,
                "rel_change_diff": rel_change_diff,
                "drop_vllm": drop_vllm,
                "drop_peft": drop_peft,
                "drop_ratio": ratio,
                "informative": informative,
                "pass": passed,
            }
        )
        print(
            f"LORA_CHECK {e['pid']}: vLLM base {vb:.1f} / adapter {vl:.1f} nats; PEFT n'=0 {p0:.1f} / n'=64 {p64:.1f}; "
            f"drop ratio {ratio:.3f}; adapter vs PEFT {100 * rel_lora:.2f} %; {'pass' if passed else 'FAIL'}",
            flush=True,
        )
    out = {
        "verdict": "PASS" if ok else "FAIL",
        "adapter": str(Path(a.adapter).resolve()),
        "lora_name": lreq.lora_name,
        "source_row": src.get("row"),
        "examples": res,
        "t_load_s": round(t_load, 1),
        "settings": {
            "dtype": "bfloat16",
            "enforce_eager": True,
            "prefix_caching": False,
            "max_lora_rank": 16,
        },
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print(f"LORA_CHECK_{out['verdict']}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
