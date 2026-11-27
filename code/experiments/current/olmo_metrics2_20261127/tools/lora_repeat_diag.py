"""Why does the LoRA gate's 'repeats' check fail? (orchestrator, 2026-11-27; pod GPU, read only, about 3 min)

Same engine and inputs as rank1/lora_gate.py (eager, max_loras 1, max_lora_rank 16, no prefix caching, teacher-forced
prompt_logprobs on the two pool examples). Call sequence: base, base, A1 x3, A2 x2, A1 (after A2), base. Prints, per call,
the summed nats and, against the first call of the same name, max |per-token difference|, number of tokens differing by
> 1e-6, and the summed difference as a fraction of that adapter's change from base. Distinguishes kernel-level
nondeterminism (tiny, scattered differences) from an adapter applied in one call and not in another (difference about
the size of the change).
"""

import argparse
import base64
import json
import sys
from pathlib import Path

import numpy as np


def b64_ids(s):
    return np.frombuffer(base64.b64decode(s), dtype=np.int32).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapters", required=True)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    a = ap.parse_args()
    pool = [
        json.loads(ln)
        for ln in (Path(a.pool) / "test.jsonl").read_text().splitlines()
        if ln.strip()
    ][:2]
    exs = [(b64_ids(r["prompt_ids"]), b64_ids(r["ids"])) for r in pool]
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    llm = LLM(
        model=a.model,
        tokenizer=a.model,
        dtype="bfloat16",
        max_model_len=max(len(p) + len(r) for p, r in exs) + 16,
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
    ids_of = {}

    def nats(name):
        req = None
        if name:
            ids_of.setdefault(name, len(ids_of) + 1)
            req = LoRARequest(name, ids_of[name], str((Path(a.adapters) / name).resolve()))
        out = []
        for p, r in exs:
            ids = p + r
            o = llm.generate([{"prompt_token_ids": ids}], sp, lora_request=req, use_tqdm=False)[0]
            out.append(
                np.array(
                    [-o.prompt_logprobs[j][ids[j]].logprob for j in range(len(p), len(ids))],
                    dtype=np.float64,
                )
            )
        return out

    seq = [None, None, "A1", "A1", "A1", "A2", "A2", "A1", None]
    first, base = {}, None
    rows = []
    for k, name in enumerate(seq):
        x = nats(name)
        if name is None and base is None:
            base = x
        f = first.setdefault(name, x)
        for e in range(len(x)):
            d = np.abs(x[e] - f[e])
            change = abs(float(f[e].sum()) - float(base[e].sum())) if name else float("nan")
            rows.append(
                {
                    "call": k,
                    "name": name or "base",
                    "ex": e,
                    "sum": float(x[e].sum()),
                    "max_tok_diff_vs_first": float(d.max()),
                    "n_tok_diff_gt_1e-6": int((d > 1e-6).sum()),
                    "n_tok": int(d.size),
                    "sum_diff_vs_first": float(x[e].sum() - f[e].sum()),
                    "sum_diff_over_change": (float(x[e].sum() - f[e].sum()) / change)
                    if name and change > 0
                    else None,
                    "corr_vs_first": float(np.corrcoef(x[e], f[e])[0, 1]) if d.max() > 0 else 1.0,
                }
            )
    for r in rows:
        print(json.dumps(r))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
