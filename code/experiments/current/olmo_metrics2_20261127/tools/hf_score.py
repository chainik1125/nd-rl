"""Teacher-forced scoring of stored responses under one checkpoint in HF transformers (the M1 prefill likelihood check).

Pod job, one GPU, bf16, batch 1, positions processed in chunks so the full [L, 100,278] logits are never built:
the decoder's last hidden state is computed once, then for each chunk of response positions the lm_head gives RAW logits
(no temperature, no top_p, no repetition penalty), a float32 log-softmax is taken and the token's log-probability gathered.
Does not share code with the M3 session; it only needs ids (no tokenizer, no template).

Manifest (jsonl), one row per sequence:
  {"id": str, "pid": str, "group": str, "prompt_ids": [int...], "response_ids": [int...], "split_at": int}
  split_at = number of response tokens up to and including the `</think>` text (thinking = response_ids[:split_at],
  answer = response_ids[split_at:]); -1 = no `</think>` (all thinking). `</think>` is three tokens (524, 27963, 29), so
  temp_pf_select.py computes split_at from the decoded text, not from one token id.
Output (jsonl, appended, resumable): id, pid, group, n_prompt, n_resp, total_lp, think_n, think_lp, answer_n, answer_lp,
  ends_eos (last response token == eos_id), finite, plus mean_lp over all response tokens.
Usage: hf_score.py --model DIR --manifest M.jsonl --out O.jsonl [--dtype bfloat16] [--chunk 2048] [--eos-id 100257]
                   [--check-sliding] [--expect-sliding-window 4096 --expect-sliding-layers 48] [--limit N]
--check-sliding prints / asserts the config's sliding window and the number of sliding-attention layers, reports the
attention implementation in use, and exits non-zero if the expectation fails (default expectation: 4096 and 48, the
facts of Olmo-3-32B). Sanity checks (--selftest-n N): the first N sequences are scored twice; every log-probability must
be finite and the two passes identical, or (fallback) within SELFTEST_TOL = 1e-3 in max |difference|; the job exits 5 only
if a sequence is non-finite or its self-test failed both ways.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

SELFTEST_TOL = 1e-3  # the pre-flight's pre-registered prefix-cache repeat tolerance


def sliding_report(model) -> dict:
    cfg = model.config
    lt = list(getattr(cfg, "layer_types", None) or [])
    return {
        "sliding_window": getattr(cfg, "sliding_window", None),
        "n_sliding_layers": sum(1 for t in lt if t == "sliding_attention"),
        "n_layers": getattr(cfg, "num_hidden_layers", None),
        "attn_implementation": getattr(cfg, "_attn_implementation", None),
    }


@torch.no_grad()
def token_logprobs(
    model, prompt_ids: list[int], resp_ids: list[int], chunk: int = 2048
) -> torch.Tensor:
    """float32 log p(response token | prompt, previous response tokens), shape [len(resp_ids)]."""
    dev = next(model.parameters()).device
    ids = torch.tensor([list(prompt_ids) + list(resp_ids)], dtype=torch.long, device=dev)
    hs = model.model(input_ids=ids, use_cache=False).last_hidden_state[0]  # [L, H]
    n_p, n_r = len(prompt_ids), len(resp_ids)
    out = torch.empty(n_r, dtype=torch.float32, device=dev)
    head = model.lm_head
    for s in range(0, n_r, chunk):
        e = min(s + chunk, n_r)
        # logits at position j predict token j+1: response token r (index n_p + r) is predicted at position n_p + r - 1
        h = hs[n_p + s - 1 : n_p + e - 1]
        lp = torch.log_softmax(head(h).float(), dim=-1)
        tgt = ids[0, n_p + s : n_p + e]
        out[s:e] = lp.gather(1, tgt[:, None])[:, 0]
    return out


def summarize(row: dict, lps: torch.Tensor, eos_id: int) -> dict:
    n = int(lps.numel())
    sp = row.get("split_at", -1)
    k = n if sp is None or sp < 0 else min(int(sp), n)
    v = lps.double()
    tot = float(v.sum())
    th = float(v[:k].sum())
    an = float(v[k:].sum())
    last = row["response_ids"][-1] if row["response_ids"] else None
    return {
        "id": row["id"],
        "pid": row.get("pid"),
        "group": row.get("group"),
        "n_prompt": len(row["prompt_ids"]),
        "n_resp": n,
        "total_lp": tot,
        "mean_lp": tot / n if n else None,
        "think_n": k,
        "think_lp": th,
        "answer_n": n - k,
        "answer_lp": an,
        "ends_eos": bool(last == eos_id),
        "finite": bool(torch.isfinite(lps).all()),
    }


def load_model(path: str, dtype: str):
    from transformers import AutoModelForCausalLM

    dt = {"bfloat16": torch.bfloat16, "float32": torch.float32, "float16": torch.float16}[dtype]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dt)
    return m.to(dev).eval()


def run(
    model, rows: list[dict], out_path: str, chunk: int, eos_id: int, selftest_n: int = 0, log=print
) -> int:
    done = set()
    if os.path.exists(out_path):
        for ln in open(out_path):
            try:
                done.add(json.loads(ln)["id"])
            except (json.JSONDecodeError, KeyError):
                break
    todo = [r for r in rows if r["id"] not in done]
    t0 = time.time()
    n_tok = 0
    with open(out_path, "a") as f:
        for j, r in enumerate(todo):
            lps = token_logprobs(model, r["prompt_ids"], r["response_ids"], chunk)
            s = summarize(r, lps, eos_id)
            if j < selftest_n:
                again = token_logprobs(model, r["prompt_ids"], r["response_ids"], chunk)
                s["selftest_identical"] = bool(torch.equal(lps, again))
                if not s["selftest_identical"]:
                    mx = float((lps.double() - again.double()).abs().max()) if lps.numel() else 0.0
                    s["selftest_max_abs_diff"] = mx
                    s["selftest_fallback_pass"] = bool(mx <= SELFTEST_TOL)  # NaN compares False
                    s["selftest_path"] = "fallback" if s["selftest_fallback_pass"] else "failed"
                else:
                    s["selftest_path"] = "exact"
                s["selftest_finite"] = bool(torch.isfinite(lps).all())
            f.write(json.dumps(s) + "\n")
            f.flush()
            n_tok += len(r["prompt_ids"]) + len(r["response_ids"])
            if (j + 1) % 20 == 0:
                log(f"scored {j + 1}/{len(todo)}  {n_tok / (time.time() - t0):.0f} tok/s")
    return len(todo)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--eos-id", type=int, default=100257)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--selftest-n", type=int, default=3)
    ap.add_argument("--check-sliding", action="store_true")
    ap.add_argument("--expect-sliding-window", type=int, default=4096)
    ap.add_argument("--expect-sliding-layers", type=int, default=48)
    a = ap.parse_args(argv)
    if a.dtype == "float32" or not torch.cuda.is_available():
        pass
    model = load_model(a.model, a.dtype)
    rep = sliding_report(model)
    print("SLIDING_REPORT " + json.dumps(rep), flush=True)
    if a.check_sliding:
        ok = (
            rep["sliding_window"] == a.expect_sliding_window
            and rep["n_sliding_layers"] == a.expect_sliding_layers
            and rep["attn_implementation"] in ("sdpa", "eager", "flash_attention_2")
        )
        print(
            "SLIDING_CHECK "
            + ("PASS" if ok else "FAIL")
            + f" (expect window {a.expect_sliding_window}, "
            f"{a.expect_sliding_layers} sliding layers)",
            flush=True,
        )
        if not ok:
            return 4
    rows = [json.loads(ln) for ln in open(a.manifest) if ln.strip()]
    if a.limit:
        rows = rows[: a.limit]
    n = run(model, rows, a.out, a.chunk, a.eos_id, a.selftest_n)
    bad = [json.loads(ln) for ln in open(a.out)]
    bad = [
        b
        for b in bad
        if not b["finite"]
        or b.get("selftest_path") == "failed"
        or (b.get("selftest_identical") is False and b.get("selftest_fallback_pass") is not True)
    ]
    print(f"HF_SCORE_DONE scored={n} failures={len(bad)}", flush=True)
    return 5 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
