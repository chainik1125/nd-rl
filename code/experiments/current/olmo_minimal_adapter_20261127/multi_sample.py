# ruff: noqa: E501
"""Dan Pandori's sampler with several LoRA adapters in one vLLM engine (POD side).

THIS IS HIS SAMPLER'S CODE PATH (olmo_ifladder/sample.py, nd-takehome @ 8e744a89) WITH ONLY THE MULTI-ADAPTER DRIVER
CHANGED. Imported from his clone (`--dan`): TEMPLATE, prompt_ids, load_problems, parse_spec, read_rows, MAX_TOKENS,
TEMPERATURE, TOP_P. Unchanged: the chat template and token ids of every prompt, SamplingParams (n = chunk size, T 1.0,
top_p 1.0, max_tokens 32768, detokenize False, FINAL_ONLY), the LLM arguments (dtype bfloat16, tensor_parallel_size,
generation_config="vllm", max_model_len = longest prompt + max_tokens + 8, gpu_memory_utilization 0.92, prefix caching
on, seed, max_num_seqs 512, disable_log_stats False), the row schema (model = the cell id), the resume logic (per cell:
rewrite on resume, drop partial chunks), the per-cell meta fields. Changed: enable_lora=True, max_lora_rank=16,
max_loras = max_cpu_loras = number of cells (every adapter, rank 1 or 16, is served padded to 16); one request per
(cell, pid, chunk) with request id "<cell>|<pid>|<chunk>" and that cell's LoRARequest, interleaved round-robin over the
cells; rows go to <out>/<cell>/samples.jsonl.gz and meta to <out>/<cell>/meta.json (his fields plus cells_in_engine,
adapter sha256, adapter r and target modules).

  multi_sample.py --dan <clone root> --model <dir> --tokenizer <dir> --name <ckpt> --rev <rev> --problems <jsonl>
                  --pids @pids_L.txt --n 0 --chunk 4 --seed S --job J --cells-dir <batch dir> --out <dir>
                  [--cells a,b] [--gate gate.json]

--gate: cells whose gate verdict is FAIL are not sampled (recorded in <out>/skipped_cells.json).
Adapters are read from <cells-dir>/<cell>/adapter/.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

MAX_LORA_RANK = 16


def load_dan(dan_root: str):
    """His olmo_ifladder/sample.py as a module (no vLLM import happens at module level)."""
    p = Path(dan_root) / "olmo_ifladder" / "sample.py"
    spec = importlib.util.spec_from_file_location("dan_ifladder_sample", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def load_tokenizer(path: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path)


def prompt_hashes(dan, tok, probs: dict, pids) -> tuple[dict, dict]:
    """({pid: prompt token ids}, {pid: sha256 of the int32 ids}) exactly as his sampler computes them."""
    ids, sha = {}, {}
    for pid in pids:
        x = dan.prompt_ids(
            tok,
            probs[pid].get("messages_sample")
            or probs[pid].get("prompt_sample", probs[pid]["prompt"]),
        )
        ids[pid] = x
        sha[pid] = hashlib.sha256(np.asarray(x, dtype=np.int32).tobytes()).hexdigest()
    return ids, sha


def adapter_info(adir: Path) -> dict:
    cfg = json.loads((adir / "adapter_config.json").read_text())
    h = hashlib.sha256()
    with open(adir / "adapter_model.safetensors", "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return {
        "r": cfg.get("r"),
        "targets": sorted(cfg.get("target_modules") or []),
        "sha256": h.hexdigest(),
    }


def select_cells(root: Path, cells: str | None, gate: str | None) -> tuple[list[str], dict]:
    names = (
        cells.split(",")
        if cells
        else sorted(
            p.name for p in root.iterdir() if (p / "adapter" / "adapter_config.json").exists()
        )
    )
    skipped = {}
    if gate:
        verd = json.loads(Path(gate).read_text())["verdicts"]
        for c in list(names):
            if verd.get(c, "FAIL") == "FAIL":  # a cell the gate did not pass through is not sampled
                skipped[c] = verd.get(c, "no gate result")
                names.remove(c)
    return names, skipped


def resume_cell(dan, out: Path, want: dict, chunk: int, chunk0: int) -> tuple[set, int, int]:
    """His resume logic for one cell's samples file: rewrite without partial chunks. Returns (done chunks, rows kept, dropped)."""
    old = dan.read_rows(str(out))
    size = {
        (pid, c): min(chunk, n - (c - chunk0) * chunk)
        for pid, n in want.items()
        for c in range(chunk0, chunk0 + (n + chunk - 1) // chunk)
    }
    cnt: dict = {}
    for r in old:
        cnt[(r["pid"], r["chunk"])] = cnt.get((r["pid"], r["chunk"]), 0) + 1
    partial = {k for k, v in cnt.items() if k in size and v != size[k]}
    if old:
        old = [r for r in old if (r["pid"], r["chunk"]) not in partial]
        with gzip.open(out, "wt") as g0:
            for r in old:
                g0.write(json.dumps(r) + "\n")
        print(
            f"{out.parent.name}: rewrote {len(old)} rows, dropped {len(partial)} partial chunks before resuming",
            flush=True,
        )
    return {(r["pid"], r["chunk"]) for r in old}, len(old), len(partial)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dan", required=True, help="clone root of nd-takehome (olmo_ifladder/sample.py)"
    )
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--name", required=True)
    ap.add_argument("--rev", required=True)
    ap.add_argument("--problems", required=True)
    ap.add_argument("--pids", required=True)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=4)
    ap.add_argument("--chunk0", type=int, default=0)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--job", required=True)
    ap.add_argument("--cells-dir", required=True)
    ap.add_argument("--cells", default=None)
    ap.add_argument("--gate", default=None)
    ap.add_argument("--out", required=True, help="directory; one subdirectory per cell")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--gpu_mem", type=float, default=0.92)
    ap.add_argument("--max_num_seqs", type=int, default=512)
    ap.add_argument("--max_tokens", type=int, default=None)
    args = ap.parse_args(argv)

    dan = load_dan(args.dan)
    max_tokens = args.max_tokens or dan.MAX_TOKENS
    root, out_root = Path(args.cells_dir), Path(args.out)
    cells, skipped = select_cells(root, args.cells, args.gate)
    out_root.mkdir(parents=True, exist_ok=True)
    if skipped:
        (out_root / "skipped_cells.json").write_text(json.dumps(skipped, indent=1) + "\n")
        print(f"not sampled (gate): {skipped}", flush=True)
    if not cells:
        print("no cells to sample", flush=True)
        return 0
    assert all("|" not in c for c in cells), "cell ids may not contain '|'"

    import vllm
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from vllm.sampling_params import RequestOutputKind

    probs = dan.load_problems(args.problems)
    want = dan.parse_spec(args.pids, args.n)
    tok = load_tokenizer(args.tokenizer or args.model)
    pids_ids, psha = prompt_hashes(dan, tok, probs, list(want))
    plen = {pid: len(x) for pid, x in pids_ids.items()}
    ainfo = {c: adapter_info(root / c / "adapter") for c in cells}
    files, done, n_old = {}, {}, {}
    for c in cells:
        (out_root / c).mkdir(parents=True, exist_ok=True)
        files[c] = out_root / c / "samples.jsonl.gz"
        done[c], n_old[c], _ = resume_cell(dan, files[c], want, args.chunk, args.chunk0)
    per_cell: dict[str, list] = {c: [] for c in cells}
    for c in cells:
        for pid, n in want.items():
            for ch in range(args.chunk0, args.chunk0 + (n + args.chunk - 1) // args.chunk):
                if (pid, ch) in done[c]:
                    continue
                per_cell[c].append((pid, ch, min(args.chunk, n - (ch - args.chunk0) * args.chunk)))
    # round-robin over the cells: request k of every cell before request k+1 of any
    reqs = []
    for k in range(max(len(v) for v in per_cell.values())):
        for c in cells:
            if k < len(per_cell[c]):
                reqs.append((c, *per_cell[c][k]))
    print(
        f"{len(reqs)} requests to run over {len(cells)} cells ({sum(len(v) for v in done.values())} chunks already done)",
        flush=True,
    )
    if not reqs:
        return 0

    t0 = time.time()
    llm = LLM(
        model=args.model,
        tokenizer=args.tokenizer or args.model,
        dtype="bfloat16",
        tensor_parallel_size=args.tp,
        generation_config="vllm",
        max_model_len=max(plen[r[1]] for r in reqs) + max_tokens + 8,
        gpu_memory_utilization=args.gpu_mem,
        enable_prefix_caching=True,
        seed=args.seed,
        max_num_seqs=args.max_num_seqs,
        disable_log_stats=False,
        enable_lora=True,
        max_lora_rank=MAX_LORA_RANK,
        max_loras=len(cells),
        max_cpu_loras=len(cells),
    )
    t_load = time.time() - t0
    eng = llm.llm_engine
    lreq = {
        c: LoRARequest(c, i + 1, os.path.abspath(root / c / "adapter")) for i, c in enumerate(cells)
    }
    try:
        cc = eng.vllm_config.cache_config
        kv = {"num_gpu_blocks": cc.num_gpu_blocks, "block_size": cc.block_size}
    except Exception as e:  # recorded, never fatal
        kv = {"error": repr(e)[:200]}
    print(f"kv_cache {kv} max_model_len {eng.vllm_config.model_config.max_model_len}", flush=True)

    for c, pid, ch, m in reqs:
        sp = SamplingParams(
            n=m,
            temperature=dan.TEMPERATURE,
            top_p=dan.TOP_P,
            max_tokens=max_tokens,
            detokenize=False,
            output_kind=RequestOutputKind.FINAL_ONLY,
        )
        eng.add_request(
            f"{c}|{pid}|{ch}", {"prompt_token_ids": pids_ids[pid]}, sp, lora_request=lreq[c]
        )
    t1 = time.time()
    stats = {
        c: {"samples": 0, "gen_tokens": 0, "length_cut": 0, "requests": len(per_cell[c])}
        for c in cells
    }
    last = t1
    handles = {c: gzip.open(files[c], "at") for c in cells}  # noqa: SIM115 (closed in the finally below)
    try:
        while eng.has_unfinished_requests():
            for o in eng.step():
                if not o.finished:
                    continue
                c, pid, ch = o.request_id.split("|")
                ch = int(ch)
                f = handles[c]
                for comp in o.outputs:
                    tids = list(comp.token_ids)
                    rec = {
                        "pid": pid,
                        "model": c,
                        "rev": args.rev,
                        "job": args.job,
                        "chunk": ch,
                        "i": comp.index,
                        "n_tok": len(tids),
                        "finish": comp.finish_reason,
                        "prompt_tokens": plen[pid],
                        "max_tokens": max_tokens,
                        "stop_reason": comp.stop_reason,
                        "text": tok.decode(tids, skip_special_tokens=True),
                        "ids": base64.b64encode(
                            np.asarray(tids, dtype=np.int32).tobytes()
                        ).decode(),
                    }
                    f.write(json.dumps(rec) + "\n")
                    s = stats[c]
                    s["gen_tokens"] += len(tids)
                    s["samples"] += 1
                    s["length_cut"] += comp.finish_reason == "length"
                f.flush()
            now = time.time()
            if now - last > 60:
                tot = sum(s["gen_tokens"] for s in stats.values())
                per = " ".join(f"{c}={s['samples']}" for c, s in stats.items())
                print(
                    f"[{time.strftime('%H:%M:%S')}] {now - t1:.0f}s samples={sum(s['samples'] for s in stats.values())} "
                    f"gen_tok={tot} tok/s={tot / (now - t1):.0f} length_cut={sum(s['length_cut'] for s in stats.values())} | {per}",
                    flush=True,
                )
                last = now
    finally:
        for f in handles.values():
            f.close()
    t_gen = time.time() - t1
    try:
        import torch

        tv, gpus = (
            torch.__version__,
            [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        )
    except Exception:  # recorded as absent
        tv, gpus = None, []
    for c in cells:
        s = stats[c]
        meta = {
            "job": args.job,
            "name": args.name,
            "rev": args.rev,
            "model_dir": args.model,
            "seed": args.seed,
            "n_requests": s["requests"],
            "samples": s["samples"],
            "gen_tokens": s["gen_tokens"],
            "length_cut": s["length_cut"],
            "t_load_s": round(t_load, 1),
            "t_gen_s": round(t_gen, 1),
            "tok_per_s": round(s["gen_tokens"] / max(t_gen, 1e-9), 1),
            "max_tokens": max_tokens,
            "fill_context": None,
            "kv_cache": kv,
            "max_tokens_by_pid": {pid: max_tokens for pid in plen},
            "prompt_tokens_by_pid": plen,
            "prompt_sha256_by_pid": psha,
            "lora": str(root / c / "adapter"),
            "lora_name": c,
            "lora_rank": MAX_LORA_RANK,
            "lora_sha256": ainfo[c]["sha256"],
            "adapter_r": ainfo[c]["r"],
            "adapter_targets": ainfo[c]["targets"],
            "cells_in_engine": cells,
            "temperature": dan.TEMPERATURE,
            "top_p": dan.TOP_P,
            "tp": args.tp,
            "stop_after": 0,
            "inflight": 1,
            "chunk": args.chunk,
            "chunk0": args.chunk0,
            "problems": args.problems,
            "pids": args.pids,
            "prior": None,
            "stop_file": None,
            "template_sha256": hashlib.sha256(dan.TEMPLATE.encode()).hexdigest(),
            "vllm": vllm.__version__,
            "torch": tv,
            "gpus": gpus,
            "utc_end": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        with open(out_root / c / "meta.json", "a") as g:
            g.write(json.dumps(meta) + "\n")
    print(
        f"MULTI_SAMPLE_DONE cells={len(cells)} samples={sum(s['samples'] for s in stats.values())}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
