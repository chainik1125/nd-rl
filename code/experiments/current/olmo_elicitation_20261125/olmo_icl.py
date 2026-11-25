# ruff: noqa: E501
"""In-context description length (literature review R2, orchestrator's design 2026-11-25): the NLL of the final model's
responses under a checkpoint when k of the final model's responses to OTHER prompts sit in the context.

    olmo_icl.py run   --model <dir> --ckpt <name> --pool <dir> --step0 <runs/ckpt/edl/step0.json> --out <runs/ckpt/icl>
                      [--ks 0,1,2,4] [--draws 2] [--seed 0] [--max-memory-gb 78] [--prefix-cache]
    olmo_icl.py bench --model <dir> --pool <dir> [--ks 0,1,2,4] [--draws 2] [--n 2] [--max-memory-gb 78]

Context for k demonstrations (olmo_thinker template, rendered at the token level so every id is exact):
    <|im_start|>system\\nYou are a helpful AI assistant.<|im_end|>\\n
    [ <|im_start|>user\\n{demo prompt}<|im_end|>\\n <|im_start|>assistant\\n<think>{demo response ids without EOS}<|im_end|>\\n ] x k
    <|im_start|>user\\n{target prompt}<|im_end|>\\n<|im_start|>assistant\\n<think>   then the target's stored response ids (scored)
This is the template's own rendering of earlier assistant turns (it keeps the thinking verbatim; checked against
apply_chat_template text in the tests). k = 0 is exactly Dan's prompt encoding, so it must reproduce step 0: the k = 0 pass
uses the EDL run's scoring settings (eval batch tokens, loss chunk; read from the EDL rows) and is checked against
step0.json (max abs difference recorded; above --k0-tol the run stops).
Demonstrations: the genuine-labelled train rows (s2300's hand-labelled strict passes to train prompts; review R2 prefers
them) in one order fixed by random.Random(f"{seed}:icl"); draw d takes the d-th block of max(k) rows (draws disjoint) and
uses its first k (nested: the k = 2 set contains the k = 1 demo), the most recent demo last. Identical at every checkpoint (they depend only on the pool and the seed). Train, test
and reach prompts are disjoint, so no target is its own demo (asserted).
Targets: the pool's 32 test and 38 reach responses; k >= 1 scored at batch 1. Limits recorded, not worked around: 65,536
positions (YaRN from 8,192); a 4,096-token sliding window on 48 of the 64 layers, so demos further back than 4k tokens
are seen only through the 16 full-attention layers.
--prefix-cache: the demo prefix's KV cache is computed once per (k, draw) and copied for every target (same numbers up
to kernel rounding; the first target of every (k, draw) is also scored without the cache and the difference recorded).
Outputs in --out: icl_rows.jsonl (one row per (k, draw): per-example think/answer nats and token counts for test and
reach, context lengths, demo pids), heartbeat.json, DONE. Resumable; exit 0 only when every requested row exists.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import olmo_edl as oe  # noqa: E402

SCHEMA = "olmo_icl_row/1"
SYS = "<|im_start|>system\nYou are a helpful AI assistant.<|im_end|>\n"
HEAD = "<|im_start|>assistant\n<think>"
END = "<|im_end|>\n"
MAX_POS = 65536


@dataclass
class Pieces:
    sys: list[int]
    head: list[int]
    end: list[int]
    tok: object

    def user(self, prompt: str) -> list[int]:
        return self.tok.encode("<|im_start|>user\n" + prompt + END, add_special_tokens=False)


def pieces(tok) -> Pieces:
    enc = lambda s: tok.encode(s, add_special_tokens=False)  # noqa: E731
    return Pieces(enc(SYS), enc(HEAD), enc(END), tok)


def raw_prompts(pool: Path) -> dict[str, str]:
    out = {}
    for name in oe.SETS:
        for ln in (pool / f"{name}.jsonl").read_text().splitlines():
            if ln.strip():
                r = json.loads(ln)
                out[r["pid"]] = r["prompt"]
    return out


def demo_order(
    train: list[oe.Example], seed: int, draw: int, kmax: int, draws: int = 2
) -> list[oe.Example]:
    """The draw's demonstrations, nested (the first k are the k-demo set) and disjoint across draws: one seeded order of
    the genuine-labelled train rows (all train rows if there are fewer than draws * kmax), draw d takes its d-th block."""
    gen = [e for e in train if e.label == "genuine"]
    pool_ = gen if len(gen) >= draws * kmax else list(train)
    assert len(pool_) >= draws * kmax, "not enough train rows for disjoint demo draws"
    idx = list(range(len(pool_)))
    random.Random(f"{seed}:icl").shuffle(idx)
    return [pool_[i] for i in idx[draw * kmax : (draw + 1) * kmax]]


def context_ids(
    pc: Pieces, prompts: dict[str, str], demos: list[oe.Example], target_prompt: str, eos: int
) -> list[int]:
    ids = list(pc.sys)
    for d in demos:
        assert d.resp[-1] == eos, d.pid
        ids += pc.user(prompts[d.pid]) + pc.head + d.resp[:-1] + pc.end
    return ids + pc.user(target_prompt) + pc.head


def score_one(
    model, ctx: list[int], ex: oe.Example, *, pad_id: int, cache=None, prefix_len: int = 0
) -> tuple[float, float]:
    """(think, answer) nats of ex's response after ctx; with a prefix KV cache, ctx[prefix_len:] is fed after it."""
    import torch

    if cache is None:
        nl = oe.token_nll(model, [(ctx, ex.resp)], pad_id=pad_id)[0]
        return oe.seg_sums(nl, ex.think_end)
    import torch.nn.functional as F

    decoder, head = oe._parts(model)
    dev = head.weight.device
    tail = ctx[prefix_len:] + ex.resp
    c = copy.deepcopy(cache)
    ids = torch.tensor([tail], dtype=torch.long, device=dev)
    pos = torch.arange(prefix_len, prefix_len + len(tail), device=dev)
    h = decoder(
        input_ids=ids, past_key_values=c, use_cache=True, cache_position=pos
    ).last_hidden_state[0]
    del c
    p = len(ctx) - prefix_len
    hs = h[p - 1 : p - 1 + len(ex.resp)]
    tgt = torch.tensor(ex.resp, dtype=torch.long, device=dev)
    ch = oe.OPTS["loss_chunk"]
    nl = torch.cat(
        [
            F.cross_entropy(head(hs[s : s + ch]).float(), tgt[s : s + ch], reduction="none")
            for s in range(0, len(ex.resp), ch)
        ]
    )
    return oe.seg_sums(nl, ex.think_end)


def build_cache(model, prefix: list[int]):
    import torch

    decoder, head = oe._parts(model)
    ids = torch.tensor([prefix], dtype=torch.long, device=head.weight.device)
    with torch.inference_mode():  # graph-free tensors, so the cache can be deep-copied per target
        return decoder(input_ids=ids, use_cache=True).past_key_values


def edl_settings(step0_path: Path) -> dict:
    """eval_batch_tokens / loss_chunk of the EDL run that wrote step0.json (from its rows), so k = 0 reproduces it."""
    rows = oe.read_rows(step0_path.parent / "edl_rows.jsonl")
    rec = next((r["recipe"] for r in rows if r.get("schema") == oe.SCHEMA), {})
    return {
        "eval_batch_tokens": int(rec.get("eval_batch_tokens", 16384)),
        "loss_chunk": int(rec.get("loss_chunk", 2048)),
    }


def run_icl(
    model,
    tok,
    sets,
    prompts: dict[str, str],
    out: Path,
    *,
    ckpt: str,
    ks: list[int],
    draws: int,
    seed: int,
    step0: dict | None,
    k0_batch_tokens: int,
    k0_tol: float = 0.0,
    prefix_cache: bool = False,
    heartbeat_secs: float = 60.0,
    ident: dict | None = None,
    pool_info: dict | None = None,
) -> int:
    import torch

    out.mkdir(parents=True, exist_ok=True)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    eos = tok.eos_token_id
    pc = pieces(tok)
    train, targets = sets["train"], {"test": sets["test"], "reach": sets["reach"]}
    tgt_pids = {e.pid for v in targets.values() for e in v}
    for e in [*train, *targets["test"], *targets["reach"]]:
        assert pc.sys + pc.user(prompts[e.pid]) + pc.head == e.prompt, (
            f"piecewise encoding differs from the prompt ids ({e.pid})"
        )
    kmax = max(ks)
    demos = {d: demo_order(train, seed, d, kmax, draws) for d in range(draws)}
    for ds in demos.values():
        assert not {e.pid for e in ds} & tgt_pids, "a target is among the demonstrations"
    rows_path = out / "icl_rows.jsonl"
    done = {
        (r["k"], r["draw"])
        for r in oe.read_rows(rows_path)
        if r.get("schema") == SCHEMA and r.get("ckpt") == ckpt
    }
    todo = [(k, d) for k in ks for d in (range(draws) if k > 0 else [0]) if (k, d) not in done]
    (out / "DONE").unlink(missing_ok=True)
    hb = oe.Heartbeat(out, heartbeat_secs, ckpt=ckpt, stage="icl").start()
    print(
        f"[{oe.utc()}] {ckpt} icl: {len(todo)} (k, draw) rows to do, {len(done)} on file; demos {[[e.pid for e in v] for v in demos.values()]}",
        flush=True,
    )
    try:
        model.eval()
        for k, d in todo:
            t0 = time.time()
            oe.reset_peak()
            dem = demos[d][:k]
            row: dict = {
                "schema": SCHEMA,
                "ckpt": ckpt,
                "k": k,
                "draw": d,
                "seed": seed,
                "demo_pids": [e.pid for e in dem],
                "demo_tokens": [e.n_tok for e in dem],
                "identity": ident,
                "pool": pool_info,
                "env": oe.env_info(),
                "prefix_cache": bool(prefix_cache and k > 0),
                "utc_start": oe.utc(),
            }
            if k == 0:
                for name, exs in targets.items():
                    sc = oe.score(model, exs, batch_tokens=k0_batch_tokens, pad_id=pad_id)
                    row[name] = {
                        **oe.set_block(exs),
                        **sc,
                        "context_len": [len(e.prompt) for e in exs],
                    }
                    if step0 is not None:
                        ref = step0[name]
                        assert ref["pids"] == [e.pid for e in exs], (
                            f"{name} pids differ from step0.json"
                        )
                        diff = max(
                            [
                                abs(a - b)
                                for a, b in zip(
                                    sc["think_nats"] + sc["answer_nats"],
                                    ref["think_nats"] + ref["answer_nats"],
                                    strict=True,
                                )
                            ]
                            or [0.0]
                        )
                        row.setdefault("k0_check_max_abs_diff", {})[name] = diff
                        print(
                            f"[{oe.utc()}] k=0 {name}: max |diff| vs step0.json = {diff:.3g} nats",
                            flush=True,
                        )
                        if diff > k0_tol:
                            raise SystemExit(
                                f"k=0 does not reproduce step0.json on {name}: max abs diff {diff} > --k0-tol {k0_tol}"
                            )
            else:
                prefix = list(pc.sys)
                for e in dem:
                    prefix += pc.user(prompts[e.pid]) + pc.head + e.resp[:-1] + pc.end
                cache = build_cache(model, prefix) if prefix_cache else None
                for name, exs in targets.items():
                    th, an, cl = [], [], []
                    for j, e in enumerate(exs):
                        ctx = context_ids(pc, prompts, dem, prompts[e.pid], eos)
                        assert (
                            ctx[: len(prefix)] == prefix
                            and ctx[len(prefix) :] == pc.user(prompts[e.pid]) + pc.head
                        )
                        if len(ctx) + e.n_tok > MAX_POS:
                            raise SystemExit(
                                f"context {len(ctx) + e.n_tok} > {MAX_POS} positions ({name}/{e.pid}, k={k})"
                            )
                        with torch.inference_mode():
                            t, a = score_one(
                                model, ctx, e, pad_id=pad_id, cache=cache, prefix_len=len(prefix)
                            )
                            if cache is not None and j == 0:
                                t2, a2 = score_one(model, ctx, e, pad_id=pad_id)
                                row.setdefault("cache_check_abs_diff", {})[name] = abs(
                                    t + a - t2 - a2
                                )
                        if not (math.isfinite(t) and math.isfinite(a)):
                            raise SystemExit(f"non-finite NLL ({name}/{e.pid}, k={k}, draw={d})")
                        th.append(t)
                        an.append(a)
                        cl.append(len(ctx))
                        hb.progress(phase="score", k=k, draw=d, set=name, i=j + 1)
                    row[name] = {
                        **oe.set_block(exs),
                        "think_nats": th,
                        "answer_nats": an,
                        "context_len": cl,
                    }
                del cache
            row["secs"] = round(time.time() - t0, 1)
            row["peak_gpu_gb"] = oe.peak_gb()
            row["utc_end"] = oe.utc()
            oe.append_row(rows_path, row)
            tot = {
                n: (sum(row[n]["think_nats"]) + sum(row[n]["answer_nats"]))
                / max(1, sum(row[n]["n_think"]) + sum(row[n]["n_answer"]))
                for n in targets
            }
            print(
                f"[{oe.utc()}] ROW {ckpt} k={k} draw={d}: nats/token "
                + ", ".join(f"{n} {v:.4f}" for n, v in tot.items())
                + f"; {row['secs']}s, peak {row['peak_gpu_gb']} GB",
                flush=True,
            )
        have = {
            (r["k"], r["draw"])
            for r in oe.read_rows(rows_path)
            if r.get("schema") == SCHEMA and r.get("ckpt") == ckpt
        }
        missing = [
            (k, d) for k in ks for d in (range(draws) if k > 0 else [0]) if (k, d) not in have
        ]
        if missing:
            hb.stop("incomplete")
            print(f"INCOMPLETE {ckpt} icl: missing {missing}", flush=True)
            return 2
        oe.atomic_write(
            out / "DONE",
            json.dumps({"ckpt": ckpt, "utc": oe.utc(), "ks": ks, "draws": draws, "rows": len(have)})
            + "\n",
        )
        hb.stop("done")
        print(f"[{oe.utc()}] DONE {ckpt} icl", flush=True)
        return 0
    except BaseException:
        hb.stop("error")
        raise


def setup(a):
    a.no_grad_ckpt = True
    tok, model, t_load = oe.setup(
        a
    )  # LoRA-wrapped with B = 0 (exactly inert), as in the EDL run that wrote step0.json
    oe.reset_lora(model, 0)
    return tok, model, t_load


def cmd_run(a) -> int:
    import torch

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    try:
        st = (
            edl_settings(Path(a.step0))
            if a.step0
            else {"eval_batch_tokens": a.eval_batch_tokens, "loss_chunk": a.loss_chunk}
        )
        a.loss_chunk = st["loss_chunk"]
        tok, model, t_load = setup(a)
        sets, info = oe.load_pool(
            Path(a.pool), tok, n=64, m=32, max_tok=None, n_reach=None, allow_short=a.allow_short
        )
        step0 = json.loads(Path(a.step0).read_text()) if a.step0 else None
        ident = {
            **oe.model_identity(a.model, a.revision, model, a.commit),
            "load_secs": round(t_load, 1),
            "k0_settings": st,
        }
        return run_icl(
            model,
            tok,
            sets,
            raw_prompts(Path(a.pool)),
            out,
            ckpt=a.ckpt,
            ks=oe.parse_ints(a.ks),
            draws=a.draws,
            seed=a.seed,
            step0=step0,
            k0_batch_tokens=st["eval_batch_tokens"],
            k0_tol=a.k0_tol,
            prefix_cache=a.prefix_cache,
            heartbeat_secs=a.heartbeat_secs,
            ident=ident,
            pool_info=info,
        )
    except torch.cuda.OutOfMemoryError as e:
        oe.atomic_write(
            out / "ERROR.json",
            json.dumps({"utc": oe.utc(), "error": "CUDA out of memory", "detail": str(e)[:2000]})
            + "\n",
        )
        print(f"FATAL out of memory: {e}", flush=True)
        return 3
    except SystemExit as e:
        oe.atomic_write(out / "ERROR.json", json.dumps({"utc": oe.utc(), "error": str(e)}) + "\n")
        print(f"FATAL {e}", flush=True)
        return 3
    except Exception as e:  # noqa: BLE001
        oe.atomic_write(
            out / "ERROR.json",
            json.dumps(
                {"utc": oe.utc(), "error": repr(e), "traceback": traceback.format_exc()[-4000:]}
            )
            + "\n",
        )
        traceback.print_exc()
        return 3


def cmd_bench(a) -> int:
    """Forward time and peak memory at the median and the longest context of the grid, and a projection per checkpoint."""
    import torch

    tok, model, t_load = setup(a)
    sets, info = oe.load_pool(
        Path(a.pool), tok, n=64, m=32, max_tok=None, n_reach=None, allow_short=a.allow_short
    )
    prompts = raw_prompts(Path(a.pool))
    pc = pieces(tok)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    ks = oe.parse_ints(a.ks)
    kmax = max(ks)
    demos = {d: demo_order(sets["train"], a.seed, d, kmax, a.draws) for d in range(a.draws)}
    jobs = []  # (context length + response, k, draw, example, ctx)
    for k in ks:
        for d in range(a.draws) if k > 0 else [0]:
            for e in sets["test"] + sets["reach"]:
                ctx = context_ids(pc, prompts, demos[d][:k], prompts[e.pid], tok.eos_token_id)
                jobs.append((len(ctx) + e.n_tok, k, d, e, ctx))
    lens = sorted(j[0] for j in jobs)
    sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
    picks = {
        "median": sorted(jobs, key=lambda j: j[0])[len(jobs) // 2],
        "longest": max(jobs, key=lambda j: j[0]),
    }
    res: dict = {
        "env": oe.env_info(),
        "load_secs": round(t_load, 1),
        "footprint_gb": round(torch.cuda.memory_allocated() / 2**30, 2)
        if torch.cuda.is_available()
        else None,
        "n_scored": len(jobs),
        "total_tokens": int(sum(lens)),
        "max_len": lens[-1],
        "median_len": lens[len(lens) // 2],
        "seqs": {},
    }
    model.eval()
    for name, (L, k, _d, e, ctx) in picks.items():
        ts = []
        oe.reset_peak()
        for r in range(a.n + 1):
            sync()
            t = time.time()
            with torch.inference_mode():
                score_one(model, ctx, e, pad_id=pad_id)
            sync()
            if r > 0:
                ts.append(time.time() - t)
        res["seqs"][name] = {
            "tokens": L,
            "k": k,
            "secs_median": round(float(np.median(ts)), 3),
            "peak_gpu_gb": oe.peak_gb(),
        }
        print(
            f"[{oe.utc()}] bench icl {name}: {L} tokens (k={k}) {res['seqs'][name]['secs_median']:.2f}s peak {res['seqs'][name]['peak_gpu_gb']} GB",
            flush=True,
        )
    x1, y1 = res["seqs"]["median"]["tokens"], res["seqs"]["median"]["secs_median"]
    x2, y2 = res["seqs"]["longest"]["tokens"], res["seqs"]["longest"]["secs_median"]
    slope = (y2 - y1) / (x2 - x1) if x2 != x1 else y2 / x2
    secs = sum(max(0.0, y1 + slope * (L - x1)) for L in lens)
    # --prefix-cache: the longest job's prefix cached once, then its tail (target prompt + response) timed
    L, k, d, e, ctx = picks["longest"]
    plen = len(ctx) - len(pc.user(prompts[e.pid]) + pc.head)
    sync()
    t = time.time()
    with torch.inference_mode():
        cache = build_cache(model, ctx[:plen])
    sync()
    t_build = time.time() - t
    ts = []
    oe.reset_peak()
    for r in range(a.n + 1):
        sync()
        t = time.time()
        with torch.inference_mode():
            score_one(model, ctx, e, pad_id=pad_id, cache=cache, prefix_len=plen)
        sync()
        if r > 0:
            ts.append(time.time() - t)
    with torch.inference_mode():
        full = score_one(model, ctx, e, pad_id=pad_id)
        cached = score_one(model, ctx, e, pad_id=pad_id, cache=cache, prefix_len=plen)
    del cache
    tail = L - plen
    rate = (
        float(np.median(ts)) / tail
    )  # seconds per tail token with a long prefix (an upper bound for shorter prefixes)
    prefixes = {}
    for _L, kk, dd, ee, cc in jobs:
        if kk > 0:
            prefixes[(kk, dd)] = len(cc) - len(pc.user(prompts[ee.pid]) + pc.head)
    tails = sum(Lj - prefixes.get((kk, dd), 0) for Lj, kk, dd, _e, _c in jobs if kk > 0)
    k0 = sum(Lj for Lj, kk, *_ in jobs if kk == 0)
    secs_cache = (
        sum(max(0.0, y1 + slope * (P - x1)) for P in prefixes.values())
        + rate * tails
        + max(0.0, y1 / x1) * k0
    )
    res["prefix_cache"] = {
        "longest_prefix_tokens": plen,
        "build_secs": round(t_build, 2),
        "tail_tokens": tail,
        "tail_secs_median": round(float(np.median(ts)), 3),
        "abs_diff_nats": abs(sum(full) - sum(cached)),
        "peak_gpu_gb": oe.peak_gb(),
    }
    res["projection"] = {
        "per_ckpt_min_no_cache": round((secs + t_load) / 60, 1),
        "per_ckpt_min_prefix_cache": round((secs_cache + t_load) / 60, 1),
        "load_min": round(t_load / 60, 1),
        "assumes": f"{len(jobs)} scored responses (ks {ks}, {a.draws} draws, {len(sets['test'])} test + {len(sets['reach'])} reach), batch 1, "
        "time linear in length through the median and longest measurements; excludes the weight download",
    }
    print("PROJECTION " + json.dumps(res["projection"]), flush=True)
    print(
        "MEMORY "
        + json.dumps(
            {
                "footprint_gb": res["footprint_gb"],
                "peak_longest_gb": res["seqs"]["longest"]["peak_gpu_gb"],
                "longest_tokens": x2,
                "peak_prefix_cache_gb": res["prefix_cache"]["peak_gpu_gb"],
                "longest_prefix_tokens": res["prefix_cache"]["longest_prefix_tokens"],
                "cache_vs_full_abs_diff_nats": res["prefix_cache"]["abs_diff_nats"],
            }
        ),
        flush=True,
    )
    if a.out:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        oe.atomic_write(Path(a.out) / "icl_bench.json", json.dumps(res, indent=1) + "\n")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("cmd", choices=["run", "bench"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--commit", default=None)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--ckpt", default="model")
    ap.add_argument("--pool", required=True)
    ap.add_argument(
        "--step0", default=None, help="the checkpoint's EDL step0.json (k = 0 must reproduce it)"
    )
    ap.add_argument("--out", default=None)
    ap.add_argument("--ks", default="0,1,2,4")
    ap.add_argument("--draws", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--k0-tol",
        type=float,
        default=0.0,
        help="max abs nats difference allowed between k = 0 and step0.json",
    )
    ap.add_argument("--prefix-cache", action="store_true")
    ap.add_argument(
        "--eval-batch-tokens",
        type=int,
        default=16384,
        help="k = 0 batching when --step0 has no EDL rows next to it",
    )
    ap.add_argument("--loss-chunk", type=int, default=2048)
    ap.add_argument("--max-memory-gb", type=float, default=None)
    ap.add_argument("--heartbeat-secs", type=float, default=60.0)
    ap.add_argument("--allow-short", action="store_true")
    ap.add_argument("--n", type=int, default=2, help="bench: timed repetitions")
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    ap.add_argument("--attn", default="sdpa")
    a = ap.parse_args()
    a.ce_checkpoint = False
    if a.cmd == "run":
        if not a.out:
            ap.error("run needs --out")
        sys.exit(cmd_run(a))
    sys.exit(cmd_bench(a))


if __name__ == "__main__":
    main()
