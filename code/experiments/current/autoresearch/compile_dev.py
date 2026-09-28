# imports the fork (model, tokenizer, train, ...), which is not resolvable here
# pyright: reportMissingImports=false, reportAttributeAccessIssue=false
# pyright: reportPossiblyUnboundVariable=false
"""On a GPU pod: time the incumbent pretrain.py training step eager vs torch.compile on real
token-budget batches, and log every recompile with the guard that failed.

  ND_FORK=/workspace/nd-takehome TORCH_LOGS=recompiles python compile_dev.py
Knobs (env): MODE eager|compile, DYN none|true|false (torch.compile dynamic=), DUCK 1|0
(duck sizing), MARK 1 (mark_dynamic on B and T), PAD n (pad T up to a multiple of n),
TARGET full|simple (incumbent mtp_loss | fork GPT + fork loss_on), STEPS, CACHE (cache_size_limit).
"""

import json
import os
import random
import sys
import time

import torch

sys.path.insert(0, os.environ.get("ND_FORK", "/workspace/nd-takehome"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pretrain as P  # noqa: E402
from model import GPT  # noqa: E402  (fork)
from train import loss_on  # noqa: E402  (fork)

E = os.environ.get
MODE, DYN, TARGET = E("MODE", "compile"), E("DYN", "none"), E("TARGET", "full")
STEPS, PAD = int(E("STEPS", "60")), int(E("PAD", "0"))
torch._dynamo.config.cache_size_limit = int(E("CACHE", "8"))
if E("COALESCE", "1") == "0":  # inductor tiling analysis crashes on dynamic shapes (2.9.1)
    torch._inductor.config.triton.coalesce_tiling_analysis = False
if E("DUCK", "1") == "0":
    torch.fx.experimental._config.use_duck_shape = False

fork = E("ND_FORK", "/workspace/nd-takehome")
with open(f"{fork}/data/train.jsonl") as f:
    recs = [json.loads(line) for line in f]
tok = P.make_tokenizer(P.TOK_MODE)
data = [(tok.encode_prompt(r["prompt"]), tok.encode_proof(r["proof"])) for r in recs]
ids, mask, ref_max, lens = P.to_gpu(data, tok, "cuda")
gen = torch.Generator(device="cuda")
gen.manual_seed(0)
torch.manual_seed(0)

if TARGET == "full":
    model = P.ALiBiGPT(tok.vocab_size, P.N_LAYER, P.D, P.N_HEAD)
    model.blocks = torch.nn.ModuleList(P.PeriLNBlock(P.D, P.N_HEAD) for _ in range(P.N_LAYER))
    model.blocks.apply(model._init)
    model, mtp = model.cuda(), P.MTPHead(P.D, P.N_HEAD).cuda()

    def step_loss(x, m):
        return P.mtp_loss(model, mtp, x, m)[0]
else:
    model = GPT(tok.vocab_size, P.N_LAYER, P.D, P.N_HEAD).cuda()

    def step_loss(x, m):
        return loss_on(model, x, m)


dyn = {"none": None, "true": True, "false": False}[DYN]
if MODE == "regional":  # compile the repeated block once; the rest of the step stays eager
    for b in list(model.blocks) + ([mtp.block] if TARGET == "full" else []):
        b.compile(dynamic=dyn, mode=E("CMODE") or None)
fn = step_loss if MODE in ("eager", "regional") else torch.compile(step_loss, dynamic=dyn)
params = [p for p in model.parameters()] + (
    [p for p in mtp.parameters()] if TARGET == "full" else []
)
opt = torch.optim.AdamW(params, lr=1e-4)
autocast = torch.autocast("cuda", dtype=torch.bfloat16)
order, cuts = P.token_batches(lens, len(data) - P.VAL_N, gen, random.Random(0))
print(
    f"torch {torch.__version__} COALESCE={E('COALESCE', '1')} MODE={MODE} DYN={DYN} "
    f"DUCK={E('DUCK', '1')} MARK={E('MARK', '0')} "
    f"PAD={PAD} TARGET={TARGET}",
    flush=True,
)
t0, tokens, times = time.time(), 0, []
for i, (s, e) in enumerate(cuts[:STEPS]):
    x, m = P.gpu_batch(ids, mask, ref_max, lens, order[s:e], tok, gen)
    if PAD:
        T = min(-(-x.shape[1] // PAD) * PAD, ids.shape[1])
        x = torch.nn.functional.pad(x, (0, T - x.shape[1]), value=tok.pad)
        m = torch.nn.functional.pad(m, (0, T - m.shape[1]), value=False)
    if E("MARK") == "1":
        for t in (x, m):
            torch._dynamo.mark_dynamic(t, 0)
            torch._dynamo.mark_dynamic(t, 1)
    torch.cuda.synchronize()
    ts = time.time()
    with autocast:
        loss = fn(x, m)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    torch.cuda.synchronize()
    dt = time.time() - ts
    times.append(dt)
    tokens += x.numel()
    c = torch._dynamo.utils.counters
    if i < 12 or dt > 1.0 or i % 10 == 0:
        print(
            f"step {i:3d} B={x.shape[0]:4d} T={x.shape[1]:3d} {dt * 1000:8.1f} ms "
            f"graphs={c['stats']['unique_graphs']} frames={c['stats']['calls_captured']}",
            flush=True,
        )
if E("PROFILE") == "1":  # 5 profiled steps after warm-up: where does the step time go?
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for s, e in cuts[STEPS : STEPS + 5]:
            x, m = P.gpu_batch(ids, mask, ref_max, lens, order[s:e], tok, gen)
            with autocast:
                loss = fn(x, m)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=18), flush=True)
steady = sorted(times[10:])[len(times[10:]) // 2] if len(times) > 20 else float("nan")
print(
    f"RESULT total {time.time() - t0:.1f}s, first-10 {sum(times[:10]):.1f}s, "
    f"median steady step {steady * 1000:.1f} ms, "
    f"graphs {torch._dynamo.utils.counters['stats']['unique_graphs']}",
    flush=True,
)
