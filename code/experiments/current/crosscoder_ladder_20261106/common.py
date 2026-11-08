# imports the fork (model, sampler) and long_rl_20261014 (model file, overlay tokenizer), which
# pyright cannot resolve
# pyright: reportMissingImports=false
"""Shared pieces: the long-RL models (run 198's untied trunk + overlay tokenizer), the checkpoint
ladder, rollouts
without Lean, and residual-stream capture.

Paths (environment):
  ND_FORK   Dan's fork at long_rl's pinned commit 51604b3 (model.py, sample.py, nd2lean, nd_verify)
  CL_CKPTS  checkpoint files, flat: <run>/<file>.pt as on HF runs/long-rl-20261014-<run>/
  CL_ART    outputs: seqs/, crosscoders/, summaries/
"""

import contextlib
import gzip
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
LONG_RL = HERE.parent / "long_rl_20261014"
FORK = Path(os.environ.get("ND_FORK", "/workspace/nd-takehome"))
CKPTS = Path(os.environ.get("CL_CKPTS", "/workspace/ckpts"))
ART = Path(os.environ.get("CL_ART", HERE / "artifacts"))
DATA = Path(os.environ.get("ND_DATA", LONG_RL / "data"))
os.environ["ND_ARCH"], os.environ["ND_RL"] = "untied", "ei"  # pretrain.py reads these at import
# overlay first (tokenizer), then the fork (model, sample), then long_rl (pretrain.py): the
# harness's order
sys.path[:0] = [str(LONG_RL / "overlay"), str(FORK), str(LONG_RL)]


def _overlay(name):
    """As harness.py: register overlay/<name>.py under the fork's module name before anything
    imports it."""
    spec = importlib.util.spec_from_file_location(name, LONG_RL / "overlay" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_overlay("lean_tok")
import pretrain as agent  # noqa: E402  (long_rl: LoopedGPT, PeriLNBlock, constants)
import torch  # noqa: E402
from sample import generate_ids  # noqa: E402  (fork)
from tokenizer import make_tokenizer  # noqa: E402  (fork; the overlay LeanTokenizer for lean_seq)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
TEMP = 0.8  # the harness's sampling temperature
HF_REPO = "robbiethompson2018/nd-rl-checkpoints"
HF_PREFIX = "runs/long-rl-20261014-"

# The ladder. Each run: its base checkpoint, the RL checkpoints (file stem -> local round), and
# where its RL clock
# starts (v5 resumed v2's 3 h endpoint, so its clock starts at v2's prior_rl_secs, read from its
# summary).
# v4 and v2 are separate runs of the same recipe; v5 continues v2. RL seconds per checkpoint come
# from each run's
# summary.json (rl.rounds[i].rl_secs_used), fetched with the checkpoints.
LADDER = {
    "v4_untied-ei_s0": {
        "base": ("v4_untied-ei_s0", "pretrain"),
        "rounds": [5, 10, 15, 20, 25, 30, 35],
        "final": True,
    },
    "v4_untied-ei_s1": {
        "base": ("v4_untied-ei_s1", "pretrain"),
        "rounds": [5, 10, 15, 20, 25, 30],  # no round-35 save on HF
        "final": True,
    },
    "v5_untied-ei_s0": {
        "base": ("v2_untied-ei_s0", "pretrain"),
        "rounds": [5, 10, 15, 20, 25, 30, 35, 40],
        "final": True,
        "prior": ("v2_untied-ei_s0", "final"),
    },
    "cpt_s0": {  # control: EI's fine-tune on pretraining records only, from v4 s0's base (cpt.py)
        "base": ("v4_untied-ei_s0", "pretrain"),
        "rounds": [5, 10, 15, 20, 25, 30, 35],
        "final": True,
    },
    "v5_untied-ei_s1": {
        "base": ("v2_untied-ei_s1", "pretrain"),
        "rounds": [5, 10, 15, 20, 25, 30, 35, 40],
        "final": True,
        "prior": ("v2_untied-ei_s1", "final"),
    },
}


def ckpt_path(run, stem):
    return CKPTS / run / f"{stem}.pt"


def ladder_points(run):
    """[(label, run, stem)] for one run: its base, its prior endpoint if resumed, the rounds, the
    final."""
    spec = LADDER[run]
    pts = [("base", *spec["base"])]
    if "prior" in spec:
        pts.append(("prior", *spec["prior"]))
    pts += [(f"r{r:03d}", run, f"rl_round_{r:03d}") for r in spec["rounds"]]
    if spec["final"]:
        pts.append(("final", run, "final"))
    return pts


def ei_rounds(summary):
    """The summary's `rl.rounds` has one entry per oracle call: an expert-iteration round is a
    first pass (k = RL_K) and its retries. Grouped by first pass; the r-th group's last entry ends
    EI round r."""
    groups = []
    for x in summary["rl"]["rounds"]:
        if x["k"] == agent.RL_K:
            groups.append([])
        groups[-1].append(x)
    return groups


def rl_seconds(run, stem):
    """RL GPU-clock seconds at a checkpoint, from the run summaries. The base is 0;
    rl_round_NNN.pt was saved after EI round NNN of its run (a resumed run counts from its own
    first round); final after the last."""
    if stem == "pretrain":
        return 0.0
    s = json.loads((CKPTS / run / "summary.json").read_text())
    prior = s["pretrain"].get("prior_rl_secs") or 0.0
    groups = ei_rounds(s)
    r = len(groups) if stem == "final" else int(stem.split("_")[-1])
    return prior + groups[r - 1][-1]["rl_secs_used"]


def build_model():
    tok = make_tokenizer(agent.TOK_MODE)
    m = agent.LoopedGPT(tok.vocab_size, agent.N_LAYER, agent.D, agent.N_HEAD, 0)
    m.blocks = torch.nn.ModuleList(
        agent.PeriLNBlock(agent.D, agent.N_HEAD) for _ in range(agent.N_LAYER)
    )
    return m, tok


def load_model(path, device=DEV):
    """A long_rl checkpoint: `{"state": ...}` (pretrain.pt, final.pt) or a bare state dict
    (rl_round_NNN.pt)."""
    m, tok = build_model()
    d = torch.load(path, map_location="cpu", weights_only=False)
    m.load_state_dict(d.get("state", d))
    m.n_loops = agent.RL_FIRST_LOOPS
    return m.to(device).eval(), tok


@contextlib.contextmanager
def capture(model, layers):
    """Residual stream after block l (l = 1 .. 6; 0 is the embedding) as {l: (B, T, d)}, overwritten
    per forward."""
    out, hs = {}, []
    for layer in layers:
        mod = model.emb if layer == 0 else model.blocks[layer - 1]
        hs.append(mod.register_forward_hook(lambda m, i, o, _l=layer: out.__setitem__(_l, o)))
    try:
        yield out
    finally:
        for h in hs:
            h.remove()


def autocast():
    return torch.autocast("cuda", dtype=torch.bfloat16, enabled=(DEV == "cuda"))


def read(fn):
    op = gzip.open if str(fn).endswith(".gz") else open
    with op(fn, "rt") as f:
        return [json.loads(line) for line in f]


def new_tokens(body, max_new=2304):
    """harness.new_tokens: the per-target sampling cap."""
    return min(max_new, -(-(64 + 30 * body) // 128) * 128)


def kv_batch(n_tokens, kv_tokens=1024 * 6 * 1700):
    return max(32, min(1024, kv_tokens // (agent.N_LAYER * n_tokens)))


@torch.no_grad()
def sample_seqs(model, tok, recs, k, seed, max_new=2304, log=print):
    """harness.sample without the Lean gate: k attempts per record at TEMP. Returns [(ids,
    gen_start)] with ids =
    prompt + sampled tokens up to and including <eos> (or the cap), gen_start = prompt length."""
    prompts = [r["prompt"] for r in recs for _ in range(k)]
    caps = [new_tokens(r["body"], max_new) for r in recs for _ in range(k)]
    ids = [tok.encode_prompt(p) for p in prompts]
    gen = torch.Generator(device=DEV)
    gen.manual_seed(seed)
    outs = [None] * len(prompts)
    t0 = time.time()
    for cap in sorted(set(caps)):
        order = sorted(
            (i for i in range(len(prompts)) if caps[i] == cap), key=lambda i: len(ids[i])
        )
        batch = kv_batch(len(ids[order[-1]]) + cap)
        for s in range(0, len(order), batch):
            chunk = order[s : s + batch]
            with autocast():
                res = generate_ids(model, tok, [ids[i] for i in chunk], False, TEMP, cap, gen)
            for i, o in zip(chunk, res, strict=True):
                o = o[: o.index(tok.eos) + 1] if tok.eos in o else [x for x in o if x != tok.pad]
                outs[i] = o
    if DEV == "cuda":
        torch.cuda.empty_cache()
    log(f"sampled {len(prompts)} sequences in {time.time() - t0:.0f}s")
    return [(ids[i] + outs[i], len(ids[i])) for i in range(len(prompts))]
