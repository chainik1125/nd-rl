"""Shared pieces of the circuit-diffing experiment: checkpoint registry with RL compute ratios,
model loading, and residual-stream capture.

Every checkpoint is read-only.  Labels:
  base            S1g3_s{seed} (the Stage-1 model every RL seed starts from)
  kp16_st{n}      low-RL arm, 16 prompts x 16 per AdamW step, after n AdamW steps (seed 0)
  kdense_st{n}    low-RL arm, dense prompts, after n AdamW steps (seed 0)
  mini_it{n}      grpo_think_vs_mini20_s0_L2 rolling checkpoint at iteration n (seed 0)
  L2              grpo_think_vs_s{seed}_L2 it39 (L2 end)
  L3_it{n}        grpo_think_vs_mb_kkl0_s{seed}_L3 rolling checkpoint (n = 5 .. 35)
  L3              grpo_think_vs_mb_kkl0_s{seed}_L3 it39 (L3 end, the recipe's final model)
  ki78            grpo_think_vs_mb_ki78_s0_L3 it78 (longer L3, seed 0 only)
  cpt_lr{lr}      the weight-drift control of `cpt.py` (no RL; R is nan)

R is the RL / Stage-1 FLOP ratio of `high_rl_passk_20261030` (6 x trained tokens + 2 x sampled
tokens) / (6 x Stage-1 tokens): read from its `lowrl_compute.tsv` where listed, else recomputed
from `metrics.jsonl` with the same rule (rolled-back iterations count).
"""

from __future__ import annotations

import csv
import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

os.environ.setdefault("ND_OFFLINE", "1")
os.environ.setdefault("ND_REGISTRY_SYNC", "0")

HERE = Path(__file__).resolve().parent
#: `CIRCUITS_RUNS` points at a mirror of the files this experiment reads (Stage 3 on ynez, whose
#: own think/ tree is an older copy): base and GRPO checkpoints, their args and metrics, the pools.
RUNS = Path(os.environ.get("CIRCUITS_RUNS", "/scratch/cowifornia/nd-rl/think/runs"))
G2 = RUNS / "grpo_v2"
ART = Path("/scratch/cowifornia/nd-rl/circuits")
LOWRL = ART / "ckpts/lowrl/scratch/cowifornia/nd-rl/think_hirl/lowrl"
PASSK = Path("/home/cowifornia/code/research/nd-rl/code/experiments/current/high_rl_passk_20261030")
COMPUTE_TSV = PASSK / "lowrl_compute.tsv"
S1_TOKENS = 440559684.74494827  # S1g3_s0 Stage-1 tokens (high_rl_passk stage1_tokens.json)
EVAL = RUNS / "eval/v2"
CHARLES = RUNS / "pools/charles_1002/charles_clean.jsonl"


@dataclass(frozen=True)
class Ckpt:
    label: str
    seed: int
    path: Path
    R: float  # RL / Stage-1 FLOP ratio (0 for the base)
    read: str | None = None  # the stem of its Charles149 k1024 read, if any


def _tsv_R() -> dict[str, float]:
    out: dict[str, float] = {}
    if COMPUTE_TSV.exists():
        with open(COMPUTE_TSV) as f:
            for row in csv.DictReader(f, delimiter="\t"):
                out[row["checkpoint"]] = float(row["R"])
    return out


def _rows_upto(metrics: Path, upto: int | None) -> list[dict[str, Any]]:
    with open(metrics) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if upto is None:
        return rows
    last = max(i for i, r in enumerate(rows) if r["iter"] == upto)
    return rows[: last + 1]


def rl_R(segments: list[tuple[Path, int | None]]) -> float:
    """high_rl_passk's flop_ratio over GRPO stages (run dir, last iteration or None)."""
    samp = trained = 0.0
    for d, upto in segments:
        a = json.loads((d / "args.json").read_text())
        per_it = int(a["prompts"]) * int(a["group"])
        for r in _rows_upto(d / "metrics.jsonl", upto):
            samp += float(r["tokens_per_attempt"]) * per_it
            trained += float(r["trained_tokens"])
    return (6 * trained + 2 * samp) / (6 * S1_TOKENS)


def registry() -> dict[tuple[str, int], Ckpt]:
    """(label, seed) -> Ckpt, for every checkpoint present on disk."""
    tsv = _tsv_R()
    out: dict[tuple[str, int], Ckpt] = {}

    def add(c: Ckpt) -> None:
        if c.path.exists():
            out[(c.label, c.seed)] = c

    for s in (0, 1, 2):
        add(Ckpt("base", s, RUNS / f"stage1/S1g3_s{s}/model.pt", 0.0, f"s1g3_s{s}"))
        l2 = G2 / f"grpo_think_vs_s{s}_L2"
        l3 = G2 / f"grpo_think_vs_mb_kkl0_s{s}_L3"
        if (l2 / "metrics.jsonl").exists():
            add(
                Ckpt(
                    "L2",
                    s,
                    l2 / f"ckpts/grpo_think_vs_s{s}_L2_it39.pt",
                    rl_R([(l2, None)]),
                    f"grpo_think_vs_L2_s{s}" if s == 0 else None,
                )
            )
        if (l3 / "metrics.jsonl").exists():
            for it in (5, 10, 15, 20, 25, 30, 35):
                add(
                    Ckpt(
                        f"L3_it{it}",
                        s,
                        l3 / f"ckpts/grpo_think_vs_mb_kkl0_s{s}_L3_it{it}_rolling.pt",
                        rl_R([(l2, None), (l3, it)]),
                    )
                )
            add(
                Ckpt(
                    "L3",
                    s,
                    l3 / f"ckpts/grpo_think_vs_mb_kkl0_s{s}_L3_it39.pt",
                    rl_R([(l2, None), (l3, None)]),
                    f"grpo_think_vs_mb_kkl0_s{s}",
                )
            )
    for arm in ("kp16", "kdense"):
        for st in (1, 2, 4, 8, 16, 32):
            name = f"grpo_think_vs_mini20_{arm}_L2st{st}"
            add(
                Ckpt(
                    f"{arm}_st{st}",
                    0,
                    LOWRL
                    / f"grpo_think_vs_mini20_{arm}_s0_L2/ckpts"
                    / f"grpo_think_vs_mini20_{arm}_s0_L2_st{st}.pt",
                    tsv.get(name, float("nan")),
                    f"grpo_think_vs_mini20_{arm}_L2st{st}_s0",
                )
            )
    mini = G2 / "grpo_think_vs_mini20_s0_L2"
    for it in (2, 4, 8):
        add(
            Ckpt(
                f"mini_it{it}",
                0,
                mini / f"ckpts/grpo_think_vs_mini20_s0_L2_it{it}_rolling.pt",
                tsv.get(f"grpo_think_vs_mini20_L2it{it}", float("nan")),
                f"grpo_think_vs_mini20_L2it{it}_s0",
            )
        )
    for p in sorted((ART / "ckpts/cpt").glob("cpt_lr*.pt")):  # weight-drift control (cpt.py)
        add(Ckpt(p.stem, 0, p, float("nan")))
    ki = G2 / "grpo_think_vs_mb_ki78_s0_L3"
    add(
        Ckpt(
            "ki78",
            0,
            ki / "ckpts/grpo_think_vs_mb_ki78_s0_L3_it78.pt",
            tsv.get("grpo_think_vs_mb_ki78", float("nan")),
            "grpo_think_vs_mb_ki78_s0",
        )
    )
    return out


def state_dict(path: Path) -> dict[str, torch.Tensor]:
    ck = torch.load(str(path), map_location="cpu", weights_only=False)
    return {k: v.float() for k, v in ck["state"].items()}


def load_model(path: Path, device: str = "cpu") -> tuple[Any, Any, dict[str, Any]]:
    """`code.thinking.ckpt.load_ckpt` (eval mode, the checkpoint's tokenizer and formats)."""
    from code.thinking.ckpt import load_ckpt

    m, tok, extra = load_ckpt(path, device)
    for p in m.parameters():
        p.requires_grad_(False)
    return m, tok, extra


@contextmanager
def capture(model: Any, layers: list[int]) -> Iterator[dict[int, torch.Tensor]]:
    """Residual stream after block l (l = 1 .. n_layer; l = 0 is the embedding) on each forward
    inside the block, as {l: (B, T, d)}; overwritten by every forward."""
    out: dict[int, torch.Tensor] = {}
    hs = []
    for layer in layers:
        if layer == 0:
            hs.append(model.emb.register_forward_hook(lambda m, i, o, _l=0: out.__setitem__(_l, o)))
        else:
            hs.append(
                model.blocks[layer - 1].register_forward_hook(
                    lambda m, i, o, _l=layer: out.__setitem__(_l, o)
                )
            )
    try:
        yield out
    finally:
        for h in hs:
            h.remove()


def pt_now() -> str:
    import datetime as dt
    from zoneinfo import ZoneInfo

    return dt.datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d %I:%M %p PT")


def free_gpus(max_jobs_total: int = 3) -> list[int]:
    """GPUs with no compute process at all, if serrano's total GPU-job count allows another job
    (`max_jobs_total` counts every user's processes)."""
    import subprocess

    q = ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"]
    gpus = [line.split(", ") for line in subprocess.check_output(q, text=True).strip().splitlines()]
    apps = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], text=True
    ).strip()
    busy = {line.split(", ")[0] for line in apps.splitlines() if line.strip()}
    if len(busy) >= max_jobs_total:
        return []
    return [int(i) for i, u in gpus if u not in busy]


def assert_gpu_free(gpu: int) -> None:
    """Refuse to start on a GPU that has any compute process, or when 3 GPUs are already busy.

    With `CIRCUITS_ALLOWED_GPUS` set (Stage 3 on ynez: one reserved GPU), the GPU must be in that
    list and have no compute process; serrano's 3-job power limit does not apply. Indices are
    nvidia-smi's, so `CUDA_DEVICE_ORDER=PCI_BUS_ID` is required for CUDA to agree."""
    allowed = os.environ.get("CIRCUITS_ALLOWED_GPUS")
    if allowed is not None:
        import subprocess

        order = os.environ.get("CUDA_DEVICE_ORDER")
        assert order == "PCI_BUS_ID", "set CUDA_DEVICE_ORDER=PCI_BUS_ID"
        if gpu not in {int(x) for x in allowed.split(",") if x}:
            raise SystemExit(f"GPU {gpu} is not in CIRCUITS_ALLOWED_GPUS={allowed}: not starting")
        q = ["nvidia-smi", "-i", str(gpu), "--query-compute-apps=pid", "--format=csv,noheader"]
        if subprocess.check_output(q, text=True).strip():
            raise SystemExit(f"GPU {gpu} has a compute process: not starting")
        return
    if gpu not in free_gpus():
        raise SystemExit(f"GPU {gpu} is busy (or serrano already has 3 GPU jobs): not starting")
