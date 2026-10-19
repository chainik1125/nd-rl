"""Collapse audit, probe 3: which parameters moved, checkpoint to checkpoint (CPU, read-only).

Run from the repo root, with M = code.experiments.current.thinking_traces_20261005.collapse_audit:

    code/.venv/bin/python -m $M.param_deltas

Per pair of checkpoints: the relative change ||dW|| / ||W|| of every tensor (largest first), and for
the vocabulary-sized tensors (`emb.weight`, `head.weight`) the rows that moved most, by token, plus
the change of each head row's projection on the mean final hidden state direction is not needed:
the head row norm and the row's change norm are printed. Seeds 0 and 1 over the same iterations.
"""

from __future__ import annotations

from pathlib import Path

import torch

from code.thinking.tok import make_tokenizer

R = Path("/scratch/cowifornia/nd-rl/think/runs/restart3_grpofix")
S0 = R / "grpo_T1v1e/ckpts"
S1 = R / "grpo_T1v1e_s1/ckpts"


def sd(p: Path) -> dict[str, torch.Tensor]:
    return {k: v.float() for k, v in torch.load(str(p), map_location="cpu")["state"].items()}


PAIRS = [
    ("s0 it0->it39", S0 / "grpo_T1v1e_it0.pt", S0 / "grpo_T1v1e_it39.pt"),
    ("s0 it39->it49", S0 / "grpo_T1v1e_it39.pt", S0 / "grpo_T1v1e_it49_alarm.pt"),
    ("s0 it49->it50", S0 / "grpo_T1v1e_it49_alarm.pt", S0 / "grpo_T1v1e_it50_alarm.pt"),
    ("s0 it50->it52", S0 / "grpo_T1v1e_it50_alarm.pt", S0 / "grpo_T1v1e_it52_alarm.pt"),
    ("s0 it52->it60", S0 / "grpo_T1v1e_it52_alarm.pt", S0 / "grpo_T1v1e_it60_rolling.pt"),
    ("s0 it60->it78", S0 / "grpo_T1v1e_it60_rolling.pt", S0 / "grpo_T1v1e_it78.pt"),
    ("s1 it0->it39", S1 / "grpo_T1v1e_s1_it0.pt", S1 / "grpo_T1v1e_s1_it39.pt"),
    ("s1 it39->it49", S1 / "grpo_T1v1e_s1_it39.pt", S1 / "grpo_T1v1e_s1_it49_alarm.pt"),
    ("s1 it49->it50", S1 / "grpo_T1v1e_s1_it49_alarm.pt", S1 / "grpo_T1v1e_s1_it50_alarm.pt"),
    ("s1 it50->it52", S1 / "grpo_T1v1e_s1_it50_alarm.pt", S1 / "grpo_T1v1e_s1_it52_alarm.pt"),
    ("s1 it52->it60", S1 / "grpo_T1v1e_s1_it52_alarm.pt", S1 / "grpo_T1v1e_s1_it60_rolling.pt"),
    ("s1 it60->it78", S1 / "grpo_T1v1e_s1_it60_rolling.pt", S1 / "grpo_T1v1e_s1_it78.pt"),
]


def main() -> int:
    torch.set_num_threads(8)
    tok = make_tokenizer("lean_thinke")
    for name, a, b in PAIRS:
        A, B = sd(a), sd(b)
        rel = {k: float((B[k] - A[k]).norm() / A[k].norm().clamp_min(1e-12)) for k in A}
        tot = float(
            torch.sqrt(sum(((B[k] - A[k]) ** 2).sum() for k in A))
            / torch.sqrt(sum((A[k] ** 2).sum() for k in A))
        )
        top = sorted(rel.items(), key=lambda kv: -kv[1])[:6]
        print(f"\n== {name}: total rel change {tot:.4f}")
        print("  top tensors: " + ", ".join(f"{k} {v:.4f}" for k, v in top))
        for key in ("head.weight", "emb.weight"):
            d = (B[key] - A[key]).norm(dim=1)
            order = torch.argsort(d, descending=True)[:8].tolist()
            med = float(d.median())
            print(
                f"  {key} row change, median {med:.4f}; top: "
                + ", ".join(f"{tok.itos[i]!r} {float(d[i]):.4f}" for i in order)
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
