#!/usr/bin/env python3
"""The low-RL arm's compute table, think_hirl/lowrl/compute.tsv (2026-10-30, agent:claude for Leon).

    python3 hirl_compute.py

R = (6 * trained-token passes + 2 * sampled tokens) / (6 * Stage-1 tokens), Stage-1 = S1g3_s0's
440,559,685 tokens (code/experiments/current/high_rl_passk_20261030/stage1_tokens.json);
R_train = trained-token passes / Stage-1 tokens. As in that folder's README: trained-token passes
are the policy loss's trained tokens (each sampled token is trained once: --updates 4 splits a
batch into 4 minibatches); sampled tokens are tokens_per_attempt x attempts; prefill, the
old-log-prob pass, KL-anchor reference passes, canary scoring and evals are left out; rolled-back
iterations count. R_anchor adds the KL anchor's scored positions on the policy (6 x, forward and
backward), which the README's ratio leaves out: it matters for the 16-prompt variant, where the
anchor's 512 steps per AdamW step are a large share of the trained tokens.

Rows:
- the step checkpoints of grpo_think_vs_mini20_kdense / _kp16 (steps.jsonl: exact per AdamW step;
  a checkpoint after step n of iteration i counts all of iteration i's sampling);
- grpo_think_vs_mini20 s0 L2 it2 / it4 / it8 (the original run's metrics.jsonl, iteration ends);
- grpo_think_vs_L2 s0 end, grpo_think_vs_mb_kkl0 s0 L3 end and _ki78 s0 end (their stages'
  metrics.jsonl in file order: L2 then L3), when lowrl/ref/ has those files.
"""

from __future__ import annotations

import json
from pathlib import Path

H = Path("/scratch/cowifornia/nd-rl/think_hirl")
LOW = H / "lowrl"
STAGE1 = 440_559_684.74494827  # S1g3_s0: epochs x (state + completion tokens per epoch)


def rows(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def ratio(trained: float, sampled: float, anchor: float | None) -> tuple[float, float, str]:
    r = (6 * trained + 2 * sampled) / (6 * STAGE1)
    ra = "" if anchor is None else f"{(6 * (trained + anchor) + 2 * sampled) / (6 * STAGE1):.3e}"
    return r, trained / STAGE1, ra


def metric_cum(paths: list[Path], upto: int, prompts_group: int) -> tuple[int, float, float, int]:
    """(iterations, sampled, trained, AdamW steps) summed in file order over the stages; the last
    stage up to iteration `upto` (every row of the earlier ones)."""
    samp = tr = 0.0
    n = steps = 0
    for i, p in enumerate(paths):
        for r in rows(p):
            if i == len(paths) - 1 and r["iter"] > upto:
                continue
            samp += r["tokens_per_attempt"] * prompts_group
            tr += r["trained_tokens"]
            steps += r.get("steps", 0)
            n += 1
    return n, samp, tr, steps


def main() -> None:
    out = [
        "\t".join(
            [
                "checkpoint",
                "run",
                "iteration",
                "minibatch",
                "adamw_step",
                "attempts",
                "sampled_tokens",
                "trained_tokens",
                "anchor_tokens",
                "R",
                "R_train",
                "R_anchor",
                "source",
            ]
        )
    ]
    for run, arm in (
        ("grpo_think_vs_mini20_kp16_s0_L2", "grpo_think_vs_mini20_kp16"),
        ("grpo_think_vs_mini20_kdense_s0_L2", "grpo_think_vs_mini20_kdense"),
    ):
        sp = LOW / run / "steps.jsonl"
        if not sp.exists():
            continue
        for s in rows(sp):
            if not s.get("ckpt"):
                continue
            tag = Path(s["ckpt"]).stem.removeprefix(run + "_")  # st4, st4_r1
            r, rt, ra = ratio(s["trained_tokens"], s["sampled_tokens"], s["anchor_tokens"])
            out.append(
                "\t".join(
                    map(
                        str,
                        [
                            f"{arm}_L2{tag}",
                            run,
                            s["iter"],
                            f"{s['minibatch']}/{s['updates']}",
                            s["step"],
                            s["attempts"],
                            s["sampled_tokens"],
                            s["trained_tokens"],
                            s["anchor_tokens"],
                            f"{r:.3e}",
                            f"{rt:.3e}",
                            ra,
                            f"{sp} (rollbacks {s['rollbacks']})",
                        ],
                    )
                )
            )
    orig = LOW / "mini20_s0_L2_orig_metrics.jsonl"
    if orig.exists():
        for it in (2, 4, 8):
            n, samp, tr, steps = metric_cum([orig], it, 256 * 16)
            r, rt, _ = ratio(tr, samp, None)
            out.append(
                "\t".join(
                    map(
                        str,
                        [
                            f"grpo_think_vs_mini20_L2it{it}",
                            "grpo_think_vs_mini20_s0_L2",
                            it,
                            "end",
                            steps,
                            n * 4096,
                            round(samp),
                            round(tr),
                            "",
                            f"{r:.3e}",
                            f"{rt:.3e}",
                            "",
                            f"{orig} (serrano's metrics.jsonl)",
                        ],
                    )
                )
            )
    ref = LOW / "ref"
    for name, paths, upto in (
        ("grpo_think_vs_L2", ["grpo_think_vs_s0_L2"], 39),
        ("grpo_think_vs_mb_kkl0", ["grpo_think_vs_s0_L2", "grpo_think_vs_mb_kkl0_s0_L3"], 39),
        ("grpo_think_vs_mb_ki78", ["grpo_think_vs_s0_L2", "grpo_think_vs_mb_ki78_s0_L3"], 78),
    ):
        ps = [ref / f"{p}_metrics.jsonl" for p in paths]
        if not all(p.exists() for p in ps):
            continue
        n, samp, tr, steps = metric_cum(ps, upto, 256 * 16)
        r, rt, _ = ratio(tr, samp, None)
        out.append(
            "\t".join(
                map(
                    str,
                    [
                        name,
                        "+".join(paths),
                        n,
                        "end",
                        steps,
                        n * 4096,
                        round(samp),
                        round(tr),
                        "",
                        f"{r:.3e}",
                        f"{rt:.3e}",
                        "",
                        "serrano's metrics.jsonl, stages in order",
                    ],
                )
            )
        )
    (LOW / "compute.tsv").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
