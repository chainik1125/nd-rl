"""Freeze and submit matched direct/scratchpad experiment pipelines on simplex1."""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data/users/dmitry/nd-rl/r1-cot-20260909")
    ap.add_argument("--name", required=True)
    ap.add_argument("--parent", required=True)
    ap.add_argument("--snapshot", default="v2")
    ap.add_argument("--sft-data", default="sft_cap6.jsonl")
    ap.add_argument("--rl-data", default="rl_mixed_50short_45long_5book.jsonl")
    ap.add_argument("--sft-steps", type=int, default=3000)
    ap.add_argument("--sft-lr", type=float, default=3e-5)
    ap.add_argument(
        "--skip-sft",
        action="store_true",
        help="Use parent directly; create no SFT run or SFT training statistics",
    )
    ap.add_argument("--rl-steps", type=int, default=256)
    ap.add_argument("--rl-lr", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=903)
    ap.add_argument(
        "--scratch-style", choices=("reverse", "dependency", "correction"), default="dependency"
    )
    ap.add_argument("--cap", type=int, default=6)
    ap.add_argument("--inherited-initial-cap", type=int, default=6)
    ap.add_argument("--modes", choices=("direct", "cot"), nargs="+", default=["direct", "cot"])
    ap.add_argument("--rl-precision", choices=("fp32", "bf16"), default="fp32")
    ap.add_argument("--eval-precision", choices=("fp32", "bf16"), default="fp32")
    ap.add_argument("--rl-budget", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--prompts-per-step", type=int, default=8)
    ap.add_argument("--retention-weight", type=float, default=0.1)
    ap.add_argument("--advantage", choices=("loo", "standardized"), default="loo")
    ap.add_argument("--loss-normalization", choices=("fixed", "sequence"), default="fixed")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--microbatch", type=int, default=32)
    ap.add_argument("--decode-batch", type=int, default=64)
    ap.add_argument("--book-samples", type=int, default=16)
    ap.add_argument("--timeout", type=int, default=10800)
    a = ap.parse_args()
    if len(a.modes) != len(set(a.modes)):
        ap.error("--modes must not contain duplicates")
    if a.temperature <= 0 or a.group_size < 2 or a.prompts_per_step < 1:
        ap.error("RL needs positive temperature, >=2 samples, and >=1 prompt")
    if a.retention_weight < 0 or min(a.cap, a.inherited_initial_cap) < 1:
        ap.error("Retention weight and proof caps must be valid")
    root = Path(a.root)
    py = "/data/users/dmitry/simplex-research/.venv/bin/python"
    run = root / "snapshots" / a.snapshot / "run.py"
    for mode in a.modes:
        name = a.name + "_" + mode
        sft, rl = (root / "runs" / (name + suffix) for suffix in ("_sft", "_rl"))
        sft_endpoint = Path(a.parent) if a.skip_sft else sft / "final.pt"
        starting_phase = "parent" if a.skip_sft else "sft"
        release = root / "plans" / (name + "_eval_release.json")

        def cmd(stage, args):
            return dict(kind="command", name=stage, command=[py, str(run), *map(str, args)])

        common = [
            "--mode",
            mode,
            "--seed",
            a.seed,
            "--guard-json",
            root / "data/heldout_guards.json",
            "--save-every",
            0,
            "--scratch-style",
            a.scratch_style,
            "--max-proof-lines",
            a.cap,
        ]
        stages = [
            cmd(
                "sft",
                [
                    "sft",
                    "--checkpoint",
                    a.parent,
                    "--data",
                    root / "data" / a.sft_data,
                    "--output",
                    sft,
                    "--steps",
                    a.sft_steps,
                    "--lr",
                    a.sft_lr,
                    "--batch-size",
                    a.batch,
                    "--microbatch",
                    a.microbatch,
                    "--precision",
                    "bf16",
                    *common,
                ],
            ),
            cmd(
                "rl",
                [
                    "rl",
                    "--checkpoint",
                    sft_endpoint,
                    "--data",
                    root / "data" / a.rl_data,
                    "--output",
                    rl,
                    "--steps",
                    a.rl_steps,
                    "--lr",
                    a.rl_lr,
                    "--prompts-per-step",
                    a.prompts_per_step,
                    "--group-size",
                    a.group_size,
                    "--decode-batch",
                    a.decode_batch,
                    "--microbatch",
                    16,
                    "--max-new-tokens",
                    a.rl_budget,
                    "--temperature",
                    a.temperature,
                    "--precision",
                    a.rl_precision,
                    "--max-logprob-error",
                    0.01 if a.rl_precision == "fp32" else 0.5,
                    "--retention-data",
                    root / "data/retention.jsonl",
                    "--retention-weight",
                    a.retention_weight,
                    "--advantage",
                    a.advantage,
                    "--loss-normalization",
                    a.loss_normalization,
                    *common,
                ],
            ),
            dict(
                kind="release_development",
                completed_rl_checkpoint=str(rl / "final.pt"),
                checkpoints=list(
                    dict.fromkeys([str(sft_endpoint), str(rl / "final.pt"), a.parent])
                ),
                data=[str(root / "data/textbook_dev.jsonl")],
                output=str(release),
            ),
        ]
        if a.skip_sft:
            stages = stages[1:]
        for phase, checkpoint in ((starting_phase, sft_endpoint), ("rl", rl / "final.pt")):
            for panel, data, k, temp in (
                ("short", "synthetic_dev", 1, 0),
                ("long", "synthetic_long_dev", 8, 0.8),
                ("book", "textbook_dev", a.book_samples, 0.8),
                ("book_greedy", "textbook_dev", 1, 0),
            ):
                stages.append(
                    cmd(
                        phase + "_" + panel,
                        [
                            "eval",
                            "--checkpoint",
                            checkpoint,
                            "--mode",
                            mode,
                            "--data",
                            root / "data" / (data + ".jsonl"),
                            "--output",
                            root / "runs" / (name + "_" + phase + "_" + panel),
                            "--samples",
                            k,
                            "--temperature",
                            temp,
                            "--max-new-tokens",
                            1536,
                            "--decode-batch",
                            a.decode_batch,
                            "--precision",
                            a.eval_precision,
                            "--seed",
                            927,
                            "--max-proof-lines",
                            a.cap,
                            "--scratch-style",
                            a.scratch_style,
                            "--evaluation-release-json",
                            release,
                        ],
                    )
                )
        config = root / "plans" / (name + "_pipeline.json")
        with config.open("x") as f:
            json.dump(
                dict(
                    stages=stages,
                    completed=str(root / "runs" / (name + "_pipeline_completed.json")),
                    declared_args=vars(a),
                    experiment_metadata=dict(
                        mode=mode,
                        parent_checkpoint=a.parent,
                        inherited_initial_proof_cap=a.inherited_initial_cap,
                        current_supervised_proof_cap=a.cap,
                        scratch_style=a.scratch_style,
                        sft_performed=not a.skip_sft,
                        starting_checkpoint=str(sft_endpoint),
                        starting_endpoint_role=starting_phase,
                    ),
                ),
                f,
                indent=2,
            )
        subprocess.run(
            [
                "python3",
                str(root / "code/submit.py"),
                "--name",
                name,
                "--snapshot",
                a.snapshot,
                "--timeout",
                str(a.timeout),
                "--",
                py,
                str(run.parent / "pipeline.py"),
                "--config",
                str(config),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
