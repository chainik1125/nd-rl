"""Create six matched-compute configs, then freeze inputs after source review."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path

from pipeline import local, protected, sha, write


PYTHON = "/data/users/dmitry/simplex-research/.venv/bin/python"
SEED = 20260910
ORDER = ((10, "cot"), (6, "cot"), (12, "cot"), (10, "direct"), (6, "direct"), (12, "direct"))
DEV_SHA = "1b04192d8a2d91b488a0884ccb096004003dbfcf40bb7aea55caa81c3f3cf53c"


def integer(value):
    amount = Decimal(str(value))
    if not amount.is_finite() or amount <= 0 or amount != amount.to_integral_value():
        raise argparse.ArgumentTypeError("A positive exact integer is required")
    return int(amount)


def milestones(budget, divisors):
    return sorted({max(1, budget // divisor) for divisor in divisors} | {budget})


def build_specs(root, sft_budget, rl_budget):
    root = Path(root).resolve()
    specs = []
    for cap, mode in ORDER:
        name = f"cap{cap}_{mode}_s{SEED}"
        base = Path("runs") / name
        source = "runs/shared_init/final.pt"
        runner = "snapshots/main_v1/run.py"
        sft_data = f"data/sft_cap{cap}.jsonl"
        spec = {"schema_version": 1, "name": name,
                "label": f"{'Direct' if mode == 'direct' else 'CoT'} cap{cap} (8.9M)",
                "root": str(root), "run_base": str(base), "python": PYTHON,
                "runner": runner, "mode": mode, "cap": cap, "seed": SEED,
                "shared_init": source, "training_stages": [],
                "data": {"sft": sft_data, "retention": sft_data, "rl": "data/rl_common.jsonl",
                         "pretraining": "data/pretraining_probe_common_short.jsonl",
                         "validation_synthetic": "evaluation/synthetic_dev.jsonl"},
                "development_data": {"path": "evaluation/textbook_dev.jsonl", "sha256": DEV_SHA,
                                     "role": "textbook_dev", "n_targets": 58,
                                     "hash_source": "prior split manifest; payload excluded from preparation"},
                "development_release": str(base / "development_release.json"),
                "evaluation": {"seed": 0, "max_seq_len": 8192, "max_new_tokens": 1536,
                               "decode_batch": 64, "greedy_samples": 1, "sampled_samples": 16},
                "checkpoint_policy": "Final SFT starts one continuous full-budget RL trajectory. All scheduled checkpoints retained.",
                "step_safety_ceiling": 100000}
        for command, budget, divisors in (("sft", sft_budget, (4, 2)), ("rl", rl_budget, (8, 4, 2))):
            checkpoint = source if command == "sft" else str(base / "sft/final.pt")
            output = str(base / command)
            guard = "data/supervised_guards.json" if command == "sft" else "data/rl_guards.json"
            data = sft_data if command == "sft" else "data/rl_common.jsonl"
            points = milestones(budget, divisors)
            expected = {"command": command, "checkpoint": str(root / checkpoint),
                        "output": str(root / output), "data": str(root / data),
                        "guard_json": str(root / guard), "seed": SEED, "mode": mode,
                        "max_proof_lines": cap, "max_seq_len": 8192,
                        "max_flop_proxy": budget, "flop_milestones": points,
                        "steps": 100000, "scratch_style": "dependency"}
            args = [command, "--checkpoint", str(root / checkpoint), "--output", str(root / output),
                    "--data", str(root / data), "--guard-json", str(root / guard),
                    "--mode", mode, "--seed", str(SEED), "--max-proof-lines", str(cap),
                    "--scratch-style", "dependency", "--device", "cuda", "--max-seq-len", "8192",
                    "--steps", "100000", "--max-flop-proxy", str(budget),
                    "--flop-milestones", *map(str, points), "--save-every", "0"]
            if command == "sft":
                expected.update(lr=3e-4, batch_size=128, microbatch=32, precision="bf16")
                args += ["--lr", "0.0003", "--batch-size", "128", "--microbatch", "32",
                         "--precision", "bf16", "--log-every", "25"]
            else:
                expected.update(lr=1e-5, prompts_per_step=8, group_size=8, microbatch=16,
                                max_new_tokens=1536, decode_batch=64, temperature=.8, policy_epochs=1,
                                precision="fp32", retention_data=str(root / sft_data), retention_weight=.1,
                                retention_guard_json=str(root / "data/supervised_guards.json"),
                                retention_batch=16, prompt_schedule="uniform", advantage="loo",
                                loss_normalization="fixed", policy_weight=1.)
                args += ["--lr", "0.00001", "--prompts-per-step", "8", "--group-size", "8",
                         "--microbatch", "16", "--decode-batch", "64", "--max-new-tokens", "1536",
                         "--temperature", "0.8", "--precision", "fp32", "--policy-epochs", "1",
                         "--prompt-schedule", "uniform", "--advantage", "loo", "--loss-normalization", "fixed",
                         "--retention-data", str(root / sft_data), "--retention-weight", "0.1",
                         "--retention-guard-json", str(root / "data/supervised_guards.json"),
                         "--retention-batch", "16", "--max-logprob-error", "0.01", "--log-every", "1"]
            spec["training_stages"].append({"command_name": command, "checkpoint": checkpoint,
                                            "output": output, "target_flop_proxy": budget,
                                            "milestones": points, "expected_args": expected,
                                            "command": [PYTHON, str(root / runner), *args]})
        specs.append(spec)
    return specs


def make_configs(root, sft_budget, rl_budget):
    root = Path(root).resolve()
    specs = build_specs(root, sft_budget, rl_budget)
    paths = [root / "plans" / f"{spec['name']}.json" for spec in specs]
    if any(path.exists() for path in paths) or (root / "plans/protocol.json").exists():
        raise ValueError("Configs/protocol already exist; do not overwrite a frozen candidate")
    for path, spec in zip(paths, specs, strict=True):
        write(path, spec)
    return [str(path) for path in paths]


def freeze(root, sft_budget, rl_budget):
    root = Path(root).resolve()
    destination = root / "plans/protocol.json"
    if destination.exists():
        raise ValueError("Protocol already exists")
    specs = build_specs(root, sft_budget, rl_budget)
    paths = [root / "plans" / f"{spec['name']}.json" for spec in specs]
    for path, expected in zip(paths, specs, strict=True):
        if json.loads(path.read_text()) != expected:
            raise ValueError(f"Config does not reconstruct exactly: {path.name}")
    manifest = json.loads((root / "data/manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("Data preparation is incomplete")
    inputs = {}
    def bind(path, expected=None):
        path = local(root, path)
        if protected(path):
            raise ValueError("Protected theorem payload cannot enter preparation hashes")
        value = sha(path)
        if expected is not None and value != expected:
            raise ValueError(f"Prepared input changed: {path}")
        inputs[str(path.relative_to(root))] = value
    for path in paths:
        bind(path)
    for path, digest in manifest["output_sha256"].items():
        bind(root / "data" / path, digest)
    for path in ("data/manifest.json", "runs/shared_init/final.pt", "runs/shared_init/manifest.json", "plans/model.json",
                 "code/plan.py", "code/pipeline.py", "code/dispatch.py", "code/schedule.py",
                 "code/calibrate_reference.py", "reports/compute_calibration.json"):
        bind(path)
    snapshot = root / "snapshots/main_v1"
    for name in ("run.py", "scratch_variants.py", "pipeline.py"):
        bind(root / "code" / name)
        bind(snapshot / name)
        if sha(root / "code" / name) != sha(snapshot / name):
            raise ValueError(f"Source snapshot differs from reviewed code: {name}")
    vendor = sorted((snapshot / "vendor").rglob("*.py"))
    if not vendor:
        raise ValueError("Frozen vendor sources are absent")
    for path in vendor:
        bind(path)
    bind("evaluation/synthetic_dev.jsonl", manifest["synthetic_dev"]["sha256_copied_from_prior_metadata"])
    # Read ONLY our fresh initialization, never a protected theorem payload.
    import torch
    initial = torch.load(root / "runs/shared_init/final.pt", map_location="cpu", weights_only=True)
    if (initial["metadata"].get("stage") != "prepared" or initial["metadata"]["args"].get("seed") != SEED
            or initial["config"].get("max_seq_len") != 8192 or initial["config"].get("vocab_size") != 43):
        raise ValueError("Shared initialization architecture/seed/tokenizer is inconsistent")
    unique_tensors = {(value.untyped_storage().data_ptr(), value.storage_offset(), tuple(value.shape), tuple(value.stride())): value
                      for value in initial["model"].values()}
    parameter_count = sum(value.numel() for value in unique_tensors.values())
    if parameter_count != 8872320:
        raise ValueError("Shared tied-weight parameter count differs from the fixed architecture")
    protocol = {"schema_version": 1, "kind": "fresh_cap_compute_comparison", "created_utc": datetime.now(timezone.utc).isoformat(),
                "arms": [spec["name"] for spec in specs], "all_six_arms_required": True,
                "training_seeds": [SEED], "input_sha256": dict(sorted(inputs.items())),
                "shared_initial_config": initial["config"],
                "shared_initial_parameter_count": parameter_count,
                "shared_init": "runs/shared_init/final.pt",
                "common_sft_flop_proxy": sft_budget, "common_rl_flop_proxy": rl_budget,
                "compute_definition": "2P*B*Q + 4*layers*d_model*B*Q*K per forward; backward adds twice forward; actual padding, generation, teacher audit, policy and retention included",
                "crossing_policy": "First complete update reaching target; report actual work and overshoot; step-limit failure is not a matched endpoint.",
                "context_tokens": 8192, "rl_max_new_tokens": 1536, "eval_max_new_tokens": 1536,
                "policy_fixed_loss_denominator": 1536,
                "historical_recipe_difference": "Historical work calibrates a scalar only. All new arms use common1536 generation and fixed-loss denominator,8192 context and weighted capped supervision.",
                "development_data": specs[0]["development_data"],
                "development_gate": "Per-arm actual full RL budget and endpoint identities must validate before any textbook development payload hash/read. SFT/fullRL only; all results retained.",
                "final_textbook_payload": "not included or evaluated",
                "primary_endpoints": ["sft", "rl"],
                "synthetic_probe_endpoints": ["sft", "rl_12.5pct", "rl_25pct", "rl_50pct", "rl"],
                "metric_protocol": {"own_cap_threshold": True, "common_reference_thresholds": [6, 10, 12],
                                    "sample_budgets": [1, 4, 16], "separate_greedy": True,
                                    "inference_axes": ["attempts", "actual_generated_tokens"],
                                    "inference_flop_proxy_available": False,
                                    "strict_replay_and_compression_required_for_claims": True,
                                    "proof_lengths_are_upper_bounds_not_minimum_difficulty": True,
                                    "paired_target_uncertainty_only_one_training_seed": True},
                "data_scope": {"short_rl_sft_overlap_intentional": True, "long_rl_union_disjoint": True,
                               "textbook_train_prompt_classes": 14, "textbook_sft_labels": 0,
                               "caps_also_change_family_support_and_stratum_mixture": True}}
    write(destination, protocol)
    return {"path": str(destination), "sha256": sha(destination), "inputs": len(inputs), "arms": protocol["arms"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("configs", "freeze"))
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--sft-flop-budget", type=integer, required=True)
    parser.add_argument("--rl-flop-budget", type=integer, required=True)
    args = parser.parse_args()
    function = make_configs if args.command == "configs" else freeze
    print(json.dumps(function(args.root, args.sft_flop_budget, args.rl_flop_budget), indent=2))


if __name__ == "__main__":
    main()
