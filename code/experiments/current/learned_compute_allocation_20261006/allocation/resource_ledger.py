"""Summarize measured allocation-owned GPU phases without claiming kernel time."""

import json
from pathlib import Path

ROOT = Path("artifacts/learned_compute_allocation_20261006/allocation")
OWNED = {
    "smoke_cost_s11": 2,
    "live_safe_smoke_cost_s11": 2,
    "live_safe_smoke_fixed2_s11": 2,
    "live_head_teacher_reload_smoke_s11": 2,
    "live_path_protocol_smoke_s52006": 2,
    "format_safe_smoke_s61": 2,
    "format_safe_pilot200_s61": 2,
    "format_dev_audit512_s61": 2,
    "main_format_train96_k4_s61006": 2,
    "main_format_train96_k5_s61006": 2,
    "format_main96_clean4_initial25_s61": 2,
    "format_main96_lr1e5_clean8_initial50_s61": 2,
    "format_lr1e5_1000_main_dev16_paths_s52006": 2,
    "no_cost/dev_headlr1e4_s71": 2,
    "no_cost/dev_headlr1e4_s71_dev32_paths_s52006": 2,
    "no_cost/main_s101": 2,
    "no_cost/main_s102": 4,
    "no_cost/main_s103": 2,
    "no_cost/main_s101_dev100_paths_s52006": 2,
    "no_cost/main_s101_dev200_paths_s52006": 2,
    "fixed4/main_s101": 4,
    "no_cost/main_s103_dev100_paths_s52006": 2,
    "no_cost/main_s103_dev200_paths_s52006": 2,
    "fixed4/main_s101_dev100_paths_s52006": 4,
    "fixed4/main_s101_dev200_paths_s52006": 4,
    "format_only/dev32_paths_s52006": 4,
    **{
        f"final_{benchmark}/{name}": gpu
        for benchmark in ("main", "historical")
        for name, gpu in (
            ("cost_s101", 2),
            ("no_cost_s102", 2),
            ("format_only", 2),
            ("no_cost_s101", 4),
            ("no_cost_s103", 4),
            ("fixed4_control_s101", 4),
            ("cost_s102", 3),
            ("fixed1_control_s101", 3),
            ("cost_s103", 1),
        )
    },
}


def main():
    results = []
    for name, gpu in OWNED.items():
        folder = ROOT / name
        if not folder.exists():
            continue
        config_file, metrics_file = folder / "config.json", folder / "metrics.jsonl"
        config = json.loads(config_file.read_text()) if config_file.exists() else {}
        metrics = (
            [json.loads(line) for line in metrics_file.read_text().splitlines()]
            if metrics_file.exists()
            else []
        )
        result = {
            "run": name,
            "gpu": gpu,
            "command": config.get("command"),
            "pid": config.get("pid"),
            "started_utc": config.get("started_utc"),
            "completed_updates": len(metrics),
            "training_episode_count": (
                len(metrics) * config["episodes"] if "episodes" in config else None
            ),
            "sampled_attempt_count": (
                sum(row.get("attempts", 0) * config["episodes"] for row in metrics)
                if "episodes" in config
                else None
            ),
        }
        result["measured_phase_wall_seconds"] = sum(row.get("seconds", 0) for row in metrics)
        result["timing_scope"] = (
            "summed monotonic updates; excludes process setup/shutdown and is not CUDA kernel time"
        )
        if not metrics and (folder / "summary.json").exists():
            summary = json.loads((folder / "summary.json").read_text())
            measure = (
                summary.get("benchmark_measurement", summary) if isinstance(summary, dict) else {}
            )
            result["measured_phase_wall_seconds"] = measure.get(
                "seconds", measure.get("actual_policy_wall_seconds", 0)
            )
            result["timing_scope"] = (
                "saved collection or policy execution phase only; "
                "excludes unavailable setup/checker overhead"
            )
        result["checker_cpu_process_seconds"] = sum(
            row.get("lean_process_seconds", 0) for row in metrics
        )
        result["diagnostic_checker_cpu_process_seconds"] = sum(
            row.get("first_attempt_diagnostic_lean_process_seconds", 0) for row in metrics
        )
        result["teacher_diagnostic_wall_seconds"] = sum(
            row.get("teacher_diagnostic_wall_seconds", 0) for row in metrics
        )
        episode_file = folder / "episodes.jsonl"
        if not episode_file.exists():
            episode_file = folder / "public_paths.jsonl"
        phases = dict.fromkeys(
            ("sampling_wall_seconds", "controller_wall_seconds", "diagnostic_replay_wall_seconds"),
            0.0,
        )
        if episode_file.exists():
            policy_wall = 0.0
            with episode_file.open() as stream:
                for line in stream:
                    episode = json.loads(line)
                    policy_wall += episode.get("policy_execution_wall_seconds", 0)
                    for key in phases:
                        phases[key] += episode.get(key, 0)
            if not result["measured_phase_wall_seconds"] and policy_wall:
                result["measured_phase_wall_seconds"] = policy_wall
                result["timing_scope"] = (
                    "summed saved policy execution only; excludes update/checker/setup"
                )
        result["episode_phase_seconds"] = phases
        result["unmeasured_or_incomplete"] = result["measured_phase_wall_seconds"] == 0
        results.append(result)
    print(
        json.dumps(
            {
                "owner": "/root/allocation",
                "scope": "owned runs only; root-owned cost/chooser/probe runs excluded",
                "runs": results,
                "measured_device_phase_seconds_lower_bound": sum(
                    row["measured_phase_wall_seconds"] for row in results
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
