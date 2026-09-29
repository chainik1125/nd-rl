"""Audit completed LM pilots from raw attempts and immutable manifests, CPU only."""
from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
import random
import statistics

import common
from nd_environment import Tokenizer, judge, strict_complete

ROOT = Path(__file__).resolve().parents[1]


def percentile(xs, p):
    values = sorted(xs)
    i = (len(values) - 1) * p
    return values[int(i)] * (1 - i % 1) + values[math.ceil(i)] * (i % 1)


def main():
    tok = Tokenizer()
    reports, schedules = {}, {}
    for objective in ("relative", "ppo"):
        for start in ("full", "reset"):
            name = f"pilot_{objective}_{start}_v1"
            path = ROOT / "results" / name
            result = json.loads((path / "result.json").read_text())
            manifest = json.loads((path / "manifest.json").read_text())
            cfg = manifest["config"]
            updates = common.read_jsonl(path / "updates.jsonl")
            groups = common.read_jsonl(path / "groups.jsonl")
            attempts = common.read_jsonl(path / "attempts.jsonl")
            assert len(updates) == result["step"] == 100
            assert len(attempts) == 100 * cfg["prompts_per_step"] * cfg["group_size"]
            assert len(groups) == 100 * cfg["prompts_per_step"]
            successes = Counter()
            lengths = Counter()
            for row in attempts:
                verdict = judge(row, row["response_ids"], tok)
                assert verdict["reward"] == row["reward"]
                if row["reward"]:
                    assert verdict["body"] == row["body"]
                    successes[row["training_kind"]] += 1
                    if row["training_kind"] == "target":
                        lengths[row["n_lines"]] += 1
                n = len(row["response_ids"])
                assert n == len(row["rollout_logprobs"]) == len(row["old_values"]) == len(row["advantages"])
                assert row["bootstrap_value"] == 0 and row["terminated"] and not row["truncated"]
                assert row["token_rewards"] == [0.] * (n - 1) + [float(row["reward"])]
                assert row["reset_index"] == 0 or row["training_kind"] == "source"
                if objective == "ppo":
                    assert max(abs(x - row["reward"]) for x in row["returns"]) < 1e-5
                    assert max(abs(a - (row["reward"] - v)) for a, v in zip(row["advantages"], row["old_values"])) < 1e-5
            for g, group in zip(groups, [attempts[i:i + cfg["group_size"]] for i in range(0, len(attempts), cfg["group_size"])]):
                assert {r["source_id"] for r in group} == {g["source_id"]}
                assert sum(r["reward"] for r in group) == g["successes"]
                if objective == "relative":
                    for row in group:
                        expected = row["reward"] - (g["successes"] - row["reward"]) / (cfg["group_size"] - 1)
                        assert all(abs(a - expected) < 1e-6 for a in row["advantages"])
            assert dict(successes) == result["successes"]
            discoveries = common.read_jsonl(path / "discoveries.jsonl") if (path / "discoveries.jsonl").exists() else []
            assert len(discoveries) == len({r["witness_id"] for r in discoveries}) == result["unique_target_discoveries"]
            for row in discoveries:
                assert strict_complete(row["prompt"], row["body"])["ok"]
                assert not row["admitted_for_training"]
            epochs = [e for u in updates for e in u["epochs"]]
            assert len(epochs) == 400
            first = [u["epochs"][0] for u in updates]
            assert max(abs(e["ratio_min"] - 1) for e in first) < .001
            assert max(abs(e["ratio_max"] - 1) for e in first) < .001
            assert all(math.isfinite(float(v)) for e in epochs for v in e.values())
            scheduled = [(g["step"], g["group"], g["training_kind"], g["source_id"]) for g in groups]
            schedules[name] = scheduled
            retention = common.read_jsonl(cfg["retention"])
            rng = random.Random(cfg["seed"] + 1000003)
            digest = ""
            for u in updates:
                selected = [retention[rng.randrange(len(retention))] for _ in range(cfg["retention_batch"])]
                digest = common.digest([digest, [r["source_id"] for r in selected]])
                assert u["retention_schedule_digest"] == digest
            rate = result["cumulative_flop_proxy"] / result["elapsed_seconds"]
            reports[name] = {
                "strict_raw_attempts_reverified": len(attempts), "strict_target_witnesses_reverified": len(discoveries),
                "unique_target_theorems": len({r["source_id"] for r in discoveries}),
                "target_success_written_lengths": dict(lengths), "successes": dict(successes),
                "max_behavior_abs_error": max(u["behavior_audit"]["max_abs_error"] for u in updates),
                "all_epochs_ratio_min": min(e["ratio_min"] for e in epochs),
                "all_epochs_ratio_max": max(e["ratio_max"] for e in epochs),
                "kl_median": statistics.median(e["approx_kl_per_action"] for e in epochs),
                "kl_p95": percentile([e["approx_kl_per_action"] for e in epochs], .95),
                "kl_max": max(e["approx_kl_per_action"] for e in epochs),
                "clip_fraction_median": statistics.median(e["clip_fraction"] for e in epochs),
                "clip_fraction_max": max(e["clip_fraction"] for e in epochs),
                "gradient_norm_max": max(e["grad_norm"] for e in epochs),
                "value_loss_max": max(e["value_loss"] for e in epochs),
                "retention_nll_first": updates[0]["epochs"][0]["retention_loss"],
                "retention_nll_last": updates[-1]["epochs"][-1]["retention_loss"],
                "elapsed_seconds": result["elapsed_seconds"], "generated_tokens": result["generated_tokens"],
                "flop_proxy": result["cumulative_flop_proxy"], "flop_proxy_per_second": rate,
                "projection_seconds_per_1e16_proxy": 1e16 / rate,
                "theorem_schedule_sha256": common.digest(scheduled), "retention_schedule_digest": digest,
                "curriculum_decisions": sum("curriculum_window" in g for g in groups),
                "curriculum_backward_moves": sum(g.get("curriculum_window", {}).get("new", 0) < g.get("curriculum_window", {}).get("old", 0) for g in groups),
                "result_sha256": common.sha(path / "result.json"),
                "attempts_sha256": common.sha(path / "attempts.jsonl"),
            }
    baseline = next(iter(schedules.values()))
    assert all(s == baseline for s in schedules.values())
    assert len({r["retention_schedule_digest"] for r in reports.values()}) == 1
    report = {"ok": True, "pilots": reports,
              "all_arms_same_theorem_source_target_and_retention_schedules": True,
              "projected_total_seconds_1e16_each_two_seeds": 2 * sum(r["projection_seconds_per_1e16_proxy"] for r in reports.values()),
              "limitations": ["Rates are 100-update pilot extrapolations; changing generated proof lengths changes padding and throughput.",
                              "Pilot comparisons are readiness checks, not heldout primary evidence.",
                              "Positive retention-only control is required to attribute absolute changes to RL rather than continued cap-six fitting."]}
    common.write_json(ROOT / "reports/review_pilots.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
