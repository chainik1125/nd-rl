"""Independent arithmetic and training-provenance audit; DEV/TRAIN only."""

import hashlib
import json
import statistics as st
from pathlib import Path

import torch

torch.set_num_threads(2)
B = Path("artifacts/learned_compute_allocation_20261006")
OUT = Path(__file__).with_name("staged_v3_cost_training_independent_audit.json")


def read(p):
    return json.loads(p.read_text())


def rows(p):
    return [json.loads(line) for line in p.open()]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


report = {"staged": {}, "training": {}}
for name in ["threshold1600_main_dev_v3"] + [f"bce_threshold_main_dev_s{s}_v3" for s in range(3)]:
    d = B / "analysis" / name
    f = read(d / "frozen_policy.json")
    sm = read(d / "summary.json")
    rs = rows(d / "per_theorem.jsonl")
    idx = {(r["theorem_id"], r["policy"]): r for r in rs}
    err = []
    for r in rs:
        err.append(
            abs(
                r["total_model_input_tokens_lower_bound"]
                - r["generator_input_tokens_lower_bound"]
                - r["required_feature_replay_tokens_unpadded"]
                - r.get("critic_prefill_tokens_proxy", 0)
            )
        )
        if r["policy"] == "fixed1":
            assert (
                r["required_feature_replay_tokens_unpadded"]
                == r.get("critic_prefill_tokens_proxy", 0)
                == 0
            )
        pol = f["policies"][r["policy"]]
        if "mixture" in pol:
            m = pol["mixture"]
            w = m["right_weight"]
            left = idx[r["theorem_id"], f"fixed{m['left_count']}"]
            right = idx[r["theorem_id"], f"fixed{m['right_count']}"]
            for k, v in r.items():
                if isinstance(v, (float, int)):
                    err.append(abs(v - ((1 - w) * left[k] + w * right[k])))
    for policy, metric in sm["metrics"].items():
        rr = [r for r in rs if r["policy"] == policy]
        for k, v in metric.items():
            if isinstance(v, (float, int)):
                err.append(abs(v - st.fmean(r[k] for r in rr)))
    choice = f["choices"]["0.03"]
    t = sm["metrics"][choice["threshold"]]
    m = sm["metrics"][choice["matched_mixtures"]["total_model_input_tokens_lower_bound"]]
    report["staged"][name] = {
        "max_arithmetic_error": max(err),
        "rows": len(rs),
        "cost03_threshold": choice["threshold"],
        "threshold": t,
        "total_work_matched_fixed_mixture": m,
        "accuracy_gap": t["accuracy"] - m["accuracy"],
        "matched_work_residual": t["total_model_input_tokens_lower_bound"]
        - m["total_model_input_tokens_lower_bound"],
    }
configs = []
for seed in [101, 102]:
    d = B / "allocation/cost" / f"main_s{seed}"
    c = read(d / "config.json")
    configs.append(c)
    ms = rows(d / "metrics.jsonl")
    es = rows(d / "episodes.jsonl")
    errors = []
    assert len(ms) == 200 and len(es) == 1600
    assert all(sha(d / "runner_snapshot" / n) == h for n, h in c["runner_hashes"].items())
    assert all(sha(Path(c["source"]) / n) == h for n, h in c["source_hashes"].items())
    assert sha(Path(c["base"])) == c["base_sha256"] and sha(Path(c["data"])) == c["data_sha256"]
    for i, metric in enumerate(ms):
        ep = es[i * 8 : (i + 1) * 8]
        assert all(e["update"] == i for e in ep)
        for k, ek in [
            ("accuracy", "correct"),
            ("attempts", "attempts"),
            ("reward", "reward"),
            ("first_attempt_accuracy_diagnostic", "first_attempt_correct_diagnostic"),
        ]:
            errors.append(abs(metric[k] - st.fmean(e[ek] for e in ep)))
        for e in ep:
            errors.append(abs(e["reward"] - (e["correct"] - 0.03 * (e["attempts"] - 1))))
    windows = {}
    for lo, hi in [(0, 50), (50, 100), (100, 150), (150, 200)]:
        ee = es[lo * 8 : hi * 8]
        p = [e["decisions"][0]["accept_probability"] for e in ee]
        windows[f"{lo + 1}-{hi}"] = {
            "episodes": len(ee),
            "attempts": st.fmean(e["attempts"] for e in ee),
            "one_attempt_fraction": st.fmean(e["attempts"] == 1 for e in ee),
            "accuracy": st.fmean(e["correct"] for e in ee),
            "first_accuracy": st.fmean(e["first_attempt_correct_diagnostic"] for e in ee),
            "first_accept_mean": st.fmean(p),
            "first_accept_range": [min(p), max(p)],
            "first_accept_sd": st.pstdev(p),
        }
    checkpoints = {}
    for n in [50, 100, 150, 200]:
        path = d / f"checkpoint_{n:05}.pt"
        q = torch.load(path, map_location="cpu", weights_only=True)
        assert q["config"] == c and q["completed_updates"] == n and q["update"] == n - 1
        assert all(torch.isfinite(v).all() for v in q["state"].values())
        checkpoints[str(n)] = {
            "sha256": sha(path),
            "parameters": sum(v.numel() for v in q["state"].values()),
        }
    report["training"][str(seed)] = {
        "max_metric_error": max(errors),
        "windows": windows,
        "checkpoint_checks": checkpoints,
        "max_replay_error": max(m["max_replay_logprob_error"] for m in ms),
    }
report["configuration_differences"] = [k for k in configs[0] if configs[0][k] != configs[1][k]]
OUT.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({k: v for k, v in report.items() if k != "training"}, indent=2))
for s, v in report["training"].items():
    print(
        s,
        json.dumps(
            {
                "max_metric_error": v["max_metric_error"],
                "windows": v["windows"],
                "max_replay_error": v["max_replay_error"],
            }
        ),
    )
