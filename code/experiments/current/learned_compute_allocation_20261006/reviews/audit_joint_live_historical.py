"""Independent historical live audit requiring explicit all-nine release."""

import argparse
import hashlib
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from audit_joint_live_paths import audit, read, rows

ROOT = Path(__file__).resolve().parents[5]
ART = ROOT / "artifacts/learned_compute_allocation_20261006"
OUT = Path(__file__).with_name("joint_live_historical_independent_audit.json")


def linprog(objective, A_eq, b_eq, **_kwargs):
    solutions = []
    for size in range(1, min(A_eq.shape) + 1):
        for support in combinations(range(len(objective)), size):
            vector, *_ = np.linalg.lstsq(A_eq[:, support], b_eq, rcond=None)
            if min(vector) >= -1e-9 and np.allclose(
                A_eq[:, support] @ vector, b_eq, rtol=1e-9, atol=1e-8
            ):
                solutions.append(float(objective[list(support)] @ vector))
    return SimpleNamespace(success=bool(solutions), fun=min(solutions) if solutions else None)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


parser = argparse.ArgumentParser()
parser.add_argument("--root-historical-release", required=True)
parser.add_argument("--expected-summary-sha256", required=True)
args = parser.parse_args()
assert args.root_historical_release.strip()
registry = read(ART / "final_evaluation/historical_protocol.json")
manifest = read(ART / "freeze_inputs/allocation_v1/historical_live_runs.json")["runs"]
assert len(manifest) == 9
for run in manifest:
    launch = read(Path(run["paths"]).with_suffix(".launch.json"))
    assert launch["status"] == "complete" and all(x["returncode"] == 0 for x in launch["phases"])
    assert launch["registry_sha256"] == sha(ART / "final_evaluation/historical_protocol.json")
summary_path = ART / "allocation/final_historical_summary_fp32_v2.json"
assert sha(summary_path) == args.expected_summary_sha256
historical_summary = read(summary_path)
summary = historical_summary["combined"]
summary["provenance"] = historical_summary["provenance"]
metadata = rows(ART / "data/textbook72_historical_public_v1/public.jsonl")
families = {r["theorem_id"]: r["family"] for r in metadata}
ids = sorted(families)
assert len(ids) == 72 and ids == summary["theorem_ids"]
names = list(next(iter(next(iter(summary["arms"].values())).values())))
primary_arms = ("cost", "no_cost")
tables = defaultdict(dict)
oracles = defaultdict(list)
path_audits = []
mixtures = []
checkpoint_ages = {}
lean_sample = []
max_point_error = 0.0
for run in manifest:
    arm, seed = run["arm"], run["seed"]
    model = next(
        m for m in registry["registered_models"] if m["arm"] == arm and m["rl_seed"] == seed
    )
    directory = Path(run["paths"])
    launch = read(directory.with_suffix(".launch.json"))
    assert launch["status"] == "complete" and all(x["returncode"] == 0 for x in launch["phases"])
    config = read(directory / "config.json")
    assert config["checkpoint_sha256"] == model["checkpoint_sha256"]
    assert config["base_sha256"] == model["base_sha256"]
    assert config["inference_sha256"] == model["inference_sha256"]
    assert config["seed"] == 92006 and config["repeats"] == 2 and config["evaluated_theorems"] == 72
    assert config["frozen_protocol_sha256"] == summary["provenance"]["frozen_protocol_sha256"]
    checkpoint_ages[f"{arm}:{seed}"] = config["checkpoint"]
    prov = summary["provenance"]["inputs"][f"{arm}:{seed}"]
    for name, filename in [
        ("config", "config.json"),
        ("integrated", "integrated.json"),
        ("private_labels", "private_labels.jsonl"),
    ]:
        assert sha(directory / filename) == prov[name + "_sha256"]
    for name in ("counter", "budgets"):
        if name in run:
            assert sha(run[name]) == prov[name + "_sha256"]
    result = audit(
        directory, True, run.get("counter"), run.get("budgets"), model.get("counter_hazards_path")
    )
    path_audits.append(result)
    print("raw audit", arm, seed, result["policy_rows"], flush=True)
    integrated = read(directory / "integrated.json")
    for name in ("counter", "budgets"):
        if name in run:
            derived = read(run[name])
            assert derived["provenance"]["paths_sha256"] == sha(directory / "public_paths.jsonl")
            assert derived["provenance"]["labels_sha256"] == sha(directory / "private_labels.jsonl")
            integrated.update(derived["integrated"])
    if "budgets" in run:
        fit = read(run["budgets"])["fit"]
        assert fit == read(model["budget_baselines_path"])
        assert sha(model["budget_baselines_path"]) == model["budget_baselines_sha256"]
        assert fit["fit_split"] == "dev"
        dev = fit["dev_summaries"]
        reward = np.array([dev[f"fixed{k}"]["reward"] for k in (1, 2, 4, 8)])
        for name, fields in [
            ("attempt_matched_mixture", ["attempts"]),
            ("forward_token_matched_mixture", ["total_forward_tokens"]),
            ("joint_matched_mixture", ["attempts", "total_forward_tokens"]),
        ]:
            a = np.array(
                [[1.0] * 4] + [[dev[f"fixed{k}"][f] for k in (1, 2, 4, 8)] for f in fields]
            )
            b = np.array([1.0] + [dev["learned"][f] for f in fields])
            opt = linprog(-reward, A_eq=a, b_eq=b, bounds=(0, None), method="highs")
            assert bool(opt.success) == fit[name]["feasible"]
            if opt.success:
                w = np.array([fit[name]["weights"][str(k)] for k in (1, 2, 4, 8)])
                assert np.allclose(a @ w, b, rtol=1e-8, atol=1e-8)
                assert abs(reward @ w + opt.fun) < 1e-8
            mixtures.append(
                {"arm": arm, "seed": seed, "method": name, "feasible": bool(opt.success)}
            )
    for method, records in integrated.items():
        lookup = {}
        for r in records:
            key = r["theorem_id"], r["repeat"]
            assert key not in lookup
            w = r["expected_policy_work"]
            vals = dict(
                correct_probability=r["correct_probability"],
                attempts=r["attempts"],
                reward_native=r["reward"],
                forced_termination_probability=r["forced_termination_probability"],
                generated_tokens=w["generated_tokens"],
                generator_forward_tokens=w["forward_tokens"],
                controller_forward_tokens=w["controller_forward_tokens"],
                total_forward_tokens=w["forward_tokens"] + w["controller_forward_tokens"],
                generator_forward_calls=w["forward_calls"],
                controller_forward_calls=w["controller_forward_calls"],
                total_forward_calls=w["forward_calls"] + w["controller_forward_calls"],
                generator_attention_cells_per_layer=w["attention_score_cells_per_layer"],
                controller_attention_cells_per_layer=w[
                    "controller_attention_score_cells_per_layer"
                ],
                total_attention_cells_per_layer=w["attention_score_cells_per_layer"]
                + w["controller_attention_score_cells_per_layer"],
                model_section_wall_seconds=w["policy_wall_seconds"],
            )
            for c in (0.0, 0.01, 0.03, 0.1):
                vals[f"utility_at_cost_{c}"] = r["correct_probability"] - c * (r["attempts"] - 1)
            assert (
                abs(r["reward"] - (r["correct_probability"] - config["cost"] * (r["attempts"] - 1)))
                < 1e-12
            )
            lookup[key] = np.array([vals[n] for n in names])
        assert set(lookup) == {(tid, r) for tid in ids for r in range(2)}
        array = np.array([(lookup[tid, 0] + lookup[tid, 1]) / 2 for tid in ids])
        tables[arm, seed][method] = array
        target = summary["per_seed"][f"{arm}:{seed}"][method]
        max_point_error = max(
            max_point_error,
            max(
                abs(array.mean(0)[j] - target[n]) / max(1, abs(target[n]))
                for j, n in enumerate(names)
            ),
        )
    labelrows = rows(directory / "private_labels.jsonl")
    labelmap = {(r["theorem_id"], r["repeat"]): r["validities"] for r in labelrows}
    assert len(labelmap) == len(labelrows) == 144
    oracles[arm].append(
        np.array(
            [
                [np.mean([any(labelmap[tid, r][:k]) for r in (0, 1)]) for k in (1, 2, 4, 8)]
                for tid in ids
            ]
        )
    )
    chosen = sorted(
        ids, key=lambda tid: hashlib.sha256(f"{arm}:{seed}|{tid}|repeat0".encode()).hexdigest()
    )[:2]
    # Deterministic sample identities are committed without inspecting their labels here.
    lean_sample += [
        {
            "arm": arm,
            "seed": seed,
            "directory": str(directory),
            "theorem_id": tid,
            "repeat": 0,
            "candidate_rule": "first and last, deduplicate",
        }
        for tid in chosen
    ]

arms = {
    arm: {
        method: np.mean([methods[method] for (a, _), methods in tables.items() if a == arm], axis=0)
        for method in set.intersection(*(set(m) for (a, _), m in tables.items() if a == arm))
    }
    for arm in sorted({a for a, _ in tables})
}
checks = 0
max_ci_error = 0.0
cohort_checks = {}
for cohort, target_summary in [("combined", historical_summary["combined"])] + list(
    historical_summary["by_source_role"].items()
):
    selected = [i for i, tid in enumerate(ids) if tid in set(target_summary["theorem_ids"])]
    expected = 72 if cohort == "combined" else 58 if cohort == "textbook_dev" else 14
    assert len(selected) == expected
    rng = np.random.default_rng(102006)
    weights = np.zeros((10000, expected))
    roles = [families[ids[i]] for i in selected]
    for role in sorted(set(roles)):
        ix = np.flatnonzero(np.array(roles) == role)
        weights[:, ix] = (
            rng.multinomial(len(ix), np.full(len(ix), 1 / len(ix)), size=10000) / expected
        )

    def verify(array, target, fields=names, selected=selected, weights=weights):
        global checks, max_ci_error
        values = array[selected]
        center = values.mean(0)
        interval = np.quantile(weights @ values, [0.025, 0.975], axis=0)
        for j, field in enumerate(fields):
            t = target[field]
            error = max(
                abs(center[j] - t["mean"]),
                abs(interval[0, j] - t["ci_low"]),
                abs(interval[1, j] - t["ci_high"]),
            ) / max(1, abs(t["mean"]), abs(t["ci_low"]), abs(t["ci_high"]))
            max_ci_error = max(max_ci_error, error)
            checks += 1

    for arm, methods in arms.items():
        for method, array in methods.items():
            verify(array, target_summary["arms"][arm][method])
        verify(
            np.mean(oracles[arm], axis=0),
            target_summary["finite_path_oracle_availability"][arm],
            ["after1", "after2", "after4", "after8"],
        )
    for (arm, seed), methods in tables.items():
        for method, array in methods.items():
            target = target_summary["per_seed"][f"{arm}:{seed}"][method]
            for j, field in enumerate(names):
                assert np.isclose(array[selected, j].mean(), target[field], rtol=1e-12, atol=1e-12)
    for key, target in target_summary["contrasts"].items():
        label = key.removeprefix("secondary:")
        if label == "cost-minus-no_cost":
            array = arms["cost"]["learned"] - arms["no_cost"]["learned"]
        else:
            arm, comparison = label.split(":")
            left, right = comparison.split("-minus-")
            rightarm = arm
            for control in ("format_only", "fixed1_control", "fixed4_control"):
                if right.startswith(control + "-"):
                    rightarm = control
                    right = right[len(control) + 1 :]
                    break
            array = arms[arm][left] - arms[rightarm][right]
        fields = list(target)
        verify(array[:, [names.index(n) for n in fields]], target, fields)
    cohort_checks[cohort] = expected
assert max(max_point_error, max_ci_error) < 1e-11
result = {
    "root_release": args.root_historical_release,
    "models": len(manifest),
    "policy_rows": sum(r["policy_rows"] for r in path_audits),
    "path_audits": path_audits,
    "cohort_sizes": cohort_checks,
    "aggregate_metric_interval_checks": checks,
    "max_relative_point_error": max_point_error,
    "max_relative_interval_error": max_ci_error,
    "dev_mixtures": mixtures,
    "selected_checkpoint_paths": checkpoint_ages,
    "summary_sha256": sha(summary_path),
}
OUT.write_text(json.dumps(result, indent=2) + "\n")
print(
    json.dumps(
        {
            k: v
            for k, v in result.items()
            if k not in ("path_audits", "dev_mixtures", "selected_checkpoint_paths")
        },
        indent=2,
    )
)
