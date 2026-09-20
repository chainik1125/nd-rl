"""Check dashboard accuracy records against the verified comparison report."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sqlite3

from analyze import read, sha, write


def audit(root, imports, partial=False, database=None):
    results_path = root / "reports/analysis_results.json"
    results = read(results_path)
    if not partial and not results["complete"]:
        raise ValueError("Wait for all six completed arms before the final audit")
    fixed_path = root / "reports/fixed_probe_results.json"
    fixed = read(fixed_path)
    assert fixed["analysis_results_sha256"] == sha(results_path)
    protocol = read(root / "plans/protocol.json")
    expected = {}
    for point in fixed["records"]:
        key = (point["arm"], point["split"], point["step"])
        assert key not in expected
        expected[key] = {"solved": point["solved"], "problems": point["problems"],
                         "accuracy": point["accuracy"], "checkpoint_sha256": point["checkpoint_sha256"],
                         "dataset_sha256": point["data_sha256"],
                         "compute": point["total_flop_proxy"]}
    for name, arm in results["arms"].items():
        for phase in ("sft", "rl"):
            endpoint = arm["endpoints"][phase + "_greedy"]
            work = arm["training"]["rl"]["actual_flop_proxy"] if phase == "rl" else 0
            step = arm["training"]["rl"]["actual_steps"] if phase == "rl" else 0
            expected[name, "validation_textbook", step] = {
                **{k: endpoint[k] for k in ("solved", "problems", "accuracy", "checkpoint_sha256")},
                "dataset_sha256": protocol["development_data"]["sha256"], "compute": work}
    seen, records, labels = set(), [], {}
    for path in sorted(imports.glob("*/accuracy/*.json")):
        point = read(path)
        name = Path(point["evaluation_directory"]).parts[-3]
        if name not in results["arms"]:
            assert partial and name in results["inventory"]
            continue
        key = (name, point["split"], point["step"])
        assert key in expected and key not in seen, key
        seen.add(key)
        check = expected[key]
        for field in ("solved", "problems", "accuracy", "checkpoint_sha256", "dataset_sha256"):
            assert point[field] == check[field], (key, field)
        assert point["protocol"]["samples"] == 1 and point["protocol"]["temperature"] == 0
        assert point["protocol"]["max_new_tokens"] == 1536
        assert point["compute_axes"]["compute/rl_flop_proxy"] == check["compute"]
        target = results["arms"][name]["training"]["rl"]["target_flop_proxy"]
        assert math.isclose(point["compute_axes"]["compute/rl_budget_fraction"], check["compute"] / target,
                            rel_tol=1e-12, abs_tol=1e-12)
        display_path = path.parent.parent / "display.json"
        display = read(display_path)
        arm = results["arms"][name]
        label = f"{'Direct' if arm['mode'] == 'direct' else 'CoT'} cap {arm['cap']} (8.9M)"
        assert display["name"] == label
        run_id = path.parent.parent.name
        assert name not in labels or labels[name]["run_id"] == run_id
        labels[name] = {"run_id": run_id, "label": label, "display_sha256": sha(display_path)}
        records.append({"arm": name, "split": point["split"], "step": point["step"],
                        "record_path": str(path), "record_sha256": sha(path)})
    assert seen == set(expected), {"missing": sorted(set(expected) - seen)}
    assert len(records) == 12 * len(results["arms"])
    database = database or imports.parent.parent / "trackio" / (imports.name + ".db")
    db_seen = set()
    by_run = {label["run_id"]: name for name, label in labels.items()}
    with sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        for row in conn.execute("SELECT * FROM metrics WHERE log_id LIKE '%:accuracy:%'"):
            if row["run_id"] not in by_run:
                assert partial
                continue
            name = by_run[row["run_id"]]
            assert row["run_name"] == labels[name]["label"]
            metrics = json.loads(row["metrics"])
            accuracy_keys = [key for key in metrics if key.startswith("accuracy/")]
            assert len(accuracy_keys) == 1
            metric = accuracy_keys[0]
            split = metric.split("/", 1)[1]
            key = (name, split, row["step"])
            assert key in expected and key not in db_seen, key
            db_seen.add(key)
            check = expected[key]
            assert metrics[metric] == check["accuracy"]
            assert metrics["compute/rl_flop_proxy"] == check["compute"]
            target = results["arms"][name]["training"]["rl"]["target_flop_proxy"]
            assert math.isclose(metrics["compute/rl_budget_fraction"], check["compute"] / target,
                                rel_tol=1e-12, abs_tol=1e-12)
            detail = metrics["accuracy_details"]["_value"]
            assert len(detail) == 1
            for field in ("solved", "problems", "checkpoint_sha256", "dataset_sha256"):
                assert detail[0][field] == check[field], (key, field)
    assert db_seen == set(expected), {"missing_in_database": sorted(set(expected) - db_seen)}
    return {"checked_utc": datetime.now(timezone.utc).isoformat(), "complete": results["complete"],
            "analysis_results_sha256": sha(results_path), "fixed_probe_results_sha256": sha(fixed_path),
            "source_sha256": sha(Path(__file__)), "records_checked": len(records), "labels": labels,
            "database": str(database.resolve()), "native_database_points_checked": len(db_seen),
            "scope": "Imported greedy accuracy records and native SQLite metrics: readable labels, scores, checkpoints, datasets and compute axes. Browser checks are separate.",
            "records": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--imports", type=Path, required=True)
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    result = audit(root, args.imports.resolve(), args.partial)
    write(root / "reports/dashboard_accuracy_audit.json", result)
    print(json.dumps({"complete": result["complete"], "records_checked": result["records_checked"],
                      "labels": result["labels"]}))


if __name__ == "__main__":
    main()
