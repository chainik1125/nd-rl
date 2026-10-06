"""Temporary-copy adversarial checks of the staged aggregator; DEV artifacts only."""

import contextlib
import copy
import importlib.util
import io
import json
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path.cwd()
WORK = ROOT / "code/experiments/current/learned_compute_allocation_20261006"
B = ROOT / "artifacts/learned_compute_allocation_20261006/analysis"
source = WORK / "analysis/aggregate_staged.py"
spec = importlib.util.spec_from_file_location("staged_guard_audit", source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def read(p):
    return json.loads(Path(p).read_text())


def write(p, v):
    Path(p).write_text(json.dumps(v) + "\n")


inventory = read(B / "staged_dev_inventory.json")
public = ROOT / "artifacts/learned_compute_allocation_20261006/pools/main_dev/public.jsonl"
public_rows = [json.loads(line) for line in public.open()]
results = {"source_sha256": module.digest(source), "clean_runs": [], "mutations": {}}
for run in inventory["runs"]:
    module.load_run(run, public_rows, None)
    results["clean_runs"].append([run["group"], run["training_seed"]])
for case in [
    "omit_seed",
    "relabel_checkpoint",
    "alter_summary",
    "nonfinite_fixed_metric",
    "duplicate_public_id",
    "duplicate_source_public_id",
]:
    with tempfile.TemporaryDirectory(prefix="staged_guard_", dir=WORK / "reviews") as temp:
        temp = Path(temp)
        inv = copy.deepcopy(inventory)
        pub = public
        if case == "omit_seed":
            inv["runs"].pop()
        elif case == "relabel_checkpoint":
            inv["runs"][2] = dict(inv["runs"][1], training_seed=1)
        elif case == "duplicate_public_id":
            lines = public.read_text().splitlines()
            lines[1] = lines[0]
            pub = temp / "public.jsonl"
            pub.write_text("\n".join(lines) + "\n")
        else:
            original = Path(inv["runs"][0]["evaluation"])
            folder = temp / "evaluation"
            folder.mkdir()
            for name in ["summary.json", "per_theorem.jsonl"]:
                shutil.copy2(original / name, folder / name)
            (folder / "source_snapshot").symlink_to(
                (original / "source_snapshot").resolve(), target_is_directory=True
            )
            inv["runs"][0]["evaluation"] = str(folder)
            if case == "alter_summary":
                report = read(folder / "summary.json")
                report["metrics"]["fixed4"]["accuracy"] += 0.01
                write(folder / "summary.json", report)
            elif case == "nonfinite_fixed_metric":
                rr = [json.loads(line) for line in (folder / "per_theorem.jsonl").open()]
                next(r for r in rr if r["policy"] == "fixed4")[
                    "total_model_input_tokens_lower_bound"
                ] = float("nan")
                (folder / "per_theorem.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rr))
            else:
                report = read(folder / "summary.json")
                lines = Path(report["inputs"]["public"]["path"]).read_text().splitlines()
                lines[1] = lines[0]
                duplicate = temp / "source_public.jsonl"
                duplicate.write_text("\n".join(lines) + "\n")
                report["inputs"]["public"] = {
                    "path": str(duplicate),
                    "sha256": module.digest(duplicate),
                }
                write(folder / "summary.json", report)
        invpath = temp / "inventory.json"
        write(invpath, inv)
        args = SimpleNamespace(
            public=str(pub),
            inventory=str(invpath),
            out=str(temp / "output"),
            split="dev",
            frozen_test_protocol=None,
            bootstrap=20,
            seed=61006,
        )
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                module.aggregate(args)
        except ValueError as exc:
            results["mutations"][case] = {
                "rejected": True,
                "message": str(exc),
                "output_created": Path(args.out).exists(),
            }
        else:
            results["mutations"][case] = {"rejected": False}
assert all(v["rejected"] and not v["output_created"] for v in results["mutations"].values())
v1 = read(B / "staged_dev_aggregate_v1/metrics.json")
v2 = read(B / "staged_dev_aggregate_v2/metrics.json")
results["v2_estimates_unchanged"] = all(
    v2["groups"][g][cost][key] == value
    for g, cs in v1["groups"].items()
    for cost, data in cs.items()
    for key, value in data.items()
)
assert results["v2_estimates_unchanged"]
Path(__file__).with_name("staged_aggregation_guard_audit.json").write_text(
    json.dumps(results, indent=2) + "\n"
)
print(json.dumps(results, indent=2))
