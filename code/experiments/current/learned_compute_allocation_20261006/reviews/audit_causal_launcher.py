"""Synthetic registry, copied DEV metadata, preflight only; no GPU/model execution."""

import copy
import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path.cwd()
W = ROOT / "code/experiments/current/learned_compute_allocation_20261006"
sys.path.insert(0, str(W / "context"))
spec = importlib.util.spec_from_file_location(
    "causal_launcher_review", W / "context/run_registered_causal.py"
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
from evaluation_freeze import signature_sha256  # noqa: E402 - runtime source path

B = ROOT / "artifacts/learned_compute_allocation_20261006"
original = json.loads((B / "freeze_inputs/selection_v1/causal_test_inventory.json").read_text())
dev = B / "context/causal_dev_panel_v1/public.jsonl"
results = {"launcher_sha256": m.causal_probe.digest(m.__file__), "cases": {}}


def write(p, v):
    p.write_text(json.dumps(v) + "\n")


for case in [
    "clean",
    "unselected_bad_condition",
    "wrong_launcher",
    "missing_analysis_registration",
    "wrong_checkpoint",
]:
    with tempfile.TemporaryDirectory(prefix="causal_launch_", dir=W / "reviews") as td:
        td = Path(td)
        public = td / "synthetic_public.jsonl"
        public.write_bytes(dev.read_bytes())
        registry = td / "registry.json"
        inventory = copy.deepcopy(original)
        inventory["public"] = str(public)
        model = None
        for e in inventory["runs"]:
            e["run"] = str(td / (str(e["training_seed"]) + "_" + e["condition"]))
            e["analysis"] = e["run"] + "_analysis"
            e["generation_registry"] = str(registry)
            sig = e["intervention_signature"]
            sig["public_sha256"] = m.causal_probe.digest(public)
            if e["training_seed"] == 101:
                ck = B / "allocation/no_cost/main_s101/checkpoint_00200.pt"
                base = B / "allocation/format_main96_lr1e5_clean8_initial50_s61/checkpoint_00999.pt"
                model = dict(
                    arm="no_cost", rl_seed=101, base_path=str(base), checkpoint_path=str(ck)
                )
                assert m.causal_probe.digest(ck) == sig["checkpoint_sha256"]
        ip = td / "inventory.json"
        write(ip, inventory)
        sigs = [m.aggregate_causal.signature(ip)]
        for e in inventory["runs"]:
            sigs.extend(
                [
                    e["intervention_signature"],
                    m.analyze_causal.analysis_signature(e["intervention_signature"]),
                ]
            )
        freeze = dict(
            status="frozen",
            selection_split="dev",
            causal_launch_helper_sha256=m.causal_probe.digest(m.__file__),
            registered_models=[model],
            registered_public_pool_sha256=[m.causal_probe.digest(public)],
            registered_analyses=[
                dict(signature=s, signature_sha256=signature_sha256(s)) for s in sigs
            ],
        )
        write(registry, freeze)
        if case == "unselected_bad_condition":
            e = next(
                e
                for e in inventory["runs"]
                if e["training_seed"] == 101 and e["condition"] == "policy_forced4_16"
            )
            e["intervention_signature"]["implementation_sha256"] = "bad"
            write(ip, inventory)
            sigs[0] = m.aggregate_causal.signature(ip)
            freeze["registered_analyses"][0] = dict(
                signature=sigs[0], signature_sha256=signature_sha256(sigs[0])
            )
            write(registry, freeze)
        elif case == "wrong_launcher":
            freeze["causal_launch_helper_sha256"] = "bad"
            write(registry, freeze)
        elif case == "missing_analysis_registration":
            freeze["registered_analyses"] = [
                e
                for e in freeze["registered_analyses"]
                if e["signature"].get("method") != "causal_intervention_analysis"
            ]
            write(registry, freeze)
        elif case == "wrong_checkpoint":
            freeze["registered_models"][0]["checkpoint_path"] = str(base)
            write(registry, freeze)
        args = SimpleNamespace(
            frozen_test_protocol=str(registry),
            inventory=str(ip),
            training_seed=101,
            condition="one_extra64" if case == "unselected_bad_condition" else None,
        )
        try:
            jobs = m.prepare(args)
        except ValueError as exc:
            r = {"rejected": True, "message": str(exc)}
        else:
            r = {
                "rejected": False,
                "jobs": len(jobs),
                "outputs_created": any(Path(e["run"]).exists() for e in inventory["runs"]),
            }
        results["cases"][case] = r
assert not results["cases"]["clean"]["rejected"] and results["cases"]["clean"]["jobs"] == 4
assert all(v["rejected"] for k, v in results["cases"].items() if k != "clean")
Path(__file__).with_name("causal_launcher_independent_audit.json").write_text(
    json.dumps(results, indent=2) + "\n"
)
print(json.dumps(results, indent=2))
