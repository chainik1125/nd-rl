"""Synthetic TEST-registration checks around copied DEV smoke; no real TEST data."""

import contextlib
import hashlib
import importlib.util
import io
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path.cwd()
W = ROOT / "code/experiments/current/learned_compute_allocation_20261006"
src = W / "context/analyze_causal.py"
spec = importlib.util.spec_from_file_location("causal_registration_audit", src)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
base = (
    ROOT / "artifacts/learned_compute_allocation_20261006/context/causal_forced4_protocol_smoke_v4"
)


def read(p):
    return json.loads(Path(p).read_text())


def write(p, v):
    Path(p).write_text(json.dumps(v) + "\n")


def sigsha(s):
    return hashlib.sha256(json.dumps(s, sort_keys=True).encode()).hexdigest()


results = {
    "analysis_sha256": m.digest(src),
    "scope": "Only copied DEV smoke outcomes and synthetic TEST metadata/registry",
    "cases": {},
}
for case in [
    "clean",
    "wrong_bootstrap",
    "wrong_registry",
    "wrong_arguments",
    "wrong_input_hash",
    "wrong_upstream",
    "wrong_helper",
    "wrong_generator",
    "split_downgrade",
]:
    with tempfile.TemporaryDirectory(prefix="causal_registration_", dir=W / "reviews") as td:
        td = Path(td)
        run = td / "run"
        run.mkdir()
        c = read(base / "config.json")
        for name in ["completion.json", "branches_scored.jsonl", "states_public.jsonl"]:
            shutil.copy2(base / name, run / name)
        snapshot = run / "runner_snapshot"
        snapshot.mkdir()
        for name in ["live.py", "controller.py", "train_controller.py", "causal_probe.py"]:
            shutil.copy2(base / "runner_snapshot" / name, snapshot / name)
        public = [json.loads(line) for line in Path(c["public"]).read_text().splitlines()]
        for r in public:
            r["split"] = "test"
        pp = td / "synthetic_public.jsonl"
        pp.write_text("".join(json.dumps(r) + "\n" for r in public))
        c["public"] = str(pp)
        c["public_sha256"] = m.digest(pp)
        c["split"] = "test"
        protocol = {
            k: c[k]
            for k in [
                "limit",
                "replicates",
                "continuation",
                "forced_extra",
                "history_removal_limit",
                "counts",
                "seed",
                "context",
                "attempt_bound",
                "state_reserve",
                "max_attempts",
                "max_action",
                "max_steps",
                "temperature",
            ]
        }
        intervention = {
            "analysis_sha256": m.digest(src),
            "protocol": protocol,
            "base_sha256": c["base_sha256"],
            "checkpoint_sha256": c["checkpoint_sha256"],
            "public_sha256": c["public_sha256"],
            "source_hashes": c["upstream_hashes"].copy(),
            "runner_hashes": {
                n: m.digest(snapshot / n)
                for n in ["live.py", "controller.py", "train_controller.py"]
            },
            "implementation_sha256": m.digest(snapshot / "causal_probe.py"),
        }
        c["registered_signature"] = intervention
        analysis = m.analysis_signature(intervention, 20, 81006)
        freeze = {
            "status": "frozen",
            "selection_split": "dev",
            "registered_public_pool_sha256": [m.digest(pp)],
            "registered_analyses": [{"signature": analysis, "signature_sha256": sigsha(analysis)}],
        }
        fp = td / "synthetic_registry.json"
        write(fp, freeze)
        c["frozen_test_protocol_sha256"] = m.digest(fp)
        bootstrap = "20"
        if case == "wrong_bootstrap":
            bootstrap = "21"
        elif case == "wrong_registry":
            fp.write_text(fp.read_text() + " ")
        elif case == "wrong_arguments":
            c["seed"] += 1
        elif case == "wrong_input_hash":
            c["checkpoint_sha256"] = "0" * 64
        elif case == "wrong_upstream":
            c["upstream_hashes"]["model.py"] = "0" * 64
        elif case == "wrong_helper":
            (snapshot / "live.py").write_text("# corrupted\n")
        elif case == "wrong_generator":
            (snapshot / "causal_probe.py").write_text("# corrupted\n")
        elif case == "split_downgrade":
            c["split"] = "dev"
        write(run / "config.json", c)
        opened = []
        original = m.load_rows

        def tracking(path, opened=opened, original=original):
            opened.append(Path(path).name)
            return original(path)

        m.load_rows = tracking
        old = sys.argv
        sys.argv = [
            "analyze_causal.py",
            "--run",
            str(run),
            "--out",
            str(td / "out"),
            "--bootstrap",
            bootstrap,
            "--frozen-test-protocol",
            str(fp),
        ]
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                m.main()
        except ValueError as exc:
            result = {"rejected": True, "message": str(exc)}
        else:
            result = {"rejected": False}
        finally:
            sys.argv = old
            m.load_rows = original
        result["scored_branches_opened"] = "branches_scored.jsonl" in opened
        results["cases"][case] = result
assert not results["cases"]["clean"]["rejected"]
assert all(
    v["rejected"] and not v["scored_branches_opened"]
    for k, v in results["cases"].items()
    if k != "clean"
)
Path(__file__).with_name("causal_registration_audit.json").write_text(
    json.dumps(results, indent=2) + "\n"
)
print(json.dumps(results, indent=2))
