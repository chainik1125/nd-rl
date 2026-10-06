"""DEV smoke mutations only; no generation, final outcomes, or implementation edits."""

import contextlib
import copy
import importlib.util
import io
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path.cwd()
W = ROOT / "code/experiments/current/learned_compute_allocation_20261006"
source = W / "context/analyze_causal.py"
spec = importlib.util.spec_from_file_location("causal_guard_audit", source)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
base = (
    ROOT / "artifacts/learned_compute_allocation_20261006/context/causal_forced4_protocol_smoke_v4"
)
inputs = {
    name: (base / name).read_text()
    for name in ["config.json", "completion.json", "branches_scored.jsonl", "states_public.jsonl"]
}
results = {"source_sha256": m.digest(source), "cases": {}}


def rows(txt):
    return [json.loads(line) for line in txt.splitlines()]


for case in [
    "clean",
    "missing_branch",
    "missing_state_and_branch",
    "accept_out_of_bounds",
    "wrong_branch_family",
    "duplicate_state_alias",
    "completion_state_mismatch",
    "negative_new_count",
    "wrong_forced_requested",
    "wrong_replicate_id",
]:
    with tempfile.TemporaryDirectory(prefix="causal_guard_", dir=W / "reviews") as td:
        d = Path(td)
        run = d / "run"
        run.mkdir()
        config = json.loads(inputs["config.json"])
        comp = json.loads(inputs["completion.json"])
        bs = rows(inputs["branches_scored.jsonl"])
        ss = rows(inputs["states_public.jsonl"])
        if case in ["missing_branch", "missing_state_and_branch"]:
            removed = bs.pop()
            comp["branches"] -= 1
            if case == "missing_state_and_branch":
                ss = [s for s in ss if s["state_id"] != removed["state_id"]]
                comp["states"] -= 1
        elif case == "accept_out_of_bounds":
            bs[0]["accept_probability"] = 2.0
        elif case == "wrong_branch_family":
            bs[0]["family"] = "fake_family"
        elif case == "duplicate_state_alias":
            added = copy.deepcopy(bs[0])
            added["state_id"] += "alias"
            bs.append(added)
            comp["branches"] += 1
        elif case == "completion_state_mismatch":
            comp["states"] += 1
        elif case == "negative_new_count":
            bs[0]["new_attempt_count"] = -1
        elif case == "wrong_forced_requested":
            bs[0]["forced_extra_requested"] = 1
        elif case == "wrong_replicate_id":
            bs[0]["repeat"] = 999
        for name, obj in [("config.json", config), ("completion.json", comp)]:
            (run / name).write_text(json.dumps(obj))
        for name, rs in [("branches_scored.jsonl", bs), ("states_public.jsonl", ss)]:
            (run / name).write_text("".join(json.dumps(r) + "\n" for r in rs))
        old = sys.argv
        sys.argv = [
            "analyze_causal.py",
            "--run",
            str(run),
            "--out",
            str(d / "out"),
            "--bootstrap",
            "20",
        ]
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                m.main()
        except (ValueError, KeyError) as exc:
            result = {"rejected": True, "error": str(exc)}
        else:
            summary = json.loads((d / "out/summary.json").read_text())
            result = {
                "rejected": False,
                "complete_panel": summary["complete_prespecified_state_panel"],
                "states": summary["pooled_actual_history"]["delta_continuation"]["states"],
            }
        finally:
            sys.argv = old
        results["cases"][case] = result
Path(__file__).with_name("causal_analysis_guard_audit.json").write_text(
    json.dumps(results, indent=2) + "\n"
)
print(json.dumps(results, indent=2))
