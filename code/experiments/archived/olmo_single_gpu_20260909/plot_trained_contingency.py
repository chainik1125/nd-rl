"""Three complete trained endpoints only; no five-arm identity or raw-output reads."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

from plot_budget_pilot import ANALYZER_SHA, COHORTS, SPRINT, digest, metric_check, require

ARMS = ["sft", "dpo", "rl"]
LABELS = ["SFT", "SFT → DPO", "SFT → DPO → RL"]
FREEZE = "plans/trained_contingency_source_freeze_v1.json"
FREEZE_SHA = "1bbf4eee9a0a81967af388f4f8cc6b2aab3d9707cbe8955e70b41a6672a8890a"
PLOT_METRIC_SHA = "0063a02be8401fe22d25834f2a35a8a2f4a64070006c9b58408a0b652266615a"


def validate(identity, summary, rows, manifest, statuses):
    require(identity["kind"] == "complete_trained_subset_of_incomplete_five_arm_pilot"
            and identity["status"] == "complete_for_three_trained_arms_only" and identity["arms"] == ARMS
            and identity["strictly_replayed_attempts"] == 72 and identity["analysis_exit_code"] == 0,
            "Expected an audited complete three-trained-arm contingency")
    require(all(identity[key] is False for key in ("original_five_arm_pilot_complete", "original_primary120_complete",
            "zero_or_partial_icl_in_comparisons", "pooled24_headline")), "Broader completion/comparison claim forbidden")
    require(0 <= identity["icl_attempts_committed"] < identity["icl_planned_attempts"] == 24, "Expected unfinished ICL metadata")
    require(identity["cohort_counts"] == manifest["cohort_counts"] == COHORTS
            and manifest["cohort_order"] == list(COHORTS), "Cohort counts/order mismatch")
    require(summary["status"] == "complete" and summary["analyzer_sha256"] == ANALYZER_SHA
            and summary["raw_success_is_primary"] is True, "Expected unchanged completed strict analyzer")
    require([run["arm"] for run in summary["runs"]] == ARMS and len(rows) == 72
            and [row["arm"] for row in rows] == [arm for arm in ARMS for _ in range(24)], "Need all 72 ordered trained outcomes")
    expected = [(cohort, key) for cohort in COHORTS for key in manifest["selection"]["eval_" + cohort + ".jsonl"]["canonical_hashes"]]
    require(len(expected) == len({key for _, key in expected}) == 24, "Frozen target membership is not unique")
    reference, panels = rows[:24], {}
    for index, run in enumerate(summary["runs"]):
        block, status = rows[index * 24:(index + 1) * 24], statuses[run["arm"]]
        require([(row["cohort"], row["canonical_hash"]) for row in block] == expected, "Target membership/order changed or duplicated")
        require(run["status"] == status["status"] == "complete" and run["pending_theorem_mode_groups"] == 0
                and run["planned_attempts"] == run["observed_attempts"] == status["attempts"] == status["planned_attempts"] == 24,
                "Missing/partial endpoints cannot be drawn as failures")
        require(run["settings"]["modes"] == ["greedy"] and run["settings"]["max_new_tokens"] == 2048
                and run["demonstrations_per_query"] == 0, "Expected trained endpoints without ICL, greedy cap2048")
        require(set(run["groups"]) == set(COHORTS), "Unexpected summary cohort")
        for row, original in zip(block, reference, strict=True):
            require(row["mode"] == "greedy" and row["metric"] == "greedy_pass1"
                    and type(row["attempts"]) is int and row["attempts"] == 1
                    and type(row["solved"]) is bool and type(row["successful_attempts"]) is int
                    and row["successful_attempts"] == int(row["solved"]), "Invalid or missing completed greedy outcome")
            require(all(row[key] == original[key] for key in ("family", "reference_lines")), "Cross-arm target metadata differs")
        for cohort, n in COHORTS.items():
            selected = [row for row in block if row["cohort"] == cohort]
            selection = manifest["selection"]["eval_" + cohort + ".jsonl"]
            require(len(selected) == selection["rows"] == n, "Cohort denominator changed")
            require(dict(Counter(str(row["reference_lines"]) for row in selected)) == selection["reference_length_counts"]
                    and dict(Counter(row["family"] for row in selected)) == selection["family_counts"], "Frozen target strata changed")
            require(set(run["groups"][cohort]) == {"greedy"}, "Unexpected decode mode")
            group = run["groups"][cohort]["greedy"]
            metric_check(group["overall"], selected, True)
            for field, key in (("by_reference_length", "reference_lines"), ("by_family", "family")):
                require(set(group[field]) == {str(row[key]) for row in selected}, "Summary bins mismatch")
                for value, metric in group[field].items():
                    metric_check(metric, [row for row in selected if str(row[key]) == value])
            if cohort not in panels:
                panels[cohort] = {"targets": [{key: row[key] for key in ("canonical_hash", "reference_lines", "family")} for row in selected], "methods": []}
            panels[cohort]["methods"].append({"arm": run["arm"], "label": LABELS[index], "n": n,
                "correct": sum(row["solved"] for row in selected), "outcomes": [int(row["solved"]) for row in selected]})
    require(manifest["selection"]["eval_long_panel.jsonl"]["reference_length_counts"] == {str(n): 2 for n in range(7, 15)}, "Expected n=2 per long reference length")
    return panels


def draw(panels, out, icl_committed):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    fig = plt.figure(figsize=(13.5, 6.8), facecolor="white")
    fig.suptitle("Three completed trained endpoints — separate 24-target contingency", fontsize=15, fontweight="bold", y=.98)
    fig.text(.5, .924, f"Original five-arm pilot incomplete: ICL {icl_committed}/24 committed; zero-shot excluded from this comparison.", ha="center", fontsize=11)
    fig.text(.5, .88, "OLMo 32B · one greedy attempt per theorem · 2,048-token generation cap · no ICL in these three rows", ha="center", fontsize=10)
    for cohort, bottom, width in (("long_panel", .565, .70), ("short_panel", .245, .35)):
        panel, n = panels[cohort], COHORTS[cohort]
        axis = fig.add_axes([.16, bottom, width, .165])
        axis.imshow([method["outcomes"] for method in panel["methods"]], cmap=ListedColormap(["#edf0f2", "#146b73"]), vmin=0, vmax=1, aspect="auto")
        axis.set_yticks(range(3), LABELS, fontsize=10)
        code = "L" if cohort == "long_panel" else "S"
        axis.set_xticks(range(n), [f"{code}{i+1:02d}\nref {row['reference_lines']}" for i, row in enumerate(panel["targets"])], fontsize=8)
        axis.tick_params(length=0, pad=6)
        axis.set_title(("Long-reference problems (7–14 lines)" if code == "L" else "Short-reference problems (2–6 lines)") + f" · n = {n}", loc="left", pad=23, fontsize=11)
        axis.text(n - .18, -.9, "Correct / n", fontsize=9)
        for index, method in enumerate(panel["methods"]):
            axis.text(n - .18, index, f"{method['correct']} / {n}", va="center", fontsize=11, fontweight="bold")
            for column, value in enumerate(method["outcomes"]):
                axis.text(column, index, str(value), ha="center", va="center", color="white" if value else "#68737d", fontsize=10)
        axis.set_xticks([i - .5 for i in range(n + 1)], minor=True)
        axis.set_yticks([i - .5 for i in range(4)], minor=True)
        axis.grid(which="minor", color="white", linewidth=2)
        axis.tick_params(which="minor", length=0)
        for spine in axis.spines.values():
            spine.set_visible(False)
    fig.legend(handles=[Patch(facecolor="#146b73", label="1: strict valid proof"), Patch(facecolor="#edf0f2", edgecolor="#aeb6bc", label="0: completed attempt, not valid")],
               loc="lower left", bbox_to_anchor=(.155, .12), ncol=2, frameon=False, fontsize=10)
    fig.text(.16, .102, "Same theorem columns across methods, in fixed target order; no outcome sorting. Missing methods are not scored as failures.", fontsize=9)
    fig.text(.16, .074, "Reference lengths are upper bounds, not minimum complexity. Only n = 2 per long length; descriptive single-seed results.", fontsize=9)
    long = panels["long_panel"]["methods"]
    same_long = all(method["outcomes"] == long[0]["outcomes"] for method in long)
    detail = f"All three solve the same {long[0]['correct']} of 16 long theorems." if same_long else "Long-theorem success patterns differ."
    fig.text(.16, .046, detail + " Short-panel differences do not establish a reliable RL improvement.", fontsize=9)
    fig.text(.16, .018, "This completes neither the five-arm comparison nor the original 120-target evaluation or sampled K4 phase.", fontsize=9)
    names = []
    for extension in ("png", "pdf", "svg"):
        name = "paired_trained_outcomes." + extension
        fig.savefig(out / name, dpi=180, bbox_inches="tight")
        names.append(name)
    plt.close(fig)
    return names


def run(report_dir, out):
    report_dir, out = report_dir.resolve(), out.resolve()
    require(not out.exists(), "Use a fresh figure directory; preserve existing reports")
    receipts = {}
    def read(path, expected=None, lines=False):
        path = Path(path)
        raw = path.read_bytes()
        actual = digest(raw)
        require(expected is None or actual == expected, "Input/source digest mismatch: " + str(path))
        receipts[str(path)] = actual
        if lines:
            require(raw.endswith(b"\n"), "Uncommitted report/event tail")
            return [json.loads(line) for line in raw.splitlines() if line.strip()]
        return json.loads(raw)
    freeze = read(SPRINT / FREEZE, FREEZE_SHA)
    identity = read(report_dir / "CONTINGENCY_IDENTITY.json")
    def bound(path, lines=False):
        path = Path(path)
        require(str(path) in identity["sha256"], "Contingency identity omitted required receipt")
        return read(path, identity["sha256"][str(path)], lines)
    summary = bound(report_dir / "summary.json")
    rows = bound(report_dir / "per_theorem.jsonl", True)
    manifest_path = SPRINT / "data_budget_pilot_v1/manifest.json"
    manifest = read(manifest_path, freeze["sha256"][str(manifest_path)])
    plan_path = SPRINT / "plans/eval_trained_contingency_v1.json"
    plan = bound(plan_path)
    require(receipts[str(plan_path)] == freeze["sha256"][str(plan_path)], "Pipeline plan differs from freeze")
    job = SPRINT / "jobs/eval_trained_contingency_v1"
    require(bound(job / "completed.json")["exit_code"] == 0, "Owning job did not complete successfully")
    spec_path = SPRINT / "plans/eval_trained_contingency_job_v1.json"
    require(bound(job / "spec.json") == bound(spec_path) and receipts[str(spec_path)] == freeze["sha256"][str(spec_path)], "Owning job spec differs from frozen plan")
    pipeline = SPRINT / "runs/eval_trained_contingency_pipeline_v1"
    require(bound(pipeline / "plan.json") == plan, "Executed plan changed")
    require(bound(pipeline / "completed.json") == {"status": "complete", "stages": ["sft", "rl", "dpo"],
            "plan_sha256": receipts[str(plan_path)], "source_freeze_sha256": FREEZE_SHA}, "Pipeline completion mismatch")
    events = bound(pipeline / "events.jsonl", True)
    exits = [row for row in events if row["event"] == "stage_exit"]
    require([row["name"] for row in exits] == ["sft", "rl", "dpo"] and all(row["exit_code"] == 0 for row in exits), "Stage completion mismatch")
    for step in plan["steps"]:
        starts = [row for row in events if row["event"] == "stage_start" and row["name"] == step["name"]]
        require(len(starts) == 1 and starts[0]["command"] == [str(SPRINT / ".venv/bin/python"), "-u", str(SPRINT / "code/evaluate.py"), *step["args"]], "Executed stage command mismatch")
    statuses = {}
    for result in summary["runs"]:
        require(result["arm"] in ARMS, "Only three trained endpoints are permitted")
        directory = SPRINT / "runs/eval_trained_contingency_v1" / result["arm"]
        require(result["run_directory"] == str(directory), "Unexpected outcome directory")
        statuses[result["arm"]] = bound(directory / "evaluation_status.json")
        eval_plan = bound(directory / "evaluation_plan.json")
        require(receipts[str(directory / "evaluation_plan.json")] == result["evaluation_plan_sha256"]
                and eval_plan["settings"] == result["settings"], "Summary evaluation plan identity mismatch")
    panels = validate(identity, summary, rows, manifest, statuses)
    for path, expected in ((SPRINT / "code/analyze.py", ANALYZER_SHA), (SPRINT / "code/plot_budget_pilot.py", PLOT_METRIC_SHA),
                           (Path(__file__), None), (SPRINT / "code/test_plot_trained_contingency.py", None)):
        actual = digest(path.read_bytes())
        require(expected is None or actual == expected, "Plot source identity mismatch")
        receipts[str(path)] = actual
    out.mkdir(parents=True)
    names = draw(panels, out, identity["icl_attempts_committed"])
    long = panels["long_panel"]["methods"]
    same_long = all(method["outcomes"] == long[0]["outcomes"] for method in long)
    common = [i for i in range(16) if all(method["outcomes"][i] for method in long)]
    common_description = "; ".join(f"L{i+1:02d} ({panels['long_panel']['targets'][i]['family']}; reference {panels['long_panel']['targets'][i]['reference_lines']} lines)" for i in common) or "none"
    caption = ("Three completed trained endpoints from a separate latency-only contingency, not a complete five-arm comparison. "
        f"ICL remains unfinished ({identity['icl_attempts_committed']}/24 committed); earlier zero-shot results are excluded. "
        "Each method has one completed greedy attempt on exactly the same 16 long-reference and 8 short-reference targets, "
        "with a 2,048-token generation cap and no ICL demonstrations. Cells show strict valid raw proofs (1) or invalid completed attempts (0). "
        "Columns retain the original frozen target order; absent methods are not failures.\n\n"
        + "; ".join(f"{method['label']}: {method['correct']}/{method['n']} short, {long[index]['correct']}/16 long"
                    for index, method in enumerate(panels["short_panel"]["methods"]))
        + f". Long outcome vectors identical across the three methods: {same_long}. "
        f"Shared long successes: {common_description}; short successes occur on different subsets. "
        "These small, single-seed results do not establish a reliable RL gain. Reference lengths are witness upper bounds, "
        "not certified minimum complexity; n=2 per long length is descriptive.\n\n"
        "This figure validates the complete three-arm report, exact fixed membership, summary counts, frozen manifest and successful owning/stage receipts. "
        "It relies on the unchanged upstream native-token/verifier replay and does not repeat that audit or compute model likelihoods. "
        "It neither completes the original 120-target evaluation nor the sampled K4 phase. No zero-shot/ICL outcome payloads or protected data were read.\n")
    (out / "CAPTION.md").write_text(caption)
    for path, expected in receipts.items():
        require(digest(Path(path).read_bytes()) == expected, "Static figure input/source changed during rendering")
    provenance = {"created_utc": datetime.now(timezone.utc).isoformat(), "kind": "three_complete_trained_endpoints_only",
        "panels": panels, "input_and_source_sha256": receipts, "same_long_outcome_vectors": same_long,
        "zero_or_icl_outcomes_read": False, "five_arm_or_primary120_completion_claim": False,
        "output_sha256": {name: digest((out / name).read_bytes()) for name in [*names, "CAPTION.md"]}}
    (out / "plot_provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    return {"out": str(out), "figures": names, "status": "complete_three_trained_arms_only"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.report, args.out), sort_keys=True))
