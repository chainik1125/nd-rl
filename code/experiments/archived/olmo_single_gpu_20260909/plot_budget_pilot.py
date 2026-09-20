"""Plot only a complete audited 24-target budget pilot; never read predictions.

This is a descriptive visualization, not another model/verifier audit or release
gate. Production inputs must come from budget_pilot_v1.py analyze. Fabricated
fixtures require an explicit flag and retain a prominent non-results label.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

SPRINT = Path(__file__).resolve().parents[1]
ARMS = ["base_zero", "base_icl32", "sft", "dpo", "rl"]
LABELS = ["Base zero-shot", "Base + ICL32", "SFT", "SFT → DPO", "SFT → DPO → RL"]
COHORTS = {"long_panel": 16, "short_panel": 8}
FREEZE = SPRINT / "plans/budget_pilot_source_freeze_v1.json"
FREEZE_SHA = "a258d5b26227ba32f775263ee9d0a7739f76732ebb5aef684e923fba66d8c5b9"
ANALYZER_SHA = "37f02d136a9a215fabdc1f3c3de738433f389349036eeaec4bf59884ca10017e"
PILOT_SHA = "f11f875ee33d6040dbd1588a58974b3e726c02b64291825e856bbfa9ac233cea"


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def metric_check(metric, rows, overall=False):
    n, solved = len(rows), sum(row["solved"] for row in rows)
    for key, expected in {"n": n, "solved": solved, "sample_attempts": n,
                          "successful_samples": solved, **({"intended_n": n} if overall else {})}.items():
        require(type(metric.get(key)) is int and metric[key] == expected, "Summary count mismatch: " + key)
    for key in ("rate", "mean_sample_success"):
        value = metric.get(key)
        require(type(value) in (int, float) and math.isfinite(value)
                and math.isclose(value, solved / n, rel_tol=0, abs_tol=1e-12), "Summary rate mismatch")


def validate(identity, summary, rows, summary_ref, fixture=False):
    require(identity.get("fabricated_fixture", False) is fixture,
            "Fabricated fixtures require explicit fixture mode; real reports cannot use it")
    require(identity.get("kind") == "separate_budget_contingency_pilot_report"
            and identity.get("status") == "complete" and identity.get("cohort_counts") == COHORTS
            and identity.get("primary120_completed_by_this_report") is False
            and identity.get("pooled24_headline") is False, "Wrong or incomplete pilot identity")
    require(identity.get("pilot_source_freeze") == {"path": str(FREEZE), "sha256": FREEZE_SHA},
            "Pilot source freeze identity mismatch")
    require(identity.get("analysis_summary") == summary_ref, "Pilot identity does not bind this exact summary")
    require(summary.get("status") == "complete" and summary.get("analyzer_sha256") == ANALYZER_SHA
            and summary.get("raw_success_is_primary") is True, "Need the complete frozen-analyzer report")
    runs = summary.get("runs", [])
    require([run["arm"] for run in runs] == ARMS, "Exactly five ordered methods required")
    require(len(rows) == 120 and [row["arm"] for row in rows] == [arm for arm in ARMS for _ in range(24)],
            "All five methods × 24 ordered outcomes are required; missing is not failure")
    reference = rows[:24]
    order = [(row["cohort"], row["canonical_hash"]) for row in reference]
    require(len({key for _, key in order}) == 24 and [cohort for cohort, _ in order]
            == [cohort for cohort, n in COHORTS.items() for _ in range(n)], "Pilot cohort counts/order or uniqueness changed")
    for row in reference:
        require(isinstance(row["canonical_hash"], str) and len(row["canonical_hash"]) == 64
                and all(char in "0123456789abcdef" for char in row["canonical_hash"]), "Invalid theorem identity")
        require(type(row["reference_lines"]) is int, "Reference length must be an integer")
    require(Counter(row["reference_lines"] for row in reference[:16]) == {length: 2 for length in range(7, 15)}
            and all(2 <= row["reference_lines"] <= 6 for row in reference[16:]), "Reference-length strata changed")
    panels = {cohort: {"targets": [dict(index=i + 1, **{key: row[key] for key in
              ("canonical_hash", "family", "reference_lines")}) for i, row in
              enumerate(value for value in reference if value["cohort"] == cohort)], "methods": []}
              for cohort in COHORTS}
    for arm_index, run in enumerate(runs):
        block = rows[arm_index * 24:(arm_index + 1) * 24]
        require([(row["cohort"], row["canonical_hash"]) for row in block] == order,
                "Method target membership/order differs; no outcome-based reordering allowed")
        require(run["status"] == "complete" and run["planned_attempts"] == run["observed_attempts"] == 24
                and run["pending_theorem_mode_groups"] == 0, "Incomplete method must not be plotted")
        require(run["settings"]["modes"] == ["greedy"] and run["settings"]["max_new_tokens"] == 2048,
                "Expected one greedy attempt with the frozen 2048-token cap")
        require(set(run["groups"]) == set(COHORTS), "Unexpected summary cohorts")
        for source, row in zip(reference, block, strict=True):
            require(all(row[key] == source[key] for key in ("family", "reference_lines")), "Method target metadata differs")
            require(row["mode"] == "greedy" and row["metric"] == "greedy_pass1"
                    and type(row["attempts"]) is int and row["attempts"] == 1
                    and type(row["solved"]) is bool and type(row["successful_attempts"]) is int
                    and row["successful_attempts"] == int(row["solved"]), "Missing or inconsistent binary greedy outcome")
        for cohort, n in COHORTS.items():
            selected = [row for row in block if row["cohort"] == cohort]
            require(set(run["groups"][cohort]) == {"greedy"}, "Extra decoding mode in summary")
            group = run["groups"][cohort]["greedy"]
            metric_check(group["overall"], selected, overall=True)
            for field, key in (("by_reference_length", "reference_lines"), ("by_family", "family")):
                bins = group[field]
                require(set(bins) == {str(row[key]) for row in selected}, "Summary bins do not match theorem rows")
                for value, metric in bins.items():
                    metric_check(metric, [row for row in selected if str(row[key]) == value])
            panels[cohort]["methods"].append({"arm": run["arm"], "label": LABELS[arm_index],
                "correct": sum(row["solved"] for row in selected), "n": n,
                "outcomes_in_original_order": [int(row["solved"]) for row in selected]})
    return panels


def draw(panels, out, fixture=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    fig = plt.figure(figsize=(13.5, 7.7), facecolor="white")
    prefix = "FABRICATED FIXTURE — NOT MODEL RESULTS\n" if fixture else ""
    fig.suptitle(prefix + "Separate budget pilot — not completion of the original 120-target evaluation",
                 fontsize=14, fontweight="bold", y=.985)
    fig.text(.5, .87 if fixture else .91, "OLMo 32B · five methods on the same problems · one greedy attempt · 2,048-token generation cap",
             ha="center", fontsize=10)
    for cohort, bottom, width in (("long_panel", .55, .70), ("short_panel", .21, .35)):
        panel, n = panels[cohort], COHORTS[cohort]
        axis = fig.add_axes([.16, bottom, width, .22])
        matrix = [method["outcomes_in_original_order"] for method in panel["methods"]]
        axis.imshow(matrix, cmap=ListedColormap(["#edf0f2", "#146b73"]), vmin=0, vmax=1, aspect="auto")
        axis.set_yticks(range(5), LABELS, fontsize=10)
        code = "L" if cohort == "long_panel" else "S"
        axis.set_xticks(range(n), [f"{code}{target['index']:02d}\nref {target['reference_lines']}"
                                  for target in panel["targets"]], fontsize=8)
        axis.tick_params(length=0, pad=6)
        axis.set_title(("Long-reference problems (7–14 lines)" if code == "L" else "Short-reference problems (2–6 lines)")
                       + f" · n = {n}", loc="left", fontsize=11, pad=23)
        axis.text(n - .18, -.92, "Correct / n", fontsize=9, ha="left")
        for method_index, method in enumerate(panel["methods"]):
            axis.text(n - .18, method_index, f"{method['correct']} / {n}", va="center", fontsize=11, fontweight="bold")
            for target_index, value in enumerate(method["outcomes_in_original_order"]):
                axis.text(target_index, method_index, str(value), color="white" if value else "#68737d",
                          ha="center", va="center", fontsize=10)
        axis.set_xticks([index - .5 for index in range(n + 1)], minor=True)
        axis.set_yticks([index - .5 for index in range(6)], minor=True)
        axis.grid(which="minor", color="white", linewidth=2)
        axis.tick_params(which="minor", length=0)
        for spine in axis.spines.values():
            spine.set_visible(False)
    fig.legend(handles=[Patch(facecolor="#146b73", label="1: strict valid proof"),
                        Patch(facecolor="#edf0f2", edgecolor="#aeb6bc", label="0: completed attempt, not valid")],
               loc="lower left", bbox_to_anchor=(.155, .095), ncol=2, frameon=False, fontsize=10)
    fig.text(.16, .072, "Each column is the same theorem across all five methods, in audited input order; no sorting by outcomes.", fontsize=9)
    fig.text(.16, .049, "Reference proof lengths are upper bounds, not minimum complexity. Only n = 2 per long length: descriptive, not a trend estimate.", fontsize=9)
    fig.text(.16, .026, "Cohorts are reported separately. Missing/incomplete results are rejected, never shown as failures. No original-120 completion claim.", fontsize=9)
    names = []
    for extension in ("png", "pdf", "svg"):
        name = "paired_pilot_outcomes." + extension
        fig.savefig(out / name, dpi=180, bbox_inches="tight")
        names.append(name)
    plt.close(fig)
    return names


def run(report_dir, out, fixture=False):
    report_dir, out = report_dir.resolve(), out.resolve()
    require(not out.exists(), "Figure output must be fresh; existing outputs are preserved")
    require(not fixture or "FABRICATED" in str(out), "Fixture output path must explicitly say FABRICATED")
    sources = {}
    for path, expected in ((FREEZE, FREEZE_SHA), (SPRINT / "code/analyze.py", ANALYZER_SHA),
                           (SPRINT / "code/budget_pilot_v1.py", PILOT_SHA), (Path(__file__), None),
                           (SPRINT / "code/test_plot_budget_pilot.py", None)):
        actual = digest(path.read_bytes())
        require(expected is None or actual == expected, "Pinned plot input/source changed: " + str(path))
        sources[str(path)] = actual
    paths = [report_dir / name for name in ("PILOT_IDENTITY.json", "summary.json", "per_theorem.jsonl")]
    payloads = [path.read_bytes() for path in paths]
    require(payloads[2].endswith(b"\n"), "Per-theorem report has an uncommitted tail")
    refs = {str(path): digest(raw) for path, raw in zip(paths, payloads, strict=True)}
    identity, summary = (json.loads(raw) for raw in payloads[:2])
    rows = [json.loads(line) for line in payloads[2].splitlines() if line.strip()]
    panels = validate(identity, summary, rows, {"path": str(paths[1]), "sha256": refs[str(paths[1])]}, fixture)
    for path, expected in {**sources, **refs}.items():
        require(digest(Path(path).read_bytes()) == expected, "Plot source/input changed during validation")
    out.mkdir(parents=True)
    names = draw(panels, out, fixture)
    caption = ("FABRICATED FIXTURE — NOT MODEL RESULTS.\n\n" if fixture else "") + (
        "Separate 24-target budget pilot, not completion of the original 120-target evaluation. "
        "Five methods share all 16 long-reference and 8 short-reference theorems. Each cell is one completed greedy attempt; "
        "1 means a strictly valid raw proof, 0 an invalid completed attempt. Exact correct/total counts are shown separately "
        "for each cohort. Columns retain the audited per-theorem order and are never selected or sorted by success. "
        "Reference witness lengths are upper bounds, not certified minimum complexity; n=2 per long length is descriptive only. "
        "All five methods must be complete; absent observations are rejected, not converted to failures. "
        "This helper checks report identities, membership and summary consistency; it does not repeat the upstream "
        "native-token/verifier audit or independently recompute model likelihoods. Later full-panel overlap is not independent replication.\n")
    (out / "CAPTION.md").write_text(caption)
    provenance = {"created_utc": datetime.now(timezone.utc).isoformat(), "kind": "paired_budget_pilot_plot",
        "fabricated_fixture": fixture, "status": "fabricated_demo" if fixture else "complete_audited_report_visualization",
        "input_sha256": refs, "source_sha256": sources, "panels": panels,
        "output_sha256": {name: digest((out / name).read_bytes()) for name in [*names, "CAPTION.md"]},
        "primary120_completion_claim": False, "model_or_verifier_reaudit": False,
        "order": "Exact per_theorem input order within each cohort, identical across methods; never outcome-sorted"}
    (out / "plot_provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    return {"status": provenance["status"], "out": str(out), "figures": names}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fabricated-fixture", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.report, args.out, args.fabricated_fixture), sort_keys=True))
