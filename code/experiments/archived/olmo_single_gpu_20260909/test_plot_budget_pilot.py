"""Fabricated outcomes only: no model prediction files are accessed."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from plot_budget_pilot import ARMS, ANALYZER_SHA, COHORTS, FREEZE, FREEZE_SHA, digest, run, validate


def metric(rows, overall=False):
    n, solved = len(rows), sum(row["solved"] for row in rows)
    return {"n": n, "solved": solved, "rate": solved/n, "mean_sample_success": solved/n,
            "sample_attempts": n, "successful_samples": solved, **({"intended_n": n} if overall else {})}


def fabricated():
    rows, runs = [], []
    # Deliberately non-monotone reference order; plotting must preserve it.
    lengths = [11, 7, 14, 9, 12, 8, 10, 13] * 2 + [6, 2, 5, 3, 4, 2, 6, 5]
    for arm_index, arm in enumerate(ARMS):
        block = []
        for index, length in enumerate(lengths):
            solved = (index * 3 + arm_index * 5) % 11 < 3 + arm_index
            block.append({"arm": arm, "cohort": "long_panel" if index < 16 else "short_panel",
                "canonical_hash": digest(f"FABRICATED THEOREM {index}".encode()), "family": f"fixture{index % 3}",
                "reference_lines": length, "mode": "greedy", "metric": "greedy_pass1", "attempts": 1,
                "successful_attempts": int(solved), "solved": solved})
        groups = {}
        for cohort in COHORTS:
            selected = [row for row in block if row["cohort"] == cohort]
            groups[cohort] = {"greedy": {"overall": metric(selected, True),
                **{field: {str(value): metric([row for row in selected if row[key] == value])
                   for value in {row[key] for row in selected}} for field, key in
                   (("by_reference_length", "reference_lines"), ("by_family", "family"))}}}
        rows.extend(block)
        runs.append({"arm": arm, "status": "complete", "planned_attempts": 24, "observed_attempts": 24,
            "pending_theorem_mode_groups": 0, "settings": {"modes": ["greedy"], "max_new_tokens": 2048}, "groups": groups})
    summary = {"status": "complete", "analyzer_sha256": ANALYZER_SHA, "raw_success_is_primary": True, "runs": runs}
    summary_ref = {"path": "/FABRICATED/summary.json", "sha256": "fixture"}
    identity = {"kind": "separate_budget_contingency_pilot_report", "status": "complete", "cohort_counts": COHORTS,
        "primary120_completed_by_this_report": False, "pooled24_headline": False, "fabricated_fixture": True,
        "pilot_source_freeze": {"path": str(FREEZE), "sha256": FREEZE_SHA}, "analysis_summary": summary_ref}
    return identity, summary, rows, summary_ref


def write_fabricated(directory):
    directory.mkdir(parents=True)
    identity, summary, rows, _ = fabricated()
    payload = (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode()
    path = directory / "summary.json"
    path.write_bytes(payload)
    identity["analysis_summary"] = {"path": str(path.resolve()), "sha256": digest(payload)}
    (directory / "PILOT_IDENTITY.json").write_text(json.dumps(identity, indent=2) + "\n")
    (directory / "per_theorem.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))


class PlotTests(unittest.TestCase):
    def test_order_and_exact_counts_preserved(self):
        panels = validate(*fabricated(), fixture=True)
        self.assertEqual([row["reference_lines"] for row in panels["long_panel"]["targets"]][:4], [11, 7, 14, 9])
        for cohort, n in COHORTS.items():
            for method in panels[cohort]["methods"]:
                self.assertEqual(method["correct"], sum(method["outcomes_in_original_order"]))
                self.assertEqual(method["n"], n)

    def test_fixture_rejected_in_production(self):
        with self.assertRaisesRegex(ValueError, "Fabricated"):
            validate(*fabricated())

    def test_incomplete_or_duplicate_rows_rejected(self):
        for action in (lambda rows: rows.pop(), lambda rows: rows.__setitem__(1, deepcopy(rows[0]))):
            identity, summary, rows, ref = fabricated()
            action(rows)
            with self.assertRaises(ValueError):
                validate(identity, summary, rows, ref, True)

    def test_cross_arm_reordering_rejected(self):
        identity, summary, rows, ref = fabricated()
        rows[24], rows[25] = rows[25], rows[24]
        with self.assertRaisesRegex(ValueError, "membership/order"):
            validate(identity, summary, rows, ref, True)

    def test_summary_status_and_count_mismatches_rejected(self):
        for action in (lambda summary: summary.update(status="partial_diagnostic"),
                       lambda summary: summary["runs"][0].update(pending_theorem_mode_groups=1),
                       lambda summary: summary["runs"][0]["groups"]["long_panel"]["greedy"]["overall"].update(solved=16),
                       lambda summary: summary["runs"][0]["groups"]["long_panel"]["greedy"]["by_reference_length"]["7"].update(n=3),
                       lambda summary: summary["runs"][0]["groups"]["short_panel"]["greedy"]["by_family"]["fixture0"].update(rate=.123)):
            identity, summary, rows, ref = fabricated()
            action(summary)
            with self.assertRaises(ValueError):
                validate(identity, summary, rows, ref, True)

    def test_wrong_identity_and_missing_outcome_rejected(self):
        for action in (lambda identity, rows: identity.update(primary120_completed_by_this_report=True),
                       lambda identity, rows: identity["analysis_summary"].update(sha256="other"),
                       lambda identity, rows: rows[0].update(solved=None),
                       lambda identity, rows: rows[0].update(mode="sampled")):
            identity, summary, rows, ref = fabricated()
            ref = deepcopy(ref)
            action(identity, rows)
            with self.assertRaises(ValueError):
                validate(identity, summary, rows, ref, True)

    def test_full_fixture_exports_and_preserves_inputs(self):
        with tempfile.TemporaryDirectory(prefix="FABRICATED-pilot-") as directory:
            root = Path(directory)
            write_fabricated(root / "input")
            before = {path.name: digest(path.read_bytes()) for path in (root / "input").iterdir()}
            result = run(root / "input", root / "figure", fixture=True)
            self.assertEqual(result["status"], "fabricated_demo")
            self.assertEqual(len(result["figures"]), 3)
            self.assertTrue(all((root / "figure" / name).stat().st_size > 1000 for name in result["figures"]))
            self.assertIn("FABRICATED FIXTURE", (root / "figure/CAPTION.md").read_text())
            self.assertEqual(before, {path.name: digest(path.read_bytes()) for path in (root / "input").iterdir()})
            with self.assertRaisesRegex(ValueError, "fresh"):
                run(root / "input", root / "figure", fixture=True)


if __name__ == "__main__":
    unittest.main()
