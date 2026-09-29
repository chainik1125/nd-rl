"""Small fabricated-data tests for the three-arm contingency plot validator."""
from collections import Counter
from copy import deepcopy
import unittest

from plot_trained_contingency import ARMS, COHORTS, validate
from test_plot_budget_pilot import fabricated


def fixture():
    _, summary, rows, _ = fabricated()
    summary["runs"] = summary["runs"][2:]
    for result in summary["runs"]:
        result["demonstrations_per_query"] = 0
    rows = rows[48:]
    manifest = {"cohort_counts": COHORTS, "cohort_order": list(COHORTS), "selection": {}}
    for cohort, n in COHORTS.items():
        selected = [row for row in rows[:24] if row["cohort"] == cohort]
        manifest["selection"]["eval_" + cohort + ".jsonl"] = {"rows": n,
            "canonical_hashes": [row["canonical_hash"] for row in selected],
            "reference_length_counts": dict(Counter(str(row["reference_lines"]) for row in selected)),
            "family_counts": dict(Counter(row["family"] for row in selected))}
    identity = {"kind": "complete_trained_subset_of_incomplete_five_arm_pilot",
        "status": "complete_for_three_trained_arms_only", "arms": ARMS, "strictly_replayed_attempts": 72,
        "analysis_exit_code": 0, "original_five_arm_pilot_complete": False, "original_primary120_complete": False,
        "zero_or_partial_icl_in_comparisons": False, "pooled24_headline": False,
        "icl_attempts_committed": 12, "icl_planned_attempts": 24, "cohort_counts": COHORTS}
    statuses = {arm: {"status": "complete", "attempts": 24, "planned_attempts": 24} for arm in ARMS}
    return identity, summary, rows, manifest, statuses


class ContingencyTests(unittest.TestCase):
    def test_three_arm_counts_and_order(self):
        panels = validate(*fixture())
        self.assertEqual([row["arm"] for row in panels["long_panel"]["methods"]], ARMS)
        self.assertEqual([row["reference_lines"] for row in panels["long_panel"]["targets"]][:4], [11, 7, 14, 9])
        self.assertEqual(len(panels["short_panel"]["targets"]), 8)

    def test_missing_duplicate_or_reordered_rows_rejected(self):
        for mutation in (lambda rows: rows.pop(), lambda rows: rows.__setitem__(1, deepcopy(rows[0])),
                         lambda rows: rows.__setitem__(slice(24, 26), list(reversed(rows[24:26])))):
            identity, summary, rows, manifest, statuses = fixture()
            mutation(rows)
            with self.assertRaises(ValueError):
                validate(identity, summary, rows, manifest, statuses)

    def test_partial_or_wrong_summary_counts_rejected(self):
        for mutate in (lambda summary, statuses: statuses["rl"].update(status="partial"),
                       lambda summary, statuses: summary["runs"][0]["groups"]["long_panel"]["greedy"]["overall"].update(solved=16)):
            identity, summary, rows, manifest, statuses = fixture()
            mutate(summary, statuses)
            with self.assertRaises(ValueError):
                validate(identity, summary, rows, manifest, statuses)

    def test_wrong_membership_or_completion_claim_rejected(self):
        for mutate in (lambda identity, manifest: identity.update(original_five_arm_pilot_complete=True),
                       lambda identity, manifest: manifest["selection"]["eval_long_panel.jsonl"]["canonical_hashes"].__setitem__(0, "f" * 64)):
            identity, summary, rows, manifest, statuses = fixture()
            mutate(identity, manifest)
            with self.assertRaises(ValueError):
                validate(identity, summary, rows, manifest, statuses)


if __name__ == "__main__":
    unittest.main()
