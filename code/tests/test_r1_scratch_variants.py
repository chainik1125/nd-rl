"""Synthetic tests for dependency-goal cold starts; no research input reads."""

import unittest

from r1_cot.scratch_variants import plan


class DependencyPlanTests(unittest.TestCase):
    def test_goal_first_drop_unreachable_and_premises(self):
        body = (
            "N1 P : PR ; N2 Q : PR ; N3 ( P v R ) : ORI1 N1 ; "
            "N4 ( P & Q ) : ANDI N1 N2 ; N5 ( ( P & Q ) v S ) : ORI1 N4 ; QED"
        )
        self.assertEqual(
            plan(body),
            [
                "ORI1",
                "(",
                "(",
                "P",
                "&",
                "Q",
                ")",
                "v",
                "S",
                ")",
                ";",
                "ANDI",
                "(",
                "P",
                "&",
                "Q",
                ")",
                ";",
            ],
        )

    def test_preserve_assumption_and_scope(self):
        body = "N1 | P : AS ; N2 | ( P v Q ) : ORI1 N1 ; N3 ( P > ( P v Q ) ) : IMPI N1 N2 ; QED"
        self.assertEqual(
            plan(body),
            [
                "IMPI",
                "(",
                "P",
                ">",
                "(",
                "P",
                "v",
                "Q",
                ")",
                ")",
                ";",
                "AS",
                "|",
                "P",
                ";",
                "ORI1",
                "|",
                "(",
                "P",
                "v",
                "Q",
                ")",
                ";",
            ],
        )

    def test_shared_dependency_once_and_reiteration_bypass(self):
        body = (
            "N1 P : PR ; N2 Q : PR ; N3 ( P & Q ) : ANDI N1 N2 ; "
            "N4 ( P & Q ) : R N3 ; N5 ( ( P & Q ) & ( P & Q ) ) : ANDI N3 N4 ; QED"
        )
        result = plan(body)
        self.assertEqual(result.count("ANDI"), 2)
        self.assertNotIn("R", result)
        self.assertEqual(result.count(";"), 2)

    def test_identity_fallback_and_offset_invariance(self):
        self.assertEqual(plan("N1 P : PR ; N2 P : R N1 ; QED"), ["PR", "P", ";"])
        self.assertEqual(plan("N8 P : PR ; N9 P : R N8 ; QED"), ["PR", "P", ";"])

    def test_reject_future_reference_and_long_label(self):
        with self.assertRaises(ValueError):
            plan("N1 P : R N1 ; QED")
        body = "N1 P : PR ; " + " ".join(f"N{i} P : R N{i - 1} ;" for i in range(2, 8)) + " QED"
        with self.assertRaisesRegex(ValueError, "1..6"):
            plan(body)

    def test_same_formula_under_distinct_assumptions_retained(self):
        body = "N1 ( P v P ) : PR ; N2 | P : AS ; N3 | P : AS ; N4 P : ORE N1 N2 N2 N3 N3 ; QED"
        self.assertEqual(plan(body).count("AS"), 2)


if __name__ == "__main__":
    unittest.main()
