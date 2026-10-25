import unittest
from fractions import Fraction

from scripts.audit_checkpoint_labels import exact_arithmetic, pairwise_auc


class CheckpointAuditTests(unittest.TestCase):
    def test_decimal_checks_are_exact_and_keep_rounding_distinct(self):
        self.assertEqual(exact_arithmetic("1.60+2.40"), Fraction(4))
        self.assertEqual(exact_arithmetic("2/3*160"), Fraction(320, 3))
        self.assertNotEqual(exact_arithmetic("2/3*160"), exact_arithmetic("106.67"))
        self.assertEqual(exact_arithmetic("(1,200 - 300) / 3"), Fraction(300))

    def test_expressions_are_whitelisted_without_execution(self):
        for text in ("abs(-1)", "x+1", "2**3", "[1][0]"):
            with self.assertRaises(ValueError):
                exact_arithmetic(text)

    def test_auc_counts_ties_as_half(self):
        self.assertEqual(pairwise_auc([0, 1], [0.1, 0.9]), 1)
        self.assertEqual(pairwise_auc([0, 1], [0.9, 0.1]), 0)
        self.assertEqual(pairwise_auc([0, 1], [0.5, 0.5]), 0.5)
        self.assertEqual(pairwise_auc([1, 0, 1, 0], [0.2, 0.2, 0.8, 0.8]), 0.5)


if __name__ == "__main__":
    unittest.main()
