import unittest
from dsa.evaluation.evaluate import extract_answer, score_answer


class EvaluationTests(unittest.TestCase):
    def test_pinned_lotus_extraction(self):
        self.assertEqual(extract_answer("reason 4 ### 1,200"), "1200")
        self.assertEqual(extract_answer("first 5 then -12.5"), "-12.5")
        self.assertEqual(extract_answer("### 10 ### 14"), "14")
        self.assertEqual(extract_answer("nonsense"), "nonsense")

    def test_pinned_lotus_numeric_comparison(self):
        self.assertTrue(score_answer("12.0", "12"))
        self.assertFalse(score_answer("12.1", "12"))
        self.assertFalse(score_answer("nonsense", "12"))
        self.assertTrue(score_answer(" label ", "label"))


if __name__ == "__main__":
    unittest.main()
