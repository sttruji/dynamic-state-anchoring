import copy
import unittest
from dsa.data.adapters import canonical_record
from scripts.assemble_lotus_personal_review import validate_reviews


class ReviewLedgerTests(unittest.TestCase):
    def setUp(self):
        self.source=[{'question':'A one-dollar pen and two-dollar pad cost how much?',
                      'steps':['<<1+2=3>>'],'answer':'$3'}]
        self.review={'row_number':1,'answer_format':'currency','verdict':'correct',
                     'expected_answer':'$3','reasoning':'One plus two equals three dollars.',
                     'confidence':'high','suggested_handling':'keep'}

    def test_identity_derived_from_authoritative_source(self):
        result=validate_reviews(self.source,[self.review])
        self.assertEqual(result[0]['sample_id'],canonical_record(self.source[0])['sample_id'])
        self.assertEqual(result[0]['source_answer'],'$3')
        self.assertEqual(result[0]['question'],self.source[0]['question'])

    def test_wrong_supplied_identity_or_incomplete_coverage_rejected(self):
        wrong={**self.review,'sample_id':'wrong-source'}
        with self.assertRaisesRegex(ValueError,'identity mismatch'):
            validate_reviews(self.source,[wrong])
        for records in ([],[self.review,self.review]):
            with self.assertRaisesRegex(ValueError,'exactly once'):
                validate_reviews(self.source,records)

    def test_unsupported_verdict_missing_reason_and_inconsistent_handling_rejected(self):
        for update in ({'verdict':'looks_ok'},{'reasoning':''},{'suggested_handling':'exclude_from_clean_sensitivity'}):
            with self.assertRaises(ValueError):
                validate_reviews(self.source,[{**self.review,**update}])


if __name__=='__main__':
    unittest.main()
