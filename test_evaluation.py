import unittest

import evaluation
class TestEvaluationMethods(unittest.TestCase):

    def test_get_optimal_selection_indices_1(self):
        ranked_to_category = [('a',1),('b',0.9),('c',0.8),('d',0.7),('e',0.6),('f',0.5)]
        documents_in_category=['a','b','f']
        precission_levels=[0.2,0.3,0.5,1]
        expected = {1.0:2, 0.5:4, 0.3:5, 0.2:5}
        res = evaluation.get_optimal_selection_indices(ranked_to_category, documents_in_category, precission_levels)
        self.assertEqual(res, expected)
        
if __name__ == '__main__':
    unittest.main()