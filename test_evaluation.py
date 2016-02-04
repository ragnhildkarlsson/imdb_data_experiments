import unittest
import numpy as np

import evaluation
class TestEvaluationMethods(unittest.TestCase):

    def test_get_precission_selection_indices_1(self):
        ranked_to_category = [('a',1),('b',0.9),('c',0.8),('d',0.7),('e',0.6),('f',0.5)]
        documents_in_category=['a','b','f']
        precission_levels=[0.1, 0.5, 0.6, 1.0]
        expected = {0:6,1:6,2:3,3:2}
        res = evaluation.get_precission_selection_indices(ranked_to_category, documents_in_category, precission_levels)
        self.assertEqual(res, expected)

    def test_get_precission_selection_indices_2(self):
        ranked_to_category = [('a',1),('b',0.9),('c',0.8),('d',0.7),('e',0.6),('f',0.5)]
        documents_in_category=['f']
        precission_levels=[0.1,1.0]
        expected = {0:6,1:0}
        res = evaluation.get_precission_selection_indices(ranked_to_category, documents_in_category, precission_levels)
        self.assertEqual(res, expected)

    def test_get_precission_selection_indices_3(self):
        ranked_to_category = []
        documents_in_category=['f']
        precission_levels=[0.1,1.0]
        expected = {0:0,1:0}
        res = evaluation.get_precission_selection_indices(ranked_to_category, documents_in_category, precission_levels)
        self.assertEqual(res, expected)

    def test_get_percentage_selection_indices_1(self):
        ranked_to_category = [('a',1),('b',0.9),('c',0.8),('d',0.7),('e',0.4),('f',0)]
        percentage_levels=[0.2, 0.25, 0.5, 1.0]
        expected = {0:3,1:3,2:4,3:6,}
        res = evaluation.get_percentage_selection_indices(ranked_to_category, percentage_levels)
        self.assertEqual(res, expected)

    def test_get_evaluation(self):
        category = "category_1"
        ranked_documents = {category:[('a',1),('b',0.9),('c',0.8),('d',0.7),('e',0.4),('f',0)]}
        documents_in_category_1=['a','b']
        documents_in_category_2=['a','c']
        documents_in_category_3=['e','f']
        gold_standard_categorization = {}
        gold_standard_categorization[category] = documents_in_category_1
        category_hierarchy = []
        evaluation_levels = [0.2]
        evaluation_selections = {category:{0:3}}
        precission_key = "precission_key"
        recall_key = "recall_key"
        n_ranked_docs_key = "n_ranked_docs_key"
        n_correct_ranked_docs_key = "n_correct_ranked_docs_key"
        n_docs_in_category_key = "n_docs_in_category_key"
        res = evaluation.get_evaluation([category],
                                        ranked_documents,
                                        gold_standard_categorization,
                                        category_hierarchy,
                                        evaluation_selections,
                                        precission_key, recall_key, n_ranked_docs_key,
                                        n_correct_ranked_docs_key,
                                        n_docs_in_category_key,
                                        evaluation_levels)
        print(res)
        
        np.testing.assert_almost_equal(res[category][0][precission_key], 2/3, 4)
        np.testing.assert_almost_equal(res[category][0][recall_key], 2/2, 4)
        self.assertEqual(res[category][0][n_correct_ranked_docs_key], 2)


if __name__ == '__main__':
    unittest.main()