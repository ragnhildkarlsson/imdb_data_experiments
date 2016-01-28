import numpy as np

import pprint
import keyword_setups
import document_categorizer
import pickle_handler
import evaluation


TEST_DATA_ALL_CATEGORIES_LIST = "data/test_data_pickles/all_categories_list"
TEST_DATA_TF_IDF_MAP_PICKLE = "data/test_data_pickles/tf_idf_map_pickle"
TEST_DATA_GOLD_STANDARD_CATEGORIZATION = "data/test_data_pickles/gold_standard_categorization_pickle"

TEST_DATA_DICE_BASED_KEYWORD_RANKING = "data/test_data_pickles/dice_based_keyword_rankning"
TEST_DATA_TF_IDF_MAP = "data/test_data_pickles/tf_idf_map_pickle"

TEST_CATEGORIES = "data/test_data/test_categories"
RESULT_DICE_BASED_RANKING = "result/dice_based_ranking"

BASIC_REFERENCE_WORDS_DICE = "data/test_data_pickles/default_reference_words_dice"
BASIC_CONTEXT_WORDS_DICE = "data/test_data_pickles/default_context_words_dice"

FREQUENT_WORDS_SET = "data/index/frequent_words"

EVAL_SCALE = 0.2
DEFAULT_WEIGHT_FILTER_LIMIT_DICE = 0.05
PRECISSION_KEY = "precission"
RECALL_KEY = "recall"
N_RANKED_DOCS_KEY ="n_ranked_docs"
N_CORRECT_RANKED_DOCS_KEY = "n_correct_ranked_docs_key"
N_DOCS_IN_CATEGORY_KEY = "n_docs_in_category_key"
BIGRAM_DELIMETER = "_"

TEST_DATA_CATEGORY_HIEARACHY = {'religion':{'christianity','buddhism','hinduism','islam','judaism'},'christianity':{'christmas'},'sport':{'baseball','basketball','bicycle','boxing','football','golf','hockey','skiing','soccer','surfing','swimming','tennis','wrestling','horseracing','olympic_games'},\
                                'water_sport':{'surfing','swimming'}, 'motoring':{'motorcycle','car'}, 'nature':{'animals'},'art':{'cinema','theater','music','opera','classical_music', 'jazz', 'country_music','hip_hop', 'dance'}, 'music':{'opera','classical_music','jazz', 'country_music','hip_hop'},\
                                'science':{'medicine','technology','psychology'},'medicine':{'disability'}, 'education':{'school','college'}, 'crime':{'prison','mafia','drugs','fraud','gambling','terrorism'}}

class Experiment:
    def __init__(self,id,
                 test_categories,
                 reference_words,
                 context_words,
                 categorization,
                 affected_categories,
                 evaluation,
                 evaluation_points,
                 summarized_precissions,
                 summarized_recalls):

        self.id = id
        self.test_categories = test_categories
        self.reference_words = reference_words
        self.context_words = context_words
        self.categorization = categorization
        self.affected_categories = affected_categories
        self.evaluation = evaluation
        self.evaluation_points=evaluation_points
        self.summarized_precissions = summarized_precissions
        self.summarized_recalls =  summarized_recalls

# BUILD EXPERIMENTS

#GENERAL_RESOURCES
test_categories = pickle_handler.load_pickle(TEST_CATEGORIES)
default_reference_words_dice = pickle_handler.load_pickle(BASIC_REFERENCE_WORDS_DICE)
default_context_words_dice = pickle_handler.load_pickle(BASIC_CONTEXT_WORDS_DICE)
tf_idf_map = pickle_handler.load_pickle(TEST_DATA_TF_IDF_MAP)
gold_standard_categorization = pickle_handler.load_pickle(TEST_DATA_GOLD_STANDARD_CATEGORIZATION)
category_hierarchy = TEST_DATA_CATEGORY_HIEARACHY

evaluation_points = list(np.arange(0,1,EVAL_SCALE))
evaluation_points.append(1.0)
evaluation_points.pop(0)

category_hierarchy = TEST_DATA_CATEGORY_HIEARACHY

precission_key = PRECISSION_KEY
recall_key = RECALL_KEY
n_ranked_docs_key = N_RANKED_DOCS_KEY
n_correct_ranked_docs_key = N_CORRECT_RANKED_DOCS_KEY
n_docs_in_category_key = N_DOCS_IN_CATEGORY_KEY


#EXPERIMENT 0
reference_words_exp_0 = default_reference_words_dice
context_words_exp_0 = default_context_words_dice
ranked_documents_exp_0 = document_categorizer.categorize(test_categories, tf_idf_map, reference_words_exp_0, context_words_exp_0)
evaluation_exp_0 = evaluation.evaluate_categorization(test_categories,
                                                      ranked_documents_exp_0, gold_standard_categorization,
                                                      category_hierarchy, evaluation_points,
                                                      precission_key, recall_key, n_ranked_docs_key,
                                                      n_correct_ranked_docs_key,
                                                      n_docs_in_category_key)

summarized_precissions_exp_0 = evaluation.get_summarized_precissions(evaluation_exp_0, 
                                                                     evaluation_points,
                                                                     n_ranked_docs_key,
                                                                     n_correct_ranked_docs_key)

summarized_recalls_exp_0 = evaluation.get_summarized_recalls(evaluation_exp_0,
                                                             evaluation_points,
                                                             n_correct_ranked_docs_key,
                                                             n_docs_in_category_key)
exp_0 = Experiment(0,test_categories,
                   reference_words_exp_0,
                   context_words_exp_0,
                   reference_words_exp_0,
                   test_categories,
                   evaluation_exp_0,
                   evaluation_points,
                   summarized_precissions_exp_0,
                   summarized_recalls_exp_0)

pprint.pprint(evaluation_exp_0)
pprint.pprint(summarized_recalls)
pprint.pprint(summarized_precissions)


