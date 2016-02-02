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

TEST_CATEGORIES = "data/test_data_pickles/test_categories"
RESULT_DICE_BASED_RANKING = "result/dice_based_ranking"

BASIC_REFERENCE_WORDS_DICE = "data/test_data_pickles/default_reference_words_dice"
BASIC_CONTEXT_WORDS_DICE = "data/test_data_pickles/default_context_words_dice"
GAVAGAI_COSINUS_SIMILARE_TERMS = "data/gavagai_pickles/gavagai_cosinus_similare_terms"

FREQUENT_WORDS_SET = "data/index/frequent_words"

EVAL_SCALE = 0.1
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

def get_experiment(id,
                   test_categories_exp,
                   affected_categories_exp,
                   tf_idf_map,
                   reference_words_exp,
                   context_words_exp,
                   gold_standard_categorization,
                   category_hierarchy,
                   evaluation_points,
                   precission_key,
                   recall_key,
                   n_ranked_docs_key,
                   n_correct_ranked_docs_key,
                   n_docs_in_category_key,
                   ):
    ranked_documents_exp = document_categorizer.categorize(test_categories_exp, tf_idf_map, reference_words_exp, context_words_exp)
    evaluation_exp = evaluation.get_threshold_optimized_evaluation(test_categories_exp,
                                                        ranked_documents_exp, gold_standard_categorization,
                                                        category_hierarchy, evaluation_points,
                                                        precission_key, recall_key, n_ranked_docs_key,
                                                        n_correct_ranked_docs_key,
                                                        n_docs_in_category_key)

    summarized_precissions_exp = evaluation.get_summarized_precissions(evaluation_exp, 
                                                                     evaluation_points,
                                                                     n_ranked_docs_key,
                                                                     n_correct_ranked_docs_key)

    summarized_recalls_exp = evaluation.get_summarized_recalls(evaluation_exp,
                                                             evaluation_points,
                                                             n_correct_ranked_docs_key,
                                                             n_docs_in_category_key)
    experiment = Experiment(id,test_categories_exp,
                            reference_words_exp,
                            context_words_exp,
                            reference_words_exp,
                            affected_categories_exp,
                            evaluation_exp,
                            evaluation_points,
                            summarized_precissions_exp,
                            summarized_recalls_exp)
    return experiment

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
bigram_delimeter = BIGRAM_DELIMETER


# Experiment 0.0

test_categories_exp_0 = [test_categories[0]]
affected_categories_exp_0 = test_categories
reference_words_exp_0 = default_reference_words_dice
context_words_exp_0 = default_context_words_dice
experiment_0 = get_experiment(id,
                              test_categories_exp_0,
                              affected_categories_exp_0,
                              tf_idf_map,
                              reference_words_exp_0,
                              context_words_exp_0,
                              gold_standard_categorization,
                              category_hierarchy,
                              evaluation_points,
                              precission_key,
                              recall_key,
                              n_ranked_docs_key,
                              n_correct_ranked_docs_key,
                              n_docs_in_category_key,
                              )

pprint.pprint(experiment_0.summarized_precissions)
pprint.pprint(experiment_0.summarized_recalls)

# EXPERIMENT 0.1

# test_categories_exp_01 = test_categories
# affected_categories_exp_01 = test_categories
# reference_words_exp_01 = {} 

# for category in test_categories:
#   reference_words_exp_01[category] = [category]
  
# context_words_exp_01 = default_context_words_dice
# experiment_01 = get_experiment(id,
#                               test_categories_exp_01,
#                               affected_categories_exp_01,
#                               tf_idf_map,
#                               reference_words_exp_01,
#                               context_words_exp_01,
#                               gold_standard_categorization,
#                               category_hierarchy,
#                               evaluation_points,
#                               precission_key,
#                               recall_key,
#                               n_ranked_docs_key,
#                               n_correct_ranked_docs_key,
#                               n_docs_in_category_key,
#                               )

# pprint.pprint(experiment_01.summarized_precissions)
# pprint.pprint(experiment_01.summarized_recalls)

# EXPERIMENT 1

# Test move word in bigrams with category name to context words

# test_categories_exp_1 = test_categories
# affected_categories_exp_1 = test_categories
# gavagai_suggested_terms = pickle_handler.load_pickle(GAVAGAI_COSINUS_SIMILARE_TERMS)

# reference_words_exp_1 = {} 
# context_words_exp_1 = {}

# for category in test_categories:
#   r,c = keyword_setups.get_only_gavagai_paradigmatic_similare_keywords(gavagai_suggested_terms[category])
#   reference_words_exp_1[category] = r
#   context_words_exp_1[category] = c

# experiment_1 = get_experiment(1,
#                               test_categories_exp_1,
#                               affected_categories_exp_1,
#                               tf_idf_map,
#                               reference_words_exp_1,
#                               context_words_exp_1,
#                               gold_standard_categorization,
#                               category_hierarchy,
#                               evaluation_points,
#                               precission_key,
#                               recall_key,
#                               n_ranked_docs_key,
#                               n_correct_ranked_docs_key,
#                               n_docs_in_category_key,
#                               )

# pprint.pprint(experiment_1.summarized_precissions)
# pprint.pprint(experiment_1.summarized_recalls)

# # Experiment 2 

# test_categories_exp_2 = test_categories
# affected_categories_exp_2 = test_categories
# gavagai_suggested_terms = pickle_handler.load_pickle(GAVAGAI_COSINUS_SIMILARE_TERMS)

# reference_words_exp_2 = {} 
# context_words_exp_2 = default_context_words_dice

# for category in test_categories:
#   r,_ = keyword_setups.get_only_gavagai_paradigmatic_similare_keywords(gavagai_suggested_terms[category])
#   r.append(category)
#   reference_words_exp_2[category] = r


# experiment_2 = get_experiment(1,
#                               test_categories_exp_2,
#                               affected_categories_exp_2,
#                               tf_idf_map,
#                               reference_words_exp_2,
#                               context_words_exp_2,
#                               gold_standard_categorization,
#                               category_hierarchy,
#                               evaluation_points,
#                               precission_key,
#                               recall_key,
#                               n_ranked_docs_key,
#                               n_correct_ranked_docs_key,
#                               n_docs_in_category_key,
#                               )

# pprint.pprint(experiment_2.summarized_precissions)
# pprint.pprint(experiment_2.summarized_recalls)


# # Experiment 3 Combine all gavagai paradigmatic neighbours + dice as reference words + dice words as keywords 

# test_categories_exp_3 = test_categories
# affected_categories_exp_3 = test_categories
# gavagai_suggested_terms = pickle_handler.load_pickle(GAVAGAI_COSINUS_SIMILARE_TERMS)

# reference_words_exp_3 = {} 
# context_words_exp_3 = default_context_words_dice

# for category in test_categories:
#   r,_ = keyword_setups.get_only_gavagai_paradigmatic_similare_keywords(gavagai_suggested_terms[category])
#   r.append(category)
#   r = r + default_reference_words_dice[category]
#   reference_words_exp_3[category] = r


# experiment_3 = get_experiment(1,
#                               test_categories_exp_3,
#                               affected_categories_exp_3,
#                               tf_idf_map,
#                               reference_words_exp_3,
#                               context_words_exp_3,
#                               gold_standard_categorization,
#                               category_hierarchy,
#                               evaluation_points,
#                               precission_key,
#                               recall_key,
#                               n_ranked_docs_key,
#                               n_correct_ranked_docs_key,
#                               n_docs_in_category_key,
#                               )

# pprint.pprint(experiment_3.summarized_precissions)
# pprint.pprint(experiment_3.summarized_recalls)

# # Experiment_4  Move reference words occuring in bigram with category name to context words
# test_categories_exp_4 = test_categories
# affected_categories_exp_4 = []

# reference_words_exp_4 = {}
# context_words_exp_4 ={}
# for category in test_categories:
#   filtered_reference_words, filtered_context_words = keyword_setups.get_dice_keyword_filter_1(category, BIGRAM_DELIMETER, default_reference_words_dice[category], default_context_words_dice[category])
#   if filtered_reference_words and filtered_context_words:
#     reference_words_exp_4[category] = filtered_reference_words
#     context_words_exp_4[category] = filtered_context_words
#     affected_categories_exp_4.append(category) 
#   else:
#     reference_words_exp_4[category] = default_reference_words_dice[category]
#     context_words_exp_4[category] = default_context_words_dice[category]

# experiment_4 = get_experiment(4,
#                               test_categories_exp_4,
#                               affected_categories_exp_4,
#                               tf_idf_map,
#                               reference_words_exp_4,
#                               context_words_exp_4,
#                               gold_standard_categorization,
#                               category_hierarchy,
#                               evaluation_points,
#                               precission_key,
#                               recall_key,
#                               n_ranked_docs_key,
#                               n_correct_ranked_docs_key,
#                               n_docs_in_category_key,
#                               )

# pprint.pprint(experiment_4.summarized_precissions)
# pprint.pprint(experiment_4.summarized_recalls)