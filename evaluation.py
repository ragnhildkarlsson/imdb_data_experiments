import math
import nltk
import numpy as np
import os
from operator import itemgetter
import pprint

import index
import document_processer
import doc_word_simularity
import pickle_handler
import word_simularities


TEST_DATA_ALL_CATEGORIES_PICKLE = "data/test_data_pickles/all_categories_pickle"
TEST_DATA_CATEGORIZED_DOCUMENTS_PICKLE = "data/test_data_pickles/categorized_documents_pickle"
TEST_DATA_REFERENCE_WORDS_DICE = "data/test_data_pickles/test_reference_words_dice"
TEST_DATA_CONTEXT_WORDS_DICE = "data/test_data_pickles/test_context_words_dice"
TEST_CATEGORIES = "data/test_data/test_categories"
RESULT_DICE_BASED_RANKING = "result/dice_based_ranking"


EVAL_SCALE = 0.2
PRECISSION_KEY = "precission"
RECALL_KEY = "recall"
N_RANKED_DOCS_KEY ="n_ranked_docs"
N_CORRECT_RANKED_DOCS_KEY = "n_correct_ranked_docs:key"
N_DOCS_IN_CATEGORY_KEY = "n_docs_in_category_key"

TEST_DATA_CATEGORY_HIEARACHY = {'religion':{'christianity','buddhism','hinduism','islam','judaism'},'christianity':{'christmas'},'sport':{'baseball','basketball','bicycle','boxing','football','golf','hockey','skiing','soccer','surfing','swimming','tennis','wrestling','horseracing','olympic_games'},\
                                'water_sport':{'surfing','swimming'}, 'motoring':{'motorcycle','car'}, 'nature':{'animals'},'art':{'cinema','theater','music','opera','classical_music', 'jazz', 'country_music','hip_hop', 'dance'}, 'music':{'opera','classical_music','jazz', 'country_music','hip_hop'},\
                                'science':{'medicine','technology','psychology'},'medicine':{'disability'}, 'education':{'school','college'}, 'crime':{'prison','mafia','drugs','fraud','gambling','terrorism'}}

def evaluate_categorization(test_categories,
                            categorized_documents, correct_categorization,
                            category_hierarchy, evaluation_points,
                            precission_key, recall_key, n_ranked_docs_key,
                            n_correct_ranked_docs_key,
                            n_docs_in_category_key):
    evaluation = {}
    for category in test_categories:
        evaluation[category] = {}
        ranked_documents = categorized_documents[category]
        ranked_documents = [doc for doc in ranked_documents if not doc[1] == 0]
        documents_in_category = set(correct_categorization[category])
        if category in category_hierarchy:
            for sub_category in category_hierarchy[category]:
                documents_in_category.update(set(correct_categorization[sub_category]))
        n_docs_in_category = len(documents_in_category)            
        evaluation[n_docs_in_category_key] = n_docs_in_category
        
        for evaluation_point_index in range(len(evaluation_points)):

            evaluation_point = evaluation_points[evaluation_point_index]
            evaluation[category][evaluation_point_index] = {}
            n_non_zero_ranked_docs = len(ranked_documents)
            # handle the case of no ranked  docs
            if( n_non_zero_ranked_docs==0):
                evaluation[category][evaluation_point_index][recall_key] = 0
                evaluation[category][evaluation_point_index][precission_key] = 0
                evaluation[category][evaluation_point_index][n_ranked_docs_key] = 0
                evaluation[category][evaluation_point_index][n_correct_ranked_docs_key] = 0
                continue  
            # create a selection for the evaluation point
            min_score = ranked_documents[-1][1]
            max_score = ranked_documents[0][1]
            score_range = max_score-min_score
            min_score_in_selection = min_score + ((1 - evaluation_point)*score_range)
            selected_ranked_documents = [ranked_doc for ranked_doc in ranked_documents if ranked_doc[1] >= min_score_in_selection]
            n_docs_in_selection =  len(selected_ranked_documents)
            # evaluate precission and recall in selection
            n_correct_ranked_docs =0
            for doc in selected_ranked_documents:
                if doc[0] in documents_in_category:
                    n_correct_ranked_docs +=1
            evaluation[category][evaluation_point_index][precission_key] = (n_correct_ranked_docs/n_docs_in_selection)
            evaluation[category][evaluation_point_index][recall_key] = (n_correct_ranked_docs/n_docs_in_category)
            evaluation[category][evaluation_point_index][n_correct_ranked_docs_key] = n_correct_ranked_docs
            evaluation[category][evaluation_point_index][n_ranked_docs_key] = n_docs_in_selection
        print('Evaluated category: '+category)
    return evaluation

def get_summerized_precission(evaluation, evaluation_point_index, n_ranked_docs_key,
                              n_correct_ranked_docs_key):
    sum_correct_ranked_docs = sum([evaluation[category][evaluation_point_index][n_correct_ranked_docs_key] for category in evaluation])
    sum_ranked_docs = sum([evaluation[category][evaluation_point_index][n_ranked_docs_key] for category in evaluation])
    precission = sum_correct_ranked_docs/sum_ranked_docs
    return precission

def get_summerized_recall(evaluation,evaluation_point_index,
                          n_correct_ranked_docs_key, n_docs_in_category_key):
    sum_correct_ranked_docs = sum([evaluation[category][evaluation_point_index][n_correct_ranked_docs_key] for category in evaluation])
    n_categorized_docs = sum([evaluation[category][n_docs_in_category_key] for category in evaluation])
    recall = sum_correct_ranked_docs / n_categorized_docs
    return recall

        
def test_basic_setup(test_categories, categorized_documents, correct_categorization,
                     category_hierarchy, evaluation_points,
                     precission_key, recall_key, n_ranked_docs_key,
                     n_correct_ranked_docs_key,
                     n_docs_in_category_key):
    
    evaluation = evaluate_categorization(test_categories,
                                         categorized_documents, correct_categorization,
                                         category_hierarchy, evaluation_points,
                                         precission_key, recall_key, n_ranked_docs_key,
                                         n_correct_ranked_docs_key,
                                         n_docs_in_category_key)

    pprint.pprint(evaluation)
    precissions = {}
    recalls = {}
    for evaluation_point_index in range(len(evaluation_points)):
        precissions[evaluation_point_index] = get_summerized_precission(evaluation, evaluation_point_index, n_ranked_docs_key, n_correct_ranked_docs_key,)
        recalls[evaluation_point_index] = get_summerized_recall(evaluation, evaluation_point_index, n_correct_ranked_docs_key, n_docs_in_category_key)
        evaluation_point_index +=1
    
    pprint.pprint(precissions)
    pprint.pprint(recalls)
    return evaluation, precissions, recalls


#EVALUATION

reference_words_map = pickle_handler.load_pickle(TEST_DATA_REFERENCE_WORDS_DICE)
all_categories = pickle_handler.load_pickle(TEST_DATA_ALL_CATEGORIES_PICKLE)
test_categories = [category for category in all_categories if len(reference_words_map[category])<15 and '_' not in category]
# 
correct_categorization = pickle_handler.load_pickle(TEST_DATA_CATEGORIZED_DOCUMENTS_PICKLE)
categorized_documents = pickle_handler.load_pickle(RESULT_DICE_BASED_RANKING )
category_hierarchy = TEST_DATA_CATEGORY_HIEARACHY
evaluation_points = list(np.arange(0,1,EVAL_SCALE))
evaluation_points.append(1.0)
evaluation_points.pop(0)
precission_key = PRECISSION_KEY
recall_key = RECALL_KEY
n_ranked_docs_key = N_RANKED_DOCS_KEY
n_correct_ranked_docs_key = N_CORRECT_RANKED_DOCS_KEY
n_docs_in_category_key = N_DOCS_IN_CATEGORY_KEY
e,p,r = test_basic_setup(test_categories, 
                         categorized_documents, correct_categorization,
                         category_hierarchy,evaluation_points,precission_key,
                         recall_key, n_ranked_docs_key,n_correct_ranked_docs_key,n_docs_in_category_key)


# categorized_documents = pickle_handler.load_pickle(RESULT_DICE_BASED_RANKING)

# evaluation_points = list(np.arange(0,1,evaluation_scale))
# evaluation_points.append(1.0)

# correct_categorization = pickle_handler.load_pickle(TEST_DATA_CATEGORIZED_DOCUMENTS_PICKLE)


# def evaluate_categorization(categorized_documents, correct_categorization,
#                             category_hierarchy, evaluation_points,
#                             precission_key, recall_key, n_ranked_docs_key,
#                             n_correct_ranked_docs_key):










