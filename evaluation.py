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

def get_selection_of_ranked_docs(ranked_documents, evaluation_point):
    min_score = ranked_documents[-1][1]
    max_score = ranked_documents[0][1]
    score_range = max_score-min_score
    min_score_in_selection = min_score + ((1 - evaluation_point)*score_range)
    selected_ranked_documents = [ranked_doc for ranked_doc in ranked_documents if ranked_doc[1] >= min_score_in_selection]
    return selected_ranked_documents        

def get_document_in_category(category, gold_standard_categorization, category_hierarchy):
    documents_in_category = set(gold_standard_categorization[category])
    if category in category_hierarchy:
        for sub_category in category_hierarchy[category]:
            documents_in_category.update(set(gold_standard_categorization[sub_category]))
    return documents_in_category

def evaluate_categorization(test_categories,
                            ranked_documents, gold_standard_categorization,
                            category_hierarchy, evaluation_points,
                            precission_key, recall_key, n_ranked_docs_key,
                            n_correct_ranked_docs_key,
                            n_docs_in_category_key):
    evaluation = {}
    for category in test_categories:
        evaluation[category] = {}
        ranked_to_category = ranked_documents[category]
        documents_in_category = get_document_in_category(category, gold_standard_categorization,category_hierarchy)
        n_docs_in_category = len(documents_in_category)            
        evaluation[category][n_docs_in_category_key] = n_docs_in_category

        for evaluation_point_index in range(len(evaluation_points)):
            evaluation_point = evaluation_points[evaluation_point_index]
            evaluation[category][evaluation_point_index] = {}
            n_non_zero_ranked_docs = len(ranked_to_category)
            # handle the case of no ranked  docs
            if(n_non_zero_ranked_docs==0):
                evaluation[category][evaluation_point_index][recall_key] = 0
                evaluation[category][evaluation_point_index][precission_key] = 0
                evaluation[category][evaluation_point_index][n_ranked_docs_key] = 0
                evaluation[category][evaluation_point_index][n_correct_ranked_docs_key] = 0
                continue  
            # create a selection for the evaluation point
            selected_ranked_documents = get_selection_of_ranked_docs(ranked_to_category,evaluation_point)
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

def get_n_correct_ranked_documnets(ranked_to_category, documents_in_category):
    n_correct_ranked_docs = 0
    for doc in ranked_to_category:
        if doc[0] in documents_in_category:
            n_correct_ranked_docs +=1
    return n_correct_ranked_docs

def get_optimal_selection_indices(ranked_to_category, documents_in_category, precission_levels):
    precission_selction_map = {}
    selection_index_precission_tuples = []
    for selection_index in range(len(ranked_to_category)):
        if selection_index == 0:
            continue
        selected_ranked_documents = ranked_to_category[:selection_index]
        n_correct_ranked_docs =  get_n_correct_ranked_documnets(selected_ranked_documents, documents_in_category)
        if n_correct_ranked_docs == 0:
            continue
        selection_index_precission_tuples.append((selection_index,(n_correct_ranked_docs/len(selected_ranked_documents))))    
    #find optimal selections
    selection_index_precission_tuples.reverse()
    for precission_level_index in range(len(precission_levels)):
        precission_level = precission_levels[precission_level_index]
        diffs = [(selection[0], abs(selection[1] - precission_level)) for selection in selection_index_precission_tuples]
        if not diffs:
            precission_selction_map[precission_level_index]= 0
        else:    
            min_diff_selection =  min(diffs, key= itemgetter(1))
            precission_selction_map[precission_level_index]= min_diff_selection[0]
    return precission_selction_map

def get_threshold_optimized_evaluation(test_categories,
                                       ranked_documents, gold_standard_categorization,
                                       category_hierarchy, evaluation_points,
                                       precission_key, recall_key, n_ranked_docs_key,
                                       n_correct_ranked_docs_key,
                                       n_docs_in_category_key):
    evaluation = {}
    for category in test_categories:
        evaluation[category] = {}
        ranked_to_category = ranked_documents[category]
        documents_in_category = get_document_in_category(category, gold_standard_categorization,category_hierarchy)
        n_docs_in_category = len(documents_in_category)            
        optimal_selection_indices_map = get_optimal_selection_indices(ranked_to_category,documents_in_category,evaluation_points)
        evaluation[category][n_docs_in_category_key] = n_docs_in_category
        for evaluation_point_index in range(len(evaluation_points)):
            evaluation_point = evaluation_points[evaluation_point_index]
            evaluation[category][evaluation_point_index] = {}
            n_non_zero_ranked_docs = len(ranked_to_category)
            
            # create a selection for the evaluation point
            selection_index = optimal_selection_indices_map[evaluation_point_index]
            selected_ranked_documents = ranked_to_category[:selection_index]
            n_docs_in_selection =  len(selected_ranked_documents)
            
            # handle the case of no relevant ranked  docs
            if(n_docs_in_selection==0):
                evaluation[category][evaluation_point_index][recall_key] = 0
                evaluation[category][evaluation_point_index][precission_key] = 0
                evaluation[category][evaluation_point_index][n_ranked_docs_key] = 0
                evaluation[category][evaluation_point_index][n_correct_ranked_docs_key] = 0
                print("Warning:" + category)
                continue  
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

def get_summarized_precission(evaluation, evaluation_point_index, n_ranked_docs_key,
                              n_correct_ranked_docs_key):
    sum_correct_ranked_docs = sum([evaluation[category][evaluation_point_index][n_correct_ranked_docs_key] for category in evaluation])
    sum_ranked_docs = sum([evaluation[category][evaluation_point_index][n_ranked_docs_key] for category in evaluation])
    precission = sum_correct_ranked_docs/sum_ranked_docs
    return precission

def get_summarized_recall(evaluation,evaluation_point_index,
                          n_correct_ranked_docs_key, n_docs_in_category_key):
    sum_correct_ranked_docs = sum([evaluation[category][evaluation_point_index][n_correct_ranked_docs_key] for category in evaluation])
    n_categorized_docs = sum([evaluation[category][n_docs_in_category_key] for category in evaluation])
    recall = sum_correct_ranked_docs / n_categorized_docs
    return recall

def get_summarized_recalls(evaluation,evaluation_points,n_correct_ranked_docs_key,n_docs_in_category_key):
    summarized_recalls = {}
    for evaluation_point_index in range(len(evaluation_points)):           
        summarized_recalls[evaluation_point_index] = get_summarized_recall(evaluation, evaluation_point_index, n_correct_ranked_docs_key, n_docs_in_category_key)
    return summarized_recalls

def get_summarized_precissions(evaluation, evaluation_points, n_ranked_docs_key,
                              n_correct_ranked_docs_key):
    summarized_precissions = {}
    for evaluation_point_index in range(len(evaluation_points)): 
        summarized_precissions[evaluation_point_index] = get_summarized_precission(evaluation, evaluation_point_index, n_ranked_docs_key, n_correct_ranked_docs_key,)
    return summarized_precissions    

