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


def get_document_in_category(category, gold_standard_categorization, category_hierarchy):
    documents_in_category = set(gold_standard_categorization[category])
    if category in category_hierarchy:
        for sub_category in category_hierarchy[category]:
            documents_in_category.update(set(gold_standard_categorization[sub_category]))
    return documents_in_category

def get_n_correct_ranked_documents(ranked_to_category,documents_in_category):
    n_correct_ranked_docs = 0
    for doc in ranked_to_category:
        if doc[0] in documents_in_category:
            n_correct_ranked_docs +=1
    return n_correct_ranked_docs

def get_precission(ranked_to_category, documents_in_category):
    n_correct_ranked_docs = get_n_correct_ranked_documents(ranked_to_category,documents_in_category)
    n_documents_in_selection = len(ranked_to_category)                
    if n_documents_in_selection == 0:
        return 0
    else:
        return n_correct_ranked_docs/n_documents_in_selection   

def get_precission_selection_indices(ranked_to_category, documents_in_category, precission_levels):
    precission_selction_map = {}
    selection_index_precission_tuples = []
    
    # Calculate precission for different selections
    all_possible_selections = range(len(ranked_to_category)+1)

    for selection_index in all_possible_selections:
        selected_ranked_documents = ranked_to_category[:selection_index]
        precission_in_selection = get_precission(selected_ranked_documents, documents_in_category)
        selection_index_precission_tuples.append((selection_index,precission_in_selection))

    #find optimal selections
    for precission_level_index in range(len(precission_levels)):
        precission_level = precission_levels[precission_level_index]
        # Find selections that maintain precission level
        higer_precission_selections = [selection for selection in selection_index_precission_tuples if (selection[1] - precission_level) >= 0]
        # Sort on selection index deccending
        higer_precission_selections.sort(key=itemgetter(0),reverse = True)
        # Return a selection
        if not higer_precission_selections:
            precission_selction_map[precission_level_index] = 0
        else:
            best_selection = higer_precission_selections[0]   
            precission_selction_map[precission_level_index] = best_selection[0]
    return precission_selction_map

def get_percentage_selection_indices(ranked_documents, percentage_levels):
    percentage_selection_map = {}
    for percentage_level_index in range(len(percentage_levels)):
        percentage_level = percentage_levels[percentage_level_index]
        min_score = ranked_documents[-1][1]
        max_score = ranked_documents[0][1]
        score_range = max_score-min_score
        min_score_in_selection = min_score + ((1 - percentage_level)*score_range)
        higer_score_selection_indices = [selection_index+1 for selection_index in range(len(ranked_documents)) if ranked_documents[selection_index][1] >= min_score_in_selection]
        higer_score_selection_indices.sort(reverse = True)
        if not higer_score_selection_indices:
            percentage_selection_map[percentage_level_index] = 0        
        else:
            best_selection = higer_score_selection_indices[0]
            percentage_selection_map[percentage_level_index] = best_selection
    
    return percentage_selection_map        

def get_evaluation(test_categories,
                   ranked_documents, gold_standard_categorization,
                   category_hierarchy, evaluation_selections,
                   precission_key, recall_key, n_ranked_docs_key,
                   n_correct_ranked_docs_key,
                   n_docs_in_category_key,
                   ):
    evaluation = {}
    evaluation_level_indices = list(evaluation_selections.keys())
    for category in test_categories:
        evaluation[category] = {}
        
        ranked_to_category = ranked_documents[category]
        
        documents_in_category = get_document_in_category(category, gold_standard_categorization,category_hierarchy)
        
        n_docs_in_category = len(documents_in_category)            
        
        evaluation[category][n_docs_in_category_key] = n_docs_in_category
        for evaluation_level_index in evaluation_level_indices:
            evaluation_selection_index = evaluation_selections[evaluation_level_index] 
            selected_ranked_documents =ranked_documents[evaluation_selection_index]            
            n_docs_in_selection =  len(selected_ranked_documents)
            # evaluate precission and recall in selection
            n_correct_ranked_docs = get_n_correct_ranked_documents(selected_ranked_documents, documents_in_category)
            precission_in_selection = get_precission(selected_ranked_documents, documents_in_category)
            evaluation[category][evaluation_level_index][precission_key] = precission_in_selection
            evaluation[category][evaluation_level_index][recall_key] = (n_correct_ranked_docs/n_docs_in_category)
            evaluation[category][evaluation_level_index][n_correct_ranked_docs_key] = n_correct_ranked_docs
            evaluation[category][evaluation_level_index][n_ranked_docs_key] = n_docs_in_selection

        print('Evaluated category: '+ category)


def get_summarized_precissions(test_categories, evaluation, evaluation_level_indices, n_ranked_docs_key, n_correct_ranked_docs_key):
    summarized_precissions = {}

    for evaluation_level_index in evaluation_level_indices: 
        sum_correct_ranked_docs = sum([evaluation[category][evaluation_level_index][n_correct_ranked_docs_key] for category in test_categories])
        sum_ranked_docs = sum([evaluation[category][evaluation_level_index][n_ranked_docs_key] for category in test_categories])
        precission = sum_correct_ranked_docs/sum_ranked_docs
        summarized_precissions[evaluation_level_index] = precission
    return summarized_precissions    

def get_summarized_recalls(test_categories, evaluation, evaluation_indices, n_correct_ranked_docs_key, n_docs_in_category_key):
    summarized_recalls = {}    
    total_n_of_classifications = sum([evaluation[category][n_docs_in_category_key] for category in test_categories])  
    for evaluation_level_index in evaluation_indices:   
        total_n_of_correct_classified_documents =  sum([evaluation[category][n_docs_in_category_key][_correct_ranked_docs_key] for category in test_categories])
        summarized_recalls[evaluation_level_index] = total_n_of_correct_classified_documents/total_n_of_classifications
    return summarized_recalls
