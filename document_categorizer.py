import math
import nltk
import numpy as np
import os
from operator import itemgetter
import pickle
import pprint

import index
import document_processer
import doc_word_simularity
import word_simularities


TEST_DATA_FOLDER = "data/test_data"
TEST_DATA_ALL_CATEGORIES_PICKLE = "data/test_data_pickles/all_categories_pickle"
TEST_DATA_TF_IDF_MAP_PICKLE = "data/test_data_pickles/tf_idf_map_pickle"
TEST_DATA_CATEGORIZED_DOCUMENTS_PICKLE = "data/test_data_pickles/categorized_documents_pickle"
TEST_DATA_REFERENCE_WORDS_DICE = "data/test_data_pickles/test_reference_words_dice"
TEST_DATA_CONTEXT_WORDS_DICE = "data/test_data_pickles/test_context_words_dice"
TEST_CATEGORIES = "data/test_data/test_categories"
RESULT_DICE_BASED_RANKING = "result/dice_based_ranking"

TRAIN_DATA_FOLDER = "data/train_data"
WORD_INDEX_PICKLE_FILE = "data/index_word_pickle_file"
BIGRAM_INDEX_PICKLE_FILE = "data/index_bigram_pickle_file"
NUMBER_OF_DOCUMENTS = 220000
DICE_WEIGHT_FILTER_LIMIT = 0.05
DICE_WORD_FREQUENCY_LIMIT = 0.04

def print_test_data_pickle(resourse, file_path):
    pickle.dump(resourse, open(file_path,'wb'))


def create_test_data_set(test_folder):
    sub_directories = document_processer.get_names_of_subdirectory_list(test_folder)
    document_names = set()
    document_texts = {}
    category_doc_map = {}
    all_categories = []
    for subdirectory in sub_directories:
        category = subdirectory
        if not category == 'no_category': 
            all_categories.append(category)
        category_path = os.path.join(test_folder, subdirectory)
        all_docs = document_processer.get_names_of_files_in_directory(category_path)
        category_doc_map[category]=[]
        for doc in all_docs:
            category_doc_map[category].append(doc)
        
        new_docs = [doc for doc in all_docs if not doc in document_names]
        empty_string = ''
        for doc in new_docs:
            document_names.add(doc)
            doc_path = os.path.join(category_path,doc)

            document_text = document_processer.get_document_string(doc_path, empty_string)
            document_texts[doc] = document_text

    return all_categories, document_texts, category_doc_map


def create_docs_id_tf_idf_map(document_texts, word_index, bigram_index, n_docs):
    max_freq = 0
    for word in word_index:
        if(len(word_index[word])>max_freq):
            max_freq = len(word_index[word])
    doc_tf_idf_map = {}
    # word_tf_idf_values
    for doc in document_texts:
        document = document_texts[doc]
        document = document_processer.preprocess_document(document)
        doc_tf_idf_map[doc] = doc_word_simularity.get_tf_idf_map(document, max_freq, n_docs, word_index)

    for doc in document_texts:
        document = document_texts[doc]
        document = document_processer.preprocess_document(document)
        bigram_freqDist = nltk.FreqDist(nltk.bigrams(document))
        document = list(bigram_freqDist.keys())
        document = [document_processer.bigram_to_string(bigram) for bigram in document ]
        bigram_tf_idf = doc_word_simularity.get_tf_idf_map(document, max_freq, n_docs, bigram_index)
        doc_tf_idf_map[doc].update(bigram_tf_idf)

    return doc_tf_idf_map

def create_dice_keyword_maps(categories, word_index, bigram_index):
    reference_words = {}
    context_words = {}
    for category in categories:
        #default values
        r = [category]
        c = []
        if category not in word_index and category not in bigram_index:
            print('WARNING: Category not in index: '+ category)
        else:
            if '_' in category:
                category_posting_list = bigram_index[category]
                r,c = word_simularities.get_dice_based_key_words(word_index, bigram_index, TRAIN_DATA_FOLDER, category_posting_list, DICE_WEIGHT_FILTER_LIMIT, DICE_WORD_FREQUENCY_LIMIT, NUMBER_OF_DOCUMENTS)
            else:
                category_posting_list = word_index[category]
                r,c = word_simularities.get_dice_based_key_words(word_index, bigram_index, TRAIN_DATA_FOLDER, category_posting_list, DICE_WEIGHT_FILTER_LIMIT, DICE_WORD_FREQUENCY_LIMIT, NUMBER_OF_DOCUMENTS)
        
        category_in_reference_list = [reference_word for reference_word in r if r[0] == category]
        if not category_in_reference_list:
            r.append((category,1.0))
        print(r)
        print(c)
        reference_words[category] = r
        context_words[category] = c
    return reference_words, context_words    

def get_ranked_documents(category, tf_idf_map, reference_words, context_words):
    ranked_documents = []  
    for document in tf_idf_map:
        referens_simularity = doc_word_simularity.get_cosinus_simularity(tf_idf_map[document],reference_words)
        context_simularity = 0
        if not referens_simularity == 0:
            context_simularity = doc_word_simularity.get_cosinus_simularity(tf_idf_map[document], context_words)
        simularity = context_simularity*referens_simularity
        ranked_documents.append((document,simularity))  
    ranked_documents = sorted(ranked_documents, key=itemgetter(1), reverse=True)
    return ranked_documents
 
def categorize(test_categories, tf_idf_map, reference_words, context_words):
    ranked_documents = {}
    for category in test_categories:
        ranked_documents[category] = get_ranked_documents(category,tf_idf_map, reference_words[category],context_words[category])
        print('calculated ranked documents for: '+ category)
    return ranked_documents

def create_dice_based_categorization(test_categories,tf_idf_map, reference_words_map,context_words_map,pickle_file):
    reference_words = {}
    context_words = {}
    for category in test_categories:
        reference_words[category] = set([reference[0] for reference in reference_words_map[category]])
        context_words[category] = set([context_word[0] for context_word in context_words_map[category]])
    categorized_documents = categorize(test_categories,tf_idf_map,reference_words, context_words)
    pprint.pprint(categorized_documents)
    pickle_handler.print_pickle(categorized_documents, pickle_file)


# Create procedure

# all_categories, document_texts, category_doc_map = create_test_data_set(TEST_DATA_FOLDER)
# pickle_handler.print_pickle(category_doc_map, TEST_DATA_CATEGORIZED_DOCUMENTS_PICKLE)
# pickle_handler.print_pickle(all_categories, TEST_DATA_ALL_CATEGORIES_PICKLE)

# word_index = index.get_index(WORD_INDEX_PICKLE_FILE)
# bigram_index = index.get_index(BIGRAM_INDEX_PICKLE_FILE)
# print('loaded index')
# tf_idf_map = create_docs_id_tf_idf_map(document_texts,word_index, bigram_index, NUMBER_OF_DOCUMENTS)
# print('created tf_idf_map')
# pickle_handler.print_pickle(tf_idf_map, TEST_DATA_TF_IDF_MAP_PICKLE)

# all_categories = pickle_handler.load_pickle(TEST_DATA_ALL_CATEGORIES_PICKLE)
# all_categories = [c for c in all_categories if not c =="no_category"]
# print(all_categories)
# word_index = index.get_index(WORD_INDEX_PICKLE_FILE)
# bigram_index = index.get_index(BIGRAM_INDEX_PICKLE_FILE)
# reference_words, context_words =  create_dice_keyword_maps(all_categories, word_index,bigram_index)
# pickle_handler.print_pickle(reference_words,TEST_DATA_REFERENCE_WORDS_DICE)
# pickle_handler.print_pickle(context_words,TEST_DATA_CONTEXT_WORDS_DICE)

# reference_words_map = pickle_handler.load_pickle(TEST_DATA_REFERENCE_WORDS_DICE)
# context_words_map = pickle_handler.load_pickle(TEST_DATA_CONTEXT_WORDS_DICE)
# test_categories = [category for category in reference_words_map if len(reference_words_map[category])<15 and '_' not in category]
# test_categories = pickle_handler.load_pickle(TEST_DATA_ALL_CATEGORIES_PICKLE)

# tf_idf_map = pickle_handler.load_pickle(TEST_DATA_TF_IDF_MAP_PICKLE)


