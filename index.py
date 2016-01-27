import os
import pickle_handler

import document_processer 

def get_index(file_path):
    index = pickle_handler.load_pickle(file_path)
    return index

def intersection_search(posting_list_1, posting_list_2):
    result =[]
    p1_index = 0
    p2_index = 0
    while(p1_index<len(posting_list_1)) and (p2_index<len(posting_list_2)):
        p1_doc_id = posting_list_1[p1_index][0]
        p2_doc_id = posting_list_2[p2_index][0]
        if p1_doc_id == p2_doc_id:
            result.append(p1_doc_id)
            p1_index += 1
            p2_index += 1
        elif p1_doc_id < p2_doc_id:
            p1_index += 1
        else:
            p2_index += 1        

    return result;
    
def get_freequent_words(index, doc_frequency_limit, n_docs_in_corpus):
    frequent_words =set()
    for word in index:
        n_docs = len(index[word])
        if doc_frequency_limit < (n_docs / n_docs_in_corpus):
            frequent_words.add(word)
    return frequent_words