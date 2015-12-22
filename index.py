import os
import pickle

import document_processer 

def get_index_keys(index_folder):
    terms = document_processer.get_names_of_files_in_directory(index_folder)
    return set(terms) 

def get_posting_list(word, index_folder):
    file_path = os.path.join(index_folder, word)
    posting_list = pickle.load( open( file_path, "rb" ) )
    return posting_list

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
    
