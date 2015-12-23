import os
import pickle

import document_processer 

def get_index(file_path):
    index = pickle.load( open(file_path, "rb" ) )
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
    
