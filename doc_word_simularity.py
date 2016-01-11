import math


def get_tf_idf_map(document, max_freq, n_docs, index):
    tf_idf_map = {}
    
    for term in document:
        tf = 0
        idf = math.log(n_docs)
        if term in index and term not in tf_idf_map: 
            posting_list = index[term]
            freq_term = sum([post[1] for post in posting_list]) 
            tf = 0.5 + 0.5*(freq_term/max_freq)
            idf = math.log(1 + (n_docs/len(posting_list)))
        if term not in tf_idf_map:
            tf_idf_map[term] = tf * idf

    return tf_idf_map

def get_cosinus_simularity(tf_idf_map, key_words):
    sum_common_terms = 0
    sum_tf_idf_terms = 0
    for term in tf_idf_map:
        if term in key_words:
            sum_common_terms += tf_idf_map[term]
        sum_tf_idf_terms += math.pow(tf_idf_map[term],2)


    cosinus_similarity = sum_common_terms/(math.sqrt(sum_tf_idf_terms)+math.sqrt(len(key_words)))
    return cosinus_similarity    
