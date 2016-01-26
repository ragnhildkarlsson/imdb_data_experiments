def get_keywords_with_too_high_doc_frequency(frequency_limit, n_docs_in_corpus, index, key_words):
    key_words_with_too_high_doc_frequency = [k for k in key_words if frequency_limit < (len(index[k])/n_docs_in_corpus)

def get_keywords_over_score_limit(score_limit, key_words, ):
    over_limit = [k for k in key_words if score_limit <= k[1]]
    return over_limit

def get_keywords_under_score_limit(score_limit, key_words):
    under_limit = [k for k in key_words if score_limit > k[1]]
    return under_limit

def get_bigram_with_category_name(category_name, bigram_delimeter, key_words):
    bigram_with_category_name = [k for k in key_words if category_name in k[0] and bigram_delimeter in k[0]]
    return bigram_with_category_name

