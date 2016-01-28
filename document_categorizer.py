import doc_word_simularity
 
def categorize(test_categories, tf_idf_map, reference_words, context_words):
    ranked_documents = {}
    for category in test_categories:
        ranked_documents[category] = doc_word_simularity.get_cosinus_ranked_documents(category,tf_idf_map, reference_words[category],context_words[category])
        print('calculated ranked documents for: '+ category)
    return ranked_documents







