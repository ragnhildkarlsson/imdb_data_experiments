import nltk
import os
from operator import itemgetter


import index
import document_processer
import doc_word_simularity
import word_simularities


TEST_DATA_FOLDER = "data/test_sub"
TRAIN_DATA_FOLDER = "data/train_data"

WORD_INDEX_PICKLE_FILE = "data/index_word_pickle_file"
BIGRAM_INDEX_PICKLE_FILE = "data/index_bigram_pickle_file"
NUMBER_OF_DOCUMENTS = 220000
DICE_WEIGHT_FILTER_LIMIT = 0.05
DICE_WORD_FREQUENCY_LIMIT = 0.04



def get_all_test_files_list(test_folder):
    sub_directories = document_processer.get_names_of_subdirectory_list(test_folder)
    test_document_paths = []
    for subdirectory in sub_directories:
        subdir_path = os.path.join(test_folder, subdirectory)
        docs = document_processer.get_names_of_files_in_directory(subdir_path)
        test_document_paths = test_document_paths + [os.path.join(subdir_path,doc) for doc in docs]
    return test_document_paths

def get_docs_id_tfidf_map(test_files_path, word_index, bigram_index, n_docs):
    max_freq = 0
    for word in word_index:
        if(len(word_index[word])>max_freq):
            max_freq = len(word_index[word])
    doc_tfidf_map = {}
    # word_tf_idf_values
    for doc_path in test_files_path:
        empty_string = ''
        document = document_processer.get_document_string(doc_path, empty_string)
        document = document_processer.preprocess_document(document)
        doc_tfidf_map[doc_path] = doc_word_simularity.get_tf_idf_map(document, max_freq, n_docs, word_index)

    for doc_path in test_files_path:
        document = document_processer.get_document_string(doc_path, empty_string)
        document = document_processer.preprocess_document(document)
        bigram_freqDist = nltk.FreqDist(nltk.bigrams(document))
        document = list(bigram_freqDist.keys())
        document = [document_processer.bigram_to_string(bigram) for bigram in document ]
        bigram_tfidf = doc_word_simularity.get_tf_idf_map(document, max_freq, n_docs, bigram_index)
        doc_tfidf_map[doc_path].update(bigram_tfidf)

    return doc_tfidf_map


def get_ranked_documents(category, tfidf_map, n_docs, referens_words, context_words):
    ranked_documents = []
    
    n_ranked_docs = 0
    for document in tfidf_map:
        referens_simularity = doc_word_simularity.get_cosinus_simularity(tfidf_map[document],referens_words)
        context_simularity = 0
        if not referens_simularity == 0:
            context_simularity = doc_word_simularity.get_cosinus_simularity(tfidf_map[document], context_words)
        simularity = context_simularity*referens_simularity
        ranked_documents.append((document,simularity))
        if((n_ranked_docs % 100) == 0):
            print(n_ranked_docs)
        n_ranked_docs += 1

    ranked_documents = sorted(ranked_documents, key=itemgetter(1), reverse=True)
    return ranked_documents


test_docs = get_all_test_files_list(TEST_DATA_FOLDER)
print(test_docs)
word_index = index.get_index(WORD_INDEX_PICKLE_FILE)
bigram_index = index.get_index(BIGRAM_INDEX_PICKLE_FILE)
print('loaded index')
tfidf_map = get_docs_id_tfidf_map(test_docs,word_index,bigram_index,NUMBER_OF_DOCUMENTS)
category = "airplanes"
category_posting_list = word_index[category]
referens_words, context_words = word_simularities.get_dice_based_key_words(word_index, bigram_index, TRAIN_DATA_FOLDER, category_posting_list, DICE_WEIGHT_FILTER_LIMIT, DICE_WORD_FREQUENCY_LIMIT, NUMBER_OF_DOCUMENTS)
print(referens_words)
print(context_words)
ranked_docs = get_ranked_documents(category, tfidf_map, NUMBER_OF_DOCUMENTS, referens_words,context_words)
print(ranked_docs)