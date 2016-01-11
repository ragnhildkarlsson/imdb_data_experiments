import nltk
import os


import index
import document_processer
import doc_word_simularity
import word_simularities


TEST_DATA_FOLDER = "data/test_sub"

WORD_INDEX_PICKLE_FILE = "data/index_word_pickle_file"
BIGRAM_INDEX_PICKLE_FILE = "data/index_bigram_pickle_file"
NUMBER_OF_DOCUMENTS = 220000


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

    for document in tfidf_map:

    pass
    
test_docs = get_all_test_files_list(TEST_DATA_FOLDER)
print(test_docs)
word_index = index.get_index(WORD_INDEX_PICKLE_FILE)
bigram_index = index.get_index(BIGRAM_INDEX_PICKLE_FILE)
print('loaded index')
tfidf_map = get_docs_id_tfidf_map(test_docs,word_index,bigram_index,NUMBER_OF_DOCUMENTS)
print(tfidf_map)
category = "airplanes"
referens_words context_words = word_simularities() 
