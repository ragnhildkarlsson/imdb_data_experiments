import nltk
import os
from operator import itemgetter
import pickle


import index
import document_processer
import doc_word_simularity
import word_simularities


TEST_DATA_FOLDER = "data/test_data"
TEST_DATA_ALL_CATEGORIES_PICKLE = "data/all_categories_pickle"
TEST_DATA_TF_IDF_MAP_PICKLE = "data/tf_idf_map_pickle"
TEST_DATA_CATEGORIZED_DOCUMENTS_PICKLE = "data/categorized_documents_pickle"
TEST_DATA_REFERENCE_WORDS_DICE = "data/test_reference_words_dice"
TEST_DATA_CONTEXT_WORDS_DICE = "data/test_context_words_dice"

TRAIN_DATA_FOLDER = "data/train_data"
WORD_INDEX_PICKLE_FILE = "data/index_word_pickle_file"
BIGRAM_INDEX_PICKLE_FILE = "data/index_bigram_pickle_file"
NUMBER_OF_DOCUMENTS = 220000
DICE_WEIGHT_FILTER_LIMIT = 0.05
DICE_WORD_FREQUENCY_LIMIT = 0.04

TEST_DATA_CATEGORY_HIEARACHY = {'religion':{'christianity','buddhism','hinduism','islam','judaism'},'christianity':{'christmas'},'sport':{'baseball','basketball','bicyckle','boxing','football','golf','hockey','skiing','soccer','surfing','swimming','tennis','wrestling','horseracing','olympic_games'},\
                                'water_sport':{'surfing','swimming'}, 'motoring':{'motorcycle','car'}, 'nature':{'animal'},'art':{'cinema','theater','music','opera','classical_music', 'jazz','pop', 'country_music','hip hop', 'dance'}, 'music':{'opera','classical_music','jazz','pop', 'country_music','hip_hop'},\
                                'science':{'medicine','technology','pyscology'},'medicine':{'disability'}, 'education':{'school','collage'}, 'crime':{'prision','mafia','drugs','fraud','gambling','terroism'}}

def load_test_data_pickle(file_path):
    resourse = pickle.load( open(file_path, "rb" ) )
    return resourse

def print_test_data_pickle(resourse, file_path):
    pickle.dump(resourse, open(file_path,'wb'))


def create_test_data_set(test_folder):
    sub_directories = document_processer.get_names_of_subdirectory_list(test_folder)
    document_names = set()
    document_texts = {}
    doc_category_map = {}
    all_categories = []
    for subdirectory in sub_directories:
        category = subdirectory
        all_categories.append(category)
        category_path = os.path.join(test_folder, subdirectory)
        all_docs = document_processer.get_names_of_files_in_directory(category_path)
        doc_category_map[category]=[]
        for doc in all_docs:
            doc_category_map[category].append(doc)
        
        new_docs = [doc for doc in all_docs if not doc in document_names]
        empty_string = ''
        for doc in new_docs:
            document_names.add(doc)
            doc_path = os.path.join(category_path,doc)

            document_text = document_processer.get_document_string(doc_path, empty_string)
            document_texts[doc] = document_text

    return all_categories, document_texts, doc_category_map


def create_docs_id_tfidf_map(document_texts, word_index, bigram_index, n_docs):
    max_freq = 0
    for word in word_index:
        if(len(word_index[word])>max_freq):
            max_freq = len(word_index[word])
    doc_tfidf_map = {}
    # word_tf_idf_values
    for doc in document_texts:
        document = document_texts[doc]
        document = document_processer.preprocess_document(document)
        doc_tfidf_map[doc] = doc_word_simularity.get_tf_idf_map(document, max_freq, n_docs, word_index)

    for doc in document_texts:
        document = document_texts[doc]
        document = document_processer.preprocess_document(document)
        bigram_freqDist = nltk.FreqDist(nltk.bigrams(document))
        document = list(bigram_freqDist.keys())
        document = [document_processer.bigram_to_string(bigram) for bigram in document ]
        bigram_tfidf = doc_word_simularity.get_tf_idf_map(document, max_freq, n_docs, bigram_index)
        doc_tfidf_map[doc].update(bigram_tfidf)

    return doc_tfidf_map


def get_ranked_documents(category, tfidf_map, n_docs, reference_words, context_words):
    ranked_documents = []
    
    n_ranked_docs = 0
    for document in tfidf_map:
        referens_simularity = doc_word_simularity.get_cosinus_simularity(tfidf_map[document],reference_words)
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

def create_dice_keyword_maps(categories, word_index, bigram_index):
    reference_words = {}
    context_words = {}
    for category in categories:
        #default values
        r = [category]
        c = []
        if category not in word_index and category not in bigram_index:
            print('WARNING: Category not in index: '+ category)
        if '_' in category:
            category_posting_list = bigram_index[category]
            r,c = word_simularities.get_dice_based_key_words(word_index, bigram_index, TRAIN_DATA_FOLDER, category_posting_list, DICE_WEIGHT_FILTER_LIMIT, DICE_WORD_FREQUENCY_LIMIT, NUMBER_OF_DOCUMENTS)
        else:
            category_posting_list = word_index[category]
            r,c = word_simularities.get_dice_based_key_words(word_index, bigram_index, TRAIN_DATA_FOLDER, category_posting_list, DICE_WEIGHT_FILTER_LIMIT, DICE_WORD_FREQUENCY_LIMIT, NUMBER_OF_DOCUMENTS)
        print(r)
        print(c)
        reference_words[category] = r
        context_words[category] = c
    return reference_words, context_words    
 
def dice_based_categorisation(test_categories, tf_idf_map, categorized_docs, category_hierarchy, n_docs, pr_evaluation_scale):
    pass    

# test_categories = categories = {'airplane', 'aliens', 'animals', 'art', 'baseball', 'basketball','bicyckle', 'boxing', 'buddhism', 'bussiness','car','christianity', 'christmas','cinema','classical music'}
# tfidf_map = load_test_data_pickle(TEST_DATA_TF_IDF_MAP_PICKLE)
# categorized_docs = load_test_data_pickle(TEST_DATA_CATEGORIZED_DOCUMENTS_PICKLE)


# category = "airplane"
# category_posting_list = word_index[category]
# reference_words, context_words = word_simularities.get_dice_based_key_words(word_index, bigram_index, TRAIN_DATA_FOLDER, category_posting_list, DICE_WEIGHT_FILTER_LIMIT, DICE_WORD_FREQUENCY_LIMIT, NUMBER_OF_DOCUMENTS)
# print(reference_words)
# print(context_words)
# reference_words = set([reference[0] for reference in reference_words])
# context_words = set([context_word[0] for context_word in context_words])
# ranked_docs = get_ranked_documents(category, tfidf_map, NUMBER_OF_DOCUMENTS, reference_words,context_words)
# print(ranked_docs)



# Create procedure

# all_categories, document_texts, doc_category_map = create_test_data_set(TEST_DATA_FOLDER)
# print_test_data_pickle(doc_category_map, TEST_DATA_CATEGORIZED_DOCUMENTS_PICKLE)
# print_test_data_pickle(all_categories, TEST_DATA_ALL_CATEGORIES_PICKLE)

# word_index = index.get_index(WORD_INDEX_PICKLE_FILE)
# bigram_index = index.get_index(BIGRAM_INDEX_PICKLE_FILE)
# print('loaded index')
# tfidf_map = create_docs_id_tfidf_map(document_texts,word_index, bigram_index, NUMBER_OF_DOCUMENTS)
# print('created tfidf_map')
# print_test_data_pickle(tfidf_map, TEST_DATA_TF_IDF_MAP_PICKLE)
word_index = index.get_index(WORD_INDEX_PICKLE_FILE)
bigram_index = index.get_index(BIGRAM_INDEX_PICKLE_FILE)
print('loaded index')
all_categories = load_test_data_pickle(TEST_DATA_ALL_CATEGORIES_PICKLE)
all_categories = [c for c in all_categories if not c =="no_category"]
reference_words, context_words, create_dice_keyword_maps(all_categories, word_index,bigram_index)
print_test_data_pickle(reference_words,TEST_DATA_REFERENCE_WORDS_DICE)
print_test_data_pickle(reference_words,TEST_DATA_CONTEXT_WORDS_DICE)
