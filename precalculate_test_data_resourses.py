import os
import document_processer
import pickle_handler

TEST_DATA_FOLDER = "data/test_data"
TRAIN_DATA_FOLDER = "data/train_data"
TEST_DATA_GOLD_STANDARD_CATEGORIZATION = "data/test_data_pickles/gold_standard_categorization_pickle"
TEST_DATA_ALL_CATEGORIES_LIST = "data/test_data_pickles/all_categories_list"
TEST_DATA_TF_IDF_MAP = "data/test_data_pickles/tf_idf_map_pickle"
TEST_DATA_DICE_BASED_KEYWORD_RANKING = "data/test_data_pickles/dice_based_keyword_rankning"

WORD_INDEX_PICKLE_FILE = "data/index/index_word_pickle_file"
BIGRAM_INDEX_PICKLE_FILE = "data/index/index_bigram_pickle_file"
NUMBER_OF_DOCUMENTS = 220000
NUMBER_OF_RANKED_KEYWORDS_DICE = 1000

def create_test_data_set(test_folder):
    sub_directories = document_processer.get_names_of_subdirectory_list(test_folder)
    document_names = set()
    document_texts = {}
    gold_standard_categorization = {}
    all_categories = []
    for subdirectory in sub_directories:
        category = subdirectory
        if not category == 'no_category': 
            all_categories.append(category)
        category_path = os.path.join(test_folder, subdirectory)
        all_docs = document_processer.get_names_of_files_in_directory(category_path)
        gold_standard_categorization[category]=[]
        for doc in all_docs:
            gold_standard_categorization[category].append(doc)
        
        new_docs = [doc for doc in all_docs if not doc in document_names]
        empty_string = ''
        for doc in new_docs:
            document_names.add(doc)
            doc_path = os.path.join(category_path,doc)

            document_text = document_processer.get_document_string(doc_path, empty_string)
            document_texts[doc] = document_text

    return all_categories, document_texts, gold_standard_categorization


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


def create_dice_ranked_keywords(n,categories, word_index, bigram_index, train_data_folder):
    reference_words = {}
    context_words = {}
    categories_keywords_map = {}
    for category in categories:
        print("Calculate dice keywords for category "+ category)
        dice_ranked_keywords = []
        if category not in word_index and category not in bigram_index:
            print('WARNING: Category not in index: '+ category)
        else:
            if '_' in category:
                category_posting_list = bigram_index[category]
                print("Number of docs with category name: " + str(len(category_posting_list))
                dice_ranked_keywords = word_simularities.get_n_dice_based_key_words(n, word_index, bigram_index, train_data_folder, category_posting_list)
            else:
                category_posting_list = word_index[category]
                print("Number of docs with category name: " + str(len(category_posting_list))
                r,c = word_simularities.get_dice_based_key_words(word_index, bigram_index, train_data_folder, category_posting_list, DICE_WEIGHT_FILTER_LIMIT, DICE_WORD_FREQUENCY_LIMIT, NUMBER_OF_DOCUMENTS)

        #ensure category name have high rank
        dice_ranked_keywords = [ranked_word for ranked_word in dice_ranked_keywords if ranked_word[0] != category]
        dice_ranked_keywords.append((category,1.0))
        categories_keywords_map[category] = dice_ranked_keywords
    return categories_keywords_map

# CREATE PROCEDURE
# all_categories, document_texts, gold_standard_categorization  = create_test_data_set(TEST_DATA_FOLDER)
# pickle_handler.print_pickle(all_categories, TEST_DATA_ALL_CATEGORIES_LIST)
# pickle_handler.print_pickle(gold_standard_categorization,TEST_DATA_GOLD_STANDARD_CATEGORIZATION)

word_index = pickle_handler.load_pickle(WORD_INDEX_PICKLE_FILE)
bigram_index = pickle_handler.load_pickle(BIGRAM_INDEX_PICKLE_FILE)
train_data_folder = TRAIN_DATA_FOLDER
all_categories = pickle_handler.load_pickle(TEST_DATA_ALL_CATEGORIES_LIST)
dice_ranked_keywords = create_dice_ranked_keywords(NUMBER_OF_RANKED_KEYWORDS_DICE,all_categories, word_index,bigram_index,train_data_folder)
pickle_handler.print_pickle(dice_ranked_keywords,TEST_DATA_DICE_BASED_KEYWORD_RANKING)
