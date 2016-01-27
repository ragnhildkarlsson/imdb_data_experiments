import os
import document_processer
import index
import pickle_handler
import word_simularities

TEST_DATA_FOLDER = "data/test_data"
TRAIN_DATA_FOLDER = "data/train_data"
TEST_DATA_GOLD_STANDARD_CATEGORIZATION = "data/test_data_pickles/gold_standard_categorization_pickle"
TEST_DATA_ALL_CATEGORIES_LIST = "data/test_data_pickles/all_categories_list"
TEST_DATA_TF_IDF_MAP = "data/test_data_pickles/tf_idf_map_pickle"
TEST_DATA_DICE_BASED_KEYWORD_RANKING = "data/test_data_pickles/dice_based_keyword_rankning"

BASIC_REFERENCE_WORDS_DICE = "data/test_data_pickles/default_reference_words_dice"
BASIC_CONTEXT_WORDS_DICE = "data/test_data_pickles/default_contexts_words_dice"
TEST_CATEGORIES = "data/test_data/test_categories"


FREQUENT_WORDS_SET = "data/index/frequent_words"
WORD_INDEX_PICKLE_FILE = "data/index/index_word_pickle_file"
BIGRAM_INDEX_PICKLE_FILE = "data/index/index_bigram_pickle_file"
NUMBER_OF_DOCUMENTS_IN_CORPUS = 220000
NUMBER_OF_RANKED_KEYWORDS_DICE = 1000
DICE_WEIGHT_FILTER_LIMIT = 0.05

def create_frequent_words_set(index, frequent_words_limit, n_docs_in_corpus):
    frequent_words = index.get_freequent_words(index, doc_frequency_limit, n_docs_in_corpus)
    return frequent_words


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


def create_dice_ranked_keywords(n,categories, word_index, bigram_index, train_data_folder, bigram_delimeter):
    reference_words = {}
    context_words = {}
    categories_keywords_map = {}
    n_categories_left  = len(categories)
    for category in categories:
        print(str(n_categories_left) + " categories left to process")
        print("Calculate dice keywords for category "+ category)
        dice_ranked_keywords = []
        if category not in word_index and category not in bigram_index:
            print('WARNING: Category not in index: '+ category)
        else:
            category_posting_list = []
            if bigram_delimeter in category:
                category_posting_list = bigram_index[category]
            else:
                category_posting_list = word_index[category]
            
            print("Number of docs with category name: " + str(len(category_posting_list)))
            dice_ranked_keywords = word_simularities.get_n_dice_based_key_words(n, word_index, bigram_index, train_data_folder, category_posting_list)
           
        #ensure category name have high rank
        dice_ranked_keywords = [ranked_word for ranked_word in dice_ranked_keywords if ranked_word[0] != category]
        dice_ranked_keywords.append((category,1.0))
        categories_keywords_map[category] = dice_ranked_keywords
        n_categories_left = n_categories_left -1
    return categories_keywords_map


def get_basic_keywords_dice(keyword_ranking_dice,
                            weight_limit_reference_words,
                            all_categories):
    reference_words = {}
    context_words = {}
    for category in all_categories:
        r, c = keyword_experiments.get_default_keywords_dice(keyword_ranking_dice, weight_limit_reference_words)
        reference_words[category] = r
        context_words[category] = c

    return reference_words, context_words

def get_basic_test_categories(all_categories,
                              bigram_delimeter,
                              reference_words):
    test_categories = [category for category in all_categories if len(reference_words[category])<13 and not bigram_delimeter in category]
    return test_categories


# CREATE PROCEDURE

# TEST DATA SET
# all_categories, document_texts, gold_standard_categorization  = create_test_data_set(TEST_DATA_FOLDER)
# pickle_handler.print_pickle(all_categories, TEST_DATA_ALL_CATEGORIES_LIST)
# pickle_handler.print_pickle(gold_standard_categorization,TEST_DATA_GOLD_STANDARD_CATEGORIZATION)

# DICE RANKED KEYWORDS 
# word_index = pickle_handler.load_pickle(WORD_INDEX_PICKLE_FILE)
# bigram_index = pickle_handler.load_pickle(BIGRAM_INDEX_PICKLE_FILE)
# train_data_folder = TRAIN_DATA_FOLDER
# all_categories = pickle_handler.load_pickle(TEST_DATA_ALL_CATEGORIES_LIST)
# dice_ranked_keywords = create_dice_ranked_keywords(NUMBER_OF_RANKED_KEYWORDS_DICE, all_categories, word_index,bigram_index,train_data_folder, BIGRAM_DELIMETER)
# pickle_handler.print_pickle(dice_ranked_keywords,TEST_DATA_DICE_BASED_KEYWORD_RANKING)

# FREQUENT WORDS
word_index = pickle_handler.load_pickle(WORD_INDEX_PICKLE_FILE)
frequent_words = create_frequent_words_set()
pickle_handler.print_pickle(frequent_words, FREQUENT_WORDS_SET)

#BASIC DICE KEYWORDS
all_categories = pickle_handler.load_pickle(TEST_DATA_ALL_CATEGORIES_LIST)
keyword_ranking_dice = pickle_handler.load_pickle()
reference_words, context_words = get_default_keywords_dice(keyword_ranking_dice, DICE_WEIGHT_FILTER_LIMIT, all_categories)
pickle_handler.print_pickle(reference_words, BASIC_REFERENCE_WORDS_DICE)
pickle_handler.print_pickle(context_words, BASIC_CONTEXT_WORDS_DICE)

#TEST CATEGORIES
all_categories = pickle_handler.load_pickle(TEST_DATA_ALL_CATEGORIES_LIST)
reference_words = pickle_handler.load_pickle(BASIC_REFERENCE_WORDS_DICE)
test_categories = get_basic_test_categories(all_categories,BIGRAM_DELIMETER,reference_words)

