import os
import nltk
from operator import itemgetter
import index
import document_processer 

ROOT_DATA_FOLDER = "data"
TRAIN_DATA_FOLDER = "data/train_data"
FILENAME_TRAIN_DATA_SUBSET = "list_train_data_subset"
WORD_INDEX_PICKLE_FILE = "data/index_word_pickle_file"
BIGRAM_INDEX_PICKLE_FILE = "data/index_bigram_pickle_file"
NUMBER_OF_DOCUMENTS = 220000

def get_list_of_categories(test_data_folder):
    return next(os.walk(test_data_folder))[1]

def calculate_dice_coefficients_word(category_posting_list, word_index, train_data_folder):
    # returns {w1:dice(cat,w1),w2:dice(cat,w2)}
    dice_coefficents = {}
    print("Number of docs")
    print(len(category_posting_list))
    calculated_posts = 0
    for post in category_posting_list:
        doc_file_mame = str(post[0])
        document = document_processer.get_document_string(doc_file_mame, train_data_folder)
        document = document_processer.preprocess_document(document)
        for word in document:
            if word not in dice_coefficents and word in word_index:
                word_posting_list = word_index[word]
                intersection = index.intersection_search(category_posting_list, word_posting_list)
                dice_coefficents[word] = len(intersection)/(len(word_posting_list)+len(category_posting_list))
        calculated_posts +=1
        if(calculated_posts%5 ==0):
            print(calculated_posts)
    return dice_coefficents

def calculate_dice_coefficients_bigram(category_posting_list, bigram_index, train_data_folder):
    dice_coefficents = {}
    calculated_posts = 0
    for post in category_posting_list:
        doc_file_mame = str(post[0])
        document = document_processer.get_document_string(doc_file_mame,train_data_folder)
        document = document_processer.preprocess_document(document)    
        bigram_freqDist = nltk.FreqDist(nltk.bigrams(document))
        for bigram in bigram_freqDist:
            bigram_string = document_processer.bigram_to_string(bigram)
            if bigram_string not in dice_coefficents and bigram_string in bigram_index:
                bigram_posting_list = bigram_index[bigram_string]
                intersection = index.intersection_search(category_posting_list, bigram_posting_list)
                dice_coefficents[bigram_string] = len(intersection)/(len(bigram_posting_list)+len(category_posting_list))
        calculated_posts +=1
        if(calculated_posts%5 ==0):
            print(calculated_posts)
    return dice_coefficents

def too_high_doc_frequency(word, frequency_limit, n_docs_in_corpus, index):
    n_docs = len(index[word])
    return frequency_limit < (n_docs / n_docs_in_corpus)

def calculate_top_100_neigbours(word_simularity_list):
    dice_list = sorted(word_simularity_list, key=itemgetter(1), reverse=True)
    return dice_list[:100]

def get_dice_based_key_words(word_index, bigram_index, train_data_folder, category_posting_list, weight_filter_limit, frequency_limit, n_docs_in_corpus):
    dice_coefficents_word = calculate_dice_coefficients_word(category_posting_list, word_index, train_data_folder)
    top_100_dice_coefficients_word = calculate_top_100_neigbours(dice_coefficents_word.items())
    top_dice_coefficients_word = [w for w in top_100_dice_coefficients_word if not too_high_doc_frequency(w[0], frequency_limit,n_docs_in_corpus,word_index)]
    
    dice_coefficents_bigram = calculate_dice_coefficients_bigram(category_posting_list, bigram_index,train_data_folder)
    top_100_dice_coefficients_bigram = calculate_top_100_neigbours(dice_coefficents_bigram.items())
    top_dice_coefficients_bigram = [ b for b in top_100_dice_coefficients_bigram if not too_high_doc_frequency(b[0], frequency_limit,n_docs_in_corpus, bigram_index)]

    all_dice_coefficients = calculate_top_100_neigbours(top_dice_coefficients_word + top_dice_coefficients_bigram) 

    reference_words = [d for d in all_dice_coefficients if weight_filter_limit <= d[1]]
    
    context_words = [d for d in all_dice_coefficients if weight_filter_limit > d[1]]

    return reference_words, context_words


# word_index = index.get_index(WORD_INDEX_PICKLE_FILE)
# category_posting_list = word_index['karate']
# bigram_index = index.get_index(BIGRAM_INDEX_PICKLE_FILE)
# print ('loaded index')
# r, c = get_dice_based_key_words(word_index, bigram_index, TRAIN_DATA_FOLDER,category_posting_list, 0.05,0.04,NUMBER_OF_DOCUMENTS)
# print(r)
# print(c)


