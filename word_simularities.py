import os
import nltk
from operator import itemgetter
import index
import document_processer 

ROOT_DATA_FOLDER = "data"
TRAIN_DATA_FOLDER = "data/train_data"
WORD_INDEX_PICKLE_FILE = "data/index/index_word_pickle_file"
BIGRAM_INDEX_PICKLE_FILE = "data/index/index_bigram_pickle_file"

def get_list_of_categories(test_data_folder):
    return next(os.walk(test_data_folder))[1]

def calculate_dice_coefficients_word(category_posting_list, word_index, train_data_folder):
    # returns {w1:dice(cat,w1),w2:dice(cat,w2)}
    dice_coefficents = {}
    calculated_posts = 0
    for post in category_posting_list:
        doc_file_name = str(post[0])
        document = document_processer.get_document_string(doc_file_name, train_data_folder)
        document = document_processer.preprocess_document(document)
        for word in document:
            if word not in dice_coefficents and word in word_index:
                word_posting_list = word_index[word]
                intersection = index.intersection_search(category_posting_list, word_posting_list)
                dice_coefficents[word] = len(intersection)/(len(word_posting_list)+len(category_posting_list))
        calculated_posts +=1
        if(calculated_posts%50==0):
            print(calculated_posts)
    return dice_coefficents

def calculate_dice_coefficients_bigram(category_posting_list, bigram_index, train_data_folder):
    dice_coefficents = {}
    calculated_posts = 0
    for post in category_posting_list:
        doc_file_name = str(post[0])
        document = document_processer.get_document_string(doc_file_name,train_data_folder)
        document = document_processer.preprocess_document(document)    
        bigram_freqDist = nltk.FreqDist(nltk.bigrams(document))
        for bigram in bigram_freqDist:
            bigram_string = document_processer.bigram_to_string(bigram)
            if bigram_string not in dice_coefficents and bigram_string in bigram_index:
                bigram_posting_list = bigram_index[bigram_string]
                intersection = index.intersection_search(category_posting_list, bigram_posting_list)
                dice_coefficents[bigram_string] = len(intersection)/(len(bigram_posting_list)+len(category_posting_list))
        calculated_posts +=1
        if(calculated_posts%50==0):
            print(calculated_posts)
    return dice_coefficents

def calculate_top_n_neigbours(n, word_simularity_list):
    #sort on simularity
    top_list = sorted(word_simularity_list, key=itemgetter(1), reverse=True)
    return top_list[:n]

def get_n_dice_based_key_words(n, word_index, bigram_index, train_data_folder, category_posting_list):
    dice_coefficents_word = calculate_dice_coefficients_word(category_posting_list, word_index, train_data_folder)
    top_dice_coefficients_word = calculate_top_n_neigbours(n, dice_coefficents_word.items())
    dice_coefficents_bigram = calculate_dice_coefficients_bigram(category_posting_list, bigram_index, train_data_folder)
    top_dice_coefficients_bigram = calculate_top_n_neigbours(dice_coefficents_bigram.items())
    top_n_dice_keywords = calculate_top_n_neigbours(n, top_dice_coefficients_word + top_dice_coefficients_bigram) 
    return top_n_dice_keywords

