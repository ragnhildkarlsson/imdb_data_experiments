import os
import nltk
from operator import itemgetter
import index
import document_processer 

ROOT_DATA_FOLDER = "data"
TRAIN_DATA_FOLDER = "data/train_data"
TEST_DATA_FOLDER = "data/test_data"
FILENAME_TRAIN_DATA_SUBSET = "list_train_data_subset"
INDEX_FOLDER = "data/index"
WORD_INDEX_PICKLE_FILE = "data/index_word_pickle_file"
BIGRAM_INDEX_PICKLE_FILE = "data/index_bigram_pickle_file"

def get_list_of_categories(test_data_folder):
    return next(os.walk(test_data_folder))[1]


def calculate_dice_coefficients_word(category_posting_list, word_index, train_data_folder):
    # returns {w1:dice(cat,w1),w2:dice(cat,w2)}
    dice_coefficents = {}
    print("Number of docs with the category name")
    print(len(category_posting_list))
    calculated_posts = 0
    for post in category_posting_list:
        document = document_processer.get_document_string(str(post[0]),train_data_folder)
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
        document = document_processer.get_document_string(str(post[0]),train_data_folder)
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

def calculate_top_100_neigbours(word_simularity_map):
    dice_list = sorted(word_simularity_map.items(),key=itemgetter(1), reverse=True)
    return dice_list[:100]

import pdb
pdb.set_trace()
word_index = index.get_index(WORD_INDEX_PICKLE_FILE)
category_posting_list = word_index['art']
word_dice = calculate_dice_coefficients_word(category_posting_list, word_index, TRAIN_DATA_FOLDER)
print(calculate_top_100_neigbours(word_dice))
bigram_index = index.get_index(BIGRAM_INDEX_PICKLE_FILE)
bigram_dice = calculate_dice_coefficients_bigram(category_posting_list, bigram_index, TRAIN_DATA_FOLDER)
print(calculate_top_100_neigbours(bigram_dice))

