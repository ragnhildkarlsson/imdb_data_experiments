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

def get_list_of_categories(test_data_folder):
    return next(os.walk(test_data_folder))[1]


def calculate_dice_coefficients(category_name, root_data_folder, name_file_list, test_data_folder, train_data_folder, index_folder):
    # returns {w1:dice(cat,w1),w2:dice(cat,w2)}
    terms = index.get_index_keys(index_folder)
    if not category_name in terms:
        return []
    import pdb
    pdb.set_trace()
    category_posting_list = index.get_posting_list(category_name, index_folder)
    dice_coefficents = {}
    print(len(category_posting_list))
    calculated_posts = 0
    for post in category_posting_list:
        #Calculate dice coefficient for single words
        document = document_processer.get_document_string(str(post[0]),train_data_folder)
        document = document_processer.preprocess_document(document)
        for word in document:
            if word not in dice_coefficents and word in terms:
                word_posting_list = index.get_posting_list(word, index_folder)
                intersection = index.intersection_search(category_posting_list, word_posting_list)
                dice_coefficents[word] = len(intersection)/(len(word_posting_list)+len(category_posting_list))
        calculated_posts +=1        
        # Calculate dice coefficients for bigram
        bigram_freqDist = nltk.FreqDist(nltk.bigrams(document))
        for bigram in bigram_freqDist:
            bigram_string = document_processer.bigram_to_string(bigram)
            if bigram_string not in dice_coefficents and bigram_string in terms:
                bigram_posting_list = index.get_posting_list(bigram_string, index_folder)
                intersection = index.intersection_search(category_posting_list, bigram_posting_list)
                dice_coefficents[bigram_string] = len(intersection)/(len(bigram_posting_list)+len(category_posting_list))
        
        if(calculated_posts%5 ==0):
            print(calculated_posts)
    return dice_coefficents

def calculate_top_100_neigbours(word_simularity_map):
    dice_list = sorted(word_simularity_map.items(),key=itemgetter(1), reverse=True)
    return dice_list[:100]

d = calculate_dice_coefficients('art', ROOT_DATA_FOLDER, FILENAME_TRAIN_DATA_SUBSET, TEST_DATA_FOLDER, TRAIN_DATA_FOLDER, INDEX_FOLDER)

print(calculate_top_100_neigbours(d))

# art
# A) Dice(w_1,w_2) = D(w_1,w_2) / D(w_1)+D(w_2) 
# WHERE 
# a1) D(w_1,w_2) = is the number of documents that contain  w1 and w2.  And D(w_i) i number of documents that contains D(w_1)
# D(w_1,w_2) = len(intersection_search cat, term) / len(cat)+ len(w2)


# categories = get_list_of_categories(TEST_DATA_FOLDER)
# print(categories)


