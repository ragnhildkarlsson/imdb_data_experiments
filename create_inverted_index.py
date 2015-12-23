import nltk
import random
import re
import os
from nltk.corpus import stopwords
import pickle

import document_processer 

NUMBER_OF_DOCUMENTS = 458712

ROOT_DATA_FOLDER = "data"
TRAIN_DATA_FOLDER = "data/train_data"
TRAIN_DATA_FILE = "data/all_plots_train_delimeter"
FILENAME_TRAIN_DATA_SUBSET = "list_train_data_subset"
ENCODING="ISO-8859-1"
INDEX_FOLDER = "index"
STOP_WORD_FILE_PATH = "data/stop_word_list"
WORD_INDEX_PICKLE_FILE = "data/index_word_pickle_file"
BIGRAM_INDEX_PICKLE_FILE = "data/index_bigram_pickle_file"

def is_delimeter_line(line):
    break_line_matcher = '^---.*' 
    return re.match(break_line_matcher, line)

def get_doc_id(line):
    doc_id_matcher = '-*(\d+)'
    doc_id = re.search(doc_id_matcher, line).group(1)
    return int(doc_id)
    
def get_set_of_random_numbers(max_number, size):
    l = list(range(max_number))
    random.shuffle(l)
    l =  l[:size]
    s = set(l)
    return s

def print_train_data_subset_list_to_file(file_index_set, subset_data_list_file_name, list_file_folder, data_folder):
    file_id_list = list(file_index_set)
    file_path_list = []
    for file_id in file_index_set:
        file_path  = os.path.join(data_folder, str(file_id))
        file_path_list.append(file_path)

    list_file_path = os.path.join(list_file_folder, subset_data_list_file_name)
    with open(list_file_path, 'w') as f:
        for line in file_path_list:
            f.write(line+'\n')

def print_index_to_file(index, index_folder):
    if not os.path.exists(index_folder):
        os.makedirs(index_folder)
    print(len(index))
    n_printed_words = 0            
    for word in index:
        if(n_printed_words%5000)==0:
            print(n_printed_words)
        posting_list_file = os.path.join(index_folder, word)
        pickle.dump(index[word], open(posting_list_file,'wb'))
        n_printed_words +=1

def load_index_pickle(file_path):
    index = pickle.load( open(file_path, "rb" ) )
    return index

def print_index_pickle_file(index, index_file):
    pickle.dump(index, open(index_file,'wb'))


def print_stop_word_list_to_file(index, stop_word_file_path):
    n_words_in_corpus = sum([sum([post[1] for post in index[word]]) for word in index])
    print("n_words_in_corpus calcullated")
    stop_words = []
    for word in index:
        word_freq = sum([post[1] for post in index[word]]) 
        if (word_freq/n_words_in_corpus) > 0.0011:
            stop_words.append(word)
    print('word_freq_calculated')
    with open(stop_word_file_path, 'w') as f:
        for stop_word in stop_words:
                f.write(stop_word+'\n')

def create_inverted_index_word(file_path, set_trainings_data_doc_numbers):
    index = {}
    plot_lines = []
    indexed_files =0
    with open(file_path, encoding="ISO-8859-1") as f:
        for line in f:
            if is_delimeter_line(line):
                doc_id = get_doc_id(line)
                if(doc_id in set_trainings_data_doc_numbers):
                    indexed_files +=1
                    if(indexed_files%100)==0:
                        print(indexed_files)
                    document = ' '.join(plot_lines)    
                    document = document_processer.preprocess_document(document)
                    freqDist = nltk.FreqDist(document)
                    for word in freqDist:
                        if not word in index:
                            index[word] = []
                        posting = (doc_id, freqDist[word])
                        index[word].append(posting)
                plot_lines = []
            else:
                plot_lines.append(line)
    return index

def create_inverted_index_bigram(file_path, set_trainings_data_doc_numbers):
    index = {}
    plot_lines = []
    indexed_files =0
    stop_words = stopwords.words('english')
    stop_words = set(stop_words)
    with open(file_path, encoding="ISO-8859-1") as f:
        for line in f:
            if is_delimeter_line(line):
                doc_id = get_doc_id(line)
                if(doc_id in set_trainings_data_doc_numbers):
                    indexed_files +=1
                    if(indexed_files%100)==0:
                        print(indexed_files)
                    document = ' '.join(plot_lines)    
                    document = document_processer.preprocess_document(document)
                    bigram_freqDist = nltk.FreqDist(nltk.bigrams(document)) 
                    for bigram in bigram_freqDist:
                        if not (bigram[0] in stop_words) and (not bigram[1] in stop_words):            
                            bigram_string = document_processer.bigram_to_string(bigram)
                            if not bigram_string in index:
                                index[bigram_string] = []
                            posting = (doc_id, bigram_freqDist[bigram])
                            index[bigram_string].append(posting)
                plot_lines = []
            else:
                plot_lines.append(line)

    return index                

# file_index_set = get_set_of_random_numbers(NUMBER_OF_DOCUMENTS, 120000)
# print_train_data_subset_list_to_file(file_index_set, FILENAME_TRAIN_DATA_SUBSET, ROOT_DATA_FOLDER, TRAIN_DATA_FOLDER)
# index = create_inverted_index_word(TRAIN_DATA_FILE,file_index_set)
# index_folder = os.path.join(ROOT_DATA_FOLDER, INDEX_FOLDER)
# print_index_to_file(index, index_folder)
# print_index_pickle_file(index, WORD_INDEX_PICKLE_FILE)

# index = load_index_pickle(WORD_INDEX_PICKLE_FILE)
# print("INDEX LOADED")
# print_stop_word_list_to_file(index, STOP_WORD_FILE_PATH)
bigram_index = create_inverted_index_bigram(TRAIN_DATA_FILE,file_index_set)
print_index_pickle_file(bigram_index, BIGRAM_INDEX_PICKLE_FILE)
#print_index_to_file(bigram_index, index_folder)
