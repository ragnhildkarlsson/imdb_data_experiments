import nltk
import random
import re
import os
from nltk.corpus import stopwords
import pickle_handler

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


def create_inverted_index_word(file_path, set_trainings_data_doc_numbers):
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
                    freqDist = nltk.FreqDist(document)
                    for word in freqDist:
                        if not word in stop_words:
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

file_index_set = get_set_of_random_numbers(NUMBER_OF_DOCUMENTS, 220000)
print_train_data_subset_list_to_file(file_index_set, FILENAME_TRAIN_DATA_SUBSET, ROOT_DATA_FOLDER, TRAIN_DATA_FOLDER)
index = create_inverted_index_word(TRAIN_DATA_FILE,file_index_set)
index_folder = os.path.join(ROOT_DATA_FOLDER, INDEX_FOLDER)
# print_index_to_file(index, index_folder)
pickle_handler.print_pickle(index, WORD_INDEX_PICKLE_FILE)
bigram_index = create_inverted_index_bigram(TRAIN_DATA_FILE,file_index_set)
pickle_handler.print_pickle(bigram_index, BIGRAM_INDEX_PICKLE_FILE)
#print_index_to_file(bigram_index, index_folder)
