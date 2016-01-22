import nltk
import os
import re

from nltk.corpus import stopwords

def get_names_of_subdirectory_list(folder):
    subdirectories =  next(os.walk(folder))[1]
    return subdirectories

def get_names_of_files_in_directory(folder):
    files =  next(os.walk(folder))[2]
    return files 

def get_lines_in_file_by_path(path):
    with open(path,'r') as f:
        lines = f.readlines()
    return lines

def get_lines_in_file_by_name(file_name, folder):
    path = os.path.join(folder, file_name)
    lines = get_lines_in_file_by_path(path)
    return lines

def get_document_string(file_name, folder):
    path = os.path.join(folder, file_name)
    lines = get_lines_in_file_by_path(path)
    document = ' '.join(lines)
    return document

def is_stop_word(word):
    stop_words = stopwords.words('english')
    if word in stop_words:
        return True
    else:
        return False

def preprocess_document(document):
    originaldocumt = document
    document = nltk.word_tokenize(document)
    document = [w for w in document if w.isalpha()]
    document = [w.lower() for w in document]
    return document

def bigram_to_string(bigram):
    return bigram[0]+'_'+bigram[1]

